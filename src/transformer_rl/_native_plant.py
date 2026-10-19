"""Private physical-step ordering for one already admitted, owned native scene.

This component has no policy, reward, curriculum, SDK launcher or resource
allocator. The effective-response probe adds no second transport delay. Formal
use still requires an independently closed timing path and physical evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ._native_actuation import NativeEffortSubmission
from ._native_contacts import TerrainContacts
from ._native_observation import SurfaceSample
from ._native_scene import NativeSceneState
from ._output_actuation import DriveInterval, EffectiveDriveResponse, FeedbackEffort, OutputFeedbackController
from ._passive_pressure import NativeSliderPressure, PressureState
from ._physical_observation import MeasuredState, RigidBodyMotion, WheelContactHistory
from ._wheel_clearance import WheelClearanceSample


@dataclass(frozen=True)
class GroundReference:
    root_world_z_m: torch.Tensor
    wheel_world_z_m: torch.Tensor
    wheel_friction: torch.Tensor
    wheel_clearance: WheelClearanceSample | None = None


@dataclass(frozen=True)
class PhysicalTick:
    feedback: FeedbackEffort
    drive: DriveInterval
    pressure: PressureState
    submitted: NativeEffortSubmission
    explicit_actuator_buffer: dict
    contacts: TerrainContacts
    measured: MeasuredState
    contact_labels_valid: torch.Tensor
    physical_tick: int
    body_motion: RigidBodyMotion
    wheel_clearance: WheelClearanceSample | None = None


class NativePhysicsPlant:
    """Order feedback, effort writing, physics, buffer refresh and contact labels.

    Observation reads never advance physics. A failed write/step/read faults
    the plant because the partially advanced physical interval is not safe to
    replay. Reset invalidates sensor state, clears only selected drive/contact
    rows, and keeps the global physical clock. The collector owns history,
    episode IDs, ownership masks and nonrefundable sample accounting.
    """

    def __init__(self, sim, scene: NativeSceneState, controller: OutputFeedbackController,
                 drive: EffectiveDriveResponse, pressure: NativeSliderPressure, *, native_to_tensor,
                 ground_query, timing_path):
        if (not isinstance(scene, NativeSceneState) or not isinstance(controller, OutputFeedbackController)
                or not isinstance(drive, EffectiveDriveResponse) or not isinstance(pressure, NativeSliderPressure)
                or not callable(native_to_tensor) or not callable(ground_query)
                or timing_path != 'effective_response_probe_no_extra_transport'):
            raise ValueError('plant requires bound components, ground geometry query and an explicit probe timing path')
        bridge = scene.observation
        if (scene.robot is not bridge.robot or scene.efforts.robot is not scene.robot
                or any(value.num_envs != bridge.num_envs or value.device != bridge.device
                       for value in (controller, drive, pressure, scene.contacts))
                or scene.contacts.terrain_motion_reference != 'authored_static_own_terrain'
                or float(sim.cfg.dt) != bridge.physics_dt_s):
            raise ValueError('plant components must share the actual native owner, environment count, device and clock')
        self.sim, self.scene, self.controller, self.drive, self.pressure = sim, scene, controller, drive, pressure
        self.num_envs, self.device, self.dt = bridge.num_envs, bridge.device, bridge.physics_dt_s
        self._convert, self._ground = native_to_tensor, ground_query
        self._view, self._faulted, self.targets = scene.robot.root_view, False, None
        self._last_tick_packet = None
        self.contact_history = WheelContactHistory(self.num_envs, self.device, physics_dt_s=self.dt)
        self._springs = [bridge.binding.native_joint_names.index(name) for name in pressure.parameters.joint_names]
        self._active = list(bridge.binding.policy_joint_indices)
        self._identity()

    def _identity(self):
        if self._faulted:
            raise RuntimeError('partially advanced physical plant is faulted; do not replay its interval')
        bridge = self.scene.observation
        if (self.scene.robot.root_view is not self._view or bridge.robot is not self.scene.robot
                or self.scene.contacts.terrain_motion_reference != 'authored_static_own_terrain'
                or tuple(self.scene.robot.joint_names) != bridge.binding.native_joint_names
                or tuple(self.scene.robot.body_names) != bridge.binding.native_body_names):
            self._faulted = True
            raise RuntimeError('physical plant native owner, terrain-motion binding or named topology changed')

    def _matrix(self, name, value, width):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width)
                or value.device != self.device or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(name + ' must match the native finite float32 layout')
        return value

    def _feedback(self):
        # Direct getters are valid immediately after reset writes; cached SDK
        # observation buffers remain invalid until the next real sim.step.
        q = self._matrix('native q', self._convert(self._view.get_dof_positions()), 18).clone()
        dq = self._matrix('native dq', self._convert(self._view.get_dof_velocities()), 18).clone()
        return q, dq

    @torch.no_grad()
    def accept_issued_action(self, issued_action):
        self._identity()
        q, _ = self._feedback()
        self.targets = self.controller.decode(issued_action, q[:, self._active], self.scene.observation.timestamp_s())
        return self.targets

    @torch.no_grad()
    def step_physics(self, *, motor_strength=None, pressure_strength=None):
        self._identity()
        if self.targets is None:
            raise RuntimeError('an explicit policy or control-probe target is required before physics')
        bridge, robot = self.scene.observation, self.scene.robot
        try:
            bridge.before_physics_step()
            q, dq = self._feedback()
            feedback = self.controller.effort(self.targets, q[:, self._active], dq[:, self._active],
                                               bridge.timestamp_s(), strength=motor_strength)
            drive = self.drive.advance(feedback.capped_effort,
                                       torch.full((self.num_envs,), self.dt, dtype=torch.float64, device=self.device),
                                       feedback.effort_bounds)
            pressure = self.pressure.evaluate(q[:, self._springs], strength=pressure_strength)
            submitted = self.scene.efforts.submit(drive.mean_effort, pressure)
            robot.write_data_to_sim()
            explicit = self.scene.efforts.explicit_actuator_buffer()
            self.sim.step(render=False)
            bridge.after_physics_step(bridge.physical_tick + 1)
            motion = bridge.body_motion()
            contacts = self.scene.contacts.read(self.dt, bridge.timestamp_s(), body_motion=motion)
            bridge.before_state_read()
            if (motion.physical_tick != bridge.physical_tick or motion.model_epoch != bridge.model_epoch
                    or not torch.equal(motion.timestamp_s, bridge.timestamp_s())):
                raise RuntimeError('contact motion crossed a native physical or model epoch')
            ground = self._ground(robot, self.scene)
            if not isinstance(ground, GroundReference):
                raise ValueError('ground query must return named values from this scene geometry')
            clearance = ground.wheel_clearance
            if clearance is not None and (not isinstance(clearance, WheelClearanceSample)
                    or clearance.physical_tick != bridge.physical_tick or clearance.model_epoch != bridge.model_epoch
                    or not torch.equal(clearance.timestamp_s, bridge.timestamp_s())
                    or clearance.body_names != tuple(bridge.binding.native_body_names[i]
                                                     for i in bridge.binding.wheel_body_indices)):
                raise ValueError('wheel clearance differs from the actual physical tick/model/named wheel identity')
            self._matrix('ground wheel z', ground.wheel_world_z_m, 2)
            self._matrix('wheel friction', ground.wheel_friction, 2)
            if (not isinstance(ground.root_world_z_m, torch.Tensor) or ground.root_world_z_m.shape != (self.num_envs,)
                    or ground.root_world_z_m.device != self.device or ground.root_world_z_m.dtype != torch.float32
                    or not torch.isfinite(ground.root_world_z_m).all() or (ground.wheel_friction < 0).any()):
                raise ValueError('ground root height and friction must be finite scene geometry values')
            surfaces = SurfaceSample(contacts.force_world_n, ground.root_world_z_m, ground.wheel_world_z_m,
                                      self.scene.environment_origins_m, ground.wheel_friction, contacts.timestamp_s,
                                      bridge.binding.native_body_names)
            measured = bridge.measure(surfaces)
            # Positive normal patch loads are used for contact labels. The
            # critic's diagnostic force vector may legitimately cancel.
            self.contact_history.push(contacts.normal_load_n[:, list(bridge.binding.wheel_body_indices)],
                                      contacts.timestamp_s)
            _, valid = self.contact_history.contacts()
            self._identity()
        except BaseException:
            self._faulted = True
            bridge.invalidate_state()
            raise
        result = PhysicalTick(feedback, drive, pressure, submitted, explicit, contacts, measured,
                                valid, bridge.physical_tick, motion, clearance)
        self._last_tick_packet = result
        return result

    @torch.no_grad()
    def reset(self, mask, root_pose_xyzw, root_com_velocity, joint_position, joint_velocity):
        self._identity()
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError('physical reset requires a same-device row mask')
        for name, value, width in (('root pose', root_pose_xyzw, 7), ('root COM velocity', root_com_velocity, 6),
                                   ('joint position', joint_position, 18), ('joint velocity', joint_velocity, 18)):
            self._matrix(name, value, width)
        if not torch.allclose(root_pose_xyzw[:, 3:].norm(dim=-1), torch.ones(self.num_envs, device=self.device),
                              rtol=0, atol=1e-5):
            raise ValueError('physical reset quaternion must use normalized xyzw')
        if not mask.any():
            return
        bridge, robot = self.scene.observation, self.scene.robot
        ids = mask.nonzero(as_tuple=False).flatten()
        bridge.invalidate_state()
        try:
            robot.write_root_link_pose_to_sim_index(root_pose=root_pose_xyzw[ids], env_ids=ids)
            robot.write_root_com_velocity_to_sim_index(root_velocity=root_com_velocity[ids], env_ids=ids)
            robot.write_joint_position_to_sim_index(position=joint_position[ids], env_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=joint_velocity[ids], env_ids=ids)
            robot.reset(ids)
            self.drive.reset(mask)
            self.contact_history.reset(mask, bridge.timestamp_s())
            if self.targets is not None:
                fresh = self.controller.decode(torch.zeros(self.num_envs, 6, device=self.device),
                                                joint_position[:, self._active], bridge.timestamp_s())
                # Keep untouched rows' issued targets and timing exactly.
                for field in fields(fresh):
                    getattr(self.targets, field.name)[mask] = getattr(fresh, field.name)[mask]
        except BaseException:
            self._faulted = True
            raise

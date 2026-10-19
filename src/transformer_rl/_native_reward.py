"""Private native reward inputs bound to the actual model and collection clock.

This adapter reads its own admitted plant only. It creates no SDK, context,
training run or qualification. True wheel clearance and the complete task/
termination environment remain separate; contacts cannot invent that evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._control_reward import PhysicalRewardComposer, RewardPhysics, RewardTuning
from ._joint_boundaries import JointBoundaryState, LegBoundaryGeometry
from ._native_plant import NativePhysicsPlant, PhysicalTick
from ._structured_collection import StructuredCollectionConfig


@dataclass(frozen=True)
class NativeRewardSample:
    physics: RewardPhysics
    boundary: JointBoundaryState
    mass_kg_by_environment: torch.Tensor
    physical_tick: int
    model_epoch: int
    output_pair_admission_enabled: bool


class NativeRewardBridge:
    """One model snapshot per epoch; every reward input from its physical tick.

    Mechanical setters must invalidate/refresh the plant observation model,
    then explicitly refresh this snapshot at a closed policy boundary. An
    invalid snapshot blocks sampling; an old mass is never silently reused.
    """

    def __init__(self, plant: NativePhysicsPlant, geometry: LegBoundaryGeometry,
                 collection_config: StructuredCollectionConfig, *, knee_margin_rad, retired_request_ids):
        if (not isinstance(plant, NativePhysicsPlant) or not isinstance(geometry, LegBoundaryGeometry)
                or not isinstance(collection_config, StructuredCollectionConfig)
                or type(knee_margin_rad) not in (int, float) or not math.isfinite(knee_margin_rad)
                or not 0 < knee_margin_rad < (geometry.inner_limits_rad[1] - geometry.inner_limits_rad[0]) / 2):
            raise ValueError('native reward requires its actual plant, selected geometry and collection configuration')
        if (not isinstance(retired_request_ids, torch.Tensor) or retired_request_ids.shape != (plant.num_envs,)
                or retired_request_ids.dtype != torch.long or retired_request_ids.device != plant.device
                or (retired_request_ids < -1).any()):
            raise ValueError('native reward requires explicit current/restored protocol consumed IDs,including fresh-1')
        bridge = plant.scene.observation
        source = plant.scene.readback
        if (geometry.binding != bridge.binding
                or source.get('selected_asset_manifest_sha256') != geometry.manifest_sha256
                or source.get('selected_model_spec_sha256') != geometry.model_spec_sha256
                or not torch.equal(plant.controller.nominal, torch.tensor(geometry.binding.nominal_policy_position_rad,
                                                                          device=plant.device, dtype=torch.float32))
                or not torch.equal(plant.controller.action_bounds, torch.tensor(collection_config.action_bounds,
                                                                                device=plant.device, dtype=torch.float32))):
            raise ValueError('native reward geometry or output controller differs from its actual selected scene')
        self.plant, self.geometry, self.collection_config = plant, geometry, collection_config
        self._scene, self._controller = plant.scene, plant.controller
        self.num_envs, self.device, self.knee_margin_rad = plant.num_envs, plant.device, float(knee_margin_rad)
        self._view, self._physics_view = plant.scene.robot.root_view, plant.scene.contacts.physics_view
        self._epoch, self._ready, self._last_tick = None, False, bridge.physical_tick
        self.composer = None
        self.refresh_model()
        tuning = RewardTuning(geometry.nominal_mass_kg, gamma=collection_config.gamma,
                              policy_dt_s=collection_config.policy_dt_s, gravity_m_s2=self.gravity_m_s2)
        self.composer = PhysicalRewardComposer(bridge.binding, self.num_envs, tuning, device=self.device,
                                                physics_dt_s=plant.dt, retired_request_ids=retired_request_ids,
                                                robot_mass_kg_by_environment=self.mass_kg_by_environment)

    def validate_collection_config(self, config):
        if not isinstance(config, StructuredCollectionConfig) or config != self.collection_config:
            raise ValueError('native reward and collector must share the same100Hz discount and complete collection contract')

    @property
    def mass_kg_by_environment(self):
        return self._mass_kg_by_environment.clone()

    @property
    def body_mass_kg(self):
        return self._body_mass_kg.clone()

    def _identity(self):
        self.plant._identity()
        if (self.plant.scene is not self._scene or self.plant.controller is not self._controller
                or self.plant.scene.robot.root_view is not self._view
                or self.plant.scene.contacts.physics_view is not self._physics_view
                or self.plant.scene.readback.get('selected_asset_manifest_sha256') != self.geometry.manifest_sha256
                or self.plant.scene.readback.get('selected_model_spec_sha256') != self.geometry.model_spec_sha256):
            raise RuntimeError('native reward model or physics view identity changed')

    def _tensor(self, name, value, shape, dtype=torch.float32):
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                or value.device != self.device):
            raise ValueError('native reward ' + name + ' layout/device/dtype differs')

    @torch.no_grad()
    def refresh_model(self):
        self._identity()
        bridge = self.plant.scene.observation
        if (self.composer is not None and self.composer._window is not None
                or bridge.physical_tick % round(.01 / self.plant.dt) or bridge.physical_tick != self._last_tick):
            raise RuntimeError('native reward model refresh requires its drained complete policy boundary')
        self._ready = False
        bridge.before_physics_step()
        epoch, tick = bridge.model_epoch, bridge.physical_tick
        if self._epoch is not None and epoch <= self._epoch:
            raise ValueError('native model writes require a new final observation model epoch')
        mass = self.plant._convert(self._view.get_masses())
        # Model parameter tensors may legitimately reside on the native
        # parameter device. Unlike live state, they are copied once per epoch.
        if (not isinstance(mass, torch.Tensor) or mass.shape != (self.num_envs, 19) or mass.dtype != torch.float32
                or not torch.isfinite(mass).all() or (mass <= 0).any()):
            raise ValueError('native reward needs complete positive body masses')
        mass = mass.detach().clone()
        limits = self.plant._convert(self._view.get_dof_limits())
        if (not isinstance(limits, torch.Tensor) or limits.shape != (self.num_envs, 18, 2)
                or limits.dtype != torch.float32):
            raise ValueError('native reward needs explicit complete DOF limits')
        indices = [bridge.binding.native_joint_names.index(side.knee_joint) for side in self.geometry.sides]
        knee_limits = limits[:, indices].detach().to(self.device).clone()
        expected = torch.tensor([side.native_limits_rad for side in self.geometry.sides], device=self.device, dtype=torch.float32)
        if (not torch.isfinite(knee_limits).all()
                or not torch.allclose(knee_limits, expected[None].expand_as(knee_limits), atol=1e-6, rtol=0)):
            raise ValueError('actual native passive knee limits differ from the selected asset source')
        gravity = self._physics_view.get_gravity()
        configured = getattr(self.plant.sim.cfg, 'gravity', None)
        for value in (gravity, configured):
            if (not isinstance(value, (tuple, list)) or len(value) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or abs(value[0]) > 1e-8 or abs(value[1]) > 1e-8 or value[2] >= 0):
                raise ValueError('native and configured gravity must explicitly share the world-negative-Z convention')
        if any(abs(a - b) > 1e-5 for a, b in zip(gravity, configured)):
            raise ValueError('actual native gravity differs from the simulation recipe')
        g = -float(gravity[2])
        total = mass.double().sum(-1).to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(total).all():
            raise ValueError('actual total mass exceeded the native physical tensor range')
        bridge.before_physics_step()
        if epoch != bridge.model_epoch or tick != bridge.physical_tick:
            raise RuntimeError('native reward model readback crossed an epoch or physical step')
        if self.composer is not None:
            if g != self.composer.tuning.gravity_m_s2:
                raise ValueError('changing gravity requires a newly reviewed task/reference recipe')
            self.composer.update_native_mass(total)
        self._mass_kg_by_environment = total.detach().clone()
        self._body_mass_kg = mass.detach().to(self.device).clone()
        self.knee_limits_rad, self.gravity_m_s2 = knee_limits, g
        self._epoch, self._ready = epoch, True

    @torch.no_grad()
    def sample(self, tick: PhysicalTick, *, episode_id, control_epoch, policy_owned):
        self._identity()
        bridge = self.plant.scene.observation
        bridge.before_state_read()
        if (not self._ready or bridge.model_epoch != self._epoch or not isinstance(tick, PhysicalTick)
                or self.plant._last_tick_packet is not tick
                or tick.physical_tick != self._last_tick + 1 or tick.physical_tick != bridge.physical_tick
                or tick.body_motion.model_epoch != self._epoch or tick.body_motion.physical_tick != tick.physical_tick
                or self.plant.targets is None or tick.contacts.wheel_motion is None):
            raise ValueError('native reward requires the next actual owned tick and current final model snapshot')
        window = self.composer._window
        if window is None or window['steps'] >= self.composer.substeps:
            raise RuntimeError('native reward sampling requires its open complete-policy interval capacity')
        for name, value, dtype in (('episode', episode_id, torch.long), ('control', control_epoch, torch.long),
                                    ('policy ownership', policy_owned, torch.bool)):
            self._tensor(name, value, (self.num_envs,), dtype)
        state, motion = tick.measured, tick.body_motion
        slip = tick.contacts.wheel_motion
        if (motion.body_names != bridge.binding.native_body_names or slip.model_epoch != self._epoch
                or slip.physical_tick != tick.physical_tick
                or any(not torch.equal(value, bridge.timestamp_s()) for value in
                       (state.timestamp_s, motion.timestamp_s, tick.contacts.timestamp_s, slip.timestamp_s))
                or not torch.equal(tick.feedback.policy_issued_at_s, self.plant.targets.policy_issued_at_s)):
            raise ValueError('native reward motion, contact, feedback or target source identity differs')
        if (not torch.equal(episode_id, window['episode']) or not torch.equal(control_epoch, window['control'])
                or (state.timestamp_s - window['start'] - (window['steps'] + 1) * self.plant.dt).abs().gt(1e-8).any()):
            raise ValueError('native reward episode, control or interval sampling clock differs')
        boundary = self.geometry.evaluate(motion.link_pose_world_xyzw, state.tree_position_rad, self.plant.targets,
                                            knee_margin_rad=self.knee_margin_rad)
        poses = motion.link_pose_world_xyzw
        q = poses[:, 0, 3:] / poses[:, 0, 3:].norm(dim=-1, keepdim=True)
        velocity = self.geometry._rotate(q, state.velocity_origin_body_m_s)
        x, y, z, w = q.unbind(-1)
        attitude = torch.stack((torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square())),
                                 (2 * (w * y - z * x)).clamp(-1, 1).asin()), -1)
        sample = RewardPhysics(state.timestamp_s.clone(), episode_id.clone(), control_epoch.clone(), policy_owned.clone(),
                                 state.velocity_origin_body_m_s.clone(), velocity.clone(), poses[:, 0, 2].clone(),
                                 state.height_above_ground_m.clone(), attitude, state.gravity_body.clone(),
                                 state.angular_velocity_body_rad_s.clone(), state.policy_velocity_rad_s.clone(),
                                 tick.explicit_actuator_buffer['active_output_effort_nm'].clone(),
                                 tick.contacts.normal_load_n.clone(), slip.tangent_speed_rms_m_s.clone(), boundary.shortfall_rad.clone())
        bridge.before_state_read()
        if bridge.physical_tick != tick.physical_tick or bridge.model_epoch != self._epoch:
            raise RuntimeError('native reward sampling crossed a physical or mechanical model epoch')
        self._last_tick = tick.physical_tick
        return NativeRewardSample(sample, boundary, self.mass_kg_by_environment.clone(), tick.physical_tick,
                                    self._epoch, self.plant.controller.pair_domain is not None)

    @torch.no_grad()
    def observe(self, tick, reference, *, episode_id, control_epoch, policy_owned):
        value = self.sample(tick, episode_id=episode_id, control_epoch=control_epoch, policy_owned=policy_owned)
        self.composer.observe(value.physics, reference)
        return value

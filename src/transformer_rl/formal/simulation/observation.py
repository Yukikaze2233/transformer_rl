"""Private native read boundary; SDK construction and admission remain external.

The bridge reads the selected articulation's own buffers after an explicit
physical refresh. It never launches an SDK, advances physics, controls another
process or substitutes default state for missing native data.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..control.observation import MechanicalBinding, NativePhysicalSample, ObservationAssembler, RigidBodyMotion


@dataclass(frozen=True)
class SurfaceSample:
    """Same-tick own-terrain data, in native body order and world SI units."""

    terrain_contact_force_world_n: torch.Tensor
    ground_at_root_world_z_m: torch.Tensor
    ground_at_wheel_world_z_m: torch.Tensor
    environment_origin_world_m: torch.Tensor
    wheel_friction: torch.Tensor
    timestamp_s: torch.Tensor
    body_names: tuple[str, ...]


class NativeObservationBridge:
    """Convert scalar-last native poses and COM velocities to named measurements.

    Call invalidate_model BEFORE mechanical setters, then refresh_model AFTER
    their last native write. State/reset writes call invalidate_state. Only
    after the next successful sim.step may after_physics_step refresh robot
    data. The bridge's counters measure reported refreshes, not independently
    certified physical steps; the actual SDK acceptance trace must prove that
    call order. Native views cannot be replaced under a live bridge.
    """

    def __init__(self, robot, manifest, num_envs, *, native_to_tensor, device='cpu', physics_dt_s=.005):
        if (not callable(native_to_tensor) or type(physics_dt_s) not in (int, float)
                or physics_dt_s not in (.001, .005)):
            raise ValueError('native bridge requires an explicit buffer converter and selected physical clock')
        self.robot, self._view, self._native_to_tensor = robot, robot.root_view, native_to_tensor
        self.binding = MechanicalBinding.from_manifest(manifest, robot.joint_names, robot.body_names)
        if self.binding.native_body_names[0] != 'base_link':
            raise ValueError('root COM velocity API requires native body zero to be base_link')
        self.assembler = ObservationAssembler(self.binding, num_envs, device)
        self.num_envs, self.device, self.physics_dt_s = num_envs, torch.device(device), physics_dt_s
        self.physical_tick, self.model_epoch, self.refresh_attempts = 0, 0, 0
        self._model_ready = self._state_ready = self._faulted = False
        self._local_com_m = self._local_body_com_m = None
        self.refresh_model()

    def _identity(self):
        if (self.robot.root_view is not self._view
                or tuple(self.robot.joint_names) != self.binding.native_joint_names
                or tuple(self.robot.body_names) != self.binding.native_body_names):
            self._faulted = True
            raise RuntimeError('native view or joint/body identity changed; construct a new bridge')
        if self._faulted:
            raise RuntimeError('failed native refresh requires an explicit new bridge')

    def invalidate_model(self):
        self._model_ready = self._state_ready = False
        self._local_com_m = self._local_body_com_m = None

    def invalidate_state(self):
        self._state_ready = False

    def before_physics_step(self):
        self._identity()
        if not self._model_ready:
            raise RuntimeError('final native model readback is required before advancing physics')

    def before_state_read(self):
        self._identity()
        if not self._model_ready or not self._state_ready:
            raise RuntimeError('model/state write requires final native COM and post-physics refresh before reading')

    @torch.no_grad()
    def refresh_model(self):
        self._identity()
        self.invalidate_model()
        local = self._native_to_tensor(self._view.get_coms())
        if (not isinstance(local, torch.Tensor) or local.shape != (self.num_envs, 19, 7)
                or local.dtype != torch.float32 or not torch.isfinite(local).all()):
            raise ValueError('native COM readback requires finite xyz+xyzw for every named body')
        self._local_body_com_m = local[:, :, :3].detach().to(self.device).clone()
        self._local_com_m = self._local_body_com_m[:, 0].clone()
        self._model_ready = True
        self.model_epoch += 1

    def after_physics_step(self, physical_tick):
        self._identity()
        if (not self._model_ready or type(physical_tick) is not int or physical_tick != self.physical_tick + 1):
            raise ValueError('refresh requires the next reported physical tick and final model readback')
        self._state_ready = False
        self.refresh_attempts += 1
        try:
            self.robot.update(self.physics_dt_s)
        except BaseException:
            self._faulted = True
            raise
        self.physical_tick = physical_tick
        self._state_ready = True

    def timestamp_s(self):
        return torch.full((self.num_envs,), self.physical_tick * self.physics_dt_s,
                          dtype=torch.float64, device=self.device)

    def _buffer(self, name, width):
        wrapper = getattr(self.robot.data, name)
        value = wrapper.torch
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width)
                or value.device != self.device or value.dtype != torch.float32):
            raise ValueError(f'native {name} buffer shape/device/dtype differs')
        return value

    @torch.no_grad()
    def body_motion(self):
        """Read fresh COM velocities and transform final native local COMs.

        The SDK's lazily cached COM-pose property is deliberately not used:
        explicit mechanical writes require our own final get_coms readback.
        No link-origin velocity is substituted for a COM velocity.
        """
        self.before_state_read()
        epoch, tick, timestamp = self.model_epoch, self.physical_tick, self.timestamp_s()
        poses = self.robot.data.body_link_pose_w.torch
        velocity = self.robot.data.body_com_vel_w.torch
        for name, value, width in (('link poses', poses, 7), ('COM velocities', velocity, 6)):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 19, width)
                    or value.dtype != torch.float32 or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError('native complete body ' + name + ' layout or finite values differ')
        root_pose, root_velocity = self._buffer('root_link_pose_w', 7), self._buffer('root_com_vel_w', 6)
        if (not torch.isfinite(root_pose).all() or not torch.isfinite(root_velocity).all()
                or not torch.allclose(poses[:, 0, :3], root_pose[:, :3], atol=1e-5, rtol=0)
                or not torch.allclose(velocity[:, 0], root_velocity, atol=1e-5, rtol=0)):
            raise ValueError('native body-zero link pose or COM velocity differs from the actual root')
        quaternion = poses[:, :, 3:7]
        norm = quaternion.norm(dim=-1, keepdim=True)
        if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0):
            raise ValueError('native body link orientations require normalized xyzw quaternions')
        if (((quaternion[:, 0] * root_pose[:, 3:7]).sum(-1).abs() - 1).abs().gt(1e-4).any()):
            raise ValueError('native body-zero orientation differs from the actual root frame')
        quaternion = quaternion / norm
        local = self._local_body_com_m
        cross = torch.linalg.cross(quaternion[..., :3], local)
        rotated = local + 2 * (quaternion[..., 3:] * cross + torch.linalg.cross(quaternion[..., :3], cross))
        position = poses[..., :3] + rotated
        if not torch.isfinite(position).all():
            raise ValueError('native body COM position overflowed its physical tensor range')
        result = RigidBodyMotion(poses.detach().clone(), position.detach().clone(), velocity[..., :3].detach().clone(),
                                  velocity[..., 3:].detach().clone(), timestamp, self.binding.native_body_names, tick, epoch)
        self.before_state_read()
        if self.model_epoch != epoch or self.physical_tick != tick:
            raise RuntimeError('body-motion read crossed a physical or mechanical model epoch')
        return result

    @torch.no_grad()
    def measure(self, surfaces: SurfaceSample):
        self.before_state_read()
        if (not isinstance(surfaces, SurfaceSample)
                or tuple(surfaces.body_names) != self.binding.native_body_names
                or not isinstance(surfaces.timestamp_s, torch.Tensor)
                or surfaces.timestamp_s.shape != (self.num_envs,)
                or surfaces.timestamp_s.dtype != torch.float64 or surfaces.timestamp_s.device != self.device
                or not torch.isfinite(surfaces.timestamp_s).all()
                or (surfaces.timestamp_s - self.timestamp_s()).abs().gt(1e-8).any()):
            raise ValueError('surface identity/time must match this native post-physics sample')
        pose, velocity = self._buffer('root_link_pose_w', 7), self._buffer('root_com_vel_w', 6)
        quaternion = pose[:, 3:7]
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        local = self._local_com_m
        cross = torch.linalg.cross(quaternion[:, :3], local)
        offset_world = local + 2 * (quaternion[:, 3:] * cross
                                    + torch.linalg.cross(quaternion[:, :3], cross))
        sample = NativePhysicalSample(
            self._buffer('joint_pos', 18), self._buffer('joint_vel', 18), pose[:, :3], pose[:, 3:7],
            pose[:, :3] + offset_world, velocity[:, :3], velocity[:, 3:6],
            surfaces.terrain_contact_force_world_n, surfaces.ground_at_root_world_z_m,
            surfaces.ground_at_wheel_world_z_m, surfaces.environment_origin_world_m,
            surfaces.wheel_friction, self.timestamp_s(),
        )
        return self.assembler.measure(sample)

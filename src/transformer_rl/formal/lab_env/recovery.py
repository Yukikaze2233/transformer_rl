"""正式 recovery 出生姿态到 LabEnv 的复位适配。

使用已冻结的 AssetResetKinematics / RecoveryPoseCompiler / AssetCollisionGeometry
计算 16 个姿态的闭链、根姿态和碰撞净空；不修改资产、不估计缺失碰撞体。
未接 recovery FSM 时它只负责 reset，不声称任务成功或物理可恢复。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import torch

from ..control.actuation import OutputFeedbackController
from ..control.ownership import ControlSource
from ..control.observation import MechanicalBinding
from ..tasks.geometry import TerrainKind
from ..tasks.recovery import RecoveryPoseCompiler, RecoveryTuning
from ..tasks.reset_collision import AssetCollisionGeometry, FlatSpawnPatch
from ..tasks.reset_kinematics import AssetResetKinematics
from ..tasks.terrain import (
    ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning,
)
from .settings import AssetDescription


@dataclass(frozen=True)
class RecoveryResetConfig:
    """与旧正式 reset 程序一致的显式几何 recipe。"""

    crouched_inner_angle_rad: float = math.radians(50.0)
    asymmetric_leg_pitch_rad: float = 0.25
    legs_above_pitch_rad: float = math.pi
    release_clearance_m: float = 0.04
    release_downward_speed_m_s: float = 0.15
    output_hold_tolerance_rad: float = 1e-6
    loop_tolerance_m: float = 1e-6
    maximum_solver_iterations: int = 80
    posture_segment_angle_rad: float = math.pi / 24
    maximum_posture_segments: int = 32
    spawn_half_xy_m: float = 2.0
    clearance_m: float = 0.002
    wheel_support_tolerance_m: float = 0.001

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or self.spawn_half_xy_m <= self.wheel_support_tolerance_m
                or self.clearance_m < 0):
            raise ValueError("recovery reset 的几何/净空参数必须有限且一致")

    def recovery_tuning(self):
        return RecoveryTuning(
            self.crouched_inner_angle_rad, self.asymmetric_leg_pitch_rad,
            self.legs_above_pitch_rad, self.release_clearance_m,
            self.release_downward_speed_m_s, self.output_hold_tolerance_rad,
            self.loop_tolerance_m, self.maximum_solver_iterations,
            self.posture_segment_angle_rad, self.maximum_posture_segments,
        )


@dataclass(frozen=True)
class RecoveryResetPose:
    root_pose_xyzw: torch.Tensor
    joint_position: torch.Tensor
    root_com_velocity: torch.Tensor
    joint_velocity: torch.Tensor
    posture_id: torch.Tensor
    geometry_ready: torch.Tensor


class _RecoveryPreparation:
    """Move only unrepresentable recovery starts into the existing action box.

    Each explicit waypoint is a script-owned preparation hold for one policy
    interval. It never expands policy targets or supplies PPO/auxiliary credit.
    The 1rad/s reference rate is independent of the unchanged 200Hz motor PD.
    """

    def __init__(self, controller, *, policy_dt_s, timeout_s):
        if (not isinstance(controller, OutputFeedbackController) or policy_dt_s not in (.01, .02)
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('recovery preparation requires the original controller and explicit clock/timeout')
        self.controller, self.dt, self.timeout_s = controller, policy_dt_s, timeout_s
        count, device = controller.num_envs, controller.device
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.elapsed = torch.zeros(count, dtype=torch.float64, device=device)
        self.frames = torch.zeros(count, dtype=torch.long, device=device)
        self._start = torch.zeros(count, 6, device=device)
        self._delta = torch.zeros(count, 4, device=device)
        self._duration = torch.zeros(count, dtype=torch.float64, device=device)
        self._has_active = False

    @torch.no_grad()
    def reset_rows(self, ids, feedback, eligible, now_s):
        hold = self.controller.reset_hold_action(feedback, position_tolerance_rad=1e-6)
        target = self.controller.decode(torch.zeros_like(feedback), feedback, now_s)
        self.active[ids] = eligible[ids] & ~hold.admitted[ids]
        self.elapsed[ids], self.frames[ids] = 0., 0
        self._start[ids] = feedback[ids]
        self._delta[ids] = target.admitted_leg_position_rad[ids] - feedback[ids, :4]
        # One shared interpolation fraction preserves each coaxial pair's
        # relative target between its two admitted endpoint configurations.
        self._duration[ids] = self._delta[ids].abs().amax(-1).double()
        self._has_active = bool(self.active.any())

    @torch.no_grad()
    def targets(self, policy_targets, now_s):
        if not self._has_active:
            return policy_targets
        self.elapsed.add_(self.active.double() * self.dt)
        fraction = (self.elapsed / self._duration.clamp_min(self.dt)).clamp_max(1.)
        reference = self._start.clone()
        reference[:, :4] += fraction[:, None].float() * self._delta
        # Encode a commanded waypoint, not a fabricated feedback measurement.
        hold = self.controller.preparation_hold(reference, position_tolerance_rad=1e-6)
        if not bool(hold.admitted[self.active].all()):
            raise ValueError('recovery preparation waypoint violates the original coupled target domain')
        selected = self.active[:, None]
        raw = self.controller.nominal[:4] + .25 * hold.requested_action[:, :4]
        return replace(
            policy_targets,
            issued_action=torch.where(selected, hold.requested_action, policy_targets.issued_action),
            raw_leg_position_rad=torch.where(selected, raw, policy_targets.raw_leg_position_rad),
            admitted_leg_position_rad=torch.where(selected, hold.leg_position_rad,
                                                  policy_targets.admitted_leg_position_rad),
            wheel_velocity_rad_s=torch.where(selected, 0., policy_targets.wheel_velocity_rad_s),
            pair_projected=torch.where(selected, False, policy_targets.pair_projected),
            common_turns=torch.where(selected, hold.common_turns, policy_targets.common_turns),
            controller_issued_at_s=torch.where(self.active, now_s, policy_targets.controller_issued_at_s),
            control_source=torch.where(self.active, int(ControlSource.PREPARATION_HOLD),
                                       policy_targets.control_source),
        )

    @torch.no_grad()
    def finish_interval(self, feedback):
        handoff = torch.zeros_like(self.active)
        timeout = torch.zeros_like(self.active)
        if self._has_active:
            self.frames.add_(self.active.long())
            hold = self.controller.reset_hold_action(feedback, position_tolerance_rad=1e-6)
            handoff = self.active & hold.admitted & (self.frames >= 5)
            timeout = self.active & ~handoff & (self.elapsed >= self.timeout_s - 1e-9)
            self.active &= ~handoff
            self._has_active = bool(self.active.any())
        return handoff, timeout


class LabRecoveryReset:
    """按 posture id 批量生成碰撞净空合格的 reset pose。"""

    def __init__(self, asset: AssetDescription, num_envs: int, *,
                 config: RecoveryResetConfig | None = None, device="cpu"):
        if not isinstance(asset, AssetDescription) or type(num_envs) is not int or num_envs < 1:
            raise ValueError("recovery reset 需要选中资产和正整数环境数")
        if config is None:
            config = RecoveryResetConfig()
        if not isinstance(config, RecoveryResetConfig):
            raise TypeError("config 必须是 RecoveryResetConfig")
        self.asset, self.num_envs, self.device, self.config = asset, num_envs, torch.device(device), config
        raw = (asset.directory / "manifest.json").read_bytes()
        spec = asset.bound_bytes("model_spec.json", maximum_bytes=256 * 1024)
        if (hashlib.sha256(raw).hexdigest() != asset.manifest_sha256
                or asset.manifest.get("files_sha256", {}).get("model_spec.json")
                != hashlib.sha256(spec).hexdigest()):
            raise ValueError("recovery reset 的 manifest/model_spec 与选中资产不一致")
        self.kinematics = AssetResetKinematics(raw, spec, asset.binding, device="cpu")
        self.geometry = AssetCollisionGeometry.from_asset_directory(self.kinematics, asset.directory)
        self.compiler = RecoveryPoseCompiler(self.kinematics, config.recovery_tuning())
        self._flat_courses = self._build_flat_courses()
        self._local_patch = self._build_local_patch()
        self._zero_yaw_pose = None
        if num_envs >= 16:
            # Formal resets reuse these same 16 requests on translated flat
            # patches. Solve and certify them once, keeping continuous-yaw or
            # elevated-origin requests on the original geometric path.
            table = self.pose(torch.arange(16), torch.arange(16),
                              torch.zeros(16, dtype=torch.float64),
                              torch.zeros(16, 3, dtype=torch.float64))
            if table.root_pose_xyzw[:, :2].eq(0).all():
                self._zero_yaw_pose = table
        fit = json.loads(asset.bound_bytes('fit_10mpa.json', maximum_bytes=64 * 1024))
        nominal_closure = self.kinematics.closure(
            self.kinematics.nominal_tree_position[None]).norm(dim=-1).amax()
        self.receipt = {
            'format': 'selected_mechanical_asset_receipt_v1',
            'manifest_sha256': asset.manifest_sha256,
            'model_spec_sha256': self.kinematics.model_sha256,
            'body_count': len(asset.binding.native_body_names),
            'tree_joint_count': len(asset.binding.tree_joint_names),
            'constraint_count': len(self.kinematics.constraint_names),
            'native_joint_names': list(asset.binding.native_joint_names),
            'native_body_names': list(asset.binding.native_body_names),
            'nominal_closure_max_m': float(nominal_closure),
            'recovery_postures': 16,
            'zero_yaw_pose_cached': self._zero_yaw_pose is not None,
            'source_pins': {name: asset.manifest['files_sha256'][name]
                            for name in ('model_spec.json', 'fit_10mpa.json',
                                         'gas_spring_binding.json', 'mechanical_limits.json',
                                         'robot.urdf', 'robot.usda')},
            'gas_spring': {
                'pressure_mpa': fit['pressure_mpa'],
                'monomial_coefficients_n': list(fit['monomial_coefficients_n']),
                'stroke_m': fit['stroke_m'],
                'slider_damping_source': 'manifest.passive_damping.slider_N_s_per_m',
            },
            'collision_mesh_pins': list(self.geometry.source_mesh_pins),
            'manifest_training_ready': bool(asset.manifest.get('training_ready', False)),
            'manifest_physics_qualified': bool(asset.manifest.get('physics_qualified', False)),
            'physical_qualification': False,
            'scope': 'static source/kinematics/collision identity; native qualification remains separate',
        }

    def _build_flat_courses(self):
        # 每个环境都需要独立、可验证的 flat course；几何只读且不含随机起伏。
        generator = ContinuousTerrainGenerator(17, TerrainTuning(.8, (.2, 1.1), .25))
        request = TerrainRequest(TerrainKind.FLAT, TerrainExtent(
            (-self.config.spawn_half_xy_m, self.config.spawn_half_xy_m),
            (-self.config.spawn_half_xy_m, self.config.spawn_half_xy_m)))
        # 所有行共享同一个只读平地源；每个 env 只改变 origin，不复制 8192 份几何。
        return (generator.build(request),) * self.num_envs

    def _build_local_patch(self):
        origins = torch.zeros(self.num_envs, 3, dtype=torch.float64)
        bounds = torch.tensor([[[-self.config.spawn_half_xy_m, self.config.spawn_half_xy_m],
                                [-self.config.spawn_half_xy_m, self.config.spawn_half_xy_m]]],
                              dtype=torch.float64).repeat(self.num_envs, 1, 1)
        return FlatSpawnPatch.from_courses(self._flat_courses, origins, bounds,
                                           torch.zeros(self.num_envs, dtype=torch.float64))

    def _patch(self, ids, origins):
        # local patch 已在构造时验证；这里只把已证明的平面按实际 env origin 平移到世界系。
        local = self._local_patch
        hashes = tuple(local.course_sha256[index] for index in ids.tolist())
        return FlatSpawnPatch(local.bounds_world_xy_m[ids] + origins[:, :2, None],
                              local.height_world_z_m[ids] + origins[:, 2],
                              hashes, True)

    @torch.no_grad()
    def pose(self, ids, posture_id, yaw_rad, origins_world_m):
        ids = torch.as_tensor(ids, dtype=torch.long, device="cpu")
        posture_id = torch.as_tensor(posture_id, dtype=torch.long, device="cpu")
        yaw_rad = torch.as_tensor(yaw_rad, dtype=torch.float64, device="cpu")
        origins = torch.as_tensor(origins_world_m, dtype=torch.float64, device="cpu")
        n = ids.numel()
        if (ids.ndim != 1 or n < 1 or posture_id.shape != (n,) or yaw_rad.shape != (n,)
                or origins.shape != (n, 3) or ((posture_id < 0) | (posture_id >= 16)).any()
                or ids.unique().numel() != n or (ids < 0).any() or (ids >= self.num_envs).any()
                or not torch.isfinite(yaw_rad).all() or not torch.isfinite(origins).all()):
            raise ValueError("recovery reset 的 ids/posture/yaw/origin 布局不一致")
        cached = getattr(self, '_zero_yaw_pose', None)
        if cached is not None and yaw_rad.eq(0).all() and origins[:, 2].eq(0).all():
            indices = posture_id.to(self.device)
            root = cached.root_pose_xyzw.index_select(0, indices)
            root[:, :2] = origins[:, :2].to(device=self.device, dtype=torch.float32)
            return RecoveryResetPose(
                root, cached.joint_position.index_select(0, indices),
                cached.root_com_velocity.index_select(0, indices),
                cached.joint_velocity.index_select(0, indices), indices.clone(),
                cached.geometry_ready.index_select(0, indices))
        controller = OutputFeedbackController(
            self.kinematics.binding.nominal_policy_position_rad, n, pair_domain=None, device="cpu")
        candidate = self.compiler.compile(posture_id, yaw_rad, controller)
        patch = self._patch(ids, origins)
        placement = self.geometry.place(
            candidate.closure.tree_position, candidate.requested_root_orientation_xyzw,
            origins[:, :2], patch,
            clearance_m=self.config.clearance_m + candidate.release_clearance_m,
            loop_tolerance_m=self.config.loop_tolerance_m,
            wheel_support_tolerance_m=self.config.wheel_support_tolerance_m)
        if not placement.geometry_ready.all():
            raise ValueError("recovery posture 存在闭链/限位/碰撞净空未通过的 row")
        root = placement.root_pose_xyzw.float().to(self.device)
        native_order = [self.kinematics.binding.tree_joint_names.index(name)
                        for name in self.kinematics.binding.native_joint_names]
        joints = placement.tree_position[:, native_order].float()
        # The compiler expresses release velocity in the link body frame.
        # With zero angular velocity, link-origin and CoM velocities agree.
        # Applying it at reset needs no scripted action or policy handoff.
        if candidate.angular_velocity_body_rad_s.any():
            raise ValueError("recovery reset requires the compiler's zero angular velocity")
        rotation = self.geometry._rotation(placement.root_pose_xyzw[:, 3:])
        velocity = torch.zeros(n, 6, dtype=torch.float32, device=self.device)
        velocity[:, :3] = (rotation @ candidate.origin_velocity_body_m_s[..., None]).squeeze(-1).to(
            device=self.device, dtype=torch.float32)
        joint_velocity = torch.zeros_like(joints).to(self.device)
        return RecoveryResetPose(root, joints.to(self.device), velocity, joint_velocity,
                                 posture_id.to(self.device), placement.geometry_ready.to(self.device))


__all__ = ["LabRecoveryReset", "RecoveryResetConfig", "RecoveryResetPose"]

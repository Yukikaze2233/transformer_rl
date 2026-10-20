"""正式 recovery 出生姿态到 LabEnv 的复位适配。

使用已冻结的 AssetResetKinematics / RecoveryPoseCompiler / AssetCollisionGeometry
计算 16 个姿态的闭链、根姿态和碰撞净空；不修改资产、不估计缺失碰撞体。
未接 recovery FSM 时它只负责 reset，不声称任务成功或物理可恢复。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from ..control.actuation import OutputFeedbackController
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
        spec = (asset.directory / "model_spec.json").read_bytes()
        if (hashlib.sha256(raw).hexdigest() != asset.manifest_sha256
                or asset.manifest.get("files_sha256", {}).get("model_spec.json")
                != hashlib.sha256(spec).hexdigest()):
            raise ValueError("recovery reset 的 manifest/model_spec 与选中资产不一致")
        self.kinematics = AssetResetKinematics(raw, spec, asset.binding, device="cpu")
        self.geometry = AssetCollisionGeometry.from_asset_directory(self.kinematics, asset.directory)
        self.compiler = RecoveryPoseCompiler(self.kinematics, config.recovery_tuning())
        self._flat_courses = self._build_flat_courses()
        self._local_patch = self._build_local_patch()

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

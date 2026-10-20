"""把 LabEnv 的物理采样接入正式 PhysicalRewardComposer。

只转换数据，不重写 reward 数学。被动膝边界由边界 producer 提供。
``contact_slip`` 需要真实接触点相对切向速度；当前 LabEnv 没有该测量，
因此不作为可用项，也不从机身速度或轮速伪造。缺失项由 ``missing_terms``
显式记录。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..control.observation import CommandMode, MeasuredState, MechanicalBinding, NativePhysicalSample
from ..tasks.reward import (
    REWARD_COST_TERMS, REWARD_POSITIVE_TERMS, PhysicalRewardComposer, RewardPhase,
    RewardPhysics, RewardReference, RewardResult, RewardTuning,
)


LAB_AVAILABLE_TERMS = tuple(dict.fromkeys(
    name for name in REWARD_POSITIVE_TERMS + REWARD_COST_TERMS
    if name not in ("joint_boundary", "contact_slip")
))


@dataclass(frozen=True)
class FormalRewardConfig:
    """正式 reward 的显式来源与能力声明。"""

    robot_mass_kg: float
    gamma: float = 0.99498743710662
    physics_dt_s: float = 0.005
    available_terms: tuple[str, ...] = LAB_AVAILABLE_TERMS
    normal_load_source: str = "contact_force_world_z_abs"

    @classmethod
    def from_manifest(cls, manifest):
        mass = manifest.get("total_mass_kg") if isinstance(manifest, dict) else None
        if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
            raise ValueError("asset manifest 必须提供有限正 total_mass_kg")
        return cls(float(mass))

    def __post_init__(self):
        if (type(self.robot_mass_kg) not in (int, float) or not math.isfinite(self.robot_mass_kg)
                or self.robot_mass_kg <= 0
                or self.gamma != 0.99498743710662 or self.physics_dt_s not in (0.005, 0.001)
                or not isinstance(self.available_terms, tuple)
                or len(set(self.available_terms)) != len(self.available_terms)
                or any(name not in REWARD_POSITIVE_TERMS + REWARD_COST_TERMS
                       for name in self.available_terms)
                or self.normal_load_source not in ("contact_force_norm", "contact_force_world_z_abs")):
            raise ValueError("正式 reward 配置的 mass/clock/term/source 不一致")


def _rotate(quaternion_xyzw, vector):
    q = quaternion_xyzw / quaternion_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    imaginary, scalar = q[:, :3], q[:, 3:]
    cross = torch.linalg.cross(imaginary, vector)
    return vector + 2 * (scalar * cross + torch.linalg.cross(imaginary, cross))


def _roll_pitch(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    return torch.stack((roll, pitch), -1)


def reward_physics_from_lab(sample: NativePhysicalSample, state: MeasuredState, *,
                            effort_nm: torch.Tensor, episode_id: torch.Tensor,
                            control_epoch: torch.Tensor, policy_owned: torch.Tensor,
                            available_terms: tuple[str, ...],
                            joint_boundary_shortfall_rad: torch.Tensor | None = None,
                            wheel_contact_tangent_speed_m_s: torch.Tensor | None = None,
                            normal_load_source: str = "contact_force_world_z_abs") -> RewardPhysics:
    """构造正式 reward 的物理端点；不推断缺失的 slip/boundary。"""
    if not isinstance(sample, NativePhysicalSample) or not isinstance(state, MeasuredState):
        raise TypeError("正式 reward 需要 native sample 与 measured state")
    n, device = state.height_above_ground_m.shape[0], state.height_above_ground_m.device
    if (effort_nm.shape != (n, 6) or effort_nm.device != device
            or episode_id.shape != (n,) or control_epoch.shape != (n,)
            or policy_owned.shape != (n,) or policy_owned.dtype != torch.bool
            or not isinstance(available_terms, tuple)):
        raise ValueError("正式 reward 的 action/id/availability 张量布局不一致")
    force = sample.terrain_contact_force_world_n
    if normal_load_source == "contact_force_norm":
        body_load = force.norm(dim=-1)
    elif normal_load_source == "contact_force_world_z_abs":
        # 平地/当前 LabEnv 的地面法向是世界 Z；保存按体表分离的法向载荷，
        # 不用三轴合力范数冒充 normal load。
        body_load = force[..., 2].abs()
    else:
        raise ValueError("未知的 LabEnv normal load 来源")
    if not torch.isfinite(body_load).all():
        raise ValueError("接触力净范数必须有限")
    boundary = (torch.zeros(n, 6, device=device) if joint_boundary_shortfall_rad is None
                else joint_boundary_shortfall_rad)
    if boundary.shape != (n, 6) or boundary.device != device or not torch.isfinite(boundary).all():
        raise ValueError("joint boundary shortfall 需要有限 [N,6]")
    if wheel_contact_tangent_speed_m_s is None:
        if "contact_slip" in available_terms:
            raise ValueError("formal reward 缺少轮底切向速度 producer")
        wheel_speed = torch.zeros(n, 2, device=device)
    else:
        wheel_speed = wheel_contact_tangent_speed_m_s
        if (wheel_speed.shape != (n, 2) or wheel_speed.device != device
                or not torch.isfinite(wheel_speed).all() or (wheel_speed < 0).any()):
            raise ValueError("wheel contact tangent speed 需要有限非负 [N,2]")
    return RewardPhysics(
        sample.timestamp_s.clone(), episode_id.clone(), control_epoch.clone(), policy_owned.clone(),
        state.velocity_origin_body_m_s.clone(),
        _rotate(sample.root_link_orientation_xyzw, state.velocity_origin_body_m_s),
        sample.root_link_position_world_m[:, 2].clone(),
        state.height_above_ground_m.clone(), _roll_pitch(sample.root_link_orientation_xyzw),
        state.gravity_body.clone(), state.angular_velocity_body_rad_s.clone(),
        state.policy_velocity_rad_s.clone(), effort_nm.clone(), body_load,
        wheel_speed.clone(), boundary,
        available_terms=available_terms,
    )


def reward_reference_from_command(command, mode, *, phase=None, jump_height_world_z_m=None,
                                  jump_velocity_world_z_m_s=None, target_roll_pitch_rad=None):
    """普通/坡面任务的 reward-only 参考；jump 参考必须显式给出。"""
    n, device = command.shape[0], command.device
    if command.shape != (n, 4) or mode.shape != (n,) or mode.dtype != torch.long:
        raise ValueError("command/mode 布局不一致")
    zeros = torch.zeros(n, device=device)
    return RewardReference(
        command.clone(), mode.clone(),
        (torch.full((n,), int(RewardPhase.SUPPORT), dtype=torch.long, device=device)
         if phase is None else phase.clone()),
        (zeros.clone() if jump_height_world_z_m is None else jump_height_world_z_m.clone()),
        (zeros.clone() if jump_velocity_world_z_m_s is None else jump_velocity_world_z_m_s.clone()),
        (torch.zeros(n, 2, device=device) if target_roll_pitch_rad is None else target_roll_pitch_rad.clone()),
    )


class LabFormalReward:
    """正式 reward composer 的 LabEnv 适配器；每个策略区间 begin→observe×2→finish。"""

    def __init__(self, binding: MechanicalBinding, num_envs: int, config: FormalRewardConfig,
                 *, device="cpu", boundary_geometry=None):
        if not isinstance(binding, MechanicalBinding) or type(num_envs) is not int or num_envs < 1:
            raise ValueError("正式 reward 需要 binding 和正整数环境数")
        if not isinstance(config, FormalRewardConfig):
            raise TypeError("config 必须是 FormalRewardConfig")
        tuning = RewardTuning(config.robot_mass_kg, gamma=config.gamma,
                              policy_dt_s=0.01, gravity_m_s2=9.81)
        if "joint_boundary" in config.available_terms and boundary_geometry is None:
            raise ValueError("formal reward 请求 joint_boundary 但没有提供 LegBoundaryGeometry")
        self.config, self.binding, self.boundary_geometry = config, binding, boundary_geometry
        self.composer = PhysicalRewardComposer(
            binding, num_envs, tuning, device=device, physics_dt_s=config.physics_dt_s)
        self.num_envs, self.device = num_envs, torch.device(device)

    @property
    def tuning(self):
        return self.composer.tuning

    @property
    def missing_terms(self):
        return tuple(name for name in REWARD_POSITIVE_TERMS + REWARD_COST_TERMS
                     if name not in self.config.available_terms)

    def reset(self, mask, now_s, episode_id, control_epoch):
        self.composer.reset(mask, now_s, episode_id, control_epoch)

    def begin(self, issued_action, initial_policy_velocity_rad_s, *, now_s, episode_id,
              control_epoch, command, mode, policy_owned=None):
        if not isinstance(command, torch.Tensor) or command.shape != (self.num_envs, 4):
            raise ValueError("command 需要 [N,4]")
        owned = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                 if policy_owned is None else policy_owned)
        self.composer.begin(issued_action, initial_policy_velocity_rad_s, now_s=now_s,
                            episode_id=episode_id, control_epoch=control_epoch,
                            policy_owned=owned, admitted_body_command=command,
                            command_mode=mode)

    def observe(self, sample, state, *, effort_nm, episode_id, control_epoch,
                command, mode, policy_owned=None, phase=None, jump_height_world_z_m=None,
                jump_velocity_world_z_m_s=None, target_roll_pitch_rad=None,
                joint_boundary_shortfall_rad=None, wheel_contact_tangent_speed_m_s=None):
        owned = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                 if policy_owned is None else policy_owned)
        if "joint_boundary" in self.config.available_terms and joint_boundary_shortfall_rad is None:
            raise ValueError("formal reward 缺少 joint_boundary shortfall producer")
        if "contact_slip" in self.config.available_terms and wheel_contact_tangent_speed_m_s is None:
            raise ValueError("formal reward 缺少轮底切向速度 producer")
        physics = reward_physics_from_lab(
            sample, state, effort_nm=effort_nm, episode_id=episode_id,
            control_epoch=control_epoch, policy_owned=owned,
            available_terms=self.config.available_terms,
            joint_boundary_shortfall_rad=joint_boundary_shortfall_rad,
            wheel_contact_tangent_speed_m_s=wheel_contact_tangent_speed_m_s,
            normal_load_source=self.config.normal_load_source)
        reference = reward_reference_from_command(
            command, mode, phase=phase, jump_height_world_z_m=jump_height_world_z_m,
            jump_velocity_world_z_m_s=jump_velocity_world_z_m_s,
            target_roll_pitch_rad=target_roll_pitch_rad)
        self.composer.observe(physics, reference)

    def finish(self, *, terminated, truncated, physical_success_request_id=None,
               entire_skill_policy_owned=None) -> RewardResult:
        success = (torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
                   if physical_success_request_id is None else physical_success_request_id)
        owned = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                 if entire_skill_policy_owned is None else entire_skill_policy_owned)
        return self.composer.finish(
            terminated=terminated, truncated=truncated,
            physical_success_request_id=success,
            entire_skill_policy_owned=owned)


__all__ = [
    "FormalRewardConfig", "LAB_AVAILABLE_TERMS", "LabFormalReward",
    "reward_physics_from_lab", "reward_reference_from_command",
]

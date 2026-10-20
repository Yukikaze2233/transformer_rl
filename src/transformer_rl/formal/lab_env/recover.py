"""华南虎式 recover task：随机翻倒 reset、落地 action hold、逐帧 success reward。

这不是 recovery FSM；它是一个独立 task：reset 后先让机器人自然落地，
再由 policy 接管，episode 固定 5s。actor 仍使用 35D flags。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _pair(name, value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or value[0] > value[1]):
        raise ValueError(f"{name} 需要有限有序上下界")
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class RecoverTaskSpec:
    policy_dt_s: float = .01
    episode_length_s: float = 5.0
    action_hold_s: float = .5
    height_command_m: float = .25
    roll_range: tuple[float, float] = (-.5, .5)
    pitch_range: tuple[float, float] = (-.5, .5)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)
    root_height_range_m: tuple[float, float] = (.32, .38)
    lin_vel_ranges: tuple[tuple[float, float], ...] = ((-.1, .1), (-.05, .05), (-.1, .1))
    ang_vel_ranges: tuple[tuple[float, float], ...] = ((-.3, .3), (-.3, .3), (-.5, .5))
    joint_position_jitter_rad: float = .1
    joint_velocity_range_rad_s: tuple[float, float] = (-.5, .5)
    wheel_force_threshold_n: float = 5.0
    forbidden_body_load_n: float = 2.0
    success_height_margin_m: float = .05
    success_tilt_rad: float = math.radians(10.0)
    success_bonus: float = 1.0
    hold_reward_zero: bool = True

    def __post_init__(self):
        for name in ("roll_range", "pitch_range", "yaw_range", "root_height_range_m",
                     "joint_velocity_range_rad_s"):
            object.__setattr__(self, name, _pair(name, getattr(self, name)))
        for name in ("lin_vel_ranges", "ang_vel_ranges"):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or len(values) != 3):
                raise ValueError(f"{name} 需要 3 个轴范围")
            object.__setattr__(self, name, tuple(_pair(name, value) for value in values))
        if (type(self.policy_dt_s) not in (int, float) or not math.isfinite(self.policy_dt_s)
                or self.policy_dt_s <= 0 or type(self.episode_length_s) not in (int, float)
                or not math.isfinite(self.episode_length_s) or self.episode_length_s <= 0
                or type(self.action_hold_s) not in (int, float) or not math.isfinite(self.action_hold_s)
                or self.action_hold_s < 0 or self.action_hold_s >= self.episode_length_s
                or type(self.height_command_m) not in (int, float)
                or not math.isfinite(self.height_command_m) or self.height_command_m <= 0
                or type(self.joint_position_jitter_rad) not in (int, float)
                or not math.isfinite(self.joint_position_jitter_rad) or self.joint_position_jitter_rad < 0
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in
                       (self.wheel_force_threshold_n, self.forbidden_body_load_n,
                        self.success_height_margin_m, self.success_tilt_rad, self.success_bonus))
                or not 0 <= self.success_tilt_rad < math.pi / 2):
            raise ValueError("recover task 的时钟/阈值/奖励参数不一致")


@dataclass(frozen=True)
class RecoverResetPlan:
    root_pose_xyzw: torch.Tensor
    root_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


def _quaternion(roll, pitch, yaw):
    sr, cr = (roll / 2).sin(), (roll / 2).cos()
    sp, cp = (pitch / 2).sin(), (pitch / 2).cos()
    sy, cy = (yaw / 2).sin(), (yaw / 2).cos()
    return torch.stack((sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy), -1)


class LabRecoverTask:
    """只管理 recover episode 的随机 reset、hold 和 success，不控制策略。"""

    def __init__(self, num_envs: int, spec: RecoverTaskSpec, *, device="cpu", seed: int = 0):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(spec, RecoverTaskSpec):
            raise ValueError("recover task 需要正整数环境和 RecoverTaskSpec")
        if type(seed) is not int or seed < 0:
            raise ValueError("recover task seed 必须是非负整数")
        self.num_envs, self.spec, self.device = num_envs, spec, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.hold_steps = max(1, round(spec.action_hold_s / spec.policy_dt_s))
        self.hold_remaining = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def _uniform(self, count, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def reset(self, ids, origins_world_m, default_joint_position):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        origins = torch.as_tensor(origins_world_m, dtype=torch.float32, device=self.device)
        n = ids.numel()
        if (ids.ndim != 1 or n < 1 or origins.shape != (n, 3)
                or default_joint_position.shape != (n, 18)
                or not torch.isfinite(origins).all() or not torch.isfinite(default_joint_position).all()):
            raise ValueError("recover reset 的 ids/origins/default joint 布局不一致")
        roll = self._uniform(n, self.spec.roll_range)
        pitch = self._uniform(n, self.spec.pitch_range)
        yaw = self._uniform(n, self.spec.yaw_range)
        root = torch.zeros(n, 7, dtype=torch.float32, device=self.device)
        root[:, :2] = origins[:, :2]
        root[:, 2] = self._uniform(n, self.spec.root_height_range_m)
        root[:, 3:] = _quaternion(roll, pitch, yaw).float()
        velocity = torch.stack(tuple(self._uniform(n, bounds) for bounds in
                                    (*self.spec.lin_vel_ranges, *self.spec.ang_vel_ranges)), -1)
        joint_position = default_joint_position.clone()
        joint_position += (torch.rand_like(joint_position) * 2 - 1) * self.spec.joint_position_jitter_rad
        joint_velocity = self._uniform(n, self.spec.joint_velocity_range_rad_s)[:, None].expand(-1, 18).clone()
        self.hold_remaining[ids] = self.hold_steps
        self.success[ids] = False
        self.active[ids] = True
        return RecoverResetPlan(root, velocity, joint_position, joint_velocity)

    @property
    def hold_mask(self):
        return self.active & (self.hold_remaining > 0)

    def advance(self):
        self.hold_remaining.sub_((self.hold_remaining > 0).long()).clamp_min_(0)
        return self.hold_mask

    def task_mode(self, recovery_mode: int, normal_mode: int):
        return torch.where(self.hold_mask, normal_mode, recovery_mode)

    def success_mask(self, *, wheel_force_n, base_load_n, height_m, gravity_body):
        n = self.num_envs
        for name, value, shape in (("wheel force", wheel_force_n, (n, 2)),
                                   ("base load", base_load_n, (n,)),
                                   ("height", height_m, (n,)),
                                   ("gravity", gravity_body, (n, 3))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError(f"recover {name} 布局不一致")
        upright = gravity_body[:, :2].norm(dim=-1) <= math.sin(self.spec.success_tilt_rad) + 1e-9
        ok = (wheel_force_n > self.spec.wheel_force_threshold_n).all(-1)
        ok &= base_load_n <= self.spec.forbidden_body_load_n
        ok &= height_m >= self.spec.height_command_m - self.spec.success_height_margin_m
        ok &= upright
        ok &= self.active & ~self.hold_mask
        self.success.copy_(ok)
        return ok

    def reward_bonus(self):
        return self.success.to(torch.float32) * self.spec.success_bonus


__all__ = ["LabRecoverTask", "RecoverResetPlan", "RecoverTaskSpec"]

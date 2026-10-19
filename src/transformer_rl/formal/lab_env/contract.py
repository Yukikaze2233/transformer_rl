"""纯 PyTorch 任务契约：与 Isaac 无关，便于单测和复用。

本模块只定义 35D/81D/6D 接口、5 帧历史、平地奖励与终止。
仿真步进、传感器和场景由 lab_env.env 负责，避免任务公式泄漏到适配层。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


ACTOR_DIM = 35
CRITIC_DIM = 81
ACTION_DIM = 6
HISTORY_LENGTH = 5

# 策略/物理时钟来自正式契约，不在任务模块里另立默认值。
POLICY_DT_S = 0.01
PHYSICS_DT_S = 0.005


@dataclass(frozen=True)
class FlatTaskSpec:
    """平地 M1 的显式阈值与奖励尺度；量纲随字段名。"""

    action_bounds: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 9.0, 9.0)
    nominal_height_m: float = 0.305
    minimum_height_m: float = 0.08
    failure_tilt_rad: float = 1.2
    base_contact_force_n: float = 30.0
    forward_sigma_m_s: float = 0.35
    yaw_sigma_rad_s: float = 0.5
    height_sigma_m: float = 0.025
    forward_weight: float = 3.0
    yaw_weight: float = 2.0
    height_weight: float = 2.0
    lateral_cost: float = 0.5
    vertical_cost: float = 0.2
    action_rate_cost: float = 0.01
    effort_cost: float = 1e-4

    def __post_init__(self):
        if (not isinstance(self.action_bounds, (tuple, list)) or len(self.action_bounds) != ACTION_DIM
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in self.action_bounds)):
            raise ValueError("action_bounds 需要 6 个有限正数")
        for name in ("nominal_height_m", "minimum_height_m", "failure_tilt_rad",
                     "base_contact_force_n", "forward_sigma_m_s", "yaw_sigma_rad_s",
                     "height_sigma_m", "forward_weight", "yaw_weight", "height_weight",
                     "lateral_cost", "vertical_cost", "action_rate_cost", "effort_cost"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值")
        if not 0 < self.minimum_height_m < self.nominal_height_m:
            raise ValueError("minimum_height_m 必须小于 nominal_height_m")
        if not 0 < self.failure_tilt_rad < math.pi:
            raise ValueError("failure_tilt_rad 必须在 (0, pi)")


class HistoryBuffer:
    """固定 [N, 5, 35] 的历史；reset 行用当前帧填充整段。"""

    def __init__(self, num_envs: int, *, length: int = HISTORY_LENGTH,
                 frame_dim: int = ACTOR_DIM, device="cpu"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs 必须是正整数")
        if type(length) is not int or length < 1:
            raise ValueError("length 必须是正整数")
        if type(frame_dim) is not int or frame_dim < 1:
            raise ValueError("frame_dim 必须是正整数")
        self.num_envs, self.length, self.frame_dim = num_envs, length, frame_dim
        self.device = torch.device(device)
        self.frames = torch.zeros(num_envs, length, frame_dim, device=self.device)

    def _frame(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, self.frame_dim)
                or value.device != self.device or value.dtype != torch.float32):
            raise ValueError("frame 需要 [N, frame_dim] float32 且同 device")
        if not torch.isfinite(value).all():
            raise ValueError("frame 必须有限")
        return value

    def reset(self, mask, frame):
        frame = self._frame(frame)
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError("reset mask 需要 [N] bool 且同 device")
        if mask.any():
            self.frames[mask] = frame[mask, None].expand(-1, self.length, -1)
        return self.frames

    def append(self, frame, reset):
        frame = self._frame(frame)
        if (not isinstance(reset, torch.Tensor) or reset.shape != (self.num_envs,)
                or reset.dtype != torch.bool or reset.device != self.device):
            raise ValueError("reset 需要 [N] bool 且同 device")
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = frame
        if reset.any():
            self.frames[reset] = frame[reset, None].expand(-1, self.length, -1)
        return self.frames

    def owned(self):
        return self.frames.detach().clone()


def _finite(name, value, shape):
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError(f"{name} 需要有限浮点 {tuple(shape)}")
    return value


def flat_reward(*, command, velocity_body, height_m, gravity_body,
                action, previous_action, effort, spec: FlatTaskSpec, dt: float):
    """平地 M1 奖励密度，返回值已乘 dt（单位：无量纲/策略步）。

    command=[vx, yaw, height]；velocity_body=[vx, vy, vz, wz]；
    gravity_body 为单位重力在机体系的投影；action/effort 为策略顺序。
    """
    n = command.shape[0] if isinstance(command, torch.Tensor) else -1
    command = _finite("command", command, (n, 3))
    velocity_body = _finite("velocity_body", velocity_body, (n, 4))
    gravity_body = _finite("gravity_body", gravity_body, (n, 3))
    height_m = _finite("height_m", height_m, (n,))
    action = _finite("action", action, (n, ACTION_DIM))
    previous_action = _finite("previous_action", previous_action, (n, ACTION_DIM))
    effort = _finite("effort", effort, (n, ACTION_DIM))
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt 必须是有限正数")

    vx, vy, vz, wz = velocity_body.unbind(-1)
    cmd_vx, cmd_yaw, cmd_height = command.unbind(-1)
    forward = spec.forward_weight * torch.exp(-((vx - cmd_vx) / spec.forward_sigma_m_s).square())
    yaw = spec.yaw_weight * torch.exp(-((wz - cmd_yaw) / spec.yaw_sigma_rad_s).square())
    height = spec.height_weight * torch.exp(-((height_m - cmd_height) / spec.height_sigma_m).square())
    upright = (-gravity_body[:, 2]).clamp(0, 1)
    cost = (spec.lateral_cost * (vy / 0.25).square()
            + spec.vertical_cost * (vz / 0.3).square()
            + spec.action_rate_cost * (action - previous_action).square().mean(-1)
            + spec.effort_cost * effort.square().mean(-1))
    reward = dt * (forward + yaw + height + upright - cost)
    return reward, {"forward": forward, "yaw": yaw, "height": height,
                    "upright": upright, "cost": cost}


def flat_termination(*, height_m, gravity_body, base_contact_force_n, spec: FlatTaskSpec):
    """返回 (terminated, tilt_rad, base_contact)。"""
    n = height_m.shape[0] if isinstance(height_m, torch.Tensor) else -1
    height_m = _finite("height_m", height_m, (n,))
    gravity_body = _finite("gravity_body", gravity_body, (n, 3))
    base_contact_force_n = _finite("base_contact_force_n", base_contact_force_n, (n,))
    tilt = torch.atan2(gravity_body[:, :2].norm(dim=-1), -gravity_body[:, 2])
    contact = base_contact_force_n > spec.base_contact_force_n
    terminated = (height_m < spec.minimum_height_m) | (tilt > spec.failure_tilt_rad) | contact
    return terminated, tilt, contact

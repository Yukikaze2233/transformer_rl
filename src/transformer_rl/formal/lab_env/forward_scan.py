"""前向地面高度差到台阶高度的显式转换。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ForwardScanConfig:
    min_step_m: float = .005
    max_step_m: float = .20
    wall_step_m: float = .20
    hysteresis_m: float = .002

    def __post_init__(self):
        values = (self.min_step_m, self.max_step_m, self.wall_step_m, self.hysteresis_m)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or self.min_step_m < 0 or self.max_step_m <= self.min_step_m
                or self.wall_step_m < self.max_step_m or self.hysteresis_m < 0):
            raise ValueError("前向扫描阈值不一致")


@dataclass(frozen=True)
class ForwardScanResult:
    step_height_m: torch.Tensor
    is_step: torch.Tensor
    is_wall: torch.Tensor
    valid: torch.Tensor


class ForwardStepDetector:
    """把当前/前方地面高度差转成非负台阶高度；不做硬件触发。"""

    def __init__(self, num_envs: int, config: ForwardScanConfig = ForwardScanConfig(), *, device="cpu"):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, ForwardScanConfig):
            raise ValueError("前向扫描需要正整数环境数和配置")
        self.num_envs, self.config, self.device = num_envs, config, torch.device(device)
        self._previous = torch.zeros(num_envs, device=self.device)

    def reset(self, ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if ids is None
               else torch.as_tensor(ids, dtype=torch.long, device=self.device))
        if ids.ndim != 1 or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("前向扫描 reset ids 超出范围")
        self._previous[ids] = 0

    def update(self, current_ground_z_m, forward_ground_z_m):
        current = torch.as_tensor(current_ground_z_m, dtype=torch.float32, device=self.device)
        forward = torch.as_tensor(forward_ground_z_m, dtype=torch.float32, device=self.device)
        if (current.shape != (self.num_envs,) or forward.shape != (self.num_envs,)
                or not torch.isfinite(current).all() or not torch.isfinite(forward).all()):
            raise ValueError("前向地面高度需要有限 [N]")
        rise = (forward - current).clamp_min(0)
        valid = torch.isfinite(rise) & (rise >= self.config.min_step_m - self.config.hysteresis_m)
        is_wall = rise > self.config.wall_step_m + self.config.hysteresis_m
        is_step = valid & ~is_wall & (rise <= self.config.max_step_m + self.config.hysteresis_m)
        return ForwardScanResult(rise, is_step, is_wall, valid)

    @property
    def previous_ground_z_m(self):
        return self._previous.clone()


__all__ = ["ForwardScanConfig", "ForwardScanResult", "ForwardStepDetector"]

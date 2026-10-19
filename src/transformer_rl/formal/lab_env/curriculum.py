"""M1 平地命令课程：只采样 vx/yaw/height，不做自动晋级。

terrain/recovery/jump 课程需要各自的 native task，不在 M1 里伪造。
stage 由调用方显式设置，保持“人工评估、不自动晋级”的约束。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CommandRanges:
    vx_m_s: tuple[float, float]
    yaw_rad_s: tuple[float, float]
    height_m: tuple[float, float]

    def __post_init__(self):
        for name in ("vx_m_s", "yaw_rad_s", "height_m"):
            value = getattr(self, name)
            if (len(value) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or value[0] > value[1]):
                raise ValueError(f"{name} 需要两个有限有序边界")


# 只增加可跟踪范围；没有 terrain/recovery/jump。
M1_STAGES = (
    CommandRanges((0.0, 0.0), (0.0, 0.0), (0.29, 0.32)),
    CommandRanges((0.2, 0.8), (0.0, 0.0), (0.28, 0.32)),
    CommandRanges((-0.5, 1.2), (-0.5, 0.5), (0.26, 0.34)),
    CommandRanges((-1.0, 1.5), (-1.5, 1.5), (0.24, 0.36)),
)


class LabCommandCurriculum:
    """按 stage 均匀采样；reset 行才重采样，命令在回合内保持。"""

    def __init__(self, num_envs: int, device="cuda:0", *, stage: int = 0, seed: int = 0):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs 必须是正整数")
        if type(stage) is not int or not 0 <= stage < len(M1_STAGES):
            raise ValueError("stage 超出 M1 command 课程范围")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed 必须是非负整数")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.stage = stage
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @property
    def ranges(self) -> CommandRanges:
        return M1_STAGES[self.stage]

    def sample(self, env_ids) -> torch.Tensor:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = ids.numel()
        ranges = self.ranges
        vx = torch.rand(count, device=self.device, generator=self.generator)
        yaw = torch.rand(count, device=self.device, generator=self.generator)
        height = torch.rand(count, device=self.device, generator=self.generator)
        vx = ranges.vx_m_s[0] + vx * (ranges.vx_m_s[1] - ranges.vx_m_s[0])
        yaw = ranges.yaw_rad_s[0] + yaw * (ranges.yaw_rad_s[1] - ranges.yaw_rad_s[0])
        height = ranges.height_m[0] + height * (ranges.height_m[1] - ranges.height_m[0])
        return torch.stack((vx, torch.zeros_like(vx), yaw, height), -1)

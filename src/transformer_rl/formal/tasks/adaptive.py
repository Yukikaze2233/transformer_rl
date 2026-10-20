"""地形等级与指令包络的自适应课程契约。

两类课程都只产生调度状态，不修改策略、物理或奖励。地形等级按环境 reset
更新；指令包络按最近窗口的成功/失败比例更新。调用方负责提供真实测量。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _finite(name, value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    return float(value)


@dataclass(frozen=True)
class TerrainLevelConfig:
    max_level: int = 9
    initial_max_level: int = 5
    tile_length_m: float = 8.0
    advance_fraction: float = 0.25
    demote_tracking_fraction: float = 0.4

    def __post_init__(self):
        for name in ("tile_length_m", "advance_fraction", "demote_tracking_fraction"):
            _finite(name, getattr(self, name))
        if (type(self.max_level) is not int or type(self.initial_max_level) is not int
                or not 0 <= self.initial_max_level <= self.max_level
                or self.tile_length_m <= 0 or not 0 < self.advance_fraction <= 1
                or not 0 <= self.demote_tracking_fraction <= 1):
            raise ValueError("地形等级配置超出范围")


@dataclass(frozen=True)
class TerrainLevelUpdate:
    levels: torch.Tensor
    moved_up: torch.Tensor
    moved_down: torch.Tensor
    reached_top: torch.Tensor


class TerrainLevelTracker:
    """逐环境地形等级；只在 reset 行按距离和跟踪比例升降。"""

    def __init__(self, num_envs: int, *, seed: int = 0, device="cpu",
                 config: TerrainLevelConfig = TerrainLevelConfig()):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("地形等级需要正整数环境数")
        if type(seed) is not int or seed < 0:
            raise ValueError("地形等级 seed 必须是非负整数")
        if not isinstance(config, TerrainLevelConfig):
            raise TypeError("config 必须是 TerrainLevelConfig")
        self.num_envs, self.config, self.device = num_envs, config, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.levels = torch.randint(
            0, config.initial_max_level + 1, (num_envs,), device=self.device,
            generator=self.generator, dtype=torch.long)

    def reset(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or ids.numel() < 1 or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("地形等级 reset ids 超出范围")
        self.levels[ids] = torch.randint(
            0, self.config.initial_max_level + 1, (ids.numel(),), device=self.device,
            generator=self.generator, dtype=torch.long)

    def update(self, reset, distance_m, tracking_fraction):
        reset = torch.as_tensor(reset, dtype=torch.bool, device=self.device)
        distance = torch.as_tensor(distance_m, dtype=torch.float32, device=self.device)
        tracking = torch.as_tensor(tracking_fraction, dtype=torch.float32, device=self.device)
        if (reset.shape != (self.num_envs,) or distance.shape != (self.num_envs,)
                or tracking.shape != (self.num_envs,)
                or not torch.isfinite(distance).all() or not torch.isfinite(tracking).all()
                or (distance < 0).any() or ((tracking < 0) | (tracking > 1)).any()):
            raise ValueError("地形等级输入布局或数值超出范围")
        moved_up = reset & (distance > self.config.tile_length_m * self.config.advance_fraction)
        moved_down = reset & ~moved_up & (tracking < self.config.demote_tracking_fraction)
        next_levels = (self.levels + moved_up.long() - moved_down.long()).clamp_(0, self.config.max_level)
        reached_top = reset & (next_levels >= self.config.max_level)
        next_levels = torch.where(
            reached_top,
            torch.randint(0, self.config.max_level + 1, (self.num_envs,), device=self.device,
                          generator=self.generator, dtype=torch.long),
            next_levels,
        ).clamp_(0, self.config.max_level)
        self.levels = torch.where(reset, next_levels, self.levels)
        return TerrainLevelUpdate(self.levels.clone(), moved_up, moved_down, reached_top)

    def state_dict(self):
        return {'levels': self.levels.detach().cpu().clone(),
                'rng': self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'levels', 'rng'}
                or not isinstance(state['levels'], torch.Tensor)
                or state['levels'].shape != (self.num_envs,)
                or state['levels'].dtype != torch.long
                or (state['levels'] < 0).any() or (state['levels'] > self.config.max_level).any()
                or not isinstance(state['rng'], torch.Tensor)):
            raise ValueError("地形等级 checkpoint 不一致")
        self.levels = state['levels'].to(self.device).clone()
        self.generator.set_state(state['rng'].detach().cpu().clone())


@dataclass(frozen=True)
class CommandRangeConfig:
    max_forward_m_s: float = 2.7
    max_yaw_rad_s: float = 2 * math.pi
    initial_forward_m_s: float = 1.5
    initial_yaw_rad_s: float = 1.5
    min_forward_m_s: float = 0.5
    min_yaw_rad_s: float = 0.3
    forward_step_m_s: float = 0.1
    yaw_step_rad_s: float = 0.2
    success_fraction: float = 0.65
    failure_fraction: float = 0.35

    def __post_init__(self):
        for name in ("max_forward_m_s", "max_yaw_rad_s", "initial_forward_m_s",
                     "initial_yaw_rad_s", "min_forward_m_s", "min_yaw_rad_s",
                     "forward_step_m_s", "yaw_step_rad_s", "success_fraction",
                     "failure_fraction"):
            _finite(name, getattr(self, name))
        if (not 0 < self.min_forward_m_s <= self.initial_forward_m_s <= self.max_forward_m_s
                or not 0 < self.min_yaw_rad_s <= self.initial_yaw_rad_s <= self.max_yaw_rad_s
                or self.forward_step_m_s <= 0 or self.yaw_step_rad_s <= 0
                or not 0 < self.success_fraction <= 1
                or not 0 < self.failure_fraction <= 1
                or self.success_fraction + self.failure_fraction > 1):
            raise ValueError("指令包络配置超出范围")


@dataclass(frozen=True)
class CommandRangeState:
    forward_m_s: tuple[float, float]
    yaw_rad_s: tuple[float, float]
    expanded: bool
    shrunk: bool


class CommandRangeCurriculum:
    """按成功/失败比例扩展或收缩全局指令包络，不采样具体命令。"""

    def __init__(self, config: CommandRangeConfig = CommandRangeConfig()):
        if not isinstance(config, CommandRangeConfig):
            raise TypeError("config 必须是 CommandRangeConfig")
        self.config = config
        self.forward = [-config.initial_forward_m_s, config.initial_forward_m_s]
        self.yaw = [-config.initial_yaw_rad_s, config.initial_yaw_rad_s]

    def update(self, success_fraction: float, failure_fraction: float):
        success = _finite("success_fraction", success_fraction)
        failure = _finite("failure_fraction", failure_fraction)
        if not 0 <= success <= 1 or not 0 <= failure <= 1 or success + failure > 1:
            raise ValueError("成功/失败比例必须位于 [0,1] 且不重叠")
        expanded = success >= self.config.success_fraction
        shrunk = failure >= self.config.failure_fraction
        if expanded:
            self.forward[0] = max(-self.config.max_forward_m_s,
                                  self.forward[0] - self.config.forward_step_m_s)
            self.forward[1] = min(self.config.max_forward_m_s,
                                  self.forward[1] + self.config.forward_step_m_s)
            self.yaw[0] = max(-self.config.max_yaw_rad_s,
                              self.yaw[0] - self.config.yaw_step_rad_s)
            self.yaw[1] = min(self.config.max_yaw_rad_s,
                              self.yaw[1] + self.config.yaw_step_rad_s)
        elif shrunk:
            self.forward[0] = min(-self.config.min_forward_m_s,
                                  self.forward[0] + 2 * self.config.forward_step_m_s)
            self.forward[1] = max(self.config.min_forward_m_s,
                                  self.forward[1] - 2 * self.config.forward_step_m_s)
            self.yaw[0] = min(-self.config.min_yaw_rad_s,
                              self.yaw[0] + 2 * self.config.yaw_step_rad_s)
            self.yaw[1] = max(self.config.min_yaw_rad_s,
                              self.yaw[1] - 2 * self.config.yaw_step_rad_s)
        return CommandRangeState(tuple(self.forward), tuple(self.yaw), expanded, shrunk)

    @property
    def state(self):
        return CommandRangeState(tuple(self.forward), tuple(self.yaw), False, False)


__all__ = [
    "CommandRangeConfig", "CommandRangeCurriculum", "CommandRangeState",
    "TerrainLevelConfig", "TerrainLevelTracker", "TerrainLevelUpdate",
]

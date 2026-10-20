"""台阶课程分类与双级门控。

本模块只定义高度/深度和成功门控，不生成碰撞几何，也不控制策略。实际
场地资产由环境层提供；这里不假设资产文件名或网格实现。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math


class StairClass(IntEnum):
    CURB = 0
    LOW_STEP = 1
    MID_STEP = 2
    HIGH_STEP = 3
    SINGLE_LEDGE = 4
    DOUBLE_LEDGE = 5


@dataclass(frozen=True)
class StairTaskSpec:
    kind: StairClass
    rise_m: float
    platform_depth_m: float
    step_count: int
    body_bump: bool


@dataclass(frozen=True)
class StairCourseSpec:
    curb_max_m: float = .04
    low_step_max_m: float = .08
    mid_step_max_m: float = .14
    high_step_max_m: float = .18
    single_ledge_rise_m: float = .20
    double_ledge_rise_m: float = .20
    platform_depth_m: float = .80
    body_bump_min_m: float = .15
    double_unlock_success_fraction: float = .80
    double_unlock_min_attempts: int = 64

    def __post_init__(self):
        values = (self.curb_max_m, self.low_step_max_m, self.mid_step_max_m,
                  self.high_step_max_m, self.single_ledge_rise_m,
                  self.double_ledge_rise_m, self.platform_depth_m,
                  self.body_bump_min_m, self.double_unlock_success_fraction)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 <= self.curb_max_m < self.low_step_max_m < self.mid_step_max_m
                < self.high_step_max_m < self.single_ledge_rise_m
                or self.double_ledge_rise_m <= 0
                or not 0 <= self.body_bump_min_m <= self.high_step_max_m
                or self.platform_depth_m <= 0
                or not 0 < self.double_unlock_success_fraction <= 1
                or type(self.double_unlock_min_attempts) is not int
                or self.double_unlock_min_attempts < 1):
            raise ValueError("台阶课程阈值不一致")

    def classify(self, rise_m: float) -> StairClass:
        if type(rise_m) not in (int, float) or not math.isfinite(rise_m) or rise_m < 0:
            raise ValueError("台阶高度必须是非负有限值")
        if rise_m <= self.curb_max_m:
            return StairClass.CURB
        if rise_m <= self.low_step_max_m:
            return StairClass.LOW_STEP
        if rise_m <= self.mid_step_max_m:
            return StairClass.MID_STEP
        if rise_m <= self.high_step_max_m:
            return StairClass.HIGH_STEP
        if rise_m <= self.single_ledge_rise_m:
            return StairClass.SINGLE_LEDGE
        raise ValueError("台阶高度超过单级上限")

    def task(self, kind: StairClass) -> StairTaskSpec:
        if not isinstance(kind, StairClass):
            raise TypeError("kind 必须是 StairClass")
        heights = {
            StairClass.CURB: self.curb_max_m,
            StairClass.LOW_STEP: self.low_step_max_m,
            StairClass.MID_STEP: self.mid_step_max_m,
            StairClass.HIGH_STEP: self.high_step_max_m,
            StairClass.SINGLE_LEDGE: self.single_ledge_rise_m,
            StairClass.DOUBLE_LEDGE: self.double_ledge_rise_m,
        }
        if kind == StairClass.DOUBLE_LEDGE:
            return StairTaskSpec(kind, self.double_ledge_rise_m,
                                 self.platform_depth_m, 2, True)
        return StairTaskSpec(kind, heights[kind], self.platform_depth_m,
                             1, kind in (StairClass.SINGLE_LEDGE, StairClass.HIGH_STEP))

    def double_unlocked(self, successes: int, attempts: int) -> bool:
        if type(successes) is not int or type(attempts) is not int:
            raise ValueError("successes/attempts 必须是整数")
        if successes < 0 or attempts < 0 or successes > attempts:
            raise ValueError("successes/attempts 超出范围")
        return attempts >= self.double_unlock_min_attempts and (
            successes / attempts >= self.double_unlock_success_fraction)

    def state_dict(self):
        return asdict(self)


__all__ = ["StairClass", "StairCourseSpec", "StairTaskSpec"]

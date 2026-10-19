"""轮腿执行器适配：只组合既有控制核心，不复制 PD/闭链公式。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..contracts import ACTION_ORDER
from ..control.actuation import (EffectiveDriveResponse, OutputFeedbackController,
                                 PairTargetDomain, ResponseParameters)


@dataclass
class WheeledControlCore:
    """把策略动作转换为 6 轴输出目标与反馈力矩。

    q/dq 使用策略轴序 [左髋, 左膝, 右髋, 右膝, 左轮, 右轮]。
    response=None 表示 M1 平地诊断不使用额外驱动滞后，而不是实车零延迟。
    """

    nominal_position_rad: tuple[float, ...]
    num_envs: int
    response: ResponseParameters | None = None
    pair_domain: PairTargetDomain | None = None
    device: str = "cpu"

    def __post_init__(self):
        if (len(self.nominal_position_rad) != len(ACTION_ORDER)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in self.nominal_position_rad)):
            raise ValueError("nominal_position_rad 需要 6 个有限轴值")
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs 必须是正整数")
        self.device = torch.device(self.device)
        self.controller = OutputFeedbackController(self.nominal_position_rad, self.num_envs,
                                                   pair_domain=self.pair_domain, device=self.device)
        self.drive = (EffectiveDriveResponse(self.response, self.num_envs, device=self.device)
                      if self.response is not None else None)

    def decode(self, action, feedback_position_rad, controller_issued_at_s):
        return self.controller.decode(action, feedback_position_rad, controller_issued_at_s)

    def effort(self, action, feedback_position_rad, feedback_velocity_rad_s,
               controller_issued_at_s, pd_computed_at_s):
        targets = self.controller.decode(action, feedback_position_rad, controller_issued_at_s)
        feedback = self.controller.effort(targets, feedback_position_rad, feedback_velocity_rad_s,
                                          pd_computed_at_s)
        return targets, feedback

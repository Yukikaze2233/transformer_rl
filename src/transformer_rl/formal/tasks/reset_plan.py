"""任务准备层与原生执行层共享的复位计划。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..control.actuation import PreparationHold
    from .commands import TaskCommandPlan


@dataclass(frozen=True)
class NativeTaskResetPlan:
    root_pose_xyzw: torch.Tensor
    root_com_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    course_sha256: tuple[str, ...]
    commands: TaskCommandPlan
    preparation_issued_action: torch.Tensor | None = None
    source_courses: tuple | None = None  # Native order, not proof of native replacement.
    preparation_hold: PreparationHold | None = None

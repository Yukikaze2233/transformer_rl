"""Hypothesis-driven IsaacLab task skeleton.

纯任务/观测/执行器/reward 模块不导入 Isaac；只有 ``env`` 适配层接触 DirectRLEnv。
"""
from __future__ import annotations

__all__ = [
    "contract", "actuator", "observations", "settings",
    "formal_curriculum", "formal_reward", "recover",
]

"""Tensor contracts shared by the actor, collector, optimizer and environment."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Protocol

import torch

from ..common.types import ActionSample, HistoryBatch, PolicyEvaluation


@dataclass(frozen=True)
class VectorObservation:
    frame: torch.Tensor        # [N, F], exact actor features, no privileged data
    timestamp: torch.Tensor    # [N], monotonic per episode, seconds (prefer float64)
    command: torch.Tensor      # [N, C]
    critic: torch.Tensor       # [N, S], simulator state available to critic only


@dataclass(frozen=True)
class StepResult:
    observation: VectorObservation  # Next observation; reset rows are already reset
    reward: torch.Tensor            # [N], reward of the preceding issued action
    terminated: torch.Tensor        # [N], true task terminal
    truncated: torch.Tensor         # [N], time limit, distinct from true terminal
    final_critic: torch.Tensor      # [N, S], PRE-reset next state, not current state
    final_critic_valid: torch.Tensor  # [N], required for truncated & ~terminated
    info: dict[str, Any]


class VectorEnv(Protocol):
    """Synchronous, tensor-only batched environment with explicit auto-reset semantics."""

    num_envs: int
    device: torch.device

    def reset(self, seed: int | None = None) -> VectorObservation: ...

    def step(self, issued_action: torch.Tensor) -> StepResult:
        """Submit normalized commands; actual transport/application belongs to the env."""
        ...

    def close(self) -> None: ...

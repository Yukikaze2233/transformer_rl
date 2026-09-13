"""Tensor contracts shared by the actor, collector, optimizer and environment."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class HistoryBatch:
    frames: torch.Tensor       # [B, L, F], oldest to newest; owned snapshot
    times: torch.Tensor        # [B, L], policy observation availability times, seconds
    valid: torch.Tensor        # [B, L], valid history tokens
    command: torch.Tensor      # [B, C], current command, same scaling as frames
    now: torch.Tensor          # [B], current policy observation availability time

    def index(self, index) -> HistoryBatch:
        return HistoryBatch(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def clone(self) -> HistoryBatch:
        return HistoryBatch(**{f.name: getattr(self, f.name).detach().clone() for f in fields(self)})

    def to(self, device: str | torch.device) -> HistoryBatch:
        return HistoryBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


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


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: torch.Tensor      # [B], SUM over all action dimensions
    entropy: torch.Tensor       # [B], SUM over all action dimensions
    mean: torch.Tensor          # [B, A]
    std: torch.Tensor           # [B, A]


@dataclass(frozen=True)
class ActionSample:
    action: torch.Tensor       # [B, A], raw Gaussian sample for PPO likelihood
    evaluation: PolicyEvaluation


class VectorEnv(Protocol):
    """Synchronous, tensor-only batched environment with explicit auto-reset semantics."""

    num_envs: int
    device: torch.device

    def reset(self, seed: int | None = None) -> VectorObservation: ...

    def step(self, issued_action: torch.Tensor) -> StepResult:
        """Submit normalized commands; actual transport/application belongs to the env."""
        ...

    def close(self) -> None: ...

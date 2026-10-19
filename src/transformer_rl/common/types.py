"""正式与研究策略共享的概率输出类型，无运行时模型依赖。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import torch


@dataclass(frozen=True)
class PolicyEvaluation:
    log_prob: torch.Tensor      # [B], SUM over all action dimensions
    entropy: torch.Tensor       # [B], SUM over all action dimensions
    mean: torch.Tensor          # [B, A]
    std: torch.Tensor           # [B, A]


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
class ActionSample:
    action: torch.Tensor       # [B, A], raw Gaussian sample for PPO likelihood
    evaluation: PolicyEvaluation

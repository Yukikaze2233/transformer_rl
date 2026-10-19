"""Owned rollout snapshots and time-major, termination-aware GAE."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .types import HistoryBatch


def _validate_batch_tensors(
    history: HistoryBatch,
    critic: torch.Tensor,
    raw_action: torch.Tensor,
    issued_action: torch.Tensor,
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    scalars: dict[str, torch.Tensor],
) -> None:
    """Validate both collected transitions and externally constructed PPO batches."""
    if not isinstance(history, HistoryBatch):
        raise TypeError("history must be a HistoryBatch")
    if history.frames.ndim != 3 or any(size < 1 for size in history.frames.shape):
        raise ValueError("history.frames must have nonempty shape [B, L, F]")
    size, length, _ = history.frames.shape
    tensors = {f"history.{f.name}": getattr(history, f.name) for f in fields(history)}
    expected = {
        "history.times": (size, length),
        "history.valid": (size, length),
        "history.now": (size,),
    }
    for name, tensor in (("history.command", history.command), ("critic", critic),
                         ("raw_action", raw_action)):
        if tensor.ndim != 2 or tensor.shape[0] != size or tensor.shape[1] < 1:
            raise ValueError(f"{name} must have nonempty shape [B, D]")
    tensors.update(critic=critic, raw_action=raw_action, issued_action=issued_action,
                   old_mean=old_mean, old_std=old_std)
    for name in ("issued_action", "old_mean", "old_std"):
        expected[name] = tuple(raw_action.shape)
    tensors.update(scalars)
    expected.update({name: (size,) for name in scalars})
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}")
    for name, tensor in tensors.items():
        if tensor.device != history.frames.device:
            raise ValueError(f"{name} must be on the history device")
        if name in ("history.valid", "terminated", "truncated"):
            if tensor.dtype != torch.bool:
                raise ValueError(f"{name} must have bool dtype")
        else:
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have floating point dtype")
    # Validate the mask's shape/dtype/device before indexing any history tensor.
    # Padding is semantically absent; preserve its payload just as the actor does.
    for name, tensor in tensors.items():
        meaningful = tensor[history.valid] if name in ("history.frames", "history.times") else tensor
        if not torch.isfinite(meaningful).all():
            raise FloatingPointError(f"{name} contains nonfinite meaningful values")
    if not (old_std > 0).all():
        raise ValueError("old_std must be strictly positive")


@dataclass(frozen=True)
class PPOBatch:
    """Flat endpoint samples; scalar tensors have shape [B], never [B, 1]."""

    history: HistoryBatch
    critic: torch.Tensor
    raw_action: torch.Tensor
    issued_action: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.raw_action.shape[0]

    def index(self, index) -> PPOBatch:
        """Select endpoints together, preserving the batch axis for integer indices."""
        if isinstance(index, torch.Tensor) and index.ndim == 0:
            index = index.item()
        if isinstance(index, int):
            index = index + len(self) if index < 0 else index
            if not 0 <= index < len(self):
                raise IndexError("PPOBatch index out of range")
            index = slice(index, index + 1)
        return type(self)(**{
            f.name: (self.history.index(index) if f.name == "history"
                     else getattr(self, f.name)[index])
            for f in fields(self)
        })

    def validate(self) -> None:
        _validate_batch_tensors(
            self.history, self.critic, self.raw_action, self.issued_action,
            self.old_mean, self.old_std,
            {name: getattr(self, name) for name in
             ("old_log_prob", "old_value", "advantages", "returns")},
        )


@dataclass(frozen=True)
class EstimatorBatch(PPOBatch):
    """Owned next-observation targets; invalid pairs never cross episode resets."""

    next_proprio: torch.Tensor
    next_valid: torch.Tensor

    def validate(self):
        super().validate()
        if (self.next_proprio.ndim != 2 or self.next_proprio.shape[0] != len(self)
            or self.next_proprio.shape[1] < 1 or self.next_proprio.dtype != self.history.frames.dtype
            or self.next_proprio.device != self.history.frames.device):
            raise ValueError("next_proprio requires [B, D] with the history dtype/device")
        if (self.next_valid.shape != (len(self),) or self.next_valid.dtype != torch.bool
            or self.next_valid.device != self.history.frames.device):
            raise ValueError("next_valid requires bool [B] on the history device")
        if not torch.isfinite(self.next_proprio[self.next_valid]).all():
            raise FloatingPointError("valid next_proprio contains nonfinite values")


@dataclass(frozen=True)
class _Transition:
    history: HistoryBatch
    critic: torch.Tensor
    raw_action: torch.Tensor
    issued_action: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    next_value: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class RolloutBuffer:
    """Reference storage: one full, detached history snapshot per endpoint.

    Capacity counts vector steps, not flattened samples. A partial rollout can
    be finished at any time. ``finish`` neither consumes nor changes snapshots.
    """

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._steps: list[_Transition] = []

    def __len__(self) -> int:
        return len(self._steps)

    def add(
        self,
        history: HistoryBatch,
        critic: torch.Tensor,
        raw_action: torch.Tensor,
        issued_action: torch.Tensor,
        old_log_prob: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        old_value: torch.Tensor,
        reward: torch.Tensor,
        next_value: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        if len(self) >= self.capacity:
            raise RuntimeError("rollout buffer is full")
        _validate_batch_tensors(
            history, critic, raw_action, issued_action, old_mean, old_std,
            dict(old_log_prob=old_log_prob, old_value=old_value, reward=reward,
                 next_value=next_value, terminated=terminated, truncated=truncated),
        )
        transition = _Transition(
            history, critic, raw_action, issued_action, old_log_prob,
            old_mean, old_std, old_value, reward, next_value, terminated, truncated,
        )
        if self._steps:
            previous = self._steps[0]
            for field in fields(transition):
                current, first = getattr(transition, field.name), getattr(previous, field.name)
                pairs = ((getattr(current, f.name), getattr(first, f.name))
                         for f in fields(history)) if field.name == "history" else [(current, first)]
                for tensor, reference in pairs:
                    if (tensor.shape != reference.shape or tensor.dtype != reference.dtype
                            or tensor.device != reference.device):
                        raise ValueError(f"{field.name} shape, dtype and device must stay constant")
        snapshot = _Transition(**{
            f.name: (history.clone() if f.name == "history"
                     else getattr(transition, f.name).detach().clone())
            for f in fields(transition)
        })
        self._steps.append(snapshot)

    def finish(self, gamma: float, gae_lambda: float) -> PPOBatch:
        if not self._steps:
            raise ValueError("cannot finish an empty rollout")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (gamma, gae_lambda)):
            raise ValueError("gamma and gae_lambda must be finite and in [0, 1]")
        stacked = {
            f.name: torch.stack([getattr(step, f.name) for step in self._steps])
            for f in fields(_Transition) if f.name != "history"
        }
        advantage = torch.zeros_like(stacked["old_value"][-1])
        reversed_advantages = []
        for time in range(len(self) - 1, -1, -1):
            terminated = stacked["terminated"][time]
            done = terminated | stacked["truncated"][time]
            bootstrap = torch.where(terminated, 0.0, stacked["next_value"][time])
            delta = stacked["reward"][time] + gamma * bootstrap - stacked["old_value"][time]
            # A timeout bootstraps its pre-reset state but cuts the next episode's trace.
            advantage = delta + gamma * gae_lambda * torch.where(done, 0.0, advantage)
            reversed_advantages.append(advantage)
        advantages = torch.stack(reversed_advantages[::-1])
        returns = advantages + stacked["old_value"]
        if not torch.isfinite(advantages).all() or not torch.isfinite(returns).all():
            raise FloatingPointError("GAE produced nonfinite advantages or returns")
        history = HistoryBatch(**{
            f.name: torch.stack([getattr(step.history, f.name) for step in self._steps]).flatten(0, 1)
            for f in fields(HistoryBatch)
        })
        return PPOBatch(
            history=history,
            **{name: stacked[name].flatten(0, 1) for name in
               ("critic", "raw_action", "issued_action", "old_log_prob",
                "old_mean", "old_std", "old_value")},
            advantages=advantages.flatten(0, 1),
            returns=returns.flatten(0, 1),
        )

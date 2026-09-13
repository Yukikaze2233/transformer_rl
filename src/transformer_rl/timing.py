"""Explicit simulation-side command timing; no actuator or hardware assumptions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


Seconds = float | torch.Tensor


@dataclass(frozen=True)
class TimingProfile:
    """Nested, synchronous clocks with fresh physics state at controller ticks.

    These periods describe a selected simulation schedule, not measured hardware
    rates. An asynchronous or internally substepped actuator needs its own model.
    """

    policy_period_s: float
    controller_period_s: float
    physics_period_s: float

    def __post_init__(self) -> None:
        periods = (self.policy_period_s, self.controller_period_s, self.physics_period_s)
        if any(isinstance(v, bool) or not isinstance(v, Real)
               or not math.isfinite(v) or v <= 0 for v in periods):
            raise ValueError("periods must be finite, positive seconds")
        if self.controller_period_s < self.physics_period_s:
            raise ValueError("controller period must be >= physics period for fresh state")
        for name, ratio in (
            ("controller/physics", self.controller_period_s / self.physics_period_s),
            ("policy/controller", self.policy_period_s / self.controller_period_s),
        ):
            if (not math.isfinite(ratio) or ratio < 1
                    or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"{name} period ratio must be a positive integer")

    @property
    def controller_substeps(self) -> int:
        return round(self.controller_period_s / self.physics_period_s)

    @property
    def controllers_per_policy(self) -> int:
        return round(self.policy_period_s / self.controller_period_s)

    @property
    def policy_substeps(self) -> int:
        return self.controllers_per_policy * self.controller_substeps


@dataclass(frozen=True)
class CommandSubmission:
    """Owned [N] scheduling snapshots, not transport ACKs or application evidence."""

    sequence: torch.Tensor
    issued_at_s: torch.Tensor
    scheduled_at_s: torch.Tensor


@dataclass(frozen=True)
class AppliedCommandState:
    """Owned snapshots of a simulated controller-side target latch.

    ``applied_target`` is [N, A]; all other fields are [N]. Times are float64
    seconds. ``applied_at_s`` is the actual advance/reset call time, not a
    backdated arrival timestamp. Sequence -1 denotes an initial/reset target;
    its issue/schedule times are NaN. The next schedule is +inf for an empty
    queue. ``superseded_count`` counts arrived packets retired without becoming
    the selected target, since the most recent reset of each environment.

    This is not evidence of a physical motor response. Hardware actor inputs
    require corresponding, timestamped application feedback from that hardware.
    """

    applied_target: torch.Tensor
    applied_sequence: torch.Tensor
    issued_at_s: torch.Tensor
    scheduled_at_s: torch.Tensor
    applied_at_s: torch.Tensor
    now_s: torch.Tensor
    pending_count: torch.Tensor
    next_scheduled_at_s: torch.Tensor
    superseded_count: torch.Tensor


class DelayedCommandChannel:
    """Batched bounded transport model with latest-sequence-wins selection.

    ``submit(target, now_s=..., delay_s=...)`` enqueues one complete target per
    environment. ``advance(now_s)`` applies the highest sequence that has
    arrived by that time, unless a higher sequence has already been applied.
    Otherwise the previous target is held. Submission alone never applies a
    target, including zero-delay submissions; advance at the same time to latch
    them. Calling advance less often quantizes application to those calls.

    Not-yet-arrived packets occupy slots even if a newer packet has won. Arrived
    losers are consumed and counted, not applied transiently. This is an
    explicit transport model, not a claim that hardware FIFOs behave this way.

    Capacity is per environment. Overflow rejects the entire submission with
    BufferError and no mutation. It does not drain due packets automatically;
    the caller must advance the controller explicitly before retrying.

    Times may be scalar seconds or same-device [N] tensors and must be finite
    and nondecreasing per environment across all calls, including reset.
    Keep this clock separate from any episode-relative observation clock.
    Sequence counters are never reused on reset. Targets and tensor arguments
    stay on their original device. Validation uses scalar reductions (CUDA
    synchronization), but never copies batches to CPU. This reference component
    is not CUDA-graph/capture optimized and does not propagate gradients.
    """

    @torch.no_grad()
    def __init__(
        self, initial_target: torch.Tensor, capacity: int, *, now_s: Seconds = 0.0
    ) -> None:
        if (not isinstance(initial_target, torch.Tensor) or initial_target.ndim != 2
                or any(size < 1 for size in initial_target.shape)
                or not initial_target.is_floating_point()):
            raise ValueError("initial_target must be a floating tensor [N, A] with N, A > 0")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer per environment")
        self.num_envs, self.action_dim = initial_target.shape
        self.device = initial_target.device
        self.dtype = initial_target.dtype
        self.capacity = capacity
        self._validate_target(initial_target)
        now = self._seconds(now_s, "now_s")

        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._targets = initial_target.new_zeros((capacity, *initial_target.shape))
        self._sequence = torch.full(
            (capacity, self.num_envs), -1, dtype=torch.int64, device=self.device
        )
        self._issued_at = torch.full(
            (capacity, self.num_envs), math.nan, dtype=torch.float64, device=self.device
        )
        self._scheduled_at = torch.full_like(self._issued_at, math.nan)
        self._next_sequence = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._applied_target = initial_target.detach().clone()
        self._applied_sequence = torch.full_like(self._next_sequence, -1)
        self._applied_issued_at = torch.full_like(now, math.nan)
        self._applied_scheduled_at = torch.full_like(now, math.nan)
        self._applied_at = now.clone()
        self._now = now.clone()
        self._superseded_count = torch.zeros_like(self._next_sequence)

    def _validate_target(self, target: torch.Tensor) -> None:
        if (not isinstance(target, torch.Tensor)
                or target.shape != (self.num_envs, self.action_dim)
                or target.device != self.device or target.dtype != self.dtype):
            raise ValueError("target must match initial_target shape, device and dtype")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("target must contain only finite values")

    def _seconds(self, value: Seconds, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device != self.device or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating tensor on {self.device}")
            result = value.detach().to(dtype=torch.float64)
        elif isinstance(value, Real) and not isinstance(value, bool):
            result = torch.tensor(value, dtype=torch.float64, device=self.device)
        else:
            raise ValueError(f"{name} must be scalar seconds or a floating tensor [N]")
        if result.ndim == 0:
            result = result.expand(self.num_envs)
        if result.shape != (self.num_envs,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} must be finite scalar seconds or shape [N]")
        return result

    def _check_monotonic(self, now: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        backwards = now < self._now
        if mask is not None:
            backwards = backwards & mask
        if bool(backwards.any()):
            raise ValueError("now_s must be nondecreasing, including across reset")

    @torch.no_grad()
    def submit(
        self, issued_target: torch.Tensor, *, now_s: Seconds, delay_s: Seconds
    ) -> CommandSubmission:
        self._validate_target(issued_target)
        now = self._seconds(now_s, "now_s")
        delay = self._seconds(delay_s, "delay_s")
        self._check_monotonic(now)
        scheduled = now + delay
        if bool((delay < 0).any()) or not bool(torch.isfinite(scheduled).all()):
            raise ValueError("delay_s must be nonnegative and scheduled times finite")
        free = self._sequence < 0
        if bool((~free.any(dim=0)).any()):
            raise BufferError(
                "command queue full; entire batch rejected, advance or configure a larger capacity"
            )
        if bool((self._next_sequence == torch.iinfo(torch.int64).max).any()):
            raise OverflowError("command sequence exhausted; entire batch rejected")

        # Each environment can have a different free slot; no per-env Python loop.
        slots = free.to(torch.int64).argmax(dim=0)
        sequence = self._next_sequence.clone()
        self._targets[slots, self._env_ids] = issued_target.detach()
        self._sequence[slots, self._env_ids] = sequence
        self._issued_at[slots, self._env_ids] = now
        self._scheduled_at[slots, self._env_ids] = scheduled
        self._next_sequence.add_(1)
        self._now.copy_(now)
        return CommandSubmission(sequence, now.clone(), scheduled.clone())

    @torch.no_grad()
    def advance(self, now_s: Seconds) -> AppliedCommandState:
        now = self._seconds(now_s, "now_s")
        self._check_monotonic(now)
        due = (self._sequence >= 0) & (self._scheduled_at <= now.unsqueeze(0))
        sequence, slots = torch.where(due, self._sequence, -1).max(dim=0)
        apply = sequence > self._applied_sequence
        selected = (slots, self._env_ids)
        self._applied_target = torch.where(
            apply.unsqueeze(1), self._targets[selected], self._applied_target
        )
        self._applied_sequence = torch.where(apply, sequence, self._applied_sequence)
        self._applied_issued_at = torch.where(
            apply, self._issued_at[selected], self._applied_issued_at
        )
        self._applied_scheduled_at = torch.where(
            apply, self._scheduled_at[selected], self._applied_scheduled_at
        )
        self._applied_at = torch.where(apply, now, self._applied_at)
        self._superseded_count.add_(due.sum(dim=0) - apply.to(torch.int64))
        self._sequence.masked_fill_(due, -1)
        self._now.copy_(now)
        return self.snapshot()

    @torch.no_grad()
    def reset(
        self, mask: torch.Tensor, initial_target: torch.Tensor, *, now_s: Seconds
    ) -> AppliedCommandState:
        """Clear only selected rows, installing [N, A] initial_target[mask].

        Unselected environments, including their clocks and pending packets, are
        unchanged. All target values must be finite; reset time monotonicity is
        checked only for selected rows. Sequence allocation remains monotonic.
        """
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.device != self.device or mask.dtype != torch.bool):
            raise ValueError("reset mask must be boolean [N] on the channel device")
        self._validate_target(initial_target)
        now = self._seconds(now_s, "now_s")
        self._check_monotonic(now, mask)

        self._sequence[:, mask] = -1
        self._targets[:, mask] = 0
        self._issued_at[:, mask] = math.nan
        self._scheduled_at[:, mask] = math.nan
        self._applied_target[mask] = initial_target.detach()[mask]
        self._applied_sequence[mask] = -1
        self._applied_issued_at[mask] = math.nan
        self._applied_scheduled_at[mask] = math.nan
        self._applied_at[mask] = now[mask]
        self._now[mask] = now[mask]
        self._superseded_count[mask] = 0
        return self.snapshot()

    @torch.no_grad()
    def snapshot(self) -> AppliedCommandState:
        """Inspect owned state without advancing time or acknowledging a packet."""
        pending = self._sequence >= 0
        next_scheduled = torch.where(pending, self._scheduled_at, math.inf).amin(dim=0)
        return AppliedCommandState(
            applied_target=self._applied_target.detach().clone(),
            applied_sequence=self._applied_sequence.clone(),
            issued_at_s=self._applied_issued_at.clone(),
            scheduled_at_s=self._applied_scheduled_at.clone(),
            applied_at_s=self._applied_at.clone(),
            now_s=self._now.clone(),
            pending_count=pending.sum(dim=0),
            next_scheduled_at_s=next_scheduled,
            superseded_count=self._superseded_count.clone(),
        )

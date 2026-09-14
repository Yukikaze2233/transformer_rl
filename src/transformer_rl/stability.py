"""Episode-centered statistics for explicitly timestamped PRE-reset signals."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)


@dataclass
class _EpisodeSignal:
    moments: _Moments = field(default_factory=_Moments)
    max_abs: float = 0.0
    previous: float = 0.0
    derivative_norm: float = 0.0

    def add(self, value: float, delta_t: float | None) -> None:
        if self.moments.count:
            # The first retained sample has no derivative, even after warmup.
            derivative = (value - self.previous) / delta_t
            if not math.isfinite(derivative):
                raise FloatingPointError("evaluation signal derivative must be finite")
            self.derivative_norm = math.hypot(self.derivative_norm, derivative)
        self.moments.add(value)
        self.max_abs = max(self.max_abs, abs(value))
        self.previous = value


@dataclass
class _Episode:
    signals: list[_EpisodeSignal]
    samples: int = 0
    previous_time: float | None = None


@dataclass
class _SignalSummary:
    count: int = 0
    segments: int = 0
    short_segments: int = 0
    short_count: int = 0
    total_count: int = 0
    settled_count: int = 0
    completed_segments: int = 0
    partial_segments: int = 0
    short_completed_segments: int = 0
    short_partial_segments: int = 0
    mean: float = 0.0
    within_m2: float = 0.0
    max_abs: float = 0.0
    derivative_norm: float = 0.0
    derivative_count: int = 0
    episode_means: _Moments = field(default_factory=_Moments)
    episode_mean_min: float = math.inf
    episode_mean_max: float = -math.inf

    def add(self, signal: _EpisodeSignal, samples: int, minimum: int, complete: bool) -> None:
        count = signal.moments.count
        self.total_count += samples
        self.settled_count += samples - count
        if count < minimum:
            self.short_segments += 1
            self.short_count += count
            self.short_completed_segments += int(complete)
            self.short_partial_segments += int(not complete)
            return
        self.count += count
        self.segments += 1
        self.completed_segments += int(complete)
        self.partial_segments += int(not complete)
        self.mean += (signal.moments.mean - self.mean) * (count / self.count)
        # Deliberately omit between-episode mean offsets from the pooled M2.
        self.within_m2 += signal.moments.m2
        self.max_abs = max(self.max_abs, signal.max_abs)
        self.derivative_norm = math.hypot(self.derivative_norm, signal.derivative_norm)
        self.derivative_count += count - 1
        self.episode_means.add(signal.moments.mean)
        self.episode_mean_min = min(self.episode_mean_min, signal.moments.mean)
        self.episode_mean_max = max(self.episode_mean_max, signal.moments.mean)

    def report(self) -> dict:
        return {
            "mean": self.mean if self.count else None,
            "within_episode_std": math.sqrt(max(0.0, self.within_m2) / self.count)
            if self.count else None,
            "derivative_rms": self.derivative_norm / math.sqrt(self.derivative_count)
            if self.derivative_count else None,
            "max_abs": self.max_abs if self.count else None,
            "count": self.count,
            "segments": self.segments,
            "short_segments": self.short_segments,
            "short_count": self.short_count,
            "total_count": self.total_count,
            "settled_count": self.settled_count,
            "derivative_count": self.derivative_count,
            "completed_segments": self.completed_segments,
            "partial_segments": self.partial_segments,
            "short_completed_segments": self.short_completed_segments,
            "short_partial_segments": self.short_partial_segments,
            "episode_mean_min": self.episode_mean_min if self.segments else None,
            "episode_mean_max": self.episode_mean_max if self.segments else None,
            "episode_mean_std": math.sqrt(max(0.0, self.episode_means.m2) / self.segments)
            if self.segments else None,
        }


class EpisodeSignalStatistics:
    """Accumulate one vector sample per update; report() finalizes partial episodes.

    All signal tensors, time and done must share a device. A single packed CPU
    transfer per update avoids per-signal GPU synchronization. The CPU snapshot
    is reduced with Welford moments; memory does not grow with evaluation length.
    """

    def __init__(self, num_envs: int, *, settle_steps: int = 200,
                 min_steady_samples: int = 200):
        self.validate_protocol(settle_steps, min_steady_samples)
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.settle_steps = settle_steps
        self.min_steady_samples = min_steady_samples
        self._names: tuple[str, ...] | None = None
        self._episodes: list[_Episode] = []
        self._summaries: list[_SignalSummary] = []
        self._finished = False

    @staticmethod
    def validate_protocol(settle_steps: int, min_steady_samples: int) -> None:
        if type(settle_steps) is not int or settle_steps < 0:
            raise ValueError("settle_steps must be a nonnegative integer (steps)")
        if type(min_steady_samples) is not int or min_steady_samples < 1:
            raise ValueError("min_steady_samples must be a positive integer (samples)")

    def _validate_tensor(self, name: str, value: torch.Tensor, device: torch.device,
                         dtype: torch.dtype | None = None) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},)")
        if value.device != device:
            raise ValueError(f"{name} must share the environment device")
        if dtype is not None:
            if value.dtype != dtype:
                raise TypeError(f"{name} must have {dtype} dtype")
        elif not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")

    @torch.no_grad()
    def update(self, signals: dict[str, torch.Tensor], signal_time: torch.Tensor | None,
               done: torch.Tensor) -> None:
        """Include PRE-reset samples, then flush and clear only done rows.

        signal_time is the physical signal sample time in seconds, independent
        of actor events and already-reset next-observation timestamps.
        """
        if self._finished:
            raise RuntimeError("cannot update statistics after report()")
        if not isinstance(signals, dict) or any(type(k) is not str or not k for k in signals):
            raise ValueError("evaluation_signals must map nonempty names to tensors")
        names = tuple(sorted(signals))
        if self._names is not None and names != self._names:
            raise ValueError("evaluation signal names must remain constant across steps")
        if not isinstance(done, torch.Tensor):
            raise TypeError("done must be a tensor")
        self._validate_tensor("done", done, done.device, torch.bool)
        if not names:
            if signal_time is not None:
                raise ValueError("evaluation_signal_time requires nonempty evaluation_signals")
            self._names = names
            return
        if signal_time is None:
            raise ValueError("evaluation_signals require PRE-reset evaluation_signal_time")
        self._validate_tensor("evaluation_signal_time", signal_time, done.device, torch.float64)
        for name, value in signals.items():
            self._validate_tensor(f"evaluation_signals.{name}", value, done.device)
        # Pack on the source device; finiteness and time checks happen on CPU.
        packed = torch.stack([signals[name].detach().double() for name in names]
                             + [signal_time.detach(), done.double()], dim=1).cpu()
        if not torch.isfinite(packed).all():
            raise FloatingPointError("evaluation signals and signal time must be finite")
        rows = packed.tolist()
        if self._names is None:
            self._names = names
            self._episodes = [self._new_episode() for _ in range(self.num_envs)]
            self._summaries = [_SignalSummary() for _ in names]
        for episode, row in zip(self._episodes, rows):
            if episode.previous_time is not None and row[-2] <= episode.previous_time:
                raise ValueError("evaluation_signal_time must strictly increase within each episode")
        for index, (episode, row) in enumerate(zip(self._episodes, rows)):
            timestamp = row[-2]
            delta_t = None if episode.previous_time is None else timestamp - episode.previous_time
            episode.samples += 1
            if episode.samples > self.settle_steps:
                for signal, value in zip(episode.signals, row[:-2]):
                    signal.add(value, delta_t)
            episode.previous_time = timestamp
            if row[-1]:
                self._flush(index, complete=True)

    def _new_episode(self) -> _Episode:
        return _Episode([_EpisodeSignal() for _ in self._names])

    def _flush(self, index: int, *, complete: bool) -> None:
        episode = self._episodes[index]
        if episode.samples:
            for summary, signal in zip(self._summaries, episode.signals):
                summary.add(signal, episode.samples, self.min_steady_samples, complete)
        self._episodes[index] = self._new_episode()

    def report(self) -> dict:
        """Finalize once; repeated calls are idempotent and do not add empty episodes."""
        if not self._finished:
            for index in range(len(self._episodes)):
                self._flush(index, complete=False)
            self._finished = True
        signals = {name: summary.report()
                   for name, summary in zip(self._names or (), self._summaries)}
        return {
            "available": any(signal["segments"] for signal in signals.values()),
            "protocol": {
                "settle_steps": self.settle_steps,
                "min_steady_samples": self.min_steady_samples,
                "centering": "per_environment_episode",
                "settle_unit": "steps",
                "signal_time_unit": "seconds",
                "variance_denominator": "retained_sample_count",
                "derivative_weighting": "equal_weight_per_within_segment_difference",
                "episode_mean_weighting": "equal_weight_per_usable_segment",
            },
            "signals": signals,
            "scope": (
                "Fixed post-settle windows per environment episode, including failure episodes; "
                "eligible final partial episodes are included and marked partial. Availability "
                "means sufficient samples, not standing success or convergence. Consult full-interval "
                "metrics and termination counts. At 100 Hz policy sampling, the first 200 steps "
                "represent nominally 2 s; this is not a hardware measurement. Such sampling cannot "
                "measure MCU fluctuations above 50 Hz. Derivatives of actor targets/torques describe "
                "changes between policy-rate samples, not the complete low-level current loop. "
                "Signal timestamps are PRE-reset physical sample times, independent of actor events."
            ),
        }

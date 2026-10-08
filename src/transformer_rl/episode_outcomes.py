"""Fixed first-episode outcomes from explicit PRE-reset environment evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch


_INTEGER_FIELDS = ("episode_ticks", "episode_horizon_ticks")
_BOOLEAN_FIELDS = ("time_out", "environment_failure", "task_success", "boundary",
                   "blocked", "survival_applicable")
_FIELDS = _INTEGER_FIELDS + _BOOLEAN_FIELDS
_END_REASONS = ("environment_failure", "boundary", "blocked", "task_success",
                "full_horizon", "other_truncation", "other_termination")


@dataclass
class _Episode:
    samples: int = 0
    ticks: int = 0
    horizon: int | None = None
    survival_applicable: bool | None = None
    continuous_violation_s: float = 0.
    health_violation_latched: bool = False
    terminal: dict | None = None


class EpisodeOutcomeStatistics:
    """Use exactly the first observed episode after reset for every declared row.

    Completed rows remain in the fixed cohort; later automatic resets cannot
    increase the denominator. Partial first episodes never count as successes.
    The health detector applies to continuous survival tasks, not intentional
    jump/traverse phases. Neither survival metric asserts command tracking.
    """

    def __init__(self, num_envs, policy_dt_s):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if (isinstance(policy_dt_s, bool) or not isinstance(policy_dt_s, (int, float))
                or not math.isfinite(policy_dt_s) or policy_dt_s <= 0):
            raise ValueError("policy_dt_s must be finite and positive")
        self.num_envs = num_envs
        self.policy_dt_s = float(policy_dt_s)
        self._episodes = [_Episode() for _ in range(num_envs)]
        self._available = None
        self._device = None
        self._finished = False
        self._cached = None

    def update(self, metadata, height, tilt, done):
        if self._finished:
            raise RuntimeError("episode outcome statistics already finalized")
        if (not isinstance(done, torch.Tensor) or done.shape != (self.num_envs,)
                or done.dtype != torch.bool):
            raise ValueError("done must be a bool tensor [num_envs]")
        if self._device is not None and done.device != self._device:
            raise ValueError("episode outcome device must stay fixed")
        present = metadata is not None
        if self._available is not None and present != self._available:
            raise ValueError("episode outcome metadata availability changed")
        if not present:
            # Adapters without explicit outcomes need not manufacture height,
            # tilt, episode limits or timeout reasons for this statistic.
            self._available = False
            self._device = done.device
            return
        if not isinstance(metadata, dict) or set(metadata) != set(_FIELDS):
            raise ValueError("episode outcome metadata requires exactly the declared fields")
        for name in _FIELDS:
            value = metadata[name]
            dtype = torch.int64 if name in _INTEGER_FIELDS else torch.bool
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                    or value.dtype != dtype or value.device != done.device):
                raise ValueError(f"invalid episode outcome metadata.{name}")
        for name, value in (("height", height), ("tilt", tilt)):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                    or not value.is_floating_point() or value.device != done.device):
                raise ValueError(f"{name} must be floating point [num_envs] on the done device")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite episode outcome {name}")
        if (metadata["episode_ticks"] < 1).any() or (metadata["episode_horizon_ticks"] < 1).any():
            raise ValueError("episode ticks and horizon ticks must be positive")
        for name in ("time_out", "environment_failure", "task_success"):
            if (metadata[name] & ~done).any():
                raise ValueError(f"terminal {name} requires done")
        if (metadata["environment_failure"] & metadata["task_success"]).any():
            raise ValueError("environment failure and task success cannot both be true")

        # Preserve integer evidence exactly, including limits above 2**53.
        rows = torch.stack([metadata[name].detach().to(torch.int64) for name in _FIELDS]
                           + [done.detach().to(torch.int64)], -1).cpu().tolist()
        physical = torch.stack((height.detach().double(), tilt.detach().double()), -1).cpu().tolist()
        for episode, row in zip(self._episodes, rows):
            if episode.terminal is not None:
                continue
            ticks, horizon, *flags = row
            applicable = bool(flags[-2])
            expected_tick = episode.ticks + 1
            if ticks != expected_tick:
                raise ValueError("first cohort must start at tick 1 and advance by exactly one tick")
            if episode.samples and (horizon != episode.horizon or applicable != episode.survival_applicable):
                raise ValueError("first-cohort horizon and survival applicability must stay fixed")

        # Validate all active rows before changing any of the cohort state.
        self._available = True
        self._device = done.device
        for episode, row, (height_value, tilt_value) in zip(self._episodes, rows, physical):
            if episode.terminal is not None:
                continue
            ticks, horizon, timeout, failure, success, boundary, blocked, applicable, ended = row
            episode.samples += 1
            episode.ticks, episode.horizon = ticks, horizon
            episode.survival_applicable = bool(applicable)
            if applicable:
                violation = height_value < .20 or tilt_value > .60
                episode.continuous_violation_s = (episode.continuous_violation_s + self.policy_dt_s
                                                   if violation else 0.)
                episode.health_violation_latched |= episode.continuous_violation_s >= .20 - 1e-9
            if ended:
                full = bool(timeout and ticks >= horizon and not (failure or success or boundary or blocked))
                reason = ("environment_failure" if failure else "boundary" if boundary else
                          "blocked" if blocked else "task_success" if success else
                          "full_horizon" if full else "other_truncation" if timeout else "other_termination")
                episode.terminal = {"time_out": bool(timeout), "environment_failure": bool(failure),
                    "task_success": bool(success), "boundary": bool(boundary), "blocked": bool(blocked),
                    "end_reason": reason,
                    "full_horizon_survived": full if applicable else None,
                    "healthy_full_horizon": full and not episode.health_violation_latched if applicable else None}

    def _outcome(self, index, episode):
        status = "completed" if episode.terminal is not None else "censored" if episode.samples else "not_started"
        result = {"env_id": index, "status": status, "samples": episode.samples,
                  "episode_ticks": episode.ticks if episode.samples else None,
                  "episode_horizon_ticks": episode.horizon,
                  "survival_applicable": episode.survival_applicable,
                  "observed_duration_s": episode.samples * self.policy_dt_s,
                  "health_violation_latched": episode.health_violation_latched if episode.survival_applicable else None}
        result.update(episode.terminal or {"time_out": None, "environment_failure": None,
            "task_success": None, "boundary": None, "blocked": None, "end_reason": None,
            "full_horizon_survived": False if episode.survival_applicable else None,
            "healthy_full_horizon": False if episode.survival_applicable else None})
        return result

    def report(self):
        if self._cached is not None:
            return deepcopy(self._cached)
        outcomes = [self._outcome(index, episode) for index, episode in enumerate(self._episodes)]
        available = bool(self._available)
        completed = sum(row["status"] == "completed" for row in outcomes)
        censored = sum(row["status"] == "censored" for row in outcomes)
        not_started = sum(row["status"] == "not_started" for row in outcomes)
        samples = sum(row["samples"] for row in outcomes)
        censored_samples = sum(row["samples"] for row in outcomes if row["status"] == "censored")

        def counts(selected):
            return {"requested_episodes": len(selected),
                    "started_episodes": sum(row["status"] != "not_started" for row in selected),
                    "completed_episodes": sum(row["status"] == "completed" for row in selected),
                    "censored_episodes": sum(row["status"] == "censored" for row in selected),
                    "not_started_episodes": sum(row["status"] == "not_started" for row in selected)}

        survive = [row for row in outcomes if row["survival_applicable"] is True]
        tasks = [row for row in outcomes if row["survival_applicable"] is False]
        full = sum(row["full_horizon_survived"] is True for row in survive)
        healthy = sum(row["healthy_full_horizon"] is True for row in survive)
        task_successes = sum(row["task_success"] is True for row in tasks)
        survival = {**counts(survive), "applicable_rows": len(survive) if available else None,
                    "full_horizon_episodes": full, "healthy_full_horizon_episodes": healthy,
                    "health_violation_episodes": sum(row["health_violation_latched"] is True for row in survive),
                    "full_horizon_survival_rate": full / len(survive) if survive else None,
                    "healthy_full_horizon_rate": healthy / len(survive) if survive else None,
                    "status": "available" if survive else "not_applicable" if available else "unavailable"}
        task = {**counts(tasks), "task_success_episodes": task_successes,
                "task_success_rate": task_successes / len(tasks) if tasks else None,
                "status": "available" if tasks else "not_applicable" if available else "unavailable"}
        if not available:
            # The fixed total N is known, but task membership is not.
            for subset in (survival, task):
                for name in ("requested_episodes", "started_episodes", "completed_episodes",
                             "censored_episodes", "not_started_episodes"):
                    subset[name] = None
        self._cached = {"schema_version": 1, "available": available,
            "requested_episodes": self.num_envs, "started_episodes": self.num_envs - not_started,
            "completed_episodes": completed, "censored_episodes": censored,
            "not_started_episodes": not_started,
            "all_requested_accounted": available and completed == self.num_envs,
            "censored_episode_fraction": censored / self.num_envs,
            "observed_samples": samples, "censored_samples": censored_samples,
            "censored_sample_fraction": censored_samples / samples if samples else None,
            "environment_failure_episodes": sum(row["environment_failure"] is True for row in outcomes),
            "end_reasons": {reason: sum(row["end_reason"] == reason for row in outcomes) for reason in _END_REASONS},
            "survival": survival, "task": task, "first_episode_outcomes": outcomes,
            "protocol": {"id": "transformer_rl.first_episode_outcomes.v1", "policy_dt_s": self.policy_dt_s,
                "cohort": "first episode after evaluator reset for every declared environment row",
                "denominator": "fixed requested rows; never completed episodes or subsequent auto resets",
                "survival_scope": "explicit continuous survive tasks only; discrete tasks are not applicable",
                "full_horizon": "done AND time_out AND episode_ticks >= episode_horizon_ticks AND no environment failure, task success, boundary or blocked reason",
                "health_height_below_m": .20, "health_tilt_above_rad": .60, "health_continuous_violation_s": .20,
                "health_scope": "entire first continuous-survival episode including warmup; not a tracking requirement",
                "censored_counts_as_success": False, "censored_rate_interpretation": "observed success lower bound when requested outcomes are incomplete",
                "sample_scope": "first-cohort samples only; first tick represents one policy_dt_s interval",
                "terminal_failure_source": "explicit environment_failure; policy-learning terminated is not a physical-failure signal",
                "end_reason_precedence": list(_END_REASONS)}}
        self._finished = True
        return deepcopy(self._cached)

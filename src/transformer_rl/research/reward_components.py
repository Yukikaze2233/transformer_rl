"""Streaming reward contributions from explicit PRE-reset producer tensors."""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch


class RewardComponentStatistics:
    """Describe every returned environment row in a bounded rollout window.

    Continuous components are reward densities, converted with the declared
    policy timestep. Event rewards are already per-step contributions and must
    be supplied explicitly. A residual is unattributed: it can contain omitted
    continuous terms as well as events. It is never an inferred event penalty,
    a gate activation rate, or evidence that the producer accounting is wrong.
    """

    def __init__(self, num_envs, policy_dt_s):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(policy_dt_s, bool) or not isinstance(policy_dt_s, (int, float)):
            raise ValueError("policy_dt_s must be finite and positive")
        try:
            timestep = float(policy_dt_s)
        except OverflowError as error:
            raise ValueError("policy_dt_s must be finite and positive") from error
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("policy_dt_s must be finite and positive")
        self.num_envs = num_envs
        self.policy_dt_s = timestep
        self._reset()

    def _reset(self):
        self._steps = 0
        self._device = None
        self._continuous_keys = None
        self._event_keys = None
        self._moments = None

    def _tensor(self, name, value, device):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or not value.is_floating_point() or value.layout != torch.strided
                or value.device != device or value.device.type == "meta"):
            raise ValueError(f"{name} must be floating [num_envs] on the reward device")
        return value.detach().to(torch.float64)

    def _components(self, name, values, device):
        if values is None:
            return None, []
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping or None")
        keys = tuple(values)
        if any(not isinstance(key, str) or not key or key.strip() != key
               or any(ord(character) < 32 or ord(character) == 127 for character in key)
               for key in keys):
            raise ValueError(f"{name} requires nonempty, unpadded component names")
        keys = tuple(sorted(keys))
        return keys, [self._tensor(f"{name}.{key}", values[key], device) for key in keys]

    @torch.no_grad()
    def observe(self, total_reward, component_densities, event_rewards=None):
        """Consume one vector step without retaining aliases or autograd graphs.

        Availability, component keys and device must stay fixed until drain.
        Invalid observations do not alter the current window. Reductions and
        accumulation stay on the input device; validation uses one combined
        scalar synchronization rather than one per reward component.
        """
        if not isinstance(total_reward, torch.Tensor):
            raise ValueError("total_reward must be a floating tensor [num_envs]")
        device = total_reward.device
        total = self._tensor("total_reward", total_reward, device)
        if self._device is not None and device != self._device:
            raise ValueError("reward statistics device must stay fixed within a window")
        continuous_keys, densities = self._components("component_densities", component_densities,
                                                      device)
        event_keys, events = self._components("event_rewards", event_rewards, device)
        if self._steps and continuous_keys != self._continuous_keys:
            raise ValueError("continuous component availability or keys changed within a window")
        if self._steps and event_keys != self._event_keys:
            raise ValueError("event component availability or keys changed within a window")

        scaled = [density * self.policy_dt_s for density in densities]
        rows = [total, *densities, *scaled, *events]
        if continuous_keys is not None:
            known = torch.stack(scaled + events).sum(0) if scaled or events else torch.zeros_like(total)
            rows.append(total - known)
        values = torch.stack(rows)
        current = torch.stack((values.sum(1), values.square().sum(1),
                               values.amin(1), values.amax(1)), 1)
        if self._moments is None:
            accumulated = current
        else:
            accumulated = torch.cat((self._moments[:, :2] + current[:, :2],
                                     torch.minimum(self._moments[:, 2:3], current[:, 2:3]),
                                     torch.maximum(self._moments[:, 3:4], current[:, 3:4])), 1)
        if not torch.isfinite(accumulated).all().item():
            raise FloatingPointError("reward observations or accumulated moments are nonfinite")

        # The reduction result owns its storage even for float64 source tensors.
        self._moments = accumulated
        self._device = device
        self._continuous_keys, self._event_keys = continuous_keys, event_keys
        self._steps += 1

    def _statistics(self, moments, unit):
        samples = self._steps * self.num_envs
        if moments is None:
            return {"available": False, "unit": unit, "samples": 0, "steps": 0,
                    "sum": None, "mean": None, "rms": None, "min": None, "max": None}
        total, squares, minimum, maximum = moments
        return {"available": True, "unit": unit, "samples": samples, "steps": self._steps,
                "sum": total, "mean": total / samples,
                "rms": math.sqrt(squares / samples), "min": minimum, "max": maximum}

    def drain(self):
        """Return owned Python statistics and reset for the next rollout window.

        A second drain has no samples and no available component sources. All
        samples include terminal and reset transitions; these are not steady
        state, task-eligibility or policy-learning-mask statistics.
        """
        moments = self._moments.detach().cpu().tolist() if self._moments is not None else None
        continuous = self._continuous_keys is not None and self._steps > 0
        events = self._event_keys is not None and self._steps > 0
        result = {
            "scope": "whole_rollout_window_all_returned_rows_including_terminal_and_reset_samples",
            "num_envs": self.num_envs, "policy_dt_s": self.policy_dt_s,
            "steps": self._steps, "samples": self._steps * self.num_envs,
            "actual_total_reward": self._statistics(moments[0] if moments else None, "reward/step"),
            "continuous_components": {
                "available": continuous,
                "reason": None if continuous else "no_explicit_component_densities_in_window",
                "terms": {} if continuous else None},
            "event_components": {
                "available": events,
                "reason": None if events else "no_explicit_event_rewards_in_window",
                "terms": {} if events else None},
            "residual_step_reward": {
                "available": continuous,
                "reason": None if continuous else "continuous_source_unavailable",
                "statistics": None,
                "interpretation": "unattributed_events_and_omitted_continuous_terms" if not events else
                                  "unattributed_terms_after_explicit_continuous_and_event_contributions"},
            "gate_coverage": {
                "available": False,
                "reason": "producer_did_not_supply_explicit_gate_or_eligibility_masks"},
        }
        if continuous:
            count = len(self._continuous_keys)
            for index, key in enumerate(self._continuous_keys):
                result["continuous_components"]["terms"][key] = {
                    "density": self._statistics(moments[1 + index], "reward/s"),
                    "step_reward": self._statistics(moments[1 + count + index], "reward/step")}
        offset = 1 + 2 * len(self._continuous_keys or ())
        if events:
            for index, key in enumerate(self._event_keys):
                result["event_components"]["terms"][key] = self._statistics(moments[offset + index],
                                                                           "reward/step")
        if continuous:
            result["residual_step_reward"]["statistics"] = self._statistics(moments[-1], "reward/step")
        self._reset()
        return result

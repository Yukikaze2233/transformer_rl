"""Simulated forward ToF stair trigger: one stair request per detected riser.

The beam samples the terrain height profile ahead of the body along the
direction of travel. A riser is a rise of at least the stair minimum within
one edge window (slopes and roughness stay below it). Each riser draws a
trigger distance; when the riser comes within it, the measured step (riser
top minus current ground, with ToF noise) is submitted after a short
latency, exactly once. The trigger re-arms when the stair skill ends, so a
two-level course triggers once per riser. A share of risers is triggered
early, as an operator pressing the manual stair key would.

The logic only consumes a height profile, so a real ToF reading can drive the
same detector on the robot.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TofTriggerConfig:
    """Distances in m along the travel direction, latency in policy intervals."""

    trigger_distance_m: tuple[float, float] = (.3, .8)
    early_probability: float = .2
    early_distance_m: tuple[float, float] = (.8, 1.5)
    height_noise_m: float = .01
    latency_intervals: tuple[int, int] = (1, 3)
    sample_spacing_m: float = .05
    edge_window_m: float = .1
    minimum_travel_speed_m_s: float = .05

    def __post_init__(self):
        for name in ('trigger_distance_m', 'early_distance_m'):
            low, high = getattr(self, name)
            if not 0 < low < high:
                raise ValueError(f'{name} requires ordered positive bounds')
        low, high = self.latency_intervals
        if (type(low) is not int or type(high) is not int or not 0 <= low <= high
                or not 0 <= self.early_probability <= 1
                or any(not math.isfinite(v) or v <= 0
                       for v in (self.height_noise_m, self.sample_spacing_m, self.edge_window_m,
                                 self.minimum_travel_speed_m_s))
                or self.edge_window_m < self.sample_spacing_m):
            raise ValueError('ToF trigger bounds are invalid')

    @property
    def range_m(self):
        """Beam length: the farthest trigger distance plus one edge window to read the riser top."""
        return max(self.trigger_distance_m[1], self.early_distance_m[1]) + self.edge_window_m

    def sample_distances(self, device=None):
        count = math.ceil(self.range_m / self.sample_spacing_m) + 1
        return torch.arange(count, device=device, dtype=torch.float32) * self.sample_spacing_m


class SimulatedTofTrigger:
    """Per-row riser detection and one-shot stair submissions."""

    def __init__(self, num_envs, *, min_height_m, max_height_m, config=TofTriggerConfig(), seed=0, device='cpu'):
        if (type(num_envs) is not int or num_envs < 1 or not isinstance(config, TofTriggerConfig)
                or not 0 < min_height_m < max_height_m):
            raise ValueError('ToF trigger requires rows, the stair height band and a parsed configuration')
        self.num_envs, self.config, self.device = num_envs, config, torch.device(device)
        self.min_height_m, self.max_height_m = float(min_height_m), float(max_height_m)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.distances_m = config.sample_distances(self.device)
        self._window = round(config.edge_window_m / config.sample_spacing_m)
        self.armed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.trigger_distance_m = torch.zeros(num_envs, device=self.device)
        self.countdown = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.measured_height_m = torch.zeros(num_envs, device=self.device)
        self.counts = torch.zeros(3, dtype=torch.long, device=self.device)  # detected, submitted, early draws

    def _uniform(self, count, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(count, generator=self.generator, device=self.device)

    @torch.no_grad()
    def arm(self, mask):
        """Arm ``mask`` rows for their next riser and draw its trigger distance."""
        early = torch.rand(self.num_envs, generator=self.generator, device=self.device) < self.config.early_probability
        distance = torch.where(early, self._uniform(self.num_envs, self.config.early_distance_m),
                               self._uniform(self.num_envs, self.config.trigger_distance_m))
        self.trigger_distance_m.copy_(torch.where(mask, distance, self.trigger_distance_m))
        self.armed |= mask
        self.countdown.masked_fill_(mask, -1)
        self.counts[2] += (mask & early).sum()

    def reset(self, ids):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.arm(mask)

    def detect(self, height_m, valid):
        """First riser along each [n,k] profile: found [n], distance [n], step height [n].

        ``height_m`` holds terrain heights at ``distances_m``; column 0 is the
        ground under the body. Invalid samples end the usable beam.
        """
        usable = valid.cumprod(-1).bool()
        w = self._window
        rise = height_m[:, w:] - height_m[:, :-w]
        edge = usable[:, w:] & (rise >= self.min_height_m)
        found = edge.any(-1)
        first = edge.float().argmax(-1)
        # The riser lies inside the window; its top is read at the window end.
        top = (first + w).clamp_max(height_m.shape[1] - 1)
        distance = self.distances_m[first] + self.config.sample_spacing_m * w / 2
        step = height_m.gather(1, top[:, None])[:, 0] - height_m[:, 0]
        return found, torch.where(found, distance, float('inf')), torch.where(found, step, 0.)

    @torch.no_grad()
    def step(self, height_m, valid, *, idle):
        """Advance one policy interval; returns [n] fire mask and submitted heights.

        ``idle`` rows may receive a request now (no active stair or jump, a
        ready estimate). A due request waits for idle instead of being dropped.
        """
        found, distance, step = self.detect(height_m, valid)
        detected = (self.armed & found & (distance <= self.trigger_distance_m)
                    & (step >= self.min_height_m) & (step <= self.max_height_m + 1e-3))
        low, high = self.config.latency_intervals
        latency = torch.randint(low, high + 1, (self.num_envs,), generator=self.generator, device=self.device)
        noise = (torch.rand(self.num_envs, generator=self.generator, device=self.device) * 2 - 1) \
            * self.config.height_noise_m
        self.measured_height_m.copy_(torch.where(detected, step + noise, self.measured_height_m))
        self.countdown.copy_(torch.where(detected, latency, self.countdown))
        self.armed &= ~detected
        self.counts[0] += detected.sum()
        pending = self.countdown >= 0
        fire = pending & (self.countdown == 0) & idle
        self.countdown.copy_(torch.where(pending & (self.countdown > 0), self.countdown - 1, self.countdown))
        self.countdown.masked_fill_(fire, -1)
        self.counts[1] += fire.sum()
        height = self.measured_height_m.clamp(self.min_height_m, self.max_height_m)
        return fire, torch.where(fire, height, torch.zeros_like(height))

    def metrics(self):
        detected, submitted, early = self.counts.tolist()
        return {'risers_detected': detected, 'requests_submitted': submitted, 'early_trigger_draws': early,
                'trigger_distance_m': list(self.config.trigger_distance_m),
                'early': {'probability': self.config.early_probability,
                          'distance_m': list(self.config.early_distance_m)},
                'height_noise_m': self.config.height_noise_m,
                'latency_intervals': list(self.config.latency_intervals)}

    def state_dict(self):
        return {'format': 'simulated_tof_trigger_v1', 'rng': self.generator.get_state().clone()}

    def restore_after_physical_reset(self, state):
        if not isinstance(state, dict) or set(state) != {'format', 'rng'} or state['format'] != 'simulated_tof_trigger_v1':
            raise ValueError('ToF trigger checkpoint contract differs')
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].cpu())
        self.generator = generator
        self.armed.zero_()
        self.countdown.fill_(-1)

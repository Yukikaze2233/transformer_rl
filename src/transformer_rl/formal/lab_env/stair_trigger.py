"""Simulated stair request source: one request per detected riser.

A beam samples the terrain height profile ahead of the body along the
direction of travel. A riser is a rise of at least the stair minimum within
one edge window (slopes and roughness stay below it). Each riser draws a
trigger distance; when the riser comes within it, the measured step (riser
top minus current ground, with sensor noise) is submitted after a short
latency.

Request sources: driving forward, a body-mounted forward ToF reports the
riser; a share of risers is instead requested early by the operator's manual
stair key; driving backward the forward ToF cannot see the riser, so the
operator requests it. This is an idealized request generator: it queries the
terrain from above with the true root pose and does not model the ToF mount,
field of view or occlusion.

Each riser is identified by its world position. A request is re-validated
when its latency elapses (same riser still ahead along the same travel
direction) and dropped as stale otherwise. After a request, the same riser can
be requested again only once it has left the beam window, so a failed attempt
cannot re-trigger in place; a different riser (the second level of a
two-level course) can be requested at once.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch

SOURCES = ('tof', 'manual_early', 'manual_reverse')


class TofBeam(NamedTuple):
    """One interval's beam: [n,k] heights/validity along ``direction_xy`` from ``origin_xy``."""

    height_m: torch.Tensor
    valid: torch.Tensor
    origin_xy: torch.Tensor     # [n,2] world position of the beam start (the root)
    direction_xy: torch.Tensor  # [n,2] unit travel direction; zero while standing
    forward: torch.Tensor       # [n] travel along the body +x axis


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
    # Two detections closer than this (world xy) are the same riser; the second
    # level of the RMUC central steps is about 0.6 m beyond the first.
    riser_identity_m: float = .2

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
                                 self.minimum_travel_speed_m_s, self.riser_identity_m))
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
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(num_envs, *shape, dtype=dtype, device=self.device)
        self.armed = zeros(dtype=torch.bool)
        self.early = zeros(dtype=torch.bool)
        self.trigger_distance_m = zeros()
        self.countdown = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.measured_height_m = zeros()
        self.pending_riser_xy = zeros(2)
        self.pending_direction_xy = zeros(2)
        self.pending_source = zeros(dtype=torch.long)
        self.last_riser_xy = torch.full((num_envs, 2), float('inf'), device=self.device)
        self.riser_cleared = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        # detected, stale-dropped, then submitted per source, accepted, rejected
        self.counts = torch.zeros(4 + len(SOURCES), dtype=torch.long, device=self.device)

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
        self.early.copy_(torch.where(mask, early, self.early))
        self.armed |= mask
        self.countdown.masked_fill_(mask, -1)

    def reset(self, ids):
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.arm(mask)
        self.last_riser_xy[ids] = float('inf')
        self.riser_cleared[ids] = True

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
    def step(self, beam: TofBeam, *, idle):
        """Advance one policy interval; returns [n] fire mask and submitted heights.

        ``idle`` rows may receive a request now (no active stair or jump, a
        ready estimate). A due request waits for idle, then is re-validated.
        """
        found, distance, step = self.detect(beam.height_m, beam.valid)
        riser_xy = beam.origin_xy + beam.direction_xy * distance.clamp_max(1e3)[:, None]
        same = found & ((riser_xy - self.last_riser_xy).norm(dim=-1) < self.config.riser_identity_m)
        self.riser_cleared |= ~same
        in_band = (step >= self.min_height_m) & (step <= self.max_height_m + 1e-3)
        detected = (self.armed & found & self.riser_cleared & (distance <= self.trigger_distance_m) & in_band)
        low, high = self.config.latency_intervals
        latency = torch.randint(low, high + 1, (self.num_envs,), generator=self.generator, device=self.device)
        noise = (torch.rand(self.num_envs, generator=self.generator, device=self.device) * 2 - 1) \
            * self.config.height_noise_m
        source = torch.where(beam.forward, torch.where(self.early, 1, 0), 2)
        self.measured_height_m.copy_(torch.where(detected, step + noise, self.measured_height_m))
        self.pending_riser_xy.copy_(torch.where(detected[:, None], riser_xy, self.pending_riser_xy))
        self.pending_direction_xy.copy_(torch.where(detected[:, None], beam.direction_xy, self.pending_direction_xy))
        self.pending_source.copy_(torch.where(detected, source, self.pending_source))
        self.countdown.copy_(torch.where(detected, latency, self.countdown))
        self.armed &= ~detected
        self.counts[0] += detected.sum()
        due = (self.countdown == 0) & idle
        # Still the same riser ahead, along the same travel direction?
        current = (found & in_band & ((riser_xy - self.pending_riser_xy).norm(dim=-1) < self.config.riser_identity_m)
                   & ((beam.direction_xy * self.pending_direction_xy).sum(-1) > .9))
        fire, stale = due & current, due & ~current
        pending = self.countdown >= 0
        self.countdown.copy_(torch.where(pending & (self.countdown > 0), self.countdown - 1, self.countdown))
        self.countdown.masked_fill_(due, -1)
        self.counts[1] += stale.sum()
        self.counts[2:2 + len(SOURCES)] += torch.bincount(self.pending_source[fire], minlength=len(SOURCES))
        self.last_riser_xy.copy_(torch.where(fire[:, None], self.pending_riser_xy, self.last_riser_xy))
        self.riser_cleared &= ~fire
        self.arm(stale)
        height = self.measured_height_m.clamp(self.min_height_m, self.max_height_m)
        return fire, torch.where(fire, height, torch.zeros_like(height))

    @torch.no_grad()
    def record_responses(self, fired, accepted):
        """Count the protocol responses to this interval's submissions."""
        self.counts[2 + len(SOURCES)] += (fired & accepted).sum()
        self.counts[3 + len(SOURCES)] += (fired & ~accepted).sum()

    def metrics(self):
        counts = self.counts.tolist()
        return {'risers_detected': counts[0], 'stale_requests_dropped': counts[1],
                'requests_submitted': dict(zip(SOURCES, counts[2:2 + len(SOURCES)])),
                'requests_accepted': counts[2 + len(SOURCES)], 'requests_rejected': counts[3 + len(SOURCES)],
                'trigger_distance_m': list(self.config.trigger_distance_m),
                'early': {'probability': self.config.early_probability,
                          'distance_m': list(self.config.early_distance_m)},
                'height_noise_m': self.config.height_noise_m,
                'latency_intervals': list(self.config.latency_intervals),
                'scope': 'idealized request generator: true root pose, terrain queried from above; '
                         'no ToF mount, field of view or occlusion model'}

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
        self.last_riser_xy.fill_(float('inf'))
        self.riser_cleared.fill_(True)

"""Private batched request protocol shared by collection and deployment.

The state machine selects an observable task condition, never a policy or an
actuator action. Estimated completion is a protocol event, not physical success
or permission to pay a training reward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import IntEnum
import math

import torch


class RequestResponse(IntEnum):
    NONE = 0
    ACCEPTED = 1
    REPLAY = 2
    BUSY = 3
    INVALID = 4
    NOT_READY = 5
    INVALID_CLOCK = 6
    CANCELLED = 7


class RequestEnd(IntEnum):
    NONE = 0
    ESTIMATED_COMPLETE = 1
    CANCELLED = 2
    SAFETY_ABORT = 3
    TIMEOUT = 4
    RESET = 5
    COLD_RESTORE = 6


@dataclass(frozen=True)
class RequestProtocolConfig:
    policy_dt_s: float = 0.01
    estimate_max_age_s: float = 0.03
    height_min_m: float = 0.03
    height_max_m: float = 0.40
    contact_ground_probability: float = 0.8
    contact_air_probability: float = 0.2
    flight_confirm_ticks: int = 2
    cancel_protection_after_s: float = 0.10
    stable_landing_s: float = 2.0
    stable_vertical_speed_m_s: float = 0.15
    stable_tilt_rad: float = 0.25
    stable_angular_speed_rad_s: float = 0.5
    timeout_s: float = 6.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'flight_confirm_ticks':
                if type(value) is not int or value < 1:
                    raise ValueError('flight confirmation requires a positive tick count')
            elif type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name} must be finite and positive')
        if (self.estimate_max_age_s < self.policy_dt_s or self.height_min_m > self.height_max_m
                or not 0 < self.contact_air_probability < self.contact_ground_probability < 1
                or self.timeout_s <= self.stable_landing_s + self.cancel_protection_after_s):
            raise ValueError('inconsistent request protocol bounds')


@dataclass(frozen=True)
class CompletedEstimate:
    """Prior-tick SI estimates; vertical speed uses gravity-aligned WORLD-Z."""

    velocity_z_m_s: torch.Tensor
    wheel_contact_probability: torch.Tensor
    tilt_rad: torch.Tensor
    angular_speed_rad_s: torch.Tensor
    available_at_s: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def _from_actor_observation(cls, frame, raw_state, timestamp):
        """Decode the shared 35D sensor normalization, without privileged data."""
        if (not isinstance(frame, torch.Tensor) or frame.ndim != 2 or frame.shape[1] != 35
                or not isinstance(raw_state, torch.Tensor) or raw_state.shape != (len(frame), 8)
                or frame.dtype != torch.float32 or raw_state.dtype != frame.dtype
                or raw_state.device != frame.device or not isinstance(timestamp, torch.Tensor)
                or timestamp.shape != (len(frame),) or timestamp.dtype != torch.float64
                or timestamp.device != frame.device):
            raise ValueError('completed estimate requires float32 actor/state rows and their float64 clock')
        gravity = frame[:, 7:10]
        norm = gravity.norm(dim=-1)
        unit = gravity / norm[:, None].clamp_min(1e-8)
        # The actor carries gyro * 0.5; the protocol compares SI rad/s.
        angular_speed = (frame[:, 4:7] * 2).norm(dim=-1)
        return cls(-(unit * raw_state[:, :3]).sum(-1), raw_state[:, 6:8].sigmoid(),
                   (-unit[:, 2]).clamp(-1, 1).acos(), angular_speed, timestamp.clone(),
                   torch.isfinite(raw_state).all(-1) & torch.isfinite(frame[:, 4:10]).all(-1)
                   & (norm > 1e-6))


@dataclass(frozen=True)
class RequestCondition:
    jump_mode: torch.Tensor
    height_m: torch.Tensor
    clock_s: torch.Tensor
    active_id: torch.Tensor
    response: torch.Tensor
    finished_id: torch.Tensor
    end_reason: torch.Tensor
    safety_required: torch.Tensor


class RequestProtocol:
    """Monotonic request IDs per environment, preserved across physical resets.

    A newly seen ID is consumed even if rejected. Retrying after BUSY/NOT_READY
    requires a new ID; replaying an accepted, rejected or finished ID never
    executes again. This bounds deduplication storage without forgetting old IDs.
    Tensor operations are batched and contain no GPU scalar logging sync.
    """

    def __init__(self, num_envs, device='cpu', config=RequestProtocolConfig()):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, RequestProtocolConfig):
            raise ValueError('protocol requires positive environment count and parsed configuration')
        self.num_envs, self.device, self.config = num_envs, torch.device(device), config
        self.max_seen_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.active_id = torch.full_like(self.max_seen_id, -1)
        self.height_m = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.accepted_at_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.last_time_s = torch.full_like(self.accepted_at_s, -float('inf'))
        self.cancelled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.flight_seen = torch.zeros_like(self.cancelled)
        self.air_ticks = torch.zeros_like(self.max_seen_id)
        self.landing_since_s = torch.full_like(self.accepted_at_s, -float('inf'))

    def _validate_tick(self, now_s, estimate, request_id, requested_height_m, cancel_id,
                       allow_new, safety_abort):
        if not isinstance(estimate, CompletedEstimate):
            raise TypeError('request detection requires a CompletedEstimate')
        for name, value, shape, dtype in (
            ('now_s', now_s, (self.num_envs,), torch.float64),
            ('request_id', request_id, (self.num_envs,), torch.long),
            ('height', requested_height_m, (self.num_envs,), torch.float32),
            ('cancel_id', cancel_id, (self.num_envs,), torch.long),
            ('allow_new', allow_new, (self.num_envs,), torch.bool),
            ('safety_abort', safety_abort, (self.num_envs,), torch.bool),
            ('velocity_z', estimate.velocity_z_m_s, (self.num_envs,), torch.float32),
            ('contacts', estimate.wheel_contact_probability, (self.num_envs, 2), torch.float32),
            ('tilt', estimate.tilt_rad, (self.num_envs,), torch.float32),
            ('angular_speed', estimate.angular_speed_rad_s, (self.num_envs,), torch.float32),
            ('available_at', estimate.available_at_s, (self.num_envs,), torch.float64),
            ('estimate_valid', estimate.valid, (self.num_envs,), torch.bool),
        ):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                    or value.device != self.device):
                raise ValueError(f'{name} requires {shape} / {dtype} on {self.device}')

    @torch.no_grad()
    def step(self, *, now_s, estimate, request_id, requested_height_m, cancel_id, allow_new, safety_abort):
        self._validate_tick(now_s, estimate, request_id, requested_height_m, cancel_id, allow_new, safety_abort)
        cfg = self.config
        clock_valid = torch.isfinite(now_s) & (now_s > self.last_time_s)
        now = torch.where(clock_valid, now_s, self.last_time_s)
        consecutive_tick = (~torch.isfinite(self.last_time_s)
                            | ((now - self.last_time_s - cfg.policy_dt_s).abs() <= 1e-6))
        age = now - estimate.available_at_s
        contacts = estimate.wheel_contact_probability
        finite = (torch.isfinite(contacts).all(-1) & torch.isfinite(estimate.velocity_z_m_s)
                  & torch.isfinite(estimate.tilt_rad) & torch.isfinite(estimate.angular_speed_rad_s)
                  & (contacts >= 0).all(-1) & (contacts <= 1).all(-1)
                  & (estimate.tilt_rad >= 0) & (estimate.angular_speed_rad_s >= 0))
        # Strictly older excludes a same-tick estimate and an algebraic loop.
        fresh = (estimate.valid & finite & torch.isfinite(age) & (age > 0)
                 & (age <= cfg.estimate_max_age_s + 1e-9) & clock_valid)
        grounded = fresh & (contacts >= cfg.contact_ground_probability).all(-1)
        airborne = fresh & (contacts <= cfg.contact_air_probability).all(-1)
        stable = (grounded & (estimate.velocity_z_m_s.abs() <= cfg.stable_vertical_speed_m_s)
                  & (estimate.tilt_rad <= cfg.stable_tilt_rad)
                  & (estimate.angular_speed_rad_s <= cfg.stable_angular_speed_rad_s))
        was_active = self.active_id >= 0
        response = torch.zeros_like(self.active_id)
        response = torch.where(~clock_valid, int(RequestResponse.INVALID_CLOCK), response)
        invalid_id = (request_id < -1) | (cancel_id < -1)
        response = torch.where(invalid_id & clock_valid, int(RequestResponse.INVALID), response)
        new_id = (request_id >= 0) & (request_id > self.max_seen_id) & clock_valid
        replay = (request_id >= 0) & ~new_id & clock_valid
        response = torch.where(replay, int(RequestResponse.REPLAY), response)
        request_valid = (torch.isfinite(requested_height_m) & (requested_height_m >= cfg.height_min_m)
                         & (requested_height_m <= cfg.height_max_m))
        cancelled_on_arrival = new_id & (cancel_id == request_id)
        accepted = (new_id & ~was_active & request_valid & stable & consecutive_tick & allow_new & ~safety_abort
                    & ~cancelled_on_arrival & ~invalid_id)
        response = torch.where(new_id, int(RequestResponse.NOT_READY), response)
        response = torch.where(new_id & was_active, int(RequestResponse.BUSY), response)
        response = torch.where(new_id & ~request_valid, int(RequestResponse.INVALID), response)
        response = torch.where(new_id & invalid_id, int(RequestResponse.INVALID), response)
        response = torch.where(cancelled_on_arrival, int(RequestResponse.CANCELLED), response)
        response = torch.where(accepted, int(RequestResponse.ACCEPTED), response)
        self.max_seen_id.copy_(torch.where(new_id, request_id, self.max_seen_id))
        self.active_id.copy_(torch.where(accepted, request_id, self.active_id))
        self.height_m.copy_(torch.where(accepted, requested_height_m, self.height_m))
        self.accepted_at_s.copy_(torch.where(accepted, now, self.accepted_at_s))
        self.cancelled.masked_fill_(accepted, False)
        self.flight_seen.masked_fill_(accepted, False)
        self.air_ticks.masked_fill_(accepted, 0)
        self.landing_since_s.masked_fill_(accepted, -float('inf'))
        active = self.active_id >= 0
        elapsed = torch.where(active, now - self.accepted_at_s, 0)
        cancel = active & (cancel_id == self.active_id) & clock_valid
        before_takeoff = (cancel & grounded & ~self.flight_seen
                          & (elapsed < cfg.cancel_protection_after_s))
        self.cancelled.logical_or_(cancel)
        air_ticks = torch.where(active & airborne, torch.where(consecutive_tick, self.air_ticks + 1, 1), 0)
        self.air_ticks.copy_(torch.where(clock_valid, air_ticks, self.air_ticks))
        self.flight_seen.logical_or_(active & (self.air_ticks >= cfg.flight_confirm_ticks))
        eligible_landing = active & stable & consecutive_tick & (self.flight_seen | self.cancelled)
        beginning = eligible_landing & ~torch.isfinite(self.landing_since_s)
        landing_start = torch.where(beginning, now, self.landing_since_s)
        landing_start = torch.where(eligible_landing, landing_start, -float('inf'))
        self.landing_since_s.copy_(torch.where(clock_valid, landing_start, self.landing_since_s))
        landed = eligible_landing & (now - self.landing_since_s >= cfg.stable_landing_s - 1e-9)
        timed_out = active & clock_valid & (elapsed >= cfg.timeout_s)
        aborted = active & safety_abort & clock_valid
        finished = before_takeoff | landed | timed_out | aborted
        finished_id = torch.where(finished, self.active_id, -1)
        end = torch.zeros_like(self.active_id)
        end = torch.where(landed, int(RequestEnd.ESTIMATED_COMPLETE), end)
        end = torch.where(finished & self.cancelled, int(RequestEnd.CANCELLED), end)
        end = torch.where(timed_out, int(RequestEnd.TIMEOUT), end)
        end = torch.where(aborted, int(RequestEnd.SAFETY_ABORT), end)
        safety = (~clock_valid) | (active & (~fresh | ~consecutive_tick)) | timed_out | aborted
        self._clear(finished)
        self.last_time_s.copy_(now)
        active = self.active_id >= 0
        return RequestCondition(active.clone(), torch.where(active, self.height_m, 0),
                                torch.where(active, now - self.accepted_at_s, 0).float(),
                                self.active_id.clone(), response, finished_id, end, safety)

    def _clear(self, mask):
        self.active_id.masked_fill_(mask, -1)
        self.height_m.masked_fill_(mask, 0)
        self.accepted_at_s.masked_fill_(mask, 0)
        self.cancelled.masked_fill_(mask, False)
        self.flight_seen.masked_fill_(mask, False)
        self.air_ticks.masked_fill_(mask, 0)
        self.landing_since_s.masked_fill_(mask, -float('inf'))

    @torch.no_grad()
    def reset(self, mask, now_s):
        if (mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device
                or now_s.shape != mask.shape or now_s.dtype != torch.float64 or now_s.device != self.device
                or not torch.isfinite(now_s).all()):
            raise ValueError('reset requires a bool row mask and finite float64 clock')
        aborted_id = torch.where(mask & (self.active_id >= 0), self.active_id, -1)
        self._clear(mask)
        self.last_time_s[mask] = now_s[mask] - self.config.policy_dt_s
        return aborted_id

    def state_dict(self):
        names = ('max_seen_id', 'active_id', 'height_m', 'accepted_at_s', 'last_time_s',
                 'cancelled', 'flight_seen', 'air_ticks', 'landing_since_s')
        return {'configuration': asdict(self.config),
                'tensors': {name: getattr(self, name).detach().clone() for name in names}}

    @torch.no_grad()
    def restore_after_physical_reset(self, state, now_s):
        """Cold recovery preserves deduplication and aborts every physical skill."""
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or state['configuration'] != expected['configuration']
                or not isinstance(state['tensors'], dict)
                or state['tensors'].keys() != expected['tensors'].keys()):
            raise ValueError('request checkpoint configuration or state fields differ')
        prepared = {}
        for name, reference in expected['tensors'].items():
            tensor = state['tensors'][name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f'invalid saved request tensor: {name}')
            prepared[name] = tensor.detach().to(self.device).clone()
        if ((prepared['max_seen_id'] < -1).any() or (prepared['active_id'] < -1).any()
                or (prepared['active_id'] > prepared['max_seen_id']).any()):
            raise ValueError('invalid saved request ID ordering')
        active = prepared['active_id'] >= 0
        height = prepared['height_m']
        accepted_at = prepared['accepted_at_s']
        last_time = prepared['last_time_s']
        landing = prepared['landing_since_s']
        if (not torch.isfinite(height).all() or not torch.isfinite(accepted_at).all()
                or not (torch.isfinite(last_time) | torch.isneginf(last_time)).all()
                or not (torch.isfinite(landing) | torch.isneginf(landing)).all()
                or (active & ((height < self.config.height_min_m) | (height > self.config.height_max_m)
                              | (accepted_at > last_time))).any()
                or (~active & ((height != 0) | (accepted_at != 0) | prepared['cancelled']
                               | prepared['flight_seen'] | (prepared['air_ticks'] != 0)
                               | torch.isfinite(landing))).any()
                or (prepared['air_ticks'] < 0).any()
                or (prepared['air_ticks'] > math.ceil(self.config.timeout_s / self.config.policy_dt_s) + 1).any()):
            raise ValueError('invalid saved request phase state')
        if (now_s.shape != (self.num_envs,) or now_s.dtype != torch.float64 or now_s.device != self.device
                or not torch.isfinite(now_s).all()):
            raise ValueError('cold recovery requires a finite current float64 clock')
        aborted_id = prepared['active_id'].clone()
        # Other physical-phase fields are deliberately not restored or replayed.
        self.max_seen_id.copy_(prepared['max_seen_id'])
        self._clear(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.last_time_s.copy_(now_s - self.config.policy_dt_s)
        return aborted_id

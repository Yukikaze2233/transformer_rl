"""Private physical jump evidence, separate from the deployable request FSM.

Only explicit PRE-reset world-Z, actual collider clearance and own-terrain
normal loads may establish flight. This tracker never accepts a request on the
policy's behalf, changes a command, starts physics, or grants qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import math

import torch

from .protocol import RequestCondition, RequestEnd, RequestProtocol, RequestResponse


class SkillPhase(IntEnum):
    IDLE = 0
    PREPARING = 1
    FLIGHT = 2
    LANDING = 3
    SETTLING = 4


class SkillOutcome(IntEnum):
    NONE = 0
    PHYSICAL_SUCCESS = 1
    HEIGHT_MISSED = 2
    BAD_START = 3
    UNHEALTHY = 4
    CANCELLED = 5
    TIMEOUT = 6
    TRUE_FAILURE = 7
    RESET = 8
    SUPERSEDED = 9
    SCRIPT_ASSISTED = 10
    SAFETY_ABORT = 11


@dataclass(frozen=True)
class SkillEvidenceConfig:
    physics_dt_s: float = .005
    flight_confirm_s: float = .01
    landing_confirm_s: float = .015
    stable_landing_s: float = 2.
    airborne_load_max_n: float = 2.
    grounded_load_min_n: float = 5.
    minimum_clearance_m: float = .005
    support_gap_max_m: float = .01
    support_penetration_max_m: float = .005
    height_tolerance_m: float = .03
    small_jump_tolerance_fraction: float = .5
    start_height_tolerance_m: float = .015
    stable_vertical_speed_m_s: float = .15
    stable_tilt_rad: float = .25
    stable_angular_speed_rad_s: float = .5

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in vars(self).values())
                or self.physics_dt_s not in (.005, .001) or not 0 < self.small_jump_tolerance_fraction <= .5
                or self.airborne_load_max_n >= self.grounded_load_min_n):
            raise ValueError('physical skill evidence requires explicit positive timing/contact/height bounds')


@dataclass(frozen=True)
class SkillMeasurement:
    physical_tick: int
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    root_world_z_m: torch.Tensor
    ground_world_z_m: torch.Tensor
    vertical_velocity_world_m_s: torch.Tensor
    wheel_normal_load_n: torch.Tensor
    wheel_clearance_m: torch.Tensor
    tilt_rad: torch.Tensor
    angular_speed_rad_s: torch.Tensor
    policy_owned: torch.Tensor
    healthy: torch.Tensor
    true_failure: torch.Tensor
    velocity_reference: str = 'base_link_origin_world_z'
    clearance_reference: str = 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint'


@dataclass(frozen=True)
class SkillReceipt:
    accepted_id: torch.Tensor
    finished_id: torch.Tensor
    outcome: torch.Tensor
    observed_at_s: torch.Tensor


@dataclass(frozen=True)
class SkillEvidence:
    physical_tick: int
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    physical_success_request_id: torch.Tensor
    finished_request_id: torch.Tensor
    finished_at_s: torch.Tensor
    outcome: torch.Tensor
    entire_skill_policy_owned: torch.Tensor
    entire_skill_healthy: torch.Tensor
    phase: torch.Tensor
    active_id: torch.Tensor
    requested_rise_m: torch.Tensor
    apex_rise_m: torch.Tensor
    height_tolerance_m: torch.Tensor
    supported_start: torch.Tensor
    flight_seen: torch.Tensor
    stable_landing_s: torch.Tensor


class PhysicalJumpEvidence:
    """Observe every physics tick; finalize positive evidence at policy endpoints.

    Protocol estimated completion is deliberately ignored for success. An early
    new acceptance supersedes the still-unfinished physical attempt explicitly.
    Cancel/reset/timeout records never become success, and IDs remain retired.
    """

    def __init__(self, protocol: RequestProtocol, config=SkillEvidenceConfig()):
        if (not isinstance(protocol, RequestProtocol) or not isinstance(config, SkillEvidenceConfig)
                or protocol.config.policy_dt_s != .01
                or config.stable_landing_s != protocol.config.stable_landing_s
                or config.stable_vertical_speed_m_s != protocol.config.stable_vertical_speed_m_s
                or config.stable_tilt_rad != protocol.config.stable_tilt_rad
                or config.stable_angular_speed_rad_s != protocol.config.stable_angular_speed_rad_s):
            raise ValueError('physical evidence and deployable request protocol must share stable-landing criteria')
        self.protocol, self.config = protocol, config
        self.num_envs, self.device = protocol.num_envs, protocol.device
        self.substeps = round(protocol.config.policy_dt_s / config.physics_dt_s)
        self._tick, self._drained_tick = 0, 0
        n, device = self.num_envs, self.device
        self._episode = torch.full((n,), -1, dtype=torch.long, device=device)
        self._control = torch.zeros(n, dtype=torch.long, device=device)
        self._time = torch.zeros(n, dtype=torch.float64, device=device)
        self._cache_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._sample = None
        self._retired = protocol.max_seen_id.detach().clone()
        self.active_id = torch.full_like(self._episode, -1)
        self.phase = torch.zeros_like(self._episode)
        self._rise = torch.zeros(n, device=device)
        self._baseline = torch.zeros(n, device=device)
        self._accepted_at = torch.zeros(n, dtype=torch.float64, device=device)
        self._air_time = torch.zeros_like(self._accepted_at)
        self._ground_time = torch.zeros_like(self._accepted_at)
        self._stable_time = torch.zeros_like(self._accepted_at)
        self._candidate_apex = torch.full((n,), -float('inf'), device=device)
        self._apex = torch.full_like(self._candidate_apex, -float('inf'))
        self._flight_seen = torch.zeros(n, dtype=torch.bool, device=device)
        self._supported_start = torch.zeros_like(self._flight_seen)
        self._whole_owned = torch.zeros_like(self._flight_seen)
        self._whole_healthy = torch.zeros_like(self._flight_seen)
        self._cancelled = torch.zeros_like(self._flight_seen)
        self._pending_id = torch.full_like(self.active_id, -1)
        self._pending_outcome = torch.zeros_like(self.active_id)
        self._pending_at = torch.zeros_like(self._accepted_at)

    def _tensor(self, name, value, width=None, dtype=torch.float32):
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device or value.dtype != dtype):
            raise ValueError(name + ' differs from physical evidence rows/device/dtype')

    def _grounded(self, sample):
        c = self.config
        return ((sample.wheel_normal_load_n >= c.grounded_load_min_n).all(-1)
                & (sample.wheel_clearance_m <= c.support_gap_max_m).all(-1)
                & (sample.wheel_clearance_m >= -c.support_penetration_max_m).all(-1))

    def _stable(self, sample):
        c = self.config
        return (self._grounded(sample) & (sample.vertical_velocity_world_m_s.abs() <= c.stable_vertical_speed_m_s)
                & (sample.tilt_rad <= c.stable_tilt_rad) & (sample.angular_speed_rad_s <= c.stable_angular_speed_rad_s))

    @torch.no_grad()
    def reset(self, mask, now_s, episode_id, control_epoch, *, physical_tick):
        for name, value, dtype in (('mask', mask, torch.bool), ('now', now_s, torch.float64),
                                    ('episode', episode_id, torch.long), ('control', control_epoch, torch.long)):
            self._tensor(name, value, dtype=dtype)
        if (type(physical_tick) is not int or physical_tick < 0 or physical_tick % self.substeps
                or physical_tick != self._tick or self._drained_tick != self._tick
                or ((self._episode < 0) & ~mask).any() or (mask & (self._pending_id >= 0)).any()
                or (~torch.isfinite(now_s) | (now_s < 0)
                    | (now_s - physical_tick * self.config.physics_dt_s).abs().gt(1e-8)
                    | (mask & ((episode_id <= self._episode) | (control_epoch < self._control)))).any()):
            raise ValueError('physical evidence reset requires its drained native boundary and new selected episodes')
        ended = torch.where(mask, self.active_id, -1)
        outcome = torch.where(ended >= 0, int(SkillOutcome.RESET), 0)
        self.active_id[mask], self.phase[mask] = -1, int(SkillPhase.IDLE)
        self._episode[mask], self._control[mask], self._time[mask] = episode_id[mask], control_epoch[mask], now_s[mask]
        self._cache_valid[mask] = False
        self._retired = torch.maximum(self._retired, self.protocol.max_seen_id)
        return SkillReceipt(torch.full_like(ended, -1), ended.clone(), outcome, now_s.clone())

    @torch.no_grad()
    def observe(self, sample: SkillMeasurement):
        if (not isinstance(sample, SkillMeasurement) or type(sample.physical_tick) is not int
                or sample.physical_tick != self._tick + 1
                or self._tick % self.substeps == 0 and self._drained_tick != self._tick):
            raise ValueError('physical evidence requires every next native tick and one drain per policy interval')
        if (sample.velocity_reference != 'base_link_origin_world_z'
                or sample.clearance_reference != 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint'):
            raise ValueError('physical evidence requires actual link-origin velocity and collision clearance')
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, width in (('root_world_z_m', None), ('ground_world_z_m', None),
                            ('vertical_velocity_world_m_s', None), ('wheel_normal_load_n', 2),
                            ('wheel_clearance_m', 2), ('tilt_rad', None), ('angular_speed_rad_s', None)):
            value = getattr(sample, name)
            self._tensor(name, value, width)
            invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        for name, dtype in (('timestamp_s', torch.float64), ('episode_id', torch.long), ('control_epoch', torch.long),
                            ('policy_owned', torch.bool), ('healthy', torch.bool), ('true_failure', torch.bool)):
            self._tensor(name, getattr(sample, name), dtype=dtype)
        c = self.config
        invalid |= (~torch.isfinite(sample.timestamp_s)
                    | (sample.timestamp_s - sample.physical_tick * c.physics_dt_s).abs().gt(1e-8)
                    | (sample.timestamp_s - self._time - c.physics_dt_s).abs().gt(1e-8)
                    | (sample.episode_id != self._episode) | (self._episode < 0)
                    | (sample.control_epoch < self._control) | (sample.wheel_normal_load_n < 0).any(-1)
                    | (sample.tilt_rad < 0) | (sample.tilt_rad > math.pi) | (sample.angular_speed_rad_s < 0))
        if invalid.any():
            raise ValueError('physical evidence sample has invalid identity, units, clock or finite values')
        active = self.active_id >= 0
        air = ((sample.wheel_normal_load_n <= c.airborne_load_max_n).all(-1)
               & (sample.wheel_clearance_m >= c.minimum_clearance_m).all(-1))
        grounded = self._grounded(sample)
        self._whole_owned &= ~active | sample.policy_owned
        self._whole_healthy &= ~active | sample.healthy
        self._air_time = torch.where(active & air, self._air_time + c.physics_dt_s, 0)
        candidate = torch.maximum(self._candidate_apex, sample.root_world_z_m)
        self._candidate_apex = torch.where(active & air, candidate, -float('inf'))
        confirmed = active & ~self._flight_seen & (self._air_time >= c.flight_confirm_s - 1e-9)
        self._apex = torch.where(confirmed, self._candidate_apex, self._apex)
        self._flight_seen |= confirmed
        self._apex = torch.where(active & self._flight_seen & air,
                                  torch.maximum(self._apex, sample.root_world_z_m), self._apex)
        self._ground_time = torch.where(active & self._flight_seen & grounded, self._ground_time + c.physics_dt_s, 0)
        landed = active & self._flight_seen & (self._ground_time >= c.landing_confirm_s - 1e-9)
        stable = landed & self._stable(sample)
        self._stable_time = torch.where(stable, self._stable_time + c.physics_dt_s, 0)
        self.phase = torch.where(active & self._flight_seen, int(SkillPhase.FLIGHT), self.phase)
        self.phase = torch.where(active & self._flight_seen & grounded, int(SkillPhase.LANDING), self.phase)
        self.phase = torch.where(stable, int(SkillPhase.SETTLING), self.phase)
        failed = active & sample.true_failure
        self._pending_id = torch.where(failed, self.active_id, self._pending_id)
        self._pending_outcome = torch.where(failed, int(SkillOutcome.TRUE_FAILURE), self._pending_outcome)
        self._pending_at = torch.where(failed, sample.timestamp_s, self._pending_at)
        self.active_id[failed] = -1
        self._time, self._control, self._tick = sample.timestamp_s.clone(), sample.control_epoch.clone(), sample.physical_tick
        self._sample = SkillMeasurement(**{field.name: (getattr(sample, field.name).detach().clone()
                                                        if isinstance(getattr(sample, field.name), torch.Tensor)
                                                        else getattr(sample, field.name)) for field in fields(sample)})
        self._cache_valid.fill_(True)

    @torch.no_grad()
    def drain_policy_evidence(self):
        if self._sample is None or self._tick % self.substeps or self._drained_tick == self._tick:
            raise RuntimeError('physical skill evidence requires one completed, undrained policy interval')
        active = self.active_id >= 0
        complete = active & self._flight_seen & (self._stable_time >= self.config.stable_landing_s - 1e-9)
        timeout = active & (self._time - self._accepted_at >= self.protocol.config.timeout_s - 1e-9)
        tolerance = torch.minimum(torch.full_like(self._rise, self.config.height_tolerance_m),
                                  self._rise * self.config.small_jump_tolerance_fraction)
        apex_rise = self._apex - self._baseline
        height_ok = self._flight_seen & ((apex_rise - self._rise).abs() <= tolerance + 1e-6)
        outcome = torch.zeros_like(self.active_id)
        outcome = torch.where(complete, int(SkillOutcome.PHYSICAL_SUCCESS), outcome)
        outcome = torch.where(complete & ~height_ok, int(SkillOutcome.HEIGHT_MISSED), outcome)
        outcome = torch.where(complete & ~self._supported_start, int(SkillOutcome.BAD_START), outcome)
        outcome = torch.where(complete & ~self._whole_healthy, int(SkillOutcome.UNHEALTHY), outcome)
        outcome = torch.where(complete & ~self._whole_owned, int(SkillOutcome.SCRIPT_ASSISTED), outcome)
        outcome = torch.where(complete & self._cancelled, int(SkillOutcome.CANCELLED), outcome)
        outcome = torch.where(timeout, int(SkillOutcome.TIMEOUT), outcome)
        finished = complete | timeout
        finished_id = torch.where(finished, self.active_id, self._pending_id)
        outcome = torch.where(self._pending_id >= 0, self._pending_outcome, outcome)
        finished_at = torch.where(finished, self._time, self._pending_at)
        success = torch.where(outcome == int(SkillOutcome.PHYSICAL_SUCCESS), finished_id, -1)
        remaining_id = torch.where(finished, -1, self.active_id)
        result = SkillEvidence(self._tick, self._time.clone(), self._episode.clone(), self._control.clone(), success,
                               finished_id.clone(), finished_at.clone(), outcome.clone(), self._whole_owned.clone(),
                               self._whole_healthy.clone(), self.phase.clone(), remaining_id, self._rise.clone(),
                               apex_rise.clone(), tolerance.clone(), self._supported_start.clone(), self._flight_seen.clone(),
                               self._stable_time.clone())
        self.active_id[finished] = -1
        self.phase[finished | (self._pending_id >= 0)] = int(SkillPhase.IDLE)
        self._pending_id.fill_(-1)
        self._pending_outcome.zero_()
        self._pending_at.zero_()
        self._drained_tick = self._tick
        return result

    @torch.no_grad()
    def accept_protocol_condition(self, condition: RequestCondition, latched_height_target_m):
        """Record real protocol receipts without using truth to change admission."""
        if (not isinstance(condition, RequestCondition) or self._sample is None
                or self._tick % self.substeps or self._drained_tick != self._tick):
            raise ValueError('physical evidence accepts protocol receipts only after its drained physical endpoint')
        self._tensor('latched visible height', latched_height_target_m)
        for name, width, dtype in (('jump_mode', None, torch.bool), ('height_m', None, torch.float32),
                                   ('clock_s', None, torch.float32), ('active_id', None, torch.long),
                                   ('response', None, torch.long), ('finished_id', None, torch.long),
                                   ('end_reason', None, torch.long), ('safety_required', None, torch.bool)):
            self._tensor(name, getattr(condition, name), width, dtype)
        protocol, sample = self.protocol, self._sample
        accepted = condition.response == int(RequestResponse.ACCEPTED)
        active = protocol.active_id >= 0
        expected_clock = torch.where(active, self._time - protocol.accepted_at_s, 0).float()
        invalid = ((protocol.last_time_s - self._time).abs().gt(1e-8)
                   | (condition.active_id != protocol.active_id)
                   | (condition.jump_mode != active) | (condition.height_m != protocol.height_m)
                   | ~torch.isfinite(condition.clock_s) | (condition.clock_s - expected_clock).abs().gt(1e-6)
                   | (condition.response < 0) | (condition.response > int(RequestResponse.CANCELLED))
                   | (condition.end_reason < 0) | (condition.end_reason > int(RequestEnd.COLD_RESTORE))
                   | (condition.finished_id < -1) | (condition.finished_id > protocol.max_seen_id)
                   | (accepted & ((condition.active_id <= self._retired) | (condition.active_id < 0)
                                  | ~self._cache_valid | condition.clock_s.abs().gt(1e-6)
                                  | (protocol.accepted_at_s - self._time).abs().gt(1e-8)))
                   | ~torch.isfinite(latched_height_target_m) | (latched_height_target_m <= 0))
        if invalid.any():
            raise ValueError('physical evidence receipt differs from its real shared request protocol and physical clock')
        finished_id = torch.full_like(self.active_id, -1)
        outcome = torch.zeros_like(self.active_id)
        existing = self.active_id >= 0
        matching_end = existing & (condition.finished_id == self.active_id)
        matching_active = existing & (protocol.active_id == self.active_id)
        self._cancelled |= matching_active & protocol.cancelled
        cancelled = matching_end & (condition.end_reason == int(RequestEnd.CANCELLED))
        self._cancelled |= cancelled
        cancel_on_ground = cancelled & ~self._flight_seen & self._grounded(sample)
        abort = matching_end & ((condition.end_reason == int(RequestEnd.SAFETY_ABORT))
                                | (condition.end_reason == int(RequestEnd.TIMEOUT)))
        superseded = accepted & existing
        close = cancel_on_ground | abort | superseded
        finished_id = torch.where(close, self.active_id, finished_id)
        outcome = torch.where(cancel_on_ground, int(SkillOutcome.CANCELLED), outcome)
        outcome = torch.where(abort, int(SkillOutcome.TIMEOUT), outcome)
        outcome = torch.where(abort & (condition.end_reason == int(RequestEnd.SAFETY_ABORT)),
                              int(SkillOutcome.SAFETY_ABORT), outcome)
        outcome = torch.where(superseded, int(SkillOutcome.SUPERSEDED), outcome)
        self.active_id[close] = -1
        self.phase[close] = int(SkillPhase.IDLE)
        # ESTIMATED_COMPLETE never closes or rewards the physical attempt.
        self.active_id[accepted] = condition.active_id[accepted]
        self.phase[accepted] = int(SkillPhase.PREPARING)
        self._rise[accepted] = condition.height_m[accepted]
        self._baseline[accepted] = sample.ground_world_z_m[accepted] + latched_height_target_m[accepted]
        self._accepted_at[accepted] = self._time[accepted]
        self._supported_start[accepted] = (self._stable(sample) & sample.healthy
                                           & ((sample.root_world_z_m - sample.ground_world_z_m - latched_height_target_m).abs()
                                              <= self.config.start_height_tolerance_m))[accepted]
        self._whole_owned[accepted], self._whole_healthy[accepted] = sample.policy_owned[accepted], sample.healthy[accepted]
        self._cancelled[accepted], self._flight_seen[accepted] = False, False
        for value in (self._air_time, self._ground_time, self._stable_time):
            value[accepted] = 0
        self._candidate_apex[accepted], self._apex[accepted] = -float('inf'), -float('inf')
        self._retired = torch.maximum(self._retired, protocol.max_seen_id)
        return SkillReceipt(torch.where(accepted, condition.active_id, -1), finished_id, outcome, self._time.clone())

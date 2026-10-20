"""Episode command trajectories and deployable jump-request scheduling.

Actor-visible request decisions use only the preceding completed estimate.
Physical flight/landing labels and success are a separate reward-only channel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch

from ..control.observation import CommandMode
from ..tasks.curriculum import TaskFamily
from ..tasks.jump import JumpReferenceProgram
from ..tasks.protocol import CompletedEstimate, RequestEnd, RequestProtocol, RequestResponse
from ..tasks.reward import RewardPhase
from ..tasks.trajectory import CommandRates, CommandStream, CommandTrajectory, DriveEnvelope


@dataclass(frozen=True)
class RequestCommand:
    """External/manual or ToF request submitted through the same ID protocol.

    Only the preceding completed estimate and this command are deployable
    inputs. Physical terrain height and flight/landing truth never enter the
    request decision.
    """

    request_id: torch.Tensor
    height_m: torch.Tensor
    cancel_id: torch.Tensor | None = None
    safety_abort: torch.Tensor | None = None
    source: str = 'manual'

    def owned(self):
        return replace(self, request_id=self.request_id.clone(), height_m=self.height_m.clone(),
                       cancel_id=None if self.cancel_id is None else self.cancel_id.clone(),
                       safety_abort=None if self.safety_abort is None else self.safety_abort.clone())

    def validate(self, count, device):
        if self.source not in ('manual', 'tof'):
            raise ValueError('request source must be manual or tof')
        for name, value, dtype in (
            ('request_id', self.request_id, torch.long),
            ('height_m', self.height_m, torch.float32),
            ('cancel_id', self.cancel_id, torch.long),
            ('safety_abort', self.safety_abort, torch.bool),
        ):
            if value is None:
                if name in ('request_id', 'height_m'):
                    raise ValueError(f'{name} is required')
                continue
            if (not isinstance(value, torch.Tensor) or value.shape != (count,)
                    or value.dtype != dtype or value.device != device):
                raise ValueError(f'{name} requires [{count}] {dtype} on {device}')


@dataclass(frozen=True)
class StairCondition:
    active: torch.Tensor
    height_m: torch.Tensor
    clock_s: torch.Tensor
    active_id: torch.Tensor
    response: torch.Tensor
    finished_id: torch.Tensor
    end_reason: torch.Tensor


@dataclass(frozen=True)
class StairRequestConfig:
    policy_dt_s: float = .01
    min_height_m: float = .04
    max_height_m: float = .20
    timeout_s: float = 6.

    def __post_init__(self):
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in vars(self).values())
                or self.policy_dt_s != .01 or self.min_height_m >= self.max_height_m
                or self.timeout_s <= self.policy_dt_s):
            raise ValueError('stair request bounds require finite 100Hz timing and ordered heights')


class StairRequestProtocol:
    """One-shot stair request IDs, sharing BUSY/cancel/timeout semantics."""

    def __init__(self, num_envs, device="cpu", config=StairRequestConfig()):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, StairRequestConfig):
            raise ValueError('stair requests require positive row count and parsed configuration')
        self.num_envs, self.device, self.config = num_envs, torch.device(device), config
        self.max_seen_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.active_id = torch.full_like(self.max_seen_id, -1)
        self.height_m = torch.zeros(num_envs, device=self.device)
        self.accepted_at_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.last_time_s = torch.full_like(self.accepted_at_s, -float("inf"))

    def _check(self, now_s, request_id, height_m, cancel_id, jump_active):
        n = self.num_envs
        for name, value, dtype, shape in (
            ("now", now_s, torch.float64, (n,)), ("request", request_id, torch.long, (n,)),
            ("height", height_m, torch.float32, (n,)), ("cancel", cancel_id, torch.long, (n,)),
            ("jump", jump_active, torch.bool, (n,))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                    or value.device != self.device):
                raise ValueError(f"stair {name} requires {shape}/{dtype} on {self.device}")

    @torch.no_grad()
    def step(self, *, now_s, request_id, height_m, cancel_id, jump_active, ready=None):
        self._check(now_s, request_id, height_m, cancel_id, jump_active)
        ready = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                 if ready is None else ready)
        if (not isinstance(ready, torch.Tensor) or ready.shape != (self.num_envs,)
                or ready.dtype != torch.bool or ready.device != self.device):
            raise ValueError('stair readiness requires one bool per row')
        clock = torch.isfinite(now_s) & (now_s > self.last_time_s)
        now = torch.where(clock, now_s, self.last_time_s)
        invalid_id = (request_id < -1) | (cancel_id < -1)
        new = (request_id >= 0) & (request_id > self.max_seen_id) & clock
        replay = (request_id >= 0) & ~new & clock
        valid_height = torch.isfinite(height_m) & (height_m >= self.config.min_height_m) & (height_m <= self.config.max_height_m)
        busy = new & ((self.active_id >= 0) | jump_active)
        cancelled_on_arrival = new & (cancel_id == request_id)
        accept = new & ~busy & valid_height & ~cancelled_on_arrival & ~invalid_id & ready
        response = torch.where(new, int(RequestResponse.NOT_READY), torch.zeros_like(self.active_id))
        response = torch.where(~clock, int(RequestResponse.INVALID_CLOCK), response)
        response = torch.where(invalid_id & clock, int(RequestResponse.INVALID), response)
        response = torch.where(replay, int(RequestResponse.REPLAY), response)
        response = torch.where(new & ~valid_height, int(RequestResponse.INVALID), response)
        response = torch.where(busy, int(RequestResponse.BUSY), response)
        response = torch.where(cancelled_on_arrival, int(RequestResponse.CANCELLED), response)
        response = torch.where(accept, int(RequestResponse.ACCEPTED), response)
        self.max_seen_id.copy_(torch.where(new, request_id, self.max_seen_id))
        self.active_id.copy_(torch.where(accept, request_id, self.active_id))
        self.height_m.copy_(torch.where(accept, height_m, self.height_m))
        self.accepted_at_s.copy_(torch.where(accept, now, self.accepted_at_s))
        active = self.active_id >= 0
        elapsed = torch.where(active, now - self.accepted_at_s, 0)
        cancel = active & (cancel_id == self.active_id) & clock
        timeout = active & clock & (elapsed >= self.config.timeout_s)
        aborted = active & clock & (~ready | jump_active)
        finished = cancel | timeout | aborted
        finished_id = torch.where(finished, self.active_id, -1)
        end = torch.zeros_like(self.active_id)
        end = torch.where(cancel, int(RequestEnd.CANCELLED), end)
        end = torch.where(timeout, int(RequestEnd.TIMEOUT), end)
        end = torch.where(aborted, int(RequestEnd.SAFETY_ABORT), end)
        self.active_id.masked_fill_(finished, -1)
        self.height_m.masked_fill_(finished, 0)
        self.accepted_at_s.masked_fill_(finished, 0)
        self.last_time_s.copy_(now)
        active = self.active_id >= 0
        return StairCondition(active, torch.where(active, self.height_m, 0),
                              torch.where(active, elapsed, 0).float(), self.active_id.clone(),
                              response, finished_id, end)

    def reset(self, ids, now_s):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.active_id[ids] = -1
        self.height_m[ids] = 0
        self.accepted_at_s[ids] = 0
        self.last_time_s[ids] = now_s[ids] - self.config.policy_dt_s

    def clear(self, mask):
        self.active_id.masked_fill_(mask, -1)
        self.height_m.masked_fill_(mask, 0)
        self.accepted_at_s.masked_fill_(mask, 0)

    def state_dict(self):
        return {'format': 'lab_stair_requests_v1', 'configuration': asdict(self.config),
                'max_seen_id': self.max_seen_id.clone(), 'active_id': self.active_id.clone()}

    def restore_after_physical_reset(self, state, now_s):
        """Keep consumed IDs; an old physical stair command is explicitly aborted."""
        if (not isinstance(state, dict)
                or set(state) != {'format', 'configuration', 'max_seen_id', 'active_id'}
                or state['format'] != 'lab_stair_requests_v1'
                or state['configuration'] != asdict(self.config)
                or not isinstance(now_s, torch.Tensor) or now_s.shape != (self.num_envs,)
                or now_s.dtype != torch.float64 or now_s.device != self.device
                or not torch.isfinite(now_s).all()):
            raise ValueError('stair restore requires its exact request contract and fresh clock')
        for name in ('max_seen_id', 'active_id'):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != self.max_seen_id.shape
                    or value.dtype != torch.long or (value < -1).any()):
                raise ValueError('stair restore request watermarks differ')
        maximum = state['max_seen_id'].to(self.device)
        aborted = state['active_id'].to(self.device)
        if (aborted > maximum).any():
            raise ValueError('stair restore request watermarks differ')
        self.max_seen_id.copy_(maximum)
        self.clear(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.last_time_s.copy_(now_s - self.config.policy_dt_s)
        return aborted.clone()


class LabCourseTasks:
    _COUNTERS = ('request_counters', 'external_request_counters', 'kind_counts', 'mode_counts',
                 'request_end_counts', 'physical_outcome_counts')

    def __init__(self, num_envs, envelope: DriveEnvelope, *, seed=2102, device='cpu'):
        self.num_envs, self.device = num_envs, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        initial = torch.zeros(num_envs, 4, device=self.device)
        initial[:, 3] = .32
        self.stream = CommandStream(initial, envelope, CommandRates(2., 3., .2))
        self.trajectory = CommandTrajectory(initial, initial, torch.ones(num_envs, device=self.device),
                                             torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self.protocol = RequestProtocol(num_envs, self.device)
        self.reference = JumpReferenceProgram()
        self.family = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.base_mode = self.family.clone()
        self.elapsed = torch.zeros(num_envs, device=self.device)
        self.next_request_s = torch.zeros_like(self.elapsed)
        self.next_id = torch.zeros_like(self.family)
        self.rise = torch.zeros_like(self.elapsed)
        self.active_rise = torch.zeros_like(self.elapsed)
        self.latched_height = torch.full_like(self.elapsed, .32)
        self.ground_anchor = torch.zeros_like(self.elapsed)
        self.request_clock = torch.zeros_like(self.elapsed)
        self.active_id = torch.full_like(self.family, -1)
        self.physical_id = torch.full_like(self.family, -1)
        self._physical_accepted_at_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self._physical_cancelled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros_like(self.family)
        self.flight_seen = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.air_ticks = torch.zeros_like(self.family)
        self.ascent_seen = torch.zeros_like(self.flight_seen)
        self.supported_start = torch.zeros_like(self.flight_seen)
        self._last_supported = torch.zeros_like(self.flight_seen)
        self._last_world_height = torch.zeros_like(self.elapsed)
        self.peak_rise = torch.zeros_like(self.elapsed)
        self.landed_s = torch.zeros_like(self.elapsed)
        self.success_id = torch.full_like(self.family, -1)
        self.paid_id = torch.full_like(self.family, -1)
        self.request_counters = torch.zeros(9, dtype=torch.long, device=self.device)
        self.kind_counts = torch.zeros(8, dtype=torch.long, device=self.device)
        self.mode_counts = torch.zeros(6, dtype=torch.long, device=self.device)
        self.external_request_counters = torch.zeros(9, dtype=torch.long, device=self.device)
        self.request_end_counts = torch.zeros(len(RequestEnd), dtype=torch.long, device=self.device)
        # Verified success, reset while unverified, closed while unverified.
        # The latter two are incomplete evidence, not invented physical failures.
        self.physical_outcome_counts = torch.zeros(3, dtype=torch.long, device=self.device)

    def reset(self, ids, task, now):
        self.physical_outcome_counts[1].add_((self.physical_id[ids] >= 0).sum())
        self.kind_counts.add_(torch.bincount(task.terrain_kind, minlength=8))
        self.mode_counts.add_(torch.bincount(task.mode, minlength=6))
        self.family[ids] = task.draw.family
        self.base_mode[ids] = torch.where(task.mode == int(CommandMode.JUMP),
                                         int(CommandMode.NORMAL), task.mode)
        self.rise[ids] = task.request_height_m
        self.active_rise[ids] = 0
        self.elapsed[ids] = 0
        self.next_request_s[ids] = 3 + 5 * torch.rand(len(ids), generator=self.generator, device=self.device)
        target = task.command.clone()
        target[task.draw.nominal] = target.new_tensor((0., 0., 0., .32))
        start = torch.zeros_like(target)
        start[:, 3] = .32
        hold = task.draw.trajectory_kind == 0
        start[hold] = target[hold]
        self.trajectory.start[ids], self.trajectory.end[ids] = start, target
        duration = .5 + 2.5 * torch.rand(len(ids), generator=self.generator, device=self.device)
        self.trajectory.duration_s[ids] = duration
        self.trajectory.frequency_start_hz[ids] = 1 / duration
        self.trajectory.frequency_end_hz[ids] = 2 / duration
        self.trajectory.kind[ids] = task.draw.trajectory_kind
        self.stream.current[ids] = self.stream.envelope.project(start)[0]
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        aborted = self.protocol.reset(mask, now)
        self.request_end_counts[int(RequestEnd.RESET)].add_((aborted >= 0).sum())
        self.active_id[ids] = -1
        self.physical_id[ids] = -1
        self._physical_accepted_at_s[ids] = 0
        self._physical_cancelled[ids] = False
        self.request_clock[ids] = 0
        self.flight_seen[ids] = False
        self.air_ticks[ids] = 0
        self.ascent_seen[ids] = False
        self.supported_start[ids] = False
        self._last_supported[ids] = False
        self.peak_rise[ids] = 0
        self.landed_s[ids] = 0
        self.phase[ids] = 0
        self.success_id[ids] = -1

    def step(self, now, estimate, ground, *, command: RequestCommand | None = None,
             blocked_by_stair=None):
        self.success_id.fill_(-1)
        if command is not None:
            command.validate(self.num_envs, self.device)
        self.elapsed.add_(.01)
        cyclic = (self.trajectory.kind == 2) | (self.trajectory.kind == 3) | (self.trajectory.kind == 5)
        clock = torch.where(cyclic, self.elapsed.remainder(self.trajectory.duration_s), self.elapsed)
        admission = self.stream.step(self.trajectory.at(clock))
        if estimate is None:
            estimate = CompletedEstimate(
                torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device),
                now.clone(), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        blocked = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                   if blocked_by_stair is None else torch.as_tensor(blocked_by_stair, dtype=torch.bool, device=self.device))
        if blocked.shape != (self.num_envs,):
            raise ValueError('blocked_by_stair requires [N] bool')
        if command is None:
            due = ((self.family == int(TaskFamily.REQUEST)) & ~blocked & (self.elapsed >= self.next_request_s)
                   & (self.protocol.active_id < 0))
            automatic_id = torch.where(due, self.next_id, -1)
            self.next_id.add_(due.long())
            interval = 3 + 5 * torch.rand(self.num_envs, generator=self.generator, device=self.device)
            self.next_request_s.copy_(torch.where(due, self.elapsed + interval, self.next_request_s))
            request_id = automatic_id
            requested_height = self.rise
            cancel_id = torch.full_like(request_id, -1)
            safety_abort = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            # A supplied command owns its IDs. Auto IDs remain dormant while a
            # manual/ToF command is present, so BUSY/replay behavior is explicit.
            request_id = command.request_id
            requested_height = command.height_m
            cancel_id = (torch.full_like(request_id, -1) if command.cancel_id is None
                         else command.cancel_id)
            safety_abort = (torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                            if command.safety_abort is None else command.safety_abort)
        allowed_family = ((self.family == int(TaskFamily.REQUEST)) if command is None
                          else (self.base_mode != int(CommandMode.RECOVERY)))
        condition = self.protocol.step(
            now_s=now, estimate=estimate, request_id=request_id, requested_height_m=requested_height,
            cancel_id=cancel_id, allow_new=allowed_family & ~blocked,
            safety_abort=safety_abort)
        # External IDs may skip far ahead. Auto scheduling must resume above
        # the highest consumed ID, otherwise its next ID is a replay.
        self.next_id.copy_(torch.maximum(self.next_id, self.protocol.max_seen_id + 1))
        accepted = condition.response == int(RequestResponse.ACCEPTED)
        matching = condition.finished_id == self.physical_id
        closed = ((matching & (condition.end_reason != int(RequestEnd.ESTIMATED_COMPLETE))
                   & (condition.end_reason != int(RequestEnd.NONE)))
                  | ((self.physical_id >= 0)
                     & (now - self._physical_accepted_at_s >= self.protocol.config.timeout_s)))
        self.physical_outcome_counts[2].add_(((self.physical_id >= 0) & closed).sum())
        ended = torch.bincount(condition.end_reason, minlength=len(RequestEnd))
        ended[0] = 0
        self.request_end_counts.add_(ended)
        self.physical_id.masked_fill_(closed, -1)
        self.physical_id.copy_(torch.where(accepted, condition.active_id, self.physical_id))
        self._physical_accepted_at_s.copy_(torch.where(accepted, now, self._physical_accepted_at_s))
        self._physical_cancelled |= ((self.physical_id == self.protocol.active_id)
                                     & self.protocol.cancelled)
        self._physical_cancelled |= (self.physical_id >= 0) & blocked
        self._physical_cancelled.masked_fill_(accepted, False)
        self.request_counters.add_(torch.bincount(condition.response, minlength=9))
        if command is not None:
            self.external_request_counters.add_(torch.bincount(condition.response, minlength=9))
        self.latched_height.copy_(torch.where(accepted, admission.admitted[:, 3], self.latched_height))
        self.active_rise.copy_(torch.where(accepted, condition.height_m, self.active_rise))
        self.ground_anchor.copy_(torch.where(accepted, ground, self.ground_anchor))
        # Truth certifies reward evidence only; it cannot accept/reject a
        # deployable request or select a command on the policy's behalf.
        supported = (self._last_supported
                     & (self._last_world_height - ground - admission.admitted[:, 3]).abs().le(.015))
        self.supported_start.copy_(torch.where(accepted, supported, self.supported_start))
        self.flight_seen.masked_fill_(accepted, False)
        self.air_ticks.masked_fill_(accepted, 0)
        self.ascent_seen.masked_fill_(accepted, False)
        self.peak_rise.masked_fill_(accepted, 0)
        self.landed_s.masked_fill_(accepted, 0)
        self.active_id.copy_(condition.active_id)
        self.request_clock.copy_(condition.clock_s)
        mode = torch.where(condition.jump_mode, int(CommandMode.JUMP), self.base_mode)
        # The reference baseline is actor-visible throughout an active request.
        # A continuing sine/ramp must not hide a latched baseline outside H5.
        self.stream.current[:, 3] = torch.where(condition.jump_mode, self.latched_height,
                                                self.stream.current[:, 3])
        admission = replace(admission, admitted=self.stream.current.clone())
        return admission, mode, condition.height_m, condition.clock_s

    def observe_physics(self, force, velocity_z, tilt, world_height, *, angular_speed_rad_s, dt=.005):
        active = self.physical_id >= 0
        air = (force <= 2).all(-1)
        self.air_ticks.copy_(torch.where(active & air, self.air_ticks + 1, 0))
        self.flight_seen |= active & (self.air_ticks >= 2)
        actual_rise = world_height - self.ground_anchor - self.latched_height
        self.ascent_seen |= active & (velocity_z > .15) & (actual_rise > .01)
        self.peak_rise.copy_(torch.where(active, torch.maximum(self.peak_rise, actual_rise), self.peak_rise))
        contact = (force >= 5).all(-1)
        tolerance = torch.minimum(torch.full_like(self.active_rise, .03), self.active_rise * .5)
        achieved = (self.peak_rise - self.active_rise).abs() <= tolerance + 1e-6
        quiet = (velocity_z.abs() <= .15) & (tilt <= .25) & (angular_speed_rad_s <= .5)
        self._last_supported.copy_(contact & quiet)
        self._last_world_height.copy_(world_height)
        stable = (active & self.supported_start & self.flight_seen & self.ascent_seen & achieved & contact
                  & quiet
                  & (actual_rise.abs() <= .05))
        self.landed_s.copy_(torch.where(stable, self.landed_s + dt, 0))
        self.phase.copy_(torch.where(
            self.active_id >= 0, torch.where(air & self.flight_seen, int(RewardPhase.FLIGHT),
                                torch.where(self.flight_seen, int(RewardPhase.LANDING), int(RewardPhase.TAKEOFF))),
            torch.where(self.base_mode == int(CommandMode.RECOVERY), int(RewardPhase.RECOVERY), int(RewardPhase.SUPPORT))))
        # Physical truth acceptance is reward-only. The deployment FSM may
        # leave FLIGHT on a timer, but it cannot manufacture this event.
        success = (stable & ~self._physical_cancelled & (self.landed_s >= 2.)
                   & (self.physical_id > self.paid_id))
        self.physical_outcome_counts[0].add_(success.sum())
        self.success_id.copy_(torch.where(success, self.physical_id, self.success_id))
        self.paid_id.copy_(torch.where(success, self.physical_id, self.paid_id))
        self.physical_id.masked_fill_(success, -1)

    def jump_reference(self):
        return self.reference.sample(self.latched_height, self.active_rise, self.request_clock.double(),
                                     self.ground_anchor, active=self.active_id >= 0)

    def state_dict(self):
        # Cold restoration aborts physical requests, preserving IDs and RNG.
        return {'format': 'lab_course_tasks_v3', 'protocol': self.protocol.state_dict(),
                'next_id': self.next_id.clone(), 'paid_id': self.paid_id.clone(),
                'rng': self.generator.get_state().clone(),
                **{name: getattr(self, name).clone() for name in self._COUNTERS}}

    def restore_after_physical_reset(self, state, now):
        expected = {'format', 'protocol', 'next_id', 'paid_id', 'rng', *self._COUNTERS}
        if (not isinstance(state, dict) or set(state) != expected
                or state.get('format') != 'lab_course_tasks_v3'):
            raise ValueError('course request checkpoint differs')
        for name in ('next_id', 'paid_id', *self._COUNTERS):
            value, reference = state[name], getattr(self, name)
            minimum = -1 if name == 'paid_id' else 0
            if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                    or value.dtype != reference.dtype or (value < minimum).any()):
                raise ValueError('course request checkpoint tensor differs: ' + name)
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].cpu())
        self.protocol.restore_after_physical_reset(state['protocol'], now)
        self.next_id.copy_(torch.maximum(state['next_id'].to(self.device),
                                        self.protocol.max_seen_id + 1))
        self.paid_id.copy_(state['paid_id'])
        for name in self._COUNTERS:
            getattr(self, name).copy_(state[name])
        self.generator = generator
        self.active_id.fill_(-1)
        self.physical_id.fill_(-1)
        self.supported_start.zero_()
        self._last_supported.zero_()
        self.success_id.fill_(-1)


def completed_policy_estimate(frame, raw_state, timestamp):
    """Deployment-reproducible estimate: no critic, terrain truth, or future state."""
    return CompletedEstimate._from_actor_observation(frame, raw_state, timestamp)

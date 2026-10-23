"""Private causal command programs; no physics truth or actuator policy.

Requests, target trajectories and command admission have deployment-visible
state. Ground anchors belong to the environment's reward context,not this
controller. The protocol is the very same object used by the environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory import CommandStream, CommandTrajectory, TrajectoryKind
from .jump import JumpReferenceProgram
from ..control.observation import CommandFeatures, CommandMode
from .curriculum import TaskDomain, TaskDraw, TaskFamily
from .protocol import RequestCondition, RequestProtocol, RequestResponse
from .geometry import TerrainKind


@dataclass(frozen=True)
class TaskCommandPlan:
    """Full task clock starts at first policy handoff and includes later scripts."""
    trajectory: CommandTrajectory
    base_mode: torch.Tensor
    task_horizon_s: torch.Tensor
    jump_height_m: torch.Tensor
    request_interval_s: torch.Tensor

    @classmethod
    def from_draw(cls, draw: TaskDraw, start, trajectory_duration_s, task_horizon_s, request_interval_s,
                  *, domain=TaskDomain()):
        if not isinstance(draw, TaskDraw) or not isinstance(domain, TaskDomain):
            raise ValueError('command plan requires the original continuous task draw and domain')
        end = draw.requested_command.detach().clone()
        initial = start.detach().clone()
        nominal = end.new_tensor((0., 0., 0., domain.nominal_height_m))
        end[draw.nominal], initial[draw.nominal] = nominal, nominal
        kind = torch.where(draw.nominal, int(TrajectoryKind.HOLD), draw.trajectory_kind)
        # HOLD means holding its sampled target,not silently keeping the
        # caller's generic trajectory starting point for the whole episode.
        initial[kind == int(TrajectoryKind.HOLD)] = end[kind == int(TrajectoryKind.HOLD)]
        mode = torch.full_like(draw.family, int(CommandMode.NORMAL))
        stairs = ((draw.terrain_kind == int(TerrainKind.STAIRS)) | (draw.terrain_kind == int(TerrainKind.STEP_UP))
                  | (draw.terrain_kind == int(TerrainKind.STEP_DOWN))
                  | (draw.terrain_kind == int(TerrainKind.TWO_PLATFORMS)))
        mode = torch.where(stairs, int(CommandMode.STAIR), mode)
        mode = torch.where(draw.terrain_kind == int(TerrainKind.SLOPE), int(CommandMode.SLOPE), mode)
        mode = torch.where(draw.family == int(TaskFamily.RECOVERY), int(CommandMode.RECOVERY), mode)
        rise = torch.where(draw.family == int(TaskFamily.REQUEST), draw.jump_height_m, 0)
        return cls(CommandTrajectory(initial, end, trajectory_duration_s, kind), mode, task_horizon_s,
                   rise, request_interval_s)


@dataclass(frozen=True)
class RequestInputs:
    request_id: torch.Tensor
    requested_height_m: torch.Tensor
    cancel_id: torch.Tensor
    allow_new: torch.Tensor
    safety_abort: torch.Tensor
    release_script: torch.Tensor

    @classmethod
    def empty(cls, num_envs, device):
        ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        no = torch.zeros(num_envs, dtype=torch.bool, device=device)
        return cls(ids, torch.zeros(num_envs, device=device), ids.clone(), ~no, no, no.clone())

    def owned(self):
        return type(self)(*(value.detach().clone() for value in vars(self).values()))


class TaskCommandProgram:
    """Advance once AFTER physics; hold the preceding packet during physics.

    Automatic training requests use fresh monotonic IDs, never replay a
    rejected ID, and do not inspect true contact, height or skill completion.
    Explicit inputs can exercise repeat/cancel/safety sequences independently.
    """

    def __init__(self, protocol: RequestProtocol, stream: CommandStream, jump: JumpReferenceProgram,
                 *, domain=TaskDomain()):
        if (not isinstance(protocol, RequestProtocol) or not isinstance(stream, CommandStream)
                or not isinstance(jump, JumpReferenceProgram) or not isinstance(domain, TaskDomain)
                or stream.current.shape != (protocol.num_envs, 4) or stream.current.device != protocol.device
                or stream.current.dtype != torch.float32 or stream.policy_dt_s != protocol.config.policy_dt_s):
            raise ValueError('task program requires one shared protocol, command stream and visible-clock curve')
        self.protocol, self.stream, self.jump, self.domain = protocol, stream, jump, domain
        self.num_envs, self.device = protocol.num_envs, protocol.device
        n, device = self.num_envs, self.device
        self._trajectory = None
        self._bound = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_time = torch.zeros(n, dtype=torch.float64, device=device)
        self._started = self._last_time.clone()
        self._base_mode = torch.zeros(n, dtype=torch.long, device=device)
        self.task_horizon_s = self._last_time.clone()
        self._rise = torch.zeros(n, device=device)
        self._period = self._last_time.clone()
        self._next_request = self._last_time.clone()
        self._latched_height = torch.full((n,), domain.nominal_height_m, device=device)
        self._latched_rise = self._rise.clone()
        self._latched_at = self._last_time.clone()
        self._latched_id = torch.full_like(protocol.active_id, -1)
        self.last_admission = None

    def _tensor(self, value, width=None, dtype=torch.float32):
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device
                or value.dtype != dtype):
            raise ValueError('task command layout/device/dtype differs')

    def validate_plan(self, plan):
        if not isinstance(plan, TaskCommandPlan) or not isinstance(plan.trajectory, CommandTrajectory):
            raise ValueError('task program requires a complete explicit command plan')
        t = plan.trajectory
        for value in (t.start, t.end):
            self._tensor(value, 4)
            if (not torch.isfinite(value).all() or value[:, 1].abs().gt(1e-8).any()
                    or ((value[:, 3] < self.domain.height_m[0]) | (value[:, 3] > self.domain.height_m[1])).any()):
                raise ValueError('command plan must retain finite body-frame requests and the full declared height domain')
        for value, dtype in ((plan.base_mode, torch.long), (plan.task_horizon_s, torch.float64),
                               (plan.jump_height_m, torch.float32), (plan.request_interval_s, torch.float64)):
            self._tensor(value, dtype=dtype)
        policy_dt_s = self.protocol.config.policy_dt_s
        if (((plan.base_mode < 0) | (plan.base_mode > int(CommandMode.RECOVERY))).any()
                or not torch.isfinite(plan.task_horizon_s).all() or (plan.task_horizon_s <= 0).any()
                or (plan.task_horizon_s / policy_dt_s
                    - (plan.task_horizon_s / policy_dt_s).round()).abs().gt(1e-7).any()
                or not torch.isfinite(plan.jump_height_m).all()
                or ((plan.jump_height_m != 0) & ((plan.jump_height_m < self.domain.jump_m[0])
                                                | (plan.jump_height_m > self.domain.jump_m[1]))).any()
                or not torch.isfinite(plan.request_interval_s).all() or (plan.request_interval_s < 0).any()
                or ((plan.jump_height_m > 0) & (plan.request_interval_s < policy_dt_s)).any()):
            raise ValueError('task horizon, non-jump base mode or request schedule differs from the declared contract')

    @torch.no_grad()
    def reset(self, mask, plan, now_s):
        self.validate_plan(plan)
        self._tensor(mask, dtype=torch.bool)
        self._tensor(now_s, dtype=torch.float64)
        if (not torch.isfinite(now_s).all() or (now_s < 0).any() or (~self._bound & ~mask).any()
                or (mask & (now_s < self._last_time)).any() or (mask & (self.protocol.active_id >= 0)).any()):
            raise ValueError('task reset requires every initial row, a nonrewound clock and aborted physical requests')
        fresh = plan.trajectory
        if self._trajectory is None:
            self._trajectory = CommandTrajectory(fresh.start, fresh.end, fresh.duration_s, fresh.kind,
                                                  fresh.frequency_start_hz, fresh.frequency_end_hz)
        else:
            for name in ('start', 'end', 'duration_s', 'kind', 'frequency_start_hz', 'frequency_end_hz'):
                getattr(self._trajectory, name)[mask] = getattr(fresh, name)[mask]
        self.stream.current[mask] = self.stream.envelope.project(fresh.start)[0][mask]
        self._base_mode[mask] = plan.base_mode[mask]
        self.task_horizon_s[mask] = plan.task_horizon_s[mask]
        self._rise[mask], self._period[mask] = plan.jump_height_m[mask], plan.request_interval_s[mask]
        self._last_time[mask], self._started[mask] = now_s[mask], now_s[mask]
        self._next_request[mask] = now_s[mask] + self._period[mask]
        self._latched_height[mask] = self.stream.current[mask, 3]
        self._latched_rise[mask], self._latched_id[mask] = 0, -1
        self._latched_at[mask] = now_s[mask]
        self._bound[mask] = True

    def validate_requests(self, value):
        if not isinstance(value, RequestInputs):
            raise ValueError('request delivery requires explicit controller inputs')
        for field, dtype in (('request_id', torch.long), ('requested_height_m', torch.float32),
                              ('cancel_id', torch.long), ('allow_new', torch.bool),
                              ('safety_abort', torch.bool), ('release_script', torch.bool)):
            self._tensor(getattr(value, field), dtype=dtype)

    @torch.no_grad()
    def requests(self, now_s, policy_ready, override):
        self._tensor(now_s, dtype=torch.float64)
        self._tensor(policy_ready, dtype=torch.bool)
        self.validate_requests(override)
        if (not self._bound.all()
                or (now_s - self._last_time - self.protocol.config.policy_dt_s).abs().gt(1e-8).any()):
            raise ValueError('request generation requires the next completed policy boundary')
        auto = (policy_ready & (self._rise > 0) & (self._period > 0) & (now_s >= self._next_request - 1e-9)
                & (self.protocol.active_id < 0) & (override.request_id == -1) & (override.cancel_id == -1)
                & override.allow_new & ~override.safety_abort)
        ids = torch.where(auto, self.protocol.max_seen_id + 1, override.request_id)
        rise = torch.where(auto, self._rise, override.requested_height_m)
        self._next_request[auto] = now_s[auto] + self._period[auto]
        return RequestInputs(ids, rise, override.cancel_id.clone(), override.allow_new.clone(),
                              override.safety_abort.clone(), override.release_script.clone())

    @torch.no_grad()
    def advance(self, now_s, condition: RequestCondition):
        self._tensor(now_s, dtype=torch.float64)
        if (not self._bound.all() or not isinstance(condition, RequestCondition)
                or (now_s - self._last_time - self.protocol.config.policy_dt_s).abs().gt(1e-8).any()
                or not torch.equal(self.protocol.last_time_s, now_s)
                or not torch.equal(condition.active_id, self.protocol.active_id)):
            raise ValueError('next commands require the actual once-per-boundary protocol result')
        accepted = condition.response == int(RequestResponse.ACCEPTED)
        self._latched_height[accepted] = self.stream.current[accepted, 3]
        self._latched_rise[accepted] = condition.height_m[accepted]
        self._latched_at[accepted] = now_s[accepted]
        self._latched_id[accepted] = condition.active_id[accepted]
        requested = self._trajectory.at(now_s - self._started)
        active = self.protocol.active_id >= 0
        requested[active, 3] = self._latched_height[active]
        self.last_admission = self.stream.step(requested)
        self._last_time.copy_(now_s)

    def commands(self, now_s, previous_issued, previous_control_source=None):
        self._tensor(now_s, dtype=torch.float64)
        self._tensor(previous_issued, 6)
        active = self.protocol.active_id >= 0
        if (not self._bound.all() or not torch.equal(now_s, self._last_time)
                or (active & (self._latched_id != self.protocol.active_id)).any()
                or (active & (self.stream.current[:, 3] != self._latched_height)).any()):
            raise ValueError('actor and reward must share the current admitted command and visible height latch')
        mode = torch.where(active, int(CommandMode.JUMP), self._base_mode)
        clock = torch.where(active, now_s - self._latched_at, now_s - self._started).float()
        return CommandFeatures(self.stream.current.clone(), mode.clone(), torch.where(active, self._latched_rise, 0),
                                 clock, previous_issued.clone(),
                                 None if previous_control_source is None else previous_control_source.detach().clone())

    def curve(self, commands, now_s, *, reward_ground_anchor_world_z_m):
        """Reward-only adapter; controller advance never needs this truth."""
        self._tensor(now_s, dtype=torch.float64)
        active = commands.mode == int(CommandMode.JUMP)
        elapsed = torch.where(active, now_s - self._latched_at, 0)
        return self.jump.sample(self._latched_height, self._latched_rise, elapsed,
                                  reward_ground_anchor_world_z_m, active=active)

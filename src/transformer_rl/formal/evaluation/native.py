"""Private deterministic evaluation of first physical episodes.

Native substeps, support masks and PRE-reset outcomes remain explicit. No
checkpoint is loaded, optimizer is run, SDK is launched or qualification is
granted. Scene, curriculum and request-sequence binding remain the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from ..tasks.reward import RewardPhase
from ..simulation.task import NativeTaskEnvironment
from ..control.observation import CommandMode
from ..tasks.skills import SkillOutcome
from ..contracts import StructuredCollectionConfig
from ..learning.collection import StructuredCollector
from ..tasks.protocol import RequestResponse


class _Moments:
    """Per-row, time-weighted Welford moments; no arithmetic mean of RMS values."""

    def __init__(self, count, width, device):
        self.seconds = torch.zeros(count, width, dtype=torch.float64, device=device)
        self.samples = torch.zeros(count, width, dtype=torch.long, device=device)
        self.mean = torch.zeros(count, width, dtype=torch.float64, device=device)
        self.m2 = torch.zeros_like(self.mean)
        self.minimum = torch.full_like(self.mean, float('inf'))
        self.maximum = torch.full_like(self.mean, -float('inf'))

    def add(self, values, mask, dt):
        values = values.double()
        weight = mask.double() * dt
        total = self.seconds + weight
        delta = values - self.mean
        new_mean = self.mean + delta * (weight / total.clamp_min(dt))
        self.m2 += weight * delta * (values - new_mean)
        self.mean = new_mean
        self.minimum = torch.where(mask, torch.minimum(self.minimum, values), self.minimum)
        self.maximum = torch.where(mask, torch.maximum(self.maximum, values), self.maximum)
        self.seconds = total
        self.samples += mask.long()

    def report(self, row, columns):
        seconds, count = self.seconds[row, columns.start].item(), self.samples[row, columns.start].item()
        if not count:
            return {'physical_samples': 0, 'support_s': 0., 'mean': None, 'rms': None,
                    'centered_std': None, 'min': None, 'max': None}
        variance = (self.m2[row, columns] / seconds).clamp_min(0)
        return {'physical_samples': count, 'support_s': seconds,
                'mean': self.mean[row, columns].tolist(),
                'rms': (variance + self.mean[row, columns].square()).sqrt().tolist(),
                'centered_std': variance.sqrt().tolist(),
                'min': self.minimum[row, columns].tolist(), 'max': self.maximum[row, columns].tolist()}


@dataclass(frozen=True)
class EvaluationPhysicalSample:
    physical_tick: int
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    policy_owned: torch.Tensor
    healthy: torch.Tensor
    true_failure: torch.Tensor
    requested_command: torch.Tensor
    admitted_command: torch.Tensor
    command_mode: torch.Tensor
    phase: torch.Tensor
    velocity_body_m_s: torch.Tensor
    velocity_world_m_s: torch.Tensor
    root_position_world_m: torch.Tensor
    support_height_m: torch.Tensor
    roll_pitch_error_rad: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor
    jump_world_z_error_m: torch.Tensor
    active_effort_nm: torch.Tensor
    output_velocity_rad_s: torch.Tensor
    effort_saturated: torch.Tensor
    issued_action: torch.Tensor
    wheel_tangent_speed_m_s: torch.Tensor
    wheel_clearance_m: torch.Tensor
    wheel_normal_load_n: torch.Tensor
    non_wheel_body_load_n: torch.Tensor
    knee_clearance_rad: torch.Tensor
    knee_pivot_residual_m: torch.Tensor
    wheel_target_projected: torch.Tensor


class FirstEpisodeStatistics:
    """Consume every physical tick and close only the first episode in each row.

    Initial sensor preparation precedes the task clock. Later script intervals
    remain inside it and prevent a policy-owned healthy-horizon claim. Health
    and tracking support are different quantities; neither hides failed rows.
    """

    _WIDTHS = {
        'tracking_error': (6, ('vx_m_s', 'vy_m_s', 'yaw_rad_s', 'height_m', 'roll_rad', 'pitch_rad')),
        'requested_tracking_error': (4, ('vx_m_s', 'vy_m_s', 'yaw_rad_s', 'height_m')),
        'world_xy_velocity': (2, ('x_m_s', 'y_m_s')),
        'world_xy_speed': (1, ('m_s',)),
        'jump_world_z_error': (1, ('m',)),
        'active_effort': (6, ('Nm',) * 6),
        'effort_slew': (6, ('Nm_s',) * 6),
        'absolute_mechanical_power': (6, ('W',) * 6),
        'feedback_effort_saturation': (6, ('fraction',) * 6),
        'wheel_contact_slip': (2, ('m_s',) * 2),
        'wheel_clearance': (2, ('m',) * 2),
        'wheel_normal_load': (2, ('N',) * 2),
        'non_wheel_body_load': (1, ('N',)),
        'knee_clearance': (2, ('rad',) * 2),
        'knee_pivot_residual': (2, ('m',) * 2),
        'wheel_target_projection': (1, ('fraction',)),
    }

    def __init__(self, episode_id, expected_horizon_s, *, physical_tick, physics_dt_s, initial_world_position_m=None):
        if (not isinstance(episode_id, torch.Tensor) or episode_id.ndim != 1 or episode_id.numel() < 1
                or episode_id.dtype != torch.long or (episode_id < 0).any()
                or physics_dt_s not in (.005, .001) or type(physical_tick) is not int
                or physical_tick < 0 or physical_tick % round(.01 / physics_dt_s)):
            raise ValueError('first-episode statistics require actual rows and a published native tick')
        self.num_envs, self.device = episode_id.numel(), episode_id.device
        self.dt, self.substeps, self.tick = physics_dt_s, round(.01 / physics_dt_s), physical_tick
        self._episode = episode_id.detach().clone()
        self._tensor('task horizon', expected_horizon_s, dtype=torch.float64)
        if initial_world_position_m is not None:
            self._tensor('initial link-origin position', initial_world_position_m, 3)
        if (not torch.isfinite(expected_horizon_s).all() or (expected_horizon_s <= 0).any()
                or initial_world_position_m is not None and not torch.isfinite(initial_world_position_m).all()
                or (expected_horizon_s / .01 - (expected_horizon_s / .01).round()).abs().gt(1e-7).any()):
            raise ValueError('evaluation task horizons and initial positions must be finite and explicit')
        self.horizon = expected_horizon_s.detach().clone()
        n, device = self.num_envs, self.device
        if initial_world_position_m is None:
            # Fresh reset buffers are not measurements. Actual preparation
            # substeps must supply the preceding position before policy starts.
            initial_world_position_m = torch.full((n, 3), float('nan'), device=device)
        self.started = torch.zeros(n, dtype=torch.bool, device=device)
        self.finished = torch.zeros_like(self.started)
        self.terminated = torch.zeros_like(self.started)
        self.truncated = torch.zeros_like(self.started)
        self.whole_healthy = torch.ones_like(self.started)
        self.whole_owned = torch.ones_like(self.started)
        self.failure_reasons = torch.zeros(n, dtype=torch.long, device=device)
        self._start = torch.full((n,), float('inf'), dtype=torch.float64, device=device)
        self._end = self._start.clone()
        self._seconds = torch.zeros_like(self._start)
        self._policy_seconds = self._seconds.clone()
        self._steady_run_s = self._seconds.clone()
        self._longest_steady_s = self._seconds.clone()
        self._previous_position = initial_world_position_m.detach().clone()
        self._origin = initial_world_position_m.detach().clone()
        self._last_position = initial_world_position_m.detach().clone()
        self._previous_effort = torch.zeros(n, 6, device=device)
        self._effort_valid = False
        self._last_time = None
        self._interval_closed = True
        self._endpoint_index = 0
        self._position_ring = torch.zeros(1001, n, 2, dtype=torch.float64, device=device)
        self._position_ring[0] = initial_world_position_m[:, :2].double()
        self._standing_ticks = torch.zeros(n, dtype=torch.long, device=device)
        self._standing_interval = torch.ones_like(self.started)
        self._drift_count = torch.zeros_like(self._standing_ticks)
        self._max_drift = torch.zeros_like(self._seconds)
        self._columns = {}
        width = 0
        for name, (size, _) in self._WIDTHS.items():
            self._columns[name] = slice(width, width + size)
            width += size
        self._moments = {scope: _Moments(n, width, device) for scope in ('full_task', 'policy', 'steady_support')}

    def _tensor(self, name, value, width=None, dtype=torch.float32):
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.device != self.device or value.dtype != dtype):
            raise ValueError('evaluation ' + name + ' rows/device/dtype differ')

    @torch.no_grad()
    def observe(self, sample: EvaluationPhysicalSample):
        if (not isinstance(sample, EvaluationPhysicalSample) or type(sample.physical_tick) is not int
                or sample.physical_tick != self.tick + 1
                or self.tick % self.substeps == 0 and not self._interval_closed):
            raise ValueError('evaluation needs every next native tick and exactly one closed endpoint per interval')
        widths = {'requested_command': 4, 'admitted_command': 4, 'velocity_body_m_s': 3, 'velocity_world_m_s': 3,
                  'root_position_world_m': 3, 'roll_pitch_error_rad': 2, 'angular_velocity_body_rad_s': 3,
                  'active_effort_nm': 6, 'output_velocity_rad_s': 6, 'issued_action': 6,
                  'wheel_tangent_speed_m_s': 2, 'wheel_clearance_m': 2, 'wheel_normal_load_n': 2,
                  'knee_clearance_rad': 2, 'knee_pivot_residual_m': 2}
        bools = ('policy_owned', 'healthy', 'true_failure', 'wheel_target_projected')
        longs = ('episode_id', 'control_epoch', 'command_mode', 'phase')
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for field in fields(sample):
            name, value = field.name, getattr(sample, field.name)
            if name == 'physical_tick':
                continue
            dtype = torch.float64 if name == 'timestamp_s' else torch.bool if name in bools or name == 'effort_saturated' \
                else torch.long if name in longs else torch.float32
            self._tensor(name, value, 6 if name == 'effort_saturated' else widths.get(name), dtype)
            if value.is_floating_point():
                invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        active = ~self.finished & (self.started | sample.policy_owned)
        new = active & ~self.started
        invalid |= ((sample.timestamp_s - sample.physical_tick * self.dt).abs().gt(1e-8)
                    | (sample.control_epoch < 0) | (sample.episode_id < 0)
                    | (sample.command_mode < 0) | (sample.command_mode > 4)
                    | (sample.phase < 0) | (sample.phase > int(RewardPhase.RECOVERY))
                    | sample.wheel_tangent_speed_m_s.lt(0).any(-1) | sample.wheel_normal_load_n.lt(0).any(-1)
                    | (sample.non_wheel_body_load_n < 0) | sample.knee_pivot_residual_m.lt(0).any(-1)
                    | (sample.healthy & sample.true_failure)
                    | (~self.finished & (sample.episode_id != self._episode))
                    | (new & ((self.tick % self.substeps != 0) | ~torch.isfinite(self._previous_position).all(-1))))
        if invalid.any():
            raise ValueError('evaluation physical sample,clock,episode or preceding position differs;no zero substitution')
        self._start[new] = sample.timestamp_s[new] - self.dt
        self._origin[new] = self._previous_position[new]
        self.started |= new
        self._seconds += active.double() * self.dt
        self._policy_seconds += (active & sample.policy_owned).double() * self.dt
        self.whole_healthy &= ~active | (sample.healthy & ~sample.true_failure)
        self.whole_owned &= ~active | sample.policy_owned
        self._last_position[active] = sample.root_position_world_m[active]
        actual = torch.stack((sample.velocity_body_m_s[:, 0], sample.velocity_body_m_s[:, 1],
                              sample.angular_velocity_body_rad_s[:, 2], sample.support_height_m), -1)
        error = torch.cat((actual - sample.admitted_command, sample.roll_pitch_error_rad), -1)
        slew = (sample.active_effort_nm - self._previous_effort) / self.dt
        values = {'tracking_error': error, 'requested_tracking_error': actual - sample.requested_command,
                  'world_xy_velocity': sample.velocity_world_m_s[:, :2],
                  'world_xy_speed': sample.velocity_world_m_s[:, :2].norm(dim=-1, keepdim=True),
                  'jump_world_z_error': sample.jump_world_z_error_m[:, None],
                  'active_effort': sample.active_effort_nm, 'effort_slew': slew,
                  'absolute_mechanical_power': (sample.active_effort_nm * sample.output_velocity_rad_s).abs(),
                  'feedback_effort_saturation': sample.effort_saturated.float(),
                  'wheel_contact_slip': sample.wheel_tangent_speed_m_s, 'wheel_clearance': sample.wheel_clearance_m,
                  'wheel_normal_load': sample.wheel_normal_load_n, 'non_wheel_body_load': sample.non_wheel_body_load_n[:, None],
                  'knee_clearance': sample.knee_clearance_rad, 'knee_pivot_residual': sample.knee_pivot_residual_m,
                  'wheel_target_projection': sample.wheel_target_projected.float()[:, None]}
        steady = (active & sample.policy_owned & sample.healthy & ~sample.true_failure
                  & (sample.timestamp_s - self._start - self.dt >= 2. - 1e-9)
                  & (sample.command_mode != int(CommandMode.JUMP)) & (sample.phase == int(RewardPhase.SUPPORT)))
        self._steady_run_s = torch.where(steady, self._steady_run_s + self.dt, 0)
        self._longest_steady_s = torch.maximum(self._longest_steady_s, self._steady_run_s)
        masks = {'full_task': active, 'policy': active & sample.policy_owned, 'steady_support': steady}
        packed = torch.cat(tuple(values[name] for name in self._columns), -1)
        for scope, mask in masks.items():
            support = mask[:, None].expand_as(packed).clone()
            if not self._effort_valid:
                support[:, self._columns['effort_slew']] = False
            support[:, self._columns['jump_world_z_error']] &= sample.command_mode[:, None] == int(CommandMode.JUMP)
            self._moments[scope].add(packed, support, self.dt)
        standing = (active & sample.policy_owned & sample.healthy & ~sample.true_failure
                    & sample.requested_command[:, :3].abs().le(1e-7).all(-1)
                    & sample.admitted_command[:, :3].abs().le(1e-7).all(-1)
                    & (sample.phase == int(RewardPhase.SUPPORT)))
        if self.tick % self.substeps == 0:
            self._standing_interval.fill_(True)
        self._standing_interval &= standing
        self._previous_effort = sample.active_effort_nm.detach().clone()
        self._previous_position = sample.root_position_world_m.detach().clone()
        self._effort_valid, self._interval_closed = True, False
        self._last_time, self.tick = sample.timestamp_s.detach().clone(), sample.physical_tick

    @torch.no_grad()
    def close_interval(self, terminated, truncated, failure_reasons):
        if self._interval_closed or self.tick % self.substeps or self._last_time is None:
            raise ValueError('evaluation endpoint requires one newly complete100Hz native interval')
        for name, value, dtype in (('terminated', terminated, torch.bool), ('truncated', truncated, torch.bool),
                                    ('failure reasons', failure_reasons, torch.long)):
            self._tensor(name, value, dtype=dtype)
        if ((terminated & truncated).any() or (failure_reasons < 0).any()
                or (~self.finished & ((failure_reasons != 0) != terminated)).any()):
            raise ValueError('first endpoint true-failure reasons and truncation semantics differ')
        self._endpoint_index += 1
        index = self._endpoint_index % 1001
        self._position_ring[index] = self._previous_position[:, :2].double()
        self._standing_ticks = torch.where(self._standing_interval, self._standing_ticks + 1, 0)
        drift = (self._position_ring[index] - self._position_ring[(self._endpoint_index - 1000) % 1001]).norm(dim=-1)
        valid = self._standing_ticks >= 1000
        self._drift_count += valid.long()
        self._max_drift = torch.where(valid, torch.maximum(self._max_drift, drift), self._max_drift)
        ended = ~self.finished & (terminated | truncated)
        self.terminated[ended], self.truncated[ended] = terminated[ended], truncated[ended]
        self.failure_reasons[ended] = failure_reasons[ended]
        self._end[ended] = self._last_time[ended]
        self.finished |= ended
        self._interval_closed = True

    def reports(self):
        if not self._interval_closed:
            raise RuntimeError('cannot publish a partly observed physical interval')
        result = []
        for row in range(self.num_envs):
            started, finished = self.started[row].item(), self.finished[row].item()
            horizon = self.horizon[row].item()
            seconds = self._seconds[row].item()
            full = (started and finished and self.truncated[row].item() and not self.terminated[row].item()
                    and seconds + 1e-8 >= horizon and self.whole_healthy[row].item() and self.whole_owned[row].item())
            drift_count = self._drift_count[row].item()
            result.append({
                'episode_id': self._episode[row].item(), 'started': started, 'finished': finished,
                'outcome': 'true_failure' if self.terminated[row].item() else 'time_limit' if self.truncated[row].item()
                else 'incomplete', 'failure_reasons': self.failure_reasons[row].item(),
                'declared_horizon_s': horizon, 'observed_task_s': seconds,
                'policy_owned_s': self._policy_seconds[row].item(),
                'scripted_task_s': seconds - self._policy_seconds[row].item(),
                'whole_task_healthy': self.whole_healthy[row].item() if started else False,
                'whole_task_policy_owned': self.whole_owned[row].item() if started else False,
                'healthy_policy_owned_full_horizon': bool(full),
                'longest_steady_support_s': self._longest_steady_s[row].item(),
                'steady_support_has_2s_contiguous_interval': self._longest_steady_s[row].item() >= 2. - 1e-8,
                'steady_support_scope': 'after2s;healthy policy support only;moments pool retained segments',
                'world_xy_net_displacement_m': ((self._last_position[row, :2] - self._origin[row, :2]).double().norm().item()
                                                if started else None),
                'standing_10s_drift': {'windows': drift_count,
                                      'max_net_displacement_m': self._max_drift[row].item() if drift_count else None,
                                      'endpoint_rate_hz': 100, 'window_s': 10.},
                'moments': {scope: {name: {**moment.report(row, columns), 'units': list(self._WIDTHS[name][1])}
                                    for name, columns in self._columns.items()} for scope, moment in self._moments.items()},
                'mechanical_power_is_not_electrical_energy': True,
                'skill_or_terrain_success_inferred_from_health': False,
                'physical_qualification': False,
            })
        return result


class NativeEvaluationTrace:
    """Read-only adapter from this environment's actual substeps to statistics."""

    def __init__(self, env: NativeTaskEnvironment, statistics: FirstEpisodeStatistics):
        if not isinstance(env, NativeTaskEnvironment) or not isinstance(statistics, FirstEpisodeStatistics):
            raise TypeError('native evaluation trace requires its actual task and first-episode statistics')
        self.env, self.statistics = env, statistics
        self._epoch = env.lifecycle.plant.scene.observation.model_epoch

    def __call__(self, tick, sample, condition, reference):
        env, plant = self.env, self.env.lifecycle.plant
        if (tick is not plant._last_tick_packet or sample is not env.rewards._last_sample
                or sample.physical_tick != tick.physical_tick or sample.model_epoch != self._epoch
                or plant.scene.observation.model_epoch != self._epoch):
            raise ValueError('evaluation trace received a detached,stale or changed-model physical packet')
        p, boundary = sample.physics, sample.boundary
        admission = env.commands.last_admission
        requested = reference.admitted_body_command if admission is None else admission.requested
        projected = (torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) if admission is None
                     else ~admission.wheel_necessary_at_request)
        wheels = list(plant.scene.observation.binding.wheel_body_indices)
        harmful = [i for i in range(19) if i not in wheels]
        self.statistics.observe(EvaluationPhysicalSample(
            tick.physical_tick, p.timestamp_s, p.episode_id, p.control_epoch, p.policy_owned,
            condition.healthy, condition.true_failure, requested, reference.admitted_body_command,
            reference.mode, reference.phase, p.velocity_origin_body_m_s, p.velocity_origin_world_m_s,
            tick.body_motion.link_pose_world_xyzw[:, 0, :3], p.height_above_ground_m,
            p.roll_pitch_rad - reference.target_roll_pitch_rad, p.angular_velocity_body_rad_s,
            p.root_world_z_m - reference.jump_height_world_z_m, p.active_explicit_effort_nm,
            p.policy_velocity_rad_s, tick.feedback.saturated, plant.targets.issued_action,
            p.wheel_contact_tangent_speed_m_s, tick.wheel_clearance.geometry.vertical_gap_m,
            p.body_normal_load_n[:, wheels], p.body_normal_load_n[:, harmful].sum(-1),
            boundary.knee_signed_clearance_rad, boundary.knee_pivot_residual_m, projected))


@torch.no_grad()
def evaluate_first_episodes(model, env: NativeTaskEnvironment, *, max_vector_steps,
                            config=StructuredCollectionConfig(), boundary_inputs=None):
    """Fixed mean actions, shared collection contract, five-frame causal resets.

    This lower layer does not claim that a scene implements a catalogue case.
    The formal runner must separately bind the recipe, request schedule, selected
    model/split, nonrefundable evaluation reservation and per-case completion.
    """
    if (not isinstance(env, NativeTaskEnvironment) or type(max_vector_steps) is not int or max_vector_steps < 1
            or boundary_inputs is not None and not callable(boundary_inputs)):
        raise ValueError('native evaluation requires an owned task and an explicit bounded vector-step budget')
    contract = StructuredCollector(env, model, config)
    modes = [(module, module.training) for module in model.modules()]
    previous_sink = env.trace_sink
    vector_steps = 0
    requests = [[] for _ in range(env.num_envs)]
    action_moments = {name: _Moments(env.num_envs, 6, env.device)
                      for name in ('raw_mean', 'issued', 'issued_rate', 'lower_bound_fraction', 'upper_bound_fraction')}
    try:
        model.eval()
        version = contract._version()
        current = contract.reset()
        bridge = env.lifecycle.plant.scene.observation
        statistics = FirstEpisodeStatistics(current.episode_id, env.commands.task_horizon_s,
                                             physical_tick=bridge.physical_tick, physics_dt_s=env.lifecycle.plant.dt)
        trace = NativeEvaluationTrace(env, statistics)

        def observe(tick, sample, condition, reference):
            trace(tick, sample, condition, reference)
            if previous_sink is not None:
                previous_sink(tick, sample, condition, reference)

        env.trace_sink = observe
        history = contract._history
        for index in range(max_vector_steps):
            contract._check_version(version)
            mean, state, _ = model.actor.policy.outputs(history)
            if not torch.isfinite(mean).all() or not torch.isfinite(state).all():
                raise FloatingPointError('nonfinite evaluation output before physical action submission')
            issued = mean.clamp(-contract.bounds, contract.bounds)
            estimate = contract.completed_estimate(current, state)
            if boundary_inputs is not None:
                incoming = boundary_inputs(index, current.owned(), statistics.finished.clone())
                if incoming is not None:
                    env.queue_requests(incoming)
            vector_steps += 1
            result = contract._step(env.step(issued.clone(), estimate), current, issued)
            contract._check_version(version)
            outcome = env.last_outcome
            action_support = (~statistics.finished & result.fully_policy_owned)[:, None].expand_as(mean)
            action_values = {'raw_mean': mean, 'issued': issued,
                             'issued_rate': (issued - current.frame[:, 22:28]) / config.policy_dt_s,
                             'lower_bound_fraction': (mean < -contract.bounds).float(),
                             'upper_bound_fraction': (mean > contract.bounds).float()}
            for name, values in action_values.items():
                action_moments[name].add(values, action_support, config.policy_dt_s)
            evidence, receipt, condition = (outcome['physical_evidence'], outcome['skill_receipt'],
                                             outcome['request_condition'])
            delivered = outcome['request_input']
            reset_ids = (torch.full_like(delivered.request_id, -1) if outcome['reset'] is None
                         else outcome['reset'].aborted_request_ids)
            event_rows = (~statistics.finished & ((delivered.request_id >= 0) | (delivered.cancel_id >= 0)
                          | (receipt.accepted_id >= 0) | (evidence.finished_request_id >= 0)
                          | (receipt.finished_id >= 0) | (condition.finished_id >= 0) | (reset_ids >= 0)))
            for row in event_rows.nonzero().flatten().tolist():
                if delivered.request_id[row] >= 0 or delivered.cancel_id[row] >= 0:
                    requests[row].append({'source': 'protocol_request',
                                          'request_id': delivered.request_id[row].item(),
                                          'requested_height_m': delivered.requested_height_m[row].item(),
                                          'cancel_id': delivered.cancel_id[row].item(),
                                          'response': RequestResponse(condition.response[row].item()).name,
                                          'at_s': receipt.observed_at_s[row].item()})
                if receipt.accepted_id[row] >= 0:
                    requests[row].append({'source': 'protocol_acceptance',
                                          'request_id': receipt.accepted_id[row].item(),
                                          'at_s': receipt.observed_at_s[row].item()})
                if evidence.finished_request_id[row] >= 0:
                    apex = evidence.apex_rise_m[row].item()
                    requests[row].append({'source': 'physical_evidence',
                                          'request_id': evidence.finished_request_id[row].item(),
                                          'outcome': SkillOutcome(evidence.outcome[row].item()).name,
                                          'at_s': evidence.finished_at_s[row].item(),
                                          'apex_rise_m': apex if math.isfinite(apex) else None,
                                          'apex_observed': math.isfinite(apex),
                                          'height_tolerance_m': evidence.height_tolerance_m[row].item(),
                                          'flight_seen': evidence.flight_seen[row].item(),
                                          'stable_landing_s': evidence.stable_landing_s[row].item(),
                                          'entire_skill_policy_owned': evidence.entire_skill_policy_owned[row].item(),
                                          'entire_skill_healthy': evidence.entire_skill_healthy[row].item()})
                if receipt.finished_id[row] >= 0:
                    requests[row].append({'source': 'physical_tracker_protocol_transition',
                                          'request_id': receipt.finished_id[row].item(),
                                          'outcome': SkillOutcome(receipt.outcome[row].item()).name,
                                          'at_s': receipt.observed_at_s[row].item()})
                if condition.finished_id[row] >= 0:
                    requests[row].append({'source': 'estimated_protocol_end',
                                          'request_id': condition.finished_id[row].item(),
                                          'end_reason': condition.end_reason[row].item(),
                                          'is_physical_success': False})
                if reset_ids[row] >= 0:
                    requests[row].append({'source': 'episode_reset',
                                          'request_id': reset_ids[row].item(),
                                          'outcome': 'RESET'})
            statistics.close_interval(outcome['terminated'], outcome['truncated'], outcome['failure_reasons'])
            next_observation = result.observation
            reset = ((next_observation.episode_id != current.episode_id)
                     | (next_observation.control_epoch != current.control_epoch))
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = next_observation.frame
            history[reset] = next_observation.frame[reset, None]
            current = next_observation
            if statistics.finished.all():
                break
        rows = statistics.reports()
        for index, (row, events) in enumerate(zip(rows, requests)):
            row['request_events'] = events
            actions = {}
            for name, moment in action_moments.items():
                value = moment.report(index, slice(0, 6))
                value['policy_intervals'] = value.pop('physical_samples')
                value['units'] = ['fraction' if name.endswith('fraction') else 'normalized_action_s'
                                  if name == 'issued_rate' else 'normalized_action'] * 6
                actions[name] = value
            row['policy_action_moments_100hz'] = actions
            binding = env.lifecycle.plant.scene.observation.binding
            row['policy_action_joint_order'] = [binding.native_joint_names[i] for i in binding.policy_joint_indices]
        return {'policy': 'deterministic_mean', 'vector_steps': vector_steps,
                'attempted_physical_transitions': vector_steps * env.num_envs,
                'first_episode_only': True, 'rows': rows,
                'all_first_episodes_finished': statistics.finished.all().item(),
                'scene_case_binding_verified': False, 'physical_qualification': False,
                'optimizer_steps': 0, 'budget_includes_preparation_and_post_first_rows': True}
    finally:
        env.trace_sink = previous_sink
        for module, training in modes:
            module.training = training

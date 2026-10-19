"""Private physical reward production with explicit ownership and event credit.

These functions do not infer contact-point slip, geometric knee margins, skill
success or physical qualification. Their named PRE-reset producer inputs must
come from the separately verified native environment. No simulator is started.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from ._physical_observation import CommandMode, MechanicalBinding


class RewardPhase(IntEnum):
    SUPPORT = 0
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3
    SETTLING = 4
    RECOVERY = 5


@dataclass(frozen=True)
class RewardTuning:
    robot_mass_kg: float
    gamma: float = .99498743710662
    policy_dt_s: float = .01
    cost_density_scale: float = 10.
    absorbing_density: float = -11.
    failure_event: float = -4.
    success_event: float = 1.
    gravity_m_s2: float = 9.81
    landing_load_multiple: float = 3.

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or min(self.robot_mass_kg, self.gravity_m_s2, self.landing_load_multiple) <= 0
                or self.policy_dt_s != .01 or not 0 < self.gamma < 1 or self.cost_density_scale != 10.
                or self.absorbing_density >= -self.cost_density_scale or self.failure_event >= 0
                or self.success_event < 0):
            raise ValueError('reward tuning requires explicit mass,100Hz discount and a strictly worse absorbing density')
        # Even the best terminal density must be worse than safe continuation
        # at its worst continuous density. This is a reward-bound assertion,
        # not a claim about an imperfect learned critic or finite evaluation.
        best_failure = 8 * self.policy_dt_s + self.failure_event + self.gamma * self.absorbing_value
        safe_lower = -self.cost_density_scale * self.policy_dt_s / (1 - self.gamma)
        if best_failure >= safe_lower:
            raise ValueError('failure reward could beat the declared safe-continuation lower bound')

    @property
    def absorbing_value(self):
        return self.absorbing_density * self.policy_dt_s / (1 - self.gamma)


@dataclass(frozen=True)
class RewardReference:
    """Current interval's visible command and reward-only physical skill target.

    Jump height is a WORLD-Z reference, while support height is the admitted
    base_link height above its declared ground. True phases are reward-only;
    neither these references nor true contact enter the deployable actor/FSM.
    """

    admitted_body_command: torch.Tensor
    mode: torch.Tensor
    phase: torch.Tensor
    jump_height_world_z_m: torch.Tensor
    jump_velocity_world_z_m_s: torch.Tensor
    target_roll_pitch_rad: torch.Tensor


@dataclass(frozen=True)
class RewardPhysics:
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    policy_owned: torch.Tensor
    velocity_origin_body_m_s: torch.Tensor
    velocity_origin_world_m_s: torch.Tensor
    root_world_z_m: torch.Tensor
    height_above_ground_m: torch.Tensor
    roll_pitch_rad: torch.Tensor
    gravity_body: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor
    policy_velocity_rad_s: torch.Tensor
    active_explicit_effort_nm: torch.Tensor
    body_normal_load_n: torch.Tensor
    wheel_contact_tangent_speed_m_s: torch.Tensor
    joint_boundary_shortfall_rad: torch.Tensor
    velocity_reference: str = 'base_link_origin'
    effort_reference: str = 'native_explicit_active_output_buffer'
    slip_reference: str = 'actual_contact_point_relative_tangent_speed'
    boundary_reference: str = 'geometric_passive_knees_then_four_output_targets'


@dataclass(frozen=True)
class RewardResult:
    learning_reward: torch.Tensor
    physical_step_reward: torch.Tensor
    component_densities: dict
    event_rewards: dict
    raw_positive_densities: dict
    raw_cost_densities: dict
    raw_error_means: dict
    raw_error_mean_squares: dict
    gate_fractions: dict
    fully_policy_owned: torch.Tensor
    success_credited: torch.Tensor
    physical_success_request_id: torch.Tensor
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor


class PhysicalRewardComposer:
    """Integrate exactly one policy interval, then publish one reward.

    Physical terms use every200Hz/1kHz sample with its actual dt. Issued-action
    differences are100Hz densities held over that interval, not differentiated
    again at the physics rate. All raw costs survive bounded composition.
    Events occur once; absorbing value adds no synthetic physical transition.
    """

    def __init__(self, binding: MechanicalBinding, num_envs, tuning: RewardTuning,
                 *, device='cpu', physics_dt_s=.005, retired_request_ids=None):
        if (not isinstance(binding, MechanicalBinding) or not isinstance(tuning, RewardTuning)
                or type(num_envs) is not int or num_envs < 1 or physics_dt_s not in (.005, .001)):
            raise ValueError('reward producer needs named mechanics, mass/discount and the selected physics rate')
        self.binding, self.num_envs, self.tuning = binding, num_envs, tuning
        self.device, self.physics_dt_s = torch.device(device), physics_dt_s
        self.substeps = round(tuning.policy_dt_s / physics_dt_s)
        self._wheels = list(binding.wheel_body_indices)
        self._harmful = [i for i in range(19) if i not in self._wheels]
        self._episode = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._control = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._time = torch.full((num_envs,), -float('inf'), dtype=torch.float64, device=self.device)
        self._ended = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._previous = torch.zeros(num_envs, 6, device=self.device)
        self._previous2 = torch.zeros_like(self._previous)
        self._action_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._last_dq = torch.zeros(num_envs, 6, device=self.device)
        self._dq_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._success_seen = torch.full_like(self._episode, -1)
        if retired_request_ids is not None:
            # Cold setup supplies the restored protocol's max_seen IDs. All
            # prior active physical requests were aborted, not replayed.
            self._tensor('retired request IDs', retired_request_ids, dtype=torch.long)
            if (retired_request_ids < -1).any():
                raise ValueError('retired request IDs cannot be below the explicit-1 sentinel')
            self._success_seen.copy_(retired_request_ids)
        self._window = None

    def _tensor(self, name, value, width=None, dtype=torch.float32):
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                or value.device != self.device):
            raise ValueError(name + ' differs from the named reward batch/device/dtype')

    @torch.no_grad()
    def reset(self, mask, now_s, episode_id, control_epoch):
        if self._window is not None:
            raise RuntimeError('preserve the PRE-reset reward endpoint before resetting its history')
        for name, value, dtype in (('mask', mask, torch.bool), ('now', now_s, torch.float64),
                                    ('episode', episode_id, torch.long), ('control', control_epoch, torch.long)):
            self._tensor(name, value, dtype=dtype)
        invalid = (~torch.isfinite(now_s) | (now_s < 0) | (episode_id < 0) | (control_epoch < 0)
                   | (mask & ((episode_id <= self._episode) | (control_epoch < self._control)
                              | (now_s < self._time))))
        if invalid.any():
            raise ValueError('reward reset must advance selected episode identities without rewinding physical time')
        self._episode[mask], self._control[mask], self._time[mask] = episode_id[mask], control_epoch[mask], now_s[mask]
        self._ended[mask] = False
        self._previous[mask], self._previous2[mask], self._action_count[mask] = 0, 0, 0
        self._dq_valid[mask] = False
        # Completed request IDs are deliberately not refunded by physical reset.

    @torch.no_grad()
    def begin(self, issued_action, initial_policy_velocity_rad_s, *, now_s, episode_id, control_epoch,
              policy_owned, admitted_body_command, command_mode):
        if self._window is not None:
            raise RuntimeError('one reward interval is already in progress')
        for name, value in (('issued action', issued_action), ('initial output velocity', initial_policy_velocity_rad_s)):
            self._tensor(name, value, 6)
        for name, value, dtype in (('now', now_s, torch.float64), ('episode', episode_id, torch.long),
                                    ('control', control_epoch, torch.long), ('owned', policy_owned, torch.bool)):
            self._tensor(name, value, dtype=dtype)
        self._tensor('actor-visible admitted command', admitted_body_command, 4)
        self._tensor('actor-visible mode', command_mode, dtype=torch.long)
        invalid = (self._ended | (episode_id != self._episode) | (control_epoch < self._control)
                   | ~torch.isfinite(now_s) | ((now_s - self._time).abs() > 1e-8)
                   | ~torch.isfinite(issued_action).all(-1) | ~torch.isfinite(initial_policy_velocity_rad_s).all(-1)
                   | (self._dq_valid & (initial_policy_velocity_rad_s - self._last_dq).abs().gt(1e-6).any(-1))
                   | (issued_action.abs() > issued_action.new_tensor((3, 3, 3, 3, 9, 9)) + 1e-6).any(-1)
                   | ~torch.isfinite(admitted_body_command).all(-1) | admitted_body_command[:, 1].abs().gt(1e-8)
                   | (admitted_body_command[:, 3] <= 0) | (command_mode < 0) | (command_mode > 4))
        if invalid.any():
            raise ValueError('reward interval needs an initialized same-time PRE-reset episode and clipped issued action')
        first = .04 * (issued_action - self._previous).square().sum(-1)
        second = (issued_action - 2 * self._previous + self._previous2).square()
        second = .8 * second[:, :4].sum(-1) + .16 * second[:, 4:].sum(-1)
        self._window = {'start': now_s.clone(), 'episode': episode_id.clone(), 'control': control_epoch.clone(),
                        'issued': issued_action.clone(), 'dq': initial_policy_velocity_rad_s.clone(),
                        'owned': policy_owned.clone(), 'steps': 0, 'positive': {}, 'raw_cost': {},
                        'command': admitted_body_command.clone(), 'mode': command_mode.clone(),
                        'components': {}, 'gates': {}, 'errors': {}, 'error_squares': {},
                        'action_cost': torch.where(self._action_count > 0, first, 0),
                        'action2_cost': torch.where(self._action_count > 1, second, 0)}

    @staticmethod
    def _huber(value):
        return torch.hypot(torch.ones_like(value), value) - 1

    def _inputs(self, sample, reference):
        if not isinstance(sample, RewardPhysics) or not isinstance(reference, RewardReference):
            raise TypeError('physical reward requires named PRE-reset sample and current reference')
        if (sample.velocity_reference != 'base_link_origin' or sample.effort_reference != 'native_explicit_active_output_buffer'
                or sample.slip_reference != 'actual_contact_point_relative_tangent_speed'
                or sample.boundary_reference != 'geometric_passive_knees_then_four_output_targets'):
            raise ValueError('reward source reference or output-axis units differ')
        widths = {'velocity_origin_body_m_s': 3, 'velocity_origin_world_m_s': 3, 'root_world_z_m': None,
                  'height_above_ground_m': None, 'roll_pitch_rad': 2, 'gravity_body': 3,
                  'angular_velocity_body_rad_s': 3, 'policy_velocity_rad_s': 6, 'active_explicit_effort_nm': 6,
                  'body_normal_load_n': 19, 'wheel_contact_tangent_speed_m_s': 2, 'joint_boundary_shortfall_rad': 6}
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, width in widths.items():
            value = getattr(sample, name)
            self._tensor(name, value, width)
            invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        for name, dtype in (('timestamp_s', torch.float64), ('episode_id', torch.long),
                            ('control_epoch', torch.long), ('policy_owned', torch.bool)):
            self._tensor(name, getattr(sample, name), dtype=dtype)
        for name, width, dtype in (('admitted_body_command', 4, torch.float32), ('mode', None, torch.long),
                                   ('phase', None, torch.long), ('jump_height_world_z_m', None, torch.float32),
                                   ('jump_velocity_world_z_m_s', None, torch.float32),
                                   ('target_roll_pitch_rad', 2, torch.float32)):
            value = getattr(reference, name)
            self._tensor('reference.' + name, value, width, dtype)
            invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        invalid |= ((sample.body_normal_load_n < 0).any(-1) | (sample.wheel_contact_tangent_speed_m_s < 0).any(-1)
                    | (sample.joint_boundary_shortfall_rad < 0).any(-1)
                    | (sample.gravity_body.norm(dim=-1) - 1).abs().gt(1e-4)
                    | (reference.mode < 0) | (reference.mode > 4)
                    | reference.admitted_body_command[:, 1].abs().gt(1e-8)
                    | (reference.admitted_body_command[:, 3] <= 0))
        return invalid

    @torch.no_grad()
    def observe(self, sample: RewardPhysics, reference: RewardReference):
        window = self._window
        if window is None or window['steps'] >= self.substeps:
            raise RuntimeError('physical reward has no open complete-policy interval capacity')
        invalid = self._inputs(sample, reference)
        invalid |= (~torch.isfinite(sample.timestamp_s)
                    | (sample.timestamp_s - window['start'] - (window['steps'] + 1) * self.physics_dt_s).abs().gt(1e-8)
                    | (sample.episode_id != window['episode']) | (sample.control_epoch != window['control'])
                    | (reference.admitted_body_command != window['command']).any(-1)
                    | (reference.mode != window['mode']))
        jump = reference.mode == int(CommandMode.JUMP)
        recovery = reference.mode == int(CommandMode.RECOVERY)
        ordinary = ~jump & ~recovery
        phase = reference.phase
        invalid |= ((jump & ((phase < int(RewardPhase.TAKEOFF)) | (phase > int(RewardPhase.SETTLING))))
                    | (recovery & (phase != int(RewardPhase.RECOVERY)))
                    | (ordinary & (phase != int(RewardPhase.SUPPORT))))
        velocity, command = sample.velocity_origin_body_m_s, reference.admitted_body_command
        ev, ew = velocity[:, 0] - command[:, 0], sample.angular_velocity_body_rad_s[:, 2] - command[:, 2]
        support_eh = sample.height_above_ground_m - command[:, 3]
        jump_eh = sample.root_world_z_m - reference.jump_height_world_z_m
        eh = torch.where(jump, jump_eh, support_eh)
        evz = sample.velocity_origin_world_m_s[:, 2] - reference.jump_velocity_world_z_m_s
        pose = sample.roll_pitch_rad - reference.target_roll_pitch_rad
        upright = (-sample.gravity_body[:, 2]).clamp(-1, 1)
        height_gate = torch.where(recovery, upright.clamp_min(0), torch.ones_like(upright))
        # Jump locomotion2+2 plus height2/vertical1/attitude1 is at most8.
        positive = {'forward': torch.where(ordinary, 3., torch.where(jump, 2., 0.)) * torch.exp(-(ev / .35).square()),
                    'yaw': torch.where(recovery, 0., 2.) * torch.exp(-(ew / .5).square()),
                    'height': 2 * height_gate * torch.exp(-(eh / .025).square()),
                    'jump_vertical': jump * torch.exp(-(evz / .35).square()),
                    'jump_attitude': jump * torch.exp(-(pose / .25).square().sum(-1)),
                    'recovery_upright': recovery * (2 * (1 + upright))}
        h = self._huber
        effort, dq = sample.active_explicit_effort_nm, sample.policy_velocity_rad_s
        ddq = (dq - window['dq']) / self.physics_dt_s
        still = ordinary & command[:, :3].eq(0).all(-1)
        wheel_load = sample.body_normal_load_n[:, self._wheels]
        slip = wheel_load > 5
        landing = jump & ((phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING)))
        weight_n = self.tuning.robot_mass_kg * self.tuning.gravity_m_s2
        excess_landing = (wheel_load.sum(-1) / weight_n - self.tuning.landing_load_multiple).clamp_min(0)
        cost = {'forward': (~recovery) * h(ev / .5), 'yaw': (~recovery) * h(ew),
                'height': 2 * height_gate * h(eh / .04),
                'attitude': torch.where(recovery, 2 * h(pose / .8).sum(-1),
                                         torch.where(jump, .25 * h(pose / .5).sum(-1), (pose / .2).square().sum(-1))),
                'stand_world_xy': still * 1.5 * h(sample.velocity_origin_world_m_s[:, :2].norm(dim=-1) / .05),
                'lateral_velocity': .5 * (velocity[:, 1] / .25).square(),
                'ordinary_vertical': ordinary * .2 * (velocity[:, 2] / .3).square(),
                'jump_vertical': jump * h(evz / .5),
                'effort': 1e-4 * effort[:, :4].square().sum(-1) + 1e-3 * effort[:, 4:].square().sum(-1),
                'mechanical_power_proxy': .002 * (effort * dq).abs().sum(-1),
                'leg_velocity': 1e-4 * dq[:, :4].square().sum(-1),
                'leg_acceleration': 1e-7 * ddq[:, :4].square().sum(-1),
                'issued_first_difference': window['action_cost'], 'issued_second_difference': window['action2_cost'],
                'harmful_contact': 2 * (sample.body_normal_load_n[:, self._harmful] > 3).sum(-1),
                'contact_slip': .5 * (slip * (sample.wheel_contact_tangent_speed_m_s / .2).square()).sum(-1),
                'joint_boundary': 2 * (sample.joint_boundary_shortfall_rad / .05).square().sum(-1),
                'flight_wheel_effort': .2 * (jump & (phase == int(RewardPhase.FLIGHT))) * (effort[:, 4:] / 4.5).square().sum(-1),
                'landing_impact': .5 * landing * h(excess_landing)}
        p, c = torch.stack(tuple(positive.values())).sum(0), torch.stack(tuple(cost.values())).sum(0)
        scale = self.tuning.cost_density_scale
        attenuation = torch.where(c > 0, scale * torch.tanh(c / scale) / c.clamp_min(1e-20), torch.ones_like(c))
        components = {**{'positive.' + key: value for key, value in positive.items()},
                      **{'bounded_cost.' + key: -value * attenuation for key, value in cost.items()}}
        invalid |= (~torch.isfinite(p) | ~torch.isfinite(c) | (p < 0) | (p > 8 + 1e-5) | (c < 0)
                    | ~torch.isfinite(torch.stack(tuple(components.values()))).all(0))
        if invalid.any():
            raise ValueError('reward input, PRE-reset identity, timing, phase or finite density bound differs')
        gates = {'ordinary': ordinary, 'jump': jump, 'recovery': recovery, 'exact_stationary': still,
                 'slip_left': slip[:, 0], 'slip_right': slip[:, 1], 'landing': landing,
                 'height_reference_weight': height_gate,
                 'first_action_difference': self._action_count > 0, 'second_action_difference': self._action_count > 1}
        errors = {'forward_m_s': ev, 'yaw_rad_s': ew, 'support_height_m': support_eh,
                  'jump_world_height_m': jump_eh, 'jump_world_vertical_m_s': evz,
                  'roll_rad': pose[:, 0], 'pitch_rad': pose[:, 1],
                  'stationary_world_xy_speed_m_s': sample.velocity_origin_world_m_s[:, :2].norm(dim=-1)}
        for key, values in (('positive', positive), ('raw_cost', cost), ('components', components), ('gates', gates),
                            ('errors', errors), ('error_squares', {key: value.double().square() for key, value in errors.items()})):
            for name, value in values.items():
                contribution = value.to(torch.float64) * self.physics_dt_s
                if name in window[key]:
                    window[key][name].add_(contribution)
                else:
                    window[key][name] = contribution.clone()
        window['owned'] &= sample.policy_owned
        window['dq'], window['steps'] = dq.detach().clone(), window['steps'] + 1
        window['last_timestamp'] = sample.timestamp_s.detach().clone()

    @torch.no_grad()
    def finish_from_evidence(self, evidence, *, terminated, truncated):
        """Only complete, matching physical receipts may pay skill credit.

        The lower-level finish kernel stays independently testable. Native
        environment composition uses this bridge, not a request-FSM estimate.
        Python packet types are ownership contracts, not qualification proof.
        """
        from ._skill_evidence import SkillEvidence, SkillOutcome

        window = self._window
        if window is None or window['steps'] != self.substeps:
            raise RuntimeError('physical evidence publication requires a complete reward interval')
        if not isinstance(evidence, SkillEvidence):
            raise TypeError('skill credit requires independent physical evidence, not protocol completion')
        for name, dtype in (('timestamp_s', torch.float64), ('episode_id', torch.long),
                            ('control_epoch', torch.long), ('physical_success_request_id', torch.long),
                            ('finished_request_id', torch.long), ('finished_at_s', torch.float64),
                            ('outcome', torch.long), ('entire_skill_policy_owned', torch.bool),
                            ('entire_skill_healthy', torch.bool), ('supported_start', torch.bool),
                            ('flight_seen', torch.bool), ('requested_rise_m', torch.float32),
                            ('apex_rise_m', torch.float32), ('height_tolerance_m', torch.float32),
                            ('stable_landing_s', torch.float64)):
            self._tensor('physical evidence.' + name, getattr(evidence, name), dtype=dtype)
        success = evidence.physical_success_request_id >= 0
        completed = evidence.outcome == int(SkillOutcome.PHYSICAL_SUCCESS)
        invalid = (~torch.isfinite(evidence.timestamp_s) | ~torch.isfinite(evidence.finished_at_s)
                   | (evidence.timestamp_s != window['last_timestamp'])
                   | (evidence.episode_id != window['episode']) | (evidence.control_epoch != window['control'])
                   | (evidence.physical_success_request_id < -1) | (evidence.finished_request_id < -1)
                   | (evidence.outcome < 0) | (evidence.outcome > int(SkillOutcome.SAFETY_ABORT))
                   | ((evidence.finished_request_id >= 0) != (evidence.outcome != int(SkillOutcome.NONE)))
                   | ((evidence.finished_request_id >= 0)
                      & ((evidence.finished_at_s > evidence.timestamp_s) | (evidence.finished_at_s < 0)))
                   | (success != completed))
        if (type(evidence.physical_tick) is not int or evidence.physical_tick < 1
                or evidence.physical_tick % self.substeps):
            raise ValueError('physical evidence needs its actual positive native tick')
        invalid |= (evidence.timestamp_s - evidence.physical_tick * self.physics_dt_s).abs().gt(1e-8)
        invalid |= success & ((evidence.physical_success_request_id != evidence.finished_request_id)
                              | (evidence.finished_at_s != evidence.timestamp_s)
                              | ~evidence.entire_skill_policy_owned | ~evidence.entire_skill_healthy
                              | ~evidence.supported_start | ~evidence.flight_seen
                              | ~torch.isfinite(evidence.apex_rise_m) | ~torch.isfinite(evidence.requested_rise_m)
                              | ~torch.isfinite(evidence.height_tolerance_m) | ~torch.isfinite(evidence.stable_landing_s)
                              | (evidence.requested_rise_m < .03) | (evidence.requested_rise_m > .30)
                              | (evidence.height_tolerance_m <= 0) | (evidence.height_tolerance_m > .03 + 1e-6)
                              | (evidence.height_tolerance_m > .5 * evidence.requested_rise_m + 1e-6)
                              | ((evidence.apex_rise_m - evidence.requested_rise_m).abs()
                                 > evidence.height_tolerance_m + 1e-6)
                              | (evidence.stable_landing_s < 2. - 1e-9))
        if invalid.any():
            raise ValueError('physical evidence differs from reward identity, endpoint or complete-skill criteria')
        return self.finish(terminated=terminated, truncated=truncated,
                            physical_success_request_id=evidence.physical_success_request_id,
                            entire_skill_policy_owned=evidence.entire_skill_policy_owned)

    @torch.no_grad()
    def finish(self, *, terminated, truncated, physical_success_request_id, entire_skill_policy_owned):
        window = self._window
        if window is None or window['steps'] != self.substeps:
            raise RuntimeError('reward publication requires every declared physical sample exactly once')
        for name, value, dtype in (('terminated', terminated, torch.bool), ('truncated', truncated, torch.bool),
                                    ('success ID', physical_success_request_id, torch.long),
                                    ('skill owned', entire_skill_policy_owned, torch.bool)):
            self._tensor(name, value, dtype=dtype)
        if (physical_success_request_id < -1).any():
            raise ValueError('physical success IDs require a real request or the explicit-1 sentinel')
        fresh_success = (physical_success_request_id >= 0) & (physical_success_request_id > self._success_seen)
        credited = fresh_success & entire_skill_policy_owned & window['owned'] & ~terminated
        failure = terminated.to(torch.float64)
        events = {'failure': failure * self.tuning.failure_event,
                  'failure_absorbing_value_tail': failure * self.tuning.gamma * self.tuning.absorbing_value,
                  'physical_skill_success': credited.to(torch.float64) * self.tuning.success_event}
        physical = torch.stack(tuple(window['components'].values())).sum(0) + torch.stack(tuple(events.values())).sum(0)
        learning = torch.where(window['owned'], physical, 0).to(torch.float32)
        density = lambda values: {key: value.detach().clone() / self.tuning.policy_dt_s for key, value in values.items()}
        # Preserve the actual producer clock. Repeated floating additions of
        # policy_dt can drift from tick*physics_dt even at the same boundary.
        timestamp = window['last_timestamp']
        result = RewardResult(learning, physical.clone(), density(window['components']), events,
                              density(window['positive']), density(window['raw_cost']), density(window['errors']),
                              density(window['error_squares']), density(window['gates']),
                              window['owned'].clone(), credited.clone(), physical_success_request_id.clone(), timestamp.clone(),
                              window['episode'].clone(), window['control'].clone())
        self._previous2.copy_(self._previous)
        self._previous.copy_(window['issued'])
        self._last_dq.copy_(window['dq'])
        self._dq_valid.fill_(True)
        self._action_count.add_(1).clamp_max_(2)
        self._time, self._control = timestamp, window['control'].clone()
        self._ended = terminated | truncated
        self._success_seen = torch.maximum(self._success_seen, physical_success_request_id)
        self._window = None
        return result

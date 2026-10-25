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

from ..control.ownership import ControlSource, validate_issued_control

from ..control.observation import CommandMode, MechanicalBinding


REWARD_VERSION = 'locomotion_v4'
REWARD_POSITIVE_TERMS = (
    'forward', 'yaw', 'height', 'attitude', 'jump_vertical', 'jump_attitude', 'recovery_upright',
)
REWARD_COST_TERMS = (
    'forward', 'yaw', 'height', 'attitude', 'stand_world_xy',
    'ordinary_vertical', 'jump_vertical', 'leg_fork', 'effort', 'mechanical_power_proxy',
    'wheel_velocity', 'wheel_acceleration',
    'leg_velocity', 'leg_acceleration', 'issued_first_difference',
    'issued_second_difference', 'harmful_contact', 'contact_slip', 'joint_boundary',
    'roll_pitch_angular_velocity', 'flight_wheel_effort', 'landing_impact',
)
REWARD_TERMS = REWARD_POSITIVE_TERMS + REWARD_COST_TERMS
# Reward-side command deadband (SCUT): smaller requests are tracked as zero and
# open the standing gates. The actor still observes the admitted command.
DEADBAND_FORWARD_M_S = .1
DEADBAND_YAW_RAD_S = .1
# Fore-aft wheel split (SCUT no_fork): base-frame x(L) - x(R), metres.
FORK_TOLERANCE_M = .05
# SCUT base run vel_height_gate (linear band): NORMAL-mode velocity tracking
# pays in full while |e_h| <= 0.05 m and nothing beyond 0.10 m, so the support
# height is held before velocity is tracked. SCUT's rough run drops the gate;
# stair/slope modes command heights away from the support height on purpose.
VELOCITY_HEIGHT_GATE_M = (.05, .10)
# SCUT trained runs: track_{lin_vel_xy,ang_vel_z}_square -0.1, stand_still_lin_vel
# -2.0 (from-scratch flat run); per second, before the bounded composition.
VELOCITY_SQUARE_WEIGHT = .1
STAND_STILL_WEIGHT = 2.


def wheel_positions_in_base(link_pose_world_xyzw, base_index, wheel_indices):
    """Left/right wheel link origins in the base_link frame, metres.

    ``link_pose_world_xyzw`` is the native [N, bodies, 7] world pose with an
    xyzw quaternion. The inverse root rotation is v + 2w(c x v) + 2c x (c x v)
    with c = -xyz, i.e. the conjugate quaternion.
    """
    base = link_pose_world_xyzw[:, base_index]
    offset = link_pose_world_xyzw[:, list(wheel_indices), :3] - base[:, None, :3]
    q = base[:, None, 3:7] / base[:, None, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-12)
    conjugate, scalar = (-q[..., :3]).expand_as(offset), q[..., 3:]
    cross = torch.linalg.cross(conjugate, offset)
    return offset + 2 * (scalar * cross + torch.linalg.cross(conjugate, cross))


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
    # SCUT termination -200 x 0.02 s: one event per failure, no absorbing tail.
    failure_event: float = -4.
    success_event: float = 1.
    gravity_m_s2: float = 9.81
    landing_load_multiple: float = 3.
    angular_damping_ordinary: float = .05
    angular_damping_jump: float = .10
    angular_damping_recovery: float = 0.

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or min(self.robot_mass_kg, self.gravity_m_s2, self.landing_load_multiple) <= 0
                or min(self.angular_damping_ordinary, self.angular_damping_jump,
                       self.angular_damping_recovery) < 0
                or self.policy_dt_s not in (.01, .02) or not 0 < self.gamma < 1 or self.failure_event >= 0
                or self.success_event < 0):
            raise ValueError('reward tuning requires explicit mass, policy-period discount and a negative failure event')


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
    intentional_base_contact: torch.Tensor | None = None


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
    # [N,2,3] wheel link origins in the base frame; required by ``leg_fork``.
    wheel_position_base_m: torch.Tensor | None = None
    # None preserves the original full formal reward. A tuple names the terms
    # whose physical producers exist; unavailable terms are not silently faked.
    available_terms: tuple[str, ...] | None = None
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
    differences are policy-period densities held over that interval, not differentiated
    again at the physics rate. As in SCUT, the reward is the plain sum of the
    positive and cost terms (no cost bound) plus once-only events.
    """

    def __init__(self, binding: MechanicalBinding, num_envs, tuning: RewardTuning,
                 *, device='cpu', physics_dt_s=.005, retired_request_ids=None, robot_mass_kg_by_environment=None):
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
        self._mass_kg = torch.full((num_envs,), tuning.robot_mass_kg, device=self.device, dtype=torch.float32)
        if robot_mass_kg_by_environment is not None:
            self.update_native_mass(robot_mass_kg_by_environment)

    def _tensor(self, name, value, width=None, dtype=torch.float32):
        shape = (self.num_envs,) if width is None else (self.num_envs, width)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                or value.device != self.device):
            raise ValueError(name + ' differs from the named reward batch/device/dtype')

    @torch.no_grad()
    def update_native_mass(self, mass_kg):
        if self._window is not None:
            raise RuntimeError('mass readback cannot change an open physical reward interval')
        self._tensor('native total mass', mass_kg)
        if (~torch.isfinite(mass_kg) | (mass_kg <= 0)).any():
            raise ValueError('native per-environment mass must be finite and positive')
        self._mass_kg = mass_kg.detach().clone()

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
              policy_owned, admitted_body_command, command_mode, control_source=None):
        if self._window is not None:
            raise RuntimeError('one reward interval is already in progress')
        for name, value in (('issued action', issued_action), ('initial output velocity', initial_policy_velocity_rad_s)):
            self._tensor(name, value, 6)
        for name, value, dtype in (('now', now_s, torch.float64), ('episode', episode_id, torch.long),
                                    ('control', control_epoch, torch.long), ('owned', policy_owned, torch.bool)):
            self._tensor(name, value, dtype=dtype)
        self._tensor('actor-visible admitted command', admitted_body_command, 4)
        self._tensor('actor-visible mode', command_mode, dtype=torch.long)
        if control_source is None:
            control_source = torch.where(policy_owned, int(ControlSource.POLICY), int(ControlSource.BOUNDED_SCRIPT))
        validate_issued_control(issued_action, control_source, num_envs=self.num_envs,
                                device=self.device, policy_owned=policy_owned)
        invalid = (self._ended | (episode_id != self._episode) | (control_epoch < self._control)
                   | ~torch.isfinite(now_s) | ((now_s - self._time).abs() > 1e-8)
                   | ~torch.isfinite(issued_action).all(-1) | ~torch.isfinite(initial_policy_velocity_rad_s).all(-1)
                   | (self._dq_valid & (initial_policy_velocity_rad_s - self._last_dq).abs().gt(1e-6).any(-1))
                   | ~torch.isfinite(admitted_body_command).all(-1) | admitted_body_command[:, 1].abs().gt(1e-8)
                   | (admitted_body_command[:, 3] <= 0) | (command_mode < 0) | (command_mode > 4))
        if invalid.any():
            raise ValueError('reward interval needs an initialized same-time PRE-reset episode and validated issued control')
        # SCUT action_rate 0.01 and smoothness 0.05 (legs) / 0.01 (wheels) per
        # second, per physical radian: its leg unit is 0.5 rad, ours 0.25 rad,
        # so leg weights are divided by 4; wheel units (10 rad/s) are equal.
        first = (issued_action - self._previous).square()
        first = .0025 * first[:, :4].sum(-1) + .01 * first[:, 4:].sum(-1)
        second = (issued_action - 2 * self._previous + self._previous2).square()
        second = .0125 * second[:, :4].sum(-1) + .01 * second[:, 4:].sum(-1)
        self._window = {'start': now_s.clone(), 'episode': episode_id.clone(), 'control': control_epoch.clone(),
                        'issued': issued_action.clone(), 'dq': initial_policy_velocity_rad_s.clone(),
                        'control_source': control_source.detach().clone(),
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
        if sample.available_terms is not None:
            if (not isinstance(sample.available_terms, tuple)
                    or len(set(sample.available_terms)) != len(sample.available_terms)
                    or any(name not in REWARD_TERMS for name in sample.available_terms)):
                raise ValueError('reward available term list is unknown or duplicated')
        fork_paid = sample.available_terms is None or 'leg_fork' in sample.available_terms
        if sample.wheel_position_base_m is not None or fork_paid:
            value = sample.wheel_position_base_m
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 2, 3)
                    or value.dtype != torch.float32 or value.device != self.device):
                raise ValueError('leg_fork requires [N,2,3] base-frame wheel positions')
            invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        if reference.intentional_base_contact is not None:
            self._tensor('intentional base contact', reference.intentional_base_contact, dtype=torch.bool)
            invalid |= reference.intentional_base_contact & (reference.mode != int(CommandMode.STAIR))
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
        # Heading frame from the measured up axis u = -g_body: forward f is
        # body x projected normal to u, left = u x f. Velocities in m/s, yaw
        # rate about the world vertical in rad/s; pitching cannot leak v_up.
        up = -sample.gravity_body
        forward = torch.zeros_like(up)
        forward[:, 0] = 1
        forward = forward - (forward * up).sum(-1, keepdim=True) * up
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.linalg.cross(up, forward)
        v_forward, v_left = (velocity * forward).sum(-1), (velocity * left).sum(-1)
        v_up = (velocity * up).sum(-1)
        yaw_rate = (sample.angular_velocity_body_rad_s * up).sum(-1)
        target_v = torch.where(command[:, 0].abs() < DEADBAND_FORWARD_M_S, 0., command[:, 0])
        target_w = torch.where(command[:, 2].abs() < DEADBAND_YAW_RAD_S, 0., command[:, 2])
        ev, ew = v_forward - target_v, yaw_rate - target_w
        support_eh = sample.height_above_ground_m - command[:, 3]
        full, zero = VELOCITY_HEIGHT_GATE_M
        velocity_gate = torch.where(reference.mode == int(CommandMode.NORMAL),
                                    ((zero - support_eh.abs()) / (zero - full)).clamp(0, 1), 1.)
        jump_eh = sample.root_world_z_m - reference.jump_height_world_z_m
        eh = torch.where(jump, jump_eh, support_eh)
        evz = sample.velocity_origin_world_m_s[:, 2] - reference.jump_velocity_world_z_m_s
        pose = sample.roll_pitch_rad - reference.target_roll_pitch_rad
        upright = (-sample.gravity_body[:, 2]).clamp(-1, 1)
        height_gate = torch.where(recovery, upright.clamp_min(0), torch.ones_like(upright))
        # SCUT tracking (per second): g [exp(-clip(e,1)^2/0.5) - 0.1 (0.25 e)^2] for
        # forward, g [exp(-clip(e,0.8)^2/0.25) - 0.1 (0.5 e)^2] for yaw, with g the
        # velocity height gate, and exp(-clip(e_h,0.15)^2/0.001) - (10 e_h)^2 for height.
        locomotion = torch.where(ordinary, 1., torch.where(jump, .8, 0.))
        gx, gy = sample.gravity_body[:, 0], sample.gravity_body[:, 1]
        # Densities per second. Ordinary <= 1+1+1+2 = 5; jump <= 0.8+1+1+1+1;
        # recovery <= 1+4; all below the composer's 8/s bound.
        positive = {'forward': locomotion * velocity_gate * torch.exp(-ev.clamp(-1., 1.).square() / .5),
                    'yaw': (~recovery) * velocity_gate * torch.exp(-ew.clamp(-.8, .8).square() / .25),
                    'height': height_gate * torch.exp(-eh.clamp(-.15, .15).square() / .001),
                    # SCUT flat_orientation_{y,x}_exp on projected gravity.
                    'attitude': ordinary * (torch.exp(-gx.square() / .02) + torch.exp(-gy.square() / .01)),
                    'jump_vertical': jump * torch.exp(-(evz / .35).square()),
                    'jump_attitude': jump * torch.exp(-(pose / .25).square().sum(-1)),
                    'recovery_upright': recovery * (2 * (1 + upright))}
        h = self._huber
        effort, dq = sample.active_explicit_effort_nm, sample.policy_velocity_rad_s
        ddq = (dq - window['dq']) / self.physics_dt_s
        exact_still = ordinary & command[:, :3].eq(0).all(-1)
        still = ordinary & (target_v == 0)
        horizontal = torch.stack((v_forward, v_left), -1)
        # SCUT flat_orientation_{y,x}_v: 2[(2 exp(-v^2/3) + 2) g]^2 = 32 c(v)^2 g^2
        # with c(v) = (exp(-v^2/3) + 1)/2, 1 at rest and 0.54 at 2.7 m/s.
        relief = .5 * (torch.exp(-command[:, 0].square() / 3) + 1)
        fork = (torch.zeros_like(v_forward) if sample.wheel_position_base_m is None
                else sample.wheel_position_base_m[:, 0, 0] - sample.wheel_position_base_m[:, 1, 0])
        wheel_load = sample.body_normal_load_n[:, self._wheels]
        slip = wheel_load > 5
        landing = jump & ((phase == int(RewardPhase.LANDING)) | (phase == int(RewardPhase.SETTLING)))
        weight_n = self._mass_kg * self.tuning.gravity_m_s2
        excess_landing = (wheel_load.sum(-1) / weight_n - self.tuning.landing_load_multiple).clamp_min(0)
        cost = {'forward': (~recovery) * velocity_gate * VELOCITY_SQUARE_WEIGHT * (.25 * ev).square(),
                'yaw': (~recovery) * velocity_gate * VELOCITY_SQUARE_WEIGHT * (.5 * ew).square(),
                'height': height_gate * (10 * eh).square(),
                'attitude': torch.where(recovery, 2 * h(pose / .8).sum(-1),
                                         torch.where(jump, .25 * h(pose / .5).sum(-1),
                                                     32 * relief.square() * (gx.square() + gy.square()))),
                # SCUT ang_vel_xy 0.05; recovery flips are ungated, jump uses its own gain.
                'roll_pitch_angular_velocity': torch.where(
                    recovery, self.tuning.angular_damping_recovery,
                    torch.where(jump, self.tuning.angular_damping_jump,
                                self.tuning.angular_damping_ordinary))
                    * sample.angular_velocity_body_rad_s[:, :2].square().sum(-1),
                # SCUT stand_still_lin_vel: |v_x| + |v_y| inside the forward deadband.
                'stand_world_xy': STAND_STILL_WEIGHT * still * horizontal.abs().sum(-1),
                'ordinary_vertical': ordinary * .5 * v_up.square(),
                'jump_vertical': jump * h(evz / .5),
                # SCUT no_fork + no_fork_square: 1(|dx| > 0.05 m) + (5 dx)^2, dx = x_L - x_R.
                'leg_fork': (~recovery) * ((fork.abs() > FORK_TOLERANCE_M).float() + (5 * fork).square()),
                'effort': 1e-4 * effort.square().sum(-1),
                # SCUT wheel_power: positive wheel mechanical power only, W.
                'mechanical_power_proxy': 1e-4 * (effort[:, 4:] * dq[:, 4:]).clamp_min(0).sum(-1),
                'wheel_velocity': 1e-5 * dq[:, 4:].square().sum(-1),
                'wheel_acceleration': 1e-8 * ddq[:, 4:].square().sum(-1),
                'leg_velocity': 5e-3 * dq[:, :4].square().sum(-1),
                'leg_acceleration': 5e-7 * ddq[:, :4].square().sum(-1),
                'issued_first_difference': window['action_cost'], 'issued_second_difference': window['action2_cost'],
                # SCUT undesired_contact: any non-wheel body above 3 N.
                'harmful_contact': 2 * (sample.body_normal_load_n[:, self._harmful] > 3).any(-1).float(),
                'contact_slip': .5 * (slip * (sample.wheel_contact_tangent_speed_m_s / .2).square()).sum(-1),
                'joint_boundary': 2 * (sample.joint_boundary_shortfall_rad / .05).square().sum(-1),
                'flight_wheel_effort': .2 * (jump & (phase == int(RewardPhase.FLIGHT))) * (effort[:, 4:] / 4.5).square().sum(-1),
                'landing_impact': .5 * landing * h(excess_landing)}
        if reference.intentional_base_contact is not None:
            base = self.binding.native_body_names.index('base_link')
            # Intentional stair base contact is exempt; other bodies still count.
            others = [index for index in self._harmful if index != base]
            cost['harmful_contact'] = torch.where(
                reference.intentional_base_contact,
                2 * (sample.body_normal_load_n[:, others] > 3).any(-1).float(), cost['harmful_contact'])
        if sample.available_terms is not None:
            available = set(sample.available_terms)
            positive = {key: value if key in available else torch.zeros_like(value)
                        for key, value in positive.items()}
            cost = {key: value if key in available else torch.zeros_like(value)
                    for key, value in cost.items()}
        p, c = torch.stack(tuple(positive.values())).sum(0), torch.stack(tuple(cost.values())).sum(0)
        components = {**{'positive.' + key: value for key, value in positive.items()},
                      **{'cost.' + key: -value for key, value in cost.items()}}
        invalid |= (~torch.isfinite(p) | ~torch.isfinite(c) | (p < 0) | (p > 8 + 1e-5) | (c < 0)
                    | ~torch.isfinite(torch.stack(tuple(components.values()))).all(0))
        if invalid.any():
            raise ValueError('reward input, PRE-reset identity, timing, phase or finite density bound differs')
        gates = {'ordinary': ordinary, 'jump': jump, 'recovery': recovery, 'exact_stationary': exact_still,
                 'stationary_deadband': still,
                 'slip_left': slip[:, 0], 'slip_right': slip[:, 1], 'landing': landing,
                 'height_reference_weight': height_gate, 'velocity_height_gate': velocity_gate,
                 'first_action_difference': self._action_count > 0, 'second_action_difference': self._action_count > 1}
        errors = {'forward_m_s': ev, 'yaw_rad_s': ew, 'support_height_m': support_eh,
                  'jump_world_height_m': jump_eh, 'jump_world_vertical_m_s': evz,
                  'roll_rad': pose[:, 0], 'pitch_rad': pose[:, 1],
                  'stationary_world_xy_speed_m_s': horizontal.norm(dim=-1), 'leg_fork_m': fork}
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
        from .skills import SkillEvidence, SkillOutcome

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
                              | (evidence.requested_rise_m < .03) | (evidence.requested_rise_m > .40)
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

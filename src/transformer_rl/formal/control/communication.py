"""Shared round-trip allocation, delayed feedback and interval torque delivery.

The substep model preserves impulse, not an unobserved high-rate trajectory.
USB RTT and the identified motor response are separate evidence sources.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class CommunicationConfig:
    enabled: bool = False
    total_delay_range_s: tuple[float, float] = (.0001, .005)
    uplink_fraction: float = .5

    def __post_init__(self):
        bounds = self.total_delay_range_s
        if (type(self.enabled) is not bool or not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 <= bounds[0] <= bounds[1] <= .005
                or type(self.uplink_fraction) not in (int, float)
                or not math.isfinite(self.uplink_fraction) or not 0 <= self.uplink_fraction <= 1):
            raise ValueError('communication needs a bounded 200Hz round-trip prior and one allocation')
        object.__setattr__(self, 'total_delay_range_s', tuple(bounds))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(asdict(cls())):
            raise ValueError('communication requires its complete frozen configuration')
        return cls(**value)


_LEGACY_EPISODE_FIELDS = frozenset({'enabled', 'full_at_update', 'spring_force_scale'})


def _scale_bounds(value, *, maximum=None):
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and 0 < value[0] <= value[1] and (maximum is None or value[1] <= maximum))


@dataclass(frozen=True)
class EpisodeRandomizationConfig:
    """Reset-local actuator, encoder, spring and push randomization.

    Startup mass and materials stay fixed. Each draw is blended toward its
    nominal value by the shared exposure scale. Strength is a torque-gain
    derating (<= 1) of command and cap; the encoder bias b is added to the
    measured leg angles used by the actor and the PD loop (rad), so the
    physical pose settles at target - b, as a mis-zeroed encoder would; the
    no-load scale multiplies the motor back-EMF speed (battery voltage); the
    time-constant scale multiplies the identified leg drive lag; pushes add
    a world-frame horizontal velocity and yaw-rate step every interval.
    Defaults disable every term added after the spring scale.
    """

    enabled: bool = False
    full_at_update: int = 5000
    spring_force_scale: tuple[float, float] = (.9, 1.1)
    leg_strength: tuple[float, float] = (1., 1.)
    wheel_strength: tuple[float, float] = (1., 1.)
    leg_encoder_bias_rad: float = 0.
    no_load_speed_scale: tuple[float, float] = (1., 1.)
    leg_time_constant_scale: tuple[float, float] = (1., 1.)
    push_interval_s: tuple[float, float] = (5., 10.)
    push_linear_m_s: float = 0.
    push_yaw_rad_s: float = 0.
    # SCUT base_external_force_torque_xyz: per-axis base-frame force (N) and
    # torque (N·m), redrawn every interval and held between draws.
    wrench_interval_s: tuple[float, float] = (5., 10.)
    external_force_n: float = 0.
    external_torque_nm: float = 0.

    def __post_init__(self):
        bounds = self.spring_force_scale
        if (type(self.enabled) is not bool or type(self.full_at_update) is not int
                or self.full_at_update < 1 or not isinstance(bounds, (tuple, list))
                or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 < bounds[0] <= 1 <= bounds[1]):
            raise ValueError('episode randomization needs an exposure horizon and positive force bounds')
        if (not _scale_bounds(self.leg_strength, maximum=1) or not _scale_bounds(self.wheel_strength, maximum=1)
                or not _scale_bounds(self.no_load_speed_scale) or not _scale_bounds(self.leg_time_constant_scale)
                or not _scale_bounds(self.push_interval_s) or not _scale_bounds(self.wrench_interval_s)
                or any(type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name))
                       or getattr(self, name) < 0
                       for name in ('leg_encoder_bias_rad', 'push_linear_m_s', 'push_yaw_rad_s',
                                    'external_force_n', 'external_torque_nm'))
                or self.leg_encoder_bias_rad > .1):
            raise ValueError('actuator/encoder/push randomization bounds are invalid')
        for name in ('spring_force_scale', 'leg_strength', 'wheel_strength', 'no_load_speed_scale',
                     'leg_time_constant_scale', 'push_interval_s', 'wrench_interval_s'):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))

    @classmethod
    def from_dict(cls, value):
        # Frozen pre-actuator runtimes contain only the legacy fields.
        if (not isinstance(value, dict) or not _LEGACY_EPISODE_FIELDS <= set(value) <= set(asdict(cls()))):
            raise ValueError('episode randomization requires its complete frozen configuration')
        return cls(**value)

    def scale_at(self, update):
        if type(update) is not int or update < 0:
            raise ValueError('randomization exposure requires a nonnegative update count')
        if not self.enabled:
            return 1.
        value = min(update / self.full_at_update, 1.)
        return value * value * (3 - 2 * value)


class PhysicalCommunication:
    """One delay draw per reset; feedback and downlink borrow the same budget."""

    def __init__(self, num_envs, config, *, seed, device='cpu', physics_dt_s=.005):
        if (type(num_envs) is not int or num_envs < 1 or not isinstance(config, CommunicationConfig)
                or not config.enabled or physics_dt_s != .005
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError('physical communication requires an explicit enabled 200Hz configuration')
        self.config, self.num_envs, self.dt = config, num_envs, physics_dt_s
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.total_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.uplink_s, self.downlink_s = self.total_s.clone(), self.total_s.clone()
        self.current = torch.zeros(num_envs, 12, device=self.device)
        self.previous = self.current.clone()
        self.has_previous = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.held_effort = torch.zeros(num_envs, 6, device=self.device)
        self.held_origins = torch.zeros(num_envs, 4, dtype=torch.float64, device=self.device)
        self.applied_origins = self.held_origins.clone()
        self.applied_at_s = self.total_s.clone()
        self.policy_allowed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.reset_rows_consumed = 0

    @torch.no_grad()
    def reset(self, mask, q, dq, now_s, *, difficulty=1.):
        rows = mask.nonzero().flatten()
        if not len(rows):
            return
        if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise ValueError('communication difficulty must be in [0,1]')
        unit = torch.rand(len(rows), device=self.device, dtype=torch.float64, generator=self.generator)
        low, high = self.config.total_delay_range_s
        total = (low + (high - low) * unit) * difficulty
        self.total_s[rows] = total
        self.uplink_s[rows] = total * self.config.uplink_fraction
        self.downlink_s[rows] = total - self.uplink_s[rows]
        self.current[rows] = torch.cat((q[rows], dq[rows]), -1)
        self.previous[rows] = self.current[rows]
        self.has_previous[rows] = False
        self.flush_commands(mask, now_s)
        self.policy_allowed[rows] = False
        self.reset_rows_consumed += len(rows)

    @torch.no_grad()
    def push(self, q, dq):
        # q comes from the unwrapped joint coordinate reader, so interpolation
        # across a 2pi cut cannot create a false half-turn in PD feedback.
        self.previous.copy_(self.current)
        self.current.copy_(torch.cat((q, dq), -1))
        self.has_previous.fill_(True)

    def feedback(self, now_s):
        age = self.uplink_s * self.has_previous
        fraction = (age / self.dt).float()[:, None]
        value = self.current + fraction * (self.previous - self.current)
        return value[:, :6], value[:, 6:], now_s - age

    @torch.no_grad()
    def flush_commands(self, mask, now_s):
        self.held_effort[mask] = 0
        self.held_origins[mask] = now_s[mask, None]
        self.applied_origins[mask] = now_s[mask, None]
        self.applied_at_s[mask] = now_s[mask]

    @torch.no_grad()
    def begin_interval(self, policy_allowed, now_s):
        self.flush_commands(policy_allowed != self.policy_allowed, now_s)
        self.policy_allowed.copy_(policy_allowed)

    @torch.no_grad()
    def deliver(self, command, bounds, now_s, policy_generated_s, feedback_sample_s, *, response=None,
                target_arrived_at_s=None):
        """Integrate old torque for D, then new torque for dt-D exactly once."""
        old_duration = self.downlink_s
        new_duration = self.dt - old_duration
        if response is None:
            mean = (self.held_effort * old_duration[:, None]
                    + command * new_duration[:, None]) / self.dt
        else:
            old = response.advance(self.held_effort, old_duration, bounds)
            new = response.advance(command, new_duration, bounds)
            mean = (old.impulse + new.impulse) / self.dt
        target_arrived_at_s = now_s if target_arrived_at_s is None else target_arrived_at_s
        origins = torch.stack((now_s, policy_generated_s, feedback_sample_s, target_arrived_at_s), -1)
        fraction = (old_duration / self.dt)[:, None]
        self.applied_origins.copy_(origins + fraction * (self.held_origins - origins))
        self.held_origins.copy_(origins)
        self.held_effort.copy_(command)
        self.applied_at_s.copy_(now_s + self.dt)
        return mean.float()

    def action_age_steps(self, physics_dt_s):
        age = (self.applied_at_s - self.applied_origins[:, 1]).clamp_min(0) / physics_dt_s
        return age.float()[:, None].expand(-1, 2)

    def metrics(self):
        # Ages describe duration-weighted request provenance, before motor lag.
        # They do not assert a physical-response timestamp for a nonlinear body.
        return {'round_trip_s': self.total_s, 'uplink_s': self.uplink_s, 'downlink_s': self.downlink_s,
                'applied_pd_age_s': self.applied_at_s - self.applied_origins[:, 0],
                'applied_policy_age_s': self.applied_at_s - self.applied_origins[:, 1],
                'applied_feedback_age_s': self.applied_at_s - self.applied_origins[:, 2],
                'applied_target_latch_age_s': self.applied_at_s - self.applied_origins[:, 3]}

    def state_dict(self):
        return {'configuration': asdict(self.config), 'rng': self.generator.get_state().clone(),
                'reset_rows_consumed': self.reset_rows_consumed}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict) or set(state) != {'configuration', 'rng', 'reset_rows_consumed'}
                or CommunicationConfig(**state['configuration']) != self.config
                or type(state['reset_rows_consumed']) is not int or state['reset_rows_consumed'] < 0):
            raise ValueError('communication checkpoint contract differs')
        self.generator.set_state(state['rng'].cpu())
        self.reset_rows_consumed = state['reset_rows_consumed']
        self.current.zero_()
        self.previous.zero_()
        self.has_previous.zero_()
        self.held_effort.zero_()
        self.held_origins.zero_()
        self.applied_origins.zero_()
        self.applied_at_s.zero_()
        self.policy_allowed.zero_()


class EpisodeRandomization:
    """Reset-local randomization shares exposure, but owns its independent RNG."""

    def __init__(self, num_envs, config, *, seed, device='cpu'):
        self.config, self.device = config, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.strength = torch.ones(num_envs, 1, device=self.device)
        self.scale = torch.ones(num_envs, device=self.device)
        self.motor_strength = torch.ones(num_envs, 6, device=self.device)
        self.encoder_bias_rad = torch.zeros(num_envs, 6, device=self.device)
        self.no_load_scale = torch.ones(num_envs, 6, device=self.device)
        self.time_constant_scale = torch.ones(num_envs, 6, device=self.device)
        self.push_timer_s = torch.full((num_envs,), float('inf'), device=self.device)
        self.wrench_timer_s = torch.full((num_envs,), float('inf'), device=self.device)
        self.base_wrench = torch.zeros(num_envs, 6, device=self.device)
        self.reset_rows_consumed = 0

    def _blend(self, bounds, count, width, scale):
        low, high = bounds
        unit = torch.rand(count, width, device=self.device, generator=self.generator)
        return 1 + scale * (low + (high - low) * unit - 1)

    def _interval(self, count, bounds=None):
        low, high = self.config.push_interval_s if bounds is None else bounds
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def _wrench(self, rows):
        unit = torch.rand(len(rows), 6, device=self.device, generator=self.generator) * 2 - 1
        limit = unit.new_tensor([self.config.external_force_n] * 3 + [self.config.external_torque_nm] * 3)
        self.base_wrench[rows] = unit * limit * self.scale[rows, None]
        self.wrench_timer_s[rows] = self._interval(len(rows), self.config.wrench_interval_s)

    def reset(self, mask, update):
        rows = mask.nonzero().flatten()
        scale = self.config.scale_at(update)
        unit = torch.rand(len(rows), 1, device=self.device, generator=self.generator)
        low, high = self.config.spring_force_scale
        self.strength[rows] = 1 + scale * (low + (high - low) * unit - 1)
        self.scale[rows] = scale
        c, count = self.config, len(rows)
        # Spring draws come first so legacy configurations keep their stream.
        self.motor_strength[rows] = torch.cat((self._blend(c.leg_strength, count, 1, scale).expand(-1, 4),
                                               self._blend(c.wheel_strength, count, 1, scale).expand(-1, 2)), -1)
        bias = (torch.rand(count, 4, device=self.device, generator=self.generator) * 2 - 1) * c.leg_encoder_bias_rad
        self.encoder_bias_rad[rows] = torch.cat((scale * bias, torch.zeros(count, 2, device=self.device)), -1)
        self.no_load_scale[rows] = self._blend(c.no_load_speed_scale, count, 1, scale).expand(-1, 6)
        self.time_constant_scale[rows] = torch.cat((self._blend(c.leg_time_constant_scale, count, 1, scale)
                                                    .expand(-1, 4), torch.ones(count, 2, device=self.device)), -1)
        self.push_timer_s[rows] = self._interval(count) if (c.push_linear_m_s or c.push_yaw_rad_s) else float('inf')
        self.base_wrench[rows] = 0
        self.wrench_timer_s[rows] = (self._interval(count, c.wrench_interval_s)
                                     if (c.external_force_n or c.external_torque_nm) else float('inf'))
        self.reset_rows_consumed += len(rows)
        return scale

    def advance_wrench(self, dt_s):
        """Redraw due base wrenches; returns the held [N,6] base-frame force/torque."""
        self.wrench_timer_s.sub_(dt_s)
        rows = (self.wrench_timer_s <= 0).nonzero().flatten()
        if rows.numel():
            self._wrench(rows)
        return self.base_wrench

    def push(self, dt_s, eligible):
        """Rows due for a push and their world [vx, vy, wz] velocity steps."""
        self.push_timer_s.sub_(dt_s)
        due = (self.push_timer_s <= 0) & eligible
        rows = due.nonzero().flatten()
        delta = (torch.rand(len(rows), 3, device=self.device, generator=self.generator) * 2 - 1)
        delta *= delta.new_tensor((self.config.push_linear_m_s, self.config.push_linear_m_s,
                                   self.config.push_yaw_rad_s)) * self.scale[rows, None]
        self.push_timer_s[rows] = self._interval(len(rows))
        return rows, delta

    def state_dict(self):
        return {'configuration': asdict(self.config), 'rng': self.generator.get_state().clone(),
                'reset_rows_consumed': self.reset_rows_consumed}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict) or set(state) != {'configuration', 'rng', 'reset_rows_consumed'}
                or EpisodeRandomizationConfig(**state['configuration']) != self.config
                or type(state['reset_rows_consumed']) is not int or state['reset_rows_consumed'] < 0):
            raise ValueError('episode randomization checkpoint contract differs')
        self.generator.set_state(state['rng'].cpu())
        self.reset_rows_consumed = state['reset_rows_consumed']
        self.strength.fill_(1)
        self.scale.zero_()
        self.motor_strength.fill_(1)
        self.encoder_bias_rad.zero_()
        self.no_load_scale.fill_(1)
        self.time_constant_scale.fill_(1)
        self.push_timer_s.fill_(float('inf'))
        self.wrench_timer_s.fill_(float('inf'))
        self.base_wrench.zero_()

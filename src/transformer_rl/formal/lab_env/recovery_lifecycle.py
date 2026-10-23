"""Recovery-to-tracking protocol with separate deployable and physical evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ..tasks.protocol import CompletedEstimate


@dataclass(frozen=True)
class RecoveryLifecycleConfig:
    policy_dt_s: float = .01
    physics_dt_s: float = .005
    estimate_max_age_s: float = .03
    stable_s: float = .5
    timeout_s: float = 5.
    tilt_rad: float = math.radians(10.)
    angular_speed_rad_s: float = .5
    vertical_speed_m_s: float = .15
    contact_probability: float = .8
    wheel_load_n: float = 5.
    base_load_n: float = 2.
    height_tolerance_m: float = .05

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in vars(self).values())
                or self.policy_dt_s not in (.01, .02) or self.physics_dt_s != .005
                or not self.physics_dt_s <= self.policy_dt_s <= self.estimate_max_age_s
                or self.stable_s >= self.timeout_s or not 0 < self.contact_probability < 1
                or self.tilt_rad >= math.pi / 2):
            raise ValueError('recovery lifecycle requires explicit 50/100Hz policy and 200Hz physics timing')


@dataclass(frozen=True)
class RecoveryBoundary:
    handoff: torch.Tensor
    timeout: torch.Tensor


class RecoveryLifecycle:
    """One policy owns recovery and subsequent tracking; no action scripts.

    Only a fresh prior-tick estimate selects the next actor-visible mode.
    Terrain-relative height and true contact loads certify diagnostic success
    separately; they never select commands or manufacture a reward event.
    """

    COUNTERS = ('births', 'estimated_handoffs', 'verified_physical_successes',
                'handoffs_without_verified_success', 'timeouts', 'failed_before_handoff',
                'resets_before_completion')

    def __init__(self, num_envs, *, device='cpu', config=RecoveryLifecycleConfig()):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, RecoveryLifecycleConfig):
            raise ValueError('recovery lifecycle requires parsed configuration and positive row count')
        self.num_envs, self.device, self.config = num_envs, torch.device(device), config
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.policy_elapsed_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.estimated_stable_s = torch.zeros_like(self.policy_elapsed_s)
        self.physical_stable_s = torch.zeros_like(self.policy_elapsed_s)
        self._last_estimate_s = torch.full_like(self.policy_elapsed_s, -float('inf'))
        self._verified = torch.zeros_like(self.active)
        self._counts = torch.zeros(len(self.COUNTERS), dtype=torch.long, device=self.device)

    def _count(self, name, rows):
        self._counts[self.COUNTERS.index(name)].add_(rows.sum())

    @torch.no_grad()
    def reset(self, ids, recovery):
        if (ids.ndim != 1 or ids.dtype != torch.long or ids.device != self.device
                or recovery.shape != ids.shape or recovery.dtype != torch.bool
                or recovery.device != self.device):
            raise ValueError('recovery reset requires bound row IDs and their sampled recovery mask')
        self._count('resets_before_completion', self.active[ids])
        self._count('births', recovery)
        self.active[ids] = recovery
        self.policy_elapsed_s[ids] = 0
        self.estimated_stable_s[ids] = 0
        self.physical_stable_s[ids] = 0
        self._last_estimate_s[ids] = -float('inf')
        self._verified[ids] = False

    @torch.no_grad()
    def observe_physics(self, *, height_m, command_height_m, tilt_rad,
                        angular_speed_rad_s, velocity_z_m_s, wheel_load_n,
                        base_load_n, policy_owned):
        n, c = self.num_envs, self.config
        for value, shape in ((height_m, (n,)), (command_height_m, (n,)),
                             (tilt_rad, (n,)), (angular_speed_rad_s, (n,)),
                             (velocity_z_m_s, (n,)), (wheel_load_n, (n, 2)),
                             (base_load_n, (n,))):
            if value.shape != shape or value.device != self.device or not value.is_floating_point():
                raise ValueError('recovery physical evidence differs from its bound rows')
        if (policy_owned.shape != (n,) or policy_owned.dtype != torch.bool
                or policy_owned.device != self.device):
            raise ValueError('recovery evidence requires actual policy ownership')
        good = (self.active & policy_owned
                & torch.isfinite(height_m) & torch.isfinite(command_height_m)
                & ((height_m - command_height_m).abs() <= c.height_tolerance_m)
                & (tilt_rad >= 0) & (tilt_rad <= c.tilt_rad)
                & (angular_speed_rad_s >= 0) & (angular_speed_rad_s <= c.angular_speed_rad_s)
                & (velocity_z_m_s.abs() <= c.vertical_speed_m_s)
                & torch.isfinite(wheel_load_n).all(-1) & (wheel_load_n >= c.wheel_load_n).all(-1)
                & (base_load_n >= 0) & (base_load_n <= c.base_load_n))
        self.physical_stable_s.copy_(torch.where(good, self.physical_stable_s + c.physics_dt_s, 0))

    @torch.no_grad()
    def finish_interval(self, now_s, estimate, *, policy_owned, failed):
        n, c = self.num_envs, self.config
        if (now_s.shape != (n,) or now_s.dtype != torch.float64 or now_s.device != self.device
                or any(v.shape != (n,) or v.dtype != torch.bool or v.device != self.device
                       for v in (policy_owned, failed))):
            raise ValueError('recovery boundary requires its clock, ownership and latched failures')
        eligible = self.active & policy_owned & ~failed
        self.policy_elapsed_s.add_(eligible.double() * c.policy_dt_s)
        ready = torch.zeros_like(eligible)
        if estimate is not None:
            if not isinstance(estimate, CompletedEstimate):
                raise TypeError('recovery mode accepts only completed deployable estimates')
            for name, shape, dtype in (
                    ('available_at_s', (n,), torch.float64), ('valid', (n,), torch.bool),
                    ('wheel_contact_probability', (n, 2), torch.float32),
                    ('tilt_rad', (n,), torch.float32),
                    ('angular_speed_rad_s', (n,), torch.float32),
                    ('velocity_z_m_s', (n,), torch.float32)):
                value = getattr(estimate, name)
                if value.shape != shape or value.dtype != dtype or value.device != self.device:
                    raise ValueError('completed recovery estimate differs from the bound rows')
            age = now_s - estimate.available_at_s
            probability = estimate.wheel_contact_probability
            clock_valid = (torch.isfinite(age) & (age > 0)
                           & (age <= c.estimate_max_age_s + 1e-9))
            ready = (estimate.valid & clock_valid
                     & (estimate.available_at_s > self._last_estimate_s)
                     & torch.isfinite(probability).all(-1)
                     & (probability >= c.contact_probability).all(-1) & (probability <= 1).all(-1)
                     & (estimate.tilt_rad >= 0) & (estimate.tilt_rad <= c.tilt_rad)
                     & (estimate.angular_speed_rad_s >= 0)
                     & (estimate.angular_speed_rad_s <= c.angular_speed_rad_s)
                     & (estimate.velocity_z_m_s.abs() <= c.vertical_speed_m_s))
            sampled_at = torch.where(clock_valid,
                                     estimate.available_at_s, self._last_estimate_s)
            self._last_estimate_s.copy_(torch.maximum(self._last_estimate_s, sampled_at))
        self.estimated_stable_s.copy_(torch.where(
            eligible & ready, self.estimated_stable_s + c.policy_dt_s, 0))
        handoff = eligible & (self.estimated_stable_s >= c.stable_s - 1e-9)
        timeout = eligible & ~handoff & (self.policy_elapsed_s >= c.timeout_s - 1e-9)
        verified = (eligible & (self.physical_stable_s >= c.stable_s - 1e-9)
                    & ~self._verified)
        self._count('verified_physical_successes', verified)
        self._verified |= verified
        self._count('estimated_handoffs', handoff)
        self._count('handoffs_without_verified_success', handoff & ~self._verified)
        self._count('timeouts', timeout)
        self._count('failed_before_handoff', self.active & failed)
        self.active &= ~(handoff | timeout | failed)
        return RecoveryBoundary(handoff, timeout)

    def metrics(self):
        return {'configuration': asdict(self.config),
                'counts': dict(zip(self.COUNTERS, self._counts.tolist())),
                'scope': 'recovery births; prior-estimate handoff; separate physical stability evidence',
                'success_reward_added': False}

    def state_dict(self):
        return {'format': 'lab_recovery_lifecycle_v1', 'configuration': asdict(self.config),
                'counts': self._counts.clone()}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict) or set(state) != {'format', 'configuration', 'counts'}
                or state['format'] != 'lab_recovery_lifecycle_v1'
                or state['configuration'] != asdict(self.config)):
            raise ValueError('recovery checkpoint contract differs')
        counts = state['counts']
        if (not isinstance(counts, torch.Tensor) or counts.shape != self._counts.shape
                or counts.dtype != torch.long or (counts < 0).any()):
            raise ValueError('recovery checkpoint counters differ')
        self._counts.copy_(counts)
        # A cold restore keeps counters, never an old physical stability clock.
        self.policy_elapsed_s.zero_()
        self.estimated_stable_s.zero_()
        self.physical_stable_s.zero_()
        self._last_estimate_s.fill_(-float('inf'))
        self._verified.zero_()

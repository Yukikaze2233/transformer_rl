"""Tensor-only physical failure limits shared by native and Lab collectors.

Every caller supplies one sample per physical substep. Contact exceptions only
suppress that contact event; they cannot clear a previously latched failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math

import torch


class FailureReason(IntFlag):
    LOW_SUPPORT_HEIGHT = 1
    EXCESSIVE_TILT = 2
    FORBIDDEN_BODY_CONTACT = 4
    PASSIVE_KNEE_OVERRUN = 8
    KNEE_PIVOT_RESIDUAL = 16
    PRESSURE_STROKE = 32


@dataclass(frozen=True)
class NativeTaskLimits:
    """Explicit recipe inputs; there are no guessed hardware-safe defaults."""

    minimum_normal_height_m: float
    minimum_recovery_height_m: float
    healthy_tilt_rad: float
    failure_tilt_rad: float
    forbidden_body_names: tuple[str, ...]
    forbidden_body_load_n: float
    forbidden_contact_confirm_s: float
    passive_knee_overrun_rad: float
    knee_pivot_residual_m: float
    preparation_timeout_s: float
    # SCUT: tilt and low-support failures must persist (0.4 s) before ending
    # an episode, so a landing or stand-up may touch down. 0 keeps instant.
    pose_failure_confirm_s: float = 0.

    def __post_init__(self):
        positive = ('healthy_tilt_rad', 'failure_tilt_rad', 'forbidden_body_load_n',
                    'forbidden_contact_confirm_s', 'passive_knee_overrun_rad', 'knee_pivot_residual_m',
                    'preparation_timeout_s')
        if (any(type(getattr(self, k)) not in (int, float) or not math.isfinite(getattr(self, k))
                or getattr(self, k) <= 0 for k in positive)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (self.minimum_normal_height_m, self.minimum_recovery_height_m))
                or not self.minimum_recovery_height_m <= self.minimum_normal_height_m
                or not self.healthy_tilt_rad < self.failure_tilt_rad <= math.pi
                or self.preparation_timeout_s < .05
                or type(self.pose_failure_confirm_s) not in (int, float)
                or not math.isfinite(self.pose_failure_confirm_s) or self.pose_failure_confirm_s < 0
                or not isinstance(self.forbidden_body_names, tuple) or not self.forbidden_body_names
                or len(set(self.forbidden_body_names)) != len(self.forbidden_body_names)
                or any(not isinstance(v, str) or not v for v in self.forbidden_body_names)):
            raise ValueError('native task limits require explicit finite,ordered physical/contact/preparation bounds')


@dataclass(frozen=True)
class PhysicalConditions:
    healthy: torch.Tensor
    true_failure: torch.Tensor
    failure_reasons: torch.Tensor


class PhysicalFailureLatch:
    """Latch explicit mechanical/pose failures until an actual row reset."""

    def __init__(self, limits, num_envs, *, physics_dt_s, device='cpu'):
        if (not isinstance(limits, NativeTaskLimits) or type(num_envs) is not int or num_envs < 1
                or type(physics_dt_s) not in (int, float) or not math.isfinite(physics_dt_s)
                or physics_dt_s <= 0):
            raise ValueError('physical conditions require explicit limits, rows and a substep clock')
        self.limits, self.num_envs = limits, num_envs
        self.device, self.dt = torch.device(device), float(physics_dt_s)
        self.contact_time_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.pose_time_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.failure_reasons = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Episodes ended per reason (all bits set at the first failing substep).
        self.reason_counts = torch.zeros(len(FailureReason), dtype=torch.long, device=self.device)
        self._reason_bits = torch.tensor([int(reason) for reason in FailureReason], device=self.device)

    def reset(self, mask):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError('physical condition reset needs its actual row mask')
        self.contact_time_s[mask] = 0
        self.pose_time_s[mask] = 0
        self.failure_reasons[mask] = 0

    @torch.no_grad()
    def observe(self, *, height_m, gravity_body, forbidden_body_load_n,
                knee_signed_clearance_rad, knee_pivot_residual_m,
                within_pressure_stroke, recovery, body_contact_allowed=None):
        n = self.num_envs
        shapes = ((height_m, (n,), True), (gravity_body, (n, 3), True),
                  (forbidden_body_load_n, (n, len(self.limits.forbidden_body_names)), True),
                  (knee_signed_clearance_rad, (n, 2), True),
                  (knee_pivot_residual_m, (n, 2), True),
                  (within_pressure_stroke, (n, 2), False), (recovery, (n,), False))
        # Native producers already validate numeric samples. Shape/dtype checks
        # do not add per-substep CUDA fences to the collector's hot path.
        for value, shape, floating in shapes:
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device
                    or (not value.is_floating_point() if floating else value.dtype != torch.bool)):
                raise ValueError('physical condition tensor layout differs from the bound producer')
        if body_contact_allowed is None:
            body_contact_allowed = torch.zeros_like(recovery)
        if (body_contact_allowed.shape != (n,) or body_contact_allowed.dtype != torch.bool
                or body_contact_allowed.device != self.device):
            raise ValueError('contact exception requires an explicit per-row boolean')
        limits = self.limits
        tilt = torch.atan2(gravity_body[:, :2].norm(dim=-1), -gravity_body[:, 2])
        contacts = forbidden_body_load_n.gt(limits.forbidden_body_load_n)
        if 'base_link' in limits.forbidden_body_names:
            contacts[:, limits.forbidden_body_names.index('base_link')] &= ~body_contact_allowed
        contact = contacts.any(-1) & ~recovery
        self.contact_time_s.copy_(torch.where(contact, self.contact_time_s + self.dt, 0))
        minimum = torch.where(recovery, limits.minimum_recovery_height_m, limits.minimum_normal_height_m)
        low, tilted = height_m < minimum, (tilt > limits.failure_tilt_rad) & ~recovery
        self.pose_time_s.copy_(torch.where(low | tilted, self.pose_time_s + self.dt, 0))
        confirmed = self.pose_time_s >= limits.pose_failure_confirm_s - 1e-9
        events = (
            (low & confirmed, FailureReason.LOW_SUPPORT_HEIGHT),
            (tilted & confirmed, FailureReason.EXCESSIVE_TILT),
            (self.contact_time_s >= limits.forbidden_contact_confirm_s - 1e-9,
             FailureReason.FORBIDDEN_BODY_CONTACT),
            (knee_signed_clearance_rad.lt(-limits.passive_knee_overrun_rad).any(-1),
             FailureReason.PASSIVE_KNEE_OVERRUN),
            (knee_pivot_residual_m.gt(limits.knee_pivot_residual_m).any(-1),
             FailureReason.KNEE_PIVOT_RESIDUAL),
            (~within_pressure_stroke.all(-1), FailureReason.PRESSURE_STROKE),
        )
        before = self.failure_reasons != 0
        for event, reason in events:
            self.failure_reasons.bitwise_or_(event.long() * int(reason))
        failed = self.failure_reasons != 0
        first = (failed & ~before)[:, None]
        self.reason_counts += (first & (self.failure_reasons[:, None] & self._reason_bits).ne(0)).sum(0)
        healthy = (~failed & ~contact & (tilt <= limits.healthy_tilt_rad)
                   & knee_signed_clearance_rad.ge(0).all(-1))
        return PhysicalConditions(healthy, failed.clone(), self.failure_reasons.clone())

    def metrics(self):
        return {'episodes_ended_by_reason': dict(zip((reason.name.lower() for reason in FailureReason),
                                                     self.reason_counts.tolist())),
                'scope': 'all reasons set at the first failing substep; one episode may count several'}

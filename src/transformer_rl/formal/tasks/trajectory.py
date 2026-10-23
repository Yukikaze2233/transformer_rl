"""Private requested/admitted command trajectories in body-frame SI units.

The wheel-speed diamond is a necessary target bound only. It does not qualify
height, torque, traction, flight or hardware capability for curriculum promotion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math

import torch


@dataclass(frozen=True)
class DriveEnvelope:
    wheel_radius_m: float
    track_width_m: float
    target_wheel_speed_rad_s: float = 90.0

    def __post_init__(self):
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in (self.wheel_radius_m, self.track_width_m, self.target_wheel_speed_rad_s)):
            raise ValueError('wheel target envelope requires positive measured geometry and speed bound')

    @staticmethod
    def validate_commands(commands):
        if (not isinstance(commands, torch.Tensor) or commands.ndim != 2 or commands.shape[1] != 4
                or commands.shape[0] < 1 or not commands.is_floating_point()
                or not torch.isfinite(commands).all()):
            raise ValueError('commands require finite [N,4] body vx/vy/yaw-rate/height in SI units')

    def wheel_speeds(self, command):
        # Signed rolling-equivalent speeds, not motor-API encoder coordinates.
        forward, yaw = command[:, 0], command[:, 2]
        return torch.stack((forward - yaw * self.track_width_m / 2,
                            forward + yaw * self.track_width_m / 2), -1) / self.wheel_radius_m

    def project(self, requested):
        self.validate_commands(requested)
        wheels = self.wheel_speeds(requested)
        peak = wheels.abs().amax(-1)
        scale = (self.target_wheel_speed_rad_s / peak.clamp_min(1e-20)).clamp_max(1)
        target = requested.clone()
        target[:, 0] *= scale
        target[:, 1] = 0
        target[:, 2] *= scale
        unsupported_lateral = requested[:, 1].abs() > 1e-8
        necessary = (peak <= self.target_wheel_speed_rad_s + 1e-4) & ~unsupported_lateral
        return target, wheels, scale, necessary, unsupported_lateral


@dataclass(frozen=True)
class CommandRates:
    forward_acceleration_m_s2: float
    yaw_acceleration_rad_s2: float
    height_rate_m_s: float

    def __post_init__(self):
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in asdict(self).values()):
            raise ValueError('command rate bounds must be positive and explicitly supplied')


@dataclass(frozen=True)
class CommandAdmission:
    requested: torch.Tensor
    wheel_projected: torch.Tensor
    admitted: torch.Tensor
    requested_wheel_speeds: torch.Tensor
    wheel_necessary_at_request: torch.Tensor
    unsupported_lateral: torch.Tensor
    wheel_projection_fraction: torch.Tensor
    slew_fraction: torch.Tensor


class CommandStream:
    """Common interpolation fraction preserves the convex wheel target envelope.

    Independent component slew limits can leave the diamond even when both
    endpoints are inside. A common interpolation fraction enforces each rate
    bound and keeps the whole path inside. Geometry/rates require their own
    asset/actuator evidence and are not inferred from an old robot version.
    """

    def __init__(self, initial, envelope: DriveEnvelope, rates: CommandRates, *, policy_dt_s=0.01):
        if (not isinstance(envelope, DriveEnvelope) or not isinstance(rates, CommandRates)
                or policy_dt_s not in (0.01, 0.02)):
            raise ValueError('command stream requires parsed bounds and a 50/100Hz period')
        _, _, _, necessary, _ = envelope.project(initial)
        if not necessary.all():
            raise ValueError('initial admitted command must satisfy the wheel target bound and body vy=0')
        self.envelope, self.rates, self.policy_dt_s = envelope, rates, policy_dt_s
        self.current = initial.detach().clone()
        self._indices = torch.tensor((0, 2, 3), dtype=torch.long, device=initial.device)
        self._rates = initial.new_tensor((rates.forward_acceleration_m_s2, rates.yaw_acceleration_rad_s2,
                                         rates.height_rate_m_s))

    @torch.no_grad()
    def step(self, requested):
        if (not isinstance(requested, torch.Tensor) or requested.shape != self.current.shape
                or requested.dtype != self.current.dtype or requested.device != self.current.device):
            raise ValueError('command stream input shape/dtype/device differs from its state')
        target, wheels, scale, necessary, lateral = self.envelope.project(requested)
        delta = target - self.current
        ratios = self._rates * self.policy_dt_s / delta.index_select(1, self._indices).abs().clamp_min(1e-20)
        fraction = ratios.amin(-1).clamp_max(1)
        self.current.add_(delta * fraction[:, None])
        return CommandAdmission(requested.detach().clone(), target, self.current.clone(), wheels, necessary,
                                 lateral, scale, fraction)

    def state_dict(self):
        return {'envelope': asdict(self.envelope), 'rates': asdict(self.rates),
                'policy_dt_s': self.policy_dt_s, 'current': self.current.detach().clone()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[name] != expected[name] for name in ('envelope', 'rates', 'policy_dt_s'))):
            raise ValueError('command stream checkpoint bounds differ')
        command = state['current']
        if (not isinstance(command, torch.Tensor) or command.shape != self.current.shape
                or command.dtype != self.current.dtype):
            raise ValueError('command stream checkpoint state differs')
        command = command.detach().to(self.current.device).clone()
        if not self.envelope.project(command)[3].all():
            raise ValueError('saved admitted command exceeds its declared wheel target bound')
        self.current.copy_(command)


class TrajectoryKind(IntEnum):
    HOLD = 0
    RAMP = 1
    SINE = 2
    CHIRP = 3
    SMOOTHSTEP = 4
    START_STOP = 5
    BOUNDED_STEP = 6


class CommandTrajectory:
    """Seven raw-request families; rate-limited delivery belongs to CommandStream."""

    def __init__(self, start, end, duration_s, kind, frequency_start_hz=None, frequency_end_hz=None):
        DriveEnvelope.validate_commands(start)
        DriveEnvelope.validate_commands(end)
        count = start.shape[0]
        if (end.shape != start.shape or end.dtype != start.dtype or end.device != start.device
                or not isinstance(duration_s, torch.Tensor) or duration_s.shape != (count,)
                or duration_s.device != start.device or duration_s.dtype != start.dtype
                or not torch.isfinite(duration_s).all() or (duration_s <= 0).any()
                or not isinstance(kind, torch.Tensor) or kind.shape != (count,) or kind.dtype != torch.long
                or kind.device != start.device or ((kind < 0) | (kind > 6)).any()):
            raise ValueError('trajectory endpoints, duration and kind must share a complete row contract')
        first = 1 / duration_s if frequency_start_hz is None else frequency_start_hz
        last = 2 / duration_s if frequency_end_hz is None else frequency_end_hz
        for frequency in (first, last):
            if (not isinstance(frequency, torch.Tensor) or frequency.shape != (count,)
                    or frequency.device != start.device or frequency.dtype != start.dtype
                    or not torch.isfinite(frequency).all() or (frequency <= 0).any()):
                raise ValueError('trajectory frequencies require finite positive per-row Hz')
        self.start, self.end = start.detach().clone(), end.detach().clone()
        self.duration_s, self.kind = duration_s.detach().clone(), kind.detach().clone()
        self.frequency_start_hz, self.frequency_end_hz = first.detach().clone(), last.detach().clone()

    @torch.no_grad()
    def at(self, elapsed_s):
        if (not isinstance(elapsed_s, torch.Tensor) or elapsed_s.shape != self.duration_s.shape
                or elapsed_s.device != self.start.device or not elapsed_s.is_floating_point()
                or not torch.isfinite(elapsed_s).all()):
            raise ValueError('trajectory evaluation requires one finite elapsed time per row')
        elapsed = elapsed_s.to(self.start.dtype).clamp_min(0).minimum(self.duration_s)
        fraction = elapsed / self.duration_s
        sine = (1 - (2 * math.pi * self.frequency_start_hz * elapsed).cos()) / 2
        phase = (self.frequency_start_hz * elapsed
                 + (self.frequency_end_hz - self.frequency_start_hz) * elapsed.square() / (2 * self.duration_s))
        chirp = (1 - (2 * math.pi * phase).cos()) / 2
        smooth = fraction.square() * (3 - 2 * fraction)
        stop = torch.where(fraction < 0.1, fraction / 0.1,
                           torch.where(fraction < 0.5, 1.0,
                                       torch.where(fraction < 0.6, (0.6 - fraction) / 0.1, 0.0)))
        step = (fraction >= 0.5).to(fraction.dtype)
        weights = torch.stack((torch.zeros_like(fraction), fraction, sine, chirp, smooth, stop, step), -1)
        weight = weights.gather(1, self.kind[:, None]).clamp(0, 1)
        return self.start + (self.end - self.start) * weight

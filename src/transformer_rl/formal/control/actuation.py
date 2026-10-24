"""Private output-coordinate targets, explicit PC feedback and drive response.

Native armature/friction/viscosity and software drive lag are separate. All
six-axis tensors use policy order. No gearbox factor, gas pressure, USB delay
or native loss is secretly added to an effort calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .ownership import ControlSource, canonical_coaxial_displacement, validate_issued_control
from ..contracts import ACTION_ORDER


@dataclass(frozen=True)
class ResponseParameters:
    additional_armature: tuple[float, ...]
    coulomb_effort: tuple[float, ...]
    viscous_effort_per_speed: tuple[float, ...]
    effective_time_constant_s: tuple[float, ...]

    @classmethod
    def from_recorded_response(cls, response):
        names = tuple(response['joint_names'])
        if (len(names) != 6 or len(set(names)) != 6 or set(names) != set(ACTION_ORDER)
                or response.get('enabled') is not True or type(response.get('gain')) not in (int, float)
                or response.get('gain') != 1.):
            raise ValueError('response requires the complete six named output axes and unit gain')
        values = []
        for field in ('armature', 'frictionloss', 'damping', 'time_constant_s'):
            raw = response[field]
            if (not isinstance(raw, (tuple, list)) or len(raw) != 6
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in raw)):
                raise ValueError(f'response requires six finite nonnegative {field} values')
            values.append(tuple(float(raw[names.index(name)]) for name in ACTION_ORDER))
        return cls(*values)

    def native_parameters(self):
        # Static and dynamic friction are alternative native regimes, not two
        # software Coulomb subtractions. CAD inertia is retained separately.
        return {name: {'armature': self.additional_armature[index],
                       'friction': self.coulomb_effort[index],
                       'dynamic_friction': self.coulomb_effort[index],
                       'viscous_friction': self.viscous_effort_per_speed[index]}
                for index, name in enumerate(ACTION_ORDER)}


@dataclass(frozen=True)
class PairTargetDomain:
    """Predeclared signed auxiliary-minus-hip bounds, not native qualification.

    The selected geometry supplies each interval and its mirror sign. Both
    intervals contain zero, allowing inward float rounding without leaving the
    interval. Native collision/closure/travel validation remains independent.
    """

    signed_difference_rad: tuple[tuple[float, float], tuple[float, float]]
    signs: tuple[int, int]

    def __post_init__(self):
        if (len(self.signed_difference_rad) != 2 or len(self.signs) != 2
                or any(type(sign) is not int or sign not in (-1, 1) for sign in self.signs)):
            raise ValueError('pair domain requires two intervals and explicit mirror signs')
        parsed = []
        for bounds in self.signed_difference_rad:
            if (len(bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                    or not bounds[0] <= 0 <= bounds[1] or bounds[0] >= bounds[1]):
                raise ValueError('signed pair intervals must be finite, ordered and contain zero')
            parsed.append(tuple(float(v) for v in bounds))
        object.__setattr__(self, 'signed_difference_rad', tuple(parsed))
        object.__setattr__(self, 'signs', tuple(self.signs))


@dataclass(frozen=True)
class OutputTargets:
    issued_action: torch.Tensor
    raw_leg_position_rad: torch.Tensor
    admitted_leg_position_rad: torch.Tensor
    wheel_velocity_rad_s: torch.Tensor
    pair_projected: torch.Tensor
    common_turns: torch.Tensor
    controller_issued_at_s: torch.Tensor
    control_source: torch.Tensor


@dataclass(frozen=True)
class MotorEnvelope:
    """Wheel torque-speed limit (SCUT M3508 curve form), N·m; legs unlimited.

    tau_max(w) = clip(tau_sat (1 - w/w0), 0, cap), tau_min(w) = clip(tau_sat (-1 - w/w0), -cap, 0),
    with w the wheel output speed in rad/s and w0 the no-load speed; braking
    keeps the full cap. M3508 with the measured 15.62 reduction and 0.244 N·m/A
    at the wheel: 24 V stall about 30.2 N·m and no-load about 62 rad/s; the C620
    20 A limit (4.88 N·m) and the 4.5 N·m cap bind below about 53 rad/s. The
    leg drives keep SCUT's ideal PD (no envelope); their curve is unidentified.
    """

    saturation_nm: tuple[float, float] = (30.2, 30.2)
    no_load_rad_s: tuple[float, float] = (62., 62.)

    def __post_init__(self):
        for name in ('saturation_nm', 'no_load_rad_s'):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value)):
                raise ValueError('motor envelope requires two positive finite wheel values per field')
            object.__setattr__(self, name, tuple(float(v) for v in value))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'enabled'} or value['enabled'] is not True:
            raise ValueError('motor envelope configuration is {"enabled": true}; constants are frozen in code')
        return cls()

    def limits(self, speed_rad_s, no_load_scale=None):
        """[N,6] lower/upper limits; the four leg axes are unbounded here."""
        saturation = speed_rad_s.new_tensor(self.saturation_nm)
        no_load = speed_rad_s.new_tensor(self.no_load_rad_s)
        if no_load_scale is not None:
            no_load = no_load * no_load_scale[:, 4:]
        ratio = speed_rad_s[:, 4:] / no_load
        unbounded = torch.full_like(speed_rad_s[:, :4], float('inf'))
        return (torch.cat((-unbounded, (saturation * (-1 - ratio)).clamp_max(0)), -1),
                torch.cat((unbounded, (saturation * (1 - ratio)).clamp_min(0)), -1))


@dataclass(frozen=True)
class FeedbackEffort:
    requested_effort: torch.Tensor
    capped_effort: torch.Tensor
    effort_bounds: torch.Tensor
    saturated: torch.Tensor
    controller_issued_at_s: torch.Tensor
    pd_computed_at_s: torch.Tensor
    control_source: torch.Tensor


@dataclass(frozen=True)
class ResetHoldAction:
    requested_action: torch.Tensor
    admitted: torch.Tensor
    target_error_rad: torch.Tensor
    pair_projected: torch.Tensor


@dataclass(frozen=True)
class PreparationHold:
    requested_action: torch.Tensor
    leg_position_rad: torch.Tensor
    common_turns: torch.Tensor
    admitted: torch.Tensor
    target_error_rad: torch.Tensor
    pair_admitted: torch.Tensor
    position_tolerance_rad: float


class OutputFeedbackController:
    """Position PD on four root output axes; wheel output speed feedback on two.

    Same-side root axes receive a COMMON full-turn winding. Independently
    wrapping the two position errors can reverse the relative closed-chain
    motion at the angular cut. Pair-domain projection is explicit and logged;
    omitting it is permitted only for a separately declared control probe.
    """

    def __init__(self, nominal_position_rad, num_envs, *, pair_domain, device='cpu'):
        if (type(num_envs) is not int or num_envs < 1 or len(nominal_position_rad) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in nominal_position_rad)
                or pair_domain is not None and not isinstance(pair_domain, PairTargetDomain)):
            raise ValueError('controller requires a finite complete nominal pose and explicit pair-domain choice')
        self.num_envs, self.device = num_envs, torch.device(device)
        self.nominal = torch.tensor(nominal_position_rad, dtype=torch.float32, device=self.device)
        self.action_bounds = torch.tensor((3., 3., 3., 3., 9., 9.), device=self.device)
        self.effort_bounds = torch.tensor((40., 40., 40., 40., 4.5, 4.5), device=self.device)
        # Active output-axis gains used by effort(); wheel speed feedback is
        # damping, not a position stiffness. These are the actual controller
        # constants, not randomized or inferred values.
        self.stiffness = torch.tensor((160., 160., 160., 160., 0., 0.),
                                      dtype=torch.float32, device=self.device)
        self.damping = torch.tensor((2.5, 2.5, 2.5, 2.5, .6, .6),
                                    dtype=torch.float32, device=self.device)
        self.pair_domain = pair_domain
        if pair_domain is not None:
            self._difference = torch.tensor(pair_domain.signed_difference_rad, dtype=torch.float64, device=self.device)
            self._signs = torch.tensor(pair_domain.signs, dtype=torch.float64, device=self.device)

    def _matrix(self, name, value, width):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width)
                or value.device != self.device or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f'{name} requires a finite float32 {(self.num_envs, width)} tensor')

    def _clock(self, name, value):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or value.device != self.device or value.dtype != torch.float64
                or not torch.isfinite(value).all() or (value < 0).any()):
            raise ValueError(f'{name} requires a finite nonnegative float64 clock')

    @torch.no_grad()
    def decode(self, issued_action, feedback_position_rad, controller_issued_at_s):
        self._matrix('issued_action', issued_action, 6)
        self._matrix('feedback_position_rad', feedback_position_rad, 6)
        self._clock('controller_issued_at_s', controller_issued_at_s)
        issued = issued_action.clamp(-self.action_bounds, self.action_bounds)
        desired = self.nominal[:4] + .25 * issued[:, :4]
        pairs, feedback = desired.double().unflatten(-1, (2, 2)), feedback_position_rad[:, :4].double().unflatten(-1, (2, 2))
        center = pairs.mean(-1)
        projected = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        if self.pair_domain is not None:
            original = self._signs * (pairs[..., 1] - pairs[..., 0])
            difference = original.clamp(self._difference[:, 0], self._difference[:, 1])
            projected = difference != original
            pairs = torch.stack((center - self._signs * difference / 2,
                                 center + self._signs * difference / 2), -1)
        turns = torch.floor((feedback.mean(-1) - center) / (2 * math.pi) + .5)
        admitted = (pairs + turns[..., None] * (2 * math.pi)).to(torch.float32)
        if self.pair_domain is not None:
            # Native float rounding can put a projected endpoint just outside
            # its interval. Rebuild the auxiliary coordinate and move one ULP
            # toward the hip; zero lies in both intervals by construction.
            hip = admitted[..., 0]
            actual = self._signs * (admitted[..., 1].double() - hip.double())
            outside = (actual < self._difference[:, 0]) | (actual > self._difference[:, 1])
            repaired = (hip.double() + self._signs * difference).to(torch.float32)
            repaired = torch.nextafter(repaired, hip)
            admitted[..., 1] = torch.where(outside, repaired, admitted[..., 1])
        if not torch.isfinite(admitted).all():
            raise ValueError('position target winding overflowed its native representation')
        return OutputTargets(issued.clone(), desired.clone(), admitted.flatten(-2),
                              issued[:, 4:] * 10, projected, turns.clone(), controller_issued_at_s.clone(),
                              torch.full((self.num_envs,), int(ControlSource.POLICY), dtype=torch.long, device=self.device))

    @torch.no_grad()
    def preparation_hold(self, feedback_position_rad, *, position_tolerance_rad):
        """Describe an exact script-owned hold without expanding policy targets.

        Only a common pair winding is removed from the logged normalized
        command. No independent wrapping,position clipping or pair projection
        is allowed to turn a rejected physical pose into an admitted hold.
        """
        self._matrix('preparation output coordinates', feedback_position_rad, 6)
        if (type(position_tolerance_rad) not in (int, float) or not math.isfinite(position_tolerance_rad)
                or position_tolerance_rad <= 0):
            raise ValueError('preparation hold needs an explicit finite target comparison tolerance')
        q = feedback_position_rad[:, :4]
        difference = (q.double() - self.nominal[:4].double()).unflatten(-1, (2, 2))
        canonical_position, turns = canonical_coaxial_displacement(q, self.nominal[:4])
        common = (difference[..., 1] - difference[..., 0]).abs().lt(math.pi).all(-1)
        common &= turns.abs().lt(2**52).all(-1)
        action = torch.zeros_like(feedback_position_rad)
        action[:, :4] = (canonical_position/.25).float()
        reconstructed = (self.nominal[:4].double() + .25*action[:, :4].double()
                         + turns.repeat_interleave(2, -1)*(2*math.pi)).float()
        error = (reconstructed - q).abs()
        pair = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        if self.pair_domain is not None:
            positions = q.double().unflatten(-1, (2, 2))
            actual = self._signs*(positions[..., 1] - positions[..., 0])
            pair = (actual >= self._difference[:, 0]) & (actual <= self._difference[:, 1])
        canonical = (torch.isfinite(action).all(-1) & action[:, :4].abs().le(6*math.pi + 1e-6).all(-1))
        admitted = common & canonical & pair.all(-1) & (error <= position_tolerance_rad).all(-1)
        return PreparationHold(action, q.clone(), turns, admitted, error, pair, float(position_tolerance_rad))

    def validate_preparation_hold(self, hold, feedback_position_rad, mask):
        if (not isinstance(hold, PreparationHold) or not isinstance(mask, torch.Tensor)
                or mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError('preparation hold validation needs its exact native row mask')
        expected = self.preparation_hold(feedback_position_rad, position_tolerance_rad=hold.position_tolerance_rad)
        for field in fields(expected):
            actual, wanted = getattr(hold, field.name), getattr(expected, field.name)
            if isinstance(wanted, torch.Tensor):
                if (not isinstance(actual, torch.Tensor) or actual.shape != wanted.shape or actual.dtype != wanted.dtype
                        or actual.device != wanted.device or not torch.equal(actual[mask], wanted[mask])):
                    raise ValueError('preparation hold differs from the actual encoded reset coordinates or controller')
            elif actual != wanted:
                raise ValueError('preparation hold comparison recipe changed')
        if not expected.admitted[mask].all():
            raise ValueError('preparation hold violates common winding,pair bounds or encoded target tolerance')
        return expected

    @torch.no_grad()
    def reset_hold_action(self, feedback_position_rad, *, position_tolerance_rad):
        """Describe a zero-wheel-speed hold;never silently clip a recovery pose.

        Common pair winding is the inverse of decode. Independent axis wrapping
        would accept an impossible knee geometry or change a chain's branch.
        The returned admission includes the actual pair-domain projection.
        """
        self._matrix('reset output coordinates', feedback_position_rad, 6)
        if (type(position_tolerance_rad) not in (int, float) or not math.isfinite(position_tolerance_rad)
                or position_tolerance_rad <= 0):
            raise ValueError('reset output hold needs an explicit numeric target comparison tolerance')
        difference, _ = canonical_coaxial_displacement(feedback_position_rad[:, :4], self.nominal[:4])
        action = torch.zeros_like(feedback_position_rad)
        action[:, :4] = (difference / .25).float()
        decoded = self.decode(action, feedback_position_rad, torch.zeros(self.num_envs, dtype=torch.float64,
                                                                         device=self.device))
        error = (decoded.admitted_leg_position_rad - feedback_position_rad[:, :4]).abs()
        admitted = (action.abs() <= self.action_bounds).all(-1) & (error <= position_tolerance_rad).all(-1)
        return ResetHoldAction(action, admitted, error, decoded.pair_projected)

    @torch.no_grad()
    def effort(self, targets: OutputTargets, feedback_position_rad, feedback_velocity_rad_s,
               pd_computed_at_s, *, strength=None, envelope: MotorEnvelope | None = None,
               motor_velocity_rad_s=None, no_load_scale=None):
        if not isinstance(targets, OutputTargets):
            raise TypeError('feedback requires decoded named output targets')
        self._matrix('feedback_position_rad', feedback_position_rad, 6)
        self._matrix('feedback_velocity_rad_s', feedback_velocity_rad_s, 6)
        self._matrix('admitted_leg_position_rad', targets.admitted_leg_position_rad, 4)
        self._matrix('wheel_velocity_rad_s', targets.wheel_velocity_rad_s, 2)
        validate_issued_control(targets.issued_action, targets.control_source,
                                num_envs=self.num_envs, device=self.device)
        self._clock('controller_issued_at_s', targets.controller_issued_at_s)
        self._clock('pd_computed_at_s', pd_computed_at_s)
        if (targets.controller_issued_at_s > pd_computed_at_s + 1e-9).any():
            raise ValueError('feedback cannot use a policy target issued in the future')
        scale = torch.ones_like(feedback_position_rad) if strength is None else strength
        self._matrix('strength', scale, 6)
        if ((scale <= 0) | (scale > 1)).any():
            raise ValueError('strength is an explicit positive derating, never an increased nominal output cap')
        legs = (160 * (targets.admitted_leg_position_rad - feedback_position_rad[:, :4])
                - 2.5 * feedback_velocity_rad_s[:, :4])
        wheels = .6 * (targets.wheel_velocity_rad_s - feedback_velocity_rad_s[:, 4:])
        requested = torch.cat((legs, wheels), -1) * scale
        bounds = self.effort_bounds * scale
        if not torch.isfinite(requested).all() or not torch.isfinite(bounds).all():
            raise ValueError('feedback effort or its scaled envelope overflowed before saturation')
        lower, upper = -bounds, bounds
        if envelope is not None:
            # Back-EMF depends on the true shaft speed, not delayed feedback.
            speed = feedback_velocity_rad_s if motor_velocity_rad_s is None else motor_velocity_rad_s
            self._matrix('motor_velocity_rad_s', speed, 6)
            low, high = envelope.limits(speed, no_load_scale)
            lower, upper = torch.maximum(lower, low * scale), torch.minimum(upper, high * scale)
        capped = torch.maximum(torch.minimum(requested, upper), lower)
        return FeedbackEffort(requested, capped, bounds, requested != capped,
                               targets.controller_issued_at_s.clone(), pd_computed_at_s.clone(), targets.control_source.clone())


@dataclass(frozen=True)
class DriveInterval:
    mean_effort: torch.Tensor
    endpoint_drive_effort: torch.Tensor
    impulse: torch.Tensor


class EffectiveDriveResponse:
    """Analytic held-command torque response, independent of native losses.

    Zero tau means no ADDITIONAL modeled lag, not measured zero hardware
    latency. Split an interval at actual ordered arrivals, then sum impulses.
    This does not invent intermediate feedback or additional physics steps.
    """

    def __init__(self, parameters: ResponseParameters, num_envs, device='cpu'):
        if (not isinstance(parameters, ResponseParameters) or type(num_envs) is not int or num_envs < 1
                or any(len(values) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                                              for v in values)
                       for values in (parameters.additional_armature, parameters.coulomb_effort,
                                      parameters.viscous_effort_per_speed, parameters.effective_time_constant_s))):
            raise ValueError('drive response requires complete finite named parameters')
        self.parameters, self.num_envs, self.device = parameters, num_envs, torch.device(device)
        self.state = torch.zeros(num_envs, 6, device=self.device)
        self._tau = torch.tensor(parameters.effective_time_constant_s, dtype=torch.float64, device=self.device)[None]
        self._direct = self._tau == 0
        self._tau_scale = torch.ones(num_envs, 6, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def set_time_constant_scale(self, mask, scale):
        """Per-row multiplier of the identified lag; zero-lag axes stay direct."""
        if (not isinstance(scale, torch.Tensor) or scale.shape != self.state.shape
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('drive time-constant scale requires positive finite [N,6] values')
        self._tau_scale[mask] = scale[mask].double()

    @torch.no_grad()
    def advance(self, command, duration_s, effort_bounds):
        for name, value in (('command', command), ('effort_bounds', effort_bounds)):
            if (not isinstance(value, torch.Tensor) or value.shape != self.state.shape
                    or value.device != self.device or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f'{name} must match the finite native effort shape/dtype/device')
        if (not isinstance(duration_s, torch.Tensor) or duration_s.shape != (self.num_envs,)
                or duration_s.device != self.device or duration_s.dtype != torch.float64
                or not torch.isfinite(duration_s).all() or (duration_s < 0).any() or (effort_bounds <= 0).any()):
            raise ValueError('drive segment requires finite nonnegative durations and positive effort bounds')
        if not torch.isfinite(self.state).all():
            raise ValueError('drive endpoint state is invalid')
        seconds, bound = duration_s[:, None], effort_bounds.double()
        initial = self.state.double().clamp(-bound, bound)
        held = command.double().clamp(-bound, bound)
        ratio = seconds / torch.where(self._direct, 1., self._tau * self._tau_scale)
        beta = -torch.expm1(-ratio)
        small_weight = ratio * (.5 + ratio * (-1 / 6 + ratio * (1 / 24 + ratio * (-1 / 120 + ratio / 720))))
        ratio_nonzero = torch.where(ratio == 0, 1., ratio)
        input_weight = torch.where(ratio.abs() < 1e-3, small_weight, 1 - beta / ratio_nonzero)
        mean = torch.where(self._direct, held, initial + input_weight * (held - initial))
        advanced = torch.where(self._direct, held, initial + beta * (held - initial))
        endpoint = torch.where(seconds > 0, advanced, initial).clamp(-bound, bound)
        impulse = seconds * mean
        if not torch.isfinite(impulse).all() or not torch.isfinite(endpoint).all():
            raise ValueError('drive integration overflowed before publishing a segment')
        self.state.copy_(endpoint.to(torch.float32))
        return DriveInterval(mean.to(torch.float32), self.state.clone(), impulse)

    @torch.no_grad()
    def reset(self, mask):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.device != self.device or mask.dtype != torch.bool):
            raise ValueError('drive reset requires a same-device row mask')
        self.state[mask] = 0

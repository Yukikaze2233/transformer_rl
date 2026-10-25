"""Private pressure curve on the selected asset's native prismatic coordinates.

Pressure is applied once as slider effort. SDK actuator damping, strut mass
and closed-chain constraints stay in the physics model. The printed force
curve is an engineering prior, not a measured complete spring/hysteresis model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def gas_spring_force_n(u, coefficients, fit_end):
    """Datasheet cubic inside its fitted domain, continued linearly beyond it.

    ``u`` is compression / stroke already clamped to [0, 1]. The printed curve
    stops at the reserve-stroke mark (u = 0.9 for BKB0.45-080-205); the rest of
    the mechanical stroke continues with the end slope, without extrapolating
    the cubic.
    """
    a, b, c, d = coefficients
    inside = u.clamp_max(fit_end)
    end_slope = b + fit_end * (2 * c + 3 * d * fit_end)
    return a + inside * (b + inside * (c + inside * d)) + end_slope * (u - fit_end).clamp_min(0)


def gas_spring_work_j_per_m(u, coefficients, fit_end):
    """Integral of ``gas_spring_force_n`` over normalized compression from 0 to ``u``."""
    a, b, c, d = coefficients
    inside = u.clamp_max(fit_end)
    tail = (u - fit_end).clamp_min(0)
    end_force = a + fit_end * (b + fit_end * (c + fit_end * d))
    end_slope = b + fit_end * (2 * c + 3 * d * fit_end)
    return inside * (a + inside * (b / 2 + inside * (c / 3 + inside * d / 4))) + tail * (end_force + end_slope * tail / 2)


@dataclass(frozen=True)
class PressureParameters:
    joint_names: tuple[str, str]
    compression_at_zero_m: tuple[float, float]
    stroke_m: tuple[float, float]
    force_coefficients_n: tuple[float, float, float, float]
    fitted_u_domain: tuple[float, float]
    slider_actuator_damping_n_s_m: float

    @classmethod
    def from_asset(cls, manifest, model_spec, fit):
        names = tuple(manifest['spring_joint_names'])
        if names != ('L_spring_slide', 'R_spring_slide'):
            raise ValueError('pressure binding requires the two named native sliders')
        joints = {joint['name']: joint for joint in model_spec['joints']}
        compression, strokes = [], []
        for name in names:
            binding = model_spec['spring_binding'][name]
            if (joints[name]['type'] != 'prismatic' or binding['effort_sign'] != 1
                    or binding['pressure_mpa'] != fit['pressure_mpa']
                    or binding['stroke_m'] != fit['stroke_m']
                    or not math.isclose(binding['source_mount_distance_m'] + binding['compression_at_q_zero_m'],
                                         binding['full_extension_pin_distance_m'], abs_tol=1e-12, rel_tol=0)):
                raise ValueError('pressure fit and actual slider datum do not match')
            compression.append(binding['compression_at_q_zero_m'])
            strokes.append(binding['stroke_m'])
        coefficients = tuple(fit['monomial_coefficients_n'])
        domain = tuple(fit['normalized_fit_domain'])
        damping = manifest['passive_damping']['slider_N_s_per_m']
        if (len(coefficients) != 4 or len(domain) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in (*compression, *strokes, *coefficients, *domain, damping))
                or min(strokes) <= 0 or min(compression) < 0 or damping < 0
                or not 0 <= domain[0] < domain[1] <= 1
                or not math.isclose(fit['max_model_compression_m'], domain[1] * fit['stroke_m'],
                                     abs_tol=1e-12, rel_tol=0)):
            raise ValueError('pressure coefficients, fitted domain, stroke or native damping are invalid')
        return cls(names, tuple(compression), tuple(strokes), coefficients, domain, float(damping))


@dataclass(frozen=True)
class PressureState:
    slider_effort_n: torch.Tensor
    compression_m: torch.Tensor
    normalized_compression: torch.Tensor
    within_fitted_curve: torch.Tensor
    within_mechanical_stroke: torch.Tensor
    endpoint_clamped: torch.Tensor
    model_pressure_potential_j: torch.Tensor


class NativeSliderPressure:
    """Evaluate pressure without adding a second external mounting wrench.

    The fitted curve may end before the full geometric stroke. Extrapolated
    points and stop penetration remain explicit flags. Holding force at curve
    endpoints outside the stroke protects a numerical evaluation; it does not
    make those states mechanically admissible or physically qualified.
    """

    def __init__(self, parameters: PressureParameters, num_envs, device='cpu'):
        if not isinstance(parameters, PressureParameters) or type(num_envs) is not int or num_envs < 1:
            raise ValueError('pressure requires bound parameters and a positive environment count')
        self.parameters, self.num_envs, self.device = parameters, num_envs, torch.device(device)
        self._zero = torch.tensor(parameters.compression_at_zero_m, dtype=torch.float64, device=self.device)
        self._stroke = torch.tensor(parameters.stroke_m, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def evaluate(self, slider_position_m, *, strength=None):
        if (not isinstance(slider_position_m, torch.Tensor) or slider_position_m.shape != (self.num_envs, 2)
                or slider_position_m.dtype != torch.float32 or slider_position_m.device != self.device
                or not torch.isfinite(slider_position_m).all()):
            raise ValueError('pressure requires finite native float32 slider positions')
        scale = torch.ones(self.num_envs, 2, device=self.device) if strength is None else strength
        if (not isinstance(scale, torch.Tensor) or scale.shape not in ((self.num_envs, 1), (self.num_envs, 2))
                or scale.device != self.device or scale.dtype != torch.float32
                or not torch.isfinite(scale).all() or (scale < 0).any()):
            raise ValueError('pressure strength requires explicit finite nonnegative common or per-side values')
        compression = self._zero - slider_position_m.double()
        raw_u = compression / self._stroke
        u = raw_u.clamp(0, 1)
        coefficients, fit_end = self.parameters.force_coefficients_n, self.parameters.fitted_u_domain[1]
        force = gas_spring_force_n(u, coefficients, fit_end) * scale
        # Integral of F(compression) from full extension. Linear tails preserve
        # virtual work for endpoint-held numerical forces outside the stroke.
        full = gas_spring_force_n(torch.ones_like(u), coefficients, fit_end)
        potential = self._stroke * (gas_spring_work_j_per_m(u, coefficients, fit_end)
                                    + torch.where(raw_u < 0, coefficients[0] * raw_u, 0)
                                    + torch.where(raw_u > 1, full * (raw_u - 1), 0)) * scale
        if (not torch.isfinite(force).all() or (force < 0).any() or not torch.isfinite(potential).all()):
            raise ValueError('pressure evaluation produced invalid effort or model potential')
        native_force = force.to(torch.float32)
        if not torch.isfinite(native_force).all():
            raise ValueError('pressure effort overflowed its native representation')
        lo, hi = self.parameters.fitted_u_domain
        return PressureState(native_force, compression, raw_u,
                              (raw_u >= lo) & (raw_u <= hi), (raw_u >= 0) & (raw_u <= 1),
                              raw_u != u, potential)

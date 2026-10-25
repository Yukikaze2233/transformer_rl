"""Pressure-coordinate and virtual-work checks, without a simulator."""
from copy import deepcopy

import pytest
import torch

from transformer_rl.formal.control.pressure import NativeSliderPressure, PressureParameters


def fixture(count=4):
    names = ('L_spring_slide', 'R_spring_slide')
    manifest = {'spring_joint_names': list(names), 'passive_damping': {'slider_N_s_per_m': .002}}
    spec = {'joints': [{'name': name, 'type': 'prismatic'} for name in names],
            'spring_binding': {name: {'effort_sign': 1, 'pressure_mpa': 10., 'stroke_m': .08,
                                      'source_mount_distance_m': .16227281990887452,
                                      'compression_at_q_zero_m': .06972718009112547,
                                      'full_extension_pin_distance_m': .232} for name in names}}
    fit = {'pressure_mpa': 10., 'stroke_m': .08,
           'monomial_coefficients_n': [280., 0., 122.73591849113129, 53.96980440930969],
           'normalized_fit_domain': [0., .9], 'max_model_compression_m': .072}
    parameters = PressureParameters.from_asset(manifest, spec, fit)
    return manifest, spec, fit, parameters, NativeSliderPressure(parameters, count)


def test_slider_coordinate_pressure_and_sdk_actuator_damping_remain_separate():
    _, _, fit, parameters, model = fixture(1)
    q = torch.tensor([[.01763681903690114, .01763681903690117]])
    result = model.evaluate(q)
    expected_u = (parameters.compression_at_zero_m[0] - float(q[0, 0])) / .08
    a, b, c, d = fit['monomial_coefficients_n']
    expected_force = a + expected_u * (b + expected_u * (c + expected_u * d))
    assert result.slider_effort_n[0, 0].item() == pytest.approx(expected_force, abs=3e-5)
    assert result.within_fitted_curve.all() and result.within_mechanical_stroke.all()
    assert not result.endpoint_clamped.any()
    assert parameters.slider_actuator_damping_n_s_m == .002
    assert parameters.joint_names == ('L_spring_slide', 'R_spring_slide')


def test_curve_tail_and_mechanical_overtravel_have_distinct_explicit_flags():
    _, _, _, parameters, model = fixture()
    requested_u = torch.tensor([.1, .85, .95, 1.1])[:, None].expand(-1, 2)
    q = torch.tensor(parameters.compression_at_zero_m)[None] - .08 * requested_u
    result = model.evaluate(q)
    assert torch.equal(result.within_fitted_curve[:, 0], torch.tensor([True, True, False, False]))
    assert torch.equal(result.within_mechanical_stroke[:, 0], torch.tensor([True, True, True, False]))
    assert torch.equal(result.endpoint_clamped[:, 0], torch.tensor([False, False, False, True]))
    # Beyond the printed curve (u > 0.9) the force continues linearly from the
    # end slope; past the stroke it is held at the u = 1 value.
    a, b, c, d = parameters.force_coefficients_n
    end = parameters.fitted_u_domain[1]
    end_force = a + end * (b + end * (c + end * d))
    slope = b + end * (2 * c + 3 * d * end)
    assert result.slider_effort_n[2, 0].item() == pytest.approx(end_force + slope * (.95 - end), abs=1e-3)
    assert result.slider_effort_n[-1, 0].item() == pytest.approx(end_force + slope * (1 - end), abs=1e-3)
    assert result.normalized_compression[-1, 0] > 1


def test_pressure_potential_gradient_matches_positive_native_extension_effort():
    _, _, _, _, model = fixture(1)
    q = torch.tensor([[.02, .03]])
    epsilon = 1e-5
    plus, minus = model.evaluate(q + epsilon), model.evaluate(q - epsilon)
    actual_step = (q + epsilon).double() - (q - epsilon).double()
    force_from_work = -(plus.model_pressure_potential_j - minus.model_pressure_potential_j) / actual_step
    torch.testing.assert_close(force_from_work.float(), model.evaluate(q).slider_effort_n, atol=.01, rtol=1e-5)


def test_endpoint_held_tail_preserves_model_virtual_work_without_claiming_admissibility():
    _, _, _, _, model = fixture(1)
    q = torch.tensor([[-.03, .09]])
    result = model.evaluate(q)
    assert not result.within_mechanical_stroke.any() and result.endpoint_clamped.all()
    epsilon = 1e-5
    plus, minus = model.evaluate(q + epsilon), model.evaluate(q - epsilon)
    force = -(plus.model_pressure_potential_j - minus.model_pressure_potential_j) / (
        (q + epsilon).double() - (q - epsilon).double())
    torch.testing.assert_close(force.float(), result.slider_effort_n, atol=.01, rtol=1e-5)


def test_common_and_single_side_pressure_events_are_explicit_and_do_not_mutate_results():
    _, _, _, _, model = fixture(2)
    q = torch.full((2, 2), .02)
    nominal = model.evaluate(q)
    common = model.evaluate(q, strength=torch.full((2, 1), 1.1))
    torch.testing.assert_close(common.slider_effort_n, nominal.slider_effort_n * 1.1)
    fault = model.evaluate(q, strength=torch.tensor([[0., 1.], [1., 0.]]))
    assert fault.slider_effort_n[0, 0] == 0 and fault.slider_effort_n[1, 1] == 0
    saved = nominal.slider_effort_n.clone()
    q.zero_()
    torch.testing.assert_close(nominal.slider_effort_n, saved)


@pytest.mark.parametrize('change', ['names', 'sign', 'pressure', 'stroke', 'mount', 'type', 'domain', 'coefficient'])
def test_mismatched_geometry_and_curve_are_not_silently_combined(change):
    manifest, spec, fit, *_ = fixture()
    manifest, spec, fit = deepcopy(manifest), deepcopy(spec), deepcopy(fit)
    if change == 'names':
        manifest['spring_joint_names'].reverse()
    elif change == 'sign':
        spec['spring_binding']['L_spring_slide']['effort_sign'] = -1
    elif change == 'pressure':
        fit['pressure_mpa'] = 9
    elif change == 'stroke':
        fit['stroke_m'] = .081
    elif change == 'mount':
        spec['spring_binding']['L_spring_slide']['source_mount_distance_m'] += .001
    elif change == 'type':
        spec['joints'][0]['type'] = 'revolute'
    elif change == 'domain':
        fit['normalized_fit_domain'][1] = 1.1
    else:
        fit['monomial_coefficients_n'][0] = float('nan')
    with pytest.raises(ValueError):
        PressureParameters.from_asset(manifest, spec, fit)


def test_nonfinite_native_effort_is_rejected_even_when_double_curve_evaluation_is_finite():
    _, _, _, _, model = fixture(1)
    with pytest.raises(ValueError, match='native representation'):
        model.evaluate(torch.full((1, 2), .02), strength=torch.full((1, 1), torch.finfo(torch.float32).max))

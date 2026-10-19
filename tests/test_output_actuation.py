"""Output-coordinate and drive-impulse checks without native physics."""
from dataclasses import replace
import math

import pytest
import torch

from transformer_rl._control_sources import ControlSource, validate_issued_control

from transformer_rl._output_actuation import (
    EffectiveDriveResponse, OutputFeedbackController, PairTargetDomain, ResponseParameters,
)
from transformer_rl._structured_collection import ACTION_ORDER


NOMINAL = (-.42, .13742282595395358, .42, -.1374155762580851, 0., 0.)


def parameters():
    return ResponseParameters((.01, .01, .007, .007, .0038, .0038),
                               (1., 2., 3., 4., .05, .1), (.1, .2, .3, .4, .001, .002),
                               (.00386, .00386, .00441, .00441, 0., 0.))


def controller(count=2, domain=True):
    # Synthetic intervals test the projection, not selected-asset reachability.
    bounds = PairTargetDomain(((-.1, 1.2), (-.1, 1.2)), (1, -1)) if domain else None
    return OutputFeedbackController(NOMINAL, count, pair_domain=bounds)


def test_complete_response_is_mapped_by_name_and_native_losses_are_not_gearbox_reflected():
    names = ('L_joint1', 'LL_joint1', 'L_joint3', 'R_joint1', 'RR_joint1', 'R_joint3')
    raw = {'enabled': True, 'gain': 1., 'joint_names': names,
           **{key: list(range(1, 7)) for key in ('armature', 'frictionloss', 'damping', 'time_constant_s')}}
    parsed = ResponseParameters.from_recorded_response(raw)
    assert parsed.additional_armature == (1., 2., 4., 5., 3., 6.)
    assert tuple(parsed.native_parameters()) == ACTION_ORDER
    assert parsed.native_parameters()['L_joint3'] == {
        'armature': 3., 'friction': 3., 'dynamic_friction': 3., 'viscous_friction': 3.}
    raw['armature'][2] = 100
    assert parsed.native_parameters()['L_joint3']['armature'] == 3.


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'gain', 'bool_gain', 'nan', 'negative'])
def test_incomplete_or_ambiguous_response_is_not_completed_with_zero_defaults(change):
    raw = {'enabled': True, 'gain': 1., 'joint_names': list(ACTION_ORDER),
           **{key: [0.] * 6 for key in ('armature', 'frictionloss', 'damping', 'time_constant_s')}}
    if change == 'partial':
        raw['damping'].pop()
    elif change == 'duplicate':
        raw['joint_names'][0] = raw['joint_names'][1]
    elif change in ('gain', 'bool_gain'):
        raw['gain'] = 1.1 if change == 'gain' else True
    else:
        raw['frictionloss'][0] = float('nan') if change == 'nan' else -1
    with pytest.raises(ValueError):
        ResponseParameters.from_recorded_response(raw)


def test_nominal_targets_use_four_output_positions_and_two_wheel_speeds_only():
    control = controller()
    q = torch.tensor(NOMINAL)[None].repeat(2, 1)
    q[:, 4:] = torch.tensor([123456., -876543.])
    targets = control.decode(torch.zeros(2, 6), q, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(targets.admitted_leg_position_rad, torch.tensor(NOMINAL[:4])[None].repeat(2, 1))
    assert not targets.wheel_velocity_rad_s.any() and not targets.pair_projected.any()
    raw = torch.tensor([[20., -20., 20., -20., 20., -20.]]).repeat(2, 1)
    decoded = control.decode(raw, q, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(decoded.issued_action, torch.tensor([[3., -3., 3., -3., 9., -9.]]).repeat(2, 1))
    torch.testing.assert_close(decoded.wheel_velocity_rad_s, torch.tensor([[90., -90.]]).repeat(2, 1))


def test_paired_winding_preserves_relative_motion_across_the_angular_cut():
    control = controller(domain=False)
    q = torch.tensor(NOMINAL)[None].repeat(2, 1)
    q[:, :2] += 2 * math.pi
    q[:, 2:4] -= 2 * math.pi
    targets = control.decode(torch.zeros(2, 6), q, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(targets.admitted_leg_position_rad, q[:, :4], atol=1e-6, rtol=0)
    assert torch.equal(targets.common_turns, torch.tensor([[1., -1.]], dtype=torch.float64).repeat(2, 1))
    raw_delta = targets.raw_leg_position_rad.unflatten(-1, (2, 2)).diff(dim=-1)
    native_delta = targets.admitted_leg_position_rad.unflatten(-1, (2, 2)).diff(dim=-1)
    torch.testing.assert_close(raw_delta, native_delta, atol=1e-6, rtol=0)


def test_pair_projection_preserves_center_and_reports_saturation_without_changing_raw_actions():
    control = controller()
    q = torch.tensor(NOMINAL)[None].repeat(2, 1)
    actions = torch.tensor([[-3., 3., 3., -3.], [3., -3., -3., 3.]])
    targets = control.decode(torch.cat((actions, torch.zeros(2, 2)), -1), q, torch.zeros(2, dtype=torch.float64))
    assert targets.pair_projected.all()
    raw = targets.raw_leg_position_rad.double().unflatten(-1, (2, 2))
    admitted = targets.admitted_leg_position_rad.double().unflatten(-1, (2, 2))
    torch.testing.assert_close(admitted.mean(-1), raw.mean(-1), atol=1e-7, rtol=0)
    delta = torch.tensor([1., -1.]) * (admitted[..., 1] - admitted[..., 0])
    assert (delta >= -.1).all() and (delta <= 1.2).all()
    torch.testing.assert_close(targets.issued_action[:, :4], actions)


def test_feedback_uses_output_axis_pd_and_keeps_requested_capped_and_derated_effort_separate():
    control = controller()
    q = torch.tensor(NOMINAL)[None].repeat(2, 1)
    action = torch.zeros(2, 6)
    action[:, 0] = 1
    action[:, 4:] = torch.tensor([2., -2.])
    targets = control.decode(action, q, torch.zeros(2, dtype=torch.float64))
    dq = torch.zeros_like(q)
    dq[:, 1] = 2
    effort = control.effort(targets, q, dq, torch.full((2,), .005, dtype=torch.float64))
    torch.testing.assert_close(effort.requested_effort, torch.tensor([[40., -5., 0., 0., 12., -12.]]).repeat(2, 1))
    torch.testing.assert_close(effort.capped_effort, torch.tensor([[40., -5., 0., 0., 4.5, -4.5]]).repeat(2, 1))
    reduced = control.effort(targets, q, dq, torch.full((2,), .005, dtype=torch.float64), strength=torch.full_like(q, .8))
    torch.testing.assert_close(reduced.capped_effort, effort.capped_effort * .8)
    torch.testing.assert_close(reduced.effort_bounds, effort.effort_bounds * .8)
    assert reduced.controller_issued_at_s.dtype == torch.float64


def test_feedback_rejects_future_targets_and_cannot_increase_nominal_caps_through_strength():
    control = controller()
    q = torch.tensor(NOMINAL)[None].repeat(2, 1)
    targets = control.decode(torch.zeros_like(q), q, torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match='future'):
        control.effort(targets, q, torch.zeros_like(q), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match='derating'):
        control.effort(targets, q, torch.zeros_like(q), torch.ones(2, dtype=torch.float64), strength=torch.full_like(q, 1.1))


def test_drive_endpoint_and_interval_impulse_match_an_independent_closed_form():
    params = parameters()
    response = EffectiveDriveResponse(params, 2)
    command, duration = torch.full((2, 6), 2.), torch.tensor([.005, .001], dtype=torch.float64)
    bound = torch.full_like(command, 40.)
    result = response.advance(command, duration, bound)
    for row, seconds in enumerate(duration.tolist()):
        for axis, tau in enumerate(params.effective_time_constant_s):
            endpoint = 2 if tau == 0 else 2 * -math.expm1(-seconds / tau)
            impulse = seconds * 2 if tau == 0 else 2 * (seconds + tau * math.expm1(-seconds / tau))
            assert result.endpoint_drive_effort[row, axis].item() == pytest.approx(endpoint, abs=2e-7)
            assert result.impulse[row, axis].item() == pytest.approx(impulse, abs=1e-12)
    assert not torch.equal(result.mean_effort, result.endpoint_drive_effort)


def test_drive_segment_splitting_preserves_impulse_and_endpoint_without_inventing_feedback():
    whole, split = EffectiveDriveResponse(parameters(), 1), EffectiveDriveResponse(parameters(), 1)
    command, bound = torch.full((1, 6), 3.), torch.full((1, 6), 40.)
    once = whole.advance(command, torch.tensor([.005], dtype=torch.float64), bound)
    impulse = torch.zeros(1, 6, dtype=torch.float64)
    for _ in range(5):
        step = split.advance(command, torch.tensor([.001], dtype=torch.float64), bound)
        impulse += step.impulse
    torch.testing.assert_close(impulse, once.impulse, atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(split.state, whole.state, atol=2e-7, rtol=1e-6)


def test_small_duration_keeps_the_first_nonzero_impulse_term():
    response = EffectiveDriveResponse(parameters(), 1)
    result = response.advance(torch.ones(1, 6), torch.tensor([1e-10], dtype=torch.float64), torch.full((1, 6), 40.))
    expected = 1e-20 / (2 * parameters().effective_time_constant_s[0])
    assert result.impulse[0, 0].item() == pytest.approx(expected, rel=1e-6, abs=1e-30)


def test_zero_duration_cannot_overwrite_a_drive_endpoint_and_outputs_own_storage():
    response = EffectiveDriveResponse(parameters(), 2)
    bound = torch.full((2, 6), 40.)
    first = response.advance(torch.ones(2, 6), torch.full((2,), .005, dtype=torch.float64), bound)
    saved = first.endpoint_drive_effort.clone()
    empty = response.advance(torch.full((2, 6), -10.), torch.zeros(2, dtype=torch.float64), bound)
    assert not empty.impulse.any()
    torch.testing.assert_close(response.state, saved)
    response.reset(torch.tensor([True, False]))
    assert not response.state[0].any()
    torch.testing.assert_close(first.endpoint_drive_effort, saved)
    torch.testing.assert_close(response.state[1], saved[1])


def test_response_does_not_subtract_native_losses_or_apply_an_extra_reduction_ratio():
    direct = replace(parameters(), effective_time_constant_s=(0.,) * 6)
    response = EffectiveDriveResponse(direct, 1)
    requested, bound = torch.tensor([[1., -2., 3., -4., 4.5, -4.5]]), torch.tensor([[40., 40., 40., 40., 4.5, 4.5]])
    result = response.advance(requested, torch.tensor([.005], dtype=torch.float64), bound)
    torch.testing.assert_close(result.mean_effort, requested)
    torch.testing.assert_close(result.endpoint_drive_effort, requested)


@pytest.mark.parametrize('bad', ['nan', 'negative_time', 'bound'])
def test_bad_drive_segment_is_rejected_before_state_changes(bad):
    response = EffectiveDriveResponse(parameters(), 1)
    command, duration, bound = torch.ones(1, 6), torch.tensor([.005], dtype=torch.float64), torch.full((1, 6), 40.)
    if bad == 'nan':
        command[0, 0] = float('nan')
    elif bad == 'negative_time':
        duration[0] = -1
    else:
        bound[0, 0] = 0
    with pytest.raises(ValueError):
        response.advance(command, duration, bound)
    assert not response.state.any()

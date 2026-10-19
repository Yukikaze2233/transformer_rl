"""Necessary wheel bounds and raw/admitted command paths, with synthetic geometry."""
from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from transformer_rl._command_trajectory import (
    CommandAdmission, CommandRates, CommandStream, CommandTrajectory, DriveEnvelope, TrajectoryKind,
)
from transformer_rl._task_curriculum import ContinuousTaskSampler


def envelope():
    # Test geometry is not a measurement of the selected robot asset.
    return DriveEnvelope(wheel_radius_m=0.06, track_width_m=0.4)


def test_joint_request_is_preserved_and_projection_preserves_signed_curvature():
    bounds = envelope()
    raw = torch.tensor([[5.0, 0, 6 * torch.pi, 0.30], [-5.0, 0, -6 * torch.pi, 0.30], [0., 0, 0, 0.17]])
    before = raw.clone()
    target, wheels, scale, necessary, lateral = bounds.project(raw)
    assert torch.equal(raw, before)
    assert necessary.tolist() == [False, False, True]
    assert not lateral.any()
    assert (bounds.wheel_speeds(target).abs().amax(-1) <= 90.0001).all()
    torch.testing.assert_close(target[:2, 0] / target[:2, 2], raw[:2, 0] / raw[:2, 2])
    torch.testing.assert_close(wheels[:2].abs().amax(-1) * scale[:2], torch.full((2,), 90.0))
    torch.testing.assert_close(target[:, 3], raw[:, 3], atol=0, rtol=0)


def test_unsupported_body_lateral_request_is_reported_not_claimed_reachable():
    raw = torch.tensor([[1.0, 0.5, 0.0, 0.3]])
    target, _, _, necessary, lateral = envelope().project(raw)
    assert not necessary.item() and lateral.item()
    assert raw[0, 1] == 0.5 and target[0, 1] == 0


def test_common_slew_fraction_stays_inside_diamond_when_independent_clamps_do_not():
    bounds = envelope()
    initial = torch.tensor([[5.0, 0, 2.0, 0.30]])
    target = torch.tensor([[0.0, 0, 18.0, 0.17]])
    rates = CommandRates(0.1, 100.0, 0.2)
    naive = initial.clone()
    naive[:, 0] -= rates.forward_acceleration_m_s2 * 0.01
    naive[:, 2] += rates.yaw_acceleration_rad_s2 * 0.01
    assert bounds.wheel_speeds(naive).abs().max() > 90
    stream = CommandStream(initial, bounds, rates)
    admitted = stream.step(target)
    assert admitted.wheel_necessary_at_request.item()
    assert bounds.wheel_speeds(admitted.admitted).abs().max() <= 90.0001
    delta = (admitted.admitted - initial).abs()
    assert delta[0, 0] <= rates.forward_acceleration_m_s2 * 0.01 + 1e-6
    assert delta[0, 2] <= rates.yaw_acceleration_rad_s2 * 0.01 + 1e-6
    assert delta[0, 3] <= rates.height_rate_m_s * 0.01 + 1e-6


def test_full_request_sampler_composes_with_admission_without_dropping_denominator():
    draw = ContinuousTaskSampler(2101).sample(1024, 5)
    initial = torch.zeros(1024, 4)
    initial[:, 3] = 0.305
    bounds = envelope()
    stream = CommandStream(initial, bounds, CommandRates(2.0, 5.0, 0.1))
    admitted = stream.step(draw.requested_command)
    assert admitted.requested.shape[0] == 1024
    torch.testing.assert_close(admitted.requested, draw.requested_command, atol=0, rtol=0)
    assert (~admitted.wheel_necessary_at_request).any()
    assert (bounds.wheel_speeds(admitted.admitted).abs().amax(-1) <= 90.0001).all()
    assert (admitted.admitted[:, 0].abs() <= 0.02 + 1e-6).all()


def test_command_stream_resume_preserves_rate_phase_and_owns_returned_state():
    initial = torch.tensor([[0., 0, 0, 0.3]])
    rates = CommandRates(1.0, 3.0, 0.1)
    first = CommandStream(initial, envelope(), rates)
    target = torch.tensor([[2., 0, 4, 0.2]])
    first.step(target)
    saved = first.state_dict()
    restored = CommandStream(initial, envelope(), rates)
    restored.load_state_dict(saved)
    saved['current'].fill_(999)
    left, right = first.step(target), restored.step(target)
    for field in fields(CommandAdmission):
        torch.testing.assert_close(getattr(left, field.name), getattr(right, field.name), atol=0, rtol=0)
    left.admitted.fill_(888)
    assert first.current.max() < 1


def test_bad_command_restore_is_atomic_and_cannot_increase_bounds():
    stream = CommandStream(torch.tensor([[0., 0, 0, 0.3]]), envelope(), CommandRates(1, 3, 0.1))
    saved = stream.state_dict()
    wrong = deepcopy(saved)
    wrong['current'][0, 0] = 100
    with pytest.raises(ValueError):
        stream.load_state_dict(wrong)
    assert torch.equal(stream.current, saved['current'])
    wrong = deepcopy(saved)
    wrong['rates']['forward_acceleration_m_s2'] = 100
    with pytest.raises(ValueError):
        stream.load_state_dict(wrong)


def trajectory_rows():
    times = torch.linspace(0, 4, 401).repeat(7)
    kinds = torch.arange(7).repeat_interleave(401)
    start = torch.tensor([[-2., 0, -3., 0.17]]).expand(times.numel(), -1)
    end = torch.tensor([[5., 0, 6., 0.4]]).expand_as(start)
    return CommandTrajectory(start, end, torch.full_like(times, 4.0), kinds), times


def test_all_seven_trajectory_families_remain_inside_endpoint_request_box():
    trajectory, elapsed = trajectory_rows()
    values = trajectory.at(elapsed)
    assert (values >= trajectory.start - 1e-6).all() and (values <= trajectory.end + 1e-6).all()
    for kind in range(7):
        selected = values[trajectory.kind == kind]
        torch.testing.assert_close(selected[0], trajectory.start[0], atol=1e-6, rtol=0)
        if kind == TrajectoryKind.HOLD:
            torch.testing.assert_close(selected, trajectory.start[:401], atol=0, rtol=0)
        else:
            assert selected[:, 0].max() > 4.99
    for kind in (TrajectoryKind.RAMP, TrajectoryKind.SMOOTHSTEP, TrajectoryKind.BOUNDED_STEP, TrajectoryKind.CHIRP):
        selected = values[trajectory.kind == kind]
        torch.testing.assert_close(selected[-1], trajectory.end[0], atol=2e-6, rtol=0)


def test_start_stop_has_exact_hold_and_return_intervals():
    start = torch.tensor([[0., 0, 0., 0.3]]).expand(5, -1)
    end = torch.tensor([[3., 0, 2., 0.4]]).expand_as(start)
    trajectory = CommandTrajectory(start, end, torch.full((5,), 10.0),
                                    torch.full((5,), int(TrajectoryKind.START_STOP), dtype=torch.long))
    values = trajectory.at(torch.tensor([0., 1., 4., 6., 10.]))
    torch.testing.assert_close(values[[0, 3, 4]], start[[0, 3, 4]], atol=1e-6, rtol=0)
    torch.testing.assert_close(values[[1, 2]], end[[1, 2]], atol=1e-6, rtol=0)


def test_raw_step_is_not_an_unbounded_admitted_acceleration():
    trajectory = CommandTrajectory(torch.tensor([[0., 0, 0, 0.3]]), torch.tensor([[5., 0, 10., 0.17]]),
                                    torch.tensor([2.0]), torch.tensor([int(TrajectoryKind.BOUNDED_STEP)]))
    stream = CommandStream(torch.tensor([[0., 0, 0, 0.3]]), envelope(), CommandRates(2., 5., 0.1))
    result = stream.step(trajectory.at(torch.tensor([1.0])))
    assert result.requested[0, 0] == 5
    assert result.admitted[0, 0] <= 0.02 + 1e-6
    assert result.admitted[0, 2] <= 0.05 + 1e-6


def test_trajectory_owns_endpoint_and_frequency_inputs():
    start = torch.tensor([[0., 0, 0, 0.3]])
    end = torch.tensor([[2., 0, 1, 0.2]])
    duration = torch.tensor([2.0])
    frequency = torch.tensor([0.5])
    trajectory = CommandTrajectory(start, end, duration, torch.tensor([int(TrajectoryKind.SINE)]), frequency, frequency)
    expected = trajectory.at(torch.tensor([1.0]))
    start.fill_(99)
    end.fill_(88)
    duration.fill_(0.1)
    frequency.fill_(1)
    torch.testing.assert_close(trajectory.at(torch.tensor([1.0])), expected, atol=0, rtol=0)


@pytest.mark.parametrize('bounds', [(0, 0.4, 90), (0.06, -1, 90), (0.06, 0.4, float('nan'))])
def test_invalid_geometry_is_not_replaced_by_an_old_asset_default(bounds):
    with pytest.raises(ValueError):
        DriveEnvelope(*bounds)


def test_nonfinite_requests_or_invalid_clock_are_rejected_before_admission():
    stream = CommandStream(torch.tensor([[0., 0, 0, 0.3]]), envelope(), CommandRates(2, 5, 0.1))
    with pytest.raises(ValueError):
        stream.step(torch.tensor([[float('nan'), 0, 0, 0.3]]))
    trajectory, _ = trajectory_rows()
    with pytest.raises(ValueError):
        trajectory.at(torch.full((2807,), float('nan')))

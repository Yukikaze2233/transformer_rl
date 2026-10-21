"""Native angle wraps must not become coupled-leg command or history jumps."""
import math

import pytest
import torch

from transformer_rl.formal.control.actuation import OutputFeedbackController
from transformer_rl.formal.lab_env.joint_coordinates import ContinuousJointCoordinates


NAMES = ('hip', 'aux', 'knee', 'slide')


def coordinates(initial):
    return ContinuousJointCoordinates(initial, NAMES, ('hip', 'aux'), .005)


def test_no_wrap_is_bitwise_identical_and_repeated_reads_are_idempotent():
    raw = torch.tensor([[.1, -.2, .3, .04], [.2, -.3, .4, .05]])
    tracked = coordinates(raw)
    velocity = torch.zeros_like(raw)
    torch.testing.assert_close(tracked.read(raw, velocity, 0), raw, rtol=0, atol=0)
    raw.add_(.001)
    first = tracked.read(raw, velocity, 1)
    again = tracked.read(raw, velocity, 1)
    assert first is again
    torch.testing.assert_close(first, raw, rtol=0, atol=0)
    raw.add_(.01)
    torch.testing.assert_close(first, raw-.01, rtol=0, atol=1e-7)


@pytest.mark.parametrize('direction', [-1, 1])
def test_native_four_pi_wrap_preserves_continuous_motion(direction):
    raw = torch.tensor([[direction * 6.26, direction * 6.0, .3, .04]])
    tracked = coordinates(raw)
    velocity = torch.zeros_like(raw)
    tracked.read(raw, velocity, 0)
    wrapped = raw.clone()
    wrapped[:, 0] = direction * (6.36 - 4*math.pi)
    velocity[:, 0] = direction * 20
    actual = tracked.read(wrapped, velocity, 1)
    assert actual[0, 0].item() == pytest.approx(direction * 6.36, abs=1e-6)
    torch.testing.assert_close(actual[:, 1:], raw[:, 1:], rtol=0, atol=0)


def test_partial_reset_reseeds_only_reset_row_and_preserves_pre_reset_sample():
    raw = torch.tensor([[6.26, 6., .3, .04], [6.26, 6., .3, .04]])
    tracked = coordinates(raw)
    velocity = torch.zeros_like(raw)
    tracked.read(raw, velocity, 0)
    wrapped = raw.clone()
    wrapped[:, 0] = 6.36 - 4*math.pi
    before_reset = tracked.read(wrapped, velocity, 1)
    before_copy = before_reset.clone()
    reset = torch.tensor([[-3.5, -3.4, .3, .04]])
    tracked.reset_rows(torch.tensor([0]), reset)
    after = tracked.read(wrapped, velocity, 1)
    torch.testing.assert_close(before_reset, before_copy, rtol=0, atol=0)
    torch.testing.assert_close(after[0], reset[0], rtol=0, atol=0)
    torch.testing.assert_close(after[1], before_copy[1], rtol=0, atol=0)
    wrapped[0] = reset[0]
    wrapped[:, 0] += .01
    next_sample = tracked.read(wrapped, velocity, 2)
    assert next_sample[0, 0].item() == pytest.approx(-3.49, abs=1e-6)
    assert next_sample[1, 0].item() == pytest.approx(6.37, abs=1e-6)


def test_native_mutable_view_cannot_overwrite_the_last_raw_sample():
    raw = torch.tensor([[6.26, 6., .3, .04]])
    tracked = coordinates(raw)
    velocity = torch.zeros_like(raw)
    tracked.read(raw, velocity, 0)
    raw[0, 0] = 6.36 - 4*math.pi
    assert tracked.read(raw, velocity, 1)[0, 0].item() == pytest.approx(6.36, abs=1e-6)


def test_finite_knee_and_slider_positions_are_never_periodically_rewritten():
    raw = torch.zeros(1, 4)
    tracked = coordinates(raw)
    tracked.read(raw, raw, 0)
    moved = torch.tensor([[.1, .1, 7., 8.]])
    actual = tracked.read(moved, raw, 1)
    torch.testing.assert_close(actual[:, 2:], moved[:, 2:], rtol=0, atol=0)


def test_missing_tick_or_ambiguous_motion_fails_instead_of_hiding_a_physics_failure():
    raw = torch.zeros(1, 4)
    tracked = coordinates(raw)
    tracked.read(raw, raw, 0)
    with pytest.raises(RuntimeError, match='sample'):
        tracked.read(raw, raw, 2)
    bad = raw.clone()
    bad[0, 0] = math.pi + .1
    with pytest.raises(RuntimeError, match='ambiguous'):
        tracked.read(bad, raw, 1)
    velocity = torch.full_like(raw, 1000.)
    with pytest.raises(RuntimeError, match='ambiguous'):
        tracked.read(raw, velocity, 1)


def test_single_axis_wrap_does_not_change_common_pair_target_or_feedback_effort():
    # Reproduce the recorded right-leg crossing: one continuous coordinate
    # crosses +2*pi while its coupled output remains on the same revolution.
    before = torch.tensor([[6.26, 5., 0., 0., 0., 0.]])
    physical = torch.tensor([[6.36, 5.01, 0., 0., 0., 0.]])
    raw = physical.clone()
    raw[:, 0] -= 4*math.pi
    names = tuple(str(i) for i in range(6))
    tracked = ContinuousJointCoordinates(before, names, names[:4], .005)
    velocity = torch.zeros_like(raw)
    tracked.read(before, velocity, 0)
    repaired = tracked.read(raw, velocity, 1)
    controller = OutputFeedbackController((0.,)*6, 1, pair_domain=None)
    action, now = torch.zeros_like(raw), torch.zeros(1, dtype=torch.float64)
    expected = controller.decode(action, physical, now)
    actual = controller.decode(action, repaired, now)
    broken = controller.decode(action, raw, now)
    torch.testing.assert_close(actual.admitted_leg_position_rad, expected.admitted_leg_position_rad, rtol=0, atol=0)
    assert not torch.equal(broken.common_turns, expected.common_turns)
    expected_effort = controller.effort(expected, physical, velocity, now)
    repaired_effort = controller.effort(actual, repaired, velocity, now)
    torch.testing.assert_close(repaired_effort.capped_effort, expected_effort.capped_effort, rtol=0, atol=1e-4)

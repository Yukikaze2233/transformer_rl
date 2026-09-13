"""Targeted transport and clock checks; no simulator or training dependencies."""
from dataclasses import fields
import math

import pytest
import torch

from transformer_rl.timing import DelayedCommandChannel, TimingProfile


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def assert_snapshot_equal(actual, expected):
    for field in fields(actual):
        torch.testing.assert_close(
            getattr(actual, field.name), getattr(expected, field.name), equal_nan=True
        )


def test_zero_delay_requires_explicit_application_and_holds(device):
    channel = DelayedCommandChannel(torch.zeros(2, 3, device=device), capacity=2)
    target = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    receipt = channel.submit(target, now_s=0.0, delay_s=0.0)
    torch.testing.assert_close(receipt.sequence, torch.zeros(2, dtype=torch.int64, device=device))
    assert (channel.snapshot().applied_sequence == -1).all()
    applied = channel.advance(0.0)
    torch.testing.assert_close(applied.applied_target, target)
    torch.testing.assert_close(applied.applied_sequence, receipt.sequence)
    assert (applied.pending_count == 0).all()
    assert torch.isinf(applied.next_scheduled_at_s).all()
    again = channel.advance(0.0)
    assert_snapshot_equal(again, applied)
    held = channel.advance(1.0)
    torch.testing.assert_close(held.applied_target, target)
    torch.testing.assert_close(held.applied_at_s, applied.applied_at_s)


def test_half_policy_period_delay_and_late_controller_tick(device):
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), capacity=1)
    target = torch.ones(1, 1, device=device)
    receipt = channel.submit(target, now_s=0.0, delay_s=0.005)
    assert channel.advance(0.0049).applied_target.item() == 0
    applied = channel.advance(0.005)
    assert applied.applied_target.item() == 1
    assert applied.scheduled_at_s.item() == 0.005
    assert applied.applied_at_s.item() == 0.005
    channel.submit(2 * target, now_s=0.010, delay_s=0.005)
    late = channel.advance(0.020)
    assert late.scheduled_at_s.item() == 0.015
    assert late.applied_at_s.item() == 0.020
    # A submission snapshot does not change when its slot is reused.
    assert receipt.issued_at_s.item() == 0.0
    assert receipt.scheduled_at_s.item() == 0.005


def test_per_environment_lag_and_clocks(device):
    channel = DelayedCommandChannel(torch.zeros(3, 2, device=device), capacity=2)
    issued = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
    delay = torch.tensor([0.0, 0.005, 0.025], dtype=torch.float64, device=device)
    target = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32, device=device)
    channel.submit(target, now_s=issued, delay_s=delay)
    first = channel.advance(issued + 0.01)
    torch.testing.assert_close(first.applied_target[:2], target[:2])
    assert (first.applied_target[2] == 0).all()
    torch.testing.assert_close(
        first.pending_count, torch.tensor([0, 0, 1], device=device)
    )
    last = channel.advance(issued + delay.clamp_min(0.01))
    torch.testing.assert_close(last.applied_target, target)
    assert (last.applied_sequence == 0).all()
    assert last.now_s.dtype == torch.float64
    assert all(getattr(last, field.name).device == device for field in fields(last))


def test_half_controller_period_arrival_waits_for_controller_poll(device):
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), capacity=1)
    channel.submit(torch.ones(1, 1, device=device), now_s=0.0, delay_s=0.0025)
    assert channel.advance(0.0).applied_sequence.item() == -1
    # A 5 ms controller only samples at its next tick, despite arrival at 2.5 ms.
    state = channel.advance(0.005)
    assert state.applied_sequence.item() == 0
    assert state.scheduled_at_s.item() == 0.0025
    assert state.applied_at_s.item() == 0.005


def test_out_of_order_arrival_does_not_roll_back_newer_target(device):
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), capacity=3)
    target = torch.ones(1, 1, device=device)
    old = channel.submit(target, now_s=0.0, delay_s=0.03)
    new = channel.submit(2 * target, now_s=0.01, delay_s=0.005)
    state = channel.advance(0.015)
    assert state.applied_target.item() == 2
    torch.testing.assert_close(state.applied_sequence, new.sequence)
    assert state.pending_count.item() == 1  # Old packet is still in flight.
    state = channel.advance(0.03)
    assert state.applied_target.item() == 2
    assert state.applied_at_s.item() == 0.015
    assert state.applied_sequence.item() > old.sequence.item()
    assert state.pending_count.item() == 0
    assert state.superseded_count.item() == 1


@pytest.mark.parametrize("delay", [0.03, 0.04])
def test_latest_sequence_wins_all_due_even_if_older_arrived_last(device, delay):
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), capacity=3)
    target = torch.ones(1, 1, device=device)
    channel.submit(target, now_s=0.0, delay_s=delay)
    channel.submit(2 * target, now_s=0.01, delay_s=0.02)
    receipt = channel.submit(3 * target, now_s=0.02, delay_s=0.01)
    state = channel.advance(0.05)
    torch.testing.assert_close(state.applied_sequence, receipt.sequence)
    assert state.applied_target.item() == 3
    assert state.superseded_count.item() == 2
    assert state.pending_count.item() == 0
    assert state.applied_at_s.item() == 0.05
    assert channel.advance(0.06).superseded_count.item() == 2


def test_independent_slot_reuse_after_different_arrivals(device):
    channel = DelayedCommandChannel(torch.zeros(2, 1, device=device), capacity=2)
    target = torch.ones(2, 1, device=device)
    lag = torch.tensor([0.03, 0.0], dtype=torch.float64, device=device)
    channel.submit(target, now_s=0.0, delay_s=lag)
    channel.advance(0.0)
    channel.submit(2 * target, now_s=0.01, delay_s=lag.flip(0))
    state = channel.advance(0.01)
    torch.testing.assert_close(
        state.applied_target, torch.tensor([[2.0], [1.0]], device=device)
    )
    channel.submit(3 * target, now_s=0.02, delay_s=0.0)
    state = channel.advance(0.02)
    torch.testing.assert_close(state.applied_target, 3 * target)
    assert (state.pending_count == 1).all()
    state = channel.advance(0.05)
    torch.testing.assert_close(state.applied_target, 3 * target)
    assert (state.superseded_count == 1).all()


def test_partial_reset_clears_old_episode_but_preserves_other_environment(device):
    channel = DelayedCommandChannel(torch.zeros(2, 2, device=device), capacity=2)
    target = torch.ones(2, 2, device=device)
    channel.submit(target, now_s=0.0, delay_s=0.0)
    channel.advance(0.0)
    old = channel.submit(2 * target, now_s=0.01, delay_s=0.09)
    mask = torch.tensor([True, False], device=device)
    reset = channel.reset(mask, -target, now_s=0.02)
    torch.testing.assert_close(
        reset.applied_target, torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device)
    )
    assert reset.applied_sequence[0].item() == -1
    assert torch.isnan(reset.issued_at_s[0])
    assert torch.isnan(reset.scheduled_at_s[0])
    assert reset.applied_at_s[0].item() == 0.02
    assert reset.now_s[1].item() == 0.01
    assert reset.pending_count.tolist() == [0, 1]
    later = channel.advance(0.1)
    torch.testing.assert_close(
        later.applied_target, torch.tensor([[-1.0, -1.0], [2.0, 2.0]], device=device)
    )
    new = channel.submit(3 * target, now_s=0.1, delay_s=0.0)
    assert (new.sequence > old.sequence).all()
    torch.testing.assert_close(channel.advance(0.1).applied_target, 3 * target)


def test_overflow_is_atomic_even_if_only_one_environment_is_full(device):
    channel = DelayedCommandChannel(torch.zeros(2, 1, device=device), capacity=1)
    target = torch.ones(2, 1, device=device)
    delay = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    first = channel.submit(target, now_s=0.0, delay_s=delay)
    channel.advance(0.0)
    before = channel.snapshot()
    with pytest.raises(BufferError, match="entire batch rejected"):
        channel.submit(2 * target, now_s=0.5, delay_s=0.0)
    assert_snapshot_equal(channel.snapshot(), before)
    channel.advance(0.25)  # Failed submission must not advance the clock.
    torch.testing.assert_close(channel.advance(1.0).applied_target, target)
    retry = channel.submit(2 * target, now_s=1.0, delay_s=0.0)
    torch.testing.assert_close(retry.sequence, first.sequence + 1)
    torch.testing.assert_close(channel.advance(1.0).applied_target, 2 * target)


def test_due_packets_are_not_implicitly_drained_by_submit(device):
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), capacity=1)
    target = torch.ones(1, 1, device=device)
    channel.submit(target, now_s=0.0, delay_s=0.0)
    with pytest.raises(BufferError):
        channel.submit(2 * target, now_s=0.0, delay_s=0.0)
    assert channel.snapshot().applied_sequence.item() == -1
    channel.advance(0.0)
    channel.submit(2 * target, now_s=0.0, delay_s=0.0)
    assert channel.advance(0.0).applied_target.item() == 2


@pytest.mark.parametrize("operation", ["submit", "advance", "reset"])
def test_rejects_backwards_time_without_mutation(device, operation):
    target = torch.ones(2, 1, device=device)
    channel = DelayedCommandChannel(target, capacity=2, now_s=1.0)
    before = channel.snapshot()
    backwards = torch.tensor([1.1, 0.9], dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="nondecreasing"):
        if operation == "submit":
            channel.submit(2 * target, now_s=backwards, delay_s=0.0)
        elif operation == "advance":
            channel.advance(backwards)
        else:
            channel.reset(torch.ones(2, dtype=torch.bool, device=device), target, now_s=backwards)
    assert_snapshot_equal(channel.snapshot(), before)


def test_reset_monotonicity_only_checks_selected_rows_and_empty_reset_is_noop(device):
    target = torch.ones(2, 1, device=device)
    channel = DelayedCommandChannel(target, capacity=1, now_s=1.0)
    times = torch.tensor([2.0, 0.0], dtype=torch.float64, device=device)
    channel.reset(torch.tensor([True, False], device=device), 2 * target, now_s=times)
    state = channel.snapshot()
    torch.testing.assert_close(
        state.now_s, torch.tensor([2.0, 1.0], dtype=torch.float64, device=device)
    )
    reset = channel.reset(torch.zeros(2, dtype=torch.bool, device=device), target, now_s=0.0)
    assert_snapshot_equal(reset, state)


def test_owned_snapshots_inputs_and_no_autograd_alias(device):
    initial = torch.zeros(2, 1, device=device, requires_grad=True)
    channel = DelayedCommandChannel(initial, capacity=2)
    with torch.no_grad():
        initial.fill_(10)
    assert (channel.snapshot().applied_target == 0).all()
    target = torch.ones(2, 1, device=device, requires_grad=True)
    times = torch.zeros(2, dtype=torch.float64, device=device)
    delay = torch.full_like(times, 0.01)
    receipt = channel.submit(target, now_s=times, delay_s=delay)
    with torch.no_grad():
        target.fill_(20)
    times.fill_(30)
    delay.fill_(40)
    for field in fields(receipt):
        getattr(receipt, field.name).fill_(50)
    state = channel.advance(0.01)
    assert (state.applied_target == 1).all()
    assert not state.applied_target.requires_grad
    for field in fields(state):
        getattr(state, field.name).fill_(60)
    held = channel.advance(0.01)
    assert (held.applied_target == 1).all()
    assert (held.applied_sequence == 0).all()
    assert (held.pending_count == 0).all()
    assert (held.superseded_count == 0).all()
    assert (held.issued_at_s == 0.0).all()
    assert (held.applied_at_s == 0.01).all()


def test_float64_uptime_retains_submillisecond_schedule(device):
    uptime = 1_000_000_000.0
    channel = DelayedCommandChannel(torch.zeros(1, 1, device=device), 1, now_s=uptime)
    channel.submit(torch.ones(1, 1, device=device), now_s=uptime, delay_s=0.0005)
    assert channel.advance(uptime + 0.0004).applied_sequence.item() == -1
    state = channel.advance(uptime + 0.0005)
    assert state.applied_sequence.item() == 0
    assert state.scheduled_at_s.item() > uptime


@pytest.mark.parametrize("delay", [-0.1, math.nan, math.inf])
def test_invalid_delay_is_atomic(device, delay):
    target = torch.ones(2, 1, device=device)
    channel = DelayedCommandChannel(target, capacity=1)
    before = channel.snapshot()
    with pytest.raises(ValueError):
        channel.submit(target, now_s=0.1, delay_s=delay)
    assert_snapshot_equal(channel.snapshot(), before)


def test_validation_and_nonfinite_scheduled_time(device):
    target = torch.ones(2, 1, device=device)
    for capacity in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="capacity"):
            DelayedCommandChannel(target, capacity)
    with pytest.raises(ValueError, match="floating tensor"):
        DelayedCommandChannel(target.to(torch.int64), 1)
    channel = DelayedCommandChannel(target, 2)
    before = channel.snapshot()
    for invalid_target in (target.flatten(), target.to(torch.float64), target * math.nan):
        with pytest.raises(ValueError, match="target"):
            channel.submit(invalid_target, now_s=0.0, delay_s=0.0)
    for invalid_time in (math.inf, math.nan, torch.zeros(3, device=device)):
        with pytest.raises(ValueError, match="now_s"):
            channel.advance(invalid_time)
    with pytest.raises(ValueError, match="scheduled times finite"):
        channel.submit(target, now_s=1e308, delay_s=1e308)
    with pytest.raises(ValueError, match="mask"):
        channel.reset(torch.ones(2, device=device), target, now_s=0.0)
    assert_snapshot_equal(channel.snapshot(), before)


@pytest.mark.parametrize(
    "periods, expected",
    [((0.01, 0.005, 0.005), (2, 1, 2)),
     ((0.02, 0.002, 0.001), (20, 2, 10)),
     ((0.03, 0.01, 0.005), (6, 2, 3))],
)
def test_timing_profile_integer_substeps(periods, expected):
    profile = TimingProfile(*periods)
    assert (profile.policy_substeps, profile.controller_substeps,
            profile.controllers_per_policy) == expected


@pytest.mark.parametrize(
    "periods",
    [(0.01, 0.001, 0.005),  # Repeating a controller on stale 5 ms state is not 1 kHz physics.
     (0.01, 0.003, 0.002),
     (0.01, 0.004, 0.002),
     (0.001, 0.005, 0.005),
     (0.0, 0.005, 0.005),
     (math.nan, 0.005, 0.005),
     (0.01, math.inf, 0.005),
     (0.01, 0.005, -0.005),
     (True, 0.005, 0.005)],
)
def test_timing_profile_rejects_unrepresentable_clocks(periods):
    with pytest.raises(ValueError):
        TimingProfile(*periods)

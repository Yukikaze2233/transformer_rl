"""Request protocol checks from deployable estimates, not simulator phases."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.tasks.protocol import (
    CompletedEstimate, RequestEnd, RequestProtocol, RequestProtocolConfig, RequestResponse,
)


def tick(protocol, now, *, request=-1, height=0.12, cancel=-1, contact=1.0,
         velocity=0.0, tilt=0.0, angular_speed=0.0, estimate_time=None,
         valid=True, allowed=True, abort=False):
    count = protocol.num_envs

    def vector(value, dtype=torch.float32):
        tensor = torch.as_tensor(value, dtype=dtype)
        return tensor.expand(count).clone() if tensor.ndim == 0 else tensor.clone()

    time = vector(now, torch.float64)
    contacts = torch.as_tensor(contact, dtype=torch.float32)
    if contacts.ndim == 0:
        contacts = contacts.expand(count, 2).clone()
    estimate = CompletedEstimate(vector(velocity), contacts, vector(tilt), vector(angular_speed),
                                 time - 0.01 if estimate_time is None else vector(estimate_time, torch.float64),
                                 vector(valid, torch.bool))
    return protocol.step(now_s=time, estimate=estimate, request_id=vector(request, torch.long),
                         requested_height_m=vector(height), cancel_id=vector(cancel, torch.long),
                         allow_new=vector(allowed, torch.bool), safety_abort=vector(abort, torch.bool))


def flight(protocol):
    first = tick(protocol, 0.0, request=1)
    assert first.response.item() == RequestResponse.ACCEPTED
    tick(protocol, 0.01, contact=0)
    tick(protocol, 0.02, contact=0)
    assert protocol.flight_seen.item()


def assert_state_equal(left, right):
    assert left['configuration'] == right['configuration']
    for name in left['tensors']:
        assert torch.equal(left['tensors'][name], right['tensors'][name]), name


def test_duplicate_request_preserves_height_and_clock_without_restarting():
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=10, height=0.12)
    repeated = tick(protocol, 0.01, request=10, height=0.30)
    assert repeated.response.item() == RequestResponse.REPLAY
    assert repeated.active_id.item() == 10
    torch.testing.assert_close(repeated.height_m, torch.tensor([0.12]))
    torch.testing.assert_close(repeated.clock_s, torch.tensor([0.01]))


def test_busy_rejection_consumes_id_and_does_not_replace_current_skill():
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=10)
    busy = tick(protocol, 0.01, request=11, height=0.25)
    assert busy.response.item() == RequestResponse.BUSY
    assert busy.active_id.item() == 10
    tick(protocol, 0.02, abort=True)
    retry = tick(protocol, 0.03, request=11)
    assert retry.response.item() == RequestResponse.REPLAY
    assert not retry.jump_mode.item()
    assert tick(protocol, 0.04, request=12).response.item() == RequestResponse.ACCEPTED


def test_pre_takeoff_cancel_consumes_request_without_success_event():
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=1)
    end = tick(protocol, 0.01, cancel=1)
    assert not end.jump_mode.item()
    assert end.finished_id.item() == 1
    assert end.end_reason.item() == RequestEnd.CANCELLED
    assert tick(protocol, 0.02, request=1).response.item() == RequestResponse.REPLAY


def test_inflight_cancel_retains_landing_task_until_continuously_stable():
    protocol = RequestProtocol(1)
    flight(protocol)
    cancelled = tick(protocol, 0.03, cancel=1, contact=0)
    assert cancelled.jump_mode.item() and cancelled.active_id.item() == 1
    assert cancelled.end_reason.item() == RequestEnd.NONE
    for index in range(4, 204):
        result = tick(protocol, index * 0.01, contact=1)
        assert result.jump_mode.item()
    result = tick(protocol, 2.04, contact=1)
    assert not result.jump_mode.item()
    assert result.end_reason.item() == RequestEnd.CANCELLED


def test_estimated_landing_completion_is_emitted_once_after_two_seconds():
    protocol = RequestProtocol(1)
    flight(protocol)
    for index in range(3, 203):
        result = tick(protocol, index * 0.01, contact=1)
        assert result.end_reason.item() == RequestEnd.NONE
    complete = tick(protocol, 2.03, contact=1)
    assert complete.finished_id.item() == 1
    assert complete.end_reason.item() == RequestEnd.ESTIMATED_COMPLETE
    assert not complete.jump_mode.item()
    after = tick(protocol, 2.04, request=1)
    assert after.response.item() == RequestResponse.REPLAY
    assert after.finished_id.item() == -1 and after.end_reason.item() == RequestEnd.NONE
    assert tick(protocol, 2.05, request=2).response.item() == RequestResponse.ACCEPTED


def test_ground_contact_without_observed_flight_does_not_complete_a_jump():
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=1)
    for index in range(1, 600):
        result = tick(protocol, index * 0.01)
        assert result.end_reason.item() == RequestEnd.NONE
    result = tick(protocol, 6.0)
    assert result.end_reason.item() == RequestEnd.TIMEOUT
    assert result.safety_required.item() and not result.jump_mode.item()


def test_unobserved_time_gap_does_not_count_as_continuous_stable_landing():
    protocol = RequestProtocol(1)
    flight(protocol)
    for index in range(3, 51):
        tick(protocol, index * 0.01)
    gap = tick(protocol, 1.5)
    assert gap.safety_required.item()
    for index in range(151, 351):
        result = tick(protocol, index * 0.01)
        assert result.end_reason.item() == RequestEnd.NONE
    result = tick(protocol, 3.51)
    assert result.end_reason.item() == RequestEnd.ESTIMATED_COMPLETE


def test_repeated_clock_does_not_advance_detectors_or_mutate_state():
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=1)
    tick(protocol, 0.01, contact=0)
    state = protocol.state_dict()
    repeated = tick(protocol, 0.01, contact=0)
    assert repeated.response.item() == RequestResponse.INVALID_CLOCK
    assert repeated.safety_required.item()
    assert_state_equal(protocol.state_dict(), state)
    tick(protocol, 0.02, contact=0)
    assert protocol.flight_seen.item()


@pytest.mark.parametrize('estimate_time', [0.0, 0.01, -1.0, float('nan')])
def test_same_tick_future_stale_or_nonfinite_estimates_cannot_admit_requests(estimate_time):
    protocol = RequestProtocol(1)
    result = tick(protocol, 0.0, request=1, estimate_time=estimate_time)
    assert result.response.item() == RequestResponse.NOT_READY
    assert not result.jump_mode.item()


@pytest.mark.parametrize('contact,velocity,tilt,speed', [
    (0.5, 0.0, 0.0, 0.0), (1.0, 0.3, 0.0, 0.0), (1.0, 0.0, 0.4, 0.0),
    (1.0, 0.0, 0.0, 1.0), (float('nan'), 0.0, 0.0, 0.0),
])
def test_unstable_or_unknown_motion_is_not_ready(contact, velocity, tilt, speed):
    protocol = RequestProtocol(1)
    result = tick(protocol, 0, request=1, contact=contact, velocity=velocity, tilt=tilt, angular_speed=speed)
    assert result.response.item() == RequestResponse.NOT_READY
    assert not result.jump_mode.item()


def test_cancelled_on_arrival_does_not_briefly_start_a_skill():
    protocol = RequestProtocol(1)
    result = tick(protocol, 0, request=1, cancel=1)
    assert result.response.item() == RequestResponse.CANCELLED
    assert not result.jump_mode.item()
    assert tick(protocol, 0.01, request=1).response.item() == RequestResponse.REPLAY


@pytest.mark.parametrize('height', [0.0, 0.31, float('nan')])
def test_invalid_height_is_rejected_instead_of_silently_clipped(height):
    protocol = RequestProtocol(1)
    result = tick(protocol, 0, request=1, height=height)
    assert result.response.item() == RequestResponse.INVALID
    assert not result.jump_mode.item()
    assert protocol.max_seen_id.item() == 1


def test_safety_abort_and_physical_reset_preserve_request_deduplication():
    protocol = RequestProtocol(2)
    tick(protocol, 0, request=[1, 7])
    aborted = tick(protocol, 0.01, abort=[True, False])
    assert aborted.end_reason.tolist() == [RequestEnd.SAFETY_ABORT, RequestEnd.NONE]
    reset_ids = protocol.reset(torch.tensor([False, True]), torch.tensor([0.02, 0.02], dtype=torch.float64))
    assert reset_ids.tolist() == [-1, 7]
    replay = tick(protocol, 0.02, request=[1, 7])
    assert replay.response.tolist() == [RequestResponse.REPLAY, RequestResponse.REPLAY]
    assert not replay.jump_mode.any()


def test_cold_restore_aborts_in_progress_physics_and_keeps_consumed_ids():
    protocol = RequestProtocol(1)
    flight(protocol)
    state = protocol.state_dict()
    recovered = RequestProtocol(1)
    aborted = recovered.restore_after_physical_reset(state, torch.tensor([0.0], dtype=torch.float64))
    assert aborted.item() == 1
    replay = tick(recovered, 0.0, request=1)
    assert replay.response.item() == RequestResponse.REPLAY and not replay.jump_mode.item()
    accepted = tick(recovered, 0.01, request=2)
    assert accepted.response.item() == RequestResponse.ACCEPTED
    assert accepted.clock_s.item() == 0
    state['tensors']['max_seen_id'].fill_(100)
    assert recovered.max_seen_id.item() == 2


@pytest.mark.parametrize('corruption', ['id', 'phase', 'config'])
def test_corrupt_restore_is_rejected_before_state_is_changed(corruption):
    protocol = RequestProtocol(1)
    tick(protocol, 0, request=1)
    original = protocol.state_dict()
    bad = deepcopy(original)
    if corruption == 'id':
        bad['tensors']['active_id'].fill_(2)
    elif corruption == 'phase':
        bad['tensors']['height_m'].fill_(float('nan'))
    else:
        bad['configuration']['stable_landing_s'] = 0.1
    with pytest.raises(ValueError):
        protocol.restore_after_physical_reset(bad, torch.tensor([1.0], dtype=torch.float64))
    assert_state_equal(protocol.state_dict(), original)


def test_request_identity_does_not_change_actor_visible_task_semantics():
    left, right = RequestProtocol(1), RequestProtocol(1)
    a, b = tick(left, 0, request=10), tick(right, 0, request=1000)
    for name in ('jump_mode', 'height_m', 'clock_s'):
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize('changes', [
    {'stable_landing_s': 0}, {'timeout_s': 1}, {'flight_confirm_ticks': True},
    {'contact_air_probability': 0.9}, {'height_max_m': 0.01}, {'estimate_max_age_s': 0.001},
])
def test_invalid_protocol_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(RequestProtocolConfig(), **changes)

"""Deployable stair request IDs and force-free deployment FSM."""
import torch
import pytest

from transformer_rl.formal.lab_env.course_tasks import StairRequestConfig, StairRequestProtocol
from transformer_rl.formal.lab_env.skill_fsm import StairFSM
from transformer_rl.formal.tasks.protocol import RequestResponse


def test_stair_request_is_one_shot_busy_cancelled_and_timed_out():
    protocol = StairRequestProtocol(1)
    t = torch.zeros(1, dtype=torch.float64)
    first = protocol.step(now_s=t, request_id=torch.tensor([0]), height_m=torch.tensor([.2]),
                           cancel_id=torch.tensor([-1]), jump_active=torch.tensor([False]))
    assert first.response.item() == int(RequestResponse.ACCEPTED)
    assert first.active_id.item() == 0
    replay = protocol.step(now_s=torch.full((1,), .01, dtype=torch.float64),
                            request_id=torch.tensor([0]), height_m=torch.tensor([.2]),
                            cancel_id=torch.tensor([-1]), jump_active=torch.tensor([False]))
    assert replay.response.item() == int(RequestResponse.REPLAY)
    busy = protocol.step(now_s=torch.full((1,), .02, dtype=torch.float64),
                          request_id=torch.tensor([1]), height_m=torch.tensor([.2]),
                          cancel_id=torch.tensor([-1]), jump_active=torch.tensor([False]))
    assert busy.response.item() == int(RequestResponse.BUSY)
    cancelled = protocol.step(now_s=torch.full((1,), .03, dtype=torch.float64),
                               request_id=torch.tensor([-1]), height_m=torch.tensor([0.]),
                               cancel_id=torch.tensor([0]), jump_active=torch.tensor([False]))
    assert cancelled.finished_id.item() == 0 and not cancelled.active.item()
    accepted = protocol.step(now_s=torch.full((1,), .04, dtype=torch.float64),
                              request_id=torch.tensor([2]), height_m=torch.tensor([.2]),
                              cancel_id=torch.tensor([-1]), jump_active=torch.tensor([False]))
    assert accepted.response.item() == int(RequestResponse.ACCEPTED)
    timeout = protocol.step(now_s=torch.full((1,), 6.1, dtype=torch.float64),
                            request_id=torch.tensor([-1]), height_m=torch.tensor([0.]),
                            cancel_id=torch.tensor([-1]), jump_active=torch.tensor([False]))
    assert timeout.finished_id.item() == 2 and not timeout.active.item()


def test_stair_request_is_busy_while_jump_is_active():
    protocol = StairRequestProtocol(1)
    result = protocol.step(now_s=torch.zeros(1, dtype=torch.float64),
                           request_id=torch.tensor([0]), height_m=torch.tensor([.2]),
                           cancel_id=torch.tensor([-1]), jump_active=torch.tensor([True]))
    assert result.response.item() == int(RequestResponse.BUSY)
    assert not result.active.item()


def test_deployment_stair_fsm_uses_probability_and_never_claims_success():
    fsm = StairFSM(1)
    out = fsm.deployment_step(
        step_height_m=torch.tensor([.20]), wheel_contact_probability=torch.tensor([[1., 1.]]),
        tilt_rad=torch.zeros(1), command_height_m=torch.tensor([.32]),
        manual_request=torch.tensor([True]), manual_height_m=torch.tensor([.20]))
    assert out.mode.item() == 1  # CommandMode.STAIR
    assert not out.success.item()
    for _ in range(20):
        out = fsm.deployment_step(
            step_height_m=torch.tensor([.20]), wheel_contact_probability=torch.tensor([[1., 1.]]),
            tilt_rad=torch.zeros(1), command_height_m=torch.tensor([.32]),
            manual_request=torch.tensor([False]), manual_height_m=torch.tensor([0.]))
    assert not out.success.item()


def test_cold_restore_keeps_consumed_stair_ids_but_aborts_active_command():
    protocol = StairRequestProtocol(2)
    protocol.step(now_s=torch.zeros(2, dtype=torch.float64),
                  request_id=torch.tensor([17, 29]), height_m=torch.tensor([.2, .1]),
                  cancel_id=torch.full((2,), -1), jump_active=torch.zeros(2, dtype=torch.bool))
    saved = protocol.state_dict()
    cold = StairRequestProtocol(2)
    aborted = cold.restore_after_physical_reset(saved, torch.zeros(2, dtype=torch.float64))
    assert aborted.tolist() == [17, 29]
    assert cold.active_id.eq(-1).all()
    replay = cold.step(now_s=torch.zeros(2, dtype=torch.float64),
                       request_id=torch.tensor([17, 29]), height_m=torch.tensor([.2, .1]),
                       cancel_id=torch.full((2,), -1), jump_active=torch.zeros(2, dtype=torch.bool))
    assert replay.response.eq(int(RequestResponse.REPLAY)).all()


def test_invalid_stair_clock_and_height_are_rejected_without_activation():
    protocol = StairRequestProtocol(2)
    invalid = protocol.step(now_s=torch.tensor([float('nan'), 0.], dtype=torch.float64),
                            request_id=torch.tensor([0, 0]), height_m=torch.tensor([.1, .25]),
                            cancel_id=torch.full((2,), -1), jump_active=torch.zeros(2, dtype=torch.bool))
    assert invalid.response.tolist() == [int(RequestResponse.INVALID_CLOCK), int(RequestResponse.INVALID)]
    assert not invalid.active.any()
    with pytest.raises(ValueError):
        StairRequestConfig(timeout_s=float('nan'))


def test_command_height_changes_cannot_fabricate_stair_drop_or_success():
    fsm = StairFSM(1)
    for height in (.35, .17):
        result = fsm.deployment_step(
            step_height_m=torch.tensor([.2]), wheel_contact_probability=torch.ones(1, 2),
            tilt_rad=torch.zeros(1), command_height_m=torch.tensor([height]),
            manual_request=torch.tensor([height == .35]), manual_height_m=torch.tensor([.2]))
        assert not result.failure.item()
        assert not result.success.item()

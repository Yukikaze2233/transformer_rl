"""lab_env 纯契约单测；不需要 Isaac，可在 CPU 上运行。"""
import pytest
import torch

from transformer_rl.formal.lab_env.actuator import WheeledControlCore
from transformer_rl.formal.lab_env.contract import (ACTOR_DIM, FlatTaskSpec,
                                                    HistoryBuffer, flat_reward,
                                                    flat_termination)


def test_history_reset_and_append_keep_five_frames():
    history = HistoryBuffer(2, length=5, frame_dim=ACTOR_DIM)
    frame = torch.arange(2 * ACTOR_DIM, dtype=torch.float32).reshape(2, ACTOR_DIM)
    mask = torch.tensor([True, False])
    history.reset(mask, frame)
    assert torch.equal(history.frames[0], frame[0][None].expand(5, -1))
    assert torch.equal(history.frames[1], torch.zeros(5, ACTOR_DIM))

    next_frame = frame + 1
    reset = torch.tensor([False, True])
    history.append(next_frame, reset)
    assert torch.equal(history.frames[0, -1], next_frame[0])
    assert torch.equal(history.frames[1], next_frame[1][None].expand(5, -1))


def test_flat_reward_and_termination_are_finite_and_bounded():
    spec = FlatTaskSpec()
    command = torch.tensor([[0., 0., spec.nominal_height_m]])
    velocity = torch.zeros(1, 4)
    gravity = torch.tensor([[0., 0., -1.]])
    height = torch.tensor([spec.nominal_height_m])
    action = torch.zeros(1, 6)
    effort = torch.zeros(1, 6)
    reward, parts = flat_reward(command=command, velocity_body=velocity, height_m=height,
                                gravity_body=gravity, action=action, previous_action=action,
                                effort=effort, spec=spec, dt=0.01)
    assert reward.shape == (1,) and torch.isfinite(reward).all()
    assert set(parts) == {"forward", "yaw", "height", "upright", "cost"}
    terminated, tilt, contact = flat_termination(
        height_m=height, gravity_body=gravity, base_contact_force_n=torch.zeros(1), spec=spec)
    assert terminated.shape == (1,) and not terminated.any()
    assert tilt.shape == (1,) and contact.shape == (1,)


def test_control_core_matches_existing_output_contract():
    core = WheeledControlCore((0., 0., 0., 0., 0., 0.), 2)
    action = torch.zeros(2, 6)
    q = torch.zeros(2, 6)
    dq = torch.zeros(2, 6)
    now = torch.zeros(2, dtype=torch.float64)
    targets, feedback = core.effort(action, q, dq, now, now)
    assert targets.admitted_leg_position_rad.shape == (2, 4)
    assert targets.wheel_velocity_rad_s.shape == (2, 2)
    assert feedback.capped_effort.shape == (2, 6)
    assert (feedback.capped_effort == 0).all()


def test_control_core_rejects_future_feedback_clock():
    core = WheeledControlCore((0., 0., 0., 0., 0., 0.), 1)
    action = torch.zeros(1, 6)
    q = torch.zeros(1, 6)
    dq = torch.zeros(1, 6)
    issued = torch.ones(1, dtype=torch.float64)
    computed = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="future"):
        core.effort(action, q, dq, issued, computed)

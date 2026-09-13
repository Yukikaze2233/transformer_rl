"""Hand-calculated GAE and ownership checks, without an environment."""
from dataclasses import fields, replace

import pytest
import torch

from transformer_rl.storage import PPOBatch, RolloutBuffer
from transformer_rl.types import HistoryBatch


def make_step(size=4, time=0):
    return dict(
        history=HistoryBatch(
            frames=torch.arange(size * 6, dtype=torch.float32).reshape(size, 2, 3) + time,
            times=torch.full((size, 2), 1_000_000.0 + time * 0.01, dtype=torch.float64),
            valid=torch.tensor([[False, True]]).expand(size, -1).clone(),
            command=torch.full((size, 2), float(time)),
            now=torch.full((size,), 1_000_000.0 + time * 0.01, dtype=torch.float64),
        ),
        critic=torch.full((size, 3), float(time)),
        raw_action=torch.full((size, 2), 2.0 + time),
        issued_action=torch.ones(size, 2),
        old_log_prob=torch.full((size,), -0.5),
        old_mean=torch.zeros(size, 2),
        old_std=torch.ones(size, 2),
        old_value=torch.arange(1, size + 1, dtype=torch.float32),
        reward=torch.ones(size),
        next_value=torch.full((size,), 2.0),
        terminated=torch.zeros(size, dtype=torch.bool),
        truncated=torch.zeros(size, dtype=torch.bool),
    )


def assert_batches_equal(actual, expected):
    for field in fields(PPOBatch):
        if field.name == "history":
            for history_field in fields(HistoryBatch):
                torch.testing.assert_close(
                    getattr(actual.history, history_field.name),
                    getattr(expected.history, history_field.name),
                )
        else:
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name))


def test_hand_calculated_gae_before_time_major_flattening():
    buffer = RolloutBuffer(8)
    for time in range(3):
        step = make_step(time=time)
        step["old_value"] = torch.arange(1.0, 5.0) + 4 * time
        step["reward"] = torch.arange(1.0, 5.0) + time
        step["next_value"] = torch.arange(5.0, 9.0) + 4 * time
        if time == 1:
            step["terminated"] = torch.tensor([True, False, True, False])
            step["truncated"] = torch.tensor([False, True, True, False])
        buffer.add(**step)
    batch = buffer.finish(gamma=0.5, gae_lambda=0.5)
    # t=2 has bootstrap but no following trace. At t=1: terminal, timeout,
    # terminal+timeout, and continuing respectively. Trace multiplier is 1/4.
    expected = torch.tensor([
        [1.75, 3.5, 2.75, 4.875],
        [-3.0, 2.0, -3.0, 3.5],
        [0.5, 1.0, 1.5, 2.0],
    ])
    torch.testing.assert_close(batch.advantages, expected.flatten())
    torch.testing.assert_close(batch.old_value, torch.arange(1.0, 13.0))
    torch.testing.assert_close(batch.returns, expected.flatten() + torch.arange(1.0, 13.0))
    assert batch.advantages.mean().item() != pytest.approx(0.0)
    assert len(buffer) == 3
    assert len(batch) == 12
    assert batch.history.frames.shape == (12, 2, 3)
    assert batch.history.times.dtype == torch.float64
    torch.testing.assert_close(batch.critic[:, 0], torch.arange(3.0).repeat_interleave(4))


@pytest.mark.parametrize("terminated,truncated,expected", [
    (False, False, 4.0), (True, False, -1.0),
    (False, True, 4.0), (True, True, -1.0),
])
def test_single_tail_bootstrap_masks(terminated, truncated, expected):
    step = make_step(size=1)
    step.update(old_value=torch.tensor([3.0]), reward=torch.tensor([2.0]),
                next_value=torch.tensor([10.0]), terminated=torch.tensor([terminated]),
                truncated=torch.tensor([truncated]))
    buffer = RolloutBuffer(4)
    buffer.add(**step)
    batch = buffer.finish(0.5, 1.0)
    torch.testing.assert_close(batch.advantages, torch.tensor([expected]))
    torch.testing.assert_close(batch.returns, torch.tensor([expected + 3.0]))
    assert batch.old_value.shape == batch.advantages.shape == batch.returns.shape == (1,)


def test_next_episode_reward_cannot_leak_across_either_done_flag():
    results = []
    for future_reward in (1.0, 1_000_000.0):
        buffer = RolloutBuffer(3)
        for time in range(3):
            step = make_step(time=time)
            if time == 1:
                step["terminated"] = torch.tensor([True, False, True, False])
                step["truncated"] = torch.tensor([False, True, True, False])
            if time == 2:
                step["reward"].fill_(future_reward)
            buffer.add(**step)
        results.append(buffer.finish(0.9, 0.8).advantages.reshape(3, 4))
    torch.testing.assert_close(results[0][:2, :3], results[1][:2, :3])
    assert results[1][0, 3] > results[0][0, 3]


def test_every_input_and_history_field_is_an_owned_detached_snapshot():
    step = make_step()
    for field in fields(HistoryBatch):
        tensor = getattr(step["history"], field.name)
        if tensor.is_floating_point():
            tensor.requires_grad_()
    for tensor in step.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            tensor.requires_grad_()
    buffer = RolloutBuffer(2)
    buffer.add(**step)
    expected = buffer.finish(0.5, 0.5)
    with torch.no_grad():
        for field in fields(HistoryBatch):
            tensor = getattr(step["history"], field.name)
            tensor.logical_not_() if tensor.dtype == torch.bool else tensor.fill_(-100.0)
        for tensor in step.values():
            if isinstance(tensor, torch.Tensor):
                tensor.logical_not_() if tensor.dtype == torch.bool else tensor.fill_(-100.0)
    actual = buffer.finish(0.5, 0.5)
    assert_batches_equal(actual, expected)
    for field in fields(PPOBatch):
        tensors = ([getattr(actual.history, f.name) for f in fields(HistoryBatch)]
                   if field.name == "history" else [getattr(actual, field.name)])
        for tensor in tensors:
            assert not tensor.requires_grad
            assert tensor.grad_fn is None
            tensor.fill_(0)
    assert_batches_equal(buffer.finish(0.5, 0.5), expected)


def test_partial_final_batch_preserves_raw_action_and_index_alignment():
    buffer = RolloutBuffer(10)
    buffer.add(**make_step(size=3, time=0))
    buffer.add(**make_step(size=3, time=1))
    batch = buffer.finish(0.99, 0.95)
    assert len(batch) == 6
    assert batch.raw_action.shape == batch.issued_action.shape == (6, 2)
    assert (batch.raw_action > batch.issued_action).all()
    for field in ("old_log_prob", "old_value", "advantages", "returns"):
        assert getattr(batch, field).shape == (6,)
    for index in (torch.tensor([5, 0, 3]), slice(1, 4), torch.tensor([True, False] * 3)):
        selected = batch.index(index)
        for field in fields(PPOBatch):
            if field.name == "history":
                for history_field in fields(HistoryBatch):
                    torch.testing.assert_close(
                        getattr(selected.history, history_field.name),
                        getattr(batch.history, history_field.name)[index],
                    )
            else:
                torch.testing.assert_close(getattr(selected, field.name), getattr(batch, field.name)[index])
    for index in (-1, 0, torch.tensor(2)):
        selected = batch.index(index)
        assert len(selected) == 1
        selected.validate()
    with pytest.raises(IndexError):
        batch.index(6)


@pytest.mark.parametrize("field", ["old_log_prob", "old_value", "reward", "next_value", "terminated", "truncated"])
def test_rejects_scalar_columns_instead_of_broadcasting(field):
    step = make_step()
    step[field] = step[field].unsqueeze(-1)
    buffer = RolloutBuffer(2)
    with pytest.raises(ValueError, match=field):
        buffer.add(**step)
    assert len(buffer) == 0


@pytest.mark.parametrize("field", ["reward", "next_value", "old_log_prob", "raw_action", "old_std",
                                 "critic", "issued_action", "old_mean", "old_value"])
def test_nonfinite_input_is_not_silently_masked(field):
    step = make_step()
    step[field].flatten()[0] = float("nan")
    step["terminated"].fill_(True)
    with pytest.raises(FloatingPointError, match=field):
        RolloutBuffer(1).add(**step)


def test_contract_errors_and_capacity():
    for capacity in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="capacity"):
            RolloutBuffer(capacity)
    buffer = RolloutBuffer(1)
    with pytest.raises(ValueError, match="empty"):
        buffer.finish(0.99, 0.95)
    step = make_step()
    with pytest.raises(ValueError, match="old_std"):
        buffer.add(**{**step, "old_std": torch.zeros(4, 2)})
    with pytest.raises(ValueError, match="bool"):
        buffer.add(**{**step, "terminated": torch.zeros(4)})
    with pytest.raises(ValueError, match="history.now"):
        buffer.add(**{**step, "history": replace(step["history"], now=torch.zeros(4, 1))})
    buffer.add(**step)
    with pytest.raises(RuntimeError, match="full"):
        buffer.add(**step)
    for gamma, gae_lambda in ((-0.1, 1.0), (1.0, 1.1), (float("nan"), 0.5)):
        with pytest.raises(ValueError, match="gamma"):
            buffer.finish(gamma, gae_lambda)


def test_temporal_schema_must_remain_constant():
    buffer = RolloutBuffer(4)
    buffer.add(**make_step(size=2))
    with pytest.raises(ValueError, match="constant"):
        buffer.add(**make_step(size=3))
    step = make_step(size=2)
    step["history"] = replace(step["history"], times=step["history"].times.float())
    with pytest.raises(ValueError, match="constant"):
        buffer.add(**step)
    assert len(buffer) == 1


def test_gae_overflow_is_reported():
    step = make_step(size=1)
    step.update(reward=torch.tensor([3e38]), next_value=torch.tensor([3e38]))
    buffer = RolloutBuffer(1)
    buffer.add(**step)
    with pytest.raises(FloatingPointError, match="GAE"):
        buffer.finish(1.0, 1.0)


@pytest.mark.parametrize("sentinel", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_padding_is_owned_and_preserved_through_finish_and_index(sentinel):
    step = make_step(size=2)
    history = step["history"]
    history.frames[~history.valid] = sentinel
    history.times[~history.valid] = sentinel
    expected = history.clone()
    buffer = RolloutBuffer(2)
    buffer.add(**step)
    history.frames[~history.valid] = 0
    history.times[~history.valid] = 0
    batch = buffer.finish(0.99, 0.95)
    batch.validate()
    batch.index(1).validate()
    torch.testing.assert_close(batch.history.frames, expected.frames, equal_nan=True)
    torch.testing.assert_close(batch.history.times, expected.times, equal_nan=True)
    assert torch.isfinite(batch.advantages).all() and torch.isfinite(batch.returns).all()


@pytest.mark.parametrize("field", ["frames", "times", "now", "command"])
@pytest.mark.parametrize("sentinel", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_meaningful_history_is_still_rejected(field, sentinel):
    step = make_step(size=2)
    tensor = getattr(step["history"], field)
    if field in ("frames", "times"):
        tensor[0, -1] = sentinel
    else:
        tensor[0] = sentinel
    with pytest.raises(FloatingPointError, match=f"history.{field}"):
        RolloutBuffer(1).add(**step)


def test_invalid_padding_mask_is_rejected_before_masked_finite_checks():
    step = make_step(size=2)
    step["history"] = replace(step["history"], valid=step["history"].valid.float())
    with pytest.raises(ValueError, match="history.valid must have bool dtype"):
        RolloutBuffer(1).add(**step)

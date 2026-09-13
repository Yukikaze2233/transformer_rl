"""History ownership, event identity, reset and feature-validity contracts."""
from dataclasses import fields, replace

import pytest
import torch

from transformer_rl.config import ModelConfig
from transformer_rl.history import HistoryBuffer, pack_frame
from transformer_rl.types import VectorObservation


def _observation(config, timestamps=(1.0, 2.0), value=1.0, dtype=torch.float32, device="cpu"):
    count = len(timestamps)
    return VectorObservation(
        frame=torch.full((count, config.frame_dim), value, dtype=dtype, device=device),
        timestamp=torch.tensor(timestamps, dtype=torch.float64, device=device),
        command=torch.full((count, config.command_dim), value + 0.1, dtype=dtype, device=device),
        critic=torch.full((count, config.critic_dim), value + 0.2, dtype=dtype, device=device),
    )


def _pack_inputs(config, dtype=torch.float32):
    return {
        "proprio": torch.full((2, config.proprio_dim), 1.0, dtype=dtype),
        "command": torch.full((2, config.command_dim), 2.0, dtype=dtype),
        "previous_issued_action": torch.full((2, config.action_dim), 3.0, dtype=dtype),
        "sensor_age_s": torch.tensor([[0.02, float("nan")], [0.0, 0.1]], dtype=dtype),
        "sensor_age_known": torch.tensor([[True, False], [True, True]]),
        "policy_dt_s": torch.tensor([0.0, 0.01], dtype=dtype),
    }


def _assert_history_equal(actual, expected):
    for field in fields(actual):
        torch.testing.assert_close(
            getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frame_layout_units_unknown_ages_and_owned_storage(dtype):
    config = ModelConfig()
    inputs = _pack_inputs(config, dtype=dtype)
    frame = pack_frame(config, **inputs)
    assert frame.shape == (2, 30)
    assert frame.dtype == dtype
    torch.testing.assert_close(frame[:, :16], inputs["proprio"])
    torch.testing.assert_close(frame[:, 16:19], inputs["command"])
    torch.testing.assert_close(frame[:, 19:25], inputs["previous_issued_action"])
    torch.testing.assert_close(frame[:, 25:27], torch.tensor([[0.02, 0], [0, 0.1]], dtype=dtype))
    torch.testing.assert_close(frame[:, 27:29], inputs["sensor_age_known"].to(dtype))
    torch.testing.assert_close(frame[:, 29], inputs["policy_dt_s"])
    assert frame[0, 26] == frame[1, 25] == 0
    assert frame[0, 28] == 0 and frame[1, 27] == 1
    saved = frame.clone()
    inputs["proprio"].zero_()
    inputs["previous_issued_action"].zero_()
    torch.testing.assert_close(frame, saved)


def test_pack_frame_preserves_gradients_and_accepts_column_interval():
    config = ModelConfig()
    inputs = _pack_inputs(config)
    inputs["policy_dt_s"] = inputs["policy_dt_s"][:, None]
    for tensor in inputs.values():
        if tensor.is_floating_point():
            tensor.requires_grad_()
    frame = pack_frame(config, **inputs)
    frame.sum().backward()
    for name in ("proprio", "command", "previous_issued_action", "policy_dt_s"):
        torch.testing.assert_close(inputs[name].grad, torch.ones_like(inputs[name]))
    torch.testing.assert_close(inputs["sensor_age_s"].grad, inputs["sensor_age_known"].float())


@pytest.mark.parametrize("sentinel", [float("nan"), float("inf"), -100.0])
def test_unknown_age_sentinel_never_masquerades_as_known_zero(sentinel):
    config = ModelConfig()
    inputs = _pack_inputs(config)
    inputs["sensor_age_known"][:] = False
    inputs["sensor_age_s"][:] = sentinel
    frame = pack_frame(config, **inputs)
    assert torch.isfinite(frame).all()
    assert torch.count_nonzero(frame[:, 25:29]) == 0


@pytest.mark.parametrize(
    "name,mutation,error,match",
    [
        ("proprio", lambda x: x[:, :-1], ValueError, "shape"),
        ("proprio", lambda x: x.long(), TypeError, "floating"),
        ("proprio", lambda x: x * float("nan"), ValueError, "finite"),
        ("command", lambda x: x[:, :1], ValueError, "shape"),
        ("command", lambda x: x.double(), TypeError, "dtype"),
        ("command", lambda x: x.to("meta"), ValueError, "device"),
        ("command", lambda x: x * float("inf"), ValueError, "finite"),
        ("previous_issued_action", lambda x: x * float("nan"), ValueError, "finite"),
        ("sensor_age_known", lambda x: x.float(), TypeError, "dtype"),
        ("sensor_age_s", lambda x: torch.full_like(x, -0.1), ValueError, "nonnegative"),
        ("sensor_age_s", lambda x: torch.full_like(x, float("nan")), ValueError, "finite"),
        ("policy_dt_s", lambda x: x[:, None, None], ValueError, "shape"),
        ("policy_dt_s", lambda x: torch.full_like(x, -0.1), ValueError, "nonnegative"),
        ("policy_dt_s", lambda x: x * float("inf"), ValueError, "finite"),
    ],
)
def test_invalid_frame_inputs(name, mutation, error, match):
    config = ModelConfig()
    inputs = _pack_inputs(config)
    inputs[name] = mutation(inputs[name])
    with pytest.raises(error, match=match):
        pack_frame(config, **inputs)


def test_left_padding_rollover_and_per_environment_timestamps():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    empty = buffer.snapshot()
    assert not empty.valid.any()
    first = buffer.append(_observation(config, (0.0, 1000.0), value=1))
    torch.testing.assert_close(first.valid, torch.tensor([[False, False, True]] * 2))
    assert torch.count_nonzero(first.frames[:, :-1]) == 0
    for tick in range(1, 4):
        result = buffer.append(
            _observation(config, (tick * 0.01, 1000 + tick * 0.02), value=tick + 1)
        )
    assert result.valid.all()
    torch.testing.assert_close(
        result.frames[:, :, 0],
        torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        result.times,
        torch.tensor([[0.01, 0.02, 0.03], [1000.02, 1000.04, 1000.06]], dtype=torch.float64),
    )
    torch.testing.assert_close(result.now, result.times[:, -1])
    torch.testing.assert_close(result.command, torch.full((2, 3), 4.1))
    assert first.valid.sum() == 2
    assert first.frames[0, -1, 0] == 1


def test_identical_same_tick_is_idempotent_and_returns_fresh_storage():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    observation = _observation(config)
    first = buffer.append(observation)
    second = buffer.append(observation)
    _assert_history_equal(first, second)
    for field in fields(first):
        assert getattr(first, field.name).data_ptr() != getattr(second, field.name).data_ptr()
    second.frames.fill_(100)
    _assert_history_equal(buffer.append(observation), first)


def test_same_tick_and_advancing_tick_can_coexist_across_environments():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    first_observation = _observation(config, (10, 20), value=1)
    buffer.append(first_observation)
    next_observation = _observation(config, (10, 21), value=2)
    for name in ("frame", "command", "critic"):
        getattr(next_observation, name)[0] = getattr(first_observation, name)[0]
    result = buffer.append(next_observation)
    torch.testing.assert_close(result.valid.sum(-1), torch.tensor([1, 2]))
    torch.testing.assert_close(result.frames[:, -1, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.now, torch.tensor([10, 21], dtype=torch.float64))


@pytest.mark.parametrize("changed_field", ["frame", "command", "critic"])
def test_changed_same_tick_rejects_entire_batch_atomically(changed_field):
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    observation = _observation(config, (10, 20))
    before = buffer.append(observation)
    next_observation = _observation(config, (10, 21))
    getattr(next_observation, changed_field)[0, 0] += 1
    with pytest.raises(ValueError, match="same timestamp"):
        buffer.append(next_observation)
    _assert_history_equal(buffer.snapshot(), before)


def test_timestamp_regression_requires_reset_and_does_not_partially_write():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    before = buffer.append(_observation(config, (10, 20)))
    with pytest.raises(ValueError, match="decrease"):
        buffer.append(_observation(config, (11, 19), value=2))
    _assert_history_equal(buffer.snapshot(), before)


def test_partial_reset_clears_all_storage_only_for_selected_environments():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    buffer.append(_observation(config, (10, 20), value=1))
    before = buffer.append(_observation(config, (11, 21), value=2))
    buffer.reset(torch.tensor([True, False]))
    reset = buffer.snapshot()
    for field in fields(reset):
        tensor = getattr(reset, field.name)
        assert torch.count_nonzero(tensor[0]) == 0
        torch.testing.assert_close(tensor[1], getattr(before, field.name)[1], rtol=0, atol=0)
    resumed = buffer.append(_observation(config, (0, 22), value=3))
    torch.testing.assert_close(resumed.valid.sum(-1), torch.tensor([1, 3]))
    assert torch.count_nonzero(resumed.frames[0, :-1]) == 0
    assert resumed.frames[1, -2, 0] == 2
    buffer.reset(torch.zeros(2, dtype=torch.bool))
    _assert_history_equal(buffer.snapshot(), resumed)
    buffer.reset(torch.ones(2, dtype=torch.bool))
    assert not buffer.snapshot().valid.any()


def test_snapshots_do_not_alias_inputs_buffer_or_their_own_time_fields():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    observation = _observation(config)
    for name in ("frame", "command", "critic"):
        getattr(observation, name).requires_grad_()
    snapshot = buffer.append(observation)
    expected = snapshot.clone()
    assert all(not getattr(snapshot, field.name).requires_grad for field in fields(snapshot))
    with torch.no_grad():
        for field in fields(observation):
            getattr(observation, field.name).zero_()
    _assert_history_equal(snapshot, expected)
    _assert_history_equal(buffer.snapshot(), expected)
    snapshot.now.add_(100)
    torch.testing.assert_close(snapshot.times, expected.times, rtol=0, atol=0)
    for field in fields(snapshot):
        getattr(snapshot, field.name).zero_()
    _assert_history_equal(buffer.snapshot(), expected)
    buffer.append(_observation(config, (3, 4), value=9))
    buffer.reset(torch.ones(2, dtype=torch.bool))
    assert expected.frames[:, -1, 0].eq(1).all()


@pytest.mark.parametrize("length", [1, 4])
def test_first_append_infers_double_features_and_preserves_precise_timestamps(length):
    config = ModelConfig(history_length=length)
    buffer = HistoryBuffer(config, 2, "cpu")
    observation = _observation(config, (1e9, 1e9 + 0.001), dtype=torch.float64)
    first = buffer.append(observation)
    second = buffer.append(
        _observation(config, (1e9 + 0.001, 1e9 + 0.002), value=2, dtype=torch.float64)
    )
    assert second.frames.dtype == second.command.dtype == torch.float64
    assert second.times.dtype == second.now.dtype == torch.float64
    assert (second.now > first.now).all()
    with pytest.raises(TypeError, match="previous appends"):
        buffer.append(_observation(config, (1e9 + 1, 1e9 + 2)))


@pytest.mark.parametrize(
    "field,mutation,error,match",
    [
        ("frame", lambda x: x[:, :-1], ValueError, "shape"),
        ("frame", lambda x: x.long(), TypeError, "floating"),
        ("frame", lambda x: x.to("meta"), ValueError, "device"),
        ("frame", lambda x: x * float("nan"), ValueError, "finite"),
        ("timestamp", lambda x: x[:, None], ValueError, "shape"),
        ("timestamp", lambda x: x.long(), TypeError, "float32 or float64"),
        ("timestamp", lambda x: x * float("inf"), ValueError, "finite"),
        ("command", lambda x: x.double(), TypeError, "floating dtype"),
        ("command", lambda x: x * float("nan"), ValueError, "finite"),
        ("critic", lambda x: x[:, :-1], ValueError, "shape"),
        ("critic", lambda x: x * float("nan"), ValueError, "finite"),
    ],
)
def test_invalid_observations_are_rejected_before_mutation(field, mutation, error, match):
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    before = buffer.append(_observation(config))
    observation = _observation(config, (3, 4))
    bad = replace(observation, **{field: mutation(getattr(observation, field))})
    with pytest.raises(error, match=match):
        buffer.append(bad)
    _assert_history_equal(buffer.snapshot(), before)


@pytest.mark.parametrize(
    "mask,error,match",
    [
        (torch.zeros(2), TypeError, "bool"),
        (torch.zeros(2, 1, dtype=torch.bool), ValueError, "shape"),
        (torch.zeros(2, dtype=torch.bool, device="meta"), ValueError, "device"),
    ],
)
def test_invalid_reset_masks(mask, error, match):
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cpu")
    before = buffer.append(_observation(config))
    with pytest.raises(error, match=match):
        buffer.reset(mask)
    _assert_history_equal(buffer.snapshot(), before)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_environment_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        HistoryBuffer(ModelConfig(), count, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable in CPU verification environment",
)
def test_cuda_history_append_and_partial_reset():
    config = ModelConfig(history_length=3)
    buffer = HistoryBuffer(config, 2, "cuda")
    history = buffer.append(_observation(config, device="cuda"))
    assert all(getattr(history, field.name).is_cuda for field in fields(history))
    buffer.reset(torch.tensor([True, False], device="cuda"))
    torch.testing.assert_close(
        buffer.snapshot().valid.sum(-1), torch.tensor([0, 1], device="cuda")
    )

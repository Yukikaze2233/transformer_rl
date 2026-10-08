"""Prearm graph preparation and fail-stop checks without a device or real graph."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from transformer_rl import frame_runtime
from transformer_rl.frame_runtime import FrameRuntime, run_control_loop


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def advance(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def perf_counter_ns(self):
        return round(self.now * 1e9)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


class Session:
    def __init__(self, *, clock=None, first_seconds=0.0, later_seconds=0.0):
        self.clock = clock
        self.first_seconds = first_seconds
        self.later_seconds = later_seconds
        self.inputs = []
        self.input_references = []
        self.output = np.zeros((1, 2), dtype=np.float32)
        self.reply = None

    def run(self, names, feeds):
        assert names == ["mean"] and set(feeds) == {"frames"}
        frames = feeds["frames"]
        self.input_references.append(frames)
        self.inputs.append(frames.copy())
        if self.clock is not None:
            self.clock.advance(self.first_seconds if len(self.inputs) == 1 else self.later_seconds)
        if self.reply is not None:
            return [self.reply(frames, len(self.inputs))]
        self.output[:] = frames[:, -1, :2]
        return [self.output]


@pytest.fixture
def bundle(tmp_path):
    graph = b"synthetic backend fixture; not an executable policy"
    (tmp_path / "policy.onnx").write_bytes(graph)
    manifest = {
        "format": "transformer_rl.packed_policy",
        "schema_version": 1,
        "control": {
            "observation_schema": "tensor_fixture",
            "policy_dt_s": 0.01,
            "action_bounds": [1.0, 2.0],
            "target_scale": [2.0, 3.0],
            "target_offset": [0.1, 0.2],
        },
        "model": {"policy": {"history_length": 3, "frame_dim": 3, "action_dim": 2}},
        "files": {"policy.onnx": hashlib.sha256(graph).hexdigest()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def install_session(monkeypatch, session):
    def load(path, *, sess_options, providers):
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.intra_op_num_threads == 1
        assert sess_options.inter_op_num_threads == 1
        return session

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=load))


def construct(bundle, **kwargs):
    return FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=0.01, **kwargs)


def assert_cleared(runtime):
    assert runtime.last_time is None
    np.testing.assert_array_equal(runtime.frames, np.zeros((1, 3, 3), dtype=np.float32))


def test_automatic_preparation_has_no_step_or_history_and_real_frame_repeats_first(bundle, monkeypatch):
    session = Session()
    install_session(monkeypatch, session)
    with monkeypatch.context() as patch:
        def forbidden_step(*args, **kwargs):
            pytest.fail("prearm preparation must not create a policy tick or issued action")
        patch.setattr(FrameRuntime, "step", forbidden_step)
        runtime = construct(bundle)

    assert len(session.inputs) == 50
    assert all(value.dtype == np.float32 and value.shape == (1, 3, 3)
               and not value.any() for value in session.inputs)
    assert all(not np.shares_memory(value, runtime.frames) for value in session.input_references)
    assert runtime.preparation["iterations"] == 50
    assert runtime.preparation["duration_ms"] >= 0
    assert isinstance(runtime.preparation["scope"], str) and runtime.preparation["scope"]
    assert_cleared(runtime)

    first = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    result = runtime.step(first, 17.25)
    np.testing.assert_array_equal(session.inputs[-1], np.broadcast_to(first, (1, 3, 3)))
    np.testing.assert_array_equal(result["mean"], first[:2])
    assert not np.shares_memory(result["mean"], session.output)
    runtime.step(first + 0.1, 17.26)
    np.testing.assert_array_equal(result["mean"], first[:2])

    calls_before_reset = len(session.inputs)
    runtime.reset()
    assert_cleared(runtime)
    assert len(session.inputs) == calls_before_reset
    next_first = np.array([-0.75, 0.5, 0.25], dtype=np.float32)
    runtime.step(next_first, 99.0)
    np.testing.assert_array_equal(session.inputs[-1], np.broadcast_to(next_first, (1, 3, 3)))


@pytest.mark.parametrize("iterations", [True, False, 0, -1, 50.0])
def test_preparation_requires_a_positive_integer(bundle, monkeypatch, iterations):
    session = Session()
    install_session(monkeypatch, session)
    with pytest.raises(ValueError):
        construct(bundle, priming_iterations=iterations)
    assert not session.inputs


@pytest.mark.parametrize("fault", ["extra_batch", "extra_axis", "missing_batch", "float64", "nan", "backend"])
def test_failed_preparation_rejects_the_instance_and_clears_state(bundle, monkeypatch, fault):
    session = Session()
    install_session(monkeypatch, session)
    error = RuntimeError("synthetic backend failed during preparation")
    invalid = {
        "extra_batch": np.zeros((2, 2), dtype=np.float32),
        "extra_axis": np.zeros((1, 1, 2), dtype=np.float32),
        "missing_batch": np.zeros(2, dtype=np.float32),
        "float64": np.zeros((1, 2), dtype=np.float64),
        "nan": np.full((1, 2), np.nan, dtype=np.float32),
    }

    def reply(frames, call):
        if call < 3:
            return np.zeros((1, 2), dtype=np.float32)
        if fault == "backend":
            raise error
        return invalid[fault]

    session.reply = reply
    cleared = []
    original_reset = FrameRuntime.reset

    def record_reset(runtime):
        original_reset(runtime)
        assert_cleared(runtime)
        cleared.append(runtime)

    monkeypatch.setattr(FrameRuntime, "reset", record_reset)
    expected = RuntimeError if fault == "backend" else FloatingPointError
    with pytest.raises(expected) as caught:
        construct(bundle)
    if fault == "backend":
        assert caught.value is error
    assert len(session.inputs) == 3
    assert len(cleared) == 1
    assert_cleared(cleared[0])


@pytest.mark.parametrize("fault", ["nan", "backend"])
def test_real_forward_failure_clears_observed_history_before_resuming(bundle, monkeypatch, fault):
    session = Session()
    install_session(monkeypatch, session)
    runtime = construct(bundle, priming_iterations=1)
    first = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    runtime.step(first, 42.0)
    assert runtime.last_time == 42.0 and runtime.frames.any()
    error = RuntimeError("synthetic backend failed on a real tick")

    def failed_reply(frames, call):
        if fault == "backend":
            raise error
        return np.full((1, 2), np.nan, dtype=np.float32)

    session.reply = failed_reply
    expected = RuntimeError if fault == "backend" else FloatingPointError
    with pytest.raises(expected) as caught:
        runtime.step(first + 0.1, 42.01)
    if fault == "backend":
        assert caught.value is error
    assert_cleared(runtime)
    assert len(session.inputs) == 3
    session.reply = None
    next_first = np.array([-0.75, 0.5, 0.25], dtype=np.float32)
    runtime.step(next_first, 99.0)
    np.testing.assert_array_equal(session.inputs[-1], np.broadcast_to(next_first, (1, 3, 3)))


def test_slow_fake_first_forward_occurs_before_the_armed_control_budget(bundle, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(frame_runtime, "time", clock)
    session = Session(clock=clock, first_seconds=0.020, later_seconds=0.001)
    install_session(monkeypatch, session)
    runtime = construct(bundle)
    assert len(session.inputs) == 50
    assert runtime.preparation["duration_ms"] == pytest.approx(69.0)
    assert_cleared(runtime)
    armed_at = clock.now
    reads, sent, faults = [], [], []
    first = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    def read_frame():
        reads.append(clock.now)
        return first, clock.now

    def hand_off(error):
        assert_cleared(runtime)
        faults.append(error)

    status = run_control_loop(runtime, read_frame, lambda *values: sent.append(values),
        should_stop=lambda: len(sent) == 2, on_fault=hand_off)
    assert status == "stopped" and len(reads) == len(sent) == 2
    assert len(session.inputs) == 52
    assert clock.now - armed_at == pytest.approx(0.011)
    assert len(clock.sleeps) == 1 and clock.sleeps[0] == pytest.approx(0.009)
    assert len(faults) == 1 and isinstance(faults[0], InterruptedError)
    np.testing.assert_array_equal(session.inputs[50], np.broadcast_to(first, (1, 3, 3)))


@pytest.mark.parametrize("gate,expected_reads,expected_sends,message", [
    ("release", 0, 0, "missed policy release"),
    ("inference", 1, 0, "observation and inference exceeded control budget"),
    ("callback", 1, 1, "control callback exceeded policy deadline"),
])
def test_preparation_does_not_relax_release_inference_or_callback_fail_stop(
        bundle, monkeypatch, gate, expected_reads, expected_sends, message):
    clock = Clock()
    monkeypatch.setattr(frame_runtime, "time", clock)
    session = Session(clock=clock, later_seconds=0.007 if gate == "inference" else 0.001)
    install_session(monkeypatch, session)
    runtime = construct(bundle, priming_iterations=1)
    reads, sent, faults = [], [], []
    stop_calls = 0
    frame = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    def should_stop():
        nonlocal stop_calls
        stop_calls += 1
        if gate == "release" and stop_calls == 1:
            clock.advance(0.0021)
        return stop_calls > 3

    def read_frame():
        reads.append(clock.now)
        clock.advance(0.0011 if gate == "inference" else 0.001)
        return frame, clock.now

    def send_targets(*values):
        sent.append(values)
        if gate == "callback":
            clock.advance(0.0081)

    def hand_off(error):
        assert_cleared(runtime)
        faults.append(error)

    status = run_control_loop(runtime, read_frame, send_targets,
        should_stop=should_stop, on_fault=hand_off)
    assert status == "fault"
    assert len(reads) == expected_reads and len(sent) == expected_sends
    assert len(session.inputs) == 1 + expected_reads
    assert len(faults) == 1 and isinstance(faults[0], TimeoutError)
    assert str(faults[0]) == message
    assert stop_calls == 1

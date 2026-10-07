"""Real small CPU learning and failure accounting; no robot or simulator runs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_rl.config import PPOConfig
import transformer_rl.exposure_training as exposure
from transformer_rl.frame_checkpoint import capture_rng
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.ppo import PPOTrainer

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from packed_env import make_env


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(architecture="mlp", history=1, readout="last", num_envs=2):
    policy = FramePolicyConfig(architecture=architecture, frame_dim=5, action_dim=2,
        history_length=history, actor_hidden_dims=(8,), encoder_hidden_dims=(8,),
        history_latent_dim=3, d_model=8, num_heads=2, num_layers=1, ffn_dim=12,
        residual_type="gated" if architecture == "transformer" else "add",
        readout_type=readout, mean_init_scale=.1)
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)],
        "action_names": ["position", "velocity"], "action_bounds": [.2, .5],
        "target_scale": [.25, 10.], "target_offset": [.1, -.2],
        "target_units": ["rad", "rad/s"]}
    return FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(8,),
        command_indices=(0,), initial_std=.8),
        PPOConfig(epochs=2, num_minibatches=2, learning_rate=1e-3, target_kl=.5),
        control, {"control": control, "num_envs": num_envs})


def stages(configuration=None, updates=2):
    configuration = configuration or config()
    return [{"name": f"exposure_{i}", "updates": updates,
        "config": replace(configuration, environment={**deepcopy(configuration.environment),
                                                   "num_envs": count})}
        for i, count in enumerate((2, 3, 4))]


def run_job(output, declared=None, factory=make_env, **overrides):
    declared = declared if declared is not None else stages()
    options = dict(job_id="cpu_exposure_fixture", rollout_steps=2, training_seed=71,
        retention_seed=901, evaluation_seeds=(801, 802), device="cpu",
        expected_initial_model_sha256=exposure.initial_model_sha256(declared[0]["config"], 71),
        max_seconds=30.)
    options.update(overrides)
    return exposure.train_exposure_job(declared, factory, "packed_env:make_env", output, **options)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def completion_on_disk(output, result):
    assert json.loads((output / "completion.json").read_bytes()) == result
    reservation = json.loads((output / "reservation.json").read_bytes())
    assert result["charged_updates"] == reservation["charged_updates"] == 6
    assert result["charged_fresh_transition_budget"] == reservation["charged_fresh_transition_budget"] == 36
    assert result["refund"] is False and result["automatic_retries"] == 0
    assert result["independent_evaluation_performed"] is False


@pytest.mark.parametrize("architecture,history,readout", [
    ("mlp", 1, "last"), ("history_mlp", 3, "last"),
    ("history_mlp", 5, "last"), ("transformer", 3, "last"),
    ("transformer", 5, "query"),
])
def test_real_three_stage_learning_preserves_Adam_RNG_and_fixed_low_reward_exposure(
        tmp_path, monkeypatch, architecture, history, readout):
    declared = stages(config(architecture, history, readout))
    opened, environments, progress = [], [], []
    original_open = FrameContinuation.open

    def noisy_low_reward_factory(**kwargs):
        torch.rand(9)
        random.random()
        np.random.rand(3)
        env = make_env(**kwargs)
        environments.append(env)
        env.set_training_progress = lambda updates, transitions: progress.append(
            (env.num_envs, updates, transitions))
        reset, step = env.reset, env.step

        def noisy_reset(seed=None):
            torch.rand(7)
            random.random()
            np.random.rand(5)
            return reset(seed)

        def low_reward(action):
            return replace(step(action), reward=torch.full((env.num_envs,), -1000.))

        env.reset, env.step = noisy_reset, low_reward
        return env

    def inspected_open(cls, configuration, factory, reference, checkpoint, **kwargs):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        session = original_open(configuration, factory, reference, checkpoint, **kwargs)
        same(session.model.state_dict(), payload["model"])
        same(session.trainer.optimizer.state_dict(), payload["optimizer"])
        same(capture_rng(), payload["rng"])
        same(session.sampler.state_dict(), payload["metadata"]["continuation"]["retention"])
        assert payload["optimizer"]["state"]
        assert any(torch.count_nonzero(state["exp_avg"]) for state in payload["optimizer"]["state"].values())
        assert session.update == kwargs["parent_update"]
        assert session.consumed_updates == kwargs["consumed_updates"]
        assert session.collected_transitions == kwargs["cumulative_transitions"]
        assert kwargs["resume"] is True and kwargs["environment_transition"] is True
        history_state = session.collector._history.snapshot()
        assert torch.equal(history_state.frames, history_state.frames[:, :1].expand_as(history_state.frames))
        opened.append(payload)
        return session

    monkeypatch.setattr(FrameContinuation, "open", classmethod(inspected_open))
    output = tmp_path / "job"
    result = run_job(output, declared, noisy_low_reward_factory)
    completion_on_disk(output, result)
    assert result["status"] == "completed" and result["error"] is None
    assert result["successful_updates"] == result["attempted_updates"] == result["recorded_metric_updates"] == 6
    assert result["actual_collected_transitions"] == result["successful_full_rollout_samples"] == 36
    assert result["recorded_full_rollout_samples"] == 36
    assert result["unverified_or_unoptimized_samples"] == result["unsealed_successful_updates"] == 0
    assert result["optimizer_steps"] > 0 and result["failed_update_optimizer_steps"] == 0
    assert len(opened) == 2 and all(env.closed for env in environments)
    assert [item["cumulative_successful_updates"] for item in result["endpoints"]] == [2, 4, 6]
    assert [item["cumulative_collected_transitions"] for item in result["endpoints"]] == [8, 20, 36]
    assert (3, 2, 8) in progress and (4, 4, 20) in progress
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["stage_index"] for row in rows] == [0, 0, 1, 1, 2, 2]
    assert [row["batch_samples"] for row in rows] == [4, 4, 6, 6, 8, 8]
    assert all(row["collection"]["reward_mean"] == -1000 for row in rows)
    assert sum(row["optimization"]["optimizer_steps"] for row in rows) == result["optimizer_steps"]
    assert sum(row["optimization"]["sample_count"] for row in rows) == result["optimization_sample_uses"]
    previous_steps = 0
    for endpoint in result["endpoints"]:
        checkpoint = Path(endpoint["checkpoint"]["path"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == endpoint["checkpoint"]["sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        steps = min(int(state["step"]) for state in payload["optimizer"]["state"].values())
        assert steps > previous_steps
        previous_steps = steps
        assert payload["metadata"]["continuation"]["clock"]["collected_transitions"] == endpoint["cumulative_collected_transitions"]
        assert payload["metadata"]["continuation"]["retention"]["coefficient"] == 0.


def test_initial_model_guard_preserves_caller_random_streams():
    before = capture_rng()
    value = exposure.initial_model_sha256(config("transformer", 5, "query"), 71)
    assert len(value) == 64
    same(capture_rng(), before)
    assert exposure.initial_model_sha256(config("transformer", 5, "query"), 71) == value


def test_partial_rollout_is_counted_without_optimizer_or_endpoint(tmp_path, monkeypatch):
    stop = {"value": False}
    optimizer_calls = []

    def factory(**kwargs):
        env = make_env(**kwargs)
        step = env.step

        def partial(action):
            result = step(action)
            stop["value"] = True
            return result

        env.step = partial
        return env

    monkeypatch.setattr(PPOTrainer, "update", lambda *args, **kwargs: optimizer_calls.append(True))
    output = tmp_path / "partial"
    result = run_job(output, factory=factory, should_stop=lambda: stop["value"])
    completion_on_disk(output, result)
    assert result["status"] == "partial_rollout" and result["stop_reason"] == "caller_stop"
    assert result["successful_updates"] == result["attempted_updates"] == 0
    assert result["actual_collected_transitions"] == result["unverified_or_unoptimized_samples"] == 2
    assert result["successful_full_rollout_samples"] == result["optimizer_steps"] == 0
    assert result["endpoints"] == [] and not list(output.rglob("endpoint*"))
    assert result["missing_stage_endpoints"] == [stage["name"] for stage in stages()]
    assert optimizer_calls == []


def test_partly_modified_Adam_failure_keeps_charge_actual_samples_and_prior_endpoint(tmp_path, monkeypatch):
    original_update = PPOTrainer.update
    calls, modified = [], []

    def fail_after_real_Adam(self, batch, **kwargs):
        before = deepcopy(self.model.state_dict())
        metrics = original_update(self, batch, **kwargs)
        calls.append(metrics)
        if len(calls) == 3:
            modified.append(any(not torch.equal(before[key], value) for key, value in self.model.state_dict().items()))
            assert self.optimizer.state_dict()["state"]
            raise RuntimeError("real Adam changed before failure")
        return metrics

    monkeypatch.setattr(PPOTrainer, "update", fail_after_real_Adam)
    output = tmp_path / "partial_optimizer"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and result["error"]["phase"] == "collect_optimize"
    assert result["error"]["message"] == "real Adam changed before failure"
    assert modified == [True]
    assert result["successful_updates"] == result["recorded_metric_updates"] == 2
    assert result["attempted_updates"] == 3 and result["failed_update_optimizer_steps"] is None
    assert result["optimizer_update_may_be_partial"] is True
    assert result["actual_collected_transitions"] == 14
    assert result["successful_full_rollout_samples"] == result["recorded_full_rollout_samples"] == 8
    assert result["unverified_or_unoptimized_samples"] == 6
    assert result["optimizer_steps"] == sum(row["optimizer_steps"] for row in calls[:2])
    assert len(calls) == 3 and len(result["endpoints"]) == 1
    assert result["last_sealed_checkpoint"] == result["endpoints"][0]["checkpoint"]
    assert not (output / "stage_0001_exposure_1" / "endpoint.pt").exists()


@pytest.mark.parametrize("failure", ["write", "short_write", "flush"])
def test_metrics_publication_failure_retains_real_success_and_optimizer_counts(tmp_path, monkeypatch, failure):
    original_open = Path.open
    observed = []
    original_update = PPOTrainer.update

    def capture_update(self, batch, **kwargs):
        value = original_update(self, batch, **kwargs)
        observed.append(value)
        return value

    class BrokenLog:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, text):
            if failure == "write":
                raise OSError("injected log write failure")
            written = self.stream.write(text)
            return written - 1 if failure == "short_write" else written

        def flush(self):
            raise OSError("injected log flush failure")

        def fileno(self):
            return self.stream.fileno()

    def broken_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return BrokenLog(stream) if path.name == "metrics.jsonl" and args and args[0] == "x" else stream

    monkeypatch.setattr(Path, "open", broken_open)
    monkeypatch.setattr(PPOTrainer, "update", capture_update)
    output = tmp_path / "log_failure"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and result["error"]["phase"] == "publish_metrics"
    assert len(observed) == result["successful_updates"] == result["attempted_updates"] == 1
    assert result["actual_collected_transitions"] == result["successful_full_rollout_samples"] == 4
    assert result["recorded_metric_updates"] == result["recorded_full_rollout_samples"] == 0
    assert result["unverified_or_unoptimized_samples"] == 0
    assert result["unsealed_successful_updates"] == 1
    assert result["failed_update_optimizer_steps"] == 0
    assert result["optimizer_steps"] == observed[0]["optimizer_steps"]
    assert result["optimization_sample_uses"] == observed[0]["sample_count"]
    assert result["endpoints"] == [] and not list(output.rglob("endpoint*"))


def test_primary_PPO_failure_is_not_replaced_by_close_error(tmp_path, monkeypatch):
    def factory(**kwargs):
        env = make_env(**kwargs)
        env.close = lambda: (_ for _ in ()).throw(ValueError("secondary close failure"))
        return env

    monkeypatch.setattr(PPOTrainer, "update", lambda *args, **kwargs:
        (_ for _ in ()).throw(RuntimeError("primary PPO failure")))
    output = tmp_path / "double_failure"
    result = run_job(output, factory=factory)
    completion_on_disk(output, result)
    assert result["status"] == "failed"
    assert result["error"] == {"type": "RuntimeError", "message": "primary PPO failure", "phase": "collect_optimize"}
    assert result["shutdown_errors"] == [{"type": "ValueError", "message": "secondary close failure", "during": "close"}]
    assert result["attempted_updates"] == 1 and result["successful_updates"] == 0
    assert result["failed_update_optimizer_steps"] is None
    assert result["actual_collected_transitions"] == 4


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_user_interrupt_is_terminal_without_retry_or_fake_success(tmp_path, exception):
    calls = []

    def factory(**kwargs):
        calls.append(True)
        env = make_env(**kwargs)
        env.step = lambda action: (_ for _ in ()).throw(exception("caller interrupted"))
        return env

    output = tmp_path / "interrupt"
    result = run_job(output, factory=factory)
    completion_on_disk(output, result)
    assert result["status"] == "user_interrupted" and result["error"]["type"] == exception.__name__
    assert calls == [True] and result["successful_updates"] == result["attempted_updates"] == 0
    assert result["endpoints"] == []


@pytest.mark.parametrize("reason,during_rollout", [("caller_stop", False), ("deadline", False), ("deadline", True)])
def test_stop_and_deadline_do_not_optimize_or_advance_exposure(tmp_path, monkeypatch, reason, during_rollout):
    clock = SimpleNamespace(value=0.)
    monkeypatch.setattr(exposure, "time", SimpleNamespace(monotonic=lambda: clock.value))
    calls = []

    def factory(**kwargs):
        calls.append(True)
        env = make_env(**kwargs)
        original_step = env.step

        def step(action):
            value = original_step(action)
            clock.value = 11.
            return value

        env.step = step
        return env

    if reason == "deadline" and not during_rollout:
        values = iter((0., 11., 11.))
        monkeypatch.setattr(exposure, "time", SimpleNamespace(monotonic=lambda: next(values, 11.)))
    output = tmp_path / "stop"
    result = run_job(output, factory=factory, max_seconds=10.,
        should_stop=(lambda: True) if reason == "caller_stop" else None)
    completion_on_disk(output, result)
    assert result["stop_reason"] == reason
    assert result["status"] == ("partial_rollout" if during_rollout else "interrupted")
    assert result["successful_updates"] == result["attempted_updates"] == result["optimizer_steps"] == 0
    assert result["actual_collected_transitions"] == (2 if during_rollout else 0)
    assert result["endpoints"] == [] and calls == ([True] if during_rollout else [])


@pytest.mark.parametrize("override", [
    {"training_seed": -1}, {"training_seed": True}, {"training_seed": 2**32},
    {"retention_seed": 71}, {"retention_seed": 801}, {"retention_seed": False},
    {"evaluation_seeds": [71]}, {"evaluation_seeds": [801, 801]},
    {"evaluation_seeds": [True]}, {"expected_initial_model_sha256": "0" * 64},
    {"expected_initial_model_sha256": "A" * 64}, {"device": "cuda"},
    {"rollout_steps": False}, {"max_seconds": 0}, {"max_seconds": float("nan")},
])
def test_invalid_identity_or_budget_rejected_before_output_and_factory(tmp_path, override):
    calls = []
    output = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError)):
        run_job(output, factory=lambda **kwargs: calls.append(True), **override)
    assert calls == [] and not output.exists()


@pytest.mark.parametrize("mutation", ["zero_envs", "mutable_bad_control", "changed_ppo", "duplicate_stage", "boolean_updates"])
def test_invalid_stage_configs_rejected_before_output_and_factory(tmp_path, mutation):
    declared = stages()
    if mutation == "zero_envs":
        declared[1]["config"].environment["num_envs"] = 0
    elif mutation == "mutable_bad_control":
        declared[1]["config"].control["action_bounds"][0] = 0.
    elif mutation == "changed_ppo":
        declared[1]["config"] = replace(declared[1]["config"], ppo=replace(declared[1]["config"].ppo, learning_rate=1e-2))
    elif mutation == "duplicate_stage":
        declared[1]["name"] = declared[0]["name"]
    else:
        declared[1]["updates"] = True
    calls, output = [], tmp_path / "invalid_stage"
    with pytest.raises((ValueError, TypeError)):
        run_job(output, declared, factory=lambda **kwargs: calls.append(True))
    assert calls == [] and not output.exists()


def test_factory_mutating_owned_runtime_configuration_is_rejected(tmp_path):
    def factory(**kwargs):
        env = make_env(**kwargs)
        kwargs["environment_config"]["num_envs"] = 99
        return env

    output = tmp_path / "mutated_runtime"
    result = run_job(output, factory=factory)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and result["error"]["phase"] == "verify_runtime_budget"
    assert result["successful_updates"] == result["actual_collected_transitions"] == 0
    assert result["endpoints"] == []


def test_caller_config_mutation_does_not_modify_frozen_owned_stages(tmp_path):
    declared = stages()
    original = [deepcopy(item["config"].to_dict()) for item in declared]

    def factory(**kwargs):
        declared[0]["config"].environment["num_envs"] = 999
        declared[1]["config"].control["action_bounds"][0] = 0.
        return make_env(**kwargs)

    output = tmp_path / "owned_config"
    result = run_job(output, declared, factory)
    assert result["status"] == "completed"
    request = json.loads((output / "request.json").read_bytes())
    assert [item["config"] for item in request["stages"]] == original
    assert result["actual_collected_transitions"] == 36


@pytest.mark.parametrize("key,value", [("source", {"sha256": "0" * 64, "files": {}}),
    ("seed", 72), ("environment_factory", "other_env:factory")])
def test_mutated_runtime_learning_metadata_is_rejected_before_first_endpoint(tmp_path, monkeypatch, key, value):
    original_start = FrameContinuation.start

    def poisoned_start(cls, *args, **kwargs):
        session = original_start(*args, **kwargs)
        session.metadata[key] = value
        return session

    monkeypatch.setattr(FrameContinuation, "start", classmethod(poisoned_start))
    output = tmp_path / "poisoned_metadata"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed"
    assert result["successful_updates"] == result["attempted_updates"] == 0
    assert result["endpoints"] == [] and not list(output.rglob("endpoint*"))


@pytest.mark.parametrize("mutation,after_update", [("source", 1), ("source", 2),
    ("config", 1), ("config", 2)])
def test_runtime_mutation_after_real_learning_preserves_accounting_without_sealing(
        tmp_path, monkeypatch, mutation, after_update):
    original_step = FrameContinuation.step
    observed = []

    def mutate_after_success(session, **kwargs):
        row = original_step(session, **kwargs)
        observed.append(row)
        if session.update == after_update:
            if mutation == "source":
                session.metadata["source"] = {"sha256": "0" * 64, "files": {}}
            else:
                session.config.environment["num_envs"] = 999
        return row

    monkeypatch.setattr(FrameContinuation, "step", mutate_after_success)
    output = tmp_path / "mutation_after_learning"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed"
    assert result["error"]["phase"] == ("collect_optimize" if after_update == 1 else "verify_source")
    assert len(observed) == result["successful_updates"] == result["attempted_updates"] == after_update
    assert result["recorded_metric_updates"] == result["unsealed_successful_updates"] == after_update
    assert result["actual_collected_transitions"] == result["successful_full_rollout_samples"] == 4 * after_update
    assert result["recorded_full_rollout_samples"] == 4 * after_update
    assert result["unverified_or_unoptimized_samples"] == 0
    assert result["optimizer_steps"] == sum(row["optimization"]["optimizer_steps"] for row in observed)
    assert result["failed_update_optimizer_steps"] == 0
    assert result["endpoints"] == [] and not list(output.rglob("endpoint*"))


@pytest.mark.parametrize("changed_at,expected_success,expected_endpoints", [(2, 0, 0), (4, 2, 0), (5, 2, 1)])
def test_changed_source_identity_refuses_construction_or_endpoint(tmp_path, monkeypatch, changed_at, expected_success, expected_endpoints):
    actual_source = exposure.source_identity()
    counter = {"value": 0}

    def changed_source():
        counter["value"] += 1
        return actual_source if counter["value"] < changed_at else {"sha256": "0" * 64, "files": {}}

    monkeypatch.setattr(exposure, "source_identity", changed_source)
    output = tmp_path / "changed_source"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and "source changed" in result["error"]["message"]
    assert result["successful_updates"] == expected_success
    assert len(result["endpoints"]) == expected_endpoints
    assert result["unsealed_successful_updates"] == expected_success - 2 * expected_endpoints


@pytest.mark.parametrize("publication", ["request", "reservation", "checkpoint", "endpoint", "completion"])
def test_publication_failure_never_reports_completed_and_directory_cannot_be_reused(tmp_path, monkeypatch, publication):
    original_new = exposure._new
    calls = []

    def broken_new(path, value):
        if path.name == publication + ".json":
            raise OSError("injected " + publication + " publication failure")
        return original_new(path, value)

    def factory(**kwargs):
        calls.append(True)
        return make_env(**kwargs)

    monkeypatch.setattr(exposure, "_new", broken_new)
    if publication == "checkpoint":
        monkeypatch.setattr(FrameContinuation, "save", lambda *args, **kwargs:
            (_ for _ in ()).throw(OSError("injected checkpoint publication failure")))
    output = tmp_path / "publication_failure"
    if publication in ("request", "reservation", "completion"):
        with pytest.raises(OSError, match="publication failure"):
            run_job(output, factory=factory)
        assert not (output / "completion.json").exists()
        if publication in ("request", "reservation"):
            assert calls == []
    else:
        result = run_job(output, factory=factory)
        completion_on_disk(output, result)
        assert result["status"] == "failed"
        assert result["error"]["phase"] == "publish_" + publication
        assert result["successful_updates"] == result["recorded_metric_updates"] == 2
        assert result["actual_collected_transitions"] == 8
        assert result["unsealed_successful_updates"] == 2 and result["endpoints"] == []
        assert len(result["unsealed_checkpoint_paths"]) == (1 if publication == "endpoint" else 0)
    previous_calls = len(calls)
    with pytest.raises(ValueError, match="new output"):
        run_job(output, factory=factory)
    assert len(calls) == previous_calls


def test_metrics_fsync_failure_keeps_sealed_stages_and_actual_counts(tmp_path, monkeypatch):
    copied_os = SimpleNamespace(**vars(exposure.os))
    copied_os.fsync = lambda *_: (_ for _ in ()).throw(OSError("injected metrics fsync failure"))
    monkeypatch.setattr(exposure, "os", copied_os)
    output = tmp_path / "fsync_failure"
    result = run_job(output)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and result["error"]["phase"] == "fsync_metrics"
    assert result["successful_updates"] == result["recorded_metric_updates"] == 6
    assert result["actual_collected_transitions"] == result["recorded_full_rollout_samples"] == 36
    assert len(result["endpoints"]) == 3 and result["missing_stage_endpoints"] == []


def test_close_between_stages_failure_keeps_prior_endpoint_and_does_not_retry(tmp_path):
    calls = []

    def factory(**kwargs):
        calls.append(True)
        env = make_env(**kwargs)

        def close():
            env.closed = True
            raise RuntimeError("previous stage close failed")

        env.close = close
        return env

    output = tmp_path / "stage_close_failure"
    result = run_job(output, factory=factory)
    completion_on_disk(output, result)
    assert result["status"] == "failed" and result["error"]["phase"] == "close_previous_environment"
    assert calls == [True] and result["successful_updates"] == 2
    assert len(result["endpoints"]) == 1 and result["last_sealed_checkpoint"] is not None


@pytest.mark.parametrize("overlap", ["inside", "ancestor"])
def test_output_overlap_rejected_before_factory(tmp_path, overlap):
    protected = tmp_path / "protected"
    protected.mkdir()
    output = protected / "job" if overlap == "inside" else tmp_path / "job"
    protected_path = protected if overlap == "inside" else output / "future_input"
    calls = []
    with pytest.raises(ValueError, match="overlaps"):
        run_job(output, factory=lambda **kwargs: calls.append(True), protected_paths=(protected_path,))
    assert calls == [] and not output.exists()

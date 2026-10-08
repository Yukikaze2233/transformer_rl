"""Tiny CPU worker checks; synthetic authorization never qualifies real branches."""
from copy import deepcopy
from dataclasses import dataclass
import builtins
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.config import PPOConfig
from transformer_rl.experiments import source_identity
from transformer_rl.frame_checkpoint import load_frame_checkpoint, save_frame_checkpoint
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest, json_bytes
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.frame_training import FrameActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.retention import save_anchors

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from packed_env import PackedFixture


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reseal(request):
    request["sha256"] = digest({key: value for key, value in request.items() if key != "sha256"})
    return request


def put(request, route, value):
    target = request
    for name in route[:-1]:
        target = target[name]
    target[route[-1]] = value


def configuration():
    policy = FramePolicyConfig(architecture="transformer", residual_type="gated",
        frame_dim=5, action_dim=2, history_length=4, actor_hidden_dims=(8,),
        d_model=8, num_heads=2, num_layers=1, ffn_dim=12)
    control = {"policy_dt_s": .01, "observation_schema": "small_worker_tensor_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)],
        "action_names": ["position", "velocity"], "action_bounds": [.2, .5],
        "target_scale": [.25, 10.], "target_offset": [.1, -.2],
        "target_units": ["rad", "rad/s"]}
    return FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(8,),
        command_indices=(0,), initial_std=.8),
        PPOConfig(epochs=1, num_minibatches=1, learning_rate=1e-3, target_kl=.5),
        control, {"control": control, "num_envs": 2})


@dataclass
class Rig:
    worker: object
    request: dict
    config: FrameTrainConfig
    model: object
    parent: Path
    events: list
    environments: list

    @property
    def directory(self):
        return Path(self.request["execution"]["run_directory"])

    def anchor(self, tmp_path):
        path = tmp_path / "small_anchors.pt"
        frames = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 100
        with torch.no_grad():
            mean = self.model.actor.policy(frames) + .1
            std = self.model.actor.log_std.exp().expand_as(mean).clone()
        save_anchors(path, self.config, frames, mean, std, self.request["checkpoint"]["sha256"])
        self.request["retention"].update(coefficient=.3,
            anchors=[{"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}])
        reseal(self.request)
        return path


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    worker = importlib.import_module("transformer_rl.continuation_process")
    campaign = importlib.import_module("transformer_rl.retention_campaign")
    config = configuration()
    parent = tmp_path / "parent.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = FrameActorCritic(config.model)
        trainer = PPOTrainer(model, config.ppo)
        save_frame_checkpoint(parent, model, trainer, config, 0,
            {"seed": 71, "environment_factory": "packed_env:make_env",
             "environment_provenance": {"identity": "synthetic_tensor_fixture",
                                        "control_sha256": digest(config.control)},
             "collected_transitions": 0, "source": source_identity()})
    assert parent.stat().st_size < 150_000
    controller = tmp_path / "synthetic_controller.json"
    controller.write_bytes(json_bytes({"synthetic_unit_fixture": True}))
    resource = tmp_path / "synthetic_resource.lock"
    resource.write_bytes(b"synthetic unit lease placeholder")
    events, environments = [], []
    body = {"format": "transformer_rl.continuation_request", "schema_version": 1,
        "config": config.to_dict(), "environment_factory": "packed_env:make_env",
        "checkpoint": {"path": str(parent), "sha256": file_sha(parent), "update": 0,
                       "cumulative_transitions": 0, "consumed_updates": 0},
        "training_seed": 71,
        "retention": {"seed": 901, "coefficient": 0., "batch_size": 3,
                      "anchors": [], "evaluation_seeds": [801, 802]},
        "execution": {"updates": 2, "rollout_steps": 2, "transitions_per_update": 4,
            "fresh_transition_budget": 8, "consumed_update_budget": 2,
            "max_seconds": 60., "checkpoint_interval": 1, "device": "cpu",
            "tensorboard": False, "run_directory": str(tmp_path / "worker"), "resume": False},
        "provenance": {"controller_protocol_path": str(controller),
            "controller_protocol_sha256": file_sha(controller),
            "preparation_protocol_sha256": "1" * 64,
            "preparation_manifest_sha256": "2" * 64,
            "branch_id": "branch_" + "a" * 24,
            "controller_lease": [{"path": str(controller), "descriptor": 3,
                "device": controller.stat().st_dev, "inode": controller.stat().st_ino,
                "controller_pid": os.getpid(), "controller_start": 123},
                {"path": str(resource), "descriptor": 4,
                "device": resource.stat().st_dev, "inode": resource.stat().st_ino,
                "controller_pid": os.getpid(), "controller_start": 123}]},
        "source": source_identity()}

    def authorize_synthetic(request):
        # This replaces the fixed production auditor only for this tiny unit fixture.
        events.append("synthetic_authorization")
        return request

    monkeypatch.setattr(campaign, "validate_worker_request", authorize_synthetic)

    def factory(**kwargs):
        events.append("factory")
        env = PackedFixture(**kwargs)
        environments.append(env)
        env.set_training_progress = lambda updates, samples: events.append(("progress", updates, samples))
        original_reset, original_close = env.reset, env.close

        def reset(seed=None):
            events.append(("reset", seed))
            return original_reset(seed)

        def close():
            events.append("env_close")
            return original_close()

        env.reset, env.close = reset, close
        return env

    def lazy_factory(reference):
        events.append(("factory_import", reference))
        assert reference == "packed_env:make_env"
        return factory

    monkeypatch.setattr(worker, "_factory", lazy_factory)
    return Rig(worker, reseal(body), config, model, parent, events, environments)


def assert_preflight_rejection(rig, request):
    with pytest.raises((ValueError, TypeError, FileExistsError, FileNotFoundError)):
        rig.worker.run_request(request)
    assert "factory" not in rig.events
    assert not any(isinstance(event, tuple) and event[0] == "factory_import" for event in rig.events)


@pytest.mark.parametrize("route,value", [
    (("format",), "unknown"), (("schema_version",), True),
    (("checkpoint", "update"), True), (("checkpoint", "cumulative_transitions"), True),
    (("checkpoint", "consumed_updates"), True), (("training_seed",), True),
    (("training_seed",), 2**32), (("retention", "seed"), True),
    (("retention", "seed"), 71), (("retention", "seed"), 801),
    (("retention", "seed"), 2**32), (("retention", "coefficient"), True),
    (("retention", "coefficient"), -.1), (("retention", "batch_size"), True),
    (("retention", "batch_size"), 0), (("retention", "evaluation_seeds"), [True]),
    (("execution", "updates"), True), (("execution", "updates"), 3),
    (("execution", "rollout_steps"), True), (("execution", "transitions_per_update"), 5),
    (("execution", "fresh_transition_budget"), 7), (("execution", "consumed_update_budget"), True),
    (("execution", "consumed_update_budget"), 1), (("execution", "max_seconds"), True),
    (("execution", "max_seconds"), 0.), (("execution", "checkpoint_interval"), True),
    (("execution", "checkpoint_interval"), 0), (("execution", "tensorboard"), 0),
    (("execution", "resume"), 0), (("execution", "device"), "cuda"),
    (("environment_factory",), "packed_env"), (("environment_factory",), "packed_env:bad-name"),
    (("provenance", "controller_protocol_path"), "relative.json"),
    (("provenance", "branch_id"), ""), (("source", "sha256"), "0" * 64),
    (("provenance", "controller_lease"), []),
])
def test_resealed_invalid_request_rejected_before_factory(rig, route, value):
    request = deepcopy(rig.request)
    put(request, route, value)
    assert_preflight_rejection(rig, reseal(request))


@pytest.mark.parametrize("section", [None, "checkpoint", "retention", "execution", "provenance"])
def test_unknown_request_fields_rejected_before_factory(rig, section):
    request = deepcopy(rig.request)
    target = request if section is None else request[section]
    target["eligible"] = True
    assert_preflight_rejection(rig, reseal(request))


def test_request_checksum_rejected_before_factory(rig):
    request = deepcopy(rig.request)
    request["sha256"] = "0" * 64
    assert_preflight_rejection(rig, request)


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("update", 1), ("cumulative_transitions", 1),
])
def test_actual_parent_binding_rejected_before_factory(rig, field, value):
    request = deepcopy(rig.request)
    request["checkpoint"][field] = value
    assert_preflight_rejection(rig, reseal(request))


@pytest.mark.parametrize("mutation", ["training_seed", "factory", "transition_bool", "ppo"])
def test_rehashed_parent_metadata_or_recipe_rejected_before_factory(rig, tmp_path, mutation):
    payload = torch.load(rig.parent, map_location="cpu", weights_only=True)
    if mutation == "training_seed":
        payload["metadata"]["seed"] = 72
    elif mutation == "factory":
        payload["metadata"]["environment_factory"] = "another:factory"
    elif mutation == "transition_bool":
        payload["metadata"]["collected_transitions"] = False
    else:
        payload["config"]["ppo"]["learning_rate"] = 3e-4
    path = tmp_path / "changed_parent.pt"
    torch.save(payload, path)
    request = deepcopy(rig.request)
    request["checkpoint"].update(path=str(path), sha256=file_sha(path))
    assert_preflight_rejection(rig, reseal(request))


@pytest.mark.parametrize("mutation", ["sha", "bytes", "file", "zero_lambda", "missing_positive"])
def test_anchor_receipts_rejected_before_factory(rig, tmp_path, mutation):
    path = rig.anchor(tmp_path)
    request = deepcopy(rig.request)
    if mutation == "sha":
        request["retention"]["anchors"][0]["sha256"] = "0" * 64
    elif mutation == "bytes":
        request["retention"]["anchors"][0]["bytes"] += 1
    elif mutation == "file":
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation == "zero_lambda":
        request["retention"]["coefficient"] = 0.
    else:
        request["retention"]["anchors"] = []
    assert_preflight_rejection(rig, reseal(request))


@pytest.mark.parametrize("occupied", ["directory", "dangling_symlink", "file"])
def test_output_occupancy_rejected_without_overwrite_or_factory(rig, tmp_path, occupied):
    directory = rig.directory
    if occupied == "directory":
        directory.mkdir()
        (directory / "sentinel").write_text("preserve")
    elif occupied == "file":
        directory.write_text("preserve")
    else:
        directory.symlink_to(tmp_path / "absent_target", target_is_directory=True)
    assert_preflight_rejection(rig, rig.request)
    if occupied == "directory":
        assert (directory / "sentinel").read_text() == "preserve"
    elif occupied == "file":
        assert directory.read_text() == "preserve"
    else:
        assert directory.is_symlink() and not directory.exists()
        assert not (tmp_path / "absent_target").exists()


def test_fixed_authorization_rejection_precedes_factory(rig, monkeypatch):
    campaign = importlib.import_module("transformer_rl.retention_campaign")
    failure = ValueError("synthetic authorization refused")

    def refuse(request):
        raise failure

    monkeypatch.setattr(campaign, "validate_worker_request", refuse)
    with pytest.raises(ValueError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    assert "factory" not in rig.events
    assert not rig.directory.exists()


def test_complete_updates_publish_interval_and_final_without_tensorboard(rig, monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard" or name.startswith("tensorboard"):
            pytest.fail("disabled TensorBoard was imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    report = rig.worker.run_request(rig.request)
    assert report["status"] == "completed"
    assert report["completed_updates"] == report["attempted_updates"] == 2
    assert report["final_update"] == 2
    assert report["consumed_transitions"] == report["cumulative_transitions"] == 8
    assert file_sha(report["checkpoint"]) == report["checkpoint_sha256"]
    assert rig.environments[0].closed
    assert rig.events.index("synthetic_authorization") < rig.events.index(("factory_import", "packed_env:make_env"))
    assert rig.events.index(("progress", 0, 0)) < rig.events.index(("reset", 71))
    checkpoints = rig.directory / "checkpoints"
    assert (checkpoints / "update_000001.pt").is_file()
    assert (checkpoints / "update_000002.pt").is_file()
    assert (checkpoints / "final.pt").is_file()
    _, trainer, _, update, metadata, _ = load_frame_checkpoint(checkpoints / "final.pt")
    assert update == 2 and trainer.optimizer.state_dict()["state"]
    assert metadata["continuation"]["clock"] == {
        "consumed_updates": 2, "collected_transitions": 8, "rollout_steps": 2}
    assert metadata["continuation"]["retention"]["draw_count"] == 0
    assert metadata["continuation_branch"] == {
        key: rig.request["provenance"][key] for key in (
            "controller_protocol_sha256", "preparation_protocol_sha256",
            "preparation_manifest_sha256", "branch_id")}
    assert metadata["source"] == rig.request["source"]
    records = [json.loads(line) for line in (rig.directory / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert [entry["batch_samples"] for entry in records] == [4, 4]
    assert sum(path.stat().st_size for path in rig.directory.rglob("*") if path.is_file()) < 1_000_000


@pytest.mark.parametrize("outcome", ["completed", "stopped", "invalid_source"])
def test_worker_preserves_reward_windows_and_partial_failure_accounting(rig, monkeypatch, outcome):
    from transformer_rl.reward_components import RewardComponentStatistics

    original_init, original_step = PackedFixture.__init__, PackedFixture.step
    budgets, writers = [], []

    class ControlledBudget(rig.worker._StopBudget):
        def __init__(self, seconds):
            super().__init__(seconds)
            budgets.append(self)

    class Writer:
        def __init__(self, path):
            self.scalars, self.closed = [], False
            writers.append(self)

        def add_scalar(self, tag, value, update):
            self.scalars.append((tag, value, update))

        def close(self):
            self.closed = True

    def initialize(self, **kwargs):
        original_init(self, **kwargs)
        self.enable_reward_components = False
        self.reward_statistics = RewardComponentStatistics(self.num_envs, self.dt)
        self.reward_source_error = None
        self.reward_observed_steps = 0

    def step(self, action):
        result = original_step(self, action)
        if self.enable_reward_components:
            self.reward_observed_steps += 1
            density = torch.full_like(result.reward, 2. * self.tick)
            if outcome == "invalid_source" and self.tick == 2:
                density[0] = float("nan")
            if self.reward_source_error is None:
                try:
                    self.reward_statistics.observe(result.reward, {"height": density})
                except ValueError as error:
                    self.reward_source_error = str(error)
                except FloatingPointError as error:
                    self.reward_source_error = str(error)
        if outcome == "stopped":
            budgets[0].deadline = -1.
        return result

    def drain(self):
        result = self.reward_statistics.drain()
        result.update(observation_error=self.reward_source_error,
                      observed_vector_steps=self.reward_observed_steps)
        self.reward_source_error, self.reward_observed_steps = None, 0
        return result

    monkeypatch.setattr(PackedFixture, "__init__", initialize)
    monkeypatch.setattr(PackedFixture, "step", step)
    monkeypatch.setattr(PackedFixture, "drain_reward_components", drain, raising=False)
    monkeypatch.setattr(rig.worker, "_StopBudget", ControlledBudget)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=Writer))
    request = deepcopy(rig.request)
    request["execution"]["tensorboard"] = True
    reseal(request)
    if outcome == "invalid_source":
        with pytest.raises(ValueError, match="invalid reward component telemetry.*nonfinite"):
            rig.worker.run_request(request)
        report = json.loads((rig.directory / "failure.json").read_text())
        assert report["actual_samples"] == report["failed_collection_samples"] == 4
        assert report["attempted_updates"] == report["completed_updates"] == 0
        assert report["failed_optimizer_samples"] == 0
        window = report["last_collection_reward_components"]
        assert (window["steps"], window["samples"], window["observed_vector_steps"]) == (1, 2, 2)
        assert "nonfinite" in window["observation_error"]
    else:
        report = rig.worker.run_request(request)
        assert report["status"] == outcome
        window = report["last_collection_reward_components"]
        records = [json.loads(line) for line in (rig.directory / "metrics.jsonl").read_text().splitlines()]
        if outcome == "stopped":
            assert report["actual_samples"] == report["partial_samples"] == 2
            assert report["attempted_updates"] == report["completed_updates"] == 0
            assert (window["steps"], window["samples"]) == (1, 2)
            assert not records
        else:
            assert report["actual_samples"] == 8 and report["completed_updates"] == 2
            assert len(records) == 2
            assert window == records[-1]["collection"]["reward_components"]
            densities = [entry["collection"]["reward_components"]["continuous_components"]
                         ["terms"]["height"]["density"]["mean"] for entry in records]
            assert densities == [3., 7.]
            tags = {(tag, update): value for tag, value, update in writers[0].scalars}
            assert tags["reward_components/density_per_s/height", 1] == 3.
            assert tags["reward_components/step_contribution/height", 2] == pytest.approx(.07)
    assert window["event_components"]["available"] is False
    assert window["gate_coverage"]["available"] is False
    assert writers[0].closed and rig.environments[0].closed


def test_positive_retention_checkpoint_keeps_private_sampler_state(rig, tmp_path):
    rig.anchor(tmp_path)
    report = rig.worker.run_request(rig.request)
    metadata = load_frame_checkpoint(report["checkpoint"])[4]
    state = metadata["continuation"]["retention"]
    assert state["coefficient"] == .3 and state["seed"] == 901
    assert state["draw_count"] == 2 and state["endpoint_draw_count"] == 6
    assert state["identities"] == [
        {key: entry[key] for key in ("path", "sha256")}
        for entry in rig.request["retention"]["anchors"]]


def test_ppo_failure_keeps_original_error_and_never_publishes_final(rig, monkeypatch):
    failure = RuntimeError("synthetic PPO failure after full collection")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(PPOTrainer, "update", fail)
    with pytest.raises(RuntimeError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    assert rig.environments[0].closed
    assert not (rig.directory / "checkpoints" / "final.pt").exists()
    assert not list((rig.directory / "checkpoints").glob("*.pt"))
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert ledger["status"] == "failed"
    assert ledger["completed_updates"] == 0 and ledger["attempted_updates"] == 1
    assert ledger["consumed_transitions"] == 4
    assert ledger["actual_samples"] == ledger["failed_optimizer_samples"] == 4
    assert ledger["partial_samples"] == ledger["failed_collection_samples"] == 0
    assert ledger["discarded_samples"] == 4
    assert ledger["unsealed_successful_updates"] == 0
    assert ledger["reserved_update_budget"] + ledger["attempted_updates"] == 2
    assert ledger["reserved_transition_budget"] + ledger["actual_samples"] == 8
    assert "synthetic PPO failure" in ledger["error"]


def test_runtime_environment_count_mismatch_closes_environment(rig, monkeypatch):
    original = PackedFixture.__init__

    def wrong_count(self, **kwargs):
        original(self, **kwargs)
        self.num_envs = 3

    monkeypatch.setattr(PackedFixture, "__init__", wrong_count)
    with pytest.raises((ValueError, TypeError)):
        rig.worker.run_request(rig.request)
    assert len(rig.environments) == 1 and rig.environments[0].closed
    assert not (rig.directory / "checkpoints" / "final.pt").exists()
    assert (rig.directory / "failure.json").exists()


def test_lazy_factory_close_failure_keeps_runtime_count_error(rig, monkeypatch):
    original = PackedFixture.__init__

    def wrong_count(self, **kwargs):
        original(self, **kwargs)
        self.num_envs = 3

    def bad_close(self):
        self.closed = True
        raise OSError("synthetic lazy environment close failure")

    monkeypatch.setattr(PackedFixture, "__init__", wrong_count)
    monkeypatch.setattr(PackedFixture, "close", bad_close)
    with pytest.raises(ValueError, match="runtime environment num_envs"):
        rig.worker.run_request(rig.request)
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert "runtime environment num_envs" in ledger["error"]
    assert ledger["shutdown_errors"] == [{"owner": "environment",
        "error": "OSError: synthetic lazy environment close failure"}]
    assert ledger["actual_samples"] == ledger["attempted_updates"] == 0
    assert ledger["checkpoint"] is None and ledger["checkpoint_sha256"] is None
    assert rig.environments[0].closed


def test_failed_final_publication_does_not_pair_residual_path_with_interval_sha(rig, monkeypatch):
    original = rig.worker.FrameContinuation.save
    failure = OSError("synthetic final publication failure")

    def fail_final(self, path):
        path = Path(path)
        if path.name == "final.pt":
            # Simulate a publication failure leaving an untrusted file. The
            # failure ledger must retain only the earlier complete checkpoint.
            path.write_bytes(b"unsealed synthetic residual")
            raise failure
        return original(self, path)

    monkeypatch.setattr(rig.worker.FrameContinuation, "save", fail_final)
    with pytest.raises(OSError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert ledger["failure_stage"] == "final_checkpoint"
    assert ledger["checkpoint"] is None and ledger["checkpoint_sha256"] is None
    sealed = rig.directory / "checkpoints" / "update_000002.pt"
    assert ledger["last_sealed_checkpoint"]["path"] == str(sealed)
    assert ledger["last_sealed_checkpoint"]["sha256"] == file_sha(sealed)
    assert ledger["actual_samples"] == 8 and ledger["completed_updates"] == 2
    assert rig.environments[0].closed


def test_worker_owns_only_new_apps_and_restores_old_registry(rig, monkeypatch):
    process = importlib.import_module("transformer_rl.frame_process")
    calls = []

    class App:
        def __init__(self, name):
            self.name = name

        def close(self, *, wait_for_replicator, exit_code):
            calls.append((self.name, wait_for_replicator, exit_code, rig.environments[0].closed))

    original_apps = [App("unrelated")]
    monkeypatch.setattr(process, "_apps", original_apps)
    monkeypatch.setattr(process, "_active", False)
    original = PackedFixture.__init__

    def register(self, **kwargs):
        process.require_worker()
        original(self, **kwargs)
        process.register_app(App("first"))
        process.register_app(App("second"))

    monkeypatch.setattr(PackedFixture, "__init__", register)
    rig.worker.run_request(rig.request)
    assert calls == [("second", False, 0, True), ("first", False, 0, True)]
    assert process._apps is original_apps and len(original_apps) == 1
    assert process._active is False


def test_shutdown_errors_do_not_mask_ppo_failure_and_all_apps_close(rig, monkeypatch):
    process = importlib.import_module("transformer_rl.frame_process")
    original = PackedFixture.__init__
    calls = []
    failure = RuntimeError("original synthetic PPO failure")

    class App:
        def __init__(self, name, fail):
            self.name, self.fail = name, fail

        def close(self, **kwargs):
            calls.append(self.name)
            if self.fail:
                raise OSError("synthetic app close failure")

    def register(self, **kwargs):
        original(self, **kwargs)
        process.register_app(App("first", False))
        process.register_app(App("second", True))

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(PackedFixture, "__init__", register)
    monkeypatch.setattr(PPOTrainer, "update", fail)
    with pytest.raises(RuntimeError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    assert calls == ["second", "first"]
    assert rig.environments[0].closed
    assert "original synthetic PPO failure" in json.loads((rig.directory / "failure.json").read_text())["error"]


@pytest.mark.parametrize("contents", ["[]", '{"format":"x","format":"y"}', '{"x":NaN}'])
def test_cli_strict_json_rejected_before_factory(rig, tmp_path, contents):
    path = tmp_path / "bad_request.json"
    path.write_text(contents)
    try:
        result = rig.worker.main(["--request", str(path)])
    except (ValueError, TypeError, SystemExit) as error:
        if isinstance(error, SystemExit):
            assert error.code not in (None, 0)
    else:
        assert result != 0
    assert "factory" not in rig.events
    assert "synthetic_authorization" not in rig.events


def test_cli_requires_absolute_request_path(rig, tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_bytes(json_bytes(rig.request) + b"\n")
    monkeypatch.chdir(tmp_path)
    try:
        result = rig.worker.main(["--request", "request.json"])
    except (ValueError, TypeError, SystemExit) as error:
        if isinstance(error, SystemExit):
            assert error.code not in (None, 0)
    else:
        assert result != 0
    assert "factory" not in rig.events
    assert "synthetic_authorization" not in rig.events


@pytest.mark.parametrize("reason", ["SIGTERM", "time_budget"])
def test_soft_stop_discards_partial_rollout_and_charges_actual_samples(rig, monkeypatch, reason):
    cli = importlib.import_module("transformer_rl.cli")
    budgets = []
    original_step = PackedFixture.step
    previous_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    class ControlledBudget(cli._StopBudget):
        def __init__(self, seconds):
            super().__init__(seconds)
            budgets.append(self)

    def stop_after_one_step(self, action):
        result = original_step(self, action)
        if reason == "SIGTERM":
            budgets[0]._signal(signal.SIGTERM, None)
        else:
            budgets[0].deadline = -1.
        return result

    monkeypatch.setattr(rig.worker, "_StopBudget", ControlledBudget)
    monkeypatch.setattr(PackedFixture, "step", stop_after_one_step)
    report = rig.worker.run_request(rig.request)
    assert report["status"] == "stopped" and report["stop_reason"] == reason
    assert report["completed_updates"] == report["attempted_updates"] == report["final_update"] == 0
    assert report["actual_samples"] == report["consumed_transitions"] == 2
    assert report["partial_samples"] == report["discarded_samples"] == 2
    assert report["failed_optimizer_samples"] == report["failed_collection_samples"] == 0
    assert report["reserved_update_budget"] == 2
    assert report["reserved_transition_budget"] == 6
    assert rig.environments[0].tick == 1 and rig.environments[0].closed
    _, trainer, _, update, metadata, _ = load_frame_checkpoint(report["checkpoint"])
    assert update == 0 and not trainer.optimizer.state_dict()["state"]
    assert metadata["continuation"]["clock"] == {
        "consumed_updates": 0, "collected_transitions": 2, "rollout_steps": 2}
    assert metadata["continuation_segment"]["discarded_transitions"] == 2
    assert all(signal.getsignal(number) == handler for number, handler in previous_handlers.items())


def test_sigterm_during_ppo_finishes_update_before_final_save(rig, monkeypatch):
    cli = importlib.import_module("transformer_rl.cli")
    budgets = []
    original = PPOTrainer.update

    class ControlledBudget(cli._StopBudget):
        def __init__(self, seconds):
            super().__init__(seconds)
            budgets.append(self)

    def signal_inside_update(self, *args, **kwargs):
        budgets[0]._signal(signal.SIGTERM, None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rig.worker, "_StopBudget", ControlledBudget)
    monkeypatch.setattr(PPOTrainer, "update", signal_inside_update)
    report = rig.worker.run_request(rig.request)
    assert report["status"] == "stopped" and report["stop_reason"] == "SIGTERM"
    assert report["completed_updates"] == report["attempted_updates"] == report["final_update"] == 1
    assert report["actual_samples"] == 4 and report["partial_samples"] == 0
    assert report["reserved_update_budget"] == 1 and report["reserved_transition_budget"] == 4
    assert load_frame_checkpoint(report["checkpoint"])[3] == 1


def test_collection_failure_counts_completed_environment_steps(rig, monkeypatch):
    original_step = PackedFixture.step

    def invalid_observation(self, action):
        result = original_step(self, action)
        # The collector counts the completed vector step before contract validation.
        result.observation.frame.fill_(float("nan"))
        return result

    monkeypatch.setattr(PackedFixture, "step", invalid_observation)
    with pytest.raises(FloatingPointError, match="observation.frame must be finite"):
        rig.worker.run_request(rig.request)
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert ledger["actual_samples"] == ledger["failed_collection_samples"] == 2
    assert ledger["attempted_updates"] == ledger["completed_updates"] == 0
    assert ledger["failed_optimizer_samples"] == 0
    assert ledger["discarded_samples"] == 2
    assert ledger["reserved_update_budget"] == 2 and ledger["reserved_transition_budget"] == 6
    assert not (rig.directory / "checkpoints" / "final.pt").exists()
    assert rig.environments[0].closed


def test_generic_authorized_resume_preserves_private_and_consumed_clocks(rig, tmp_path):
    rig.anchor(tmp_path)
    first = rig.worker.run_request(rig.request)
    request = deepcopy(rig.request)
    request["checkpoint"].update(path=first["checkpoint"], sha256=first["checkpoint_sha256"],
        update=2, cumulative_transitions=8, consumed_updates=2)
    request["execution"].update(resume=True, run_directory=str(tmp_path / "resumed_worker"))
    second = rig.worker.run_request(reseal(request))
    assert second["completed_updates"] == second["attempted_updates"] == 2
    assert second["final_update"] == 4 and second["actual_samples"] == 8
    assert second["cumulative_transitions"] == 16
    assert second["reserved_update_budget"] == second["reserved_transition_budget"] == 0
    metadata = load_frame_checkpoint(second["checkpoint"])[4]
    assert metadata["continuation"]["clock"] == {
        "consumed_updates": 4, "collected_transitions": 16, "rollout_steps": 2}
    assert metadata["continuation"]["retention"]["draw_count"] == 4
    assert metadata["continuation"]["retention"]["endpoint_draw_count"] == 12
    assert metadata["episode_state_restored"] is False and metadata["history_reset"] == "repeat_first"
    assert len(rig.environments) == 2 and all(env.closed for env in rig.environments)


@pytest.mark.parametrize("field", ["controller_protocol_sha256", "preparation_protocol_sha256",
                                   "preparation_manifest_sha256", "branch_id"])
def test_resume_rejects_changed_controller_branch_identity_before_factory(rig, tmp_path, field):
    first = rig.worker.run_request(rig.request)
    request = deepcopy(rig.request)
    request["checkpoint"].update(path=first["checkpoint"], sha256=first["checkpoint_sha256"],
        update=2, cumulative_transitions=8, consumed_updates=2)
    request["execution"].update(resume=True, run_directory=str(tmp_path / "changed_resume"))
    request["provenance"][field] = "branch_" + "b" * 24 if field == "branch_id" else "3" * 64
    rig.events.clear()
    assert_preflight_rejection(rig, reseal(request))


def test_logging_failure_charges_completed_but_unsealed_update(rig, monkeypatch):
    original = PPOTrainer.update

    def nonfinite_log(self, *args, **kwargs):
        metrics = original(self, *args, **kwargs)
        return {**metrics, "synthetic_nonfinite_metric": float("nan")}

    monkeypatch.setattr(PPOTrainer, "update", nonfinite_log)
    with pytest.raises(ValueError):
        rig.worker.run_request(rig.request)
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert ledger["attempted_updates"] == ledger["completed_updates"] == 1
    assert ledger["actual_samples"] == 4
    assert ledger["unsealed_successful_updates"] == 1
    assert ledger["failed_optimizer_samples"] == ledger["failed_collection_samples"] == 0
    assert not (rig.directory / "checkpoints" / "final.pt").exists()
    assert rig.environments[0].closed


def test_environment_close_failure_does_not_mask_ppo_failure(rig, monkeypatch):
    failure = RuntimeError("original PPO failure before environment shutdown")

    def fail(*args, **kwargs):
        raise failure

    def bad_close(self):
        self.closed = True
        raise OSError("synthetic environment close failure")

    monkeypatch.setattr(PPOTrainer, "update", fail)
    monkeypatch.setattr(PackedFixture, "close", bad_close)
    with pytest.raises(RuntimeError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    assert rig.environments[0].closed
    assert "original PPO failure" in json.loads((rig.directory / "failure.json").read_text())["error"]


def test_failed_second_update_preserves_only_complete_interval_checkpoint(rig, monkeypatch):
    original = PPOTrainer.update
    calls = []
    failure = RuntimeError("synthetic partially mutated second optimizer update")

    def fail_second(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            with torch.no_grad():
                next(self.model.parameters()).add_(.25)
            raise failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PPOTrainer, "update", fail_second)
    with pytest.raises(RuntimeError) as caught:
        rig.worker.run_request(rig.request)
    assert caught.value is failure
    ledger = json.loads((rig.directory / "failure.json").read_text())
    assert ledger["actual_samples"] == 8 and ledger["completed_updates"] == 1
    assert ledger["attempted_updates"] == 2 and ledger["failed_optimizer_samples"] == 4
    assert ledger["optimizer_update_may_be_partial"] is True
    assert ledger["unsealed_successful_updates"] == 0
    assert ledger["reserved_update_budget"] == ledger["reserved_transition_budget"] == 0
    assert ledger["last_sealed_checkpoint"]["update"] == 1
    assert ledger["checkpoint"] is None and ledger["checkpoint_sha256"] is None
    sealed = rig.directory / "checkpoints" / "update_000001.pt"
    assert file_sha(sealed) == ledger["last_sealed_checkpoint"]["sha256"]
    assert load_frame_checkpoint(sealed)[3] == 1
    assert not (rig.directory / "checkpoints" / "update_000002.pt").exists()
    assert not (rig.directory / "checkpoints" / "final.pt").exists()


def test_worker_reclaims_sdk_signal_handler_before_reset_and_restores_caller(rig, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    original_init, original_reset = PackedFixture.__init__, PackedFixture.reset
    checked = []

    def sdk_handler(number, frame):
        raise AssertionError("synthetic SDK signal handler should have been replaced")

    def install_sdk(self, **kwargs):
        original_init(self, **kwargs)
        signal.signal(signal.SIGTERM, sdk_handler)

    def check_reset(self, seed=None):
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not sdk_handler
        assert isinstance(getattr(handler, "__self__", None), rig.worker._StopBudget)
        checked.append(True)
        return original_reset(self, seed)

    monkeypatch.setattr(PackedFixture, "__init__", install_sdk)
    monkeypatch.setattr(PackedFixture, "reset", check_reset)
    rig.worker.run_request(rig.request)
    assert checked == [True]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("field", ["descriptor", "device", "inode", "controller_pid", "controller_start"])
def test_lease_integer_bools_rejected_before_authorization_and_factory(rig, field):
    request = deepcopy(rig.request)
    request["provenance"]["controller_lease"][0][field] = True
    assert_preflight_rejection(rig, reseal(request))
    assert "synthetic_authorization" not in rig.events


@pytest.mark.parametrize("mutation", ["duplicate_path", "duplicate_descriptor", "extra_field"])
def test_lease_identity_shape_rejected_before_authorization_and_factory(rig, mutation):
    request = deepcopy(rig.request)
    leases = request["provenance"]["controller_lease"]
    if mutation == "duplicate_path":
        leases[1]["path"] = leases[0]["path"]
    elif mutation == "duplicate_descriptor":
        leases[1]["descriptor"] = leases[0]["descriptor"]
    else:
        leases[0]["eligible"] = True
    assert_preflight_rejection(rig, reseal(request))
    assert "synthetic_authorization" not in rig.events

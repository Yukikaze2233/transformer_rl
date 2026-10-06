"""Anchor capacity contracts with synthetic tensors; no PPO updates or simulator."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading

import pytest
import torch

from transformer_rl import chassis_adapter, frame_cli, frame_study, frame_workflow
from transformer_rl.frame_checkpoint import save_frame_checkpoint
from transformer_rl.frame_config import digest
from transformer_rl.frame_training import FrameActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.retention import AnchorRegularizer
from test_frame_study import specification
from test_frame_workflow import configuration
from packed_env import make_env


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint(tmp_path):
    config = configuration()
    model = FrameActorCritic(config.model)
    path = tmp_path / "checkpoint.pt"
    save_frame_checkpoint(path, model, PPOTrainer(model, config.ppo), config, 0,
        {"environment_provenance": {"identity": "synthetic_tensor_fixture", "control_sha256": digest(config.control)}})
    return config, model, path


def test_legacy_study_serialization_is_not_normalized_and_explicit_capacity_is_frozen(tmp_path):
    path = specification(tmp_path)
    old = json.loads(path.read_text())
    serialized, identity = json.dumps(old, sort_keys=True), digest(old)
    frame_study._validate_spec(old)
    assert json.dumps(old, sort_keys=True) == serialized and digest(old) == identity
    root = tmp_path / "legacy"
    frame_study.plan_study(path, root)
    assert frame_study.validate_study(root)["spec"] == old
    assert "max_anchors" not in old["training"]
    for capacity in (256, 512):
        value = json.loads(serialized)
        value["training"]["max_anchors"] = capacity
        path.write_text(json.dumps(value))
        root = tmp_path / f"capacity_{capacity}"
        frame_study.plan_study(path, root)
        assert frame_study.validate_study(root)["spec"] == value
        assert digest(value) != identity


@pytest.mark.parametrize("capacity", (0, -1, True, 256., "512", None))
def test_study_rejects_invalid_anchor_capacity(tmp_path, capacity):
    spec = json.loads(specification(tmp_path).read_text())
    spec["training"]["max_anchors"] = capacity
    with pytest.raises(ValueError, match="max_anchors"):
        frame_study._validate_spec(spec)


@pytest.mark.parametrize("capacity", (256, 512))
def test_single_evaluation_collects_and_regularizer_reads_the_whole_pool(tmp_path, monkeypatch, capacity):
    config, model, path = checkpoint(tmp_path)
    environment = {**config.environment, "num_envs": 16}
    anchors = tmp_path / "anchors.pt"
    report = frame_workflow.evaluate_frame_policy(path, make_env, environment,
        steps=64, seed=4101, settle_steps=1, min_steady_samples=1, anchor_output=anchors, max_anchors=capacity)
    receipt = report["anchors"]
    assert receipt["max_samples"] == receipt["samples"] == capacity
    assert receipt["sha256"] == hashlib.sha256(anchors.read_bytes()).hexdigest()
    regularizer = AnchorRegularizer(model.actor, config, [anchors], .2)
    assert regularizer.batch_size == 256  # KL minibatch size is distinct from stored capacity.
    assert len(regularizer.pools[0][0]) == capacity
    loaded = torch.load(anchors, weights_only=True)
    for actual, expected in zip(regularizer.pools[0], (loaded["frames"], loaded["mean"], loaded["std"])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    seen = []
    original_forward = model.actor.policy.forward
    def observe(frames):
        seen.append(frames.clone())
        return original_forward(frames)
    monkeypatch.setattr(model.actor.policy, "forward", observe)
    sampled_bounds = []
    def last_endpoint(high, shape):
        sampled_bounds.append(high)
        return torch.full(shape, high - 1, dtype=torch.long)
    monkeypatch.setattr(torch, "randint", last_endpoint)
    assert regularizer().abs() < 1e-6
    assert sampled_bounds == [1, capacity]
    torch.testing.assert_close(seen[0], loaded["frames"][-1:].expand(256, -1, -1), rtol=0, atol=0)


def suite_fixture(tmp_path, monkeypatch):
    config, model, path = checkpoint(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    configs, outputs = [], []
    for name in ("first", "other"):
        contract = {"evaluation_exact_cases": True, "target_num_envs": 8, "physics_dt": .001,
            "scene_groups": [{"name": name, "fraction": 1.}],
            "evaluation": {"episodes_per_case": 8, "cases": [{"name": name, "terrain": "flat"}]}}
        route = snapshot / f"{name}.json"
        route.write_text(json.dumps(contract))
        environment = {"snapshot": str(snapshot), "snapshot_sha256": "0" * 64, "contract": route.name,
            "contract_sha256": hashlib.sha256(route.read_bytes()).hexdigest(), "num_envs": 8}
        candidate = replace(config, environment=environment)
        config_path = tmp_path / f"config_{name}.json"
        config_path.write_text(json.dumps(candidate.to_dict()))
        configs.append(config_path)
        outputs.append(tmp_path / f"report_{name}.json")
    def grouped(model_config, environment_config, device):
        env = make_env(model_config, {**environment_config, "control": config.control}, device)
        env.metadata["evaluation_groups"] = ["first"] * 8 + ["other"] * 8
        return env
    monkeypatch.setattr(chassis_adapter, "make_env", grouped)
    return config, model, path, configs, outputs


@pytest.mark.parametrize("capacity", (None, 256, 512))
def test_suite_uses_requested_capacity_per_case_and_publishes_actual_pools(tmp_path, monkeypatch, capacity):
    config, model, path, configs, outputs = suite_fixture(tmp_path, monkeypatch)
    calls = []
    evaluate = frame_workflow.evaluate_frame_policy
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(frame_workflow, "evaluate_frame_policy", spy)
    kwargs = {} if capacity is None else {"max_anchors": capacity}
    chassis_adapter.evaluate_suite(path, configs, outputs, steps=64, seed=4101, device="cpu",
        settle_steps=1, min_steady_samples=1, anchor_directory=tmp_path / "anchors", **kwargs)
    expected = 256 if capacity is None else capacity
    assert calls[0]["max_anchors"] == expected
    reports = []
    for output in outputs:
        report = json.loads(output.read_text())
        reports.append(report)
        assert report["num_envs"] == 8 and report["transitions"] == 512
        assert report["anchors"]["max_samples"] == report["anchors"]["samples"] == expected
        pool = AnchorRegularizer(model.actor, config, [report["anchors"]["path"]], .2)
        assert len(pool.pools[0][0]) == expected
        assert pool.identities[0]["sha256"] == report["anchors"]["sha256"]
    combined = AnchorRegularizer(model.actor, config, [report["anchors"]["path"] for report in reports], .2)
    assert len(combined.pools) == 2
    assert sum(len(pool[0]) for pool in combined.pools) == 2 * expected
    assert combined.batch_size == 256


@pytest.mark.parametrize("capacity", (0, -1, True, 256., "512", None))
def test_suite_api_rejects_invalid_capacity_before_loading_checkpoint(tmp_path, capacity):
    with pytest.raises(ValueError, match="max_anchors"):
        chassis_adapter.evaluate_suite(tmp_path / "unused", [tmp_path / "unused"], [tmp_path / "unused"],
            steps=1, seed=1, device="cpu", settle_steps=1, min_steady_samples=1, max_anchors=capacity)


def test_capacity_is_an_upper_bound_and_absent_without_anchor_collection(tmp_path):
    config, _, path = checkpoint(tmp_path)
    report = frame_workflow.evaluate_frame_policy(path, make_env, config.environment,
        steps=2, seed=4101, settle_steps=1, min_steady_samples=1,
        anchor_output=tmp_path / "anchors.pt", max_anchors=512)
    assert report["anchors"]["samples"] == 6 and report["anchors"]["max_samples"] == 512
    report = frame_workflow.evaluate_frame_policy(path, make_env, config.environment,
        steps=2, seed=4101, settle_steps=1, min_steady_samples=1, max_anchors=512)
    assert "anchors" not in report and "max_anchors" not in report and "max_samples" not in report


@pytest.mark.parametrize("capacity", (None, 256, 512))
def test_suite_cli_passes_legacy_default_or_requested_capacity(tmp_path, monkeypatch, capacity):
    calls = []
    monkeypatch.setattr(chassis_adapter, "evaluate_suite", lambda *args, **kwargs: calls.append(kwargs) or {})
    arguments = ["evaluate-suite", "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--configs", str(tmp_path / "config.json"), "--outputs", str(tmp_path / "output.json"),
        "--steps", "64", "--seed", "4101", "--anchor-directory", str(tmp_path / "anchors")]
    if capacity is not None:
        arguments += ["--max-anchors", str(capacity)]
    assert frame_cli.main(arguments) == 0
    assert calls[0]["max_anchors"] == (256 if capacity is None else capacity)


@pytest.mark.parametrize("capacity", ("0", "-1", "1.5"))
def test_suite_cli_rejects_invalid_capacity_without_calling_evaluator(tmp_path, monkeypatch, capacity):
    monkeypatch.setattr(chassis_adapter, "evaluate_suite", lambda *a, **k: pytest.fail("evaluator started"))
    with pytest.raises(SystemExit):
        frame_cli.main(["evaluate-suite", "--checkpoint", "unused", "--configs", "unused",
            "--outputs", "unused", "--steps", "1", "--seed", "1", "--max-anchors", capacity])


def scheduler_fixture(tmp_path, monkeypatch, *, capacity, batch, coefficient=.2, change_report=None):
    spec = json.loads(specification(tmp_path).read_text())
    spec["training"]["retention_coef"] = coefficient
    if capacity is not None:
        spec["training"]["max_anchors"] = capacity
    if batch:
        spec["environment_factory"] = "transformer_rl.chassis_adapter:make_env"
        for scenario in spec["scenarios"]:
            scenario["environment"]["evaluation_batch"] = "combined"
    frame_study._validate_spec(spec)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    checkpoint_path = attempt / "checkpoint.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint identity, never loaded")
    pending = {"directory": "attempt", "checkpoint": frame_study._receipt(checkpoint_path, tmp_path)}
    state = {"status": "running", "protected": {}, "rollbacks": 0, "stages": [pending]}
    calls = []
    def process(command, *args):
        calls.append(command)
        option = lambda name: command[command.index(name) + 1]
        paths = command[command.index("--outputs") + 1:command.index("--outputs") + 1 + len(spec["scenarios"])] if "evaluate-suite" in command else [option("--output")]
        for path in paths:
            report = {"checkpoint_sha256": pending["checkpoint"]["sha256"], "completed_episodes": 8,
                "success_rate": 1., "metrics": {"tracking_error": {"mean": 0.}}}
            if "--anchor-directory" in command or "--anchor-output" in command:
                anchor = Path(path).with_suffix(".pt")
                anchor.write_bytes(b"synthetic anchor identity, never loaded")
                report["anchors"] = {"path": str(anchor), "sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(), "samples": 24}
                if capacity is not None:
                    report["anchors"]["max_samples"] = capacity
                if change_report:
                    change_report(report["anchors"])
            Path(path).write_text(json.dumps(report))
        return True
    monkeypatch.setattr(frame_study, "_process", process)
    def evaluate(anchor_only):
        return frame_study._evaluate_attempt(tmp_path, {"spec": spec}, {"name": "transformer"},
            [case["name"] for case in spec["scenarios"]], pending, ["synthetic_worker"], "cpu",
            threading.Event(), tmp_path / "state.json", state, anchor_only=anchor_only)
    return evaluate, calls, pending


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("capacity", (None, 256, 512))
def test_scheduler_passes_capacity_only_to_anchor_collection_and_accepts_legacy_reports(tmp_path, monkeypatch, batch, capacity):
    evaluate, calls, pending = scheduler_fixture(tmp_path, monkeypatch, capacity=capacity, batch=batch)
    assert evaluate(False)
    assert all("--max-anchors" not in call and "--anchor-output" not in call and "--anchor-directory" not in call for call in calls)
    calls.clear()
    assert evaluate(True) and pending["anchors_complete"] and len(pending["anchors"]) == 2
    assert all(("evaluate-suite" in call) == batch for call in calls)
    for call in calls:
        assert ("--max-anchors" in call) == (capacity is not None)
        if capacity is not None:
            assert call[call.index("--max-anchors") + 1] == str(capacity)


@pytest.mark.parametrize("change_report", (
    lambda report: report.pop("max_samples"),
    lambda report: report.update(max_samples=256),
    lambda report: report.update(max_samples=True),
    lambda report: report.update(samples=513),
    lambda report: report.update(samples=0),
    lambda report: report.update(samples=True),
    lambda report: report.update(sha256="0" * 64),
))
def test_scheduler_rejects_false_capacity_or_anchor_identity(tmp_path, monkeypatch, change_report):
    evaluate, _, pending = scheduler_fixture(tmp_path, monkeypatch, capacity=512, batch=True, change_report=change_report)
    with pytest.raises(ValueError, match="anchor"):
        evaluate(True)
    assert not pending.get("anchors_complete")


def test_disabled_retention_never_collects_or_claims_effective_capacity(tmp_path, monkeypatch):
    evaluate, calls, pending = scheduler_fixture(tmp_path, monkeypatch, capacity=512, batch=True, coefficient=0.)
    assert evaluate(False)
    assert all("--max-anchors" not in call and "--anchor-directory" not in call for call in calls)
    assert "anchors" not in pending and "anchor_reports" not in pending and "anchors_complete" not in pending
    count = len(calls)
    with pytest.raises(ValueError, match="enabled retention"):
        evaluate(True)
    assert len(calls) == count

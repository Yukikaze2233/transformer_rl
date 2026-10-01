"""Workflow evidence checks using synthetic tensor workers only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import pytest
import torch

from transformer_rl.frame_config import digest
from transformer_rl import frame_study as study
from transformer_rl.frame_workflow import evaluate_frame_policy, train_frame_policy
from transformer_rl.chassis_adapter import merge_evaluation_contracts
from test_frame_workflow import configuration
from packed_env import make_env


def specification(tmp_path):
    config = configuration()
    (tmp_path / "base.json").write_text(json.dumps(config.to_dict()))
    spec = {"base_config": "base.json", "environment_factory": "packed_env:make_env",
        "variants": [{"name": "transformer", "policy": {}}], "seeds": [71, 97],
        "stages": [{"name": "first", "updates": 1, "environment": {}, "scenarios": ["normal"]},
                   {"name": "second", "updates": 1, "environment": {"stage": "second"}, "scenarios": ["new_skill"]}],
        "scenarios": [{"name": name, "environment": {"case": name},
            "gates": [{"path": "success_rate", "operator": "min", "value": 1.}]} for name in ("normal", "new_skill")],
        "training": {"rollout_steps": 4, "checkpoint_interval": 1, "max_seconds": 60., "retention_coef": .2,
                     "anchor_seeds": [4101]},
        "evaluation": {"validation_seeds": [701], "seeds": [2701], "steps": 12, "settle_steps": 1,
                       "min_steady_samples": 1, "min_completed_episodes": 6},
        "execution": {"devices": ["cpu"], "worker_module": "transformer_rl.frame_process", "job_timeout_seconds": 60.},
        "selection": {"min_training_seeds": 2, "objectives": [{"path": "metrics.tracking_error.mean",
             "direction": "minimize", "scale": 1., "weight": 1.}], "std_penalty": 1.,
             "latency_p99_ms": 9.9, "latency_max_ms": 1000., "max_deadline_misses": 1000,
             "retention_score_tolerance": .1, "rollback_limit": 1}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_real_worker_curriculum_eval_retry_selection_and_integrity(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("tensorboard")
    fixture_dir = Path(__file__).parent / "fixtures"
    source_dir = Path(__file__).parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(fixture_dir), str(source_dir))))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    root = tmp_path / "study"
    study.plan_study(specification(tmp_path), root)
    manifest = study.validate_study(root, source=True)
    variant = manifest["spec"]["variants"][0]
    real_process, interrupted_once = study._process, []

    def lose_first_evaluation(command, *args):
        if "evaluate" in command and not interrupted_once:
            interrupted_once.append(True)
            return False
        return real_process(command, *args)

    monkeypatch.setattr(study, "_process", lose_first_evaluation)
    first = study._run_job(root, manifest, variant, 71, "cpu", threading.Event())
    assert first["status"] == "failed"
    assert first["stages"][0]["consumed_updates"] == 1
    assert first["stages"][0]["attempts"][0]["status"] == "trained"
    recovered = study._run_job(root, manifest, variant, 71, "cpu", threading.Event())
    assert recovered["status"] == "completed"
    assert len(recovered["stages"][0]["attempts"]) == 1
    assert recovered["stages"][0]["attempts"][0]["evaluation_attempts"] == 2
    protected_attempt = recovered["stages"][0]["attempts"][0]
    assert set(protected_attempt["anchor_reports"]) == {"normal/4101"}
    eval_report = study._load_receipt(root, protected_attempt["evaluations"]["normal/701"])
    assert "anchors" not in eval_report
    assert set(recovered["final_evaluations"]) == {"normal/2701", "new_skill/2701"}
    run_receipt = json.loads((root / recovered["stages"][1]["attempts"][0]["directory"] / "train/run.json").read_text())
    assert run_receipt["initialize_from"] and run_receipt["retention_coef"] == .2
    # An incomplete comparison cannot select a single lucky seed.
    incomplete = study.select_transformer(root)
    assert incomplete["best_transformer"] is None
    assert "missing seed 97" in incomplete["candidates"][0]["reasons"]
    second = study._run_job(root, manifest, variant, 97, "cpu", threading.Event())
    assert second["status"] == "completed"
    report = study.select_transformer(root)
    winner = report["best_transformer"]
    assert winner and winner["score_sample_std"] is not None
    assert winner["representative_seed"] in (71, 97)
    assert report["transformer_is_best_overall"]
    # Completed jobs are idempotent; their checkpoints and ledgers are not overwritten.
    assert study._run_job(root, manifest, variant, 71, "cpu", threading.Event()) == recovered
    graph = root / winner["deployment"]["path"]
    onnx_path = graph.parent / "policy.onnx"
    onnx_path.write_bytes(onnx_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="graph evidence"):
        study.select_transformer(root)


def test_grade_requires_task_completion_and_post_settle_samples():
    scenario = {"gates": [{"path": "success_rate", "operator": "min", "value": .95}], "require_steady": True}
    evaluation = {"min_completed_episodes": 5}
    objectives = [{"path": "metrics.height.mean", "direction": "minimize", "scale": .03, "weight": 1.}]
    report = {"completed_episodes": 6, "success_rate": 1., "stability": {"available": True}, "metrics": {"height": {"mean": .02}}}
    assert study.grade_report(report, scenario, evaluation, objectives)["passed"]
    report["stability"]["available"] = False
    assert not study.grade_report(report, scenario, evaluation, objectives)["passed"]
    assert study.grade_report(report, {**scenario, "require_steady": False}, evaluation, objectives)["passed"]
    report["metrics"]["height"]["mean"] = None
    assert study.grade_report(report, scenario, evaluation, objectives)["score"] is None


def test_frozen_plan_rejects_configuration_and_package_changes(tmp_path):
    root = tmp_path / "study"
    study.plan_study(specification(tmp_path), root)
    config = root / "configs/transformer.train.first.json"
    value = json.loads(config.read_text())
    value["ppo"]["learning_rate"] *= 2
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuration changed"):
        study.validate_study(root)


def test_unsealed_worker_crash_charges_budget_once(tmp_path):
    root = tmp_path
    attempt = root / "attempt"
    attempt.mkdir()
    pending = {"directory": "attempt", "requested_updates": 4, "status": "running"}
    entry = {"consumed_updates": 0}
    assert not study._account_training(root, entry, pending, "unused")
    assert not study._account_training(root, entry, pending, "unused")
    assert entry["consumed_updates"] == 4 and pending["budget_charged"]


def test_worker_timeout_reaps_its_process_group(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    log = tmp_path / "worker.log"
    assert not study._process(command, log, .1, threading.Event())
    record = json.loads(Path(str(log) + ".process.json").read_text())
    assert record["status"] == "finished" and study._process_start(record["pid"]) is None


def test_grouped_evaluation_keeps_per_case_counts_and_statistics(tmp_path):
    config = configuration()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trained = train_frame_policy(config, make_env, "packed_env:make_env", tmp_path / "train", updates=1,
                                     rollout_steps=4, tensorboard=False)
        def grouped(model_config, environment_config, device):
            env = make_env(model_config, environment_config, device)
            env.metadata["evaluation_groups"] = ["first", "other", "other"]
            return env
        report = evaluate_frame_policy(trained["checkpoint"], grouped, config.environment, steps=12, seed=71,
            settle_steps=1, min_steady_samples=1, group_anchor_directory=tmp_path / "anchors", max_anchors=8)
        first, other = report["groups"]["first"], report["groups"]["other"]
        assert first["num_envs"] == 1 and other["num_envs"] == 2
        assert first["completed_episodes"] == 4 and other["completed_episodes"] == 6
        assert first["stability"]["signals"]["tracking_error"]["count"] == 8
        assert other["stability"]["signals"]["tracking_error"]["count"] == 18
        assert first["anchors"]["samples"] <= 8 and other["anchors"]["samples"] <= 8
    finally:
        torch.set_num_threads(previous_threads)


def test_batch_contract_merge_preserves_dynamics_and_fixed_case_layout(tmp_path):
    environments = []
    for name in ("first", "second"):
        config = {"evaluation_exact_cases": True, "target_num_envs": 2, "physics_dt": .001,
                  "scene_groups": [{"name": name, "fraction": 1.}],
                  "evaluation": {"episodes_per_case": 2, "cases": [{"name": name, "terrain": "flat"}]}}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        environments.append({"contract": path.name, "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "num_envs": 2})
    merged = merge_evaluation_contracts(tmp_path, environments)
    assert merged["target_num_envs"] == 4
    assert [case["name"] for case in merged["evaluation"]["cases"]] == ["first", "second"]
    value = json.loads((tmp_path / "second.json").read_text())
    value["physics_dt"] = .002
    (tmp_path / "second.json").write_text(json.dumps(value))
    environments[1]["contract_sha256"] = hashlib.sha256((tmp_path / "second.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="differ in dynamics"):
        merge_evaluation_contracts(tmp_path, environments)


def test_frozen_worker_source_is_hash_checked(tmp_path):
    root = tmp_path / "study"
    study.plan_study(specification(tmp_path), root)
    policy = root / "policy_source/transformer_rl/frame_policy.py"
    policy.write_text(policy.read_text() + "\n# accidental modification\n")
    with pytest.raises(ValueError, match="frozen policy source changed"):
        study.validate_study(root)


def test_sdk_adapter_requires_dedicated_worker_before_imports():
    from transformer_rl.chassis_adapter import make_env as make_chassis
    imported = {name for name in sys.modules if name.startswith("isaaclab")}
    with pytest.raises(RuntimeError, match="frame_process"):
        make_chassis(configuration().model, {}, "cpu")
    assert imported == {name for name in sys.modules if name.startswith("isaaclab")}


def test_retention_regression_restores_parent_without_refunding_budget(tmp_path, monkeypatch):
    spec_path = specification(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["training"]["retention_coef"] = 0.
    spec["stages"][1]["updates"] = 3
    spec_path.write_text(json.dumps(spec))
    root = tmp_path / "study"
    study.plan_study(spec_path, root)
    manifest = study.validate_study(root)
    commands, training_count = [], []
    def option(command, name):
        return command[command.index(name) + 1]
    def synthetic_receipts(command, log, seconds, stop):
        commands.append(command)
        if "train" in command:
            training_count.append(1)
            directory = Path(option(command, "--run-dir"))
            directory.mkdir()
            checkpoint = directory / "synthetic_state.pt"
            checkpoint.write_bytes(f"synthetic-state-{len(training_count)}".encode())
            study._write(directory / "completion.json", {"status": "completed", "attempted_updates": 1,
                "completed_updates": 1, "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "config_sha256": digest(study._read(option(command, "--config")))})
        elif "evaluate" in command:
            checkpoint = Path(option(command, "--checkpoint"))
            error = .5 if len(training_count) == 2 else 0.
            study._write(option(command, "--output"), {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "completed_episodes": 10, "success_rate": 1., "stability": {"available": True},
                "metrics": {"tracking_error": {"mean": error}}})
        else:
            # This test covers the curriculum ledger, not export evidence.
            return False
        return True
    monkeypatch.setattr(study, "_process", synthetic_receipts)
    state = study._run_job(root, manifest, spec["variants"][0], 71, "cpu", threading.Event())
    stage = state["stages"][1]
    assert state["rollbacks"] == 1 and stage["consumed_updates"] == 3
    assert len(stage["attempts"]) == 3 and stage["attempts"][0]["retention_regression"]
    training = [command for command in commands if "train" in command]
    assert "--initialize-from" in training[1]
    assert "--restore-learning-from" in training[2]
    restored = option(training[2], "--restore-learning-from")
    assert restored == str(root / state["stages"][0]["checkpoint"]["path"])
    assert option(training[2], "--consumed-update-offset") == "1"
    assert "--resume" in training[3]
    assert state["final_evaluation_complete"]

"""Transfer queue ownership and actual PPO evidence without simulator imports."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).parents[1] / "tools/run_transfer_campaign.py"
SPEC = importlib.util.spec_from_file_location("transfer_campaign", PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
control = campaign.control


def test_atomic_sealed_json_is_never_replaced(tmp_path):
    path = tmp_path / "sealed.json"
    campaign.write(path, {"value": 1})
    with pytest.raises(FileExistsError):
        campaign.write(path, {"value": 2})
    assert control.read(path) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))
    campaign.write(path, {"value": 3}, replace=True)
    assert control.read(path) == {"value": 3}


def fixture(tmp_path, *, dependency_status="completed"):
    study, output, source = tmp_path / "new", tmp_path / "output", tmp_path / "source"
    predecessor = tmp_path / "old"
    predecessor.mkdir()
    resource = predecessor / ".run.lock"
    resource.touch()
    receipt = tmp_path / "old_control/summary.json"
    control.write(receipt, {"status": dependency_status})
    (source / "src/transformer_rl").mkdir(parents=True)
    (source / "src/transformer_rl/frame_process.py").write_text("# synthetic source\n")
    spec = {"variants": [{"name": "mlp"}, {"name": "transformer"}], "seeds": [1101],
            "stages": [{"name": "transfer", "updates": 3, "scenarios": ["stand", "delay"]}],
            "evaluation": {"seeds": [1701], "validation_seeds": [701]}, "training": {"anchor_seeds": []}}
    plan = {"spec": spec, "configs": {}}
    for variant in spec["variants"]:
        for scenario in spec["stages"][0]["scenarios"]:
            route = f"configs/{variant['name']}.eval.{scenario}.json"
            configuration = {"variant": variant["name"], "scenario": scenario}
            control.write(study / route, configuration)
            plan["configs"][route] = control.digest(configuration)
    plan["sha256"] = control.digest(plan)
    control.write(study / "plan.json", plan)
    args = argparse.Namespace(study_root=study, output_root=output, source_root=source,
        resource_lock=resource, dependency_receipt=receipt, updates=3, seeds=[5701], steps=401,
        device="cpu", benchmark_iterations=1000, max_run_attempts=3,
        max_wait_seconds=0., worker_timeout_seconds=60., poll_seconds=.01)
    return args, plan


def seal_training(study, plan, variant, final, *, status="rejected", stop_reason=None):
    path = study / "jobs" / variant / "seed_1101/state.json"
    if path.exists():
        state = control.read(path)
        attempts = state["stages"][0]["attempts"]
        start = control.read(control.checked(study, attempts[-1]["training"]))["final_update"]
    else:
        state = {"variant": variant, "seed": 1101, "plan_sha256": plan["sha256"],
                 "stages": [{"name": "transfer", "attempts": []}]}
        attempts, start = state["stages"][0]["attempts"], 0
    directory = study / "jobs" / variant / "seed_1101/transfer" / f"attempt_{len(attempts):04d}"
    checkpoint = directory / "train/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{variant}-{final}".encode())
    completion = directory / "train/completion.json"
    control.write(completion, {"status": "stopped" if stop_reason else "completed", "stop_reason": stop_reason,
        "start_update": start, "final_update": final, "completed_updates": final - start,
        "checkpoint": str(checkpoint), "checkpoint_sha256": control.file_sha(checkpoint)})
    records = [{"update": number, "batch_samples": 64,
                "optimization": {"optimizer_steps": 5, "grad_norm": .1}}
               for number in range(start + 1, final + 1)]
    (directory / "train/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    attempts.append({"directory": str(directory.relative_to(study)), "budget_charged": True,
                     "training": control.artifact(completion, study), "checkpoint": control.artifact(checkpoint, study)})
    state["status"] = status
    control.write(path, state, replace=path.exists())


def synthetic_worker(args, plan, calls, *, partial_first=False, incomplete_variant=None, tampered_benchmark=False):
    def worker(invocation, directory, environment, timeout, heartbeat):
        calls.append(invocation)
        operation = invocation[3]
        assert "transformer_rl.frame_process" in invocation[2]
        option = lambda name: invocation[invocation.index(name) + 1]
        if operation == "run":
            assert invocation[-2:] == ["--max-parallel", "1"]
            # The external lock is held; the new study lock is available to its executor.
            with args.resource_lock.open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (args.study_root / ".run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_count = sum(call[3] == "run" for call in calls)
            for variant in ("mlp", "transformer"):
                final = 1 if (partial_first and run_count == 1) or variant == incomplete_variant else 3
                state = args.study_root / "jobs" / variant / "seed_1101/state.json"
                if state.exists():
                    prior = campaign.training_snapshot(args.study_root, plan)[f"{variant}/seed_1101"]["checkpoint"]["update"]
                    if final <= prior:
                        continue
                seal_training(args.study_root, plan, variant, final,
                              status="failed" if final < 3 else "rejected",
                              stop_reason="time_budget" if final < 3 else None)
            heartbeat()
            return {"returncode": 2 if partial_first and run_count == 1 else 0, "timed_out": False}
        if operation == "evaluate-suite":
            checkpoint = Path(option("--checkpoint"))
            outputs = invocation[invocation.index("--outputs") + 1:invocation.index("--steps")]
            seed = int(option("--seed"))
            for output in outputs:
                control.write(Path(output), {"seed": seed, "checkpoint_sha256": control.file_sha(checkpoint),
                                             "success_rate": 0., "metrics": {"height_error": {"mean": .2}}})
            control.write(directory / "control.json", {"seed": seed,
                          "checkpoint_sha256": control.file_sha(checkpoint), "control": {"rmse": .2}})
            (directory / "trace.npz").write_bytes(b"synthetic trace")
        elif operation == "export":
            bundle = Path(option("--directory"))
            bundle.mkdir()
            runtime = bundle / "policy.onnx"
            runtime.write_bytes(b"synthetic runtime")
            control.write(bundle / "manifest.json", {"checkpoint_sha256": control.file_sha(option("--checkpoint")),
                          "files": {"policy.onnx": control.file_sha(runtime)}})
        elif operation == "benchmark":
            manifest = Path(option("--directory")) / "manifest.json"
            control.write(Path(option("--output")), {"manifest_sha256": "wrong" if tampered_benchmark else control.file_sha(manifest),
                "backend": "onnx", "threads": 1, "iterations": 1000,
                "mean_ms": .2, "p99_ms": .4, "max_ms": .5, "deadline_misses": 0})
        else:
            raise AssertionError(operation)
        return {"returncode": 0, "timed_out": False}
    return worker


def test_dependency_wait_does_not_launch_or_change_predecessor(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path, dependency_status="running")
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls))
    original = args.dependency_receipt.read_bytes()
    summary = campaign.run_campaign(args)
    assert summary["status"] == "blocked" and not calls
    assert args.dependency_receipt.read_bytes() == original


def test_external_resource_lock_blocks_even_after_dependency_completion(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls))
    with args.resource_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert campaign.run_campaign(args)["status"] == "blocked"
    assert not calls


def test_lock_must_not_equal_new_executor_lock(tmp_path):
    args, _ = fixture(tmp_path)
    args.resource_lock = args.study_root / ".run.lock"
    with pytest.raises(ValueError, match="must differ"):
        campaign.run_campaign(args)


def test_legal_time_budget_resume_proves_actual_ppo_and_checks_rejected_models(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls, partial_first=True))
    summary = campaign.run_campaign(args)
    assert summary["status"] == "completed"
    assert sum(call[3] == "run" for call in calls) == 2
    assert sum(call[3] == "evaluate-suite" for call in calls) == 2
    assert not any(call[3] == "select" for call in calls)
    for value in summary["training"].values():
        assert value["status"] == "rejected" and value["ppo_verified"]
        assert value["sealed_updates"] == 3 and value["batch_samples"] == 192 and value["optimizer_steps"] == 15
    assert not summary["target_hardware_verified"]
    sealed = {str(path): path.read_bytes() for path in args.output_root.rglob("receipt.json")}
    count = len(calls)
    assert campaign.run_campaign(args)["status"] == "completed" and len(calls) == count
    assert sealed == {str(path): path.read_bytes() for path in args.output_root.rglob("receipt.json")}


def test_failed_partial_model_retains_latest_endpoint_after_bounded_retries(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls, incomplete_variant="transformer"))
    summary = campaign.run_campaign(args)
    assert summary["status"] == "incomplete" and not summary["training_target_completed"]
    assert sum(call[3] == "run" for call in calls) == args.max_run_attempts
    model = summary["results"]["transformer/seed_1101"]
    assert model["training"]["checkpoint"]["update"] == 1
    assert model["evaluations"]["5701"]["status"] == "completed"
    assert model["deployment"]["status"] == "completed"


def test_pid_without_optimization_evidence_is_not_training(tmp_path):
    args, plan = fixture(tmp_path)
    path = args.study_root / "jobs/mlp/seed_1101/state.json"
    control.write(path, {"variant": "mlp", "seed": 1101, "plan_sha256": plan["sha256"],
                        "status": "running", "stages": []})
    item = campaign.training_snapshot(args.study_root, plan)["mlp/seed_1101"]
    assert not item["ppo_verified"] and item["checkpoint"] is None


def test_live_applied_updates_are_distinct_from_sealed_progress(tmp_path):
    args, plan = fixture(tmp_path)
    directory = args.study_root / "jobs/mlp/seed_1101/transfer/attempt_0000"
    (directory / "train").mkdir(parents=True)
    (directory / "train/metrics.jsonl").write_text(
        '{"update": 1, "batch_samples": 8, "optimization": {"optimizer_steps": 2}}\n')
    control.write(args.study_root / "jobs/mlp/seed_1101/state.json", {"variant": "mlp", "seed": 1101,
        "plan_sha256": plan["sha256"], "status": "running", "stages": [{"name": "transfer",
        "attempts": [{"directory": str(directory.relative_to(args.study_root))}]}]})
    item = campaign.training_snapshot(args.study_root, plan)["mlp/seed_1101"]
    assert item["live_ppo_verified"] and item["live_updates"] == 1 and item["live_optimizer_steps"] == 2
    assert item["sealed_updates"] == 0 and item["checkpoint"] is None and not item["ppo_verified"]


def test_sealed_receipt_requires_exact_optimization_log(tmp_path):
    args, plan = fixture(tmp_path)
    seal_training(args.study_root, plan, "mlp", 3)
    path = next((args.study_root / "jobs").rglob("metrics.jsonl"))
    path.write_text(path.read_text().splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="receipt and PPO records differ"):
        campaign.training_snapshot(args.study_root, plan)


def test_live_partial_line_is_not_counted_as_an_applied_update(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"update": 1, "batch_samples": 8, "optimization": {"optimizer_steps": 1}}\n{"update":')
    assert campaign.metrics_evidence(path)["updates"] == 1


def test_resume_fork_and_checkpoint_hash_mismatch_are_rejected(tmp_path):
    args, plan = fixture(tmp_path)
    seal_training(args.study_root, plan, "mlp", 1, status="failed", stop_reason="time_budget")
    seal_training(args.study_root, plan, "mlp", 3)
    checkpoint = Path(campaign.training_snapshot(args.study_root, plan)["mlp/seed_1101"]["checkpoint"]["checkpoint"])
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        campaign.training_snapshot(args.study_root, plan)


def test_resume_branch_start_must_match_previous_sealed_endpoint(tmp_path):
    args, plan = fixture(tmp_path)
    seal_training(args.study_root, plan, "mlp", 1, status="failed", stop_reason="time_budget")
    seal_training(args.study_root, plan, "mlp", 3)
    state_path = args.study_root / "jobs/mlp/seed_1101/state.json"
    state = control.read(state_path)
    attempt = state["stages"][0]["attempts"][-1]
    completion = control.checked(args.study_root, attempt["training"])
    receipt = control.read(completion)
    receipt["start_update"] = 0
    control.write(completion, receipt, replace=True)
    attempt["training"] = control.artifact(completion, args.study_root)
    control.write(state_path, state, replace=True)
    with pytest.raises(ValueError, match="continuous resume chain"):
        campaign.training_snapshot(args.study_root, plan)


def test_tampered_cpu_benchmark_does_not_claim_deployment_success(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls, tampered_benchmark=True))
    summary = campaign.run_campaign(args)
    assert summary["status"] == "incomplete"
    assert summary["results"]["mlp/seed_1101"]["deployment"]["status"] == "failed"


def test_live_predecessor_and_orphan_workers_block_resource_use(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, calls))
    monkeypatch.setattr(control, "process_start", lambda pid: "456")
    path = args.dependency_receipt.parent / "old/worker.process.json"
    control.write(path, {"status": "running", "pid": 123, "start": "456"})
    assert campaign.run_campaign(args)["status"] == "blocked" and not calls
    path.unlink()
    path = args.output_root / "executor/attempt_0000/worker.process.json"
    control.write(path, {"status": "running", "pid": 124, "start": "456"})
    assert campaign.run_campaign(args)["status"] == "blocked" and not calls


def test_source_and_eval_config_changes_reject_reuse(tmp_path, monkeypatch):
    args, plan = fixture(tmp_path)
    monkeypatch.setattr(control, "run_worker", synthetic_worker(args, plan, []))
    assert campaign.run_campaign(args)["status"] == "completed"
    (args.source_root / "src/transformer_rl/frame_process.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="protocol or source changed"):
        campaign.run_campaign(args)


def test_real_worker_timeout_uses_only_owned_process(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    result = control.run_worker([sys.executable, "-c", "import time; time.sleep(60)"], directory,
                               {"PYTHONPATH": "synthetic"}, .1, lambda: None)
    assert result["timed_out"] and result["returncode"] != 0
    receipt = control.read(directory / "worker.process.json")
    assert receipt["start"] and receipt["status"] == "finished"
    assert control.process_start(receipt["pid"]) is None

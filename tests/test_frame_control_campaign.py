"""Campaign scheduling and artifact ownership with synthetic worker reports."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools/run_frame_control_campaign.py"
SPEC = importlib.util.spec_from_file_location("frame_control_campaign", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def fixture(tmp_path, counts=(200, 200), statuses=("completed", "rejected")):
    study, output, source = tmp_path / "study", tmp_path / "output", tmp_path / "source"
    source.mkdir()
    (source / "src/transformer_rl").mkdir(parents=True)
    (source / "src/transformer_rl/frame_process.py").write_text("# synthetic source identity\n")
    names = ["mlp", "transformer"]
    scenarios = ["stand", "move"]
    spec = {"variants": [{"name": name} for name in names], "seeds": [1101],
            "stages": [{"name": "s1", "scenarios": scenarios}],
            "training": {"anchor_seeds": [4101]},
            "evaluation": {"seeds": [2701], "validation_seeds": [701]}}
    plan = {"spec": spec, "configs": {}}
    for name in names:
        for scenario in scenarios:
            route = f"configs/{name}.eval.{scenario}.json"
            config = {"model": name, "scenario": scenario}
            campaign.write(study / route, config)
            plan["configs"][route] = campaign.digest(config)
    plan["sha256"] = campaign.digest(plan)
    campaign.write(study / "plan.json", plan)
    (study / ".run.lock").touch()
    for name, count, status in zip(names, counts, statuses):
        relative = f"jobs/{name}/seed_1101/s1/attempt_0001"
        checkpoint = study / relative / "train/checkpoints/final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(name.encode())
        completion = checkpoint.parent.parent / "completion.json"
        campaign.write(completion, {"status": "completed", "final_update": count,
                                    "checkpoint": str(checkpoint),
                                    "checkpoint_sha256": campaign.file_sha(checkpoint)})
        attempt = {"directory": relative, "budget_charged": True,
                   "training": campaign.artifact(completion, study),
                   "checkpoint": campaign.artifact(checkpoint, study)}
        state = {"variant": name, "seed": 1101, "plan_sha256": plan["sha256"],
                 "status": status, "stages": [{"name": "s1", "attempts": [attempt]}]}
        campaign.write(study / f"jobs/{name}/seed_1101/state.json", state)
    args = argparse.Namespace(study_root=study, output_root=output, source_root=source,
                              updates=200, training_seed=1101, seeds=[5701, 6701], steps=4001,
                              device="cpu", max_wait_seconds=0., worker_timeout_seconds=60., poll_seconds=.01)
    return args, plan


def synthetic_worker(calls, fail_variant=None):
    def worker(command, directory, environment, timeout, heartbeat):
        calls.append(command)
        option = lambda name: command[command.index(name) + 1]
        checkpoint = Path(option("--checkpoint"))
        if fail_variant and fail_variant in checkpoint.parts:
            (directory / "partial-output.txt").write_text("retained partial output")
            return {"returncode": 1, "timed_out": False}
        seed = int(option("--seed"))
        outputs = command[command.index("--outputs") + 1:command.index("--steps")]
        for output in outputs:
            campaign.write(Path(output), {"seed": seed, "checkpoint_sha256": campaign.file_sha(checkpoint),
                                          "success_rate": 1., "metrics": {"height_error": {"mean": .01}}})
        campaign.write(directory / "control.json", {"seed": seed,
                       "checkpoint_sha256": campaign.file_sha(checkpoint), "task_metrics": {"height_rmse": .01}})
        (directory / "trace.npz").write_bytes(b"synthetic trace archive")
        assert environment["PYTHONPATH"].split(os.pathsep)[0].endswith("/source/src")
        assert "alter_sys=True" in command[2]
        return {"returncode": 0, "timed_out": False}
    return worker


def test_ready_checkpoint_keeps_rejected_model_and_checks_sealed_hashes(tmp_path):
    args, plan = fixture(tmp_path)
    result = campaign.ready_checkpoint(args.study_root, "transformer", 1101, 200, plan)
    assert result["status"] == "ready" and result["old_job_status"] == "rejected"
    Path(result["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.ready_checkpoint(args.study_root, "transformer", 1101, 200, plan)


def test_partial_and_unsealed_training_cannot_become_ready(tmp_path):
    args, plan = fixture(tmp_path, counts=(162, 200))
    assert campaign.ready_checkpoint(args.study_root, "mlp", 1101, 200, plan)["status"] == "not_ready"
    state_path = args.study_root / "jobs/transformer/seed_1101/state.json"
    state = campaign.read(state_path)
    state["stages"][0]["attempts"][0]["budget_charged"] = False
    campaign.write(state_path, state, replace=True)
    assert campaign.ready_checkpoint(args.study_root, "transformer", 1101, 200, plan)["status"] == "not_ready"


def test_all_models_evaluated_and_resume_does_not_overwrite_sealed_outputs(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    before = {str(path): path.read_bytes() for path in args.study_root.rglob("*.json")}
    result = campaign.run_campaign(args)
    assert result["status"] == "completed" and len(calls) == 4
    assert set(result["results"]) == {"mlp", "transformer"}
    assert result["results"]["transformer"]["training"]["old_job_status"] == "rejected"
    sealed = {str(path): path.read_bytes() for path in args.output_root.rglob("receipt.json")}
    assert campaign.run_campaign(args)["status"] == "completed"
    assert len(calls) == 4
    assert sealed == {str(path): path.read_bytes() for path in args.output_root.rglob("receipt.json")}
    assert before == {str(path): path.read_bytes() for path in args.study_root.rglob("*.json")}


def test_unready_failed_candidate_is_preserved_not_declared_complete(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, counts=(200, 121), statuses=("completed", "failed"))
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    result = campaign.run_campaign(args)
    assert result["status"] == "blocked" and len(calls) == 2
    assert result["results"]["transformer"]["training"]["status"] == "not_ready"
    assert result["blockers"]["unready_variants"] == ["transformer"]


def test_old_failed_job_with_complete_checkpoint_is_evaluated(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, statuses=("completed", "failed"))
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    result = campaign.run_campaign(args)
    assert result["status"] == "completed" and len(calls) == 4
    assert result["results"]["transformer"]["training"]["old_job_status"] == "failed"


def test_old_executor_lock_prevents_evaluation_even_between_workers(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    with (args.study_root / ".run.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = campaign.run_campaign(args)
    assert result["status"] == "blocked" and not calls
    assert result["blockers"]["old_executor_lock_held"]


def test_failed_worker_retained_and_next_invocation_uses_new_attempt(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls, fail_variant="transformer"))
    result = campaign.run_campaign(args)
    assert result["status"] == "incomplete" and len(calls) == 4
    failed = args.output_root / "transformer/seed_5701/attempt_0000/receipt.json"
    original = failed.read_bytes()
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    assert campaign.run_campaign(args)["status"] == "completed"
    assert len(calls) == 6 and failed.read_bytes() == original
    assert (failed.parent.parent / "attempt_0001/receipt.json").is_file()


def test_pid_start_identity_prevents_pid_reuse_misclassification(monkeypatch):
    monkeypatch.setattr(campaign, "process_start", lambda pid: "456")
    assert campaign.live_process({"status": "running", "pid": 123, "start": "456"})
    assert not campaign.live_process({"status": "running", "pid": 123, "start": "455"})
    assert not campaign.live_process({"status": "running", "pid": 123, "start": None})


def test_live_old_and_orphan_campaign_workers_block_all_launches(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    monkeypatch.setattr(campaign, "process_start", lambda pid: "456")
    old = args.study_root / "jobs/mlp/seed_1101/s1/attempt_0001/train.log.process.json"
    campaign.write(old, {"status": "running", "pid": 123, "start": "456"})
    result = campaign.run_campaign(args)
    assert result["status"] == "blocked" and result["blockers"]["old_live_workers"] and not calls
    old.unlink()
    orphan = args.output_root / "mlp/seed_5701/attempt_0000/worker.process.json"
    campaign.write(orphan, {"status": "running", "pid": 123, "start": "456"})
    result = campaign.run_campaign(args)
    assert result["status"] == "blocked" and result["blockers"]["previous_campaign_live_workers"] and not calls


def test_real_worker_timeout_records_owned_pid_without_changing_old_records(tmp_path):
    directory = tmp_path / "worker"
    directory.mkdir()
    environment = dict(os.environ, PYTHONPATH="synthetic")
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    result = campaign.run_worker(command, directory, environment, .1, lambda: None)
    assert result["timed_out"] and result["returncode"] != 0
    record = campaign.read(directory / "worker.process.json")
    assert record["status"] == "finished" and record["command"] == command
    assert record["pid"] > 0 and record["start"] is not None
    assert campaign.process_start(record["pid"]) is None


def test_tampered_sealed_report_is_retained_as_invalid(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker(calls))
    assert campaign.run_campaign(args)["status"] == "completed"
    output = args.output_root / "mlp/seed_5701/attempt_0000/stand.json"
    output.write_text("tampered")
    result = campaign.run_campaign(args)
    assert result["status"] == "incomplete" and len(calls) == 4
    assert result["results"]["mlp"]["evaluations"]["5701"]["status"] == "invalid"


def test_changed_source_and_old_evaluation_seed_rejected(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    monkeypatch.setattr(campaign, "run_worker", synthetic_worker([]))
    assert campaign.run_campaign(args)["status"] == "completed"
    (args.source_root / "src/transformer_rl/frame_process.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="protocol or source changed"):
        campaign.run_campaign(args)
    args.seeds = [2701]
    with pytest.raises(ValueError, match="disjoint"):
        campaign.run_campaign(args)

"""Curriculum sample budgets, paired initialization and queue ownership."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "tools/run_curriculum_campaign.py"
SPEC = importlib.util.spec_from_file_location("curriculum_campaign", PATH)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
control = campaign.control


def fixture(tmp_path, *, dependency_status="completed"):
    study, source, output = tmp_path / "study", tmp_path / "source", tmp_path / "output"
    (source / "src/transformer_rl").mkdir(parents=True)
    (source / "src/transformer_rl/frame_process.py").write_text("# synthetic learner\n")
    resource = tmp_path / "old/.run.lock"
    resource.parent.mkdir()
    resource.touch()
    dependency = tmp_path / "transfer/summary.json"
    campaign.write(dependency, {"status": dependency_status})
    manifest = {"format": "transformer_rl.curriculum_study", "schema_version": 1,
        "source_identity": {"learner_source": campaign.learner_source_identity(source)},
        "training_seeds": [1101, 1102, 1103],
        "training": {"rollout_steps": 48, "checkpoint_interval": 100, "max_seconds": 100.},
        "evaluation": {"seeds": [8701, 9701], "steps": 401, "settle_steps": 200,
                       "min_steady_samples": 200, "trace_replicas": 8},
        "arms": [], "scenarios": [{"name": "standing", "config": "configs/standing.json"}], "configs": {}}
    for name, pools in (("mixed", ["mixed", "mixed"]), ("stationary", ["stationary", "stationary"]),
                        ("pretrain", ["stationary", "mixed"])):
        manifest["arms"].append({"name": name, "phases": [
            {"name": f"phase_{i + 1}", "updates": [400, 800][i], "start_update": [0, 400][i],
             "config": f"configs/{pool}.json"} for i, pool in enumerate(pools)]})
    for pool in ("mixed", "stationary", "standing"):
        config = {"model": {"policy": {"architecture": "gated"}}, "ppo": {"learning_rate": 3e-5},
            "control": {"policy_dt_s": .01}, "environment": {"snapshot_sha256": "same_snapshot",
            "num_envs": 8, "pool": pool}}
        route = f"configs/{pool}.json"
        campaign.write(study / route, config)
        manifest["configs"][route] = control.digest(config)
    manifest["sha256"] = control.digest(manifest)
    path = study / "curriculum.json"
    campaign.write(path, manifest)
    args = argparse.Namespace(manifest=path, output_root=output, source_root=source,
        resource_lock=resource, dependency_receipt=dependency, device="cpu", max_run_attempts=4,
        max_wait_seconds=0., worker_timeout_seconds=60., poll_seconds=.01)
    return args, manifest


def worker(args, calls, *, partial=False, extra_samples=False, fail=False, mismatch_init=False, reject_evaluation=False):
    stopped_once = False
    def execute(invocation, directory, environment, timeout, heartbeat):
        nonlocal stopped_once
        calls.append(invocation)
        option = lambda name: invocation[invocation.index(name) + 1]
        operation = invocation[3]
        # Both controller ownership and shared predecessor GPU ownership are held.
        for lock_path in (args.resource_lock, args.output_root / ".curriculum.lock"):
            with lock_path.open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if operation == "train":
            request = control.read(directory / "request.json")
            run_dir = Path(option("--run-dir"))
            run_dir.mkdir()
            if fail:
                campaign.write(run_dir / "failure.json", {"attempted_updates": 1, "consumed_transitions": 384})
                return {"returncode": 1, "timed_out": False}
            desired, offset = int(option("--updates")), int(option("--consumed-update-offset"))
            actual = min(100, desired) if partial and not stopped_once else desired
            stopped_once = True
            checkpoint = run_dir / "final.pt"
            checkpoint.write_bytes(f"{directory}-{offset + actual}".encode())
            initializer = {key.replace("-", "_").removeprefix("__"): option(key)
                           for key in ("--resume", "--restore-learning-from", "--initialize-from") if key in invocation}
            seed = int(option("--seed"))
            initial = control.digest({"seed": seed, "arm": request["identity"]["arm"] if mismatch_init else "same"})
            campaign.write(run_dir / "run.json", {"seed": seed, "config": request["config"],
                "initial_model_sha256": initial, **initializer})
            batch = request["batch_samples_per_update"]
            consumed = batch * actual + (1 if extra_samples else 0)
            campaign.write(run_dir / "completion.json", {"status": "completed" if actual == desired else "stopped",
                "stop_reason": None if actual == desired else "time_budget", "start_update": offset,
                "final_update": offset + actual, "completed_updates": actual, "attempted_updates": actual,
                "consumed_transitions": consumed, "cumulative_transitions": request["prior_transitions"] + consumed,
                "checkpoint": str(checkpoint), "checkpoint_sha256": control.file_sha(checkpoint),
                "config_sha256": request["identity"]["config_sha256"]})
            rows = [{"update": offset + i, "batch_samples": batch,
                     "optimization": {"optimizer_steps": 2, "sample_count": batch * 2, "grad_norm": .1}}
                    for i in range(1, actual + 1)]
            (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        elif operation == "evaluate-suite":
            outputs = invocation[invocation.index("--outputs") + 1:invocation.index("--steps")]
            seed, checkpoint = int(option("--seed")), Path(option("--checkpoint"))
            sha = control.file_sha(checkpoint)
            for output in outputs:
                campaign.write(Path(output), {"seed": seed, "checkpoint_sha256": sha,
                    "success_rate": 0. if reject_evaluation else 1., "metrics": {"drift": {"mean": .5}},
                    "stability": {}})
            campaign.write(directory / "control.json", {"seed": seed, "checkpoint_sha256": sha,
                "control": {"planar_motion": {"speed_mean": .5}}})
            (directory / "trace.npz").write_bytes(b"all eight evaluation replicas")
            assert option("--trace-replicas") == "8"
        else:
            raise AssertionError(operation)
        return {"returncode": 0, "timed_out": False}
    return execute


def test_equal_samples_restore_parent_clock_and_never_gate_phase_two(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls, reject_evaluation=True))
    result = campaign.run_campaign(args)
    assert result["status"] == "completed" and len(result["results"]) == 9
    training = [call for call in calls if call[3] == "train"]
    assert len(training) == 18
    assert [call[call.index("--seed") + 1] for call in training[:6]] == ["1101"] * 6
    for first, second in zip(training[::2], training[1::2]):
        assert not any(flag in first for flag in ("--resume", "--initialize-from", "--restore-learning-from"))
        assert "--restore-learning-from" in second and "--resume" not in second and "--initialize-from" not in second
        assert second[second.index("--consumed-update-offset") + 1] == "400"
        assert Path(second[second.index("--restore-learning-from") + 1]).exists()
    assert all(item["checkpoint"]["update"] == 1200 and item["checkpoint"]["cumulative_transitions"] == 1200 * 8 * 48
               for item in result["results"].values())
    calls.clear()
    again = campaign.run_campaign(args)
    assert again["status"] == "completed" and not calls


def test_full_rollout_stop_resumes_adam_with_consumed_clock(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls, partial=True))
    result = campaign.run_campaign(args)
    assert result["status"] == "completed"
    first, resumed = [call for call in calls if call[3] == "train"][:2]
    assert first[first.index("--updates") + 1] == "400"
    assert resumed[resumed.index("--updates") + 1] == "300"
    assert resumed[resumed.index("--consumed-update-offset") + 1] == "100"
    assert "--resume" in resumed and "--initialize-from" not in resumed


@pytest.mark.parametrize("kwargs", [{"extra_samples": True}, {"fail": True}])
def test_unsealed_or_partial_samples_never_refunded_or_retrained(tmp_path, monkeypatch, kwargs):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls, **kwargs))
    result = campaign.run_campaign(args)
    assert result["status"] == "incomplete"
    assert len(calls) == 9 and all(call[3] == "train" for call in calls)
    assert all(item["phases"][0]["training"]["consumed_updates"] == 400 for item in result["results"].values())
    calls.clear()
    assert campaign.run_campaign(args)["status"] == "incomplete" and not calls


def test_paired_scratch_initialization_is_checked(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls, mismatch_init=True))
    with pytest.raises(ValueError, match="different learner parameters"):
        campaign.run_campaign(args)


@pytest.mark.parametrize("status", ["training", "incomplete"])
def test_dependency_must_complete_before_launch(tmp_path, monkeypatch, status):
    args, _ = fixture(tmp_path, dependency_status=status)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    before = args.dependency_receipt.read_bytes()
    assert campaign.run_campaign(args)["status"] == "blocked" and not calls
    assert args.dependency_receipt.read_bytes() == before


def test_shared_lock_prevents_overlap_even_after_completed_receipt(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    with args.resource_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert campaign.run_campaign(args)["status"] == "blocked" and not calls


def test_frozen_config_and_checkpoint_tampering_stop_restart(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    result = campaign.run_campaign(args)
    checkpoint = Path(next(iter(result["results"].values()))["checkpoint"]["checkpoint"])
    checkpoint.write_bytes(b"tampered")
    calls.clear()
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.run_campaign(args)
    assert not calls
    (args.manifest.parent / "configs/mixed.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.run_campaign(args)


def test_source_change_while_queued_prevents_training(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    original = campaign.transfer.acquire_resource
    from contextlib import contextmanager
    @contextmanager
    def changed(*values):
        with original(*values) as dependency:
            (args.source_root / "src/transformer_rl/frame_process.py").write_text("# changed queued source\n")
            yield dependency
    monkeypatch.setattr(campaign.transfer, "acquire_resource", changed)
    with pytest.raises(ValueError, match="changed while queued"):
        campaign.run_campaign(args)
    assert not calls


def test_preparation_and_launch_sources_must_match(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    (args.source_root / "src/transformer_rl/frame_process.py").write_text("# different prepared learner\n")
    with pytest.raises(ValueError, match="prepared learner source differs"):
        campaign.run_campaign(args)
    assert not calls and not args.output_root.exists()


def test_full_package_source_includes_non_python_files_but_ignores_caches(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path)
    package = args.source_root / "src/transformer_rl"
    (package / "calibration.json").write_text('{"gain": 1}')
    manifest["source_identity"]["learner_source"] = campaign.learner_source_identity(args.source_root)
    manifest.pop("sha256")
    manifest["sha256"] = control.digest(manifest)
    campaign.write(args.manifest, manifest, replace=True)
    (package / "__pycache__").mkdir()
    (package / "__pycache__/frame_process.pyc").write_bytes(b"ignored cache")
    (package / "leftover.pyc").write_bytes(b"also ignored")
    campaign.verify_learner_source(args.source_root, manifest)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    original = campaign.transfer.acquire_resource
    from contextlib import contextmanager
    @contextmanager
    def changed(*values):
        with original(*values) as dependency:
            (package / "calibration.json").write_text('{"gain": 2}')
            yield dependency
    monkeypatch.setattr(campaign.transfer, "acquire_resource", changed)
    with pytest.raises(ValueError, match="prepared learner source differs"):
        campaign.run_campaign(args)
    assert not calls


def test_missing_prepared_learner_source_is_rejected(tmp_path):
    args, manifest = fixture(tmp_path)
    manifest.pop("source_identity")
    manifest.pop("sha256")
    manifest["sha256"] = control.digest(manifest)
    campaign.write(args.manifest, manifest, replace=True)
    with pytest.raises(ValueError, match="prepared learner source differs"):
        campaign.run_campaign(args)


def test_configured_continuous_phase_budgets_are_supported(tmp_path, monkeypatch):
    args, manifest = fixture(tmp_path)
    for arm in manifest["arms"]:
        arm["phases"][0].update(updates=40)
        arm["phases"][1].update(start_update=40, updates=80)
    manifest.pop("sha256")
    manifest["sha256"] = control.digest(manifest)
    campaign.write(args.manifest, manifest, replace=True)
    calls = []
    monkeypatch.setattr(control, "run_worker", worker(args, calls))
    result = campaign.run_campaign(args)
    assert result["status"] == "completed"
    assert all(item["checkpoint"]["update"] == 120 and item["checkpoint"]["cumulative_transitions"] == 120 * 8 * 48
               for item in result["results"].values())


def test_raw_source_artifact_and_snapshot_hashes_are_verified(tmp_path):
    args, manifest = fixture(tmp_path)
    source_file = args.manifest.parent / "parent_base.json"
    campaign.write(source_file, {"unchanged_parent": True})
    manifest["artifacts"] = {"parent_base.json": control.file_sha(source_file)}
    campaign.verify_configs(args.manifest.parent, manifest)
    source_file.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.verify_configs(args.manifest.parent, manifest)
    manifest.pop("artifacts")
    snapshot = args.manifest.parent / "snapshot"
    snapshot.mkdir()
    (snapshot / "environment.py").write_text("# frozen dynamics\n")
    files = {"environment.py": control.file_sha(snapshot / "environment.py")}
    source = {"files": files, "sha256": control.digest(files)}
    campaign.write(snapshot / "snapshot.json", source)
    for pool in ("mixed", "stationary"):
        path = args.manifest.parent / f"configs/{pool}.json"
        config = control.read(path)
        config["environment"].update(snapshot=str(snapshot), snapshot_sha256=source["sha256"])
        campaign.write(path, config, replace=True)
        manifest["configs"][f"configs/{pool}.json"] = control.digest(config)
    campaign.verify_configs(args.manifest.parent, manifest)
    (snapshot / "environment.py").write_text("# changed dynamics\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.verify_configs(args.manifest.parent, manifest)

"""Orchestration tests use fake subprocesses only, never environment factories."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from transformer_rl import experiments as ex
from transformer_rl.experiment_cli import main


FAKE = r'''
import hashlib, json, pathlib, sys, time
args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
def write(path, data):
    path.write_text(json.dumps(data))
mode = args[0]
time.sleep(0.03)
if mode == 'train':
    root = pathlib.Path(arg('--run-dir'))
    (root / 'checkpoints').mkdir(parents=True)
    checkpoint = root / 'checkpoints/last.pt'
    checkpoint.write_bytes(b'fake checkpoint, not a model')
    updates = int(arg('--updates'))
    write(root / 'completion.json', dict(status='completed', updates_completed=updates,
        cumulative_update=updates, collected_transitions=updates*int(arg('--rollout-steps'))*2,
        checkpoints=[dict(path=str(checkpoint), update=updates,
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())]))
    (root / 'metrics.jsonl').write_text(json.dumps({'collection': {'reward_mean': 999}})+'\n')
else:
    checkpoint = pathlib.Path(arg('--checkpoint'))
    train_seed = int(checkpoint.parents[2].name.split('_')[1])
    seed = int(arg('--seed'))
    config = json.loads(pathlib.Path(arg('--config')).read_text())
    value = train_seed + seed
    write(pathlib.Path(arg('--output')), dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        seed=seed, vector_steps=int(arg('--steps')), transitions=20, reward_mean=value,
        terminated_count=0, truncated_count=0, environment=config['environment'],
        action_clip=float(arg('--action-clip')) if '--action-clip' in args else None,
        policy='deterministic_mean', metrics={'tracking_error': dict(mean=value, rms=value, min=value, max=value, count=20)}))
'''


@pytest.fixture
def specification(tmp_path):
    ex._write(tmp_path / "base.json", {"environment": {"scene": "fixed", "speed": 1}})
    spec = {
        "base_config": "base.json", "environment_factory": "unverified_robot:factory",
        "seeds": [11, 22, 33],
        "variants": [{"name": "attention", "model": {}, "ppo": {}, "group": "architecture"}],
        "training": {"updates": 2, "rollout_steps": 4, "max_seconds": 10,
                     "checkpoint_interval": 1, "action_clip": None},
        "execution": {"devices": ["cpu", "cuda:0"], "max_parallel": 2, "job_timeout_seconds": 10},
        "evaluation": {"steps": 10, "seeds": [101, 102, 103], "environment": {"speed": 2}},
    }
    path = tmp_path / "spec.json"
    ex._write(path, spec)
    return path, tmp_path / "experiment", spec


def fake_runner(argv, log, deadline, stop):
    # Retain the exact main CLI argv after replacing the module entrypoint.
    return ex._process([sys.executable, "-c", FAKE, *argv[3:]], log, deadline, stop)


def test_freeze_and_reject_overwrite(specification):
    path, root, spec = specification
    manifest = ex.plan(path, root)
    assert len(manifest["jobs"]) == 3
    assert ex.validate_plan(root, True) == manifest
    config = ex._read(root / "configs/attention.evaluation.json")
    assert config["environment"] == {"scene": "fixed", "speed": 2}
    assert "learning_rate" in config["ppo"]
    with pytest.raises(FileExistsError):
        ex.plan(path, root)
    config["environment"]["speed"] = 9
    (root / "configs/attention.evaluation.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config hash"):
        ex.validate_plan(root)


def test_null_factory_and_source_change(specification, monkeypatch):
    path, root, spec = specification
    spec["environment_factory"] = None
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    with pytest.raises(ValueError, match="plan-only"):
        ex.run(root)
    assert not (root / "execution.json").exists()
    monkeypatch.setattr(ex, "source_identity", lambda: {"sha256": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        ex.run(root)


def test_custom_process_owner_is_used_for_training_and_evaluation(specification):
    path, root, spec = specification
    spec["seeds"] = [11]
    spec["evaluation"]["seeds"] = [101]
    spec["execution"]["worker_module"] = "examples._isaaclab_process"
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    calls = []

    def runner(argv, log, deadline, stop):
        assert argv[1:3] == ["-m", "examples._isaaclab_process"]
        calls.append(argv[3])
        fake_runner(argv, log, deadline, stop)

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == "completed"
    assert calls == ["train", "evaluate"]


@pytest.mark.parametrize("worker", [None, "", "module:factory", "module; command", 1])
def test_invalid_process_owner_is_rejected_before_planning(specification, worker):
    path, root, spec = specification
    spec["execution"]["worker_module"] = worker
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="worker_module"):
        ex.plan(path, root)
    assert not root.exists()


def test_bounded_parallel_sequence_and_hierarchical_summary(specification):
    path, root, spec = specification
    ex.plan(path, root)
    active = 0
    peak = 0
    lock = threading.Lock()
    commands = []

    def runner(argv, log, deadline, stop):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            commands.append(argv)
        try:
            fake_runner(argv, log, deadline, stop)
        finally:
            with lock:
                active -= 1

    result = ex.run(root, runner=runner)
    assert not result["interrupted"]
    assert all(j["status"] == "completed" for j in result["jobs"])
    assert peak == 2
    assert len(commands) == 12
    assert {j["device"] for j in result["jobs"]} == {"cpu", "cuda:0"}
    summary = ex.summarize(root)["variants"][0]
    assert summary["completed"] == summary["requested"] == 3
    assert summary["evaluation"]["reward_mean"] == {"n": 3, "mean": 124, "std": 11}
    assert summary["physical_metrics_complete"]
    assert summary["comparison_available"] and not summary["partial"]
    assert summary["training_reward_diagnostic"]["mean"] == 999
    assert "requested,completed,failed,timedout,missing" in (root / "summary.csv").read_text()
    with pytest.raises(FileExistsError):
        ex.run(root, runner=fake_runner)
    (root / "jobs/attention/seed_22/evaluation_102.json").unlink()
    summary = ex.summarize(root)["variants"][0]
    assert summary["missing"] == 1 and summary["completed"] == 2
    assert not summary["evaluation_complete"] and not summary["physical_metrics_complete"]
    assert summary["evaluation"]["reward_mean"]["n"] == 2
    assert summary["partial"] and not summary["comparison_available"]


@pytest.mark.parametrize("failure,expected", [("exit", "failed"), ("timeout", "timedout"),
                                                ("missing", "missing"), ("sha", "failed"),
                                                ("incomplete", "failed")])
def test_failed_training_never_evaluates(specification, failure, expected):
    path, root, spec = specification
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        assert argv[3] == "train"
        if failure in ("exit", "timeout", "missing"):
            code = {"exit": "raise SystemExit(7)", "timeout": "import time; time.sleep(30)", "missing": "pass"}[failure]
            return ex._process([sys.executable, "-c", code], log,
                               time.monotonic() + 0.1 if failure == "timeout" else deadline, stop)
        fake_runner(argv, log, deadline, stop)
        directory = Path(argv[argv.index("--run-dir") + 1])
        if failure == "sha":
            (directory / "checkpoints/last.pt").write_bytes(b"corrupt")
        else:
            completion = ex._read(directory / "completion.json")
            completion["updates_completed"] = 1
            (directory / "completion.json").write_text(json.dumps(completion))

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == expected
    assert ex.summarize(root)["variants"][0][expected] == 1


def test_missing_evaluation_and_wrong_protocol(specification):
    path, root, spec = specification
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        if argv[3] == "evaluate":
            return ex._process([sys.executable, "-c", "pass"], log, deadline, stop)
        fake_runner(argv, log, deadline, stop)

    assert all(j["status"] == "missing" for j in ex.run(root, runner=runner)["jobs"])
    assert ex.summarize(root)["variants"][0]["evaluation"] == {}


@pytest.mark.parametrize("mutation", ["hash", "route", "shape"])
def test_plan_artifact_validation(specification, mutation):
    path, root, spec = specification
    manifest = ex.plan(path, root)
    if mutation == "hash":
        manifest["spec"]["seeds"] = [9]
    elif mutation == "route":
        manifest["jobs"][0]["directory"] = "../escape"
        manifest["plan_sha256"] = ex._digest({k: v for k, v in manifest.items() if k != "plan_sha256"})
    else:
        manifest = {"spec": []}
    (root / "plan.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ex.validate_plan(root)


def test_symlink_config_rejected(specification, tmp_path):
    path, root, spec = specification
    ex.plan(path, root)
    config = root / "configs/attention.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        ex.validate_plan(root)


def test_timeout_reaps_owned_process_without_touching_other_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "pid"
    code = "import os,pathlib,time; pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(30)"
    try:
        with pytest.raises(TimeoutError):
            ex._process([sys.executable, "-c", code], tmp_path / "log", time.monotonic() + 0.3, threading.Event())
        pid = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_sigterm_cleans_children(specification, tmp_path):
    path, root, spec = specification
    ex.plan(path, root)
    pidfile = tmp_path / "child_pid"
    code = '''
import pathlib, sys
from transformer_rl import experiments as ex
def runner(argv, log, deadline, stop):
    child = "import os,pathlib,time; pathlib.Path(" + repr(sys.argv[2]) + ").write_text(str(os.getpid())); time.sleep(30)"
    ex._process([sys.executable, '-c', child], log, deadline, stop)
ex.run(sys.argv[1], max_parallel=1, runner=runner)
'''
    env = {**os.environ, "PYTHONPATH": str(Path(ex.__file__).parents[1])}
    process = subprocess.Popen([sys.executable, "-c", code, str(root), str(pidfile)], env=env)
    try:
        deadline = time.monotonic() + 15
        while not pidfile.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        assert pidfile.exists()
        child = int(pidfile.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        summary = ex.summarize(root)["variants"][0]
        assert summary["failed"] == 1 and summary["missing"] == 2
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_cli_plan_summary_and_null_run(specification):
    path, root, spec = specification
    spec["environment_factory"] = None
    path.write_text(json.dumps(spec))
    assert main(["plan", "--spec", str(path), "--root", str(root)]) == 0
    assert main(["summarize", "--root", str(root)]) == 0
    assert main(["run", "--root", str(root)]) == 1


def test_default_comparison_freezes_all_variants(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    manifest = ex.plan(repo / "configs/comparison.json", tmp_path / "comparison")
    assert len(manifest["jobs"]) == 18
    assert manifest["spec"]["evaluation"]["seeds"] == [101, 102, 103]
    config = ex._read(tmp_path / "comparison/configs/supervised_attention.json")
    assert config["model"]["auxiliary_indices"] == [25, 26, 27]
    assert config["ppo"]["auxiliary_coef"] == 0.1
    assert {j["group"] for j in manifest["jobs"] if j["variant"] == "supervised_attention"} == {"supervision"}
    for name, actor in (("history_mlp", "mlp"), ("history_gru", "gru")):
        assert ex._read(tmp_path / f"comparison/configs/{name}.json")["model"]["actor_type"] == actor
    with pytest.raises(ValueError, match="plan-only"):
        ex.run(tmp_path / "comparison")


@pytest.mark.parametrize("field,value", [("seed", 999), ("checkpoint_sha256", "bad"),
                                        ("policy", "stochastic"), ("environment", {})])
def test_evaluation_protocol_mismatch(specification, field, value):
    path, root, spec = specification
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            report_path = Path(argv[argv.index("--output") + 1])
            report = ex._read(report_path)
            report[field] = value
            report_path.write_text(json.dumps(report))

    result = ex.run(root, runner=runner)
    assert result["jobs"][0]["status"] == "failed"
    assert ex.summarize(root)["variants"][0]["completed"] == 0


def test_reward_only_report_is_not_physical_success(specification):
    path, root, spec = specification
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            report_path = Path(argv[argv.index("--output") + 1])
            report = ex._read(report_path)
            report["metrics"] = {}
            report_path.write_text(json.dumps(report))

    ex.run(root, runner=runner)
    row = ex.summarize(root)["variants"][0]
    assert row["evaluation_complete"] and not row["physical_metrics_complete"]
    assert row["evaluation"]["reward_mean"]["std"] is None


@pytest.mark.parametrize("group", ["architecture", "supervision"])
@pytest.mark.parametrize("section,field,value", [
    ("model", "history_length", 32), ("model", "proprio_dim", 18),
    ("model", "action_dim", 8), ("model", "critic_hidden", [64]),
    ("model", "initial_std", 0.5), ("ppo", "learning_rate", 0.002),
    ("ppo", "epochs", 8),
])
def test_unfair_overrides_rejected_before_root_creation(specification, group, section, field, value):
    path, root, spec = specification
    variant = spec["variants"][0]
    variant["group"] = group
    variant[section][field] = value
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fair comparison"):
        ex.plan(path, root)
    assert not root.exists()


def test_fair_actor_capacity_changes_and_identical_overrides(specification):
    path, root, spec = specification
    spec["variants"][0]["model"] = {"d_model": 128, "num_layers": 3, "num_heads": 8,
                                      "ffn_dim": 256, "history_length": 16, "critic_hidden": [256, 128, 64]}
    spec["variants"][0]["ppo"] = {"learning_rate": 0.0001}
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    ex.validate_plan(root)


@pytest.mark.parametrize("difference", ["training", "evaluation", "eval_seed", "partial"])
def test_group_budget_comparability(specification, difference):
    path, root, spec = specification
    spec["seeds"] = [11]
    spec["variants"] += [
        {"name": "mlp", "model": {"actor_type": "mlp"}, "ppo": {}, "group": "architecture"},
        {"name": "supervised", "model": {"auxiliary_indices": [25]},
         "ppo": {"auxiliary_coef": 0.1}, "group": "supervision"},
    ]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    ex.run(root, runner=fake_runner)
    directory = root / "jobs/mlp/seed_11"
    if difference == "training":
        target = directory / "train/completion.json"
        completion = ex._read(target)
        completion["collected_transitions"] *= 2
        target.write_text(json.dumps(completion))
    elif difference in ("evaluation", "eval_seed"):
        for seed in ([101] if difference == "eval_seed" else spec["evaluation"]["seeds"]):
            target = directory / f"evaluation_{seed}.json"
            report = ex._read(target)
            report["transitions"] *= 2
            target.write_text(json.dumps(report))
    else:
        (directory / "evaluation_101.json").unlink()
    summary = ex.summarize(root)
    checks = summary["fairness_checks"]
    assert not checks["architecture"]["comparison_available"]
    assert checks["supervision"]["comparison_available"]
    rows = summary["variants"]
    assert not rows[0]["comparison_available"] and not rows[1]["comparison_available"]
    assert rows[2]["comparison_available"]
    if difference == "partial":
        assert checks["architecture"]["partial"] and rows[1]["partial"]
    else:
        assert rows[1]["completed"] == 1 and not rows[1]["partial"]
        assert "inconsistent" in checks["architecture"]["reasons"][0]
    assert "comparison_available" in (root / "summary.csv").read_text()


@pytest.mark.parametrize("field,value", [("min", 3), ("mean", 4), ("max", 0),
                                        ("rms", -1), ("count", 21)])
def test_inconsistent_metric_statistics_rejected(specification, tmp_path, field, value):
    _, _, spec = specification
    metric = {"min": 1, "mean": 2, "max": 3, "rms": 2.1, "count": 20}
    metric[field] = value
    report = {"checkpoint_sha256": "digest", "seed": 101, "vector_steps": 10,
              "transitions": 20, "reward_mean": 0, "terminated_count": 0,
              "truncated_count": 0, "environment": {}, "action_clip": None,
              "policy": "deterministic_mean", "metrics": {"tracking": metric}}
    target = tmp_path / "report.json"
    ex._write(target, report)
    with pytest.raises(ValueError, match="inconsistent evaluation metric"):
        ex._report(target, "digest", 101, spec, {})


@pytest.mark.parametrize("cooperative", [True, False])
def test_termination_grace_then_kill(tmp_path, monkeypatch, cooperative):
    monkeypatch.setattr(ex, "_TERMINATION_GRACE_SECONDS", 0.25)
    marker = tmp_path / "terminated"
    pidfile = tmp_path / "pid"
    code = '''
import os, pathlib, signal, sys, time
def terminate(*_):
    pathlib.Path(sys.argv[1]).write_text('SIGTERM received')
    time.sleep(0.05)
    if sys.argv[3] == 'True':
        raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
pathlib.Path(sys.argv[2]).write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
'''
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        ex._process([sys.executable, "-c", code, str(marker), str(pidfile), str(cooperative)],
                    tmp_path / "log", started + 0.4, threading.Event())
    elapsed = time.monotonic() - started
    assert marker.read_text() == "SIGTERM received"
    assert elapsed < 3
    if not cooperative:
        assert elapsed >= 0.65
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)

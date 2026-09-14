"""Named scenarios exercise the CLI protocol through fake subprocesses only."""
import csv
import json
from pathlib import Path
import shutil
import sys

import pytest

from transformer_rl import experiments as ex
from test_experiments import fake_runner, specification  # noqa: F401


FAKE_SCENARIOS = r'''
import hashlib, json, pathlib, sys
args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
def write(path, data):
    path.write_text(json.dumps(data))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
config = json.loads(pathlib.Path(arg('--config')).read_text())
if args[0] == 'train':
    root = pathlib.Path(arg('--run-dir'))
    (root / 'checkpoints').mkdir(parents=True)
    updates = int(arg('--updates'))
    interval = int(arg('--checkpoint-interval'))
    saved = []
    for update in sorted({*range(interval, updates + 1, interval), updates}):
        checkpoint = root / f'checkpoints/checkpoint_{update:06d}.pt'
        checkpoint.write_bytes(f'fake checkpoint {update}'.encode())
        saved.append(dict(path=str(checkpoint), update=update, sha256=sha(checkpoint)))
    write(root / 'completion.json', dict(status='completed', updates_completed=updates,
        cumulative_update=updates, collected_transitions=updates*int(arg('--rollout-steps'))*2,
        checkpoints=saved))
else:
    checkpoint = pathlib.Path(arg('--checkpoint'))
    update = int(checkpoint.stem.split('_')[1])
    train_seed = int(checkpoint.parents[2].name.split('_')[1])
    seed = int(arg('--seed'))
    steps = int(arg('--steps'))
    environment = config['environment']
    value = train_seed + seed + environment['fixed_command'][2]*1000 + update*10
    transitions = steps * environment['num_envs']
    write(pathlib.Path(arg('--output')), dict(checkpoint_sha256=sha(checkpoint),
        seed=seed, vector_steps=steps, transitions=transitions, reward_mean=value,
        terminated_count=0, truncated_count=0, environment=environment,
        action_clip=float(arg('--action-clip')) if '--action-clip' in args else None,
        policy='deterministic_mean', metrics={'tracking_error': dict(
            mean=value, rms=value, min=value, max=value, count=transitions)}))
'''


def scenario_runner(argv, log, deadline, stop):
    return ex._process([sys.executable, "-c", FAKE_SCENARIOS, *argv[3:]], log, deadline, stop)


@pytest.fixture
def scenario_spec(specification):
    path, root, spec = specification
    spec["variants"].append({"name": "mlp", "model": {"actor_type": "mlp"},
                             "ppo": {}, "group": "architecture"})
    spec["evaluation"]["seeds"] = [101, 103]
    spec["evaluation"]["environment"] = {
        "speed": 2, "num_envs": 2, "fixed_command": [0, 0, 0.3]}
    spec["evaluation"]["scenarios"] = [
        {"name": "stand_low", "environment": {"fixed_command": [0, 0, 0.28]}},
        {"name": "stand_high", "environment": {"fixed_command": [0, 0, 0.32], "num_envs": 4}},
    ]
    path.write_text(json.dumps(spec))
    return path, root, spec


def test_scenario_plan_merges_and_freezes_all_variant_configs(scenario_spec):
    path, root, spec = scenario_spec
    manifest = ex.plan(path, root)
    assert manifest["spec"] == spec
    assert len(manifest["jobs"]) == 6
    assert len(manifest["configs"]) == 6
    assert ex.validate_plan(root, check_source=True) == manifest
    for job in manifest["jobs"]:
        assert "eval_config" not in job
        assert set(job["eval_configs"]) == {"stand_low", "stand_high"}
        train = ex._read(root / job["config"])
        assert train["environment"] == {"scene": "fixed", "speed": 1}
        for scene, route in job["eval_configs"].items():
            config = ex._read(root / route)
            assert config["model"] == train["model"]
            assert config["ppo"] == train["ppo"]
            assert config["environment"] == {
                "scene": "fixed", "speed": 2,
                "fixed_command": [0, 0, 0.28 if scene == "stand_low" else 0.32],
                "num_envs": 2 if scene == "stand_low" else 4}
    assert not (root / "configs/attention.evaluation.json").exists()


@pytest.mark.parametrize("scenarios", [
    None, [], {}, [None], [{"name": "low"}],
    [{"name": "low", "environment": []}],
    [{"name": "../escape", "environment": {}}],
    [{"name": "/absolute", "environment": {}}],
    [{"name": "low;exit", "environment": {}}],
    [{"name": "", "environment": {}}],
    [{"name": 1, "environment": {}}],
    [{"name": "low", "environment": {}}, {"name": "low", "environment": {}}],
    [{"name": "low", "environment": {}, "steps": 20}],
    [{"name": "low", "environment": {}, "seeds": [99]}],
])
def test_invalid_scenarios_rejected_before_root_creation(scenario_spec, scenarios):
    path, root, spec = scenario_spec
    spec["evaluation"]["scenarios"] = scenarios
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="scenario"):
        ex.plan(path, root)
    assert not root.exists()


def test_ten_saved_checkpoints_only_final_evaluated_with_hierarchical_statistics(scenario_spec):
    path, root, spec = scenario_spec
    spec["training"].update(updates=977, checkpoint_interval=100)
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    commands = []

    def runner(argv, log, deadline, stop):
        commands.append(argv)
        assert "--diagnostics" not in argv
        scenario_runner(argv, log, deadline, stop)

    assert all(job["status"] == "completed" for job in ex.run(root, runner=runner)["jobs"])
    trains = [argv for argv in commands if argv[3] == "train"]
    evaluations = [argv for argv in commands if argv[3] == "evaluate"]
    assert len(trains) == 6
    assert len(evaluations) == 6 * 2 * 2
    assert {argv[argv.index("--seed") + 1] for argv in evaluations} == {"101", "103"}
    for job in ex.validate_plan(root)["jobs"]:
        directory = root / job["directory"]
        train = next(argv for argv in trains if str(directory / "train") in argv)
        saved = ex._read(directory / "train/completion.json")["checkpoints"]
        assert [checkpoint["update"] for checkpoint in saved] == [
            100, 200, 300, 400, 500, 600, 700, 800, 900, 977]
        assert all(Path(checkpoint["path"]).is_file() for checkpoint in saved)
        assert not (directory / "evaluations").exists()
        for scene in ("stand_low", "stand_high"):
            for seed in spec["evaluation"]["seeds"]:
                output = directory / f"evaluation_{scene}_{seed}.json"
                assert output.is_file() and output.with_suffix(".log").is_file()
                evaluation = next(argv for argv in evaluations if str(output) in argv)
                assert commands.index(train) < commands.index(evaluation)
                assert str(root / job["eval_configs"][scene]) in evaluation
                assert evaluation[evaluation.index("--checkpoint") + 1] == saved[-1]["path"]
    summary = ex.summarize(root)
    assert "evaluation_updates" not in summary
    for row in summary["variants"]:
        assert row["evaluation_complete"] and row["physical_metrics_complete"]
        assert row["comparison_available"] and not row["partial"]
        assert "reward_mean" not in row["evaluation"]
        assert all(key.startswith("scenarios.") for key in row["evaluation"])
        for prefix, mean in (("scenarios.stand_low.", 10174), ("scenarios.stand_high.", 10214)):
            assert row["evaluation"][prefix + "reward_mean"] == {"n": 3, "mean": mean, "std": 11}
            assert row["evaluation"][prefix + "metrics.tracking_error.rms"]["mean"] == mean
        assert row["seeds"][0]["evaluation_seed_means"]["scenarios.stand_low.reward_mean"] == 10163
    checks = summary["fairness_checks"]["architecture"]["scenarios"]
    assert checks["scenarios.stand_low"]["evaluation_transition_counts"] == [20]
    assert checks["scenarios.stand_high"]["evaluation_transition_counts"] == [40]
    assert all(check["comparison_available"] for check in checks.values())
    with (root / "summary.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert any(row["metric"] == "scenarios.stand_low.reward_mean" for row in csv_rows)
    assert not any(row["metric"].startswith("checkpoints.") for row in csv_rows)
    assert all(row["n"] == "3" for row in csv_rows if row["metric"].endswith("reward_mean"))


@pytest.mark.parametrize("missing", ["final_seed", "whole_scenario"])
def test_missing_scenario_or_seed_is_partial_and_excludes_training_seed(scenario_spec, missing):
    path, root, _ = scenario_spec
    ex.plan(path, root)
    ex.run(root, runner=scenario_runner)
    directory = root / "jobs/attention/seed_22"
    routes = {
        "final_seed": ["evaluation_stand_high_101.json"],
        "whole_scenario": ["evaluation_stand_high_101.json", "evaluation_stand_high_103.json"],
    }
    for route in routes[missing]:
        (directory / route).unlink()
    summary = ex.summarize(root)
    row = summary["variants"][0]
    assert row["partial"] and not row["evaluation_complete"]
    assert not row["physical_metrics_complete"] and not row["comparison_available"]
    assert row["completed"] == 2 and row["missing"] == 1
    assert row["seeds"][1]["partial"]
    assert all(stats["n"] == 2 for stats in row["evaluation"].values())
    assert summary["fairness_checks"]["architecture"]["partial"]
    assert all(check["partial"] for check in summary["fairness_checks"]["architecture"]["scenarios"].values())


def test_missing_worker_report_never_completes(scenario_spec):
    path, root, spec = scenario_spec
    spec["seeds"] = [11]
    spec["variants"] = spec["variants"][:1]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        if argv[3] == "evaluate" and "stand_high" in argv[argv.index("--output") + 1]:
            return ex._process([sys.executable, "-c", "pass"], log, deadline, stop)
        scenario_runner(argv, log, deadline, stop)

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == "missing"
    row = ex.summarize(root)["variants"][0]
    assert row["partial"] and row["evaluation"] == {}


@pytest.mark.parametrize("mutation", ["budget", "environment", "checkpoint_sha", "seed", "steps"])
def test_scene_protocol_and_within_scene_budget_checks(scenario_spec, mutation):
    path, root, spec = scenario_spec
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    ex.run(root, runner=scenario_runner)
    target = root / "jobs/mlp/seed_11/evaluation_stand_high_101.json"
    report = ex._read(target)
    field, value = {
        "budget": ("transitions", 80), "environment": ("environment", {}),
        "checkpoint_sha": ("checkpoint_sha256", "wrong"), "seed": ("seed", 999),
        "steps": ("vector_steps", 99),
    }[mutation]
    report[field] = value
    target.write_text(json.dumps(report))
    summary = ex.summarize(root)
    assert not summary["fairness_checks"]["architecture"]["comparison_available"]
    if mutation == "budget":
        checks = summary["fairness_checks"]["architecture"]["scenarios"]
        assert checks["scenarios.stand_low"]["comparison_available"]
        assert not checks["scenarios.stand_high"]["comparison_available"]
        assert checks["scenarios.stand_high"]["evaluation_transition_counts"] == [40, 80]
        assert summary["variants"][1]["evaluation_complete"]
    else:
        assert summary["variants"][1]["failed"] == 1
        assert summary["variants"][1]["partial"]


@pytest.mark.parametrize("mutation", ["config", "route", "plan", "symlink"])
def test_scenario_hash_and_root_safety(scenario_spec, tmp_path, mutation):
    path, root, _ = scenario_spec
    manifest = ex.plan(path, root)
    config_path = root / "configs/attention.evaluation.stand_low.json"
    if mutation == "config":
        config = ex._read(config_path)
        config["environment"]["fixed_command"] = [0, 0, 9]
        config_path.write_text(json.dumps(config))
    elif mutation == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(config_path.read_bytes())
        config_path.unlink()
        config_path.symlink_to(outside)
    else:
        if mutation == "route":
            manifest["jobs"][0]["eval_configs"]["stand_low"] = "../outside.json"
            manifest["plan_sha256"] = ex._digest({k: v for k, v in manifest.items() if k != "plan_sha256"})
        else:
            manifest["spec"]["evaluation"]["scenarios"][0]["environment"] = {}
        (root / "plan.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ex.validate_plan(root)


@pytest.mark.parametrize("mutation", ["missing", "sha", "escape"])
def test_final_checkpoint_validation_precedes_evaluation(scenario_spec, tmp_path, mutation):
    path, root, spec = scenario_spec
    spec["seeds"] = [11]
    spec["variants"] = spec["variants"][:1]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        assert argv[3] == "train"
        scenario_runner(argv, log, deadline, stop)
        completion_path = Path(argv[argv.index("--run-dir") + 1]) / "completion.json"
        completion = ex._read(completion_path)
        final = completion["checkpoints"][-1]
        if mutation == "missing":
            Path(final["path"]).unlink()
        elif mutation == "sha":
            Path(final["path"]).write_bytes(b"corrupt")
        else:
            outside = tmp_path / "outside.pt"
            outside.write_bytes(Path(final["path"]).read_bytes())
            final["path"] = str(outside)
        completion_path.write_text(json.dumps(completion))

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == (
        "missing" if mutation == "missing" else "failed")
    assert ex.summarize(root)["variants"][0]["partial"]


def test_scenario_plan_portability_and_source_validation(scenario_spec, tmp_path, monkeypatch):
    path, root, spec = scenario_spec
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    manifest = ex.plan(path, root)
    remote = tmp_path / "remote"
    shutil.copytree(root, remote)
    assert ex.validate_plan(remote, check_source=True) == manifest
    assert all(job["status"] == "completed" for job in ex.run(remote, runner=scenario_runner)["jobs"])
    assert ex.summarize(remote)["fairness_checks"]["architecture"]["comparison_available"]
    monkeypatch.setattr(ex, "source_identity", lambda: {"sha256": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        ex.run(root)


def test_legacy_spec_keeps_exact_config_job_and_output_layout(specification):
    path, root, spec = specification
    spec["seeds"] = [11]
    path.write_text(json.dumps(spec))
    manifest = ex.plan(path, root)
    assert manifest["jobs"] == [{
        "id": "attention/seed_11", "variant": "attention", "group": "architecture", "seed": 11,
        "config": "configs/attention.json", "eval_config": "configs/attention.evaluation.json",
        "directory": "jobs/attention/seed_11"}]
    expected_configs = {
        "configs/attention.json": ex._digest(ex._configuration(manifest["base_config"], spec["variants"][0])),
        "configs/attention.evaluation.json": ex._digest(ex._configuration(
            manifest["base_config"], spec["variants"][0], spec["evaluation"]["environment"]))}
    assert manifest["configs"] == expected_configs
    assert "scenarios" not in manifest["spec"]["evaluation"]
    ex.run(root, runner=fake_runner)
    directory = root / "jobs/attention/seed_11"
    assert {p.name for p in directory.glob("evaluation_*.json")} == {
        "evaluation_101.json", "evaluation_102.json", "evaluation_103.json"}
    assert not (directory / "evaluations").exists()
    summary = ex.summarize(root)
    assert "evaluation_scenarios" not in summary
    assert summary["variants"][0]["evaluation"]["reward_mean"] == {"n": 1, "mean": 113, "std": None}


def test_learning_curves_recipe_and_complete_plan(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    control = ex._read(repo / "configs/control.json")
    optimized = ex._read(repo / "configs/optimized_control.json")
    control["model"]["mean_init_scale"] = 0.1
    control["ppo"].update(learning_rate=3e-5, epochs=2)
    assert optimized == control
    root = tmp_path / "learning"
    manifest = ex.plan(repo / "configs/learning_curves.json", root)
    spec = manifest["spec"]
    assert spec["variants"] == ex._read(repo / "configs/comparison.json")["variants"]
    assert spec["seeds"] == [1011, 1022, 1033]
    assert spec["training"] == {"updates": 977, "rollout_steps": 32, "max_seconds": 7200,
                                "checkpoint_interval": 100, "diagnostics": False, "action_clip": None}
    assert spec["training"]["updates"] * spec["training"]["rollout_steps"] * 512 == 16_007_168
    assert spec["execution"] == {"devices": ["cuda:0"], "max_parallel": 2, "job_timeout_seconds": 10800}
    assert spec["evaluation"]["steps"] == 2000 and spec["evaluation"]["seeds"] == [301]
    assert spec["evaluation"]["environment"] == {}
    assert {scene["name"]: scene["environment"] for scene in spec["evaluation"]["scenarios"]} == {
        "stand_low": {"fixed_command": [0, 0, 0.28]}, "stand_mid": {"fixed_command": [0, 0, 0.30]},
        "stand_high": {"fixed_command": [0, 0, 0.32]}, "forward": {"fixed_command": [0.5, 0, 0.30]},
        "reverse": {"fixed_command": [-0.5, 0, 0.30]}, "turn_left": {"fixed_command": [0, 1, 0.30]},
        "turn_right": {"fixed_command": [0, -1, 0.30]}}
    assert len(manifest["jobs"]) == 18 and len(manifest["configs"]) == 48
    assert ex.validate_plan(root, check_source=True) == manifest
    summary = ex.summarize(root)
    assert "evaluation_updates" not in summary
    assert all(row["partial"] for row in summary["variants"])
    with pytest.raises(ValueError, match="plan-only"):
        ex.run(root, runner=lambda *_: pytest.fail("must not launch training"))


def test_learning_curves_runs_18_train_and_126_final_scenario_evaluations(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    spec = ex._read(repo / "configs/learning_curves.json")
    spec["base_config"] = str(repo / "configs/optimized_control.json")
    spec["environment_factory"] = "unverified_robot:factory"
    spec["evaluation"]["environment"] = {"num_envs": 8}
    spec["execution"]["devices"] = ["cpu"]
    path, root = tmp_path / "fake_spec.json", tmp_path / "fake_learning"
    ex._write(path, spec)
    manifest = ex.plan(path, root)
    commands = []

    def runner(argv, log, deadline, stop):
        commands.append(argv)
        if argv[3] == "evaluate":
            assert Path(argv[argv.index("--checkpoint") + 1]).name == "checkpoint_000977.pt"
            assert argv[argv.index("--seed") + 1] == "301"
        scenario_runner(argv, log, deadline, stop)

    result = ex.run(root, runner=runner)
    assert all(job["status"] == "completed" for job in result["jobs"])
    assert sum(argv[3] == "train" for argv in commands) == 18
    assert sum(argv[3] == "evaluate" for argv in commands) == 126
    assert ex._read(root / "execution.json")["max_parallel"] == 2
    for job in manifest["jobs"]:
        directory = root / job["directory"]
        assert len(ex._read(directory / "train/completion.json")["checkpoints"]) == 10
        assert len(list((directory / "train/checkpoints").glob("*.pt"))) == 10
        assert len(list(directory.glob("evaluation_*.json"))) == 7
        assert not (directory / "evaluations").exists()
    summary = ex.summarize(root)
    assert all(row["evaluation_complete"] and not row["partial"] for row in summary["variants"])
    assert all(check["comparison_available"] for check in summary["fairness_checks"].values())

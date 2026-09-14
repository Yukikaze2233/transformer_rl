"""Sensitivity contracts use frozen configurations and fake workers only."""
import csv
import json
from pathlib import Path
import shutil

import pytest

from transformer_rl import experiments as ex
from transformer_rl.config import ModelConfig, PPOConfig
from test_experiments import fake_runner, specification  # noqa: F401


@pytest.fixture
def sensitivity_spec(specification):
    path, root, spec = specification
    spec["variants"] = [
        {"name": "baseline", "model": {}, "ppo": {}, "group": "sensitivity"},
        {"name": "low_lr", "model": {}, "ppo": {"learning_rate": 3e-5}, "group": "sensitivity"},
    ]
    path.write_text(json.dumps(spec))
    return path, root, spec


@pytest.mark.parametrize("section,field,value", [
    ("model", "actor_type", "mlp"), ("model", "time_encoding", "index"),
    ("model", "residual_type", "gated"), ("model", "d_model", 128),
    ("model", "num_layers", 3), ("model", "num_heads", 8),
    ("model", "ffn_dim", 256), ("model", "baseline_hidden", [64]),
    ("model", "gru_hidden", 128), ("model", "history_length", 32),
    ("model", "proprio_dim", 18), ("model", "command_dim", 4),
    ("model", "action_dim", 8), ("model", "sensor_groups", 3),
    ("model", "critic_dim", 30), ("model", "critic_hidden", [64]),
    ("model", "time_scale_s", 0.2), ("model", "auxiliary_indices", [25]),
    ("ppo", "epochs", 8), ("ppo", "num_minibatches", 8),
    ("ppo", "gamma", 0.95), ("ppo", "gae_lambda", 0.9),
    ("ppo", "clip_ratio", 0.3), ("ppo", "target_kl", 0.02),
    ("ppo", "value_clip", 0.3), ("ppo", "value_coef", 0.5),
    ("ppo", "entropy_coef", 0.01), ("ppo", "max_grad_norm", 0.5),
    ("ppo", "normalize_advantage", False), ("ppo", "auxiliary_coef", 0.1),
])
def test_sensitivity_freezes_nonfactor_configuration(sensitivity_spec, section, field, value):
    path, root, spec = sensitivity_spec
    spec["variants"][1][section][field] = value
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fair comparison"):
        ex.plan(path, root)
    assert not root.exists()


@pytest.mark.parametrize("group", ["architecture", "supervision"])
@pytest.mark.parametrize("model,ppo", [
    ({"mean_init_scale": 0.1}, {}), ({"initial_std": 0.1}, {}),
    ({}, {"learning_rate": 3e-5}),
])
def test_existing_groups_do_not_gain_sensitivity_overrides(sensitivity_spec, group, model, ppo):
    path, root, spec = sensitivity_spec
    spec["variants"] = [{"name": "candidate", "model": model, "ppo": ppo, "group": group}]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fair comparison"):
        ex.plan(path, root)
    assert not root.exists()


def test_canonical_changes_omit_explicit_defaults_and_distinguish_combinations(sensitivity_spec):
    path, root, spec = sensitivity_spec
    model, ppo = ModelConfig(), PPOConfig()
    spec["variants"] += [
        {"name": "explicit_defaults", "model": {
            "mean_init_scale": 1, "initial_std": model.initial_std,
            "critic_hidden": list(model.critic_hidden)},
         "ppo": {"learning_rate": ppo.learning_rate, "value_coef": 1}, "group": "sensitivity"},
        {"name": "small_head", "model": {"mean_init_scale": 0.1}, "ppo": {}, "group": "sensitivity"},
        {"name": "small_std", "model": {"initial_std": 0.1}, "ppo": {}, "group": "sensitivity"},
        {"name": "combined", "model": {"mean_init_scale": 0.1, "initial_std": 0.4},
         "ppo": {"learning_rate": 1e-5}, "group": "sensitivity"},
    ]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    summary = ex.summarize(root)
    rows = {row["variant"]: row for row in summary["variants"]}
    for name in ("baseline", "explicit_defaults"):
        assert rows[name]["factor_changes"] == {}
        assert rows[name]["single_factor"] is False
    assert rows["low_lr"]["factor_changes"] == {
        "ppo.learning_rate": {"base": ppo.learning_rate, "value": 3e-5}}
    assert rows["small_head"]["factor_changes"] == {
        "model.mean_init_scale": {"base": model.mean_init_scale, "value": 0.1}}
    assert rows["small_std"]["factor_changes"] == {
        "model.initial_std": {"base": model.initial_std, "value": 0.1}}
    assert all(rows[name]["single_factor"] for name in ("low_lr", "small_head", "small_std"))
    assert len(rows["combined"]["factor_changes"]) == 3
    assert rows["combined"]["single_factor"] is False
    assert "combinations do not establish single-factor effects" in summary["interpretation"]
    with (root / "summary.csv").open() as stream:
        csv_rows = {row["variant"]: row for row in csv.DictReader(stream)}
    assert json.loads(csv_rows["combined"]["factor_changes"]) == rows["combined"]["factor_changes"]
    assert csv_rows["combined"]["single_factor"] == "False"


def test_factors_are_relative_to_nondefault_base(sensitivity_spec):
    path, root, spec = sensitivity_spec
    base = ex._read(path.parent / "base.json")
    base.update(model={"mean_init_scale": 0.1, "initial_std": 0.4}, ppo={"learning_rate": 3e-5})
    (path.parent / "base.json").write_text(json.dumps(base))
    spec["variants"][1]["model"] = {"mean_init_scale": 0.1, "initial_std": 0.4}
    spec["variants"] += [{"name": "restored", "model": {"mean_init_scale": 1},
                          "ppo": {}, "group": "sensitivity"}]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)
    rows = ex.summarize(root)["variants"]
    assert rows[1]["factor_changes"] == {}
    assert rows[2]["factor_changes"] == {"model.mean_init_scale": {"base": 0.1, "value": 1}}
    assert rows[2]["single_factor"]


@pytest.mark.parametrize("setting", [None, False, True])
def test_diagnostics_only_adds_training_flag_when_true(sensitivity_spec, setting):
    path, root, spec = sensitivity_spec
    spec["seeds"] = [11]
    spec["evaluation"]["seeds"] = [101]
    if setting is not None:
        spec["training"]["diagnostics"] = setting
    path.write_text(json.dumps(spec))
    manifest = ex.plan(path, root)
    assert manifest["spec"] == spec
    commands = []

    def runner(argv, log, deadline, stop):
        commands.append(argv)
        assert ("--diagnostics" in argv) == (setting is True and argv[3] == "train")
        fake_runner(argv, log, deadline, stop)

    assert all(job["status"] == "completed" for job in ex.run(root, runner=runner)["jobs"])
    assert len(commands) == 4


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_diagnostics_requires_boolean(sensitivity_spec, value):
    path, root, spec = sensitivity_spec
    spec["training"]["diagnostics"] = value
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="diagnostics must be boolean"):
        ex.plan(path, root)
    assert not root.exists()


def test_default_sensitivity_is_seven_plan_only_screening_candidates(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "screening"
    manifest = ex.plan(repo / "configs/sensitivity.json", root)
    assert manifest["spec"]["seeds"] == [11]
    assert len(manifest["jobs"]) == 7
    assert manifest["spec"]["training"]["diagnostics"] is True
    assert manifest["spec"]["training"]["updates"] == 1000
    rows = ex.summarize(root)["variants"]
    assert rows[0]["factor_changes"] == {}
    factors = [row["factor_changes"] for row in rows[1:]]
    assert factors == [
        {"ppo.learning_rate": {"base": 1e-4, "value": 3e-5}},
        {"ppo.learning_rate": {"base": 1e-4, "value": 1e-5}},
        {"model.mean_init_scale": {"base": 1.0, "value": 0.1}},
        {"model.mean_init_scale": {"base": 1.0, "value": 0.01}},
        {"model.initial_std": {"base": 0.2, "value": 0.1}},
        {"model.initial_std": {"base": 0.2, "value": 0.4}},
    ]
    for row in rows[1:]:
        assert row["single_factor"] and row["group"] == "sensitivity"
    with pytest.raises(ValueError, match="plan-only"):
        ex.run(root, runner=lambda *_: pytest.fail("null factory must never start a worker"))
    assert not (root / "execution.json").exists()


def test_optimizer_summary_uses_endpoints_and_ratio_of_total_steps(sensitivity_spec):
    path, root, spec = sensitivity_spec
    spec["variants"] = spec["variants"][:1]
    spec["training"]["diagnostics"] = True
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "train":
            seed = int(argv[argv.index("--seed") + 1])
            first = {"initial_mean_abs": 0.3, "initial_std_mean": 0.2,
                     "first_step_kl": seed / 1000, "first_step_mean_kl": seed / 2000,
                     "first_step_std_kl": seed / 2000, "final_kl": 0.05,
                     "optimizer_steps": 1, "planned_optimizer_steps": 4,
                     "early_stopped": True, "stop_kl": 0.08}
            last = {"initial_mean_abs": 0.1, "initial_std_mean": 0.15,
                    "first_step_kl": 0.001, "final_kl": seed / 10000,
                    "final_mean_kl": seed / 20000, "final_std_kl": seed / 20000,
                    "optimizer_steps": 2, "planned_optimizer_steps": 2,
                    "early_stopped": False, "stop_kl": 0}
            records = [{"update": i, "collection": {"reward_mean": i}, "optimization": value}
                       for i, value in enumerate((first, last), 1)]
            metrics = Path(argv[argv.index("--run-dir") + 1]) / "metrics.jsonl"
            metrics.write_text("".join(json.dumps(r) + "\n" for r in records) + '{"update":')

    ex.run(root, runner=runner)
    summary = ex.summarize(root)
    row = summary["variants"][0]
    stats = row["optimizer_diagnostics"]
    assert stats["first_update.first_step_kl"]["mean"] == pytest.approx(0.022)
    assert stats["first_update.first_step_kl"]["std"] == pytest.approx(0.011)
    assert stats["last_update.final_kl"]["mean"] == pytest.approx(0.0022)
    assert stats["optimizer_steps"] == {"n": 3, "mean": 3, "std": 0}
    assert stats["planned_optimizer_steps"]["mean"] == 6
    assert stats["optimizer_step_utilization"]["mean"] == 0.5
    assert stats["early_stopped_fraction"]["mean"] == 0.5
    assert stats["invalid_metric_rows"]["mean"] == 1
    assert row["training_reward_diagnostic"]["mean"] == 2
    assert row["comparison_available"]
    assert "zero LR" in summary["interpretation"]
    assert "winner" not in summary
    assert "optimizer_diagnostics.last_update.final_kl" in (root / "summary.csv").read_text()


@pytest.mark.parametrize("planned,actual", [(None, 2), (0, 0), (4, 0), (1, 2)])
def test_missing_diagnostics_and_zero_steps_are_not_fabricated(tmp_path, planned, actual):
    path = tmp_path / "metrics.jsonl"
    optimization = {"optimizer_steps": actual, "kl": 0, "first_step_kl": None}
    if planned is not None:
        optimization["planned_optimizer_steps"] = planned
    path.write_text(json.dumps({"optimization": optimization}) + "\n")
    row = ex._training_diagnostics(path)["optimizer_diagnostics"]
    assert row["first_update.first_step_kl"] is None
    assert row["last_update.final_kl"] is None
    assert row["optimizer_steps"] == actual
    assert row["planned_optimizer_steps"] == planned
    assert row["optimizer_step_utilization"] == (0 if planned == 4 else None)


@pytest.mark.parametrize("difference", ["training", "evaluation", "partial"])
def test_sensitivity_checks_actual_budgets_and_all_seeds(sensitivity_spec, difference):
    path, root, spec = sensitivity_spec
    ex.plan(path, root)
    ex.run(root, runner=fake_runner)
    directory = root / "jobs/low_lr/seed_22"
    if difference == "partial":
        (directory / "evaluation_101.json").unlink()
    else:
        target = directory / ("train/completion.json" if difference == "training" else "evaluation_101.json")
        data = ex._read(target)
        data["collected_transitions" if difference == "training" else "transitions"] *= 2
        target.write_text(json.dumps(data))
    summary = ex.summarize(root)
    assert not summary["fairness_checks"]["sensitivity"]["comparison_available"]
    assert all(not row["comparison_available"] for row in summary["variants"])
    if difference == "partial":
        assert summary["variants"][1]["evaluation"]["reward_mean"]["n"] == 2


def test_same_source_hash_is_location_independent_and_plan_is_portable(sensitivity_spec, tmp_path, monkeypatch):
    path, root, _ = sensitivity_spec
    package = Path(ex.__file__).resolve().parent
    remote_package = tmp_path / "remote_package"
    shutil.copytree(package, remote_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = ex.plan(path, root)
    remote_root = tmp_path / "remote_plan"
    shutil.copytree(root, remote_root)
    monkeypatch.setattr(ex, "__file__", str(remote_package / "experiments.py"))
    assert ex.source_identity() == manifest["source"]
    assert ex.validate_plan(remote_root, check_source=True) == manifest
    assert all(job["status"] == "completed" for job in ex.run(remote_root, runner=fake_runner)["jobs"])
    assert ex.summarize(remote_root)["fairness_checks"]["sensitivity"]["comparison_available"]


def test_remote_hash_can_be_summarized_but_different_local_source_cannot_run(sensitivity_spec, monkeypatch):
    path, root, _ = sensitivity_spec
    ex.plan(path, root)
    ex.run(root, runner=fake_runner)
    remote_source = ex.source_identity()
    local_files = {**remote_source["files"], "experiments.py": "a" * 64}
    monkeypatch.setattr(ex, "source_identity", lambda: {"files": local_files, "sha256": ex._digest(local_files)})
    assert ex.validate_plan(root)["source"] == remote_source
    assert ex.summarize(root)["variants"][0]["completed"] == 3
    with pytest.raises(ValueError, match="source changed"):
        ex.run(root, runner=lambda *_: pytest.fail("source mismatch must not launch"))

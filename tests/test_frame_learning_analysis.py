"""Read-only learning-log analysis, without importing PyTorch or the SDK."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools/analyze_frame_learning.py"
SPEC = importlib.util.spec_from_file_location("frame_learning_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record(update, samples=8, applied=4, stopped=False):
    chunks, epochs = min(2, samples), 2
    complete_epochs, leftover = divmod(applied, chunks)
    quotient, remainder = divmod(samples, chunks)
    return {"update": update, "batch_samples": samples,
            "collection": {"vector_steps": samples // 2, "transitions": samples,
                           "early_stopped": samples < 8, "reward_mean": -1.},
            "optimization": {"optimizer_steps": applied, "planned_optimizer_steps": chunks * epochs,
                             "sample_count": complete_epochs * samples + leftover * quotient + min(leftover, remainder),
                             "early_stopped": stopped, "grad_norm": 1., "kl": .002,
                             "first_step_kl": .0001 if applied else None,
                             "final_kl": .003, "stop_kl": .02 if stopped else 0., "clip_fraction": .1}}


def logs(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def fixture(tmp_path, second_records=None, resume=False):
    root = tmp_path / "study"
    source = {"files": {"policy.py": "synthetic-source-hash"}}
    source["sha256"] = analysis.digest(source["files"])
    base = {"model": {"policy": {"architecture": "mlp", "history_length": 1,
                                  "actor_hidden_dims": [8], "frame_dim": 35}},
            "environment": {"num_envs": 2},
            "control": {"policy_dt_s": .01},
            "ppo": {"learning_rate": 1e-4, "target_kl": .01, "gamma": .99,
                    "gae_lambda": .95, "epochs": 2, "num_minibatches": 2}}
    spec = {"variants": [{"name": "mlp", "policy": {}},
                         {"name": "transformer", "policy": {"architecture": "transformer", "history_length": 31}}],
            "seeds": [1101], "stages": [{"name": "s1", "environment": {}, "scenarios": ["stand"]}],
            "scenarios": [{"name": "stand", "environment": {}}],
            "environment_factory": "synthetic:make_env",
            "training": {"rollout_steps": 4, "retention_coef": 0.}}
    plan = {"format": "transformer_rl.packed_study", "schema_version": 1,
            "spec": spec, "base": base, "source": source, "configs": {}}
    configs = {}
    for variant in spec["variants"]:
        config = deepcopy(base)
        config["model"]["policy"].update(variant["policy"])
        configs[variant["name"]] = config
        for kind, names in (("train", ["s1"]), ("eval", ["stand"])):
            for name in names:
                route = f"configs/{variant['name']}.{kind}.{name}.json"
                write(root / route, config)
                plan["configs"][route] = analysis.digest(config)
    plan["sha256"] = analysis.digest(plan)
    write(root / "plan.json", plan)
    for variant in spec["variants"]:
        name = variant["name"]
        rows = [record(1), record(2)] if name == "mlp" or second_records is None else second_records
        parts = [rows[:1], rows[1:]] if resume else [rows]
        attempts = []
        for index, part in enumerate(parts):
            route = f"jobs/{name}/seed_1101/s1/attempt_{index:04d}"
            attempts.append({"directory": route, "status": "stopped" if index == 0 and resume else "completed"})
            run = {"config": configs[name], "source": source, "seed": 1101,
                   "environment_factory": spec["environment_factory"], "rollout_steps": 4,
                   "retention_coef": 0., "updates": len(part), "history_reset": "repeat_first",
                   "episode_state_restored": False, "resume": "remote/final.pt" if index else None}
            write(root / route / "train/run.json", run)
            logs(root / route / "train/metrics.jsonl", part)
            write(root / route / "train/completion.json", {
                "config_sha256": analysis.digest(configs[name]), "start_update": part[0]["update"] - 1,
                "final_update": part[-1]["update"], "completed_updates": len(part),
                "consumed_transitions": sum(r["batch_samples"] for r in part)})
        write(root / f"jobs/{name}/seed_1101/state.json", {
            "variant": name, "seed": 1101, "plan_sha256": plan["sha256"], "status": "completed",
            "stages": [{"name": "s1", "attempts": attempts}]})
    return root


def test_full_window_and_credit_horizons_are_not_memory(tmp_path):
    root = fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = analysis.analyze(root, 1, 2)
    assert report["comparisons"]["s1"]["complete_common_update_interval"]
    assert report["comparisons"]["s1"]["equal_endpoint_uses"]
    transformer = report["jobs"][1]
    assert transformer["history_span_s"] == .3
    assert transformer["window"]["endpoint_use_fraction"] == 1.
    assert transformer["reward_credit_e_fold_horizon"]["seconds"] == pytest.approx(.9949916247)
    assert not report["forgetting"]["available"]
    assert report["forgetting"]["skill_retention_rate"] is None
    assert report["source_files"]["plan.json"]["sha256"]
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_early_stop_records_unequal_optimization_with_equal_samples(tmp_path):
    root = fixture(tmp_path, [record(1, applied=1, stopped=True), record(2, applied=1, stopped=True)])
    report = analysis.analyze(root, 1, 2)
    fair = report["comparisons"]["s1"]
    assert fair["equal_selected_rollout_samples"]
    assert not fair["equal_actual_optimizer_steps"] and not fair["equal_endpoint_uses"]
    window = report["jobs"][1]["window"]
    assert window["endpoint_use_fraction"] == .25 and window["early_stop_fraction"] == 1.
    assert window["optimizer_step_fraction"] == .25


def test_partial_is_excluded_and_does_not_claim_complete_interval(tmp_path):
    root = fixture(tmp_path, [record(1), record(2, samples=4)])
    report = analysis.analyze(root, 1, 2)
    job = report["jobs"][1]
    assert job["actual_logged_samples"] == 12
    assert job["actual_partial_rollouts"] == [{"update": 2, "batch_samples": 4, "in_requested_interval": True}]
    assert job["window"]["selected_updates"] == [1]
    assert job["window"]["excluded_partial_updates"] == [2]
    assert not report["comparisons"]["s1"]["complete_common_update_interval"]
    assert not report["comparisons"]["s1"]["equal_selected_rollout_samples"]


def test_missing_interval_returns_unready_and_null_not_zero(tmp_path):
    report = analysis.analyze(fixture(tmp_path), 3, 4)
    assert report["jobs"][0]["window"]["missing_updates"] == [3, 4]
    assert report["jobs"][0]["window"]["endpoint_use_fraction"] is None
    assert report["jobs"][0]["window"]["optimizer_step_fraction"] is None
    assert report["jobs"][0]["window"]["optimization"]["final_kl"]["mean"] is None
    assert not report["comparisons"]["s1"]["complete_common_update_interval"]


def test_nonoverlapping_resume_is_combined_with_reset_metadata(tmp_path):
    report = analysis.analyze(fixture(tmp_path, resume=True), 1, 2)
    assert report["jobs"][0]["window"]["selected_updates"] == [1, 2]
    assert report["jobs"][0]["attempts"][1]["resume_mode"] == "resume"
    assert report["jobs"][0]["attempts"][1]["episode_state_restored"] is False


@pytest.mark.parametrize("conflicting", [False, True])
def test_overlapping_resume_never_silently_merges(tmp_path, conflicting):
    root = fixture(tmp_path, resume=True)
    path = root / "jobs/mlp/seed_1101/s1/attempt_0001/train/metrics.jsonl"
    row = record(1)
    if conflicting:
        row["collection"]["reward_mean"] = -10.
    logs(path, [row])
    with pytest.raises(ValueError, match="duplicate/conflicting resumed update"):
        analysis.analyze(root, 1, 2)


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "1e999"])
def test_nonfinite_anywhere_in_log_is_rejected(tmp_path, invalid):
    root = fixture(tmp_path)
    path = root / "jobs/mlp/seed_1101/s1/attempt_0000/train/metrics.jsonl"
    path.write_text(path.read_text().replace('"reward_mean": -1.0', f'"reward_mean": {invalid}', 1))
    with pytest.raises(ValueError):
        analysis.analyze(root, 1, 2)


def test_plan_hash_is_checked(tmp_path):
    root = fixture(tmp_path)
    plan = json.loads((root / "plan.json").read_text())
    plan["spec"]["seeds"] = [1102]
    write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="plan SHA"):
        analysis.analyze(root, 1, 2)


def test_config_and_run_identity_are_checked(tmp_path):
    root = fixture(tmp_path)
    path = root / "jobs/mlp/seed_1101/s1/attempt_0000/train/run.json"
    run = json.loads(path.read_text())
    run["config"]["ppo"]["learning_rate"] = 3e-5
    write(path, run)
    with pytest.raises(ValueError, match="run identity/config"):
        analysis.analyze(root, 1, 2)


@pytest.mark.parametrize("field", ["batch_samples", "optimizer_steps", "sample_count"])
def test_row_identity_is_checked(tmp_path, field):
    root = fixture(tmp_path)
    path = root / "jobs/mlp/seed_1101/s1/attempt_0000/train/metrics.jsonl"
    row = record(1)
    if field == "batch_samples":
        row[field] = 7
    else:
        row["optimization"][field] += 1
    logs(path, [row, record(2)])
    with pytest.raises(ValueError, match="identity mismatch"):
        analysis.analyze(root, 1, 2)


def test_snapshot_time_is_fixed_and_file_bytes_verified(tmp_path):
    root = fixture(tmp_path)
    snapshot = {"captured_at": "2026-10-03T01:44:53+00:00",
                "plan_sha256": json.loads((root / "plan.json").read_text())["sha256"],
                "files": {}}
    for path in root.rglob("*"):
        if path.is_file():
            snapshot["files"][str(path.relative_to(root))] = {
                "sha256": analysis.hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    write(root / "snapshot.json", snapshot)
    first, second = analysis.analyze(root, 1, 2), analysis.analyze(root, 1, 2)
    assert first == second and first["snapshot_is_file_sealed"]
    (root / "configs/mlp.train.s1.json").write_text((root / "configs/mlp.train.s1.json").read_text() + " ")
    with pytest.raises(ValueError, match="snapshot file SHA"):
        analysis.analyze(root, 1, 2)


def test_partial_jsonl_duplicate_keys_and_path_escape_are_rejected(tmp_path):
    root = fixture(tmp_path)
    path = root / "jobs/mlp/seed_1101/s1/attempt_0000/train/metrics.jsonl"
    path.write_text(path.read_text().rstrip("\n"))
    with pytest.raises(ValueError, match="incomplete JSONL"):
        analysis.analyze(root, 1, 2)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analysis.parse('{"seed": 1, "seed": 2}')
    with pytest.raises(ValueError, match="escapes root"):
        analysis.Reader(root).path("../elsewhere.json")


def test_cli_never_overwrites_output_or_mutates_study(tmp_path):
    root = fixture(tmp_path)
    output = tmp_path / "report.json"
    command = [sys.executable, str(MODULE_PATH), "--study-root", str(root), "--output", str(output),
               "--first-update", "1", "--last-update", "2"]
    assert subprocess.run(command, capture_output=True).returncode == 0
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
    command[command.index("--output") + 1] = str(root / "report.json")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not (root / "report.json").exists()


def test_gamma_boundary_horizons_do_not_emit_infinity():
    assert analysis._horizon(0., .01) == {"seconds": 0., "infinite": False}
    assert analysis._horizon(1., .01) == {"seconds": None, "infinite": True}
    with pytest.raises(ValueError):
        analysis._horizon(1.01, .01)


def test_positive_configured_retention_starts_without_anchors_then_activates(tmp_path):
    root = fixture(tmp_path)
    plan = json.loads((root / "plan.json").read_text())
    plan["spec"]["training"]["retention_coef"] = .1
    plan["spec"]["stages"].append({"name": "s2", "environment": {}, "scenarios": ["stand"]})
    for variant in plan["spec"]["variants"]:
        key = variant["name"]
        source_config = root / f"configs/{key}.train.s1.json"
        route = f"configs/{key}.train.s2.json"
        write(root / route, json.loads(source_config.read_text()))
        plan["configs"][route] = plan["configs"][f"configs/{key}.train.s1.json"]
    plan["sha256"] = analysis.digest({k: v for k, v in plan.items() if k != "sha256"})
    write(root / "plan.json", plan)
    for variant in plan["spec"]["variants"]:
        key = variant["name"]
        state_path = root / f"jobs/{key}/seed_1101/state.json"
        state = json.loads(state_path.read_text())
        state["plan_sha256"] = plan["sha256"]
        route = f"jobs/{key}/seed_1101/s2/attempt_0000"
        state["stages"].append({"name": "s2", "anchors": [{"path": "anchor.pt", "sha256": "synthetic"}],
                                "attempts": [{"directory": route, "status": "completed"}]})
        write(state_path, state)
        original = root / f"jobs/{key}/seed_1101/s1/attempt_0000/train"
        run = json.loads((original / "run.json").read_text())
        run["retention_coef"] = .1
        write(root / route / "train/run.json", run)
        logs(root / route / "train/metrics.jsonl", [record(1), record(2)])
        write(root / route / "train/completion.json", json.loads((original / "completion.json").read_text()))
    report = analysis.analyze(root, 1, 2)
    for job in report["jobs"]:
        assert job["configured_retention_coef"] == .1
        assert job["effective_retention_coef"] == (0. if job["stage"] == "s1" else .1)
    assert not report["forgetting"]["available"] and report["forgetting"]["stage_count"] == 2

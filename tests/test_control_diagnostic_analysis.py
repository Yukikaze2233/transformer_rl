"""Fault-injection checks for a JSON-only, fixed-denominator diagnostic analysis."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("control_diagnostic_analysis", Path(__file__).parents[1] / "tools/analyze_control_diagnostics.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root, source, study, snapshot = [tmp_path / n for n in ("campaign", "source", "study", "snapshot")]
    code = source / "src/transformer_rl/synthetic.py"
    code.parent.mkdir(parents=True)
    code.write_text("# inert fixture; must never be executed\n")
    source_files = {"src/transformer_rl/synthetic.py": ref(code)["sha256"]}
    monkeypatch.setattr(analysis, "SOURCE_SHA256", analysis.digest(source_files))
    source_identity = {"files": source_files, "sha256": analysis.SOURCE_SHA256, "git_head": analysis.SOURCE_COMMIT}
    controllers = {}
    pins = {}
    for name in analysis.CONTROLLERS:
        file = tmp_path / "controllers" / name
        file.parent.mkdir(exist_ok=True)
        file.write_text("raise RuntimeError('historical controller import forbidden')\n")
        controllers[str(file)] = pins[name] = ref(file)["sha256"]
    monkeypatch.setattr(analysis, "CONTROLLERS", pins)
    snapshot_files = {}
    for case in analysis.CASES:
        receipt = write(snapshot / f"{case}.json", {"evaluation_exact_cases": True, "target_num_envs": 8,
            "evaluation": {"cases": [{"name": case}]}})
        snapshot_files[case + ".json"] = receipt["sha256"]
    snapshot_sha = analysis.digest(snapshot_files)
    snapshot_receipt = write(snapshot / "snapshot.json", {"files": snapshot_files, "sha256": snapshot_sha})
    control = {"policy_dt_s": .01, "action_names": list("abcdef")}
    configs, content_shas, environments = {}, {}, {}
    for variant in analysis.VARIANTS:
        environments[variant] = {}
        for case in analysis.CASES:
            route = f"configs/{variant}.eval.{case}.json"
            env = {"num_envs": 8, "snapshot": str(snapshot), "snapshot_sha256": snapshot_sha,
                "contract": case + ".json", "contract_sha256": snapshot_files[case + ".json"]}
            cfg = {"control": control, "environment": env}
            configs[route] = write(study / route, cfg)
            content_shas[route] = analysis.digest(cfg)
            environments[variant][case] = env
    plan = {"spec": {"variants": [{"name": v} for v in analysis.VARIANTS], "seeds": [1101],
        "stages": [{"updates": 1200, "scenarios": list(analysis.CASES)}]}, "configs": content_shas}
    plan["sha256"] = analysis.digest(plan)
    plan_receipt = write(study / "plan.json", plan)
    checkpoints = {}
    for variant in analysis.VARIANTS:
        file = study / "jobs" / variant / "seed_1101/final.pt"
        file.parent.mkdir(parents=True)
        file.write_bytes(b"not a loadable checkpoint " + variant.encode())
        cp = {"checkpoint": str(file), "checkpoint_sha256": ref(file)["sha256"], "update": 1200,
            "control_sha256": analysis.digest(control)}
        cp["sidecar"] = write(str(file) + ".json", {"update": 1200, "sha256": cp["checkpoint_sha256"],
            "config": {"control": control}, "metadata": {"seed": 1101, "environment_provenance": {"identity": snapshot_sha}}})
        cp["completion"] = write(file.parent / "completion.json", {"status": "completed", "final_update": 1200,
            "checkpoint": str(file), "checkpoint_sha256": cp["checkpoint_sha256"]})
        cp["training_state"] = write(file.parent / "state.json", {"seed": 1101, "variant": variant, "plan_sha256": plan["sha256"]})
        checkpoints[variant] = cp
    dependencies = {}
    for name in ("transfer", "curriculum"):
        definition = {"name": name}
        dependencies[name] = {"campaign": write(tmp_path / name / "campaign.json", definition),
            "campaign_sha256": analysis.digest(definition)}
    manifest = {"format": "transformer_rl.frame_diagnostic_campaign", "schema_version": 1,
        "output_root": str(root), "protocol": analysis.PROTOCOL, "formal_architecture_selection": False,
        "hardware_deployment_ready": False, "controllers": controllers, "source": source_identity,
        "source_commit": analysis.SOURCE_COMMIT, "source_root": str(source), "source_origin": None,
        "dependencies": dependencies, "inputs": {"variants": list(analysis.VARIANTS), "cases": list(analysis.CASES),
            "training_seed": 1101, "study_root": str(study), "plan": plan_receipt, "plan_sha256": plan["sha256"],
            "configs": configs, "checkpoints": checkpoints, "environments": environments,
            "snapshots": {str(snapshot): {"root": str(snapshot), "sha256": snapshot_sha, "receipt": snapshot_receipt}}}}
    manifest["sha256"] = analysis.digest(manifest)
    monkeypatch.setattr(analysis, "MANIFEST_SHA256", manifest["sha256"])
    path = root / "manifest.json"
    write(path, manifest)
    write(root / "summary.json", {"status": "waiting", "manifest_sha256": manifest["sha256"], "results": {},
        "paired_diagnostic_retest": True, "formal_architecture_selection": False})
    return path, manifest


def metric(rows, *, steady=True, failed=0, speed=3., stationary=True):
    samples = 4001 * rows
    partial = rows
    full_intervals = samples - rows - failed
    steady_samples = 3801 * rows if steady else 0
    segments = rows + failed
    retained_segments = rows if steady else 0
    def pool(intervals):
        duration = intervals * .01
        return {"available": bool(intervals), "intervals": intervals, "observed_duration_s": duration,
            "path_length_m": speed * duration, "mean_speed_m_s": speed if intervals else None,
            "rms_speed_m_s": speed if intervals else None, "max_speed_m_s": speed if intervals else None,
            "p95_speed_m_s": speed + .0005 if intervals else None, "p95_bin_m_s": [speed, speed + .001] if intervals else [None, None],
            "p95_method": "duration-weighted histogram bin midpoint; null if the quantile is in overflow",
            "p95_bin_width_m_s": .001, "p95_overflow_from_m_s": 10., "overflow_duration_s": 0.,
            "velocity_world": {"vx": {"mean_m_s": 0. if intervals else None, "rms_m_s": 0. if intervals else None},
                               "vy": {"mean_m_s": speed if intervals else None, "rms_m_s": speed if intervals else None}}}
    def tracking(count):
        return {"samples": count, "axes": {a: {"count": count,
            **{k: 0. if count else None for k in ("bias", "mae", "rmse", "within_group_std", "group_mean_std")}}
            for a in ("vx", "wz", "height")}}
    def displacement(count):
        return {"count": count, **{k: 4. if count else None for k in ("mean", "rms", "mean_abs", "max_abs")}}
    stationary_runs = segments if stationary else 0
    stationary_samples = samples if stationary else 0
    stationary_intervals = stationary_samples - stationary_runs
    return {"available": True, "protocol": {"policy_dt_s": .01, "settle_steps": 200, "min_steady_samples": 200,
        "signal_time": "PRE-reset physical sample time", "axis_order": ["vx", "wz", "height"]},
        "full_interval": tracking(samples), "steady": {**tracking(steady_samples), "available": steady,
            "total": segments, "eligible": retained_segments, "short": segments - retained_segments,
            "failed": failed, "partial": partial, "discarded_settle_samples": samples - steady_samples,
            "discarded_short_samples": 0},
        "episodes": {"completed": failed, "failed": failed, "success_flags": 0, "partial": partial, "short": failed},
        "planar_motion": {"coordinate_frame": "world_xy", "num_envs": rows, "physical_samples": samples,
            "velocity_source": analysis.PLANAR_SOURCE, "scope": analysis.PLANAR_SCOPE, "weighting": analysis.PLANAR_WEIGHTING,
            "steady_eligibility": analysis.STEADY_ELIGIBILITY, "full_interval": pool(full_intervals),
            "steady": pool(steady_samples - retained_segments), "stationary_steady": pool(steady_samples - retained_segments if stationary else 0),
            "stationary": {**pool(stationary_intervals), "samples": stationary_samples, "runs": stationary_runs,
                "endpoint_displacement_m": displacement(stationary_runs), "max_excursion_m": displacement(stationary_runs),
                "reference": "effective vx and wz both within command_tolerance of zero; height may vary",
                "origin": "first stationary sample; run ends at first nonzero command, reset or evaluation cut"}},
        "actuation": {"sample_count": samples, "scaled_nominal_envelope": {"available": True, "sample_count": samples,
            "active_bound_samples": [samples] * 6, "applied_at_bound_fraction": [.25] * 6,
            "requested_outside_bounds_fraction": [.5] * 6, "applied_outside_bounds_fraction": [.1] * 6,
            "unit": "N*m", "semantics": analysis.SCALED_SEMANTICS, "scope": analysis.SCALED_SCOPE}}}


def suite(bundle, variant="mlp", seed=8701, *, complete=True, steady=True):
    path, manifest = bundle
    root = path.parent
    cp = manifest["inputs"]["checkpoints"][variant]
    directory = root / "evaluations" / variant / f"seed_{seed}" / "attempt_0000"
    directory.mkdir(parents=True)
    identity = {"checkpoint_sha256": cp["checkpoint_sha256"], "checkpoint_update": 1200, "seed": seed, "steps": 4001}
    groups, artifacts = {}, {}
    for case in analysis.CASES:
        groups[case] = metric(8, steady=steady, stationary=case.split("__")[0] in ("stand_305mm", "height_scan", "start_stop_05"))
        report = {**identity, "format": "transformer_rl.packed_evaluation", "schema_version": 1, "num_envs": 8, "transitions": 32008, "control_sha256": cp["control_sha256"],
            "environment": manifest["inputs"]["environments"][variant][case], "control": groups[case]}
        artifacts[case] = write(directory / f"{case}.json", report)
    trace_path = directory / "trace.npz"
    trace_path.write_bytes(b"deliberately invalid archive; JSON-only analyzer must never open it")
    artifacts["trace"] = ref(trace_path)
    labels = [case for case in analysis.CASES for _ in range(8)]
    report = {**identity, "format": "transformer_rl.control_evaluation", "schema_version": 1, "groups": groups, "control": metric(400, steady=steady),
        "environment_provenance": {"identity": next(iter(manifest["inputs"]["snapshots"].values()))["sha256"], "evaluation_groups": labels},
        "trace": {**identity, "policy_dt_s": .01, "sampling_hz": 100., "control_sha256": cp["control_sha256"],
            "group_labels": [case for case in analysis.CASES for _ in range(2)],
            "sha256": artifacts["trace"]["sha256"], "row_indices": [j for i in range(50) for j in (i * 8, i * 8 + 1)]}}
    artifacts["control"] = write(directory / "control.json", report)
    receipt = {"identity": {"manifest_sha256": manifest["sha256"], "variant": variant,
        "checkpoint_sha256": cp["checkpoint_sha256"], "seed": seed}, "status": "completed" if complete else "failed",
        "directory": str(directory.relative_to(root)), "worker": {"returncode": 0 if complete else 1, "timed_out": False}, "artifacts": artifacts}
    write(directory / "receipt.json", receipt)
    summary = json.loads((root / "summary.json").read_bytes())
    summary["results"][f"{variant}/seed_{seed}"] = receipt
    write(root / "summary.json", summary)
    return directory, receipt


def rewrite_suite(bundle, directory, receipt, case, transform):
    path, _ = bundle
    report = json.loads((directory / f"{case}.json").read_bytes())
    transform(report)
    receipt["artifacts"][case] = write(directory / f"{case}.json", report)
    control = json.loads((directory / "control.json").read_bytes())
    control["groups"][case] = report["control"]
    receipt["artifacts"]["control"] = write(directory / "control.json", control)
    write(directory / "receipt.json", receipt)
    summary = json.loads((path.parent / "summary.json").read_bytes())
    summary["results"][f"mlp/seed_8701"] = receipt
    write(path.parent / "summary.json", summary)


def test_waiting_retains_all_fixed_cells_and_planar_nulls(bundle):
    path, _ = bundle
    result = analysis.analyze(path)
    assert result["status"] == "not_ready"
    assert result["coverage"]["expected_case_reports"] == len(result["cells"]) == 1000
    assert len(result["suites"]) == 20 and len(result["case_pairs"]) == 500
    assert result["coverage"]["available_case_reports"] == 0
    assert all(row["metrics"] is None and row["case_report"] is None for row in result["cells"])
    for row in result["case_pairs"]:
        assert row["metrics"]["world_xy_stationary_mean_speed_m_s"] == {"expected_cells": 2, "available_cells": 0, "equal_cell_mean": None}
    assert all(row["metrics"]["full_interval_vx_mae"]["expected_cells"] == 12 for row in result["profile_macros"])
    assert "torch" not in analysis.__dict__ and "numpy" not in analysis.__dict__


def test_partial_suite_keeps_missing_seed_and_world_lateral_speed(bundle):
    directory, _ = suite(bundle)
    result = analysis.analyze(bundle[0])
    assert result["coverage"]["completed_suites"] == 1
    assert result["coverage"]["available_case_reports"] == 50
    row = next(r for r in result["cells"] if r["variant"] == "mlp" and r["evaluation_seed"] == 8701 and r["case"] == analysis.CASES[0])
    assert row["metrics"]["world_xy_stationary_mean_speed_m_s"] == 3.
    assert row["metrics"]["full_interval_vx_mae"] == 0.
    assert row["metrics"]["stationary_run_endpoint_displacement_m_mean"] == 4.
    assert row["case_report"]["sha256"] == ref(directory / f"{analysis.CASES[0]}.json")["sha256"]
    pair = result["case_pairs"][0]["metrics"]["world_xy_stationary_mean_speed_m_s"]
    assert pair == {"expected_cells": 2, "available_cells": 1, "equal_cell_mean": None}
    assert result["profile_macros"][0]["metrics"]["full_interval_vx_mae"]["equal_cell_mean"] is None


def test_paired_cells_are_equal_seed_means_and_steady_absence_is_not_zero(bundle):
    suite(bundle, seed=8701, steady=False)
    suite(bundle, seed=9701)
    result = analysis.analyze(bundle[0])
    pair = result["case_pairs"][0]["metrics"]
    assert pair["world_xy_stationary_mean_speed_m_s"]["equal_cell_mean"] == 3.
    assert pair["steady_vx_mae"] == {"expected_cells": 2, "available_cells": 1, "equal_cell_mean": None}
    assert pair["steady_sample_fraction"]["equal_cell_mean"] == pytest.approx(.5 * 3801 / 4001)
    forward = next(r for r in result["case_pairs"] if r["variant"] == "mlp" and r["case"] == "forward_05__nominal")
    assert forward["metrics"]["world_xy_stationary_mean_speed_m_s"]["equal_cell_mean"] is None
    macro = result["profile_macros"][0]["metrics"]
    assert macro["full_interval_vx_mae"]["expected_cells"] == 12
    assert macro["full_interval_vx_mae"]["equal_cell_mean"] == 0.
    assert not any("stationary" in key for key in macro)


def test_failed_suite_is_missing_evidence_without_reducing_denominator(bundle):
    suite(bundle, complete=False)
    result = analysis.analyze(bundle[0])
    assert result["coverage"]["available_case_reports"] == 0
    assert sum(row["status"] == "failed" for row in result["cells"]) == 50
    assert len(result["cells"]) == 1000


def test_rehashed_manifest_cannot_bless_omitted_model(bundle):
    path, manifest = bundle
    manifest["inputs"]["variants"].pop()
    manifest["sha256"] = analysis.digest({k: v for k, v in manifest.items() if k != "sha256"})
    write(path, manifest)
    with pytest.raises(ValueError, match="independently pinned"):
        analysis.analyze(path)


@pytest.mark.parametrize("corruption", ["checkpoint", "controller", "source", "snapshot", "config"])
def test_sealed_input_byte_changes_are_rejected(bundle, corruption):
    path, m = bundle
    if corruption == "checkpoint":
        target = Path(m["inputs"]["checkpoints"]["mlp"]["checkpoint"])
    elif corruption == "controller":
        target = Path(next(iter(m["controllers"])))
    elif corruption == "source":
        target = Path(m["source_root"]) / next(iter(m["source"]["files"]))
    elif corruption == "snapshot":
        target = Path(next(iter(m["inputs"]["snapshots"]))) / (analysis.CASES[0] + ".json")
    else:
        target = Path(next(iter(m["inputs"]["configs"].values()))["path"])
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        analysis.analyze(path)


@pytest.mark.parametrize("field,value", [("num_envs", 2), ("transitions", 8002), ("seed", 9701),
    ("checkpoint_update", 1199), ("checkpoint_sha256", "0" * 64), ("control_sha256", "0" * 64)])
def test_rehashed_case_with_wrong_identity_or_trace_subsample_coverage_is_rejected(bundle, field, value):
    directory, receipt = suite(bundle)
    rewrite_suite(bundle, directory, receipt, analysis.CASES[0], lambda r: r.update({field: value}))
    with pytest.raises(ValueError, match="case identity"):
        analysis.analyze(bundle[0])


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["planar_motion"].update(coordinate_frame="body"), "world planar"),
    (lambda c: c["planar_motion"].update(velocity_source="body vx"), "world planar"),
    (lambda c: c["planar_motion"]["stationary"].update(mean_speed_m_s=400.), "speed and path"),
    (lambda c: c["planar_motion"]["stationary"]["velocity_world"].pop("vy"), "both world"),
    (lambda c: c["steady"].update(samples=0), "steady sample"),
    (lambda c: c["actuation"]["scaled_nominal_envelope"].update(semantics="physical current-loop limit"), "envelope provenance"),
    (lambda c: c["actuation"]["scaled_nominal_envelope"].update(sample_count=8002), "envelope provenance"),
    (lambda c: c["actuation"]["scaled_nominal_envelope"].update(applied_at_bound_fraction=[2.] * 6), "finite diagnostic"),
    (lambda c: c["episodes"].update(failed=100), "partial episodes"),
])
def test_rehashed_measurement_corruptions_are_rejected(bundle, mutation, match):
    directory, receipt = suite(bundle)
    rewrite_suite(bundle, directory, receipt, analysis.CASES[0], lambda r: mutation(r["control"]))
    with pytest.raises(ValueError, match=match):
        analysis.analyze(bundle[0])


def test_trace_is_not_read_and_checkpoint_is_never_deserialized(bundle, monkeypatch):
    suite(bundle)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".npz":
            raise AssertionError("trace payload must not be opened")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    assert analysis.analyze(bundle[0])["coverage"]["available_case_reports"] == 50


def test_nonfinite_and_duplicate_json_rejected():
    for text in ('{"x": NaN}', '{"x": 1e999}', '{"x":1,"x":2}'):
        with pytest.raises(ValueError):
            analysis.parse_json(text)


def test_full_interval_failures_are_reported_separately_from_steady_metrics():
    control = metric(8, steady=False, failed=12, speed=0.)
    analysis.validate_control(control, 8)
    values = analysis.extract(control)
    assert values["world_xy_full_interval_mean_speed_m_s"] == 0.
    assert values["failed_resets_per_policy_robot_minute"] == pytest.approx(12 * 60 / 320.08)
    assert values["steady_vx_mae"] is None and values["steady_sample_fraction"] == 0.


def test_overflow_quantile_is_null_not_imputed_at_limit():
    pool = metric(8)["planar_motion"]["full_interval"]
    pool.update(p95_bin_m_s=[10., None], p95_speed_m_s=None, overflow_duration_s=pool["observed_duration_s"])
    analysis.pool(pool, 32000)
    pool["p95_speed_m_s"] = 10.
    with pytest.raises(ValueError, match="overflow quantile"):
        analysis.pool(pool, 32000)


def test_completed_summary_cannot_hide_missing_suites(bundle):
    path, manifest = bundle
    write(path.parent / "summary.json", {"status": "completed", "results": {}, "manifest_sha256": manifest["sha256"],
        "paired_diagnostic_retest": True, "formal_architecture_selection": False})
    with pytest.raises(ValueError, match="missing fixed suites"):
        analysis.analyze(path)


def test_no_winner_or_hardware_qualification_and_no_input_mutations(bundle, tmp_path):
    path, _ = bundle
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    result = analysis.analyze(path)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert result["formal_architecture_selection"] is False and result["hardware_deployment_ready"] is False
    assert "winner" not in result
    assert result["sha256"] == analysis.digest({k: v for k, v in result.items() if k != "sha256"})


def test_full_fixed_grid_is_ready_only_with_all_twenty_sealed_suites(bundle):
    for variant in analysis.VARIANTS:
        for seed in analysis.SEEDS:
            suite(bundle, variant, seed)
    result = analysis.analyze(bundle[0])
    assert result["status"] == "ready"
    assert result["coverage"]["completed_suites"] == 20
    assert result["coverage"]["available_case_reports"] == 1000
    assert result["coverage"]["expected_policy_samples"] == 32008000
    assert all(r["metrics"]["full_interval_height_mae"]["available_cells"] == 12 for r in result["profile_macros"])
    assert result["metric_units"]["world_xy_stationary_mean_speed_m_s"] == "m/s"
    assert result["metric_units"]["stationary_run_max_excursion_m_mean"] == "m"


def test_trace_row_receipt_cannot_replace_full_case_coverage(bundle):
    directory, receipt = suite(bundle)
    value = json.loads((directory / "control.json").read_bytes())
    value["trace"]["row_indices"][-1] = 399
    receipt["artifacts"]["control"] = write(directory / "control.json", value)
    write(directory / "receipt.json", receipt)
    summary_path = bundle[0].parent / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    summary["results"]["mlp/seed_8701"] = receipt
    write(summary_path, summary)
    with pytest.raises(ValueError, match="two-of-eight trace"):
        analysis.analyze(bundle[0])


def test_old_schema_missing_planar_metrics_has_explicit_error_no_vx_substitution():
    control = metric(8)
    del control["planar_motion"]
    with pytest.raises(ValueError, match="unsupported diagnostic schema.*planar_motion.*no measurement substituted"):
        analysis.validate_control(control, 8)


@pytest.mark.parametrize("field", ["samples", "physical_samples", "scaled_count"])
def test_measurement_counts_require_integers(field):
    control = metric(8)
    if field == "samples":
        control["full_interval"]["samples"] = 32008.
    elif field == "physical_samples":
        control["planar_motion"]["physical_samples"] = 32008.
    else:
        control["actuation"]["scaled_nominal_envelope"]["sample_count"] = 32008.
    with pytest.raises(ValueError, match="invalid diagnostic sample count"):
        analysis.validate_control(control, 8)

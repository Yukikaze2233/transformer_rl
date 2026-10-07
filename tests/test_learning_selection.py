"""CPU fault tests for fixed-grid LR selection; no learner or simulator runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SPEC = importlib.util.spec_from_file_location("selection_under_test", Path(__file__).parents[1] / "tools/select_learning_rate.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
campaign, control = selection.campaign, selection.control
REAL_LOAD_SUITE = selection.load_suite


def put(path, value):
    campaign.write(path, value, replace=Path(path).exists())
    return campaign.artifact(path)


def resign(path, value):
    value = deepcopy(value)
    value.pop("sha256", None)
    return put(path, selection.seal(value))


def terminal(command):
    return {"status": "finished", "returncode": 0, "timed_out": False, "command": command,
            "pid": 999999991, "start": "1"}


def physical_report(height=.01):
    return {"completed_episodes": 8, "success_rate": 1., "success_metric_available": True, "stability": {"available": True},
        "metrics": {"height_abs_error": {"mean": height}, "vx_abs_error": {"mean": .03},
            "wz_abs_error": {"mean": .04}, "tilt_angle": {"mean": .04}, "drift_m": {"max": .02},
            "issued_action_rate_rms": {"mean": 10.}},
        "control": {"full_interval": {"samples": 32008}, "steady": {"total": 8, "eligible": 8, "short": 0, "failed": 0, "partial": 8},
            "episodes": {"completed": 8, "failed": 0, "success_flags": 0},
            "response": {"axes": {axis: {key: 0 for key in ("events", "eligible_steps", "short_hold_candidates", "nonsteady_start",
                "failed", "partial", "rise_censored", "settling_censored")} for axis in ("vx", "wz", "height")}}}}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    root, prepared, source = [tmp_path / name for name in ("runtime", "prepared", "source")]
    names = ["mlp", "mlp_medium", "history_mlp", "history_mlp_wide", "transformer_small", "transformer",
             "transformer_query", "transformer_gated", "transformer_large", "transformer_xlarge"]
    tasks = ["stand_305mm", "forward_05", "backward_05", "rotate_1", "start_stop_05", "height_scan"]
    profiles = ["nominal", "delay20_15", "delay40_30", "delay80_60", "noise", "payload_com", "low_grip", "combined"]
    cases = [f"{task}__{profile}" for task in tasks for profile in profiles] + ["stand_305mm__motor_weak", "stand_305mm__spring_weak"]
    seeds, rates = [1101, 1102, 1103], [1e-5, 3e-5, 1e-4]
    confirmation = {"noise_seeds": selection.NOISE_SEEDS, "scope": "heldout_noise_stream_only", "deterministic_cases": 38,
        "stochastic_noise_or_combined_cases": 12, "new_initial_conditions": False, "new_perturbation_domains": False,
        "no_reselection_on_confirmation": True}
    evaluation = {"validation_seeds": [701, 1701], "seeds": [2701, 3701], "steps": 4001, "settle_steps": 200,
                  "min_steady_samples": 200, "min_completed_episodes": 8}
    scenarios = [{"name": case, "environment": {}, "gates": deepcopy(selection.GATES) + ([deepcopy(selection.DRIFT_GATE)] if case.startswith("stand") else []),
                  "require_steady": True} for case in cases]
    recipe = {"objectives": deepcopy(selection.OBJECTIVES), "min_training_seeds": 3, "std_penalty": 1.,
              "latency_p99_ms": 8., "latency_max_ms": 10., "max_deadline_misses": 0}
    manifest = {"sha256": "1" * 64, "output_root": str(root), "source_root": str(source), "device": "cuda:0",
        "controllers": {}, "runtime": {"checkpoint_runtime": {"torch": "fixture", "numpy": "fixture"}},
        "inputs": {"root": str(prepared), "source": {"sha256": "2" * 64},
            "protocol": {"evaluation": {"case_names": cases}, "variants": names, "training_seeds": seeds, "learning_rates": rates,
                         "confirmation": confirmation}, "children": {}, "cells": []}}
    audited = {"manifest_sha256": manifest["sha256"], "expected_cells": 90, "actual_cells": 90,
        "completed_training_cells": 90, "completed_development_cells": 90, "status": "development_complete", "cells": {}}
    index = {"format": "transformer_rl.learning_latency_index", "schema_version": 1, "campaign_sha256": manifest["sha256"], "cells": {}}
    for rate_index, rate in enumerate(rates):
        rate_id = f"rate_{rate_index:03d}"
        manifest["inputs"]["children"][rate_id] = {"spec": {"selection": deepcopy(recipe), "evaluation": evaluation,
                                                          "scenarios": deepcopy(scenarios)}}
        for variant in names:
            config = {"model": {"variant": variant}, "control": {"policy_dt_s": .01}, "ppo": {"learning_rate": rate},
                      "environment": {"snapshot_sha256": "a" * 64, "num_envs": 1024}}
            config_path = prepared / f"{rate_id}/study/configs/{variant}.train.transfer.json"
            config_receipt = put(config_path, config)
            for case in cases:
                put(prepared / f"{rate_id}/study/configs/{variant}.eval.{case}.json", {**config, "environment": {**config["environment"], "num_envs": 8}})
            for seed in seeds:
                cell = {"rate_id": rate_id, "learning_rate": rate, "variant": variant, "training_seed": seed,
                    "training_config": {"path": str(config_path.relative_to(prepared)), "canonical_sha256": control.digest(config), **config_receipt}}
                cell["training_config"]["path"] = str(config_path.relative_to(prepared))
                key = campaign.cell_key(cell)
                checkpoint_path = root / "cells" / key / "endpoint.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_bytes(key.encode())
                checkpoint = campaign.artifact(checkpoint_path)
                manifest["inputs"]["cells"].append(cell)
                audited["cells"][key] = {"status": "completed", "training": {"status": "completed", "completed_updates": 1200,
                    "charged_updates": 1200, "verified_fresh_samples": campaign.TOTAL, "checkpoint": checkpoint},
                    "development": {str(value): {"status": "completed", "receipt": {"path": str(root / f"dev/{key}/{value}/receipt.json"),
                                                                                 "sha256": "d" * 64}} for value in campaign.DEVELOPMENT_SEEDS}}
                bundle_dir = tmp_path / "latency" / key / "bundle"
                bundle_dir.mkdir(parents=True)
                for name in ("policy.pt", "policy.onnx"):
                    (bundle_dir / name).write_bytes(key.encode() + name.encode())
                bundle = {"format": "transformer_rl.packed_policy", "schema_version": 1, "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_update": 1200, "model": config["model"], "control": config["control"],
                    "history": {"order": "oldest_to_newest", "reset": "repeat_first", "state": "external_window_only"},
                    "files": {name: control.file_sha(bundle_dir / name) for name in ("policy.pt", "policy.onnx")},
                    "runtime_versions": {"torch": "fixture", "numpy": "fixture", "onnx": "fixture", "onnxruntime": "test"},
                    "validation": {"cases": 5, "batch_sizes": [1, 7], "inputs": "zeros, random windows and repeat-first windows",
                                   "torchscript_max_abs_error": 0., "onnx_max_abs_error": 0.}}
                bundle_item = put(bundle_dir / "manifest.json", bundle)
                benchmark_path = bundle_dir.parent / "benchmark.json"
                benchmark = {"backend": "onnx", "threads": 1, "iterations": 1000, "scope": selection.LATENCY_SCOPE,
                    "manifest_sha256": bundle_item["sha256"], "mean_ms": .5, "p99_ms": .8, "max_ms": 1., "deadline_misses": 0,
                    "machine": {"node": "cpu-test", "system": "Linux", "architecture": "x86_64", "processor": "fixture",
                                "cpu_count": 4, "cpu_affinity": [0], "onnxruntime": "test"}}
                benchmark_item = put(benchmark_path, benchmark)
                export_command = campaign.transfer.command(source, "export", "--checkpoint", checkpoint["path"], "--directory", bundle_dir)
                benchmark_command = campaign.transfer.command(source, "benchmark", "--directory", bundle_dir, "--output", benchmark_path,
                                                              "--backend", "onnx", "--threads", 1, "--iterations", 1000)
                receipt = {"format": "transformer_rl.learning_latency_cell", "schema_version": 1, "status": "completed",
                    "identity": {"campaign_sha256": manifest["sha256"], "cell": key, "checkpoint_sha256": checkpoint["sha256"],
                        "checkpoint_update": 1200, "source_sha256": "2" * 64,
                        "configuration_sha256": control.digest(config), "control_sha256": control.digest(config["control"])},
                    "controllers": {}, "parity_reference_max_abs": 10., "bundle": bundle_item, "benchmark": benchmark_item,
                    "export_process": put(bundle_dir.parent / "export.process.json", terminal(export_command)),
                    "benchmark_process": put(bundle_dir.parent / "benchmark.process.json", terminal(benchmark_command))}
                parity_output = bundle_dir.parent / "parity.json"
                parity_request_path = bundle_dir.parent / "parity.request.json"
                parity = selection.parity_request(manifest, cell, checkpoint, parity_output)
                receipt["parity"] = {
                    "request": put(parity_request_path, parity),
                    "output": put(parity_output, {"format": "transformer_rl.learning_export_parity_reference", "schema_version": 1,
                        "identity": parity["identity"], "protocol": selection.PARITY_INPUTS, "reference_max_abs": 10., "cuda_initialized": False}),
                    "process": put(bundle_dir.parent / "parity.process.json", terminal(selection.parity_command(manifest, parity_request_path)))}
                receipt_item = put(bundle_dir.parent / "receipt.json", selection.seal(receipt))
                index["cells"][key] = {"status": "completed", "receipt": receipt_item}
    campaign_manifest = root / "manifest.json"
    put(campaign_manifest, manifest)
    index_path = tmp_path / "latency/index.json"
    resign(index_path, index)
    overrides = {}
    def load_suite(_manifest, cell, checkpoint, item, seed, **kwargs):
        key = campaign.cell_key(cell)
        require = selection.require
        require(seed in (selection.NOISE_SEEDS if kwargs else campaign.DEVELOPMENT_SEEDS), "undeclared eval seed")
        height = .01 + int(cell["rate_id"].rsplit("_", 1)[1]) * .003 + (cell["training_seed"] - 1101) * .001
        base = physical_report(height)
        result = {case: deepcopy(base) for case in cases}
        for (target, eval_seed, case), changes in overrides.items():
            if target == key and eval_seed == seed:
                result[case].update(deepcopy(changes))
        return result
    monkeypatch.setattr(campaign, "validate", lambda path: manifest)
    monkeypatch.setattr(campaign, "audit", lambda path: deepcopy(audited))
    monkeypatch.setattr(selection, "load_suite", load_suite)
    monkeypatch.setattr(campaign, "workers_in", lambda roots: [])
    monkeypatch.setattr(campaign, "evaluation_inputs", lambda m, cell, checkpoint: ({case:
        prepared / f"{cell['rate_id']}/study/configs/{cell['variant']}.eval.{case}.json" for case in cases}, {}))
    output = tmp_path / "selection"
    definition = selection.prepare(campaign_manifest, output, index_path)
    return {"manifest": manifest, "audit": audited, "definition": definition, "selection_path": output / "manifest.json",
            "index": index, "index_path": index_path, "overrides": overrides, "root": tmp_path,
            "names": names, "cases": cases}


def test_full_ninety_cell_selection_uses_original_score_and_seed_sample_std(ready):
    result = selection.assess(ready["selection_path"])
    assert result["status"] == "selected" and len(result["cells"]) == 90
    assert len(result["choices"]) == 10
    assert {item["rate_id"] for item in result["choices"].values()} == {"rate_000"}
    candidate = result["candidates"]["mlp"][0]
    assert candidate["score"]["count"] == 3
    assert candidate["score"]["mean"] == pytest.approx(-4 + .011 / .03 + .03 / .15 + .1 * 10 / 100)
    assert candidate["score"]["sample_std"] == pytest.approx(.001 / .03)
    assert candidate["rank_score"] == pytest.approx(candidate["score"]["mean"] + candidate["score"]["sample_std"])
    assert "best_transformer" not in result and "best_overall" not in result
    assert result["success_semantics"]["denominator"].startswith("all ended")
    drift = result["cells"]["rate_000/mlp/seed_1101"]["metrics"]["metrics.drift_m.max"]
    assert drift["count"] == drift["expected_count"] == 10 * 4 and drift["censored_or_missing_count"] == 0


def test_case_and_development_seed_weights_do_not_depend_on_episode_counts(ready):
    key = "rate_000/mlp/seed_1101"
    metrics = deepcopy(physical_report()["metrics"])
    metrics["height_abs_error"]["mean"] = .016
    ready["overrides"][(key, 701, ready["cases"][0])] = {"metrics": metrics, "completed_episodes": 80}
    result = selection.assess(ready["selection_path"])
    seed_scores = result["candidates"]["mlp"][0]["training_seed_scores"]
    baseline = -4 + .01 / .03 + .03 / .15 + .1 * 10 / 100
    assert seed_scores["1101"] == pytest.approx(baseline + (.006 / .03) / 50 / 4)
    assert seed_scores["1102"] == pytest.approx(baseline + .001 / .03)


def test_real_suite_bridge_checks_all_fifty_model_control_and_seed_identities(ready, monkeypatch):
    manifest = ready["manifest"]
    cell = manifest["inputs"]["cells"][0]
    key = campaign.cell_key(cell)
    checkpoint = ready["audit"]["cells"][key]["training"]["checkpoint"]
    root = Path(manifest["output_root"])
    directory = root / "cells" / key / "development/seed_701/attempt_0000"
    artifacts = {}
    configs, _ = campaign.evaluation_inputs(manifest, cell, checkpoint)
    for case in ready["cases"]:
        configured = selection.read(configs[case])
        report = physical_report()
        report.update(format="transformer_rl.packed_evaluation", schema_version=1, model=configured["model"],
                      control_sha256=control.digest(configured["control"]),
                      environment_provenance={"identity": configured["environment"]["snapshot_sha256"]})
        report["stability"]["protocol"] = {"settle_steps": 200, "min_steady_samples": 200}
        put(directory / f"{case}.json", report)
        artifacts[case] = control.artifact(directory / f"{case}.json", root)
    identity = {"manifest_sha256": manifest["sha256"], "cell": key, "evaluation_seed": 701,
                "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": 1200, "use": "development_only"}
    receipt_item = put(directory / "receipt.json", {"identity": identity, "status": "completed",
        "directory": str(directory.relative_to(root)), "artifacts": artifacts})
    monkeypatch.setattr(selection.diagnostic, "verify_outputs", lambda *args: deepcopy(artifacts))
    assert len(REAL_LOAD_SUITE(manifest, cell, checkpoint, receipt_item, 701)) == 50
    report_path = directory / f"{ready['cases'][0]}.json"
    report = selection.read(report_path)
    report["model"]["variant"] = "older_transfer"
    put(report_path, report)
    artifacts[ready["cases"][0]] = control.artifact(report_path, root)
    receipt = selection.read(receipt_item["path"])
    receipt["artifacts"] = artifacts
    receipt_item = put(receipt_item["path"], receipt)
    with pytest.raises(ValueError, match="model/control/source"):
        REAL_LOAD_SUITE(manifest, cell, checkpoint, receipt_item, 701)


@pytest.mark.parametrize("failure", ["runtime", "development", "latency_index", "latency_cell"])
def test_any_missing_evidence_is_global_not_ready_and_cannot_write_choice(ready, failure):
    key = next(iter(ready["audit"]["cells"]))
    if failure in {"runtime", "development"}:
        ready["audit"]["cells"][key]["status"] = "not_ready"
        ready["audit"]["status"] = "not_ready"
    elif failure == "latency_index":
        ready["index_path"].unlink()
    else:
        del ready["index"]["cells"][key]
        resign(ready["index_path"], ready["index"])
    result = selection.seal_choice(ready["selection_path"])
    assert result["status"] == "not_ready" and result["missing"]
    assert not (Path(ready["definition"]["output_root"]) / "choice.json").exists()


def test_missing_numeric_objective_is_not_ready_not_no_eligible(ready):
    key = next(iter(ready["audit"]["cells"]))
    broken = deepcopy(physical_report()["metrics"])
    broken["issued_action_rate_rms"]["mean"] = None
    ready["overrides"][(key, 701, ready["cases"][0])] = {"metrics": broken}
    result = selection.seal_choice(ready["selection_path"])
    assert result["status"] == "not_ready"
    assert result["choices"] == {}


def test_full_numeric_gate_failures_are_no_eligible_not_missing(ready):
    for key in ready["audit"]["cells"]:
        ready["overrides"][(key, 701, ready["cases"][0])] = {"success_rate": .94}
    result = selection.seal_choice(ready["selection_path"])
    assert result["status"] == "no_eligible_rate" and not result["missing"]
    assert all(item["status"] == "no_eligible_rate" for item in result["choices"].values())
    assert not (Path(ready["definition"]["output_root"]) / "choice.json").exists()


@pytest.mark.parametrize("gate", ["wz_abs_error", "tilt_angle", "drift_m"])
def test_yaw_tilt_and_stationary_drift_remain_joint_gates(ready, gate):
    key = "rate_000/mlp/seed_1101"
    metrics = deepcopy(physical_report()["metrics"])
    metrics[gate]["max" if gate == "drift_m" else "mean"] = .3
    ready["overrides"][(key, 701, ready["cases"][0])] = {"metrics": metrics}
    result = selection.assess(ready["selection_path"])
    assert result["choices"]["mlp"]["rate_id"] == "rate_001"
    assert result["choices"]["transformer"]["rate_id"] == "rate_000"
    assert not result["candidates"]["mlp"][0]["eligible"]


def test_low_episode_count_and_observed_no_steady_are_numeric_ineligibility(ready):
    for key in ready["audit"]["cells"]:
        ready["overrides"][(key, 701, ready["cases"][0])] = {"completed_episodes": 7, "stability": {"available": False}}
    result = selection.assess(ready["selection_path"])
    assert result["status"] == "no_eligible_rate"
    reasons = result["candidates"]["mlp"][0]["failed_gates"]
    assert any("completed_episodes" in reason for reason in reasons)
    assert any("stability.available" in reason for reason in reasons)


def test_valid_zero_ended_null_success_is_censored_ineligible_without_global_missing(ready):
    for key in ready["audit"]["cells"]:
        ready["overrides"][(key, 701, ready["cases"][0])] = {"completed_episodes": 0, "success_rate": None}
    result = selection.assess(ready["selection_path"])
    assert result["status"] == "no_eligible_rate" and not result["missing"]
    candidate = result["candidates"]["mlp"][0]
    assert candidate["rank_score"] is None and candidate["score"] is None
    assert candidate["training_seed_scores"] == {"1101": None, "1102": None, "1103": None}
    assert any("no_ended_episodes" in reason for reason in candidate["score_unavailable"])
    assert any("completed_episodes" in reason for reason in candidate["failed_gates"])
    training_metric = candidate["metrics_equal_training_seed_summary"]["success_rate"]
    assert training_metric["mean"] is None and training_metric["sample_std"] is None
    assert training_metric["count"] == 0 and training_metric["expected_count"] == 3 and training_metric["censored_or_missing_count"] == 3
    measured = result["cells"]["rate_000/mlp/seed_1101"]["metrics"]["success_rate"]
    assert measured["mean"] is None and measured["sample_std"] is None
    assert measured["count"] == 199 and measured["expected_count"] == 200 and measured["censored_or_missing_count"] == 1


def test_zero_ended_censoring_does_not_hide_other_missing_metrics(ready):
    key = "rate_000/mlp/seed_1101"
    metrics = deepcopy(physical_report()["metrics"])
    metrics["height_abs_error"]["mean"] = None
    ready["overrides"][(key, 701, ready["cases"][0])] = {"completed_episodes": 0, "success_rate": None, "metrics": metrics}
    result = selection.assess(ready["selection_path"])
    assert result["status"] == "not_ready" and any("height_abs_error" in item for item in result["missing"])


def test_positive_episode_count_null_success_is_missing(ready):
    ready["overrides"][("rate_000/mlp/seed_1101", 701, ready["cases"][0])] = {"completed_episodes": 8, "success_rate": None}
    assert selection.assess(ready["selection_path"])["status"] == "not_ready"


@pytest.mark.parametrize("fault", ["checkpoint", "source", "config", "machine", "graph", "backend", "process", "extra_cell"])
def test_old_or_corrupt_latency_is_rejected_never_borrowed(ready, fault):
    key = next(iter(ready["index"]["cells"]))
    item = ready["index"]["cells"][key]["receipt"]
    receipt = selection.read(item["path"])
    if fault in {"checkpoint", "source", "config"}:
        field = {"checkpoint": "checkpoint_sha256", "source": "source_sha256", "config": "configuration_sha256"}[fault]
        receipt["identity"][field] = "f" * 64
    elif fault == "graph":
        (Path(receipt["bundle"]["path"]).parent / "policy.onnx").write_bytes(b"old transfer graph")
    elif fault in {"machine", "backend"}:
        benchmark = selection.read(receipt["benchmark"]["path"])
        if fault == "machine":
            benchmark["machine"]["node"] = "another-host"
        else:
            benchmark["backend"] = "torchscript"
        receipt["benchmark"] = put(receipt["benchmark"]["path"], benchmark)
    elif fault == "process":
        process = selection.read(receipt["export_process"]["path"])
        process["returncode"] = 1
        receipt["export_process"] = put(receipt["export_process"]["path"], process)
    else:
        ready["index"]["cells"]["foreign/mlp/seed_1101"] = ready["index"]["cells"][key]
    ready["index"]["cells"][key]["receipt"] = resign(item["path"], receipt)
    resign(ready["index_path"], ready["index"])
    with pytest.raises(ValueError):
        selection.seal_choice(ready["selection_path"])
    assert not (Path(ready["definition"]["output_root"]) / "choice.json").exists()


def test_complete_latency_over_deadline_is_ineligible_not_not_ready(ready):
    for key, index_item in ready["index"]["cells"].items():
        receipt = selection.read(index_item["receipt"]["path"])
        benchmark = selection.read(receipt["benchmark"]["path"])
        benchmark.update(p99_ms=8.1, max_ms=10.1, deadline_misses=1)
        receipt["benchmark"] = put(receipt["benchmark"]["path"], benchmark)
        index_item["receipt"] = resign(index_item["receipt"]["path"], receipt)
    resign(ready["index_path"], ready["index"])
    assert selection.assess(ready["selection_path"])["status"] == "no_eligible_rate"


@pytest.mark.parametrize("fault", ["iterations", "large_error", "abs_only_error"])
def test_latency_exact_iterations_and_original_relative_absolute_parity(ready, fault):
    key = next(iter(ready["index"]["cells"]))
    receipt_item = ready["index"]["cells"][key]["receipt"]
    receipt = selection.read(receipt_item["path"])
    if fault == "iterations":
        benchmark = selection.read(receipt["benchmark"]["path"])
        benchmark["iterations"] = 1001
        receipt["benchmark"] = put(receipt["benchmark"]["path"], benchmark)
    else:
        bundle = selection.read(receipt["bundle"]["path"])
        bundle["validation"]["torchscript_max_abs_error"] = 1000. if fault == "large_error" else 9e-5
        bundle["validation"]["onnx_max_abs_error"] = 1000. if fault == "large_error" else 9e-4
        receipt["bundle"] = put(receipt["bundle"]["path"], bundle)
        benchmark = selection.read(receipt["benchmark"]["path"])
        benchmark["manifest_sha256"] = receipt["bundle"]["sha256"]
        receipt["benchmark"] = put(receipt["benchmark"]["path"], benchmark)
    ready["index"]["cells"][key]["receipt"] = resign(receipt_item["path"], receipt)
    resign(ready["index_path"], ready["index"])
    if fault == "abs_only_error":
        assert selection.assess(ready["selection_path"])["status"] == "selected"
    else:
        with pytest.raises(ValueError):
            selection.assess(ready["selection_path"])


@pytest.mark.parametrize("fault", ["missing", "torch", "numpy", "onnx_empty", "onnxruntime"])
def test_actual_export_runtime_versions_must_match_training_and_benchmark(ready, fault):
    key = next(iter(ready["index"]["cells"]))
    item = ready["index"]["cells"][key]["receipt"]
    receipt = selection.read(item["path"])
    bundle = selection.read(receipt["bundle"]["path"])
    if fault == "missing":
        del bundle["runtime_versions"]
    else:
        field = "onnx" if fault == "onnx_empty" else fault
        bundle["runtime_versions"][field] = "" if fault == "onnx_empty" else "other-version"
    receipt["bundle"] = put(receipt["bundle"]["path"], bundle)
    benchmark = selection.read(receipt["benchmark"]["path"])
    benchmark["manifest_sha256"] = receipt["bundle"]["sha256"]
    receipt["benchmark"] = put(receipt["benchmark"]["path"], benchmark)
    ready["index"]["cells"][key]["receipt"] = resign(item["path"], receipt)
    resign(ready["index_path"], ready["index"])
    with pytest.raises(ValueError, match="runtime"):
        selection.seal_choice(ready["selection_path"])
    assert not (Path(ready["definition"]["output_root"]) / "choice.json").exists()


@pytest.mark.parametrize("fault", ["reference", "protocol", "checkpoint", "command"])
def test_parity_probe_cannot_substitute_old_cp_or_unmeasured_range(ready, fault):
    key = next(iter(ready["index"]["cells"]))
    item = ready["index"]["cells"][key]["receipt"]
    receipt = selection.read(item["path"])
    if fault == "reference":
        receipt["parity_reference_max_abs"] = 100000.
    elif fault == "command":
        process = selection.read(receipt["parity"]["process"]["path"])
        process["command"][-1] = "/wrong/request.json"
        receipt["parity"]["process"] = put(receipt["parity"]["process"]["path"], process)
    else:
        output = selection.read(receipt["parity"]["output"]["path"])
        if fault == "protocol":
            output["protocol"]["seed"] = 1
        else:
            output["identity"]["checkpoint_sha256"] = "f" * 64
        receipt["parity"]["output"] = put(receipt["parity"]["output"]["path"], output)
    ready["index"]["cells"][key]["receipt"] = resign(item["path"], receipt)
    resign(ready["index_path"], ready["index"])
    with pytest.raises(ValueError):
        selection.assess(ready["selection_path"])


def test_post_audit_incomplete_suite_remains_global_not_ready(ready, monkeypatch):
    original = selection.load_suite
    def incomplete(manifest, cell, checkpoint, item, seed, **kwargs):
        if campaign.cell_key(cell) == "rate_000/mlp/seed_1101" and seed == 701:
            return None
        return original(manifest, cell, checkpoint, item, seed, **kwargs)
    monkeypatch.setattr(selection, "load_suite", incomplete)
    result = selection.seal_choice(ready["selection_path"])
    assert result["status"] == "not_ready" and result["choices"] == {}


def test_choice_is_exclusive_and_later_evidence_mutation_blocks_confirmation(ready):
    choice = selection.seal_choice(ready["selection_path"])
    path = Path(ready["definition"]["output_root"]) / "choice.json"
    before = path.read_bytes()
    assert selection.sealed(selection.read(path)) == choice
    with pytest.raises(ValueError, match="immutable choice"):
        selection.seal_choice(ready["selection_path"])
    assert path.read_bytes() == before
    ready["overrides"][("rate_000/mlp/seed_1101", 701, ready["cases"][0])] = {"success_rate": .99}
    with pytest.raises(ValueError, match="changed after LR choice"):
        selection.prepare_confirmation(ready["selection_path"], path, ready["root"] / "confirm")
    assert path.read_bytes() == before


def sealed_confirmation(ready):
    selection.seal_choice(ready["selection_path"])
    choice = Path(ready["definition"]["output_root"]) / "choice.json"
    confirm = selection.prepare_confirmation(ready["selection_path"], choice, ready["root"] / "confirm")
    return choice, confirm, Path(confirm["output_root"]) / "manifest.json"


def test_confirmation_preparation_is_unexecuted_same_lr_original_seeds_noise_only(ready):
    choice, confirm, path = sealed_confirmation(ready)
    before = choice.read_bytes()
    assert confirm["execution_implemented"] is False and confirm["no_reselection"] is True
    assert len(confirm["requests"]) == 60
    assert {request["evaluation_seed"] for request in confirm["requests"]} == {11701, 12701}
    assert {request["training_seed"] for request in confirm["requests"]} == {1101, 1102, 1103}
    assert {request["cell"].split("/")[0] for request in confirm["requests"]} == {"rate_000"}
    assert all("evaluate-suite" in request["command"] and "train" not in request["command"] for request in confirm["requests"])
    assert selection.audit_confirmation(path)["status"] == "not_ready"
    assert choice.read_bytes() == before
    assert not (Path(confirm["output_root"]) / "evaluations").exists()


@pytest.mark.parametrize("fault", ["rate", "seed", "training_seed", "checkpoint", "command", "scope"])
def test_confirmation_cannot_reselect_resample_or_change_scope(ready, fault):
    choice, confirm, path = sealed_confirmation(ready)
    if fault == "rate":
        confirm["requests"][0]["cell"] = confirm["requests"][0]["cell"].replace("rate_000", "rate_001")
    elif fault == "seed":
        confirm["requests"][0]["evaluation_seed"] = 3701
    elif fault == "training_seed":
        confirm["requests"][0]["training_seed"] = 1104
    elif fault == "checkpoint":
        confirm["requests"][0]["checkpoint"] = confirm["requests"][2]["checkpoint"]
    elif fault == "command":
        confirm["requests"][0]["command"][-1] = "1"
    else:
        confirm["protocol"]["new_initial_conditions"] = True
    resign(path, confirm)
    with pytest.raises(ValueError):
        selection.audit_confirmation(path)
    assert selection.read(choice)["choices"]["mlp"]["rate_id"] == "rate_000"


def test_confirmation_failure_retains_original_choice_and_has_no_reselection(ready):
    choice, confirm, path = sealed_confirmation(ready)
    choice_bytes = choice.read_bytes()
    for request in confirm["requests"]:
        directory = Path(confirm["output_root"]) / request["directory"]
        put(directory / "receipt.json", {"test": "suite reader is isolated in this CPU fault test"})
        put(directory / "worker.process.json", terminal(request["command"]))
        ready["overrides"][(request["cell"], request["evaluation_seed"], ready["cases"][0])] = {"success_rate": .94}
    result = selection.audit_confirmation(path)
    assert result["status"] == "not_confirmed" and result["no_reselection"]
    assert result["expected_suites"] == result["actual_suites"] == 60
    assert choice.read_bytes() == choice_bytes
    assert "candidates" not in result and result["original_choices"]["mlp"]["rate_id"] == "rate_000"


def test_complete_confirmation_reports_only_noise_stream_confirmation(ready):
    choice, confirm, path = sealed_confirmation(ready)
    before = choice.read_bytes()
    for request in confirm["requests"]:
        directory = Path(confirm["output_root"]) / request["directory"]
        put(directory / "receipt.json", {"test": "suite reader is isolated in this CPU fault test"})
        put(directory / "worker.process.json", terminal(request["command"]))
    result = selection.audit_confirmation(path)
    assert result["status"] == "confirmed"
    assert result["scope"] == "heldout_noise_stream_only"
    assert result["new_initial_conditions"] is False and result["new_perturbation_domains"] is False
    assert result["formal_architecture_selection"] is False and result["hardware_deployment_ready"] is False
    assert choice.read_bytes() == before


def test_selector_helper_and_original_objectives_are_frozen(ready, monkeypatch):
    old = selection.helper_identity()
    monkeypatch.setattr(selection, "helper_identity", lambda: {**old, "other": "f" * 64})
    with pytest.raises(ValueError, match="bytes changed"):
        selection.validate(ready["selection_path"])
    monkeypatch.setattr(selection, "helper_identity", lambda: old)
    ready["manifest"]["inputs"]["children"]["rate_001"]["spec"]["selection"]["objectives"][0]["weight"] = 1.
    with pytest.raises(ValueError):
        selection.validate(ready["selection_path"])


def test_duplicate_json_nonfinite_and_symlink_receipts_reject(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"status": "completed", "status": "failed"}')
    with pytest.raises(ValueError, match="duplicate"):
        selection.read(path)
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        selection.read(path)
    path.write_text('{"value": 1e999}')
    with pytest.raises(ValueError):
        selection.read(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        selection.read(link)


def test_actual_cpu_parity_probe_measures_raw_mean_without_clipping_or_training(tmp_path):
    import torch
    from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig
    from transformer_rl.frame_policy import FramePolicyConfig
    from transformer_rl.frame_training import FrameActorCritic
    from transformer_rl.frame_checkpoint import save_frame_checkpoint
    from transformer_rl.ppo import PPOTrainer
    from transformer_rl.config import PPOConfig
    source = Path(__file__).parents[1]
    contract = {"policy_dt_s": .01, "observation_schema": "cpu_parity_fixture", "feature_names": [f"f_{i}" for i in range(5)],
        "action_names": ["leg", "wheel"], "action_bounds": [.2, .5], "target_scale": [.25, 10.],
        "target_offset": [.1, -.2], "target_units": ["rad", "rad/s"]}
    config = FrameTrainConfig(FrameModelConfig(policy=FramePolicyConfig(architecture="mlp", frame_dim=5, action_dim=2,
        history_length=1, actor_hidden_dims=(4,)), critic_dim=3, critic_hidden=(4,)), PPOConfig(), contract, {})
    model = FrameActorCritic(config.model)
    with torch.no_grad():
        for parameter in model.actor.policy.parameters():
            parameter.zero_()
        model.actor.policy.output_layer.bias.fill_(32.)
    checkpoint = tmp_path / "cpu_fixture.pt"
    # An untrained synthetic CPU fixture with a 1200 label tests the probe only;
    # no campaign can admit this fixture without the real runtime training ledger.
    save_frame_checkpoint(checkpoint, model, PPOTrainer(model, config.ppo), config, 1200, {"fixture": True})
    output, request_path, cache = tmp_path / "reference.json", tmp_path / "request.json", tmp_path / "empty-cache"
    cache.mkdir()
    request = {"format": "transformer_rl.learning_export_parity_request", "schema_version": 1,
        "identity": {"fixture": "CPU-only"}, "source_root": str(source), "checkpoint": campaign.artifact(checkpoint),
        "output": str(output), "protocol": selection.PARITY_INPUTS}
    put(request_path, request)
    environment = campaign.cpu_environment(source)
    environment["PYTHONPYCACHEPREFIX"] = str(cache)
    process = subprocess.run(selection.parity_command({"source_root": str(source)}, request_path),
                             env=environment, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    result = selection.read(output)
    assert result["reference_max_abs"] == 32. and not result["cuda_initialized"]
    assert result["protocol"] == selection.PARITY_INPUTS and not list(cache.iterdir())
    assert result["reference_max_abs"] > max(abs(offset) + abs(scale) * bound for offset, scale, bound in
                                            zip(contract["target_offset"], contract["target_scale"], contract["action_bounds"]))

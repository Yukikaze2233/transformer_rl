#!/usr/bin/env python3
"""Architecture-neutral LR selection and immutable noise-only confirmation plans.

This file never trains, exports, benchmarks, or launches a simulator. The real
90-cell campaign audit is mandatory; latency and confirmation are separately
produced evidence. No generic architecture selector is invoked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import statistics
import uuid


_SPEC = importlib.util.spec_from_file_location("learning_selection_campaign", Path(__file__).with_name("run_learning_campaign.py"))
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)
control, diagnostic = campaign.control, campaign.diagnostic
FORMAT = "transformer_rl.learning_rate_selection"
CONFIRMATION_FORMAT = "transformer_rl.learning_rate_confirmation"
NOISE_SEEDS = [11701, 12701]
LATENCY_SCOPE = "CPU history + mean inference + target mapping; excludes sensor I/O, transport and torque control"
OBJECTIVES = [
    {"path": "success_rate", "direction": "maximize", "scale": 1., "weight": 4.},
    {"path": "metrics.height_abs_error.mean", "direction": "minimize", "scale": .03, "weight": 1.},
    {"path": "metrics.vx_abs_error.mean", "direction": "minimize", "scale": .15, "weight": 1.},
    {"path": "metrics.issued_action_rate_rms.mean", "direction": "minimize", "scale": 100., "weight": .1},
]
GATES = [
    {"path": "success_rate", "operator": "min", "value": .95},
    {"path": "metrics.height_abs_error.mean", "operator": "max", "value": .03},
    {"path": "metrics.vx_abs_error.mean", "operator": "max", "value": .15},
    {"path": "metrics.wz_abs_error.mean", "operator": "max", "value": .25},
    {"path": "metrics.tilt_angle.mean", "operator": "max", "value": .25},
]
DRIFT_GATE = {"path": "metrics.drift_m.max", "operator": "max", "value": .20}
METRICS = ["success_rate", "metrics.height_abs_error.mean", "metrics.vx_abs_error.mean",
           "metrics.wz_abs_error.mean", "metrics.tilt_angle.mean", "metrics.drift_m.max",
           "metrics.issued_action_rate_rms.mean"]
SUCCESS_SEMANTICS = {
    "label": "ended_nonterminated_task_or_ordinary_truncation",
    "predicate": "done & ~terminated & (diagnostic.success | (truncated & ordinary))",
    "denominator": "all ended episodes; boundary and blocked truncations are retained",
    "interpretation": "not joint tracking, task-complete-only, or uninterrupted full-horizon survival",
    "control_success_flags": "physical packet diagnostic.success; distinct from evaluation success_rate",
}
PARITY_INPUTS = {"seed": 9271, "cases": 5, "batch_sizes": [1, 7],
                 "inputs": "zeros, random windows and repeat-first windows", "output": "raw mean before clipping or target mapping"}
# This is a reproducible CPU-only producer command definition. The selector
# never runs it; the independent latency producer owns its request/process/output.
PARITY_PROBE_SCRIPT = """import hashlib,json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0,str(Path(request['source_root'])/'src'))
import torch
from transformer_rl.frame_checkpoint import load_frame_checkpoint
assert not torch.cuda.is_initialized()
checkpoint=Path(request['checkpoint']['path'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==request['checkpoint']['sha256']
model,_,config,update,_,_=load_frame_checkpoint(checkpoint)
assert update==1200
policy=model.actor.policy.eval()
generator=torch.Generator().manual_seed(9271)
cases=[torch.zeros(1,config.model.history_length,config.model.frame_dim)]
for batch in (1,7):
    frames=torch.randn(batch,config.model.history_length,config.model.frame_dim,generator=generator)
    cases.extend((frames,frames[:,-1:].expand_as(frames).clone()))
with torch.no_grad():
    outputs=[policy(frames) for frames in cases]
assert all(torch.isfinite(value).all() for value in outputs)
assert not torch.cuda.is_initialized()
result={'format':'transformer_rl.learning_export_parity_reference','schema_version':1,
        'identity':request['identity'],'protocol':request['protocol'],
        'reference_max_abs':max(float(value.abs().max()) for value in outputs),'cuda_initialized':False}
with Path(request['output']).open('x') as stream:
    stream.write(json.dumps(result,allow_nan=False)+'\\n')
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    """Strict JSON, including duplicate-key and non-finite rejection."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    result = json.loads(campaign.plain(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(result, allow_nan=False)
    return result


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def value_at(report, route):
    value = report
    for part in route.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if number(value) else None


def sealed(value):
    require(isinstance(value, dict) and value.get("sha256") == control.digest(
        {key: item for key, item in value.items() if key != "sha256"}), "canonical object seal differs")
    return value


def seal(value):
    require("sha256" not in value, "cannot reseal an existing object")
    return {**value, "sha256": control.digest(value)}


def helper_identity():
    return {**campaign.controllers(), str(campaign.plain(__file__)): control.file_sha(__file__)}


def stats(values):
    require(bool(values) and all(number(value) for value in values), "finite nonempty observations required")
    return {"count": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def terminal_process(process, command):
    return (process.get("command") == command and process.get("status") == "finished"
            and type(process.get("returncode")) is int and process["returncode"] == 0 and not process.get("timed_out", False)
            and type(process.get("pid")) is int and process["pid"] > 0
            and isinstance(process.get("start"), str) and process["start"].isdigit()
            and control.process_start(process["pid"]) != process["start"])


def protocol(manifest):
    inputs = manifest["inputs"]
    specs = [child["spec"] for child in inputs["children"].values()]
    reference = specs[0]
    require(all(spec["selection"] == reference["selection"] and spec["scenarios"] == reference["scenarios"]
                and spec["evaluation"] == reference["evaluation"] for spec in specs), "paired LR selection recipes differ")
    recipe = reference["selection"]
    require(recipe["objectives"] == OBJECTIVES and recipe["min_training_seeds"] == 3
            and recipe["std_penalty"] == 1. and recipe["latency_p99_ms"] == 8.
            and recipe["latency_max_ms"] == 10. and recipe["max_deadline_misses"] == 0,
            "frozen parent objectives or latency recipe differs")
    cases = inputs["protocol"]["evaluation"]["case_names"]
    require([case["name"] for case in reference["scenarios"]] == cases and len(cases) == 50,
            "fifty ordered case gates required")
    for case in reference["scenarios"]:
        require(case["gates"] == GATES + ([DRIFT_GATE] if case["name"].startswith("stand") else [])
                and case.get("require_steady") is True, "original joint task gates differ")
    evaluation = reference["evaluation"]
    require(evaluation == {"validation_seeds": [701, 1701], "seeds": [2701, 3701], "steps": 4001,
            "settle_steps": 200, "min_steady_samples": 200, "min_completed_episodes": 8}, "evaluation coverage differs")
    confirm = inputs["protocol"]["confirmation"]
    require(confirm["noise_seeds"] == NOISE_SEEDS and confirm["scope"] == "heldout_noise_stream_only"
            and confirm["deterministic_cases"] == 38 and confirm["stochastic_noise_or_combined_cases"] == 12
            and confirm["new_initial_conditions"] is False and confirm["new_perturbation_domains"] is False
            and confirm["no_reselection_on_confirmation"] is True, "noise-only confirmation scope differs")
    return {"expected_cells": 90, "development_seeds": list(campaign.DEVELOPMENT_SEEDS),
        "case_names": cases, "training_seeds": inputs["protocol"]["training_seeds"],
        "variants": inputs["protocol"]["variants"], "learning_rates": inputs["protocol"]["learning_rates"],
        "objectives": OBJECTIVES, "objectives_sha256": control.digest(OBJECTIVES),
        "scenarios": reference["scenarios"], "evaluation": evaluation, "std_penalty": 1.,
        "weighting": "equal cases then equal development seeds then equal independent training seeds",
        "ranking": "all original gates first; original objective mean + training-seed sample std; lower LR breaks exact ties",
        "architecture_neutral": True, "architecture_winner_implemented": False,
        "latency": {"backend": "onnx", "threads": 1, "iterations": 1000, "warmup": 50,
            "p99_ms": 8., "max_ms": 10., "max_deadline_misses": 0, "scope": LATENCY_SCOPE,
            "same_machine_for_all_ninety_cells": True,
            "parity": {"reference": "measured raw mean maximum over frozen export generator9271 five input cases",
                "torchscript": {"rtol": 1e-5, "atol": 1e-6}, "onnx": {"rtol": 1e-4, "atol": 1e-5},
                "selector_check": "necessary manifest sanity bound; elementwise allclose performed by successful frozen export producer"}},
        "confirmation": {**confirm, "noise_seeds": list(NOISE_SEEDS)}, "success_semantics": SUCCESS_SEMANTICS}


def independent_root(root, forbidden):
    root = campaign.plain(root)
    require(all(root != path and not root.is_relative_to(path) and not path.is_relative_to(root)
                for path in map(campaign.plain, forbidden)), "outputs overlap immutable evidence")
    require(not root.exists(), "output root already exists")
    return root


def prepare(campaign_manifest, output_root, latency_index):
    manifest = campaign.validate(campaign_manifest)
    root = independent_root(output_root, [manifest["output_root"], manifest["source_root"], manifest["inputs"]["root"]])
    latency_index = campaign.plain(latency_index)
    require(not latency_index.is_relative_to(root), "latency input cannot be a selection output")
    definition = {"format": FORMAT, "schema_version": 1, "output_root": str(root),
        "campaign_manifest": campaign.artifact(campaign_manifest), "campaign_sha256": manifest["sha256"],
        "helpers": helper_identity(), "protocol": protocol(manifest), "latency_index": str(latency_index),
        "formal_architecture_selection": False, "hardware_deployment_ready": False}
    root.mkdir(parents=True, exist_ok=False)
    lock = root / ".selection.lock"
    lock.touch(exist_ok=False)
    definition["lock"] = {"path": str(lock), **diagnostic.lock_identity(lock)}
    definition = seal(definition)
    campaign.write(root / "manifest.json", definition)
    return definition


def validate(path):
    path = campaign.plain(path)
    definition = sealed(read(path))
    require(definition.get("format") == FORMAT and definition.get("schema_version") == 1
            and path == Path(definition["output_root"]) / "manifest.json", "selection manifest identity differs")
    require(definition["helpers"] == helper_identity(), "selector or execution auditor bytes changed")
    source = campaign.checked(definition["campaign_manifest"])
    manifest = campaign.validate(source)
    require(manifest["sha256"] == definition["campaign_sha256"] and protocol(manifest) == definition["protocol"],
            "selection recipe or input campaign changed")
    lock = definition["lock"]
    require(diagnostic.lock_identity(lock["path"]) == {key: lock[key] for key in ("device", "inode")},
            "selection lock inode changed")
    require(definition["formal_architecture_selection"] is False and definition["hardware_deployment_ready"] is False,
            "LR evidence cannot claim hardware or architecture qualification")
    return definition, manifest


@contextmanager
def selection_lock(definition):
    lock = definition["lock"]
    with campaign.plain(lock["path"]).open("r+") as stream:
        diagnostic.check_open_lock(stream, {key: lock[key] for key in ("device", "inode")})
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            diagnostic.check_open_lock(stream, {key: lock[key] for key in ("device", "inode")})
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_suite(manifest, cell, checkpoint, receipt_item, seed, *, confirmation=None):
    receipt = read(campaign.checked(receipt_item))
    expected = {"manifest_sha256": manifest["sha256"], "cell": campaign.cell_key(cell), "evaluation_seed": seed,
                "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": 1200, "use": "development_only"}
    root = Path(manifest["output_root"])
    if confirmation:
        root = Path(confirmation["output_root"])
        expected.update(use="heldout_noise_stream_only", choice_sha256=confirmation["choice_sha256"],
                        confirmation_sha256=confirmation["sha256"])
    require(receipt["identity"] == expected, "evaluation receipt identity differs")
    if receipt.get("status") != "completed":
        return None
    directory = campaign.inside(root, receipt["directory"])
    require(campaign.plain(receipt_item["path"]) == directory / "receipt.json", "evaluation receipt route differs")
    configs, shim = campaign.evaluation_inputs(manifest, cell, checkpoint)
    shim["output_root"] = str(root)
    verified = diagnostic.verify_outputs(directory, shim, cell["variant"], seed)
    require(verified == receipt["artifacts"], "evaluation artifact coverage differs")
    cases = manifest["inputs"]["protocol"]["evaluation"]["case_names"]
    reports = {case: read(control.checked(root, receipt["artifacts"][case])) for case in cases}
    for case, report in reports.items():
        config = read(configs[case])
        require(report.get("format") == "transformer_rl.packed_evaluation" and report.get("schema_version") == 1
                and report["model"] == config["model"] and report["control_sha256"] == control.digest(config["control"])
                and report["environment_provenance"]["identity"] == config["environment"]["snapshot_sha256"]
                and report["stability"]["protocol"]["settle_steps"] == 200
                and report["stability"]["protocol"]["min_steady_samples"] == 200,
                "case model/control/source or stability protocol differs")
    return reports


def latency_cell(manifest, cell, checkpoint, item):
    """Accept independently produced CPU evidence bound to this exact new CP."""
    receipt = sealed(read(campaign.checked(item)))
    config_path = campaign.inside(manifest["inputs"]["root"], cell["training_config"]["path"])
    config = read(config_path)
    expected = {"campaign_sha256": manifest["sha256"], "cell": campaign.cell_key(cell),
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": 1200,
        "source_sha256": manifest["inputs"]["source"]["sha256"],
        "configuration_sha256": cell["training_config"]["canonical_sha256"],
        "control_sha256": control.digest(config["control"])}
    require(receipt.get("format") == "transformer_rl.learning_latency_cell" and receipt.get("schema_version") == 1
            and receipt["identity"] == expected and receipt["controllers"] == manifest["controllers"],
            "latency belongs to another source/cell/checkpoint/controller")
    if receipt.get("status") != "completed":
        return None
    bundle_path = campaign.checked(receipt["bundle"])
    bundle, benchmark = read(bundle_path), read(campaign.checked(receipt["benchmark"]))
    commands = {
        "export_process": campaign.transfer.command(Path(manifest["source_root"]), "export", "--checkpoint", checkpoint["path"],
            "--directory", bundle_path.parent),
        "benchmark_process": campaign.transfer.command(Path(manifest["source_root"]), "benchmark", "--directory", bundle_path.parent,
            "--output", receipt["benchmark"]["path"], "--backend", "onnx", "--threads", 1, "--iterations", 1000),
    }
    for name, command in commands.items():
        process = read(campaign.checked(receipt[name]))
        require(terminal_process(process, command), "CPU export/benchmark successful terminal process identity differs")
    require(bundle.get("format") == "transformer_rl.packed_policy" and bundle.get("schema_version") == 1
            and bundle["checkpoint_sha256"] == checkpoint["sha256"] and bundle["checkpoint_update"] == 1200
            and bundle["model"] == config["model"] and bundle["control"] == config["control"], "export identity differs")
    require(bundle["history"] == {"order": "oldest_to_newest", "reset": "repeat_first", "state": "external_window_only"}
            and "policy.onnx" in bundle["files"] and "policy.pt" in bundle["files"], "export histories or runtimes differ")
    for name, sha in bundle["files"].items():
        require(Path(name).name == name and bool(name), "export filename escapes its bundle")
        require(control.file_sha(campaign.inside(bundle_path.parent, name)) == diagnostic.sha(sha), "export graph bytes changed")
    validation = bundle["validation"]
    require(validation.get("batch_sizes") == [1, 7] and validation.get("cases") == 5
            and validation.get("inputs") == "zeros, random windows and repeat-first windows"
            and number(validation.get("torchscript_max_abs_error")) and validation["torchscript_max_abs_error"] >= 0
            and number(validation.get("onnx_max_abs_error")) and validation["onnx_max_abs_error"] >= 0,
            "independent export parity evidence absent")
    reference = receipt.get("parity_reference_max_abs")
    require(number(reference) and reference >= 0, "actual frozen-input raw mean parity reference range absent")
    parity = receipt["parity"]
    parity_request_path = campaign.checked(parity["request"])
    parity_output_path = campaign.checked(parity["output"])
    require(read(parity_request_path) == parity_request(manifest, cell, checkpoint, parity_output_path),
            "parity probe request changed its actual source/checkpoint/fixed inputs")
    probe = read(parity_output_path)
    require(probe.get("format") == "transformer_rl.learning_export_parity_reference" and probe.get("schema_version") == 1
            and probe["identity"] == expected and probe["protocol"] == PARITY_INPUTS and probe["cuda_initialized"] is False
            and number(probe.get("reference_max_abs")) and probe["reference_max_abs"] == reference,
            "raw mean reference is not the sealed real CPU probe result")
    process = read(campaign.checked(parity["process"]))
    require(terminal_process(process, parity_command(manifest, parity_request_path)),
            "parity probe process did not complete the original CPU command")
    # The exported raw mean is unbounded: action/target limits cannot bound it.
    # A maximum-reference range gives a necessary sanity bound, not a substitute
    # for the producer's original per-element relative + absolute assertions.
    require(validation["torchscript_max_abs_error"] <= 1e-6 + 1e-5 * reference
            and validation["onnx_max_abs_error"] <= 1e-5 + 1e-4 * reference,
            "export parity maxima contradict the original relative + absolute gates")
    require(benchmark.get("backend") == "onnx" and type(benchmark.get("threads")) is int and benchmark["threads"] == 1
            and type(benchmark.get("iterations")) is int and benchmark["iterations"] == 1000
            and benchmark.get("scope") == LATENCY_SCOPE and benchmark.get("manifest_sha256") == receipt["bundle"]["sha256"],
            "CPU latency protocol or bundle linkage differs")
    require(all(number(benchmark.get(key)) and benchmark[key] >= 0 for key in ("mean_ms", "p99_ms", "max_ms"))
            and benchmark["mean_ms"] <= benchmark["max_ms"] and benchmark["p99_ms"] <= benchmark["max_ms"]
            and type(benchmark.get("deadline_misses")) is int and 0 <= benchmark["deadline_misses"] <= benchmark["iterations"],
            "latency observations invalid")
    machine = benchmark.get("machine")
    require(isinstance(machine, dict) and set(machine) == {"node", "system", "architecture", "processor", "cpu_count", "cpu_affinity", "onnxruntime"}
            and all(isinstance(machine[key], str) and machine[key] for key in ("node", "system", "architecture", "onnxruntime"))
            and type(machine["cpu_count"]) is int and machine["cpu_count"] > 0
            and isinstance(machine["cpu_affinity"], list) and bool(machine["cpu_affinity"])
            and all(type(index) is int and 0 <= index < machine["cpu_count"] for index in machine["cpu_affinity"])
            and machine["cpu_affinity"] == sorted(set(machine["cpu_affinity"])), "measured CPU machine absent")
    versions = bundle.get("runtime_versions")
    checkpoint_runtime = manifest["runtime"]["checkpoint_runtime"]
    require(isinstance(versions, dict) and set(versions) == {"torch", "numpy", "onnx", "onnxruntime"}
            and all(isinstance(versions[name], str) and bool(versions[name]) for name in versions)
            and versions["torch"] == checkpoint_runtime["torch"] and versions["numpy"] == checkpoint_runtime["numpy"]
            and versions["onnxruntime"] == machine["onnxruntime"],
            "export Torch/NumPy or benchmark ONNXRuntime versions differ from the actual sealed runtimes")
    return {"receipt": item, "bundle": receipt["bundle"], "benchmark": receipt["benchmark"],
        "machine_sha256": control.digest(machine), "p99_ms": benchmark["p99_ms"], "max_ms": benchmark["max_ms"],
        "deadline_misses": benchmark["deadline_misses"], "passed": benchmark["p99_ms"] <= 8.
        and benchmark["max_ms"] <= 10. and benchmark["deadline_misses"] == 0}


def parity_request(manifest, cell, checkpoint, output):
    config = read(campaign.inside(manifest["inputs"]["root"], cell["training_config"]["path"]))
    return {"format": "transformer_rl.learning_export_parity_request", "schema_version": 1,
        "identity": {"campaign_sha256": manifest["sha256"], "cell": campaign.cell_key(cell),
            "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": 1200,
            "source_sha256": manifest["inputs"]["source"]["sha256"],
            "configuration_sha256": cell["training_config"]["canonical_sha256"],
            "control_sha256": control.digest(config["control"])},
        "source_root": manifest["source_root"], "checkpoint": checkpoint,
        "output": str(campaign.plain(output)), "protocol": PARITY_INPUTS}


def parity_command(manifest, request_path):
    executable = campaign.transfer.command(Path(manifest["source_root"]), "export")[0]
    return [executable, "-c", PARITY_PROBE_SCRIPT, str(campaign.plain(request_path))]


def load_latency(definition, manifest, audit):
    path = Path(definition["latency_index"])
    if not path.exists():
        return {}, ["latency_index_absent"], None
    index = sealed(read(path))
    require(index.get("format") == "transformer_rl.learning_latency_index" and index.get("schema_version") == 1
            and index.get("campaign_sha256") == manifest["sha256"], "latency index identity differs")
    keys = {campaign.cell_key(cell) for cell in manifest["inputs"]["cells"]}
    require(isinstance(index.get("cells"), dict) and not set(index["cells"]) - keys, "latency index has foreign cells")
    result, missing = {}, []
    for cell in manifest["inputs"]["cells"]:
        key = campaign.cell_key(cell)
        item = index["cells"].get(key)
        if not item or item.get("status") != "completed" or not item.get("receipt"):
            missing.append(f"{key}:latency")
            continue
        checkpoint = audit["cells"][key]["training"]["checkpoint"]
        measured = latency_cell(manifest, cell, checkpoint, item["receipt"])
        if measured is None:
            missing.append(f"{key}:latency")
        else:
            result[key] = measured
    require(len({item["machine_sha256"] for item in result.values()}) <= 1, "all ninety latency cells require the same measured CPU")
    return result, missing, campaign.artifact(path)


def grade(report, scenario, evaluation):
    missing, failures, unavailable = [], [], []
    completed = report.get("completed_episodes")
    if type(completed) is not int or completed < 0:
        missing.append("completed_episodes")
    elif completed < evaluation["min_completed_episodes"]:
        failures.append("completed_episodes")
    zero_ended = (type(completed) is int and completed == 0 and "success_rate" in report
                  and report["success_rate"] is None and report.get("success_metric_available") is True)
    if zero_ended:
        unavailable.append("success_rate:no_ended_episodes")
    stability = report.get("stability", {})
    if type(stability.get("available")) is not bool:
        missing.append("stability.available")
    elif scenario["require_steady"] and not stability["available"]:
        failures.append("stability.available")
    for gate in scenario["gates"]:
        value = value_at(report, gate["path"])
        if value is None:
            if not (zero_ended and gate["path"] == "success_rate"):
                missing.append(gate["path"])
        elif (value < gate["value"] if gate["operator"] == "min" else value > gate["value"]):
            failures.append(gate["path"])
        if value is not None:
            require(value >= 0 and (gate["path"] != "success_rate" or value <= 1), "task metric has an impossible numeric range")
    score = 0.
    for objective in OBJECTIVES:
        value = value_at(report, objective["path"])
        if value is None:
            if not (zero_ended and objective["path"] == "success_rate"):
                missing.append(objective["path"])
        else:
            require(value >= 0 and (objective["path"] != "success_rate" or value <= 1), "objective has an impossible numeric range")
            score += objective["weight"] * value / objective["scale"] * (1 if objective["direction"] == "minimize" else -1)
    return {"missing": sorted(set(missing)), "failed": sorted(set(failures)),
            "score_unavailable": unavailable, "score": score if not missing and not unavailable else None}


def summarize_training_seed(reports, scenarios, evaluation):
    """No episodes, seeds, or unsteady cases are removed from any denominator."""
    require(set(reports) in (set(campaign.DEVELOPMENT_SEEDS), set(NOISE_SEEDS)), "evaluation seed pool differs")
    cases = [case["name"] for case in scenarios]
    require(all(set(pool) == set(cases) for pool in reports.values()), "fifty-case evaluation coverage differs")
    scores, missing, failed, unavailable, metrics = [], [], [], [], {key: [] for key in METRICS}
    counters = {"full_interval_samples": 0, "steady_total_segments": 0, "steady_eligible_segments": 0,
                "steady_short_segments": 0, "steady_failed_segments": 0, "steady_partial_segments": 0,
                "packet_completed_episodes": 0, "packet_failed_episodes": 0, "packet_success_flags": 0}
    responses = {axis: {key: 0 for key in ("events", "eligible_steps", "short_hold_candidates", "nonsteady_start",
                "failed", "partial", "rise_censored", "settling_censored")} for axis in ("vx", "wz", "height")}
    for seed, pool in reports.items():
        seed_scores = []
        for scenario in scenarios:
            case, report = scenario["name"], pool[scenario["name"]]
            result = grade(report, scenario, evaluation)
            missing.extend(f"{seed}/{case}:{key}" for key in result["missing"])
            failed.extend(f"{seed}/{case}:{key}" for key in result["failed"])
            unavailable.extend(f"{seed}/{case}:{key}" for key in result["score_unavailable"])
            if result["score"] is not None:
                seed_scores.append(result["score"])
            for key in metrics:
                if key == "metrics.drift_m.max" and not case.startswith("stand"):
                    continue
                value = value_at(report, key)
                if value is not None:
                    metrics[key].append(value)
            ctl = report["control"]
            counters["full_interval_samples"] += ctl["full_interval"]["samples"]
            for suffix, field in (("total", "total"), ("eligible", "eligible"), ("short", "short"), ("failed", "failed"), ("partial", "partial")):
                counters[f"steady_{suffix}_segments"] += ctl["steady"][field]
            for key, field in (("packet_completed_episodes", "completed"), ("packet_failed_episodes", "failed"), ("packet_success_flags", "success_flags")):
                counters[key] += ctl["episodes"][field]
            for axis in responses:
                for key in responses[axis]:
                    responses[axis][key] += ctl["response"]["axes"][axis][key]
        if len(seed_scores) == 50:
            scores.append(statistics.mean(seed_scores))
    metric_summaries = {}
    for key, values in metrics.items():
        expected_count = len(reports) * sum(key != "metrics.drift_m.max" or case.startswith("stand") for case in cases)
        complete = len(values) == expected_count and expected_count > 0
        measured = stats(values) if complete else {"count": len(values), "mean": None, "sample_std": None, "min": None, "max": None}
        metric_summaries[key] = {**measured, "expected_count": expected_count, "censored_or_missing_count": expected_count - len(values)}
    return {"score": statistics.mean(scores) if not missing and len(scores) == len(reports) else None,
        "missing": missing, "failed": failed, "score_unavailable": unavailable,
        "metrics": metric_summaries,
        "control_diagnostics": {"counts": counters, "response_censoring": responses,
            "count_weighting": "raw observed coverage counts; not used as performance or LR weights",
            "contact_and_actuation": "complete verified per-case control and metrics artifacts retained in evidence; no hardware envelope qualification"}}


def rank_candidates(definition, manifest, audit, latency):
    prepared = definition["protocol"]
    cells, missing = {}, []
    for cell in manifest["inputs"]["cells"]:
        key = campaign.cell_key(cell)
        audited = audit["cells"][key]
        pools, receipts = {}, {}
        for seed in campaign.DEVELOPMENT_SEEDS:
            receipt = audited["development"][str(seed)]["receipt"]
            pools[seed] = load_suite(manifest, cell, audited["training"]["checkpoint"], receipt, seed)
            receipts[str(seed)] = receipt
            if pools[seed] is None:
                missing.append(f"{key}/{seed}:development")
        if any(pool is None for pool in pools.values()):
            continue
        summary = summarize_training_seed(pools, prepared["scenarios"], prepared["evaluation"])
        missing.extend(f"{key}/{item}" for item in summary["missing"])
        cells[key] = {"checkpoint": audited["training"]["checkpoint"], "development_receipts": receipts,
                      "latency": latency[key], **summary}
    candidates = {}
    for variant in prepared["variants"]:
        entries = []
        for rate_id, child in manifest["inputs"]["children"].items():
            jobs = [cells.get(f"{rate_id}/{variant}/seed_{seed}") for seed in prepared["training_seeds"]]
            failures = [f"seed_{seed}/{reason}" for seed, job in zip(prepared["training_seeds"], jobs)
                        if job for reason in job["failed"]]
            failures += [f"seed_{seed}/latency" for seed, job in zip(prepared["training_seeds"], jobs)
                         if job and not job["latency"]["passed"]]
            scores = [job["score"] for job in jobs if job and job["score"] is not None]
            measured = stats(scores) if len(scores) == 3 else None
            metrics = {}
            for metric in METRICS:
                values = [job["metrics"][metric]["mean"] for job in jobs
                          if job and number(job["metrics"][metric]["mean"])]
                measured_metric = stats(values) if len(values) == 3 else {
                    "count": len(values), "mean": None, "sample_std": None, "min": None, "max": None}
                metrics[metric] = {**measured_metric, "expected_count": 3, "censored_or_missing_count": 3 - len(values)}
            entries.append({"rate_id": rate_id, "learning_rate": prepared["learning_rates"][int(rate_id.rsplit("_", 1)[1])],
                "training_seed_scores": {str(seed): job["score"] if job else None
                                         for seed, job in zip(prepared["training_seeds"], jobs)},
                "score_unavailable": [f"seed_{seed}/{reason}" for seed, job in zip(prepared["training_seeds"], jobs)
                                      if job for reason in job["score_unavailable"]],
                "score": measured, "rank_score": measured["mean"] + measured["sample_std"] if measured else None,
                "eligible": measured is not None and not failures and all(job and not job["missing"] for job in jobs),
                "failed_gates": failures, "metrics_equal_training_seed_summary": metrics})
        candidates[variant] = entries
    return cells, candidates, missing


def assess(path):
    definition, manifest = validate(path)
    audited = campaign.audit(definition["campaign_manifest"]["path"])
    keys = {campaign.cell_key(cell) for cell in manifest["inputs"]["cells"]}
    require(audited["manifest_sha256"] == manifest["sha256"] and audited["expected_cells"] == audited["actual_cells"] == 90
            and set(audited["cells"]) == keys, "real runtime audit lacks the complete fixed ninety-cell grid")
    report = {"format": "transformer_rl.learning_rate_assessment", "schema_version": 1,
        "selection_sha256": definition["sha256"], "campaign_sha256": manifest["sha256"],
        "runtime_audit": audited, "runtime_audit_sha256": control.digest(audited),
        "status": "not_ready", "missing": [], "candidates": {}, "choices": {}, "cells": {},
        "success_semantics": SUCCESS_SEMANTICS, "formal_architecture_selection": False, "hardware_deployment_ready": False}
    if audited["status"] != "development_complete" or any(item["status"] != "completed" for item in audited["cells"].values()):
        report["missing"] = [key for key, item in audited["cells"].items() if item["status"] != "completed"]
        return report
    latency, missing, index = load_latency(definition, manifest, audited)
    report["latency_index"] = index
    if missing:
        report["missing"] = missing
        return report
    cells, candidates, missing = rank_candidates(definition, manifest, audited, latency)
    report.update(cells=cells, candidates=candidates, missing=missing)
    if missing:
        return report
    for variant, options in candidates.items():
        eligible = [item for item in options if item["eligible"]]
        best = min(eligible, key=lambda item: (item["rank_score"], item["learning_rate"])) if eligible else None
        report["choices"][variant] = {"status": "selected" if best else "no_eligible_rate",
            "rate_id": best["rate_id"] if best else None, "learning_rate": best["learning_rate"] if best else None,
            "rank_score": best["rank_score"] if best else None}
    report["status"] = "selected" if any(item["status"] == "selected" for item in report["choices"].values()) else "no_eligible_rate"
    return report


def seal_choice(path):
    definition, _ = validate(path)
    root = Path(definition["output_root"])
    with selection_lock(definition):
        require(not (root / "choice.json").exists(), "immutable choice already exists; reselection is forbidden")
        report = assess(path)
        if report["status"] != "selected":
            return report
        snapshot = seal(report)
        # Atomic exclusive publication: failures never overwrite a prior seal.
        assessment = root / f"assessment-{uuid.uuid4().hex}.json"
        campaign.write(assessment, snapshot)
        choice = seal({"format": "transformer_rl.learning_rate_choice", "schema_version": 1,
            "selection_sha256": definition["sha256"], "campaign_sha256": definition["campaign_sha256"],
            "assessment": campaign.artifact(assessment), "assessment_sha256": snapshot["sha256"],
            "protocol_sha256": control.digest(definition["protocol"]), "choices": report["choices"],
            "confirmation": definition["protocol"]["confirmation"], "formal_architecture_selection": False,
            "hardware_deployment_ready": False})
        campaign.write(root / "choice.json", choice)
        return choice


def load_choice(path, definition):
    path = campaign.plain(path)
    require(path == Path(definition["output_root"]) / "choice.json", "choice must be the original immutable seal")
    choice = sealed(read(path))
    require(choice.get("format") == "transformer_rl.learning_rate_choice" and choice.get("schema_version") == 1
            and choice["selection_sha256"] == definition["sha256"] and choice["campaign_sha256"] == definition["campaign_sha256"]
            and choice["protocol_sha256"] == control.digest(definition["protocol"])
            and choice["confirmation"] == definition["protocol"]["confirmation"]
            and choice["formal_architecture_selection"] is False and choice["hardware_deployment_ready"] is False,
            "choice identity or protocol differs")
    assessment = sealed(read(campaign.checked(choice["assessment"])))
    require(assessment["sha256"] == choice["assessment_sha256"] and assessment["status"] == "selected"
            and assessment["choices"] == choice["choices"] and not assessment["missing"], "choice assessment changed")
    current = assess(Path(definition["output_root"]) / "manifest.json")
    require({key: value for key, value in assessment.items() if key != "sha256"} == current,
            "development/runtime/latency evidence changed after LR choice")
    return choice, assessment


def confirmation_requests(manifest, choice, assessment, root):
    requests = []
    for cell in manifest["inputs"]["cells"]:
        selected = choice["choices"][cell["variant"]]
        if selected["status"] != "selected" or selected["rate_id"] != cell["rate_id"]:
            continue
        checkpoint = assessment["cells"][campaign.cell_key(cell)]["checkpoint"]
        for seed in NOISE_SEEDS:
            directory = root / "evaluations" / campaign.cell_key(cell) / f"seed_{seed}" / "attempt_0000"
            configs, _ = campaign.evaluation_inputs(manifest, cell, checkpoint)
            command = campaign.transfer.command(Path(manifest["source_root"]), "evaluate-suite", "--checkpoint", checkpoint["path"],
                "--configs", *configs.values(), "--outputs", *(directory / f"{case}.json" for case in configs),
                "--steps", 4001, "--seed", seed, "--device", manifest["device"], "--control-output", directory / "control.json",
                "--trace-output", directory / "trace.npz", "--settle-steps", 200, "--min-steady-samples", 200, "--trace-replicas", 2)
            requests.append({"cell": campaign.cell_key(cell), "training_seed": cell["training_seed"], "evaluation_seed": seed,
                "checkpoint": checkpoint, "configs": {case: campaign.artifact(path) for case, path in configs.items()},
                "directory": str(directory.relative_to(root)), "command": command, "use": "heldout_noise_stream_only"})
    return requests


def prepare_confirmation(selection_manifest, choice_path, output_root):
    definition, manifest = validate(selection_manifest)
    choice, assessment = load_choice(choice_path, definition)
    root = independent_root(output_root, [definition["output_root"], manifest["output_root"], manifest["source_root"], manifest["inputs"]["root"]])
    requests = confirmation_requests(manifest, choice, assessment, root)
    definition_out = seal({"format": CONFIRMATION_FORMAT, "schema_version": 1, "status": "prepared_not_queued", "output_root": str(root),
        "selection_manifest": campaign.artifact(selection_manifest), "choice": campaign.artifact(choice_path),
        "choice_sha256": choice["sha256"], "campaign_sha256": manifest["sha256"], "helpers": helper_identity(),
        "protocol": definition["protocol"]["confirmation"], "requests": requests,
        "execution_implemented": False, "no_reselection": True,
        "formal_architecture_selection": False, "hardware_deployment_ready": False})
    root.mkdir(parents=True, exist_ok=False)
    campaign.write(root / "manifest.json", definition_out)
    return definition_out


def audit_confirmation(path):
    """Read one original attempt per selected CP/seed; never emit a new LR."""
    path = campaign.plain(path)
    confirm = sealed(read(path))
    require(confirm.get("format") == CONFIRMATION_FORMAT and confirm.get("schema_version") == 1
            and path == Path(confirm["output_root"]) / "manifest.json" and confirm["helpers"] == helper_identity()
            and confirm["status"] == "prepared_not_queued" and confirm["no_reselection"] is True and confirm["execution_implemented"] is False
            and confirm["formal_architecture_selection"] is False and confirm["hardware_deployment_ready"] is False,
            "confirmation identity or implementation scope differs")
    selection_path = campaign.checked(confirm["selection_manifest"])
    definition, manifest = validate(selection_path)
    choice, assessment = load_choice(campaign.checked(confirm["choice"]), definition)
    require(confirm["choice_sha256"] == choice["sha256"] and confirm["campaign_sha256"] == manifest["sha256"]
            and confirm["protocol"] == definition["protocol"]["confirmation"], "confirmation cannot change the chosen LR or noise-only scope")
    expected = {(campaign.cell_key(cell), seed) for cell in manifest["inputs"]["cells"] for seed in NOISE_SEEDS
                if choice["choices"][cell["variant"]]["status"] == "selected"
                and choice["choices"][cell["variant"]]["rate_id"] == cell["rate_id"]}
    require(confirm["requests"] == confirmation_requests(manifest, choice, assessment, Path(confirm["output_root"]))
            and len(confirm["requests"]) == len(expected) and {(request["cell"], request["evaluation_seed"]) for request in confirm["requests"]} == expected,
            "confirmation must preserve all original training seeds and exactly the two heldout noise streams")
    cells = {campaign.cell_key(cell): cell for cell in manifest["inputs"]["cells"]}
    pools, evidence, missing = {}, {}, []
    root = Path(confirm["output_root"])
    for request in confirm["requests"]:
        key, seed = request["cell"], request["evaluation_seed"]
        cell = cells[key]
        directory = campaign.inside(root, request["directory"])
        require(directory == root / "evaluations" / key / f"seed_{seed}" / "attempt_0000"
                and request["training_seed"] == cell["training_seed"] and request["use"] == "heldout_noise_stream_only", "confirmation request route or role differs")
        require(len(list(directory.parent.glob("attempt_*"))) <= 1, "confirmation retries cannot select a favorable outcome")
        campaign.checked(request["checkpoint"])
        configs, _ = campaign.evaluation_inputs(manifest, cell, request["checkpoint"])
        require(request["configs"] == {case: campaign.artifact(path) for case, path in configs.items()}, "confirmation evaluation configs changed")
        item_path = directory / "receipt.json"
        if not item_path.exists() or campaign.workers_in([directory]):
            missing.append(f"{key}/{seed}")
            continue
        process = read(directory / "worker.process.json")
        require(terminal_process(process, request["command"]), "confirmation worker endpoint or frozen command differs")
        item = campaign.artifact(item_path)
        reports = load_suite(manifest, cell, request["checkpoint"], item, seed, confirmation=confirm)
        if reports is None:
            missing.append(f"{key}/{seed}")
            continue
        pools.setdefault(key, {})[seed] = reports
        evidence[f"{key}/{seed}"] = item
    summaries = {}
    for key, reports in pools.items():
        if set(reports) != set(NOISE_SEEDS):
            continue
        summary = summarize_training_seed(reports, definition["protocol"]["scenarios"], definition["protocol"]["evaluation"])
        missing.extend(f"{key}/{item}" for item in summary["missing"])
        summaries[key] = summary
    passed = not missing and len(summaries) * 2 == len(expected) and all(not summary["failed"] for summary in summaries.values())
    return {"format": "transformer_rl.learning_rate_confirmation_audit", "schema_version": 1,
        "confirmation_sha256": confirm["sha256"], "choice_sha256": choice["sha256"],
        "status": "not_ready" if missing else "confirmed" if passed else "not_confirmed",
        "missing": missing, "expected_suites": len(expected), "actual_suites": len(evidence), "cells": summaries,
        "evaluation_receipts": evidence, "original_choices": choice["choices"], "no_reselection": True,
        "scope": "heldout_noise_stream_only", "new_initial_conditions": False, "new_perturbation_domains": False,
        "formal_architecture_selection": False, "hardware_deployment_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    item = commands.add_parser("prepare")
    item.add_argument("--campaign-manifest", type=Path, required=True)
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--latency-index", type=Path, required=True)
    for name in ("assess", "seal-choice"):
        item = commands.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
    item = commands.add_parser("prepare-confirmation")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--choice", type=Path, required=True)
    item.add_argument("--output-root", type=Path, required=True)
    item = commands.add_parser("audit-confirmation")
    item.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
            result = prepare(args.campaign_manifest, args.output_root, args.latency_index)
        elif args.operation == "assess":
            result = assess(args.manifest)
        elif args.operation == "seal-choice":
            result = seal_choice(args.manifest)
        elif args.operation == "prepare-confirmation":
            result = prepare_confirmation(args.manifest, args.choice, args.output_root)
        else:
            result = audit_confirmation(args.manifest)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("status") not in {"not_ready", "no_eligible_rate", "not_confirmed"} else 3


if __name__ == "__main__":
    raise SystemExit(main())

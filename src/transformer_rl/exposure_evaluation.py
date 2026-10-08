"""Independent physical evaluation provider for the complete exposure matrix."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import sys

from . import exposure_campaign as campaign
from . import exposure_protocol as definition
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest


FORMAT = "transformer_rl.exposure_evaluation_request"
METRIC_NAMES = sorted(("height_abs_error", "vx_abs_error", "wz_abs_error", "tilt_angle", "drift_m",
                       "issued_action_rate_rms"))
SIGNAL_NAMES = sorted(["height_error", "vx_error", "wz_error",
                      *[f"leg_target_{i}" for i in range(4)],
                      *[f"wheel_target_{i}" for i in range(2)], *[f"effort_{i}" for i in range(6)]])


def _require(value, reason):
    if not value:
        raise campaign.CampaignIntegrityError(reason)


def _batch(protocol, endpoint_receipt, cells, directory, selection_receipt=None):
    """Reconstruct a whole declared role/seed batch, including poor scenarios."""
    _require(type(cells) is list and bool(cells), "nonempty declared evaluation batch required")
    first = cells[0]
    _require(type(first) is dict and all(type(first.get(k)) is int for k in ("stage_index", "seed")),
             "explicit integer evaluation stage and seed required")
    key = {k: first[k] for k in ("job_id", "stage_index", "role", "seed")}
    _require(key["role"] in ("validation", "held_out"), "unknown evaluation role")
    if key["role"] == "held_out":
        _require(selection_receipt is not None, "held-out evaluation requires an immutable validation choice")
        from .exposure_selection import authorize_heldout_batch
        authorize_heldout_batch(selection_receipt, endpoint_receipt, cells, directory)
    else:
        _require(selection_receipt is None, "validation cannot depend on a later selection")
    expected = [c for c in protocol["evaluation_cells"] if all(c[k] == v for k, v in key.items())]
    _require(digest(cells) == digest(expected) and bool(expected),
             "evaluation batch omits, reorders or replaces declared cells")
    job = next((j for j in protocol["jobs"] if j["id"] == key["job_id"]), None)
    _require(job is not None, "evaluation job does not belong to the complete grid")
    endpoint = campaign.verify_segment_endpoint(protocol, job, key["stage_index"], endpoint_receipt)
    stage = job["stages"][key["stage_index"]]
    configs = [FrameTrainConfig.from_dict(definition._read(campaign._checked(c["config_receipt"]))) for c in cells]
    saved = FrameTrainConfig.from_dict(stage["config"])
    _require(protocol["environment_factory"] == "transformer_rl.chassis_adapter:make_env"
             and saved.model.action_dim == 6, "physical provider requires the frozen six-motor chassis adapter")
    _require(all(c.model == saved.model and c.control == saved.control for c in configs),
             "evaluation changes the trained model or control contract")
    _require(len({c["scenario"] for c in cells}) == len(cells)
             and len({c["num_envs"] for c in cells}) == 1
             and all(c["num_envs"] == cfg.environment["num_envs"] for c, cfg in zip(cells, configs)),
             "all declared scenarios need their complete equal replica sets")
    # This adapter merges immutable exact-case contracts without starting Isaac.
    # Architectures and H are generic; physical packets retain the six-motor
    # chassis contract rather than inventing another robot's measurement units.
    from .chassis_adapter import merge_evaluation_contracts
    environments = [c.environment for c in configs]
    snapshot = definition._path(environments[0]["snapshot"])
    _require(all(e["snapshot"] == str(snapshot) and e["snapshot_sha256"] == environments[0]["snapshot_sha256"]
                 for e in environments), "evaluation mixes snapshots")
    merged = merge_evaluation_contracts(snapshot, environments)
    _require([c["name"] for c in merged["evaluation"]["cases"]] == [c["scenario"] for c in cells],
             "effective cases differ from the declared batch")
    environment = {"snapshot": str(snapshot), "snapshot_sha256": environments[0]["snapshot_sha256"],
                   "contracts": deepcopy(environments), "num_envs": sum(e["num_envs"] for e in environments)}
    expected_directory = Path(protocol["output_root"]) / job["id"] / f"stage_{key['stage_index']:04d}" \
        / f"evaluation_{key['role']}_{key['seed']}"
    _require(definition._path(directory) == expected_directory, "evaluation escapes its fixed role/seed output")
    return {"job": job, "endpoint": endpoint, "config": saved, "environment": environment,
            "effective_contract_sha256": digest(merged), "key": key, "configs": configs}


def _body(protocol, endpoint_receipt, cells, directory, leases, controller_receipt, selection_receipt=None,
          storage_progress=None):
    batch = _batch(protocol, endpoint_receipt, cells, directory, selection_receipt)
    controller_path = campaign._checked(controller_receipt)
    _require(controller_path == Path(protocol["output_root"]) / "controller.json",
             "evaluator controller must be the original canonical campaign owner")
    controller = campaign._read(controller_path)
    _require(controller.get("format") == "transformer_rl.exposure_controller"
             and type(controller.get("schema_version")) is int and controller["schema_version"] == 1
             and controller["protocol_sha256"] == protocol["sha256"]
             and controller["source"] == protocol["source"] == source_identity()
             and controller["runtime"] == protocol["runtime"], "evaluator source or runtime declaration differs")
    owner = controller["process"]
    _require(type(owner) is dict and set(owner) == {"pid", "start", "argv"}
             and type(owner["pid"]) is int and owner["pid"] > 0
             and type(owner["start"]) is int and owner["start"] > 0
             and type(owner["argv"]) is list and bool(owner["argv"]), "controller process record differs")
    pins = protocol["execution"]["resource_locks"]
    _require(type(leases) is list and len(leases) == len(pins) == 2, "two declared evaluation leases required")
    descriptors = set()
    for lease, pin in zip(leases, pins):
        _require(type(lease) is dict and set(lease) == {"path", "device", "inode", "descriptor",
            "controller_pid", "controller_start"} and all(type(lease[k]) is int for k in
                ("device", "inode", "descriptor", "controller_pid", "controller_start"))
            and lease["descriptor"] >= 0 and lease["controller_pid"] == owner["pid"]
            and lease["controller_start"] == owner["start"]
            and digest({k: lease[k] for k in ("path", "device", "inode")}) == digest(pin),
            "evaluation lease declarations differ from the original controller and locks")
        descriptors.add(lease["descriptor"])
    _require(len(descriptors) == 2, "evaluation lease descriptors must be distinct")
    raw = controller["protocol_raw_receipt"]
    _require(campaign._checked(raw) and raw["sha256"] == controller["expected_protocol_sha256"]
             and definition._read(raw["path"]) == protocol, "external raw protocol authorization differs")
    storage = controller["storage_contract_receipt"]
    contract = campaign._validate_storage(campaign._read(campaign._checked(storage)), protocol, raw)
    progress = deepcopy(storage_progress) if storage_progress is not None else {"completed_stages": [], "closed_cells": []}
    _require(type(progress) is dict and set(progress) == {"completed_stages", "closed_cells"}
             and type(progress["completed_stages"]) is list and type(progress["closed_cells"]) is list,
             "fixed storage progress required")
    declared_stages = {(j["id"], s["index"]) for j in protocol["jobs"] for s in j["stages"]}
    declared_cells = {c["id"] for c in protocol["evaluation_cells"]}
    _require(all(type(s) is list and len(s) == 2 and type(s[0]) is str and type(s[1]) is int
                 and tuple(s) in declared_stages for s in progress["completed_stages"])
             and len(set(map(tuple, progress["completed_stages"]))) == len(progress["completed_stages"])
             and all(type(c) is str and c in declared_cells for c in progress["closed_cells"])
             and len(set(progress["closed_cells"])) == len(progress["closed_cells"]),
             "storage progress changes the fixed grid or repeats budget items")
    remaining = campaign.remaining_storage_bytes(protocol, contract,
        map(tuple, progress["completed_stages"]), progress["closed_cells"])
    return {"format": FORMAT, "schema_version": 1, "protocol": deepcopy(raw),
            "endpoint": deepcopy(endpoint_receipt), "cells": deepcopy(cells),
            "directory": str(definition._path(directory)), "leases": deepcopy(leases),
            "controller": deepcopy(controller_receipt), "source": deepcopy(protocol["source"]),
            "environment": batch["environment"], "effective_contract_sha256": batch["effective_contract_sha256"],
            "evaluation": deepcopy(protocol["evaluation"]), "key": batch["key"],
            "storage_contract": deepcopy(storage), "selection": deepcopy(selection_receipt),
            "storage_progress": progress, "required_remaining_bytes": remaining}


def plan_request(protocol, endpoint_receipt, cells, directory, leases, controller_receipt, *,
                 selection_receipt=None, storage_progress=None):
    body = _body(protocol, endpoint_receipt, cells, directory, leases, controller_receipt,
                 selection_receipt, storage_progress)
    directory = definition._path(directory)
    _require(not os.path.lexists(directory), "evaluation directory cannot be reused")
    directory.mkdir(mode=0o700)
    path = directory / "request.json"
    campaign._new(path, body)
    receipt = definition._receipt(path)
    command = [sys.executable, "-B", "-m", "transformer_rl.exposure_evaluation", "worker",
               "--request", str(path), "--expected-request-sha256", receipt["sha256"]]
    return {"command": command, "request": receipt}


def _request(receipt, *, child=False):
    path = campaign._checked(receipt)
    value = campaign._read(path)
    _require(value.get("format") == FORMAT and type(value.get("schema_version")) is int
             and value["schema_version"] == 1 and path == Path(value["directory"]) / "request.json",
             "evaluation request identity differs")
    protocol, raw = campaign.read_authorized_protocol(value["protocol"]["path"], value["protocol"]["sha256"])
    _require(raw == value["protocol"], "evaluation protocol receipt differs")
    expected = _body(protocol, value["endpoint"], value["cells"], value["directory"],
                     value["leases"], value["controller"], value.get("selection"), value.get("storage_progress"))
    _require(digest(value) == digest(expected), "evaluation request differs from actual complete inputs")
    if child:
        campaign.validate_controller_lease(protocol, value["leases"], value["controller"])
    return value, protocol, _batch(protocol, value["endpoint"], value["cells"], value["directory"], value.get("selection"))


def _same_metrics(actual, expected):
    """Float tolerance permits CPU/GPU reduction rounding, not missing fields."""
    if isinstance(expected, dict):
        return (type(actual) is dict and set(actual) == set(expected)
                and all(_same_metrics(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return type(actual) is list and len(actual) == len(expected) and all(
            _same_metrics(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-10)
    return type(actual) is type(expected) and actual == expected


def verify_result(protocol, endpoint_receipt, cells, directory, request_receipt):
    """Reconstruct every reported statistic from all declared physical samples."""
    request, actual_protocol, batch = _request(request_receipt)
    _require(actual_protocol == protocol and request["endpoint"] == endpoint_receipt
             and request["cells"] == cells and request["directory"] == str(definition._path(directory)),
             "result belongs to a different authorized evaluation")
    directory = definition._path(directory)
    from . import runtime_paths
    artifact_names = ("report.json", "trace.npz", "evaluation.completion.json",
                      "worker.completion.json", "worker.process.json", runtime_paths.PROFILE_FILENAME)
    artifacts = {name: definition._receipt(directory / name) for name in artifact_names}
    control = campaign._read(directory / "report.json")
    completed = campaign._read(directory / "evaluation.completion.json")
    worker = campaign._read(directory / "worker.completion.json")
    process = campaign._read(directory / "worker.process.json")
    runtime_profile = definition._receipt(directory / runtime_paths.PROFILE_FILENAME)
    runtime_paths.validate_runtime_artifact(directory, runtime_profile)
    _require(worker.get("runtime_profile") == process.get("runtime_profile") == runtime_profile,
             "physical worker runtime profile differs from its parent publication")
    command = [sys.executable, "-B", "-m", "transformer_rl.exposure_evaluation", "worker",
               "--request", request_receipt["path"], "--expected-request-sha256", request_receipt["sha256"]]
    handle = worker.get("process", {})
    _require(type(handle) is dict and set(handle) == {"pid", "start", "argv"}
             and type(handle["pid"]) is int and handle["pid"] > 0
             and type(handle["start"]) is int and handle["start"] > 0 and handle["argv"] == command,
             "evaluation worker kernel-handle record or actual argv differs")
    live = campaign.predecessors._process(handle["pid"])
    _require(worker["format"] == "transformer_rl.exposure_worker_completion"
             and type(worker.get("schema_version")) is int and worker["schema_version"] == 1
             and type(process.get("schema_version")) is int and process["schema_version"] == 1
             and process["format"] == "transformer_rl.exposure_worker_process"
             and type(worker["returncode"]) is int and worker["returncode"] == 0
             and worker["timed_out"] is False
             and worker["command"] == process["command"] == command
             and worker["process"] == process["process"]
             and process["leases"] == request["leases"]
             and (live is None or live["start"] != handle["start"]),
             "evaluation worker has no exact normal terminal evidence")
    _require(digest(completed) == digest({"format": "transformer_rl.exposure_evaluation_completion", "schema_version": 1,
        "request": request_receipt, "report": definition._receipt(directory / "report.json"),
        "source": protocol["source"], "status": "completed", "runtime_profile": runtime_profile,
        "hardware_verified": False}),
        "physical completion differs from actual publication")
    checkpoint = batch["endpoint"]["checkpoint"]
    config, evaluation = batch["config"], protocol["evaluation"]
    labels = control.get("environment_provenance", {}).get("evaluation_groups")
    _require(type(labels) is list and len(labels) == batch["environment"]["num_envs"]
             and Counter(labels) == Counter({c["scenario"]: c["num_envs"] for c in cells}),
             "evaluation has missing or undeclared physical rows")
    provenance = control["environment_provenance"]
    _require(provenance["identity"] == batch["environment"]["snapshot_sha256"]
             and provenance["control_sha256"] == digest(config.control)
             and provenance["contract_sha256"] == batch["effective_contract_sha256"],
             "physical snapshot, effective contract or timing provenance differs")
    identity = {"format": "transformer_rl.packed_evaluation", "schema_version": 1,
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": batch["endpoint"]["cumulative_successful_updates"],
        "model": config.to_dict()["model"], "control_sha256": digest(config.control),
        "environment": batch["environment"], "seed": request["key"]["seed"],
        "steps": evaluation["steps"], "num_envs": len(labels), "transitions": evaluation["steps"] * len(labels),
        "policy": "deterministic_raw_mean_then_declared_action_limits"}
    _require(all(_same_metrics(control.get(k), v) for k, v in identity.items())
             and set(control.get("groups", {})) == set(labels),
             "evaluation policy/checkpoint/clock or scenario identity differs")
    _require(sorted(control["metrics"]) == METRIC_NAMES and sorted(control["stability"]["signals"]) == SIGNAL_NAMES,
             "physical report omits or replaces the provider's declared measurement schema")
    trace_expected = {"checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": identity["checkpoint_update"],
        "seed": identity["seed"], "steps": evaluation["steps"], "policy_dt_s": config.control["policy_dt_s"],
        "sampling_hz": 1 / config.control["policy_dt_s"], "control_sha256": digest(config.control),
        "row_indices": list(range(len(labels))), "group_labels": labels,
        "history_length": config.model.history_length,
        "pre_inference_age_semantics": "policy steps since reset, captured before actor inference",
        "evaluation_metric_names": METRIC_NAMES, "evaluation_signal_names": SIGNAL_NAMES}
    from .exposure_trace import verify_trace_archive
    replay = verify_trace_archive(directory / "trace.npz", control["trace"], trace_expected,
        steps=evaluation["steps"], rows=len(labels), history_length=config.model.history_length,
        action_bounds=config.control["action_bounds"], settle_steps=evaluation["settle_steps"],
        min_steady_samples=evaluation["min_steady_samples"])
    from .frame_study import grade_report
    records = {}
    fields = ("control", "history_control", "metrics", "reward_mean", "stability", "completed_episodes",
              "failed_episodes", "success_metric_available", "success_rate")
    _require(all(k in replay and _same_metrics(control.get(k), replay[k]) for k in fields),
             "reported aggregate metrics disagree with full trace replay")
    for cell in cells:
        name = cell["scenario"]
        grouped = control["groups"][name]
        group_identity = {**identity, "num_envs": cell["num_envs"],
                          "transitions": cell["expected_policy_samples"], "environment_provenance": provenance}
        _require(all(_same_metrics(grouped.get(k), v) for k, v in group_identity.items()),
                 "scenario report checkpoint, seed, clock or row identity differs")
        _require(all(k in replay["groups"][name] and _same_metrics(grouped.get(k), replay["groups"][name][k])
                     for k in fields), "reported scenario metrics disagree with full trace replay")
        scenario = next(s for s in protocol["scenarios"] if s["name"] == name)
        grade = grade_report(grouped, scenario, evaluation, protocol["selection"]["objectives"])
        records[cell["id"]] = {"status": "completed", "identity": deepcopy(cell),
            "checkpoint": deepcopy(checkpoint), "endpoint": deepcopy(endpoint_receipt),
            "metrics": {k: deepcopy(grouped[k]) for k in fields}, "grade": grade,
            "hardware_verified": False}
    _request(request_receipt)
    _require(all(definition._receipt(directory / name) == receipt for name, receipt in artifacts.items()),
             "physical output bytes changed during verification")
    return {"format": "transformer_rl.exposure_physical_evaluation", "schema_version": 1,
        "status": "completed", "request": request_receipt, "cells": records,
        "trace_validation": replay["trace_validation"], "source": protocol["source"],
        "artifacts": artifacts,
        "formal_architecture_selection": False, "hardware_verified": False}


def run_worker(request_receipt):
    request, protocol, batch = _request(request_receipt, child=True)
    directory = Path(request["directory"])
    from . import runtime_paths
    runtime_profile = runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))
    cache = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
    _require(cache == directory / "empty_python_cache" and cache.is_dir() and not list(cache.iterdir()),
             "evaluation worker did not start with a new empty bytecode cache")
    _require(not any((directory / name).exists() for name in
        ("report.json", "trace.npz", "evaluation.completion.json")), "evaluation output cannot be overwritten")
    from .cli import _factory
    from .frame_workflow import evaluate_frame_policy
    evaluation = protocol["evaluation"]
    replicas = request["cells"][0]["num_envs"]
    contract = campaign._read(campaign._checked(request["storage_contract"]))
    stop_requested = False

    def request_stop(number, frame):
        nonlocal stop_requested
        stop_requested = True

    def guard():
        if stop_requested:
            return True
        campaign._checked(request_receipt)
        _require(runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory)) == runtime_profile,
                 "physical worker runtime profile changed")
        campaign._checked(request["storage_contract"])
        campaign.validate_controller_lease(protocol, request["leases"], request["controller"])
        _require(source_identity() == protocol["source"], "physical worker source changed")
        campaign.worker_storage_guard(protocol, contract, directory, request["required_remaining_bytes"], active_worker=True,
            trace_prefix=directory / "trace.npz", trace_sample_limit=sum(c["expected_policy_samples"] for c in request["cells"]))
        return False

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        report = evaluate_frame_policy(batch["endpoint"]["checkpoint"]["path"], _factory(protocol["environment_factory"]),
            batch["environment"], steps=evaluation["steps"], seed=request["key"]["seed"],
            device=protocol["execution"]["device"], settle_steps=evaluation["settle_steps"],
            min_steady_samples=evaluation["min_steady_samples"], control_metrics=True, history_control=True,
            trace_output=directory / "trace.npz", trace_replicas=replicas, should_stop=guard)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    _request(request_receipt, child=True)
    _require(not guard(), "physical worker stopped before result publication")
    campaign._new(directory / "report.json", report)
    campaign._new(directory / "evaluation.completion.json", {
        "format": "transformer_rl.exposure_evaluation_completion", "schema_version": 1,
        "request": request_receipt, "report": definition._receipt(directory / "report.json"),
        "source": protocol["source"], "status": "completed", "runtime_profile": runtime_profile,
        "hardware_verified": False})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["worker"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    args = parser.parse_args(argv)
    receipt = definition._receipt(args.request)
    _require(receipt["sha256"] == args.expected_request_sha256, "worker request raw authorization differs")
    from . import frame_process
    frame_process._active = True
    code = 1
    try:
        code = run_worker(receipt)
        return code
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for app in reversed(frame_process._apps):
            app.close(wait_for_replicator=False, exit_code=code)


if __name__ == "__main__":
    raise SystemExit(main())

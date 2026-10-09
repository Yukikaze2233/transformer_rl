"""Independent, byte-authorized storage calibration before full grid execution.

Small calibration jobs retain the real model, control, rollout and complete
scenario batch. Their private evaluations never enter architecture selection.
Measurements are facts, not permission to spend the production disk budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import stat
import sys
import time

from . import exposure_campaign as campaign
from . import exposure_protocol as definition
from . import runtime_paths
from .calibration_storage import CalibrationObserver, guard_owned_storage, helper_inventory
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest


FORMAT = "transformer_rl.calibration_plan"
REQUEST_FORMAT = "transformer_rl.calibration_worker_request"
LIMIT_FIELDS = {"max_owned_bytes", "free_margin_bytes", "interval_s", "timeout_s"}


def _require(condition, reason):
    campaign._require(condition, reason)


def _signed(value):
    return {**value, "sha256": digest(value)}


def _check_signed(value, format_name):
    _require(type(value) is dict and value.get("format") == format_name
             and type(value.get("schema_version")) is int and value["schema_version"] == 1
             and value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
             "calibration identity or self-signature differs")


def _limits(value):
    _require(type(value) is dict and set(value) == LIMIT_FIELDS, "explicit calibration limits required")
    _require(all(type(value[k]) is int and value[k] > 0
                 for k in ("max_owned_bytes", "free_margin_bytes")), "positive byte limits required")
    _require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] > 0
                 for k in ("interval_s", "timeout_s")), "positive finite sampling/deadline limits required")
    _require(value["interval_s"] <= 1 and value["timeout_s"] > 2,
             "sampling interval must be <= one second and deadline > two seconds")
    return deepcopy(value)


def _cells(protocol, job):
    seed = protocol["evaluation"]["validation_seeds"][0]
    # Continuous checkpoint schedules provide several complete batches. Use
    # the earliest declared first-stage batch only as a scenario template;
    # private training updates and evaluation seeds remain independent.
    checkpoint = None
    if protocol["schema_version"] == 2:
        checkpoint = min(record["checkpoint_update"] for record in
                         campaign.learning_checkpoint_records(protocol, job, job["stages"][0]))
    cells = [c for c in protocol["evaluation_cells"] if c["job_id"] == job["id"]
             and c["stage_index"] == 0 and c["role"] == "validation" and c["seed"] == seed
             and (checkpoint is None or c["checkpoint_update"] == checkpoint)]
    _require([c["scenario"] for c in cells] == [s["name"] for s in protocol["scenarios"]],
             "calibration must retain every original scenario in order")
    return deepcopy(cells)


def _merge_environment(protocol, job, cells):
    """Read the real exact-case contracts without starting Isaac."""
    from .chassis_adapter import merge_evaluation_contracts
    configs = [FrameTrainConfig.from_dict(definition._read(campaign._checked(c["config_receipt"])))
               for c in cells]
    saved = FrameTrainConfig.from_dict(job["stages"][0]["config"])
    _require(all(c.model == saved.model and c.control == saved.control for c in configs),
             "calibration evaluation changes trained model or control")
    environments = [c.environment for c in configs]
    snapshot = definition._path(environments[0]["snapshot"])
    _require(all(e["snapshot"] == str(snapshot) and e["snapshot_sha256"] == environments[0]["snapshot_sha256"]
                 and e["num_envs"] == cell["num_envs"] for e, cell in zip(environments, cells)),
             "calibration mixes snapshots or changes replicas")
    merged = merge_evaluation_contracts(snapshot, environments)
    _require([c["name"] for c in merged["evaluation"]["cases"]] == [c["scenario"] for c in cells],
             "effective calibration scenarios differ")
    return {"snapshot": str(snapshot), "snapshot_sha256": environments[0]["snapshot_sha256"],
            "contracts": deepcopy(environments), "num_envs": sum(c["num_envs"] for c in cells)}


def build_plan(protocol_path, *, expected_protocol_sha256, output_root, candidates=None,
               training_seed, updates=2, evaluation_seed=92001, max_owned_bytes,
               free_margin_bytes, interval_s=.1, timeout_s=3600.):
    """Freeze an independent prior budget; do not create outputs or start jobs."""
    protocol, raw = campaign.read_authorized_protocol(protocol_path, expected_protocol_sha256)
    return _build_plan(protocol, raw, output_root=output_root, candidates=candidates,
        training_seed=training_seed, updates=updates, evaluation_seed=evaluation_seed,
        max_owned_bytes=max_owned_bytes, free_margin_bytes=free_margin_bytes,
        interval_s=interval_s, timeout_s=timeout_s)


def _build_plan(protocol, raw, *, output_root, candidates, training_seed, updates, evaluation_seed,
                max_owned_bytes, free_margin_bytes, interval_s, timeout_s, allow_existing=False):
    root = definition._path(output_root)
    _require(root.parent.is_dir() and (allow_existing or not os.path.lexists(root)), "new calibration output required")
    protected = [*protocol["protected_roots"], protocol["output_root"], raw["path"]]
    _require(all(not root.is_relative_to(definition._path(p))
                 and not definition._path(p).is_relative_to(root) for p in protected),
             "calibration output overlaps protected production inputs or output")
    _require(type(training_seed) is int and 0 <= training_seed < 2**32,
             "calibration training seed must be an original uint32 seed")
    available = list(dict.fromkeys(j["candidate"] for j in protocol["jobs"]))
    candidates = available if candidates is None else list(candidates)
    _require(bool(candidates) and len(candidates) == len(set(candidates))
             and all(type(c) is str and c in available for c in candidates),
             "distinct original calibration candidates required")
    selected = [j for j in protocol["jobs"] if j["candidate"] in candidates and j["training_seed"] == training_seed]
    _require(len(selected) == len(candidates), "calibration seed is missing an original candidate")
    _require(type(updates) is int and updates > 0
             and all(updates <= j["stages"][0]["updates"] for j in selected),
             "calibration updates must fit each first original stage")
    used_seeds = {protocol["retention_seed"], *(j["training_seed"] for j in protocol["jobs"]),
                  *protocol["evaluation"]["validation_seeds"], *protocol["evaluation"]["seeds"]}
    packed = definition._read(Path(protocol["history_root"]) / "study" / "plan.json")
    used_seeds.update(packed["spec"]["training"]["anchor_seeds"])
    _require(type(evaluation_seed) is int and 0 <= evaluation_seed < 2**32
             and evaluation_seed not in used_seeds, "private calibration evaluation seed required")
    limits = _limits({"max_owned_bytes": max_owned_bytes, "free_margin_bytes": free_margin_bytes,
                      "interval_s": interval_s, "timeout_s": timeout_s})
    batches = [_cells(protocol, job) for job in selected]
    for job, cells in zip(selected, batches):
        _merge_environment(protocol, job, cells)
    value = {"format": FORMAT, "schema_version": 1, "protocol": raw,
        "source": deepcopy(protocol["source"]), "output_root": str(root),
        "candidates": [j["candidate"] for j in selected], "job_ids": [j["id"] for j in selected],
        "training_seed": training_seed, "updates": updates, "evaluation_seed": evaluation_seed,
        "limits": limits, "coverage": {"selected_candidates": len(selected), "original_candidates": len(available),
            "all_original_candidates": len(selected) == len(available), "original_training_seeds_per_candidate": 1,
            "training_updates": len(selected) * updates,
            "fresh_transition_budget": sum(updates * protocol["execution"]["rollout_steps"]
                                            * j["stages"][0]["config"]["environment"]["num_envs"] for j in selected),
            "evaluation_policy_samples": sum(c["expected_policy_samples"] for cells in batches for c in cells),
            "worker_namespaces": 2 * len(selected)},
        "scope": "independent small whole-job reservations and complete private scenario batches; sampled storage facts",
        "retry_policy": "stop_on_first_failure_no_retry_no_refund_no_directory_reuse",
        "formal_architecture_selection": False, "production_storage_authorized": False, "hardware_verified": False}
    _require(definition._receipt(raw["path"]) == raw and source_identity() == protocol["source"],
             "calibration inputs changed during planning")
    return _signed(value)


def validate_plan(receipt):
    path = campaign._checked(receipt)
    plan = campaign._read(path)
    _check_signed(plan, FORMAT)
    _require(not path.is_relative_to(Path(plan["output_root"])), "plan must be an external immutable input")
    # Reconstruction accepts an existing owned output only for validation;
    # creation and launch still require a genuinely unused directory.
    protocol, raw = campaign.read_authorized_protocol(plan["protocol"]["path"], plan["protocol"]["sha256"])
    _require(raw == plan["protocol"] and plan["source"] == protocol["source"] == source_identity(),
             "calibration source or raw protocol authorization changed")
    expected = _build_plan(protocol, raw, output_root=plan["output_root"], candidates=plan["candidates"],
        training_seed=plan["training_seed"], updates=plan["updates"], evaluation_seed=plan["evaluation_seed"],
        allow_existing=True, **plan["limits"])
    _require(plan == expected, "calibration jobs, limits, seed, coverage or recipe changed")
    campaign._checked(receipt)
    return plan, protocol


def validate_controller_lease(protocol, leases, controller_receipt):
    """Require a direct live controller and both actual inherited original flocks."""
    controller_path = campaign._checked(controller_receipt)
    controller = campaign._read(controller_path)
    root = definition._path(controller["root"])
    _require(controller_path == root / "controller.json"
             and controller.get("format") == "transformer_rl.calibration_controller"
             and controller.get("schema_version") == 1 and controller["source"] == protocol["source"]
             and controller["runtime"] == protocol["runtime"]
             and controller["process"] == campaign._identity(os.getppid()),
             "calibration direct parent, source or runtime differs")
    _require(definition._read(campaign._checked(controller["protocol"])) == protocol,
             "controller original protocol bytes differ")
    campaign._checked(controller["plan"])
    guard_owned_storage(root, controller["limits"]["max_owned_bytes"], controller["limits"]["free_margin_bytes"],
                        expected_root_identity=controller["root_identity"])
    pins = protocol["execution"]["resource_locks"]
    parent = controller["process"]
    _require(type(leases) is list and len(leases) == len(pins) == 2
             and len({x.get("descriptor") for x in leases}) == 2, "two distinct calibration leases required")
    for item, pin in zip(leases, pins):
        _require(type(item) is dict and set(item) == {"path", "device", "inode", "descriptor", "controller_pid", "controller_start"}
                 and all(type(item[k]) is int for k in ("device", "inode", "descriptor", "controller_pid", "controller_start"))
                 and item["descriptor"] >= 0 and item["controller_pid"] == parent["pid"]
                 and item["controller_start"] == parent["start"]
                 and {k: item[k] for k in ("path", "device", "inode")} == pin
                 and campaign.predecessors._lock_signature(pin["path"]) == pin,
                 "calibration inherited lock identity differs")
        own = os.fstat(item["descriptor"])
        parent_fd = Path(f"/proc/{parent['pid']}/fd/{item['descriptor']}").stat()
        _require((own.st_dev, own.st_ino) == (parent_fd.st_dev, parent_fd.st_ino) == (pin["device"], pin["inode"]),
                 "calibration inherited descriptor was replaced")
        for pid in ("self", str(parent["pid"])):
            info = Path(f"/proc/{pid}/fdinfo/{item['descriptor']}").read_text()
            _require(any("FLOCK" in line and "ADVISORY" in line and "WRITE" in line
                         for line in info.splitlines() if line.startswith("lock:")), "calibration exclusive flock is absent")
        check = os.open(pin["path"], os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(check, fcntl.LOCK_UN)
                raise campaign.CampaignIntegrityError("calibration lock is not exclusively held")
        finally:
            os.close(check)
    campaign._checked(controller_receipt)


def _batch(plan, protocol, job, request, controller):
    config = FrameTrainConfig.from_dict(job["stages"][0]["config"])
    cells = _cells(protocol, job)
    return {"config": config, "stage": {"name": job["stages"][0]["name"], "config": config.to_dict(), "updates": plan["updates"]},
        "evaluation_environment": _merge_environment(protocol, job, cells), "cells": cells,
        "evaluation": deepcopy(protocol["evaluation"]), "evaluation_seed": plan["evaluation_seed"],
        "trace_replicas": max(c["num_envs"] for c in cells),
        "expected_policy_samples": sum(c["expected_policy_samples"] for c in cells),
        "checkpoint_receipt": request["checkpoint"], "storage_root": plan["output_root"],
        "storage_root_identity": controller["root_identity"]}


def validate_request(receipt, *, child=False):
    request = campaign._read(campaign._checked(receipt))
    _check_signed(request, REQUEST_FORMAT)
    fields = {"format", "schema_version", "plan", "protocol", "source", "job_id", "kind", "directory", "updates",
              "controller", "leases", "checkpoint", "limits", "sha256"}
    _require(set(request) == fields, "calibration request fields differ")
    plan, protocol = validate_plan(request["plan"])
    controller = campaign._read(campaign._checked(request["controller"]))
    _require(controller["root"] == plan["output_root"] and controller["plan"] == request["plan"]
             and controller["protocol"] == request["protocol"] == plan["protocol"]
             and request["source"] == plan["source"] and request["limits"] == plan["limits"]
             and request["updates"] == plan["updates"], "calibration controller/request authorization differs")
    job = next((j for j in protocol["jobs"] if j["id"] == request["job_id"] and j["id"] in plan["job_ids"]), None)
    _require(job is not None and request["kind"] in ("train", "evaluate"), "calibration job or kind differs")
    directory = definition._path(request["directory"])
    _require(directory == Path(plan["output_root"]) / job["id"] / request["kind"]
             and Path(receipt["path"]) == directory / "request.json", "calibration request directory differs")
    if request["kind"] == "train":
        _require(request["checkpoint"] is None, "calibration training cannot resume a checkpoint")
    else:
        expected = _training_checkpoint(Path(plan["output_root"]) / job["id"] / "train", request["plan"], protocol, job, plan)
        _require(request["checkpoint"] == expected, "calibration evaluation checkpoint differs from its closed train worker")
    if child:
        validate_controller_lease(protocol, request["leases"], request["controller"])
    campaign._checked(receipt)
    return request, protocol, job, _batch(plan, protocol, job, request, controller)


def _training_checkpoint(directory, plan_receipt, protocol, job, plan):
    outcome = campaign._read(directory / "outcome.json")
    process = campaign._read(directory / "worker.process.json")
    closed = campaign._read(directory / "worker.completion.json")
    _require(outcome.get("format") == "transformer_rl.calibration_worker_outcome" and outcome["status"] == "completed"
             and not outcome["shutdown_errors"] and outcome["error"] is None and closed["returncode"] == 0
             and closed["timed_out"] is False and outcome["process"] == process["process"] == closed["process"]
             and outcome["runtime_profile"] == process["runtime_profile"] == closed["runtime_profile"],
             "calibration training worker did not close normally")
    request = campaign._read(campaign._checked(outcome["request"]))
    _require(request["plan"] == plan_receipt and request["job_id"] == job["id"] and request["kind"] == "train"
             and request["source"] == protocol["source"], "calibration training producer differs")
    runtime_paths.validate_runtime_artifact(directory, outcome["runtime_profile"])
    completion = campaign._read(campaign._checked(outcome["training_completion"]))
    expected_samples = plan["updates"] * protocol["execution"]["rollout_steps"] * job["stages"][0]["config"]["environment"]["num_envs"]
    _require(completion["status"] == "completed" and completion["job_id"] == "calibration_" + job["id"]
             and completion["source"] == protocol["source"] and not completion["shutdown_errors"]
             and completion["successful_updates"] == completion["attempted_updates"] == plan["updates"]
             and completion["actual_collected_transitions"] == completion["successful_full_rollout_samples"] == expected_samples
             and len(completion["endpoints"]) == 1, "calibration complete rollout or update proof differs")
    checkpoint = completion["endpoints"][0]["checkpoint"]
    _require(checkpoint == outcome["checkpoint"] and campaign._checked(checkpoint)
             and Path(checkpoint["path"]).is_relative_to(directory / "train"), "calibration checkpoint publication differs")
    from .frame_checkpoint import load_frame_checkpoint
    _, _, config, update, metadata, _ = load_frame_checkpoint(checkpoint["path"])
    _require(config.to_dict() == job["stages"][0]["config"] and update == plan["updates"]
             and metadata["source"] == protocol["source"] and metadata["seed"] == job["training_seed"]
             and metadata["collected_transitions"] == expected_samples,
             "calibration actual checkpoint recipe or counters differ")
    # Reuse the complete learner proof: initialization, private seeds, Adam
    # options/steps, RNG/clocks, actual metric rows and the whole-job charge.
    from . import exposure_training as training
    expected, configs = training._definition([
        {"name": job["stages"][0]["name"], "config": job["stages"][0]["config"], "updates": plan["updates"]}],
        job_id="calibration_" + job["id"], rollout_steps=protocol["execution"]["rollout_steps"],
        training_seed=job["training_seed"], retention_seed=job["retention_seed"],
        evaluation_seeds=[plan["evaluation_seed"]], device=protocol["execution"]["device"],
        expected_initial_model_sha256=job["initial_model_sha256"],
        max_seconds=min(protocol["execution"]["max_seconds"], plan["limits"]["timeout_s"] - 2.),
        environment_reference=protocol["environment_factory"])
    actual, _ = training._read_input(directory / "train" / "request.json")
    _require(actual == expected, "calibration actual learner request differs from its frozen recipe")
    proof = training._segment_parent(definition._receipt(Path(checkpoint["path"]).with_name("endpoint.json")),
                                     expected, configs[0])
    _require(proof["completion"] == outcome["training_completion"]
             and proof["endpoint"]["checkpoint"] == checkpoint,
             "calibration complete learning proof differs from its worker outcome")
    return checkpoint


def _request(plan_receipt, plan, protocol, job, kind, root, controller_receipt, leases, checkpoint):
    directory = root / job["id"] / kind
    _require(not os.path.lexists(directory), "calibration worker directory cannot be reused")
    directory.mkdir(mode=0o700)
    request = _signed({"format": REQUEST_FORMAT, "schema_version": 1, "plan": plan_receipt,
        "protocol": plan["protocol"], "source": protocol["source"], "job_id": job["id"], "kind": kind,
        "directory": str(directory), "updates": plan["updates"], "controller": controller_receipt,
        "leases": leases, "checkpoint": checkpoint, "limits": plan["limits"]})
    path = directory / "request.json"
    campaign._new(path, request)
    receipt = definition._receipt(path)
    return directory, receipt, [sys.executable, "-B", "-m", "transformer_rl.calibration_worker", "worker",
                               "--request", str(path), "--expected-request-sha256", receipt["sha256"]]


def _closed_outcome(directory, receipt, result, protocol):
    _require(type(result["returncode"]) is int and result["returncode"] == 0
             and result["timed_out"] is False, "calibration worker failed or timed out")
    outcome = campaign._read(directory / "outcome.json")
    _require(outcome.get("format") == "transformer_rl.calibration_worker_outcome" and outcome["request"] == receipt
             and outcome["status"] == "completed" and outcome["error"] is None and not outcome["shutdown_errors"]
             and outcome["source"] == protocol["source"] and outcome["process"] == result["process"]
             and outcome["runtime_profile"] == result["runtime_profile"]
             and outcome["formal_architecture_selection"] is False
             and outcome["production_storage_authorized"] is False and outcome["hardware_verified"] is False,
             "calibration closed outcome differs")
    runtime_paths.validate_runtime_artifact(directory, outcome["runtime_profile"])
    return outcome


def _verify_private_evaluation(plan, protocol, job, checkpoint, outcome):
    """Replay the complete trace and bind every private calibration row."""
    from .chassis_adapter import merge_evaluation_contracts
    from .episode_outcomes import TRACE_METADATA_KEY, trace_metadata
    from .exposure_evaluation import METRIC_NAMES, SIGNAL_NAMES, _episode_outcome_rows, _same_metrics
    from .exposure_trace import verify_trace_archive
    report = campaign._read(campaign._checked(outcome["report"]))
    trace_path = campaign._checked(outcome["trace"])
    cells = _cells(protocol, job)
    environment = _merge_environment(protocol, job, cells)
    config = FrameTrainConfig.from_dict(job["stages"][0]["config"])
    provenance = report.get("environment_provenance")
    labels = provenance.get("evaluation_groups") if type(provenance) is dict else None
    _require(type(labels) is list and all(type(label) is str for label in labels)
             and len(labels) == environment["num_envs"]
             and Counter(labels) == Counter({c["scenario"]: c["num_envs"] for c in cells}),
             "private calibration report changes declared physical rows")
    merged = merge_evaluation_contracts(definition._path(environment["snapshot"]), environment["contracts"])
    _require(provenance.get("identity") == environment["snapshot_sha256"]
             and provenance.get("control_sha256") == digest(config.control)
             and provenance.get("contract_sha256") == digest(merged),
             "private calibration physical snapshot, effective contract or timing provenance differs")
    _require(merged.get("policy_dt") == config.control["policy_dt_s"],
             "private calibration case horizon clock differs from the trained policy interval")
    outcome_cases = {}
    for case in merged["evaluation"]["cases"]:
        seconds = case.get("episode_seconds", merged.get("episode_seconds"))
        _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
                 and type(case.get("task")) is str and case["task"],
                 "private calibration requires explicit positive case horizons and tasks")
        ticks = round(seconds / config.control["policy_dt_s"])
        _require(0 < ticks < 2**63, "private calibration case horizon exceeds int64 or one policy tick")
        outcome_cases[case["name"]] = {"episode_horizon_ticks": ticks,
                                       "survival_applicable": case["task"] == "survive"}
    expected = {"format": "transformer_rl.packed_evaluation", "schema_version": 1,
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": plan["updates"],
        "model": config.to_dict()["model"], "control_sha256": digest(config.control),
        "environment": environment, "seed": plan["evaluation_seed"], "steps": protocol["evaluation"]["steps"],
        "num_envs": len(labels), "transitions": protocol["evaluation"]["steps"] * len(labels),
        "policy": "deterministic_raw_mean_then_declared_action_limits"}
    _require(all(_same_metrics(report.get(k), v) for k, v in expected.items())
             and set(report.get("groups", {})) == set(labels)
             and sorted(report["metrics"]) == METRIC_NAMES
             and sorted(report["stability"]["signals"]) == SIGNAL_NAMES,
             "private calibration report changes checkpoint, seed, control, scenarios or complete rows")
    trace_expected = {"checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": plan["updates"],
        "seed": plan["evaluation_seed"], "steps": protocol["evaluation"]["steps"],
        "policy_dt_s": config.control["policy_dt_s"], "sampling_hz": 1 / config.control["policy_dt_s"],
        "control_sha256": digest(config.control), "row_indices": list(range(len(labels))), "group_labels": labels,
        "history_length": config.model.history_length,
        "pre_inference_age_semantics": "policy steps since reset, captured before actor inference",
        "evaluation_metric_names": METRIC_NAMES, "evaluation_signal_names": SIGNAL_NAMES}
    trace_expected[TRACE_METADATA_KEY] = trace_metadata(True)
    trace_expected["episode_outcome_contract"] = _episode_outcome_rows(outcome_cases, labels)
    replay = verify_trace_archive(trace_path, report["trace"], trace_expected,
        steps=protocol["evaluation"]["steps"], rows=len(labels), history_length=config.model.history_length,
        action_bounds=config.control["action_bounds"], settle_steps=protocol["evaluation"]["settle_steps"],
        min_steady_samples=protocol["evaluation"]["min_steady_samples"])
    fields = ("control", "history_control", "metrics", "reward_mean", "stability", "completed_episodes",
              "failed_episodes", "success_metric_available", "success_rate", "episode_outcomes")
    _require(all(k in replay and _same_metrics(report.get(k), replay[k]) for k in fields),
             "private calibration aggregate report differs from trace replay")
    for cell in cells:
        grouped = report["groups"][cell["scenario"]]
        group_identity = {**expected, "num_envs": cell["num_envs"],
                          "transitions": cell["expected_policy_samples"], "environment_provenance": provenance}
        _require(all(_same_metrics(grouped.get(k), v) for k, v in group_identity.items()),
                 "private calibration scenario checkpoint, seed, clock or row identity differs")
        _require(all(k in replay["groups"][cell["scenario"]]
                     and _same_metrics(grouped.get(k), replay["groups"][cell["scenario"]][k]) for k in fields),
                 "private calibration scenario metrics or replicas differ")
    campaign._checked(outcome["report"])
    campaign._checked(outcome["trace"])
    return replay["trace_validation"]


def _cache_inventory(directory):
    root = directory / "runtime"
    files = []
    for base, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            p = definition._path(Path(base) / name)
            _require(stat.S_ISDIR(p.lstat().st_mode), "runtime cache contains special directory")
        for name in names:
            files.append(definition._receipt(Path(base) / name))
    files.sort(key=lambda r: r["path"])
    inventory = {"format": "transformer_rl.exposure_runtime_cache_measurement", "schema_version": 1,
                 "root": str(root), "files": files, "total_bytes": sum(r["bytes"] for r in files)}
    path = directory / "runtime.measurement.json"
    campaign._new(path, inventory)
    return {"kind": "runtime_cache", "receipt": definition._receipt(path), "units": 1,
            "observed_final_bytes": inventory["total_bytes"], "nonempty": bool(files)}


def run(plan_path, *, expected_plan_sha256):
    """Run serial fresh SDK processes only after original queues really close."""
    receipt = definition._receipt(plan_path)
    _require(receipt["sha256"] == expected_plan_sha256, "calibration plan differs from external authorization")
    plan, protocol = validate_plan(receipt)
    root = Path(plan["output_root"])
    _require(not os.path.lexists(root), "calibration output cannot be reused")
    campaign.check_disk(root.parent, plan["limits"]["max_owned_bytes"] + plan["limits"]["free_margin_bytes"])
    root.mkdir(mode=0o700)
    limits = plan["limits"]
    observer = CalibrationObserver(root, limits["max_owned_bytes"], limits["free_margin_bytes"], limits["interval_s"])
    started = time.monotonic()
    worker_records, measurements, helpers = [], [], {}
    status, error, observed_helpers, storage = "failed", None, None, None
    helper_closure_failures = {"count": 0, "first": None, "last": None}

    def validate_output_identity():
        from .calibration_storage import _root
        _require(_root(root)[1] == observer.root_identity, "terminal owned root identity changed")

    def publish(state, **values):
        validate_output_identity()
        campaign._publish(root / "progress.json", {"status": state, "plan": receipt,
            "elapsed_s": time.monotonic() - started, **values})

    def monitor():
        campaign._checked(receipt)
        campaign._checked(plan["protocol"])
        _require(source_identity() == protocol["source"], "calibration producer source changed")
        observer.guard()
        inventory = helper_inventory(root, tuple(helpers.values()))
        _require(inventory["observer_ancestry_complete"], "calibration helper ancestry observation is incomplete")
        for handle in inventory["handles"]:
            helpers[(handle["pid"], handle["start"], handle["uid"])] = handle
        return inventory

    def close_helpers():
        nonlocal observed_helpers, status, error

        def record_failure(failure):
            nonlocal status, error
            issue = {"type": type(failure).__name__, "message": str(failure)}
            status = "failed"
            if error is None:
                error = issue
            helper_closure_failures["count"] += 1
            if helper_closure_failures["first"] is None:
                helper_closure_failures["first"] = issue
            helper_closure_failures["last"] = issue

        # Keep both controller flocks while an actually observed helper is
        # still alive. A timeout or incomplete observation never releases an
        # SDK helper into the next queue. Only exact PID/start/UID facts count.
        while True:
            try:
                campaign._checked(receipt)
                campaign._checked(plan["protocol"])
                _require(source_identity() == protocol["source"], "calibration source changed while closing helpers")
                observer.guard()
            except BaseException as failure:
                record_failure(failure)
            try:
                inventory = helper_inventory(root, tuple(helpers.values()))
                for handle in inventory["handles"]:
                    helpers[(handle["pid"], handle["start"], handle["uid"])] = handle
                observed_helpers = inventory
                if inventory["observer_ancestry_complete"] and inventory["all_tracked_terminal"]:
                    return
                publish("waiting_for_observed_helpers", helper_inventory=inventory)
            except BaseException as failure:
                # Observation or progress I/O failure, including interruption,
                # cannot release the original leases around a live helper.
                # Bound the error record while continuing only closure checks.
                record_failure(failure)
            try:
                time.sleep(min(1., limits["interval_s"]))
            except BaseException as failure:
                record_failure(failure)

    try:
        # Waiting consumes no SDK namespace; the prior whole-calibration cap
        # includes the observer/progress records created during that wait.
        observer.start()
        with campaign.resource_lease(protocol, publish) as leases:
            validate_plan(receipt)
            try:
                controller = {"format": "transformer_rl.calibration_controller", "schema_version": 1,
                    "root": str(root), "root_identity": observer.root_identity, "plan": receipt,
                    "protocol": plan["protocol"], "source": protocol["source"], "runtime": protocol["runtime"],
                    "process": campaign._identity(os.getpid()), "limits": limits,
                    "formal_architecture_selection": False, "production_storage_authorized": False, "hardware_verified": False}
                campaign._new(root / "controller.json", controller)
                controller_receipt = definition._receipt(root / "controller.json")
                for job_id in plan["job_ids"]:
                    job = next(j for j in protocol["jobs"] if j["id"] == job_id)
                    (root / job_id).mkdir(mode=0o700)
                    checkpoint = None
                    for kind in ("train", "evaluate"):
                        monitor()
                        directory, request_receipt, command = _request(receipt, plan, protocol, job, kind, root,
                            controller_receipt, leases, checkpoint)
                        result = campaign.launch_owned_worker(command, directory, leases, limits["timeout_s"], publish,
                                                              monitor=monitor)
                        outcome = _closed_outcome(directory, request_receipt, result, protocol)
                        inventory = monitor()
                        _require(inventory["all_tracked_terminal"], "calibration worker left a live observed helper")
                        observed_helpers = inventory
                        cache = _cache_inventory(directory)
                        measurements.append(cache)
                        if kind == "train":
                            checkpoint = _training_checkpoint(directory, receipt, protocol, job, plan)
                            completion = campaign._read(campaign._checked(outcome["training_completion"]))
                            metric = directory / "train" / "metrics.jsonl"
                            measurements.extend(({"kind": "checkpoint", "receipt": checkpoint, "units": 1},
                                {"kind": "metric", "receipt": definition._receipt(metric), "units": completion["recorded_metric_updates"]}))
                        else:
                            for key in ("report", "trace"):
                                campaign._checked(outcome[key])
                            units = sum(c["expected_policy_samples"] for c in _cells(protocol, job))
                            replay = _verify_private_evaluation(plan, protocol, job, checkpoint, outcome)
                            raw_bytes = campaign.measured_trace_bytes(outcome["trace"]["path"], units)
                            measurements.append({"kind": "trace", "receipt": outcome["trace"], "units": units,
                                                 "uncompressed_bytes": raw_bytes, "trace_validation": replay})
                        worker_records.append({"job_id": job_id, "candidate": job["candidate"], "kind": kind,
                            "request": request_receipt, "process_completion": definition._receipt(directory / "worker.completion.json"),
                            "outcome": definition._receipt(directory / "outcome.json"), "actual_counters": outcome["actual_counters"]})
                        publish("calibrating", completed_workers=len(worker_records), expected_workers=plan["coverage"]["worker_namespaces"])
                validate_plan(receipt)
                _require(len(worker_records) == plan["coverage"]["worker_namespaces"], "calibration worker coverage incomplete")
                status = "completed"
            finally:
                close_helpers()
    except BaseException as failure:
        status = "failed"
        error = {"type": type(failure).__name__, "message": str(failure)}
    # Publish the worker facts while the observer still sees publication.
    value = {"format": "transformer_rl.calibration_result", "schema_version": 1, "status": status,
        "error": error, "plan": receipt, "source": protocol["source"], "coverage": plan["coverage"],
        "workers": worker_records, "measurements": measurements, "sampled_storage": observer.report(),
        "sampled_storage_scope": "pre-result snapshot; storage.final.json contains sampling through result/progress publication",
        "observed_helpers": observed_helpers, "helper_closure_failures": helper_closure_failures,
        "elapsed_s": time.monotonic() - started,
        "continuous_peak_verified": False, "all_system_cap_verified": False,
        "formal_architecture_selection": False, "production_storage_authorized": False, "hardware_verified": False}
    result_receipt = storage_receipt = completion_receipt = None
    try:
        validate_output_identity()
        campaign._new(root / "result.json", value)
        result_receipt = definition._receipt(root / "result.json")
        publish(status, result=result_receipt)
    except BaseException as failure:
        status = "failed"
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        try:
            storage = observer.stop()
        except BaseException as failure:
            storage = observer.report()
            status = "failed"
            if error is None:
                error = {"type": type(failure).__name__, "message": str(failure)}
    try:
        validate_output_identity()
        campaign._new(root / "storage.final.json", storage)
        storage_receipt = definition._receipt(root / "storage.final.json")
    except BaseException as failure:
        status = "failed"
        error = {"type": type(failure).__name__, "message": str(failure)}
    final = {"format": "transformer_rl.calibration_completion", "schema_version": 1,
        "status": status, "error": error, "plan": receipt, "source": protocol["source"],
        "result": result_receipt, "sampled_storage": storage_receipt,
        "terminal_storage_guard_after_publication": False,
        "formal_architecture_selection": False, "production_storage_authorized": False, "hardware_verified": False}
    try:
        validate_plan(receipt)
        guard_owned_storage(root, limits["max_owned_bytes"], limits["free_margin_bytes"], observer.root_identity)
        final["terminal_storage_guard_after_publication"] = True
        validate_output_identity()
        campaign._new(root / "completion.json", final)
        # This last actual scan includes completion.json, storage.final.json,
        # result.json and progress.json; no file is written on success after it.
        guard_owned_storage(root, limits["max_owned_bytes"], limits["free_margin_bytes"], observer.root_identity)
        completion_receipt = definition._receipt(root / "completion.json")
    except BaseException as failure:
        final.update(status="failed", error={"type": type(failure).__name__, "message": str(failure)},
                     terminal_storage_guard_after_publication=False)
        # Invalidate a provisional successful completion before best-effort
        # failure publication. Persistent I/O failure must not leave success.
        try:
            validate_output_identity()
            (root / "completion.json").unlink(missing_ok=True)
            campaign._publish(root / "completion.json", final)
            completion_receipt = definition._receipt(root / "completion.json")
        except BaseException as publication_failure:
            final["publication_error"] = {"type": type(publication_failure).__name__,
                                          "message": str(publication_failure)}
    return {**value, "status": final["status"], "error": final["error"], "sampled_storage": storage,
            "completion": completion_receipt, "terminal_publication_error": final.get("publication_error")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("plan")
    prepare.add_argument("--protocol", required=True)
    prepare.add_argument("--expected-protocol-sha256", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--candidate", action="append", dest="candidates")
    prepare.add_argument("--training-seed", type=int, required=True)
    prepare.add_argument("--updates", type=int, default=2)
    prepare.add_argument("--evaluation-seed", type=int, default=92001)
    prepare.add_argument("--max-owned-bytes", type=int, required=True)
    prepare.add_argument("--free-margin-bytes", type=int, required=True)
    prepare.add_argument("--interval-s", type=float, default=.1)
    prepare.add_argument("--timeout-s", type=float, default=3600.)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    if args.operation == "plan":
        path = definition._path(args.plan)
        value = build_plan(args.protocol, expected_protocol_sha256=args.expected_protocol_sha256,
            output_root=args.output_root, candidates=args.candidates, training_seed=args.training_seed,
            updates=args.updates, evaluation_seed=args.evaluation_seed, max_owned_bytes=args.max_owned_bytes,
            free_margin_bytes=args.free_margin_bytes, interval_s=args.interval_s, timeout_s=args.timeout_s)
        _require(not path.is_relative_to(Path(value["output_root"])), "calibration plan must be external")
        campaign._new(path, value)
        print(definition._receipt(path)["sha256"])
        return 0
    value = run(args.plan, expected_plan_sha256=args.expected_plan_sha256)
    return 0 if value["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

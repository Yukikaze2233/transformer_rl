"""Immutable validation-only choices from complete, replayed physical evidence.

This module fixes exact final-stage artifacts before any held-out observations.
It provides a provisional control-quality choice, never deployment qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import stat
import statistics
import sys

from . import exposure_campaign as campaign
from . import exposure_evaluation as evaluation
from . import exposure_protocol as definition
from .frame_config import FrameTrainConfig, digest
from . import runtime_paths


FORMAT = "transformer_rl.exposure_validation_choice"
_RULE = {
    "ranking": "final_stage_equal_scenario_and_validation_seed_mean_then_training_mean_plus_sample_std_penalty",
    "eligibility": "every_declared_training_seed_complete_final_task_gates_and_acquired_skill_retention",
    "checkpoint": "final_stage_of_training_seed_nearest_candidate_median_score_then_seed_number",
    "retention": "earlier_passed_cell_must_remain_passed_and_not_exceed_best_acquired_score_plus_tolerance",
    "early_stages": "learning_and_acquisition_conditioned_retention_only_never_final_ranking",
    "held_out": "every_original_candidate_seed_stage_case_seed_once_after_immutable_validation_choice",
}


def _rule(protocol):
    rule = deepcopy(_RULE)
    if protocol.get("schema_version") == 2:
        rule.update(
            checkpoint="final_endpoint_of_final_stage_of_training_seed_nearest_candidate_median_score_then_seed_number",
            early_stages="all_scheduled_checkpoints_learning_and_acquisition_conditioned_retention_only_never_final_ranking",
            held_out="every_original_candidate_seed_stage_checkpoint_case_seed_once_after_immutable_validation_choice")
    return rule


def _require(condition, reason):
    campaign._require(condition, reason)


def _signature(path):
    """Actual metadata of an owned canonical regular file, never a receipt claim."""
    path = definition._path(path)
    value = path.lstat()
    _require(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid(),
             "selection evidence must be an owned regular file")
    return {"device": value.st_dev, "inode": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns, "nlink": value.st_nlink}


class _Pins:
    """Bind published bytes and absences without copying raw physical metrics."""

    def __init__(self):
        self.files = {}
        self.signatures = {}
        self.missing = set()

    def receipt(self, path):
        before = _signature(path)
        pin = definition._receipt(path)
        _require(_signature(path) == before and before["size"] == pin["bytes"],
                 "selection input metadata changed while reading")
        old = self.files.setdefault(pin["path"], pin)
        original = self.signatures.setdefault(pin["path"], before)
        _require(old == pin, "selection input bytes changed while reconstructing")
        _require(original == before, "selection input metadata changed while reconstructing")
        return deepcopy(pin)

    def optional(self, path):
        path = definition._path(path)
        if os.path.lexists(path):
            return self.receipt(path)
        self.missing.add(str(path))
        return None

    def checked(self, pin):
        campaign._checked(pin)
        _require(self.receipt(pin["path"]) == pin, "selection publication receipt differs")
        return Path(pin["path"])

    def read(self, path):
        pin = self.receipt(path)
        value = campaign._read(path)
        _require(definition._receipt(path) == pin, "selection JSON changed while reading")
        return value

    def verify(self):
        for pin in self.files.values():
            campaign._checked(pin)
            _require(_signature(pin["path"]) == self.signatures[pin["path"]],
                     "selection input metadata changed during full replay")
        _require(not any(os.path.lexists(path) for path in self.missing),
                 "a missing validation/training publication appeared during selection")


def _terminal(directory, command, leases, pins):
    worker = pins.read(directory / "worker.completion.json")
    process = pins.read(directory / "worker.process.json")
    profile_receipt = pins.receipt(directory / runtime_paths.PROFILE_FILENAME)
    runtime_paths.validate_runtime_artifact(directory, profile_receipt)
    _require(worker.get("runtime_profile") == process.get("runtime_profile") == profile_receipt,
             "closed worker runtime profile differs from parent and child bindings")
    handle = worker.get("process", {})
    _require(type(handle) is dict and set(handle) == {"pid", "start", "argv"}
             and type(handle["pid"]) is int and handle["pid"] > 0
             and type(handle["start"]) is int and handle["start"] > 0
             and handle["argv"] == command, "closed worker kernel identity differs")
    live = campaign.predecessors._process(handle["pid"])
    _require(worker.get("format") == "transformer_rl.exposure_worker_completion"
             and process.get("format") == "transformer_rl.exposure_worker_process"
             and type(worker.get("schema_version")) is int and worker["schema_version"] == 1
             and type(process.get("schema_version")) is int and process["schema_version"] == 1
             and type(worker.get("returncode")) is int and type(worker.get("timed_out")) is bool
             and worker["command"] == process["command"] == command
             and digest(worker["process"]) == digest(process["process"])
             and digest(process["leases"]) == digest(leases)
             and (live is None or live["start"] != handle["start"]),
             "selection requires an exact terminal worker publication")
    return worker


def _controller(protocol, raw, pins):
    root = Path(protocol["output_root"])
    value = pins.read(root / "controller.json")
    _require(value.get("format") == "transformer_rl.exposure_controller"
             and type(value.get("schema_version")) is int and value["schema_version"] == 1
             and value["protocol_raw_receipt"] == raw
             and value["expected_protocol_sha256"] == raw["sha256"]
             and value["protocol_sha256"] == protocol["sha256"]
             and value["source"] == protocol["source"] and value["runtime"] == protocol["runtime"],
             "choice does not belong to the externally authorized campaign")
    owner = value.get("process", {})
    _require(type(owner) is dict and set(owner) == {"pid", "start", "argv"}
             and type(owner["pid"]) is int and owner["pid"] > 0
             and type(owner["start"]) is int and owner["start"] > 0
             and type(owner["argv"]) is list and bool(owner["argv"])
             and all(type(argument) is str for argument in owner["argv"]),
             "selection controller process declaration differs")
    storage_path = pins.checked(value["storage_contract_receipt"])
    storage = campaign._validate_storage(pins.read(storage_path), protocol, raw)
    for item in storage["measurements"]:
        pins.checked(item["receipt"])
        if item["kind"] == "runtime_cache":
            for receipt in pins.read(item["receipt"]["path"])["files"]:
                pins.checked(receipt)
    return value, pins.receipt(root / "controller.json")


def _endpoint_path(protocol, job, stage):
    return Path(protocol["output_root"]) / job["id"] / f"stage_{stage['index']:04d}" / "train" \
        / f"stage_{stage['index']:04d}_{stage['name']}" / "endpoint.json"


def _training_checkpoints(protocol, job, stage, item, completion_receipt, pins, *, completed):
    """Replay every scheduled save, including a failed learner's sealed prefix.

    A checkpoint is an observation point, never permission to resume a learner.
    The campaign verifier proves its actual model, optimizer and metric prefix;
    the selection seal also pins the containing immutable files.
    """
    if "checkpoint_updates" not in stage:
        return None
    verified = campaign.verified_stage_checkpoints(protocol, job, stage, completion_receipt,
                                                  completed=completed)
    _require(digest(item.get("checkpoints")) == digest(verified),
             "job ledger changes its actual sealed learning checkpoints")
    pins.checked(completion_receipt)
    directory = Path(protocol["output_root"]) / job["id"] / f"stage_{stage['index']:04d}" / "train"
    for name in ("request.json", "reservation.json", "metrics.jsonl"):
        pins.receipt(directory / name)
    for entry in verified:
        pins.checked(entry["record"])
        pins.checked(entry["checkpoint"])
        record = pins.read(entry["record"]["path"])
        pins.receipt(record["sidecar"]["path"])
    return deepcopy(verified)


def _observation_record(stage_record, cell):
    """Route a cell to its exact scheduled save without replacing a missing one."""
    if "checkpoint_update" not in cell:
        return stage_record
    checkpoint_update = cell["checkpoint_update"]
    _require(type(checkpoint_update) is int, "explicit integer observation checkpoint required")
    matches = [entry for entry in stage_record.get("checkpoints", [])
               if entry["checkpoint_update"] == checkpoint_update]
    _require(len(matches) <= 1, "duplicate training checkpoint observation")
    entry = matches[0] if matches else None
    return {"stage_index": cell["stage_index"], "checkpoint_update": checkpoint_update,
            "endpoint": deepcopy(entry["record"]) if entry else None,
            "checkpoint": deepcopy(entry["checkpoint"]) if entry else None}


def _training(protocol, raw, job, controller_receipt, pins):
    root = Path(protocol["output_root"]) / job["id"]
    reservation_receipt = pins.receipt(root / "reservation.json")
    _require(digest(pins.read(root / "reservation.json")) == digest(campaign.reservation_for(protocol, raw, job)),
             "selection whole-job reservation differs")
    published = pins.read(root / "receipt.json")
    _require(published.get("status") in ("training_completed", "numerical_failure")
             and published.get("reservation") == reservation_receipt
             and type(published.get("stages")) is list, "training job is not closed")
    records, parent, failed_at = [], None, None
    for stage in job["stages"]:
        directory = root / f"stage_{stage['index']:04d}"
        if failed_at is not None:
            _require(not os.path.lexists(directory), "a numerical failure was retried or bypassed")
            pins.missing.add(str(directory))
            record = {"stage_index": stage["index"], "status": "missing", "reason": "prior_training_numerical_failure",
                      "endpoint": None, "checkpoint": None}
            if "checkpoint_updates" in stage:
                record["checkpoints"] = []
            records.append(record)
            continue
        request_receipt = pins.receipt(directory / "request.json")
        request = pins.read(directory / "request.json")
        expected = campaign.build_stage_request(protocol, raw, job, stage, directory, request["leases"],
            controller_receipt, reservation_receipt, parent, request["storage_contract"], request["required_remaining_bytes"])
        _require(digest(request) == digest(expected), "closed training request changes the fixed job/stage")
        _require(type(request["required_remaining_bytes"]) is int and request["required_remaining_bytes"] > 0,
                 "closed training lacks its positive original disk reservation")
        controller = campaign._read(controller_receipt["path"])
        _require(request["storage_contract"] == controller["storage_contract_receipt"], "training storage authorization differs")
        owner = controller["process"]
        leases, lock_pins = request["leases"], protocol["execution"]["resource_locks"]
        _require(type(leases) is list and len(leases) == len(lock_pins) == 2,
                 "closed training requires both original lock declarations")
        descriptors = set()
        for lease, pin in zip(leases, lock_pins):
            _require(type(lease) is dict and set(lease) == {"path", "device", "inode", "descriptor",
                     "controller_pid", "controller_start"}
                     and all(type(lease[k]) is int for k in
                             ("device", "inode", "descriptor", "controller_pid", "controller_start"))
                     and lease["descriptor"] >= 0 and lease["controller_pid"] == owner["pid"]
                     and lease["controller_start"] == owner["start"]
                     and digest({k: lease[k] for k in ("path", "device", "inode")}) == digest(pin),
                     "closed training lock/controller identity differs")
            descriptors.add(lease["descriptor"])
        _require(len(descriptors) == 2, "closed training lock descriptors differ")
        command = [sys.executable, "-B", "-m", "transformer_rl.exposure_process", "--request", request_receipt["path"],
                   "--expected-request-sha256", request_receipt["sha256"]]
        worker = _terminal(directory, command, request["leases"], pins)
        outcome_path = directory / "outcome.json"
        outcome = pins.read(outcome_path)
        _require(outcome.get("format") == "transformer_rl.exposure_stage_outcome"
                 and type(outcome.get("schema_version")) is int and outcome["schema_version"] == 1
                 and outcome["request"] == request_receipt and outcome["source"] == protocol["source"]
                 and outcome.get("runtime_profile") == worker["runtime_profile"]
                 and digest(outcome["process"]) == digest(worker["process"])
                 and outcome["shutdown_errors"] == [] and outcome["error"] is None,
                 "closed training outcome differs from actual worker")
        _require(len(published["stages"]) > stage["index"], "closed job omits its actual training stage")
        item = published["stages"][stage["index"]]
        _require(type(item["stage_index"]) is int and item["stage_index"] == stage["index"]
                 and digest(item["worker"]) == digest(worker)
                 and item["outcome"] == pins.receipt(outcome_path), "job ledger changes its actual terminal stage")
        if outcome["status"] == "numerical_failure":
            _require(worker["returncode"] == 20 and worker["timed_out"] is False
                     and outcome["typed_numerical_origin"] in ("environment_step", "optimizer_update"),
                     "failed training is not a typed numerical outcome")
            completion = pins.read(pins.checked(outcome["training_completion"]))
            _require(completion.get("status") == "failed" and completion.get("job_id") == job["id"]
                     and completion.get("source") == protocol["source"] and completion.get("shutdown_errors") == []
                     and completion.get("error", {}).get("type") == "FloatingPointError"
                     and completion.get("error", {}).get("phase") == "collect_optimize", "actual numerical completion differs")
            for name, upper in (("successful_updates", stage["updates"]), ("attempted_updates", stage["updates"]),
                                ("actual_collected_transitions", stage["fresh_transitions"])):
                _require(type(completion.get(name)) is int and 0 <= completion[name] <= upper,
                         "numerical outcome counters exceed the reserved stage")
            optimizer_steps = completion.get("failed_update_optimizer_steps")
            _require((type(optimizer_steps) is int and optimizer_steps >= 0)
                     or (optimizer_steps is None and completion.get("optimizer_update_may_be_partial") is True),
                     "numerical outcome lacks actual failed optimizer accounting")
            failed_training = {"completion": outcome["training_completion"],
                "successful_updates": completion["successful_updates"],
                "attempted_updates": completion["attempted_updates"],
                "actual_collected_transitions": completion["actual_collected_transitions"],
                "optimizer_steps_of_failed_update": completion["failed_update_optimizer_steps"],
                "accounting_scope": "actual counters; no sealed endpoint for this stage"}
            _require(digest(item.get("failed_training")) == digest(failed_training),
                     "job ledger changes actual numerical accounting")
            failed_at = stage["index"]
            record = {"stage_index": stage["index"], "status": "numerical_failure", "endpoint": None,
                      "checkpoint": None, "completion": deepcopy(outcome["training_completion"])}
            checkpoints = _training_checkpoints(protocol, job, stage, item,
                outcome["training_completion"], pins, completed=False)
            if checkpoints is not None:
                record["checkpoints"] = checkpoints
            records.append(record)
        else:
            _require(outcome["status"] == "completed" and worker["returncode"] == 0
                     and worker["timed_out"] is False, "unknown/interrupted training cannot close selection")
            proof = campaign.prove_stage_endpoint(protocol, job, stage, _endpoint_path(protocol, job, stage))
            _require(outcome["training_completion"] == proof["completion"] and digest(item.get("training")) == digest(proof),
                     "job ledger differs from actual complete learning proof")
            endpoint = campaign.verify_segment_endpoint(protocol, job, stage["index"], proof["endpoint"])
            for pin in (proof["endpoint"], proof["checkpoint"], proof["completion"], proof["metrics"]):
                pins.checked(pin)
            pins.receipt(endpoint["sidecar"]["path"])
            for name in ("request.json", "reservation.json"):
                pins.receipt(directory / "train" / name)
            parent = proof["endpoint"]
            record = {"stage_index": stage["index"], "status": "completed", "endpoint": deepcopy(parent),
                      "checkpoint": deepcopy(proof["checkpoint"]), "successful_updates": proof["actual_updates"],
                      "fresh_transitions": proof["actual_fresh_transitions"]}
            checkpoints = _training_checkpoints(protocol, job, stage, item, proof["completion"], pins, completed=True)
            if checkpoints is not None:
                record["checkpoints"] = checkpoints
            records.append(record)
    _require(len(published["stages"]) == (failed_at + 1 if failed_at is not None else len(job["stages"]))
             and published["status"] == ("numerical_failure" if failed_at is not None else "training_completed"),
             "closed job stages or final status differ")
    return {"status": published["status"], "stages": records, "reservation": reservation_receipt,
            "receipt": pins.receipt(root / "receipt.json")}


def _directory(protocol, first):
    return campaign.evaluation_directory(protocol, first)


def _validation_batch(protocol, cells, stage_record, pins):
    directory = _directory(protocol, cells[0])
    batch_id = "batch_" + digest(campaign.evaluation_batch_key(cells[0]))[:24]
    paths = {name: directory / name for name in ("request.json", "report.json", "trace.npz",
        "evaluation.completion.json", "worker.process.json", "worker.completion.json")}
    existing = {name: pins.optional(path) for name, path in paths.items()}
    if stage_record["endpoint"] is None:
        _require(not any(existing.values()), "physical evaluation substituted a checkpoint after failed training")
        return batch_id, {"status": "missing", "reason": "no_sealed_training_endpoint", "cells": [c["id"] for c in cells]}, {}
    _require(existing["request.json"] is not None or not any(existing.values()),
             "evaluation publications have no original request")
    if existing["request.json"] is not None:
        request, actual, _ = evaluation._request(existing["request.json"])
        _require(actual == protocol and request["endpoint"] == stage_record["endpoint"]
                 and digest(request["cells"]) == digest(cells), "evaluation changes its declared batch")
        if existing["worker.process.json"] and existing["worker.completion.json"]:
            command = [sys.executable, "-B", "-m", "transformer_rl.exposure_evaluation", "worker", "--request",
                       existing["request.json"]["path"], "--expected-request-sha256", existing["request.json"]["sha256"]]
            worker = _terminal(directory, command, request["leases"], pins)
            if worker["returncode"] != 0 or worker["timed_out"]:
                return batch_id, {"status": "failed", "reason": "physical_worker_did_not_complete",
                    "cells": [c["id"] for c in cells], "worker": worker,
                    "request": existing["request.json"]}, {}
    if not all(existing.values()):
        if existing["request.json"] is not None:
            request, actual, _ = evaluation._request(existing["request.json"])
            _require(actual == protocol and request["endpoint"] == stage_record["endpoint"]
                     and digest(request["cells"]) == digest(cells), "partial evaluation changes its declared batch")
            if existing["worker.process.json"] and existing["worker.completion.json"]:
                command = [sys.executable, "-B", "-m", "transformer_rl.exposure_evaluation", "worker", "--request",
                           existing["request.json"]["path"], "--expected-request-sha256", existing["request.json"]["sha256"]]
                _terminal(directory, command, request["leases"], pins)
            elif existing["worker.process.json"]:
                process = pins.read(paths["worker.process.json"])["process"]
                live = campaign.predecessors._process(process["pid"])
                _require(live is None or live["start"] != process["start"], "evaluation is still running")
        return batch_id, {"status": "missing", "reason": "incomplete_physical_publication", "cells": [c["id"] for c in cells]}, {}
    result = evaluation.verify_result(protocol, stage_record["endpoint"], cells, directory, existing["request.json"])
    _require(set(result["cells"]) == {c["id"] for c in cells}, "physical replay shrank the declared cell denominator")
    for pin in result["artifacts"].values():
        pins.checked(pin)
    records = {}
    for cell in cells:
        measured = result["cells"][cell["id"]]
        _require(digest(measured["identity"]) == digest(cell), "physical cell identity differs")
        grade, history_gate = _physical_grade(protocol, cell, measured)
        records[cell["id"]] = {"identity": deepcopy(cell), "status": "completed", "batch": batch_id,
                               "grade": grade, "full_history_gate": history_gate}
    return batch_id, {"status": "completed", "cells": [c["id"] for c in cells],
        "request": existing["request.json"], "endpoint": deepcopy(stage_record["endpoint"]),
        "artifacts": deepcopy(result["artifacts"]), "trace_validation": deepcopy(result["trace_validation"])}, records


def _physical_grade(protocol, cell, measured):
    """Require enough genuinely observed history, using already replayed rows."""
    from .frame_study import grade_report
    config = FrameTrainConfig.from_dict(definition._read(campaign._checked(cell["config_receipt"])))
    metrics, evaluation_spec = measured["metrics"], protocol["evaluation"]
    scenario = next(item for item in protocol["scenarios"] if item["name"] == cell["scenario"])
    grade = grade_report(metrics, scenario, evaluation_spec, protocol["selection"]["objectives"])
    _require(digest(grade) == digest(measured["grade"]), "provider grade differs from original declared objectives")
    history = metrics["history_control"]
    _require(type(history["history_length"]) is int and history["history_length"] == config.model.history_length
             and type(history["minimum_full_age"]) is int
             and history["minimum_full_age"] == config.model.history_length - 1
             and evaluation._same_metrics(history["policy_dt_s"], float(config.control["policy_dt_s"]))
             and type(history["settle_steps"]) is int and history["settle_steps"] == evaluation_spec["settle_steps"]
             and type(history["min_steady_samples"]) is int
             and history["min_steady_samples"] == evaluation_spec["min_steady_samples"],
             "full-history measurement uses a different model/age/clock contract")
    window = history["windows"]["full_history"]
    minimum = evaluation_spec["min_steady_samples"] * cell["num_envs"]
    samples, steady = window["samples"], window["steady_tracking"]["samples"]
    _require(type(samples) is int and samples >= 0 and type(steady) is int and 0 <= steady <= samples,
             "full-history sample counters differ")
    reasons = []
    if samples < minimum:
        reasons.append("insufficient_full_history_samples")
    if scenario.get("require_steady", False) and steady < minimum:
        reasons.append("insufficient_full_history_steady_samples")
    grade = {**grade, "passed": grade["passed"] and not reasons, "reasons": [*grade["reasons"], *reasons]}
    return grade, {"passed": not reasons, "reasons": reasons, "minimum_samples": minimum,
                   "actual_samples": samples, "actual_steady_samples": steady}


def _rank(protocol, training_records, cells):
    candidates = []
    names = list(dict.fromkeys(job["candidate"] for job in protocol["jobs"]))
    for name in names:
        jobs = [job for job in protocol["jobs"] if job["candidate"] == name]
        scores, seed_records, reasons = [], [], []
        for job in jobs:
            data = training_records[job["id"]]
            observations = [cells[cell["id"]] for cell in protocol["evaluation_cells"]
                            if cell["job_id"] == job["id"] and cell["role"] == "validation"]
            last = len(job["stages"]) - 1
            final_update = job["stages"][-1].get("expected_cumulative_updates")
            final = [record for record in observations if record["identity"]["stage_index"] == last
                     and ("checkpoint_update" not in record["identity"]
                          or record["identity"]["checkpoint_update"] == final_update)]
            failures = []
            if data["status"] != "training_completed" or any(record["status"] != "completed" for record in observations):
                failures.append("incomplete_declared_training_or_validation")
            if any(record["status"] != "completed" or not record.get("grade", {}).get("passed") for record in final):
                failures.append("final_stage_task_gate_failed")
            acquired, regressions, learning = {}, [], []
            ordered = (sorted(observations, key=lambda record: (
                record["identity"]["stage_index"], record["identity"].get("checkpoint_update", 0),
                record["identity"]["seed"], record["identity"]["scenario"]))
                if any("checkpoint_update" in record["identity"] for record in observations) else observations)
            for record in ordered:
                if record["status"] != "completed":
                    if "checkpoint_update" in record["identity"]:
                        learning.append({"cell": record["identity"]["id"], "status": record["status"],
                                         "acquisition_status": "unobserved"})
                    continue
                grade, identity = record["grade"], record["identity"]
                key = (identity["scenario"], identity["seed"])
                old = acquired.get(key)
                regressed = old is not None and (not grade["passed"] or grade["score"] is None
                    or grade["score"] > old["score"] + protocol["selection"]["retention_score_tolerance"])
                if regressed:
                    regressions.append({"cell": identity["id"], "acquired_cell": old["cell"],
                                        "acquired_score": old["score"], "current_score": grade["score"]})
                if "checkpoint_update" in identity:
                    learning.append({"cell": identity["id"], "stage_index": identity["stage_index"],
                        "checkpoint_update": identity["checkpoint_update"], "status": "completed",
                        "passed_gate": grade["passed"], "score": grade["score"],
                        "acquisition_status": ("regressed_after_acquisition" if regressed else
                            "retained" if old is not None else
                            "acquired" if grade["passed"] and grade["score"] is not None else
                            "not_yet_acquired")})
                if grade["passed"] and grade["score"] is not None and (old is None or grade["score"] < old["score"]):
                    acquired[key] = {"cell": identity["id"], "score": grade["score"]}
            if regressions:
                failures.append("acquired_skill_retention_failed")
            final_scores = [record["grade"]["score"] for record in final if record["status"] == "completed"
                            and record["grade"]["score"] is not None]
            score = statistics.mean(final_scores) if len(final_scores) == len(final) and final else None
            if score is None:
                failures.append("missing_final_stage_objective")
            else:
                scores.append(score)
            reasons.extend(f"seed_{job['training_seed']}:{failure}" for failure in failures)
            seed_record = {"training_seed": job["training_seed"], "job_id": job["id"],
                "score": score, "eligible": not failures, "reasons": failures,
                "expected_validation_cells": len(observations),
                "completed_validation_cells": sum(record["status"] == "completed" for record in observations),
                "acquired_skills": [{"scenario": key[0], "evaluation_seed": key[1], **value} for key, value in acquired.items()],
                "retention_regressions": regressions}
            if any("checkpoint_update" in record["identity"] for record in observations):
                seed_record["learning_observations"] = learning
            seed_records.append(seed_record)
        complete_scores = len(scores) == len(jobs)
        mean = statistics.mean(scores) if complete_scores else None
        std = statistics.stdev(scores) if complete_scores and len(scores) > 1 else None
        if len(jobs) < protocol["selection"]["min_training_seeds"]:
            reasons.append("insufficient_declared_training_seeds")
        architecture = jobs[0]["stages"][0]["config"]["model"]["policy"]["architecture"]
        candidate = {"candidate": name, "architecture": architecture, "eligible": not reasons,
            "reasons": reasons, "training_seeds": seed_records, "score_mean": mean, "score_sample_std": std,
            "rank_score": mean + protocol["selection"]["std_penalty"] * (std or 0.) if mean is not None else None}
        if not reasons:
            median = statistics.median(scores)
            chosen = min(seed_records, key=lambda record: (abs(record["score"] - median), record["training_seed"]))
            selected = training_records[chosen["job_id"]]["stages"][-1]
            candidate["representative"] = {"job_id": chosen["job_id"], "training_seed": chosen["training_seed"],
                "stage_index": selected["stage_index"], "endpoint": deepcopy(selected["endpoint"]),
                "checkpoint": deepcopy(selected["checkpoint"])}
            selected_job = next(job for job in jobs if job["id"] == chosen["job_id"])
            if "checkpoint_updates" in selected_job["stages"][-1]:
                candidate["representative"]["checkpoint_update"] = selected_job["stages"][-1]["expected_cumulative_updates"]
        candidates.append(candidate)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    transformers = [candidate for candidate in eligible if candidate["architecture"] == "transformer"]
    best = lambda items: min(items, key=lambda candidate: (candidate["rank_score"], candidate["candidate"])) if items else None
    return {"candidates": candidates, "best_transformer": deepcopy(best(transformers)),
            "best_overall": deepcopy(best(eligible)),
            "mlp_controls": [deepcopy(candidate) for candidate in candidates if candidate["architecture"] in ("mlp", "history_mlp")]}


def _build(protocol, raw, *, require_pristine_held_out):
    pins = _Pins()
    pins.checked(raw)
    _, controller_receipt = _controller(protocol, raw, pins)
    pristine_held_out = []
    if require_pristine_held_out:
        for cell in protocol["evaluation_cells"]:
            if cell["role"] == "held_out":
                directory = _directory(protocol, cell)
                _require(not os.path.lexists(directory), "held-out output existed before immutable validation choice")
                pristine_held_out.append(directory)
    training_records = {job["id"]: _training(protocol, raw, job, controller_receipt, pins) for job in protocol["jobs"]}
    validation = {cell["id"]: {"identity": deepcopy(cell), "status": "missing", "reason": "not_evaluated"}
                  for cell in protocol["evaluation_cells"] if cell["role"] == "validation"}
    batches, seen = {}, set()
    for cell in protocol["evaluation_cells"]:
        if cell["role"] != "validation":
            continue
        key = digest(campaign.evaluation_batch_key(cell))
        if key in seen:
            continue
        seen.add(key)
        declared = [item for item in protocol["evaluation_cells"]
                    if digest(campaign.evaluation_batch_key(item)) == key]
        stage_record = _observation_record(training_records[cell["job_id"]]["stages"][cell["stage_index"]], cell)
        batch_id, batch, measured = _validation_batch(protocol, declared, stage_record, pins)
        batches[batch_id] = batch
        for item in declared:
            validation[item["id"]] = measured.get(item["id"], {"identity": deepcopy(item), "status": "missing",
                "batch": batch_id, "reason": batch.get("reason", "not_evaluated"), "grade": None})
    ranking = _rank(protocol, training_records, validation)
    held_out = {}
    for cell in protocol["evaluation_cells"]:
        if cell["role"] != "held_out":
            continue
        record = _observation_record(training_records[cell["job_id"]]["stages"][cell["stage_index"]], cell)
        held_out[cell["id"]] = {"identity": deepcopy(cell), "endpoint": deepcopy(record["endpoint"]),
                               "checkpoint": deepcopy(record["checkpoint"])}
    actual, actual_raw = campaign.read_authorized_protocol(raw["path"], raw["sha256"])
    _require(actual == protocol and actual_raw == raw, "authorized protocol changed during selection")
    pins.verify()
    _require(not any(os.path.lexists(directory) for directory in pristine_held_out),
             "held-out output appeared while freezing validation choice")
    winner = ranking["best_transformer"]
    status = "selected_provisional" if winner is not None else "no_eligible_transformer" if ranking["best_overall"] else "no_eligible"
    value = {"format": FORMAT, "schema_version": 1, "protocol": deepcopy(raw), "source": deepcopy(protocol["source"]),
        "status": status, "rule": _rule(protocol), "selection_parameters": deepcopy(protocol["selection"]),
        "training": training_records, "validation_cells": validation, "validation_batches": batches, **ranking,
        "denominator": {"jobs": len(protocol["jobs"]), "validation_cells": len(validation),
            "completed_validation_cells": sum(cell["status"] == "completed" for cell in validation.values()),
            "held_out_cells": len(held_out)}, "held_out_cells": held_out,
        "artifacts": list(pins.files.values()), "input_signatures": deepcopy(pins.signatures),
        "missing_validation_paths": sorted(pins.missing),
        "deployment": {"status": "unverified", "latency_gate_applied": False,
            "required_limits": {name: protocol["selection"][name] for name in
                                ("latency_p99_ms", "latency_max_ms", "max_deadline_misses")},
            "reason": "validation choice is provisional until exact artifact export and real deployment runtime benchmark"},
        "formal_architecture_selection": False, "hardware_verified": False}
    return {**value, "sha256": digest(value)}


def freeze_selection(protocol_path, *, expected_protocol_sha256):
    """Seal once after closed training; poor/missing measurements stay visible."""
    protocol, raw = campaign.read_authorized_protocol(protocol_path, expected_protocol_sha256)
    directory = Path(protocol["output_root"]) / "selection"
    _require(not os.path.lexists(directory), "validation choice cannot be overwritten or retried")
    value = _build(protocol, raw, require_pristine_held_out=True)
    directory.mkdir(mode=0o700)
    path = directory / "choice.json"
    campaign._new(path, value)
    return {"choice": value, "receipt": definition._receipt(path)}


def verify_selection(selection_receipt):
    """Rebuild from all actual validation traces and the complete learning grid."""
    path = campaign._checked(selection_receipt)
    value = campaign._read(path)
    _require(value.get("format") == FORMAT and type(value.get("schema_version")) is int
             and value["schema_version"] == 1, "unsupported immutable validation choice")
    protocol, raw = campaign.read_authorized_protocol(value["protocol"]["path"], value["protocol"]["sha256"])
    _require(raw == value["protocol"] and path == Path(protocol["output_root"]) / "selection" / "choice.json",
             "choice is outside its original authorized campaign")
    expected = _build(protocol, raw, require_pristine_held_out=False)
    _require(digest(value) == digest(expected), "validation choice differs from actual full evidence")
    campaign._checked(selection_receipt)
    return value


def _sealed_inputs(selection_receipt):
    """Bounded stat authorization of an externally pinned, fully replayed seal.

    The caller must retain the exact receipt returned by freeze_selection.
    This does not replace its initial or final full hash/replay verification.
    Byte restoration cannot restore ctime/inode and therefore cannot reseal an
    altered validation input. Large NPZ files are not rehashed on this path.
    """
    path = campaign._checked(selection_receipt)
    before = _signature(path)
    choice = campaign._read(path)
    _require(choice.get("format") == FORMAT and type(choice.get("schema_version")) is int
             and choice["schema_version"] == 1
             and digest({k: v for k, v in choice.items() if k != "sha256"}) == choice.get("sha256"),
             "unsupported or changed externally pinned validation seal")
    protocol, raw = campaign.read_authorized_protocol(choice["protocol"]["path"], choice["protocol"]["sha256"])
    _require(path == Path(protocol["output_root"]) / "selection" / "choice.json"
             and choice["protocol"] == raw and choice["source"] == protocol["source"]
             and choice["rule"] == _rule(protocol) and choice["selection_parameters"] == protocol["selection"],
             "validation seal escapes its original protocol/source/rule")
    artifacts, signatures = choice["artifacts"], choice["input_signatures"]
    _require(type(artifacts) is list and type(signatures) is dict
             and len({pin["path"] for pin in artifacts}) == len(artifacts)
             and set(signatures) == {pin["path"] for pin in artifacts},
             "validation seal omits frozen input signatures")
    for pin in artifacts:
        _require(type(pin) is dict and set(pin) == {"path", "sha256", "bytes"}
                 and type(pin["bytes"]) is int and pin["bytes"] >= 0
                 and type(pin["sha256"]) is str and len(pin["sha256"]) == 64
                 and all(character in "0123456789abcdef" for character in pin["sha256"]),
                 "validation seal input receipt differs")
        signature = signatures[pin["path"]]
        _require(type(signature) is dict and set(signature) == {"device", "inode", "size", "mtime_ns", "ctime_ns", "nlink"}
                 and all(type(value) is int and value >= 0 for value in signature.values())
                 and signature["size"] == pin["bytes"] and signature["nlink"] >= 1
                 and _signature(pin["path"]) == signature,
                 "frozen validation input signature changed; no automatic reseal")
    _require(not any(os.path.lexists(definition._path(path)) for path in choice["missing_validation_paths"]),
             "a frozen missing validation/training publication appeared")
    declared = {cell["id"]: cell for cell in protocol["evaluation_cells"] if cell["role"] == "held_out"}
    _require(set(choice["held_out_cells"]) == set(declared)
             and all(digest(entry["identity"]) == digest(declared[cell_id])
                     for cell_id, entry in choice["held_out_cells"].items()),
             "validation seal changes the original held-out denominator")
    campaign._checked(selection_receipt)
    _require(_signature(path) == before, "validation seal metadata changed while authorizing")
    return choice, protocol, raw


def authorize_heldout_batch(selection_receipt, endpoint_receipt, cells, directory):
    """Permit only the original full batch and previously sealed exact endpoint."""
    choice, protocol, raw = _sealed_inputs(selection_receipt)
    _require(type(cells) is list and bool(cells), "nonempty held-out batch required")
    first = cells[0]
    _require(type(first) is dict and first.get("role") == "held_out"
             and all(type(first.get(name)) is int for name in ("stage_index", "seed")), "held-out role/stage/seed differs")
    key = campaign.evaluation_batch_key(first)
    _require("checkpoint_update" not in key or type(key["checkpoint_update"]) is int,
             "explicit integer held-out checkpoint update required")
    expected = [cell for cell in protocol["evaluation_cells"] if campaign.evaluation_batch_key(cell) == key]
    _require(bool(expected) and digest(cells) == digest(expected), "held-out batch shrinks or changes the original matrix")
    _require(definition._path(directory) == _directory(protocol, first), "held-out output route differs")
    fixed = [choice["held_out_cells"][cell["id"]] for cell in cells]
    _require(all(item["endpoint"] is not None and item["endpoint"] == endpoint_receipt for item in fixed),
             "held-out evaluation changes its sealed checkpoint or lacks a trained endpoint")
    job = next(job for job in protocol["jobs"] if job["id"] == first["job_id"])
    stage = job["stages"][first["stage_index"]]
    endpoint = campaign.verify_learning_checkpoint(protocol, job, first["stage_index"],
        first.get("checkpoint_update", stage["expected_cumulative_updates"]), endpoint_receipt)
    _require(all(endpoint["checkpoint"] == item["checkpoint"] for item in fixed), "held-out actual checkpoint differs from choice")
    campaign._checked(selection_receipt)
    return {"selection": deepcopy(selection_receipt), "protocol": raw, "endpoint": deepcopy(endpoint_receipt),
            "cells": deepcopy(cells), "directory": str(definition._path(directory)),
            "choice_status": choice["status"], "selection_may_change": False, "hardware_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args(argv)
    freeze_selection(args.protocol, expected_protocol_sha256=args.expected_protocol_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

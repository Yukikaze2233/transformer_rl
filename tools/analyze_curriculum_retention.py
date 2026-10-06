#!/usr/bin/env python3
"""Freeze and apply read-only, standard-library curriculum retention analysis.

No learner, simulator, checkpoint deserializer or historical source is imported.
The protocol is separate from the frozen campaign; absent evidence stays absent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ARMS = {"mixed": ("mixed", "mixed"), "stationary": ("stationary", "stationary"),
        "pretrain": ("stationary", "mixed")}
PROFILES = ("nominal", "delay20_15", "delay40_30", "delay80_60", "noise",
            "payload_com", "low_grip", "combined")
NEW_SKILLS = ("forward_05", "backward_05", "rotate_1", "start_stop_05", "height_scan")
OLD_CASES = [f"stand_305mm__{p}" for p in (*PROFILES, "motor_weak", "spring_weak")]
NEW_CASES = [f"{s}__{p}" for s in NEW_SKILLS for p in PROFILES]
METRICS = {"height_mae_m": "metrics.height_abs_error.mean",
           "vx_mae_m_s": "metrics.vx_abs_error.mean",
           "yaw_mae_rad_s": "metrics.wz_abs_error.mean",
           "tilt_mean_rad": "metrics.tilt_angle.mean", "drift_max_m": "metrics.drift_m.max"}
SHA = re.compile(r"[0-9a-f]{64}")


def digest(value, *, ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=ascii,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def floating(value):
        result = float(value)
        _require(math.isfinite(result), "non-finite JSON number")
        return result
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid, parse_float=floating)


def _inside(root, route):
    _require(isinstance(route, str) and route, "artifact path must be a nonempty string")
    root = Path(root).resolve()
    path = Path(route)
    path = (path if path.is_absolute() else root / path).resolve()
    _require(path != root and path.is_relative_to(root), "artifact path escapes its root")
    return path


class Reader:
    def __init__(self):
        self.files = {}

    def raw(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        receipt = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        known = self.files.setdefault(str(path), receipt)
        _require(known == receipt, f"input changed while reading: {path}")
        return raw

    def hash(self, path):
        path = Path(path).resolve()
        sha, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        receipt = {"sha256": sha.hexdigest(), "bytes": size}
        known = self.files.setdefault(str(path), receipt)
        _require(known == receipt, f"input changed while reading: {path}")
        return receipt

    def read(self, path):
        return _json(self.raw(path))

    def checked(self, root, receipt):
        _require(isinstance(receipt, dict) and SHA.fullmatch(str(receipt.get("sha256", ""))),
                 "artifact receipt requires SHA256")
        path = _inside(root, receipt.get("path"))
        self.hash(path)
        _require(self.files[str(path)]["sha256"] == receipt["sha256"], f"artifact SHA mismatch: {path}")
        if "bytes" in receipt:
            _require(self.files[str(path)]["bytes"] == receipt["bytes"], f"artifact size mismatch: {path}")
        return path

    def receipt(self, path):
        self.hash(path)
        return {"path": str(Path(path).resolve()), **self.files[str(Path(path).resolve())]}

    def unchanged(self):
        for path, receipt in list(self.files.items()):
            self.hash(path)


def _inventory(root, *, python_only=False):
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and p.suffix != ".pyc"
            and (p.suffix == ".py" if python_only else p != root / "snapshot.json")}


def _bundle(reader, campaign_root):
    root = Path(campaign_root).resolve()
    experiment = root.parent
    campaign = reader.read(root / "campaign.json")
    _require((campaign.get("format"), campaign.get("schema_version")) ==
             ("transformer_rl.curriculum_campaign", 1), "unsupported curriculum campaign")
    manifest_path = _inside(experiment, campaign.get("manifest"))
    reader.checked(experiment, {"path": str(manifest_path), "sha256": campaign["manifest_file_sha256"]})
    manifest = reader.read(manifest_path)
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    _require((manifest.get("format"), manifest.get("schema_version")) ==
             ("transformer_rl.curriculum_study", 1) and digest(body) == manifest.get("sha256")
             == campaign.get("manifest_sha256"), "curriculum manifest identity differs")
    _require(reader.read(root / "manifest.json") == manifest, "campaign manifest copy differs")
    study = manifest_path.parent
    source = campaign["source"]
    source_root = _inside(experiment, source["root"])
    _require(digest(source["files"]) == source["sha256"], "learner source manifest SHA differs")
    for route, sha in source["files"].items():
        _require(route.startswith("src/transformer_rl/"), "source route is outside learner package")
        reader.checked(source_root, {"path": route, "sha256": sha})
    package = source_root / "src/transformer_rl"
    actual_source = {"src/transformer_rl/" + p for p in _inventory(package, python_only=True)}
    _require(actual_source == set(source["files"]), "learner source contains unsealed files")
    learner_files = {k.removeprefix("src/transformer_rl/"): v for k, v in source["files"].items()}
    learner = manifest["source_identity"]["learner_source"]
    _require(learner["files"] == learner_files and learner["sha256"] == digest(learner_files, ascii=True),
             "prepared learner and campaign source differ")
    for key, route in {"controller_sha256": "tools/run_frame_control_campaign.py",
                       "transfer_controller_sha256": "tools/run_transfer_campaign.py",
                       "curriculum_controller_sha256": "tools/run_curriculum_campaign.py",
                       "transfer_helper_sha256": "tools/run_transfer_campaign.py"}.items():
        reader.checked(source_root, {"path": route, "sha256": source[key]})
    snapshot = _inside(study, manifest["snapshot_identity"]["path"])
    reader.checked(snapshot, {"path": "snapshot.json",
        "sha256": manifest["snapshot_identity"]["receipt_sha256"]})
    snapshot_data = reader.read(snapshot / "snapshot.json")
    _require(digest(snapshot_data["files"]) == snapshot_data["sha256"] ==
             manifest["snapshot_identity"]["sha256"], "snapshot identity differs")
    for route, sha in snapshot_data["files"].items():
        reader.checked(snapshot, {"path": route, "sha256": sha})
    _require(_inventory(snapshot) == set(snapshot_data["files"]), "snapshot contains unsealed files")
    _require("parent_study.json" in snapshot_data["files"], "original skill gates are not sealed")
    parent = reader.read(snapshot / "parent_study.json")
    configs = {}
    _require(set(manifest["configs"]) == set(manifest["artifacts"]), "configuration coverage differs")
    for route, sha in manifest["configs"].items():
        artifact = manifest["artifacts"][route]
        reader.checked(study, artifact if isinstance(artifact, dict) else {"path": route, "sha256": artifact})
        cfg = reader.read(_inside(study, route))
        _require(digest(cfg) == sha, "configuration content SHA differs")
        env = cfg["environment"]
        _require(Path(env["snapshot"]).resolve() == snapshot and env["snapshot_sha256"] == snapshot_data["sha256"],
                 "configuration snapshot differs")
        reader.checked(snapshot, {"path": env["contract"], "sha256": env["contract_sha256"]})
        configs[route] = cfg
    return {"root": root, "experiment": experiment, "study": study, "snapshot": snapshot,
            "source_root": source_root, "campaign": campaign, "manifest": manifest,
            "manifest_path": manifest_path, "parent": parent, "configs": configs}


def _protocol_body(reader, bundle):
    m, parent, configs = bundle["manifest"], bundle["parent"], bundle["configs"]
    seeds = m["training_seeds"]
    _require(len(seeds) >= 3 and len(set(seeds)) == len(seeds)
             and all(_integer(s) and s < 2**32 for s in seeds), "at least three independent training seeds required")
    evaluation = {"seeds": [8701, 9701], "steps": 4001, "settle_steps": 200,
                  "min_steady_samples": 200, "trace_replicas": 8}
    _require(m["evaluation"] == evaluation and not set(seeds).intersection(evaluation["seeds"]),
             "paired evaluation protocol differs")
    _require(all(parent["evaluation"].get(k) == evaluation[k] for k in
                 ("steps", "settle_steps", "min_steady_samples"))
             and parent["evaluation"]["min_completed_episodes"] == 8, "original evaluation qualification differs")
    protocol = m["protocol"]
    _require(all(protocol.get(k) == v for k, v in {
        "warmup_updates": 400, "total_updates": 1200, "transitions_per_update": 49152,
        "planned_transitions_per_arm_seed": 58982400, "planned_total_transitions": 58982400 * 3 * len(seeds),
        "same_phase_boundary_reset_all_arms": True,
        "phase2_learning_state": "restore_model_optimizer_rng_and_update_counter",
        "phase2_clock": "cumulative_consumed_updates_with_zero_global_offset",
        "phase_evaluation_does_not_gate_progress": True, "task_dose_matched": False,
        "causal_scope": "task_schedule_and_exposure_intervention_not_order_only",
        "policy_hz": 100, "physics_feedback_pd_hz": 200}.items()), "curriculum budget or causal scope differs")
    _require(m["training"]["rollout_steps"] == 48, "rollout length differs")
    _require([a["name"] for a in m["arms"]] == list(ARMS), "curriculum arm coverage differs")
    reference = None
    for arm in m["arms"]:
        _require(len(arm["phases"]) == 2, "two phases required")
        for i, phase in enumerate(arm["phases"]):
            _require(all(phase.get(k) == v for k, v in {"name": f"phase{i + 1}",
                "domain": ARMS[arm["name"]][i], "updates": (400, 800)[i],
                "start_update": (0, 400)[i]}.items()), "phase identity or clock differs")
            cfg = configs[phase["config"]]
            invariant = {k: cfg[k] for k in ("model", "ppo", "control")}
            _require(cfg["environment"]["num_envs"] == 1024, "training environment count differs")
            reference = invariant if reference is None else reference
            _require(invariant == reference, "arms differ in network, PPO or control")
    policy = reference["model"]["policy"]
    _require(policy["architecture"] == "transformer" and policy["residual_type"] == "gated"
             and policy["history_length"] == 31 and reference["control"]["policy_dt_s"] == .01,
             "this protocol supports the frozen Gated H31 pilot only")
    names = [c["name"] for c in m["scenarios"]]
    _require(len(names) == 50 and set(names) == set(OLD_CASES + NEW_CASES), "old10/new40 scenario coverage differs")
    used_configs = {p["config"] for a in m["arms"] for p in a["phases"]} | {c["config"] for c in m["scenarios"]}
    _require(len(used_configs) == 56 and used_configs == set(configs), "configuration coverage must be exactly six phases and fifty cases")
    original = {c["name"]: c for c in parent["scenarios"]}
    _require(len(original) == 50 and set(original) == set(names), "parent scenario coverage differs")
    gates = {}
    for case in m["scenarios"]:
        cfg, prior = configs[case["config"]], original[case["name"]]
        _require(cfg["model"] == reference["model"] and cfg["control"] == reference["control"],
                 "evaluation policy/control differs")
        env = {k: v for k, v in cfg["environment"].items() if k not in ("snapshot", "snapshot_sha256")}
        _require(env == prior["environment"] and env["num_envs"] == 8, "evaluation case differs from original")
        seen = set()
        for gate in prior["gates"]:
            _require(set(gate) == {"path", "operator", "value"} and gate["operator"] in ("min", "max")
                     and _number(gate["value"]) and gate["path"] not in seen, "invalid original gate")
            seen.add(gate["path"])
        required = {"success_rate", *[v for k, v in METRICS.items() if k != "drift_max_m"]}
        if case["name"] in OLD_CASES:
            required.add(METRICS["drift_max_m"])
        _require(required.issubset(seen), "original gates do not establish physical skill acquisition")
        _require(type(prior.get("require_steady")) is bool, "original steady qualification is missing")
        gates[case["name"]] = {"gates": prior["gates"], "require_steady": prior["require_steady"]}
    return {"format": "transformer_rl.curriculum_retention_protocol", "schema_version": 1,
        "campaign_root": str(bundle["root"]),
        "campaign": reader.receipt(bundle["root"] / "campaign.json"),
        "manifest": reader.receipt(bundle["manifest_path"]),
        "parent_study": reader.receipt(bundle["snapshot"] / "parent_study.json"),
        "source_sha256": bundle["campaign"]["source"]["sha256"],
        "snapshot_sha256": m["snapshot_identity"]["sha256"],
        "architecture": "transformer_gated", "model": reference["model"],
        "training_seeds": seeds, "evaluation": evaluation, "old_cases": OLD_CASES, "new_cases": NEW_CASES,
        "gates": gates, "min_completed_episodes": 8, "metrics": METRICS,
        "weighting": "equal evaluation seeds within case; equal cases within declared set; equal training seeds",
        "acquisition": "case passes original gates on both phase1 evaluations",
        "retention": "same acquired case passes both phase2 evaluations; unknown post-evaluation makes rate null",
        "missing": "unknown, never failed or removed from acquired denominator",
        "skill_scope": "10 old perturbation cases of one standing skill; case fraction is not a ten-skill capacity",
        "causal_scope": protocol["causal_scope"], "formal_architecture_selection": False}


def prepare(campaign_root):
    reader = Reader()
    body = _protocol_body(reader, _bundle(reader, campaign_root))
    protocol = {**body, "prepared_at": datetime.now(timezone.utc).isoformat()}
    protocol["sha256"] = digest(protocol)
    reader.unchanged()
    return protocol


def _value(report, route):
    for key in route.split("."):
        if not isinstance(report, dict) or key not in report:
            return None
        report = report[key]
    return report if _number(report) else None


def _training(reader, bundle, arm, phase, seed, item, parent):
    root, m = bundle["root"], bundle["manifest"]
    training = item.get("training", {})
    if training.get("status") != "completed":
        raise FileNotFoundError("training endpoint not completed")
    cursor, prior, initial = phase["start_update"], parent, None
    cfg = bundle["configs"][phase["config"]]
    for index, artifact in enumerate(training["attempts"]):
        receipt_path = reader.checked(root, artifact)
        expected_job = root / "jobs" / arm / f"seed_{seed}" / phase["name"] / "training"
        _require(receipt_path.parent.parent == expected_job and receipt_path.name == "receipt.json"
                 and re.fullmatch(r"attempt_[0-9]+", receipt_path.parent.name), "training receipt belongs to another job")
        receipt = reader.read(receipt_path)
        request_path = receipt_path.parent / "request.json"
        reader.checked(root, {"path": str(request_path), "sha256": receipt["request_sha256"]})
        request = reader.read(request_path)
        expected = {"manifest_sha256": m["sha256"], "arm": arm, "phase": phase["name"],
            "training_seed": seed, "config_sha256": m["configs"][phase["config"]],
            "parent_checkpoint_sha256": prior["checkpoint_sha256"] if prior else None,
            "consumed_update_offset": cursor}
        _require(request["identity"] == receipt["identity"] == expected and request["config"] == cfg,
                 "training request identity/configuration differs")
        initializer = "restore_learning_from" if phase["start_update"] > 0 and index == 0 else "resume"
        initialization = {initializer: prior["checkpoint"]} if prior else {}
        _require(request["initialization"] == initialization and request["start_update"] == cursor
                 and request["updates"] == phase["start_update"] + phase["updates"] - cursor
                 and request["prior_transitions"] == cursor * 49152
                 and request["batch_samples_per_update"] == 49152, "training parent or sample clock differs")
        artifacts = receipt["artifacts"]
        files = {k: reader.checked(root, artifacts[k]) for k in ("completion", "metrics", "run", "checkpoint")}
        completion, run = reader.read(files["completion"]), reader.read(files["run"])
        count = completion["completed_updates"]
        _require(_integer(count, 1) and count <= request["updates"] and completion["attempted_updates"] == count
                 and completion["start_update"] == cursor and completion["final_update"] == cursor + count
                 and completion["config_sha256"] == expected["config_sha256"]
                 and completion["consumed_transitions"] == count * 49152
                 and completion["cumulative_transitions"] == (cursor + count) * 49152,
                 "training completion or unique sample budget differs")
        _require(completion["status"] == ("completed" if count == request["updates"] else "stopped")
                 and (completion["status"] == "completed" or completion.get("stop_reason") in
                      ("time_budget", "SIGINT", "SIGTERM")), "invalid sealed training endpoint")
        _require(run["seed"] == seed and run["config"] == cfg and all(run.get(k) == initialization.get(k)
                 for k in ("resume", "restore_learning_from", "initialize_from")), "run initialization differs")
        _require(all(run.get(k) == v for k, v in {
            "environment_factory": m["environment_factory"], "rollout_steps": 48,
            "source": m["source_identity"]["learner_source"], "retention_coef": 0.,
            "episode_state_restored": False, "history_reset": "repeat_first",
            "initial_model_hash_format": "sorted_named_tensor_contents_v1",
            "updates": request["updates"], "max_seconds": m["training"]["max_seconds"],
            "checkpoint_interval": m["training"]["checkpoint_interval"],
            "device": bundle["campaign"]["device"]}.items()), "run source, recipe or reset identity differs")
        if index == 0:
            initial = run.get("initial_model_sha256")
            _require(isinstance(initial, str) and SHA.fullmatch(initial), "initial model identity missing")
        rows = [_json(line) for line in reader.raw(files["metrics"]).splitlines()]
        _require(len(rows) == count, "training log has missing or extra updates")
        optimization_samples, optimizer_steps = 0, 0
        for offset, row in enumerate(rows, 1):
            opt = row["optimization"]
            _require(_integer(row["update"], 1) and row["update"] == cursor + offset
                     and _integer(row["batch_samples"], 1) and row["batch_samples"] == 49152
                     and _integer(opt["optimizer_steps"], 1) and _integer(opt["sample_count"], 1),
                     "invalid applied PPO record")
            optimization_samples += opt["sample_count"]
            optimizer_steps += opt["optimizer_steps"]
        checkpoint = {"checkpoint": str(files["checkpoint"]), "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
            "update": cursor + count, "cumulative_transitions": (cursor + count) * 49152}
        sidecar = reader.read(_inside(root, str(files["checkpoint"]) + ".json"))
        _require((sidecar.get("format"), sidecar.get("schema_version")) == ("transformer_rl.packed_checkpoint", 1)
                 and sidecar["sha256"] == checkpoint["checkpoint_sha256"] and sidecar["config"] == cfg
                 and sidecar["update"] == checkpoint["update"] and all(sidecar["metadata"].get(k) == v for k, v in {
                     "source": m["source_identity"]["learner_source"], "seed": seed,
                     "environment_factory": m["environment_factory"], "retention_coef": 0., "anchors": [],
                     "episode_state_restored": False, "initial_model_sha256": run["initial_model_sha256"],
                     "initial_model_hash_format": "sorted_named_tensor_contents_v1",
                     "collected_transitions": checkpoint["cumulative_transitions"]}.items()),
                 "checkpoint sidecar source, configuration or sample identity differs")
        _require(_inside(root, completion["checkpoint"]) == files["checkpoint"]
                 and completion["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                 and receipt["checkpoint"] == checkpoint
                 and receipt["status"] == ("completed" if count == request["updates"] else "resumable")
                 and receipt["consumed_updates"] == count and receipt["consumed_transitions"] == count * 49152
                 and receipt["cumulative_transitions"] == checkpoint["cumulative_transitions"]
                 and receipt["initial_model_sha256"] == run["initial_model_sha256"]
                 and all(receipt["evidence"].get(k) == v for k, v in {
                     "updates": count, "last_update": cursor + count, "optimizer_steps": optimizer_steps,
                     "batch_samples": count * 49152, "ppo_verified": True,
                     "metrics_sha256": artifacts["metrics"]["sha256"],
                     "optimization_samples": optimization_samples}.items()),
                 "sealed training receipt differs")
        cursor, prior = cursor + count, checkpoint
    _require(cursor == phase["start_update"] + phase["updates"] and training["checkpoint"] == prior
             and training["consumed_updates"] == phase["updates"]
             and training["initial_model_sha256"] == initial, "phase endpoint differs from sealed attempts")
    return prior, initial


def _evaluation(reader, bundle, protocol, arm, phase, seed, eval_seed, checkpoint, receipt):
    if receipt.get("status") != "completed":
        raise FileNotFoundError("evaluation not completed")
    root, m = bundle["root"], bundle["manifest"]
    _require(receipt["identity"] == {"manifest_sha256": m["sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "evaluation_seed": eval_seed,
        "protocol": m["evaluation"]}, "evaluation receipt identity differs")
    artifacts = receipt["artifacts"]
    _require(set(artifacts) == set(OLD_CASES + NEW_CASES + ["control", "trace"]), "evaluation artifact coverage differs")
    paths = {k: _inside(root, v["path"]) for k, v in artifacts.items()}
    for name in ("control", "trace"):
        reader.checked(root, artifacts[name])
    directory = paths["control"].parent
    expected_job = root / "control" / arm / f"train_{seed}" / phase["name"] / f"seed_{eval_seed}"
    _require(directory.parent == expected_job and re.fullmatch(r"attempt_[0-9]+", directory.name)
             and all(p.parent == directory for p in paths.values()), "evaluation belongs to another job or attempt")
    _require(reader.read(directory / "receipt.json") == receipt, "summary and sealed evaluation receipt differ")
    control = reader.read(paths["control"])
    _require((control.get("format"), control.get("schema_version")) == ("transformer_rl.control_evaluation", 1)
             and control["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
             and control["checkpoint_update"] == checkpoint["update"] and control["seed"] == eval_seed
             and control["steps"] == 4001 and set(control["groups"]) == set(OLD_CASES + NEW_CASES),
             "control suite identity or scenario coverage differs")
    trace = control["trace"]
    _require(trace["sha256"] == artifacts["trace"]["sha256"] and _inside(root, trace["path"]) == paths["trace"]
             and trace["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
             and trace["checkpoint_update"] == checkpoint["update"] and trace["seed"] == eval_seed
             and trace["steps"] == 4001 and trace["rows"] == trace["row_indices"] == list(range(400))
             and Counter(trace["group_labels"]) == Counter({c: 8 for c in OLD_CASES + NEW_CASES}),
             "trace provenance differs")
    return paths, control


def _cell(reader, bundle, protocol, case, eval_seed, checkpoint, path, control):
    report = reader.read(path)
    entry = next(c for c in bundle["manifest"]["scenarios"] if c["name"] == case)
    cfg = bundle["configs"][entry["config"]]
    _require((report.get("format"), report.get("schema_version")) == ("transformer_rl.packed_evaluation", 1)
             and report["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
             and report["checkpoint_update"] == checkpoint["update"] and report["seed"] == eval_seed
             and report["model"] == cfg["model"] and report["control_sha256"] == digest(cfg["control"])
             and report["environment"] == cfg["environment"] and report["steps"] == 4001
             and report["num_envs"] == 8 and report["transitions"] == 32008,
             "case report policy, configuration, sample or checkpoint identity differs")
    for k in ("completed_episodes", "failed_episodes"):
        _require(_integer(report.get(k)), "episode counts missing or invalid")
    _require(report["failed_episodes"] <= report["completed_episodes"], "failure count exceeds completed episodes")
    quality = report.get("control")
    if not isinstance(quality, dict):
        raise FileNotFoundError("case control metrics missing")
    _require(quality == control["groups"][case], "case and suite control metrics differ")
    _require(quality["full_interval"]["samples"] == quality["actuation"]["sample_count"] == 32008,
             "control sample coverage differs")
    _require(all(quality["protocol"].get(k) == v for k, v in {"policy_dt_s": .01,
        "settle_steps": 200, "min_steady_samples": 200, "tracking_tolerance": [.15, .25, .03]}.items()),
             "case control protocol differs")
    _require(all(report["stability"]["protocol"].get(k) == v for k, v in
                 {"settle_steps": 200, "min_steady_samples": 200}.items()), "stability protocol differs")
    names = [k for k in METRICS if k != "drift_max_m" or case in OLD_CASES]
    values = {k: _value(report, METRICS[k]) for k in names}
    gates = protocol["gates"][case]
    gate_values = {g["path"]: _value(report, g["path"]) for g in gates["gates"]}
    if any(v is None for v in (*values.values(), *gate_values.values())):
        raise FileNotFoundError("required finite physical gate/metric unavailable")
    for route in {METRICS[k].split(".")[1] for k in names}:
        _require(report["metrics"][route]["count"] == 32008, "physical metric sample coverage differs")
    _require(all(v >= 0 for v in values.values()), "absolute physical errors cannot be negative")
    _require(0 <= gate_values["success_rate"] <= 1, "success rate outside [0,1]")
    reasons = [g["path"] for g in gates["gates"] if (gate_values[g["path"]] < g["value"]
               if g["operator"] == "min" else gate_values[g["path"]] > g["value"])]
    if report["completed_episodes"] < protocol["min_completed_episodes"]:
        reasons.append("insufficient_completed_episodes")
    if gates["require_steady"] and not report["stability"].get("available"):
        reasons.append("insufficient_steady_samples")
    return {"status": "ready", "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_update": checkpoint["update"], "metrics": values, "passed": not reasons,
        "gate_reasons": reasons, "gate_values": gate_values,
        "completed_episodes": report["completed_episodes"], "failed_episodes": report["failed_episodes"],
        "steady_available": report["stability"].get("available"), "report": reader.receipt(path)}


def _mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def _changes(protocol, cells):
    index = {(c["arm"], c["training_seed"], c["phase"], c["evaluation_seed"], c["case"]): c for c in cells}
    changes, results = [], []
    for seed in protocol["training_seeds"]:
        for case in OLD_CASES + NEW_CASES:
            arm_values, acquired, retained = {}, {}, {}
            metrics = [k for k in METRICS if k != "drift_max_m" or case in OLD_CASES]
            for arm in ARMS:
                before = [index[(arm, seed, "phase1", e, case)] for e in protocol["evaluation"]["seeds"]]
                after = [index[(arm, seed, "phase2", e, case)] for e in protocol["evaluation"]["seeds"]]
                acquired[arm] = all(c["passed"] for c in before) if all(c["status"] == "ready" for c in before) else None
                retained[arm] = all(c["passed"] for c in after) if all(c["status"] == "ready" for c in after) else None
                arm_values[arm] = {k: _mean([b["metrics"][k] - a["metrics"][k]
                    if a["status"] == b["status"] == "ready" else None for a, b in zip(before, after)]) for k in metrics}
            did = {k: arm_values["pretrain"][k] - arm_values["stationary"][k]
                   if arm_values["pretrain"][k] is not None and arm_values["stationary"][k] is not None else None for k in metrics}
            eligible = acquired["pretrain"] is True and retained["pretrain"] is not None
            qualified = case in OLD_CASES and eligible
            forgetting = arm_values["pretrain"] if qualified else None
            qualified_did = did if qualified and acquired["stationary"] is True and retained["stationary"] is not None else None
            changes.append({"training_seed": seed, "case": case, "role": "old" if case in OLD_CASES else "new",
                "arm_error_changes": arm_values, "phase1_passed": acquired, "phase2_passed": retained,
                "old_error_increase": arm_values["pretrain"] if case in OLD_CASES else None,
                "new_error_reduction": {k: -v if v is not None else None for k, v in arm_values["pretrain"].items()}
                    if case in NEW_CASES else None,
                "pretrain_minus_stationary_error_change": did,
                "new_pretrain_minus_stationary_gain": {k: -v if v is not None else None for k, v in did.items()}
                    if case in NEW_CASES else None,
                "qualified_forgetting": forgetting,
                "positive_forgetting": {k: max(0., v) for k, v in forgetting.items()} if forgetting is not None else None,
                "qualified_forgetting_difference": qualified_did,
                "retained": retained["pretrain"] if qualified else None,
                "qualification_reason": None if qualified else ("new_case" if case in NEW_CASES else
                    "old_task_not_acquired" if acquired["pretrain"] is False else "missing_matched_evidence")})
        seed_changes = [c for c in changes if c["training_seed"] == seed]
        old = [c for c in seed_changes if c["role"] == "old"]
        new = [c for c in seed_changes if c["role"] == "new"]
        acquired_cases = [c for c in old if c["phase1_passed"]["pretrain"] is True]
        unknown_reference = [c["case"] for c in old if c["phase1_passed"]["pretrain"] is None]
        unknown_post = [c["case"] for c in acquired_cases if c["phase2_passed"]["pretrain"] is None]
        rate = sum(c["phase2_passed"]["pretrain"] is True for c in acquired_cases) / len(acquired_cases) \
            if acquired_cases and not unknown_reference and not unknown_post else None
        results.append({"training_seed": seed, "acquired_old_cases": [c["case"] for c in acquired_cases],
            "acquired_old_case_count": len(acquired_cases), "retained_case_fraction": rate,
            "known_retained_case_count": sum(c["phase2_passed"]["pretrain"] is True for c in acquired_cases),
            "known_lost_case_count": sum(c["phase2_passed"]["pretrain"] is False for c in acquired_cases),
            "unknown_reference_cases": unknown_reference, "unknown_post_cases": unknown_post,
            "retention_reason": "missing_matched_evidence" if unknown_reference or unknown_post else
                "old_task_not_acquired" if not acquired_cases else None,
            "old_error_increase": {k: _mean([c["old_error_increase"][k] for c in old]) for k in METRICS},
            "qualified_old_forgetting": {k: _mean([c["qualified_forgetting"][k] if c["qualified_forgetting"] else None
                for c in acquired_cases]) for k in METRICS} if acquired_cases and not unknown_reference else None,
            "old_pretrain_minus_stationary_change": {k: _mean([c["pretrain_minus_stationary_error_change"][k] for c in old]) for k in METRICS},
            "new_error_reduction": {k: _mean([c["new_error_reduction"][k] for c in new]) for k in METRICS if k != "drift_max_m"},
            "new_pretrain_minus_stationary_gain": {k: _mean([c["new_pretrain_minus_stationary_gain"][k] for c in new])
                                                  for k in METRICS if k != "drift_max_m"},
            "newly_acquired_cases": [c["case"] for c in new if c["phase1_passed"]["pretrain"] is False
                                     and c["phase2_passed"]["pretrain"] is True]})
    return changes, results


def _aggregate(results):
    def summarize(values):
        valid = all(v is not None for v in values.values())
        return {"training_seed_values": values, "available": valid, "n_training_seeds": len(values),
            "mean": statistics.mean(values.values()) if valid else None,
            "sample_std": statistics.stdev(values.values()) if valid and len(values) > 1 else None}
    aggregate = {"retained_case_fraction": summarize({str(r["training_seed"]): r["retained_case_fraction"] for r in results})}
    for group in ("old_error_increase", "old_pretrain_minus_stationary_change", "new_error_reduction",
                  "new_pretrain_minus_stationary_gain"):
        aggregate[group] = {k: summarize({str(r["training_seed"]): r[group][k] for r in results}) for k in results[0][group]}
    return aggregate


def analyze(protocol, *, analyzer_source=None):
    reader = Reader()
    _require(isinstance(protocol, dict) and digest({k: v for k, v in protocol.items() if k != "sha256"}) ==
             protocol.get("sha256"), "analysis protocol SHA differs")
    bundle = _bundle(reader, protocol["campaign_root"])
    expected = _protocol_body(reader, bundle)
    _require({k: v for k, v in protocol.items() if k not in ("prepared_at", "sha256")} == expected,
             "analysis protocol changed its frozen gates, cases, weights or inputs")
    summary_path = bundle["root"] / "summary.json"
    try:
        summary = reader.read(summary_path)
    except FileNotFoundError:
        summary = {"status": "not_started", "results": {}}
    else:
        _require(summary.get("campaign_sha256") == digest(bundle["campaign"]), "summary belongs to another campaign")
    _require(not summary.get("formal_architecture_selection", False), "Gated pilot cannot select all architectures")
    cells, initial = [], {}
    known_keys = {f"{a}/seed_{s}" for a in ARMS for s in protocol["training_seeds"]}
    _require(set(summary["results"]).issubset(known_keys), "unexpected curriculum result job")
    for seed in protocol["training_seeds"]:
        for arm_entry in bundle["manifest"]["arms"]:
            arm = arm_entry["name"]
            result = summary["results"].get(f"{arm}/seed_{seed}", {})
            if result:
                _require(result.get("arm") == arm and result.get("training_seed") == seed, "summary job identity differs")
            items = {p["name"]: p for p in result.get("phases", [])}
            _require(len(items) == len(result.get("phases", [])) and set(items).issubset({"phase1", "phase2"}),
                     "duplicate or unknown phase")
            parent = None
            for phase in arm_entry["phases"]:
                item, endpoint, reason = items.get(phase["name"], {}), None, None
                try:
                    if phase["start_update"] and parent is None:
                        raise FileNotFoundError("verified phase1 parent unavailable")
                    endpoint, model = _training(reader, bundle, arm, phase, seed, item, parent)
                    if phase["start_update"] == 0:
                        _require(initial.setdefault(str(seed), model) == model, "paired arms initialized different models")
                    parent = endpoint
                except FileNotFoundError as error:
                    reason = str(error)
                for eval_seed in protocol["evaluation"]["seeds"]:
                    paths, control, eval_reason = None, None, reason
                    if endpoint is not None:
                        try:
                            paths, control = _evaluation(reader, bundle, protocol, arm, phase, seed, eval_seed,
                                endpoint, item.get("evaluations", {}).get(str(eval_seed), {}))
                        except FileNotFoundError as error:
                            eval_reason = str(error)
                    for case in OLD_CASES + NEW_CASES:
                        cell = {"arm": arm, "training_seed": seed, "phase": phase["name"],
                                "evaluation_seed": eval_seed, "case": case}
                        try:
                            if paths is None:
                                raise FileNotFoundError(eval_reason or "matched evaluation unavailable")
                            receipt = item["evaluations"][str(eval_seed)]
                            reader.checked(bundle["root"], receipt["artifacts"][case])
                            cell.update(_cell(reader, bundle, protocol, case, eval_seed, endpoint, paths[case], control))
                        except FileNotFoundError as error:
                            cell.update(status="not_ready", passed=None, metrics=None, reason=str(error))
                        cells.append(cell)
    changes, results = _changes(protocol, cells)
    unready = [{k: c[k] for k in ("arm", "training_seed", "phase", "evaluation_seed", "case", "reason")}
               for c in cells if c["status"] != "ready"]
    reader.unchanged()
    return {"format": "transformer_rl.curriculum_retention_analysis", "schema_version": 1,
        "status": "not_ready" if unready else "complete", "campaign_status": summary["status"],
        "captured_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol["sha256"],
        "analyzer_sha256": hashlib.sha256(analyzer_source if analyzer_source is not None else Path(__file__).read_bytes()).hexdigest(),
        "architecture": "transformer_gated", "formal_architecture_selection": False,
        "expected_cells": 3 * len(protocol["training_seeds"]) * 2 * 2 * 50,
        "ready_cells": len(cells) - len(unready), "cells": cells, "paired_changes": changes,
        "training_seed_results": results, "aggregate": _aggregate(results), "unready": unready,
        "initial_models": initial, "input_receipts": reader.files,
        "forgetting": {"available": any(c["qualified_forgetting"] is not None for c in changes),
            "qualified_old_case_count": sum(c["qualified_forgetting"] is not None for c in changes),
            "scope": "availability for explicitly qualified case/training-seed pairs; see null aggregate and missing cells"},
        "interpretation": {"error_change": "after minus before; positive is worse, in original physical units",
            "new_gain": "before minus after; positive is better", "controlled_change": "pretrain change minus stationary change",
            "qualification": "only actually acquired old cases support forgetting and retention",
            "weighting": protocol["weighting"], "independent_unit": "training seed; evaluation repetitions are not new trained policies",
            "causal_scope": protocol["causal_scope"], "case_fraction": protocol["skill_scope"],
            "trace": "SHA and JSON provenance verified; NPZ arrays are not used for these report-based differences",
            "unlearned": "raw error differences remain descriptive; qualified forgetting and retention are null"}}


def write_external(path, value, campaign_root):
    path = Path(path).resolve()
    _require(not path.is_relative_to(Path(campaign_root).resolve().parent),
             "output must be outside the read-only experiment")
    if path.exists():
        raise FileExistsError(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare_parser = sub.add_parser("prepare", help="freeze a separate analysis protocol, without training")
    prepare_parser.add_argument("--campaign-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze", help="read sealed curriculum evidence")
    analyze_parser.add_argument("--protocol", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        value, root = prepare(args.campaign_root), args.campaign_root
    else:
        raw_protocol = args.protocol.read_bytes()
        protocol = _json(raw_protocol)
        value, root = analyze(protocol), protocol["campaign_root"]
        _require(args.protocol.read_bytes() == raw_protocol, "protocol file changed while analyzing")
        value["protocol_file"] = {"path": str(args.protocol.resolve()),
            "sha256": hashlib.sha256(raw_protocol).hexdigest()}
    write_external(args.output, value, root)
    print(json.dumps({"output": str(args.output.resolve()), "status": value.get("status", "protocol_frozen"),
                      "ready_cells": value.get("ready_cells"), "formal_architecture_selection": False}))


if __name__ == "__main__":
    main()

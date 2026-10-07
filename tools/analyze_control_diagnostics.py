#!/usr/bin/env python3
"""Analyze the sealed paired control diagnostics without importing a learner.

Inputs are JSON and ordinary file SHA256 checks. Traces and checkpoints are
never deserialized; trace payloads are not read. Missing fixed cells stay null.
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


MANIFEST_SHA256 = "eb8541267bb8c92460867492ebbf02068a254cfda572b2bee5f71b32f3e8bbf6"
SOURCE_COMMIT = "9b576e7d8f72e90ad361df83139bc2bb78e05165"
SOURCE_SHA256 = "45fce95d3abae0291d04d325baa3b759eb8c4ab1d41be2d6c145f5d6c5677c1a"
CONTROLLERS = {
    "run_frame_diagnostic_campaign.py": "eca93a7aea6c312cad0ee8dde37c35f7277e38003cc8a0028f51e5efedbb2ec0",
    "run_transfer_campaign.py": "d9f887c7e2db39e40b71c4fbf951e25edcf5cea9aef1c0b5b72bc92c346f0465",
    "run_frame_control_campaign.py": "6ac98583ec65be72416f828eaa48f847308404311ead103526373459f3effc9d",
}
VARIANTS = ("mlp", "mlp_medium", "history_mlp", "history_mlp_wide", "transformer_small",
            "transformer", "transformer_query", "transformer_gated", "transformer_large", "transformer_xlarge")
PROFILES = ("nominal", "delay20_15", "delay40_30", "delay80_60", "noise", "payload_com", "low_grip", "combined")
TASKS = ("stand_305mm", "forward_05", "backward_05", "rotate_1", "start_stop_05", "height_scan")
CASES = tuple(f"{task}__{profile}" for task in TASKS for profile in PROFILES) + (
    "stand_305mm__motor_weak", "stand_305mm__spring_weak")
SEEDS = (8701, 9701)
PROTOCOL = {"checkpoint_update": 1200, "steps": 4001, "seeds": list(SEEDS),
    "settle_steps": 200, "min_steady_samples": 200, "cases": 50, "case_replicas": 8,
    "trace_replicas": 2, "policy_dt_s": .01,
    "seed_role": "paired_diagnostic_retest_of_existing_control_seeds", "independent_holdout": False}
SHA = re.compile(r"[0-9a-f]{64}")
PLANAR_SOURCE = "consecutive PRE-reset world position_xy divided by actual within-episode time_s difference"
PLANAR_SCOPE = "all declared environment rows; no trace subsampling; first isolated sample has no velocity interval"
PLANAR_WEIGHTING = "pooled observed duration; report scenarios and training seeds separately"
STEADY_ELIGIBILITY = "same unchanged-reference segments and retained-sample threshold as control.steady; includes eligible failed and final partial segments"
SCALED_SEMANTICS = "nominal PD request and envelope scaled together by motor_strength * schedule.motor_scale; not final physical wheel output limits"
SCALED_SCOPE = "all PRE-reset policy-rate physical samples, including transients, failed and final partial episodes; no steady filtering"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parse_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value
    def constant(text):
        raise ValueError(f"nonfinite JSON constant: {text}")
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def inside(root, route):
    require(isinstance(route, str) and bool(route), "nonempty artifact path required")
    root, path = Path(root).resolve(), Path(route)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path != root and path.is_relative_to(root), "artifact path escapes its declared root")
    return path


class Reader:
    def __init__(self):
        self.files = {}

    def _remember(self, path, sha, size):
        receipt = {"sha256": sha, "bytes": size}
        old = self.files.setdefault(str(path), receipt)
        require(old == receipt, f"input changed during analysis: {path}")
        return receipt

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        self._remember(path, hashlib.sha256(raw).hexdigest(), len(raw))
        return parse_json(raw)

    def hash(self, path):
        path = Path(path).resolve()
        require(path.suffix != ".npz", "trace payloads must not be read by this analyzer")
        sha, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        return self._remember(path, sha.hexdigest(), size)

    def checked(self, root, receipt):
        require(isinstance(receipt, dict) and isinstance(receipt.get("sha256"), str)
                and SHA.fullmatch(receipt["sha256"]), "artifact requires lowercase SHA256")
        path = inside(root, receipt.get("path"))
        actual = self.hash(path)
        require(actual["sha256"] == receipt["sha256"], f"artifact SHA mismatch: {path}")
        return path

    def absolute(self, receipt):
        path = Path(receipt.get("path", ""))
        require(path.is_absolute(), "sealed input path must be absolute")
        return self.checked(path.parent, receipt)

    def unchanged(self):
        for path in list(self.files):
            self.hash(path)


def inventory(root, *, source=False):
    root = Path(root)
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"
            and (path.suffix == ".py" if source else path.name != "snapshot.json")}


def load_manifest(reader, manifest_path):
    path = Path(manifest_path).resolve()
    manifest = reader.read(path)
    require(manifest.get("format") == "transformer_rl.frame_diagnostic_campaign"
            and manifest.get("schema_version") == 1, "unsupported diagnostic manifest")
    require(digest({k: v for k, v in manifest.items() if k != "sha256"}) == manifest.get("sha256")
            == MANIFEST_SHA256, "manifest differs from the independently pinned campaign")
    root = Path(manifest["output_root"]).resolve()
    require(path == root / "manifest.json" and manifest["protocol"] == PROTOCOL,
            "sealed output root or diagnostic protocol differs")
    require(manifest.get("formal_architecture_selection") is False
            and manifest.get("hardware_deployment_ready") is False, "diagnostic scope differs")
    controls = manifest["controllers"]
    require(len(controls) == 3 and {Path(p).name: s for p, s in controls.items()} == CONTROLLERS,
            "controller or helper differs from independently pinned bytes")
    for route, sha in controls.items():
        reader.absolute({"path": route, "sha256": sha})
    source = manifest["source"]
    require(manifest["source_commit"] == source["git_head"] == SOURCE_COMMIT
            and digest(source["files"]) == source["sha256"] == SOURCE_SHA256, "sealed diagnostic source differs")
    source_root = Path(manifest["source_root"]).resolve()
    require(all(route.startswith("src/transformer_rl/") for route in source["files"]), "source routes differ")
    for route, sha in source["files"].items():
        reader.checked(source_root, {"path": route, "sha256": sha})
    actual = {"src/transformer_rl/" + route for route in inventory(source_root / "src/transformer_rl", source=True)}
    require(actual == set(source["files"]), "diagnostic source has unsealed Python files")
    if manifest["source_origin"] is not None:
        origin = reader.read(reader.absolute(manifest["source_origin"]))
        require(all(origin.get(k) == source[k] for k in ("files", "sha256", "git_head")), "source origin receipt differs")
    inputs = manifest["inputs"]
    require(inputs["variants"] == list(VARIANTS) and inputs["cases"] == list(CASES)
            and type(inputs["training_seed"]) is int and inputs["training_seed"] == 1101,
            "all ten variants, fifty cases and training seed 1101 are required")
    require(set(inputs["checkpoints"]) == set(VARIANTS)
            and set(inputs["environments"]) == set(VARIANTS), "checkpoint or environment coverage differs")
    study = Path(inputs["study_root"]).resolve()
    plan = reader.read(reader.absolute(inputs["plan"]))
    require(digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"] == inputs["plan_sha256"],
            "parent plan identity differs")
    require([v["name"] for v in plan["spec"]["variants"]] == list(VARIANTS)
            and plan["spec"]["seeds"] == [1101] and len(plan["spec"]["stages"]) == 1
            and plan["spec"]["stages"][0]["updates"] == 1200
            and plan["spec"]["stages"][0]["scenarios"] == list(CASES), "parent plan coverage differs")
    require(set(inputs["configs"]) == set(plan["configs"]), "parent configuration inventory differs")
    configs = {}
    for route, receipt in inputs["configs"].items():
        actual_path = reader.checked(study, receipt)
        require(actual_path == inside(study, route), "configuration route differs")
        value = reader.read(actual_path)
        require(digest(value) == plan["configs"][route], "parent configuration content differs")
        configs[route] = value
    require(len(inputs["snapshots"]) == 1, "one frozen environment snapshot required")
    for route, snapshot in inputs["snapshots"].items():
        snapshot_root = Path(route).resolve()
        require(str(snapshot_root) == snapshot["root"], "snapshot route differs")
        data = reader.read(reader.checked(snapshot_root, snapshot["receipt"]))
        require(digest(data["files"]) == data["sha256"] == snapshot["sha256"], "snapshot inventory identity differs")
        for item, sha in data["files"].items():
            reader.checked(snapshot_root, {"path": item, "sha256": sha})
        require(inventory(snapshot_root) == set(data["files"]), "snapshot contains unsealed files")
    snapshot_sha = next(iter(inputs["snapshots"].values()))["sha256"]
    for variant in VARIANTS:
        checkpoint = inputs["checkpoints"][variant]
        require(checkpoint["update"] == 1200, "parent checkpoint update differs")
        checkpoint_path = reader.checked(study, {"path": checkpoint["checkpoint"], "sha256": checkpoint["checkpoint_sha256"]})
        sidecar = reader.read(reader.checked(study, checkpoint["sidecar"]))
        completion = reader.read(reader.checked(study, checkpoint["completion"]))
        state = reader.read(reader.checked(study, checkpoint["training_state"]))
        require(sidecar["update"] == 1200 and sidecar["sha256"] == checkpoint["checkpoint_sha256"]
                and sidecar["metadata"]["seed"] == 1101
                and sidecar["metadata"]["environment_provenance"]["identity"] == snapshot_sha
                and digest(sidecar["config"]["control"]) == checkpoint["control_sha256"], "parent checkpoint sidecar differs")
        require(completion["status"] == "completed" and completion["final_update"] == 1200
                and completion["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                and Path(completion["checkpoint"]).resolve() == checkpoint_path, "parent completion differs")
        require(state["seed"] == 1101 and state["variant"] == variant
                and state["plan_sha256"] == inputs["plan_sha256"], "parent training ledger identity differs")
        require(set(inputs["environments"][variant]) == set(CASES), "case environment inventory differs")
        for case in CASES:
            cfg = configs[f"configs/{variant}.eval.{case}.json"]
            environment = cfg["environment"]
            require(environment == inputs["environments"][variant][case]
                    and environment["num_envs"] == 8 and environment["snapshot_sha256"] == snapshot_sha
                    and Path(environment["snapshot"]).resolve() == snapshot_root
                    and cfg["control"] == sidecar["config"]["control"] and cfg["control"]["policy_dt_s"] == .01,
                    "case environment or policy contract differs")
            contract = reader.read(reader.checked(snapshot_root, {"path": environment["contract"],
                "sha256": environment["contract_sha256"]}))
            require(contract.get("evaluation_exact_cases") is True and contract["target_num_envs"] == 8
                    and [v["name"] for v in contract["evaluation"]["cases"]] == [case], "case contract does not declare eight replicas")
    for dependency in manifest["dependencies"].values():
        definition = reader.read(reader.absolute(dependency["campaign"]))
        require(digest(definition) == dependency["campaign_sha256"], "dependency campaign identity differs")
    return manifest


def integer(value, maximum=None):
    require(type(value) is int and value >= 0 and (maximum is None or value <= maximum), "invalid diagnostic sample count")


def finite(value, *, nonnegative=False, fraction=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0) and (not fraction or 0 <= value <= 1), "invalid finite diagnostic statistic")


def scalar(signal, count):
    integer(signal["count"])
    require(signal["count"] == count, "stationary run signal count differs")
    for key in ("mean", "rms", "mean_abs", "max_abs"):
        if count:
            finite(signal[key], nonnegative=True)
        else:
            require(signal[key] is None, "absent stationary run measurements must remain null")


def pool(value, maximum):
    require(type(value["available"]) is bool, "planar availability must be explicit")
    integer(value["intervals"], maximum)
    require(value["available"] == bool(value["intervals"]), "planar interval availability differs")
    for key in ("observed_duration_s", "path_length_m", "overflow_duration_s"):
        finite(value[key], nonnegative=True)
    require(value["overflow_duration_s"] <= value["observed_duration_s"] + 1e-7, "planar overflow duration differs")
    require(value["p95_bin_width_m_s"] == .001 and value["p95_overflow_from_m_s"] == 10.
            and value["p95_method"] == "duration-weighted histogram bin midpoint; null if the quantile is in overflow",
            "planar histogram method differs")
    require(set(value["velocity_world"]) == {"vx", "vy"}, "both world-XY velocity axes required")
    for key in ("mean_speed_m_s", "rms_speed_m_s", "max_speed_m_s"):
        if value["available"]:
            finite(value[key], nonnegative=True)
        else:
            require(value[key] is None, "absent planar speed must remain null")
    for axis in ("vx", "vy"):
        for key in ("mean_m_s", "rms_m_s"):
            if value["available"]:
                finite(value["velocity_world"][axis][key], nonnegative=key == "rms_m_s")
            else:
                require(value["velocity_world"][axis][key] is None, "absent world velocity must remain null")
    lower, upper = value["p95_bin_m_s"]
    if not value["available"]:
        require(value["observed_duration_s"] == value["path_length_m"] == value["overflow_duration_s"] == 0
                and lower is upper is value["p95_speed_m_s"] is None, "empty planar pool contains measurements")
    else:
        require(value["observed_duration_s"] > 0, "planar observation duration must be positive")
        finite(lower, nonnegative=True)
        if upper is None:
            require(lower == 10. and value["p95_speed_m_s"] is None and value["overflow_duration_s"] > 0,
                    "overflow quantile must remain null with a lower bound")
        else:
            finite(upper, nonnegative=True)
            finite(value["p95_speed_m_s"], nonnegative=True)
            require(abs(upper - lower - .001) < 1e-10 and 0 <= lower < 10.
                    and abs(value["p95_speed_m_s"] - (lower + upper) / 2) < 1e-10,
                    "planar histogram midpoint differs")
        require(math.isclose(value["mean_speed_m_s"] * value["observed_duration_s"], value["path_length_m"], rel_tol=1e-7, abs_tol=1e-7),
                "planar speed and path duration disagree")
        require(value["mean_speed_m_s"] <= value["rms_speed_m_s"] + 1e-7
                <= value["max_speed_m_s"] + 2e-7, "planar speed moment ordering differs")


def validate_control(value, rows):
    try:
        return _validate_control(value, rows)
    except KeyError as error:
        raise ValueError(f"unsupported diagnostic schema: required control field {error} is missing; no measurement substituted") from error


def _validate_control(value, rows):
    samples = rows * 4001
    for count in (value["full_interval"]["samples"], value["planar_motion"]["physical_samples"],
                  value["planar_motion"]["num_envs"], value["actuation"]["sample_count"],
                  value["actuation"]["scaled_nominal_envelope"]["sample_count"]):
        integer(count)
    require(value["available"] is True and value["full_interval"]["samples"] == samples,
            "full control samples differ from 4001 policy ticks per replica")
    protocol = value["protocol"]
    require(all(protocol[k] == v for k, v in {"policy_dt_s": .01, "settle_steps": 200,
            "min_steady_samples": 200, "signal_time": "PRE-reset physical sample time",
            "axis_order": ["vx", "wz", "height"]}.items()), "control sampling protocol differs")
    planar = value["planar_motion"]
    require(planar["coordinate_frame"] == "world_xy" and planar["num_envs"] == rows
            and planar["physical_samples"] == samples and planar["velocity_source"] == PLANAR_SOURCE
            and planar["scope"] == PLANAR_SCOPE and planar["weighting"] == PLANAR_WEIGHTING
            and planar["steady_eligibility"] == STEADY_ELIGIBILITY, "world planar measurement provenance differs")
    for name in ("full_interval", "steady", "stationary", "stationary_steady"):
        pool(planar[name], 4000 * rows)
    require(planar["full_interval"]["available"] is True, "full interval has no planar velocity observations")
    stationary = planar["stationary"]
    integer(stationary["samples"], samples)
    integer(stationary["runs"], stationary["samples"])
    require(stationary["reference"] == "effective vx and wz both within command_tolerance of zero; height may vary"
            and stationary["origin"] == "first stationary sample; run ends at first nonzero command, reset or evaluation cut",
            "stationary runs do not use the effective zero planar reference")
    for name in ("endpoint_displacement_m", "max_excursion_m"):
        scalar(stationary[name], stationary["runs"])
    steady = value["steady"]
    integer(steady["samples"], samples)
    require(type(steady["available"]) is bool and steady["available"] == bool(steady["samples"]), "steady sample availability differs")
    for name in ("total", "eligible", "short", "failed", "partial", "discarded_settle_samples", "discarded_short_samples"):
        integer(steady[name])
    require(steady["eligible"] + steady["short"] == steady["total"]
            and steady["samples"] + steady["discarded_settle_samples"] + steady["discarded_short_samples"] == samples,
            "steady eligibility or discarded sample accounting differs")
    require(planar["steady"]["intervals"] <= steady["samples"]
            and planar["stationary_steady"]["intervals"] <= planar["steady"]["intervals"]
            and planar["stationary"]["intervals"] <= stationary["samples"], "planar retained coverage differs")
    for section in ("full_interval", "steady"):
        for name in ("vx", "wz", "height"):
            signal = value[section]["axes"][name]
            integer(signal["count"])
            require(signal["count"] == value[section]["samples"], "tracking axis coverage differs")
            for key in ("bias", "mae", "rmse", "within_group_std", "group_mean_std"):
                if signal["count"]:
                    finite(signal[key], nonnegative=key != "bias")
                else:
                    require(signal[key] is None, "ineligible steady tracking must remain null")
    episodes = value["episodes"]
    for name in ("completed", "failed", "success_flags", "partial", "short"):
        integer(episodes[name], samples)
    require(episodes["failed"] + episodes["success_flags"] <= episodes["completed"]
            and episodes["partial"] <= rows, "failed, completed or partial episodes differ")
    require(planar["full_interval"]["intervals"] == samples - episodes["completed"] - episodes["partial"]
            and planar["steady"]["intervals"] == steady["samples"] - steady["eligible"]
            and planar["stationary"]["intervals"] == stationary["samples"] - stationary["runs"],
            "within-episode or within-segment interval accounting differs")
    actuation = value["actuation"]
    scaled = actuation["scaled_nominal_envelope"]
    require(scaled["available"] is True and scaled["sample_count"] == actuation["sample_count"] == samples
            and scaled["unit"] == "N*m" and scaled["semantics"] == SCALED_SEMANTICS
            and scaled["scope"] == SCALED_SCOPE, "scaled nominal envelope provenance differs")
    for count in scaled["active_bound_samples"]:
        integer(count)
    require(scaled["active_bound_samples"] == [samples] * 6, "six scaled PD channels must cover all physical samples")
    for name in ("applied_at_bound_fraction", "requested_outside_bounds_fraction", "applied_outside_bounds_fraction"):
        require(len(scaled[name]) == 6, "six scaled envelope channels required")
        for fraction in scaled[name]:
            finite(fraction, fraction=True)
    return value


def extract(control):
    p, s, a = control["planar_motion"], control["steady"], control["actuation"]["scaled_nominal_envelope"]
    result = {"steady_sample_fraction": s["samples"] / control["full_interval"]["samples"],
        "failed_resets_per_policy_robot_minute": control["episodes"]["failed"] * 60 / (control["full_interval"]["samples"] * .01)}
    for section in ("full_interval", "steady"):
        for axis in ("vx", "wz", "height"):
            result[f"{section}_{axis}_mae"] = control[section]["axes"][axis]["mae"]
    for section in ("full_interval", "steady", "stationary", "stationary_steady"):
        for key in ("mean_speed_m_s", "rms_speed_m_s", "max_speed_m_s", "p95_speed_m_s"):
            result[f"world_xy_{section}_{key}"] = p[section][key]
    for name in ("endpoint_displacement_m", "max_excursion_m"):
        for key in ("mean", "rms", "max_abs"):
            result[f"stationary_run_{name}_{key}"] = p["stationary"][name][key]
    for name in ("applied_at_bound_fraction", "requested_outside_bounds_fraction", "applied_outside_bounds_fraction"):
        for channel, value in enumerate(a[name]):
            result[f"scaled_nominal_{name}_channel_{channel}"] = value
    return result


# Profile macros use only all-interval tracking, coverage and actuation. Drift
# and stationary-run displacement remain at the declared task/case level.
MACRO_METRICS = ("steady_sample_fraction", "failed_resets_per_policy_robot_minute") + tuple(
    f"{section}_{axis}_mae" for section in ("full_interval", "steady") for axis in ("vx", "wz", "height")) + tuple(
    f"scaled_nominal_{name}_channel_{channel}" for name in ("applied_at_bound_fraction",
        "requested_outside_bounds_fraction", "applied_outside_bounds_fraction") for channel in range(6))


ALL_METRICS = MACRO_METRICS + tuple(
    f"world_xy_{section}_{key}" for section in ("full_interval", "steady", "stationary", "stationary_steady")
    for key in ("mean_speed_m_s", "rms_speed_m_s", "max_speed_m_s", "p95_speed_m_s")) + tuple(
    f"stationary_run_{name}_{key}" for name in ("endpoint_displacement_m", "max_excursion_m")
    for key in ("mean", "rms", "max_abs"))


def summarize(rows, metrics):
    result = {}
    for metric in metrics:
        values = [row["metrics"].get(metric) if row["metrics"] is not None else None for row in rows]
        known = [v for v in values if v is not None]
        result[metric] = {"expected_cells": len(rows), "available_cells": len(known),
            "equal_cell_mean": math.fsum(known) / len(rows) if len(known) == len(rows) else None}
    return result


def analyze(manifest_path):
    reader = Reader()
    manifest = load_manifest(reader, manifest_path)
    root = Path(manifest["output_root"]).resolve()
    summary_path = root / "summary.json"
    summary = reader.read(summary_path) if summary_path.exists() else None
    if summary is not None:
        require(summary["manifest_sha256"] == manifest["sha256"] and summary.get("paired_diagnostic_retest") is True
                and summary.get("formal_architecture_selection") is False, "diagnostic summary identity differs")
    results = summary.get("results", {}) if summary else {}
    expected_suites = {f"{variant}/seed_{seed}" for variant in VARIANTS for seed in SEEDS}
    require(set(results).issubset(expected_suites), "summary contains undeclared suites")
    cells, suites = [], []
    for variant in VARIANTS:
        cp = manifest["inputs"]["checkpoints"][variant]
        for seed in SEEDS:
            key = f"{variant}/seed_{seed}"
            receipt = results.get(key)
            status, reports, provenance = "missing", {}, None
            if receipt is not None:
                require(receipt.get("identity") == {"manifest_sha256": manifest["sha256"], "variant": variant,
                    "checkpoint_sha256": cp["checkpoint_sha256"], "seed": seed}, "suite identity differs")
                status = receipt["status"]
                require(status in ("completed", "failed"), "unsealed result status")
                directory = inside(root, receipt["directory"])
                require(directory.parent == root / "evaluations" / variant / f"seed_{seed}"
                        and re.fullmatch(r"attempt_[0-9]{4,}", directory.name), "suite attempt path differs")
                sealed = reader.read(directory / "receipt.json")
                require(sealed == receipt, "sealed suite receipt and summary differ")
                provenance = {"path": str(directory / "receipt.json"), **reader.files[str(directory / "receipt.json")]}
                if status == "completed":
                    require(receipt["worker"]["returncode"] == 0 and receipt["worker"]["timed_out"] is False,
                            "completed suite has no successful worker completion")
                    artifacts = receipt["artifacts"]
                    require(set(artifacts) == set(CASES) | {"control", "trace"}, "suite artifact coverage differs")
                    control_path = reader.checked(root, artifacts["control"])
                    require(control_path == directory / "control.json", "control report route differs")
                    suite = reader.read(control_path)
                    identity = {"checkpoint_sha256": cp["checkpoint_sha256"], "checkpoint_update": 1200, "seed": seed, "steps": 4001}
                    require(suite.get("format") == "transformer_rl.control_evaluation" and suite.get("schema_version") == 1
                            and all(suite.get(k) == v for k, v in identity.items())
                            and all(type(suite[k]) is int for k in ("checkpoint_update", "seed", "steps")) and set(suite["groups"]) == set(CASES)
                            and suite["environment_provenance"]["identity"] == next(iter(manifest["inputs"]["snapshots"].values()))["sha256"],
                            "suite report identity or complete group inventory differs")
                    labels = suite["environment_provenance"]["evaluation_groups"]
                    require(len(labels) == 400 and Counter(labels) == Counter({case: 8 for case in CASES}), "all eight case replicas required")
                    expected_rows = sorted(i for case in CASES for i in [j for j, label in enumerate(labels) if label == case][:2])
                    trace = artifacts["trace"]
                    require(inside(root, trace["path"]) == directory / "trace.npz" and SHA.fullmatch(str(trace["sha256"]))
                            and suite["trace"]["sha256"] == trace["sha256"]
                            and suite["trace"]["row_indices"] == expected_rows
                            and all(suite["trace"].get(k) == v for k, v in {**identity, "policy_dt_s": .01,
                                "sampling_hz": 100., "control_sha256": cp["control_sha256"],
                                "group_labels": [labels[i] for i in expected_rows]}.items()),
                            "declared two-of-eight trace receipt differs")
                    validate_control(suite["control"], 400)
                    for case in CASES:
                        report_path = reader.checked(root, artifacts[case])
                        require(report_path == directory / f"{case}.json", "case report route differs")
                        report = reader.read(report_path)
                        require(report.get("format") == "transformer_rl.packed_evaluation" and report.get("schema_version") == 1
                                and all(report.get(k) == v for k, v in identity.items())
                                and all(type(report[k]) is int for k in ("checkpoint_update", "seed", "steps", "num_envs", "transitions")) and report["num_envs"] == 8
                                and report["transitions"] == 32008
                                and report["control_sha256"] == cp["control_sha256"]
                                and report["environment"] == manifest["inputs"]["environments"][variant][case]
                                and report["control"] == suite["groups"][case], "case identity, control group or sample coverage differs")
                        validate_control(report["control"], 8)
                        reports[case] = (report, {"path": str(report_path), **reader.files[str(report_path)]})
            suites.append({"variant": variant, "evaluation_seed": seed, "status": status, "receipt": provenance})
            for case in CASES:
                report, report_receipt = reports.get(case, (None, None))
                control = report["control"] if report is not None else None
                metrics = extract(control) if control is not None else None
                task, profile = case.split("__")
                cells.append({"variant": variant, "training_seed": 1101, "evaluation_seed": seed,
                    "case": case, "task": task, "profile": profile, "status": "available" if report else status,
                    "case_report": report_receipt, "parent_checkpoint_sha256": cp["checkpoint_sha256"],
                    "metrics": metrics, "control": control,
                    "planar_label": "zero effective planar command runs" if task in ("stand_305mm", "height_scan") else
                                    "motion speed plus separately observed zero-command runs; not all-interval drift"})
    complete = sum(s["status"] == "completed" for s in suites)
    if summary and summary["status"] == "completed":
        require(complete == 20, "completed summary has missing fixed suites")
    pairs = []
    for variant in VARIANTS:
        for case in CASES:
            rows = [row for row in cells if row["variant"] == variant and row["case"] == case]
            pairs.append({"variant": variant, "case": case, "expected_evaluation_seeds": list(SEEDS),
                "metrics": summarize(rows, ALL_METRICS)})
    macros = []
    for variant in VARIANTS:
        for profile in PROFILES:
            rows = [row for row in cells if row["variant"] == variant and row["profile"] == profile]
            require(len(rows) == 12, "profile macro must contain six declared tasks and both evaluation seeds")
            macros.append({"variant": variant, "profile": profile, "expected_cases": [f"{t}__{profile}" for t in TASKS],
                "expected_evaluation_seeds": list(SEEDS), "metrics": summarize(rows, MACRO_METRICS)})
    reader.unchanged()
    result = {"format": "transformer_rl.control_diagnostic_analysis", "schema_version": 1,
        "status": "ready" if complete == 20 else "not_ready", "captured_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["sha256"], "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": {"expected_suites": 20, "completed_suites": complete, "expected_case_reports": 1000,
                     "available_case_reports": sum(row["status"] == "available" for row in cells),
                     "expected_policy_samples": 32008000, "expected_case_replicas": 8, "trace_replicas": 2},
        "protocol": {"case_seed_pair_weighting": "equal evaluation seed weight; null unless both declared cells have the metric",
            "profile_macro_weighting": "equal weight to six fixed tasks and two evaluation seeds; null unless all twelve cells have the metric",
            "extra_cases": "motor_weak and spring_weak are separate standing cases; excluded from six-task profile macros",
            "planar_speed": "duration-weighted within-case world XY finite difference; m/s; not body vx or displacement/time approximation",
            "stationary_displacement": "endpoint and maximum excursion in m per zero-command run; runs can end at command change, failure, reset or cut",
            "steady": "eligibility after 200 settle samples and at least 200 retained samples; eligible failed/partial segments remain included; not success",
            "adapter_success_scope": "packed success_rate counts adapter-success among ended episodes; survive cases can count healthy ordinary truncations including boundary/blocked truncations; not tracking qualification or proven full-horizon survival; not used for ranking here",
            "maximum_or_quantile_aggregation": "equal_cell_mean is a mean of supplied per-case maxima or binned p95 estimates; it does not reconstruct a pooled maximum or pooled quantile",
            "failure_frequency": "failed resets per policy robot minute; repeated resets included; not episode probability",
            "trace": "declared receipt only, two of eight replicas; payload not read or independently verified by this analyzer; statistics use all eight rows",
            "scaled_nominal_envelope": SCALED_SEMANTICS,
            "causal_scope": "one training seed, paired retests of existing evaluation seeds; no new training replicates, no independent held-out domains",
            "ranking": "no winner or hardware qualification; low errors in short failed intervals do not establish robustness"},
        "metric_units": {key: ("m/s" if "speed_m_s" in key or key.endswith("vx_mae") else
            "rad/s" if key.endswith("wz_mae") else "m" if "displacement_m" in key or "excursion_m" in key or key.endswith("height_mae") else
            "failed_resets/policy_robot_minute" if key == "failed_resets_per_policy_robot_minute" else "fraction") for key in ALL_METRICS},
        "suites": suites, "cells": cells, "case_pairs": pairs, "profile_macros": macros,
        "inputs": reader.files, "formal_architecture_selection": False, "hardware_deployment_ready": False}
    result["sha256"] = digest(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.manifest)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        roots = [Path(p).parent for p in result["inputs"]]
        require(not any(output == p or output.is_relative_to(p) for p in roots), "analysis output overlaps frozen inputs")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(encoded)
        print(json.dumps({"status": result["status"], "sha256": result["sha256"], "output": str(output), "coverage": result["coverage"]}))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

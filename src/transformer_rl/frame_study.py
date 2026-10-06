"""Frozen multi-stage/seed studies, retention gates and measured model selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest, json_bytes


def _read(path):
    value = json.loads(Path(path).read_text())
    json_bytes(value)
    return value


def _write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _replace(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path(root, route):
    if not isinstance(route, str) or Path(route).is_absolute():
        raise ValueError("study artifacts require relative paths")
    result = (root / route).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError("artifact path escapes the study")
    return result


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("names require lowercase safe identifiers")


def _seeds(value):
    if (not isinstance(value, list) or not value or len(set(value)) != len(value)
            or any(type(n) is not int or not 0 <= n < 2**32 for n in value)):
        raise ValueError("seeds require unique unsigned 32-bit integers")


def _positive(value, name, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_spec(spec):
    keys = {"base_config", "environment_factory", "variants", "seeds", "stages", "scenarios",
            "training", "evaluation", "execution", "selection"}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError("study requires the documented specification sections")
    factory = spec["environment_factory"]
    if not isinstance(factory, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory):
        raise ValueError("environment_factory requires module:callable")
    _seeds(spec["seeds"])
    _seeds(spec["evaluation"]["seeds"])
    _seeds(spec["evaluation"]["validation_seeds"])
    _seeds(spec["training"]["anchor_seeds"])
    seed_sets = [set(spec["seeds"]), set(spec["training"]["anchor_seeds"]),
                 set(spec["evaluation"]["validation_seeds"]), set(spec["evaluation"]["seeds"])]
    if any(first & second for i, first in enumerate(seed_sets) for second in seed_sets[i + 1:]):
        raise ValueError("training, anchor, validation and held-out evaluation seeds must be disjoint")
    for section, fields in (("training", {"rollout_steps", "checkpoint_interval", "max_seconds", "retention_coef", "anchor_seeds"}),
                            ("evaluation", {"seeds", "validation_seeds", "steps", "settle_steps", "min_steady_samples", "min_completed_episodes"}),
                            ("execution", {"devices", "worker_module", "job_timeout_seconds"}),
                            ("selection", {"min_training_seeds", "objectives", "std_penalty", "latency_p99_ms", "latency_max_ms",
                                           "max_deadline_misses", "retention_score_tolerance", "rollback_limit"})):
        optional = {"max_anchors"} if section == "training" else set()
        if not isinstance(spec[section], dict) or not fields <= set(spec[section]) or set(spec[section]) - fields - optional:
            raise ValueError(f"invalid {section} fields")
    if "max_anchors" in spec["training"]:
        _positive(spec["training"]["max_anchors"], "max_anchors", True)
    for section, fields in (("training", ("rollout_steps", "checkpoint_interval")),
                            ("evaluation", ("steps", "min_steady_samples", "min_completed_episodes")),
                            ("selection", ("min_training_seeds",))):
        for name in fields:
            _positive(spec[section][name], name, True)
    for section, fields in (("training", ("max_seconds",)), ("execution", ("job_timeout_seconds",)),
                            ("selection", ("latency_p99_ms", "latency_max_ms"))):
        for name in fields:
            _positive(spec[section][name], name)
    for section, names in (("training", ("retention_coef",)), ("evaluation", ("settle_steps",)),
                           ("selection", ("std_penalty", "retention_score_tolerance", "rollback_limit", "max_deadline_misses"))):
        for name in names:
            value = spec[section][name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
    for section, name in (("evaluation", "settle_steps"), ("selection", "rollback_limit"), ("selection", "max_deadline_misses")):
        if type(spec[section][name]) is not int:
            raise ValueError(f"{name} must be an integer")
    devices = spec["execution"]["devices"]
    if (not isinstance(devices, list) or not devices or len(set(devices)) != len(devices)
            or any(not isinstance(d, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", d) for d in devices)):
        raise ValueError("devices require unique cpu/cuda:index routes")
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", spec["execution"]["worker_module"]):
        raise ValueError("invalid worker module")
    for entries, shape in ((spec["variants"], {"name", "policy"}),
                           (spec["scenarios"], {"name", "environment", "gates"}),
                           (spec["stages"], {"name", "updates", "environment", "scenarios"})):
        if not isinstance(entries, list) or not entries:
            raise ValueError("variants, scenarios and stages must be nonempty")
        names = set()
        for entry in entries:
            allowed = shape | {"require_steady"} if "gates" in shape else shape
            if not isinstance(entry, dict) or not shape <= set(entry) or set(entry) - allowed:
                raise ValueError("invalid variant/scenario/stage fields")
            _name(entry["name"])
            if entry["name"] in names:
                raise ValueError("duplicate name")
            names.add(entry["name"])
    scenario_names = {entry["name"] for entry in spec["scenarios"]}
    for stage in spec["stages"]:
        _positive(stage["updates"], "stage updates", True)
        if (not isinstance(stage["environment"], dict) or not isinstance(stage["scenarios"], list)
                or not stage["scenarios"] or len(set(stage["scenarios"])) != len(stage["scenarios"])
                or set(stage["scenarios"]) - scenario_names):
            raise ValueError("stages require environment overrides and known scenarios")
    if set().union(*(set(stage["scenarios"]) for stage in spec["stages"])) != scenario_names:
        raise ValueError("every scenario must be evaluated by a stage")
    for scenario in spec["scenarios"]:
        if type(scenario.get("require_steady", False)) is not bool:
            raise ValueError("require_steady must be boolean")
        if not isinstance(scenario["environment"], dict) or not isinstance(scenario["gates"], list) or not scenario["gates"]:
            raise ValueError("scenarios require environment overrides and explicit gates")
        for gate in scenario["gates"]:
            if (not isinstance(gate, dict) or set(gate) != {"path", "operator", "value"}
                    or gate["operator"] not in ("min", "max") or not isinstance(gate["path"], str)
                    or type(gate["value"]) not in (int, float) or not math.isfinite(gate["value"])):
                raise ValueError("invalid scenario gate")
    objectives = spec["selection"]["objectives"]
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("selection needs explicit scaled objectives")
    for item in objectives:
        if (not isinstance(item, dict) or set(item) != {"path", "direction", "scale", "weight"}
                or item["direction"] not in ("minimize", "maximize") or not isinstance(item["path"], str)):
            raise ValueError("invalid selection objective")
        _positive(item["scale"], "objective scale")
        _positive(item["weight"], "objective weight")


def _configs(base, spec):
    configs = {}
    for variant in spec["variants"]:
        policy = variant["policy"]
        if not isinstance(policy, dict) or set(policy) & {"frame_dim", "action_dim", "mean_init_scale"}:
            raise ValueError("comparison overrides must preserve the observation, action and mean initialization contract")
        candidate = base.with_policy(policy)
        for kind, entries in (("train", spec["stages"]), ("eval", spec["scenarios"])):
            for entry in entries:
                value = candidate.to_dict()
                value["environment"].update(entry["environment"])
                configs[f"configs/{variant['name']}.{kind}.{entry['name']}.json"] = FrameTrainConfig.from_dict(value).to_dict()
    return configs


def plan_study(spec_path, root):
    spec_path, root = Path(spec_path).resolve(), Path(root).resolve()
    spec = _read(spec_path)
    _validate_spec(spec)
    base = FrameTrainConfig.load(spec_path.parent / spec["base_config"])
    configs = _configs(base, spec)
    if spec["selection"]["latency_p99_ms"] >= base.control["policy_dt_s"] * 1000:
        raise ValueError("latency gate must reserve some policy-period control-chain margin")
    manifest = {"format": "transformer_rl.packed_study", "schema_version": 1,
                "spec": spec, "base": base.to_dict(), "source": source_identity(),
                "configs": {route: digest(value) for route, value in configs.items()}}
    manifest["sha256"] = digest(manifest)
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    (root / "jobs").mkdir()
    shutil.copytree(Path(__file__).resolve().parent, root / "policy_source/transformer_rl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for route, config in configs.items():
        _write(root / route, config)
    _write(root / "plan.json", manifest)
    return {"root": str(root), "sha256": manifest["sha256"],
            "jobs": len(spec["variants"]) * len(spec["seeds"]),
            "updates_per_job": sum(stage["updates"] for stage in spec["stages"]),
            "policy_hz": 1 / base.control["policy_dt_s"]}


def validate_study(root, *, source=False):
    root = Path(root).resolve()
    manifest = _read(root / "plan.json")
    if (manifest.get("format") != "transformer_rl.packed_study" or manifest.get("schema_version") != 1
            or digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest.get("sha256")):
        raise ValueError("study manifest hash/format mismatch")
    _validate_spec(manifest["spec"])
    package = root / "policy_source/transformer_rl"
    source_files = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(package.rglob("*")) if path.is_file()
                    and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if source_files != manifest["source"]["files"]:
        raise ValueError("frozen policy source changed")
    expected = _configs(FrameTrainConfig.from_dict(manifest["base"]), manifest["spec"])
    if manifest["configs"] != {path: digest(config) for path, config in expected.items()}:
        raise ValueError("study configurations differ from frozen specification")
    for route, expected_hash in manifest["configs"].items():
        if digest(_read(_path(root, route))) != expected_hash:
            raise ValueError(f"configuration changed: {route}")
    if source and source_identity() != manifest["source"]:
        raise ValueError("package source changed since planning; create a new study")
    return manifest


def _value(report, path):
    try:
        for part in path.split("."):
            report = report[part]
    except (TypeError, KeyError):
        return None
    return float(report) if type(report) in (int, float) and math.isfinite(report) else None


def grade_report(report, scenario, evaluation, objectives):
    errors = []
    if _value(report, "completed_episodes") is None or report["completed_episodes"] < evaluation["min_completed_episodes"]:
        errors.append("insufficient completed episodes")
    if scenario.get("require_steady", False) and not report.get("stability", {}).get("available"):
        errors.append("insufficient post-settle episode samples")
    for gate in scenario["gates"]:
        value = _value(report, gate["path"])
        if value is None or (value < gate["value"] if gate["operator"] == "min" else value > gate["value"]):
            errors.append(gate["path"])
    score = 0.
    for objective in objectives:
        value = _value(report, objective["path"])
        if value is None:
            errors.append("missing objective: " + objective["path"])
        else:
            score += objective["weight"] * value / objective["scale"] * (1 if objective["direction"] == "minimize" else -1)
    return {"passed": not errors, "reasons": errors, "score": score if not any(e.startswith("missing") for e in errors) else None}


def _process_start(pid):
    try:
        # The command name may itself contain spaces or parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _recover_processes(directory):
    """Reap only workers whose PID, process-start identity and argv still match."""
    for path in directory.rglob("*.process.json"):
        record = _read(path)
        if record["status"] != "running":
            continue
        pid = record["pid"]
        if record["start"] is not None and _process_start(pid) == record["start"]:
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
                if argv != record["command"] or os.getpgid(pid) != pid:
                    raise ValueError("worker process ownership cannot be verified")
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while _process_start(pid) == record["start"] and time.monotonic() < deadline:
                    time.sleep(.05)
                if _process_start(pid) == record["start"]:
                    os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        record["status"] = "abandoned"
        _replace(path, record)


def _process(command, log_path, seconds, stop):
    ownership = Path(str(log_path) + ".process.json")
    with Path(log_path).open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": process.pid, "start": _process_start(process.pid), "command": command, "status": "running"}
        _replace(ownership, record)
        deadline = time.monotonic() + seconds
        try:
            while process.poll() is None:
                if stop.is_set() or time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return False
                stop.wait(0.1)
            return process.returncode == 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record.update(status="finished", returncode=process.returncode)
            _replace(ownership, record)


def _receipt(path, root):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load_receipt(root, item):
    path = _path(root, item["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("study evidence hash mismatch")
    return _read(path)


def _checked_path(root, item):
    path = _path(root, item["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("study evidence hash mismatch")
    return path


def _account_training(root, entry, pending, config):
    """Seal the worker outcome and its budget in one atomic study-state update."""
    attempt = _path(root, pending["directory"])
    completion, failure = attempt / "train/completion.json", attempt / "train/failure.json"
    receipt_path = completion if completion.exists() else failure if failure.exists() else None
    receipt = _read(receipt_path) if receipt_path else None
    consumed = receipt["attempted_updates"] if receipt else pending["requested_updates"]
    if type(consumed) is not int or not 0 <= consumed <= pending["requested_updates"]:
        raise ValueError("worker budget receipt is invalid")
    if not pending.get("budget_charged"):
        entry["consumed_updates"] += consumed
        pending.update(consumed_updates=consumed, budget_charged=True)
    if receipt_path:
        pending["training"] = _receipt(receipt_path, root)
    else:
        pending["budget_scope"] = "whole chunk charged after unsealed worker crash"
    if receipt_path == completion:
        if (receipt["config_sha256"] != digest(_read(config))
                or receipt["completed_updates"] > consumed):
            raise ValueError("worker configuration/update receipt differs")
        checkpoint = _receipt(receipt["checkpoint"], root)
        if checkpoint["sha256"] != receipt["checkpoint_sha256"]:
            raise ValueError("training checkpoint identity differs")
        pending["checkpoint"] = checkpoint
        entry.update(resume_checkpoint=checkpoint, resume_mode="--resume")
        if receipt["status"] == "completed" and consumed == pending["requested_updates"]:
            pending["status"] = "trained"
            return True
    pending["status"] = "stopped" if receipt and receipt["status"] == "stopped" else "failed"
    return False


def _evaluate_attempt(root, manifest, variant, required, pending, worker, device, stop, state_path, state, *, anchor_only=False, final_only=False):
    spec = manifest["spec"]
    if anchor_only and spec["training"]["retention_coef"] <= 0:
        raise ValueError("anchor collection requires enabled retention")
    factory = spec["environment_factory"]
    scenarios = {case["name"]: case for case in spec["scenarios"]}
    attempt = _path(root, pending["directory"])
    checkpoint_path = _checked_path(root, pending["checkpoint"])
    kind = "anchor" if anchor_only else "final" if final_only else "evaluation"
    retry = pending.get(f"{kind}_attempts", 0)
    evaluation_dir = attempt / f"{kind}_{retry:04d}"
    while evaluation_dir.exists():
        retry += 1
        evaluation_dir = attempt / f"{kind}_{retry:04d}"
    evaluation_dir.mkdir(exist_ok=False)
    pending[f"{kind}_attempts"] = retry + 1
    if anchor_only:
        pending["anchor_reports"], pending["anchors"] = {}, []
    elif final_only:
        pending["final_evaluations"] = {}
    else:
        pending["evaluations"] = {}
    _replace(state_path, state)
    grades, batches = {}, {}
    for name in required:
        key = scenarios[name]["environment"].get("evaluation_batch", name) if factory == "transformer_rl.chassis_adapter:make_env" else name
        _name(key)
        batches.setdefault(key, []).append(name)
    for batch_name, names in batches.items():
        seeds = spec["training"]["anchor_seeds"] if anchor_only else spec["evaluation"]["seeds" if final_only else "validation_seeds"]
        for eval_seed in seeds:
            output_paths = [evaluation_dir / f"{name}_{eval_seed}.json" for name in names]
            if len(names) > 1:
                command = worker + ["evaluate-suite", "--checkpoint", str(checkpoint_path), "--configs"]
                command += [str(root / f"configs/{variant['name']}.eval.{name}.json") for name in names]
                command += ["--outputs", *(str(path) for path in output_paths)]
                if anchor_only:
                    command.extend(("--anchor-directory", str(evaluation_dir / f"anchors_{batch_name}_{eval_seed}")))
            else:
                name = names[0]
                command = worker + ["evaluate", "--checkpoint", str(checkpoint_path), "--config",
                    str(root / f"configs/{variant['name']}.eval.{name}.json"), "--env-factory", factory,
                    "--output", str(output_paths[0])]
                if anchor_only:
                    command.extend(("--anchor-output", str(evaluation_dir / f"anchors_{name}_{eval_seed}.pt")))
            command += ["--steps", str(spec["evaluation"]["steps"]), "--seed", str(eval_seed), "--device", device,
                "--settle-steps", str(spec["evaluation"]["settle_steps"]),
                "--min-steady-samples", str(spec["evaluation"]["min_steady_samples"])]
            if anchor_only and "max_anchors" in spec["training"]:
                # Omission retains the legacy command protocol for frozen workers.
                command.extend(("--max-anchors", str(spec["training"]["max_anchors"])))
            if not _process(command, evaluation_dir / f"{batch_name}_{eval_seed}.log", spec["execution"]["job_timeout_seconds"], stop):
                # The trained checkpoint/budget are already sealed. A retry only evaluates.
                state["status"] = "stopped" if stop.is_set() else "failed"
                _replace(state_path, state)
                return False
            for name, path in zip(names, output_paths):
                report = _read(path)
                if report["checkpoint_sha256"] != pending["checkpoint"]["sha256"]:
                    raise ValueError("evaluation checkpoint identity differs")
                key = f"{name}/{eval_seed}"
                grades[key] = grade_report(report, scenarios[name], spec["evaluation"], spec["selection"]["objectives"])
                if anchor_only:
                    pending["anchor_reports"][key] = _receipt(path, root)
                    if grades[key]["passed"]:
                        anchors = report["anchors"]
                        capacity = spec["training"].get("max_anchors", 256)
                        # Legacy reports omit the cap; only legacy specs may accept them.
                        reported_capacity = anchors.get("max_samples", 256 if "max_anchors" not in spec["training"] else None)
                        if (type(reported_capacity) is not int or reported_capacity != capacity
                                or type(anchors.get("samples")) is not int or not 1 <= anchors["samples"] <= capacity):
                            raise ValueError("anchor sample count/capacity differs from specification")
                        receipt = _receipt(anchors["path"], root)
                        if receipt["sha256"] != anchors.get("sha256"):
                            raise ValueError("anchor artifact identity differs from evaluation")
                        pending["anchors"].append(receipt)
                else:
                    pending["final_evaluations" if final_only else "evaluations"][key] = _receipt(path, root)
    if anchor_only:
        pending["anchors_complete"] = True
        _replace(state_path, state)
        return True
    if final_only:
        state["final_evaluations"] = pending["final_evaluations"]
        state["final_evaluation_complete"] = True
        _replace(state_path, state)
        return True
    pending.update(grades=grades, status="evaluated")
    regression = any(not grades[key]["passed"] or grades[key]["score"] is None
                     or grades[key]["score"] > score + spec["selection"]["retention_score_tolerance"]
                     for key, score in state["protected"].items())
    pending["retention_regression"] = regression
    if regression:
        state["rollbacks"] += 1
    else:
        state["checkpoint"] = pending["checkpoint"]
        if all(grade["passed"] for grade in grades.values()):
            # Keep the best observed reference for each old skill; no cumulative ratchet.
            state["protected"] = {key: min(state["protected"].get(key, grade["score"]), grade["score"])
                                  for key, grade in grades.items()}
            state["protected_checkpoint"] = pending["checkpoint"]
    _replace(state_path, state)
    return True


def _run_job(root, manifest, variant, seed, device, stop):
    spec = manifest["spec"]
    job_dir = root / "jobs" / variant["name"] / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    state_path = job_dir / "state.json"
    state = _read(state_path) if state_path.exists() else {
        "variant": variant["name"], "seed": seed, "plan_sha256": manifest["sha256"],
        "status": "running", "stages": [], "protected": {}, "rollbacks": 0, "anchors": []}
    if state["plan_sha256"] != manifest["sha256"]:
        raise ValueError("job state belongs to another plan")
    if state["status"] in ("completed", "rejected"):
        return state
    _recover_processes(job_dir)
    bootstrap = ("import runpy, sys; sys.path.insert(0, " + repr(str(root / "policy_source"))
                 + "); runpy.run_module(" + repr(spec["execution"]["worker_module"])
                 + ", run_name='__main__', alter_sys=True)")
    worker = [sys.executable, "-c", bootstrap]
    required = []
    for stage_index, stage in enumerate(spec["stages"]):
        required.extend(name for name in stage["scenarios"] if name not in required)
        if len(state["stages"]) <= stage_index:
            state["stages"].append({"name": stage["name"], "consumed_updates": 0, "attempts": [],
                "anchors": list(state["anchors"]), "resume_checkpoint": state.get("checkpoint"),
                "resume_mode": "--initialize-from"})
        entry = state["stages"][stage_index]
        if entry.get("passed"):
            _checked_path(root, entry["checkpoint"])
            continue
        config_path = root / f"configs/{variant['name']}.train.{stage['name']}.json"
        if entry["attempts"] and entry["attempts"][-1]["status"] == "running":
            _account_training(root, entry, entry["attempts"][-1], config_path)
            _replace(state_path, state)
        while not stop.is_set():
            pending = entry["attempts"][-1] if entry["attempts"] else None
            if pending and pending["status"] == "evaluated" and pending.get("retention_regression"):
                if state["rollbacks"] > spec["selection"]["rollback_limit"]:
                    state["status"] = "rejected"
                    _replace(state_path, state)
                    return state
                entry.update(resume_checkpoint=state["protected_checkpoint"], resume_mode="--restore-learning-from")
            if not pending or pending["status"] != "trained":
                remaining = stage["updates"] - entry["consumed_updates"]
                if remaining <= 0:
                    break
                number = len(entry["attempts"])
                attempt = job_dir / stage["name"] / f"attempt_{number:04d}"
                while attempt.exists():
                    number += 1
                    attempt = job_dir / stage["name"] / f"attempt_{number:04d}"
                attempt.mkdir(parents=True, exist_ok=False)
                count = min(remaining, spec["training"]["checkpoint_interval"])
                command = worker + ["train", "--config", str(config_path), "--env-factory", spec["environment_factory"],
                    "--run-dir", str(attempt / "train"), "--updates", str(count),
                    "--rollout-steps", str(spec["training"]["rollout_steps"]), "--seed", str(seed), "--device", device,
                    "--max-seconds", str(spec["training"]["max_seconds"]),
                    "--checkpoint-interval", str(spec["training"]["checkpoint_interval"]),
                    "--consumed-update-offset", str(entry["consumed_updates"])]
                parent = entry["resume_checkpoint"]
                if parent:
                    command.extend((entry["resume_mode"], str(_checked_path(root, parent))))
                coefficient = spec["training"]["retention_coef"]
                if coefficient > 0 and entry["anchors"]:
                    command.extend(("--retention-coef", str(coefficient), "--anchors"))
                    command.extend(str(_checked_path(root, receipt)) for receipt in entry["anchors"])
                pending = {"directory": str(attempt.relative_to(root)), "requested_updates": count, "status": "running"}
                entry["attempts"].append(pending)
                state["status"] = "running"
                _replace(state_path, state)
                _process(command, attempt / "train.log", spec["execution"]["job_timeout_seconds"], stop)
                trained = _account_training(root, entry, pending, config_path)
                _replace(state_path, state)
                if not trained:
                    state["status"] = "stopped" if stop.is_set() else "failed"
                    _replace(state_path, state)
                    return state
            if stop.is_set():
                break
            if not _evaluate_attempt(root, manifest, variant, required, pending, worker, device, stop, state_path, state):
                return state
        if stop.is_set():
            state["status"] = "stopped"
            _replace(state_path, state)
            return state
        last = entry["attempts"][-1]
        passed = (last["status"] == "evaluated" and not last.get("retention_regression")
                  and bool(last.get("grades")) and all(g["passed"] for g in last["grades"].values()))
        if not passed:
            state["status"] = "rejected"
            _replace(state_path, state)
            return state
        entry["checkpoint"] = state["checkpoint"]
        state["curriculum_evaluations"] = last["evaluations"]
        if spec["training"]["retention_coef"] > 0:
            if not last.get("anchors_complete") and not _evaluate_attempt(
                    root, manifest, variant, required, last, worker, device, stop, state_path, state, anchor_only=True):
                # Promotion waits for independent anchor collection, so resuming cannot skip it.
                _replace(state_path, state)
                return state
            state["anchors"] = last["anchors"]
        entry["passed"] = True
        _replace(state_path, state)
    if not state.get("final_evaluation_complete") and not _evaluate_attempt(
            root, manifest, variant, required, state["stages"][-1]["attempts"][-1], worker, device,
            stop, state_path, state, final_only=True):
        return state
    # Each retry has its own directory/log. Incomplete exports are preserved for diagnosis.
    parent = state["checkpoint"]
    bundle = _path(root, state["deployment"]["path"]).parent if "deployment" in state else None
    if bundle is None:
        retry = state.get("export_attempts", 0)
        bundle = job_dir / f"deployment_{retry:04d}"
        state["export_attempts"] = retry + 1
        _replace(state_path, state)
        command = worker + ["export", "--checkpoint", str(_checked_path(root, parent)), "--directory", str(bundle)]
        if not _process(command, job_dir / f"export_{retry:04d}.log", spec["execution"]["job_timeout_seconds"], stop):
            state["status"] = "failed"
            _replace(state_path, state)
            return state
        state["deployment"] = _receipt(bundle / "manifest.json", root)
        _replace(state_path, state)
    else:
        _load_receipt(root, state["deployment"])
    if "benchmark" not in state:
        retry = state.get("benchmark_attempts", 0)
        benchmark_path = job_dir / f"benchmark_{retry:04d}.json"
        state["benchmark_attempts"] = retry + 1
        _replace(state_path, state)
        command = worker + ["benchmark", "--directory", str(bundle), "--output", str(benchmark_path)]
        if not _process(command, job_dir / f"benchmark_{retry:04d}.log", spec["execution"]["job_timeout_seconds"], stop):
            state["status"] = "failed"
            _replace(state_path, state)
            return state
        state["benchmark"] = _receipt(benchmark_path, root)
    state["status"] = "completed"
    _replace(state_path, state)
    return state


def run_study(root, *, max_parallel=1):
    root = Path(root).resolve()
    manifest = validate_study(root, source=True)
    _positive(max_parallel, "max_parallel", True)
    devices = manifest["spec"]["execution"]["devices"]
    if max_parallel > len(devices):
        raise ValueError("one worker per device; add device routes before increasing parallelism")
    stop = threading.Event()
    previous = {}
    def interrupted(*_):
        stop.set()
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, interrupted)
    try:
        with (root / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            jobs = [(variant, seed) for variant in manifest["spec"]["variants"] for seed in manifest["spec"]["seeds"]]
            # A lane retains exclusive ownership of its device throughout each full curriculum.
            def lane(index):
                return [_run_job(root, manifest, variant, seed, devices[index], stop)
                        for variant, seed in jobs[index::max_parallel] if not stop.is_set()]
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                results = [job for lane_result in executor.map(lane, range(max_parallel)) for job in lane_result]
        return {"status": "completed" if len(results) == len(jobs) and all(j["status"] in ("completed", "rejected") for j in results) else "incomplete",
                "jobs": [{"variant": j["variant"], "seed": j["seed"], "status": j["status"]} for j in results]}
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def select_transformer(root):
    root = Path(root).resolve()
    manifest = validate_study(root)
    spec = manifest["spec"]
    selection = spec["selection"]
    candidates = []
    machines = set()
    for variant in spec["variants"]:
        scores, jobs, errors = [], [], []
        for seed in spec["seeds"]:
            path = root / "jobs" / variant["name"] / f"seed_{seed}" / "state.json"
            if not path.exists():
                errors.append(f"missing seed {seed}")
                continue
            state = _read(path)
            if state.get("plan_sha256") != manifest["sha256"] or state.get("status") != "completed":
                errors.append(f"incomplete/rejected seed {seed}")
                continue
            if state.get("variant") != variant["name"] or state.get("seed") != seed:
                raise ValueError("job identity differs from comparison")
            if len(state["stages"]) != len(spec["stages"]):
                raise ValueError("completed job has incomplete curriculum")
            for entry, stage in zip(state["stages"], spec["stages"]):
                if (entry["name"] != stage["name"] or not entry.get("passed")
                        or entry["consumed_updates"] != stage["updates"]
                        or sum(a["consumed_updates"] for a in entry["attempts"]) != stage["updates"]):
                    raise ValueError("curriculum budget/promotion ledger differs")
                for attempt in entry["attempts"]:
                    if "training" in attempt:
                        receipt = _load_receipt(root, attempt["training"])
                        if receipt["attempted_updates"] != attempt["consumed_updates"]:
                            raise ValueError("training budget evidence differs")
                    elif attempt["consumed_updates"] != attempt["requested_updates"]:
                        raise ValueError("unsealed worker attempts must consume their reserved budget")
                _checked_path(root, entry["checkpoint"])
            _checked_path(root, state["checkpoint"])
            evidence = state.get("final_evaluations", {})
            expected = {f"{s['name']}/{seed_value}" for s in spec["scenarios"] for seed_value in spec["evaluation"]["seeds"]}
            if set(evidence) != expected:
                errors.append(f"missing evaluation coverage seed {seed}")
                continue
            grades = []
            for scenario in spec["scenarios"]:
                for eval_seed in spec["evaluation"]["seeds"]:
                    report = _load_receipt(root, evidence[f"{scenario['name']}/{eval_seed}"])
                    config = FrameTrainConfig.load(root / f"configs/{variant['name']}.eval.{scenario['name']}.json")
                    if (report["checkpoint_sha256"] != state["checkpoint"]["sha256"] or report["seed"] != eval_seed
                            or report["model"] != config.to_dict()["model"] or report["environment"] != config.environment
                            or report["control_sha256"] != digest(config.control)
                            or report["steps"] != spec["evaluation"]["steps"]
                            or report["stability"]["protocol"]["settle_steps"] != spec["evaluation"]["settle_steps"]
                            or report["stability"]["protocol"]["min_steady_samples"] != spec["evaluation"]["min_steady_samples"]):
                        raise ValueError("selection report identity/protocol mismatch")
                    grades.append(grade_report(report, scenario, spec["evaluation"], selection["objectives"]))
            bundle = _load_receipt(root, state["deployment"])
            benchmark = _load_receipt(root, state["benchmark"])
            if bundle["checkpoint_sha256"] != state["checkpoint"]["sha256"] or benchmark["manifest_sha256"] != state["deployment"]["sha256"]:
                raise ValueError("deployment evidence differs from evaluated policy")
            bundle_directory = _checked_path(root, state["deployment"]).parent
            for name, expected_hash in bundle["files"].items():
                if Path(name).name != name or hashlib.sha256((bundle_directory / name).read_bytes()).hexdigest() != expected_hash:
                    raise ValueError("deployment graph evidence differs")
            if (benchmark.get("backend") != "onnx" or benchmark.get("threads") != 1
                    or type(benchmark.get("iterations")) is not int or benchmark["iterations"] < 1000
                    or any(_value(benchmark, name) is None or benchmark[name] < 0
                           for name in ("p99_ms", "max_ms", "deadline_misses"))):
                raise ValueError("latency benchmark protocol/values differ")
            machines.add(digest(benchmark["machine"]))
            if any(not g["passed"] for g in grades):
                errors.append(f"task gate failed seed {seed}")
            if (benchmark["p99_ms"] > selection["latency_p99_ms"] or benchmark["max_ms"] > selection["latency_max_ms"]
                    or benchmark["deadline_misses"] > selection["max_deadline_misses"]):
                errors.append(f"latency gate failed seed {seed}")
            if all(g["score"] is not None for g in grades):
                score = statistics.mean(g["score"] for g in grades)
                scores.append(score)
                jobs.append((score, state))
        if len(scores) < selection["min_training_seeds"]:
            errors.append("insufficient independent training seeds")
        mean = statistics.mean(scores) if scores else None
        std = statistics.stdev(scores) if len(scores) > 1 else None
        model = FrameTrainConfig.load(root / f"configs/{variant['name']}.train.{spec['stages'][0]['name']}.json").model
        entry = {"variant": variant["name"], "architecture": model.policy.architecture,
                 "eligible": not errors, "reasons": errors, "training_seed_scores": scores,
                 "score_mean": mean, "score_sample_std": std,
                 "rank_score": mean + selection["std_penalty"] * (std or 0.) if mean is not None else None}
        if not errors:
            median = statistics.median(scores)
            chosen = min(jobs, key=lambda item: (abs(item[0] - median), item[1]["seed"]))[1]
            entry["representative_seed"] = chosen["seed"]
            entry["checkpoint"] = chosen["checkpoint"]
            entry["deployment"] = chosen["deployment"]
        candidates.append(entry)
    if len(machines) > 1:
        raise ValueError("latency comparisons require the same measured machine")
    eligible = [c for c in candidates if c["eligible"]]
    transformers = [c for c in eligible if c["architecture"] == "transformer"]
    winner = min(transformers, key=lambda c: (c["rank_score"], c["variant"])) if transformers else None
    overall = min(eligible, key=lambda c: (c["rank_score"], c["variant"])) if eligible else None
    report = {"plan_sha256": manifest["sha256"], "candidates": candidates, "best_transformer": winner,
              "best_overall": overall, "transformer_is_best_overall": bool(winner and overall and winner["variant"] == overall["variant"]),
              "selection_rule": "all task/retention/latency gates; equal scenario and evaluation-seed weights; training mean + std penalty; median seed artifact",
              "status": "selected" if winner else "no_eligible_transformer",
              "latency_scope": "measured CPU inference pipeline; final sensor/transport/torque-loop verification is separate"}
    _replace(root / "selection.json", report)
    return report

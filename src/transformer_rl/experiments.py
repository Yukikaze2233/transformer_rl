"""Frozen experiment plans and bounded, process-isolated execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

from .config import ModelConfig, PPOConfig, config_dict


_ACTOR_FIELDS = {"actor_type", "time_encoding", "readout_type", "residual_type", "baseline_hidden",
                 "gru_hidden", "d_model", "num_layers", "num_heads", "ffn_dim"}
_SENSITIVITY_FIELDS = {"model": {"mean_init_scale", "initial_std"},
                       "ppo": {"learning_rate"}}
_OPTIMIZER_DIAGNOSTICS = (
    "initial_mean_abs", "initial_std_mean", "initial_std_min", "initial_std_max",
    "first_step_kl", "first_step_mean_kl", "first_step_std_kl",
    "first_step_mean_change_rms", "first_step_normalized_mean_change_rms",
    "final_kl", "final_mean_kl", "final_std_kl", "final_mean_change_rms", "final_std_mean",
    "kl", "stop_kl", "optimizer_steps", "planned_optimizer_steps",
)
_TERMINATION_GRACE_SECONDS = 3.0
_STABILITY_WINDOWS = ("settle_steps", "min_steady_samples")
_STABILITY_STATS = ("mean", "within_episode_std", "derivative_rms", "max_abs")
_STABILITY_COUNTS = (
    "count", "segments", "short_count", "short_segments", "total_count", "settled_count",
    "derivative_count", "completed_segments", "partial_segments",
    "short_completed_segments", "short_partial_segments",
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    _encoded(value)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("artifact route must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("artifact route escapes experiment root")
    return path


def source_identity():
    """Hash actual imported package files, including uncommitted source."""
    package = Path(__file__).resolve().parent
    files = {str(p.relative_to(package)): _sha(p)
             for p in sorted(package.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    return {"files": files, "sha256": _digest(files)}


def _positive(value, name, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be positive")


def _seeds(values):
    if (not isinstance(values, list) or not values
            or any(type(v) is not int or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("seeds must be unique nonnegative integers")


def _stability_protocol(evaluation):
    # Resolve omitted windows from the core; its defaults are bound by source_identity.
    from .stability import EpisodeSignalStatistics

    return EpisodeSignalStatistics(
        1, **{key: evaluation[key] for key in _STABILITY_WINDOWS if key in evaluation}
    ).report()["protocol"]


def _validate_spec(spec):
    expected = {"base_config", "environment_factory", "seeds", "variants",
                "training", "execution", "evaluation"}
    if set(spec) != expected:
        raise ValueError("spec requires exactly the documented sections")
    if not isinstance(spec["base_config"], str):
        raise ValueError("base_config must be a path")
    factory = spec["environment_factory"]
    if factory is not None and (not isinstance(factory, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory)):
        raise ValueError("environment_factory must be null or module:callable")
    _seeds(spec["seeds"])
    names = set()
    if not isinstance(spec["variants"], list) or not spec["variants"]:
        raise ValueError("variants must be nonempty")
    for variant in spec["variants"]:
        if not isinstance(variant, dict) or set(variant) != {"name", "model", "ppo", "group"}:
            raise ValueError("invalid variant shape")
        name = variant["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("variant names must be unique safe identifiers")
        names.add(name)
        if variant["group"] not in ("architecture", "supervision", "sensitivity"):
            raise ValueError("unknown comparison group")
        if not all(isinstance(variant[k], dict) for k in ("model", "ppo")):
            raise ValueError("variant overrides must be objects")
    for section, keys in (
        ("training", {"updates", "rollout_steps", "max_seconds", "checkpoint_interval", "action_clip"}),
        ("execution", {"devices", "max_parallel", "job_timeout_seconds"}),
        ("evaluation", {"steps", "seeds", "environment"}),
    ):
        optional = {"execution": {"worker_module"}, "training": {"diagnostics"},
                    "evaluation": {"scenarios", *_STABILITY_WINDOWS}}.get(section, set())
        if (not isinstance(spec[section], dict) or not keys <= set(spec[section])
                or set(spec[section]) - keys - optional):
            raise ValueError(f"invalid {section} shape")
    for key in ("updates", "rollout_steps", "checkpoint_interval"):
        _positive(spec["training"][key], key, True)
    _positive(spec["training"]["max_seconds"], "max_seconds")
    if type(spec["training"].get("diagnostics", False)) is not bool:
        raise ValueError("training.diagnostics must be boolean")
    if spec["training"]["action_clip"] is not None:
        _positive(spec["training"]["action_clip"], "action_clip")
    execution = spec["execution"]
    worker = execution.get("worker_module", "transformer_rl")
    if not isinstance(worker, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", worker):
        raise ValueError("worker_module must be a qualified Python module name")
    _positive(execution["max_parallel"], "max_parallel", True)
    _positive(execution["job_timeout_seconds"], "job_timeout_seconds")
    if (not isinstance(execution["devices"], list) or not execution["devices"]
            or any(not isinstance(d, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", d)
                   for d in execution["devices"])):
        raise ValueError("devices must be a nonempty list of cpu/cuda devices")
    _positive(spec["evaluation"]["steps"], "evaluation.steps", True)
    _stability_protocol(spec["evaluation"])
    _seeds(spec["evaluation"]["seeds"])
    if not isinstance(spec["evaluation"]["environment"], dict):
        raise ValueError("evaluation.environment must be an object")
    if "scenarios" in spec["evaluation"]:
        scenarios = spec["evaluation"]["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("evaluation.scenarios must be a nonempty list")
        names = set()
        for scenario in scenarios:
            if not isinstance(scenario, dict) or set(scenario) != {"name", "environment"}:
                raise ValueError("scenario requires exactly name and environment; steps are uniform")
            name = scenario["name"]
            if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                    or name in names):
                raise ValueError("scenario names must be unique safe identifiers")
            names.add(name)
            if not isinstance(scenario["environment"], dict):
                raise ValueError("scenario.environment must be an object")


def _configuration(base, variant, evaluation=None):
    if set(base) - {"model", "ppo", "environment"}:
        raise ValueError("unknown base config section")
    configs = []
    for key, cls in (("model", ModelConfig), ("ppo", PPOConfig)):
        section = base.get(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"base {key} must be an object")
        values = {**section, **variant[key]}
        if set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown {key} fields; ensure configuration implementation is available")
        defaults = cls()
        allowed = set(_ACTOR_FIELDS) if key == "model" else set()
        if variant["group"] == "supervision":
            allowed.add("auxiliary_indices" if key == "model" else "auxiliary_coef")
        elif variant["group"] == "sensitivity":
            allowed = _SENSITIVITY_FIELDS[key]
        reference_values = dict(section)
        for field in fields(cls):
            if isinstance(getattr(defaults, field.name), tuple):
                for candidate in (reference_values, values):
                    if field.name in candidate:
                        candidate[field.name] = tuple(candidate[field.name])
        reference, configured = cls(**reference_values), cls(**values)
        for name in variant[key]:
            if name not in allowed and getattr(configured, name) != getattr(reference, name):
                raise ValueError(f"fair comparison forbids changing {key}.{name}")
        configs.append(configured)
    if variant["group"] == "architecture" and getattr(configs[0], "auxiliary_indices", ()):
        raise ValueError("architecture group must not contain auxiliary supervision heads")
    if getattr(configs[1], "auxiliary_coef", 0) > 0:
        if (not getattr(configs[0], "auxiliary_indices", ())
                or variant["group"] not in ("supervision", "sensitivity")):
            raise ValueError("auxiliary supervision requires targets and supervision group")
    environment = base.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("base environment must be an object")
    return json.loads(_encoded(config_dict(*configs, {**environment, **(evaluation or {})})))


def _evaluation_configs(spec, variant):
    """Resolve shared scenario overrides without changing legacy config routes."""
    evaluation = spec["evaluation"]
    if "scenarios" not in evaluation:
        return {None: (f"configs/{variant}.evaluation.json", evaluation["environment"])}
    return {scene["name"]: (f"configs/{variant}.evaluation.{scene['name']}.json",
                            {**evaluation["environment"], **scene["environment"]})
            for scene in evaluation["scenarios"]}


def _plan_configs(base, spec):
    configs = {}
    for variant in spec["variants"]:
        configs[f"configs/{variant['name']}.json"] = _configuration(base, variant)
        for route, environment in _evaluation_configs(spec, variant["name"]).values():
            configs[route] = _configuration(base, variant, environment)
    return configs


def _jobs(spec):
    return [{"id": f"{v['name']}/seed_{seed}", "variant": v["name"],
             "group": v["group"], "seed": seed,
             "config": f"configs/{v['name']}.json",
             **({"eval_configs": {name: route for name, (route, _) in
                                  _evaluation_configs(spec, v["name"]).items()}}
                if "scenarios" in spec["evaluation"] else
                {"eval_config": f"configs/{v['name']}.evaluation.json"}),
             "directory": f"jobs/{v['name']}/seed_{seed}"}
            for v in spec["variants"] for seed in spec["seeds"]]


def plan(spec_path, root):
    spec_path, root = Path(spec_path).resolve(), Path(root).absolute()
    spec = _read(spec_path)
    _validate_spec(spec)
    base = _read(spec_path.parent / spec["base_config"])
    configs = _plan_configs(base, spec)
    manifest = {"spec": spec, "spec_sha256": _digest(spec), "base_config": base,
                "source": source_identity(), "jobs": _jobs(spec),
                "configs": {k: _digest(v) for k, v in configs.items()}}
    manifest["plan_sha256"] = _digest(manifest)
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    for route, config in configs.items():
        _write(root / route, config)
    _write(root / "plan.json", manifest)
    return manifest


def validate_plan(root, check_source=False):
    root = Path(root).resolve()
    manifest = _read(root / "plan.json")
    if set(manifest) != {"spec", "spec_sha256", "base_config", "source", "jobs", "configs", "plan_sha256"}:
        raise ValueError("invalid plan shape")
    content = {k: v for k, v in manifest.items() if k != "plan_sha256"}
    if _digest(content) != manifest["plan_sha256"]:
        raise ValueError("plan hash mismatch")
    source = manifest["source"]
    if (not isinstance(source, dict) or set(source) != {"files", "sha256"}
            or not isinstance(source["files"], dict) or not source["files"]
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                   for v in source["files"].values())
            or _digest(source["files"]) != source["sha256"]):
        raise ValueError("invalid source snapshot")
    spec = manifest["spec"]
    _validate_spec(spec)
    if _digest(spec) != manifest["spec_sha256"] or manifest["jobs"] != _jobs(spec):
        raise ValueError("spec hash or job routes mismatch")
    expected = {route: _digest(config) for route, config in
                _plan_configs(manifest["base_config"], spec).items()}
    if manifest["configs"] != expected:
        raise ValueError("config manifest mismatch")
    for route, digest in expected.items():
        if _digest(_read(_inside(root, route))) != digest:
            raise ValueError(f"frozen config hash mismatch: {route}")
    for job in manifest["jobs"]:
        _inside(root, job["directory"])
    if check_source and manifest["source"] != source_identity():
        raise ValueError("package source changed since plan; create a new plan")
    return manifest


class _Interrupted(Exception):
    pass


def _process(argv, log_path, deadline, stop):
    """Own one session; kill the whole session group and reap its direct child."""
    if stop.is_set():
        raise _Interrupted("execution interrupted")
    if time.monotonic() >= deadline:
        raise TimeoutError("job deadline exceeded before subprocess launch")
    with log_path.open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                if stop.wait(0.05):
                    raise _Interrupted("execution interrupted")
                if time.monotonic() >= deadline:
                    raise TimeoutError("job deadline exceeded")
            if child.returncode:
                raise RuntimeError(f"subprocess exited {child.returncode}; see {log_path}")
        finally:
            # Also remove descendants left behind by a normally exiting leader.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                grace_deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
                while time.monotonic() < grace_deadline:
                    child.poll()  # Reap the leader so an empty group disappears.
                    os.killpg(child.pid, 0)
                    time.sleep(min(0.05, max(0, grace_deadline - time.monotonic())))
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def _checkpoint(directory, spec):
    completion = _read(_inside(directory, "train/completion.json"))
    updates = spec["training"]["updates"]
    if (completion.get("status") != "completed" or completion.get("updates_completed") != updates
            or completion.get("cumulative_update") != updates
            or any(type(completion.get(k)) is not int for k in ("updates_completed", "cumulative_update"))):
        raise ValueError("training did not complete requested updates")
    _positive(completion.get("collected_transitions"), "collected_transitions", True)
    checkpoints = completion.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or not isinstance(checkpoints[-1], dict):
        raise ValueError("completion has no last checkpoint")
    last = checkpoints[-1]
    path = Path(last["path"])
    if not path.is_absolute():
        path = directory / "train" / path
    path = path.resolve()
    if not path.is_relative_to(_inside(directory, "train/checkpoints")):
        raise ValueError("checkpoint escapes training checkpoint directory")
    if last.get("update") != updates or _sha(path) != last.get("sha256"):
        raise ValueError("last checkpoint update or SHA mismatch")
    return path, last["sha256"]


def _evaluation_route(scenario, seed):
    return f"evaluation_{scenario}_{seed}.json" if scenario is not None else f"evaluation_{seed}.json"


def _evaluation_prefix(scenario):
    return f"scenarios.{scenario}." if scenario is not None else ""


def _validate_stability(stability, evaluation, transitions):
    if (not isinstance(stability, dict) or type(stability.get("available")) is not bool
            or not isinstance(stability.get("signals"), dict)):
        raise ValueError("invalid stability report shape")
    protocol = stability.get("protocol")
    if (not isinstance(protocol, dict)
            or any(type(protocol.get(k)) is not int for k in _STABILITY_WINDOWS)
            or protocol != _stability_protocol(evaluation)):
        raise ValueError("stability protocol mismatch")
    for name, signal in stability["signals"].items():
        if not isinstance(name, str) or not name or not isinstance(signal, dict):
            raise ValueError("invalid stability signal")
        if any(type(signal.get(k)) is not int or signal[k] < 0 for k in _STABILITY_COUNTS):
            raise ValueError("invalid stability count")
        count, segments = signal["count"], signal["segments"]
        if (bool(count) != bool(segments)
                or count < segments * protocol["min_steady_samples"]
                or signal["derivative_count"] != count - segments
                or segments != signal["completed_segments"] + signal["partial_segments"]
                or signal["short_segments"] != (
                    signal["short_completed_segments"] + signal["short_partial_segments"])
                or signal["short_count"] > signal["short_segments"] * (protocol["min_steady_samples"] - 1)
                or signal["total_count"] != signal["settled_count"] + signal["short_count"] + count
                or signal["total_count"] != transitions
                or segments + signal["short_segments"] > transitions
                or signal["settled_count"] > (
                    segments + signal["short_segments"]) * protocol["settle_steps"]):
            raise ValueError("inconsistent stability counts")
        for key in (*_STABILITY_STATS, "episode_mean_min", "episode_mean_max", "episode_mean_std"):
            if key not in signal:
                raise ValueError(f"missing stability statistic: {key}")
            value = signal[key]
            available = signal["derivative_count"] > 0 if key == "derivative_rms" else count > 0
            if ((not available and value is not None)
                    or (available and (type(value) not in (int, float) or not math.isfinite(value)))
                    or (value is not None and key in (
                        "within_episode_std", "derivative_rms", "max_abs", "episode_mean_std") and value < 0)):
                raise ValueError(f"invalid stability finite-or-null statistic: {key}")
        if count:
            tolerance = 1e-7 * max(1, signal["max_abs"])
            if (abs(signal["mean"]) > signal["max_abs"] + tolerance
                    or signal["within_episode_std"] > signal["max_abs"] + tolerance
                    or not signal["episode_mean_min"] - tolerance <= signal["mean"] <= (
                        signal["episode_mean_max"] + tolerance)):
                raise ValueError("inconsistent stability statistics")
    if stability["available"] != any(s["count"] > 0 for s in stability["signals"].values()):
        raise ValueError("stability availability contradicts counts")


def _report(path, digest, seed, spec, environment):
    report = _read(path)
    expected = {"checkpoint_sha256": digest, "seed": seed,
                "vector_steps": spec["evaluation"]["steps"], "environment": environment,
                "action_clip": spec["training"]["action_clip"], "policy": "deterministic_mean"}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("evaluation report protocol mismatch")
    if any(type(report.get(k)) is not int for k in ("seed", "vector_steps")):
        raise ValueError("evaluation seed and vector_steps must be integers")
    for key in ("reward_mean", "transitions", "terminated_count", "truncated_count"):
        if type(report.get(key)) not in (int, float) or not math.isfinite(report[key]):
            raise ValueError(f"invalid evaluation {key}")
    for key in ("transitions", "terminated_count", "truncated_count"):
        if type(report[key]) is not int or report[key] < 0:
            raise ValueError(f"invalid evaluation count: {key}")
    _positive(report["transitions"], "transitions", True)
    if any(report[k] > report["transitions"] for k in ("terminated_count", "truncated_count")):
        raise ValueError("evaluation termination count exceeds transitions")
    if "done_count" in report:
        done = report["done_count"]
        if (type(done) is not int or not max(report["terminated_count"], report["truncated_count"])
                <= done <= min(report["transitions"], report["terminated_count"] + report["truncated_count"])):
            raise ValueError("invalid evaluation done_count")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("evaluation metrics must be an object")
    for metric in metrics.values():
        if not isinstance(metric, dict) or set(metric) != {"mean", "rms", "min", "max", "count"}:
            raise ValueError("invalid metric shape")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in metric.values()):
            raise ValueError("invalid metric value")
        _positive(metric["count"], "metric count", True)
        tolerance = 1e-7 * max(1, abs(metric["min"]), abs(metric["mean"]), abs(metric["max"]))
        if (metric["min"] > metric["max"] or metric["mean"] < metric["min"] - tolerance
                or metric["mean"] > metric["max"] + tolerance or metric["rms"] < 0
                or metric["count"] > report["transitions"]):
            raise ValueError("inconsistent evaluation metric statistics")
    if "stability" in report:
        _validate_stability(report["stability"], spec["evaluation"], report["transitions"])
    report["stability_missing"] = "stability" not in report
    return report


def _run_job(root, manifest, job, device, stop, runner):
    spec = manifest["spec"]
    directory = _inside(root, job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    result = {"job": job["id"], "plan_sha256": manifest["plan_sha256"], "device": device}
    deadline = time.monotonic() + spec["execution"]["job_timeout_seconds"]
    common = ["--env-factory", spec["environment_factory"], "--device", device]
    clip = spec["training"]["action_clip"]
    if clip is not None:
        common += ["--action-clip", str(clip)]
    try:
        worker = spec["execution"].get("worker_module", "transformer_rl")
        train = [sys.executable, "-m", worker, "train", "--config",
                 str(_inside(root, job["config"])), "--run-dir", str(directory / "train"),
                 "--seed", str(job["seed"]), *common]
        for key in ("updates", "rollout_steps", "max_seconds", "checkpoint_interval"):
            train += ["--" + key.replace("_", "-"), str(spec["training"][key])]
        if spec["training"].get("diagnostics", False):
            train.append("--diagnostics")
        runner(train, directory / "train.log", deadline, stop)
        if manifest["source"] != source_identity():
            raise ValueError("package source changed during execution")
        checkpoint, digest = _checkpoint(directory, spec)
        for scenario, (route, _) in _evaluation_configs(spec, job["variant"]).items():
            config = _inside(root, route)
            environment = _read(config)["environment"]
            for seed in spec["evaluation"]["seeds"]:
                output = _inside(directory, _evaluation_route(scenario, seed))
                argv = [sys.executable, "-m", worker, "evaluate", "--checkpoint",
                        str(checkpoint), "--config", str(config),
                        "--steps", str(spec["evaluation"]["steps"]), "--seed", str(seed),
                        "--output", str(output), *common]
                for key in _STABILITY_WINDOWS:
                    if key in spec["evaluation"]:
                        argv += ["--" + key.replace("_", "-"), str(spec["evaluation"][key])]
                runner(argv, output.with_suffix(".log"), deadline, stop)
                _report(output, digest, seed, spec, environment)
        if manifest["source"] != source_identity():
            raise ValueError("package source changed during execution")
        result["status"] = "completed"
    except TimeoutError as error:
        result.update(status="timedout", error=str(error))
    except FileNotFoundError as error:
        result.update(status="missing", error=str(error))
    except Exception as error:
        result.update(status="failed", error=str(error))
    _write(directory / "result.json", result)
    return result


def run(root, max_parallel=None, *, runner=_process):
    root = Path(root).resolve()
    manifest = validate_plan(root, check_source=True)
    spec = manifest["spec"]
    if spec["environment_factory"] is None:
        raise ValueError("run requires a verified environment_factory; null is plan-only")
    limit = spec["execution"]["max_parallel"] if max_parallel is None else max_parallel
    _positive(limit, "max_parallel", True)
    # Claim once, before any child starts. A failed/interrupted attempt is not resumable.
    for job in manifest["jobs"]:
        if _inside(root, job["directory"]).exists():
            raise FileExistsError("existing job directory; create a new experiment root")
    _write(root / "execution.json", {"plan_sha256": manifest["plan_sha256"], "max_parallel": limit,
                                     "termination_grace_seconds": _TERMINATION_GRACE_SECONDS})
    slots = queue.Queue()
    devices = spec["execution"]["devices"]
    for index in range(limit):
        slots.put(devices[index % len(devices)])
    stop = threading.Event()
    previous = {}

    def interrupt(_number, _frame):
        stop.set()

    def worker(job):
        device = slots.get()
        try:
            if stop.is_set():
                return None
            return _run_job(root, manifest, job, device, stop, runner)
        finally:
            slots.put(device)

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.getsignal(number)
            signal.signal(number, interrupt)
        with ThreadPoolExecutor(max_workers=limit) as executor:
            try:
                futures = [executor.submit(worker, job) for job in manifest["jobs"]]
                results = [future.result() for future in futures]
            except BaseException:
                stop.set()
                raise
        interrupted = stop.is_set()
    finally:
        stop.set()
        for number, handler in previous.items():
            signal.signal(number, handler)
    return {"interrupted": interrupted,
            "jobs": results}


def _statistics(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def _training_diagnostics(path):
    """Describe available log rows without making them evaluation evidence."""
    result = {}
    updates = []
    invalid_rows = 0
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("metric row must be an object")
            except ValueError:
                invalid_rows += 1
                continue
            collection = record.get("collection", {})
            reward = collection.get("reward_mean") if isinstance(collection, dict) else None
            if type(reward) in (int, float) and math.isfinite(reward):
                result["training_reward_diagnostic"] = reward
            optimization = record.get("optimization")
            if isinstance(optimization, dict):
                updates.append(optimization)
    if not updates:
        return result
    diagnostic = {"updates_observed": len(updates), "invalid_metric_rows": invalid_rows}
    for label, update in (("first_update", updates[0]), ("last_update", updates[-1])):
        for key in _OPTIMIZER_DIAGNOSTICS:
            value = update.get(key)
            diagnostic[f"{label}.{key}"] = (
                value if type(value) in (int, float) and math.isfinite(value) else None)
    for key in ("optimizer_steps", "planned_optimizer_steps"):
        counts = [update.get(key) for update in updates]
        diagnostic[key] = (sum(counts) if all(type(n) is int and n >= 0 for n in counts)
                           else None)
    actual, planned = diagnostic["optimizer_steps"], diagnostic["planned_optimizer_steps"]
    valid_budget = all(
        type(update.get("optimizer_steps")) is int
        and type(update.get("planned_optimizer_steps")) is int
        and 0 <= update["optimizer_steps"] <= update["planned_optimizer_steps"]
        for update in updates)
    diagnostic["optimizer_step_utilization"] = (
        actual / planned if valid_budget and planned else None)
    early_stops = [update.get("early_stopped") for update in updates]
    diagnostic["early_stopped_fraction"] = (
        sum(early_stops) / len(updates) if all(type(v) is bool for v in early_stops) else None)
    result["optimizer_diagnostics"] = diagnostic
    return result


def _stability_seed_means(reports, prefix):
    """Keep missing runs explicit while describing the available subset equally."""
    means, coverage = {}, {}

    def record(available):
        return {"requested_eval_runs": len(reports), "available_eval_runs": sum(available),
                "transitions": sum(r["transitions"] for r in reports),
                "missing_eval_seeds": [r["seed"] for r, ok in zip(reports, available) if not ok]}

    blocks = [r.get("stability", {}) for r in reports]
    scope = record([b.get("available", False) for b in blocks])
    scope["missing_report_eval_seeds"] = [r["seed"] for r in reports if "stability" not in r]
    coverage[prefix + "stability"] = scope
    names = sorted({name for block in blocks for name in block.get("signals", {})})
    for name in names:
        signals = [b.get("signals", {}).get(name, {}) for b in blocks]
        counts = {key: sum(s.get(key, 0) for s in signals) for key in _STABILITY_COUNTS}
        # The denominator includes reports without this signal; no missing value is a zero measurement.
        counts["transitions"] = sum(r["transitions"] for r in reports)
        counts["sample_coverage"] = counts["count"] / counts["transitions"]
        for stat in _STABILITY_STATS:
            key = f"{prefix}stability.{name}.{stat}"
            values = [s.get(stat) for s in signals]
            means[key] = _statistics([v for v in values if v is not None])["mean"]
            coverage[key] = {**record([v is not None for v in values]), **counts}
    return means, coverage


def _stability_training_coverage(spec, seeds):
    """Summarize metric coverage without treating eval runs as independent train seeds."""
    keys = {_evaluation_prefix(scene) + "stability"
            for scene in _evaluation_configs(spec, spec["variants"][0]["name"])}
    keys.update(key for seed in seeds for key in seed.get("stability_coverage", {}))
    result = {}
    for key in sorted(keys):
        scope_key = ".".join(key.split(".")[:3]) if key.startswith("scenarios.") else "stability"
        entries = [s.get("stability_coverage", {}).get(key, {
            "transitions": s.get("stability_coverage", {}).get(scope_key, {}).get("transitions", 0)
        }) for s in seeds]
        available = [e.get("available_eval_runs", 0) for e in entries]
        requested = len(seeds) * len(spec["evaluation"]["seeds"])
        result[key] = {
            "requested_training_seeds": len(seeds),
            "available_training_seeds": sum(n > 0 for n in available),
            "missing_training_seeds": [s["seed"] for s, n in zip(seeds, available) if not n],
            "incomplete_training_seeds": [s["seed"] for s, n in zip(seeds, available)
                                          if n != len(spec["evaluation"]["seeds"])],
            "requested_eval_runs": requested, "available_eval_runs": sum(available),
            "missing_eval_runs": requested - sum(available),
            "complete": sum(available) == requested,
        }
        if any("count" in e for e in entries):
            counts = {k: sum(e.get(k, 0) for e in entries) for k in (*_STABILITY_COUNTS, "transitions")}
            result[key].update(counts, sample_coverage=(
                counts["count"] / counts["transitions"] if counts["transitions"] else None))
    return result


def _summarize_evaluations(root, directory, spec, job):
    """Aggregate a final checkpoint only after all its scenario/seed reports validate."""
    means, transitions = {}, {}
    stability_coverage = {}
    physical_metrics_available = True
    _, digest = _checkpoint(directory, spec)
    for scenario, (route, _) in _evaluation_configs(spec, job["variant"]).items():
        environment = _read(_inside(root, route))["environment"]
        reports = [_report(_inside(directory, _evaluation_route(scenario, seed)),
                           digest, seed, spec, environment)
                   for seed in spec["evaluation"]["seeds"]]
        keys = set(reports[0]["metrics"])
        if any(set(r["metrics"]) != keys for r in reports):
            raise ValueError("evaluation metric sets differ across seeds")
        prefix = _evaluation_prefix(scenario)
        counts = {str(r["seed"]): r["transitions"] for r in reports}
        if scenario is None:
            transitions = counts
        else:
            transitions[prefix.rstrip(".")] = counts
        for key in ("reward_mean", "terminated_count", "truncated_count", "transitions"):
            means[prefix + key] = statistics.mean(r[key] for r in reports)
        for key in sorted(keys):
            for stat in ("mean", "rms", "min", "max"):
                means[f"{prefix}metrics.{key}.{stat}"] = statistics.mean(
                    r["metrics"][key][stat] for r in reports)
        physical_metrics_available = physical_metrics_available and bool(keys)
        stability_means, coverage = _stability_seed_means(reports, prefix)
        means.update(stability_means)
        stability_coverage.update(coverage)
    return {"evaluation_transitions": transitions, "evaluation_seed_means": means,
            "physical_metrics_available": physical_metrics_available,
            "stability_protocol": _stability_protocol(spec["evaluation"]),
            "stability_coverage": stability_coverage,
            "stability_complete": all(not c["missing_eval_seeds"] for c in stability_coverage.values())}


def _scenario_budget_checks(spec, seeds, partial):
    """Compare final-checkpoint sample counts within each scenario, never pool tasks."""
    checks = {}
    for scenario in spec["evaluation"]["scenarios"]:
        key = _evaluation_prefix(scenario["name"]).rstrip(".")
        counts = sorted({count for seed in seeds
                         for count in seed["evaluation_transitions"][key].values()})
        checks[key] = {"partial": partial, "evaluation_transition_counts": counts,
                       "evaluation_budget_consistent": len(counts) == 1,
                       "comparison_available": not partial and len(counts) == 1}
    return checks


def summarize(root):
    root = Path(root).resolve()
    manifest = validate_plan(root)
    spec = manifest["spec"]
    # Resolve omitted values through the same canonical dataclasses as planning.
    base = _configuration(manifest["base_config"],
                          {"model": {}, "ppo": {}, "group": "supervision"})
    variants = []
    for variant in spec["variants"]:
        config = _read(_inside(root, f"configs/{variant['name']}.json"))
        changes = {f"{section}.{key}": {"base": base[section][key], "value": value}
                   for section in ("model", "ppo") for key, value in config[section].items()
                   if value != base[section][key]}
        row = {"variant": variant["name"], "group": variant["group"],
               "factor_changes": changes, "single_factor": len(changes) == 1,
               "requested": len(spec["seeds"]), "completed": 0, "failed": 0,
               "timedout": 0, "missing": 0, "seeds": []}
        values = {}
        diagnostics = []
        optimizer_diagnostics = {}
        for job in (j for j in manifest["jobs"] if j["variant"] == variant["name"]):
            directory = _inside(root, job["directory"])
            item = {"seed": job["seed"], "status": "missing"}
            try:
                result = _read(_inside(directory, "result.json"))
                if result.get("plan_sha256") != manifest["plan_sha256"] or result.get("job") != job["id"]:
                    raise ValueError("result provenance mismatch")
                status = result.get("status")
                if status not in ("completed", "failed", "timedout", "missing"):
                    raise ValueError("invalid result status")
                item["status"] = status
                if status == "completed":
                    evaluation = _summarize_evaluations(root, directory, spec, job)
                    item["training_transitions"] = _read(
                        _inside(directory, "train/completion.json"))["collected_transitions"]
                    item.update(evaluation)
                    for key, value in evaluation["evaluation_seed_means"].items():
                        available = values.setdefault(key, [])
                        if value is not None:
                            available.append(value)
            except FileNotFoundError as error:
                item.update(status="missing", error=str(error))
            except (ValueError, KeyError, TypeError) as error:
                item.update(status="failed", error=str(error))
            if "scenarios" in spec["evaluation"]:
                item["partial"] = item["status"] != "completed"
            metrics_path = _inside(directory, "train/metrics.jsonl")
            if metrics_path.is_file():
                item.update(_training_diagnostics(metrics_path))
                if "training_reward_diagnostic" in item:
                    diagnostics.append(item["training_reward_diagnostic"])
                for key, value in item.get("optimizer_diagnostics", {}).items():
                    available = optimizer_diagnostics.setdefault(key, [])
                    if value is not None:
                        available.append(value)
            row[item["status"]] += 1
            row["seeds"].append(item)
        row["evaluation"] = {k: _statistics(v) for k, v in values.items()}
        row["training_reward_diagnostic"] = _statistics(diagnostics)
        row["optimizer_diagnostics"] = {k: _statistics(v) for k, v in optimizer_diagnostics.items()}
        row["evaluation_complete"] = row["completed"] == row["requested"]
        row["partial"] = not row["evaluation_complete"]
        row["physical_metrics_complete"] = row["evaluation_complete"] and all(
            s.get("physical_metrics_available", False) for s in row["seeds"])
        row["stability_coverage"] = _stability_training_coverage(spec, row["seeds"])
        row["stability_complete"] = all(c["complete"] for c in row["stability_coverage"].values())
        variants.append(row)
    fairness_checks = {}
    for group in sorted({v["group"] for v in variants}):
        members = [v for v in variants if v["group"] == group]
        seeds = [s for v in members for s in v["seeds"] if s["status"] == "completed"]
        train_counts = sorted({s["training_transitions"] for s in seeds})
        partial = any(v["partial"] for v in members)
        scenario_checks = None
        if "scenarios" in spec["evaluation"]:
            scenario_checks = _scenario_budget_checks(spec, seeds, partial)
            eval_counts = {key: check["evaluation_transition_counts"]
                           for key, check in scenario_checks.items()}
            evaluation_budget_consistent = all(
                check["evaluation_budget_consistent"] for check in scenario_checks.values())
        else:
            eval_counts = sorted({count for s in seeds for count in s["evaluation_transitions"].values()})
            evaluation_budget_consistent = len(eval_counts) == 1
        reasons = []
        if partial:
            reasons.append("partial: not all requested training seeds have complete evaluation")
        if len(train_counts) != 1:
            reasons.append("training sample budget unavailable or inconsistent")
        if not evaluation_budget_consistent:
            reasons.append("evaluation sample budget unavailable or inconsistent")
        check = {"partial": partial, "training_transition_counts": train_counts,
                  "evaluation_transition_counts": eval_counts,
                  "comparison_available": not reasons, "reasons": reasons,
                 "scope": "within-group configuration and sample-budget checks, not a quality ranking"}
        stability_keys = [set(v["stability_coverage"]) for v in members]
        check["stability_complete"] = all(v["stability_complete"] for v in members)
        check["stability_comparison_available"] = (
            check["comparison_available"] and check["stability_complete"]
            and all(keys == stability_keys[0] for keys in stability_keys))
        if scenario_checks is not None:
            for scenario_key, scenario_check in scenario_checks.items():
                scenario_check["comparison_available"] = (
                    scenario_check["comparison_available"] and len(train_counts) == 1)
                scene_keys = [{k for k in keys if k.startswith(scenario_key + ".stability")}
                              for keys in stability_keys]
                scenario_check["stability_complete"] = all(
                    v["stability_coverage"][k]["complete"] for v, keys in zip(members, scene_keys) for k in keys)
                scenario_check["stability_comparison_available"] = (
                    scenario_check["comparison_available"] and scenario_check["stability_complete"]
                    and all(keys == scene_keys[0] for keys in scene_keys))
            check["scenarios"] = scenario_checks
        fairness_checks[group] = check
        for row in members:
            row["comparison_available"] = check["comparison_available"]
            row["comparison_reasons"] = reasons
            row["training_budget_consistent"] = len(train_counts) == 1
            row["evaluation_budget_consistent"] = evaluation_budget_consistent
            row["stability_comparison_available"] = check["stability_comparison_available"]
    summary = {"plan_sha256": manifest["plan_sha256"], "variants": variants,
               "stability_protocol": _stability_protocol(spec["evaluation"]),
               "fairness_checks": fairness_checks,
               "interpretation": "Descriptive available-seed statistics only; no winner or physical-success claim. "
                 "Missing seeds remain explicit. Sample std uses independent training seeds, not frames. "
                 "Stability uses available eval-run means then training-seed statistics; nulls are never zero. "
                 "Coverage counts include eligible partial and failure episodes. Low jitter, including zero jitter "
                 "with large bias, is not standing success; consult bias/max_abs, full-interval metrics and terminations. "
                "factor_changes compares canonical values to base; zero changes is baseline, one is single-factor, "
                "and combinations do not establish single-factor effects. Optimizer diagnostics describe available "
                "logged updates, including incomplete jobs; lower KL alone is not better (zero LR can mean no learning). "
                 "Formal comparisons require at least 3 (preferably 5) independent training seeds and sufficient transitions."}
    if "scenarios" in spec["evaluation"]:
        summary["evaluation_scenarios"] = spec["evaluation"]["scenarios"]
        summary["interpretation"] += (
            " Only the final checkpoint is evaluated, with each scenario aggregated separately:"
            " evaluation-seed means, then training-seed statistics."
            " Missing any final-checkpoint scenario/evaluation seed excludes that training seed from all evaluation aggregates."
            " Scenario metrics use scenarios.NAME.")
    # Reports are derived artifacts and may be refreshed; execution artifacts are exclusive.
    for route in ("summary.json", "summary.csv"):
        if (root / route).is_symlink():
            raise ValueError("summary output must not be a symlink")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    row_columns = ["variant", "group", "requested", "completed", "failed", "timedout", "missing",
                   "factor_changes", "single_factor",
                   "evaluation_complete", "physical_metrics_complete", "partial", "comparison_available",
                   "stability_complete", "stability_comparison_available",
                   "training_budget_consistent", "evaluation_budget_consistent", "comparison_reasons"]
    columns = [*row_columns, "metric", "n", "mean", "std", "stability_coverage"]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in variants:
            metrics = {**row["evaluation"], "training_reward_diagnostic": row["training_reward_diagnostic"],
                       **{f"optimizer_diagnostics.{k}": v for k, v in row["optimizer_diagnostics"].items()}}
            for key in row["stability_coverage"]:
                metrics.setdefault(key, _statistics([]))
            for key, stats in metrics.items():
                writer.writerow({**{k: row[k] for k in row_columns},
                                 "comparison_reasons": "; ".join(row["comparison_reasons"]),
                                 "factor_changes": json.dumps(row["factor_changes"], sort_keys=True),
                                 "stability_coverage": json.dumps(row["stability_coverage"][key], sort_keys=True)
                                 if key in row["stability_coverage"] else "",
                                 "metric": key, **stats})
    return summary

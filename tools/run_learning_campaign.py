"""Execute a frozen 90-cell learning-rate grid and audit real training evidence.

Prepared studies remain read-only. Training and development evaluations live in
a separate campaign root. This controller does not select a learning rate,
perform confirmation, or qualify an architecture for hardware deployment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


_HELPER = Path(__file__).with_name("run_frame_diagnostic_campaign.py")
_SPEC = importlib.util.spec_from_file_location("learning_diagnostic_helpers", _HELPER)
diagnostic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostic)
transfer, control = diagnostic.transfer, diagnostic.control
write = transfer.write
FORMAT = "transformer_rl.learning_campaign"
FACTORY = "transformer_rl.chassis_adapter:make_env"
SAMPLES = 49152
UPDATES = 1200
TOTAL = SAMPLES * UPDATES
LEARNER_MAX_SECONDS = 604800.
DEVELOPMENT_SEEDS = [701, 1701, 2701, 3701]
DEPENDENCIES = {"transfer": ("transformer_rl.transfer_campaign", "campaign_sha256"),
                "curriculum": ("transformer_rl.curriculum_campaign", "campaign_sha256"),
                "diagnostics": ("transformer_rl.frame_diagnostic_campaign", "manifest_sha256")}
TRAINING_PROTOCOL = {"expected_cells": 90, "updates": UPDATES, "samples_per_update": SAMPLES,
    "samples_per_cell": TOTAL, "planned_total_samples": 90 * TOTAL, "retention_coef": 0.,
    "zero_update_endpoint": "terminal_incomplete_no_refund_no_reseed",
    "short_or_unsealed_attempt": "terminal_incomplete_entire_reservation_charged_actual_separately_verified",
    "resume": "exact_full_learning_state_sealed_full_rollout_endpoint_only"}


def read(path):
    value = control.read(path)
    json.dumps(value, allow_nan=False)
    return value


def plain(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("symlink paths are not permitted")
    return path.resolve()


def inside(root, route):
    root, route = plain(root), Path(route)
    result = plain(route if route.is_absolute() else root / route)
    if result == root or not result.is_relative_to(root):
        raise ValueError("artifact path escapes its root")
    return result


def artifact(path):
    path = plain(path)
    return {"path": str(path), "sha256": control.file_sha(path)}


def checked(item):
    path = plain(item["path"])
    if not Path(item["path"]).is_absolute() or control.file_sha(path) != diagnostic.sha(item["sha256"]):
        raise ValueError("sealed artifact SHA differs")
    return path


def inventory(root):
    root = plain(root)
    if not root.is_dir():
        raise ValueError("input directory is absent")
    result = {}
    for path in sorted(plain(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink input files are not permitted")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            result[str(path.relative_to(root))] = control.file_sha(path)
    return result


def learner_identity(root):
    files = inventory(plain(root) / "src/transformer_rl")
    if "frame_process.py" not in files:
        raise ValueError("a frozen packed learner package is required")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def controllers():
    paths = [Path(__file__), _HELPER, _HELPER.with_name("run_transfer_campaign.py"),
             _HELPER.with_name("run_frame_control_campaign.py")]
    return {str(plain(path)): control.file_sha(path) for path in paths}


def cpu_environment(source):
    result = dict(os.environ)
    result.update(PYTHONPATH=str(plain(source) / "src"), PYTHONDONTWRITEBYTECODE="1",
                  CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return result


def python_command(executable, cache, *arguments):
    cache = plain(cache)
    if not cache.is_dir() or any(cache.iterdir()):
        raise ValueError("Python cache prefix must be an exclusive empty directory")
    return [str(executable), "-B", "-X", "pycache_prefix=" + str(cache), *arguments]


def runtime_identity(source):
    script = ("import json,platform,torch,numpy as np; "
        "print(json.dumps({'checkpoint_runtime':{'python':platform.python_version(),"
        "'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,"
        "'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},"
        "'default_dtype':str(torch.get_default_dtype()),'cuda_initialized':torch.cuda.is_initialized()},allow_nan=False))")
    with tempfile.TemporaryDirectory(prefix="learning-runtime-cpu-") as cache:
        command = python_command(sys.executable, cache, "-c", script)
        result = subprocess.run(command, env=cpu_environment(source), capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError("CPU runtime identity failed: " + result.stderr[-4000:])
    value = json.loads(result.stdout)
    executable = Path(sys.executable).resolve()
    value["executable"] = {"requested": str(Path(sys.executable).absolute()), "resolved": str(executable),
        "sha256": control.file_sha(executable), "device": executable.stat().st_dev, "inode": executable.stat().st_ino}
    if value["cuda_initialized"] is not False or value["default_dtype"] != "torch.float32":
        raise ValueError("CPU runtime preflight initialized CUDA or changed the float32 recipe")
    return value


def validate_prepared(prepared, source):
    """The preparation validator authenticates inputs, never runtime outputs."""
    script = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
              "from transformer_rl.learning_study import validate_learning_study; "
              "print(json.dumps(validate_learning_study(sys.argv[2]),allow_nan=False))")
    with tempfile.TemporaryDirectory(prefix="learning-inputs-cpu-") as cache:
        process = subprocess.run(python_command(sys.executable, cache, "-c", script,
            str(plain(source) / "src"), str(prepared)), env=cpu_environment(source),
            capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise ValueError("frozen CPU preparation validator rejected inputs: " + process.stderr[-4000:])
    return json.loads(process.stdout)


def prepared_identity(root, source, expected_sha, *, preflight=False):
    root, source = plain(root), plain(source)
    manifest = read(root / "manifest.json")
    if (manifest.get("format") != "transformer_rl.learning_rate_study"
            or manifest.get("schema_version") != 1 or manifest.get("sha256") != diagnostic.sha(expected_sha)
            or control.digest({k: v for k, v in manifest.items() if k != "sha256"}) != expected_sha):
        raise ValueError("prepared manifest identity differs")
    protocol = manifest["protocol"]
    if (protocol["expected_training_jobs"] != 90 or protocol["updates_per_job"] != UPDATES
            or protocol["planned_transitions_per_job"] != TOTAL
            or protocol["planned_total_transitions"] != TOTAL * 90
            or protocol["rollout_steps"] != 48 or protocol["num_envs"] != 1024
            or protocol["retention_coef"] != 0. or len(protocol["variants"]) != 10
            or len(set(protocol["variants"])) != 10 or len(protocol["training_seeds"]) != 3
            or len(set(protocol["training_seeds"])) != 3 or len(protocol["learning_rates"]) != 3
            or protocol["development"]["validation_seeds"] != DEVELOPMENT_SEEDS[:2]
            or protocol["development"]["packed_final_seeds"] != DEVELOPMENT_SEEDS[2:]
            or len(protocol["evaluation"]["case_names"]) != 50
            or len(set(protocol["evaluation"]["case_names"])) != 50):
        raise ValueError("complete paired 90-cell protocol is required")
    if learner_identity(source) != manifest["source"]:
        raise ValueError("actual learner source differs from prepared source")
    initializations = {(item["variant"], item["training_seed"]): item for item in manifest["initializations"]}
    if len(initializations) != 30 or len(manifest["initializations"]) != 30:
        raise ValueError("thirty distinct complete initial models are required")
    configs, expected_cells, children = {}, [], {}
    for index, child in enumerate(manifest["children"]):
        name = f"rate_{index:03d}"
        if child["id"] != name or child["learning_rate"] != protocol["learning_rates"][index]:
            raise ValueError("child rate identity differs")
        packed = inside(root, child["root"])
        if inventory(packed / "jobs"):
            raise ValueError("prepared jobs must remain empty and read-only")
        plan = control.load_plan(packed)
        if plan["source"] != manifest["source"] or plan["spec"]["environment_factory"] != FACTORY:
            raise ValueError("child plan learner/factory differs")
        children[name] = {"root": str(packed), "plan": artifact(packed / "plan.json"), "spec": plan["spec"]}
        for receipt in child["configurations"]:
            path = inside(root, receipt["path"])
            configured = read(path)
            if (control.file_sha(path) != receipt["sha256"]
                    or control.digest(configured) != receipt["canonical_sha256"]):
                raise ValueError("prepared configuration identity differs")
            configs[receipt["path"]] = configured
        for variant in protocol["variants"]:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", variant):
                raise ValueError("unsafe architecture name")
            for seed in protocol["training_seeds"]:
                init = initializations[variant, seed]
                diagnostic.sha(init["initial_model_sha256"])
                route = f"{name}/study/configs/{variant}.train.transfer.json"
                configured = configs[route]
                if (configured["ppo"]["learning_rate"] != child["learning_rate"]
                        or configured["environment"]["num_envs"] != 1024
                        or configured["control"]["policy_dt_s"] != .01
                        or configured["ppo"]["epochs"] != 5 or configured["ppo"]["num_minibatches"] != 32
                        or control.digest(configured["model"]) != init["model_sha256"]):
                    raise ValueError("training recipe/model initialization binding differs")
                expected_cells.append({"rate_id": name, "learning_rate": child["learning_rate"], "variant": variant,
                    "training_seed": seed, "job": f"{name}/study/jobs/{variant}/seed_{seed}",
                    "training_config": {"path": route, "sha256": control.file_sha(root / route),
                                        "canonical_sha256": control.digest(configured)},
                    "initial_model_sha256": init["initial_model_sha256"], "updates": UPDATES,
                    "rollout_steps": 48, "num_envs": 1024, "planned_transitions": TOTAL})
    if len(children) != 3 or manifest["cells"] != expected_cells or len(expected_cells) != 90:
        raise ValueError("complete paired 90-cell coverage differs")
    for route in configs:
        name, suffix = route.split("/", 1)
        normalized = json.loads(json.dumps(configs[route]))
        del normalized["ppo"]["learning_rate"]
        if name != "rate_000":
            reference = json.loads(json.dumps(configs["rate_000/" + suffix]))
            del reference["ppo"]["learning_rate"]
            if normalized != reference:
                raise ValueError("paired configurations differ beyond the learning rate")
    if preflight:
        validated = validate_prepared(root, source)
        if validated["sha256"] != expected_sha or validated["status"] != "prepared":
            raise ValueError("preparation validator did not authenticate this manifest")
    return {"root": str(root), "manifest": artifact(root / "manifest.json"), "sha256": expected_sha,
            "files": inventory(root), "source": manifest["source"], "protocol": protocol,
            "children": children, "cells": expected_cells}


def dependency_definition(name, item, resource):
    if name not in DEPENDENCIES or set(item) != {"definition", "sha256", "summary", "controller"}:
        raise ValueError("three explicitly named dependency definitions and handles are required")
    definition, summary = plain(item["definition"]), plain(item["summary"])
    if summary.parent != definition.parent:
        raise ValueError("dependency summary must belong to its definition root")
    value = read(definition)
    format_name, summary_field = DEPENDENCIES[name]
    identity = value.get("sha256") if name == "diagnostics" else control.digest(value)
    if (value.get("format") != format_name or identity != diagnostic.sha(item["sha256"])
            or plain(value["resource_lock"]) != resource):
        raise ValueError("dependency definition/resource identity differs")
    if name == "diagnostics" and control.digest({k: v for k, v in value.items() if k != "sha256"}) != identity:
        raise ValueError("diagnostic dependency manifest self identity differs")
    handle = item["controller"]
    if (set(handle) != {"pid", "start"} or type(handle["pid"]) is not int or handle["pid"] <= 0
            or not isinstance(handle["start"], str) or not handle["start"].isdigit()):
        raise ValueError("dependency controller requires a real PID/start handle")
    return {"definition": artifact(definition), "sha256": identity, "summary": str(summary),
            "summary_identity_field": summary_field, "controller": handle, "root": str(definition.parent),
            "immutable_inputs": dependency_inputs(value)}


def dependency_inputs(definition):
    """Bind declared predecessor sources/configs without freezing live summaries."""
    source = definition["source"]
    root = plain(definition.get("source_root", source.get("root")))
    actual_files = {str(path.relative_to(root)): control.file_sha(path)
                    for path in sorted((root / "src/transformer_rl").rglob("*.py"))}
    if actual_files != source["files"]:
        raise ValueError("predecessor complete Python source inventory changed")
    files = {}
    for route, expected in source["files"].items():
        path = inside(root, route)
        if control.file_sha(path) != diagnostic.sha(expected):
            raise ValueError("predecessor learner source changed")
        files[str(path)] = expected
    if not files:
        raise ValueError("predecessor learner source inventory is absent")
    for route, expected in definition.get("controllers", {}).items():
        path = plain(route)
        if control.file_sha(path) != diagnostic.sha(expected):
            raise ValueError("predecessor controller source changed")
        files[str(path)] = expected
    if "study_root" in definition:
        study = plain(definition["study_root"])
        files[str(study / "plan.json")] = control.file_sha(study / "plan.json")
        for receipt in definition.get("configs", {}).values():
            path = inside(study, receipt["path"])
            if control.file_sha(path) != diagnostic.sha(receipt["sha256"]):
                raise ValueError("predecessor configuration changed")
            files[str(path)] = receipt["sha256"]
    if "manifest_file_sha256" in definition:
        path = plain(definition["manifest"])
        if control.file_sha(path) != definition["manifest_file_sha256"]:
            raise ValueError("predecessor curriculum inputs changed")
        files[str(path)] = definition["manifest_file_sha256"]
    return files


def workers_in(roots):
    result = []
    paths = {path for root in roots for path in plain(root).rglob("*.process.json")}
    for path in sorted(paths):
        value = read(path)
        if value.get("status") in {"launching", "running"} and (value.get("pid") is None or value.get("start") is None):
            result.append({"receipt": str(path), "reason": "worker ownership is unresolved"})
        elif value.get("pid") is not None and value.get("start") is not None:
            if control.process_start(value["pid"]) == str(value["start"]):
                result.append({"receipt": str(path), "pid": value["pid"], "start": value["start"]})
    return result


def _verify_artifacts(value, roots):
    """Check sealed relative artifacts against their known dependency roots."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            path = Path(value["path"])
            candidates = [plain(path)] if path.is_absolute() else [inside(root, path) for root in roots]
            matches = [p for p in candidates if p.is_file() and control.file_sha(p) == diagnostic.sha(value["sha256"])]
            if len(matches) != 1:
                raise ValueError("dependency sealed artifact is missing, changed, or ambiguous")
        else:
            for item in value.values():
                _verify_artifacts(item, roots)
    elif isinstance(value, list):
        for item in value:
            _verify_artifacts(item, roots)


def _dependency_suite(receipt, cases, seed, checkpoint, update, root):
    worker = receipt.get("worker", {})
    if (receipt.get("status") != "completed" or set(receipt.get("artifacts", {})) != set(cases) | {"control", "trace"}
            or worker.get("status") != "finished" or worker.get("returncode") != 0 or worker.get("timed_out")):
        raise ValueError("dependency evaluation does not prove complete case coverage and a terminal worker")
    for case in cases:
        item = receipt["artifacts"][case]
        path = inside(root, item["path"])
        report = read(path)
        if (report.get("checkpoint_sha256") != checkpoint or report.get("checkpoint_update") != update
                or report.get("seed") != seed or report.get("steps") != 4001
                or report.get("num_envs") != 8 or report.get("transitions") != 32008):
            raise ValueError("dependency case checkpoint/seed/sample coverage differs")
    suite = read(inside(root, receipt["artifacts"]["control"]["path"]))
    if (suite.get("checkpoint_sha256") != checkpoint or suite.get("checkpoint_update") != update
            or suite.get("seed") != seed or suite.get("steps") != 4001 or set(suite.get("groups", {})) != set(cases)):
        raise ValueError("dependency control suite identity or groups differ")


def dependency_state(dependency):
    definition = read(checked(dependency["definition"]))
    if dependency_inputs(definition) != dependency["immutable_inputs"]:
        raise ValueError("predecessor source or configuration identity changed")
    summary_path = Path(dependency["summary"])
    handle = dependency["controller"]
    controller_live = control.process_start(handle["pid"]) == handle["start"]
    if not summary_path.exists():
        return {"ready": False, "reason": "completion summary is absent", "controller_live": controller_live}
    summary = read(summary_path)
    if summary.get(dependency["summary_identity_field"]) != dependency["sha256"]:
        raise ValueError("dependency summary belongs to another sealed campaign")
    live = workers_in([dependency["root"]])
    value = {"ready": False, "status": summary.get("status"), "summary": artifact(summary_path),
             "controller_live": controller_live, "live_workers": live}
    if summary.get("status") != "completed" or controller_live or live:
        return value
    results = summary.get("results", {})
    kind = definition["format"]
    expected_count = 20 if kind.endswith("frame_diagnostic_campaign") else 10 if kind.endswith("transfer_campaign") else 9
    if len(results) != expected_count or any(item.get("status") != "completed" for item in results.values()):
        raise ValueError("dependency completed summary has incomplete result coverage")
    roots = [Path(dependency["root"])]
    if "study_root" in definition:
        roots.append(plain(definition["study_root"]))
    if kind.endswith("transfer_campaign"):
        plan = control.load_plan(definition["study_root"])
        expected_keys = {f"{v['name']}/seed_{seed}" for v in plan["spec"]["variants"] for seed in plan["spec"]["seeds"]}
        if set(results) != expected_keys or plan["sha256"] != definition["plan_sha256"]:
            raise ValueError("transfer dependency architecture/training-seed identity differs")
        if not summary.get("training_target_completed") or not summary.get("all_last_sealed_endpoints_checked"):
            raise ValueError("transfer dependency training/evaluation completion is unproven")
        for item in results.values():
            if (len(item.get("evaluations", {})) != 2 or item.get("deployment", {}).get("status") != "completed"
                    or any(ev.get("status") != "completed" for ev in item["evaluations"].values())):
                raise ValueError("transfer dependency did not complete both seeds and deployment checks")
            training = item["training"]
            checkpoint = training["checkpoint"]
            if (training.get("sealed_updates") != UPDATES or training.get("batch_samples") != TOTAL
                    or training.get("ppo_verified") is not True or checkpoint.get("update") != UPDATES
                    or control.file_sha(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]
                    or set(item["evaluations"]) != {str(seed) for seed in definition["evaluation_seeds"]}
                    or not {"manifest", "benchmark"}.issubset(item["deployment"].get("artifacts", {}))):
                raise ValueError("transfer dependency actual training/checkpoint/evaluation evidence differs")
            for seed in definition["evaluation_seeds"]:
                _dependency_suite(item["evaluations"][str(seed)], plan["spec"]["stages"][0]["scenarios"], seed,
                                  checkpoint["checkpoint_sha256"], UPDATES, dependency["root"])
    elif kind.endswith("curriculum_campaign"):
        original = read(definition["manifest"])
        if control.file_sha(definition["manifest"]) != definition["manifest_file_sha256"] or original["sha256"] != definition["manifest_sha256"]:
            raise ValueError("curriculum dependency manifest changed")
        expected_keys = {f"{arm['name']}/seed_{seed}" for arm in original["arms"] for seed in original["training_seeds"]}
        if set(results) != expected_keys:
            raise ValueError("curriculum dependency arm/training-seed identity differs")
        for item in results.values():
            if (len(item.get("phases", [])) != 2 or any(phase.get("training", {}).get("status") != "completed"
                    or len(phase.get("evaluations", {})) != 2
                    or any(ev.get("status") != "completed" for ev in phase["evaluations"].values()) for phase in item["phases"])):
                raise ValueError("curriculum dependency phase/seed coverage is incomplete")
            arm = next(a for a in original["arms"] if a["name"] == item["arm"])
            for phase, expected_phase in zip(item["phases"], arm["phases"]):
                checkpoint = phase["training"]["checkpoint"]
                update = expected_phase["start_update"] + expected_phase["updates"]
                if (phase["name"] != expected_phase["name"]
                        or phase["training"].get("consumed_updates") != expected_phase["updates"]
                        or not phase["training"].get("attempts") or checkpoint.get("update") != update
                        or control.file_sha(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]
                        or set(phase["evaluations"]) != {str(seed) for seed in original["evaluation"]["seeds"]}):
                    raise ValueError("curriculum dependency actual phase budget/checkpoint/evaluation differs")
                for seed in original["evaluation"]["seeds"]:
                    _dependency_suite(phase["evaluations"][str(seed)], [c["name"] for c in original["scenarios"]],
                                      seed, checkpoint["checkpoint_sha256"], update, dependency["root"])
    else:
        expected_keys = {f"{variant}/seed_{seed}" for variant in definition["inputs"]["variants"] for seed in definition["protocol"]["seeds"]}
        if set(results) != expected_keys:
            raise ValueError("diagnostic dependency architecture/evaluation-seed identity differs")
        for variant in definition["inputs"]["variants"]:
            checkpoint = definition["inputs"]["checkpoints"][variant]
            for seed in definition["protocol"]["seeds"]:
                receipt = results[f"{variant}/seed_{seed}"]
                _dependency_suite(receipt, definition["inputs"]["cases"], seed,
                                  checkpoint["checkpoint_sha256"], UPDATES, dependency["root"])
    _verify_artifacts(results, roots)
    value["ready"] = True
    return value


def prepare(args):
    prepared, source, root, resource, study_lock = [plain(getattr(args, name)) for name in
        ("prepared_root", "source_root", "output_root", "resource_lock", "study_lock")]
    if not resource.is_file() or not study_lock.is_file() or resource == study_lock:
        raise ValueError("two distinct existing resource and study locks are required")
    inputs = prepared_identity(prepared, source, args.prepared_manifest_sha256, preflight=True)
    runtime = runtime_identity(source)
    dependency_file = plain(args.dependencies)
    declared = read(dependency_file)
    if set(declared) != set(DEPENDENCIES):
        raise ValueError("transfer, curriculum and diagnostics dependencies are all required")
    dependencies = {name: dependency_definition(name, item, resource) for name, item in declared.items()}
    transfer_definition = read(checked(dependencies["transfer"]["definition"]))
    if "study_root" in transfer_definition and study_lock != plain(Path(transfer_definition["study_root"]) / ".run.lock"):
        raise ValueError("the existing transfer study lock must be shared")
    forbidden = [prepared, source, resource.parent, study_lock.parent,
                 *(Path(item["root"]) for item in dependencies.values())]
    if any(root == path or root.is_relative_to(path) or path.is_relative_to(root) for path in forbidden):
        raise ValueError("campaign outputs must be independent of every immutable input and predecessor")
    if root.exists():
        raise FileExistsError(root)
    definition = {"format": FORMAT, "schema_version": 1, "output_root": str(root), "inputs": inputs,
        "source_root": str(source), "controllers": controllers(), "dependency_file": artifact(dependency_file),
        "dependencies": dependencies, "resource_lock": str(resource), "study_lock": str(study_lock),
        "locks": {str(path): diagnostic.lock_identity(path) for path in (resource, study_lock)},
        "device": args.device, "max_run_attempts": args.max_run_attempts,
        "learner_max_seconds": args.learner_max_seconds, "runtime": runtime,
        "worker_timeout_seconds": args.worker_timeout_seconds, "development_seeds": DEVELOPMENT_SEEDS,
        "training_protocol": dict(TRAINING_PROTOCOL),
        "selection_implemented": False, "confirmation_implemented": False,
        "formal_architecture_selection": False, "hardware_deployment_ready": False}
    if (type(args.max_run_attempts) is not int or args.max_run_attempts < 1
            or type(args.worker_timeout_seconds) not in (int, float)
            or type(args.learner_max_seconds) not in (int, float) or not math.isfinite(args.learner_max_seconds)
            or args.learner_max_seconds <= 0 or not math.isfinite(args.worker_timeout_seconds)
            or args.worker_timeout_seconds <= args.learner_max_seconds):
        raise ValueError("positive attempt limit and worker deadline beyond the learner time budget are required")
    root.mkdir(parents=True, exist_ok=False)
    (root / ".learning.lock").touch(exist_ok=False)
    (root / ".bytecode-cache").mkdir()
    definition["locks"][str(root / ".learning.lock")] = diagnostic.lock_identity(root / ".learning.lock")
    definition["sha256"] = control.digest(definition)
    write(root / "manifest.json", definition)
    return definition


def validate(path):
    path = plain(path)
    manifest = read(path)
    if (manifest.get("format") != FORMAT or manifest.get("schema_version") != 1
            or control.digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest.get("sha256")
            or path != Path(manifest["output_root"]) / "manifest.json"
            or manifest["controllers"] != controllers()):
        raise ValueError("learning controller manifest or frozen helpers changed")
    inputs = manifest["inputs"]
    actual = prepared_identity(inputs["root"], manifest["source_root"], inputs["sha256"])
    if actual != inputs:
        raise ValueError("prepared inputs changed during or after execution")
    if runtime_identity(manifest["source_root"]) != manifest["runtime"]:
        raise ValueError("Python executable or CPU runtime/version identity changed")
    cache = plain(Path(manifest["output_root"]) / ".bytecode-cache")
    if not cache.is_dir() or any(cache.iterdir()):
        raise ValueError("exclusive bytecode cache prefix is no longer empty")
    checked(manifest["dependency_file"])
    declared = read(manifest["dependency_file"]["path"])
    dependencies = {name: dependency_definition(name, item, plain(manifest["resource_lock"])) for name, item in declared.items()}
    if dependencies != manifest["dependencies"] or set(dependencies) != set(DEPENDENCIES):
        raise ValueError("dependency definition changed")
    transfer_definition = read(checked(dependencies["transfer"]["definition"]))
    if ("study_root" in transfer_definition and plain(manifest["study_lock"])
            != plain(Path(transfer_definition["study_root"]) / ".run.lock")):
        raise ValueError("the existing transfer study lock must be shared")
    if manifest["locks"] != {route: diagnostic.lock_identity(route) for route in manifest["locks"]}:
        raise ValueError("existing resource or study lock inode changed")
    if (manifest["development_seeds"] != DEVELOPMENT_SEEDS
            or manifest["training_protocol"] != TRAINING_PROTOCOL
            or type(manifest["max_run_attempts"]) is not int or manifest["max_run_attempts"] < 1
            or type(manifest["worker_timeout_seconds"]) not in (int, float)
            or type(manifest["learner_max_seconds"]) not in (int, float)
            or not math.isfinite(manifest["learner_max_seconds"]) or manifest["learner_max_seconds"] <= 0
            or not math.isfinite(manifest["worker_timeout_seconds"])
            or manifest["worker_timeout_seconds"] <= manifest["learner_max_seconds"]
            or manifest["selection_implemented"] or manifest["confirmation_implemented"]):
        raise ValueError("execution/development protocol differs")
    return manifest


@contextmanager
def acquire_resources(manifest, args, publish):
    started = time.monotonic()
    routes = [manifest["resource_lock"], manifest["study_lock"]]
    with Path(routes[0]).open("r+") as resource, Path(routes[1]).open("r+") as study:
        streams = [resource, study]
        while True:
            for stream in streams:
                diagnostic.check_open_lock(stream, manifest["locks"][stream.name])
            states = {name: dependency_state(item) for name, item in manifest["dependencies"].items()}
            live = workers_in([manifest["output_root"], *(item["root"] for item in manifest["dependencies"].values()),
                               str(Path(manifest["study_lock"]).parent)])
            acquired = []
            if all(item["ready"] for item in states.values()) and not live:
                try:
                    for stream in streams:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired.append(stream)
                except BlockingIOError:
                    pass
            if len(acquired) == len(streams):
                try:
                    after = {name: dependency_state(item) for name, item in manifest["dependencies"].items()}
                    if after == states and not workers_in([manifest["output_root"],
                            *(item["root"] for item in manifest["dependencies"].values()), str(Path(manifest["study_lock"]).parent)]):
                        for stream in streams:
                            diagnostic.check_open_lock(stream, manifest["locks"][stream.name])
                        yield states
                        return
                finally:
                    for stream in reversed(acquired):
                        fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                for stream in reversed(acquired):
                    fcntl.flock(stream, fcntl.LOCK_UN)
            remaining = args.max_wait_seconds - (time.monotonic() - started)
            blockers = {"dependencies": states, "live_workers": live, "resource_or_study_lock_unavailable": True}
            publish("waiting", blockers=blockers, active=None)
            if remaining <= 0:
                yield None
                return
            time.sleep(min(args.poll_seconds, remaining))


def cell_key(cell):
    return f"{cell['rate_id']}/{cell['variant']}/seed_{cell['training_seed']}"


def run_environment(manifest):
    result = dict(os.environ)
    result.update(PYTHONPATH=str(Path(manifest["source_root"]) / "src"), PYTHONDONTWRITEBYTECODE="1")
    return result


def learner_command(manifest, operation, *arguments):
    command = transfer.command(Path(manifest["source_root"]), operation, *arguments)
    return python_command(manifest["runtime"]["executable"]["requested"],
                          Path(manifest["output_root"]) / ".bytecode-cache", *command[1:])


def worker(command, directory, environment, timeout, publish):
    """Unknown launch ownership is retained, never interpreted as permission to restart."""
    record = {"command": command, "status": "launching", "pid": None, "start": None,
              "started_at": control.now(), "timeout_seconds": timeout}
    write(directory / "worker.process.json", record)
    process = None
    with (directory / "worker.log").open("x") as log:
        try:
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, start=control.process_start(process.pid), status="running")
            write(directory / "worker.process.json", record, replace=True)
            started, heartbeat = time.monotonic(), time.monotonic()
            timed_out = False
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    control.terminate_owned(process, record["start"])
                    break
                if time.monotonic() >= heartbeat:
                    publish()
                    heartbeat = time.monotonic() + 30
                time.sleep(.2)
            record.update(status="finished", returncode=process.poll(), timed_out=timed_out, finished_at=control.now())
            write(directory / "worker.process.json", record, replace=True)
            return record
        except BaseException:
            if process is not None:
                control.terminate_owned(process, record["start"])
                record.update(status="finished", returncode=process.poll(), interrupted=True, finished_at=control.now())
            else:
                record.update(status="launch_failed", finished_at=control.now())
            write(directory / "worker.process.json", record, replace=True)
            raise


def training_request(manifest, cell, directory, consumed, parent):
    config_path = inside(manifest["inputs"]["root"], cell["training_config"]["path"])
    config = read(config_path)
    remaining = UPDATES - consumed
    command = learner_command(manifest, "train", "--config", config_path,
        "--env-factory", FACTORY, "--run-dir", directory / "train", "--updates", remaining,
        "--rollout-steps", 48, "--seed", cell["training_seed"], "--device", manifest["device"],
        "--max-seconds", manifest["learner_max_seconds"], "--checkpoint-interval", UPDATES,
        "--consumed-update-offset", consumed, "--retention-coef", 0.)
    if parent is None:
        command.extend(["--expected-initial-model-sha256", cell["initial_model_sha256"]])
    else:
        command.extend(["--resume", parent["path"]])
    return {"manifest_sha256": manifest["sha256"], "cell": cell_key(cell), "seed": cell["training_seed"],
        "source": manifest["inputs"]["source"], "environment_factory": FACTORY, "config": config,
        "runtime": manifest["runtime"],
        "config_sha256": control.digest(config), "initial_model_sha256": cell["initial_model_sha256"],
        "parent": parent, "start_update": consumed, "prior_transitions": consumed * SAMPLES,
        "updates": remaining, "reserved_samples": remaining * SAMPLES, "command": command}


_CHECKPOINT_SCRIPT = r'''
import hashlib, json, sys
sys.path.insert(0,sys.argv[1])
import torch
from transformer_rl.frame_checkpoint import load_frame_checkpoint
from transformer_rl.frame_workflow import _model_state_sha256
from transformer_rl.ppo import PPOTrainer
def encode(value):
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu().contiguous()
        return {"tensor":str(value.dtype),"shape":list(value.shape),"sha256":hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value,dict):
        return {"dict":[[encode(k),encode(v)] for k,v in sorted(value.items(),key=lambda p:(type(p[0]).__name__,str(p[0])))]}
    if isinstance(value,(list,tuple)):
        return {type(value).__name__:[encode(v) for v in value]}
    return value
def digest(value):
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
model,trainer,config,update,metadata,rng=load_frame_checkpoint(sys.argv[2],device="cpu")
state=trainer.optimizer.state_dict()
steps=[float(v["step"]) for v in state["state"].values()]
groups=[{k:v for k,v in group.items() if k!="params"} for group in state["param_groups"]]
reference=PPOTrainer(model,config.ppo).optimizer.state_dict()
reference_groups=[{k:v for k,v in group.items() if k!="params"} for group in reference["param_groups"]]
print(json.dumps({"checkpoint_sha256":hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(),
 "config":config.to_dict(),"update":update,"metadata":metadata,
 "model_sha256":_model_state_sha256(model),"optimizer_sha256":digest(state),"rng_sha256":digest(rng),
 "optimizer_groups":groups,"optimizer_state_count":len(steps),
 "optimizer_parameter_count":sum(len(group["params"]) for group in state["param_groups"]),
 "optimizer_recipe_matches_fresh":groups==reference_groups,
 "optimizer_step_min":min(steps) if steps else None,"optimizer_step_max":max(steps) if steps else None,
 "rng_validated":True,"model_validated":True,"optimizer_validated":True,
 "cuda_initialized":torch.cuda.is_initialized()},allow_nan=False))
'''


def checkpoint_probe(manifest, path):
    with tempfile.TemporaryDirectory(prefix="learning-checkpoint-cpu-") as cache:
        result = subprocess.run(python_command(sys.executable, cache, "-c", _CHECKPOINT_SCRIPT,
            str(Path(manifest["source_root"]) / "src"), str(path)), env=cpu_environment(manifest["source_root"]),
            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError("CPU full-learning-state validation failed: " + result.stderr[-4000:])
    return json.loads(result.stdout)


def optimization_records(path, start, *, strict=True):
    path = Path(path)
    if not path.exists():
        return {"updates": 0, "fresh_samples": 0, "optimizer_steps": 0, "optimization_samples": 0,
                "early_stop_updates": 0, "first_kl": None, "final_kl": None, "complete": False}
    rows, fresh, steps, optimized, early = [], 0, 0, 0, 0
    complete = True
    for line in path.read_text().splitlines(keepends=True):
        try:
            if not line.endswith("\n"):
                raise ValueError("unsealed final metric line")
            row = json.loads(line)
            json.dumps(row, allow_nan=False)
            optimization, collection = row["optimization"], row["collection"]
            size, applied, count = row["batch_samples"], optimization["optimizer_steps"], optimization["sample_count"]
            if (type(row["update"]) is not int or row["update"] != start + len(rows) + 1
                    or type(size) is not int or size <= 0 or type(applied) is not int or not 0 < applied <= 160
                    or type(count) is not int or count <= 0
                    or type(collection["vector_steps"]) is not int or collection["vector_steps"] <= 0
                    or any(type(collection[key]) is not int for key in ("transitions", "total_steps", "total_transitions"))
                    or collection["transitions"] != size or size != collection["vector_steps"] * 1024
                    or collection["total_transitions"] != fresh + size
                    or collection["total_steps"] != (fresh + size) // 1024
                    or optimization["planned_optimizer_steps"] != 160
                    or type(optimization["planned_optimizer_steps"]) is not int
                    or type(optimization["early_stopped"]) is not bool
                    or type(collection["early_stopped"]) is not bool
                    or any(type(optimization[key]) not in (float, int) or not math.isfinite(optimization[key])
                           for key in ("first_step_kl", "final_kl"))):
                raise ValueError("invalid applied PPO optimization/sample record")
            if count != applied * (size // 32):
                raise ValueError("recorded PPO repeated endpoints differ from minibatch sizes")
            if strict and (size != SAMPLES or collection["vector_steps"] != 48 or collection["early_stopped"]):
                raise ValueError("short rollout or repeated PPO sample budget differs")
            if strict and (optimization["early_stopped"] != (applied < 160)):
                # The frozen PPO marks a stop only after a KL-triggered skipped minibatch.
                raise ValueError("PPO early-stop flag differs from actual optimizer steps")
            rows.append(row)
            fresh += size
            steps += applied
            optimized += count
            early += int(optimization["early_stopped"])
        except (ValueError, KeyError, TypeError) as error:
            if strict:
                raise ValueError(f"optimization ledger rejected row {len(rows) + 1}: {error}") from error
            complete = False
            break
    return {"updates": len(rows), "fresh_samples": fresh, "optimizer_steps": steps,
            "optimization_samples": optimized, "early_stop_updates": early,
            "first_kl": rows[0]["optimization"]["first_step_kl"] if rows else None,
            "final_kl": rows[-1]["optimization"]["final_kl"] if rows else None, "complete": complete}


def training_evidence(manifest, request, directory, parent_evidence=None):
    root = Path(manifest["output_root"])
    run, completion = read(directory / "train/run.json"), read(directory / "train/completion.json")
    parent = request["parent"]
    process = read(directory / "worker.process.json")
    if (process.get("command") != request["command"] or process.get("status") not in {"finished", "running"}
            or type(process.get("pid")) is not int or process["pid"] <= 0
            or not isinstance(process.get("start"), str) or not process["start"].isdigit()
            or control.process_start(process["pid"]) == process["start"]
            or process["status"] == "finished" and type(process.get("returncode")) is not int):
        raise ValueError("actual worker command/clock or original terminal handle is unverified")
    if request["runtime"] != manifest["runtime"]:
        raise ValueError("training request changed its executable/runtime identity")
    expected = {"config": request["config"], "seed": request["seed"], "environment_factory": FACTORY,
        "updates": request["updates"], "rollout_steps": 48, "device": manifest["device"], "source": request["source"],
        "resume": parent["path"] if parent else None, "initialize_from": None, "restore_learning_from": None,
        "episode_state_restored": False, "history_reset": "repeat_first", "retention_coef": 0.,
        "initial_model_hash_format": "sorted_named_tensor_contents_v1", "max_seconds": manifest["learner_max_seconds"], "checkpoint_interval": UPDATES}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("actual worker run config/source/factory/seed/initialization differs")
    if parent is None:
        guard = {"expected_sha256": request["initial_model_sha256"],
                 "actual_sha256": request["initial_model_sha256"], "verified": True}
        if run.get("initialization_guard") != guard or run.get("initial_model_sha256") != request["initial_model_sha256"]:
            raise ValueError("fresh complete initial model SHA guard was not verified")
    else:
        checked(parent)
        if parent_evidence is None or run.get("initialization_guard") is not None:
            raise ValueError("resume requires an audited full-learning-state predecessor, without a fresh guard")
        if run.get("initial_model_sha256") != parent_evidence["model_sha256"]:
            raise ValueError("resume did not load the previous complete model including fixed buffers")
    evidence = optimization_records(directory / "train/metrics.jsonl", request["start_update"])
    count = evidence["updates"]
    if (any(type(completion[key]) is not int for key in ("completed_updates", "attempted_updates", "start_update",
             "final_update", "consumed_transitions", "cumulative_transitions"))
            or count == 0 or count > request["updates"] or completion["completed_updates"] != count
            or completion["attempted_updates"] != count or completion["start_update"] != request["start_update"]
            or completion["final_update"] != request["start_update"] + count
            or completion["consumed_transitions"] != evidence["fresh_samples"]
            or completion["cumulative_transitions"] != request["prior_transitions"] + count * SAMPLES
            or completion["config_sha256"] != request["config_sha256"]):
        raise ValueError("sealed completion does not prove the exact successful-update/fresh-sample continuum")
    if (completion["status"] not in {"completed", "stopped"}
            or completion["status"] == "completed" and count != request["updates"]
            or completion["status"] == "stopped" and completion.get("stop_reason") not in {"time_budget", "SIGINT", "SIGTERM"}):
        raise ValueError("unrecognized or inconsistent completion endpoint")
    checkpoint = inside(root, completion["checkpoint"])
    if checkpoint != directory / "train/checkpoints/final.pt" or control.file_sha(checkpoint) != completion["checkpoint_sha256"]:
        raise ValueError("sealed checkpoint SHA or output path differs")
    sidecar = read(str(checkpoint) + ".json")
    probe = checkpoint_probe(manifest, checkpoint)
    metadata = probe["metadata"]
    environment = read(directory / "train/environment.json")
    population = read(Path(manifest["inputs"]["root"]) / "parent/transfer_design.json")["scene_allocation"]["requested_counts"]
    expected_metadata = {"environment_factory": FACTORY, "environment_provenance": environment,
        "seed": request["seed"], "source": request["source"], "anchors": [], "retention_coef": 0.,
        "initial_model_sha256": run["initial_model_sha256"], "initial_model_hash_format": "sorted_named_tensor_contents_v1",
        "episode_state_restored": False, "collected_transitions": completion["cumulative_transitions"]}
    if (any(metadata.get(key) != value for key, value in expected_metadata.items())
            or metadata.get("runtime") != manifest["runtime"]["checkpoint_runtime"]
            or metadata.get("initialization_guard") != run.get("initialization_guard")
            or environment.get("identity") != request["config"]["environment"]["snapshot_sha256"]
            or environment.get("control_sha256") != control.digest(request["config"]["control"])
            or environment.get("contract_sha256") != request["config"]["environment"]["contract_sha256"]
            or environment.get("startup", {}).get("scene_group_counts") != population
            or environment.get("policy_hz") != 100. or environment.get("physics_hz") != 200.
            or probe["update"] != completion["final_update"] or probe["config"] != request["config"]
            or probe["checkpoint_sha256"] != completion["checkpoint_sha256"]
            or sidecar.get("sha256") != completion["checkpoint_sha256"] or sidecar.get("config") != probe["config"]
            or sidecar.get("metadata") != metadata or sidecar.get("update") != probe["update"]
            or sidecar.get("format") != "transformer_rl.packed_checkpoint" or sidecar.get("schema_version") != 1
            or any(probe.get(key) is not True for key in ("rng_validated", "model_validated", "optimizer_validated"))
            or probe.get("cuda_initialized") is not False):
        raise ValueError("checkpoint/sidecar/model/Adam/RNG/provenance validation differs")
    prior_steps = parent_evidence["cumulative_optimizer_steps"] if parent_evidence else 0
    cumulative_steps = prior_steps + evidence["optimizer_steps"]
    groups = probe["optimizer_groups"]
    if (len(groups) != 1 or groups[0].get("lr") != request["config"]["ppo"]["learning_rate"]
            or probe.get("optimizer_recipe_matches_fresh") is not True
            or groups[0].get("betas") != [.9, .999] or groups[0].get("eps") != 1e-8
            or groups[0].get("weight_decay") != 0 or groups[0].get("amsgrad") is not False
            or probe["optimizer_state_count"] <= 0 or probe["optimizer_state_count"] != probe["optimizer_parameter_count"]
            or probe["optimizer_step_min"] != cumulative_steps
            or probe["optimizer_step_max"] != cumulative_steps):
        raise ValueError("actual Adam learning rate, hyperparameters or optimizer-step continuity differs")
    return {**evidence, "cumulative_optimizer_steps": cumulative_steps,
        "final_update": completion["final_update"], "cumulative_transitions": completion["cumulative_transitions"],
        "checkpoint": artifact(checkpoint), "model_sha256": probe["model_sha256"],
        "optimizer_sha256": probe["optimizer_sha256"], "rng_sha256": probe["rng_sha256"],
        "full_learning_state_validated": True, "stop_reason": completion.get("stop_reason"),
        "status": "completed" if count == request["updates"] else "resumable"}


def seal_training(manifest, request, directory, parent_evidence=None):
    path = directory / "receipt.json"
    if workers_in([directory]):
        return {"status": "waiting", "reason": "a verified or unresolved original worker is still present"}
    if path.exists():
        receipt = read(path)
        if receipt["request_sha256"] != control.file_sha(directory / "request.json"):
            raise ValueError("sealed training request changed")
        for item in receipt["artifacts"].values():
            checked(item)
        if receipt["status"] in {"completed", "resumable"}:
            if training_evidence(manifest, request, directory, parent_evidence) != receipt["evidence"]:
                raise ValueError("previous sealed runtime evidence no longer validates")
        elif receipt["status"] == "incomplete":
            if (receipt["charged_updates"] != request["updates"]
                    or receipt["charged_samples"] != request["reserved_samples"]
                    or receipt["verified_actual"] != optimization_records(directory / "train/metrics.jsonl", request["start_update"], strict=False)):
                raise ValueError("failed reservation accounting or verified actual evidence changed")
        else:
            raise ValueError("unknown sealed training status")
        return receipt
    receipt = {"request_sha256": control.file_sha(directory / "request.json"), "status": "incomplete",
        "reserved_updates": request["updates"], "charged_updates": request["updates"],
        "charged_samples": request["reserved_samples"], "artifacts": {}, "finished_at": control.now()}
    try:
        evidence = training_evidence(manifest, request, directory, parent_evidence)
        receipt.update(status=evidence["status"], charged_updates=evidence["updates"],
                       charged_samples=evidence["fresh_samples"], evidence=evidence,
                       verified_actual=evidence, accounting="sealed_exact_full_rollout_endpoint")
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        receipt.update(error=f"{type(error).__name__}: {error}",
            verified_actual=optimization_records(directory / "train/metrics.jsonl", request["start_update"], strict=False),
            accounting="entire_reservation_charged_not_actual_consumption_no_refund")
    for name in ("run.json", "environment.json", "completion.json", "failure.json", "metrics.jsonl"):
        target = directory / "train" / name
        if target.exists():
            receipt["artifacts"][name] = artifact(target)
    if receipt["status"] != "incomplete":
        checkpoint = receipt["evidence"]["checkpoint"]["path"]
        receipt["artifacts"].update(checkpoint=artifact(checkpoint), sidecar=artifact(str(checkpoint) + ".json"))
    process = directory / "worker.process.json"
    if process.exists():
        receipt["artifacts"]["worker"] = artifact(process)
    write(path, receipt)
    return receipt


def train_cell(manifest, cell, publish):
    root = Path(manifest["output_root"]) / "cells" / cell_key(cell) / "training"
    completed, charged, parent, prior, attempts = 0, 0, None, None, []
    directories = sorted(root.glob("attempt_*"))
    while completed < UPDATES:
        if len(attempts) >= manifest["max_run_attempts"]:
            return {"status": "incomplete", "reason": "sealed resume attempt limit", "completed_updates": completed,
                    "charged_updates": charged, "verified_fresh_samples": completed * SAMPLES, "attempts": attempts}
        directory = directories[len(attempts)] if len(attempts) < len(directories) else None
        if directory is None:
            validate(Path(manifest["output_root"]) / "manifest.json")
            if workers_in([manifest["output_root"]]):
                return {"status": "waiting", "reason": "another original worker remains alive"}
            directory = transfer.next_attempt(root)
            request = training_request(manifest, cell, directory, completed, parent)
            write(directory / "request.json", request)
            publish("training", active={"cell": cell_key(cell), "directory": str(directory)})
            try:
                worker(request["command"], directory, run_environment(manifest), manifest["worker_timeout_seconds"], publish)
            finally:
                receipt = seal_training(manifest, request, directory, prior)
        else:
            request = read(directory / "request.json")
            if request != training_request(manifest, cell, directory, completed, parent):
                raise ValueError("training request/parent/config/source/clock changed")
            receipt = seal_training(manifest, request, directory, prior)
        if receipt["status"] == "waiting":
            return receipt
        attempts.append(artifact(directory / "receipt.json"))
        charged += receipt["charged_updates"]
        if receipt["status"] == "incomplete":
            if len(directories) > len(attempts):
                raise ValueError("unexpected extra attempt after a terminal incomplete reservation")
            return {"status": "incomplete", "reason": receipt["error"], "completed_updates": completed,
                "charged_updates": charged, "charged_samples": charged * SAMPLES,
                "verified_fresh_samples_before_failed_attempt": completed * SAMPLES,
                "verified_known_prefix_fresh_samples": completed * SAMPLES + receipt["verified_actual"]["fresh_samples"],
                "failed_attempt_verified_actual": receipt["verified_actual"], "attempts": attempts}
        completed += receipt["evidence"]["updates"]
        parent, prior = receipt["evidence"]["checkpoint"], receipt["evidence"]
        publish(active=None)
    if completed != UPDATES or charged != UPDATES or prior["cumulative_transitions"] != TOTAL or len(directories) > len(attempts):
        raise ValueError("final cell budget/update/sample/attempt continuum differs")
    return {"status": "completed", "completed_updates": completed, "charged_updates": charged,
        "verified_fresh_samples": TOTAL, "checkpoint": parent, "learning_state": prior,
        "initial_model_sha256": cell["initial_model_sha256"], "attempts": attempts}


def evaluation_inputs(manifest, cell, checkpoint):
    prepared = Path(manifest["inputs"]["root"])
    cases = manifest["inputs"]["protocol"]["evaluation"]["case_names"]
    configs = {case: prepared / f"{cell['rate_id']}/study/configs/{cell['variant']}.eval.{case}.json" for case in cases}
    environment = {case: read(path)["environment"] for case, path in configs.items()}
    ctl = read(configs[cases[0]])["control"]
    shim = {"output_root": manifest["output_root"], "inputs": {"cases": cases,
        "environments": {cell["variant"]: environment}, "checkpoints": {cell["variant"]: {
            "checkpoint_sha256": checkpoint["sha256"], "control_sha256": control.digest(ctl)}},
        "snapshots": {environment[cases[0]]["snapshot"]: {"sha256": environment[cases[0]]["snapshot_sha256"]}}}}
    return configs, shim


def evaluate_cell(manifest, cell, training, seed, publish):
    root = Path(manifest["output_root"])
    job = root / "cells" / cell_key(cell) / "development" / f"seed_{seed}"
    checkpoint = training["checkpoint"]
    checked(checkpoint)
    configs, shim = evaluation_inputs(manifest, cell, checkpoint)
    identity = {"manifest_sha256": manifest["sha256"], "cell": cell_key(cell), "evaluation_seed": seed,
                "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": UPDATES, "use": "development_only"}
    existing = sorted(job.glob("attempt_*"))
    # Evaluation failure is retained, rather than selecting a favorable retry.
    if existing:
        if len(existing) != 1:
            raise ValueError("development evaluation has undeclared extra attempts")
        directory = existing[0]
        if workers_in([directory]):
            return {"status": "waiting", "reason": "original evaluation worker remains present"}
        if (directory / "receipt.json").exists():
            receipt = read(directory / "receipt.json")
            if receipt["identity"] != identity:
                raise ValueError("development evaluation identity changed")
            for item in receipt.get("artifacts", {}).values():
                control.checked(root, item)
            if receipt["status"] == "completed":
                evaluation_execution(manifest, cell, checkpoint, seed, directory, receipt)
                diagnostic.verify_outputs(directory, shim, cell["variant"], seed)
            return receipt
        request = read(directory / "request.json")
        if request["identity"] != identity:
            raise ValueError("unsealed development request identity changed")
    else:
        validate(root / "manifest.json")
        if workers_in([manifest["output_root"]]):
            return {"status": "waiting", "reason": "a verified or unresolved original worker remains present"}
        directory = transfer.next_attempt(job)
        command = learner_command(manifest, "evaluate-suite", "--checkpoint", checkpoint["path"],
            "--configs", *configs.values(), "--outputs", *(directory / f"{case}.json" for case in configs),
            "--steps", 4001, "--seed", seed, "--device", manifest["device"],
            "--control-output", directory / "control.json", "--trace-output", directory / "trace.npz",
            "--settle-steps", 200, "--min-steady-samples", 200, "--trace-replicas", 2)
        write(directory / "request.json", {"identity": identity, "command": command,
            "configurations": {case: artifact(path) for case, path in configs.items()}})
        publish("evaluating", active={"cell": cell_key(cell), "seed": seed, "directory": str(directory)})
        worker(command, directory, run_environment(manifest), manifest["worker_timeout_seconds"], publish)
    receipt = {"identity": identity, "status": "incomplete", "finished_at": control.now(),
               "artifacts": {}, "directory": str(directory.relative_to(root))}
    try:
        receipt["execution"] = evaluation_execution(manifest, cell, checkpoint, seed, directory)
        receipt["artifacts"] = diagnostic.verify_outputs(directory, shim, cell["variant"], seed)
        receipt["status"] = "completed"
    except (ValueError, KeyError, TypeError, OSError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    write(directory / "receipt.json", receipt)
    return receipt


def evaluation_execution(manifest, cell, checkpoint, seed, directory, receipt=None):
    configs, _ = evaluation_inputs(manifest, cell, checkpoint)
    identity = {"manifest_sha256": manifest["sha256"], "cell": cell_key(cell), "evaluation_seed": seed,
        "checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": UPDATES, "use": "development_only"}
    command = learner_command(manifest, "evaluate-suite", "--checkpoint", checkpoint["path"],
        "--configs", *configs.values(), "--outputs", *(directory / f"{case}.json" for case in configs),
        "--steps", 4001, "--seed", seed, "--device", manifest["device"],
        "--control-output", directory / "control.json", "--trace-output", directory / "trace.npz",
        "--settle-steps", 200, "--min-steady-samples", 200, "--trace-replicas", 2)
    request = {"identity": identity, "command": command,
        "configurations": {case: artifact(path) for case, path in configs.items()}}
    if read(directory / "request.json") != request:
        raise ValueError("actual development command/configurations changed")
    process = read(directory / "worker.process.json")
    if (process.get("status") != "finished" or type(process.get("returncode")) is not int
            or process["returncode"] != 0 or process.get("timed_out", False) or process.get("command") != command
            or type(process.get("pid")) is not int or process["pid"] <= 0
            or not isinstance(process.get("start"), str) or not process["start"].isdigit()
            or control.process_start(process["pid"]) == process["start"]):
        raise ValueError("development worker command or successful terminal handle is unverified")
    execution = {"request": artifact(directory / "request.json"), "worker": artifact(directory / "worker.process.json")}
    if receipt is not None and receipt.get("execution") != execution:
        raise ValueError("development execution seal changed")
    return execution


def audit(path):
    """Read all ninety fixed cells; missing evidence remains explicit and unranked."""
    manifest = validate(path)
    results = {}
    for cell in manifest["inputs"]["cells"]:
        key = cell_key(cell)
        directory = Path(manifest["output_root"]) / "cells" / key
        attempts = sorted((directory / "training").glob("attempt_*"))
        if not attempts:
            results[key] = {"status": "not_started", "training": None, "development": {str(seed): None for seed in DEVELOPMENT_SEEDS}}
            continue
        completed, charged, parent, prior = 0, 0, None, None
        training = {"status": "not_ready"}
        for index, attempt in enumerate(attempts):
            request = read(attempt / "request.json")
            if request != training_request(manifest, cell, attempt, completed, parent):
                raise ValueError("actual request chain differs from the frozen cell")
            if workers_in([attempt]):
                training = {"status": "running", "live_workers": workers_in([attempt])}
                break
            if not (attempt / "receipt.json").exists():
                if len(attempts) != index + 1:
                    raise ValueError("unexpected extra attempt after an unsealed reservation")
                actual = optimization_records(attempt / "train/metrics.jsonl", completed, strict=False)
                training = {"status": "unsealed", "reserved_updates": request["updates"],
                    "charged_updates": charged + request["updates"], "charged_samples": (charged + request["updates"]) * SAMPLES,
                    "completed_updates": completed, "verified_fresh_samples_before_failed_attempt": completed * SAMPLES,
                    "verified_actual": actual, "verified_known_prefix_fresh_samples": completed * SAMPLES + actual["fresh_samples"],
                    "accounting": "unsealed_entire_reservation_conservatively_charged_not_actual_consumption"}
                break
            receipt = read(attempt / "receipt.json")
            if receipt["request_sha256"] != control.file_sha(attempt / "request.json"):
                raise ValueError("runtime ledger request SHA differs")
            for item in receipt["artifacts"].values():
                checked(item)
            charged += receipt["charged_updates"]
            if receipt["status"] == "incomplete":
                if len(attempts) != index + 1:
                    raise ValueError("unexpected extra attempt after a terminal incomplete reservation")
                if receipt["charged_updates"] != request["updates"] or receipt["charged_samples"] != request["reserved_samples"]:
                    raise ValueError("failed reservation was refunded")
                if receipt["verified_actual"] != optimization_records(attempt / "train/metrics.jsonl", request["start_update"], strict=False):
                    raise ValueError("failed attempt actual sample evidence changed")
                training = {"status": "incomplete", "reason": receipt["error"], "charged_updates": charged,
                    "charged_samples": charged * SAMPLES, "completed_updates": completed,
                    "verified_fresh_samples_before_failed_attempt": completed * SAMPLES,
                    "verified_known_prefix_fresh_samples": completed * SAMPLES + receipt["verified_actual"]["fresh_samples"],
                    "verified_actual": receipt["verified_actual"],
                    "accounting": "entire_failed_reservation_charged_not_actual_consumption"}
                break
            if receipt["status"] not in {"completed", "resumable"}:
                raise ValueError("unknown training ledger status")
            evidence = training_evidence(manifest, request, attempt, prior)
            if (evidence != receipt["evidence"] or receipt["charged_updates"] != evidence["updates"]
                    or receipt["charged_samples"] != evidence["fresh_samples"]):
                raise ValueError("runtime ledger PPO/budget evidence changed")
            completed += evidence["updates"]
            parent, prior = evidence["checkpoint"], evidence
            training = {"status": "completed" if completed == UPDATES else "resumable", "completed_updates": completed,
                "charged_updates": charged, "verified_fresh_samples": completed * SAMPLES, "checkpoint": parent}
        development = {str(seed): None for seed in DEVELOPMENT_SEEDS}
        if training["status"] == "completed":
            if completed != UPDATES or charged != UPDATES or prior["cumulative_transitions"] != TOTAL:
                raise ValueError("completed grid cell does not have exact1200/58982400 evidence")
            for seed in DEVELOPMENT_SEEDS:
                candidates = list((directory / "development" / f"seed_{seed}").glob("attempt_*"))
                if len(candidates) > 1:
                    raise ValueError("undeclared evaluation retries cannot replace a failed endpoint")
                if candidates and (candidates[0] / "receipt.json").exists():
                    receipt = read(candidates[0] / "receipt.json")
                    expected = {"manifest_sha256": manifest["sha256"], "cell": key, "evaluation_seed": seed,
                        "checkpoint_sha256": parent["sha256"], "checkpoint_update": UPDATES, "use": "development_only"}
                    if receipt["identity"] != expected:
                        raise ValueError("development evaluation refers to another checkpoint/cell/seed")
                    for item in receipt["artifacts"].values():
                        control.checked(manifest["output_root"], item)
                    if receipt["status"] == "completed":
                        evaluation_execution(manifest, cell, parent, seed, candidates[0], receipt)
                        _, shim = evaluation_inputs(manifest, cell, parent)
                        diagnostic.verify_outputs(candidates[0], shim, cell["variant"], seed)
                    development[str(seed)] = {"status": receipt["status"], "receipt": artifact(candidates[0] / "receipt.json")}
        ready = training["status"] == "completed" and all(item and item["status"] == "completed" for item in development.values())
        results[key] = {"status": "completed" if ready else "not_ready", "training": training, "development": development}
    return {"manifest_sha256": manifest["sha256"], "expected_cells": 90, "actual_cells": len(results),
        "completed_training_cells": sum(item["training"] is not None and item["training"]["status"] == "completed" for item in results.values()),
        "completed_development_cells": sum(item["status"] == "completed" for item in results.values()),
        "status": "development_complete" if all(item["status"] == "completed" for item in results.values()) else "not_ready",
        "cells": results, "selection_implemented": False, "confirmation_implemented": False,
        "formal_architecture_selection": False, "hardware_deployment_ready": False}


def run(args):
    manifest = validate(args.manifest)
    root = Path(manifest["output_root"])
    summary = {"status": "waiting", "manifest_sha256": manifest["sha256"], "started_at": control.now(),
               "results": {}, "expected_cells": 90, "selection_implemented": False, "confirmation_implemented": False}
    def publish(status=None, **values):
        if status:
            summary["status"] = status
        summary.update(values, updated_at=control.now())
        write(root / "summary.json", summary, replace=True)
    with (root / ".learning.lock").open("r+") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic.check_open_lock(own, manifest["locks"][own.name])
        write(root / "controller.json", {"pid": os.getpid(), "start": control.process_start(os.getpid()),
            "status": "running", "manifest_sha256": manifest["sha256"], "started_at": control.now()}, replace=True)
        with acquire_resources(manifest, args, publish) as dependencies:
            if dependencies is None:
                return summary
            validate(args.manifest)
            publish("training", dependencies=dependencies, active=None)
            for cell in manifest["inputs"]["cells"]:
                training = train_cell(manifest, cell, publish)
                result = {"training": training, "development": {str(seed): None for seed in DEVELOPMENT_SEEDS}, "status": "incomplete"}
                summary["results"][cell_key(cell)] = result
                if training["status"] == "waiting":
                    publish("waiting", active=None)
                    return summary
                if training["status"] == "completed":
                    for seed in DEVELOPMENT_SEEDS:
                        result["development"][str(seed)] = evaluate_cell(manifest, cell, training, seed, publish)
                        if result["development"][str(seed)]["status"] == "waiting":
                            publish("waiting", active=None)
                            return summary
                    result["status"] = "completed" if all(item["status"] == "completed" for item in result["development"].values()) else "incomplete"
                publish(active=None)
            report = audit(args.manifest)
            write(root / "audit.json", report, replace=True)
            publish("completed" if report["status"] == "development_complete" else "incomplete", active=None,
                    audit=artifact(root / "audit.json"), grid_status=report["status"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    preparer = operations.add_parser("prepare", help="freeze execution metadata; never start training")
    for name in ("prepared-root", "source-root", "output-root", "resource-lock", "study-lock", "dependencies"):
        preparer.add_argument("--" + name, required=True, type=Path)
    preparer.add_argument("--prepared-manifest-sha256", required=True)
    preparer.add_argument("--device", default="cuda:0")
    preparer.add_argument("--max-run-attempts", type=int, default=8)
    preparer.add_argument("--learner-max-seconds", type=float, default=LEARNER_MAX_SECONDS)
    preparer.add_argument("--worker-timeout-seconds", type=float, default=LEARNER_MAX_SECONDS + 300.)
    for operation in ("validate", "audit", "run"):
        sub = operations.add_parser(operation)
        sub.add_argument("--manifest", type=Path, required=True)
        if operation == "run":
            sub.add_argument("--max-wait-seconds", type=float, default=604800.)
            sub.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        result = prepare(args)
    elif args.operation == "validate":
        result = {"status": "validated", "sha256": validate(args.manifest)["sha256"]}
    elif args.operation == "audit":
        result = audit(args.manifest)
    else:
        if not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0 or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30:
            parser.error("finite nonnegative observation budget and bounded polling are required")
        old = {}
        def interrupted(*_):
            raise KeyboardInterrupt
        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                old[number] = signal.signal(number, interrupted)
            result = run(args)
        except KeyboardInterrupt:
            path = args.manifest.parent / "summary.json"
            if path.exists():
                summary = read(path)
                summary.update(status="interrupted", active=None, updated_at=control.now())
                write(path, summary, replace=True)
            return 130
        finally:
            for number, handler in old.items():
                signal.signal(number, handler)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if args.operation != "run" or result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

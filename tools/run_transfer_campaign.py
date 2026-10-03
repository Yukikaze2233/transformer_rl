"""Queue a frozen transfer study, control evaluation and CPU deployment checks.

The external resource lock belongs to the predecessor study. The new frame
executor owns its separate study lock, so acquiring both cannot self-deadlock.
Rejected or partially trained models retain their last sealed evaluation. No
architecture selection, hardware qualification or TensorBoard server is started.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid


_SPEC = importlib.util.spec_from_file_location(
    "transfer_control_helpers", Path(__file__).with_name("run_frame_control_campaign.py"))
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)


def write(path, value, *, replace=False):
    """Publish complete JSON bytes atomically, without replacing sealed evidence."""
    if replace:
        return control.write(path, value, replace=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command(source, operation, *arguments):
    bootstrap = ("import runpy, sys; sys.path.insert(0, " + repr(str(source / "src"))
                 + "); runpy.run_module('transformer_rl.frame_process', "
                 "run_name='__main__', alter_sys=True)")
    return [sys.executable, "-c", bootstrap, operation, *map(str, arguments)]


def source_identity(source):
    result = control.source_identity(source)
    result["transfer_controller_sha256"] = control.file_sha(__file__)
    return result


def metrics_evidence(path, *, sealed=None):
    """Applied optimization records, not a PID, establish actual PPO progress."""
    path = Path(path)
    if not path.exists():
        if sealed and sealed.get("completed_updates", 0):
            raise ValueError("sealed training has no optimization records")
        return {"updates": 0, "optimizer_steps": 0, "batch_samples": 0, "ppo_verified": False}
    records, steps, samples, last = [], 0, 0, None
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith("\n") and sealed is None and index == len(lines) - 1:
            continue
        row = json.loads(line)
        update, size, optimization = row["update"], row["batch_samples"], row["optimization"]
        applied = optimization["optimizer_steps"]
        if (type(update) is not int or type(size) is not int or size <= 0
                or type(applied) is not int or applied <= 0
                or (last is not None and update != last + 1)
                or not all(type(value) in (float, int, bool) and math.isfinite(value)
                           for value in optimization.values())):
            raise ValueError("invalid applied PPO optimization record")
        records.append(update)
        steps, samples, last = steps + applied, samples + size, update
    if sealed is not None:
        start, final = sealed["start_update"], sealed["final_update"]
        if records != list(range(start + 1, final + 1)) or len(records) != sealed["completed_updates"]:
            raise ValueError("sealed update receipt and PPO records differ")
    return {"updates": len(records), "last_update": last, "optimizer_steps": steps,
            "batch_samples": samples, "ppo_verified": bool(records),
            "metrics_sha256": control.file_sha(path) if sealed is not None else None}


def training_snapshot(study, plan):
    result = {}
    for variant in plan["spec"]["variants"]:
        for seed in plan["spec"]["seeds"]:
            key = f"{variant['name']}/seed_{seed}"
            state_path = study / "jobs" / variant["name"] / f"seed_{seed}" / "state.json"
            item = {"variant": variant["name"], "training_seed": seed, "status": "not_started",
                    "checkpoint": None, "sealed_updates": 0, "optimizer_steps": 0,
                    "batch_samples": 0, "ppo_verified": False, "live_updates": 0,
                    "live_optimizer_steps": 0, "live_batch_samples": 0, "live_ppo_verified": False}
            result[key] = item
            if not state_path.exists():
                continue
            state = control.read(state_path)
            if (state.get("plan_sha256") != plan["sha256"] or state.get("seed") != seed
                    or state.get("variant") != variant["name"]):
                raise ValueError("training job belongs to another frozen plan")
            item.update(status=state["status"], state=control.artifact(state_path, study))
            last_update = 0
            for stage in state.get("stages", []):
                for attempt in stage.get("attempts", []):
                    directory = control.inside(study, attempt["directory"])
                    receipt_path = control.checked(study, attempt["training"]) if "training" in attempt else None
                    if receipt_path is not None and receipt_path.name == "completion.json":
                        receipt = control.read(receipt_path)
                        if not attempt.get("budget_charged"):
                            raise ValueError("sealed checkpoint has no charged budget")
                        if receipt["start_update"] != last_update:
                            raise ValueError("training checkpoints do not form a continuous resume chain")
                        evidence = metrics_evidence(directory / "train/metrics.jsonl", sealed=receipt)
                        checkpoint = control.inside(study, receipt["checkpoint"])
                        if control.file_sha(checkpoint) != receipt["checkpoint_sha256"]:
                            raise ValueError("sealed checkpoint hash differs")
                        if "checkpoint" in attempt and control.checked(study, attempt["checkpoint"]) != checkpoint:
                            raise ValueError("ledger checkpoint differs from the completion")
                        item["checkpoint"] = {"checkpoint": str(checkpoint),
                            "checkpoint_sha256": receipt["checkpoint_sha256"],
                            "update": receipt["final_update"],
                            "completion": control.artifact(receipt_path, study)}
                        for field in ("optimizer_steps", "batch_samples"):
                            item[field] += evidence[field]
                        item["sealed_updates"] += evidence["updates"]
                        item["ppo_verified"] = item["ppo_verified"] or evidence["ppo_verified"]
                        last_update = receipt["final_update"]
                    elif receipt_path is None:
                        evidence = metrics_evidence(directory / "train/metrics.jsonl")
                        item["live_updates"] += evidence["updates"]
                        item["live_optimizer_steps"] += evidence["optimizer_steps"]
                        item["live_batch_samples"] += evidence["batch_samples"]
                        item["live_ppo_verified"] = item["live_ppo_verified"] or evidence["ppo_verified"]
            item["checkpoint_update"] = last_update
    return result


def dependency_ready(path):
    if not path.exists():
        return {"ready": False, "reason": "predecessor receipt has not been written"}
    try:
        receipt = control.read(path)
    except (OSError, ValueError) as error:
        return {"ready": False, "reason": str(error)}
    return {"ready": receipt.get("status") == "completed", "status": receipt.get("status"),
            "path": str(path), "sha256": control.file_sha(path)}


@contextmanager
def acquire_resource(args, publish):
    started = time.monotonic()
    with args.resource_lock.open("a") as resource:
        while True:
            dependency = dependency_ready(args.dependency_receipt)
            blockers = {"dependency": dependency,
                "predecessor_workers": control.active_study_workers(args.resource_lock.parent),
                "predecessor_control_workers": control.active_campaign_workers(args.dependency_receipt.parent),
                "study_workers": control.active_study_workers(args.study_root),
                "previous_transfer_workers": control.active_campaign_workers(args.output_root)}
            locked = False
            if dependency["ready"] and not any(blockers[name] for name in blockers if name != "dependency"):
                try:
                    fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    pass
            blockers["resource_lock_held"] = not locked
            if locked:
                # Recheck after acquisition: the receipt and process scan preceded the lock.
                if (dependency_ready(args.dependency_receipt) == dependency
                        and not control.active_study_workers(args.resource_lock.parent)
                        and not control.active_campaign_workers(args.dependency_receipt.parent)):
                    yield dependency
                    return
                fcntl.flock(resource, fcntl.LOCK_UN)
            remaining = args.max_wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                publish("blocked", blockers=blockers, active=None, reason="resource wait budget exhausted")
                yield None
                return
            publish("waiting", blockers=blockers, active=None)
            time.sleep(min(args.poll_seconds, remaining))


def next_attempt(job):
    job.mkdir(parents=True, exist_ok=True)
    number = 0
    while (job / f"attempt_{number:04d}").exists():
        number += 1
    directory = job / f"attempt_{number:04d}"
    directory.mkdir()
    return directory


def evaluation(args, plan, item, seed, environment, publish):
    scenarios = plan["spec"]["stages"][0]["scenarios"]
    job = args.output_root / "control" / item["variant"] / f"train_{item['training_seed']}" / f"seed_{seed}"
    checkpoint = item["checkpoint"]
    identity = {"plan_sha256": plan["sha256"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "evaluation_seed": seed, "steps": args.steps}
    previous = control.previous_success(job, identity, args.output_root)
    if previous:
        return previous
    directory = next_attempt(job)
    configs = [args.study_root / f"configs/{item['variant']}.eval.{case}.json" for case in scenarios]
    outputs = [directory / f"{case}.json" for case in scenarios]
    invocation = command(args.source_root, "evaluate-suite", "--checkpoint", checkpoint["checkpoint"],
        "--configs", *configs, "--outputs", *outputs, "--steps", args.steps, "--seed", seed,
        "--device", args.device, "--control-output", directory / "control.json", "--trace-output",
        directory / "trace.npz", "--settle-steps", 200, "--min-steady-samples", 200)
    publish("evaluating", active={"variant": item["variant"], "training_seed": item["training_seed"],
                                  "evaluation_seed": seed, "directory": str(directory)})
    receipt = {"identity": identity, "status": "failed", "started_at": control.now()}
    try:
        worker = control.run_worker(invocation, directory, environment, args.worker_timeout_seconds, publish)
        receipt["worker"] = worker
        if worker["returncode"] != 0 or worker["timed_out"]:
            raise RuntimeError("control evaluation failed or timed out")
        reports, artifacts = control.verify_outputs(directory, checkpoint, seed, scenarios, args.output_root)
        receipt.update(status="completed", scenarios=reports, artifacts=artifacts)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    except BaseException:
        receipt.update(error="interrupted", finished_at=control.now())
        write(directory / "receipt.json", receipt)
        raise
    receipt["finished_at"] = control.now()
    write(directory / "receipt.json", receipt)
    return receipt


def deployment_checks(args, plan, item, environment, publish):
    job = args.output_root / "deployment" / item["variant"] / f"seed_{item['training_seed']}"
    checkpoint = item["checkpoint"]
    identity = {"plan_sha256": plan["sha256"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "backend": "onnx", "threads": 1, "iterations": args.benchmark_iterations}
    previous = control.previous_success(job, identity, args.output_root)
    if previous:
        return previous
    directory = next_attempt(job)
    bundle, benchmark = directory / "bundle", directory / "benchmark.json"
    receipt = {"identity": identity, "status": "failed", "started_at": control.now(),
               "target_hardware_verified": False, "hardware_deployment_ready": False}
    try:
        workers = []
        for operation, arguments in (("export", ["--checkpoint", checkpoint["checkpoint"], "--directory", bundle]),
                ("benchmark", ["--directory", bundle, "--output", benchmark, "--backend", "onnx",
                               "--threads", 1, "--iterations", args.benchmark_iterations])):
            work = directory / operation
            work.mkdir()
            publish("deployment_checks", active={"variant": item["variant"], "operation": operation})
            worker = control.run_worker(command(args.source_root, operation, *arguments), work,
                                        environment, args.worker_timeout_seconds, publish)
            workers.append(worker)
            receipt["workers"] = workers
            if worker["returncode"] != 0 or worker["timed_out"]:
                raise RuntimeError(f"{operation} failed or timed out")
        manifest = control.read(bundle / "manifest.json")
        if manifest["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
            raise ValueError("export differs from the trained endpoint")
        artifacts = {"manifest": control.artifact(bundle / "manifest.json", args.output_root),
                     "benchmark": control.artifact(benchmark, args.output_root)}
        for name, expected in manifest["files"].items():
            path = control.inside(bundle, name)
            if control.file_sha(path) != expected:
                raise ValueError("exported runtime file hash differs")
            artifacts[name] = control.artifact(path, args.output_root)
        report = control.read(benchmark)
        if (report["manifest_sha256"] != artifacts["manifest"]["sha256"]
                or report.get("backend") != "onnx" or report.get("threads") != 1
                or report.get("iterations") != args.benchmark_iterations
                or any(type(report.get(name)) not in (float, int) or not math.isfinite(report[name])
                       or report[name] < 0 for name in ("mean_ms", "p99_ms", "max_ms", "deadline_misses"))):
            raise ValueError("CPU benchmark protocol or values differ")
        receipt.update(status="completed", artifacts=artifacts, benchmark=report)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    except BaseException:
        receipt.update(error="interrupted", finished_at=control.now())
        write(directory / "receipt.json", receipt)
        raise
    receipt["finished_at"] = control.now()
    write(directory / "receipt.json", receipt)
    return receipt


def run_campaign(args):
    for name in ("study_root", "output_root", "source_root", "resource_lock", "dependency_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    if args.resource_lock == args.study_root / ".run.lock":
        raise ValueError("external resource lock must differ from the new study lock")
    plan = control.load_plan(args.study_root)
    if (len(plan["spec"]["stages"]) != 1 or plan["spec"]["stages"][0]["updates"] != args.updates):
        raise ValueError("transfer controller requires one stage with the declared update target")
    used_seeds = set(plan["spec"]["seeds"])
    for section, field in (("training", "anchor_seeds"), ("evaluation", "validation_seeds"), ("evaluation", "seeds")):
        used_seeds.update(plan["spec"].get(section, {}).get(field, []))
    if used_seeds.intersection(args.seeds):
        raise ValueError("independent control seeds must be disjoint from the study seeds")
    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = {route: control.artifact(control.inside(args.study_root, route), args.study_root)
               for route in plan["configs"]}
    for route in configs:
        if control.digest(control.read(args.study_root / route)) != plan["configs"][route]:
            raise ValueError("frozen study configuration changed")
    definition = {"format": "transformer_rl.transfer_campaign", "schema_version": 1,
        "study_root": str(args.study_root), "plan_sha256": plan["sha256"],
        "source": source_identity(args.source_root), "configs": configs,
        "resource_lock": str(args.resource_lock), "dependency_receipt": str(args.dependency_receipt),
        "updates": args.updates, "evaluation_seeds": args.seeds, "steps": args.steps,
        "device": args.device, "benchmark_iterations": args.benchmark_iterations,
        "max_run_attempts": args.max_run_attempts, "worker_timeout_seconds": args.worker_timeout_seconds,
        "all_variants_retained": True, "formal_architecture_selection": False,
        "target_hardware_verified": False, "hardware_deployment_ready": False}
    manifest_path = args.output_root / "campaign.json"
    if manifest_path.exists():
        if control.read(manifest_path) != definition:
            raise ValueError("frozen campaign protocol or source changed; use a new output root")
    else:
        write(manifest_path, definition)
    summary = {"status": "waiting", "started_at": control.now(), "campaign_sha256": control.digest(definition),
               "training": {}, "results": {}, "all_variants_retained": True,
               "formal_architecture_selection": False, "target_hardware_verified": False}

    def publish(status=None, **values):
        if status is not None:
            summary["status"] = status
        summary.update(values, updated_at=control.now())
        # A live record may be partial; sealed training must pass the stricter audit.
        summary["training"] = training_snapshot(args.study_root, plan)
        write(args.output_root / "summary.json", summary, replace=True)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.source_root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with (args.output_root / ".transfer.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with acquire_resource(args, publish) as dependency:
            if dependency is None:
                return summary
            if source_identity(args.source_root) != definition["source"]:
                raise ValueError("frozen source changed while queued")
            for item in configs.values():
                control.checked(args.study_root, item)
            publish("training", dependency=dependency, blockers=[], active=None)
            runs = args.output_root / "executor"
            while not all(item["checkpoint"] and item["checkpoint"]["update"] >= args.updates
                          and item["ppo_verified"] for item in summary["training"].values()):
                if len(list(runs.glob("attempt_*"))) >= args.max_run_attempts:
                    break
                if control.active_study_workers(args.study_root):
                    publish("blocked", reason="previous new-study worker remains alive", active=None)
                    return summary
                directory = next_attempt(runs)
                publish("training", active={"directory": str(directory)})
                worker = control.run_worker(command(args.source_root, "run", "--root", args.study_root,
                    "--max-parallel", 1), directory, environment, args.worker_timeout_seconds, publish)
                publish(active=None)
                write(directory / "receipt.json", {"status": "completed" if worker["returncode"] == 0 else "incomplete",
                    "worker": worker, "training": summary["training"], "finished_at": control.now()})
                if worker["timed_out"]:
                    # Do not overlap an independently sessioned child left behind by a killed executor.
                    if control.active_study_workers(args.study_root):
                        publish("blocked", reason="timed-out executor left a live owned study worker")
                        return summary
                # frame run legally resumes charged time-budget endpoints without changing its plan.
            for key, item in summary["training"].items():
                result = {"training": item, "evaluations": {}}
                summary["results"][key] = result
                if not item["checkpoint"]:
                    result["status"] = "no_sealed_checkpoint"
                    continue
                for seed in args.seeds:
                    result["evaluations"][str(seed)] = evaluation(args, plan, item, seed, environment, publish)
                result["deployment"] = deployment_checks(args, plan, item, environment, publish)
                result["status"] = "completed" if (all(value["status"] == "completed" for value in result["evaluations"].values())
                    and result["deployment"]["status"] == "completed") else "incomplete"
                publish(active=None)
            trained = all(item["checkpoint"] and item["checkpoint"]["update"] >= args.updates
                          and item["ppo_verified"] for item in summary["training"].values())
            evaluated = all(value["status"] == "completed" for value in summary["results"].values())
            publish("completed" if trained and evaluated else "incomplete", active=None,
                    training_target_completed=trained, all_last_sealed_endpoints_checked=evaluated)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "output-root", "source-root", "resource-lock", "dependency-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[8701, 9701])
    parser.add_argument("--steps", type=int, default=4001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    parser.add_argument("--max-run-attempts", type=int, default=8)
    parser.add_argument("--max-wait-seconds", type=float, default=172800.)
    parser.add_argument("--worker-timeout-seconds", type=float, default=172800.)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if (any(value < 1 for value in (args.updates, args.steps, args.max_run_attempts, args.benchmark_iterations))
            or args.benchmark_iterations < 1000 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in args.seeds)
            or not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0
            or not math.isfinite(args.worker_timeout_seconds) or args.worker_timeout_seconds <= 0
            or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30):
        parser.error("finite positive budgets, at least 1000 benchmark iterations and distinct unsigned seeds required")
    handlers = {}
    def interrupted(*_):
        raise KeyboardInterrupt
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        result = run_campaign(args)
        print(json.dumps({"status": result["status"], "summary": str(args.output_root / "summary.json")}), flush=True)
        return 0 if result["status"] == "completed" else 2
    except KeyboardInterrupt:
        path = args.output_root / "summary.json"
        if path.exists():
            summary = control.read(path)
            summary.update(status="interrupted", active=None, updated_at=control.now())
            write(path, summary, replace=True)
        return 130
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())

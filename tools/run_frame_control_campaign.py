"""Run independent control evaluations without changing an existing training study.

The controller waits for the old study's exclusive lock and verified workers,
then evaluates sealed checkpoints serially. Rejected training gates do not
remove models from this campaign. Failed attempts and their partial outputs are
retained; a later invocation creates a new attempt instead of overwriting them.
Only the Python standard library is needed by this controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if not replace:
        with path.open("x") as stream:
            stream.write(encoded)
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("artifact path escapes its root")
    return path


def checked(root, receipt):
    path = inside(root, receipt["path"])
    if file_sha(path) != receipt["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def artifact(path, root):
    return {"path": str(Path(path).resolve().relative_to(Path(root).resolve())),
            "sha256": file_sha(path)}


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def live_process(record):
    """A reused PID is not evidence that the original worker remains alive."""
    return (record.get("status") == "running" and record.get("start") is not None
            and process_start(record.get("pid")) == str(record["start"]))


def active_study_workers(study_root):
    result = []
    for path in sorted((Path(study_root) / "jobs").rglob("*.process.json")):
        record = read(path)
        if live_process(record):
            result.append({"receipt": str(path), "pid": record["pid"], "start": record["start"]})
    return result


def active_campaign_workers(root):
    result = []
    for path in sorted(Path(root).rglob("worker.process.json")):
        record = read(path)
        if live_process(record):
            result.append({"receipt": str(path), "pid": record["pid"], "start": record["start"]})
    return result


def load_plan(study_root):
    plan = read(Path(study_root) / "plan.json")
    if digest({key: value for key, value in plan.items() if key != "sha256"}) != plan.get("sha256"):
        raise ValueError("study plan identity differs from its sealed hash")
    names = [entry["name"] for entry in plan["spec"]["variants"]]
    scenarios = plan["spec"]["stages"][0]["scenarios"]
    if (not names or len(names) != len(set(names)) or not scenarios
            or len(scenarios) != len(set(scenarios))
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names + scenarios)):
        raise ValueError("variants and first-stage scenarios require unique safe names")
    return plan


def source_identity(source_root):
    root = Path(source_root).resolve()
    package = root / "src/transformer_rl"
    files = {str(path.relative_to(root)): file_sha(path) for path in sorted(package.rglob("*.py"))}
    if not files or not (package / "frame_process.py").is_file():
        raise ValueError("source root does not contain src/transformer_rl/frame_process.py")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return {"root": str(root), "files": files, "sha256": digest(files),
            "git_head": commit.stdout.strip() if commit.returncode == 0 else None,
            "controller_sha256": file_sha(__file__)}


def ready_checkpoint(study_root, variant, training_seed, updates, plan):
    """Read sealed first-stage attempts even when their old gates rejected them."""
    root = Path(study_root).resolve()
    state_path = root / "jobs" / variant / f"seed_{training_seed}" / "state.json"
    if not state_path.exists():
        return {"status": "not_ready", "reason": "training job has no state"}
    state = read(state_path)
    if (state.get("plan_sha256") != plan["sha256"] or state.get("variant") != variant
            or state.get("seed") != training_seed):
        raise ValueError("training job identity differs from the plan")
    stages = [stage for stage in state.get("stages", [])
              if stage["name"] == plan["spec"]["stages"][0]["name"]]
    candidates = []
    for attempt in stages[0].get("attempts", []) if stages else []:
        if not attempt.get("budget_charged") or "training" not in attempt:
            continue
        completion_path = checked(root, attempt["training"])
        if completion_path.name != "completion.json":
            continue
        completion = read(completion_path)
        count = completion.get("final_update")
        if type(count) is not int or count < updates:
            continue
        checkpoint = inside(root, completion["checkpoint"])
        if file_sha(checkpoint) != completion["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        if "checkpoint" in attempt and checked(root, attempt["checkpoint"]) != checkpoint:
            raise ValueError("ledger and completion checkpoint identities differ")
        candidates.append({"status": "ready", "checkpoint": str(checkpoint),
                           "checkpoint_sha256": completion["checkpoint_sha256"], "update": count,
                           "training_completion": artifact(completion_path, root),
                           "training_state": artifact(state_path, root),
                           "old_job_status": state["status"]})
    if candidates:
        return min(candidates, key=lambda candidate: candidate["update"])
    return {"status": "not_ready", "reason": f"no sealed checkpoint at update >= {updates}",
            "old_job_status": state.get("status")}


def study_jobs_terminal(study_root, plan):
    terminal = {"completed", "rejected", "failed", "stopped"}
    pending = []
    for variant in plan["spec"]["variants"]:
        for seed in plan["spec"]["seeds"]:
            path = Path(study_root) / "jobs" / variant["name"] / f"seed_{seed}" / "state.json"
            status = read(path).get("status") if path.exists() else "not_started"
            if status not in terminal:
                pending.append({"variant": variant["name"], "training_seed": seed, "status": status})
    return pending


def build_command(args, checkpoint, configs, outputs, control_output, trace_output):
    bootstrap = ("import runpy, sys; sys.path.insert(0, "
                 + repr(str(args.source_root / "src"))
                 + "); runpy.run_module('transformer_rl.frame_process', "
                 "run_name='__main__', alter_sys=True)")
    return [sys.executable, "-c", bootstrap, "evaluate-suite", "--checkpoint", checkpoint,
            "--configs", *map(str, configs), "--outputs", *map(str, outputs),
            "--steps", str(args.steps), "--seed", str(args.current_seed), "--device", args.device,
            "--control-output", str(control_output), "--trace-output", str(trace_output),
            "--settle-steps", "200", "--min-steady-samples", "200"]


def terminate_owned(process, start):
    if process.poll() is not None:
        return
    if start is None or process_start(process.pid) != start or os.getpgid(process.pid) != process.pid:
        raise RuntimeError("campaign worker ownership cannot be verified")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if process_start(process.pid) == start:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_worker(command, directory, environment, timeout, heartbeat):
    record = {"command": command, "status": "launching", "started_at": now(), "timeout_seconds": timeout}
    write(directory / "command.json", {"command": command, "pythonpath": environment["PYTHONPATH"]})
    timed_out = False
    with (directory / "worker.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        record.update(pid=process.pid, start=process_start(process.pid), status="running")
        write(directory / "worker.process.json", record)
        started, next_heartbeat = time.monotonic(), time.monotonic() + 30
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    terminate_owned(process, record["start"])
                    break
                if time.monotonic() >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = time.monotonic() + 30
                time.sleep(.2)
        except BaseException:
            terminate_owned(process, record["start"])
            raise
        finally:
            record.update(status="finished", returncode=process.poll(), timed_out=timed_out, finished_at=now())
            write(directory / "worker.process.json", record, replace=True)
    return record


def verify_outputs(directory, checkpoint, eval_seed, scenarios, root):
    reports, receipts = {}, {}
    for scenario in scenarios:
        path = directory / f"{scenario}.json"
        report = read(path)
        if report.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"] or report.get("seed") != eval_seed:
            raise ValueError(f"evaluation identity mismatch: {scenario}")
        reports[scenario] = {"success_rate": report.get("success_rate"),
                             "completed_episodes": report.get("completed_episodes"),
                             "failed_episodes": report.get("failed_episodes"),
                             "metrics": report.get("metrics"), "stability": report.get("stability"),
                             "report": artifact(path, root)}
        receipts[scenario] = artifact(path, root)
    control = read(directory / "control.json")
    if control.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"] or control.get("seed") != eval_seed:
        raise ValueError("control report identity mismatch")
    receipts["control"] = artifact(directory / "control.json", root)
    receipts["trace"] = artifact(directory / "trace.npz", root)
    return reports, receipts


def previous_success(job, identity, root):
    for attempt in sorted(job.glob("attempt_*"), reverse=True):
        receipt_path = attempt / "receipt.json"
        if not receipt_path.exists():
            process_path = attempt / "worker.process.json"
            if process_path.exists() and live_process(read(process_path)):
                return {"status": "waiting", "reason": "previous campaign worker is still alive"}
            continue
        receipt = read(receipt_path)
        if receipt["status"] != "completed":
            continue
        if receipt["identity"] != identity:
            raise ValueError("sealed evaluation identity differs from the current checkpoint/protocol")
        for item in receipt["artifacts"].values():
            checked(root, item)
        return receipt
    return None


def run_campaign(args):
    root, study = args.output_root.resolve(), args.study_root.resolve()
    args.source_root = args.source_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = load_plan(study)
    variants = [entry["name"] for entry in plan["spec"]["variants"]]
    scenarios = list(plan["spec"]["stages"][0]["scenarios"])
    if args.training_seed not in plan["spec"]["seeds"]:
        raise ValueError("training seed is not present in the frozen study")
    used_seeds = set(plan["spec"]["seeds"])
    for section, field in (("training", "anchor_seeds"), ("evaluation", "validation_seeds"), ("evaluation", "seeds")):
        used_seeds.update(plan["spec"].get(section, {}).get(field, []))
    if used_seeds.intersection(args.seeds):
        raise ValueError("supplemental evaluation seeds must be disjoint from all old study seeds")
    config_receipts = {}
    for variant in variants:
        for scenario in scenarios:
            route = f"configs/{variant}.eval.{scenario}.json"
            path = inside(study, route)
            if digest(read(path)) != plan["configs"].get(route):
                raise ValueError(f"frozen evaluation configuration changed: {route}")
            config_receipts[route] = artifact(path, study)
    definition = {"format": "transformer_rl.control_campaign", "schema_version": 1,
                  "study_root": str(study), "plan_sha256": plan["sha256"],
                  "source": source_identity(args.source_root), "variants": variants, "scenarios": scenarios,
                  "training_seed": args.training_seed, "minimum_checkpoint_update": args.updates,
                  "evaluation_seeds": args.seeds, "steps": args.steps, "settle_steps": 200,
                  "min_steady_samples": 200, "device": args.device, "configs": config_receipts}
    manifest = root / "campaign.json"
    if manifest.exists():
        if read(manifest) != definition:
            raise ValueError("campaign protocol or source changed; use a new output root")
    else:
        write(manifest, definition)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.source_root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    results = {variant: {"training": {}, "evaluations": {}} for variant in variants}
    started, tried = time.monotonic(), set()
    summary = {"status": "waiting", "started_at": now(), "updated_at": now(),
               "campaign_sha256": digest(definition), "results": results,
               "all_variants_retained": True, "formal_architecture_selection": False}

    def publish(status=None, **values):
        if status is not None:
            summary["status"] = status
        summary.update(values, updated_at=now())
        write(root / "summary.json", summary, replace=True)

    with (root / ".campaign.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for variant in variants:
                try:
                    results[variant]["training"] = ready_checkpoint(study, variant, args.training_seed, args.updates, plan)
                except (OSError, ValueError, KeyError) as error:
                    results[variant]["training"] = {"status": "invalid", "reason": str(error)}
            pending = study_jobs_terminal(study, plan)
            live = active_study_workers(study)
            own_live = active_campaign_workers(root)
            # The existing executor holds this lock across the gaps between workers.
            # Use the same lock without writing it; no existing study files are changed.
            locked = False
            with (study / ".run.lock").open("r+") as study_lock:
                try:
                    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    pass
                if locked and not pending and not live and not own_live:
                    if source_identity(args.source_root) != definition["source"]:
                        publish("blocked", active=None, reason="evaluation source changed while waiting")
                        return summary
                    for route, item in config_receipts.items():
                        checked(study, item)
                    for variant in variants:
                        checkpoint = results[variant]["training"]
                        if checkpoint["status"] != "ready":
                            continue
                        configs = [study / f"configs/{variant}.eval.{scenario}.json" for scenario in scenarios]
                        for eval_seed in args.seeds:
                            identity = {"campaign_sha256": digest(definition), "variant": variant,
                                        "evaluation_seed": eval_seed,
                                        "checkpoint_sha256": checkpoint["checkpoint_sha256"]}
                            job = root / variant / f"seed_{eval_seed}"
                            job.mkdir(parents=True, exist_ok=True)
                            try:
                                previous = previous_success(job, identity, root)
                            except (OSError, ValueError, KeyError) as error:
                                results[variant]["evaluations"][str(eval_seed)] = {"status": "invalid", "reason": str(error)}
                                continue
                            if previous:
                                results[variant]["evaluations"][str(eval_seed)] = previous
                                continue
                            if (variant, eval_seed) in tried:
                                continue
                            attempt_index = 0
                            while (job / f"attempt_{attempt_index:04d}").exists():
                                attempt_index += 1
                            directory = job / f"attempt_{attempt_index:04d}"
                            directory.mkdir()
                            args.current_seed = eval_seed
                            outputs = [directory / f"{scenario}.json" for scenario in scenarios]
                            command = build_command(args, checkpoint["checkpoint"], configs, outputs,
                                                    directory / "control.json", directory / "trace.npz")
                            publish("running", active={"variant": variant, "evaluation_seed": eval_seed,
                                                       "directory": str(directory)})
                            receipt = {"identity": identity, "started_at": now(), "status": "failed"}
                            try:
                                worker = run_worker(command, directory, environment, args.worker_timeout_seconds, publish)
                                receipt["worker"] = worker
                                if worker["returncode"] != 0 or worker["timed_out"]:
                                    raise RuntimeError("evaluation worker failed or exceeded its time limit")
                                reports, artifacts = verify_outputs(directory, checkpoint, eval_seed, scenarios, root)
                                receipt.update(status="completed", scenarios=reports, artifacts=artifacts,
                                               control=artifacts["control"], trace=artifacts["trace"])
                            except Exception as error:
                                receipt["error"] = f"{type(error).__name__}: {error}"
                            except BaseException:
                                receipt.update(error="campaign interrupted", finished_at=now())
                                write(directory / "receipt.json", receipt)
                                results[variant]["evaluations"][str(eval_seed)] = receipt
                                publish("interrupted", active=None)
                                raise
                            receipt["finished_at"] = now()
                            write(directory / "receipt.json", receipt)
                            results[variant]["evaluations"][str(eval_seed)] = receipt
                            tried.add((variant, eval_seed))
                            publish(active=None)
            complete = all(results[variant]["evaluations"].get(str(seed), {}).get("status") == "completed"
                           for variant in variants for seed in args.seeds)
            if complete:
                publish("completed", active=None, blockers=[])
                return summary
            blockers = {"old_executor_lock_held": not locked, "old_nonterminal_jobs": pending,
                        "old_live_workers": live, "previous_campaign_live_workers": own_live,
                        "unready_variants": [variant for variant in variants
                                             if results[variant]["training"].get("status") != "ready"]}
            attempted_all = all(results[variant]["evaluations"].get(str(seed), {}).get("status")
                                in {"completed", "failed", "invalid"}
                                for variant in variants for seed in args.seeds)
            if attempted_all:
                publish("incomplete", active=None, blockers=blockers,
                        reason="failed evaluation evidence retained; a later invocation may create new attempts")
                return summary
            remaining = args.max_wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                publish("blocked", active=None, blockers=blockers,
                        reason="wait budget exhausted; incomplete and failed candidates remain listed")
                return summary
            # Every waiting interval persists status, even with no training progress.
            publish("waiting", active=None, blockers=blockers)
            time.sleep(min(args.poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--training-seed", type=int, default=1101)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5701, 6701])
    parser.add_argument("--steps", type=int, default=4001)
    parser.add_argument("--max-wait-seconds", type=float, default=86400.)
    parser.add_argument("--worker-timeout-seconds", type=float, default=3600.)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if (args.updates < 1 or args.steps < 1 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**32 for seed in [args.training_seed, *args.seeds])
            or not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0
            or not math.isfinite(args.worker_timeout_seconds) or args.worker_timeout_seconds <= 0
            or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30):
        parser.error("budgets must be finite and positive; seeds must be unique unsigned integers")
    old_handlers = {}
    def interrupted(*_):
        raise KeyboardInterrupt
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, interrupted)
        result = run_campaign(args)
        print(json.dumps({"status": result["status"], "summary": str(args.output_root / "summary.json")}), flush=True)
        return 0 if result["status"] == "completed" else 2
    except KeyboardInterrupt:
        path = args.output_root / "summary.json"
        if path.exists():
            summary = read(path)
            summary.update(status="interrupted", active=None, updated_at=now())
            write(path, summary, replace=True)
        return 130
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())

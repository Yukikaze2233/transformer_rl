"""Queue paired Gated curricula with equal samples and preserved learning state.

Every arm resets the environment at the same phase boundary. Task evaluation
never controls promotion: poor control remains evidence, rather than cancelling
the second phase. Only sealed, full-rollout time-budget endpoints can resume.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal


_SPEC = importlib.util.spec_from_file_location(
    "curriculum_transfer_helpers", Path(__file__).with_name("run_transfer_campaign.py"))
transfer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(transfer)
control, write, command = transfer.control, transfer.write, transfer.command


def load_manifest(path):
    manifest = control.read(path)
    if (manifest.get("format") != "transformer_rl.curriculum_study"
            or manifest.get("schema_version") != 1
            or control.digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest.get("sha256")):
        raise ValueError("curriculum manifest identity differs")
    arms, seeds = manifest["arms"], manifest["training_seeds"]
    if (len(arms) != 3 or len({arm["name"] for arm in arms}) != 3
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("three distinct arms and distinct training seeds are required")
    schedule = [(phase["start_update"], phase["updates"]) for phase in arms[0]["phases"]]
    cursor = 0
    for start, updates in schedule:
        if type(start) is not int or type(updates) is not int or start != cursor or updates <= 0:
            raise ValueError("phase budgets must be positive and their clocks continuous from zero")
        cursor += updates
    for arm in arms:
        if len(arm["phases"]) != 2 or [(p["start_update"], p["updates"]) for p in arm["phases"]] != schedule:
            raise ValueError("paired curricula require the same continuous two-phase budget")
        if len({phase["name"] for phase in arm["phases"]}) != 2:
            raise ValueError("phase names must be distinct")
        for phase in arm["phases"]:
            if phase["config"] not in manifest["configs"]:
                raise ValueError("training configuration is not frozen")
    cases = manifest["scenarios"]
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("evaluation cases must be distinct")
    for case in cases:
        if case["config"] not in manifest["configs"]:
            raise ValueError("evaluation configuration is not frozen")
    for name in [arm["name"] for arm in arms] + [p["name"] for a in arms for p in a["phases"]] + [c["name"] for c in cases]:
        if not control.re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("unsafe curriculum artifact name")
    evaluation = manifest["evaluation"]
    if (not evaluation["seeds"] or len(set(evaluation["seeds"])) != len(evaluation["seeds"])
            or set(evaluation["seeds"]).intersection(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in evaluation["seeds"])):
        raise ValueError("evaluation seeds must be distinct from training seeds")
    for value in (manifest["training"]["rollout_steps"], manifest["training"]["checkpoint_interval"],
                  evaluation["steps"], evaluation["min_steady_samples"], evaluation["trace_replicas"]):
        if type(value) is not int or value <= 0:
            raise ValueError("sampling budgets must be positive integers")
    if type(evaluation["settle_steps"]) is not int or evaluation["settle_steps"] < 0:
        raise ValueError("invalid evaluation settle period")
    seconds = manifest["training"]["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid worker time budget")
    return manifest


def source_identity(root):
    result = transfer.source_identity(root)
    result["curriculum_controller_sha256"] = control.file_sha(__file__)
    result["transfer_helper_sha256"] = control.file_sha(Path(__file__).with_name("run_transfer_campaign.py"))
    return result


def learner_source_identity(root):
    """Match experiments.source_identity without importing Torch or simulation."""
    package = Path(root).resolve() / "src/transformer_rl"
    files = {str(path.relative_to(package)): control.file_sha(path)
             for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    # experiments._digest uses JSON's default ASCII escaping, unlike the
    # controller's unicode JSON digest. Retain its exact byte convention.
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def verify_learner_source(root, manifest):
    expected = manifest.get("source_identity", {}).get("learner_source")
    if not isinstance(expected, dict) or learner_source_identity(root) != expected:
        raise ValueError("prepared learner source differs from the campaign source")


def verify_configs(root, manifest):
    for route, item in manifest.get("artifacts", {}).items():
        receipt = item if isinstance(item, dict) else {"path": route, "sha256": item}
        control.checked(root, receipt)
    parsed = {}
    for route, expected in manifest["configs"].items():
        path = control.inside(root, route)
        value = control.read(path)
        if control.digest(value) != expected:
            raise ValueError(f"frozen curriculum configuration changed: {route}")
        parsed[route] = value
    reference, verified_snapshots = None, set()
    for arm in manifest["arms"]:
        for phase in arm["phases"]:
            cfg = parsed[phase["config"]]
            snapshot = Path(cfg["environment"]["snapshot"]).resolve() if "snapshot" in cfg["environment"] else None
            if snapshot is not None and snapshot not in verified_snapshots:
                if not snapshot.is_relative_to(root.resolve()):
                    raise ValueError("curriculum snapshot must belong to its frozen root")
                source = control.read(snapshot / "snapshot.json")
                if (source["sha256"] != cfg["environment"]["snapshot_sha256"]
                        or control.digest(source["files"]) != source["sha256"]):
                    raise ValueError("snapshot identity changed")
                for route, sha in source["files"].items():
                    control.checked(snapshot, {"path": route, "sha256": sha})
                verified_snapshots.add(snapshot)
            invariant = {key: cfg[key] for key in ("model", "ppo", "control")}
            invariant["snapshot_sha256"] = cfg["environment"]["snapshot_sha256"]
            invariant["num_envs"] = cfg["environment"]["num_envs"]
            if reference is None:
                reference = invariant
            elif invariant != reference:
                raise ValueError("curriculum arms must share network, PPO, control, snapshot and environment count")
    return parsed


def train_command(args, manifest, config, directory, seed, remaining, offset, parent, *, phase_transfer):
    invocation = command(args.source_root, "train", "--config", args.study_root / config,
        "--env-factory", manifest.get("environment_factory", "transformer_rl.chassis_adapter:make_env"),
        "--run-dir", directory / "train", "--updates", remaining,
        "--rollout-steps", manifest["training"]["rollout_steps"], "--seed", seed,
        "--device", args.device, "--max-seconds", manifest["training"]["max_seconds"],
        "--checkpoint-interval", manifest["training"]["checkpoint_interval"],
        "--consumed-update-offset", offset)
    if parent is not None:
        invocation.extend(("--restore-learning-from" if phase_transfer else "--resume", parent["checkpoint"]))
    return invocation


def seal_training(directory, request, root, worker=None):
    """A crash reserves its whole budget; partial collection cannot be refunded."""
    path = directory / "receipt.json"
    if path.exists():
        receipt = control.read(path)
        if receipt["request_sha256"] != control.file_sha(directory / "request.json"):
            raise ValueError("sealed training request changed")
        for item in receipt.get("artifacts", {}).values():
            control.checked(root, item)
        return receipt
    receipt = {"identity": request["identity"], "request_sha256": control.file_sha(directory / "request.json"),
        "status": "incomplete", "reserved_updates": request["updates"], "worker": worker,
        "finished_at": control.now(), "artifacts": {}}
    completion = directory / "train/completion.json"
    try:
        if not completion.exists():
            raise ValueError("unsealed training attempt retains its entire reserved budget")
        report = control.read(completion)
        evidence = transfer.metrics_evidence(directory / "train/metrics.jsonl", sealed=report)
        completed = report["completed_updates"]
        expected_samples = completed * request["batch_samples_per_update"]
        if (type(completed) is not int or not 0 < completed <= request["updates"]
                or report["attempted_updates"] != completed
                or report["start_update"] != request["start_update"]
                or report["final_update"] != request["start_update"] + completed
                or report["config_sha256"] != request["identity"]["config_sha256"]
                or report["consumed_transitions"] != expected_samples
                or report["cumulative_transitions"] != request["prior_transitions"] + expected_samples
                or evidence["batch_samples"] != expected_samples):
            raise ValueError("training budget, sample or parent continuum differs")
        if report["status"] not in ("completed", "stopped"):
            raise ValueError("training completion has invalid status")
        if report["status"] == "completed" and completed != request["updates"]:
            raise ValueError("completed training did not exhaust its requested updates")
        if report["status"] == "stopped" and report.get("stop_reason") not in ("time_budget", "SIGINT", "SIGTERM"):
            raise ValueError("unrecognized sealed stop reason")
        run = control.read(directory / "train/run.json")
        if run.get("seed") != request["identity"]["training_seed"] or run.get("config") != request["config"]:
            raise ValueError("worker training seed or configuration differs")
        for key in ("resume", "restore_learning_from", "initialize_from"):
            if run.get(key) != request["initialization"].get(key):
                raise ValueError("worker learning-state initialization differs")
        initial_model = run.get("initial_model_sha256")
        if not request["initialization"] and (not isinstance(initial_model, str)
                or not control.re.fullmatch(r"[0-9a-f]{64}", initial_model)):
            raise ValueError("scratch learner initialization has no verified identity")
        checkpoint = control.inside(root, report["checkpoint"])
        if control.file_sha(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("training checkpoint hash differs")
        optimized_samples = 0
        for row in (json.loads(line) for line in (directory / "train/metrics.jsonl").read_text().splitlines()):
            count = row["optimization"].get("sample_count")
            if type(count) is not int or count <= 0:
                raise ValueError("applied PPO samples must be recorded")
            optimized_samples += count
        receipt.update(status="completed" if completed == request["updates"] else "resumable",
            initial_model_sha256=initial_model,
            consumed_updates=completed, consumed_transitions=expected_samples,
            cumulative_transitions=report["cumulative_transitions"], evidence={**evidence, "optimization_samples": optimized_samples},
            checkpoint={"checkpoint": str(checkpoint), "checkpoint_sha256": report["checkpoint_sha256"],
                        "update": report["final_update"], "cumulative_transitions": report["cumulative_transitions"]},
            artifacts={"completion": control.artifact(completion, root),
                       "metrics": control.artifact(directory / "train/metrics.jsonl", root),
                       "run": control.artifact(directory / "train/run.json", root),
                       "checkpoint": control.artifact(checkpoint, root)})
    except (ValueError, KeyError, OSError, TypeError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["consumed_updates"] = request["updates"]
        for name in ("completion.json", "failure.json", "metrics.jsonl", "run.json"):
            artifact = directory / "train" / name
            if artifact.exists():
                receipt["artifacts"][name] = control.artifact(artifact, root)
    write(path, receipt)
    return receipt


def train_phase(args, manifest, parsed, arm, phase, seed, phase_offset, parent, environment, publish):
    job = args.output_root / "jobs" / arm["name"] / f"seed_{seed}" / phase["name"] / "training"
    consumed, attempts = 0, []
    prior = parent
    directories = sorted(job.glob("attempt_*"))
    while consumed < phase["updates"]:
        remaining = phase["updates"] - consumed
        identity = {"manifest_sha256": manifest["sha256"], "arm": arm["name"], "phase": phase["name"],
            "training_seed": seed, "config_sha256": manifest["configs"][phase["config"]],
            "parent_checkpoint_sha256": prior["checkpoint_sha256"] if prior else None,
            "consumed_update_offset": phase_offset + consumed}
        phase_transfer = phase_offset > 0 and consumed == 0
        directory = directories[len(attempts)] if len(attempts) < len(directories) else None
        if directory is None:
            if len(attempts) >= args.max_run_attempts:
                return {"status": "incomplete", "reason": "training attempt limit", "attempts": attempts}
            directory = transfer.next_attempt(job)
            cfg = parsed[phase["config"]]
            invocation = train_command(args, manifest, phase["config"], directory, seed, remaining,
                phase_offset + consumed, prior, phase_transfer=phase_transfer)
            initializer = "restore_learning_from" if phase_transfer else "resume"
            request = {"identity": identity, "updates": remaining, "start_update": phase_offset + consumed,
                "prior_transitions": prior["cumulative_transitions"] if prior else 0,
                "batch_samples_per_update": cfg["environment"]["num_envs"] * manifest["training"]["rollout_steps"],
                "config": cfg, "command": invocation,
                "initialization": {initializer: prior["checkpoint"]} if prior else {}}
            write(directory / "request.json", request)
            publish("training", active={**identity, "directory": str(directory)})
            worker = None
            try:
                worker = control.run_worker(invocation, directory, environment, args.worker_timeout_seconds, publish)
            finally:
                # Seal even after interruption; a full, saved endpoint remains resumable.
                receipt = seal_training(directory, request, args.output_root, worker)
        else:
            request = control.read(directory / "request.json")
            if request["identity"] != identity or request["updates"] != remaining:
                raise ValueError("training attempt parent, clock or reservation changed")
            receipt = seal_training(directory, request, args.output_root)
        attempts.append(control.artifact(directory / "receipt.json", args.output_root))
        if receipt["status"] == "incomplete":
            return {"status": "incomplete", "reason": receipt["error"], "attempts": attempts,
                    "consumed_updates": consumed + receipt["consumed_updates"]}
        consumed += receipt["consumed_updates"]
        prior = receipt["checkpoint"]
        publish(active=None)
    if len(directories) > len(attempts):
        raise ValueError("unexpected extra training attempts after exhausted phase budget")
    initial_model = control.read(control.checked(args.output_root, attempts[0])).get("initial_model_sha256")
    return {"status": "completed", "consumed_updates": consumed, "checkpoint": prior, "attempts": attempts,
            "initial_model_sha256": initial_model}


def evaluate_phase(args, manifest, arm, phase, training_seed, checkpoint, seed, environment, publish):
    cfg = manifest["evaluation"]
    job = args.output_root / "control" / arm["name"] / f"train_{training_seed}" / phase["name"] / f"seed_{seed}"
    identity = {"manifest_sha256": manifest["sha256"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "evaluation_seed": seed, "protocol": cfg}
    previous = control.previous_success(job, identity, args.output_root)
    if previous:
        return previous
    directory = transfer.next_attempt(job)
    names = [case["name"] for case in manifest["scenarios"]]
    configs = [args.study_root / case["config"] for case in manifest["scenarios"]]
    outputs = [directory / f"{name}.json" for name in names]
    invocation = command(args.source_root, "evaluate-suite", "--checkpoint", checkpoint["checkpoint"],
        "--configs", *configs, "--outputs", *outputs, "--steps", cfg["steps"], "--seed", seed,
        "--device", args.device, "--control-output", directory / "control.json",
        "--trace-output", directory / "trace.npz", "--trace-replicas", cfg["trace_replicas"],
        "--settle-steps", cfg["settle_steps"], "--min-steady-samples", cfg["min_steady_samples"])
    receipt = {"identity": identity, "status": "failed", "started_at": control.now()}
    publish("evaluating", active={"arm": arm["name"], "phase": phase["name"],
        "training_seed": training_seed, "evaluation_seed": seed, "directory": str(directory)})
    try:
        worker = control.run_worker(invocation, directory, environment, args.worker_timeout_seconds, publish)
        receipt["worker"] = worker
        if worker["returncode"] != 0 or worker["timed_out"]:
            raise RuntimeError("control evaluation failed or timed out")
        reports, artifacts = control.verify_outputs(directory, checkpoint, seed, names, args.output_root)
        receipt.update(status="completed", scenarios=reports, artifacts=artifacts)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    except BaseException:
        receipt.update(error="interrupted", finished_at=control.now())
        write(directory / "receipt.json", receipt)
        raise
    receipt["finished_at"] = control.now()
    write(directory / "receipt.json", receipt)
    publish(active=None)
    return receipt


def run_campaign(args):
    for name in ("manifest", "output_root", "source_root", "resource_lock", "dependency_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    args.study_root = args.manifest.parent
    manifest = load_manifest(args.manifest)
    verify_learner_source(args.source_root, manifest)
    parsed = verify_configs(args.study_root, manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.resource_lock == args.output_root / ".curriculum.lock":
        raise ValueError("resource and controller locks must differ")
    definition = {"format": "transformer_rl.curriculum_campaign", "schema_version": 1,
        "manifest": str(args.manifest), "manifest_sha256": manifest["sha256"],
        "manifest_file_sha256": control.file_sha(args.manifest), "source": source_identity(args.source_root),
        "device": args.device, "resource_lock": str(args.resource_lock),
        "dependency_receipt": str(args.dependency_receipt), "max_run_attempts": args.max_run_attempts,
        "worker_timeout_seconds": args.worker_timeout_seconds,
        "training_promotion": "fixed_budget_no_task_gates", "learning_state": "preserve_adam_rng_std",
        "phase_boundary": "all_arms_reset_episodes_repeat_first_history", "hardware_deployment_ready": False}
    frozen = args.output_root / "campaign.json"
    if frozen.exists():
        if control.read(frozen) != definition:
            raise ValueError("frozen campaign protocol/source changed")
        if control.read(args.output_root / "manifest.json") != manifest:
            raise ValueError("frozen manifest copy changed")
    else:
        write(args.output_root / "manifest.json", manifest)
        write(frozen, definition)
    summary = {"status": "waiting", "started_at": control.now(), "campaign_sha256": control.digest(definition),
               "results": {}, "initial_models": {}, "formal_architecture_selection": False, "hardware_deployment_ready": False}

    def publish(status=None, **values):
        if status is not None:
            summary["status"] = status
        summary.update(values, updated_at=control.now())
        write(args.output_root / "summary.json", summary, replace=True)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.source_root / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with (args.output_root / ".curriculum.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with transfer.acquire_resource(args, publish) as dependency:
            if dependency is None:
                return summary
            if source_identity(args.source_root) != definition["source"] or control.file_sha(args.manifest) != definition["manifest_file_sha256"]:
                raise ValueError("frozen source or manifest changed while queued")
            verify_learner_source(args.source_root, manifest)
            verify_configs(args.study_root, manifest)
            # Seed-major order gives all three arms a first-seed pilot before replication.
            for seed in manifest["training_seeds"]:
                for arm in manifest["arms"]:
                    key, parent, offset = f"{arm['name']}/seed_{seed}", None, 0
                    result = {"arm": arm["name"], "training_seed": seed, "phases": [], "status": "running"}
                    summary["results"][key] = result
                    for phase in arm["phases"]:
                        training = train_phase(args, manifest, parsed, arm, phase, seed, offset, parent, environment, publish)
                        item = {"name": phase["name"], "training": training, "evaluations": {}}
                        result["phases"].append(item)
                        if training["status"] != "completed":
                            result["status"] = "incomplete"
                            break
                        if offset == 0:
                            initial_model = training["initial_model_sha256"]
                            known = summary["initial_models"].setdefault(str(seed), initial_model)
                            if known != initial_model:
                                raise ValueError("paired arms started from different learner parameters")
                        parent = training["checkpoint"]
                        for eval_seed in manifest["evaluation"]["seeds"]:
                            item["evaluations"][str(eval_seed)] = evaluate_phase(args, manifest, arm, phase, seed,
                                parent, eval_seed, environment, publish)
                        offset += phase["updates"]
                    else:
                        result["status"] = "completed" if all(
                            ev["status"] == "completed" for p in result["phases"] for ev in p["evaluations"].values()) else "incomplete"
                        result["checkpoint"] = parent
                    publish(active=None)
                    if control.active_campaign_workers(args.output_root):
                        publish("blocked", reason="owned worker remains alive")
                        return summary
            completed = all(result["status"] == "completed" for result in summary["results"].values())
            publish("completed" if completed else "incomplete", active=None, dependency=dependency)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output-root", "source-root", "resource-lock", "dependency-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-run-attempts", type=int, default=8)
    parser.add_argument("--max-wait-seconds", type=float, default=604800.)
    parser.add_argument("--worker-timeout-seconds", type=float, default=173100.)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if (args.max_run_attempts < 1 or not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0
            or not math.isfinite(args.worker_timeout_seconds) or args.worker_timeout_seconds <= 0
            or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30):
        parser.error("finite positive worker budgets and bounded polling required")
    def interrupted(*_):
        raise KeyboardInterrupt
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    try:
        result = run_campaign(args)
        print(json.dumps({"status": result["status"], "summary": str(args.output_root / "summary.json")}), flush=True)
        return 0 if result["status"] == "completed" else 2
    except KeyboardInterrupt:
        path = args.output_root / "summary.json"
        if path.exists():
            summary = control.read(path)
            summary.update(status="stopped", active=None, updated_at=control.now())
            write(path, summary, replace=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

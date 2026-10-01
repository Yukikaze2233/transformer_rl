"""Bounded simulator integration checks; never a model-quality comparison.

Run this explicitly in an Isaac Lab runtime after prepare-chassis. Each network
gets the same tiny update budget, an exact learning-state resume, a fixed-case
evaluation, export validation and CPU runtime timing. Simulator workers are
isolated processes. The output records pipeline readiness, not a trained winner.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from transformer_rl.chassis_adapter import network_variants
from transformer_rl.frame_checkpoint import load_frame_checkpoint
from transformer_rl.frame_config import FrameTrainConfig
from transformer_rl.frame_policy import FramePolicyConfig


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(arguments, log, timeout):
    command = [sys.executable, "-m", "transformer_rl.frame_process", *map(str, arguments)]
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            raise
    if code:
        raise RuntimeError(f"worker exited {code}; inspect {log}")


def optimizer_steps(trainer):
    return [int(state["step"].item()) for state in trainer.optimizer.state.values() if "step" in state]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--evaluation-steps", type=int, default=256)
    parser.add_argument("--worker-timeout", type=float, default=600.)
    args = parser.parse_args()
    if args.updates < 1 or args.evaluation_steps < 1 or not math.isfinite(args.worker_timeout) or args.worker_timeout <= 30:
        parser.error("step budgets must be positive; worker timeout must be finite and >30 seconds")
    base = FrameTrainConfig.load(args.prepared / "base.json")
    spec = json.loads((args.prepared / "study.json").read_text())
    variants = network_variants()
    if args.variants is not None:
        unknown = set(args.variants) - {entry["name"] for entry in variants}
        if unknown or len(set(args.variants)) != len(args.variants):
            parser.error(f"unknown or repeated variants: {args.variants}")
        variants = [entry for entry in variants if entry["name"] in args.variants]
    continuous = [case for case in spec["scenarios"] if case.get("require_steady")]
    scenario = next((case for case in continuous if "305" in case["name"]), continuous[0])
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in variants:
        started = time.monotonic()
        root = directory / variant["name"]
        root.mkdir()
        policy = FramePolicyConfig.from_dict(variant["policy"])
        config = replace(base, model=replace(base.model, policy=policy))
        config_path = root / "training.json"
        write(config_path, config.to_dict())
        common = ["train", "--config", config_path, "--env-factory", spec["environment_factory"],
                  "--rollout-steps", spec["training"]["rollout_steps"], "--seed", 1101,
                  "--device", args.device, "--max-seconds", args.worker_timeout - 30]
        worker([*common, "--run-dir", root / "train", "--updates", args.updates], root / "train.log", args.worker_timeout)
        first = json.loads((root / "train/completion.json").read_text())
        worker([*common, "--run-dir", root / "resume", "--updates", 1, "--resume", first["checkpoint"],
                "--consumed-update-offset", args.updates], root / "resume.log", args.worker_timeout)
        resumed = json.loads((root / "resume/completion.json").read_text())
        old_model, old_trainer, *_ = load_frame_checkpoint(first["checkpoint"])
        new_model, new_trainer, *_ = load_frame_checkpoint(resumed["checkpoint"])
        changed = any(not torch.equal(parameter, dict(old_model.actor.policy.named_parameters())[name])
                      for name, parameter in new_model.actor.policy.named_parameters())
        old_steps, new_steps = optimizer_steps(old_trainer), optimizer_steps(new_trainer)
        if (resumed["final_update"] != args.updates + 1 or not changed or not old_steps
                or len(old_steps) != len(new_steps) or any(new <= old for old, new in zip(old_steps, new_steps))):
            raise RuntimeError("resume did not advance actor weights and the existing optimizer")
        evaluated = replace(config, environment={**base.environment, **scenario["environment"]})
        evaluation_path = root / "evaluation.json"
        write(evaluation_path, evaluated.to_dict())
        worker(["evaluate", "--checkpoint", resumed["checkpoint"], "--config", evaluation_path,
                "--env-factory", spec["environment_factory"], "--steps", args.evaluation_steps,
                "--seed", 701, "--device", args.device, "--output", root / "evaluation-report.json"],
               root / "evaluation.log", args.worker_timeout)
        worker(["export", "--checkpoint", resumed["checkpoint"], "--directory", root / "deployment"],
               root / "export.log", args.worker_timeout)
        worker(["benchmark", "--directory", root / "deployment", "--output", root / "latency.json"],
               root / "benchmark.log", args.worker_timeout)
        results.append({"variant": variant["name"], "architecture": policy.architecture,
            "mean_parameters": sum(parameter.numel() for parameter in new_model.actor.policy.parameters()),
            "training": first, "resume": resumed, "actor_mean_changed_during_resume": changed,
            "old_optimizer_step_min": min(old_steps), "resumed_optimizer_step_min": min(new_steps),
            "evaluation": str(root / "evaluation-report.json"), "latency": str(root / "latency.json"),
            "export_manifest": str(root / "deployment/manifest.json"), "elapsed_s": time.monotonic() - started})
        write(root / "result.json", results[-1])
        print(json.dumps({"completed": variant["name"], "elapsed_s": results[-1]["elapsed_s"]}), flush=True)
    write(directory / "completion.json", {"status": "passed", "purpose": "bounded_pipeline_integration",
        "model_quality_comparison": False, "selected_model": None, "results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

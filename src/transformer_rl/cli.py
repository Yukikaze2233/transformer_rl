"""Inspect, train with an explicit environment factory, or export a saved actor."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import traceback

import torch

from . import __version__
from .config import ModelConfig, PPOConfig, config_dict, load_config
from .model import ActorCritic


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    results = {}
    for key, arguments in (("commit", ["rev-parse", "HEAD"]),
                           ("status", ["status", "--porcelain"])):
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments], capture_output=True,
                text=True, timeout=5, check=False,
            )
            results[key] = completed.stdout.strip() if completed.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            results[key] = None
    return {"commit": results["commit"], "dirty": bool(results["status"])
            if results["status"] is not None else None}


class _StopBudget:
    """Flag-only signals and a soft deadline checked at collection/update boundaries."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds
        self.reason: str | None = None
        self.previous: dict[int, object] = {}

    def _signal(self, number, _frame):
        self.reason = signal.Signals(number).name

    def install(self) -> None:
        for number in (signal.SIGINT, signal.SIGTERM):
            if number not in self.previous:
                self.previous[number] = signal.getsignal(number)
            signal.signal(number, self._signal)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *_):
        for number, handler in self.previous.items():
            signal.signal(number, handler)

    def stopped(self) -> bool:
        if self.reason is None and time.monotonic() >= self.deadline:
            self.reason = "time_budget"
        return self.reason is not None


def _factory(reference: str):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("env-factory must be module_name:callable_name")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("env-factory must resolve to a callable")
    return factory


def _train(args, model_config: ModelConfig, ppo_config: PPOConfig, environment: dict) -> dict:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .ppo import PPOTrainer
    from .runner import RolloutCollector

    run_dir = args.run_dir
    # exist_ok=False also rejects symlinks and protects previous runs.
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    metadata = {
        "environment_factory": args.env_factory,
        "environment": environment,
        "action_clip": args.action_clip,
        "seed": args.seed,
        "source": _source_identity(),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    _write_json(run_dir / "run.json", {
        **config_dict(model_config, ppo_config, environment),
        "metadata": metadata,
        "requested_updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "max_seconds": args.max_seconds,
        "device": args.device,
        "resume": str(args.resume) if args.resume else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    model = trainer = collector = env = None
    update = start_update = 0
    update_in_progress = False
    saved: list[dict] = []
    started = time.monotonic()
    with _StopBudget(args.max_seconds) as budget:
        try:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.resume:
                model, trainer, update, origin = load_checkpoint(args.resume, device="cpu")
                if model.config != model_config or trainer.config != ppo_config:
                    raise ValueError("resume model/PPO configuration differs from checkpoint")
                for key in ("environment_factory", "environment", "action_clip"):
                    if key not in origin or origin[key] != metadata[key]:
                        raise ValueError(f"resume environment contract differs: {key}")
                start_update = update
                metadata["resume_source"] = str(args.resume)
            # The factory owns simulator startup. CUDA model placement follows it.
            factory = _factory(args.env_factory)
            if budget.stopped():
                raise TimeoutError("budget expired before environment construction")
            env = factory(model_config=model_config, environment_config=environment,
                          device=torch.device(args.device))
            budget.install()  # Reclaim flag-only handlers if an SDK installed its own.
            provenance = json.loads(json.dumps(getattr(env, "metadata", {}), allow_nan=False))
            if not isinstance(provenance, dict):
                raise ValueError("environment metadata must be a JSON object")
            if args.resume:
                expected_identity = origin.get("environment_provenance", {}).get("identity")
                if expected_identity is not None and provenance.get("identity") != expected_identity:
                    raise ValueError("resume environment identity differs from checkpoint")
            metadata["environment_provenance"] = provenance
            _write_json(run_dir / "environment.json", provenance)
            if args.resume:
                # Rebind Adam after device conversion via the checkpoint loader.
                model, trainer, _, _ = load_checkpoint(args.resume, device=args.device)
            else:
                model = ActorCritic(model_config).to(args.device)
                trainer = PPOTrainer(model, ppo_config)
            collector = RolloutCollector(env, model, ppo_config, args.action_clip)
            if not budget.stopped():
                collector.reset(seed=args.seed)
            with (run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
                for _ in range(args.updates):
                    if budget.stopped():
                        break
                    batch = collector.collect(args.rollout_steps, should_stop=budget.stopped)
                    if batch is None or budget.stopped():
                        break
                    update_in_progress = True
                    learned = trainer.update(batch)
                    update_in_progress = False
                    update += 1
                    row = {"update": update, "collection": collector.last_metrics,
                           "optimization": learned, "elapsed_s": time.monotonic() - started}
                    encoded = json.dumps(row, allow_nan=False)
                    metrics.write(encoded + "\n")
                    metrics.flush()
                    print(encoded, flush=True)
                    if update % args.checkpoint_interval == 0:
                        saved.append(save_checkpoint(
                            checkpoints / f"checkpoint_{update:06d}.pt", model, trainer,
                            update, {**metadata, "collected_transitions": collector.total_transitions},
                        ))
            # Environment cleanup is part of successful completion, not a detached process.
            closing, env = env, None
            closing.close()
            if not saved or saved[-1]["update"] != update:
                saved.append(save_checkpoint(
                    checkpoints / f"checkpoint_{update:06d}.pt", model, trainer,
                    update, {**metadata, "collected_transitions": collector.total_transitions},
                ))
            result = {
                "status": "completed" if update - start_update == args.updates else "stopped",
                "stop_reason": budget.reason or "updates_completed",
                "updates_completed": update - start_update,
                "cumulative_update": update,
                "collected_transitions": collector.total_transitions,
                "elapsed_s": time.monotonic() - started,
                "checkpoints": saved,
                "environment_provenance": provenance,
                "scope": "optimization execution; control quality requires independent evaluation",
            }
            _write_json(run_dir / "completion.json", result)
            return result
        except BaseException as error:
            _write_json(run_dir / "failure.json", {
                "exception": type(error).__name__, "message": str(error),
                "cumulative_completed_update": update,
                "optimizer_update_may_be_partial": update_in_progress,
                "collected_transitions": collector.total_transitions if collector else 0,
                "elapsed_s": time.monotonic() - started,
            })
            raise
        finally:
            if env is not None:
                env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    inspect = subparsers.add_parser("inspect", help="Inspect architecture without environment interaction")
    inspect.add_argument("--config", type=Path)
    train = subparsers.add_parser("train", help="Train using an explicitly provided vector environment")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--env-factory", required=True, help="module:factory returning the VectorEnv protocol")
    train.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    train.add_argument("--updates", type=_positive_int, required=True, help="Additional updates in this invocation")
    train.add_argument("--rollout-steps", type=_positive_int, default=48)
    train.add_argument("--max-seconds", type=_positive_seconds, default=3600)
    train.add_argument("--checkpoint-interval", type=_positive_int, default=100)
    train.add_argument("--action-clip", type=_positive_seconds, default=None)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--resume", type=Path)
    export = subparsers.add_parser("export", help="Export and verify deterministic ONNX policy")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint with independent scenario seeds")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--env-factory", required=True)
    evaluate.add_argument("--steps", type=_positive_int, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--action-clip", type=_positive_seconds)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "export":
            from .export import export_policy
            result = export_policy(args.checkpoint, args.output)
        elif args.operation == "evaluate":
            from .checkpoint import load_checkpoint
            from .evaluation import evaluate_policy
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("evaluation output already exists")
            if not args.output.parent.is_dir():
                raise FileNotFoundError("evaluation output parent does not exist")
            model_config, _, environment = load_config(args.config)
            model, _, _, _ = load_checkpoint(args.checkpoint)
            if model.config != model_config:
                raise ValueError("evaluation model configuration differs from checkpoint")
            result = evaluate_policy(args.checkpoint, _factory(args.env_factory), environment,
                                     args.steps, args.seed, args.device, args.action_clip)
            _write_json(args.output, result)
        else:
            model_config, ppo_config, environment = (load_config(args.config) if args.config
                                                      else (ModelConfig(), PPOConfig(), {}))
            if args.operation == "inspect":
                with torch.random.fork_rng(devices=[]):
                    model = ActorCritic(model_config)
                result = {
                    "architecture": model.actor.describe(),
                    "model": asdict(model_config),
                    "frame_dim": model_config.frame_dim,
                    "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
                    "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
                    "environment_started": False,
                }
            else:
                result = _train(args, model_config, ppo_config, environment)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1

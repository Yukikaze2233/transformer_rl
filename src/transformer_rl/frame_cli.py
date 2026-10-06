"""Train, compare, evaluate and export packed-frame control networks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from .cli import _factory, _nonnegative_int, _positive_int, _positive_seconds, _write_json
from .frame_config import FrameTrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    inspect = commands.add_parser("inspect", help="Validate a packed training configuration without simulation")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--num-envs", type=_positive_int, default=4096)
    inspect.add_argument("--rollout-steps", type=_positive_int, default=48)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--env-factory", required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--updates", type=_positive_int, required=True)
    train.add_argument("--rollout-steps", type=_positive_int, default=48)
    train.add_argument("--seed", type=_nonnegative_int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-seconds", type=_positive_seconds, default=3600.)
    train.add_argument("--checkpoint-interval", type=_positive_int, default=40)
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-from", type=Path)
    initialization.add_argument("--restore-learning-from", type=Path)
    train.add_argument("--anchors", type=Path, nargs="*", default=[])
    train.add_argument("--retention-coef", type=float, default=0.)
    train.add_argument("--no-tensorboard", action="store_true")
    train.add_argument("--consumed-update-offset", type=_nonnegative_int, default=0)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--env-factory", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--steps", type=_positive_int, required=True)
    evaluate.add_argument("--seed", type=_nonnegative_int, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--settle-steps", type=_nonnegative_int, default=200)
    evaluate.add_argument("--min-steady-samples", type=_positive_int, default=200)
    evaluate.add_argument("--anchor-output", type=Path)
    evaluate.add_argument("--max-anchors", type=_positive_int, default=256)
    evaluate.add_argument("--control-output", type=Path)
    evaluate.add_argument("--trace-output", type=Path)
    evaluate.add_argument("--trace-replicas", type=_positive_int, default=2)
    suite = commands.add_parser("evaluate-suite", help="Batch compatible frozen chassis scenarios into one simulation")
    suite.add_argument("--checkpoint", type=Path, required=True)
    suite.add_argument("--configs", type=Path, nargs="+", required=True)
    suite.add_argument("--outputs", type=Path, nargs="+", required=True)
    suite.add_argument("--steps", type=_positive_int, required=True)
    suite.add_argument("--seed", type=_nonnegative_int, required=True)
    suite.add_argument("--device", default="cpu")
    suite.add_argument("--settle-steps", type=_nonnegative_int, default=200)
    suite.add_argument("--min-steady-samples", type=_positive_int, default=200)
    suite.add_argument("--anchor-directory", type=Path)
    suite.add_argument("--max-anchors", type=_positive_int, default=256)
    suite.add_argument("--control-output", type=Path)
    suite.add_argument("--trace-output", type=Path)
    suite.add_argument("--trace-replicas", type=_positive_int, default=2)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--directory", type=Path, required=True)
    export.add_argument("--torchscript-only", action="store_true")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--directory", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--backend", choices=("onnx", "torchscript"), default="onnx")
    benchmark.add_argument("--threads", type=_positive_int, default=1)
    benchmark.add_argument("--iterations", type=_positive_int, default=1000)
    planner = commands.add_parser("plan")
    planner.add_argument("--spec", type=Path, required=True)
    planner.add_argument("--root", type=Path, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--max-parallel", type=_positive_int, default=1)
    select = commands.add_parser("select")
    select.add_argument("--root", type=Path, required=True)
    prepare = commands.add_parser("prepare-chassis", help="Freeze a new 100 Hz study from the existing curriculum")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--curriculum", type=Path, required=True)
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--num-envs", type=_positive_int, default=4096)
    prepare.add_argument("--evaluation-replicas", type=_positive_int, default=16)
    prepare.add_argument("--round", choices=("screen", "confirm"), default="screen")
    transfer = commands.add_parser("prepare-transfer", help="Freeze a 100 Hz transfer study from a materialized V6 task")
    transfer.add_argument("--source-root", type=Path, required=True)
    transfer.add_argument("--task-contract", type=Path, required=True)
    transfer.add_argument("--directory", type=Path, required=True)
    transfer.add_argument("--num-envs", type=_positive_int, default=1024)
    transfer.add_argument("--evaluation-replicas", type=_positive_int, default=8)
    transfer.add_argument("--updates", type=_positive_int, default=1200)
    transfer.add_argument("--seeds", type=_nonnegative_int, nargs="+", default=[1101])
    curriculum = commands.add_parser("prepare-curriculum", help="Freeze matched Gated task curricula from a prepared transfer study")
    curriculum.add_argument("--base-config", type=Path, required=True)
    curriculum.add_argument("--directory", type=Path, required=True)
    curriculum.add_argument("--warmup-updates", type=_positive_int, default=400)
    curriculum.add_argument("--updates", type=_positive_int, default=1200)
    curriculum.add_argument("--seeds", type=_nonnegative_int, nargs="+", default=[1101, 1102, 1103])
    args = parser.parse_args(argv)
    try:
        if args.operation == "inspect":
            from .frame_training import FrameActorCritic
            import torch
            config = FrameTrainConfig.load(args.config)
            with torch.random.fork_rng(devices=[]):
                model = FrameActorCritic(config.model)
            # Reference storage holds and then flattens endpoint histories; this excludes activations/optimizer.
            history_bytes = args.num_envs * args.rollout_steps * config.model.history_length * (4 * config.model.frame_dim + 9)
            result = {"actor": model.actor.describe(), "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
                      "policy_hz": 1 / config.control["policy_dt_s"],
                      "history_span_s": (config.model.history_length - 1) * config.control["policy_dt_s"],
                      "rollout_seconds_per_env": args.rollout_steps * config.control["policy_dt_s"],
                      "history_peak_gib_lower_bound": 2 * history_bytes / 1024**3, "environment_started": False}
        elif args.operation == "train":
            from .frame_workflow import train_frame_policy
            result = train_frame_policy(FrameTrainConfig.load(args.config), _factory(args.env_factory), args.env_factory,
                args.run_dir, updates=args.updates, rollout_steps=args.rollout_steps, seed=args.seed, device=args.device,
                max_seconds=args.max_seconds, checkpoint_interval=args.checkpoint_interval, resume=args.resume,
                initialize_from=args.initialize_from, anchors=args.anchors, retention_coef=args.retention_coef,
                restore_learning_from=args.restore_learning_from,
                tensorboard=not args.no_tensorboard, consumed_update_offset=args.consumed_update_offset)
            if result["status"] != "completed":
                print(json.dumps(result, indent=2))
                return 2
        elif args.operation == "evaluate":
            from .frame_checkpoint import load_frame_checkpoint
            from .frame_workflow import evaluate_frame_policy
            config = FrameTrainConfig.load(args.config)
            _, _, saved, _, _, _ = load_frame_checkpoint(args.checkpoint)
            if config.model != saved.model or config.control != saved.control:
                raise ValueError("evaluation policy/control configuration differs from checkpoint")
            if args.output.exists():
                raise FileExistsError(args.output)
            for path in (args.control_output, args.trace_output):
                if path is not None and path.exists():
                    raise FileExistsError(path)
            if args.trace_output is not None and args.control_output is None:
                raise ValueError("--trace-output requires --control-output")
            result = evaluate_frame_policy(args.checkpoint, _factory(args.env_factory), config.environment,
                steps=args.steps, seed=args.seed, device=args.device, settle_steps=args.settle_steps,
                min_steady_samples=args.min_steady_samples, anchor_output=args.anchor_output, max_anchors=args.max_anchors,
                control_metrics=args.control_output is not None, trace_output=args.trace_output, trace_replicas=args.trace_replicas)
            _write_json(args.output, result)
            if args.control_output is not None:
                _write_json(args.control_output, {"format": "transformer_rl.control_evaluation", "schema_version": 1,
                    "checkpoint_sha256": result["checkpoint_sha256"], "checkpoint_update": result["checkpoint_update"],
                    "seed": args.seed, "steps": args.steps, "environment_provenance": result["environment_provenance"],
                    "control": result["control"], "trace": result.get("trace")})
        elif args.operation == "evaluate-suite":
            from .chassis_adapter import evaluate_suite
            result = evaluate_suite(args.checkpoint, args.configs, args.outputs, steps=args.steps, seed=args.seed,
                device=args.device, settle_steps=args.settle_steps, min_steady_samples=args.min_steady_samples,
                anchor_directory=args.anchor_directory, max_anchors=args.max_anchors, control_output=args.control_output,
                trace_output=args.trace_output, trace_replicas=args.trace_replicas)
        elif args.operation == "export":
            from .frame_export import export_frame_policy
            result = export_frame_policy(args.checkpoint, args.directory, onnx=not args.torchscript_only)
        elif args.operation == "benchmark":
            import hashlib
            import platform
            import os
            from importlib.metadata import PackageNotFoundError, version
            from .frame_runtime import FrameRuntime
            manifest = json.loads((args.directory / "manifest.json").read_text())
            control = manifest["control"]
            runtime = FrameRuntime(args.directory, observation_schema=control["observation_schema"],
                policy_dt_s=control["policy_dt_s"], backend=args.backend, threads=args.threads)
            try:
                ort_version = version("onnxruntime")
            except PackageNotFoundError:
                ort_version = None
            result = {**runtime.benchmark(iterations=args.iterations), "threads": args.threads,
                "machine": {"node": platform.node(), "system": platform.system(), "architecture": platform.machine(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count(),
                            "cpu_affinity": sorted(os.sched_getaffinity(0)),
                            "onnxruntime": ort_version},
                "manifest_sha256": hashlib.sha256((args.directory / "manifest.json").read_bytes()).hexdigest()}
            _write_json(args.output, result)
        elif args.operation == "prepare-chassis":
            from .chassis_adapter import prepare_study
            result = prepare_study(args.source_root, args.curriculum, args.directory,
                num_envs=args.num_envs, evaluation_replicas=args.evaluation_replicas, round_name=args.round)
        elif args.operation == "prepare-transfer":
            from .transfer_study import prepare_transfer_study
            result = prepare_transfer_study(args.source_root, args.task_contract, args.directory,
                num_envs=args.num_envs, evaluation_replicas=args.evaluation_replicas,
                updates=args.updates, seeds=args.seeds)
        elif args.operation == "prepare-curriculum":
            from .curriculum_study import prepare_curriculum_study
            result = prepare_curriculum_study(args.base_config, args.directory,
                warmup_updates=args.warmup_updates, total_updates=args.updates, seeds=args.seeds)
        else:
            from .frame_study import plan_study, run_study, select_transformer
            if args.operation == "plan":
                result = plan_study(args.spec, args.root)
            elif args.operation == "run":
                result = run_study(args.root, max_parallel=args.max_parallel)
                if result["status"] != "completed":
                    print(json.dumps(result, indent=2))
                    return 2
            else:
                result = select_transformer(args.root)
                if result["status"] != "selected":
                    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
                    return 2
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

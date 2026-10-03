"""Bounded packed-policy training and independent deterministic evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import platform
import time

import numpy as np
import torch

from .adapters import _TensorEnvContract
from .cli import _StopBudget, _write_json
from .evaluation import _MetricAccumulator
from .experiments import source_identity
from .frame_checkpoint import load_frame_checkpoint, restore_rng, save_frame_checkpoint
from .frame_config import FrameTrainConfig, digest
from .frame_training import FrameActorCritic, FrameCollector, FrameHistory
from .ppo import PPOTrainer
from .retention import AnchorRegularizer, save_anchors
from .stability import EpisodeSignalStatistics


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _provenance(env, control):
    value = getattr(env, "metadata", {})
    if not isinstance(value, dict) or not value.get("identity"):
        raise ValueError("environment metadata requires a reproducible identity")
    if value.get("control_sha256") != digest(control):
        raise ValueError("environment observation/action/timing contract differs from training")
    return json.loads(json.dumps(value, allow_nan=False))


def train_frame_policy(config, env_factory, env_reference, run_dir, *, updates, rollout_steps=48,
                       seed=0, device="cpu", max_seconds=3600., checkpoint_interval=40,
                       resume=None, initialize_from=None, restore_learning_from=None, anchors=(), retention_coef=0., tensorboard=True,
                       consumed_update_offset=0):
    """Learning-state resume resets episodes/history; weights transfer starts new Adam.

    Updates are additional successful PPO updates. The external study ledger
    accounts for all consumed updates, including failed stages and rollbacks.
    SIGINT/SIGTERM stop at a collector/update boundary and save learning state.
    """
    if not isinstance(config, FrameTrainConfig) or not callable(env_factory):
        raise TypeError("training requires a parsed config and environment factory")
    for name, value in (("updates", updates), ("rollout_steps", rollout_steps), ("checkpoint_interval", checkpoint_interval)):
        _positive_integer(value, name)
    if type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if sum(path is not None for path in (resume, initialize_from, restore_learning_from)) > 1:
        raise ValueError("choose exact learning-state resume or weights initialization")
    if type(consumed_update_offset) is not int or consumed_update_offset < 0:
        raise ValueError("consumed_update_offset must be nonnegative")
    if (type(retention_coef) not in (int, float) or not math.isfinite(retention_coef) or retention_coef < 0
            or (anchors and retention_coef == 0) or (retention_coef > 0 and not anchors)):
        raise ValueError("retention coefficient and explicit anchor paths must be supplied together")
    _seed(seed)
    model = trainer = parent_rng = None
    parent_metadata = {}
    start_update = 0
    restoring = resume is not None or restore_learning_from is not None
    if restoring or initialize_from is not None:
        model, trainer, parent_config, parent_update, parent_metadata, parent_rng = load_frame_checkpoint(resume or initialize_from or restore_learning_from)
        if resume is not None:
            if parent_config.to_dict() != config.to_dict() or parent_metadata.get("environment_factory") != env_reference:
                raise ValueError("resume requires the exact configuration and environment factory")
            start_update = parent_update
        elif parent_config.model != config.model or parent_config.control != config.control:
            raise ValueError("stage transfer requires the same actor, critic and control contract")
        if restore_learning_from:
            if parent_config.ppo != config.ppo:
                raise ValueError("learning-state rollback requires the same optimizer recipe")
            start_update = parent_update
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    _write_json(run_dir / "run.json", {"config": config.to_dict(), "seed": seed, "environment_factory": env_reference,
        "updates": updates, "rollout_steps": rollout_steps, "device": str(device), "source": source_identity(),
        "resume": str(resume) if resume else None, "initialize_from": str(initialize_from) if initialize_from else None,
        "restore_learning_from": str(restore_learning_from) if restore_learning_from else None,
        "episode_state_restored": False, "history_reset": "repeat_first", "retention_coef": retention_coef,
        "max_seconds": max_seconds, "checkpoint_interval": checkpoint_interval})
    env = writer = None
    started = time.monotonic()
    update = start_update
    collector = None
    final_path = None
    attempted_updates = 0
    try:
        with _StopBudget(max_seconds) as stop:
            env = env_factory(model_config=config.model, environment_config=config.environment, device=torch.device(device))
            provenance = _provenance(env, config.control)
            if restoring and parent_metadata.get("environment_provenance", {}).get("identity") != provenance["identity"]:
                raise ValueError("resume environment source/assets differ from checkpoint")
            _write_json(run_dir / "environment.json", provenance)
            if model is None:
                model = FrameActorCritic(config.model).to(device)
                trainer = PPOTrainer(model, config.ppo)
            else:
                model.to(device)
                old_optimizer = trainer.optimizer.state_dict() if restoring else None
                trainer = PPOTrainer(model, config.ppo)
                if old_optimizer is not None:
                    trainer.optimizer.load_state_dict(old_optimizer)
            regularizer = AnchorRegularizer(model.actor, config, anchors, retention_coef) if anchors else None
            anchor_identity = regularizer.identities if regularizer else []
            if resume and (parent_metadata.get("anchors", []) != anchor_identity
                           or parent_metadata.get("retention_coef", 0.) != retention_coef):
                raise ValueError("resume retention objective differs from checkpoint")
            metadata = {"environment_factory": env_reference, "environment_provenance": provenance,
                        "seed": seed, "source": source_identity(), "anchors": anchor_identity,
                        "retention_coef": retention_coef, "episode_state_restored": False,
                        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                                    "numpy": np.__version__, "cuda": torch.version.cuda,
                                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
            collector = FrameCollector(env, model, config.ppo, config.control["action_bounds"])
            collector.reset(seed=seed)
            if restoring:
                # Environment creation/reset may consume global RNG; restore the learner last.
                restore_rng(parent_rng)
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(run_dir / "tensorboard"))
            prior_transitions = parent_metadata.get("collected_transitions", 0) if restoring else 0
            with (run_dir / "metrics.jsonl").open("x", buffering=1) as log:
                while update < start_update + updates and not stop.stopped():
                    progress = getattr(env, "set_training_progress", None)
                    if progress is not None:
                        progress(consumed_update_offset + attempted_updates, prior_transitions + collector.total_transitions)
                    batch = collector.collect(rollout_steps, should_stop=stop.stopped)
                    if batch is None:
                        break
                    attempted_updates += 1
                    metrics = trainer.update(batch, diagnostics=True, regularizer=regularizer)
                    update += 1
                    record = {"update": update, "batch_samples": len(batch), "optimization": metrics,
                              "collection": collector.last_metrics, "elapsed_s": time.monotonic() - started}
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    if writer:
                        for group, items in (("ppo", metrics), ("rollout", collector.last_metrics)):
                            for name, value in items.items():
                                if type(value) in (float, int, bool):
                                    writer.add_scalar(f"{group}/{name}", value, update)
                    print(json.dumps({"update": update, "reward_mean": collector.last_metrics["reward_mean"],
                                      "transitions": collector.total_transitions}), flush=True)
                    metadata["collected_transitions"] = prior_transitions + collector.total_transitions
                    if update % checkpoint_interval == 0:
                        save_frame_checkpoint(checkpoint_dir / f"update_{update:06d}.pt", model, trainer, config, update, metadata)
            metadata["collected_transitions"] = prior_transitions + collector.total_transitions
            final_path = checkpoint_dir / "final.pt"
            save_frame_checkpoint(final_path, model, trainer, config, update, metadata)
            report = {"status": "completed" if update == start_update + updates else "stopped",
                      "stop_reason": stop.reason, "start_update": start_update, "final_update": update,
                      "completed_updates": update - start_update, "consumed_transitions": collector.total_transitions,
                      "attempted_updates": attempted_updates,
                      "cumulative_transitions": metadata["collected_transitions"], "checkpoint": str(final_path),
                      "checkpoint_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
                      "elapsed_s": time.monotonic() - started, "config_sha256": digest(config.to_dict())}
            _write_json(run_dir / "completion.json", report)
            return report
    except BaseException as error:
        # Preserve successfully applied updates, even if the next rollout or optimizer failed.
        _write_json(run_dir / "failure.json", {"status": "failed", "completed_updates": update - start_update,
            "attempted_updates": attempted_updates,
            "consumed_transitions": collector.total_transitions if collector else 0,
            "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if writer:
            writer.close()
        if env:
            env.close()


@torch.no_grad()
def evaluate_frame_policy(checkpoint, env_factory, environment, *, steps, seed, device="cpu",
                          settle_steps=200, min_steady_samples=200, anchor_output=None, max_anchors=2048,
                          group_anchor_directory=None, control_metrics=False, trace_output=None, trace_replicas=2):
    """Task metrics and episode_success are owned PRE-reset environment diagnostics."""
    _positive_integer(steps, "steps")
    _positive_integer(max_anchors, "max_anchors")
    EpisodeSignalStatistics.validate_protocol(settle_steps, min_steady_samples)
    if trace_output is not None and not control_metrics:
        raise ValueError("trace_output requires control_metrics=True")
    if trace_output is not None and Path(trace_output).exists():
        raise FileExistsError(trace_output)
    _seed(seed)
    model, _, config, update, metadata, _ = load_frame_checkpoint(checkpoint)
    env = env_factory(model_config=config.model, environment_config=environment, device=torch.device(device))
    trace = None
    try:
        provenance = _provenance(env, config.control)
        if provenance["identity"] != metadata["environment_provenance"]["identity"]:
            raise ValueError("evaluation source/assets differ from training")
        model.to(device).eval()
        controls = None
        if control_metrics:
            from .control_metrics import ControlMetrics
            env.enable_control_metrics = True
            controls = ControlMetrics(env.num_envs, config.control["policy_dt_s"],
                settle_steps=settle_steps, min_steady_samples=min_steady_samples)
        contract = _TensorEnvContract(model.config, env.num_envs, env.device)
        history = FrameHistory(model.config, env.num_envs, env.device)
        current = history.append(contract.observation(env.reset(seed=seed)))
        group_labels = provenance.get("evaluation_groups")
        groups = {}
        if group_labels is not None:
            if (not isinstance(group_labels, list) or len(group_labels) != env.num_envs
                    or any(not isinstance(name, str) or not name for name in group_labels)):
                raise ValueError("evaluation_groups must identify every environment row")
            for name in sorted(set(group_labels)):
                indices = torch.tensor([i for i, label in enumerate(group_labels) if label == name], device=device)
                groups[name] = {"indices": indices, "completed": 0, "successes": 0, "failures": 0,
                    "reward": _MetricAccumulator(), "metrics": {}, "frames": [], "mean": [], "std": [], "anchor_count": 0,
                    "statistics": EpisodeSignalStatistics(len(indices), settle_steps=settle_steps, min_steady_samples=min_steady_samples)}
                if control_metrics:
                    groups[name]["control"] = ControlMetrics(len(indices), config.control["policy_dt_s"],
                        settle_steps=settle_steps, min_steady_samples=min_steady_samples)
        checkpoint_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if trace_output is not None:
            from .control_trace import ControlTrace
            trace = ControlTrace(trace_output, steps=steps, num_envs=env.num_envs, replicas=trace_replicas,
                groups=group_labels, metadata={"checkpoint_sha256": checkpoint_sha, "checkpoint_update": update,
                    "seed": seed, "policy_dt_s": config.control["policy_dt_s"],
                    "sampling_hz": 1 / config.control["policy_dt_s"], "control_sha256": digest(config.control)})
        rewards, metrics = _MetricAccumulator(), {}
        names = None
        completed = successes = failures = 0
        success_available = None
        bounds = torch.tensor(config.control["action_bounds"], device=device)
        statistics = EpisodeSignalStatistics(env.num_envs, settle_steps=settle_steps, min_steady_samples=min_steady_samples)
        ages = torch.zeros(env.num_envs, dtype=torch.int64, device=device)
        anchor_frames, anchor_mean, anchor_std = [], [], []
        anchor_count = 0
        capture_interval = max(1, math.ceil(steps * env.num_envs / max_anchors))
        for index in range(steps):
            mean = contract.tensor("mean", model.actor(current), (env.num_envs, model.config.action_dim))
            action = mean.clamp(-bounds, bounds)
            if anchor_output is not None and index % capture_interval == 0 and anchor_count < max_anchors:
                valid = torch.arange(env.num_envs, device=device)[:max_anchors - anchor_count]
                if len(valid):
                    anchor_frames.append(current.frames[valid].cpu())
                    anchor_mean.append(mean[valid].cpu())
                    anchor_std.append(model.actor.log_std.exp().expand_as(mean)[valid].cpu())
                    anchor_count += len(valid)
            result = contract.step(env.step(action.clone()))
            done = result.terminated | result.truncated
            packet = None
            if controls is not None:
                packet = result.info.get("control_packet")
                if not isinstance(packet, dict):
                    raise ValueError("control evaluation requires PRE-reset control_packet tensors")
                controls.update(packet, done)
                if trace is not None:
                    trace.add(packet, done)
            present = "episode_success" in result.info
            if success_available is not None and present != success_available:
                raise ValueError("episode_success availability changed")
            success_available = present
            success = torch.zeros_like(done)
            if present:
                success = contract.tensor("episode_success", result.info["episode_success"], (env.num_envs,), boolean=True) & done
                successes += int(success.sum())
                failures += int((done & ~success).sum())
            completed += int(done.sum())
            rewards.add(result.reward)
            physical = result.info.get("evaluation_metrics", {})
            if not isinstance(physical, dict) or any(not isinstance(k, str) or not k for k in physical):
                raise ValueError("evaluation_metrics must map metric names to PRE-reset tensors")
            if names is None:
                names = set(physical)
                metrics = {name: _MetricAccumulator() for name in names}
            if set(physical) != names:
                raise ValueError("evaluation metric names changed")
            for name, values in physical.items():
                metrics[name].add(contract.tensor(name, values, (env.num_envs,)))
            statistics.update(result.info.get("evaluation_signals", {}), result.info.get("evaluation_signal_time"), done)
            for group in groups.values():
                rows = group["indices"]
                if controls is not None:
                    group["control"].update({name: value[rows] for name, value in packet.items()}, done[rows])
                group["completed"] += int(done[rows].sum())
                group["successes"] += int(success[rows].sum())
                group["failures"] += int((done[rows] & ~success[rows]).sum())
                group["reward"].add(result.reward[rows])
                for name, values in physical.items():
                    group["metrics"].setdefault(name, _MetricAccumulator()).add(values[rows])
                signals = {name: values[rows] for name, values in result.info.get("evaluation_signals", {}).items()}
                signal_time = result.info.get("evaluation_signal_time")
                group["statistics"].update(signals, signal_time[rows] if signal_time is not None else None, done[rows])
                interval = max(1, math.ceil(steps * len(rows) / max_anchors))
                if group_anchor_directory is not None and index % interval == 0 and group["anchor_count"] < max_anchors:
                    selected = rows[:max_anchors - group["anchor_count"]]
                    group["frames"].append(current.frames[selected].cpu())
                    group["mean"].append(mean[selected].cpu())
                    group["std"].append(model.actor.log_std.exp().expand_as(mean)[selected].cpu())
                    group["anchor_count"] += len(selected)
            ages += 1
            ages[done] = 0
            history.reset(done)
            current = history.append(result.observation)
        report = {"format": "transformer_rl.packed_evaluation", "schema_version": 1,
                  "checkpoint_sha256": checkpoint_sha, "checkpoint_update": update,
                  "model": config.to_dict()["model"], "control_sha256": digest(config.control),
                  "environment": environment, "environment_provenance": provenance,
                  "seed": seed, "steps": steps, "num_envs": env.num_envs, "transitions": steps * env.num_envs,
                  "completed_episodes": completed, "success_rate": successes / completed if present and completed else None,
                  "failed_episodes": failures if present else None, "success_metric_available": bool(present),
                  "reward_mean": rewards.report()["mean"], "metrics": {name: value.report() for name, value in metrics.items()},
                  "stability": statistics.report(), "policy": "deterministic_raw_mean_then_declared_action_limits"}
        if controls is not None:
            report["control"] = controls.report()
        if anchor_output is not None:
            if not anchor_count:
                raise ValueError("no behavior samples available for anchors")
            report["anchors"] = save_anchors(anchor_output, config, torch.cat(anchor_frames), torch.cat(anchor_mean),
                                              torch.cat(anchor_std), report["checkpoint_sha256"])
        if groups:
            report["groups"] = {}
            if group_anchor_directory is not None:
                Path(group_anchor_directory).mkdir(parents=True, exist_ok=False)
            for name, group in groups.items():
                grouped = {key: value for key, value in report.items() if key not in ("groups", "anchors")}
                grouped.update(num_envs=len(group["indices"]), transitions=steps * len(group["indices"]),
                    completed_episodes=group["completed"], failed_episodes=group["failures"] if present else None,
                    success_rate=group["successes"] / group["completed"] if present and group["completed"] else None,
                    reward_mean=group["reward"].report()["mean"],
                    metrics={key: values.report() for key, values in group["metrics"].items()},
                    stability=group["statistics"].report())
                if controls is not None:
                    grouped["control"] = group["control"].report()
                if group_anchor_directory is not None:
                    # Group names must be safe before being used as file routes.
                    import re
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                        raise ValueError("group anchor names require safe identifiers")
                    grouped["anchors"] = save_anchors(Path(group_anchor_directory) / f"{name}.pt", config,
                        torch.cat(group["frames"]), torch.cat(group["mean"]), torch.cat(group["std"]), report["checkpoint_sha256"])
                report["groups"][name] = grouped
        json.dumps(report, allow_nan=False)
        if trace is not None:
            report["trace"] = trace.publish()
        return report
    finally:
        if trace is not None:
            trace.close()
        env.close()

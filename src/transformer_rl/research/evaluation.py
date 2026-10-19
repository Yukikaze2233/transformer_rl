"""Independent deterministic policy evaluation with explicit task-owned metrics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .adapters import _TensorEnvContract
from .checkpoint import _load_checkpoint_bytes, _require_new_paths
from .control_quality import ControlQuality, EvaluationTrace
from .history import HistoryBuffer
from .stability import EpisodeSignalStatistics


class _MetricAccumulator:
    def __init__(self):
        self.count = 0
        self.total = self.square_total = 0.0
        self.minimum, self.maximum = math.inf, -math.inf

    def add(self, values):
        values = values.detach().double()
        self.count += values.numel()
        self.total += values.sum().item()
        self.square_total += values.square().sum().item()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())

    def report(self):
        return {"mean": self.total / self.count,
                "rms": math.sqrt(self.square_total / self.count),
                "min": self.minimum, "max": self.maximum, "count": self.count}


@torch.no_grad()
def evaluate_policy(checkpoint_path, env_factory, environment_config, steps, seed,
                    device, action_clip=None, *, settle_steps=200, min_steady_samples=200,
                    trace_output=None) -> dict:
    """Evaluate fixed mean actions; the factory owns scenario and physical semantics.

    Metrics must be PRE-reset float tensors [N], with the same names each step.
    No optimizer updates occur. Counts describe the complete sampled interval,
    including initial transients and any auto-resets; they do not prove a
    continuous no-reset standing interval or convergence.
    Optional evaluation_signals use separate PRE-reset float64 signal times.
    Stability discards settle_steps samples per episode, then requires at least
    min_steady_samples retained samples; these units are steps, not measured seconds.
    """
    EpisodeSignalStatistics.validate_protocol(settle_steps, min_steady_samples)
    if type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0:
        raise ValueError("steps must be positive and seed nonnegative integers")
    if not callable(env_factory) or not isinstance(environment_config, dict):
        raise TypeError("evaluation requires a factory and environment object")
    if action_clip is not None and (type(action_clip) not in (int, float)
                                   or not math.isfinite(action_clip) or action_clip <= 0):
        raise ValueError("action_clip must be None or finite and positive")
    environment_config = json.loads(json.dumps(environment_config, allow_nan=False))
    if trace_output is not None:
        _require_new_paths([Path(trace_output)])
    data = Path(checkpoint_path).read_bytes()
    model, _, update, metadata = _load_checkpoint_bytes(data, device="cpu")
    if "action_clip" in metadata and metadata["action_clip"] != action_clip:
        raise ValueError("evaluation action_clip differs from training checkpoint")
    random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    env = env_factory(model_config=model.config, environment_config=environment_config,
                      device=torch.device(device))
    try:
        # Keep SDK initialization ahead of CUDA model placement.
        model.to(device).eval()
        contract = _TensorEnvContract(model.config, env.num_envs, env.device)
        provenance = json.loads(json.dumps(getattr(env, "metadata", {}), allow_nan=False))
        if not isinstance(provenance, dict):
            raise ValueError("environment metadata must be a JSON object")
        expected_identity = metadata.get("environment_provenance", {}).get("identity")
        if expected_identity is not None and provenance.get("identity") != expected_identity:
            raise ValueError("evaluation environment identity differs from training checkpoint")
        history = HistoryBuffer(model.config, env.num_envs, env.device)
        observation = contract.observation(env.reset(seed=seed))
        current = history.append(observation)
        rewards = _MetricAccumulator()
        metrics = {}
        metric_names = None
        stability = EpisodeSignalStatistics(env.num_envs, settle_steps=settle_steps,
                                            min_steady_samples=min_steady_samples)
        terminated = truncated = done_count = 0
        quality = ControlQuality(env.num_envs, settle_steps, min_steady_samples)
        trace = EvaluationTrace(env.num_envs) if trace_output is not None else None
        estimation_errors = []
        for _ in range(steps):
            # Do not sample a distribution or use its exploration std in evaluation.
            action = contract.tensor("policy mean", model.actor(current),
                                      (env.num_envs, model.config.action_dim))
            raw_mean = action.clone()
            estimated = target = action.new_empty((env.num_envs, 0))
            if model.config.estimator_type != "none":
                estimate, _ = model.actor.estimate(current)
                estimated = estimate * model.actor.state_scale + model.actor.state_offset
                target = observation.critic[:, model.config.state_indices]
                estimation_errors.append((estimated - target).double().cpu())
            if action_clip is not None:
                action = action.clamp(-action_clip, action_clip)
            result = contract.step(env.step(action.clone()))
            rewards.add(result.reward)
            terminated += result.terminated.sum().item()
            truncated += result.truncated.sum().item()
            done = result.terminated | result.truncated
            done_count += done.sum().item()
            physical = result.info.get("evaluation_metrics", {})
            if not isinstance(physical, dict) or any(type(k) is not str or not k for k in physical):
                raise ValueError("evaluation_metrics must map nonempty names to tensors")
            if metric_names is None:
                metric_names = set(physical)
                metrics = {name: _MetricAccumulator() for name in physical}
            if set(physical) != metric_names:
                raise ValueError("evaluation metric names must remain constant across steps")
            for name, value in physical.items():
                metrics[name].add(contract.tensor(f"evaluation_metrics.{name}", value, (env.num_envs,)))
            stability.update(result.info.get("evaluation_signals", {}),
                              result.info.get("evaluation_signal_time"), done)
            state = result.info.get("evaluation_state", {})
            quality.update(state, result.info.get("evaluation_signal_time"),
                           result.terminated, result.truncated, result.info.get("evaluation_step_dt"))
            if trace is not None:
                if not state:
                    raise ValueError("physical trajectory recording requires evaluation_state")
                trace.add({
                    "observation_time": current.now, "signal_time": result.info["evaluation_signal_time"],
                    "frame": current.frames[:, -1], "raw_mean_action": raw_mean, "issued_action": action,
                    "estimated_state_t": estimated, "state_target_t": target,
                    "terminated": result.terminated, "truncated": result.truncated,
                    **state, **result.info.get("evaluation_signals", {}),
                }, done)
            history.reset(done)
            observation = result.observation
            current = history.append(observation)
        if contract.device.type == "cuda":
            torch.cuda.synchronize(contract.device)
        report = {
            "policy": "deterministic_mean", "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
            "checkpoint_update": update, "seed": seed, "vector_steps": steps,
            "num_envs": env.num_envs, "transitions": steps * env.num_envs,
            "reward_mean": rewards.report()["mean"], "terminated_count": terminated,
            "truncated_count": truncated, "done_count": done_count,
            "metrics": {name: metric.report() for name, metric in sorted(metrics.items())},
            "physical_metrics_available": bool(metrics),
            "stability": stability.report(),
            "control_quality": quality.report(),
            "environment": environment_config, "environment_provenance": provenance,
            "action_clip": action_clip, "actor": model.actor.describe(),
            "elapsed_s": time.monotonic() - started,
            "scope": "full interval including transients and auto-resets; task metrics are pre-reset",
        }
        if estimation_errors:
            errors = torch.cat(estimation_errors)
            report["state_estimation"] = {
                "indices": list(model.config.state_indices), "samples": len(errors),
                "mae": errors.abs().mean(0).tolist(), "rmse": errors.square().mean(0).sqrt().tolist(),
                "p95_abs": torch.quantile(errors.abs(), 0.95, dim=0).tolist(),
                "units": ["m/s"] * 3 + (["m"] if errors.shape[1] == 4 else []),
                "alignment": "current observation endpoint, before issuing its action",
            }
        if trace is not None:
            report["trajectory"] = trace.save(trace_output)
        json.dumps(report, allow_nan=False)
        return report
    finally:
        env.close()

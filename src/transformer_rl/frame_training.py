"""Packed-frame actor/critic, repeat-first histories and the shared PPO collector."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .adapters import _TensorEnvContract
from .frame_config import FrameModelConfig
from .frame_policy import FramePolicy
from .runner import RolloutCollector
from .types import ActionSample, HistoryBatch, PolicyEvaluation


class FrameGaussianActor(nn.Module):
    def __init__(self, config: FrameModelConfig):
        super().__init__()
        self.config = config
        self.policy = FramePolicy(config.policy)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), math.log(config.initial_std)))

    def forward(self, history: HistoryBatch):
        if not isinstance(history, HistoryBatch):
            raise TypeError("actor requires a packed HistoryBatch")
        if (history.frames.shape[1:] != (self.config.history_length, self.config.frame_dim)
                or history.valid.shape != history.frames.shape[:2] or not history.valid.all()):
            raise ValueError("packed actor requires complete repeat-first fixed windows")
        if not torch.isfinite(history.frames).all():
            raise FloatingPointError("packed frames must be finite")
        return self.policy(history.frames)

    @staticmethod
    def _evaluate(distribution, raw_action):
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), distribution.mean, distribution.stddev)

    def act(self, history, deterministic=False):
        mean = self(history)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean), validate_args=False)
        action = mean if deterministic else distribution.sample()
        return ActionSample(action, self._evaluate(distribution, action))

    def evaluate(self, history, raw_action):
        mean = self(history)
        if raw_action.shape != mean.shape or not torch.isfinite(raw_action).all():
            raise ValueError("raw_action requires finite [B, A]")
        return self._evaluate(Normal(mean, self.log_std.exp().expand_as(mean), validate_args=False), raw_action)

    def describe(self):
        config = self.config.policy
        return {"architecture": config.architecture, "policy_config": config.to_dict(),
                "residual_type": config.residual_type,
                "history": "fixed window recomputed; oldest to newest; repeat first after reset",
                "current_frame_bypass": self.policy.uses_encoder,
                "actor_parameters": sum(p.numel() for p in self.parameters()),
                "mean_parameters": sum(p.numel() for p in self.policy.parameters()),
                "readout_type": config.readout_type if self.policy.is_transformer else None,
                "latent_dim": config.history_latent_dim if self.policy.is_history_mlp
                else config.d_model if self.policy.is_transformer else None}


class FrameActorCritic(nn.Module):
    """The critic is independent and never part of an exported control policy."""

    def __init__(self, config: FrameModelConfig):
        super().__init__()
        self.config = config
        self.actor = FrameGaussianActor(config)
        layers = []
        width = config.critic_dim
        for hidden in config.critic_hidden:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.extend((nn.Linear(width, 1), nn.Flatten(0)))
        self.critic = nn.Sequential(*layers)

    def policy_parameters(self):
        return list(self.parameters())


class FrameHistory:
    """Per-row owned history with the same semantics as deployment.

    Timestamps are collection metadata, not policy features or positional inputs.
    Repeat-first slots are observations, not masked padding. Idempotent reads at
    the same timestamp do not append a second frame.
    """

    def __init__(self, config, num_envs, device):
        self.config = config
        self._contract = _TensorEnvContract(config, num_envs, device)
        self.device = self._contract.device
        self.num_envs = num_envs
        self._frames = torch.zeros(num_envs, config.history_length, config.frame_dim, device=device)
        self._times = torch.zeros(num_envs, config.history_length, dtype=torch.float64, device=device)
        self._ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._command = torch.zeros(num_envs, config.command_dim, device=device)
        self._critic = torch.zeros(num_envs, config.critic_dim, device=device)

    def reset(self, mask):
        if not isinstance(mask, torch.Tensor) or mask.shape != self._ready.shape or mask.dtype != torch.bool:
            raise ValueError("reset requires one bool per environment")
        if mask.device != self.device:
            raise ValueError("reset mask must be on the history device")
        self._ready[mask] = False

    @torch.no_grad()
    def append(self, observation):
        observation = self._contract.observation(observation)
        if observation.frame.dtype != torch.float32:
            raise TypeError("packed training and deployment use float32 features")
        if not torch.equal(observation.command, observation.frame[:, self.config.command_indices]):
            raise ValueError("command must equal the declared packed frame columns")
        times = observation.timestamp.double()
        if (self._ready & (times < self._times[:, -1])).any():
            raise ValueError("history timestamps decreased without reset")
        repeated = self._ready & (times == self._times[:, -1])
        changed = ((observation.frame != self._frames[:, -1]).any(-1)
                   | (observation.critic != self._critic).any(-1))
        if (repeated & changed).any():
            raise ValueError("different packed observation at the same timestamp")
        first = ~self._ready
        self._frames[first] = observation.frame[first, None]
        self._times[first] = times[first, None]
        advance = self._ready & ~repeated
        self._frames[advance, :-1] = self._frames[advance, 1:]
        self._times[advance, :-1] = self._times[advance, 1:]
        self._frames[advance, -1] = observation.frame[advance]
        self._times[advance, -1] = times[advance]
        self._command.copy_(observation.command)
        self._critic.copy_(observation.critic)
        self._ready.fill_(True)
        return self.snapshot()

    def snapshot(self):
        if not self._ready.all():
            raise RuntimeError("insert each reset environment's first observation before snapshot")
        return HistoryBatch(self._frames.clone(), self._times.clone(),
                            torch.ones_like(self._times, dtype=torch.bool), self._command.clone(),
                            self._times[:, -1].clone())


class FrameCollector(RolloutCollector):
    """Reuse tested ownership, terminal bootstrapping, GAE and behavior snapshots."""

    def __init__(self, env, model, ppo_config, action_bounds):
        self.bounds = torch.tensor(action_bounds, device=env.device, dtype=torch.float32)
        if (self.bounds.shape != (model.config.action_dim,) or not torch.isfinite(self.bounds).all()
                or not (self.bounds > 0).all()):
            raise ValueError("action bounds require one positive finite value per action")
        super().__init__(env, model, ppo_config)
        self._reward_component_drain_failed = False
        self.reward_components_enabled = (hasattr(env, "enable_reward_components")
                                         and callable(getattr(env, "drain_reward_components", None)))
        if self.reward_components_enabled:
            env.enable_reward_components = True

    def _new_history(self):
        return FrameHistory(self.model.config, self.num_envs, self.device)

    def _issued_action(self, raw_action):
        return raw_action.clamp(-self.bounds, self.bounds)

    def collect(self, steps, should_stop=None):
        # Reject invalid calls before draining or replacing a prior window.
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if should_stop is not None and not callable(should_stop):
            raise TypeError("should_stop must be callable")
        if self._reward_component_drain_failed:
            raise RuntimeError("reward component drain failed; use a fresh environment and collector")
        collection_error = None
        try:
            batch = super().collect(steps, should_stop)
            if batch is not None:
                import time
                from .action_statistics import action_statistics
                started = time.perf_counter()
                self.last_metrics.update(action_statistics(
                    batch.raw_action, batch.issued_action, batch.old_mean, self.bounds))
                self.last_metrics["action_statistics_elapsed_s"] = time.perf_counter() - started
                self.last_metrics["elapsed_s"] += self.last_metrics["action_statistics_elapsed_s"]
            return batch
        except BaseException as error:
            collection_error = error
            raise
        finally:
            if self.reward_components_enabled:
                import time
                started = time.perf_counter()
                drained = False
                try:
                    report = self.env.drain_reward_components()
                    drained = True
                    self.last_metrics["reward_components"] = report
                    issue = report.get("observation_error")
                    if not issue and (report["samples"] != self.last_metrics["transitions"]
                                      or report["steps"] != self.last_metrics["vector_steps"]):
                        issue = "reward component window differs from returned collection samples"
                    if issue:
                        self._observation = None
                        raise ValueError(f"invalid reward component telemetry: {issue}")
                except BaseException as error:
                    self._observation = None
                    if not drained:
                        # A failed drain may retain an earlier window. Reset
                        # must not permit those samples to enter another batch.
                        self._reward_component_drain_failed = True
                    if collection_error is None:
                        raise
                    collection_error.add_note(f"reward component drain failed: {type(error).__name__}: {error}")
                finally:
                    self.last_metrics["reward_component_drain_elapsed_s"] = time.perf_counter() - started
                    self.last_metrics["elapsed_s"] = (self.last_metrics.get("elapsed_s", 0.)
                                                       + self.last_metrics["reward_component_drain_elapsed_s"])


def reward_component_scalars(collection):
    """Separate unit-labelled rollout tags from the legacy last-step diagnostics."""
    report = collection.get("reward_components")
    if report is None or report.get("observation_error"):
        return {}
    result = {}
    total = report["actual_total_reward"]
    if total["available"]:
        result["reward_components/total_step_mean"] = total["mean"]
    for name, term in (report["continuous_components"]["terms"] or {}).items():
        result[f"reward_components/density_per_s/{name}"] = term["density"]["mean"]
        result[f"reward_components/step_contribution/{name}"] = term["step_reward"]["mean"]
    for name, term in (report["event_components"]["terms"] or {}).items():
        result[f"reward_components/event_step_contribution/{name}"] = term["mean"]
    residual = report["residual_step_reward"]
    if residual["available"]:
        result["reward_components/unattributed_step_mean"] = residual["statistics"]["mean"]
    return result

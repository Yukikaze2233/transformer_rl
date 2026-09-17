"""Synchronous rollout collection with persistent, per-environment histories."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
import math
from numbers import Real
import time

import torch

from .adapters import _TensorEnvContract
from .config import PPOConfig
from .history import HistoryBuffer
from .model import ActorCritic
from .storage import EstimatorBatch, PPOBatch, RolloutBuffer
from .types import VectorEnv, VectorObservation


class RolloutCollector:
    """Collect one policy version at a time; call ``reset`` explicitly to start.

    Histories and the next observation persist across collect/update boundaries.
    ``total_steps`` counts successful env.step returns over this collector's
    lifetime, including across explicit resets. ``last_metrics`` describes the
    most recent collect call; elapsed wall time includes callbacks and GAE.
    A failed environment transition invalidates the current state until reset.
    The caller must serialize model updates with collection.
    """

    def __init__(
        self, env: VectorEnv, model: ActorCritic, ppo_config: PPOConfig,
        action_clip: float | None = None,
    ):
        if action_clip is not None and (
            isinstance(action_clip, bool) or not isinstance(action_clip, Real)
            or not math.isfinite(action_clip) or action_clip <= 0
        ):
            raise ValueError("action_clip must be None or a finite positive number")
        self.env = env
        self.model = model
        self.ppo_config = ppo_config
        self.action_clip = action_clip
        self._contract = _TensorEnvContract(model.config, env.num_envs, env.device)
        self.num_envs = self._contract.num_envs
        self.device = self._contract.device
        self._history = HistoryBuffer(model.config, self.num_envs, self.device)
        self._observation: VectorObservation | None = None
        self.total_steps = 0
        self.last_metrics: dict[str, float | int | bool] = {}

    @property
    def total_transitions(self) -> int:
        return self.total_steps * self.num_envs

    @torch.no_grad()
    def reset(self, seed: int | None = None) -> VectorObservation:
        self._observation = None
        observation = self._contract.observation(self.env.reset(seed=seed))
        history = HistoryBuffer(self.model.config, self.num_envs, self.device)
        history.append(observation)
        self._history = history
        self._observation = observation
        # The return value must not expose the collector's live current state.
        return VectorObservation(**{
            field.name: getattr(observation, field.name).clone() for field in fields(VectorObservation)
        })

    def _model_version(self) -> tuple:
        tensors = (*self.model.named_parameters(), *self.model.named_buffers())
        return tuple((name, id(tensor), tensor._version) for name, tensor in tensors)

    def _check_model_version(self, expected: tuple) -> None:
        if self._model_version() != expected:
            raise RuntimeError("model changed during collect; discard this rollout and serialize policy updates")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("model must remain in eval mode during collect")

    def _value(self, critic: torch.Tensor) -> torch.Tensor:
        return self._contract.tensor("critic value", self.model.critic(critic), (critic.shape[0],))

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def collect(
        self, steps: int, should_stop: Callable[[], bool] | None = None,
    ) -> PPOBatch | None:
        """Collect up to ``steps`` vector steps; a zero/initially stopped call returns None.

        The callback is checked before each step, including before requiring an
        explicit reset. A nonempty stopped rollout is finished with normal tail
        bootstrapping; stopping does not create an artificial episode boundary.
        """
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if should_stop is not None and not callable(should_stop):
            raise TypeError("should_stop must be callable")
        self.model.eval()
        version = self._model_version()
        self._synchronize()
        started = time.perf_counter()
        vector_steps = terminated_count = truncated_count = done_count = 0
        reward_sum = 0.0
        early_stopped = False
        buffer = RolloutBuffer(steps) if steps else None
        auxiliary = self.model.config.estimator_type != "none"
        next_proprio, next_valid = [], []
        try:
            for _ in range(steps):
                stop = should_stop is not None and should_stop()
                self._check_model_version(version)
                if stop:
                    early_stopped = True
                    break
                if self._observation is None:
                    raise RuntimeError("call reset(seed=...) explicitly before collecting")

                history = self._history.snapshot()
                critic = self._observation.critic.detach().clone()
                sample = self.model.actor.act(history)
                raw_action = self._contract.tensor(
                    "raw_action", sample.action, (self.num_envs, self.model.config.action_dim),
                )
                issued_action = (raw_action.clone() if self.action_clip is None
                                 else raw_action.clamp(-self.action_clip, self.action_clip))
                old_log_prob = self._contract.tensor("old_log_prob", sample.evaluation.log_prob, (self.num_envs,))
                old_mean = self._contract.tensor("old_mean", sample.evaluation.mean, tuple(raw_action.shape))
                old_std = self._contract.tensor("old_std", sample.evaluation.std, tuple(raw_action.shape))
                if not (old_std > 0).all():
                    raise ValueError("policy std must be strictly positive")
                old_value = self._value(critic)
                self._check_model_version(version)

                # Every t-side tensor is owned before step can mutate input or auto-reset buffers.
                self._observation = None
                result = self.env.step(issued_action.clone())
                self.total_steps += 1
                vector_steps += 1
                result = self._contract.step(result)
                reward_sum += result.reward.double().sum().item()
                terminated_count += result.terminated.sum().item()
                truncated_count += result.truncated.sum().item()
                done = result.terminated | result.truncated
                done_count += done.sum().item()
                if auxiliary:
                    valid = (~done).clone()
                    target = result.observation.frame[:, :self.model.config.proprio_dim]
                    next_proprio.append(torch.where(valid[:, None], target, torch.zeros_like(target)).clone())
                    next_valid.append(valid)
                self._check_model_version(version)

                # Never evaluate undefined final rows, even when multiplying by zero later.
                next_value = torch.zeros_like(old_value)
                ongoing = ~done
                timeout = result.truncated & ~result.terminated
                if ongoing.any():
                    next_value[ongoing] = self._value(result.observation.critic[ongoing])
                if timeout.any():
                    next_value[timeout] = self._value(result.final_critic[timeout])
                buffer.add(
                    history=history, critic=critic, raw_action=raw_action, issued_action=issued_action,
                    old_log_prob=old_log_prob, old_mean=old_mean, old_std=old_std, old_value=old_value,
                    reward=result.reward, next_value=next_value,
                    terminated=result.terminated, truncated=result.truncated,
                )
                self._history.reset(done)
                self._history.append(result.observation)
                self._observation = result.observation
            self._check_model_version(version)
            if not buffer or not len(buffer):
                return None
            batch = buffer.finish(self.ppo_config.gamma, self.ppo_config.gae_lambda)
            return EstimatorBatch(**vars(batch), next_proprio=torch.cat(next_proprio),
                                  next_valid=torch.cat(next_valid)) if auxiliary else batch
        finally:
            self._synchronize()
            transitions = vector_steps * self.num_envs
            self.last_metrics = {
                "reward_mean": reward_sum / transitions if transitions else 0.0,
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "done_count": done_count,
                "vector_steps": vector_steps,
                "transitions": transitions,
                "elapsed_s": time.perf_counter() - started,
                "early_stopped": early_stopped,
                "total_steps": self.total_steps,
                "total_transitions": self.total_transitions,
            }

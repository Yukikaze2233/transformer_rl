"""Strict tensor-only environment boundaries; no simulator imports or startup."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

import torch

from .config import ModelConfig
from .types import StepResult, VectorObservation


class _TensorEnvContract:
    """Shared validation and ownership boundary for adapters and the collector."""

    def __init__(self, config: ModelConfig, num_envs: int, device: str | torch.device):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.config = config
        self.num_envs = num_envs
        # Resolve device aliases, e.g. cpu:0 and an unindexed cuda device.
        self.device = torch.empty(0, device=device).device

    def tensor(
        self, name: str, value: torch.Tensor, shape: tuple[int, ...],
        *, boolean: bool = False, finite: bool = True,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on environment device {self.device}")
        if boolean:
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        elif not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
        if finite and not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} must be finite")
        return value.detach().clone()

    def observation(self, observation: VectorObservation) -> VectorObservation:
        if not isinstance(observation, VectorObservation):
            raise TypeError("observation must be a VectorObservation")
        shapes = dict(frame=(self.num_envs, self.config.frame_dim),
                      timestamp=(self.num_envs,), command=(self.num_envs, self.config.command_dim),
                      critic=(self.num_envs, self.config.critic_dim))
        snapshot = VectorObservation(**{
            field.name: self.tensor(f"observation.{field.name}", getattr(observation, field.name),
                                    shapes[field.name])
            for field in fields(VectorObservation)
        })
        if snapshot.timestamp.dtype not in (torch.float32, torch.float64):
            raise TypeError("observation.timestamp must be float32 or float64")
        if snapshot.command.dtype != snapshot.frame.dtype or snapshot.critic.dtype != snapshot.frame.dtype:
            raise TypeError("observation frame, command and critic must share a floating dtype")
        return snapshot

    def final_state(
        self, final_critic: torch.Tensor | None, final_critic_valid: torch.Tensor | None,
        terminated: torch.Tensor, truncated: torch.Tensor, *, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = truncated & ~terminated
        if final_critic is None or final_critic_valid is None:
            if timeout.any():
                raise ValueError("timeouts require both final_critic and final_critic_valid")
            if final_critic is not None or final_critic_valid is not None:
                raise ValueError("final_critic and final_critic_valid must be supplied together")
            return (
                torch.zeros(self.num_envs, self.config.critic_dim, dtype=dtype, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
        final = self.tensor("final_critic", final_critic, (self.num_envs, self.config.critic_dim), finite=False)
        valid = self.tensor("final_critic_valid", final_critic_valid, (self.num_envs,), boolean=True)
        if final.dtype != dtype:
            raise TypeError("final_critic must have the observation critic dtype")
        if (timeout & ~valid).any():
            raise ValueError("final_critic_valid must be true for every truncated, nonterminated row")
        # Undefined final rows are allowed to contain NaN, including true terminals.
        if not torch.isfinite(final[timeout]).all():
            raise FloatingPointError("timeout final_critic must be finite")
        return final, valid

    def step(self, result: StepResult) -> StepResult:
        if not isinstance(result, StepResult):
            raise TypeError("env.step must return a StepResult")
        if not isinstance(result.info, dict):
            raise TypeError("step info must be a dict")
        observation = self.observation(result.observation)
        reward = self.tensor("reward", result.reward, (self.num_envs,))
        terminated = self.tensor("terminated", result.terminated, (self.num_envs,), boolean=True)
        truncated = self.tensor("truncated", result.truncated, (self.num_envs,), boolean=True)
        final, valid = self.final_state(result.final_critic, result.final_critic_valid,
                                        terminated, truncated, dtype=observation.critic.dtype)
        return StepResult(observation, reward, terminated, truncated, final, valid, dict(result.info))


class TensorEnvAdapter:
    """Adapt an already-created, auto-resetting Gymnasium-style tensor environment.

    ``reset`` must return ``(raw_observation, info)`` and ``step`` must return
    ``(raw_observation, reward, terminated, truncated, info)``. The caller owns
    simulator startup and supplies all observation semantics through the encoder.
    Timeouts require explicit pre-reset ``info['final_critic']`` [N, S] and
    ``info['final_critic_valid']`` bool [N]; reset observations are never a fallback.
    Contract tensors are owned snapshots. Other info values are passed through.
    """

    def __init__(
        self, env: Any, model_config: ModelConfig,
        encode_observation: Callable[[Any, dict[str, Any]], VectorObservation],
    ):
        if not callable(encode_observation):
            raise TypeError("encode_observation must be callable")
        self.env = env
        self.model_config = model_config
        self.encode_observation = encode_observation
        self._contract = _TensorEnvContract(model_config, env.num_envs, env.device)
        self.num_envs = self._contract.num_envs
        self.device = self._contract.device

    @torch.no_grad()
    def reset(self, seed: int | None = None) -> VectorObservation:
        result = self.env.reset(seed=seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("raw env.reset must return (observation, info)")
        raw_observation, info = result
        if not isinstance(info, dict):
            raise TypeError("reset info must be a dict")
        return self._contract.observation(self.encode_observation(raw_observation, info))

    @torch.no_grad()
    def step(self, issued_action: torch.Tensor) -> StepResult:
        action = self._contract.tensor("issued_action", issued_action,
                                       (self.num_envs, self.model_config.action_dim))
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("raw env.step must return (observation, reward, terminated, truncated, info)")
        raw_observation, reward, terminated, truncated, info = result
        if not isinstance(info, dict):
            raise TypeError("step info must be a dict")
        # Encoders can reuse scratch tensors: own transition metadata before calling one.
        reward = self._contract.tensor("reward", reward, (self.num_envs,))
        terminated = self._contract.tensor("terminated", terminated, (self.num_envs,), boolean=True)
        truncated = self._contract.tensor("truncated", truncated, (self.num_envs,), boolean=True)
        final = info.get("final_critic")
        valid = info.get("final_critic_valid")
        has_final = final is not None
        dtype = final.dtype if isinstance(final, torch.Tensor) else torch.float32
        final, valid = self._contract.final_state(final, valid, terminated, truncated, dtype=dtype)
        info = dict(info)
        observation = self._contract.observation(self.encode_observation(raw_observation, info))
        if not has_final:
            final = final.to(dtype=observation.critic.dtype)
        elif final.dtype != observation.critic.dtype:
            raise TypeError("final_critic must have the observation critic dtype")
        return StepResult(observation, reward, terminated, truncated, final, valid, info)

    def close(self) -> None:
        self.env.close()

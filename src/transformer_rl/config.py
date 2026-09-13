"""Explicit model and optimization configuration, independent of a simulator."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    proprio_dim: int = 16
    command_dim: int = 3
    action_dim: int = 6
    sensor_groups: int = 2
    critic_dim: int = 29
    history_length: int = 16
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 128
    critic_hidden: tuple[int, ...] = (256, 128, 64)
    time_scale_s: float = 0.1
    initial_std: float = 0.2

    @property
    def frame_dim(self) -> int:
        # Proprioception, historical command, previous issued action,
        # sensor ages, age-known flags, and actual policy interval.
        return self.proprio_dim + self.command_dim + self.action_dim + 2 * self.sensor_groups + 1

    def __post_init__(self) -> None:
        if type(self.critic_hidden) is not tuple:
            raise ValueError("critic_hidden must be a tuple of positive integers")
        integers = (self.proprio_dim, self.command_dim, self.action_dim,
                    self.sensor_groups, self.critic_dim, self.history_length,
                    self.d_model, self.num_heads, self.num_layers, self.ffn_dim,
                    *self.critic_hidden)
        if not self.critic_hidden or any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("model dimensions must be positive integers")
        if self.d_model % self.num_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by num_heads")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
               for v in (self.time_scale_s, self.initial_std)):
            raise ValueError("time_scale_s and initial_std must be finite and positive")


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    epochs: int = 5
    num_minibatches: int = 4
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        values = (self.learning_rate, self.gamma, self.gae_lambda, self.clip_ratio,
                  self.value_clip, self.value_coef, self.entropy_coef,
                  self.max_grad_norm, self.target_kl)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("PPO numeric parameters must be finite real numbers")
        if type(self.normalize_advantage) is not bool:
            raise ValueError("normalize_advantage must be boolean")
        positive = (self.learning_rate, self.clip_ratio, self.value_clip,
                    self.max_grad_norm, self.target_kl)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("PPO rates, clipping bounds and target KL must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        if any(not math.isfinite(v) or v < 0 for v in (self.value_coef, self.entropy_coef)):
            raise ValueError("loss coefficients must be finite and nonnegative")
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.num_minibatches)):
            raise ValueError("epochs and num_minibatches must be positive integers")


def load_config(path: str | Path) -> tuple[ModelConfig, PPOConfig, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"model", "ppo", "environment"}:
        raise ValueError("configuration requires only model, ppo and environment sections")
    configs = []
    for key, cls in (("model", ModelConfig), ("ppo", PPOConfig)):
        section = data.get(key, {})
        if not isinstance(section, dict) or set(section) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown {key} configuration fields")
        if key == "model" and "critic_hidden" in section:
            section = {**section, "critic_hidden": tuple(section["critic_hidden"])}
        configs.append(cls(**section))
    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment configuration must be an object")
    return configs[0], configs[1], environment


def config_dict(model: ModelConfig, ppo: PPOConfig, environment: dict) -> dict:
    return {"model": asdict(model), "ppo": asdict(ppo), "environment": environment}

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
    actor_type: str = "transformer"
    time_encoding: str = "elapsed"
    residual_type: str = "add"
    auxiliary_indices: tuple[int, ...] = ()
    baseline_hidden: tuple[int, ...] = (128, 64)
    gru_hidden: int = 64
    mean_init_scale: float = 1.0
    readout_type: str = "query"
    estimator_type: str = "none"
    state_indices: tuple[int, ...] = ()
    controller_hidden: tuple[int, ...] = (128, 64, 32)
    state_velocity_scale: float = 1.0
    state_height_center: float = 0.30
    state_height_scale: float = 0.10
    context_dim: int = 16
    context_target_hidden: tuple[int, ...] = (128, 64)
    context_prototypes: int = 32
    context_temperature: float = 3.0
    context_sinkhorn_epsilon: float = 0.05
    context_sinkhorn_iterations: int = 3

    @property
    def requires_current_frame(self) -> bool:
        return self.estimator_type != "none" or self.readout_type == "last"

    @property
    def frame_dim(self) -> int:
        # Proprioception, historical command, previous issued action,
        # sensor ages, age-known flags, and actual policy interval.
        return self.proprio_dim + self.command_dim + self.action_dim + 2 * self.sensor_groups + 1

    def __post_init__(self) -> None:
        if self.estimator_type not in ("none", "velocity", "context"):
            raise ValueError("estimator_type must be none, velocity or context")
        expected_states = {"none": 0, "velocity": 3, "context": 4}[self.estimator_type]
        if (type(self.state_indices) is not tuple or len(self.state_indices) != expected_states
            or any(type(i) is not int or not 0 <= i < self.critic_dim for i in self.state_indices)
            or len(set(self.state_indices)) != len(self.state_indices)):
            raise ValueError("state_indices must explicitly identify the estimator's critic targets")
        for hidden in (self.controller_hidden, self.context_target_hidden):
            if type(hidden) is not tuple or not hidden or any(type(n) is not int or n < 1 for n in hidden):
                raise ValueError("estimator hidden dimensions must be positive integers")
        if any(type(n) is not int or n < 1 for n in
               (self.context_dim, self.context_prototypes, self.context_sinkhorn_iterations)):
            raise ValueError("context dimensions and iterations must be positive integers")
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in
               (self.state_velocity_scale, self.state_height_scale,
                self.context_temperature, self.context_sinkhorn_epsilon)):
            raise ValueError("estimator scales and temperatures must be finite and positive")
        if type(self.state_height_center) not in (float, int) or not math.isfinite(self.state_height_center):
            raise ValueError("state_height_center must be finite")
        if self.estimator_type != "none":
            if self.actor_type not in ("mlp", "transformer") or self.auxiliary_indices:
                raise ValueError("detached estimators require MLP/Transformer without a shared auxiliary head")
            if self.actor_type == "transformer" and (
                self.readout_type != "last" or self.time_encoding != "index" or self.residual_type != "add"
            ):
                raise ValueError("estimator Transformer requires index/last/add")
        if self.actor_type not in ("transformer", "mlp", "gru"):
            raise ValueError("actor_type must be transformer, mlp or gru")
        if self.time_encoding not in ("elapsed", "index"):
            raise ValueError("time_encoding must be elapsed or index")
        if self.residual_type not in ("add", "gated"):
            raise ValueError("residual_type must be add or gated")
        if self.readout_type not in ("query", "last"):
            raise ValueError("readout_type must be query or last")
        if self.actor_type != "transformer" and self.readout_type != "query":
            raise ValueError("readout_type='last' requires transformer")
        if self.actor_type != "transformer" and (
            self.time_encoding != "elapsed" or self.residual_type != "add"
            or self.auxiliary_indices
        ):
            raise ValueError("position/residual variants and auxiliary heads require transformer")
        if (type(self.auxiliary_indices) is not tuple
            or any(type(i) is not int or not 0 <= i < self.critic_dim
                   for i in self.auxiliary_indices)
            or len(set(self.auxiliary_indices)) != len(self.auxiliary_indices)):
            raise ValueError("auxiliary_indices must be unique valid critic column indices")
        if (type(self.baseline_hidden) is not tuple or not self.baseline_hidden
            or any(type(v) is not int or v < 1 for v in self.baseline_hidden)
            or type(self.gru_hidden) is not int or self.gru_hidden < 1):
            raise ValueError("baseline dimensions must be positive integers")
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
        if (type(self.mean_init_scale) not in (float, int)
            or not math.isfinite(self.mean_init_scale) or self.mean_init_scale < 0):
            raise ValueError("mean_init_scale must be finite and nonnegative")


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
    auxiliary_coef: float = 0.0
    estimator_learning_rate: float = 0.001
    estimator_epochs: int = 2
    estimator_minibatches: int = 4
    estimator_max_grad_norm: float = 1.0
    estimator_target_kl: float = 0.0025
    estimator_context_coef: float = 1.0

    def __post_init__(self) -> None:
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in
               (self.estimator_learning_rate, self.estimator_max_grad_norm, self.estimator_target_kl)):
            raise ValueError("estimator optimizer rates and limits must be finite and positive")
        if (type(self.estimator_context_coef) not in (int, float)
            or not math.isfinite(self.estimator_context_coef) or self.estimator_context_coef < 0):
            raise ValueError("estimator_context_coef must be finite and nonnegative")
        if any(type(v) is not int or v < 1 for v in (self.estimator_epochs, self.estimator_minibatches)):
            raise ValueError("estimator epochs and minibatches must be positive integers")
        values = (self.learning_rate, self.gamma, self.gae_lambda, self.clip_ratio,
                  self.value_clip, self.value_coef, self.entropy_coef,
                  self.max_grad_norm, self.target_kl, self.auxiliary_coef)
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
        if any(not math.isfinite(v) or v < 0 for v in (
            self.value_coef, self.entropy_coef, self.auxiliary_coef
        )):
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
        if key == "model":
            section = dict(section)
            for name in ("critic_hidden", "baseline_hidden", "auxiliary_indices", "state_indices",
                         "controller_hidden", "context_target_hidden"):
                if name in section:
                    section[name] = tuple(section[name])
        configs.append(cls(**section))
    environment = data.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment configuration must be an object")
    if configs[1].auxiliary_coef > 0 and not configs[0].auxiliary_indices:
        raise ValueError("positive auxiliary_coef requires explicit auxiliary_indices")
    return configs[0], configs[1], environment


def config_dict(model: ModelConfig, ppo: PPOConfig, environment: dict) -> dict:
    return {"model": asdict(model), "ppo": asdict(ppo), "environment": environment}

"""Private structured policy candidate; not an admitted training endpoint.

All inference depends on one complete, oldest-to-newest observation window.
Privileged labels, sampling, control ownership and reset handling stay outside
the network. The existing research policy/config/checkpoint formats are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from ..common.types import ActionSample, HistoryBatch, PolicyEvaluation


@dataclass(frozen=True)
class StructuredPolicyConfig:
    history_length: int = 5
    body_width: int = 32
    body_heads: int = 2
    body_ffn: int = 64
    temporal_width: int = 64
    temporal_heads: int = 4
    temporal_layers: int = 2
    temporal_ffn: int = 128
    state_hidden_dim: int = 32
    leg_hidden_dims: tuple[int, int] = (64, 32)
    wheel_hidden_dims: tuple[int, int] = (64, 32)
    prediction_hidden_dim: int = 32
    critic_input_dim: int = 191
    critic_hidden_dims: tuple[int, int, int] = (256, 128, 64)
    gate_bias: float = -2.0
    mean_init_scale: float = 0.01
    estimate_log_std_min: float = -4.0
    estimate_log_std_max: float = 2.0

    def __post_init__(self):
        for name in ("history_length", "body_width", "body_heads", "body_ffn",
                     "temporal_width", "temporal_heads", "temporal_layers", "temporal_ffn",
                     "state_hidden_dim", "prediction_hidden_dim", "critic_input_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, depth in (("leg_hidden_dims", 2), ("wheel_hidden_dims", 2),
                            ("critic_hidden_dims", 3)):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != depth
                    or any(type(width) is not int or width < 1 for width in value)):
                raise ValueError(f"{name} requires {depth} positive integer widths")
            object.__setattr__(self, name, tuple(value))
        if (self.body_width % self.body_heads or self.temporal_width % self.temporal_heads
                or self.temporal_width % 2):
            raise ValueError("attention widths must divide their head counts; time width must be even")
        for name in ("gate_bias", "mean_init_scale", "estimate_log_std_min", "estimate_log_std_max"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.mean_init_scale < 0 or self.estimate_log_std_min >= self.estimate_log_std_max:
            raise ValueError("invalid mean initialization or estimation scale bounds")

    @property
    def normalization_contract(self):
        return f"fixed_manual35_{self.critic_input_dim}_no_running_statistics"


class _GatedAttentionBlock(nn.Module):
    """Pre-LN attention with scalar-per-feature gates across depth only."""

    __constants__ = ["heads", "head_width"]

    def __init__(self, width, heads, ffn_width, gate_bias):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.attention_gate = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))
        self.ffn_gate = nn.Linear(width, width)
        # Zero gate weights give the whole initial residual branch the same gain.
        for gate in (self.attention_gate, self.ffn_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias)

    def forward(self, tokens: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        normalized = self.attention_norm(tokens)
        qkv = self.qkv(normalized).reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(2)
        # SDPA boolean True means allowed. Dropout remains zero in both modes.
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=allowed, dropout_p=0.0,
        ).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention_gate(normalized).sigmoid() * self.attention_output(attended)
        normalized = self.ffn_norm(tokens)
        return tokens + self.ffn_gate(normalized).sigmoid() * self.ffn(normalized)


class _CurrentReadout(nn.Module):
    """One shared memory projection for four actuator queries and one state query."""

    __constants__ = ["heads", "head_width", "width"]

    def __init__(self, config):
        super().__init__()
        self.heads = config.temporal_heads
        self.width = config.temporal_width
        self.head_width = self.width // self.heads
        self.query = nn.Linear(config.body_width + 11, self.width)
        self.query_identity = nn.Parameter(torch.empty(5, self.width))
        nn.init.normal_(self.query_identity, std=0.02)
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, self.width)
        self.norm = nn.LayerNorm(self.width)

    def forward(self, local: torch.Tensor, command: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length = memory.shape[:2]
        query = self.query(torch.cat((local, command[:, None].expand(-1, 5, -1)), -1))
        query = query + self.query_identity
        q = query.reshape(batch, 5, self.heads, self.head_width).transpose(1, 2)
        k = self.key(memory).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = self.value(memory).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        result = result.transpose(1, 2).reshape(batch, 5, self.width)
        return self.norm(query + self.output(result))


class StructuredPolicy(nn.Module):
    """Raw action means from a fixed window; no cache or cross-call hidden state.

    ``validate_frames`` is a boundary check, not a per-minibatch GPU sync.
    The tensor-only forward deliberately excludes both wheel-position slots even
    if an upstream caller violates their required zero-valued wire semantics.
    """

    __constants__ = ["history_length", "body_width", "time_width", "log_std_min", "log_std_max"]

    def __init__(self, config=StructuredPolicyConfig(), *, prediction_head=True):
        super().__init__()
        if not isinstance(config, StructuredPolicyConfig):
            raise TypeError("config must be a StructuredPolicyConfig")
        self.history_length = config.history_length
        self.body_width = config.body_width
        self.time_width = config.temporal_width
        self.log_std_min = float(config.estimate_log_std_min)
        self.log_std_max = float(config.estimate_log_std_max)
        for name, indices in (
            ("command_indices", (0, 1, 2, 3, 28, 29, 30, 31, 32, 33, 34)),
            ("base_indices", (4, 5, 6, 7, 8, 9)),
            ("left_leg_indices", (10, 11, 16, 17, 22, 23)),
            ("right_leg_indices", (12, 13, 18, 19, 24, 25)),
            ("left_wheel_indices", (20, 26)), ("right_wheel_indices", (21, 27)),
            ("query_body_indices", (2, 3, 4, 5, 1)),
        ):
            self.register_buffer(name, torch.tensor(indices, dtype=torch.long))
        self.command_projection = nn.Linear(11, self.body_width)
        self.base_projection = nn.Linear(6, self.body_width)
        self.leg_projection = nn.Linear(6, self.body_width)
        self.wheel_projection = nn.Linear(2, self.body_width)
        self.body_identity = nn.Parameter(torch.empty(6, self.body_width))
        nn.init.normal_(self.body_identity, std=0.02)
        graph = torch.eye(6, dtype=torch.bool)
        graph[:2, :] = True
        graph[:, :2] = True
        for left, right in ((2, 4), (3, 5), (2, 3), (4, 5)):
            graph[left, right] = graph[right, left] = True
        self.register_buffer("body_allowed", graph)
        self.spatial_block = _GatedAttentionBlock(
            self.body_width, config.body_heads, config.body_ffn, config.gate_bias)
        self.body_norm = nn.LayerNorm(self.body_width)
        self.body_fusion = nn.Linear(6 * self.body_width, self.time_width)
        self.fusion_norm = nn.LayerNorm(self.time_width)
        position = torch.arange(1 - self.history_length, 1, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, self.time_width, 2) * (-math.log(10000.0) / self.time_width))
        encoding = torch.zeros(self.history_length, self.time_width)
        encoding[:, 0::2] = (position * frequencies).sin()
        encoding[:, 1::2] = (position * frequencies).cos()
        self.register_buffer("time_encoding", encoding)
        self.register_buffer("time_allowed", torch.ones(self.history_length, self.history_length,
                                                       dtype=torch.bool).tril())
        self.temporal_blocks = nn.ModuleList([
            _GatedAttentionBlock(self.time_width, config.temporal_heads, config.temporal_ffn, config.gate_bias)
            for _ in range(config.temporal_layers)
        ])
        self.time_norm = nn.LayerNorm(self.time_width)
        self.readout = _CurrentReadout(config)
        leg_first, leg_second = config.leg_hidden_dims
        wheel_first, wheel_second = config.wheel_hidden_dims
        self.state_head = nn.Sequential(nn.Linear(self.time_width, config.state_hidden_dim), nn.GELU(),
                                         nn.Linear(config.state_hidden_dim, 8))
        self.leg_decoder = nn.Sequential(nn.Linear(self.time_width + 31, leg_first), nn.GELU(),
                                         nn.Linear(leg_first, leg_second), nn.GELU(), nn.Linear(leg_second, 2))
        self.wheel_decoder = nn.Sequential(nn.Linear(self.time_width + 27, wheel_first), nn.GELU(),
                                           nn.Linear(wheel_first, wheel_second), nn.GELU(), nn.Linear(wheel_second, 1))
        self.prediction_head = (nn.Sequential(nn.Linear(self.time_width + 6, config.prediction_hidden_dim),
                                               nn.GELU(), nn.Linear(config.prediction_hidden_dim, 6))
                                if prediction_head else None)
        with torch.no_grad():
            for head in (self.leg_decoder[-1], self.wheel_decoder[-1]):
                head.weight.mul_(config.mean_init_scale)
                head.bias.mul_(config.mean_init_scale)

    @torch.jit.unused
    def validate_frames(self, frames):
        reference = self.command_projection.weight
        if (not isinstance(frames, torch.Tensor) or frames.ndim != 3 or frames.shape[0] < 1
                or frames.shape[1:] != (self.history_length, 35)):
            raise ValueError("frames require [B, history_length, 35] with B >= 1")
        if frames.dtype != reference.dtype or frames.device != reference.device:
            raise ValueError("frames must use the policy device and floating dtype")
        if not torch.isfinite(frames).all():
            raise ValueError("frames contain nonfinite values")
        if torch.count_nonzero(frames[..., 14:16]):
            raise ValueError("wheel-position compatibility slots must be zero")

    def _encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = frames.shape[:2]
        body = torch.stack((
            self.command_projection(frames.index_select(-1, self.command_indices)),
            self.base_projection(frames.index_select(-1, self.base_indices)),
            self.leg_projection(frames.index_select(-1, self.left_leg_indices)),
            self.leg_projection(frames.index_select(-1, self.right_leg_indices)),
            self.wheel_projection(frames.index_select(-1, self.left_wheel_indices)),
            self.wheel_projection(frames.index_select(-1, self.right_wheel_indices)),
        ), dim=2) + self.body_identity
        body = self.spatial_block(body.reshape(batch * length, 6, self.body_width), self.body_allowed)
        body = self.body_norm(body).reshape(batch, length, 6, self.body_width)
        tokens = self.fusion_norm(self.body_fusion(body.flatten(2))) + self.time_encoding
        for block in self.temporal_blocks:
            tokens = block(tokens, self.time_allowed)
        return self.time_norm(tokens), body[:, -1]

    @torch.jit.export
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode(frames)[0]

    def state_features(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.cat((raw[:, :3], raw[:, 3:6].clamp(self.log_std_min, self.log_std_max),
                          raw[:, 6:8].sigmoid()), -1)

    def outputs(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, current_body = self._encode(frames)
        current = frames[:, -1]
        command = current.index_select(-1, self.command_indices)
        base = current.index_select(-1, self.base_indices)
        queries = self.readout(current_body.index_select(1, self.query_body_indices), command, memory)
        raw_state = self.state_head(queries[:, 4])
        shared = torch.cat((base, command, self.state_features(raw_state)), -1)
        legs = torch.stack((current.index_select(-1, self.left_leg_indices),
                            current.index_select(-1, self.right_leg_indices)), 1)
        wheels = torch.stack((current.index_select(-1, self.left_wheel_indices),
                              current.index_select(-1, self.right_wheel_indices)), 1)
        legs = self.leg_decoder(torch.cat((queries[:, :2], legs, shared[:, None].expand(-1, 2, -1)), -1))
        wheels = self.wheel_decoder(torch.cat((queries[:, 2:4], wheels, shared[:, None].expand(-1, 2, -1)), -1))
        return torch.cat((legs.flatten(1), wheels.flatten(1)), -1), raw_state, queries[:, 4]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.outputs(frames)[0]

    @torch.jit.unused
    def predict_velocity_delta(self, state_context, issued_action):
        if self.prediction_head is None:
            raise RuntimeError("prediction head is absent from a deployment policy")
        return self.prediction_head(torch.cat((state_context, issued_action), -1))

    @torch.jit.unused
    def for_deployment(self):
        deployed = deepcopy(self).eval()
        deployed.prediction_head = None
        return deployed


class StructuredGaussianActor(nn.Module):
    """Training distribution; the deployment artifact contains only its policy."""

    def __init__(self, config=StructuredPolicyConfig()):
        super().__init__()
        self.policy = StructuredPolicy(config)
        self.log_std = nn.Parameter(torch.tensor([math.log(0.3)] * 4 + [math.log(0.5)] * 2))

    @staticmethod
    def _evaluation(mean, log_std, raw_action):
        distribution = Normal(mean, log_std.clamp(-4.0, 1.0).exp().expand_as(mean), validate_args=False)
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), mean, distribution.stddev)

    def forward(self, history: HistoryBatch):
        return self.policy(history.frames)

    def act(self, history, deterministic=False):
        mean = self(history)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        raw_action = mean if deterministic else Normal(mean, std, validate_args=False).sample()
        return ActionSample(raw_action, self._evaluation(mean, self.log_std, raw_action))

    def evaluate(self, history, raw_action):
        return self._evaluation(self(history), self.log_std, raw_action)

    def evaluate_training(self, history, raw_action, issued_action):
        mean, raw_state, context = self.policy.outputs(history.frames)
        evaluation = self._evaluation(mean, self.log_std, raw_action)
        return evaluation, raw_state, self.policy.predict_velocity_delta(context, issued_action)


class StructuredActorCritic(nn.Module):
    """Separate parameters and optimizers are required by the new trainer."""

    def __init__(self, config=StructuredPolicyConfig()):
        super().__init__()
        self.config = config
        self.actor = StructuredGaussianActor(config)
        first, second, third = config.critic_hidden_dims
        self.critic = nn.Sequential(nn.Linear(config.critic_input_dim, first), nn.ELU(),
                                    nn.Linear(first, second), nn.ELU(), nn.Linear(second, third), nn.ELU(),
                                    nn.Linear(third, 1), nn.Flatten(0))

    def critic_features(self, environment_features: torch.Tensor, raw_state: torch.Tensor) -> torch.Tensor:
        """188D物理特权量 + 本次Actor速度估计3D；价值梯度不进入Actor。"""
        if self.config.critic_input_dim != 191:
            raise ValueError('critic feature assembly requires the 191D observation contract')
        if (environment_features.ndim != 2 or environment_features.shape[1] != 188
                or raw_state.shape != (environment_features.shape[0], 8)
                or environment_features.device != raw_state.device
                or environment_features.dtype != raw_state.dtype
                or not environment_features.is_floating_point()):
            raise ValueError('critic features require matching [B,188] physical inputs and [B,8] actor state')
        return torch.cat((environment_features.detach(), raw_state[:, :3].detach()), -1)

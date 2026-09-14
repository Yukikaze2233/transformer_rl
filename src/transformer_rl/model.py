"""Stateless, time-aware Gaussian policy and an independent value network."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .config import ModelConfig
from .types import ActionSample, HistoryBatch, PolicyEvaluation


class _ResidualGate(nn.Module):
    """GTrXL-style GRU gate across depth, with an identity-favoring update bias."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.reset_x = nn.Linear(width, width, bias=False)
        self.reset_y = nn.Linear(width, width)
        self.update_x = nn.Linear(width, width, bias=False)
        self.update_y = nn.Linear(width, width)
        self.candidate_x = nn.Linear(width, width, bias=False)
        self.candidate_y = nn.Linear(width, width)
        nn.init.constant_(self.update_y.bias, -2.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        reset = torch.sigmoid(self.reset_x(x) + self.reset_y(y))
        update = torch.sigmoid(self.update_x(x) + self.update_y(y))
        candidate = torch.tanh(self.candidate_y(y) + self.candidate_x(reset * x))
        return (1 - update) * x + update * candidate


class _CausalBlock(nn.Module):
    """Explicit attention keeps grad, inference and export on the same path."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.attention_output = nn.Linear(config.d_model, config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.gated = config.residual_type == "gated"
        if self.gated:
            self.attention_gate = _ResidualGate(config.d_model)
            self.ffn_gate = _ResidualGate(config.d_model)

    def forward(
        self, tokens: torch.Tensor, allowed: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * self.head_dim**-0.5
        weights = scores.masked_fill(~allowed[:, None], float("-inf")).softmax(-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        attention = self.attention_output(attended)
        tokens = self.attention_gate(tokens, attention) if self.gated else tokens + attention
        feedforward = self.ffn(self.ffn_norm(tokens))
        tokens = self.ffn_gate(tokens, feedforward) if self.gated else tokens + feedforward
        return torch.where(valid[..., None], tokens, torch.zeros_like(tokens))


class GaussianActor(nn.Module):
    """Shared validated, stateless Gaussian policy interface.

    Construction registers no parameters or buffers. Concrete actors own their
    registration order, including log_std and preprocessing buffers, so existing
    checkpoint parameter IDs remain stable. Representation methods belong to
    the concrete architecture; only the five-input mean is shared.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def _register_frame_scale(self) -> None:
        config = self.config
        # Frames carry seconds; normalize time scalars only inside the model.
        frame_scale = torch.ones(config.frame_dim)
        age_start = config.proprio_dim + config.command_dim + config.action_dim
        frame_scale[age_start : age_start + config.sensor_groups] = 1 / config.time_scale_s
        frame_scale[-1] = 1 / config.time_scale_s
        self.register_buffer("frame_scale", frame_scale)

    def _scale_initial_mean_head(self, head: nn.Linear) -> None:
        # Preserve default initialization, RNG consumption and parameter identity/order.
        # This is initialization only: loaded or learned weights need no runtime gain.
        if self.config.mean_init_scale != 1.0:
            with torch.no_grad():
                head.weight.mul_(self.config.mean_init_scale)
                head.bias.mul_(self.config.mean_init_scale)

    def _validate_history(self, history: HistoryBatch) -> None:
        if not isinstance(history, HistoryBatch):
            raise TypeError("history must be a HistoryBatch")
        items = {
            "frames": history.frames,
            "times": history.times,
            "valid": history.valid,
            "command": history.command,
            "now": history.now,
        }
        for name, tensor in items.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"history.{name} must be a tensor")
        if history.frames.ndim != 3:
            raise ValueError("history.frames must have shape [B, L, F]")
        batch, length, _ = history.frames.shape
        if batch < 1 or not 1 <= length <= self.config.history_length:
            raise ValueError("history requires B >= 1 and 1 <= L <= history_length")
        shapes = {
            "frames": (batch, length, self.config.frame_dim),
            "times": (batch, length),
            "valid": (batch, length),
            "command": (batch, self.config.command_dim),
            "now": (batch,),
        }
        reference = next(self.parameters())
        for name, tensor in items.items():
            if tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"history.{name} must have shape {shapes[name]}")
            if tensor.device != reference.device:
                raise ValueError(f"history.{name} must be on model device {reference.device}")
        if history.valid.dtype != torch.bool:
            raise TypeError("history.valid must have bool dtype")
        if (
            history.frames.dtype != reference.dtype
            or history.command.dtype != reference.dtype
        ):
            raise TypeError(
                "history frames and command must have the model floating dtype"
            )
        if history.times.dtype not in (torch.float32, torch.float64):
            raise TypeError("history.times must be float32 or float64 (float64 recommended)")
        if history.now.dtype != history.times.dtype:
            raise TypeError("history.now and times must have the same dtype")
        meaningful = {
            "frames": history.frames[history.valid],
            "times": history.times[history.valid],
            "command": history.command,
            "now": history.now,
        }
        for name, tensor in meaningful.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"history.{name} contains nonfinite meaningful values")
        if ((history.times > history.now[:, None]) & history.valid).any():
            raise ValueError("valid history timestamps must not be later than now")
        previous = history.times.masked_fill(~history.valid, float("-inf"))
        previous = previous.cummax(1).values
        if ((history.times[:, 1:] <= previous[:, :-1]) & history.valid[:, 1:]).any():
            raise ValueError("valid history timestamps must be strictly increasing")

    def forward_tensors(
        self,
        frames: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        command: torch.Tensor,
        now: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked, exportable tensor-only mean; preserves the raw action domain."""
        raise NotImplementedError("concrete actors must implement the five-input mean")

    def forward(self, history: HistoryBatch) -> torch.Tensor:
        self._validate_history(history)
        return self.forward_tensors(
            history.frames, history.times, history.valid, history.command, history.now
        )

    def describe(self) -> dict:
        config = self.config
        transformer = config.actor_type == "transformer"
        return {
            "actor_type": config.actor_type,
            "history_length": config.history_length,
            "dimensions": (
                {"d_model": config.d_model, "num_heads": config.num_heads,
                 "num_layers": config.num_layers, "ffn_dim": config.ffn_dim}
                if transformer else {"hidden": list(config.baseline_hidden)}
                if config.actor_type == "mlp" else {"gru_hidden": config.gru_hidden}
            ),
            "compute": {
                "transformer": "dense attention quadratic in window length; gates add depth projections",
                "mlp": "flattened window first projection grows linearly with window length",
                "gru": "sequential cell recomputation linear in window length",
            }[config.actor_type],
            "state": "finite window recomputed each call; no cross-call hidden state",
            "history_encoder": {
                "transformer": "causal self-attention with appended current-command query",
                "mlp": "flattened masked history MLP with separate current command",
                "gru": "masked finite-window GRU with separate current command",
            }[config.actor_type],
            "time_encoding": {
                "position": config.time_encoding if transformer else "none",
                "history_time_features": (
                    "fixed sin/cos of elapsed age / time_scale_s" if transformer
                    and config.time_encoding == "elapsed" else
                    "fixed sin/cos of slot index" if transformer else
                    "float64(now - times), cast to model dtype, divided by time_scale_s"
                ),
                "frame_times": "sensor ages and policy interval retained, divided by time_scale_s",
                "time_scale_s": config.time_scale_s,
            },
            "residual_type": config.residual_type if transformer else None,
            "gating": "GRU-style depth residual, not streaming recurrence"
            if transformer and config.residual_type == "gated" else None,
            "auxiliary_indices": list(config.auxiliary_indices),
            "auxiliary_source": "query representation" if config.auxiliary_indices else None,
            "mean_initialization": {
                "scale": config.mean_init_scale,
                "target": "action mean output layer weight and bias only",
                "stored_in_weights": True,
                "runtime_gain": False,
                "semantics": "initialization factor already incorporated in weights; do not reapply",
            },
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "parameter_count_scope": "training actor including Gaussian std and optional auxiliary head",
            "comparison": "parameter counts and compute differ across architectures",
        }

    def _distribution(self, mean: torch.Tensor) -> Normal:
        return Normal(mean, self.log_std.exp().expand_as(mean), validate_args=False)

    @staticmethod
    def _evaluation(distribution: Normal, action: torch.Tensor) -> PolicyEvaluation:
        return PolicyEvaluation(
            log_prob=distribution.log_prob(action).sum(-1),
            entropy=distribution.entropy().sum(-1),
            mean=distribution.mean,
            std=distribution.stddev,
        )

    def act(self, history: HistoryBatch, deterministic: bool = False) -> ActionSample:
        distribution = self._distribution(self(history))
        action = distribution.mean if deterministic else distribution.sample()
        return ActionSample(action, self._evaluation(distribution, action))

    def evaluate(self, history: HistoryBatch, raw_action: torch.Tensor) -> PolicyEvaluation:
        mean = self(history)
        if not isinstance(raw_action, torch.Tensor):
            raise TypeError("raw_action must be a tensor")
        if raw_action.shape != mean.shape:
            raise ValueError(f"raw_action must have shape {tuple(mean.shape)}")
        if raw_action.dtype != mean.dtype:
            raise TypeError("raw_action must have the model floating dtype")
        if raw_action.device != mean.device:
            raise ValueError("raw_action must be on the model device")
        if not torch.isfinite(raw_action).all():
            raise ValueError("raw_action must be finite")
        return self._evaluation(self._distribution(mean), raw_action)


class TimeAwareActor(GaussianActor):
    """Causal history encoder with a current-command query and raw Gaussian actions.

    Public HistoryBatch methods validate shapes, devices and meaningful values.
    ``forward_tensors`` / ``encode_tensors`` are unchecked tensor-only export
    entry points; validate their inputs at the integration boundary. Padding may
    contain arbitrary values, including nonfinite frames and timestamps.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.actor_type != "transformer":
            raise ValueError("TimeAwareActor requires actor_type='transformer'")
        super().__init__(config)
        self.frame_projection = nn.Linear(config.frame_dim, config.d_model)
        self.command_projection = nn.Linear(config.command_dim, config.d_model)
        self.query_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.blocks = nn.ModuleList(
            _CausalBlock(config) for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.mean_head = nn.Linear(config.d_model, config.action_dim)
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), math.log(config.initial_std))
        )
        self.register_buffer(
            "time_frequencies",
            torch.exp(
                -math.log(10000.0)
                * torch.arange(0, config.d_model, 2, dtype=torch.float32)
                / config.d_model
            ),
        )
        self._register_frame_scale()
        if config.auxiliary_indices:
            self.auxiliary_head = nn.Linear(config.d_model, len(config.auxiliary_indices))
        self._scale_initial_mean_head(self.mean_head)

    def time_features_tensors(
        self, times: torch.Tensor, valid: torch.Tensor, now: torch.Tensor
    ) -> torch.Tensor:
        """Fixed sin/cos of elapsed age or slot index; padding encodes zero.

        Subtract in float64 *before* converting to the network dtype. Converting
        an already rounded float32 timestamp cannot recover lost precision.
        """
        now64 = now.to(torch.float64)[:, None]
        times64 = torch.where(valid, times.to(torch.float64), now64)
        age = (now64 - times64).to(self.frame_projection.weight.dtype)
        if self.config.time_encoding == "index":
            position = torch.arange(times.shape[1], device=times.device)
            phase = position.to(age.dtype)[None, :, None] * self.time_frequencies
        else:
            phase = (age / self.config.time_scale_s)[..., None] * self.time_frequencies
        features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return torch.where(valid[..., None], features, torch.zeros_like(features))

    def encode_tensors(
        self,
        frames: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        command: torch.Tensor,
        now: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, L+1, D] representations, with the command query last.

        This path is tensor-only and has no mutable cache or inference fast path.
        Historical tokens never attend to the appended current command query.
        """
        valid = valid & (times <= now[:, None])
        frames = torch.where(valid[..., None], frames, torch.zeros_like(frames))
        history_tokens = self.frame_projection(frames * self.frame_scale)
        history_tokens = history_tokens + self.time_features_tensors(times, valid, now)
        history_tokens = torch.where(
            valid[..., None], history_tokens, torch.zeros_like(history_tokens)
        )
        query = self.command_projection(command) + self.query_embedding
        tokens = torch.cat((history_tokens, query[:, None]), dim=1)
        query_valid = torch.ones_like(now[:, None], dtype=torch.bool)
        token_valid = torch.cat((valid, query_valid), dim=1)
        positions = torch.arange(tokens.shape[1], device=frames.device)
        causal = positions[:, None] >= positions[None, :]
        diagonal = positions[:, None] == positions[None, :]
        allowed = causal[None] & token_valid[:, None, :]
        # A padding row needs a finite softmax even though its output is erased.
        # The real query is always valid and can attend to itself for empty history.
        allowed = allowed | ((~token_valid)[:, :, None] & diagonal[None])
        for block in self.blocks:
            tokens = block(tokens, allowed, token_valid)
        tokens = self.output_norm(tokens)
        return torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))

    def encode(self, history: HistoryBatch) -> torch.Tensor:
        """Validated representations, including the query; useful for causal audits."""
        self._validate_history(history)
        return self.encode_tensors(
            history.frames, history.times, history.valid, history.command, history.now
        )

    def forward_tensors(
        self,
        frames: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        command: torch.Tensor,
        now: torch.Tensor,
    ) -> torch.Tensor:
        """Unchecked, exportable tensor-only mean; preserves the raw action domain."""
        tokens = self.encode_tensors(frames, times, valid, command, now)
        return self.mean_head(tokens[:, -1])

    def predict_auxiliary(self, history: HistoryBatch) -> torch.Tensor:
        if not self.config.auxiliary_indices:
            raise ValueError("actor has no configured auxiliary head")
        return self.auxiliary_head(self.encode(history)[:, -1])


class _WindowActor(GaussianActor):
    """Masked elapsed-time frame features shared by finite-window baselines."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), math.log(config.initial_std))
        )
        self._register_frame_scale()

    def _window_features(self, frames, times, valid, now):
        valid = valid & (times <= now[:, None])
        frames = torch.where(valid[..., None], frames, torch.zeros_like(frames))
        now64 = now.to(torch.float64)[:, None]
        times64 = torch.where(valid, times.to(torch.float64), now64)
        age = (now64 - times64).to(frames.dtype) / self.config.time_scale_s
        features = torch.cat((frames * self.frame_scale, age[..., None],
                              valid.to(frames.dtype)[..., None]), dim=-1)
        return features, valid


class HistoryMLPActor(_WindowActor):
    def __init__(self, config: ModelConfig) -> None:
        if config.actor_type != "mlp":
            raise ValueError("HistoryMLPActor requires actor_type='mlp'")
        super().__init__(config)
        width = config.history_length * (config.frame_dim + 2) + config.command_dim
        layers: list[nn.Module] = []
        for hidden in config.baseline_hidden:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, config.action_dim))
        self.network = nn.Sequential(*layers)
        self._scale_initial_mean_head(self.network[-1])

    def forward_tensors(self, frames, times, valid, command, now):
        features, _ = self._window_features(frames, times, valid, now)
        # Public short histories are left-padded to the configured fixed window.
        features = torch.nn.functional.pad(
            features, (0, 0, self.config.history_length - frames.shape[1], 0)
        )
        return self.network(torch.cat((features.flatten(1), command), dim=-1))


class WindowGRUActor(_WindowActor):
    def __init__(self, config: ModelConfig) -> None:
        if config.actor_type != "gru":
            raise ValueError("WindowGRUActor requires actor_type='gru'")
        super().__init__(config)
        self.cell = nn.GRUCell(config.frame_dim + 2, config.gru_hidden)
        self.mean_head = nn.Linear(config.gru_hidden + config.command_dim, config.action_dim)
        self._scale_initial_mean_head(self.mean_head)

    def forward_tensors(self, frames, times, valid, command, now):
        features, valid = self._window_features(frames, times, valid, now)
        hidden = frames.new_zeros((frames.shape[0], self.config.gru_hidden))
        for index in range(frames.shape[1]):
            candidate = self.cell(features[:, index], hidden)
            hidden = torch.where(valid[:, index, None], candidate, hidden)
        return self.mean_head(torch.cat((hidden, command), dim=-1))


class ValueCritic(nn.Module):
    """Independent privileged-state MLP; no shared actor parameters or history."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.critic_dim
        for hidden in config.critic_hidden:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, critic: torch.Tensor) -> torch.Tensor:
        if not isinstance(critic, torch.Tensor):
            raise TypeError("critic must be a tensor")
        if critic.ndim != 2 or critic.shape[0] < 1 or critic.shape[1] != self.config.critic_dim:
            raise ValueError(f"critic must have shape [B, {self.config.critic_dim}], B >= 1")
        reference = self.network[0].weight
        if critic.dtype != reference.dtype:
            raise TypeError("critic must have the model floating dtype")
        if critic.device != reference.device:
            raise ValueError("critic must be on the model device")
        if not torch.isfinite(critic).all():
            raise ValueError("critic must be finite")
        return self.network(critic).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = {
            "transformer": TimeAwareActor, "mlp": HistoryMLPActor, "gru": WindowGRUActor,
        }[config.actor_type](config)
        self.critic = ValueCritic(config)

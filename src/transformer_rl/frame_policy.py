"""Deterministic policies for already-packed, externally scaled observations.

This interface is separate from the time-aware ``HistoryBatch`` actor contract.
It does not pack or reinterpret sensor, command, action or timing fields. In
particular, a width of 35 does not establish V6 sensor semantics: the external
environment/deployment contract must establish feature order, units, scaling,
history sampling interval and episode reset behavior. Histories are fixed-size,
oldest-to-newest, with the current frame last and no padding mask. An external
repeat-first history buffer can supply complete windows after reset.

These networks output raw action means without clipping or squashing. Gaussian
exploration, critic, optimizers, history ownership and actuator processing stay
outside this module. ``frame_training`` supplies the packed-frame training
interface; the older time-aware checkpoint format remains a separate format.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import torch
from torch import nn

from .model import _CausalBlock


@dataclass(frozen=True)
class FramePolicyConfig:
    """Immutable architecture settings, independent of robot sensor semantics.

    ``mlp`` requires one frame. ``frame_stack_mlp`` feeds a flattened window
    directly to the head. History MLP and Transformer encoders concatenate a
    latent with the unmodified current frame before an ELU MLP head. The
    Transformer FFN uses GELU, as in the time-aware actor. History MLP latents
    have no prescribed physical meaning without a separate supervised target.

    ``mean_init_scale`` multiplies only the initialized final action linear
    weights and bias; its default preserves standard PyTorch initialization.
    ``position_reference`` defaults to the legacy oldest-frame index encoding.
    Transformers can instead use current-frame-relative integer positions, with
    the latest frame at zero and shared frame ages aligned across window lengths.
    """

    architecture: str = "mlp"
    frame_dim: int = 35
    action_dim: int = 6
    history_length: int = 1
    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    d_model: int = 96
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 192
    residual_type: str = "add"
    mean_init_scale: float = 1.0
    encoder_hidden_dims: tuple[int, ...] = (128, 128)
    history_latent_dim: int = 3
    readout_type: str = "last"
    position_reference: str = "oldest"

    def __post_init__(self) -> None:
        if self.architecture not in ("mlp", "frame_stack_mlp", "history_mlp", "transformer"):
            raise ValueError("unknown frame policy architecture")
        for name in (
            "frame_dim", "action_dim", "history_length", "d_model", "num_layers",
            "num_heads", "ffn_dim", "history_latent_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        widths = self.actor_hidden_dims
        if (not isinstance(widths, (tuple, list)) or not widths
                or any(type(width) is not int or width < 1 for width in widths)):
            raise ValueError("actor_hidden_dims must contain positive integer widths")
        object.__setattr__(self, "actor_hidden_dims", tuple(widths))
        for name in ("encoder_hidden_dims",):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or not values
                    or any(type(value) is not int or value < 1 for value in values)):
                raise ValueError(f"{name} requires positive integers")
            object.__setattr__(self, name, tuple(values))
        if self.d_model % 2 or self.d_model % self.num_heads:
            raise ValueError("d_model must be even and divisible by num_heads")
        if self.architecture == "mlp" and self.history_length != 1:
            raise ValueError("mlp requires history_length=1; use frame_stack_mlp for a window")
        if self.residual_type not in ("add", "gated"):
            raise ValueError("residual_type must be add or gated")
        if self.architecture != "transformer" and self.residual_type != "add":
            raise ValueError("gated residuals require transformer")
        if self.readout_type not in ("last", "query"):
            raise ValueError("readout_type must be last or query")
        if self.architecture != "transformer" and self.readout_type != "last":
            raise ValueError("query readout requires transformer")
        if self.position_reference not in ("oldest", "current"):
            raise ValueError("position_reference must be oldest or current")
        if self.architecture != "transformer" and self.position_reference != "oldest":
            raise ValueError("current position reference requires transformer")
        scale = self.mean_init_scale
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
            raise ValueError("mean_init_scale must be finite and nonnegative")
        object.__setattr__(self, "mean_init_scale", float(scale))

    @property
    def input_size(self) -> int:
        return self.history_length * self.frame_dim

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready data without exposing mutable config state."""
        result = asdict(self)
        for name in ("actor_hidden_dims", "encoder_hidden_dims"):
            result[name] = list(getattr(self, name))
        if self.position_reference == "oldest":
            # Preserve legacy canonical configs, checkpoint and anchor identities.
            result.pop("position_reference")
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FramePolicyConfig:
        """Parse config data, rejecting unknown fields and invalid dimensions."""
        if not isinstance(data, Mapping):
            raise ValueError("FramePolicyConfig requires a mapping")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown FramePolicyConfig fields: {sorted(map(str, unknown))}")
        return cls(**dict(data))


class _FrameCausalBlock(_CausalBlock):
    """Reuse the existing attention/gate math, with a static export branch."""

    __constants__ = ["gated"]


class _CurrentQueryReadout(nn.Module):
    """Current-frame query attends to a fully observed causal history window."""

    __constants__ = ["heads", "head_dim", "scale"]

    def __init__(self, config: FramePolicyConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scale = self.head_dim**-.5
        self.query = nn.Linear(config.frame_dim, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, current: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.size(0), tokens.size(1)
        query = self.query(current)
        q = query.reshape(batch, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(tokens).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(tokens).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        pooled = torch.matmul(weights, v).transpose(1, 2).reshape(batch, self.heads * self.head_dim)
        return self.norm(query + self.output(pooled))


class FramePolicy(nn.Module):
    """Fixed-window, stateless action mean from ``[batch, length, features]``.

    ``encode`` exposes encoder tokens; direct MLPs return the original frames.
    ``forward_features`` reads the encoder latent, the single MLP frame, or a
    flattened frame stack. Encoder variants use ``concat(current_frame, latent)``.

    Features are not normalized or otherwise rescaled. All frames must be real
    observations or an externally initialized repeat-first window; zero padding
    has no special meaning. No mutable cache or cross-call hidden state exists.
    """

    __constants__ = ["input_size", "history_length", "frame_dim", "is_transformer", "is_single_frame",
                     "architecture", "uses_encoder", "is_history_mlp", "uses_query"]

    def __init__(self, config: FramePolicyConfig) -> None:
        super().__init__()
        if not isinstance(config, FramePolicyConfig):
            raise TypeError("config must be a FramePolicyConfig")
        self.config = config
        self.input_size = config.input_size
        self.history_length = config.history_length
        self.frame_dim = config.frame_dim
        self.is_transformer = config.architecture == "transformer"
        self.is_single_frame = config.architecture == "mlp"
        self.architecture = config.architecture
        self.uses_encoder = config.architecture not in ("mlp", "frame_stack_mlp")
        self.is_history_mlp = config.architecture == "history_mlp"
        self.uses_query = self.is_transformer and config.readout_type == "query"
        if self.is_transformer:
            self.frame_projection = nn.Linear(config.frame_dim, config.d_model)
            self.blocks = nn.ModuleList(
                _FrameCausalBlock(config) for _ in range(config.num_layers)
            )
            self.output_norm = nn.LayerNorm(config.d_model)
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(0, config.d_model, 2, dtype=torch.float32)
                / config.d_model
            )
            if config.position_reference == "current":
                phase = torch.arange(1 - config.history_length, 1)[:, None] * frequencies
            else:
                phase = torch.arange(config.history_length, dtype=torch.float32)[:, None] * frequencies
            # Preserve the existing layout: all sin, then all cos.
            self.register_buffer("position_encoding", torch.cat((phase.sin(), phase.cos()), dim=-1)[None])
            positions = torch.arange(config.history_length)
            self.register_buffer("allowed", (positions[:, None] >= positions[None, :])[None])
            if self.uses_query:
                self.query_readout = _CurrentQueryReadout(config)
            width = config.frame_dim + config.d_model
        elif config.architecture == "history_mlp":
            encoder_layers: list[nn.Module] = []
            width = config.input_size
            for hidden in config.encoder_hidden_dims:
                encoder_layers.extend((nn.Linear(width, hidden), nn.ELU()))
                width = hidden
            encoder_layers.append(nn.Linear(width, config.history_latent_dim))
            self.history_encoder = nn.Sequential(*encoder_layers)
            width = config.frame_dim + config.history_latent_dim
        else:
            width = config.frame_dim if self.is_single_frame else config.input_size
        layers: list[nn.Module] = []
        for hidden in config.actor_hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ELU()))
            width = hidden
        layers.append(nn.Linear(width, config.action_dim))
        self.head = nn.Sequential(*layers)
        with torch.no_grad():
            self.head[-1].weight.mul_(config.mean_init_scale)
            self.head[-1].bias.mul_(config.mean_init_scale)

    @property
    @torch.jit.unused
    def output_layer(self) -> nn.Linear:
        return self.head[-1]

    def _validate_shape(self, frames: torch.Tensor) -> None:
        if frames.dim() != 3 or frames.size(0) < 1:
            raise ValueError("frames must have shape [batch, history_length, frame_dim] with batch >= 1")
        if frames.size(1) != self.history_length or frames.size(2) != self.frame_dim:
            raise ValueError("frames dimensions must match FramePolicyConfig exactly")

    @torch.jit.export
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Return encoder tokens, a single MLP latent, or direct-MLP frames."""
        # ONNX callers enforce the exported fixed window shape at their boundary.
        if not torch.jit.is_tracing():
            self._validate_shape(frames)
        if self.is_transformer:
            tokens = self.frame_projection(frames) + self.position_encoding
            valid = torch.ones_like(frames[:, :, 0], dtype=torch.bool)
            for block in self.blocks:
                tokens = block(tokens, self.allowed, valid)
            return self.output_norm(tokens)
        elif self.is_history_mlp:
            return self.history_encoder(frames.flatten(start_dim=1)).unsqueeze(1)
        return frames

    @torch.jit.export
    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the latent/readout before concatenating the current frame."""
        tokens = self.encode(frames)
        if self.uses_query:
            return self.query_readout(frames[:, -1], tokens)
        if self.uses_encoder or self.is_single_frame:
            return tokens[:, -1]
        return tokens.flatten(start_dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(frames)
        if self.uses_encoder:
            features = torch.cat((frames[:, -1], features), dim=-1)
        return self.head(features)

    @torch.jit.export
    def forward_flat(self, observations: torch.Tensor) -> torch.Tensor:
        """Equivalent mean for oldest-to-newest flattened ``[batch, L*F]``."""
        if not torch.jit.is_tracing():
            if observations.dim() != 2 or observations.size(0) < 1 or observations.size(1) != self.input_size:
                raise ValueError("observations must have shape [batch, history_length * frame_dim]")
        return self.forward(observations.reshape(observations.size(0), self.history_length, self.frame_dim))

"""Feed-forward history encoder with the formal actor's observation contract."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class HistoryMLPConfig:
    architecture: str = 'history_mlp'
    history_length: int = 5
    history_hidden_dims: tuple[int, int] = (256, 128)
    action_hidden_dim: int = 64
    state_hidden_dim: int = 32
    prediction_hidden_dim: int = 32
    critic_input_dim: int = 185
    critic_hidden_dims: tuple[int, int, int] = (256, 128, 64)
    mean_init_scale: float = 0.01
    estimate_log_std_min: float = -4.0
    estimate_log_std_max: float = 2.0

    def __post_init__(self):
        if self.architecture != 'history_mlp':
            raise ValueError('unknown history MLP architecture')
        for name in ('history_length', 'action_hidden_dim', 'state_hidden_dim',
                     'prediction_hidden_dim', 'critic_input_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, depth in (('history_hidden_dims', 2), ('critic_hidden_dims', 3)):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != depth
                    or any(type(width) is not int or width < 1 for width in value)):
                raise ValueError(f'{name} requires {depth} positive integer widths')
            object.__setattr__(self, name, tuple(value))
        for name in ('mean_init_scale', 'estimate_log_std_min', 'estimate_log_std_max'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.mean_init_scale < 0 or self.estimate_log_std_min >= self.estimate_log_std_max:
            raise ValueError('invalid mean initialization or estimation scale bounds')

    @property
    def normalization_contract(self):
        return f'fixed_manual35_{self.critic_input_dim}_no_running_statistics'


class HistoryMLPPolicy(nn.Module):
    """One complete oldest-to-newest window per call; no recurrent state.

    Commands already occupy the formal 35D frames. Their current values bypass
    the history encoder, alongside deployable estimates, to condition actions.
    Privileged supervision is supplied by the trainer and never by forward().
    """

    __constants__ = ['history_length', 'log_std_min', 'log_std_max']

    def __init__(self, config=HistoryMLPConfig(), *, prediction_head=True):
        super().__init__()
        if not isinstance(config, HistoryMLPConfig):
            raise TypeError('config must be a HistoryMLPConfig')
        self.history_length = config.history_length
        self.log_std_min = float(config.estimate_log_std_min)
        self.log_std_max = float(config.estimate_log_std_max)
        hidden, context = config.history_hidden_dims
        self.history_encoder = nn.Sequential(
            nn.Linear(35 * self.history_length, hidden), nn.ELU(),
            nn.Linear(hidden, context), nn.ELU(),
        )
        self.state_head = nn.Sequential(
            nn.Linear(context, config.state_hidden_dim), nn.GELU(),
            nn.Linear(config.state_hidden_dim, 8),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(context + 35 + 8, config.action_hidden_dim), nn.ELU(),
            nn.Linear(config.action_hidden_dim, 6),
        )
        self.prediction_head = (nn.Sequential(
            nn.Linear(context + 6, config.prediction_hidden_dim), nn.GELU(),
            nn.Linear(config.prediction_hidden_dim, 6),
        ) if prediction_head else None)
        with torch.no_grad():
            self.action_decoder[-1].weight.mul_(config.mean_init_scale)
            self.action_decoder[-1].bias.mul_(config.mean_init_scale)

    @torch.jit.unused
    def validate_frames(self, frames):
        reference = self.history_encoder[0].weight
        if (not isinstance(frames, torch.Tensor) or frames.ndim != 3 or frames.shape[0] < 1
                or frames.shape[1:] != (self.history_length, 35)):
            raise ValueError('frames require [B, history_length, 35] with B >= 1')
        if frames.dtype != reference.dtype or frames.device != reference.device:
            raise ValueError('frames must use the policy device and floating dtype')
        if not torch.isfinite(frames).all():
            raise ValueError('frames contain nonfinite values')
        if torch.count_nonzero(frames[..., 14:16]):
            raise ValueError('wheel-position compatibility slots must be zero')

    def _clean_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # Do not multiply by zero: invalid compatibility slots may contain NaN.
        return torch.cat((frames[..., :14], torch.zeros_like(frames[..., 14:16]),
                          frames[..., 16:]), -1)

    @torch.jit.export
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(self._clean_frames(frames).flatten(1))

    def state_features(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.cat((raw[:, :3], raw[:, 3:6].clamp(self.log_std_min, self.log_std_max),
                          raw[:, 6:8].sigmoid()), -1)

    def outputs(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self._clean_frames(frames)
        context = self.history_encoder(clean.flatten(1))
        raw_state = self.state_head(context)
        mean = self.action_decoder(torch.cat((context, clean[:, -1],
                                               self.state_features(raw_state)), -1))
        return mean, raw_state, context

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.outputs(frames)[0]

    @torch.jit.unused
    def predict_velocity_delta(self, state_context, issued_action):
        if self.prediction_head is None:
            raise RuntimeError('prediction head is absent from a deployment policy')
        return self.prediction_head(torch.cat((state_context, issued_action), -1))

    @torch.jit.unused
    def for_deployment(self):
        deployed = deepcopy(self).eval()
        deployed.prediction_head = None
        return deployed

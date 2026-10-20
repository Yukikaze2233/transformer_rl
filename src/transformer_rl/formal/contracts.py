"""控制、仿真与采样共享的数据包；各方持有独立副本。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch


ACTION_ORDER = ('L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3')


@dataclass(frozen=True)
class PhysicalLabels:
    velocity_body_m_s: torch.Tensor
    wheel_contact: torch.Tensor
    joint_velocity_rad_s: torch.Tensor
    valid: torch.Tensor
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    velocity_reference: str = 'base_link_origin_body'
    joint_order: tuple[str, ...] = ACTION_ORDER

    def owned(self):
        return type(self)(**{field.name: (getattr(self, field.name).detach().clone()
                                         if isinstance(getattr(self, field.name), torch.Tensor)
                                         else getattr(self, field.name)) for field in fields(self)})


@dataclass(frozen=True)
class LabeledObservation:
    frame: torch.Tensor
    critic: torch.Tensor
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    policy_allowed: torch.Tensor
    labels: PhysicalLabels

    def owned(self):
        return type(self)(**{field.name: (self.labels.owned() if field.name == 'labels'
                                         else getattr(self, field.name).detach().clone()) for field in fields(self)})


@dataclass(frozen=True)
class LabeledStep:
    observation: LabeledObservation
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_critic: torch.Tensor
    final_critic_valid: torch.Tensor
    final_labels: PhysicalLabels
    fully_policy_owned: torch.Tensor
    handoff_after: torch.Tensor
    # PRE-reset actor frame at the interval endpoint. Optional only for older
    # adapters; LabEnv sets it so timeout critics can use the terminal window.
    terminal_frame: torch.Tensor | None = None


@dataclass(frozen=True)
class StructuredCollectionConfig:
    policy_dt_s: float = 0.01
    gamma: float = 0.99498743710662
    gae_lambda: float = 0.9746794344808963
    action_bounds: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 9.0, 9.0)

    def __post_init__(self):
        if (type(self.policy_dt_s) not in (int, float) or self.policy_dt_s != 0.01
                or type(self.gamma) not in (int, float) or not 0 < self.gamma < 1
                or type(self.gae_lambda) not in (int, float) or not 0 <= self.gae_lambda <= 1
                or not isinstance(self.action_bounds, (tuple, list)) or len(self.action_bounds) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in self.action_bounds)):
            raise ValueError('collection requires the fixed 100Hz discount and six positive action bounds')
        object.__setattr__(self, 'action_bounds', tuple(self.action_bounds))

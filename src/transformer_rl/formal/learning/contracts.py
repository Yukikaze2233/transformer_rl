"""Private rollout ownership, supervision and return contracts for the candidate.

No simulator, command protocol, artifact format or existing PPO batch is changed.
Every label is supplied explicitly; privileged columns are never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BoundaryMasks:
    actor: torch.Tensor
    critic: torch.Tensor
    state_auxiliary: torch.Tensor
    next_auxiliary: torch.Tensor
    trace: torch.Tensor
    bootstrap: torch.Tensor

    @classmethod
    def from_events(cls, *, fully_policy_owned, terminated, truncated, final_value_valid,
                    handoff_after, rollout_end, state_label_valid, next_label_valid,
                    next_same_control_epoch):
        """Describe complete intervals, with true failure taking precedence.

        ``next_label_valid`` refers to a PRE-reset physical endpoint, including
        time-limit endpoints when available. It never refers to an auto-reset
        observation. Mid-interval override is ``fully_policy_owned=False``.
        """
        reference = fully_policy_owned
        values = (terminated, truncated, final_value_valid, handoff_after, rollout_end,
                  state_label_valid, next_label_valid, next_same_control_epoch)
        for tensor in (reference, *values):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bool
                    or tensor.shape != reference.shape or tensor.device != reference.device):
                raise ValueError("boundary events require bool tensors with one common shape/device")
        needs_final = reference & truncated & ~terminated & ~handoff_after
        if (needs_final & ~final_value_valid).any():
            raise ValueError("time-limit bootstrap requires a valid PRE-reset terminal value")
        continuing = reference & ~terminated & ~handoff_after
        return cls(
            actor=reference.clone(), critic=reference.clone(),
            state_auxiliary=reference & state_label_valid,
            next_auxiliary=continuing & next_label_valid & next_same_control_epoch,
            trace=continuing & ~truncated & ~rollout_end,
            bootstrap=continuing,
        )

    def validate(self):
        reference = self.actor
        for field in fields(self):
            value = getattr(self, field.name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                    or value.shape != reference.shape or value.device != reference.device):
                raise ValueError("masks require bool tensors with a common shape/device")
        if not torch.equal(self.actor, self.critic):
            raise ValueError("actor and critic require the same complete policy-owned intervals")
        for mask in (self.state_auxiliary, self.next_auxiliary, self.trace, self.bootstrap):
            if (mask & ~self.actor).any():
                raise ValueError("scripted or mixed intervals cannot supply policy targets or returns")
        if (self.trace & ~self.bootstrap).any():
            raise ValueError("a GAE trace requires bootstrap at the same boundary")


@torch.no_grad()
def masked_gae(reward, value, final_next_value, masks: BoundaryMasks, *, gamma, gae_lambda):
    """Time-major GAE with separate propagation, bootstrap and ownership masks."""
    masks.validate()
    if reward.ndim != 2 or reward.shape != masks.actor.shape or reward.numel() == 0:
        raise ValueError("GAE requires nonempty [vector_steps, environments] tensors")
    if (type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma < 1
            or type(gae_lambda) not in (int, float) or not math.isfinite(gae_lambda)
            or not 0 <= gae_lambda <= 1):
        raise ValueError("invalid GAE discounts")
    for tensor in (reward, value, final_next_value):
        if (tensor.shape != reward.shape or tensor.device != reward.device
                or tensor.dtype != reward.dtype or not tensor.is_floating_point()):
            raise ValueError("GAE values must share shape, floating dtype and device")
    if masks.actor.device != reward.device:
        raise ValueError("GAE masks must be on the value device")
    for tensor, valid in ((reward, masks.critic), (value, masks.critic),
                          (final_next_value, masks.bootstrap)):
        if not torch.isfinite(tensor[valid]).all():
            raise ValueError("nonfinite value in a meaningful GAE interval")
    current_value = torch.where(masks.critic, value, 0)
    next_value = torch.where(masks.bootstrap, final_next_value, 0)
    delta = torch.where(masks.critic, reward, 0) + gamma * next_value - current_value
    advantage = torch.zeros_like(reward)
    following = torch.zeros_like(reward[0])
    for step in range(reward.shape[0] - 1, -1, -1):
        following = delta[step] + gamma * gae_lambda * masks.trace[step] * following
        following = torch.where(masks.critic[step], following, 0)
        advantage[step] = following
    return advantage, advantage + current_value


class ObservationWindows:
    """Owned raw prefix plus one frame per endpoint, with reset-aware indices.

    Time-major endpoint IDs are ``step * environments + environment``. Only raw
    frames and integer window indices are cached; features are always recomputed
    under the current actor. ``reset_before`` also denotes a control-epoch change.
    """

    def __init__(self, prefix, observations, reset_before):
        if (prefix.ndim != 3 or observations.ndim != 3 or prefix.shape[1:] != observations.shape[1:]
                or observations.shape[0] < 1 or observations.shape[1] < 1 or observations.shape[2] != 35
                or prefix.device != observations.device or prefix.dtype != observations.dtype
                or not observations.is_floating_point()):
            raise ValueError("raw prefix/observations require [steps, environments, 35]")
        if (reset_before.shape != observations.shape[:2] or reset_before.dtype != torch.bool
                or reset_before.device != observations.device):
            raise ValueError("reset_before requires bool [steps, environments]")
        if (not torch.isfinite(prefix).all() or not torch.isfinite(observations).all()
                or torch.count_nonzero(prefix[..., 14:16]) or torch.count_nonzero(observations[..., 14:16])):
            raise ValueError("raw windows require finite frames and zero wheel-position slots")
        self.history_length = prefix.shape[0] + 1
        self.steps, self.num_envs = observations.shape[:2]
        self._raw = torch.cat((prefix.detach(), observations.detach()), 0)
        step = torch.arange(self.steps, device=observations.device) + self.history_length - 1
        resets = torch.where(reset_before, step[:, None], 0)
        floor = resets.cummax(0).values
        ages = torch.arange(self.history_length - 1, -1, -1, device=observations.device)
        self._indices = torch.maximum(step[:, None, None] - ages, floor[..., None]).flatten(0, 1)
        self._environments = torch.arange(self.num_envs, device=observations.device).repeat(self.steps)

    def __len__(self):
        return self.steps * self.num_envs

    @property
    def device(self):
        return self._raw.device

    @property
    def dtype(self):
        return self._raw.dtype

    def gather(self, endpoints):
        if (not isinstance(endpoints, torch.Tensor) or endpoints.ndim != 1 or endpoints.dtype != torch.long
                or endpoints.device != self.device):
            raise ValueError("endpoint IDs require an int64 vector on the window device")
        # index_select rejects negative IDs without a per-minibatch GPU scalar
        # synchronization; Python-style advanced indexing would wrap them.
        positions = self._indices.index_select(0, endpoints)
        environments = self._environments.index_select(0, endpoints)
        return self._raw[positions, environments[:, None]]

    @property
    def raw_storage_elements(self):
        return self._raw.numel()


@dataclass(frozen=True)
class AuxiliaryLabels:
    velocity: torch.Tensor
    wheel_contact: torch.Tensor
    velocity_delta: torch.Tensor

    def validate(self, state_valid, next_valid):
        count = state_valid.numel()
        if (state_valid.shape != (count,) or next_valid.shape != (count,)
                or state_valid.dtype != torch.bool or next_valid.dtype != torch.bool
                or state_valid.device != next_valid.device):
            raise ValueError("auxiliary masks require bool endpoint vectors")
        for tensor, width, valid in ((self.velocity, 3, state_valid), (self.wheel_contact, 2, state_valid),
                                     (self.velocity_delta, 6, next_valid)):
            if (tensor.shape != (count, width) or tensor.device != state_valid.device
                    or not tensor.is_floating_point() or not torch.isfinite(tensor[valid]).all()):
                raise ValueError("auxiliary labels must be finite and have their named dimensions")
        contacts = self.wheel_contact[state_valid]
        if ((contacts < 0) | (contacts > 1)).any():
            raise ValueError("contact targets must be probabilities")

    def index(self, endpoints):
        return type(self)(*(getattr(self, field.name)[endpoints] for field in fields(self)))


def auxiliary_losses(raw_state, velocity_delta, labels: AuxiliaryLabels, state_valid, next_valid,
                     *, log_std_min=-4.0, log_std_max=2.0):
    """Separate valid-count reductions; ignored NaN labels never enter arithmetic.

    Validation belongs to the batch boundary. The trainer combines these terms
    with PPO in one actor transaction; there is no auxiliary optimizer.
    """
    # Substitute only semantically absent labels before arithmetic, then apply
    # masks. This avoids NaN*0 and dynamic GPU boolean-index synchronization.
    target = torch.where(state_valid[:, None], labels.velocity.detach(), raw_state[:, :3].detach())
    log_std = raw_state[:, 3:6].clamp(log_std_min, log_std_max)
    velocity_terms = 0.5 * ((target - raw_state[:, :3]) * (-log_std).exp()).square() + log_std
    velocity = (velocity_terms * state_valid[:, None]).sum()
    velocity = velocity / (state_valid.sum() * 3).clamp_min(1)
    contact_target = torch.where(state_valid[:, None], labels.wheel_contact.detach(), 0)
    contact_terms = F.binary_cross_entropy_with_logits(raw_state[:, 6:8], contact_target, reduction='none')
    contact = (contact_terms * state_valid[:, None]).sum()
    contact = contact / (state_valid.sum() * 2).clamp_min(1)
    scales = velocity_delta.new_tensor([5.0] * 4 + [20.0] * 2)
    target_delta = torch.where(next_valid[:, None], labels.velocity_delta.detach(), velocity_delta.detach())
    following = (F.smooth_l1_loss(velocity_delta / scales, target_delta / scales, reduction='none')
                 * next_valid[:, None]).sum()
    following = following / (next_valid.sum() * 6).clamp_min(1)
    return velocity, contact, following


def bounded_task_reward(positive_density, cost_density, failed, *, policy_dt, gamma,
                        failure_event=-4.0, absorbing_density=-11.0):
    """Bound costs while charging the absorbing failure tail exactly once."""
    if (type(policy_dt) not in (int, float) or not math.isfinite(policy_dt) or policy_dt <= 0
            or type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0 < gamma < 1
            or not math.isfinite(failure_event) or failure_event >= 0
            or not math.isfinite(absorbing_density) or absorbing_density >= -10):
        raise ValueError("invalid reward time or absorbing failure contract")
    if (positive_density.shape != cost_density.shape or failed.shape != cost_density.shape
            or failed.dtype != torch.bool or failed.device != cost_density.device
            or positive_density.device != cost_density.device or positive_density.dtype != cost_density.dtype
            or not positive_density.is_floating_point()):
        raise ValueError("reward components and failure mask require one common shape/device")
    if (not torch.isfinite(positive_density).all() or not torch.isfinite(cost_density).all()
            or ((positive_density < 0) | (positive_density > 8)).any() or (cost_density < 0).any()):
        raise ValueError("positive density must be in [0,8] and cost density nonnegative")
    density = positive_density - 10.0 * (cost_density / 10.0).tanh()
    absorbing_value = absorbing_density * policy_dt / (1 - gamma)
    return density * policy_dt + failed * (failure_event + gamma * absorbing_value)

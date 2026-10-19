"""Private PPO candidate with joint actor/auxiliary KL transactions.

The existing optimizer and checkpoint formats remain independent. Probe checks
are empirical guards, not a mathematical trust region over unobserved states.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import math
import time

import torch
from torch import nn

from .contracts import AuxiliaryLabels, BoundaryMasks, ObservationWindows, auxiliary_losses


@dataclass(frozen=True)
class StructuredPPOConfig:
    epochs: int = 4
    minibatches: int = 16
    learning_rate: float = 0.0001
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 4.0
    max_gradient_norm: float = 1.0
    target_kl: float = 0.01
    reject_kl: float = 0.02
    probe_endpoints: int = 1024
    evaluation_chunk: int = 3072
    training_chunk: int = 3072
    diagnostics_interval: int = 20
    velocity_coefficient: float = 0.1
    contact_coefficient: float = 0.02
    prediction_coefficient: float = 0.05

    def __post_init__(self):
        for name in ('epochs', 'minibatches', 'probe_endpoints', 'evaluation_chunk', 'training_chunk',
                     'diagnostics_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for field in fields(self):
            if field.name not in ('epochs', 'minibatches', 'probe_endpoints', 'evaluation_chunk',
                                  'training_chunk', 'diagnostics_interval'):
                value = getattr(self, field.name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{field.name} must be finite and nonnegative')
        if (self.learning_rate <= 0 or self.max_gradient_norm <= 0 or not 0 < self.clip_ratio < 1
                or not 0 < self.target_kl < self.reject_kl):
            raise ValueError('invalid optimizer or KL bounds')


@dataclass(frozen=True)
class StructuredBatch:
    windows: ObservationWindows
    critic: torch.Tensor
    raw_action: torch.Tensor
    issued_action: torch.Tensor
    old_mean: torch.Tensor
    old_log_std: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: BoundaryMasks
    labels: AuxiliaryLabels

    def __len__(self):
        return len(self.windows)

    def owned(self):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                value = value.detach().clone()
            elif field.name in ('masks', 'labels'):
                value = type(value)(*(getattr(value, part.name).detach().clone() for part in fields(value)))
            values[field.name] = value
        return type(self)(**values)

    def validate(self, model):
        self.masks.validate()
        count = len(self)
        if self.masks.actor.shape != (count,) or self.masks.actor.device != self.windows.device:
            raise ValueError('training masks require one flat entry per raw-window endpoint')
        if self.windows.history_length != model.actor.policy.history_length:
            raise ValueError('raw-window history length differs from the policy')
        reference = next(model.actor.parameters())
        if reference.device != self.windows.device or reference.dtype != self.windows.dtype:
            raise ValueError('policy and rollout must use the same device and dtype')
        for name in ('critic', 'raw_action', 'issued_action', 'old_mean', 'old_log_std',
                     'old_log_prob', 'old_value', 'advantages', 'returns'):
            value = getattr(self, name)
            width = 81 if name == 'critic' else 6 if name in (
                'raw_action', 'issued_action', 'old_mean', 'old_log_std') else None
            shape = (count, width) if width is not None else (count,)
            if value.shape != shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f'{name} has invalid shape, dtype or device')
            valid = self.masks.critic if name in ('critic', 'old_value', 'returns') else self.masks.actor
            if not torch.isfinite(value[valid]).all():
                raise ValueError(f'{name} has nonfinite meaningful entries')
        self.labels.validate(self.masks.state_auxiliary, self.masks.next_auxiliary)
        if any(getattr(self.labels, field.name).dtype != reference.dtype for field in fields(self.labels)):
            raise ValueError('auxiliary targets must use the policy floating dtype')


class _ParameterTransaction:
    """Restoring Adam includes moments, step counters and parameter-group state."""

    def __init__(self, module, optimizer):
        self.module = module
        self.optimizer = optimizer
        self.parameters = deepcopy(module.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def restore(self):
        self.module.load_state_dict(self.parameters)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.optimizer.zero_grad(set_to_none=True)


class StructuredPPOTrainer:
    """One actor backward/step per minibatch; the critic has its own optimizer.

    Every executed step and endpoint use is charged even after a rollback.
    Metrics leave the device at update end, apart from the necessary KL admission
    decisions. Probe and rollback time is included in optimization wall time.
    """

    def __init__(self, model, config=StructuredPPOConfig()):
        self.model, self.config = model, config
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.learning_rate)
        self.update_index = 0
        self.consumed = dict(fresh_endpoints=0, actor_optimizer_steps=0, actor_endpoint_uses=0,
                             critic_optimizer_steps=0, critic_endpoint_uses=0)
        self._shared_parameters = [p for name, p in model.actor.policy.named_parameters()
                                   if not name.startswith(('state_head.', 'prediction_head.',
                                                           'leg_decoder.', 'wheel_decoder.'))]

    @staticmethod
    def diagonal_kl(old_mean, old_log_std, mean, log_std):
        """旧→新 KL = Σ[log(σ/σ₀) + (σ₀² + (μ₀−μ)²)/(2σ²) − 1/2]。"""
        return (log_std - old_log_std + (old_log_std.mul(2).exp() + (old_mean - mean).square())
                / (2 * log_std.mul(2).exp()) - 0.5).sum(-1)

    @torch.no_grad()
    def _kl(self, batch, endpoints):
        total = batch.old_log_prob.new_zeros(())
        if not endpoints.numel():
            return total
        for part in endpoints.split(self.config.evaluation_chunk):
            mean = self.model.actor.policy(batch.windows.gather(part))
            log_std = self.model.actor.log_std.clamp(-4.0, 1.0).expand_as(mean)
            total += self.diagonal_kl(batch.old_mean[part], batch.old_log_std[part], mean, log_std).sum()
        return total / max(endpoints.numel(), 1)

    @torch.no_grad()
    def _validate_behavior(self, batch, endpoints):
        for part in endpoints.split(self.config.evaluation_chunk):
            if not part.numel():
                continue
            mean = self.model.actor.policy(batch.windows.gather(part))
            evaluation = self.model.actor._evaluation(mean, self.model.actor.log_std, batch.raw_action[part])
            expected_std = batch.old_log_std[part].exp()
            for name, actual, expected in (('mean', evaluation.mean, batch.old_mean[part]),
                                           ('std', evaluation.std, expected_std),
                                           ('logp', evaluation.log_prob, batch.old_log_prob[part])):
                if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                    raise ValueError(f'behavior {name} mismatch before any optimizer step')

    def _losses(self, batch, endpoints, advantages, *, normalization=None):
        actor = self.model.actor
        mean, raw_state, context = actor.policy.outputs(batch.windows.gather(endpoints))
        evaluation = actor._evaluation(mean, actor.log_std, batch.raw_action[endpoints])
        # r = exp(logπ_new − logπ_old)，使用原始采样动作的概率。
        ratio = (evaluation.log_prob - batch.old_log_prob[endpoints]).exp()
        clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        # 最小化 −E[min(rA, clip(r, 1±ε)A)]；A 已按有效策略样本归一化。
        terms = -torch.minimum(ratio * advantages[endpoints], clipped * advantages[endpoints])
        if normalization is None:
            objective = terms.mean() - self.config.entropy_coefficient * evaluation.entropy.mean()
        else:
            objective = (terms.sum() - self.config.entropy_coefficient * evaluation.entropy.sum()) / normalization[0]
        prediction = actor.policy.predict_velocity_delta(context, batch.issued_action[endpoints])
        auxiliary = auxiliary_losses(raw_state, prediction, batch.labels.index(endpoints),
                                      batch.masks.state_auxiliary[endpoints], batch.masks.next_auxiliary[endpoints],
                                      log_std_min=actor.policy.log_std_min, log_std_max=actor.policy.log_std_max,
                                      valid_counts=None if normalization is None else normalization[1:])
        components = (objective, *auxiliary)
        loss = (objective + self.config.velocity_coefficient * auxiliary[0]
                + self.config.contact_coefficient * auxiliary[1]
                + self.config.prediction_coefficient * auxiliary[2])
        return loss, components

    def _actor_backward(self, batch, endpoints, advantages):
        """分块累积同一 minibatch 的梯度；每项仍按整批有效样本数归一化。"""
        if endpoints.numel() <= self.config.training_chunk:
            self._losses(batch, endpoints, advantages)[0].backward()
            return
        normalization = (endpoints.numel(), batch.masks.state_auxiliary[endpoints].sum(),
                         batch.masks.next_auxiliary[endpoints].sum())
        for part in endpoints.split(self.config.training_chunk):
            self._losses(batch, part, advantages, normalization=normalization)[0].backward()

    def _gradient_diagnostics(self, batch, endpoints, advantages):
        _, components = self._losses(batch, endpoints, advantages)
        vectors = []
        for component in components:
            gradients = torch.autograd.grad(component, self._shared_parameters, retain_graph=True,
                                             allow_unused=True)
            vectors.append(torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).flatten()
                                      for p, g in zip(self._shared_parameters, gradients)]))
        names = ('policy_entropy', 'velocity', 'contact', 'prediction')
        result = {f'shared_gradient_norm/{name}': vector.norm() for name, vector in zip(names, vectors)}
        weights = (1.0, self.config.velocity_coefficient, self.config.contact_coefficient,
                   self.config.prediction_coefficient)
        result['shared_gradient_norm/combined'] = sum(w * v for w, v in zip(weights, vectors)).norm()
        for name, vector in zip(names[1:], vectors[1:]):
            divisor = vectors[0].norm() * vector.norm()
            result[f'shared_gradient_cosine/{name}'] = torch.where(
                divisor > 0, torch.dot(vectors[0], vector) / divisor.clamp_min(1e-30),
                divisor.new_tensor(float('nan')))
        return result

    def update(self, batch: StructuredBatch):
        started = time.perf_counter()
        batch.validate(self.model)
        batch = batch.owned()
        actor_endpoints = batch.masks.actor.nonzero().flatten()
        critic_endpoints = batch.masks.critic.nonzero().flatten()
        self._validate_behavior(batch, actor_endpoints)
        advantages = batch.advantages.clone()
        if actor_endpoints.numel():
            selected = advantages[actor_endpoints]
            advantages[actor_endpoints] = (selected - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
        count = min(actor_endpoints.numel(), self.config.probe_endpoints)
        slots = torch.linspace(0, max(actor_endpoints.numel() - 1, 0), steps=count,
                               device=batch.windows.device).long() if count else actor_endpoints
        probe = actor_endpoints[slots]
        actor_start = _ParameterTransaction(self.model.actor, self.actor_optimizer)
        critic_start = _ParameterTransaction(self.model.critic, self.critic_optimizer)
        self.update_index += 1
        self.consumed['fresh_endpoints'] += len(batch)
        metrics = dict(fresh_endpoints=len(batch), actor_optimizer_steps=0, critic_optimizer_steps=0,
                       actor_endpoint_uses=0, critic_endpoint_uses=0, retained_actor_steps=0,
                       rolled_back_actor_steps=0, actor_stop_reason=None, whole_actor_update_rolled_back=False)
        tensors = {}
        norms = []
        critic_norms = []
        try:
            if actor_endpoints.numel() and self.update_index % self.config.diagnostics_interval == 0:
                tensors.update(self._gradient_diagnostics(batch, probe, advantages))
            actor_finished = actor_endpoints.numel() == 0
            for _ in range(self.config.epochs):
                if actor_finished:
                    break
                order = actor_endpoints[torch.randperm(actor_endpoints.numel(), device=batch.windows.device)]
                for endpoints in torch.tensor_split(order, self.config.minibatches):
                    if not endpoints.numel():
                        continue
                    transaction = _ParameterTransaction(self.model.actor, self.actor_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self._actor_backward(batch, endpoints, advantages)
                    norm = nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.max_gradient_norm)
                    self.actor_optimizer.step()
                    metrics['actor_optimizer_steps'] += 1
                    metrics['actor_endpoint_uses'] += endpoints.numel()
                    self.consumed['actor_optimizer_steps'] += 1
                    self.consumed['actor_endpoint_uses'] += endpoints.numel()
                    kl = self._kl(batch, probe)
                    norms.append(norm.detach())
                    # One admission decision is unavoidable for real early stop.
                    # The combined code also rejects auxiliary-only NaN gradients.
                    bad = ~torch.isfinite(norm) | ~torch.isfinite(kl) | (kl > self.config.reject_kl)
                    decision = torch.where(bad, kl.new_tensor(2), (kl > self.config.target_kl).to(kl.dtype))
                    status = int(decision.item())
                    if status == 2:
                        transaction.restore()
                        metrics['rolled_back_actor_steps'] += 1
                        metrics['actor_stop_reason'] = 'rejected_actor_and_auxiliary_transaction'
                    else:
                        metrics['retained_actor_steps'] += 1
                    if status:
                        actor_finished = True
                        if status == 1:
                            metrics['actor_stop_reason'] = 'target_kl_reached'
                        break
            full_kl = self._kl(batch, actor_endpoints)
            if not torch.isfinite(full_kl) or full_kl > self.config.reject_kl:
                tensors['rejected_full_rollout_kl'] = full_kl
                actor_start.restore()
                metrics['rolled_back_actor_steps'] += metrics['retained_actor_steps']
                metrics['retained_actor_steps'] = 0
                metrics['whole_actor_update_rolled_back'] = True
                metrics['actor_stop_reason'] = 'rejected_full_rollout_kl'
                full_kl = self._kl(batch, actor_endpoints)
            tensors['full_rollout_kl_after_guard'] = full_kl
            actor_finished_at = time.perf_counter()
            for _ in range(self.config.epochs):
                order = critic_endpoints[torch.randperm(critic_endpoints.numel(), device=batch.windows.device)]
                for endpoints in torch.tensor_split(order, self.config.minibatches):
                    if not endpoints.numel():
                        continue
                    value = self.model.critic(batch.critic[endpoints])
                    clipped = batch.old_value[endpoints] + (value - batch.old_value[endpoints]).clamp(
                        -self.config.value_clip, self.config.value_clip)
                    loss = torch.maximum((value - batch.returns[endpoints]).square(),
                                         (clipped - batch.returns[endpoints]).square()).mean()
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (0.5 * self.config.value_coefficient * loss).backward()
                    critic_norms.append(nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.config.max_gradient_norm).detach())
                    self.critic_optimizer.step()
                    metrics['critic_optimizer_steps'] += 1
                    metrics['critic_endpoint_uses'] += endpoints.numel()
                    self.consumed['critic_optimizer_steps'] += 1
                    self.consumed['critic_endpoint_uses'] += endpoints.numel()
            if norms:
                tensors['actor_combined_gradient_norm_mean'] = torch.stack(norms).mean()
            if critic_norms:
                all_critic_norms = torch.stack(critic_norms)
                if not torch.isfinite(all_critic_norms).all():
                    raise FloatingPointError('critic gradients became nonfinite; whole update restored')
                tensors['critic_gradient_norm_mean'] = all_critic_norms.mean()
            metrics.update({name: (value if math.isfinite(value) else None)
                            for name, value in ((name, float(value.detach())) for name, value in tensors.items())})
            metrics['actor_and_guard_elapsed_s'] = actor_finished_at - started
            metrics['critic_elapsed_s'] = time.perf_counter() - actor_finished_at
            metrics['total_optimization_elapsed_s'] = time.perf_counter() - started
            metrics['cumulative_consumed'] = dict(self.consumed)
            return metrics
        except BaseException:
            # A failed update cannot leave an uncheckpointable half-transaction.
            # Consumption/RNG advance is retained and must be recorded by the caller.
            actor_start.restore()
            critic_start.restore()
            raise

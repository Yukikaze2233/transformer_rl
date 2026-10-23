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


def _is_cuda_fatal(error):
    """A lost CUDA context cannot be repaired by replaying a CPU-side snapshot."""
    cuda_types = tuple(
        candidate for candidate in
        (getattr(torch, 'AcceleratorError', None), getattr(torch, 'CudaError', None))
        if isinstance(candidate, type)
    )
    if cuda_types and isinstance(error, cuda_types):
        return True
    text = f'{type(error).__name__}: {error}'.lower()
    return 'cuda error' in text or 'cudaerror' in text or 'device-side assert' in text


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
    kl_early_stop: bool = True
    validate_behavior: bool = True
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
        if type(self.kl_early_stop) is not bool or type(self.validate_behavior) is not bool:
            raise ValueError('kl_early_stop and validate_behavior must be explicit bools')
        for field in fields(self):
            if field.name not in ('epochs', 'minibatches', 'probe_endpoints', 'evaluation_chunk',
                                  'training_chunk', 'diagnostics_interval', 'kl_early_stop',
                                  'validate_behavior'):
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
            width = model.config.critic_input_dim if name == 'critic' else 6 if name in (
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


class _LossTelemetry:
    """Detached sufficient statistics; no state survives an update call."""

    _ACTOR_NAMES = ('surrogate', 'entropy', 'velocity', 'contact', 'prediction', 'combined_objective')

    def __init__(self, reference):
        self.attempted = reference.new_zeros((len(self._ACTOR_NAMES), 2), dtype=torch.float64)
        self.retained = torch.zeros_like(self.attempted)
        self.minibatch = torch.zeros_like(self.attempted)
        self.critic = reference.new_zeros((2, 2), dtype=torch.float64)

    def begin_actor(self):
        self.minibatch.zero_()

    @torch.no_grad()
    def observe_actor(self, terms, entropy, auxiliary, combined, state_valid, next_valid, normalization):
        state_count, next_count = state_valid.sum(), next_valid.sum()
        denominators = ((state_count, next_count) if normalization is None else normalization[1:])
        actor_count = terms.numel()
        sums = torch.stack((terms.detach().sum(), entropy.detach().sum(),
                            auxiliary[0].detach() * denominators[0],
                            auxiliary[1].detach() * denominators[0],
                            auxiliary[2].detach() * denominators[1],
                            combined.detach() * (actor_count if normalization is None else normalization[0])))
        counts = torch.stack((state_count.new_tensor(actor_count), state_count.new_tensor(actor_count),
                              state_count, state_count, next_count, state_count.new_tensor(actor_count)))
        self.minibatch[:, 0].add_(sums)
        self.minibatch[:, 1].add_(counts)

    def finish_actor(self, *, retained):
        self.attempted.add_(self.minibatch)
        if retained:
            self.retained.add_(self.minibatch)

    def reject_actor_update(self):
        self.retained.zero_()

    @torch.no_grad()
    def observe_critic(self, loss, count, value_coefficient):
        self.critic[:, 0].add_(torch.stack((loss.detach(), loss.detach() * value_coefficient)) * count)
        self.critic[:, 1].add_(count)

    def summary(self):
        arrays = torch.cat((self.attempted, self.retained, self.critic)).detach().cpu().tolist()
        width = len(self._ACTOR_NAMES)

        def report(names, rows):
            return {name: {'mean': total / count if count and math.isfinite(total) else None,
                           'sample_uses': int(count)}
                    for name, (total, count) in zip(names, rows)}

        return {
            'scope': 'pre-step losses weighted by repeated endpoint uses; auxiliary means use valid label rows',
            'retained_scope': 'actor transactions surviving minibatch and full-rollout rollback; not a final-policy re-evaluation',
            'actor_attempted': report(self._ACTOR_NAMES, arrays[:width]),
            'actor_retained': report(self._ACTOR_NAMES, arrays[width:2 * width]),
            'critic_attempted': report(('clipped_value_mse', 'weighted_objective'), arrays[2 * width:]),
        }


class StructuredPPOTrainer:
    """One actor backward/step per minibatch; the critic has its own optimizer.

    Every executed step and endpoint use is charged even after a rollback.
    Metrics leave the device at update end, apart from the necessary KL admission
    decisions. Probe and rollback time is included in optimization wall time.
    """

    MIN_LEARNING_RATE = 1e-5
    MAX_LEARNING_RATE = 1e-2

    def __init__(self, model, config=StructuredPPOConfig(), *, loss_telemetry=True):
        if type(loss_telemetry) is not bool:
            raise TypeError('loss_telemetry must be an explicit bool')
        self.model, self.config = model, config
        self._loss_telemetry = loss_telemetry
        self._base_learning_rate = float(config.learning_rate)
        self._min_learning_rate = min(self._base_learning_rate, self.MIN_LEARNING_RATE)
        self._max_learning_rate = self.MAX_LEARNING_RATE
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.learning_rate)
        self.update_index = 0
        self.consumed = dict(fresh_endpoints=0, actor_optimizer_steps=0, actor_endpoint_uses=0,
                             critic_optimizer_steps=0, critic_endpoint_uses=0)
        self._shared_parameters = [p for name, p in model.actor.policy.named_parameters()
                                   if not name.startswith(('state_head.', 'prediction_head.',
                                                           'leg_decoder.', 'wheel_decoder.',
                                                           'action_decoder.'))]

    def _learning_rate(self):
        return float(self.actor_optimizer.param_groups[0]['lr'])

    def _set_learning_rate(self, value):
        value = min(max(float(value), self._min_learning_rate), self._max_learning_rate)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = value
        return value

    def _adapt_learning_rate(self, full_kl, *, rejected):
        """Bounded KL feedback close to the adaptive RSL schedules.

        A finite rejection is a direct signal that the current step is too
        large; otherwise the next update uses a ratio of desired to observed
        KL. Nonfinite auxiliary/KL failures also reduce the step rather than
        freezing forever at one rate.
        """
        current = self._learning_rate()
        if rejected or not torch.isfinite(full_kl):
            factor = 1 / 1.5
        elif full_kl > self.config.target_kl * 2.0:
            factor = 1 / 1.5
        elif 0.0 < full_kl < self.config.target_kl * 0.5:
            factor = 1.5
        else:
            factor = 1.0
        return self._set_learning_rate(current * factor)

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
                tolerance = 1e-4 if name == 'logp' else 1e-5
                if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
                    delta = (actual - expected).abs()
                    index = int(delta.argmax().item())
                    print(f'BEHAVIOR_MISMATCH update={self.update_index} field={name} '
                          f'max_abs={float(delta.flatten()[index].item()):.9g} '
                          f'index={index} actual={float(actual.flatten()[index].item()):.9g} '
                          f'expected={float(expected.flatten()[index].item()):.9g}', flush=True)
                    raise ValueError(f'behavior {name} mismatch before any optimizer step')

    def _losses(self, batch, endpoints, advantages, *, normalization=None, telemetry=None,
                return_kl=False):
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
        if telemetry is not None:
            telemetry.observe_actor(terms, evaluation.entropy, auxiliary, loss,
                                    batch.masks.state_auxiliary[endpoints],
                                    batch.masks.next_auxiliary[endpoints], normalization)
        if return_kl:
            log_std = actor.log_std.clamp(-4.0, 1.0).expand_as(mean)
            kl = self.diagonal_kl(batch.old_mean[endpoints], batch.old_log_std[endpoints],
                                  mean.detach(), log_std.detach()).sum()
            return loss, components, kl
        return loss, components

    def _actor_backward(self, batch, endpoints, advantages, *, telemetry=None, return_kl=False):
        """分块累积同一 minibatch 的梯度；每项仍按整批有效样本数归一化。"""
        if endpoints.numel() <= self.config.training_chunk:
            result = self._losses(batch, endpoints, advantages, telemetry=telemetry,
                                  return_kl=return_kl)
            result[0].backward()
            return result[2] / max(endpoints.numel(), 1) if return_kl else None
        normalization = (endpoints.numel(), batch.masks.state_auxiliary[endpoints].sum(),
                         batch.masks.next_auxiliary[endpoints].sum())
        kl_total = endpoints.new_zeros((), dtype=torch.float32)
        for part in endpoints.split(self.config.training_chunk):
            result = self._losses(batch, part, advantages, normalization=normalization,
                                  telemetry=telemetry, return_kl=return_kl)
            result[0].backward()
            if return_kl:
                kl_total = kl_total + result[2]
        return kl_total / max(endpoints.numel(), 1) if return_kl else None

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

    def _open_steps(self, batch, actor_endpoints, critic_endpoints, advantages, metrics,
                    loss_telemetry, norms, critic_norms, timing):
        """Reference-style update: align one actor/critic logical minibatch.

        The KL is taken from the actor loss forward, the shared rate is
        scheduled once, and both optimizer steps consume that same rate.
        """
        for _ in range(self.config.epochs):
            actor_order = actor_endpoints[torch.randperm(actor_endpoints.numel(),
                                                         device=batch.windows.device)]
            critic_order = critic_endpoints[torch.randperm(critic_endpoints.numel(),
                                                           device=batch.windows.device)]
            actor_parts = torch.tensor_split(actor_order, self.config.minibatches)
            critic_parts = torch.tensor_split(critic_order, self.config.minibatches)
            for actor_part, critic_part in zip(actor_parts, critic_parts):
                actor_started = time.perf_counter()
                kl = None
                if actor_part.numel():
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    if loss_telemetry is not None:
                        loss_telemetry.begin_actor()
                    kl = self._actor_backward(batch, actor_part, advantages,
                                              telemetry=loss_telemetry, return_kl=True)
                    norm = nn.utils.clip_grad_norm_(
                        self.model.actor.parameters(), self.config.max_gradient_norm)
                    norms.append(norm.detach())
                    if not bool(torch.isfinite(norm).item()) or not bool(torch.isfinite(kl).item()):
                        raise FloatingPointError(
                            'actor gradient or KL became nonfinite; update aborted')
                critic_started = time.perf_counter()
                if critic_part.numel():
                    value = self.model.critic(batch.critic[critic_part])
                    clipped = batch.old_value[critic_part] + (
                        value - batch.old_value[critic_part]).clamp(
                        -self.config.value_clip, self.config.value_clip)
                    loss = torch.maximum((value - batch.returns[critic_part]).square(),
                                         (clipped - batch.returns[critic_part]).square()).mean()
                    if loss_telemetry is not None:
                        loss_telemetry.observe_critic(
                            loss, critic_part.numel(), self.config.value_coefficient)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coefficient * loss).backward()
                    critic_norm = nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.config.max_gradient_norm)
                    if not bool(torch.isfinite(critic_norm).item()):
                        raise FloatingPointError('critic gradient became nonfinite before optimizer step')
                    critic_norms.append(critic_norm.detach())
                # One scheduler result for the corresponding logical minibatch.
                if kl is not None:
                    self._adapt_learning_rate(kl, rejected=False)
                if actor_part.numel():
                    self.actor_optimizer.step()
                    metrics['actor_optimizer_steps'] += 1
                    metrics['actor_endpoint_uses'] += actor_part.numel()
                    metrics['retained_actor_steps'] += 1
                    self.consumed['actor_optimizer_steps'] += 1
                    self.consumed['actor_endpoint_uses'] += actor_part.numel()
                    metrics['actor_learning_rate_after_update'] = self._learning_rate()
                    if loss_telemetry is not None:
                        loss_telemetry.finish_actor(retained=True)
                if critic_part.numel():
                    self.critic_optimizer.step()
                    metrics['critic_optimizer_steps'] += 1
                    metrics['critic_endpoint_uses'] += critic_part.numel()
                    self.consumed['critic_optimizer_steps'] += 1
                    self.consumed['critic_endpoint_uses'] += critic_part.numel()
                timing['actor_backward_step_s'] += critic_started - actor_started
                timing['probe_guard_s'] += time.perf_counter() - critic_started

    def update(self, batch: StructuredBatch):
        started = time.perf_counter()
        batch.validate(self.model)
        batch = batch.owned()
        actor_endpoints = batch.masks.actor.nonzero().flatten()
        critic_endpoints = batch.masks.critic.nonzero().flatten()
        validated_at = time.perf_counter()
        if self.config.validate_behavior:
            self._validate_behavior(batch, actor_endpoints)
        behavior_at = time.perf_counter()
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
                       rolled_back_actor_steps=0, actor_stop_reason=None, whole_actor_update_rolled_back=False,
                       actor_learning_rate=self._learning_rate(),
                       actor_reject_nonfinite_gradient_steps=0,
                       actor_reject_nonfinite_kl_steps=0,
                       actor_reject_kl_excess_steps=0,
                       actor_reject_full_rollout_kl_steps=0,
                       actor_rejected_kl_last=None)
        tensors = {}
        loss_telemetry = _LossTelemetry(batch.old_log_prob) if self._loss_telemetry else None
        norms = []
        critic_norms = []
        timing = {'batch_validation_copy_s': validated_at - started,
                  'behavior_validation_s': behavior_at - validated_at,
                  'update_setup_snapshot_s': time.perf_counter() - behavior_at,
                  'minibatch_snapshot_s': 0., 'actor_backward_step_s': 0., 'probe_guard_s': 0.}
        try:
            if actor_endpoints.numel() and self.update_index % self.config.diagnostics_interval == 0:
                tensors.update(self._gradient_diagnostics(batch, probe, advantages))
            open_steps = not self.config.kl_early_stop
            critic_loop_endpoints = critic_endpoints
            if open_steps:
                self._open_steps(batch, actor_endpoints, critic_endpoints, advantages, metrics,
                                 loss_telemetry, norms, critic_norms, timing)
                actor_finished = True
                critic_loop_endpoints = actor_endpoints.new_empty((0,))
            else:
                actor_finished = actor_endpoints.numel() == 0
            for _ in range(self.config.epochs):
                if actor_finished:
                    break
                order = actor_endpoints[torch.randperm(actor_endpoints.numel(), device=batch.windows.device)]
                for endpoints in torch.tensor_split(order, self.config.minibatches):
                    if not endpoints.numel():
                        continue
                    snapshot_at = time.perf_counter()
                    transaction = (_ParameterTransaction(self.model.actor, self.actor_optimizer)
                                   if self.config.kl_early_stop else None)
                    backward_at = time.perf_counter()
                    timing['minibatch_snapshot_s'] += backward_at - snapshot_at
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    if loss_telemetry is not None:
                        loss_telemetry.begin_actor()
                    self._actor_backward(batch, endpoints, advantages, telemetry=loss_telemetry)
                    norm = nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.config.max_gradient_norm)
                    norms.append(norm.detach())
                    guard_at = time.perf_counter()
                    if not self.config.kl_early_stop:
                        # Reference PPO adjusts LR inside each minibatch before
                        # its optimizer step, then continues with all batches.
                        kl = self._kl(batch, endpoints)
                        if not bool(torch.isfinite(norm).item()) or not bool(torch.isfinite(kl).item()):
                            raise FloatingPointError(
                                'actor gradient or KL became nonfinite; update aborted')
                        self._adapt_learning_rate(kl, rejected=False)
                        self.actor_optimizer.step()
                        metrics['actor_optimizer_steps'] += 1
                        metrics['actor_endpoint_uses'] += endpoints.numel()
                        self.consumed['actor_optimizer_steps'] += 1
                        self.consumed['actor_endpoint_uses'] += endpoints.numel()
                        metrics['retained_actor_steps'] += 1
                        metrics['actor_learning_rate_after_update'] = self._learning_rate()
                        if loss_telemetry is not None:
                            loss_telemetry.finish_actor(retained=True)
                    else:
                        # Optional compatibility guard; formal training disables it
                        # to follow the reference PPO schedule (all minibatches).
                        self.actor_optimizer.step()
                        metrics['actor_optimizer_steps'] += 1
                        metrics['actor_endpoint_uses'] += endpoints.numel()
                        self.consumed['actor_optimizer_steps'] += 1
                        self.consumed['actor_endpoint_uses'] += endpoints.numel()
                        kl = self._kl(batch, probe)
                        finite_norm = bool(torch.isfinite(norm).item())
                        finite_kl = bool(torch.isfinite(kl).item())
                        bad = (not finite_norm or not finite_kl
                               or bool((kl > self.config.reject_kl).item()))
                        decision = torch.where(kl.new_tensor(bad, dtype=torch.bool), kl.new_tensor(2),
                                               (kl > self.config.target_kl).to(kl.dtype))
                        status = int(decision.item())
                        if status == 2:
                            transaction.restore()
                            metrics['rolled_back_actor_steps'] += 1
                            if not finite_norm:
                                metrics['actor_reject_nonfinite_gradient_steps'] += 1
                            elif not finite_kl:
                                metrics['actor_reject_nonfinite_kl_steps'] += 1
                            else:
                                metrics['actor_reject_kl_excess_steps'] += 1
                                metrics['actor_rejected_kl_last'] = float(kl.detach().cpu())
                            metrics['actor_stop_reason'] = 'rejected_actor_and_auxiliary_transaction'
                        else:
                            metrics['retained_actor_steps'] += 1
                        if loss_telemetry is not None:
                            loss_telemetry.finish_actor(retained=status != 2)
                        if status:
                            actor_finished = True
                            if status == 1:
                                metrics['actor_stop_reason'] = 'target_kl_reached'
                            timing['probe_guard_s'] += time.perf_counter() - guard_at
                            break
                    timing['probe_guard_s'] += time.perf_counter() - guard_at
                    timing['actor_backward_step_s'] += time.perf_counter() - backward_at
            full_guard_at = time.perf_counter()
            full_kl = self._kl(batch, actor_endpoints)
            learning_kl = full_kl
            full_kl_finite = bool(torch.isfinite(full_kl).item())
            if self.config.kl_early_stop:
                full_rejected = (not full_kl_finite
                                 or bool((full_kl > self.config.reject_kl).item()))
                if full_rejected:
                    tensors['rejected_full_rollout_kl'] = full_kl
                    metrics['actor_reject_full_rollout_kl_steps'] += 1
                    if full_kl_finite:
                        metrics['actor_rejected_kl_last'] = float(full_kl.detach().cpu())
                    actor_start.restore()
                    metrics['rolled_back_actor_steps'] += metrics['retained_actor_steps']
                    metrics['retained_actor_steps'] = 0
                    metrics['whole_actor_update_rolled_back'] = True
                    metrics['actor_stop_reason'] = 'rejected_full_rollout_kl'
                    if loss_telemetry is not None:
                        loss_telemetry.reject_actor_update()
                    full_kl = self._kl(batch, actor_endpoints)
            elif not full_kl_finite:
                raise FloatingPointError('full-rollout KL became nonfinite; update aborted')
            tensors['full_rollout_kl_after_guard'] = full_kl
            actor_finished_at = time.perf_counter()
            timing['full_rollout_guard_s'] = actor_finished_at - full_guard_at
            for _ in range(self.config.epochs):
                order = critic_loop_endpoints[torch.randperm(
                    critic_loop_endpoints.numel(), device=batch.windows.device)]
                for endpoints in torch.tensor_split(order, self.config.minibatches):
                    if not endpoints.numel():
                        continue
                    value = self.model.critic(batch.critic[endpoints])
                    clipped = batch.old_value[endpoints] + (value - batch.old_value[endpoints]).clamp(
                        -self.config.value_clip, self.config.value_clip)
                    loss = torch.maximum((value - batch.returns[endpoints]).square(),
                                         (clipped - batch.returns[endpoints]).square()).mean()
                    if loss_telemetry is not None:
                        loss_telemetry.observe_critic(loss, endpoints.numel(), self.config.value_coefficient)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coefficient * loss).backward()
                    critic_norm = nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), self.config.max_gradient_norm)
                    if not bool(torch.isfinite(critic_norm).item()):
                        raise FloatingPointError('critic gradient became nonfinite before optimizer step')
                    critic_norms.append(critic_norm.detach())
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
            if self.config.kl_early_stop:
                rejected = (metrics['rolled_back_actor_steps'] > 0
                            or metrics['whole_actor_update_rolled_back'])
                if actor_endpoints.numel():
                    new_rate = self._adapt_learning_rate(learning_kl, rejected=rejected)
                else:
                    new_rate = self._learning_rate()
            else:
                # The reference PPO updates the rate inside each minibatch;
                # no second end-of-update adjustment.
                new_rate = self._learning_rate()
            metrics['actor_learning_rate_after_update'] = new_rate
            metrics['learning_rate_scale'] = new_rate / self._base_learning_rate
            metrics.update({name: (value if math.isfinite(value) else None)
                            for name, value in ((name, float(value.detach())) for name, value in tensors.items())})
            if loss_telemetry is not None:
                summary_started = time.perf_counter()
                metrics['loss_statistics'] = loss_telemetry.summary()
                metrics['loss_telemetry_summary_elapsed_s'] = time.perf_counter() - summary_started
            metrics['actor_and_guard_elapsed_s'] = actor_finished_at - started
            metrics['critic_elapsed_s'] = time.perf_counter() - actor_finished_at
            metrics['total_optimization_elapsed_s'] = time.perf_counter() - started
            metrics['cumulative_consumed'] = dict(self.consumed)
            metrics['host_wall_s'] = timing
            return metrics
        except BaseException as error:
            # A failed update cannot leave an uncheckpointable half-transaction.
            # Consumption/RNG advance is retained and must be recorded by the caller.
            # A lost CUDA context makes state-dict copies fail again; preserve the
            # original accelerator error instead of masking it with a restore error.
            if not _is_cuda_fatal(error):
                for transaction in (actor_start, critic_start):
                    try:
                        transaction.restore()
                    except BaseException as restore_error:
                        error.add_note(
                            f'{type(restore_error).__name__} while restoring a failed update: {restore_error}')
            raise

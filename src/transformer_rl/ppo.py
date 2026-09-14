"""Self-contained PyTorch PPO over complete, immutable rollout endpoints."""
from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributions import Normal

from .config import PPOConfig
from .storage import PPOBatch
from .types import PolicyEvaluation

if TYPE_CHECKING:
    from .model import ActorCritic


class PPOTrainer:
    """Clipped PPO with an externally checkpointable Adam optimizer.

    Loss/entropy/KL/clip-fraction metrics are sample-weighted over minibatches
    that completed an optimizer step, evaluated just before that step. ``kl``
    is the analytic diagonal Gaussian KL(old || new), summed over actions.
    ``grad_norm`` is the step-mean *pre-clipping* norm. ``sample_count`` counts
    endpoint uses, including repeated uses across epochs. Rejected KL batches
    are excluded from these aggregates and reported through ``stop_kl``.
    ``auxiliary_loss`` is the unscaled MSE over configured critic target columns,
    sample-weighted like other losses; it is zero when ``auxiliary_coef`` is zero.
    Optional diagnostics compare the entire rollout to its stored behavior moments,
    after the first optimizer step and after the update (including KL early stops).
    KL terms sum over actions then average endpoints; moment means and change RMS
    average all endpoint/action entries. Normalized change uses the old std.
    """

    def __init__(self, model: ActorCritic, config: PPOConfig):
        self.model = model
        self.config = config
        self.auxiliary_indices = ()
        if config.auxiliary_coef > 0:
            self.auxiliary_indices = getattr(getattr(model, "config", None), "auxiliary_indices", ())
            if not self.auxiliary_indices:
                raise ValueError("positive auxiliary_coef requires explicit model.config.auxiliary_indices")
            if not callable(getattr(model.actor, "predict_auxiliary", None)):
                raise ValueError("positive auxiliary_coef requires an actor auxiliary prediction head")
        self.model.eval()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains nonfinite values")

    @classmethod
    def _validate_evaluation(cls, evaluation: PolicyEvaluation, batch: PPOBatch) -> None:
        for name, expected in (("log_prob", (len(batch),)), ("entropy", (len(batch),)),
                               ("mean", batch.raw_action.shape), ("std", batch.raw_action.shape)):
            tensor = getattr(evaluation, name)
            if tensor.shape != expected:
                raise ValueError(f"policy {name} must have shape {tuple(expected)}")
            if tensor.device != batch.raw_action.device or not tensor.is_floating_point():
                raise ValueError(f"policy {name} must be floating point on the batch device")
            cls._require_finite(f"policy {name}", tensor)
        if not (evaluation.std > 0).all():
            raise ValueError("policy std must be strictly positive")

    @torch.no_grad()
    def _check_old_log_prob(self, batch: PPOBatch, chunks: int) -> None:
        # Check every endpoint before *any* optimizer step, including remainder samples.
        for indices in torch.tensor_split(torch.arange(len(batch), device=batch.raw_action.device), chunks):
            minibatch = batch.index(indices)
            evaluation = self.model.actor.evaluate(minibatch.history, minibatch.raw_action)
            self._validate_evaluation(evaluation, minibatch)
            # Compare distributions before checking each density against its own moments.
            # Cross-batch-shape mean roundoff can be amplified by Gaussian log_prob.
            for name in ("mean", "std"):
                current = getattr(evaluation, name)
                old = getattr(minibatch, f"old_{name}")
                if not torch.allclose(current, old, rtol=1e-5, atol=1e-5):
                    error = (current - old).abs().max().item()
                    raise ValueError(
                        f"old_{name} mismatch before first optimizer step (max absolute error {error:.6g}); "
                        "check behavior policy statistics, weights, full history snapshots and raw_action"
                    )
            for name, mean, std, log_prob in (
                ("old_log_prob", minibatch.old_mean, minibatch.old_std, minibatch.old_log_prob),
                ("current_log_prob", evaluation.mean, evaluation.std, evaluation.log_prob),
            ):
                # Reproduce the original density arithmetic, rather than an FP64 reference.
                distribution = Normal(mean.to(log_prob.dtype), std.to(log_prob.dtype), validate_args=False)
                reconstructed = distribution.log_prob(minibatch.raw_action.to(log_prob.dtype)).sum(-1)
                self._require_finite(f"reconstructed {name}", reconstructed)
                if not torch.allclose(reconstructed, log_prob, rtol=1e-5, atol=1e-5):
                    error = (reconstructed - log_prob).abs().max().item()
                    raise ValueError(
                        f"{name} mismatch before first optimizer step (max absolute error {error:.6g}); "
                        "log_prob must match its own Gaussian mean/std and raw_action"
                    )

    @staticmethod
    def _detached_batch(batch: PPOBatch) -> PPOBatch:
        # Even caller-created batches must never backpropagate into rollout targets.
        history = type(batch.history)(**{
            f.name: getattr(batch.history, f.name).detach() for f in fields(batch.history)
        })
        return replace(batch, history=history, **{
            f.name: getattr(batch, f.name).detach() for f in fields(batch) if f.name != "history"
        })

    def _auxiliary_loss(self, batch: PPOBatch) -> torch.Tensor:
        # Privileged critic columns are supervision only, never actor inputs.
        targets = batch.critic[:, self.auxiliary_indices].detach()
        prediction = self.model.actor.predict_auxiliary(batch.history)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != targets.shape:
            raise ValueError(f"auxiliary prediction must have shape {tuple(targets.shape)}")
        if prediction.device != targets.device or not prediction.is_floating_point():
            raise ValueError("auxiliary prediction must be floating point on the batch device")
        self._require_finite("auxiliary prediction", prediction)
        loss = (prediction - targets).square().mean()
        self._require_finite("auxiliary loss", loss)
        return loss

    @torch.no_grad()
    def _initial_diagnostics(self, batch: PPOBatch, chunks: int) -> dict[str, float]:
        # Stored behavior moments are the fixed reference, not a minibatch average.
        totals = torch.zeros(2, dtype=torch.float64, device=batch.raw_action.device)
        std_min = torch.full((), float("inf"), dtype=torch.float64, device=totals.device)
        std_max = torch.zeros_like(std_min)
        for indices in torch.tensor_split(torch.arange(len(batch), device=totals.device), chunks):
            old_mean = batch.old_mean[indices].double()
            old_std = batch.old_std[indices].double()
            totals += torch.stack((old_mean.abs().sum(), old_std.sum()))
            std_min = torch.minimum(std_min, old_std.min())
            std_max = torch.maximum(std_max, old_std.max())
        mean_abs, std_mean = (totals / batch.old_mean.numel()).tolist()
        return {"initial_mean_abs": mean_abs, "initial_std_mean": std_mean,
                "initial_std_min": std_min.item(), "initial_std_max": std_max.item()}

    @torch.no_grad()
    def _distribution_diagnostics(self, batch: PPOBatch, chunks: int) -> dict[str, float]:
        # eval-mode evaluation does not sample actions or change optimizer/gradient state.
        totals = torch.zeros(5, dtype=torch.float64, device=batch.raw_action.device)
        for indices in torch.tensor_split(torch.arange(len(batch), device=totals.device), chunks):
            minibatch = batch.index(indices)
            evaluation = self.model.actor.evaluate(minibatch.history, minibatch.raw_action)
            self._validate_evaluation(evaluation, minibatch)
            old_std, new_std = minibatch.old_std.double(), evaluation.std.double()
            difference = evaluation.mean.double() - minibatch.old_mean.double()
            mean_kl = 0.5 * (difference / new_std).square()
            log_ratio = old_std.log() - new_std.log()
            # expm1 avoids subtracting nearly equal variances for small std changes.
            std_kl = 0.5 * torch.expm1(2.0 * log_ratio) - log_ratio
            totals += torch.stack((mean_kl.sum(), std_kl.sum(), difference.square().sum(),
                                   (difference / old_std).square().sum(), new_std.sum()))
        self._require_finite("distribution diagnostics", totals)
        mean_kl, std_kl = (totals[:2] / len(batch)).tolist()
        mean_rms, normalized_rms = (totals[2:4] / batch.old_mean.numel()).sqrt().tolist()
        return {"kl": mean_kl + std_kl, "mean_kl": mean_kl, "std_kl": std_kl,
                "mean_change_rms": mean_rms, "normalized_mean_change_rms": normalized_rms,
                "std_mean": (totals[4] / batch.old_mean.numel()).item()}

    @torch.enable_grad()
    def update(self, batch: PPOBatch, *, diagnostics: bool = False) -> dict[str, float | int | bool | None]:
        self.model.eval()  # Deterministic dropout behavior; this does not disable autograd.
        self.optimizer.zero_grad(set_to_none=True)
        batch.validate()
        batch = self._detached_batch(batch)
        chunks = min(self.config.num_minibatches, len(batch))
        self._check_old_log_prob(batch, chunks)
        diagnostic_metrics = {}
        first_step_fields = ("kl", "mean_kl", "std_kl", "mean_change_rms",
                             "normalized_mean_change_rms")
        if diagnostics:
            diagnostic_metrics = self._initial_diagnostics(batch, chunks)
            diagnostic_metrics.update({f"first_step_{name}": None for name in first_step_fields})
        if self.config.normalize_advantage:
            advantages = batch.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            self._require_finite("normalized advantages", advantages)
            batch = replace(batch, advantages=advantages)

        totals = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0,
                      clip_fraction=0.0, loss=0.0, auxiliary_loss=0.0)
        grad_norm_sum = 0.0
        optimizer_steps = sample_count = 0
        early_stopped = False
        stop_kl = 0.0
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        for _ in range(self.config.epochs):
            order = torch.randperm(len(batch), device=batch.raw_action.device)
            for indices in torch.tensor_split(order, chunks):
                minibatch = batch.index(indices)
                evaluation = self.model.actor.evaluate(minibatch.history, minibatch.raw_action)
                self._validate_evaluation(evaluation, minibatch)
                # Promote low precision inputs for the KL; never add eps that biases KL at equality.
                kl_dtype = torch.promote_types(evaluation.std.dtype, minibatch.old_std.dtype)
                if kl_dtype in (torch.float16, torch.bfloat16):
                    kl_dtype = torch.float32
                new_std = evaluation.std.detach().to(kl_dtype)
                old_std = minibatch.old_std.to(kl_dtype)
                mean_difference = minibatch.old_mean.to(kl_dtype) - evaluation.mean.detach().to(kl_dtype)
                kl_per_sample = (
                    new_std.log() - old_std.log()
                    + 0.5 * ((old_std / new_std).square()
                             + (mean_difference / new_std).square() - 1.0)
                ).sum(dim=-1)
                self._require_finite("KL", kl_per_sample)
                kl = kl_per_sample.mean()
                if kl.item() > self.config.target_kl:
                    early_stopped = True
                    stop_kl = kl.item()
                    break

                value = self.model.critic(minibatch.critic)
                if value.shape != (len(minibatch),):
                    raise ValueError("critic output must have shape [B]")
                if value.device != minibatch.critic.device or not value.is_floating_point():
                    raise ValueError("critic output must be floating point on the batch device")
                self._require_finite("critic value", value)
                ratio = (evaluation.log_prob - minibatch.old_log_prob).exp()
                self._require_finite("policy ratio", ratio)
                clipped_ratio = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                actor_loss = -torch.minimum(
                    ratio * minibatch.advantages, clipped_ratio * minibatch.advantages,
                ).mean()
                clipped_value = minibatch.old_value + (value - minibatch.old_value).clamp(
                    -self.config.value_clip, self.config.value_clip,
                )
                # The 1/2 factor is part of the value-loss contract, before value_coef.
                value_loss = 0.5 * torch.maximum(
                    (value - minibatch.returns).square(),
                    (clipped_value - minibatch.returns).square(),
                ).mean()
                entropy = evaluation.entropy.mean()
                loss = (actor_loss + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy)
                auxiliary_loss = loss.new_zeros(())
                if self.config.auxiliary_coef > 0:
                    auxiliary_loss = self._auxiliary_loss(minibatch)
                    loss = loss + self.config.auxiliary_coef * auxiliary_loss
                for name, component in (("actor loss", actor_loss), ("value loss", value_loss),
                                        ("entropy", entropy), ("loss", loss)):
                    self._require_finite(name, component)
                clip_fraction = (
                    (ratio < 1.0 - self.config.clip_ratio) | (ratio > 1.0 + self.config.clip_ratio)
                ).float().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Validate all gradients before clipping or allowing Adam to mutate parameters/state.
                for parameter in parameters:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("nonfinite gradient; optimizer step aborted")
                try:
                    grad_norm = nn.utils.clip_grad_norm_(
                        parameters, self.config.max_grad_norm, error_if_nonfinite=True,
                    )
                except RuntimeError:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise
                self.optimizer.step()
                if diagnostics and optimizer_steps == 0:
                    first_step = self._distribution_diagnostics(batch, chunks)
                    diagnostic_metrics.update({f"first_step_{name}": first_step[name]
                                               for name in first_step_fields})
                count = len(minibatch)
                for name, metric in dict(actor_loss=actor_loss, value_loss=value_loss,
                                         entropy=entropy, kl=kl, clip_fraction=clip_fraction,
                                         loss=loss, auxiliary_loss=auxiliary_loss).items():
                    totals[name] += metric.detach().item() * count
                grad_norm_sum += grad_norm.item()
                optimizer_steps += 1
                sample_count += count
            if early_stopped:
                break
        self.optimizer.zero_grad(set_to_none=True)
        if diagnostics:
            final = self._distribution_diagnostics(batch, chunks)
            diagnostic_metrics.update({f"final_{name}": final[name]
                                       for name in ("kl", "mean_kl", "std_kl", "mean_change_rms", "std_mean")})
        return {
            **{name: total / sample_count if sample_count else 0.0 for name, total in totals.items()},
            "auxiliary_coef": self.config.auxiliary_coef,
            "grad_norm": grad_norm_sum / optimizer_steps if optimizer_steps else 0.0,
            "optimizer_steps": optimizer_steps,
            "planned_optimizer_steps": self.config.epochs * chunks,
            "sample_count": sample_count,
            "early_stopped": early_stopped,
            "stop_kl": stop_kl,
            **diagnostic_metrics,
        }

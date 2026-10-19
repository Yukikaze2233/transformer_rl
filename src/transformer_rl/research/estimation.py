"""Detached state estimation updates with bounded policy change and rollback."""
from __future__ import annotations

import copy

import torch
from torch import nn

from .storage import EstimatorBatch


class EstimatorTrainer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.parameters = model.estimator_parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.estimator_learning_rate)

    def validate_batch(self, batch):
        if self.model.config.estimator_type == "context":
            if not isinstance(batch, EstimatorBatch):
                raise ValueError("context estimator requires owned next-observation targets")
            if batch.next_proprio.shape[1] != self.model.config.proprio_dim:
                raise ValueError("next_proprio width differs from configured proprio_dim")

    @torch.no_grad()
    def _means(self, batch):
        chunks = min(len(batch), self.config.estimator_minibatches)
        return torch.cat([
            self.model.actor(batch.history.index(indices)) for indices in
            torch.tensor_split(torch.arange(len(batch), device=batch.critic.device), chunks)
        ])

    def _snapshot(self):
        states = {"estimator": copy.deepcopy(self.model.actor.estimator.state_dict()),
                  "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                  "cpu_rng": torch.get_rng_state().clone()}
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            states["cuda_rng"] = torch.cuda.get_rng_state(device).clone()
        if self.model.config.estimator_type == "context":
            states["context"] = copy.deepcopy(self.model.context_objective.state_dict())
        return states

    def _restore(self, states):
        self.model.actor.estimator.load_state_dict(states["estimator"], strict=True)
        # Adam may retain same-device tensors from load_state_dict. A retry must
        # never mutate the saved transaction through those aliases.
        self.optimizer.load_state_dict(copy.deepcopy(states["optimizer"]))
        torch.set_rng_state(states["cpu_rng"])
        if "cuda_rng" in states:
            torch.cuda.set_rng_state(states["cuda_rng"], next(self.model.parameters()).device)
        if "context" in states:
            self.model.context_objective.load_state_dict(states["context"], strict=True)
        self.optimizer.zero_grad(set_to_none=True)

    def update(self, batch):
        self.validate_batch(batch)
        actor, cfg = self.model.actor, self.model.config
        chunks = min(len(batch), self.config.estimator_minibatches)
        before = self._means(batch)
        with torch.no_grad():
            std = actor.log_std.exp().detach().clone()
        snapshot = self._snapshot()
        attempts = attempted_steps = 0
        accepted_steps = 0
        final_kl = last_attempt_kl = 0.0
        state_total = context_total = sample_count = pair_count = 0
        try:
            for factor in (1.0, 0.5, 0.25):
                if attempts:
                    self._restore(snapshot)
                attempts += 1
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.estimator_learning_rate * factor
                state_total = context_total = sample_count = pair_count = 0
                steps = 0
                for _ in range(self.config.estimator_epochs):
                    order = torch.randperm(len(batch), device=batch.critic.device)
                    for indices in torch.tensor_split(order, chunks):
                        part = batch.index(indices)
                        prediction, latent = actor.estimate(part.history)
                        target = (part.critic[:, cfg.state_indices] - actor.state_offset) / actor.state_scale
                        state_loss = (prediction - target.detach()).square().mean()
                        context_loss = state_loss.new_zeros(())
                        valid_count = 0
                        if cfg.estimator_type == "context" and part.next_valid.any():
                            valid_count = int(part.next_valid.sum().item())
                            context_loss = self.model.context_objective(
                                latent[part.next_valid], part.next_proprio[part.next_valid].detach(),
                            )
                        loss = state_loss + self.config.estimator_context_coef * context_loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite estimator loss")
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        for parameter in self.parameters:
                            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError("nonfinite estimator gradient")
                        nn.utils.clip_grad_norm_(self.parameters, self.config.estimator_max_grad_norm,
                                                 error_if_nonfinite=True)
                        self.optimizer.step()
                        steps += 1
                        attempted_steps += 1
                        count = len(part)
                        state_total += float(state_loss.detach()) * count
                        context_total += float(context_loss.detach()) * valid_count
                        sample_count += count
                        pair_count += valid_count
                after = self._means(batch)
                last_attempt_kl = float((0.5 * ((after - before) / std).double().square().sum(-1)).mean())
                if not torch.isfinite(after).all() or not torch.isfinite(torch.tensor(last_attempt_kl)):
                    raise FloatingPointError("nonfinite estimator policy change")
                if last_attempt_kl <= self.config.estimator_target_kl:
                    accepted_steps, final_kl = steps, last_attempt_kl
                    break
            if not accepted_steps:
                self._restore(snapshot)
        except BaseException:
            self._restore(snapshot)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.estimator_learning_rate
        return {
            "estimator_state_loss": state_total / sample_count,
            "estimator_context_loss": context_total / pair_count if pair_count else 0.0,
            "estimator_context_pair_uses": pair_count,
            "estimator_loss_from_accepted_attempt": bool(accepted_steps),
            "estimator_attempts": attempts,
            "estimator_attempted_steps": attempted_steps,
            "estimator_accepted_steps": accepted_steps,
            "estimator_planned_steps": self.config.estimator_epochs * chunks,
            "estimator_policy_kl": final_kl,
            "estimator_last_attempt_kl": last_attempt_kl,
            "estimator_update_accepted": bool(accepted_steps),
        }

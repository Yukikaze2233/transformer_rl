"""Gaussian behavior contracts before any optimizer step, without environments."""
from copy import deepcopy
from dataclasses import replace
import math
import random

import pytest
import torch
from torch.distributions import Normal

from transformer_rl.ppo import PPOTrainer
from transformer_rl.types import PolicyEvaluation

from test_ppo import GaussianActor, SyntheticModel, config, make_batch


class BatchShapeGaussianActor(GaussianActor):
    """Controlled moment perturbation, not a claim about a particular CPU kernel."""

    def evaluate(self, history, raw_action):
        evaluation = super().evaluate(history, raw_action)
        mean = evaluation.mean
        if len(raw_action) != 5:
            mean = mean + 5e-7
        distribution = Normal(mean, evaluation.std, validate_args=False)
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), mean, evaluation.std)


class LegacyCheckTrainer(PPOTrainer):
    """Frozen pre-fix scan; all optimization code is inherited unchanged."""

    @torch.no_grad()
    def _check_old_log_prob(self, batch, chunks):
        for indices in torch.tensor_split(torch.arange(len(batch), device=batch.raw_action.device), chunks):
            minibatch = batch.index(indices)
            evaluation = self.model.actor.evaluate(minibatch.history, minibatch.raw_action)
            self._validate_evaluation(evaluation, minibatch)
            for name in ("log_prob", "mean", "std"):
                current = getattr(evaluation, name)
                old = getattr(minibatch, f"old_{name}")
                if not torch.allclose(current, old, rtol=1e-5, atol=1e-5):
                    error = (current - old).abs().max().item()
                    raise ValueError(
                        f"old_{name} mismatch before first optimizer step (max absolute error {error:.6g}); "
                        "check behavior policy statistics, weights, full history snapshots and raw_action"
                    )


def low_std_batch(dtype):
    model = SyntheticModel()
    model.actor = BatchShapeGaussianActor()
    model.to(dtype=dtype)
    with torch.no_grad():
        model.actor.log_std.fill_(math.log(0.1))
    # Two action dimensions, joint log density approximately -0.05.
    action = 0.1 * math.sqrt(2 * (-math.log(0.1 * math.sqrt(2 * math.pi)) + 0.025))
    batch = make_batch(model, raw_action=torch.full((5, 2), action, dtype=dtype),
                       advantages=torch.full((5,), 2.0), returns=torch.ones(5))
    return model, batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_moment_roundoff_with_amplified_cross_logp_is_legal(dtype):
    model, batch = low_std_batch(dtype)
    minibatch = batch.index(torch.arange(3))
    with torch.no_grad():
        current = model.actor.evaluate(minibatch.history, minibatch.raw_action)
    assert torch.allclose(current.mean, minibatch.old_mean, rtol=1e-5, atol=1e-5)
    assert torch.equal(current.std, minibatch.old_std)
    assert minibatch.old_log_prob.abs().max() < 0.1
    assert not torch.allclose(current.log_prob, minibatch.old_log_prob, rtol=1e-5, atol=1e-5)

    # Independently close the amplification formula in FP64 using the actual moments.
    old_mean, mean = minibatch.old_mean.double(), current.mean.double()
    std, action = minibatch.old_std.double(), minibatch.raw_action.double()
    delta = mean - old_mean
    predicted = (((action - old_mean) * delta - 0.5 * delta.square()) / std.square()).sum(-1)
    measured = (Normal(mean, std).log_prob(action) - Normal(old_mean, std).log_prob(action)).sum(-1)
    torch.testing.assert_close(measured, predicted, rtol=0, atol=1e-14)
    assert (0.5 * (delta / std).square()).sum(-1).max() < 3e-11

    with pytest.raises(ValueError, match="old_log_prob mismatch"):
        LegacyCheckTrainer(model, config())._check_old_log_prob(batch, chunks=2)
    trainer = PPOTrainer(model, config(num_minibatches=2))
    model.actor.calls.clear()
    rng = torch.get_rng_state()
    trainer._check_old_log_prob(batch, chunks=2)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [ids.tolist() for _, ids, _ in model.actor.calls] == [[0, 1, 2], [3, 4]]
    assert not any(enabled for enabled, _, _ in model.actor.calls)


def test_newly_legal_update_uses_original_stored_denominator_and_preserves_batch():
    model, batch = low_std_batch(torch.float32)
    batch = batch.index(4)
    before = deepcopy(batch)
    with torch.no_grad():
        evaluation = model.actor.evaluate(batch.history, batch.raw_action)
        ratio = (evaluation.log_prob - batch.old_log_prob).exp()
        expected_loss = -(ratio * batch.advantages).mean().item()
        expected_gradient = (-ratio[:, None] * batch.advantages[:, None]
                             * (batch.raw_action - evaluation.mean) / evaluation.std.square()).sum(0)
    assert not torch.equal(ratio, torch.ones_like(ratio))
    trainer = PPOTrainer(model, config(max_grad_norm=100.0))
    gradients = []
    model.actor.mean.register_hook(lambda gradient: gradients.append(gradient.clone()))
    metrics = trainer.update(batch)
    assert metrics["optimizer_steps"] == 1
    assert metrics["actor_loss"] == expected_loss
    assert metrics["actor_loss"] != -batch.advantages.mean().item()
    torch.testing.assert_close(gradients[0], expected_gradient)
    torch.testing.assert_close(vars(batch.history), vars(before.history), rtol=0, atol=0)
    torch.testing.assert_close({k: v for k, v in vars(batch).items() if k != "history"},
                               {k: v for k, v in vars(before).items() if k != "history"}, rtol=0, atol=0)


@pytest.mark.parametrize("corruption,message", [
    ("mean", "old_mean mismatch"),
    ("std", "old_std mismatch"),
    ("history", "old_mean mismatch"),
    ("stored_logp", "old_log_prob mismatch"),
    ("current_logp", "current_log_prob mismatch"),
    ("stored_issued", "old_log_prob mismatch"),
    ("current_issued", "current_log_prob mismatch"),
    ("both_issued", "old_log_prob mismatch"),
    ("raw_replaced_by_issued", "old_log_prob mismatch"),
])
def test_corruption_in_remainder_is_rejected_before_any_optimizer_step(monkeypatch, corruption, message):
    model = SyntheticModel()
    batch = make_batch(model, advantages=torch.ones(5))
    if corruption in ("mean", "std"):
        # Keep stored density internally consistent; the moment comparison must reject it first.
        getattr(batch, f"old_{corruption}")[-1, 0] += 0.01
        batch = replace(batch, old_log_prob=Normal(batch.old_mean, batch.old_std).log_prob(batch.raw_action).sum(-1))
    elif corruption == "history":
        batch.history.frames[-1, -1, 0] += 0.01
    elif corruption == "raw_replaced_by_issued":
        batch.raw_action[-1].copy_(batch.issued_action[-1])
    else:
        if corruption in ("stored_logp", "stored_issued", "both_issued"):
            wrong_action = batch.raw_action[-1] + 0.1 if corruption == "stored_logp" else batch.issued_action[-1]
            batch.old_log_prob[-1] = Normal(batch.old_mean[-1], batch.old_std[-1]).log_prob(wrong_action).sum(-1)
        if corruption in ("current_logp", "current_issued", "both_issued"):
            original_evaluate = model.actor.evaluate

            def wrong_action_density(history, raw_action):
                action = raw_action.clone()
                last = history.now == 4
                action[last] = action[last] + 0.1 if corruption == "current_logp" else action[last].clamp(-1, 1)
                # A real Gaussian density of the wrong action, with the correct current moments.
                return original_evaluate(history, action)

            monkeypatch.setattr(model.actor, "evaluate", wrong_action_density)
    trainer = PPOTrainer(model, config(num_minibatches=2))
    before = deepcopy(model.state_dict())
    rng, python_rng = torch.get_rng_state(), random.getstate()
    with pytest.raises(ValueError, match=f"{message} before first optimizer step"):
        trainer.update(batch)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [ids.tolist() for _, ids, _ in model.actor.calls] == [[0, 1, 2], [3, 4]]
    assert not any(enabled for enabled, _, _ in model.actor.calls)


@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_legacy_legal_updates_preserve_exact_gradients_adam_rng_and_evaluations(early_stop, diagnostics):
    model = SyntheticModel()
    cfg = config(epochs=3, num_minibatches=2, normalize_advantage=True, entropy_coef=0.03,
                 learning_rate=0.01, target_kl=1e-6 if early_stop else 10.0)
    legacy = LegacyCheckTrainer(model, cfg)
    fixed = PPOTrainer(deepcopy(model), cfg)
    gradients = [[], []]
    for trainer, recorded in zip((legacy, fixed), gradients, strict=True):
        for name, parameter in trainer.model.named_parameters():
            parameter.register_hook(lambda grad, name=name, recorded=recorded:
                                    recorded.append((name, grad.clone())))

    # The second update also compares already-populated Adam moments and step counters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        for _ in range(2):
            batch = make_batch(legacy.model, advantages=torch.tensor([-1.0, 0.0, 1.0, 2.0, 4.0]),
                               returns=torch.arange(5).float())
            fixed.model.actor.calls.clear()
            for recorded in gradients:
                recorded.clear()
            rng, python_rng = torch.get_rng_state(), random.getstate()
            expected = legacy.update(batch, diagnostics=diagnostics)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = fixed.update(batch, diagnostics=diagnostics)
            assert actual == expected
            assert actual["early_stopped"] == early_stop
            assert actual["optimizer_steps"] == (1 if early_stop else 6)
            assert gradients[0] and any(grad.count_nonzero() for _, grad in gradients[0])
            assert [name for name, _ in gradients[1]] == [name for name, _ in gradients[0]]
            torch.testing.assert_close([grad for _, grad in gradients[1]],
                                       [grad for _, grad in gradients[0]], rtol=0, atol=0)
            torch.testing.assert_close(fixed.model.state_dict(), legacy.model.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(fixed.optimizer.state_dict(), legacy.optimizer.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(fixed.model.actor.calls, legacy.model.actor.calls, rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert random.getstate() == python_rng

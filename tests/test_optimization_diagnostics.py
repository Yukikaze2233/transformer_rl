"""Full-rollout optimization diagnostics on synthetic gradients, without environments."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import random

import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from transformer_rl.config import ModelConfig
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.storage import PPOBatch
from transformer_rl.types import HistoryBatch, PolicyEvaluation

from test_ppo import SyntheticModel, config, make_batch


FIRST_FIELDS = {"first_step_kl", "first_step_mean_kl", "first_step_std_kl",
                "first_step_mean_change_rms", "first_step_normalized_mean_change_rms"}
DIAGNOSTIC_FIELDS = FIRST_FIELDS | {
    "initial_mean_abs", "initial_std_mean", "initial_std_min", "initial_std_max",
    "final_kl", "final_mean_kl", "final_std_kl", "final_mean_change_rms", "final_std_mean",
}


def assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_state_equal(a, b)
    else:
        assert actual == expected


def refresh_behavior(model, batch):
    with torch.no_grad():
        evaluation = model.actor.evaluate(batch.history, batch.raw_action)
        return replace(batch, old_mean=evaluation.mean.clone(), old_std=evaluation.std.clone(),
                       old_log_prob=evaluation.log_prob.clone(), old_value=model.critic(batch.critic).clone())


def test_first_step_is_post_optimizer_analytic_kl_and_early_stop():
    model = SyntheticModel()
    batch = make_batch(model, size=5, advantages=torch.ones(5))
    trainer = PPOTrainer(model, config(epochs=3, num_minibatches=2, target_kl=0.1, max_grad_norm=100.0))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = trainer.update(batch, diagnostics=True)
    # grad_mu=-2 and grad_log_std=-3 in both action dimensions: mu=0.2, std=exp(0.3).
    mean_kl = 0.2**2 * math.exp(-0.6)
    std_kl = 2 * (0.3 + 0.5 * math.exp(-0.6) - 0.5)
    assert metrics["kl"] == 0.0  # The original metric is pre-step, unlike the diagnostics.
    assert metrics["first_step_kl"] == pytest.approx(mean_kl + std_kl)
    assert metrics["first_step_mean_kl"] == pytest.approx(mean_kl)
    assert metrics["first_step_std_kl"] == pytest.approx(std_kl)
    assert metrics["first_step_mean_change_rms"] == pytest.approx(0.2)
    assert metrics["first_step_normalized_mean_change_rms"] == pytest.approx(0.2)
    for name in ("kl", "mean_kl", "std_kl", "mean_change_rms"):
        assert metrics[f"final_{name}"] == metrics[f"first_step_{name}"]
    assert metrics["final_std_mean"] == pytest.approx(math.exp(0.3))
    assert metrics["initial_mean_abs"] == 0.0
    assert [metrics[f"initial_std_{name}"] for name in ("mean", "min", "max")] == [1.0] * 3
    assert metrics["early_stopped"] and metrics["stop_kl"] > 0.1
    assert metrics["optimizer_steps"] == 1
    assert metrics["planned_optimizer_steps"] == 6
    assert metrics["sample_count"] == 3


def test_no_optimizer_step_reports_null_and_real_final_kl():
    model = SyntheticModel()
    with torch.no_grad():
        model.actor.log_std.fill_(math.log(1e-6))
    batch = make_batch(model, raw_action=torch.full((5, 2), 0.5e-6))
    # Within the existing absolute moment tolerance, yet significant relative to std.
    # The midpoint action has equal densities under old and current means.
    batch.old_mean.fill_(1e-6)
    trainer = PPOTrainer(model, config(num_minibatches=2, target_kl=0.1))
    before = deepcopy(model.state_dict())
    metrics = trainer.update(batch, diagnostics=True)
    assert metrics["optimizer_steps"] == 0 and metrics["planned_optimizer_steps"] == 2
    assert metrics["early_stopped"] and not trainer.optimizer.state
    assert all(metrics[name] is None for name in FIRST_FIELDS)
    assert metrics["final_kl"] == pytest.approx(1.0, rel=1e-5)
    assert metrics["final_mean_kl"] == metrics["final_kl"]
    assert metrics["final_std_kl"] == 0.0
    assert_state_equal(model.state_dict(), before)
    assert all(json.loads(json.dumps(metrics, allow_nan=False))[name] is None for name in FIRST_FIELDS)


class ConditionalActor(nn.Module):
    """Heterogeneous moments make incorrect equal-chunk weighting observable."""

    def __init__(self):
        super().__init__()
        self.mean_scale = nn.Parameter(torch.tensor([0.5, -0.2]))
        self.log_std = nn.Parameter(torch.tensor([-0.5, 0.3]))

    def evaluate(self, history, raw_action):
        feature = 1 + history.now.float()[:, None]
        mean = feature * self.mean_scale
        std = feature.sqrt() * self.log_std.exp()
        distribution = Normal(mean, std)
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), mean, std)


@pytest.mark.parametrize("chunks,size", [(2, 5), (8, 3), (1, 1)])
def test_full_batch_weighting_and_action_reductions(chunks, size):
    model = SyntheticModel()
    model.actor = ConditionalActor()
    # make_batch's call recorder is only needed by the legacy synthetic fixture.
    model.actor.calls = []
    batch = make_batch(model, size=size, advantages=torch.arange(1, size + 1).float())
    trainer = PPOTrainer(model, config(epochs=2, num_minibatches=chunks, learning_rate=0.01))
    first = []
    step = trainer.optimizer.step

    def capture_step():
        step()
        if not first:
            with torch.no_grad():
                first.append(model.actor.evaluate(batch.history, batch.raw_action))

    trainer.optimizer.step = capture_step
    metrics = trainer.update(batch, diagnostics=True)
    old_mean, old_std = batch.old_mean.double(), batch.old_std.double()
    assert metrics["initial_mean_abs"] == pytest.approx(old_mean.abs().mean().item())
    assert metrics["initial_std_mean"] == pytest.approx(old_std.mean().item())
    assert metrics["initial_std_min"] == old_std.min().item()
    assert metrics["initial_std_max"] == old_std.max().item()
    with torch.no_grad():
        final = model.actor.evaluate(batch.history, batch.raw_action)
    for prefix, evaluation in (("first_step", first[0]), ("final", final)):
        mean, std = evaluation.mean.double(), evaluation.std.double()
        kl = kl_divergence(Normal(old_mean, old_std), Normal(mean, std)).sum(-1).mean()
        mean_kl = ((mean - old_mean).square() / (2 * std.square())).sum(-1).mean()
        std_kl = kl_divergence(Normal(old_mean, old_std), Normal(old_mean, std)).sum(-1).mean()
        assert metrics[f"{prefix}_kl"] == pytest.approx(kl.item(), abs=1e-14)
        assert metrics[f"{prefix}_mean_kl"] == pytest.approx(mean_kl.item(), abs=1e-14)
        assert metrics[f"{prefix}_std_kl"] == pytest.approx(std_kl.item(), abs=1e-14)
        assert metrics[f"{prefix}_kl"] == metrics[f"{prefix}_mean_kl"] + metrics[f"{prefix}_std_kl"]
        assert metrics[f"{prefix}_mean_change_rms"] == pytest.approx((mean - old_mean).square().mean().sqrt().item())
    assert metrics["first_step_normalized_mean_change_rms"] == pytest.approx(
        ((first[0].mean.double() - old_mean) / old_std).square().mean().sqrt().item())
    assert metrics["final_std_mean"] == pytest.approx(final.std.double().mean().item())
    assert metrics["planned_optimizer_steps"] == metrics["optimizer_steps"] == 2 * min(chunks, size)
    assert metrics["final_kl"] > metrics["first_step_kl"]


@pytest.mark.parametrize("component", ["mean", "std", "neither"])
def test_isolated_components_and_tiny_std_roundoff(component):
    model = SyntheticModel()
    if component == "mean":
        model.actor.log_std.requires_grad_(False)
    else:
        model.actor.mean.requires_grad_(False)
    batch = make_batch(model, advantages=torch.ones(5) if component == "mean" else torch.zeros(5))
    trainer = PPOTrainer(model, config(entropy_coef=1.0 if component == "std" else 0.0,
                                       learning_rate=1e-7))
    metrics = trainer.update(batch, diagnostics=True)
    if component == "std":
        std = model.actor.log_std.exp().double()
        # This scale is small enough for the old float32 KL formula to lose precision.
        expected = (std.log() + 0.5 * (std.reciprocal().square() - 1)).sum().item()
        assert metrics["final_std_kl"] > 0
        assert metrics["final_std_kl"] == pytest.approx(expected, abs=1e-16)
        assert metrics["final_mean_kl"] == metrics["final_mean_change_rms"] == 0
    elif component == "mean":
        assert metrics["final_mean_kl"] > 0 and metrics["final_std_kl"] == 0
    else:
        assert metrics["first_step_kl"] == metrics["final_kl"] == 0
        assert metrics["first_step_mean_change_rms"] == 0
    assert metrics["optimizer_steps"] == 1


def real_batch(model):
    cfg = model.config
    valid = torch.tensor([[False, False, True], [False, True, True], [False, False, False],
                          [True, True, True], [False, True, True]])
    history = HistoryBatch(
        frames=torch.randn(5, 3, cfg.frame_dim),
        times=torch.arange(3, dtype=torch.float64).expand(5, -1).clone(), valid=valid,
        command=torch.randn(5, cfg.command_dim), now=torch.full((5,), 2.0, dtype=torch.float64),
    )
    history.frames[~valid] = float("nan")
    history.times[~valid] = float("inf")
    action, critic = torch.randn(5, cfg.action_dim), torch.randn(5, cfg.critic_dim)
    with torch.no_grad():
        evaluation = model.actor.evaluate(history, action)
        value = model.critic(critic)
    return PPOBatch(history=history, critic=critic, raw_action=action, issued_action=action.clamp(-1, 1),
                    old_log_prob=evaluation.log_prob, old_mean=evaluation.mean.clone(),
                    old_std=evaluation.std.clone(), old_value=value, advantages=torch.randn(5),
                    returns=torch.randn(5))


@pytest.mark.parametrize("actor,residual", [("transformer", "add"), ("transformer", "gated"),
                                           ("mlp", "add"), ("gru", "add")])
def test_diagnostics_preserve_exact_next_model_adam_rng_and_batch(actor, residual):
    torch.manual_seed(123)
    cfg = ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                      critic_dim=2, history_length=3, d_model=8, num_heads=2,
                      num_layers=1, ffn_dim=16, critic_hidden=(8,), baseline_hidden=(8,),
                      gru_hidden=8, actor_type=actor, residual_type=residual)
    plain = PPOTrainer(ActorCritic(cfg), config(epochs=2, num_minibatches=2, normalize_advantage=True))
    batch = real_batch(plain.model)
    # Warm Adam so equality covers populated moments and the next update, not just initialization.
    plain.update(batch)
    observed = PPOTrainer(deepcopy(plain.model), plain.config)
    observed.optimizer.load_state_dict(deepcopy(plain.optimizer.state_dict()))
    for _ in range(2):
        batch = refresh_behavior(plain.model, batch)
        batch_before = deepcopy(batch)
        rng, python_rng = torch.get_rng_state(), random.getstate()
        plain_metrics = plain.update(batch)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        observed_metrics = observed.update(batch, diagnostics=True)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert random.getstate() == python_rng
        assert_state_equal(observed.model.state_dict(), plain.model.state_dict())
        assert_state_equal(observed.optimizer.state_dict(), plain.optimizer.state_dict())
        assert all(p.grad is None for p in observed.model.parameters())
        assert set(observed_metrics) - set(plain_metrics) == DIAGNOSTIC_FIELDS
        assert {key: observed_metrics[key] for key in plain_metrics} == plain_metrics
        assert_state_equal(vars(batch.history), vars(batch_before.history))
        assert_state_equal({k: v for k, v in vars(batch).items() if k != "history"},
                           {k: v for k, v in vars(batch_before).items() if k != "history"})
        assert observed_metrics["first_step_kl"] > 0
        assert all(math.isfinite(value) for value in observed_metrics.values())


def test_default_and_explicit_false_have_no_extra_actor_passes():
    for kwargs in ({}, {"diagnostics": False}, {"diagnostics": True}):
        model = SyntheticModel()
        batch = make_batch(model)
        metrics = PPOTrainer(model, config(epochs=2, num_minibatches=2)).update(batch, **kwargs)
        diagnostic = kwargs.get("diagnostics", False)
        assert len(model.actor.calls) == 6 + (4 if diagnostic else 0)
        assert sum(enabled for enabled, _, _ in model.actor.calls) == 4
        assert bool(DIAGNOSTIC_FIELDS & metrics.keys()) == diagnostic

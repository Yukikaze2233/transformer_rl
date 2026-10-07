"""Directed PPO checks using differentiable synthetic Gaussian policies only."""
from dataclasses import fields, replace
from copy import deepcopy
import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.storage import PPOBatch, RolloutBuffer
from transformer_rl.types import HistoryBatch, PolicyEvaluation


class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(2))
        self.log_std = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(0.5)
        self.calls = []

    def evaluate(self, history, raw_action):
        self.calls.append((torch.is_grad_enabled(), history.now.detach().clone(), raw_action.detach().clone()))
        mean = self.dropout(self.mean + history.frames[:, -1, :2])
        std = self.log_std.exp().expand_as(mean)
        distribution = Normal(mean, std, validate_args=False)
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), mean, std)


class ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, critic):
        return self.value.expand(critic.shape[0])


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = GaussianActor()
        self.critic = ScalarCritic()


def make_batch(model, size=5, raw_action=None, advantages=None, returns=None):
    model.eval()
    history = HistoryBatch(
        frames=torch.zeros(size, 2, 3), times=torch.zeros(size, 2, dtype=torch.float64),
        valid=torch.tensor([[False, True]]).expand(size, -1).clone(),
        command=torch.zeros(size, 1), now=torch.arange(size, dtype=torch.float64),
    )
    raw_action = torch.full((size, 2), 2.0) if raw_action is None else raw_action
    critic = torch.zeros(size, 1)
    with torch.no_grad():
        evaluation = model.actor.evaluate(history, raw_action)
        old_value = model.critic(critic)
    model.actor.calls.clear()
    return PPOBatch(
        history=history, critic=critic, raw_action=raw_action,
        issued_action=raw_action.clamp(-1.0, 1.0), old_log_prob=evaluation.log_prob.detach().clone(),
        old_mean=evaluation.mean.detach().clone(), old_std=evaluation.std.detach().clone(),
        old_value=old_value.detach().clone(), advantages=torch.zeros(size) if advantages is None else advantages,
        returns=torch.zeros(size) if returns is None else returns,
    )


def config(**overrides):
    return PPOConfig(**{**dict(epochs=1, num_minibatches=1, normalize_advantage=False,
                              entropy_coef=0.0, target_kl=10.0), **overrides})


def snapshot_parameters(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


def assert_parameters_equal(model, snapshot):
    for actual, expected in zip(model.parameters(), snapshot, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("size,minibatches,epochs", [(5, 2, 3), (2, 8, 2), (1, 4, 1)])
def test_zero_update_consistency_and_no_dropped_remainder(size, minibatches, epochs):
    model = SyntheticModel()
    batch = make_batch(model, size)
    before = snapshot_parameters(model)
    trainer = PPOTrainer(model, config(epochs=epochs, num_minibatches=minibatches))
    model.train()  # update must restore eval without disabling gradients.
    with torch.no_grad():
        metrics = trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not model.training and not model.actor.dropout.training
    for name in ("actor_loss", "value_loss", "loss", "kl", "clip_fraction", "grad_norm"):
        assert metrics[name] == pytest.approx(0.0, abs=1e-7)
    assert metrics["entropy"] == pytest.approx(1.0 + math.log(2.0 * math.pi))
    assert metrics["optimizer_steps"] == min(size, minibatches) * epochs
    assert metrics["sample_count"] == size * epochs
    assert not metrics["early_stopped"]
    gradient_calls = [ids.tolist() for enabled, ids, _ in model.actor.calls if enabled]
    chunks = min(size, minibatches)
    for epoch in range(epochs):
        visited = [value for ids in gradient_calls[epoch * chunks:(epoch + 1) * chunks] for value in ids]
        assert sorted(visited) == list(range(size))


def test_old_log_prob_mismatch_in_final_sample_aborts_before_any_step():
    model = SyntheticModel()
    batch = make_batch(model, size=5, advantages=torch.ones(5))
    batch.old_log_prob[-1] += 0.1
    trainer = PPOTrainer(model, config(num_minibatches=2))
    before = snapshot_parameters(model)
    with pytest.raises(ValueError, match="old_log_prob mismatch"):
        trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state
    assert not any(enabled for enabled, _, _ in model.actor.calls)


def test_likelihood_uses_raw_action_not_clipped_issued_action():
    model = SyntheticModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    trainer.update(batch)
    assert all(torch.equal(raw, batch.raw_action) for _, _, raw in model.actor.calls)
    with torch.no_grad():
        wrong = model.actor.evaluate(batch.history, batch.issued_action).log_prob
    with pytest.raises(ValueError, match="old_log_prob mismatch"):
        trainer.update(replace(batch, old_log_prob=wrong))


def test_initial_losses_and_exact_gradient_include_half_value_and_entropy_coefficients():
    model = SyntheticModel()
    batch = make_batch(model, size=2, raw_action=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                       advantages=torch.tensor([2.0, 2.0]), returns=torch.tensor([3.0, 3.0]))
    trainer = PPOTrainer(model, config(value_coef=0.4, entropy_coef=0.3, max_grad_norm=100.0))
    gradients = {}
    for name, parameter in model.named_parameters():
        parameter.register_hook(lambda gradient, name=name: gradients.setdefault(name, gradient.clone()))
    # SGD makes the directed single-step parameter change independently calculable.
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = trainer.update(batch)
    expected_entropy = 1.0 + math.log(2.0 * math.pi)
    assert metrics["actor_loss"] == pytest.approx(-2.0)
    assert metrics["value_loss"] == pytest.approx(4.5)
    assert metrics["loss"] == pytest.approx(-2.0 + 0.4 * 4.5 - 0.3 * expected_entropy)
    torch.testing.assert_close(gradients["actor.mean"], torch.tensor([-2.0, 0.0]))
    torch.testing.assert_close(gradients["actor.log_std"], torch.tensor([-0.3, 1.7]))
    torch.testing.assert_close(gradients["critic.value"], torch.tensor(-1.2))
    torch.testing.assert_close(model.actor.mean, torch.tensor([0.02, 0.0]))
    torch.testing.assert_close(model.actor.log_std, torch.tensor([0.003, -0.017]))
    torch.testing.assert_close(model.critic.value, torch.tensor(0.012))
    assert metrics["grad_norm"] == pytest.approx(math.sqrt(4.0 + 0.09 + 2.89 + 1.44))


def test_grad_norm_is_pre_clip_and_optimizer_receives_clipped_gradients():
    model = SyntheticModel()
    batch = make_batch(model, size=1, advantages=torch.tensor([2.0]), returns=torch.tensor([3.0]))
    trainer = PPOTrainer(model, config(max_grad_norm=0.1))
    norms = []
    original_step = trainer.optimizer.step

    def checked_step(*args, **kwargs):
        norms.append(torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item())
        return original_step(*args, **kwargs)

    trainer.optimizer.step = checked_step
    metrics = trainer.update(batch)
    assert metrics["grad_norm"] > 0.1
    assert norms == pytest.approx([0.1], abs=1e-6)


@pytest.mark.parametrize("normalize", [True, False])
def test_advantage_normalization_is_learner_local_and_population_based(normalize):
    model = SyntheticModel()
    batch = make_batch(model, size=3, raw_action=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                       advantages=torch.tensor([1.0, 2.0, 6.0]),
                       returns=torch.tensor([4.0, 5.0, 8.0]))
    original_advantages, original_returns = batch.advantages.clone(), batch.returns.clone()
    trainer = PPOTrainer(model, config(normalize_advantage=normalize))
    gradients = []
    model.actor.mean.register_hook(lambda grad: gradients.append(grad.clone()))
    metrics = trainer.update(batch)
    assert metrics["actor_loss"] == pytest.approx(0.0 if normalize else -3.0, abs=1e-6)
    expected_gradient = -5.0 / (3.0 * math.sqrt(14.0 / 3.0)) if normalize else -23.0 / 3.0
    torch.testing.assert_close(gradients[0], torch.tensor([expected_gradient, 0.0]))
    torch.testing.assert_close(batch.advantages, original_advantages, rtol=0, atol=0)
    torch.testing.assert_close(batch.returns, original_returns, rtol=0, atol=0)
    assert metrics["value_loss"] == pytest.approx((16.0 + 25.0 + 64.0) / 6.0)


def test_singleton_advantage_normalization_is_finite():
    model = SyntheticModel()
    batch = make_batch(model, size=1, advantages=torch.tensor([4.0]))
    metrics = PPOTrainer(model, config(normalize_advantage=True)).update(batch)
    assert metrics["actor_loss"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_sample_weighted_metrics_with_remainders(monkeypatch):
    model = SyntheticModel()
    batch = make_batch(model, size=5, advantages=torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0]),
                       returns=torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0]))
    trainer = PPOTrainer(model, config(epochs=2, num_minibatches=2))
    # Freeze optimizer effects to isolate aggregation across unequal chunk sizes.
    monkeypatch.setattr(trainer.optimizer, "step", lambda: None)
    metrics = trainer.update(batch)
    assert metrics["actor_loss"] == pytest.approx(-4.0)
    assert metrics["value_loss"] == pytest.approx(13.0)
    assert metrics["sample_count"] == 10
    assert metrics["optimizer_steps"] == 4


def test_analytic_kl_old_to_new_and_early_stop_before_rejected_update():
    model = SyntheticModel()
    batch = make_batch(model, size=4, advantages=torch.ones(4))
    trainer = PPOTrainer(model, config(epochs=3, num_minibatches=2, target_kl=0.1, max_grad_norm=100.0))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = trainer.update(batch)
    # One real SGD step: grad_mean=-2 and grad_log_std=-3 per action dimension.
    torch.testing.assert_close(model.actor.mean, torch.full((2,), 0.2))
    torch.testing.assert_close(model.actor.log_std, torch.full((2,), 0.3))
    expected_kl = 2.0 * (0.3 + 0.5 * (1.0 + 0.2**2) * math.exp(-0.6) - 0.5)
    assert metrics["stop_kl"] == pytest.approx(expected_kl)
    reverse_kl = 2.0 * (-0.3 + 0.5 * (math.exp(0.6) + 0.2**2) - 0.5)
    assert metrics["stop_kl"] != pytest.approx(reverse_kl)
    assert metrics["early_stopped"]
    assert metrics["kl"] == 0.0
    assert metrics["optimizer_steps"] == 1
    assert metrics["sample_count"] == 2


def test_inconsistent_behavior_moments_are_rejected_before_any_step():
    model = SyntheticModel()
    batch = make_batch(model)
    batch.old_mean.fill_(10.0)
    trainer = PPOTrainer(model, config(target_kl=0.01))
    before = snapshot_parameters(model)
    with pytest.raises(ValueError, match="old_mean mismatch"):
        trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("moment", ["mean", "std"])
def test_equal_sample_log_prob_does_not_hide_changed_distribution_in_remainder(moment):
    model = SyntheticModel()
    batch = make_batch(model, size=5, raw_action=torch.ones(5, 2), advantages=torch.ones(5))
    if moment == "mean":
        # At a=1, N(0,1) and N(2,1) have identical density but KL=2 in this dimension.
        batch.old_mean[-1, 0] = 2.0
    else:
        # Swap two unequal stds at identical action coordinates: joint density is unchanged.
        with torch.no_grad():
            model.actor.log_std.copy_(torch.tensor([0.0, math.log(2.0)]))
        batch = make_batch(model, size=5, raw_action=torch.ones(5, 2), advantages=torch.ones(5))
        batch.old_std[-1] = batch.old_std[-1].flip(0)
    behavior = Normal(batch.old_mean, batch.old_std)
    torch.testing.assert_close(behavior.log_prob(batch.raw_action).sum(-1), batch.old_log_prob)
    trainer = PPOTrainer(model, config(num_minibatches=2))
    before = snapshot_parameters(model)
    with pytest.raises(ValueError, match=f"old_{moment} mismatch"):
        trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state
    assert not any(enabled for enabled, _, _ in model.actor.calls)


@pytest.mark.parametrize("mean,advantage,expected_clipped", [(0.5, 2.0, True), (0.5, -2.0, False),
                                                          (-0.5, -2.0, True), (-0.5, 2.0, False)])
def test_surrogate_clipping_selects_correct_sign_branch(monkeypatch, mean, advantage, expected_clipped):
    model = SyntheticModel()
    batch = make_batch(model, size=1, raw_action=torch.tensor([[1.0, 0.0]]),
                       advantages=torch.tensor([advantage]))
    trainer = PPOTrainer(model, config(epochs=2))
    gradients = []
    model.actor.mean.register_hook(lambda grad: gradients.append(grad.clone()))
    steps = []

    def set_mean():
        steps.append(True)
        with torch.no_grad():
            model.actor.mean[0] = mean

    monkeypatch.setattr(trainer.optimizer, "step", set_mean)
    metrics = trainer.update(batch)
    ratio = math.exp(mean - mean * mean / 2.0)
    clipped_ratio = min(max(ratio, 0.8), 1.2)
    second_loss = -min(ratio * advantage, clipped_ratio * advantage)
    assert metrics["actor_loss"] == pytest.approx((-advantage + second_loss) / 2.0)
    assert metrics["clip_fraction"] == pytest.approx(0.5)
    expected_gradient = 0.0 if expected_clipped else -advantage * ratio * (1.0 - mean)
    assert gradients[1][0].item() == pytest.approx(expected_gradient)


@pytest.mark.parametrize("new_value,target,second_loss,second_gradient", [
    (0.5, 1.0, 0.32, 0.0), (-0.5, -1.0, 0.32, 0.0), (0.5, -1.0, 1.125, 1.5),
])
def test_value_clipping_uses_max_squared_error_and_half_factor(
    monkeypatch, new_value, target, second_loss, second_gradient,
):
    model = SyntheticModel()
    batch = make_batch(model, size=1, returns=torch.tensor([target]))
    trainer = PPOTrainer(model, config(epochs=2, value_clip=0.2))
    gradients = []
    model.critic.value.register_hook(lambda grad: gradients.append(grad.clone()))

    def set_value():
        with torch.no_grad():
            model.critic.value.fill_(new_value)

    monkeypatch.setattr(trainer.optimizer, "step", set_value)
    metrics = trainer.update(batch)
    assert metrics["value_loss"] == pytest.approx((0.5 + second_loss) / 2.0)
    assert gradients[1].item() == pytest.approx(second_gradient)


@pytest.mark.parametrize("field", ["advantages", "returns", "old_value", "old_log_prob", "old_mean", "old_std"])
def test_nonfinite_batch_is_rejected_before_update(field):
    model = SyntheticModel()
    batch = make_batch(model)
    getattr(batch, field).flatten()[-1] = float("nan")
    trainer = PPOTrainer(model, config())
    before = snapshot_parameters(model)
    with pytest.raises(FloatingPointError, match=field):
        trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("bad_gradient", [float("nan"), float("inf")])
def test_nonfinite_gradient_aborts_before_parameter_or_adam_state_changes(bad_gradient, diagnostics):
    model = SyntheticModel()
    batch = make_batch(model, advantages=torch.ones(5), returns=torch.ones(5))
    trainer = PPOTrainer(model, config())
    before = snapshot_parameters(model)
    model.actor.mean.register_hook(lambda grad: torch.full_like(grad, bad_gradient))
    with pytest.raises(FloatingPointError, match="nonfinite gradient"):
        trainer.update(batch, diagnostics=diagnostics)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("diagnostics", [False, True])
def test_finite_gradient_with_overflowing_norm_aborts_optimizer(diagnostics):
    model = SyntheticModel()
    batch = make_batch(model, advantages=torch.ones(5))
    trainer = PPOTrainer(model, config())
    before = snapshot_parameters(model)
    model.actor.mean.register_hook(lambda grad: torch.full_like(grad, 3e38))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.update(batch, diagnostics=diagnostics)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("field", ["log_prob", "entropy", "mean", "std"])
def test_nonfinite_policy_evaluation_is_rejected(monkeypatch, field):
    model = SyntheticModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    original_evaluate = model.actor.evaluate

    def invalid_evaluate(history, action):
        evaluation = original_evaluate(history, action)
        return replace(evaluation, **{field: torch.full_like(getattr(evaluation, field), float("nan"))})

    monkeypatch.setattr(model.actor, "evaluate", invalid_evaluate)
    with pytest.raises(FloatingPointError, match=field):
        trainer.update(batch)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("failure", ["critic", "ratio", "loss", "KL"])
def test_forward_nonfinite_or_overflow_aborts_before_optimizer(monkeypatch, failure):
    model = SyntheticModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    before = snapshot_parameters(model)
    if failure == "critic":
        monkeypatch.setattr(model.critic, "forward", lambda value: torch.full((len(value),), float("nan")))
    elif failure == "ratio":
        original_evaluate = model.actor.evaluate

        def overflow_ratio(history, action):
            evaluation = original_evaluate(history, action)
            if torch.is_grad_enabled():
                return replace(evaluation, log_prob=evaluation.log_prob + 1000.0)
            return evaluation

        monkeypatch.setattr(model.actor, "evaluate", overflow_ratio)
    elif failure == "loss":
        batch.returns.fill_(3e38)
    else:
        original_evaluate = model.actor.evaluate

        def overflow_kl(history, action):
            evaluation = original_evaluate(history, action)
            if torch.is_grad_enabled():
                return replace(evaluation, std=torch.full_like(evaluation.std, 1e-38))
            return evaluation

        monkeypatch.setattr(model.actor, "evaluate", overflow_kl)
    with pytest.raises(FloatingPointError, match=failure):
        trainer.update(batch)
    assert_parameters_equal(model, before)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("shape", ["critic", "log_prob", "advantages"])
def test_rejects_column_scalars_in_learner(monkeypatch, shape):
    model = SyntheticModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    if shape == "critic":
        original_forward = model.critic.forward
        monkeypatch.setattr(model.critic, "forward", lambda value: original_forward(value).unsqueeze(-1))
    elif shape == "log_prob":
        original_evaluate = model.actor.evaluate

        def wrong_shape(history, action):
            evaluation = original_evaluate(history, action)
            return replace(evaluation, log_prob=evaluation.log_prob.unsqueeze(-1))

        monkeypatch.setattr(model.actor, "evaluate", wrong_shape)
    else:
        batch = replace(batch, advantages=batch.advantages.unsqueeze(-1))
    with pytest.raises(ValueError, match="shape"):
        trainer.update(batch)
    assert not trainer.optimizer.state


def test_rollout_targets_are_detached_and_optimizer_is_checkpointable():
    model = SyntheticModel()
    batch = make_batch(model, advantages=torch.ones(5), returns=torch.ones(5))
    for field in fields(PPOBatch):
        if field.name != "history":
            getattr(batch, field.name).requires_grad_()
    trainer = PPOTrainer(model, config())
    metrics = trainer.update(batch)
    assert metrics["optimizer_steps"] == 1
    assert all(getattr(batch, field.name).grad is None for field in fields(PPOBatch) if field.name != "history")
    state = trainer.optimizer.state_dict()
    restored = PPOTrainer(SyntheticModel(), config())
    restored.optimizer.load_state_dict(state)
    assert len(restored.optimizer.state) == len(list(model.parameters()))
    assert restored.optimizer.param_groups[0]["lr"] == trainer.config.learning_rate


def test_real_actor_padding_contract_through_storage_ppo_and_export_path(monkeypatch):
    model_config = ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                               critic_dim=2, history_length=3, d_model=8, num_heads=2,
                               num_layers=1, ffn_dim=16, critic_hidden=(8,))
    model = ActorCritic(model_config).eval()
    valid = torch.tensor([[False, False, True], [False, True, True], [False, False, False]])
    clean = HistoryBatch(
        frames=torch.randn(3, 3, model_config.frame_dim),
        times=torch.arange(3, dtype=torch.float64).expand(3, -1).clone(),
        valid=valid, command=torch.randn(3, 1), now=torch.full((3,), 2.0, dtype=torch.float64),
    )
    padded = clean.clone()
    for row, sentinel in enumerate((float("nan"), float("inf"), -float("inf"))):
        padded.frames[row, ~valid[row]] = sentinel
        padded.times[row, ~valid[row]] = sentinel
    critic = torch.randn(3, 2)
    with torch.no_grad():
        sample = model.actor.act(clean)
        old_value = model.critic(critic)
    buffer = RolloutBuffer(2)
    buffer.add(
        history=padded, critic=critic, raw_action=sample.action, issued_action=sample.action.clamp(-1, 1),
        old_log_prob=sample.evaluation.log_prob, old_mean=sample.evaluation.mean,
        old_std=sample.evaluation.std, old_value=old_value, reward=torch.tensor([1.0, 2.0, 3.0]),
        next_value=torch.zeros(3), terminated=torch.ones(3, dtype=torch.bool),
        truncated=torch.zeros(3, dtype=torch.bool),
    )
    batch = buffer.finish(0.99, 0.95)
    batch.validate()
    with torch.no_grad():
        evaluation = model.actor.evaluate(batch.history, batch.raw_action)
        exported_mean = model.actor.forward_tensors(
            **{field.name: getattr(batch.history, field.name) for field in fields(HistoryBatch)}
        )
    torch.testing.assert_close(evaluation.log_prob, batch.old_log_prob, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(evaluation.mean, batch.old_mean)
    torch.testing.assert_close(evaluation.std, batch.old_std)
    torch.testing.assert_close(exported_mean, batch.old_mean)

    trainer = PPOTrainer(model, config(max_grad_norm=100.0))
    gradients = []

    def capture_gradients_without_updating_weights():
        gradients.append({name: parameter.grad.clone() for name, parameter in model.named_parameters()})

    # Keep identical weights to compare learner gradients for semantic-equivalent padding.
    monkeypatch.setattr(trainer.optimizer, "step", capture_gradients_without_updating_weights)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        padded_metrics = trainer.update(batch)
        torch.manual_seed(17)
        clean_metrics = trainer.update(replace(batch, history=clean))
    assert padded_metrics == pytest.approx(clean_metrics)
    assert gradients and any(gradient.abs().sum() > 0 for gradient in gradients[0].values())
    for name in gradients[0]:
        assert torch.isfinite(gradients[0][name]).all()
        torch.testing.assert_close(gradients[0][name], gradients[1][name])


@pytest.mark.parametrize("optimizer_kind", ["sgd", "adam"])
def test_step_diagnostics_hand_calculated_gradients_and_actual_parameter_delta(optimizer_kind):
    model = SyntheticModel().double()
    with torch.no_grad():
        model.actor.mean.copy_(torch.tensor([0.5, -0.5]))
        model.actor.log_std.fill_(math.log(2.0))
        model.critic.value.fill_(2.0)
    batch = make_batch(model, size=2, raw_action=torch.tensor([[1.5, -0.5]] * 2, dtype=torch.float64),
                       advantages=torch.full((2,), 2.0), returns=torch.full((2,), 5.0))
    trainer = PPOTrainer(model, config(value_coef=0.4, entropy_coef=0.3, max_grad_norm=100.0))
    if optimizer_kind == "sgd":
        trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.01)
    else:
        # A visible epsilon makes the first Adam update independently calculable
        # and decisively different from learning_rate * gradient.
        trainer.optimizer = torch.optim.Adam(trainer.policy_parameters, lr=0.01, eps=0.5)
    metrics = trainer.update(batch, diagnostics=True)
    actor_gradient = torch.tensor([-0.5, 0.0, 1.2, 1.7], dtype=torch.float64)
    critic_gradient = torch.tensor([-1.2], dtype=torch.float64)
    gradient = torch.cat((actor_gradient, critic_gradient))
    delta = -0.01 * gradient if optimizer_kind == "sgd" else -0.01 * gradient / (gradient.abs() + 0.5)
    actor_old_norm = math.sqrt(0.5 + 2.0 * math.log(2.0) ** 2)
    assert metrics["actor_grad_norm"] == pytest.approx(math.sqrt(0.25 + 1.44 + 2.89))
    assert metrics["critic_grad_norm"] == pytest.approx(1.2)
    assert metrics["grad_norm"] == pytest.approx(gradient.norm().item())
    assert metrics["actor_param_update_l2"] == pytest.approx(delta[:4].norm().item())
    assert metrics["critic_param_update_l2"] == pytest.approx(abs(delta[-1].item()))
    assert metrics["param_update_l2"] == pytest.approx(delta.norm().item())
    assert metrics["actor_param_update_relative_l2"] == pytest.approx(delta[:4].norm().item() / actor_old_norm)
    assert metrics["critic_param_update_relative_l2"] == pytest.approx(abs(delta[-1].item()) / 2.0)
    assert metrics["param_update_relative_l2"] == pytest.approx(delta.norm().item() / math.sqrt(actor_old_norm ** 2 + 4.0))
    assert metrics["grad_clip_coef"] == 1.0 and metrics["grad_clip_step_fraction"] == 0.0
    assert metrics["param_update_relative_step_count"] == 1
    assert (metrics["actor_param_count"], metrics["critic_param_count"], metrics["param_count"]) == (4, 1, 5)
    torch.testing.assert_close(model.actor.mean, torch.tensor([0.5, -0.5], dtype=torch.float64) + delta[:2])
    torch.testing.assert_close(model.actor.log_std, torch.full((2,), math.log(2.0), dtype=torch.float64) + delta[2:4])
    torch.testing.assert_close(model.critic.value, torch.tensor(2.0, dtype=torch.float64) + delta[-1])


def test_global_critic_gradient_clips_actor_too_with_exact_coefficient():
    model = SyntheticModel().double()
    batch = make_batch(model, size=1, advantages=torch.ones(1), returns=torch.tensor([100.0]))
    trainer = PPOTrainer(model, config(max_grad_norm=0.1))
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.01)
    metrics = trainer.update(batch, diagnostics=True)
    coefficient = (0.1 / (torch.tensor(math.sqrt(10026.0), dtype=torch.float64) + 1e-6)).item()
    assert metrics["actor_grad_norm"] == pytest.approx(math.sqrt(26.0))
    assert metrics["critic_grad_norm"] == 100.0
    assert metrics["grad_clip_coef"] == pytest.approx(coefficient, rel=1e-14)
    assert metrics["grad_clip_step_fraction"] == 1.0
    assert metrics["actor_param_update_l2"] == pytest.approx(0.01 * coefficient * math.sqrt(26.0))
    assert metrics["critic_param_update_l2"] == pytest.approx(coefficient)
    assert metrics["param_update_l2"] == pytest.approx(0.01 * coefficient * math.sqrt(10026.0))
    assert metrics["param_update_relative_l2"] is None
    assert metrics["param_update_relative_step_count"] == 0


def test_clip_step_fraction_uses_epsilon_coefficient_at_exact_bound():
    model = SyntheticModel()
    model.actor.log_std.requires_grad_(False)
    batch = make_batch(model, size=1, raw_action=torch.tensor([[1.0, 0.0]]), advantages=torch.ones(1))
    trainer = PPOTrainer(model, config(max_grad_norm=1.0))
    metrics = trainer.update(batch, diagnostics=True)
    expected = (1.0 / (torch.tensor(1.0, dtype=torch.float32) + 1e-6)).item()
    assert metrics["grad_norm"] == 1.0
    assert metrics["grad_clip_coef"] == expected < 1.0
    assert metrics["grad_clip_step_fraction"] == 1.0


def test_applied_step_diagnostics_exclude_rejected_kl_batch():
    model = SyntheticModel()
    batch = make_batch(model, size=5, advantages=torch.ones(5))
    trainer = PPOTrainer(model, config(epochs=3, num_minibatches=2, target_kl=0.1, max_grad_norm=100.0))
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.1)
    metrics = trainer.update(batch, diagnostics=True)
    assert metrics["early_stopped"] and metrics["optimizer_steps"] == 1
    assert metrics["planned_optimizer_steps"] == 6 and metrics["sample_count"] == 3
    assert metrics["actor_grad_norm"] == pytest.approx(math.sqrt(26.0))
    assert metrics["actor_param_update_l2"] == pytest.approx(0.1 * math.sqrt(26.0))
    assert metrics["actor_param_update_relative_l2"] is None
    assert metrics["actor_param_update_relative_step_count"] == 0
    assert metrics["grad_clip_coef"] == 1.0 and metrics["grad_clip_step_fraction"] == 0.0


def test_no_applied_step_has_null_optimization_diagnostics():
    model = SyntheticModel()
    with torch.no_grad():
        model.actor.log_std.fill_(math.log(1e-6))
    batch = make_batch(model, raw_action=torch.full((5, 2), 0.5e-6))
    batch.old_mean.fill_(1e-6)
    trainer = PPOTrainer(model, config(num_minibatches=2, target_kl=0.1))
    before = snapshot_parameters(model)
    metrics = trainer.update(batch, diagnostics=True)
    assert metrics["optimizer_steps"] == 0 and not trainer.optimizer.state
    assert_parameters_equal(model, before)
    for prefix in ("", "actor_", "critic_", "shared_", "other_"):
        assert metrics[f"{prefix}param_update_l2"] is None
        assert metrics[f"{prefix}param_update_relative_l2"] is None
        assert metrics[f"{prefix}param_update_relative_step_count"] == 0
        if prefix:
            assert metrics[f"{prefix}grad_norm"] is None
    assert metrics["grad_clip_coef"] is None and metrics["grad_clip_step_fraction"] is None


def test_step_means_and_relative_valid_denominator_with_unequal_minibatches(monkeypatch):
    model = SyntheticModel().double()
    batch = make_batch(model, size=5, advantages=torch.tensor([1.0, 1.0, 1.0, 4.0, 4.0]))
    trainer = PPOTrainer(model, config(num_minibatches=2, max_grad_norm=10.0))
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.0001)
    monkeypatch.setattr(torch, "randperm", lambda n, device: torch.arange(n, device=device))
    gradients, deltas, relative = [], [], []
    step_gradients = {}
    for name, parameter in model.actor.named_parameters():
        parameter.register_hook(lambda grad, name=name: step_gradients.update({name: grad.detach().clone()}))
    step = trainer.optimizer.step

    def capture_actual_step():
        before = torch.cat([p.detach().flatten().clone() for p in model.actor.parameters()])
        gradients.append(torch.cat([step_gradients[name].flatten() for name, _ in model.actor.named_parameters()]).norm().item())
        step()
        after = torch.cat([p.detach().flatten() for p in model.actor.parameters()])
        delta = (after - before).norm().item()
        deltas.append(delta)
        if before.norm().item() > 0:
            relative.append(delta / before.norm().item())

    monkeypatch.setattr(trainer.optimizer, "step", capture_actual_step)
    metrics = trainer.update(batch, diagnostics=True)
    assert metrics["optimizer_steps"] == 2 and metrics["sample_count"] == 5
    assert gradients[0] < 10.0 < gradients[1]
    assert metrics["actor_grad_norm"] == pytest.approx(sum(gradients) / 2)
    assert metrics["actor_grad_norm"] != pytest.approx((3 * gradients[0] + 2 * gradients[1]) / 5)
    assert metrics["actor_param_update_l2"] == pytest.approx(sum(deltas) / 2)
    assert metrics["grad_clip_step_fraction"] == 0.5
    expected_coefficients = [min(1.0, 10.0 / (norm + 1e-6)) for norm in gradients]
    assert metrics["grad_clip_coef"] == pytest.approx(sum(expected_coefficients) / 2)
    assert len(relative) == metrics["actor_param_update_relative_step_count"] == 1
    assert metrics["actor_param_update_relative_l2"] == pytest.approx(relative[0])


class SharedCritic(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.value = nn.Parameter(torch.tensor(2.0, dtype=shared.dtype))

    def forward(self, critic):
        return (self.value + self.shared[0]).expand(len(critic))


def test_shared_parameters_are_counted_once_in_separate_group():
    model = SyntheticModel().double()
    with torch.no_grad():
        model.actor.mean.copy_(torch.tensor([0.5, -0.5]))
    model.critic = SharedCritic(model.actor.mean)
    batch = make_batch(model, size=1, raw_action=torch.tensor([[1.5, -0.5]], dtype=torch.float64),
                       advantages=torch.tensor([2.0]), returns=torch.tensor([5.5]))
    trainer = PPOTrainer(model, config(value_coef=0.4, max_grad_norm=100.0))
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.01)
    metrics = trainer.update(batch, diagnostics=True)
    assert [metrics[f"{group}_param_count"] for group in ("actor", "critic", "shared", "other")] == [2, 1, 2, 0]
    assert metrics["param_count"] == 5
    assert metrics["actor_grad_norm"] == pytest.approx(2.0)
    assert metrics["critic_grad_norm"] == pytest.approx(1.2)
    assert metrics["shared_grad_norm"] == pytest.approx(3.2)
    assert metrics["other_grad_norm"] == 0.0
    assert metrics["grad_norm"] ** 2 == pytest.approx(sum(metrics[f"{group}_grad_norm"] ** 2 for group in ("actor", "critic", "shared", "other")))
    assert metrics["actor_param_update_l2"] == pytest.approx(0.02)
    assert metrics["critic_param_update_l2"] == pytest.approx(0.012)
    assert metrics["shared_param_update_l2"] == pytest.approx(0.032)
    assert metrics["param_update_l2"] ** 2 == pytest.approx(0.02 ** 2 + 0.012 ** 2 + 0.032 ** 2)
    assert metrics["actor_param_update_relative_l2"] is None
    assert metrics["shared_param_update_relative_l2"] == pytest.approx(0.032 / math.sqrt(0.5))


class PartitionedModel(SyntheticModel):
    def __init__(self):
        super().__init__()
        self.actor.estimator = nn.Parameter(torch.tensor(100.0))
        self.actor.frozen = nn.Parameter(torch.tensor(1000.0), requires_grad=False)
        self.outside = nn.Parameter(torch.tensor(3.0))

    def policy_parameters(self):
        return [p for p in self.parameters() if p is not self.actor.estimator]


def test_only_optimized_policy_parameters_enter_actor_or_other_diagnostics():
    model = PartitionedModel().double()
    batch = make_batch(model, advantages=torch.ones(5), returns=torch.ones(5))
    trainer = PPOTrainer(model, config(max_grad_norm=1000.0))
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.01)
    metrics = trainer.update(batch, diagnostics=True,
                             regularizer=lambda: model.outside.square() + model.actor.estimator.square())
    assert metrics["actor_param_count"] == 4 and metrics["other_param_count"] == 1
    assert metrics["param_count"] == 6
    assert metrics["other_grad_norm"] == 6.0
    assert metrics["grad_norm"] == pytest.approx(math.sqrt(63.0))
    assert metrics["other_param_update_l2"] == pytest.approx(0.06)
    assert metrics["other_param_update_relative_l2"] == pytest.approx(0.02)
    assert metrics["actor_param_update_relative_l2"] is None
    assert model.actor.estimator.item() == 100.0 and model.actor.frozen.item() == 1000.0
    assert model.actor.estimator.grad.item() == 200.0


def test_optimizer_subset_narrows_observation_without_changing_clip_scope():
    model = SyntheticModel().double()
    batch = make_batch(model, size=1, advantages=torch.ones(1), returns=torch.tensor([100.0]))
    trainer = PPOTrainer(model, config(max_grad_norm=0.1))
    trainer.optimizer = torch.optim.SGD(model.actor.parameters(), lr=0.01)
    metrics = trainer.update(batch, diagnostics=True)
    assert metrics["param_count"] == metrics["actor_param_count"] == 4
    assert metrics["critic_param_count"] == 0 and metrics["critic_grad_norm"] == 0.0
    assert metrics["critic_param_update_l2"] == 0.0 and model.critic.value.item() == 0.0
    coefficient = 0.1 / (math.sqrt(10026.0) + 1e-6)
    assert metrics["grad_norm"] == pytest.approx(math.sqrt(10026.0))
    assert metrics["grad_clip_coef"] == pytest.approx(coefficient)
    assert metrics["param_update_l2"] == pytest.approx(0.01 * coefficient * math.sqrt(26.0))


def test_actor_gradient_diagnostics_include_weighted_auxiliary_loss():
    from test_auxiliary import AuxiliaryModel, config as auxiliary_config, make_batch as auxiliary_batch

    model = AuxiliaryModel()
    trainer = PPOTrainer(model, auxiliary_config())
    trainer.optimizer = torch.optim.SGD(trainer.policy_parameters, lr=0.01)
    metrics = trainer.update(auxiliary_batch(model), diagnostics=True)
    # The weighted auxiliary loss contributes shared=-9 and head=(-1.5,-2);
    # rollout advantages, returns, entropy coefficient all give zero gradient.
    assert metrics["actor_grad_norm"] == pytest.approx(math.sqrt(81.0 + 2.25 + 4.0))
    assert metrics["actor_param_update_l2"] == pytest.approx(0.01 * math.sqrt(87.25), rel=1e-5)
    assert metrics["actor_param_count"] == 5 and metrics["shared_param_count"] == 0


@pytest.mark.parametrize("family", ["velocity", "context"])
def test_actual_detached_estimator_is_excluded_and_both_optimizer_states_match(family):
    from test_estimators import batch as estimator_batch, config as estimator_config

    model = ActorCritic(estimator_config("mlp", family, baseline_hidden=(8,), controller_hidden=(8,),
                                        critic_hidden=(8,), context_target_hidden=(8,)))
    cfg = config(epochs=1, num_minibatches=1, estimator_epochs=1, estimator_minibatches=1,
                 estimator_target_kl=10.0)
    plain = PPOTrainer(model, cfg)
    data = estimator_batch(model, size=3)
    observed = PPOTrainer(deepcopy(model), cfg)
    policy_ids = {id(p) for p in model.policy_parameters() if p.requires_grad}
    actor_parameters = [p for p in model.actor.parameters() if id(p) in policy_ids]
    actor_before = torch.cat([p.detach().double().flatten().clone() for p in actor_parameters])
    estimator_before = deepcopy(model.actor.estimator.state_dict())
    rng = torch.get_rng_state()
    expected = plain.update(data, diagnostics=False)
    next_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = observed.update(data, diagnostics=True)
    assert {key: actual[key] for key in expected} == expected
    torch.testing.assert_close(observed.model.state_dict(), plain.model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(observed.optimizer.state_dict(), plain.optimizer.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(observed.estimator.optimizer.state_dict(), plain.estimator.optimizer.state_dict(), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), next_rng)
    assert actual["actor_param_count"] == sum(p.numel() for p in actor_parameters)
    assert actual["param_count"] == sum(p.numel() for p in model.policy_parameters())
    assert actual["other_param_count"] == actual["shared_param_count"] == 0
    actor_after = torch.cat([p.detach().double().flatten() for p in actor_parameters])
    assert actual["actor_param_update_l2"] == pytest.approx((actor_after - actor_before).norm().item())
    assert any(not torch.equal(value, estimator_before[key]) for key, value in model.actor.estimator.state_dict().items())


def test_diagnostics_false_never_constructs_optimizer_observer(monkeypatch):
    import transformer_rl.ppo as ppo_module

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled diagnostics must not construct a parameter observer")

    monkeypatch.setattr(ppo_module, "_OptimizerStepDiagnostics", forbidden)
    model = SyntheticModel()
    trainer = PPOTrainer(model, config())
    metrics = trainer.update(make_batch(model, advantages=torch.ones(5)), diagnostics=False)
    assert metrics["optimizer_steps"] == 1
    assert "actor_grad_norm" not in metrics and "param_update_l2" not in metrics


@pytest.mark.parametrize("early_stop", [False, True])
def test_optimization_observation_preserves_populated_adam_and_all_rng(early_stop):
    import random
    import numpy as np

    model = SyntheticModel()
    cfg = config(epochs=2, num_minibatches=2, normalize_advantage=True,
                 target_kl=1e-7 if early_stop else 10.0, learning_rate=0.01)
    plain = PPOTrainer(model, cfg)
    plain.update(make_batch(model, advantages=torch.arange(5).float(), returns=torch.ones(5)))
    observed = PPOTrainer(deepcopy(model), cfg)
    observed.optimizer.load_state_dict(deepcopy(plain.optimizer.state_dict()))
    for _ in range(2):
        batch = make_batch(model, advantages=torch.arange(5).float(), returns=torch.ones(5))
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        expected = plain.update(batch, diagnostics=False)
        next_rng = torch.get_rng_state()
        torch.set_rng_state(torch_rng)
        actual = observed.update(batch, diagnostics=True)
        assert {key: actual[key] for key in expected} == expected
        assert actual["early_stopped"] == early_stop
        torch.testing.assert_close(observed.model.state_dict(), plain.model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(observed.optimizer.state_dict(), plain.optimizer.state_dict(), rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), next_rng)
        assert random.getstate() == python_rng
        current_numpy_rng = np.random.get_state()
        assert current_numpy_rng[0] == numpy_rng[0]
        assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
        assert current_numpy_rng[2:] == numpy_rng[2:]
        assert all(p.grad is None for p in observed.model.parameters())

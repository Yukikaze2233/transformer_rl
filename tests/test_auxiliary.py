"""Auxiliary PPO supervision checks using synthetic endpoints only."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.storage import PPOBatch
from transformer_rl.types import HistoryBatch, PolicyEvaluation


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class AuxiliaryActor(nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.zeros(2))
        self.auxiliary_head = nn.Linear(1, 2, bias=False) if with_head else None
        if with_head:
            with torch.no_grad():
                self.auxiliary_head.weight.copy_(torch.tensor([[2.0], [3.0]]))
        self.auxiliary_calls = 0

    def representation(self, history):
        return self.shared * history.frames[:, -1, :1]

    def evaluate(self, history, raw_action):
        mean = self.representation(history).expand(-1, 2)
        std = self.log_std.exp().expand_as(mean)
        distribution = Normal(mean, std)
        return PolicyEvaluation(distribution.log_prob(raw_action).sum(-1),
                                distribution.entropy().sum(-1), mean, std)

    def predict_auxiliary(self, history):
        self.auxiliary_calls += 1
        assert isinstance(history, HistoryBatch)
        if self.auxiliary_head is None:
            raise ValueError("auxiliary head is not configured")
        return self.auxiliary_head(self.representation(history))


class ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, critic):
        return self.value.expand(len(critic))


class AuxiliaryModel(nn.Module):
    def __init__(self, with_head=True):
        super().__init__()
        self.config = SimpleNamespace(auxiliary_indices=(2, 0))
        self.actor = AuxiliaryActor(with_head)
        self.critic = ScalarCritic()


def config(**overrides):
    return PPOConfig(**{"auxiliary_coef": 0.5,
                              "epochs": 1, "num_minibatches": 1,
                              "normalize_advantage": False, "value_coef": 0.0,
                              "entropy_coef": 0.0, "max_grad_norm": 100.0,
                              "target_kl": 10.0, **overrides})


def make_batch(model, size=5):
    frame_dim = getattr(getattr(model, "config", None), "frame_dim", 1)
    now = torch.arange(size, dtype=torch.float64)
    history = HistoryBatch(
        frames=torch.ones(size, 2, frame_dim),
        times=now[:, None] + torch.tensor([-0.01, 0.0], dtype=torch.float64),
        valid=torch.ones(size, 2, dtype=torch.bool), command=torch.zeros(size, 1),
        now=now,
    )
    critic = torch.tensor([[7.0, 999.0, 5.0]]).repeat(size, 1)
    raw_action = torch.zeros(size, 2)
    with torch.no_grad():
        evaluation = model.actor.evaluate(history, raw_action)
        value = model.critic(critic)
    return PPOBatch(history, critic, raw_action, raw_action.clone(),
                    evaluation.log_prob.clone(), evaluation.mean.clone(), evaluation.std.clone(),
                    value.detach().clone(), torch.zeros(size), torch.zeros(size))


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_unchanged(model, before):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_auxiliary_exact_loss_target_detach_and_head_shared_gradients():
    model = AuxiliaryModel()
    batch = make_batch(model)
    for field in fields(batch):
        if field.name != "history":
            getattr(batch, field.name).requires_grad_()
    batch.history.frames.requires_grad_()
    batch.history.command.requires_grad_()
    trainer = PPOTrainer(model, config())
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}
    gradients = {}
    for name, parameter in model.named_parameters():
        parameter.register_hook(lambda grad, name=name: gradients.setdefault(name, grad.clone()))
    before = snapshot(model)
    metrics = trainer.update(batch)
    assert metrics["auxiliary_loss"] == pytest.approx(12.5)
    assert metrics["auxiliary_coef"] == 0.5
    assert metrics["loss"] == pytest.approx(6.25)
    torch.testing.assert_close(gradients["actor.auxiliary_head.weight"], torch.tensor([[-1.5], [-2.0]]))
    torch.testing.assert_close(gradients["actor.shared"], torch.tensor(-9.0))
    assert not torch.equal(before["actor.shared"], model.actor.shared)
    assert not torch.equal(before["actor.auxiliary_head.weight"], model.actor.auxiliary_head.weight)
    assert model.actor.auxiliary_head.weight in trainer.optimizer.state
    assert all(getattr(batch, f.name).grad is None for f in fields(batch) if f.name != "history")
    assert batch.history.frames.grad is None and batch.history.command.grad is None


def test_zero_coefficient_has_exact_policy_update_parity_and_no_head_gradient():
    with_head = AuxiliaryModel()
    without_head = AuxiliaryModel(with_head=False)
    del without_head.config
    batch = replace(make_batch(with_head), advantages=torch.ones(5), returns=torch.ones(5))
    trainers = [PPOTrainer(model, config(auxiliary_coef=0.0, num_minibatches=2, value_coef=0.4))
                for model in (with_head, without_head)]
    head_before = with_head.actor.auxiliary_head.weight.detach().clone()
    head_gradients = []
    with_head.actor.auxiliary_head.weight.register_hook(lambda grad: head_gradients.append(grad.clone()))
    metrics = []
    for trainer in trainers:
        with torch.random.fork_rng():
            torch.manual_seed(41)
            metrics.append(trainer.update(batch))
    assert metrics[0] == metrics[1]
    assert metrics[0]["auxiliary_loss"] == metrics[0]["auxiliary_coef"] == 0.0
    for name, parameter in without_head.named_parameters():
        torch.testing.assert_close(dict(with_head.named_parameters())[name], parameter, rtol=0, atol=0)
    torch.testing.assert_close(with_head.actor.auxiliary_head.weight, head_before, rtol=0, atol=0)
    assert not head_gradients and with_head.actor.auxiliary_calls == without_head.actor.auxiliary_calls == 0
    assert with_head.actor.auxiliary_head.weight not in trainers[0].optimizer.state


@pytest.mark.parametrize("missing", ["config", "indices", "empty", "method", "head"])
def test_missing_auxiliary_configuration_is_explicitly_rejected(missing):
    model = AuxiliaryModel(with_head=missing != "head")
    batch = make_batch(model)
    if missing == "config":
        del model.config
    elif missing == "indices":
        del model.config.auxiliary_indices
    elif missing == "empty":
        model.config.auxiliary_indices = ()
    elif missing == "method":
        model.actor.predict_auxiliary = None
    before = snapshot(model)
    with pytest.raises(ValueError, match="auxiliary"):
        trainer = PPOTrainer(model, config())
        trainer.update(batch)
    assert_unchanged(model, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 3e38])
def test_nonfinite_prediction_or_mse_rejected_before_optimizer_step(monkeypatch, bad):
    model = AuxiliaryModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    before = snapshot(model)
    monkeypatch.setattr(model.actor, "predict_auxiliary", lambda history: torch.full((len(history.now), 2), bad))
    with pytest.raises(FloatingPointError, match="auxiliary"):
        trainer.update(batch)
    assert_unchanged(model, before)
    assert not trainer.optimizer.state


@pytest.mark.parametrize("bad", [torch.zeros(5, 1), torch.zeros(5, 2, dtype=torch.long), None])
def test_invalid_prediction_contract_rejected(monkeypatch, bad):
    model = AuxiliaryModel()
    trainer = PPOTrainer(model, config())
    batch = make_batch(model)
    monkeypatch.setattr(model.actor, "predict_auxiliary", lambda history: bad)
    with pytest.raises(ValueError, match="auxiliary prediction"):
        trainer.update(batch)
    assert not trainer.optimizer.state


def test_auxiliary_nonfinite_gradient_rejected_before_adam_state_mutation():
    model = AuxiliaryModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config())
    before = snapshot(model)
    model.actor.auxiliary_head.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(FloatingPointError, match="nonfinite gradient"):
        trainer.update(batch)
    assert_unchanged(model, before)
    assert not trainer.optimizer.state
    assert all(p.grad is None for p in model.parameters())


def test_auxiliary_metrics_are_sample_weighted_over_unequal_minibatches(monkeypatch):
    model = AuxiliaryModel()
    batch = make_batch(model)
    batch.critic[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 20.0])
    batch.critic[:, 2] = torch.tensor([4.0, 1.0, 5.0, 2.0, 10.0])
    expected = ((batch.critic[:, 0] - 3).square() + (batch.critic[:, 2] - 2).square()).mean() / 2
    trainer = PPOTrainer(model, config(epochs=3, num_minibatches=2))
    monkeypatch.setattr(trainer.optimizer, "step", lambda: None)
    metrics = trainer.update(batch)
    assert metrics["auxiliary_loss"] == pytest.approx(expected.item())
    assert metrics["loss"] == pytest.approx(0.5 * expected.item())
    assert metrics["sample_count"] == 15
    assert metrics["optimizer_steps"] == 6


def test_kl_rejected_minibatch_does_not_predict_or_contribute_auxiliary_metrics(monkeypatch):
    model = AuxiliaryModel()
    batch = make_batch(model)
    trainer = PPOTrainer(model, config(epochs=3, num_minibatches=2, target_kl=0.01))

    def change_policy():
        with torch.no_grad():
            model.actor.shared.add_(10.0)

    monkeypatch.setattr(trainer.optimizer, "step", change_policy)
    metrics = trainer.update(batch)
    assert metrics["early_stopped"] and metrics["stop_kl"] > 0.01
    assert metrics["optimizer_steps"] == 1 and metrics["sample_count"] == 3
    assert model.actor.auxiliary_calls == 1
    assert metrics["auxiliary_loss"] == pytest.approx(12.5)
    assert metrics["loss"] == pytest.approx(6.25)


@pytest.mark.parametrize("field", ["old_log_prob", "old_mean", "old_std"])
def test_auxiliary_enabled_preserves_complete_behavior_checks(field):
    model = AuxiliaryModel()
    batch = make_batch(model)
    getattr(batch, field)[-1] += 0.2
    trainer = PPOTrainer(model, config(num_minibatches=2))
    before = snapshot(model)
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        trainer.update(batch)
    assert not trainer.optimizer.state and model.actor.auxiliary_calls == 0
    assert_unchanged(model, before)


@pytest.mark.parametrize("coefficient", [0.0, 0.5])
def test_real_query_representation_and_head_receive_only_enabled_auxiliary_gradients(monkeypatch, coefficient):
    model_config = ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                               critic_dim=3, history_length=2, d_model=8, num_heads=2,
                               num_layers=1, ffn_dim=16, critic_hidden=(8,), auxiliary_indices=(2, 0))
    with torch.random.fork_rng():
        torch.manual_seed(51)
        model = ActorCritic(model_config).eval()
    batch = make_batch(model)
    batch.critic.requires_grad_()
    gradients = {}
    for name, parameter in model.actor.named_parameters():
        parameter.register_hook(lambda grad, name=name: gradients.setdefault(name, grad.clone()))
    encoded = []
    original_encode = model.actor.encode

    def capture_encode(history):
        tokens = original_encode(history)
        if tokens.requires_grad:
            tokens.retain_grad()
            encoded.append(tokens)
        return tokens

    monkeypatch.setattr(model.actor, "encode", capture_encode)
    trainer = PPOTrainer(model, config(auxiliary_coef=coefficient))
    head_before = model.actor.auxiliary_head.weight.detach().clone()
    metrics = trainer.update(batch)
    assert batch.critic.grad is None
    if coefficient:
        assert metrics["auxiliary_loss"] > 0
        for name in ("auxiliary_head.weight", "query_embedding", "frame_projection.weight"):
            assert torch.isfinite(gradients[name]).all() and gradients[name].abs().sum() > 0
        assert encoded[-1].grad[:, -1].abs().sum() > 0
        assert not torch.equal(model.actor.auxiliary_head.weight, head_before)
    else:
        assert metrics["auxiliary_loss"] == 0.0
        assert "auxiliary_head.weight" not in gradients
        assert all(grad.count_nonzero() == 0 for grad in gradients.values())
        torch.testing.assert_close(model.actor.auxiliary_head.weight, head_before, rtol=0, atol=0)

"""Behavioral contracts for detached estimation, rollback, resume and export."""
from dataclasses import replace
import copy

import pytest
import torch

from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.export import _synthetic_history, export_policy
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.storage import EstimatorBatch


@pytest.fixture(autouse=True)
def isolated():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7101)
        yield
    torch.set_num_threads(previous)


def config(backbone="mlp", family="velocity", **kwargs):
    return ModelConfig(actor_type=backbone, estimator_type=family,
                       state_indices=(25, 26, 27) if family == "velocity" else (25, 26, 27, 28),
                       readout_type="last" if backbone == "transformer" else "query",
                       time_encoding="index" if backbone == "transformer" else "elapsed",
                       mean_init_scale=0.1, **kwargs)


def batch(model, size=8, invalid_next=False):
    history = _synthetic_history(model.config, [model.config.history_length] * size)
    with torch.no_grad():
        sample = model.actor.act(history)
        critic = torch.randn(size, 29) * 0.1
        critic[:, 28] = 0.3 + critic[:, 28] * 0.1
        value = model.critic(critic)
    return EstimatorBatch(history, critic, sample.action, sample.action.clone(),
                          sample.evaluation.log_prob, sample.evaluation.mean, sample.evaluation.std,
                          value, torch.linspace(-1, 1, size), value + torch.linspace(-0.2, 0.2, size),
                          torch.full((size, 16), float("nan")) if invalid_next else torch.randn(size, 16),
                          torch.full((size,), not invalid_next, dtype=torch.bool))


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
@pytest.mark.parametrize("family", ["velocity", "context"])
def test_current_bypass_and_encoder_gradient_isolation(backbone, family):
    model = ActorCritic(config(backbone, family))
    data = batch(model)
    model.actor(data.history).sum().backward()
    assert all(p.grad is None for p in model.estimator_parameters())
    assert any(p.grad is not None for p in model.actor.controller.parameters())
    assert not ({id(p) for p in model.policy_parameters()} & {id(p) for p in model.estimator_parameters()})
    altered = data.history.clone()
    altered.command.add_(0.1)
    with pytest.raises(ValueError, match="command"):
        model.actor(altered)
    altered.frames[:, -1, 16:19] = altered.command
    assert not torch.equal(model.actor(altered), model.actor(data.history))


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
def test_masked_history_sentinels_and_short_history_equivalence(backbone):
    model = ActorCritic(config(backbone))
    history = _synthetic_history(model.config, [1, 2])
    expected = model.actor(history)
    poisoned = history.clone()
    poisoned.frames[~poisoned.valid] = float("nan")
    poisoned.times[~poisoned.valid] = float("inf")
    assert torch.equal(expected, model.actor(poisoned))
    short = replace(history, frames=history.frames[:, -2:], times=history.times[:, -2:], valid=history.valid[:, -2:])
    assert torch.equal(expected, model.actor(short))


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
@pytest.mark.parametrize("family", ["velocity", "context"])
def test_estimator_updates_preserve_controller_and_reject_reset_targets(backbone, family):
    model = ActorCritic(config(backbone, family))
    trainer = PPOTrainer(model, PPOConfig(estimator_epochs=1, estimator_minibatches=2, estimator_target_kl=10.0))
    data = batch(model, invalid_next=True)
    data.validate()
    controller = copy.deepcopy(model.actor.controller.state_dict())
    before = copy.deepcopy(model.actor.estimator.state_dict())
    metrics = trainer.estimator.update(data)
    same_tree(controller, model.actor.controller.state_dict())
    assert metrics["estimator_accepted_steps"] == 2
    assert metrics["estimator_context_pair_uses"] == 0
    assert any(not torch.equal(before[k], v) for k, v in model.actor.estimator.state_dict().items())


@pytest.mark.parametrize("family", ["velocity", "context"])
def test_rejected_auxiliary_update_restores_weights_adam_and_rng(family):
    model = ActorCritic(config("transformer", family))
    trainer = PPOTrainer(model, PPOConfig(estimator_epochs=1, estimator_minibatches=2, estimator_target_kl=10.0))
    data = batch(model)
    trainer.estimator.update(data)
    trainer.estimator.config = replace(trainer.config, estimator_target_kl=1e-30)
    before = copy.deepcopy(model.state_dict())
    optimizer = copy.deepcopy(trainer.estimator.optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    result = trainer.estimator.update(data)
    assert result["estimator_accepted_steps"] == 0
    assert result["estimator_attempts"] == 3 and result["estimator_attempted_steps"] == 6
    same_tree(before, model.state_dict())
    same_tree(optimizer, trainer.estimator.optimizer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("backbone", ["mlp", "transformer"])
@pytest.mark.parametrize("family", ["velocity", "context"])
def test_full_ppo_resume_preserves_both_optimizers_and_export(tmp_path, backbone, family):
    model = ActorCritic(config(backbone, family))
    trainer = PPOTrainer(model, PPOConfig(epochs=1, num_minibatches=2,
                                        estimator_epochs=1, estimator_minibatches=2))
    result = trainer.update(batch(model))
    assert result["optimizer_steps"] == 2
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, trainer, 1, {})
    restored, resumed, update, _ = load_checkpoint(path)
    assert update == 1 and restored.checkpoint_source_schema_version == 5
    same_tree(model.state_dict(), restored.state_dict())
    same_tree(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
    same_tree(trainer.estimator.optimizer.state_dict(), resumed.estimator.optimizer.state_dict())
    data = batch(model)
    rng = torch.get_rng_state().clone()
    expected = trainer.update(data)
    torch.set_rng_state(rng)
    actual = resumed.update(data)
    assert actual == expected
    same_tree(model.state_dict(), restored.state_dict())
    same_tree(trainer.estimator.optimizer.state_dict(), resumed.estimator.optimizer.state_dict())
    pytest.importorskip("onnxruntime")
    exported = export_policy(path, tmp_path / "policy.onnx")
    assert len(exported["validation"]["cases"]) == 8


def test_corrupt_behavior_is_rejected_before_either_optimizer():
    model = ActorCritic(config("transformer", "context"))
    trainer = PPOTrainer(model, PPOConfig())
    data = batch(model)
    data.old_mean[-1, 0] += 1
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="old_mean"):
        trainer.update(data)
    same_tree(before, model.state_dict())
    assert not trainer.optimizer.state and not trainer.estimator.optimizer.state


@pytest.mark.parametrize("backbone,family,expected", [
    ("mlp", "velocity", 89001), ("transformer", "velocity", 84265),
    ("mlp", "context", 92282), ("transformer", "context", 87546),
])
def test_real_deployment_parameter_budget(backbone, family, expected):
    model = ActorCritic(config(backbone, family))
    assert sum(p.numel() for p in model.actor.parameters()) - 6 == expected
    if family == "context":
        assert sum(p.numel() for p in model.context_objective.parameters()) == 11984

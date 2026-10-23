"""Behavior replay is diagnostic; disabling it must not alter valid PPO updates."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_policy_optimization import assert_identical, batch_for, model
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.settings import FormalTrainingSettings


@pytest.fixture(autouse=True)
def seeded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2102)
        yield
    torch.set_num_threads(threads)


def formal_ppo():
    path = Path(__file__).resolve().parents[1] / 'configs/formal_training.json'
    return FormalTrainingSettings.load(path).ppo


def test_formal_configuration_disables_behavior_replay_and_finishes_all_steps(monkeypatch):
    config = formal_ppo()
    assert config.validate_behavior is False
    assert config.kl_early_stop is False
    network = model()
    trainer = StructuredPPOTrainer(network, config)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('formal training must not replay behavior as an update gate')

    monkeypatch.setattr(trainer, '_validate_behavior', forbidden)
    result = trainer.update(batch_for(network))
    expected_steps = config.epochs * config.minibatches
    assert result['retained_actor_steps'] == result['critic_optimizer_steps'] == expected_steps
    assert result['rolled_back_actor_steps'] == 0
    assert result['actor_stop_reason'] is None


def test_explicit_diagnostic_replay_still_detects_corrupt_saved_log_probability():
    network = model()
    config = replace(formal_ppo(), validate_behavior=True)
    trainer = StructuredPPOTrainer(network, config)
    batch = batch_for(network)
    batch.old_log_prob.add_(.1)
    before = deepcopy(network.state_dict())
    with pytest.raises(ValueError, match='behavior logp mismatch'):
        trainer.update(batch)
    assert_identical(network.state_dict(), before)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    assert trainer.consumed['fresh_endpoints'] == 0


@pytest.mark.parametrize('field', ['old_log_prob', 'old_mean', 'raw_action', 'critic'])
def test_disabling_replay_does_not_accept_nonfinite_learning_data(field):
    network = model()
    trainer = StructuredPPOTrainer(network, formal_ppo())
    batch = batch_for(network)
    getattr(batch, field)[0] = float('nan')
    before = deepcopy(network.state_dict())
    with pytest.raises(ValueError, match='nonfinite meaningful entries'):
        trainer.update(batch)
    assert_identical(network.state_dict(), before)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state


def test_diagnostic_switch_does_not_change_valid_gradients_adam_or_action_rng():
    first = model()
    second = deepcopy(first)
    config = replace(formal_ppo(), epochs=1, minibatches=2)
    reference = StructuredPPOTrainer(first, replace(config, validate_behavior=True))
    ordinary = StructuredPPOTrainer(second, replace(config, validate_behavior=False))
    batch = batch_for(first)
    torch.manual_seed(73)
    left = reference.update(batch)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(73)
    right = ordinary.update(batch)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert_identical(first.state_dict(), second.state_dict())
    assert_identical(reference.actor_optimizer.state_dict(), ordinary.actor_optimizer.state_dict())
    assert_identical(reference.critic_optimizer.state_dict(), ordinary.critic_optimizer.state_dict())
    assert reference.consumed == ordinary.consumed
    for key in ('retained_actor_steps', 'actor_optimizer_steps', 'critic_optimizer_steps',
                'actor_endpoint_uses', 'critic_endpoint_uses', 'rolled_back_actor_steps'):
        assert left[key] == right[key]


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_behavior_validation_switch_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match='explicit bools'):
        replace(StructuredPPOConfig(), validate_behavior=value)

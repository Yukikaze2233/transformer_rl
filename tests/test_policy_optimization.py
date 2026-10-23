"""Synthetic optimizer transactions; no robot training or simulator is started."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.learning.optimization import StructuredBatch, StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig
from transformer_rl.formal.learning.contracts import AuxiliaryLabels, BoundaryMasks, ObservationWindows


@pytest.fixture(autouse=True)
def seeded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(previous)


def model():
    # Transaction algebra uses the same modules at reduced test widths; the
    # production-width parameter/export/causality tests live separately.
    return StructuredActorCritic(StructuredPolicyConfig(body_width=8, body_heads=2, body_ffn=16,
                                                       temporal_width=16, temporal_heads=4,
                                                       temporal_layers=1, temporal_ffn=32))


def batch_for(network, *, owned=True):
    prefix = torch.randn(4, 4, 35)
    observations = torch.randn(8, 4, 35)
    prefix[..., 14:16] = observations[..., 14:16] = 0
    windows = ObservationWindows(prefix, observations, torch.zeros(8, 4, dtype=torch.bool))
    count = len(windows)
    critic = torch.randn(count, network.config.critic_input_dim)
    with torch.no_grad():
        mean = network.actor.policy(windows.gather(torch.arange(count)))
        old_log_std = network.actor.log_std.clamp(-4, 1).expand_as(mean).clone()
        raw_action = mean + torch.randn_like(mean) * old_log_std.exp()
        old_log_prob = network.actor._evaluation(mean, network.actor.log_std, raw_action).log_prob
        value = network.critic(critic)
    valid = torch.full((count,), owned, dtype=torch.bool)
    masks = BoundaryMasks(*(valid.clone() for _ in range(6)))
    labels = AuxiliaryLabels(torch.randn(count, 3), torch.randint(0, 2, (count, 2)).float(),
                             torch.randn(count, 6))
    advantages = torch.randn(count)
    return StructuredBatch(windows, critic, raw_action, raw_action.clamp(-1, 1), mean, old_log_std,
                            old_log_prob, value, advantages, value + advantages, masks, labels)


def assert_identical(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_identical(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_identical(left, right)
    else:
        assert actual == expected


def test_batch_uses_configured_critic_width_and_rejects_old_observations():
    network = model()
    batch = batch_for(network)
    assert batch.critic.shape == (len(batch), 185)
    batch.validate(network)
    with pytest.raises(ValueError, match='critic'):
        replace(batch, critic=batch.critic[:, :81]).validate(network)


def test_default_complete_minibatches_have_64_steps_and_no_sample_refund():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), learning_rate=1e-8))
    batch = batch_for(network)
    metrics = trainer.update(batch)
    assert metrics['actor_optimizer_steps'] == metrics['retained_actor_steps'] == 64
    assert metrics['critic_optimizer_steps'] == 64
    assert metrics['actor_endpoint_uses'] == metrics['critic_endpoint_uses'] == 4 * len(batch)
    assert metrics['fresh_endpoints'] == len(batch)
    assert metrics['rolled_back_actor_steps'] == 0
    assert metrics['cumulative_consumed']['actor_optimizer_steps'] == 64


@pytest.mark.parametrize('training_chunk', [3072, 3])
def test_rejected_actor_step_restores_adam_and_stops_all_auxiliary_updates(monkeypatch, training_chunk):
    network = model()
    config = replace(StructuredPPOConfig(), epochs=1, minibatches=2, learning_rate=1e-4,
                     training_chunk=training_chunk)
    trainer = StructuredPPOTrainer(network, config)
    trainer.update(batch_for(network))
    before = deepcopy(network.actor.state_dict())
    adam_before = deepcopy(trainer.actor_optimizer.state_dict())
    charged_before = trainer.consumed['actor_optimizer_steps']
    original = trainer._kl
    calls = []

    def reject_once(batch, endpoints):
        calls.append(endpoints.numel())
        return batch.old_log_prob.new_tensor(0.03) if len(calls) == 1 else original(batch, endpoints)

    monkeypatch.setattr(trainer, '_kl', reject_once)
    batch = batch_for(network)
    metrics = trainer.update(batch)
    assert_identical(network.actor.state_dict(), before)
    adam_after = deepcopy(trainer.actor_optimizer.state_dict())
    adam_expected = deepcopy(adam_before)
    for state in (adam_after, adam_expected):
        state['param_groups'][0].pop('lr')
    assert_identical(adam_after, adam_expected)
    assert trainer.actor_optimizer.param_groups[0]['lr'] == pytest.approx(
        adam_before['param_groups'][0]['lr'] / 1.5)
    assert metrics['actor_reject_kl_excess_steps'] == 1
    assert metrics['actor_rejected_kl_last'] == pytest.approx(.03)
    assert metrics['actor_optimizer_steps'] == metrics['rolled_back_actor_steps'] == 1
    assert metrics['retained_actor_steps'] == 0
    assert metrics['critic_optimizer_steps'] == 2
    assert trainer.consumed['actor_optimizer_steps'] == charged_before + 1
    assert metrics['actor_endpoint_uses'] == len(batch) // 2


def test_final_rollout_guard_can_reject_changes_missed_by_fixed_probe(monkeypatch):
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2,
                                                    probe_endpoints=8, learning_rate=1e-4))
    before = deepcopy(network.actor.state_dict())
    adam_before = deepcopy(trainer.actor_optimizer.state_dict())
    original = trainer._kl
    full_calls = 0

    def biased_probe(batch, endpoints):
        nonlocal full_calls
        if endpoints.numel() < len(batch):
            return batch.old_log_prob.new_zeros(())
        full_calls += 1
        return batch.old_log_prob.new_tensor(0.03) if full_calls == 1 else original(batch, endpoints)

    monkeypatch.setattr(trainer, '_kl', biased_probe)
    metrics = trainer.update(batch_for(network))
    assert metrics['whole_actor_update_rolled_back']
    assert metrics['actor_optimizer_steps'] == metrics['rolled_back_actor_steps'] == 2
    assert metrics['retained_actor_steps'] == 0
    assert metrics['full_rollout_kl_after_guard'] < 1e-6
    assert_identical(network.actor.state_dict(), before)
    adam_after = deepcopy(trainer.actor_optimizer.state_dict())
    adam_expected = deepcopy(adam_before)
    for state in (adam_after, adam_expected):
        state['param_groups'][0].pop('lr')
    assert_identical(adam_after, adam_expected)
    assert trainer.actor_optimizer.param_groups[0]['lr'] == pytest.approx(
        adam_before['param_groups'][0]['lr'] / 1.5)
    assert metrics['actor_reject_full_rollout_kl_steps'] == 1
    assert metrics['actor_rejected_kl_last'] == pytest.approx(.03)


def test_target_kl_stops_without_undoing_an_admitted_transaction(monkeypatch):
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=2, minibatches=2,
                                                    learning_rate=1e-6))
    before = deepcopy(network.actor.state_dict())
    original = trainer._kl
    calls = 0

    def stop_once(batch, endpoints):
        nonlocal calls
        calls += 1
        return batch.old_log_prob.new_tensor(0.015) if calls == 1 else original(batch, endpoints)

    monkeypatch.setattr(trainer, '_kl', stop_once)
    metrics = trainer.update(batch_for(network))
    assert metrics['actor_optimizer_steps'] == metrics['retained_actor_steps'] == 1
    assert metrics['critic_optimizer_steps'] == 4
    assert metrics['actor_stop_reason'] == 'target_kl_reached'
    assert any(not torch.equal(value, before[name]) for name, value in network.actor.state_dict().items())


def test_disabled_early_stop_runs_all_actor_minibatches():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(
        StructuredPPOConfig(), epochs=1, minibatches=2, learning_rate=1e-4,
        target_kl=1e-12, kl_early_stop=False))
    metrics = trainer.update(batch_for(network))
    assert metrics['actor_optimizer_steps'] == metrics['retained_actor_steps'] == 2
    assert metrics['rolled_back_actor_steps'] == 0
    assert metrics['whole_actor_update_rolled_back'] is False
    assert metrics['actor_stop_reason'] is None
    assert metrics['actor_reject_kl_excess_steps'] == 0
    assert metrics['actor_learning_rate_after_update'] < metrics['actor_learning_rate']


def test_actor_and_critic_steps_share_the_same_lr_schedule(monkeypatch):
    network = model()
    trainer = StructuredPPOTrainer(network, replace(
        StructuredPPOConfig(), epochs=1, minibatches=2, learning_rate=1e-4,
        kl_early_stop=False))
    actor_lrs, critic_lrs = [], []
    original = torch.optim.Adam.step

    def record(optimizer, *args, **kwargs):
        if optimizer is trainer.actor_optimizer:
            actor_lrs.append(optimizer.param_groups[0]['lr'])
        elif optimizer is trainer.critic_optimizer:
            critic_lrs.append(optimizer.param_groups[0]['lr'])
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, 'step', record)
    trainer.update(batch_for(network))
    assert len(actor_lrs) == len(critic_lrs) == 2
    assert actor_lrs == critic_lrs


@pytest.mark.parametrize('training_chunk', [3072, 3])
def test_auxiliary_nonfinite_gradient_is_covered_by_actor_transaction(training_chunk):
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=2, minibatches=2,
                                                    training_chunk=training_chunk))
    batch = batch_for(network)
    before = deepcopy(network.actor.state_dict())
    hook = network.actor.policy.prediction_head[-1].weight.register_hook(lambda g: torch.full_like(g, float('nan')))
    try:
        metrics = trainer.update(batch)
    finally:
        hook.remove()
    assert metrics['actor_stop_reason'] == 'rejected_actor_and_auxiliary_transaction'
    assert metrics['actor_reject_nonfinite_gradient_steps'] == 1
    assert metrics['actor_optimizer_steps'] == metrics['rolled_back_actor_steps'] == 1
    assert metrics['retained_actor_steps'] == 0
    assert metrics['critic_optimizer_steps'] == 4
    assert_identical(network.actor.state_dict(), before)
    assert not trainer.actor_optimizer.state


def test_critical_critic_failure_restores_both_models_without_refunding_consumption():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2,
                                                    learning_rate=1e-6))
    before = deepcopy(network.state_dict())
    batch = batch_for(network)
    hook = network.critic[0].weight.register_hook(lambda g: torch.full_like(g, float('nan')))
    try:
        with pytest.raises(FloatingPointError, match='critic'):
            trainer.update(batch)
    finally:
        hook.remove()
    assert_identical(network.state_dict(), before)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    assert trainer.consumed['fresh_endpoints'] == len(batch)
    assert trainer.consumed['actor_optimizer_steps'] == 2
    # A nonfinite critic gradient is rejected before any critic optimizer step.
    assert trainer.consumed['critic_optimizer_steps'] == 0


def test_script_only_rollout_supplies_no_actor_auxiliary_or_critic_updates():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2))
    batch = batch_for(network, owned=False)
    batch.raw_action.fill_(float('nan'))
    batch.critic.fill_(float('nan'))
    batch.labels.velocity.fill_(float('nan'))
    before = deepcopy(network.state_dict())
    metrics = trainer.update(batch)
    assert metrics['actor_optimizer_steps'] == metrics['critic_optimizer_steps'] == 0
    assert metrics['actor_endpoint_uses'] == metrics['critic_endpoint_uses'] == 0
    assert metrics['fresh_endpoints'] == len(batch)
    assert_identical(network.state_dict(), before)


@pytest.mark.parametrize('field', ['old_mean', 'old_log_std', 'old_log_prob'])
def test_behavior_mismatch_rejected_before_optimization(field):
    network = model()
    trainer = StructuredPPOTrainer(network)
    batch = batch_for(network)
    before = deepcopy(network.state_dict())
    getattr(batch, field).add_(0.1)
    with pytest.raises(ValueError, match='behavior'):
        trainer.update(batch)
    assert_identical(network.state_dict(), before)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state


def test_diagnostics_do_not_change_update_weights_optimizer_or_rng_consumption():
    first = model()
    second = deepcopy(first)
    config = replace(StructuredPPOConfig(), epochs=1, minibatches=2, learning_rate=1e-6)
    left = StructuredPPOTrainer(first, config)
    right = StructuredPPOTrainer(second, replace(config, diagnostics_interval=1))
    batch = batch_for(first)
    torch.manual_seed(33)
    left.update(batch)
    first_rng = torch.random.get_rng_state()
    torch.manual_seed(33)
    metrics = right.update(batch)
    assert_identical(first.state_dict(), second.state_dict())
    assert_identical(left.actor_optimizer.state_dict(), right.actor_optimizer.state_dict())
    assert_identical(left.critic_optimizer.state_dict(), right.critic_optimizer.state_dict())
    assert torch.equal(first_rng, torch.random.get_rng_state())
    assert 'shared_gradient_norm/combined' in metrics
    assert 'shared_gradient_cosine/velocity' in metrics


def test_rollout_targets_are_detached_from_optimization():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2))
    batch = batch_for(network)
    batch.returns.requires_grad_()
    batch.advantages.requires_grad_()
    batch.labels.velocity.requires_grad_()
    trainer.update(batch)
    assert batch.returns.grad is batch.advantages.grad is batch.labels.velocity.grad is None


def test_analytic_kl_is_old_to_new_and_sums_action_dimensions():
    old_mean = torch.randn(8, 6)
    mean = torch.randn(8, 6)
    old_log_std = torch.randn(8, 6) * 0.1
    log_std = torch.randn(8, 6) * 0.2
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_log_std.exp()),
        torch.distributions.Normal(mean, log_std.exp())).sum(-1)
    torch.testing.assert_close(StructuredPPOTrainer.diagonal_kl(old_mean, old_log_std, mean, log_std), expected)


def test_accumulated_update_preserves_masked_gradients_adam_steps_and_sample_budget():
    first = model()
    second = deepcopy(first)
    config = replace(StructuredPPOConfig(), epochs=2, minibatches=2, learning_rate=1e-6)
    left = StructuredPPOTrainer(first, config)
    right = StructuredPPOTrainer(second, replace(config, training_chunk=3))
    batch = batch_for(first)
    indices = torch.arange(len(batch))
    batch.masks.state_auxiliary.copy_(indices % 7 == 0)
    batch.masks.next_auxiliary.copy_(indices % 11 == 0)
    batch.labels.velocity[~batch.masks.state_auxiliary] = float('nan')
    batch.labels.wheel_contact[~batch.masks.state_auxiliary] = float('nan')
    batch.labels.velocity_delta[~batch.masks.next_auxiliary] = float('nan')
    left._actor_backward(batch, indices, batch.advantages)
    right._actor_backward(batch, indices, batch.advantages)
    for full, chunked in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(full.grad, chunked.grad, atol=2e-6, rtol=2e-5)
    torch.manual_seed(73)
    whole = left.update(batch)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(73)
    chunked = right.update(batch)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for key in ('actor_optimizer_steps', 'critic_optimizer_steps', 'actor_endpoint_uses',
                'critic_endpoint_uses', 'retained_actor_steps', 'rolled_back_actor_steps'):
        assert whole[key] == chunked[key]
    assert whole['actor_optimizer_steps'] == 4
    assert right.consumed == left.consumed
    torch.testing.assert_close(first.state_dict(), second.state_dict(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(left.actor_optimizer.state_dict(), right.actor_optimizer.state_dict(),
                               atol=2e-6, rtol=2e-5)

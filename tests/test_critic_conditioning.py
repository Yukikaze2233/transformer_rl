"""Fixed physical preprocessing and pre-step numerical fault coverage."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from transformer_rl.formal.critic import (
    CRITIC182_SLICES, CRITIC_PHYSICAL_TRANSFORM, PrivilegedValueNetwork,
)
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from test_policy_optimization import assert_identical, batch_for, model


@pytest.fixture(autouse=True)
def cpu_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2102)
        yield
    torch.set_num_threads(threads)


def test_grouped_preprocessing_uses_physical_bounds_before_scaling():
    critic = PrivilegedValueNetwork(185, (8, 8, 4))
    raw = torch.zeros(3, 185)
    for name in CRITIC_PHYSICAL_TRANSFORM:
        raw[0, CRITIC182_SLICES[name]] = 1e6
        raw[1, CRITIC182_SLICES[name]] = -1e6
        raw[2, CRITIC182_SLICES[name]] = .1
    before = raw.clone()
    scaled = critic.preprocess(raw)
    for name, (minimum, maximum, factor) in CRITIC_PHYSICAL_TRANSFORM.items():
        group = CRITIC182_SLICES[name]
        torch.testing.assert_close(scaled[:, group], raw[:, group].clamp(minimum, maximum) * factor)
    assert torch.equal(raw, before)
    assert scaled[0, 51].item() == 5.
    assert scaled[0, 167].item() == 10.


def test_actor_terrain_blocks_are_not_scaled_twice_and_estimate_is_detached():
    network = model()
    physical = torch.randn(3, 182, requires_grad=True)
    state = torch.randn(3, 8, requires_grad=True)
    assembled = network.critic_features(physical, state)
    scaled = network.critic.preprocess(assembled)
    for name in ('actor_frame', 'terrain_scan'):
        group = CRITIC182_SLICES[name]
        torch.testing.assert_close(scaled[:, group], physical[:, group])
    network.critic(assembled).sum().backward()
    assert physical.grad is state.grad is None


def test_old_mlp_state_keys_and_adam_remain_loadable_without_rng_or_actor_changes():
    network = model()
    actor = deepcopy(network.actor.state_dict())
    old = nn.Sequential(nn.Linear(185, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                        nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1), nn.Flatten(0))
    optimizer = torch.optim.Adam(old.parameters(), lr=1e-4)
    optimizer.zero_grad()
    old(torch.randn(4, 185)).square().mean().backward()
    optimizer.step()
    network.critic.load_state_dict(old.state_dict(), strict=True)
    restored = torch.optim.Adam(network.critic.parameters(), lr=1e-4)
    restored.load_state_dict(optimizer.state_dict())
    assert_identical(network.critic.state_dict(), old.state_dict())
    assert_identical(restored.state_dict(), optimizer.state_dict())
    assert_identical(network.actor.state_dict(), actor)
    rng = torch.get_rng_state().clone()
    physical = torch.ones(4, 185)
    seen = []
    handle = network.critic[0].register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    network.critic(physical)
    handle.remove()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(seen[0][:, 51:57], torch.full((4, 6), .05))
    # Compatibility preserves tensors, not the old unscaled value function.
    assert not torch.equal(network.critic(physical), old(physical))


@pytest.mark.parametrize('early_stop', [False, True])
def test_nonfinite_critic_gradient_never_reaches_critic_adam(monkeypatch, early_stop):
    network = model()
    trainer = StructuredPPOTrainer(network, replace(
        StructuredPPOConfig(), epochs=1, minibatches=2, learning_rate=1e-8,
        kl_early_stop=early_stop))
    batch = batch_for(network)
    before = deepcopy(network.state_dict())
    calls = []
    original = trainer.critic_optimizer.step

    def step(*args, **kwargs):
        calls.append('critic')
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.critic_optimizer, 'step', step)
    handle = network.critic[6].bias.register_hook(lambda grad: torch.full_like(grad, float('nan')))
    try:
        with pytest.raises(FloatingPointError, match='critic gradient.*before optimizer step'):
            trainer.update(batch)
    finally:
        handle.remove()
    assert calls == []
    assert trainer.critic_optimizer.state == {}
    assert_identical(network.state_dict(), before)
    assert trainer.consumed['fresh_endpoints'] == len(batch)
    if not early_stop:
        assert trainer.consumed['actor_optimizer_steps'] == 0


def test_conditioned_critic_still_updates_with_all_reference_minibatches():
    network = model()
    trainer = StructuredPPOTrainer(network, replace(
        StructuredPPOConfig(), epochs=2, minibatches=2, kl_early_stop=False))
    metrics = trainer.update(batch_for(network))
    assert metrics['actor_optimizer_steps'] == metrics['critic_optimizer_steps'] == 4
    assert metrics['rolled_back_actor_steps'] == 0
    assert metrics['critic_gradient_norm_mean'] >= 0


def test_source_review_does_not_implicitly_allow_policy_preprocessing(tmp_path):
    import hashlib
    from transformer_rl.formal.runtime.transition import _review_sources

    old, new = tmp_path / 'old', tmp_path / 'new'
    for package in (old, new):
        (package / 'formal').mkdir(parents=True)
        (package / 'formal/policy.py').write_text('# before\n')
    (new / 'formal/policy.py').write_text('# explicit preprocessing change\n')
    review = {'formal/policy.py': {
        'before': hashlib.sha256((old / 'formal/policy.py').read_bytes()).hexdigest(),
        'after': hashlib.sha256((new / 'formal/policy.py').read_bytes()).hexdigest(),
    }}
    with pytest.raises(ValueError, match='reviewed'):
        _review_sources(old, new, review)
    assert _review_sources(old, new, review, observation_preprocessing=True) == review

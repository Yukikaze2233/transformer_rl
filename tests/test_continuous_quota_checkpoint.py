"""Continuous quotas restore real saved history without repairing corruption."""
from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from transformer_rl.formal.tasks.curriculum import COMPLETE_STAGE_WEIGHTS, ContinuousTaskSampler


def sampled_state():
    sampler = ContinuousTaskSampler(2102, stage_weights=COMPLETE_STAGE_WEIGHTS, continuous=True)
    for weights in (COMPLETE_STAGE_WEIGHTS[2], COMPLETE_STAGE_WEIGHTS[3]):
        sampler.sample(257, 3, weights=weights)
    return sampler, sampler.state_dict()


def test_valid_quota_restore_preserves_saved_credits_rng_and_draws_exactly():
    sampler, state = sampled_state()
    restored = ContinuousTaskSampler(0, stage_weights=COMPLETE_STAGE_WEIGHTS, continuous=True)
    restored.load_state_dict(state)
    assert restored.state_dict()['dynamic_family_credit'] == state['dynamic_family_credit']
    assert torch.equal(restored.generator.get_state(), sampler.generator.get_state())
    for count in (1, 7, 511):
        left = sampler.sample(count, 3, weights=COMPLETE_STAGE_WEIGHTS[3])
        right = restored.sample(count, 3, weights=COMPLETE_STAGE_WEIGHTS[3])
        for field in fields(left):
            assert torch.equal(getattr(left, field.name), getattr(right, field.name))


@pytest.mark.parametrize('inactive', [False, True])
def test_unbalanced_credit_is_not_silently_repaired_during_restore(inactive):
    sampler, state = sampled_state()
    before = deepcopy(sampler.state_dict())
    damaged = deepcopy(state)
    damaged['dynamic_family_credit'][5 if inactive else 0] += .25
    with pytest.raises(ValueError, match='credit'):
        sampler.load_state_dict(damaged)
    assert sampler.state_dict()['dynamic_family_credit'] == before['dynamic_family_credit']
    assert torch.equal(sampler.generator.get_state(), before['rng'])


def test_balanced_legacy_inactive_credit_is_migrated_without_advancing_rng():
    sampler, state = sampled_state()
    old = deepcopy(state)
    old['dynamic_family_credit'][5] += .25
    old['dynamic_family_credit'][0] -= .25
    assert abs(sum(old['dynamic_family_credit'])) < 1e-6
    sampler.load_state_dict(old)
    after = sampler.state_dict()
    assert after['dynamic_family_credit'][5] == 0
    assert abs(sum(after['dynamic_family_credit'])) < 1e-6
    assert torch.equal(after['rng'], old['rng'])


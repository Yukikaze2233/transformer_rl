import math

import pytest
import torch

from transformer_rl.action_statistics import action_statistics


def test_action_counts_distinguish_clipping_from_exact_bound_and_policy_mean():
    bounds = torch.tensor([1., 3.])
    raw = torch.tensor([[-2., -3.], [-1., 4.], [0., 0.], [1., 2.], [2., -4.]])
    issued = raw.clamp(-bounds, bounds)
    mean = torch.tensor([[0., 0.], [2., 0.], [0., 4.], [0., 0.], [0., 0.]])
    original = tuple(value.clone() for value in (raw, issued, mean, bounds))
    rng = torch.get_rng_state().clone()
    metrics = action_statistics(raw, issued, mean, bounds)
    assert metrics['action_sample_count'] == 5
    assert metrics['raw_action_outside_count_0'] == 2
    assert metrics['raw_action_clip_fraction_0'] == .4
    assert metrics['issued_action_at_bound_count_0'] == 4
    assert metrics['issued_action_at_bound_fraction_0'] == .8
    assert metrics['action_mean_outside_count_0'] == 1
    assert metrics['raw_action_clip_fraction_1'] == .4
    assert metrics['issued_action_at_bound_fraction_1'] == .6
    assert metrics['action_mean_outside_fraction_1'] == .2
    assert metrics['raw_action_mean_0'] == 0
    assert metrics['raw_action_rms_0'] == pytest.approx(math.sqrt(2))
    assert metrics['issued_action_rms_0'] == pytest.approx(math.sqrt(.8))
    assert metrics['issued_action_std_0'] == pytest.approx(math.sqrt(.8))
    for value, before in zip((raw, issued, mean, bounds), original):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_no_samples_are_missing_moments_and_zero_counts():
    empty = torch.empty(0, 2)
    metrics = action_statistics(empty, empty, empty, torch.ones(2))
    assert metrics['action_sample_count'] == 0
    assert all(value == 0 for key, value in metrics.items() if 'count' in key)
    assert all(value is None for key, value in metrics.items() if 'count' not in key)


@pytest.mark.parametrize('bad', ['bounds', 'issued', 'mean', 'raw_shape', 'nan'])
def test_invalid_action_domain_measurements_rejected(bad):
    raw, bounds = torch.tensor([[2., 0.]]), torch.ones(2)
    issued, mean = raw.clamp(-bounds, bounds), torch.zeros_like(raw)
    if bad == 'bounds': bounds[0] = 0
    if bad == 'issued': issued.zero_()
    if bad == 'mean': mean = mean[:, :1]
    if bad == 'raw_shape': raw = raw.flatten()
    if bad == 'nan': raw[0, 0] = float('nan')
    with pytest.raises(ValueError):
        action_statistics(raw, issued, mean, bounds)

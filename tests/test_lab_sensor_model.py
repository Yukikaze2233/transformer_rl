"""Same-clock reset readings keep non-reset sensors and their RNG unchanged."""
from dataclasses import replace

import pytest
import torch

from test_sensor_model import configuration, push
from test_physical_observation import fixture
from transformer_rl.formal.lab_env.sensors import LabSensorModel


def sensor(seed=13):
    _, _, assembler, raw, _ = fixture(4)
    template = assembler.measure(raw)
    model = LabSensorModel(4, seed, configuration(nominal_episode_fraction=0., extra_delay_probability=1.))
    mask = torch.ones(4, dtype=torch.bool)
    now = torch.zeros(4, dtype=torch.float64)
    state = replace(template, timestamp_s=now)
    model.reset(mask, now)
    return model, state


def test_initial_reset_projection_has_current_timestamps_and_no_fake_older_history():
    model, state = sensor()
    original = model.generator.get_state().clone()
    model.seed_after_reset(torch.ones(4, dtype=torch.bool), state)
    output = model(state)
    assert output.sampled_at_s.eq(0).all() and model.count.eq(1).all()
    assert model.reset_projection_rows_consumed == 4
    assert model.noise_rows_consumed == 0
    assert torch.equal(original, model.generator.get_state())
    with pytest.raises(ValueError, match='just-reset'):
        model.seed_after_reset(torch.ones(4, dtype=torch.bool), state)


def test_partial_reset_replaces_only_selected_cached_rows_and_not_normal_noise_rng():
    left, state = sensor()
    right, other = sensor()
    mask = torch.ones(4, dtype=torch.bool)
    left.seed_after_reset(mask, state);right.seed_after_reset(mask, other)
    for _ in range(4):
        a, b = push(left, state), push(right, other)
    old = left(a);right(b)
    before = left.generator.get_state().clone()
    reset = torch.tensor([True, False, False, False])
    left.reset(reset, a.timestamp_s)
    changed = replace(a, policy_velocity_rad_s=torch.zeros(4, 6))
    left.seed_after_reset(reset, changed)
    current = left(changed)
    assert current.sampled_at_s[0].eq(.02).all()
    for name in ('policy_position_rad', 'policy_velocity_rad_s', 'angular_velocity_body_rad_s',
                 'gravity_body', 'sampled_at_s'):
        torch.testing.assert_close(getattr(current, name)[1:], getattr(old, name)[1:], rtol=0, atol=0)
    assert torch.equal(before, left.generator.get_state())
    for _ in range(2):
        a, b = push(left, state), push(right, other)
    first, second = left(a), right(b)
    torch.testing.assert_close(first.policy_velocity_rad_s[1:], second.policy_velocity_rad_s[1:], rtol=0, atol=0)
    assert torch.equal(left.generator.get_state(), right.generator.get_state())


def test_cold_restore_keeps_both_rng_streams_and_discards_reset_cache():
    original, state = sensor()
    original.seed_after_reset(torch.ones(4, dtype=torch.bool), state)
    push(original, state);sample = push(original, state)
    original(sample)
    cold, zero = sensor()
    # A cold model has no initialized physical rows at restoration.
    cold = LabSensorModel(4, original.seed, original.config)
    cold.load_after_explicit_cold_setup(original.checkpoint_state_dict())
    assert cold.count.eq(0).all() and not cold.initialized.any()
    assert cold._sample is None and cold.reset_projection_rows_consumed == 4
    assert torch.equal(original.generator.get_state(), cold.generator.get_state())
    assert torch.equal(original.reset_generator.get_state(), cold.reset_generator.get_state())

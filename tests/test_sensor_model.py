"""Actor传感器时钟与RNG；仅合成测量，不建立SDK或训练进程。"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_physical_observation import fixture as physical_fixture
from transformer_rl.formal.control.sensors import SensorModel, SensorModelConfig


def configuration(**changes):
    return replace(SensorModelConfig(), **changes)


def no_noise(**changes):
    return configuration(leg_position_halfwidth_rad=0., leg_velocity_halfwidth_rad_s=0.,
                         wheel_velocity_halfwidth_rad_s=0., gyro_halfwidth_rad_s=0.,
                         gravity_halfwidth=0., **changes)


def fixture(count=4, config=None, seed=211):
    _, _, assembler, raw, _ = physical_fixture(count)
    measured = assembler.measure(raw)
    model = SensorModel(count, seed, config or configuration())
    model.reset(torch.ones(count, dtype=torch.bool), torch.zeros(count, dtype=torch.float64))
    return model, measured


def push(model, template, *, value=None):
    tick = model.physical_tick + 1
    scalar = float(tick) if value is None else value
    state = replace(template, policy_position_rad=torch.full((model.num_envs, 6), scalar),
                    policy_velocity_rad_s=torch.full((model.num_envs, 6), scalar),
                    angular_velocity_body_rad_s=torch.full((model.num_envs, 3), scalar),
                    gravity_body=torch.full((model.num_envs, 3), scalar),
                    timestamp_s=torch.full((model.num_envs,), tick * model.config.physics_dt_s,
                                           dtype=torch.float64))
    model.push(state, tick)
    return state


def test_past_only_group_delays_keep_real_timestamps_and_do_not_expose_wheel_angle():
    model, template = fixture(config=no_noise(nominal_episode_fraction=0., extra_delay_probability=1.,
                                              observation_delays_s=(.005, .01, .015)))
    for _ in range(6):
        state = push(model, template)
    before = deepcopy(state)
    output = model(state)
    expected = model.physical_tick - model.lags
    torch.testing.assert_close(output.sampled_at_s, expected.double() * .005)
    for values, group in ((output.policy_position_rad[:, :4], 0), (output.policy_velocity_rad_s, 1),
                           (output.angular_velocity_body_rad_s, 2), (output.gravity_body, 3)):
        torch.testing.assert_close(values, expected[:, group, None].float().expand_as(values))
    assert output.policy_position_rad[:, 4:].eq(0).all()
    assert state.policy_position_rad.eq(before.policy_position_rad).all()


def test_noise_is_applied_once_per_policy_tick_after_delay_in_physical_units():
    model, template = fixture(config=configuration(nominal_episode_fraction=0., extra_delay_probability=1.,
                                                   observation_delays_s=(.015,)))
    for _ in range(4):
        state = push(model, template)
    global_rng = torch.random.get_rng_state()
    output = model(state)
    rng = model.generator.get_state()
    baseline = torch.ones(4, 18)
    baseline[:, 4:6] = 0
    actual = torch.cat((output.policy_position_rad, output.policy_velocity_rad_s,
                        output.angular_velocity_body_rad_s, output.gravity_body), -1)
    delta = actual - baseline
    assert (delta.abs() <= model._halfwidth + 1e-6).all() and delta[:, 4:6].eq(0).all()
    assert delta[:, 12:15].abs().sum() > 0
    output.policy_position_rad.fill_(-999)
    repeated = model(state)
    assert not repeated.policy_position_rad.eq(-999).any()
    assert torch.equal(rng, model.generator.get_state())
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert model.noise_rows_consumed == model.num_envs
    push(model, template)
    state = push(model, template)
    model(state)
    assert model.noise_rows_consumed == 2 * model.num_envs


def test_nominal_episodes_have_no_extra_sensor_delay_or_noise():
    model, template = fixture(config=configuration(nominal_episode_fraction=1., extra_delay_probability=1.))
    push(model, template)
    state = push(model, template)
    output = model(state)
    assert model.nominal.all() and model.lags.eq(0).all()
    assert output.policy_velocity_rad_s.eq(2).all()
    assert output.angular_velocity_body_rad_s.eq(2).all()
    assert output.sampled_at_s.eq(.01).all()


def test_partial_reset_flushes_only_new_episode_rows_and_never_reuses_old_sample_times():
    model, template = fixture(config=no_noise(nominal_episode_fraction=0., extra_delay_probability=1.,
                                              observation_delays_s=(.015,)))
    for _ in range(4):
        state = push(model, template)
    model(state)
    retained = model.values[:, 1:].clone()
    model.reset(torch.tensor([True, False, False, False]), state.timestamp_s)
    assert torch.equal(model.values[:, 1:], retained)
    with pytest.raises(ValueError, match='post-reset'):
        model(state)
    push(model, template)
    state = push(model, template)
    output = model(state)
    assert output.sampled_at_s[0].eq(.025).all()
    assert output.sampled_at_s[1:].eq(.015).all()
    assert output.policy_velocity_rad_s[0].eq(5).all()
    assert output.policy_velocity_rad_s[1:].eq(3).all()
    assert model.reset_rows_consumed == model.num_envs + 1


def test_cold_restore_retains_independent_rng_and_counts_but_discards_physical_cache():
    original, template = fixture()
    push(original, template, value=2.)
    state = push(original, template, value=2.)
    original(state)
    snapshot = original.checkpoint_state_dict()
    global_rng = torch.random.get_rng_state()
    cold = SensorModel(original.num_envs, original.seed, original.config)
    cold.load_after_explicit_cold_setup(snapshot)
    assert cold.count.eq(0).all() and not cold.initialized.any() and cold._sample is None
    assert torch.equal(global_rng, torch.random.get_rng_state())
    mask = torch.ones(original.num_envs, dtype=torch.bool)
    original.reset(mask, state.timestamp_s)
    cold.reset(mask, torch.zeros(cold.num_envs, dtype=torch.float64))
    assert torch.equal(original.lags, cold.lags) and torch.equal(original.nominal, cold.nominal)
    push(original, template, value=2.)
    push(cold, template, value=2.)
    first = original(push(original, template, value=2.))
    second = cold(push(cold, template, value=2.))
    torch.testing.assert_close(first.policy_velocity_rad_s, second.policy_velocity_rad_s)
    assert torch.equal(original.generator.get_state(), cold.generator.get_state())
    assert torch.equal(original.reset_generator.get_state(), cold.reset_generator.get_state())
    assert original.noise_rows_consumed == cold.noise_rows_consumed
    with pytest.raises(ValueError, match='empty cold cache'):
        cold.load_after_explicit_cold_setup(snapshot)


@pytest.mark.parametrize('change', ['seed', 'config', 'rng', 'reset_rng', 'counter'])
def test_corrupt_restore_is_rejected_without_advancing_the_new_model_rng(change):
    original, _ = fixture()
    snapshot = original.checkpoint_state_dict()
    if change == 'seed':snapshot['seed'] += 1
    elif change == 'config':snapshot['config']['gyro_halfwidth_rad_s'] *= 2
    elif change == 'rng':snapshot['generator_state'] = torch.zeros(3)
    elif change == 'reset_rng':snapshot['reset_generator_state'] = torch.zeros(3)
    else:snapshot['noise_rows_consumed'] = -1
    cold = SensorModel(original.num_envs, original.seed, original.config)
    before = cold.generator.get_state()
    reset_before = cold.reset_generator.get_state()
    with pytest.raises(ValueError):cold.load_after_explicit_cold_setup(snapshot)
    assert torch.equal(before, cold.generator.get_state())
    assert torch.equal(reset_before, cold.reset_generator.get_state())


def test_repeated_or_out_of_order_physical_samples_and_non_policy_reads_are_not_new_observations():
    model, template = fixture()
    state = push(model, template)
    rng = model.generator.get_state()
    with pytest.raises(ValueError, match='physical step'):
        model.push(state, 1)
    with pytest.raises(ValueError, match='post-reset'):
        model(state)
    assert torch.equal(rng, model.generator.get_state())
    state = push(model, template)
    model(state)
    with pytest.raises(ValueError, match='same-tick'):
        model.push(replace(state, timestamp_s=state.timestamp_s + .5), 3)
    assert model.physical_tick == 2


def test_reference_clock_preserves_delays_in_seconds_without_changing_the_policy_clock():
    model, template = fixture(config=no_noise(physics_dt_s=.001, nominal_episode_fraction=0.,
                                              extra_delay_probability=1., observation_delays_s=(.015,)))
    for _ in range(20):state = push(model, template)
    output = model(state)
    assert model.policy_substeps == 10 and model.capacity == 16
    assert output.sampled_at_s.eq(.005).all() and model.lags.eq(15).all()


def test_one_row_reset_does_not_shift_other_rows_future_sensor_noise():
    config = configuration(nominal_episode_fraction=0., extra_delay_probability=1.)
    left, template = fixture(config=config)
    right, _ = fixture(config=config)
    for _ in range(4):
        first, second = push(left, template), push(right, template)
    left(first)
    right(second)
    left.reset(torch.tensor([True, False, False, False]), first.timestamp_s)
    for _ in range(2):
        first, second = push(left, template), push(right, template)
    a, b = left(first), right(second)
    for name in ('policy_position_rad', 'policy_velocity_rad_s', 'angular_velocity_body_rad_s',
                 'gravity_body', 'sampled_at_s'):
        assert torch.equal(getattr(a, name)[1:], getattr(b, name)[1:])
    assert torch.equal(left.generator.get_state(), right.generator.get_state())

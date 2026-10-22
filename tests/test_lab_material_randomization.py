"""Startup material writes preserve native shape ownership and cold identity."""
from dataclasses import asdict

import pytest
import torch

from test_lab_materials import reader, robot
from transformer_rl.formal.lab_env.materials import WheelMaterialConfig


def native_writer(robot, *, corrupt=None):
    writes = []

    def write(values, indices):
        assert values.dtype == torch.float32 and values.device.type == 'cpu'
        assert values.is_contiguous()
        assert indices.dtype == torch.int32 and indices.tolist() == [0, 1, 2]
        writes.append(values.clone())
        robot.root_view.values.copy_(values)
        if corrupt is not None:
            corrupt(robot.root_view.values)

    robot.root_view.set_material_properties = write
    return writes


def apply(value, config=None, *, seed=31):
    value.randomize_startup(config or WheelMaterialConfig(), seed=seed,
                            tensor_to_native=lambda tensor: tensor)


def test_native_write_matches_cached_critic_and_preserves_non_wheel_shapes(robot):
    writes = native_writer(robot)
    value = reader(robot)
    original = robot.root_view.values.clone()
    rng = torch.get_rng_state().clone()
    apply(value)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(writes) == 1
    assert torch.equal(robot.root_view.values[:, 0], original[:, 0])
    for wheel, (start, end) in enumerate(((1, 3), (3, 5))):
        expected = value.features()[:, wheel * 3:(wheel + 1) * 3]
        assert torch.equal(robot.root_view.values[:, start:end], expected[:, None].expand(-1, 2, -1))
        assert (expected[:, 1] <= expected[:, 0]).all()
    receipt = value.receipt['startup_randomization']
    assert receipt['non_wheel_shapes_unchanged'] and not receipt['terrain_materials_changed']
    assert not receipt['reset_resampling'] and not receipt['effective_contact_coefficient']
    assert [sum(histogram) for histogram in receipt['bucket_histogram_by_wheel']] == [3, 3]
    reads = robot.root_view.reads
    for _ in range(100):
        value.features()
        value.receipt
    assert robot.root_view.reads == reads
    robot.root_view.values[:, 1:3] = .123
    assert not torch.equal(value.features()[:, :3], robot.root_view.values[:, 1])
    with pytest.raises(RuntimeError, match='already attempted'):
        apply(value)
    assert len(writes) == 1


def test_cold_draw_is_reproducible_independent_of_global_rng_and_default_dtype(robot):
    native_writer(robot)
    first = reader(robot)
    apply(first)
    expected = first.features().clone()
    fingerprint = first.receipt['startup_randomization']['native_parameters_sha256']
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        second = reader(robot)
        apply(second)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(second.features(), expected)
    assert second.receipt['startup_randomization']['native_parameters_sha256'] == fingerprint
    different = reader(robot)
    apply(different, seed=32)
    assert different.receipt['startup_randomization']['native_parameters_sha256'] != fingerprint


def test_explicit_lower_endpoint_has_truthful_readback_and_json_configuration(robot):
    native_writer(robot)
    value = reader(robot)
    config = WheelMaterialConfig(static_friction=(.1, 1.2), dynamic_friction=(.1, 1.),
                                 num_buckets=1, anchor_lower_endpoint=True)
    assert WheelMaterialConfig.from_dict(asdict(config)) == config
    apply(value, config)
    expected = torch.tensor([.1, .1, .02, .1, .1, .02]).expand(3, -1)
    torch.testing.assert_close(value.features(), expected, rtol=0, atol=0)
    receipt = value.receipt['startup_randomization']
    assert receipt['actual_min'] == receipt['actual_max'] == expected[0, :3].tolist()


@pytest.mark.parametrize('which_shape', (0, 1, 4))
def test_full_readback_catches_wrong_wheel_and_non_wheel_writes_before_cache_publish(robot, which_shape):
    def corrupt(values):
        values[:, which_shape, 0] += .01

    native_writer(robot, corrupt=corrupt)
    value = reader(robot)
    previous = value.features().clone()
    with pytest.raises(RuntimeError, match='full shape table'):
        apply(value)
    assert torch.equal(value.features(), previous)
    assert 'startup_randomization' not in value.receipt
    with pytest.raises(RuntimeError, match='already attempted'):
        apply(value)


@pytest.mark.parametrize('change', ({'num_buckets': True}, {'num_buckets': 0}, {'num_buckets': 257},
                                    {'anchor_lower_endpoint': 1}, {'static_friction': (.1, .5)},
                                    {'dynamic_friction': (float('nan'), 1.)},
                                    {'static_friction': (-.1, 1.)}, {'restitution': (0., 1.1)}))
def test_invalid_palette_cannot_be_constructed(change):
    with pytest.raises(ValueError):
        WheelMaterialConfig(**change)


def test_missing_setter_and_wrong_native_dtype_are_rejected_before_write(robot):
    value = reader(robot)
    with pytest.raises(ValueError, match='setter'):
        apply(value)
    writes = native_writer(robot)
    robot.root_view.values = robot.root_view.values.double()
    with pytest.raises(ValueError, match='float32'):
        apply(value)
    assert not writes


def test_runtime_material_configuration_rejects_silent_unknown_or_missing_fields():
    config = asdict(WheelMaterialConfig())
    config['other_body_friction'] = .5
    with pytest.raises(ValueError, match='fields differ'):
        WheelMaterialConfig.from_dict(config)
    config.pop('other_body_friction')
    config.pop('num_buckets')
    with pytest.raises(ValueError, match='fields differ'):
        WheelMaterialConfig.from_dict(config)

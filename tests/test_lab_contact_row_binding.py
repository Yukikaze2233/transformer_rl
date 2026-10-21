"""Raw PhysX row ordering cannot exchange bodies or environments."""
from types import SimpleNamespace

import pytest
import torch

from test_physical_observation import fixture
from transformer_rl.formal.lab_env.contact_measurement import LabContactLoads


def contact_fixture(layout, *, nested=False):
    _, binding, *_ = fixture()
    names = tuple(reversed(binding.native_body_names))
    envs = tuple(f'/World/envs/env_{index}' for index in range(3))
    pairs = ([(env, name) for env in envs for name in names] if layout == 'environment_major'
             else [(env, name) for name in names for env in envs])
    prefix = '/Robot/tree_joints/' if nested else '/Robot/'
    paths = tuple(env + prefix + name for env, name in pairs)
    filters = tuple(f'/World/ground/surface_{index:02d}' for index in range(22))
    size = len(paths)
    target = [envs.index(env) * len(names) + binding.native_body_names.index(name) for env, name in pairs]
    force = torch.full((size + 1,), float('nan'))
    force[:size] = torch.tensor(target, dtype=torch.float32) + 1
    normals = torch.zeros(size + 1, 3)
    normals[:size, 2] = 1
    for row, (_, name) in enumerate(pairs):
        if name == binding.native_body_names[binding.wheel_body_indices[0]]:
            normals[row] = torch.tensor([1., 0., 0.])
    counts = torch.zeros(size, len(filters), dtype=torch.int32)
    starts = torch.zeros_like(counts)
    for row in range(size):
        counts[row, row % len(filters)] = 1
        starts[row, row % len(filters)] = row
    reads = []

    def read(*, dt):
        reads.append(dt)
        return force, None, normals, None, counts, starts

    view = SimpleNamespace(
        sensor_paths=paths, sensor_count=size, filter_count=len(filters),
        filter_paths=(filters,) * size, max_contact_data_count=size + 1, get_contact_data=read)
    sensor = SimpleNamespace(contact_view=view, body_names=names,
                             body_physx_view=SimpleNamespace(prim_paths=paths))
    return sensor, binding, envs, filters, reads


def construct(sensor, binding, envs, filters):
    return LabContactLoads(sensor, binding, envs, collision_path=filters, device='cpu')


@pytest.mark.parametrize('nested', [False, True])
def test_environment_and_body_major_patch_rows_produce_identical_canonical_loads(nested):
    left, binding, envs, filters, _ = contact_fixture('environment_major', nested=nested)
    right, *_ = contact_fixture('body_major', nested=nested)
    a, b = construct(left, binding, envs, filters), construct(right, binding, envs, filters)
    normal, support = a.read(1, .005)
    other_normal, other_support = b.read(1, .005)
    expected = torch.arange(1, normal.numel() + 1, dtype=torch.float32).view_as(normal)
    torch.testing.assert_close(normal, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_normal, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_support, support, rtol=0, atol=0)
    assert support[:, 0].eq(0).all()  # Horizontal riser impact is never upward wheel support.
    torch.testing.assert_close(support[:, 1], expected[:, binding.wheel_body_indices[1]])


def test_body_major_partial_reset_does_not_clear_other_environments_or_refetch_the_same_tick():
    sensor, binding, envs, filters, reads = contact_fixture('body_major', nested=True)
    loads = construct(sensor, binding, envs, filters)
    normal, support = loads.read(1, .005)
    expected_normal, expected_support = normal.clone(), support.clone()
    loads.reset_rows(torch.tensor([1]), 1)
    current, current_support = loads.read(1, .005)
    assert reads == [.005]
    assert loads.valid.tolist() == [True, False, True]
    assert current[1].eq(0).all() and current_support[1].eq(0).all()
    torch.testing.assert_close(current[[0, 2]], expected_normal[[0, 2]])
    torch.testing.assert_close(current_support[[0, 2]], expected_support[[0, 2]])
    refreshed, refreshed_support = loads.read(2, .005)
    assert reads == [.005, .005]
    torch.testing.assert_close(refreshed, expected_normal)
    torch.testing.assert_close(refreshed_support, expected_support)


@pytest.mark.parametrize('fault', [
    'duplicate', 'other_environment', 'missing_body', 'duplicate_name', 'nested_alias',
    'sensor_order', 'sensor_count', 'filter_missing', 'filter_duplicate', 'filter_other',
])
def test_invalid_sensor_or_filter_binding_fails_before_a_patch_read(fault):
    sensor, binding, envs, filters, reads = contact_fixture('body_major', nested=True)
    paths = list(sensor.body_physx_view.prim_paths)
    if fault == 'duplicate':
        paths[-1] = paths[0]
    elif fault == 'other_environment':
        paths[-1] = paths[-1].replace('/env_2/', '/env_3/')
    elif fault == 'missing_body':
        paths.pop()
    elif fault == 'duplicate_name':
        sensor.body_names = (*sensor.body_names[:-1], sensor.body_names[0])
    elif fault == 'nested_alias':
        paths[-1] = paths[-1].replace('/tree_joints/', '/another_subtree/')
    elif fault == 'sensor_order':
        sensor.contact_view.sensor_paths = tuple(reversed(paths))
    elif fault == 'sensor_count':
        sensor.contact_view.sensor_count -= 1
    else:
        rows = list(sensor.contact_view.filter_paths)
        row = list(rows[-1])
        if fault == 'filter_missing':
            row.pop()
        elif fault == 'filter_duplicate':
            row[-1] = row[0]
        else:
            row[-1] = '/World/other_training/terrain'
        rows[-1] = tuple(row)
        sensor.contact_view.filter_paths = tuple(rows)
    if fault not in ('sensor_order', 'sensor_count'):
        sensor.body_physx_view.prim_paths = tuple(paths)
        sensor.contact_view.sensor_paths = tuple(paths)
    with pytest.raises(ValueError, match='exact robot bodies'):
        construct(sensor, binding, envs, filters)
    assert not reads


def test_runtime_uses_the_compiled_map_without_reaccessing_python_path_metadata():
    sensor, binding, envs, filters, _ = contact_fixture('body_major')
    loads = construct(sensor, binding, envs, filters)
    sensor.body_physx_view.prim_paths = None
    sensor.contact_view.sensor_paths = None
    sensor.contact_view.filter_paths = None
    normal, _ = loads.read(1, .005)
    torch.testing.assert_close(normal, torch.arange(1, normal.numel() + 1).float().view_as(normal))

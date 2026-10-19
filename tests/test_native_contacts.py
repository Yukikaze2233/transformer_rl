"""Shared native contact buffer checks; no simulator is imported or launched."""
from types import SimpleNamespace

import pytest
import torch

from test_physical_observation import fixture as observation_fixture
from transformer_rl.formal.simulation.contacts import ContactCapacityError, NativeTerrainContacts


def fixture():
    _, binding, *_ = observation_fixture()
    views = []

    def create(sensors, *, filter_patterns, max_contact_data_count):
        sensor_count, filter_count = len(sensors), len(filter_patterns[0])
        matrix = torch.zeros(sensor_count, filter_count, 3)
        counts = torch.zeros(sensor_count, filter_count, dtype=torch.int32)
        starts = torch.full_like(counts, -1)
        # Entries for different sensors share the same allocation. Empty pairs
        # may contain sentinel offsets and unused storage need not be initialized.
        force = torch.full((max_contact_data_count, 1), float('nan'))
        for row in range(sensor_count):
            force[row, 0] = row + 1
            counts[row, row % filter_count] = 1
            starts[row, row % filter_count] = row
            matrix[row, :, 2] = row + 1
        view = SimpleNamespace(sensor_count=sensor_count, filter_count=filter_count,
                               sensor_paths=sensors, filter_paths=filter_patterns,
                               max_contact_data_count=max_contact_data_count,
                               matrix=matrix, force=force, counts=counts, starts=starts)
        view.get_contact_force_matrix = lambda *, dt: view.matrix
        view.get_contact_data = lambda *, dt: (view.force, None, None, None, view.counts, view.starts)
        views.append(view)
        return view

    # Native clone order is deliberately different from lexicographic order.
    envs = ('/World/envs/env_2', '/World/envs/env_0', '/World/envs/env_1')
    surfaces = tuple(tuple(env + '/Terrain/mesh_' + str(i) for i in range(size))
                     for env, size in zip(envs, (2, 1, 2)))
    physics = SimpleNamespace(create_rigid_contact_view=create)
    contacts = NativeTerrainContacts(physics, binding, envs, surfaces,
                                     native_to_tensor=lambda value: value)
    return binding, contacts, views, envs, surfaces, physics


def test_grouped_contacts_preserve_native_clone_order_and_shared_buffer_indices():
    binding, contacts, views, envs, surfaces, _ = fixture()
    assert len(views) == 2
    assert views[0].sensor_paths[0].startswith(envs[1] + '/Robot/')
    assert views[1].sensor_paths[0].startswith(envs[0] + '/Robot/')
    assert views[1].sensor_paths[len(binding.native_body_names)].startswith(envs[2] + '/Robot/')
    assert tuple(views[1].filter_paths[0]) == surfaces[0]
    timestamp = torch.full((3,), .005, dtype=torch.float64)
    result = contacts.read(.005, timestamp)
    bodies = len(binding.native_body_names)
    torch.testing.assert_close(result.normal_load_n[0], torch.arange(1, bodies + 1).float())
    torch.testing.assert_close(result.normal_load_n[1], torch.arange(1, bodies + 1).float())
    torch.testing.assert_close(result.normal_load_n[2], torch.arange(bodies + 1, bodies * 2 + 1).float())
    torch.testing.assert_close(result.force_world_n[0, :, 2], result.normal_load_n[0] * 2)
    assert result.patch_count.eq(1).all()
    timestamp.zero_()
    for view in views:
        view.matrix.zero_()
        view.force.zero_()
    assert result.normal_load_n.any() and result.force_world_n.any() and result.timestamp_s.any()


def test_positive_patch_load_does_not_disappear_when_opposing_forces_cancel():
    _, contacts, views, *_ = fixture()
    two_surfaces = views[1]
    two_surfaces.matrix[:, 0, 2] = 10
    two_surfaces.matrix[:, 1, 2] = -10
    result = contacts.read(.005, torch.full((3,), .005, dtype=torch.float64))
    assert not result.force_world_n[[0, 2]].any()
    assert result.normal_load_n[[0, 2]].gt(0).all()


def test_empty_pair_sentinel_and_unused_nan_padding_do_not_affect_patch_load():
    force = torch.tensor([float('nan'), 2., 3., float('nan'), 7., float('nan')])
    counts = torch.tensor([[2, 0], [0, 1]], dtype=torch.int32)
    starts = torch.tensor([[1, -1], [99999, 4]], dtype=torch.int32)
    load, count = NativeTerrainContacts._patch_loads(force, counts, starts, 10)
    torch.testing.assert_close(load, torch.tensor([5., 7.]))
    torch.testing.assert_close(count, torch.tensor([2, 1]))


@pytest.mark.parametrize('change', ['negative_count', 'negative_start', 'past_buffer', 'overlap',
                                    'active_nan', 'negative_force', 'load_overflow', 'wrong_dtype'])
def test_invalid_active_patch_data_is_rejected(change):
    force = torch.tensor([2., 3., 4., float('nan')])
    counts = torch.tensor([[1, 1]], dtype=torch.int32)
    starts = torch.tensor([[0, 1]], dtype=torch.int32)
    if change == 'negative_count':
        counts[0, 0] = -1
    elif change == 'negative_start':
        starts[0, 0] = -1
    elif change == 'past_buffer':
        starts[0, 0] = 4
    elif change == 'overlap':
        starts[0, 1] = 0
    elif change == 'active_nan':
        force[0] = float('nan')
    elif change == 'negative_force':
        force[0] = -1
    elif change == 'load_overflow':
        force[:2] = 3e38
    else:
        counts = counts.float()
    with pytest.raises(ValueError):
        NativeTerrainContacts._patch_loads(force, counts, starts, 10)


def test_capacity_reached_is_a_measurement_error_instead_of_a_healthy_contact_result():
    with pytest.raises(ContactCapacityError):
        NativeTerrainContacts._patch_loads(torch.tensor([2., 3.]), torch.tensor([[2]]),
                                           torch.tensor([[0]]), 2)


@pytest.mark.parametrize('change', ['sensor_order', 'filter_order', 'capacity', 'sensor_count'])
def test_changed_native_view_identity_is_rejected(change):
    _, contacts, views, *_ = fixture()
    view = views[1]
    if change == 'sensor_order':
        view.sensor_paths = list(reversed(view.sensor_paths))
    elif change == 'filter_order':
        view.filter_paths[0].reverse()
    elif change == 'capacity':
        view.max_contact_data_count += 1
    else:
        view.sensor_count += 1
    with pytest.raises(RuntimeError):
        contacts.read(.005, torch.full((3,), .005, dtype=torch.float64))


@pytest.mark.parametrize('change', ['neighbor_filter', 'broad_filter', 'empty_filter', 'repeated_env'])
def test_only_exact_environment_owned_filters_can_be_bound(change):
    binding, _, _, envs, surfaces, physics = fixture()
    surfaces = list(surfaces)
    if change == 'neighbor_filter':
        surfaces[0] = surfaces[1]
    elif change == 'broad_filter':
        surfaces[0] = (envs[0] + '/Terrain/.*',)
    elif change == 'empty_filter':
        surfaces[0] = ()
    else:
        envs = (envs[0], envs[0], envs[2])
    with pytest.raises(ValueError):
        NativeTerrainContacts(physics, binding, envs, surfaces, native_to_tensor=lambda value: value)


@pytest.mark.parametrize('change', ['dt', 'clock_dtype', 'negative_time', 'matrix_nan', 'net_overflow'])
def test_invalid_clock_or_force_cannot_be_published(change):
    _, contacts, views, *_ = fixture()
    timestamp, dt = torch.full((3,), .005, dtype=torch.float64), .005
    if change == 'dt':
        dt = .01
    elif change == 'clock_dtype':
        timestamp = timestamp.float()
    elif change == 'negative_time':
        timestamp[0] = -1
    elif change == 'matrix_nan':
        views[0].matrix[0, 0, 0] = float('nan')
    else:
        views[1].matrix[:, :, 0] = 3e38
    with pytest.raises(ValueError):
        contacts.read(dt, timestamp)

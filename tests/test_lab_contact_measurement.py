"""A wheel hitting a vertical riser cannot acquire upward support."""
from types import SimpleNamespace

import pytest
import torch

from test_physical_observation import fixture
from transformer_rl.formal.lab_env.contact_measurement import LabContactLoads, LabObservationAssembler, LabPhysicalSample
from transformer_rl.formal.lab_env.formal_reward import reward_physics_from_lab
from transformer_rl.formal.simulation.contacts import ContactCapacityError


def patches():
    # Distinct sensors: upward floor, horizontal riser, opposing contacts.
    force = torch.tensor([10., 20., 7., 7., float('nan')])
    normals = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [0., 0., -1.],
                            [float('nan'), 0., 0.]])
    counts = torch.tensor([[1], [1], [2]], dtype=torch.int32)
    starts = torch.tensor([[0], [1], [2]], dtype=torch.int32)
    return force, normals, counts, starts


def test_patch_sum_and_support_are_distinct_from_net_force():
    normal, upward = LabContactLoads.reduce(*patches(), 5)
    torch.testing.assert_close(normal, torch.tensor([10., 20., 14.]))
    torch.testing.assert_close(upward, torch.tensor([10., 0., 7.]))


def test_inclined_patch_support_is_projected_in_world_z():
    force, normals, counts, starts = patches()
    normals[0] = torch.tensor([.6, 0., .8])
    normal, upward = LabContactLoads.reduce(force, normals, counts, starts, 5)
    assert normal[0] == 10 and upward[0] == 8


@pytest.mark.parametrize('bad', ('direction', 'range', 'capacity'))
def test_invalid_contact_measurement_never_becomes_a_zero_load(bad):
    force, normals, counts, starts = patches()
    capacity = 5
    if bad == 'direction':
        normals[1] = 0
    elif bad == 'range':
        starts[1] = 900
    else:
        capacity = 4
    with pytest.raises((ValueError, ContactCapacityError)):
        LabContactLoads.reduce(force, normals, counts, starts, capacity)


def test_critic_labels_and_reward_use_the_same_patch_sample_without_actor_leakage():
    _, binding, _, original, command = fixture()
    assembler = LabObservationAssembler(binding, 2)
    body = torch.zeros(2, 19)
    body[:, list(binding.wheel_body_indices)] = torch.tensor([20., 10.])
    support = torch.tensor([[0., 10.], [0., 10.]])
    sample = LabPhysicalSample(**original.__dict__, body_normal_load_n=body, wheel_support_load_n=support,
                               contact_measurement_valid=torch.ones(2, dtype=torch.bool))
    state = assembler.measure(sample)
    torch.testing.assert_close(state.wheel_ground_force_n, support)
    kwargs = dict(effort_nm=torch.zeros(2, 6), episode_id=torch.zeros(2, dtype=torch.long),
                  control_epoch=torch.zeros(2, dtype=torch.long), policy_owned=torch.ones(2, dtype=torch.bool),
                  available_terms=(), normal_load_source='native_contact_patch_normal_forces')
    physics = reward_physics_from_lab(sample, state, **kwargs)
    torch.testing.assert_close(physics.body_normal_load_n, body)
    with pytest.raises(ValueError, match='actual positive native patch loads'):
        reward_physics_from_lab(original, state, **kwargs)
    sensors = assembler.clean_sensors(state)
    before = assembler.actor(sensors, command, now_s=state.timestamp_s)
    changed = LabPhysicalSample(**original.__dict__, body_normal_load_n=body * 10,
                                wheel_support_load_n=support * 10,
                                contact_measurement_valid=torch.ones(2, dtype=torch.bool))
    changed_state = assembler.measure(changed)
    after = assembler.actor(assembler.clean_sensors(changed_state), command, now_s=state.timestamp_s)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_partial_reset_clears_only_reset_support_until_a_new_physical_tick():
    _, binding, *_ = fixture()
    names = tuple(reversed(binding.native_body_names))
    envs = ('/World/envs/env_0', '/World/envs/env_1')
    paths = tuple(env + '/Robot/' + name for env in envs for name in names)
    sensors = len(paths)
    force = torch.full((sensors + 1,), float('nan'))
    normals = torch.zeros(sensors + 1, 3)
    normals[:sensors, 2] = 1
    force[:sensors] = torch.arange(1, sensors + 1)
    counts = torch.ones(sensors, 1, dtype=torch.int32)
    starts = torch.arange(sensors, dtype=torch.int32)[:, None]
    calls = []
    def read(*, dt):
        calls.append(dt)
        return force, None, normals, None, counts, starts
    view = SimpleNamespace(sensor_paths=paths, sensor_count=sensors, filter_count=1,
                           filter_paths=(('/World/ground/terrain/Collision',),) * sensors,
                           max_contact_data_count=sensors + 1, get_contact_data=read)
    sensor = SimpleNamespace(contact_view=view, body_names=names,
                             body_physx_view=SimpleNamespace(prim_paths=paths))
    loads = LabContactLoads(sensor, binding, envs, collision_path='/World/ground/terrain/Collision', device='cpu')
    normal, support = loads.read(1, .005)
    expected = torch.tensor([[1 + names.index(name) + row * 19 for name in binding.native_body_names]
                             for row in range(2)], dtype=torch.float32)
    torch.testing.assert_close(normal, expected)
    before = support.clone()
    loads.reset_rows(torch.tensor([0]), 1)
    _, current = loads.read(1, .005)
    assert len(calls) == 1
    assert current[0].eq(0).all()
    assert loads.valid.tolist() == [False, True]
    torch.testing.assert_close(current[1], before[1])
    loads.read(2, .005)
    assert len(calls) == 2
    assert loads.valid.tolist() == [True, True]


def test_reset_contact_placeholder_is_invalid_and_pre_reset_label_validity_is_owned():
    from transformer_rl.formal.lab_env.adapter import LabLabeledEnv
    _, binding, _, original, _ = fixture()
    raw_valid = torch.tensor([True, False])
    sample = LabPhysicalSample(**original.__dict__, body_normal_load_n=torch.zeros(2, 19),
                               wheel_support_load_n=torch.zeros(2, 2), contact_measurement_valid=raw_valid)
    state = LabObservationAssembler(binding, 2).measure(sample)
    adapter = LabLabeledEnv(SimpleNamespace(num_envs=2, device='cpu', step_dt=.01))
    raw_valid.zero_()
    labels = adapter._labels(state, state.timestamp_s)
    assert labels.valid.tolist() == [True, False]

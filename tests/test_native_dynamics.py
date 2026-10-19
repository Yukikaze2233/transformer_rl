"""Kaiser上的原生参数接口替身；不声称已通过SDK物理验收。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_observation import native_fixture
from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig, apply_startup_dynamics


def fixture():
    robot, bridge, _ = native_fixture()
    view = robot.root_view
    view.masses = torch.arange(1, 20, dtype=torch.float32)[None].repeat(2, 1) / 10
    view.inertias = torch.eye(3).reshape(1, 1, 9).repeat(2, 19, 1) * .01
    view.get_masses = lambda: view.masses
    view.get_inertias = lambda: view.inertias
    writes = []

    def write(name, field, value, indices):
        with pytest.raises(RuntimeError, match='readback'):
            bridge.before_physics_step()
        assert indices.tolist() == [0, 1] and indices.dtype == torch.int32
        writes.append(name)
        getattr(view, field).copy_(value)

    view.set_masses = lambda value, indices: write('mass', 'masses', value, indices)
    view.set_coms = lambda value, indices: write('com', 'local_com', value, indices)
    view.set_inertias = lambda value, indices: write('inertia', 'inertias', value, indices)
    scene = SimpleNamespace(robot=robot, observation=bridge, readback={
        'native_mass_sum_verified': True, 'nominal_source_mass_kg': float(view.masses[0].sum())})
    return scene, view, writes


def apply(scene, config=None, seed=37):
    return apply_startup_dynamics(scene, config or StartupDynamicsConfig(nominal_fraction=0), seed,
                                  native_to_tensor=lambda value: value, tensor_to_native=lambda value: value)


def test_native_mass_com_and_inertia_are_written_before_final_model_refresh():
    scene, view, writes = fixture()
    mass, inertia, com = view.masses.clone(), view.inertias.clone(), view.local_com.clone()
    before = torch.random.get_rng_state().clone()
    receipt = apply(scene)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert writes == ['mass', 'com', 'inertia']
    scale = view.masses / mass
    assert scale[:, 0].ge(.9).all() and scale[:, 0].le(1.3).all()
    assert scale[:, 1:].ge(.9).all() and scale[:, 1:].le(1.1).all()
    torch.testing.assert_close(view.inertias, inertia * scale[..., None])
    torch.testing.assert_close(view.local_com[:, 1:], com[:, 1:])
    assert (view.local_com[:, 0, :3] - com[:, 0, :3]).abs().le(torch.tensor([.04, .02, .02])).all()
    torch.testing.assert_close(scene.observation._local_body_com_m, view.local_com[..., :3])
    assert receipt['model_epoch'] == 2 and receipt['nominal_rows'] == 0
    assert receipt['status'] == 'native_readback_matched' and not receipt['physical_qualification']
    scene.observation.before_physics_step()
    with pytest.raises(RuntimeError, match='refresh'):
        scene.observation.before_state_read()
    with pytest.raises(RuntimeError, match='fresh nominal'):
        apply(scene)
    assert len(writes) == 3


def test_cold_start_reconstructs_the_same_named_parameters_without_using_global_rng():
    first, a, _ = fixture()
    second, b, _ = fixture()
    left = apply(first)
    torch.rand(17)  # Other startup work does not alter this private draw.
    right = apply(second)
    assert left == right
    assert torch.equal(a.masses, b.masses) and torch.equal(a.local_com, b.local_com)
    other, c, _ = fixture()
    assert apply(other, seed=38)['native_parameters_sha256'] != left['native_parameters_sha256']
    assert not torch.equal(a.masses, c.masses)


def test_nominal_fraction_one_preserves_exact_native_parameters():
    scene, view, _ = fixture()
    old = (view.masses.clone(), view.local_com.clone(), view.inertias.clone())
    receipt = apply(scene, StartupDynamicsConfig(nominal_fraction=1))
    assert receipt['nominal_rows'] == 2
    assert all(torch.equal(a, b) for a, b in zip(old, (view.masses, view.local_com, view.inertias)))


def test_native_body_enumeration_does_not_reassign_named_mass_or_com_draws():
    first, a, _ = fixture()
    second, b, _ = fixture()
    order = [0, *range(18, 0, -1)]
    names = tuple(second.robot.body_names[i] for i in order)
    second.robot.body_names = names
    second.observation.binding = replace(second.observation.binding, native_body_names=names)
    b.masses = b.masses[:, order].clone()
    b.inertias = b.inertias[:, order].clone()
    b.local_com = b.local_com[:, order].clone()
    apply(first)
    apply(second)
    torch.testing.assert_close(a.masses[:, order], b.masses)
    torch.testing.assert_close(a.inertias[:, order], b.inertias)
    torch.testing.assert_close(a.local_com[:, order], b.local_com)


def test_installed_warp_accepts_the_worker_cpu_parameter_and_uint32_index_conversion():
    import warp as wp

    wp.init()  # Production initializes Warp while building the admitted SDK scene.
    values = torch.tensor([[1., 2.]], dtype=torch.float32)
    indices = torch.arange(2, dtype=torch.int32)
    native_values = wp.from_torch(values, dtype=wp.float32)
    native_indices = wp.from_torch(indices, dtype=wp.uint32)
    torch.testing.assert_close(wp.to_torch(native_values), values)
    assert wp.to_torch(native_indices).to(torch.int64).tolist() == [0, 1]


@pytest.mark.parametrize('fault', ['late', 'stale_nominal', 'mass_zero', 'bad_inertia', 'bad_com'])
def test_invalid_or_late_input_never_starts_mechanical_writes(fault):
    scene, view, writes = fixture()
    if fault == 'late':
        scene.observation.physical_tick = 2
    elif fault == 'stale_nominal':
        scene.readback['nominal_source_mass_kg'] += 1
    elif fault == 'mass_zero':
        view.masses[0, 0] = 0
    elif fault == 'bad_inertia':
        view.inertias[0, 0, 0] = -1
    else:
        view.local_com[0, 0, 6] = .5
    with pytest.raises((ValueError, RuntimeError)):
        apply(scene)
    assert not writes


@pytest.mark.parametrize('fault', ['setter', 'mass', 'com', 'inertia'])
def test_failed_native_write_or_final_readback_leaves_model_invalid(fault):
    scene, view, writes = fixture()
    original = view.set_inertias

    def corrupt(value, indices):
        if fault == 'setter':
            raise RuntimeError('native setter failure')
        original(value, indices)
        if fault == 'mass':
            view.masses[0, 0] *= 2
        elif fault == 'com':
            view.local_com[0, 0, 0] += .1
        else:
            view.inertias[0, 0, 0] *= 2

    view.set_inertias = corrupt
    with pytest.raises(RuntimeError):
        apply(scene)
    assert scene.readback['startup_dynamics']['status'] == 'writing'
    assert scene.observation.model_epoch == 1
    with pytest.raises(RuntimeError, match='readback'):
        scene.observation.before_physics_step()
    with pytest.raises(RuntimeError, match='fresh nominal'):
        apply(scene)


@pytest.mark.parametrize('change', [{'nominal_fraction': float('nan')}, {'base_mass_scale': (0, 1)},
                                    {'link_mass_scale': (.5, .9)}, {'base_com_offset_m': (.1, -.1, 0)}])
def test_engineering_prior_configuration_is_finite_and_contains_nominal(change):
    with pytest.raises(ValueError):
        replace(StartupDynamicsConfig(), **change)

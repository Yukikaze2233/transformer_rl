"""Native startup writes, actual COM projection and critic/reward mass source."""
from types import SimpleNamespace
import math

import pytest
import torch

from test_native_dynamics import fixture
from transformer_rl.formal.lab_env.dynamics import LabStartupDynamics
from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig


def native():
    scene, view, _ = fixture()
    view.count = 2
    writes = []
    for method, field in [('set_masses','masses'), ('set_coms','local_com'), ('set_inertias','inertias')]:
        def write(value, indices, field=field):
            assert indices.tolist() == [0, 1]
            writes.append(field)
            getattr(view, field).copy_(value)
        setattr(view, method, write)
    return scene.robot, scene.observation.binding, view, writes


def apply(robot, binding, view, config, *, seed=31):
    return LabStartupDynamics(robot, binding, 2, float(view.masses[0].sum()), seed, config,
                              device='cpu', native_to_tensor=lambda value:value,
                              tensor_to_native=lambda value:value)


def test_lab_startup_reuses_private_draw_and_actual_mass_without_global_rng_changes():
    robot, binding, view, writes = native()
    nominal = view.masses.sum(-1).clone()
    base_index = binding.native_body_names.index('base_link')
    nominal_com = view.local_com[:, base_index, :3].clone()
    initial_inertia, initial_mass = view.inertias.clone(), view.masses.clone()
    before = torch.get_rng_state().clone()
    first = apply(robot, binding, view, StartupDynamicsConfig(nominal_fraction=0))
    assert torch.equal(before, torch.get_rng_state())
    assert writes == ['masses', 'local_com', 'inertias']
    torch.testing.assert_close(first.mass_kg_by_environment, view.masses.sum(-1))
    torch.testing.assert_close(first.mass_scale[:,0], view.masses.sum(-1) / nominal)
    torch.testing.assert_close(first.base_com_delta_local_m,
                               view.local_com[:, base_index, :3] - nominal_com)
    assert first.base_com_delta_local_m.abs().sum() > 0
    torch.testing.assert_close(view.inertias, initial_inertia * (view.masses / initial_mass)[...,None])
    other_robot, other_binding, other_view, _ = native()
    second = apply(other_robot, other_binding, other_view, StartupDynamicsConfig(nominal_fraction=0))
    assert first.receipt == second.receipt
    assert not first.receipt['physical_qualification']


def test_nominal_path_writes_nothing_and_projects_final_com_in_native_xyzw_frame():
    robot, binding, view, writes = native()
    view.local_com[:,0,:3] = torch.tensor([.1, 0., 0.])
    actual = apply(robot, binding, view, None)
    assert not writes and actual.mass_scale.eq(1).all()
    assert actual.base_com_delta_local_m.eq(0).all()
    xyz = torch.tensor([[2., 3., 4.]]).repeat(2,1)
    rotation = torch.tensor([[0.,0.,math.sqrt(.5),math.sqrt(.5)]]).repeat(2,1)
    torch.testing.assert_close(actual.base_com_position(xyz, rotation), xyz + torch.tensor([0.,.1,0.]))


def test_bad_native_post_write_readback_is_rejected_before_any_feature_is_published():
    robot, binding, view, _ = native()
    original = view.set_inertias
    def corrupt(value, indices):
        original(value, indices)
        view.local_com[0,0,0] += .2
    view.set_inertias = corrupt
    with pytest.raises(RuntimeError, match='post-write readback'):
        apply(robot, binding, view, StartupDynamicsConfig())

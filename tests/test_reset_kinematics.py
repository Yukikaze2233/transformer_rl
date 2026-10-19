"""Independent tree/Jacobian checks; no SDK or robot simulation."""
from copy import deepcopy
import json

import pytest
import torch

from test_joint_boundaries import asset_sources, encode, forward_poses
from test_physical_observation import fixture as physical_fixture
from transformer_rl._reset_kinematics import AssetResetKinematics


def rotation(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*quaternion.shape[:-1], 3, 3)


def fixture(*, mutate=None):
    binding = physical_fixture()[1]
    manifest, spec = asset_sources(binding)
    spec['bodies'] = [{'name': name} for name in binding.native_body_names]
    spec['nominal_joint_pos'] = dict(manifest['nominal_joint_pos'])
    for joint in spec['joints']:
        if joint['type'] == 'prismatic':
            joint['limit'] = {'lower': -.3, 'upper': .3}
    root = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    poses = forward_poses(spec, binding, torch.zeros(1, 18), root)[0].double()
    matrices = rotation(poses[:, 3:])
    constraints = []
    for first, second, local in (
            ('L_link1', 'LL_link3', (.1, .1, 0.)), ('L_link2', 'LL_link4', (.1, 0., 0.)),
            ('L_link1', 'LLL_link1', (.1, .04, 0.)),
            ('R_link1', 'RR_link3', (.1, .1, 0.)), ('R_link2', 'RR_link4', (.1, 0., 0.)),
            ('R_link1', 'RRR_link1', (.1, .04, 0.))):
        a, b = (binding.native_body_names.index(name) for name in (first, second))
        target = poses[a, :3] + matrices[a] @ torch.tensor(local, dtype=torch.float64)
        local1 = matrices[b].T @ (target - poses[b, :3])
        constraints.append({'name': f'loop_{len(constraints)}', 'type': 'spherical', 'body0': first, 'body1': second,
                            'local_pos0_m': list(local), 'local_pos1_m': local1.tolist(),
                            'physics:excludeFromArticulation': True})
    spec['constraints'] = constraints
    if mutate is not None:
        mutate(spec)
    raw, model = encode(manifest, spec)
    return AssetResetKinematics(raw, model, binding), manifest, spec


def test_batched_full_tree_matches_an_independent_matrix_and_quaternion_implementation():
    geometry, _, spec = fixture()
    torch.manual_seed(2101)
    q = torch.randn(4, 18, dtype=torch.float64) * .1
    root = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).repeat(4, 1)
    expected = forward_poses(spec, geometry.binding, q, root).double()
    actual = geometry.forward(q)
    torch.testing.assert_close(actual.body_position_m, expected[:, :, :3], atol=7e-8, rtol=0)
    torch.testing.assert_close(actual.body_rotation, rotation(expected[:, :, 3:]), atol=3e-7, rtol=0)
    assert actual.body_position_m.shape == (4, 19, 3)
    assert actual.joint_axis_world.shape == (4, 18, 3)


def test_loop_jacobian_matches_central_differences_for_every_native_coordinate():
    geometry, *_ = fixture()
    q = torch.linspace(-.1, .1, 18, dtype=torch.float64)[None].repeat(2, 1)
    residual, jacobian = geometry.closure(q, jacobian=True)
    assert residual.shape == (2, 6, 3) and jacobian.shape == (2, 18, 18)
    for column in range(18):
        plus, minus = q.clone(), q.clone()
        plus[:, column] += 1e-6
        minus[:, column] -= 1e-6
        difference = (geometry.closure(plus) - geometry.closure(minus)).flatten(1) / 2e-6
        torch.testing.assert_close(jacobian[:, :, column], difference, atol=2e-10, rtol=1e-8)
    assert jacobian[:, :, list(geometry.policy_tree_indices[4:])].eq(0).all()


def test_passive_closure_keeps_all_six_prescribed_outputs_and_the_original_source_anchors():
    geometry, _, spec = fixture()
    seed = torch.zeros(3, 18, dtype=torch.float64)
    seed[:, geometry.passive_tree_indices] = .02
    output = torch.zeros(3, 6, dtype=torch.float64)
    output[:, 4:] = torch.tensor([2., -3.], dtype=torch.float64)
    original = deepcopy(spec['constraints'])
    result = geometry.solve_passive(seed, output, closure_tolerance_m=1e-7, maximum_iterations=80)
    assert result.converged.all() and result.within_native_limits.all()
    assert torch.equal(result.tree_position[:, geometry.policy_tree_indices], output)
    assert spec['constraints'] == original and not result.physical_qualification
    assert result.maximum_residual_m.max() <= 1e-7 and result.iterations > 0
    assert not torch.equal(result.tree_position, seed)


def test_one_infeasible_output_pair_does_not_move_its_motor_target_or_fail_other_rows():
    geometry, *_ = fixture()
    seed = torch.zeros(2, 18, dtype=torch.float64)
    output = torch.zeros(2, 6, dtype=torch.float64)
    output[1, 0] = 3.
    result = geometry.solve_passive(seed, output, closure_tolerance_m=1e-7, maximum_iterations=40)
    assert result.converged.tolist() == [True, False]
    assert torch.equal(result.tree_position[:, geometry.policy_tree_indices], output)
    assert result.maximum_residual_m[1] > .01


@pytest.mark.parametrize('fault', ['missing_loop', 'anchor_nan', 'body_duplicate', 'unrooted', 'bad_limit'])
def test_incomplete_or_invalid_selected_mechanism_is_rejected(fault):
    def mutate(spec):
        if fault == 'missing_loop':
            spec['constraints'].pop()
        elif fault == 'anchor_nan':
            spec['constraints'][0]['local_pos0_m'][0] = float('nan')
        elif fault == 'body_duplicate':
            spec['bodies'][0]['name'] = spec['bodies'][1]['name']
        elif fault == 'unrooted':
            next(j for j in spec['joints'] if j['name'] == 'LL_joint1')['parent'] = 'LL_link4'
        else:
            next(j for j in spec['joints'] if j['type'] == 'prismatic')['limit'] = {'lower': .3, 'upper': -.3}

    with pytest.raises(ValueError):
        fixture(mutate=mutate)

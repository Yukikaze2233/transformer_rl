"""Scene recipe and native-parameter readback fixtures; no SDK context."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from test_output_actuation import parameters
from test_physical_observation import fixture as observation_fixture
from transformer_rl._native_scene import NativeSceneRecipe, TerrainMesh


def mesh():
    return TerrainMesh(((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.)),
                       ((0, 1, 2), (0, 2, 3)), .8)


def fixture(tmp_path):
    manifest, binding, *_ = observation_fixture()
    spec = {'joints': [{'name': name, 'type': 'prismatic' if 'spring_slide' in name else 'continuous'}
                       for name in manifest['tree_joint_names']]}
    model = (json.dumps(spec, indent=2) + '\n').encode()
    (tmp_path / 'model_spec.json').write_bytes(model)
    usd = b'# synthetic fixture, never loaded by an SDK\n'
    (tmp_path / 'robot.usda').write_bytes(usd)
    manifest.update(spring_joint_names=['L_spring_slide', 'R_spring_slide'],
                    passive_damping={'active_Nm_s_per_rad': 0., 'passive_hinge_Nm_s_per_rad': .002,
                                     'slider_N_s_per_m': .002},
                    nominal_base_height_m=.32, self_collision_enabled=False, usd='robot.usda',
                    files_sha256={'model_spec.json': hashlib.sha256(model).hexdigest(),
                                  'robot.usda': hashlib.sha256(usd).hexdigest()})
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    (tmp_path / 'manifest.json').write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    recipe = NativeSceneRecipe(tmp_path, sha, parameters(), ((0., 0., 0.), (10., 0., 0.)), ((mesh(),), (mesh(),)))
    names = binding.native_joint_names
    native = parameters().native_parameters()
    armature = torch.tensor([[native[n]['armature'] if n in native else 0. for n in names]]).repeat(2, 1)
    friction = torch.tensor([[[native[n][k] if n in native else 0.
                               for k in ('friction', 'dynamic_friction', 'viscous_friction')] for n in names]]).repeat(2, 1, 1)
    zeros = torch.zeros(2, 18)
    view = SimpleNamespace(prim_paths=('/World/envs/env_1/Robot/base_link', '/World/envs/env_0/Robot/base_link'),
                           get_dof_armatures=lambda: armature,
                           get_dof_friction_properties=lambda: friction,
                           get_dof_stiffnesses=lambda: zeros,
                           get_dof_dampings=lambda: zeros)
    actuators = {}
    for name, group in recipe.groups.items():
        # Native actuator joint order may differ from configuration order.
        joints = tuple(reversed(group.names))
        actuators[name] = SimpleNamespace(joint_names=joints, is_implicit_model=False,
                                          stiffness=torch.zeros(2, len(joints)),
                                          damping=torch.full((2, len(joints)), float(group.damping)),
                                          effort_limit=torch.full((2, len(joints)), group.effort_limit))
    robot = SimpleNamespace(root_view=view, joint_names=names, body_names=binding.native_body_names,
                            actuators=actuators)
    return recipe, robot, manifest, sha, armature, friction, zeros


def test_recipe_and_configs_do_not_import_or_start_an_sdk(tmp_path):
    before = set(sys.modules)
    recipe, _, _, _, _, _, _ = fixture(tmp_path)
    configs = recipe._actuator_configs(lambda **kwargs: kwargs)
    assert recipe.initial_root_position_m == (0., 0., .32)
    assert not ({'isaaclab', 'isaaclab_physx', 'omni', 'isaacsim'} & (set(sys.modules) - before))
    assert configs['legs']['stiffness'] == configs['legs']['damping'] == 0
    assert configs['pressure']['stiffness'] == 0 and configs['pressure']['damping'] == .002
    assert configs['passive']['damping'] == .002
    assert configs['wheels']['effort_limit_sim'] == 4.5
    assert configs['legs']['effort_limit_sim'] == 40
    assert configs['pressure']['effort_limit_sim'] == 1000
    assert configs['wheels']['armature']['L_joint3'] == parameters().additional_armature[4]
    assert configs['wheels']['friction']['R_joint3'] == parameters().coulomb_effort[5]
    assert all(v == 0 for v in configs['pressure']['armature'].values())


def test_parameter_readback_separates_explicit_passive_damping_from_zero_solver_pd(tmp_path):
    recipe, robot, *_ = fixture(tmp_path)
    receipt = recipe._actuator_readback(robot, lambda x: x, 'cpu')
    assert receipt['solver_pd_gains_zero'] and receipt['named_armature_friction_verified']
    assert receipt['explicit_actuator_gains_and_caps_verified']
    assert not receipt['native_solver_caps_readback_verified'] and not receipt['hardware_calibration_verified']


@pytest.mark.parametrize('change', ['armature', 'static_friction', 'dynamic_friction', 'viscous_friction',
                                    'solver_pd', 'explicit_active_pd', 'explicit_passive_damping',
                                    'explicit_caps', 'implicit', 'wrong_joint', 'missing_group'])
def test_incorrect_native_or_explicit_parameters_are_rejected(tmp_path, change):
    recipe, robot, _, _, armature, friction, zeros = fixture(tmp_path)
    if change == 'armature':
        armature[0, 0] += .001
    elif 'friction' in change:
        column = ('static_friction', 'dynamic_friction', 'viscous_friction').index(change)
        friction[0, 0, column] += .01
    elif change == 'solver_pd':
        zeros[0, 0] = .002
    elif change == 'explicit_active_pd':
        robot.actuators['legs'].damping[0, 0] = 2.5
    elif change == 'explicit_passive_damping':
        robot.actuators['pressure'].damping.zero_()
    elif change == 'explicit_caps':
        robot.actuators['wheels'].effort_limit.fill_(40)
    elif change == 'implicit':
        robot.actuators['legs'].is_implicit_model = True
    elif change == 'wrong_joint':
        robot.actuators['legs'].joint_names = ('L_joint1',) * 4
    else:
        robot.actuators.pop('pressure')
    with pytest.raises(RuntimeError):
        recipe._actuator_readback(robot, lambda x: x, 'cpu')


@pytest.mark.parametrize('change', ['manifest_hash', 'model_hash', 'outside_dependency', 'duplicate_origins',
                                    'no_terrain', 'nan_origin', 'bad_active_damping'])
def test_changed_asset_or_incomplete_layout_is_rejected(tmp_path, change):
    _, _, manifest, sha, *_ = fixture(tmp_path)
    origins, meshes = ((0., 0., 0.), (10., 0., 0.)), ((mesh(),), (mesh(),))
    if change == 'manifest_hash':
        sha = '0' * 64
    elif change == 'model_hash':
        (tmp_path / 'model_spec.json').write_text('{}')
    elif change == 'outside_dependency':
        manifest['files_sha256']['../external.json'] = '0' * 64
    elif change == 'duplicate_origins':
        origins = (origins[0], origins[0])
    elif change == 'no_terrain':
        meshes = ((), (mesh(),))
    elif change == 'nan_origin':
        origins = ((float('nan'), 0., 0.), origins[1])
    else:
        manifest['passive_damping']['active_Nm_s_per_rad'] = .1
    if change in ('outside_dependency', 'bad_active_damping'):
        raw = (json.dumps(manifest, indent=2) + '\n').encode()
        (tmp_path / 'manifest.json').write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        NativeSceneRecipe(tmp_path, sha, parameters(), origins, meshes)


@pytest.mark.parametrize('change', ['empty', 'duplicate_face_vertex', 'out_of_range', 'degenerate', 'nan', 'friction'])
def test_meshes_reject_invalid_native_collision_geometry(change):
    vertices, faces, mu = [(-1., -1., 0.), (1., -1., 0.), (1., 1., 0.)], [(0, 1, 2)], .8
    if change == 'empty':
        faces = []
    elif change == 'duplicate_face_vertex':
        faces = [(0, 0, 2)]
    elif change == 'out_of_range':
        faces = [(0, 1, 3)]
    elif change == 'degenerate':
        vertices[2] = (0., -1., 0.)
    elif change == 'nan':
        vertices[0] = (float('nan'), 0., 0.)
    else:
        mu = -1
    with pytest.raises(ValueError):
        TerrainMesh(vertices, faces, mu)


def test_mesh_input_is_snapshotted_instead_of_changed_by_later_layout_edits():
    source = [[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.]]
    terrain = TerrainMesh(source, [[0, 1, 2]], .8)
    expected = deepcopy(terrain.vertices_m)
    source[0][0] = 500
    assert terrain.vertices_m == expected

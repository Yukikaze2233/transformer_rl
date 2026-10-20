"""Verify source-to-atlas physical materials on Kaiser without an SDK context."""
import json

import numpy as np
import pytest
import torch

from transformer_rl.formal.tasks.terrain_assets import write_collision_usda
from transformer_rl.formal.tasks.terrain_courses import build_atlas
from transformer_rl.formal.tasks.terrain_query import StaticTerrainQuery

from test_terrain_assets import real_assets


VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
FACES = np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def test_complete_materials_keep_equal_static_but_distinct_dynamic_and_restitution(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics, UsdShade

    path = tmp_path / 'terrain.usda'
    parameters = np.array([[.6, .5, 0.], [.6, .3, .1]])
    write_collision_usda(path, VERTICES, FACES, static_friction=.7, dynamic_friction=.4,
                         face_material=np.array([0, 1]), material_parameters=parameters)
    stage = Usd.Stage.Open(str(path))
    mesh = stage.GetPrimAtPath('/Terrain/Collision')
    for index, expected in enumerate(parameters):
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f'/Terrain/Surface_{index}'))
        actual = (material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get(),
                  material.GetRestitutionAttr().Get())
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
        prim = material.GetPrim()
        assert prim.GetAttribute('physxMaterial:frictionCombineMode').Get() == 'multiply'
        assert prim.GetAttribute('physxMaterial:restitutionCombineMode').Get() == 'multiply'
    subsets = UsdShade.MaterialBindingAPI(mesh).GetMaterialBindSubsets()
    assignments = {}
    for subset in subsets:
        target = subset.GetPrim().GetRelationship('material:binding:physics').GetTargets()
        assert len(target) == 1
        assignments[str(target[0])] = list(subset.GetIndicesAttr().Get())
    assert assignments == {'/Terrain/Surface_0': [0], '/Terrain/Surface_1': [1]}


@pytest.mark.parametrize('extra', [
    {'face_material': np.array([0., 0.]), 'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([False, True]), 'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([-1, 0]), 'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([0, 1]), 'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([0]), 'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, .5]]},
    {'face_material': np.array([0, 0]), 'material_parameters': []},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, float('nan'), 0.]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, float('inf'), 0.]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, -.1, 0.]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, .5, 1.1]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, .5, -.1]]},
    {'face_material': np.array([0, 0])},
    {'material_parameters': [[.6, .5, 0.]]},
    {'face_material': np.array([0, 0]), 'material_parameters': [[.6, .5, 0.]],
     'material_coefficients': [.6]},
    {'face_material': np.array([0, 0]), 'material_coefficients': [[.6]]},
])
def test_invalid_materials_are_rejected_before_writing_output(tmp_path, extra):
    path = tmp_path / 'terrain.usda'
    with pytest.raises(ValueError):
        write_collision_usda(path, VERTICES, FACES, static_friction=.7, dynamic_friction=.5, **extra)
    assert not path.exists()


@pytest.mark.parametrize('static,dynamic,restitution', [
    (float('nan'), .5, 0), (.6, float('inf'), 0), (-.1, .5, 0), (.6, .5, 2),
])
def test_default_material_is_validated_without_per_face_table(tmp_path, static, dynamic, restitution):
    path = tmp_path / 'terrain.usda'
    with pytest.raises(ValueError):
        write_collision_usda(path, VERTICES, FACES, static_friction=static,
                             dynamic_friction=dynamic, restitution=restitution)
    assert not path.exists()


def test_authored_atlas_preserves_source_pairs_and_query_static_semantics(real_assets, tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics

    _, assets, prepared = real_assets
    output = tmp_path / 'atlas'
    manifest = build_atlas(assets, output, seed=7, levels=2)
    assert manifest['material_parameter_order'] == ['static_friction', 'dynamic_friction', 'restitution']
    assert manifest['material_combine_mode'] == 'multiply'
    assert manifest['material_coefficients_semantics'] == 'static_friction_only'
    stage = Usd.Stage.Open(str(output / 'terrain.usda'))
    points, tiles, expected_friction = [], [], []
    with np.load(output / 'geometry.npz', allow_pickle=False) as data:
        table = data['material_parameters']
        np.testing.assert_array_equal(data['material_coefficients'], table[:, 0])
        for tile_id, tile in enumerate(manifest['tiles']):
            if tile['name'] not in prepared['assets']:
                continue
            source = prepared['assets'][tile['name']]['material']
            expected = np.array([source[key] for key in manifest['material_parameter_order']])
            assert tile['physical_material'] == source
            face_ids = np.flatnonzero(data['face_tile'] == tile_id)
            assigned = np.unique(data['face_material'][face_ids])
            assert len(face_ids) > 0 and len(assigned) == 1
            np.testing.assert_allclose(table[assigned[0]], expected, atol=0, rtol=0)
            material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f'/Terrain/Surface_{assigned[0]}'))
            actual = (material.GetStaticFrictionAttr().Get(), material.GetDynamicFrictionAttr().Get(),
                      material.GetRestitutionAttr().Get())
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
            points.append(np.asarray(tile['origin_m']) + [-1., 0., 5.])
            tiles.append(tile_id)
            expected_friction.append(source['static_friction'])
    assert len(points) == 2
    query = StaticTerrainQuery(output / 'geometry.npz', output / 'query_index.npz')
    _, valid, _, friction = query(torch.tensor(np.asarray(points), dtype=torch.float32),
                                  torch.tensor(tiles, dtype=torch.long))
    assert valid.all()
    torch.testing.assert_close(friction, torch.tensor(expected_friction, dtype=torch.float32))
    assert json.loads((assets / 'manifest.json').read_text()) == prepared

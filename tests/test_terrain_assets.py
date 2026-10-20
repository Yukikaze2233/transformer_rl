"""Geometry-only checks, executed on Kaiser without starting a simulator."""
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest

from transformer_rl.formal.tasks.terrain_assets import clean_collision, prepare_assets, write_collision_usda
from transformer_rl.formal.tasks.terrain_courses import surface_height


def area(vertices, faces):
    tri = np.asarray(vertices, float)[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2


def test_coplanar_overlap_is_subtracted_without_removing_real_height():
    first = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    second = first + [.5, 0, 0]
    raised = first + [0, 0, .001]
    vertices = np.concatenate((first, second, raised))
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]])
    v, f, lineage, report = clean_collision(vertices, faces)
    assert area(v, f) == pytest.approx(2.5, abs=1e-7)
    assert report['removed_overlap_area_m2'] == pytest.approx(.5, abs=1e-7)
    assert len(lineage) == len(f)
    assert surface_height(v, f, [[.25, .25]])[0] == pytest.approx(.001)


def test_underpanel_gap_and_reverse_faces_are_not_filled_or_deleted():
    vertices = np.array([[0, -1, .05], [0, 1, .05], [0, 1, .20], [0, -1, .20]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0], [0, 0, 1]])
    v, f, _, report = clean_collision(vertices, faces)
    assert report['float32_zero_area_source_faces'] == [4]
    assert area(v, f) == pytest.approx(.6)
    assert v[:, 2].min() == pytest.approx(.05)
    assert not report['filled_holes']


def test_usda_has_one_collision_mesh_and_complete_material_partition(tmp_path):
    pxr = pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    path = tmp_path / 'terrain.usda'
    write_collision_usda(path, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                         np.array([[0, 1, 2], [0, 2, 3]]), static_friction=.7, dynamic_friction=.7,
                         face_material=np.array([0, 1]), material_coefficients=np.array([.2, 1.1]))
    stage = Usd.Stage.Open(str(path))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert [str(p.GetPath()) for p in meshes] == ['/Terrain/Collision']
    assert meshes[0].HasAPI(UsdPhysics.CollisionAPI)
    subsets = UsdShade.MaterialBindingAPI(meshes[0]).GetMaterialBindSubsets()
    assert sorted(i for subset in subsets for i in subset.GetIndicesAttr().Get()) == [0, 1]
    assert len(subsets) == 2


@pytest.fixture(scope='module')
def real_assets(tmp_path_factory):
    archive = os.environ.get('TERRAIN_SOURCE_ARCHIVE')
    if not archive:
        pytest.skip('reviewed terrain archive not supplied')
    root = Path(__file__).parents[1]
    output = tmp_path_factory.mktemp('prepared') / 'assets'
    manifest = prepare_assets(archive, root / 'configs/terrain_assets.json', output)
    return archive, output, manifest


def test_source_derivatives_preserve_finite_surface_samples(real_assets):
    archive, output, manifest = real_assets
    rng = np.random.default_rng(103)
    with zipfile.ZipFile(archive) as package:
        for name, record in manifest['assets'].items():
            source_name = record['specification']['directory']
            with np.load(io.BytesIO(package.read(f'RMUC2027_RL_Terrains/{source_name}/geometry.npz'))) as source:
                old_v, old_f = source['vertices_m'], source['triangles']
            with np.load(output / name / 'collision.npz') as new:
                vertices, faces = new['vertices_m'], new['triangles']
                assert new['source_triangle_index'].shape == (len(faces),)
            low, high = old_v[:, :2].min(0), old_v[:, :2].max(0)
            points = rng.uniform(low, high, (300, 2))
            before = surface_height(old_v.astype(np.float32), old_f, points)
            after = surface_height(vertices, faces, points)
            np.testing.assert_array_equal(np.isfinite(before), np.isfinite(after))
            np.testing.assert_allclose(before[np.isfinite(before)], after[np.isfinite(after)], atol=2e-6, rtol=0)
            if name == 'central_steps':
                assert len(record['processing']['float32_zero_area_source_faces']) == 15


def test_archive_rejects_wrong_identity_without_creating_output(tmp_path):
    config = tmp_path / 'assets.json'
    config.write_text(json.dumps({'package_sha256': '0' * 64}))
    archive = tmp_path / 'input.zip'
    archive.write_bytes(b'not the reviewed package')
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='SHA256'):
        prepare_assets(archive, config, output)
    assert not output.exists()

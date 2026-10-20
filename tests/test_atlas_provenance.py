"""Every atlas face retains its origin; only encoded zero-area faces disappear."""
import numpy as np
import pytest

from transformer_rl.formal.tasks.terrain_courses import build_atlas

from test_terrain_assets import real_assets


@pytest.fixture(scope='module')
def full_atlas(real_assets, tmp_path_factory):
    _, prepared, _ = real_assets
    root = tmp_path_factory.mktemp('atlas-provenance')/'atlas'
    manifest = build_atlas(prepared, root, seed=2102, levels=10)
    return prepared, root, manifest


def test_encoded_triangles_are_nondegenerate_and_all_removed_rows_are_accounted(full_atlas):
    _, root, manifest = full_atlas
    cleanup = manifest['float32_face_cleanup']
    assert cleanup['stage'] == 'after_atlas_translation'
    assert not cleanup['vertices_moved'] and not cleanup['positive_area_faces_removed']
    with np.load(root/'geometry.npz', allow_pickle=False) as data:
        triangles = data['vertices_m'][data['triangles']].astype(float)
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        assert (normals != 0).any(-1).all()
        assert len(triangles) == cleanup['output_triangles']
        removed = [row['pre_cleanup_face'] for row in cleanup['removed']]
        surviving = data['face_pre_cleanup_index'].tolist()
        assert sorted(removed+surviving) == list(range(cleanup['input_triangles']))
        assert set(removed).isdisjoint(surviving)
        for name in ('face_tile', 'face_source_part', 'face_source_triangle'):
            assert data[name].shape == (len(triangles),)


def test_authored_face_provenance_reconstructs_every_surviving_surface_exactly(full_atlas):
    prepared, root, manifest = full_atlas
    with np.load(root/'geometry.npz', allow_pickle=False) as data:
        for tile_id, tile in enumerate(manifest['tiles']):
            if tile['name'] not in ('road_step', 'central_steps'):
                continue
            with np.load(prepared/tile['name']/'collision.npz', allow_pickle=False) as source:
                vertices, faces = source['vertices_m'], source['triangles']
                if tile['name'] == 'road_step':
                    vertices = vertices[:, [1, 0, 2]]*np.asarray((1, -1, 1), np.float32)
                vertices = vertices + np.asarray(tile['origin_m'], np.float32)
                mask = (data['face_tile'] == tile_id) & (data['face_source_part'] == 0)
                expected = vertices[faces[data['face_source_triangle'][mask]]]
                actual = data['vertices_m'][data['triangles'][mask]]
                np.testing.assert_array_equal(actual, expected)
                # Derive numerical collapse independently from the prepared
                # source after the actual float32 transform, not a size cutoff.
                triangles = vertices[faces].astype(float)
                zero = (np.cross(triangles[:, 1]-triangles[:, 0],
                                 triangles[:, 2]-triangles[:, 0]) == 0).all(-1)
                removed = [record['source_triangle'] for record in manifest['float32_face_cleanup']['removed']
                           if record['tile'] == tile_id and record['part'] == 0]
                assert sorted(removed) == np.flatnonzero(zero).tolist()
                assert sorted(data['face_source_triangle'][mask].tolist()+removed) == list(range(len(faces)))
        # The fixed reviewed atlas does expose the post-translation issue.
        assert len(manifest['float32_face_cleanup']['removed']) == 4
        assert {manifest['tiles'][row['tile']]['name'] for row in manifest['float32_face_cleanup']['removed']} == {'road_step'}


def test_atlas_keeps_thin_positive_area_faces_and_original_terrain_scope(full_atlas):
    _, root, manifest = full_atlas
    assert len(manifest['tiles']) == 73 and manifest['levels'] == 10
    assert {tile['name'] for tile in manifest['tiles']} >= {'road_step', 'central_steps', 'stairs', 'step_up', 'step_down'}
    with np.load(root/'geometry.npz', allow_pickle=False) as data:
        triangles = data['vertices_m'][data['triangles']].astype(float)
        length = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=-1).max(-1)
        area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)
        assert (length**2/area > 10000).any()

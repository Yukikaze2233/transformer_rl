"""Surface-preserving numerical bases for collision triangle preparation."""
import numpy as np
import pytest

from transformer_rl.formal.tasks.terrain_assets import stable_triangle_corners

from test_terrain_assets import real_assets


def test_narrow_triangle_uses_a_short_edge_without_moving_or_reversing_vertices():
    vertices = np.array([[0., 0, .2], [2., 0, .2], [2., .0001, .2]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    original = vertices.copy()
    result, offsets = stable_triangle_corners(vertices, faces)
    assert result.tolist() == [[1, 2, 0]] and offsets.tolist() == [1]
    np.testing.assert_array_equal(vertices, original)
    np.testing.assert_array_equal(faces, [[0, 1, 2]])
    again, second = stable_triangle_corners(vertices, result)
    np.testing.assert_array_equal(again, result)
    assert second.tolist() == [0]


def test_cyclic_input_orders_converge_to_the_same_geometric_basis():
    vertices = np.array([[0., 0, 0], [2., 0, 0], [2., .0001, 0]])
    faces = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]])
    result, offsets = stable_triangle_corners(vertices, faces)
    np.testing.assert_array_equal(result[:3], np.repeat([[1, 2, 0]], 3, axis=0))
    before, after = vertices[faces], vertices[result]
    # Reversing a source face remains reversed; no winding repair is inferred.
    np.testing.assert_allclose(np.cross(before[:, 1]-before[:, 0], before[:, 2]-before[:, 0]),
                               np.cross(after[:, 1]-after[:, 0], after[:, 2]-after[:, 0]), rtol=0, atol=0)
    restored = np.take_along_axis(result, (np.arange(3)[None] - offsets[:, None]) % 3, axis=1)
    np.testing.assert_array_equal(restored, faces)


@pytest.mark.parametrize('vertices,faces', [
    ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1]]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, -1]]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0., 1., 2.]]),
    ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], []),
    ([[0, 0, 0], [1, 0, 0], [0, float('nan'), 0]], [[0, 1, 2]]),
])
def test_invalid_or_truly_degenerate_geometry_is_not_hidden(vertices, faces):
    with pytest.raises(ValueError):
        stable_triangle_corners(vertices, faces)


def test_prepared_asset_offsets_are_explicit_reversible_and_idempotent(real_assets):
    _, root, manifest = real_assets
    for name, record in manifest['assets'].items():
        description = record['processing']['triangle_corner_order']
        assert description['offset_array'] == 'cyclic_corner_offset'
        assert not any(description[key] for key in ('vertices_moved', 'winding_changed',
                                                    'triangles_added_or_removed', 'lineage_changed'))
        with np.load(root/name/'collision.npz', allow_pickle=False) as data:
            faces, offsets = data['triangles'], data['cyclic_corner_offset']
            assert offsets.dtype == np.uint8 and offsets.shape == (len(faces),)
            assert (offsets <= 2).all()
            restored = np.take_along_axis(faces, (np.arange(3)[None] - offsets[:, None]) % 3, axis=1)
            repeated, repeated_offsets = stable_triangle_corners(data['vertices_m'], restored)
            np.testing.assert_array_equal(repeated, faces)
            np.testing.assert_array_equal(repeated_offsets, offsets)
            _, again = stable_triangle_corners(data['vertices_m'], faces)
            assert (again == 0).all()
            assert int((offsets != 0).sum()) == description['reordered_faces']
            assert data['source_triangle_index'].shape == (len(faces),)
            assert data['derived_face_kind'].shape == (len(faces),)

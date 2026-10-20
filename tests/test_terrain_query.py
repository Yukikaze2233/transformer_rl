"""Exact indexed intersections, holes and stream-local runtime queries."""
import numpy as np
import pytest
import torch

from transformer_rl.formal.tasks.terrain_courses import surface_height
from transformer_rl.formal.tasks.terrain_query import build_query_index, StaticTerrainQuery


@pytest.fixture
def geometry(tmp_path):
    # Two levels overlap in XY. A separate gap and a vertical panel remain holes
    # for a downward ray; neither may be replaced with an invented floor.
    vertices = np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                           (0, 0, .2), (1, 0, .2), (0, 1, .2),
                           (2, 0, .05), (2, 0, .2), (2, 1, .2)], np.float32)
    faces = np.asarray([(0, 1, 2), (3, 4, 5), (6, 7, 8)], np.int32)
    mesh = tmp_path / 'geometry.npz'
    np.savez(mesh, vertices_m=vertices, triangles=faces,
             face_material=np.array([0, 1, 0]), material_coefficients=np.array([.6, .9]))
    index = tmp_path / 'query_index.npz'
    np.savez(index, **build_query_index(vertices, faces, np.zeros(3, np.int32), 1))
    return mesh, index, vertices, faces


def test_indexed_downward_query_matches_full_mesh_and_preserves_holes(geometry):
    mesh, index, vertices, faces = geometry
    query = StaticTerrainQuery(mesh, index)
    xy = np.random.default_rng(3).uniform(-.2, 2.2, (100, 2)).astype(np.float32)
    points = torch.from_numpy(np.column_stack((xy, np.full(100, 5., np.float32))))
    height, valid, normal, material = query(points, torch.zeros(100, dtype=torch.long))
    expected = surface_height(vertices, faces, xy)
    assert np.array_equal(valid.numpy(), np.isfinite(expected))
    np.testing.assert_allclose(height.numpy()[valid], expected[valid], atol=1e-6)
    assert normal[valid, 2].eq(1).all()
    torch.testing.assert_close(material[valid], torch.full_like(material[valid], .9))
    assert torch.isnan(height[~valid]).all()


def test_query_starts_at_ray_height_and_rejects_invalid_tile_without_memory_access(geometry):
    mesh, index, _, _ = geometry
    query = StaticTerrainQuery(mesh, index)
    points = torch.tensor([[.1, .1, .1], [.1, .1, 5.], [2, .5, 5.], [float('nan'), 0, 5]])
    height, valid, _, _ = query(points, torch.tensor([0, -1, 0, 0]))
    assert valid.tolist() == [True, False, False, False]
    assert float(height[0]) == 0
    assert int(query.misses) == 3

"""Failure-only terrain diagnostics preserve the production intersection result."""
import numpy as np
import torch

from transformer_rl.formal.tasks.terrain_query import StaticTerrainQuery, build_query_index


def _query(tmp_path):
    # The triangular surface occupies half its XY bounding rectangle. The
    # other half is an authored hole, even though the CSR cell grid covers it.
    vertices = np.array([[0, 0, .2], [1, 0, .2], [0, 1, .2]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    geometry, index = tmp_path / 'geometry.npz', tmp_path / 'query_index.npz'
    np.savez(geometry, vertices_m=vertices, triangles=faces,
             face_material=np.array([0]), face_tile=np.array([0]),
             material_coefficients=np.array([.7], np.float32))
    np.savez(index, **build_query_index(vertices, faces, np.array([0]), 1))
    return StaticTerrainQuery(geometry, index)


def test_gap_outside_low_ray_nonfinite_and_invalid_tile_are_distinct(tmp_path):
    query = _query(tmp_path)
    points = torch.tensor([[.9, .9, 2.], [1.1, .1, 2.], [.1, .1, .1],
                           [float('nan'), .1, 2.], [.1, .1, 2.]])
    tiles = torch.tensor([0, 0, 0, 0, -1])
    height, valid, _, _ = query(points, tiles)
    misses, queries = query.misses.clone(), query.queries
    original_points, rng = points.clone(), torch.get_rng_state().clone()
    assert not valid.any() and torch.isnan(height).all()
    report = query._diagnose_misses(points, tiles, valid)
    assert [sample['reason'] for sample in report['samples']] == [
        'geometry_gap', 'spatial_outside', 'ray_origin_below_surface',
        'nonfinite_state', 'invalid_tile']
    assert report['failed_columns'] == 5 and report['diagnostic_only']
    assert query.queries == queries and torch.equal(query.misses, misses)
    assert torch.isnan(height).all()
    torch.testing.assert_close(points, original_points, equal_nan=True)
    assert torch.equal(torch.get_rng_state(), rng)


def test_a_missed_index_hit_cannot_be_reported_as_a_physical_hole(tmp_path):
    query = _query(tmp_path)
    point, tile = torch.tensor([[.1, .1, 2.]]), torch.zeros(1, dtype=torch.long)
    height, valid, _, _ = query(point, tile)
    assert valid.all() and torch.allclose(height, torch.tensor([.2]))
    # Remove the acceleration entries without editing the source triangles.
    query._buffers['offsets'].zero_()
    missed, valid, _, _ = query(point, tile)
    assert not valid.any() and torch.isnan(missed).all()
    report = query._diagnose_misses(point, tile, valid)
    assert report['samples'][0]['reason'] == 'index_miss'


def test_overwriting_ray_z_cannot_hide_nonfinite_physics_and_samples_are_bounded(tmp_path):
    query = _query(tmp_path)
    point = torch.tensor([[[.1, .1, query._ceiling_m], [.9, .9, query._ceiling_m]]])
    physical = point.clone()
    physical[0, 0, 2] = float('nan')
    report = query._diagnose_misses(point, torch.zeros(1, 2, dtype=torch.long),
                                    torch.zeros(1, 2, dtype=torch.bool),
                                    physical_points=physical, limit=1)
    assert report['failed_columns'] == 2 and len(report['samples']) == 1
    assert report['samples'][0]['flat_index'] == 0
    assert report['samples'][0]['reason'] == 'nonfinite_state'


def test_a_fallen_body_keeps_a_real_ground_hit_and_needs_no_miss_diagnostic(tmp_path):
    query = _query(tmp_path)
    physical = torch.tensor([[.1, .1, -6.]])
    ray = physical.clone()
    ray[:, 2] = query._ceiling_m
    height, valid, _, _ = query(ray, torch.zeros(1, dtype=torch.long))
    report = query._diagnose_misses(ray, torch.zeros(1, dtype=torch.long), valid,
                                    physical_points=physical)
    assert report['samples'] == [] and report['failed_columns'] == 0
    assert float(physical[0, 2] - height[0]) < -6.


def test_a_neighbouring_tile_cannot_supply_an_assigned_tiles_missing_ground(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [10, 0, .2], [11, 0, .2], [10, 1, .2]], np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    geometry, index = tmp_path / 'geometry.npz', tmp_path / 'query_index.npz'
    np.savez(geometry, vertices_m=vertices, triangles=faces,
             face_material=np.array([0, 0]), face_tile=np.array([0, 1]),
             material_coefficients=np.array([.7], np.float32))
    np.savez(index, **build_query_index(vertices, faces, np.array([0, 1]), 2))
    query = StaticTerrainQuery(geometry, index)
    points = torch.tensor([[10.1, .1, 2.], [10.9, .9, 2.]])
    tiles = torch.tensor([0, 1])
    height, valid, _, _ = query(points, tiles)
    assert not valid.any() and torch.isnan(height).all()
    report = query._diagnose_misses(points, tiles, valid)
    assert [row['reason'] for row in report['samples']] == ['spatial_outside', 'geometry_gap']

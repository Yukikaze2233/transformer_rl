"""Probe-reference checks; synthetic rays grant no native PhysX evidence."""
import numpy as np
import pytest

from tools.check_terrain_assets import TerrainAssetCheck
from test_terrain_assets import real_assets


def test_reference_uses_the_visible_surface_and_ignores_hits_behind_the_origin():
    floor = np.array([[[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]]])
    upper = floor + [0., 0., 1.]
    triangles = np.concatenate((floor, upper))
    direction = np.array([0., 0., -1.])
    np.testing.assert_allclose(
        TerrainAssetCheck._first_intersection(triangles, np.array([0., 0., 1.5]), direction, 2),
        [0., 0., 1.])
    np.testing.assert_allclose(
        TerrainAssetCheck._first_intersection(triangles, np.array([0., 0., .5]), direction, 1),
        [0., 0., 0.])
    with pytest.raises(ValueError, match='no intersection'):
        TerrainAssetCheck._first_intersection(triangles, np.array([0., 0., .5]), direction, .2)


def test_reference_intersects_a_sloped_face_from_either_side():
    triangles = np.array([[[-1., -1., -.5], [1., -1., .5], [0., 1., 0.]]])
    for start, direction in (([.2, 0., 1.], [0., 0., -1.]),
                             ([.2, 0., -1.], [0., 0., 1.])):
        actual = TerrainAssetCheck._first_intersection(
            triangles, np.asarray(start), np.asarray(direction), 2)
        np.testing.assert_allclose(actual, [.2, 0., .1], atol=1e-14)


def test_prepared_probe_set_covers_original_faces_and_the_generated_front(real_assets):
    _archive, folder, manifest = real_assets
    check = TerrainAssetCheck(folder / 'manifest.json')
    for name, (_folder, record, geometry) in check.assets.items():
        probes = list(check._probes(record, geometry))
        source = [probe for probe in probes if probe[0] == 'triangle_centroid']
        assert len(source) == len(geometry['triangles'])
        assert all(abs(np.linalg.norm(probe[3]) - 1.) < 1e-12 for probe in source)
        front = [probe for probe in probes if probe[0] == 'continuous_front']
        assert len(front) == (72 if name == 'road_step' else 0)
        assert np.isfinite(np.asarray([probe[5] for probe in probes])).all()
    assert manifest['assets']['road_step']['processing']['training_additions']

"""Named native-query adapter tests; the synthetic rays grant no SDK evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_ground import fixture
from transformer_rl._native_collision import verify_wheel_cylinders


class Query:
    def __init__(self, scene, names, *, error=0., own_hits=True):
        self.paths = [env + '/Robot/' + name + '/collider' for env in scene.environment_paths for name in names]
        self.calls = []
        self.error, self.own_hits = error, own_hits

    def raycast_all(self, origin, direction, distance, callback, both_sides):
        self.calls.append((origin, direction, distance, both_sides))
        assert abs(sum(v*v for v in direction) - 1) < 1e-10
        point = tuple(a + (.01 + self.error) * b for a, b in zip(origin, direction))
        paths = self.paths if self.own_hits else ['/other/Robot/wheel/collider']
        for path in paths:
            callback(SimpleNamespace(collision=path, position=point))
        return len(paths)


def setup():
    robot, bridge, scene, _, wheels = fixture()
    wheels = replace(wheels, urdf_sha256='a' * 64)
    scene.readback['selected_urdf_sha256'] = wheels.urdf_sha256
    poses = robot.data.body_link_pose_w.torch
    robot.root_view.get_link_transforms = lambda: poses
    return scene, wheels


def test_native_probe_uses_owned_names_caps_and_radial_points_without_advancing_physics():
    scene, wheels = setup()
    query = Query(scene, wheels.body_names)
    receipt = verify_wheel_cylinders(scene, wheels, query, native_to_tensor=lambda v: v,
                                      tolerance_m=.001)
    assert receipt['maximum_error_m'] < 1e-12
    assert len(query.calls) == 2 * 2 * 34
    assert sum(v['kind'] == 'cap' for v in receipt['records']) == 8
    assert scene.observation.physical_tick == 0
    assert receipt['physical_task_qualified'] is False
    assert scene.readback.get('native_wheel_collision_equivalence_verified') is not True


@pytest.mark.parametrize('error,own_hits', [(.002, True), (0., False)])
def test_native_probe_refuses_changed_surface_and_foreign_collider_hits(error, own_hits):
    scene, wheels = setup()
    with pytest.raises(RuntimeError, match='native collision differs'):
        verify_wheel_cylinders(scene, wheels, Query(scene, wheels.body_names, error=error, own_hits=own_hits),
                               native_to_tensor=lambda v: v, tolerance_m=.001)

"""Kaiser CPU检查：驻留几何与编辑顺序，不代替原生接触验证。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_terrain import fixture
from transformer_rl.formal.simulation.terrain import NativeTerrainInstaller
from transformer_rl.formal.simulation.terrain_stream import NativeTerrainStream
from transformer_rl.formal.tasks.geometry import TerrainKind, TerrainQueryBatch, TerrainSurface
from transformer_rl.formal.tasks.terrain import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning


def rough(extent=20.):
    generator = ContinuousTerrainGenerator(7, TerrainTuning(.7, (.2, 1.1), .25))
    request = TerrainRequest(TerrainKind.ROUGH, TerrainExtent((-extent, extent), (-extent, extent)), roughness_m=.02)
    return generator.build(request, geometry_seed=17, resident_tiles=True)


def test_resident_geometry_is_bounded_while_the_complete_request_and_source_remain_available():
    course = rough(302.)
    assert course.request.extent.x_m == (-302., 302.)
    assert course.tile_source.shape[0]*course.tile_source.shape[1] > 2_000_000
    assert sum(len(mesh.vertices_m) for mesh in course.surface.meshes) < 2000
    keys = course.tile_source.tile_keys((120., 121.), (70., 71.), cells=8, halo=1)
    moved = course.with_tiles(keys)
    assert moved.sha256 == course.sha256 and moved.tile_source is course.tile_source
    assert sum(len(mesh.vertices_m) for mesh in moved.surface.meshes) < 2000
    points = torch.tensor([[[120.25, 70.25], [120.5, 70.375], [120.875, 70.5]]])
    query = TerrainQueryBatch((moved.surface,)).at(points)
    expected = course.tile_source.at(points)
    torch.testing.assert_close(query.height_m, expected.height_m)
    assert query.valid.all()


def admitted_rough(terrain=None):
    env, robot, sim, writes, editor, storage, *_ = fixture()
    courses = (rough(), rough()) if terrain is None else (terrain, terrain)
    owner = SimpleNamespace(lifecycle=env.lifecycle, _scene=env.lifecycle.plant.scene, _identity=lambda: None)
    installer = NativeTerrainInstaller(env, owner, editor)
    result = editor.replace_courses(env.lifecycle.ground, courses, torch.ones(2, dtype=torch.bool))
    installer.adopt(result)
    # 假SDK夹具只验证编辑；不声称完成真实reset或物理验收。
    result.scene.readback['terrain_reset_required_rows'] = ()
    return env, robot, sim, writes, editor, storage, installer


def flat(extent=302.):
    generator = ContinuousTerrainGenerator(7, TerrainTuning(.7, (.2, 1.1), .25))
    request = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-extent, extent), (-extent, extent)))
    return generator.build(request, geometry_seed=17, resident_tiles=True)


def test_full_flat_domain_uses_small_exact_plane_tiles_and_one_shared_source():
    generator = ContinuousTerrainGenerator(7, TerrainTuning(.7, (.2, 1.1), .25))
    request = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-302., 302.), (-302., 302.)))
    first = generator.build(request, geometry_seed=17, resident_tiles=True)
    second = generator.build(replace(request, direction=-1), geometry_seed=18, resident_tiles=True)
    assert first.tile_source is second.tile_source
    assert len(first.surface.meshes) == 1
    assert first.request.extent == request.extent
    assert all(len(mesh.vertices_m) == 4 for mesh in first.surface.meshes)
    assert max(p.x_m[-1] for p in first.surface.patches) - min(p.x_m[0] for p in first.surface.patches) < 10
    keys = first.tile_source.tile_keys((120., 121.), (70., 71.), cells=8, halo=1)
    moved = first.with_tiles(keys)
    points = torch.tensor([[[120.25, 70.25], [120.875, 70.5]]])
    query = TerrainQueryBatch((moved.surface,)).at(points)
    assert query.valid.all() and query.height_m.eq(0).all()
    assert moved.sha256 == first.sha256


def test_metadata_only_reset_preserves_the_authored_tile_objects_before_live_streaming():
    env, _, _, _, editor, _, installer = admitted_rough(flat())
    ground = env.lifecycle.ground
    old = ground.courses
    fresh_surface = TerrainSurface(tuple(replace(p) for p in old[0].surface.patches))
    fresh = replace(old[0], geometry_seed=old[0].geometry_seed + 1, surface=fresh_surface)
    assert fresh.surface.meshes[0] is not old[0].surface.meshes[0]
    ground.replace_same_surface_courses((fresh, old[1]), torch.tensor([True, False]))
    bound = ground.courses[0]
    assert bound.sha256 == fresh.sha256 and bound.geometry_seed == fresh.geometry_seed
    assert bound.surface is old[0].surface
    keys = set(bound.resident_tiles)
    last = max(j for _, j in keys)
    moved = bound.with_tiles(keys | {(i, last + 1) for i, _ in keys})
    result = editor.replace_courses(ground, (moved, old[1]), torch.tensor([True, False]),
                                    protected_xy_m=torch.tensor([[[-1., 1.], [-1., 1.]]] * 2))
    installer.adopt(result)
    assert result.receipt['retained_collider_count'] == len(keys)
    assert result.scene.readback['terrain_reset_required_rows'] == ()


@pytest.mark.parametrize('kind', ['rough', 'flat'])
def test_live_tiles_preserve_protected_prims_and_never_request_a_reset(kind):
    env, _, sim, writes, editor, storage, installer = admitted_rough(flat() if kind == 'flat' else None)
    old = env.lifecycle.ground.courses
    old_paths = editor.scene.contacts.terrain_paths_by_environment
    count = len(storage.writes)
    keys = set(old[0].resident_tiles)
    last = max(j for _, j in keys)
    extended = keys | {(i, last+1) for i, _ in keys}
    courses = (old[0].with_tiles(extended), old[1])
    bounds = torch.tensor([[[-1., 1.], [-1., 1.]]]*2)
    result = editor.replace_courses(env.lifecycle.ground, courses, torch.tensor([True, False]), protected_xy_m=bounds)
    installer.adopt(result)
    assert result.receipt['live_tile_stream'] and result.receipt['retained_collider_count'] == len(keys)
    assert set(old_paths[0]) <= set(result.scene.contacts.terrain_paths_by_environment[0])
    assert not any(v[0] == 'enabled' and v[1] in old_paths[0] for v in storage.writes[count:])
    assert result.scene.readback['terrain_reset_required_rows'] == ()
    assert tuple(c.sha256 for c in old) == env.lifecycle.course_sha256
    assert sim.steps == 0 and not writes
    back = editor.replace_courses(result.ground, old, torch.tensor([True, False]), protected_xy_m=bounds)
    assert back.receipt['new_native_path_probe']['colliders_queried'] == 0
    assert set(back.scene.contacts.terrain_paths_by_environment[0]) == set(old_paths[0])
    assert back.scene.readback['terrain_reset_required_rows'] == ()


def test_live_tiles_cannot_remove_protected_geometry_or_replace_the_source():
    env, _, _, _, editor, _, _ = admitted_rough()
    old = env.lifecycle.ground.courses
    bounds = torch.tensor([[[-1., 1.], [-1., 1.]]]*2)
    required = old[0].tile_source.tile_keys((-1., 1.), (-1., 1.), cells=8, halo=0)
    bad = old[0].with_tiles(set(old[0].resident_tiles)-{required[0]})
    with pytest.raises(ValueError, match='protected'):
        editor.replace_courses(env.lifecycle.ground, (bad, old[1]), torch.tensor([True, False]), protected_xy_m=bounds)
    changed = replace(old[0], tile_source=replace(old[0].tile_source, geometry_seed=18),
                      surface=rough(22.).surface)
    with pytest.raises(ValueError, match='complete terrain source'):
        editor.replace_courses(env.lifecycle.ground, (changed, old[1]), torch.tensor([True, False]), protected_xy_m=bounds)


def test_flat_boundary_stream_does_no_native_edit_or_physics_step():
    env, _, sim, writes, editor, _, *_ = fixture()
    owner = SimpleNamespace(lifecycle=env.lifecycle, _scene=env.lifecycle.plant.scene, _identity=lambda: None)
    installer = NativeTerrainInstaller(env, owner, editor)
    geometry = SimpleNamespace(kinematics=SimpleNamespace(binding=env.lifecycle.plant.scene.observation.binding),
        _mesh_points=[(i, torch.tensor([[.2, 0., 0.]], dtype=torch.float64)) for i in range(19)], _cylinders=[])
    stream = NativeTerrainStream(installer, geometry)
    stream()
    assert stream.last_result is None and sim.steps == 0 and not writes


@pytest.mark.parametrize('kind', ['rough', 'flat'])
def test_stream_moves_distant_tiles_without_resetting_episode_or_control_identity(kind):
    env, robot, sim, writes, editor, _, installer = admitted_rough(flat() if kind == 'flat' else None)
    geometry = SimpleNamespace(kinematics=SimpleNamespace(binding=env.lifecycle.plant.scene.observation.binding),
        _mesh_points=[(i, torch.tensor([[.2, 0., 0.]], dtype=torch.float64)) for i in range(19)], _cylinders=[])
    stream = NativeTerrainStream(installer, geometry)
    stream()
    assert stream.last_result is None
    old = env.lifecycle.ground.courses
    before = (env.lifecycle.episode_id.clone(), env.lifecycle.control_epoch.clone(),
              env.lifecycle.plant.scene.observation.physical_tick)
    # 只在假SDK夹具中改变可见位置，驱动与PPO都没有运行。
    shift = float(stream._limits[0, 0, 1] - robot.data.body_link_pose_w.torch[0, :, 0].max()) + .01
    robot.data.root_link_pose_w.torch[0, 0] += shift
    robot.data.body_link_pose_w.torch[0, :, 0] += shift
    stream()
    assert stream.last_result.receipt['live_tile_stream']
    assert env.lifecycle.ground.courses[0].resident_tiles != old[0].resident_tiles
    assert env.lifecycle.ground.courses[1] is old[1]
    assert tuple(c.sha256 for c in old) == env.lifecycle.course_sha256
    assert torch.equal(before[0], env.lifecycle.episode_id)
    assert torch.equal(before[1], env.lifecycle.control_epoch)
    assert before[2] == env.lifecycle.plant.scene.observation.physical_tick
    assert sim.steps == 0 and not writes

"""Requested terrain bounds and completed-episode proficiency share one route."""
import json
from copy import deepcopy

import pytest
import torch

from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter


def router(tmp_path, *, count=16):
    tiles = [{'name': 'flat', 'kind': 0, 'level': 0, 'origin_m': [0., 0., 0.],
              'spawn_z_m': 0., 'valid_half_width_m': 10., 'finish_x_m': 10.,
              'start_x_m': -10., 'parameters': {}}]
    for level in range(10):
        for kind in (3, 7):
            tiles.append({**tiles[0], 'name': 'step' if kind == 3 else 'platforms',
                          'kind': kind, 'level': level,
                          'parameters': {'step_height_m': .02 * level}})
    for name, kind, profile in (('road_step', 3, [0., .2]),
                                 ('central_steps', 7, [0., .2, .35, .3])):
        tiles.append({**tiles[0], 'name': name, 'kind': kind, 'level': 9,
                      'parameters': {'step_height_m': .2, 'profile_m': profile}})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'format': 'terrain_course_atlas_v1', 'levels': 10,
                                'files': {}, 'tiles': tiles}))
    return TerrainCourseRouter(path, count, seed=71)


def request(count, up):
    return {'slope_rad': torch.zeros(count), 'roughness_m': torch.zeros(count),
            'step_up_m': torch.full((count,), float(up)), 'step_down_m': torch.full((count,), .15)}


def test_formal_request_preserves_success_and_failure_difficulty(tmp_path):
    r = router(tmp_path, count=2)
    r.level[:] = 4
    r.last_success[0] = True
    r.last_failure[1] = True
    r.reset(torch.arange(2), torch.full((2,), 3), stage=2,
            command_forward=torch.ones(2), requested=request(2, .2))
    assert r.level.tolist() == [5, 3]
    assert r.tile_levels[r.tile].tolist() == [5, 3]
    # A gentler command selects a smaller tile but cannot erase proficiency.
    r.reset(torch.arange(2), torch.full((2,), 3), stage=2,
            command_forward=torch.ones(2), requested=request(2, .025))
    assert r.level.tolist() == [5, 3]
    assert r.step_table[r.tile].le(.025).all()


@pytest.mark.parametrize('kind,name', [(3, 'road_step'), (7, 'central_steps')])
def test_authored_route_uses_real_height_and_mature_rough_rows(tmp_path, kind, name):
    r = router(tmp_path)
    ids = torch.arange(r.num_envs)
    r.level[:] = 6
    r.reset(ids, torch.full_like(ids, kind), stage=1,
            command_forward=torch.ones(len(ids)), requested=request(len(ids), .2))
    expected = next(i for i, t in enumerate(r.manifest['tiles']) if t['name'] == name)
    assert r.tile.eq(expected).all()
    assert r.level.eq(6).all()
    r.reset(ids, torch.full_like(ids, kind), stage=1,
            command_forward=torch.ones(len(ids)), requested=request(len(ids), 0))
    assert not r.authored[r.tile].any()
    assert r.step_table[r.tile].eq(0).all()
    r.level[:] = 0
    r.reset(ids, torch.full_like(ids, kind), stage=1,
            command_forward=torch.ones(len(ids)), requested=request(len(ids), .2))
    assert not r.authored[r.tile].any()


def test_old_authored_missing_profile_cannot_silently_become_zero(tmp_path):
    r = router(tmp_path)
    manifest = deepcopy(r.manifest)
    manifest['tiles'][-1]['parameters'] = {}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='real routing profile'):
        TerrainCourseRouter(path, 1)


def test_formal_routing_checkpoint_preserves_level_and_tie_rng(tmp_path):
    a = router(tmp_path)
    a.level[:] = 6
    saved = a.state_dict()
    b = router(tmp_path)
    b.load_state_dict(saved)
    for r in (a, b):
        r.reset(torch.arange(16), torch.full((16,), 3), stage=1,
                command_forward=torch.ones(16), requested=request(16, .2))
    torch.testing.assert_close(a.level, b.level)
    torch.testing.assert_close(a.tile, b.tile)
    torch.testing.assert_close(a.generator.get_state(), b.generator.get_state())

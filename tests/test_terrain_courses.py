"""Reset routing, exposure, and cold-state checks; no SDK application."""
from copy import deepcopy

import pytest
import torch

from transformer_rl.formal.tasks.curriculum import COMPLETE_STAGE_BUDGETS, TaskFamily
from transformer_rl.formal.lab_env.formal_curriculum import FormalCurriculumCapabilities, FormalTaskCurriculum
from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter, build_atlas

from test_terrain_assets import real_assets


@pytest.fixture(scope='module')
def atlas(real_assets, tmp_path_factory):
    _, assets, _ = real_assets
    output = tmp_path_factory.mktemp('atlas') / 'course'
    build_atlas(assets, output, seed=41, levels=3)
    return output / 'manifest.json'


def test_all_terrain_kinds_have_physical_tiles_and_level_endpoints(atlas):
    router = TerrainCourseRouter(atlas, 8)
    ids = torch.arange(8)
    origins, heights = router.reset(ids, ids, stage=2, command_forward=torch.ones(8))
    assert torch.equal(router.kinds[router.tile], ids)
    assert torch.isfinite(origins).all() and torch.isfinite(heights).all()
    assert {t['name'] for t in router.manifest['tiles']} >= {'road_step', 'central_steps', 'rough', 'stairs'}
    maxima = [t for t in router.manifest['tiles'] if t['name'] == 'two_platforms' and t['level'] == 2]
    assert maxima[0]['parameters']['platform_tops_m'] == [.2, .4]


def test_difficulty_uses_completed_episode_and_does_not_move_live_geometry(atlas):
    router = TerrainCourseRouter(atlas, 2)
    router.reset(torch.arange(2), torch.tensor([3, 3]), stage=2, command_forward=torch.ones(2))
    router.level[:] = 1
    original = router.origins.clone()
    router.observe_end(original + torch.tensor([[5., 0, 0], [0, 0, 0]]), torch.tensor([False, True]))
    assert torch.equal(original, router.origins)
    router.reset(torch.arange(2), torch.tensor([3, 3]), stage=2, command_forward=torch.ones(2))
    assert router.level.tolist() == [2, 0]


def test_restore_preserves_route_rng_and_rejects_different_atlas(atlas):
    first = TerrainCourseRouter(atlas, 8, seed=3)
    first.reset(torch.arange(8), torch.arange(8), stage=2, command_forward=torch.ones(8))
    saved = first.state_dict()
    second = TerrainCourseRouter(atlas, 8, seed=99)
    second.load_state_dict(saved)
    one = first.reset(torch.arange(8), torch.arange(8), stage=2, command_forward=torch.ones(8))
    two = second.reset(torch.arange(8), torch.arange(8), stage=2, command_forward=torch.ones(8))
    torch.testing.assert_close(one[0], two[0])
    wrong = deepcopy(saved)
    wrong['atlas_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        second.load_state_dict(wrong)


def test_complete_course_reaches_jump_without_reinitializing_sampler():
    curriculum = FormalTaskCurriculum(100, capabilities=FormalCurriculumCapabilities(True, True, True),
                                      include_jump=True)
    sampler = curriculum.sampler
    for _ in range(sum(COMPLETE_STAGE_BUDGETS)):
        curriculum.begin_update()
    assert curriculum.stage_name == 'jump'
    assert curriculum.budget.total_updates == 35000
    assert curriculum.status == 'training_budget_complete'
    assert curriculum.sampler is sampler
    draw = curriculum.sample(torch.arange(100))
    assert (draw.family == int(TaskFamily.REQUEST)).sum() == 50
    restored = FormalTaskCurriculum(100, capabilities=FormalCurriculumCapabilities(True, True, True), include_jump=True)
    restored.load_state_dict(curriculum.state_dict())
    assert restored.snapshot() == curriculum.snapshot()
    with pytest.raises(RuntimeError):
        restored.begin_update()

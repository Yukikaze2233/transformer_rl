"""Reset routing, exposure, and cold-state checks; no SDK application."""
from copy import deepcopy
from types import SimpleNamespace

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


def test_course_spawn_is_on_declared_approach_not_inside_first_rise(atlas):
    from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
    from transformer_rl.formal.lab_env.course_tasks import StairRequestProtocol
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.router = TerrainCourseRouter(atlas, 2)
    course.nominal_spawn_height_m = .32
    origins = torch.tensor([[3., 4, 0], [7., 8, 0]])
    course.env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins),
                                 _task_sampler=SimpleNamespace(stage=2), _as_torch=lambda value: value)
    task = SimpleNamespace(terrain_kind=torch.tensor([3, 0]), command=torch.zeros(2, 4))
    pose = torch.tensor([[3., 4, .32, 0, 0, 0, 1], [7., 8, .32, 0, 0, 0, 1]])
    course.place_reset(torch.arange(2), task, pose)
    local = pose[:, :3] - course.router.origins
    torch.testing.assert_close(local[:, 0], torch.tensor([-1., 0.]))
    assert local[:, 1].eq(0).all()
    torch.testing.assert_close(local[:, 2], course.router.spawn_z[course.router.tile] + .32)


def test_authored_assets_remain_available_below_their_atlas_level(atlas):
    from transformer_rl.formal.tasks.geometry import TerrainKind
    from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter

    router = TerrainCourseRouter(atlas, 256, seed=41, device='cpu')
    names = [tile['name'] for tile in router.manifest['tiles']]
    road = names.index('road_step')
    central = names.index('central_steps')
    assert router.authored[road] and router.authored[central]
    for kind, expected in ((int(TerrainKind.STEP_UP), road),
                           (int(TerrainKind.TWO_PLATFORMS), central)):
        ids = torch.arange(256)
        router.reset(ids, torch.full_like(ids, kind), stage=1, command_forward=torch.ones(256))
        selected = router.tile
        assert (router.kinds[selected] == kind).all()


def test_body_bump_clears_only_allowed_contact_not_other_failures():
    from types import SimpleNamespace
    from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
    from transformer_rl.formal.lab_env.course_tasks import StairRequestProtocol

    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.env = SimpleNamespace(_task=SimpleNamespace(minimum_height_m=.17, failure_tilt_rad=.9))
    course.body_contact_allowed = torch.tensor([True, True, True, True])
    terminated = torch.tensor([True, True, True, True])
    state = SimpleNamespace(height_above_ground_m=torch.tensor([.3, .3, .1, .3]))
    tilt = torch.tensor([.2, 1.2, .2, .2])
    base = torch.tensor([True, True, True, False])
    result = course.termination(terminated, state, tilt, base)
    assert result.tolist() == [False, True, True, True]


def _stair_command_with_truth(truth_height, *, return_course=False):
    from types import SimpleNamespace
    from transformer_rl.formal.lab_env.skill_fsm import SkillFSM
    from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
    from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestProtocol
    from transformer_rl.formal.lab_env.course_tasks import StairRequestProtocol

    n = 2
    fsm = SkillFSM(n)
    command = torch.tensor([[1., 0., 0., .32], [1., 0., 0., .32]])
    sample = SimpleNamespace(terrain_contact_force_world_n=torch.zeros(n, 19, 3))
    env = SimpleNamespace(
        num_envs=n,
        _device=torch.device('cpu'),
        _clock=lambda: torch.zeros(n, dtype=torch.float64),
        _completed_estimate=CompletedEstimate(
            torch.zeros(n), torch.ones(n, 2), torch.zeros(n), torch.zeros(n),
            torch.full((n,), -.01, dtype=torch.float64), torch.ones(n, dtype=torch.bool)),
        _ground_at_root=torch.full((n,), truth_height),
        _last_sample=sample,
        _wheel_body_indices=torch.tensor([1, 2]),
        _base_body_index=0,
        _last_tilt=torch.zeros(n),
        _command_base=command.clone(),
        _base_command_mode=torch.zeros(n, dtype=torch.long),
        _command=command.clone(),
        _command_mode=torch.zeros(n, dtype=torch.long),
        _request_height=torch.zeros(n),
        _skill_fsm=fsm,
    )
    admission = SimpleNamespace(admitted=command.clone())
    tasks = SimpleNamespace(
        active_id=torch.full((n,), -1, dtype=torch.long),
        protocol=RequestProtocol(n),
        step=lambda *args, **kwargs: (admission, torch.zeros(n, dtype=torch.long), torch.zeros(n), None))
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.env = env
    course.tasks = tasks
    course._skill_fsm = fsm
    course.body_contact_allowed = torch.zeros(n, dtype=torch.bool)
    course.stair_requests = StairRequestProtocol(n)
    course.manual_stair_request = torch.ones(n, dtype=torch.bool)
    course.manual_stair_height_m = torch.full((n,), .20)
    course.manual_stair_request_id = torch.zeros(n, dtype=torch.long)
    course.stair_cancel_id = torch.full((n,), -1, dtype=torch.long)
    course.tof_step_height_m = torch.zeros(n)
    course.tof_request_id = torch.full((n,), -1, dtype=torch.long)
    course.stair_next_id = torch.ones(n, dtype=torch.long)
    course._pending_requests = {'jump': None, 'stair': None}
    course._course_stair_pending = torch.zeros(n, dtype=torch.bool)
    course._course_stair_height = torch.zeros(n)
    course._last_stair_response = torch.zeros(n, dtype=torch.long)
    course.advance_skill()
    if return_course:
        return course
    return env._command.clone(), env._command_mode.clone(), fsm.stair.state.clone()


def test_stair_command_and_fsm_do_not_read_critic_terrain_truth():
    low = _stair_command_with_truth(0.)
    high = _stair_command_with_truth(10.)
    torch.testing.assert_close(low[0], high[0])
    torch.testing.assert_close(low[1], high[1])
    torch.testing.assert_close(low[2], high[2])


def test_cancelled_stair_stops_the_deployment_fsm_in_the_same_interval():
    course = _stair_command_with_truth(0., return_course=True)
    assert course.env._command_mode.eq(1).all()
    course.env._clock = lambda: torch.full((2,), .01, dtype=torch.float64)
    course.cancel_stair_request(torch.tensor([0]), torch.tensor([0]))
    course.advance_skill()
    assert course.env._command_mode.tolist() == [0, 1]
    assert course.stair_requests.active_id.tolist() == [-1, 0]


def test_stale_completed_estimate_aborts_stair_without_using_simulator_truth():
    course = _stair_command_with_truth(0., return_course=True)
    course.env._clock = lambda: torch.full((2,), .10, dtype=torch.float64)
    course.advance_skill()
    assert course.env._command_mode.eq(0).all()
    assert course.stair_requests.active_id.eq(-1).all()


def test_common_request_port_owns_packets_and_arbitrates_jump_before_new_stair():
    from test_lab_course_tasks import fixture
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    course = _stair_command_with_truth(0., return_course=True)
    course.stair_requests.clear(torch.ones(2, dtype=torch.bool))
    course.env._skill_fsm.stair.reset()
    course.tasks, _ = fixture()
    course.tasks.family.zero_()
    course.tasks.base_mode.zero_()
    course.tasks.step(torch.zeros(2, dtype=torch.float64), course.env._completed_estimate,
                      torch.zeros(2))
    course.env._clock = lambda: torch.full((2,), .01, dtype=torch.float64)
    caller_packet = RequestCommand(torch.tensor([17, 23]), torch.full((2,), .1), source='tof')
    course.submit_request('jump', caller_packet)
    caller_packet.height_m.fill_(.3)
    course.submit_request('stair', RequestCommand(torch.tensor([31, 37]), torch.full((2,), .2)))
    with pytest.raises(RuntimeError, match='unconsumed'):
        course.submit_request('jump', caller_packet)
    course.advance_skill()
    assert course.tasks.protocol.active_id.tolist() == [17, 23]
    torch.testing.assert_close(course.tasks.active_rise, torch.full((2,), .1))
    assert course.stair_requests.active_id.eq(-1).all()
    assert course.stair_requests.max_seen_id.tolist() == [31, 37]
    assert course.env._command_mode.eq(4).all()
    assert all(packet is None for packet in course._pending_requests.values())


def test_partial_reset_cancels_only_its_queued_external_request_rows():
    from dataclasses import fields, replace
    from test_lab_course_tasks import fixture
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    course = _stair_command_with_truth(0., return_course=True)
    course.tasks, sample = fixture()
    caller_packet = RequestCommand(torch.tensor([17, 23]), torch.full((2,), .1), source='manual')
    course.submit_request('jump', caller_packet)
    draw = replace(sample.draw, **{field.name: getattr(sample.draw, field.name)[:1]
                                  for field in fields(sample.draw)})
    row = replace(sample, draw=draw, command=sample.command[:1], request_height_m=sample.request_height_m[:1],
                  terrain_kind=sample.terrain_kind[:1], mode=sample.mode[:1], phase=sample.phase[:1])
    course.reset_commands(torch.tensor([0]), row)
    assert course._pending_requests['jump'].request_id.tolist() == [-1, 23]
    assert caller_packet.request_id.tolist() == [17, 23]


def _sampled_stair_course(heights=(.08, .18), kinds=(3, 7)):
    from dataclasses import replace
    from test_lab_course_tasks import fixture
    from transformer_rl.formal.control.observation import CommandMode

    course = _stair_command_with_truth(0., return_course=True)
    course.tasks, sample = fixture()
    kind = torch.tensor(kinds, dtype=torch.long)
    draw = replace(sample.draw, family=torch.full((2,), int(TaskFamily.TERRAIN)),
                   terrain_kind=kind, step_up_m=torch.tensor(heights))
    sample = replace(sample, draw=draw, terrain_kind=kind,
                     mode=torch.full((2,), int(CommandMode.STAIR)))
    course.env._skill_fsm.reset(torch.arange(2))
    course.reset_commands(torch.arange(2), sample)
    return course


def test_sampled_stair_commands_reach_the_same_request_port_once():
    from transformer_rl.formal.lab_env.skill_fsm import StairState

    course = _sampled_stair_course()
    course.advance_skill()
    accepted = course.stair_requests.active_id.clone()
    assert (accepted >= 0).all() and not course._course_stair_pending.any()
    torch.testing.assert_close(course.stair_requests.height_m, torch.tensor([.08, .18]))
    assert course.env._skill_fsm.stair.state.tolist() == [int(StairState.STEP_UP), int(StairState.BODY_BUMP)]
    course.env._clock = lambda: torch.full((2,), .01, dtype=torch.float64)
    course.advance_skill()
    torch.testing.assert_close(course.stair_requests.max_seen_id, accepted)


def test_sampled_stair_waits_for_estimate_and_external_request_owns_its_row():
    from dataclasses import replace
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    course = _sampled_stair_course()
    course.env._completed_estimate = replace(course.env._completed_estimate,
                                             valid=torch.tensor([False, True]))
    course.advance_skill()
    assert course._course_stair_pending.tolist() == [True, False]
    assert course.stair_requests.active_id[0] == -1
    course.env._clock = lambda: torch.full((2,), .01, dtype=torch.float64)
    course.env._completed_estimate = replace(course.env._completed_estimate,
                                             valid=torch.ones(2, dtype=torch.bool))
    course.submit_request('stair', RequestCommand(torch.tensor([900, -1]), torch.tensor([.15, 0.])))
    course.advance_skill()
    assert course.stair_requests.active_id[0] == 900
    assert not course._course_stair_pending.any()
    assert course.stair_next_id[0] == 901
    torch.testing.assert_close(course.stair_requests.height_m[0], torch.tensor(.15))


def test_downward_or_subcurb_courses_do_not_fabricate_upward_requests():
    course = _sampled_stair_course(heights=(.02, .18), kinds=(3, 4))
    course.advance_skill()
    assert course.stair_requests.active_id.eq(-1).all()
    assert not course._course_stair_pending.any()

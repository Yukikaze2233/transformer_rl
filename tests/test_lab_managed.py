"""Managed task-state roundtrips with real course RNG, without a simulator."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from test_policy_optimization import assert_identical
from test_structured_checkpoint import fixture, update
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, StairRequestProtocol
from transformer_rl.formal.lab_env.formal_curriculum import FormalCurriculumCapabilities, UnsupportedCurriculumTask
from transformer_rl.formal.lab_env.formal_reward import FormalRewardConfig
from transformer_rl.formal.lab_env.managed import ManagedCurriculum, ManagedLabRuntime
from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
from transformer_rl.formal.tasks.terrain_courses import TerrainCourseRouter
from transformer_rl.formal.tasks.trajectory import DriveEnvelope


def scene(tmp_path, components, *, seed=13):
    path = tmp_path / 'atlas.json'
    if not path.exists():
        tiles = [{'name': 'flat', 'kind': 0, 'level': 0, 'origin_m': [0., 0., 0.],
                  'spawn_z_m': 0., 'valid_half_width_m': 3., 'finish_x_m': 5., 'start_x_m': -1.5}]
        tiles.extend({**tiles[0], 'name': f'kind_{kind}', 'kind': kind, 'level': level,
                      'origin_m': [float(kind * 10), float(level * 10), 0.]}
                     for kind in range(1, 8) for level in range(3))
        path.write_text(json.dumps({'format': 'terrain_course_atlas_v1', 'levels': 3,
                                    'files': {}, 'tiles': tiles}))
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.router = TerrainCourseRouter(path, 4, seed=seed)
    course.tasks = LabCourseTasks(4, DriveEnvelope(.06, .4), seed=seed)
    course.stair_requests = StairRequestProtocol(4)
    course.stair_next_id = torch.zeros(4, dtype=torch.long)
    reward = SimpleNamespace(config=FormalRewardConfig(16.),
                             missing_term_reasons={'contact_slip': 'No verified contact-point relative tangent velocity producer',
                                                   'joint_boundary': 'No selected-asset geometric boundary producer'})
    env = SimpleNamespace(num_envs=4, device='cpu', _device=torch.device('cpu'), step_dt=.01, _physical_tick=0,
                          _terrain_course=course, _formal_reward=reward,
                          _clock=lambda: torch.zeros(4, dtype=torch.float64))
    course.env = env
    env.set_task_sampler = lambda sampler: setattr(env, '_task_sampler', sampler)
    return env


def identity(env):
    return {'atlas_sha256': env._terrain_course.router.identity,
            'asset_manifest_sha256': 'a' * 64, 'response_sha256': 'd' * 64}


def test_managed_course_shares_one_quota_and_refuses_double_reservation(tmp_path):
    recipe, components, journal, _ = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        course = ManagedCurriculum(recipe, components, FormalCurriculumCapabilities(True, True, True))
        assert course.budget is components.curriculum and course.sampler is components.sampler
        course.sample(torch.arange(4))
        assert components.curriculum.total_updates == 0
        with pytest.raises(RuntimeError, match='exclusively'):
            course.begin_update()
        with pytest.raises(RuntimeError, match='complete learning checkpoint'):
            course.load_state_dict(course.state_dict())
        with pytest.raises(UnsupportedCurriculumTask, match='terrain'):
            ManagedCurriculum(recipe, components, FormalCurriculumCapabilities(recovery=True, requests=True))


def test_managed_source_roundtrip_keeps_real_task_rng_ids_and_learning_budget(tmp_path):
    recipe, components, journal, store = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        env = scene(tmp_path, components)
        managed = ManagedLabRuntime(env, recipe, components, identity(env))
        assert env._terrain_course.tasks.protocol is components.protocol
        sample = env._task_sampler(torch.arange(4))
        course = env._terrain_course
        course.router.reset(torch.arange(4), sample.terrain_kind, stage=1, command_forward=sample.command[:, 0])
        course.tasks.reset(torch.arange(4), sample, env._clock())
        components.protocol.max_seen_id[:] = 7
        components.protocol.active_id[0] = 7
        components.protocol.height_m[0] = .1
        components.protocol.accepted_at_s[0] = .1
        components.protocol.last_time_s[:] = .2
        course.tasks.next_id[:] = 8
        course.tasks.paid_id[:] = 6
        update(components, journal)
        saved = managed.checkpoint_state_dict()
        assert 'protocol' not in saved['course']['tasks']
        assert 'contact_slip' in saved['reward_contract']['missing_terms']
        store.save('course.pt', components)
        restored = store.load('course.pt')
        assert restored.aborted_request_ids.tolist() == [7, -1, -1, -1]
        cold_env = scene(tmp_path, restored.components, seed=29)
        cold = ManagedLabRuntime(cold_env, recipe, restored.components, identity(cold_env))
        assert restored.components.task_source is cold
        assert restored.components.deferred_task_source_state is None
        assert restored.components.curriculum.consumed == [1, 0, 0, 0]
        assert restored.components.trainer.consumed == components.trainer.consumed
        assert_identical(restored.components.model.state_dict(), components.model.state_dict())
        assert_identical(restored.components.trainer.actor_optimizer.state_dict(),
                         components.trainer.actor_optimizer.state_dict())
        assert_identical(restored.components.trainer.critic_optimizer.state_dict(),
                         components.trainer.critic_optimizer.state_dict())
        assert_identical(cold.checkpoint_state_dict(), saved)
        assert cold_env._terrain_course.tasks.protocol is restored.components.protocol
        assert restored.components.protocol.active_id.eq(-1).all()
        assert restored.components.protocol.max_seen_id.eq(7).all()
        for first, second in ((course.router.generator, cold.course.router.generator),
                               (course.tasks.generator, cold.course.tasks.generator)):
            torch.testing.assert_close(torch.rand(16, generator=first), torch.rand(16, generator=second))
        with pytest.raises(ValueError, match='before fresh physical sampling'):
            cold.load_after_explicit_cold_setup(saved)
        restored.restore_global_rng_after_cold_setup()


@pytest.mark.parametrize('change', ('identity', 'reward', 'watermark', 'live_physics', 'legacy_format'))
def test_managed_restore_refuses_incompatible_or_live_task_state(tmp_path, change):
    recipe, components, journal, store = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        env = scene(tmp_path, components)
        managed = ManagedLabRuntime(env, recipe, components, identity(env))
        store.save('course.pt', components)
        restored = store.load('course.pt')
        saved = restored.components.deferred_task_source_state
        cold_env = scene(tmp_path, restored.components)
        if change == 'identity':
            saved['identity']['asset_manifest_sha256'] = 'e' * 64
        elif change == 'reward':
            saved['reward_contract']['missing_terms']['contact_slip'] = 'silently enabled'
        elif change == 'watermark':
            saved['course']['tasks']['next_id'][:] = -1
        elif change == 'legacy_format':
            saved['format'] = 'managed_lab_task_source_v1'
        else:
            cold_env._physical_tick = 1
        with pytest.raises(ValueError, match='original sources|watermarks'):
            ManagedLabRuntime(cold_env, recipe, restored.components, identity(cold_env))
        assert restored.components.task_source is None
        assert restored.components.deferred_task_source_state is not None


def test_cold_source_keeps_stair_deduplication_and_cumulative_request_counters(tmp_path):
    recipe, components, journal, store = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        env = scene(tmp_path, components)
        managed = ManagedLabRuntime(env, recipe, components, identity(env))
        course = managed.course
        course.stair_requests.max_seen_id[0] = 19
        course.stair_requests.active_id[0] = 19
        course.stair_next_id[0] = 20
        course.tasks.external_request_counters[1] = 4
        course.tasks.kind_counts[0] = 17
        store.save('stair.pt', components)
        restored = store.load('stair.pt')
        cold_env = scene(tmp_path, restored.components)
        cold = ManagedLabRuntime(cold_env, recipe, restored.components, identity(cold_env))
        assert cold.course.stair_requests.max_seen_id.tolist() == [19, -1, -1, -1]
        assert cold.course.stair_requests.active_id.eq(-1).all()
        assert cold.course.stair_next_id.tolist() == [20, 0, 0, 0]
        assert cold.course.tasks.external_request_counters[1] == 4
        assert cold.course.tasks.kind_counts[0] == 17


def test_managed_collector_uses_the_full_learner_without_rebuilding_it(tmp_path):
    recipe, components, journal, _ = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        env = scene(tmp_path, components)
        managed = ManagedLabRuntime(env, recipe, components, identity(env))
        collector = managed.create_collector(components.model, recipe.collection)
        assert collector.model is components.model and collector.env is managed.environment
        with pytest.raises(ValueError, match='same learner model'):
            managed.create_collector(object(), recipe.collection)
        env._task_sampler.curriculum.sampler = deepcopy(components.sampler)
        with pytest.raises(ValueError, match='shares its learner state'):
            managed.checkpoint_state_dict()


def test_low_frequency_reward_projection_labels_its_interval_scope(tmp_path):
    recipe, components, journal, _ = fixture(tmp_path, full=True, target_rollouts=35000)
    with journal:
        env = scene(tmp_path, components)
        managed = ManagedLabRuntime(env, recipe, components, identity(env))
        values = torch.tensor([1., 2., 3., 4.])
        env._formal_result = SimpleNamespace(component_densities={'tracking': values},
            raw_positive_densities={'tracking': values * 2}, raw_cost_densities={'effort': values * .1},
            event_rewards={'success': values * 0}, gate_fractions={'support': torch.ones(4)})
        env._terrain_course.metrics = lambda: {'scan_misses': 0, 'request_responses': [0] * 9}
        report = managed.diagnostics()
        assert report['scope'] == 'last_completed_policy_interval_across_environments'
        assert report['pooled_rollout_statistics'] is False
        assert report['reward']['bounded_density_mean']['tracking'] == 2.5
        assert report['reward']['raw_positive_density_mean']['tracking'] == 5.
        assert report['reward']['event_reward_mean']['success'] == 0.
        assert 'contact_slip' in report['reward_contract']['missing_terms']
        assert components.curriculum.total_updates == 0

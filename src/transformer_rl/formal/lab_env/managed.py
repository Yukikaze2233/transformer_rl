"""Fast Lab collection inside the existing managed learning transaction.

The journal owns update reservations. The Lab task samples episodes from the
same curriculum/sampler and checkpoints only task RNG and durable identities;
physics, sensor histories and active requests are rebuilt during cold setup.
This module imports no simulator.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json

import torch

from ..learning.checkpoint import LearningComponents, LearningRecipe
from .adapter import LabLabeledEnv
from .collector import LabCollector
from .formal_curriculum import FormalCurriculumCapabilities, FormalTaskCurriculum, LabTaskDrawSampler


class ManagedCurriculum(FormalTaskCurriculum):
    """Episode sampling shares the learner's sole budget and quota RNG."""

    def __init__(self, recipe, components, capabilities):
        if not isinstance(recipe, LearningRecipe) or not isinstance(components, LearningComponents):
            raise TypeError('managed course requires its parsed learning recipe and components')
        names, budgets, weights = recipe.stage_schedule
        if (components.curriculum.stage_names != names or components.curriculum.stage_budgets != budgets
                or components.sampler.stage_weights != weights or components.sampler.domain != recipe.domain
                or components.sampler.device != torch.device(recipe.device)):
            raise ValueError('managed course differs from the learner curriculum or sampler')
        super().__init__(recipe.num_envs, device=recipe.device, seed=recipe.training_seed,
                         promotion=recipe.curriculum_promotion, capabilities=capabilities,
                         domain=recipe.domain, include_jump=len(names) == 4,
                         continuous=len(names) == 4)
        if components.sampler.continuous != self.continuous:
            raise ValueError('managed course sampler continuous mode differs from its recipe')
        self.budget, self.sampler = components.curriculum, components.sampler
        for stage in range(len(names)):
            self.require_supported(stage)

    def begin_update(self):
        raise RuntimeError('managed update reservations belong exclusively to the learning journal')

    def load_state_dict(self, state):
        raise RuntimeError('managed curriculum restores through the complete learning checkpoint')


class ManagedLabRuntime:
    """Bind an already constructed Lab scene to full managed checkpoints.

    Construction happens before collector.reset. A cold source cannot replay a
    saved physical episode or request, but its deduplication IDs, difficulty,
    task quota and independent RNG streams survive the new physical reset.
    """

    _FORMAT = 'managed_lab_task_source_v4'

    def __init__(self, env, recipe, components, identity, *, profile=False, physics_health=None):
        if type(profile) is not bool or not isinstance(identity, dict):
            raise ValueError('managed Lab identity and profiling must be explicit')
        if (components.task_source is not None or env.num_envs != recipe.num_envs
                or torch.device(env.device) != torch.device(recipe.device)
                or recipe.policy.critic_input_dim != 185 or recipe.policy.history_length != 5):
            raise ValueError('managed Lab requires its sole 35D/H5/185D learner source')
        self.env, self.recipe, self.components = env, recipe, components
        from .sensors import LabSensorModel

        self.sensor_model = env._sensor_model
        if (not isinstance(self.sensor_model, LabSensorModel)
                or self.sensor_model.num_envs != recipe.num_envs
                or self.sensor_model.device != torch.device(recipe.device)):
            raise ValueError('managed Lab requires the actual explicit actor sensor model')
        self.mechanical_identity = env._startup_dynamics.receipt['native_parameters_sha256']
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self.profile = profile
        self.physics_health = physics_health
        curriculum = ManagedCurriculum(recipe, components, FormalCurriculumCapabilities(True, True, True))
        env.set_task_sampler(LabTaskDrawSampler(curriculum))
        self.course = env._terrain_course
        if (self.course is None or env._formal_reward is None
                or self.course.tasks.protocol.config != components.protocol.config):
            raise ValueError('managed Lab requires real terrain, formal reward and the original request contract')
        # There is one authoritative request watermark in the learning snapshot.
        # Do not keep an unrelated RequestProtocol inside the course task.
        self.course.tasks.protocol = components.protocol
        self.environment = LabLabeledEnv(env)
        self._restored = False
        saved = components.deferred_task_source_state
        if saved is not None:
            self.load_after_explicit_cold_setup(saved)
            components.deferred_task_source_state = None
        elif components.trainer.update_index:
            raise ValueError('consumed Lab learning requires its complete saved task source')
        components.task_source = self

    def _reward_contract(self):
        reward = self.env._formal_reward
        return json.loads(json.dumps({'configuration': asdict(reward.config),
                                     'missing_terms': reward.missing_term_reasons}, allow_nan=False))

    def _require_binding(self):
        if (self.course is not self.env._terrain_course
                or self.sensor_model is not self.env._sensor_model
                or self.course.tasks.protocol is not self.components.protocol
                or self.env._task_sampler.curriculum.budget is not self.components.curriculum
                or self.env._task_sampler.curriculum.sampler is not self.components.sampler):
            raise ValueError('managed Lab task source no longer shares its learner state')

    def checkpoint_state_dict(self):
        self._require_binding()
        course = self.course.state_dict()
        # Protocol is serialized by StructuredCheckpointStore exactly once.
        # It is restored/aborted there before this source rebuilds its episode.
        course['tasks'] = {key: value for key, value in course['tasks'].items() if key != 'protocol'}
        return {'format': self._FORMAT, 'identity': deepcopy(self.identity),
                'reward_contract': self._reward_contract(), 'course': course,
                'sensor_model': self.sensor_model.checkpoint_state_dict(),
                'startup_parameters_sha256': self.mechanical_identity,
                'cold_physical_state_restored': False}

    def load_after_explicit_cold_setup(self, state):
        self._require_binding()
        expected = {'format', 'identity', 'reward_contract', 'course', 'cold_physical_state_restored',
                    'sensor_model', 'startup_parameters_sha256'}
        if (self._restored or self.env._physical_tick != 0
                or not isinstance(state, dict) or set(state) != expected
                or state['format'] != self._FORMAT or state['identity'] != self.identity
                or state['reward_contract'] != self._reward_contract()
                or state['startup_parameters_sha256'] != self.mechanical_identity
                or state['cold_physical_state_restored'] is not False):
            raise ValueError('managed Lab restore requires the original sources before fresh physical sampling')
        course = deepcopy(state['course'])
        if not isinstance(course, dict):
            raise ValueError('managed Lab course snapshot must be a mapping')
        task = course.get('tasks', {})
        protocol = self.components.protocol
        course_keys = {'format', 'router', 'tasks', 'stair_requests', 'stair_next_id'}
        if course.get('format') == 'lab_terrain_course_v3':
            course_keys |= {'recovery', 'birth_family_counts'}
        elif course.get('format') != 'lab_terrain_course_v2':
            raise ValueError('managed Lab course snapshot format differs')
        if (set(course) != course_keys
                or not isinstance(task, dict)
                or set(task) != {'format', 'next_id', 'paid_id', 'rng',
                                *self.course.tasks._COUNTERS}):
            raise ValueError('managed Lab task snapshot fields differ')
        for name in ('next_id', 'paid_id', *self.course.tasks._COUNTERS):
            reference = getattr(self.course.tasks, name)
            value = task[name]
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError('managed Lab task snapshot tensor differs: ' + name)
        next_id, paid = task['next_id'].to(protocol.device), task['paid_id'].to(protocol.device)
        if ((next_id <= protocol.max_seen_id).any() or (paid < -1).any()
                or (paid > protocol.max_seen_id).any() or (task['request_counters'] < 0).any()
                or (protocol.active_id >= 0).any()):
            raise ValueError('managed Lab request watermarks differ from the cold learning protocol')
        task['protocol'] = protocol.state_dict()
        self.sensor_model.load_after_explicit_cold_setup(state['sensor_model'])
        self.course.restore_after_physical_reset(course)
        self._restored = True

    def create_collector(self, model, collection):
        self._require_binding()
        if model is not self.components.model or collection != self.recipe.collection:
            raise ValueError('managed Lab collector must use the same learner model and collection contract')
        return _ManagedCollector(self, model, collection)

    def diagnostics(self):
        """Sample the last complete policy interval, never label it pooled data."""
        reward = self.env._formal_result
        if reward is None:
            raise RuntimeError('managed Lab collection did not publish its formal reward interval')
        groups = {'bounded_density_mean': reward.component_densities,
                  'raw_positive_density_mean': reward.raw_positive_densities,
                  'raw_cost_density_mean': reward.raw_cost_densities,
                  'event_reward_mean': reward.event_rewards,
                  'gate_fraction_mean': reward.gate_fractions}
        keys, values = [], []
        for group, terms in groups.items():
            for name, value in terms.items():
                keys.append((group, name))
                values.append(value.detach().float().mean())
        means = torch.stack(values).cpu().tolist() if values else []
        summary = {group: {} for group in groups}
        for (group, name), value in zip(keys, means):
            summary[group][name] = value
        return {'scope': 'last_completed_policy_interval_across_environments',
                'pooled_rollout_statistics': False, 'reward': summary,
                'reward_contract': self._reward_contract(), 'course': self.course.metrics(),
                'wheel_lookahead': self.course.lookahead_summary()}


class _ManagedCollector(LabCollector):
    def __init__(self, runtime, model, collection):
        super().__init__(runtime.environment, model, collection, profile=runtime.profile)
        self.runtime = runtime

    def collect(self, steps=48):
        batch = super().collect(steps)
        if self.runtime.physics_health is not None:
            self.runtime.physics_health.check()
        update = self.runtime.components.trainer.update_index + 1
        if update == 1 or update % self.runtime.recipe.ppo.diagnostics_interval == 0:
            self.last_metrics['task_diagnostics'] = self.runtime.diagnostics()
        return batch

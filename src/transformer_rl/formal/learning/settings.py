"""正式训练的采样与学习配置；资产、SDK和资源由运行层绑定。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from .checkpoint import LearningRecipe
from .optimization import StructuredPPOConfig
from ..contracts import StructuredCollectionConfig
from ..policy import PolicyConfig, policy_config_from_dict
from ..tasks.curriculum import COMPLETE_STAGE_BUDGETS, TaskDomain
from ..tasks.protocol import RequestProtocolConfig
from ..tasks.terrain import TerrainTuning


@dataclass(frozen=True)
class FormalTrainingSettings:
    training_seeds: tuple[int, ...]
    updates_per_seed: int
    num_envs: int
    rollout_steps: int
    checkpoint_interval: int
    policy: PolicyConfig
    ppo: StructuredPPOConfig
    collection: StructuredCollectionConfig
    domain: TaskDomain
    protocol: RequestProtocolConfig
    terrain: TerrainTuning

    def __post_init__(self):
        if (not isinstance(self.training_seeds, (tuple, list)) or not self.training_seeds
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.training_seeds)
                or len(set(self.training_seeds)) != len(self.training_seeds)
                or type(self.updates_per_seed) is not int or self.updates_per_seed != sum(COMPLETE_STAGE_BUDGETS)
                or any(type(getattr(self, name)) is not int or getattr(self, name) < 1
                       for name in ('num_envs', 'rollout_steps', 'checkpoint_interval'))
                or any(not isinstance(getattr(self, name), cls)
                       for name, cls in LearningRecipe._config_types().items())):
            raise ValueError('formal settings require explicit seeds, complete course and parsed configurations')
        object.__setattr__(self, 'training_seeds', tuple(self.training_seeds))
        if self.num_envs * self.rollout_steps % self.ppo.minibatches:
            raise ValueError('formal rollout must divide into complete logical minibatches')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError('formal settings fields differ')
        arguments = dict(value)
        for name, config_type in LearningRecipe._config_types().items():
            raw = arguments[name]
            if not isinstance(raw, dict):
                raise ValueError('formal configuration must be an object: ' + name)
            if name == 'policy':
                arguments[name] = policy_config_from_dict(raw)
                continue
            try:
                arguments[name] = config_type(**raw)
            except TypeError as error:
                raise ValueError('unknown formal configuration fields: ' + name) from error
        return cls(**arguments)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def recipe(self, run_id, training_seed, *, asset_manifest_sha256, learner_source_sha256,
               task_contract_sha256, initial_model_sha256, device='cuda:0'):
        if type(training_seed) is not int or training_seed not in self.training_seeds:
            raise ValueError('training seed is not part of this formal plan')
        recipe = LearningRecipe(
            run_id, training_seed, self.num_envs, self.rollout_steps,
            asset_manifest_sha256, learner_source_sha256, task_contract_sha256, initial_model_sha256,
            **{name: getattr(self, name) for name in LearningRecipe._config_types()},
            checkpoint_interval=self.checkpoint_interval, device=device, curriculum_promotion='exposure',
            target_rollouts=self.updates_per_seed)
        if recipe.journal_spec().target_rollouts != self.updates_per_seed:
            raise ValueError('learning journal and formal course budgets differ')
        return recipe

    def budget(self):
        fresh = self.num_envs * self.rollout_steps
        return {'fresh_per_update': fresh, 'fresh_per_seed': fresh * self.updates_per_seed,
                'total_fresh': fresh * self.updates_per_seed * len(self.training_seeds),
                'rollout_duration_s': self.rollout_steps * self.collection.policy_dt_s,
                'logical_minibatch_endpoints': fresh // self.ppo.minibatches,
                'maximum_actor_steps_per_update': self.ppo.epochs * self.ppo.minibatches,
                'maximum_actor_endpoint_uses_per_update': fresh * self.ppo.epochs}

"""Private complete learning-state snapshots for the structured candidate.

This is not the existing packed checkpoint format or a registered production
endpoint. Loading validates a new object graph without changing caller RNG.
The caller separately proves cold physical setup and safe worker ownership.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace

import torch

from ...common.artifacts import _cpu_snapshot, _json_metadata, _publish_new_files, _validate_optimizer_state, _validate_tensor
from ...common.random_state import capture_rng, restore_rng, _validate_rng
from .accounting import LearningJournal, LearningJournalSpec
from .optimization import StructuredPPOConfig, StructuredPPOTrainer
from ..contracts import StructuredCollectionConfig
from ..policy import StructuredActorCritic, StructuredPolicyConfig
from ..tasks.curriculum import ContinuousTaskSampler, CurriculumBudget, TaskDomain
from ..tasks.protocol import RequestProtocol, RequestProtocolConfig
from ..tasks.terrain import ContinuousTerrainGenerator, TerrainTuning
from ...common.training_control import read_json


def _document(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


@dataclass(frozen=True)
class LearningRecipe:
    """Private source-bound recipe; no simulator or production admission flag."""

    run_id: str
    training_seed: int
    num_envs: int
    rollout_steps: int
    asset_manifest_sha256: str
    learner_source_sha256: str
    task_contract_sha256: str
    initial_model_sha256: str
    policy: StructuredPolicyConfig
    ppo: StructuredPPOConfig
    collection: StructuredCollectionConfig
    domain: TaskDomain
    protocol: RequestProtocolConfig
    terrain: TerrainTuning
    device: str = 'cpu'
    curriculum_promotion: str = 'exposure'
    checkpoint_interval: int = 25

    def __post_init__(self):
        for name in ('asset_manifest_sha256', 'learner_source_sha256', 'task_contract_sha256', 'initial_model_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError('learning recipe requires exact source/asset/initial-model hashes')
        if (not isinstance(self.run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', self.run_id)
                or type(self.training_seed) is not int or not 0 <= self.training_seed < 2**32
                or type(self.num_envs) is not int or self.num_envs < 1
                or type(self.rollout_steps) is not int or self.rollout_steps < 1
                or type(self.checkpoint_interval) is not int or self.checkpoint_interval < 1
                or self.device not in ('cpu', 'cuda:0')
                or self.curriculum_promotion not in ('exposure', 'audited')
                or any(not isinstance(getattr(self, name), cls) for name, cls in self._config_types().items())):
            raise ValueError('learning recipe requires parsed configurations and an explicit seed/device/rollout')

    @staticmethod
    def _config_types():
        return {'policy': StructuredPolicyConfig, 'ppo': StructuredPPOConfig, 'collection': StructuredCollectionConfig,
                'domain': TaskDomain, 'protocol': RequestProtocolConfig, 'terrain': TerrainTuning}

    def to_dict(self):
        return _document(asdict(self))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError('learning recipe fields differ')
        arguments = dict(value)
        try:
            for name, config_type in cls._config_types().items():
                raw = arguments[name]
                if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(config_type)}:
                    raise ValueError('learning configuration fields differ: ' + name)
                arguments[name] = config_type(**raw)
            return cls(**arguments)
        except TypeError as error:
            raise ValueError('invalid learning recipe') from error

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def journal_spec(self):
        return LearningJournalSpec(self.run_id, self.sha256, self.num_envs, self.rollout_steps,
                                   self.ppo.epochs, self.ppo.minibatches)


@dataclass
class LearningComponents:
    model: StructuredActorCritic
    trainer: StructuredPPOTrainer
    curriculum: CurriculumBudget
    sampler: ContinuousTaskSampler
    terrain: ContinuousTerrainGenerator
    protocol: RequestProtocol
    attempted_vector_steps: int = 0
    returned_vector_steps: int = 0
    task_source: object | None = None
    deferred_task_source_state: dict | None = None

    def observed_consumption(self, num_envs):
        return {'attempted_transitions': self.attempted_vector_steps * num_envs,
                'returned_transitions': self.returned_vector_steps * num_envs,
                'ppo_updates': self.trainer.update_index,
                'ppo_fresh_endpoints': self.trainer.consumed['fresh_endpoints'],
                **{key: self.trainer.consumed[key] for key in LearningJournal._ACTUAL[4:]}}

    def validate_binding(self, recipe, journal):
        if (self.trainer.model is not self.model or self.model.config != recipe.policy
                or self.trainer.config != recipe.ppo or self.sampler.domain != recipe.domain
                or self.terrain.tuning != recipe.terrain or self.protocol.config != recipe.protocol
                or self.protocol.num_envs != recipe.num_envs or self.protocol.device != torch.device(recipe.device)
                or self.curriculum.promotion != recipe.curriculum_promotion
                or self.sampler.device != torch.device(recipe.device)
                or any(type(v) is not int or v < 0 for v in
                       (self.attempted_vector_steps, self.returned_vector_steps, self.trainer.update_index))
                or self.returned_vector_steps > self.attempted_vector_steps
                or journal.spec != recipe.journal_spec()):
            raise ValueError('learning components differ from their source-bound recipe')
        if any(value.dtype != torch.float32 or value.device != torch.device(recipe.device)
               for value in self.model.parameters()):
            raise ValueError('learning model requires its original float32 device')
        boundary = journal.checkpoint_boundary()
        if (self.curriculum.total_updates != boundary['rollouts_reserved']
                or self.curriculum.consumed != boundary['reserved_by_stage']
                or any(value != boundary[key] for key, value in self.observed_consumption(recipe.num_envs).items())):
            raise ValueError('learning state and nonrefundable consumption disagree')


@dataclass
class RestoredLearning:
    recipe: LearningRecipe
    components: LearningComponents
    global_rng: dict
    aborted_request_ids: torch.Tensor
    checkpoint_boundary: dict
    _rng_restored: bool = False

    def restore_task_source(self, source):
        """Restore source RNG into an empty native lifecycle before cold setup."""
        state = self.components.deferred_task_source_state
        if (state is None or self.components.task_source is not None or self._rng_restored
                or not callable(getattr(source, 'load_after_explicit_cold_setup', None))):
            raise ValueError('task source restore requires its deferred state and an empty cold source')
        source.load_after_explicit_cold_setup(_cpu_snapshot(state))
        self.components.task_source = source
        self.components.deferred_task_source_state = None

    def restore_global_rng_after_cold_setup(self):
        """Invoke after the separately certified physical reset/warmup, once.

        This restores sampling RNG only. It does not certify a physical reset,
        authorize a cross-boot run, or restore contact/history/actuator state.
        """
        if self._rng_restored:
            raise RuntimeError('checkpoint sampling RNG was already restored')
        if self.components.deferred_task_source_state is not None:
            raise RuntimeError('restore the task source before completing cold physical setup')
        restore_rng(self.global_rng)
        self._rng_restored = True


class StructuredCheckpointStore:
    """New-file publication, weights-only loading, bounded own-directory reads."""

    _FORMAT = 'private_structured_learning_snapshot'
    _LIMIT = 64 * 1024 * 1024

    def __init__(self, directory, recipe: LearningRecipe, journal: LearningJournal):
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir() or not isinstance(recipe, LearningRecipe) or journal.spec != recipe.journal_spec():
            raise ValueError('checkpoint store must bind its own directory, full recipe and consumption journal')
        self.recipe, self.journal = recipe, journal

    @staticmethod
    def _validate_weights(state, model):
        expected = model.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError('structured snapshot model fields differ')
        buffers = dict(model.named_buffers())
        for name, reference in expected.items():
            _validate_tensor(name, state[name], reference.shape, reference.dtype)
            if name in buffers and not torch.equal(state[name].cpu(), reference.cpu()):
                raise ValueError('fixed input/mask/time buffer differs: ' + name)

    @staticmethod
    def _validate_adam(state, module, consumed_steps, learning_rate):
        _validate_optimizer_state(state, SimpleNamespace(policy_parameters=lambda: list(module.parameters())))
        group = state['param_groups'][0]
        if (group['lr'] != learning_rate or tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-8
                or group['weight_decay'] != 0 or group['amsgrad'] or group['maximize']):
            raise ValueError('structured Adam settings differ from the selected optimizer')
        if any(float(entry['step']) > consumed_steps for entry in state['state'].values()):
            raise ValueError('retained Adam steps exceed executed nonrefundable steps')

    def _path(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}\.pt', name):
            raise ValueError('checkpoint names must stay directly inside their own private store')
        return self.directory / name

    @staticmethod
    def model_sha256(model):
        digest = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            header = json.dumps({'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape)},
                                sort_keys=True, separators=(',', ':')).encode()
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            for part in (header, raw):
                digest.update(len(part).to_bytes(8, 'big'))
                digest.update(part)
        return digest.hexdigest()

    def save(self, name, components: LearningComponents, *, metadata=None):
        path = self._path(name)
        components.validate_binding(self.recipe, self.journal)
        if components.deferred_task_source_state is not None:
            raise ValueError('cannot save before the deferred task source has been restored')
        task_source = None
        if components.task_source is not None:
            task_source = components.task_source.checkpoint_state_dict()
            if not isinstance(task_source, dict):
                raise ValueError('task source snapshots must be a state dictionary')
            task_source = _cpu_snapshot(task_source)
        if (self.journal.summary()['rollouts_reserved'] == 0
                and self.model_sha256(components.model) != self.recipe.initial_model_sha256):
            raise ValueError('fresh structured model differs from its frozen initialization')
        rng = _cpu_snapshot(capture_rng())
        payload = {'format': self._FORMAT, 'recipe': self.recipe.to_dict(), 'boundary': self.journal.checkpoint_boundary(),
                   'model': _cpu_snapshot(dict(components.model.state_dict())),
                   'actor_adam': _cpu_snapshot(components.trainer.actor_optimizer.state_dict()),
                   'critic_adam': _cpu_snapshot(components.trainer.critic_optimizer.state_dict()),
                   'trainer': {'update_index': components.trainer.update_index, 'consumed': dict(components.trainer.consumed)},
                   'collection': {'attempted_vector_steps': components.attempted_vector_steps,
                                  'returned_vector_steps': components.returned_vector_steps,
                                  'task_source': task_source},
                   'curriculum': components.curriculum.state_dict(), 'sampler': _cpu_snapshot(components.sampler.state_dict()),
                   'terrain': _cpu_snapshot(components.terrain.state_dict()), 'protocol': _cpu_snapshot(components.protocol.state_dict()),
                   'global_rng': rng, 'normalization': 'fixed_manual35_81_no_running_statistics',
                   'metadata': _json_metadata({} if metadata is None else metadata)}
        return self._publish_payload(name, payload)

    def _publish_payload(self, name, payload):
        """发布完整状态；迁移同样必须经过原有全部验收谓词。"""
        path = self._path(name)
        # Validate against fresh canonical masks/buffers, not the potentially
        # drifted live buffers. Construction is invisible to caller CPU RNG.
        self._validate_payload(payload)
        stream = io.BytesIO()
        torch.save(payload, stream)
        raw = stream.getvalue()
        if len(raw) > self._LIMIT:
            raise ValueError('structured snapshot exceeds its bounded candidate size')
        receipt = {'format': self._FORMAT, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'recipe_sha256': self.recipe.sha256, 'boundary': payload['boundary'],
                   'cold_physical_state_restored': False, 'metadata': payload['metadata']}
        _publish_new_files({path: raw, Path(str(path) + '.json'): json.dumps(receipt,indent=2,allow_nan=False).encode()})
        return receipt

    def _validate_payload(self, payload):
        expected = {'format', 'recipe', 'boundary', 'model', 'actor_adam', 'critic_adam', 'trainer', 'collection',
                    'curriculum', 'sampler', 'terrain', 'protocol', 'global_rng', 'normalization', 'metadata'}
        if (not isinstance(payload, dict) or set(payload) != expected or payload['format'] != self._FORMAT
                or LearningRecipe.from_dict(payload['recipe']) != self.recipe
                or payload['normalization'] != 'fixed_manual35_81_no_running_statistics'):
            raise ValueError('structured snapshot recipe, format or fields differ')
        self.journal.require_exact_boundary(payload['boundary'])
        _validate_rng(payload['global_rng'])
        _json_metadata(payload['metadata'])
        if self.recipe.device == 'cuda:0' and not torch.cuda.is_initialized():
            raise ValueError('CUDA restore requires an already admitted and initialized device context')
        if torch.get_default_dtype() != torch.float32:
            raise ValueError('structured checkpoint validation requires the frozen float32 construction context')
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            model = StructuredActorCritic(self.recipe.policy)
        self._validate_weights(payload['model'], model)
        model.load_state_dict(payload['model'], strict=True)
        model.to(self.recipe.device).eval()
        trainer = StructuredPPOTrainer(model, self.recipe.ppo)
        stats = payload['trainer']
        if (not isinstance(stats, dict) or stats.keys() != {'update_index', 'consumed'}
                or type(stats['update_index']) is not int or stats['update_index'] < 0
                or not isinstance(stats['consumed'], dict) or stats['consumed'].keys() != trainer.consumed.keys()
                or any(type(v) is not int or v < 0 for v in stats['consumed'].values())):
            raise ValueError('structured trainer counters are invalid')
        self._validate_adam(payload['actor_adam'], model.actor, stats['consumed']['actor_optimizer_steps'], self.recipe.ppo.learning_rate)
        self._validate_adam(payload['critic_adam'], model.critic, stats['consumed']['critic_optimizer_steps'], self.recipe.ppo.learning_rate)
        trainer.actor_optimizer.load_state_dict(payload['actor_adam'])
        trainer.critic_optimizer.load_state_dict(payload['critic_adam'])
        trainer.update_index, trainer.consumed = stats['update_index'], dict(stats['consumed'])
        budget = CurriculumBudget(promotion=self.recipe.curriculum_promotion)
        budget.load_state_dict(payload['curriculum'])
        sampler = ContinuousTaskSampler(self.recipe.training_seed, device=self.recipe.device, domain=self.recipe.domain)
        sampler.load_state_dict(payload['sampler'])
        terrain = ContinuousTerrainGenerator(self.recipe.training_seed, self.recipe.terrain)
        terrain.load_state_dict(payload['terrain'])
        protocol = RequestProtocol(self.recipe.num_envs, self.recipe.device, self.recipe.protocol)
        aborted = protocol.restore_after_physical_reset(payload['protocol'], torch.zeros(
            self.recipe.num_envs, dtype=torch.float64, device=self.recipe.device))
        collection = payload['collection']
        counters = {'attempted_vector_steps', 'returned_vector_steps'}
        if (not isinstance(collection, dict) or set(collection) not in (counters, counters | {'task_source'})
                or collection.get('task_source') is not None and not isinstance(collection['task_source'], dict)):
            raise ValueError('structured collection counters are invalid')
        components = LearningComponents(model, trainer, budget, sampler, terrain, protocol,
                                         **{name: collection[name] for name in counters},
                                         deferred_task_source_state=_cpu_snapshot(collection.get('task_source')))
        components.validate_binding(self.recipe, self.journal)
        return RestoredLearning(self.recipe, components, _cpu_snapshot(payload['global_rng']), aborted,
                                _document(payload['boundary']))

    def _read_payload(self, name):
        """有界且只反序列化一次；调用者随后执行完整状态验收。"""
        path = self._path(name)
        receipt = read_json(Path(str(path) + '.json'))
        if (not isinstance(receipt, dict) or set(receipt) != {'format', 'bytes', 'sha256', 'recipe_sha256', 'boundary',
                                                            'cold_physical_state_restored', 'metadata'}
                or receipt['format'] != self._FORMAT or receipt['recipe_sha256'] != self.recipe.sha256
                or receipt['cold_physical_state_restored'] is not False):
            raise ValueError('structured snapshot completion receipt differs')
        self.journal.require_exact_boundary(receipt['boundary'])
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > self._LIMIT or info.st_size != receipt['bytes']:
                raise ValueError('structured snapshot size or regular-file identity differs')
            raw = stream.read(self._LIMIT + 1)
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('structured snapshot bytes differ from the completion marker')
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (not isinstance(payload, dict) or payload.get('boundary') != receipt['boundary']
                or payload.get('metadata') != receipt['metadata']):
            raise ValueError('structured snapshot and receipt describe different completed states')
        self._read_receipt = receipt
        return payload

    def load(self, name):
        return self._validate_payload(self._read_payload(name))

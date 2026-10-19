"""显式迁移已停止的学习端点；原会话、验收规则和已付预算保持不变。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

import torch

from ..learning.accounting import LearningJournal
from ..learning.checkpoint import LearningRecipe, StructuredCheckpointStore
from ..learning.loop import load_stopped_attempt
from ...common.artifacts import _publish_new_files
from ...common.training_control import (
    _lease, boot_id, command_lock, file_pin, read_json, verify_pin, write_json,
)


_FORMAT = 'structured_source_transition'
_LEARNING_FIELDS = ('model', 'actor_adam', 'critic_adam', 'trainer', 'collection',
                    'curriculum', 'sampler', 'terrain', 'protocol', 'global_rng', 'normalization')
_REVIEWABLE = {
    'formal/tasks/terrain.py', 'formal/runtime/reset_coordinator.py',
    'formal/simulation/terrain_stream.py', 'formal/runtime/transition.py',
    'formal/runtime/session.py', 'formal/runtime/process.py',
    'formal/learning/checkpoint.py', 'formal/learning/loop.py',
    'formal/learning/collection.py', 'formal/learning/optimization.py',
    'formal/simulation/task.py', 'formal/simulation/plant.py',
}


def source_table(package):
    return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(package).rglob('*.py'))}


def _review_sources(previous, candidate, review):
    before, after = source_table(previous), source_table(candidate)
    actual = {name: {'before': before.get(name), 'after': after.get(name)}
              for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    if (not actual or actual != review or not set(actual).issubset(_REVIEWABLE)
            or any(value['after'] is None for value in actual.values())):
        raise ValueError('source transition differs from its exact reviewed performance/support changes')
    return actual


def learning_state_sha256(payload):
    """按字段与原始张量值取摘要；不使用pickle字节作为等价证据。"""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            visit(('tensor', str(value.dtype), tuple(value.shape)))
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, 'big'))
            digest.update(raw)
        elif isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(value[key])
            digest.update(b'end')
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                visit(item)
            digest.update(b'end')
        else:
            raw = json.dumps([type(value).__name__, value], allow_nan=False).encode()
            digest.update(len(raw).to_bytes(8, 'big'))
            digest.update(raw)

    visit({name: payload[name] for name in _LEARNING_FIELDS})
    return digest.hexdigest()


class _AcceptedSourceStore(StructuredCheckpointStore):
    """复用原端点验收入口，保留其唯一一次反序列化的已接受payload。"""

    def load(self, name):
        payload = self._read_payload(name)
        restored = self._validate_payload(payload)
        self.accepted_payload = payload
        return restored


def _hold_resources(session, stack):
    descriptor = _lease(session.root)
    stack.callback(os.close, descriptor)
    for item in session.spec['resources']['locks']:
        fd = os.open(item['path'], os.O_RDWR | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino, info.st_uid) != (item['device'], item['inode'], item['uid'])):
            raise ValueError('source transition resource lease identity changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready, resources = session._resources()
    if not ready:
        raise RuntimeError('source transition waits for the original resource admission')
    return resources


def migrate_stopped_session(source_root, target_root, *, learner_source, reviewed_changes,
                            approval_pin, authorized_boot):
    """仅准备新会话，不启动SDK/训练或切换registry；调用者负责明确授权。"""
    from .session import StructuredTrainingSession

    if authorized_boot != boot_id():
        raise ValueError('source transition requires explicit authorization for the actual boot')
    verify_pin(approval_pin)
    approval = read_json(approval_pin['path'])
    if (approval.get('authorization') != 'approved_implementation_validation_then_switch'
            or approval.get('boot') != authorized_boot or not approval.get('human_answer')):
        raise ValueError('source transition lacks its explicit human approval')
    source = StructuredTrainingSession(source_root)
    candidate = Path(learner_source).resolve(strict=True)
    with command_lock(source.root), ExitStack() as stack:
        resources = _hold_resources(source, stack)
        for pin in source.spec['pins']:
            verify_pin(pin)
        changes = _review_sources(source.root / 'source/transformer_rl', candidate, reviewed_changes)
        old_recipe = LearningRecipe.from_dict(read_json(source.root / 'recipe.json'))
        if old_recipe.learner_source_sha256 != source.source_sha256(source.root / 'source/transformer_rl'):
            raise ValueError('previous frozen source differs from its original recipe')
        previous = source.attempt()
        if previous is None:
            raise ValueError('source transition cannot invent a previous learning attempt')
        exit_pin = file_pin(previous / 'exit.json')
        old_journal = stack.enter_context(LearningJournal(
            source.root / 'journal', writable=True, authorized_boot=authorized_boot))
        if old_recipe.device == 'cuda:0':
            torch.cuda.init()  # 准入和四把原lease均已持有，才初始化CUDA。
        old_store = _AcceptedSourceStore(previous / 'train/checkpoints', old_recipe, old_journal)
        _, spent = load_stopped_attempt(old_store, previous / 'control', parent_exit_pin=exit_pin,
                                        authorized_current_boot=authorized_boot)
        payload = old_store.accepted_payload
        state_hash = learning_state_sha256(payload)
        boundary = old_journal.checkpoint_boundary()
        if boundary['ppo_updates'] < 1:
            raise ValueError('source transition must inherit a nonzero completed learning boundary')
        new_recipe = replace(old_recipe, learner_source_sha256=source.source_sha256(candidate))
        target = StructuredTrainingSession.create(
            target_root, new_recipe, runtime_factory=source.spec['runtime_factory'],
            runtime_configuration=read_json(source.root / 'runtime.json'),
            resources=source.spec['resources'], python=source.spec['python'], learner_source=candidate,
            name=source.spec['name'], source_transition_required=True)
        with LearningJournal.create(target.root / 'journal', new_recipe.journal_spec(),
                                    authorized_boot=authorized_boot):
            pass
        # 事件字节与hash链逐字继承；只创建新identity，不给历史事件改source身份。
        events = {}
        for path in sorted((old_journal.directory / 'events').glob('[0-9]*.json')):
            raw = path.read_bytes()
            _publish_new_files({target.root / 'journal/events' / path.name: raw})
            events[path.name] = hashlib.sha256(raw).hexdigest()
        with LearningJournal(target.root / 'journal', writable=True, authorized_boot=authorized_boot) as journal:
            expected = dict(boundary, identity_sha256=journal.identity_sha256)
            journal.require_exact_boundary(expected)
            directory = target.root / 'transition/checkpoints'
            directory.mkdir(parents=True)
            checkpoint = {'path': str(previous / 'train/checkpoints/final.pt'),
                          'bytes': old_store._read_receipt['bytes'],
                          'sha256': old_store._read_receipt['sha256']}
            metadata = {
                'purpose': 'source_transition_bootstrap', 'authorized_boot': authorized_boot,
                'source_session': str(source.root), 'source_recipe_pin': file_pin(source.root / 'recipe.json'),
                'source_checkpoint_pin': checkpoint, 'source_exit_pin': exit_pin,
                'source_boundary': boundary, 'approval_pin': approval_pin,
                'reviewed_changes': changes, 'learning_state_sha256': state_hash,
                'physical_state_restored': False,
            }
            migrated = dict(payload, recipe=new_recipe.to_dict(), boundary=journal.checkpoint_boundary(),
                            metadata=metadata)
            if learning_state_sha256(migrated) != state_hash:
                raise ValueError('migration changed learning tensors, counters or RNG')
            store = StructuredCheckpointStore(directory, new_recipe, journal)
            receipt = store._publish_payload('bootstrap.pt', migrated)
            marker = {
                'format': _FORMAT, 'authorized_boot': authorized_boot, 'target_session': str(target.root),
                'target_recipe_sha256': new_recipe.sha256,
                'checkpoint_pin': {'path': str(directory / 'bootstrap.pt'),
                                   'bytes': receipt['bytes'], 'sha256': receipt['sha256']},
                'sidecar_pin': file_pin(directory / 'bootstrap.pt.json'),
                'boundary': journal.checkpoint_boundary(), 'update': boundary['ppo_updates'],
                'active_elapsed_s': spent, 'returned_transitions': boundary['returned_transitions'],
                'learning_state_sha256': state_hash, 'source': metadata,
                'inherited_events_sha256': events, 'resources_at_validation': resources,
                'cold_physical_state_restored': False,
            }
            assert receipt['metadata']['learning_state_sha256'] == state_hash
            write_json(target.root / 'transition.json', marker)
            write_json(target.root / 'transition.pin.json', file_pin(target.root / 'transition.json'))
        return target


def transition_metadata(root):
    root = Path(root).resolve(strict=True)
    pin = read_json(root / 'transition.pin.json')
    if not pin or pin.get('path') != str(root / 'transition.json'):
        raise ValueError('required source transition is absent or unsealed')
    verify_pin(pin)
    marker = read_json(root / 'transition.json')
    recipe = LearningRecipe.from_dict(read_json(root / 'recipe.json'))
    if (marker.get('format') != _FORMAT or marker.get('target_session') != str(root)
            or marker.get('authorized_boot') != boot_id()
            or marker.get('target_recipe_sha256') != recipe.sha256
            or marker.get('cold_physical_state_restored') is not False
            or marker.get('update', 0) < 1
            or marker.get('returned_transitions') != marker['update'] * recipe.num_envs * recipe.rollout_steps):
        raise ValueError('source transition boot, recipe or nonzero consumption differs')
    for name, relative in (('checkpoint_pin', 'bootstrap.pt'), ('sidecar_pin', 'bootstrap.pt.json')):
        if marker[name]['path'] != str(root / 'transition/checkpoints' / relative):
            raise ValueError('source transition checkpoint escapes its new session')
    return marker


def load_transition(root, recipe, journal, binding):
    """新producer唯一加载新source快照；普通旧版resume规则不变。"""
    root = Path(root).resolve(strict=True)
    marker = transition_metadata(root)
    if (binding['start_update'] != marker['update']
            or binding['prior_transitions'] != marker['returned_transitions']
            or binding['prior_active_elapsed_s'] != marker['active_elapsed_s']):
        raise ValueError('source transition launch rebased its inherited budget or clock')
    journal.require_exact_boundary(marker['boundary'])
    verify_pin(marker['sidecar_pin'])
    sidecar = read_json(marker['sidecar_pin']['path'])
    if (sidecar['bytes'] != marker['checkpoint_pin']['bytes']
            or sidecar['sha256'] != marker['checkpoint_pin']['sha256']):
        raise ValueError('source transition payload pin differs')
    store = _AcceptedSourceStore(root / 'transition/checkpoints', recipe, journal)
    restored = store.load('bootstrap.pt')
    if (store.accepted_payload['metadata'] != marker['source']
            or learning_state_sha256(store.accepted_payload) != marker['learning_state_sha256']):
        raise ValueError('source transition learning-state equivalence proof differs')
    return restored

"""Explicit private fork from a cached checkpoint behind a failed journal.

The old failure is never rewritten. Retained learning state and retired
executed work have separate counters; all reservations remain charged.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import torch

from ..learning.accounting import LearningJournal
from ..learning.checkpoint import LearningRecipe, StructuredCheckpointStore
from ...common.training_control import boot_id, file_pin, read_json, verify_pin, write_json
from .transition import _AcceptedSourceStore, _hold_resources, learning_state_sha256, source_table


_FORMAT = 'private_failed_learning_fork'
_REVIEWABLE = {
    'common/training_cli.py',
    'formal/lab_env/collector.py', 'formal/lab_env/recovery.py', 'formal/lab_env/runtime.py',
    'formal/lab_env/terrain.py', 'formal/lab_env/train.py', 'formal/tasks/terrain_query.py',
    'formal/learning/accounting.py', 'formal/learning/loop.py',
    'formal/runtime/session.py', 'formal/runtime/process.py', 'formal/runtime/recovery.py',
}


def _review_sources(previous, candidate, review):
    before, after = source_table(previous), source_table(candidate)
    actual = {name: {'before': before.get(name), 'after': after.get(name)}
              for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    if (not actual or actual != review or not set(actual).issubset(_REVIEWABLE)
            or any(item['after'] is None for item in actual.values())):
        raise ValueError('failed recovery source differs from the exact reviewed support/performance changes')
    return actual


def prepare_failed_recovery(target_root, *, cache_review_pin, previous_recipe, previous_source,
                            learner_source, reviewed_changes, authorization_pin, authorized_boot,
                            runtime_factory, runtime_configuration, resources, python, name=None):
    """Validate an immutable evidence cache and publish a new, unstarted fork.

    No original producer path or terminal process is read. This is deliberately
    separate from normal resume and from migration of a clean stopped session.
    """
    from .session import StructuredTrainingSession

    if authorized_boot != boot_id():
        raise ValueError('failed recovery requires explicit authorization for the actual boot')
    verify_pin(authorization_pin)
    approval = read_json(authorization_pin['path'])
    if (approval.get('authorization') != 'failed_checkpoint_fork_preserving_all_consumption'
            or approval.get('boot') != authorized_boot or not approval.get('human_objective')):
        raise ValueError('failed recovery lacks explicit human authorization')
    verify_pin(cache_review_pin)
    review = read_json(cache_review_pin['path'])
    cache = Path(cache_review_pin['path']).parent.resolve(strict=True)
    recipe = previous_recipe
    if (not isinstance(recipe, LearningRecipe) or review['boot'] != authorized_boot
            or recipe.sha256 != review['checkpoint_recipe_sha256']
            or approval.get('checkpoint_sha256') != review['records'][next(
                key for key, value in review['records'].items()
                if value['cached_path'] == str(cache / 'checkpoint.pt'))]['sha256']
            or approval.get('failed_head_sha256') != review['final_journal']['head']['sha256']
            or StructuredTrainingSession.source_sha256(previous_source) != recipe.learner_source_sha256):
        raise ValueError('failed recovery cache, recipe, source or approved boundary differs')
    for record in review['records'].values():
        path = Path(record['cached_path'])
        if not path.is_relative_to(cache) or path.is_symlink():
            raise ValueError('recovery input escapes the immutable evidence cache')
        verify_pin({'path': str(path), 'bytes': record['bytes'], 'sha256': record['sha256']})
    changes = _review_sources(previous_source, learner_source, reviewed_changes)
    old = LearningJournal(cache / 'journal')
    if old.spec != recipe.journal_spec() or old.summary() != review['final_journal']:
        raise ValueError('failed recovery journal differs from its pinned final chain')
    checkpoint = review['checkpoint_boundary']
    prefix = Path(tempfile.mkdtemp(prefix='verified-checkpoint-prefix-', dir=cache))
    (prefix / 'events').mkdir()
    for relative in ('identity.json', 'identity.pin.json'):
        shutil.copyfile(cache / 'journal' / relative, prefix / relative)
    for index in range(checkpoint['head']['sequence'] + 1):
        relative = f'{index:08d}.json'
        shutil.copyfile(cache / 'journal/events' / relative, prefix / 'events' / relative)
    with LearningJournal(prefix) as prefix_journal, ExitStack() as stack:
        prefix_journal.require_exact_boundary(checkpoint)
        # Admission precedes CUDA and loading; target creation does not launch.
        admission = SimpleNamespace(root=cache, spec={'resources': resources})
        admission._resources = lambda: StructuredTrainingSession._resources(admission)
        admission._resource_definition = StructuredTrainingSession._resource_definition
        observed_resources = _hold_resources(admission, stack)
        if recipe.device == 'cuda:0':
            torch.cuda.init()
        store = _AcceptedSourceStore(cache, recipe, prefix_journal)
        store.load('checkpoint.pt')  # All original strict checkpoint predicates.
        payload = store.accepted_payload
        before_hash = learning_state_sha256(payload)
        budget = recipe.make_curriculum()
        for _ in range(old.summary()['rollouts_reserved']):
            budget.charge_rollout()
        if budget.consumed != old.summary()['reserved_by_stage']:
            raise ValueError('failed reservations do not match the original continuous curriculum')
        candidate_recipe = replace(recipe, learner_source_sha256=StructuredTrainingSession.source_sha256(learner_source))
        target = StructuredTrainingSession.create(
            target_root, candidate_recipe, runtime_factory=runtime_factory,
            runtime_configuration=runtime_configuration, resources=resources, python=python,
            learner_source=learner_source, name=name, failed_recovery_required=True)
        provenance = {'cache_review_pin': cache_review_pin, 'authorization_pin': authorization_pin,
                      'reviewed_changes': changes, 'checkpoint_learning_state_sha256': before_hash,
                      'physical_state_restored': False, 'lost_learning_updates':
                      old.summary()['ppo_updates'] - checkpoint['ppo_updates']}
        with LearningJournal.create(target.root / 'journal', candidate_recipe.journal_spec(),
                                    authorized_boot=authorized_boot) as journal:
            journal.import_failed_consumption(checkpoint, old.summary(), provenance=provenance)
            metadata = dict(provenance, purpose='explicit_failed_checkpoint_fork')
            inherited = deepcopy(payload)
            inherited.update(recipe=candidate_recipe.to_dict(), boundary=journal.checkpoint_boundary(),
                             curriculum=budget.state_dict(), metadata=metadata)
            # Only budget exposure changes; every other learning field is exact.
            original_budget = inherited['curriculum']
            inherited['curriculum'] = payload['curriculum']
            if learning_state_sha256(inherited) != before_hash:
                raise ValueError('failed recovery changed model, Adam, RNG, request or source state')
            inherited['curriculum'] = original_budget
            fork_hash = learning_state_sha256(inherited)
            directory = target.root / 'recovery/checkpoints'
            directory.mkdir(parents=True)
            receipt = StructuredCheckpointStore(directory, candidate_recipe, journal)._publish_payload(
                'bootstrap.pt', inherited)
            marker = {'format': _FORMAT, 'authorized_boot': authorized_boot, 'target_session': str(target.root),
                      'target_recipe_sha256': candidate_recipe.sha256, 'update': checkpoint['ppo_updates'],
                      'budget_update': journal.summary()['rollouts_reserved'],
                      'returned_transitions': checkpoint['returned_transitions'],
                      'active_elapsed_s': old.summary()['active_elapsed_s'],
                      'boundary': journal.checkpoint_boundary(), 'source': metadata,
                      'learning_state_sha256': fork_hash, 'cold_physical_state_restored': False,
                      'resources_at_validation': observed_resources,
                      'checkpoint_pin': {'path': str(directory / 'bootstrap.pt'),
                                         'bytes': receipt['bytes'], 'sha256': receipt['sha256']},
                      'sidecar_pin': file_pin(directory / 'bootstrap.pt.json')}
            write_json(target.root / 'recovery.json', marker)
            write_json(target.root / 'recovery.pin.json', file_pin(target.root / 'recovery.json'))
        return target


def recovery_metadata(root):
    root = Path(root).resolve(strict=True)
    pin = read_json(root / 'recovery.pin.json')
    if not pin or pin.get('path') != str(root / 'recovery.json'):
        raise ValueError('required failed recovery is absent or unsealed')
    verify_pin(pin)
    marker = read_json(root / 'recovery.json')
    recipe = LearningRecipe.from_dict(read_json(root / 'recipe.json'))
    if (marker.get('format') != _FORMAT or marker.get('target_session') != str(root)
            or marker.get('authorized_boot') != boot_id() or marker.get('update', 0) <= 0
            or marker.get('target_recipe_sha256') != recipe.sha256
            or marker.get('cold_physical_state_restored') is not False
            or marker['update'] != marker['boundary']['ppo_updates']
            or marker['budget_update'] != marker['boundary']['rollouts_reserved']
            or marker['returned_transitions'] != marker['boundary']['returned_transitions']):
        raise ValueError('failed recovery boot, recipe or split model/budget counters differ')
    for key, relative in (('checkpoint_pin', 'bootstrap.pt'), ('sidecar_pin', 'bootstrap.pt.json')):
        if marker[key]['path'] != str(root / 'recovery/checkpoints' / relative):
            raise ValueError('failed recovery checkpoint escapes its own session')
    return marker


def load_recovery(root, recipe, journal, binding):
    marker = recovery_metadata(root)
    if any(binding[field] != marker[key] for field, key in (
            ('start_update', 'update'), ('prior_transitions', 'returned_transitions'),
            ('prior_active_elapsed_s', 'active_elapsed_s'))):
        raise ValueError('failed recovery launch rebased model state or consumed budget')
    journal.require_exact_boundary(marker['boundary'])
    verify_pin(marker['sidecar_pin'])
    verify_pin(marker['checkpoint_pin'])
    store = _AcceptedSourceStore(Path(root) / 'recovery/checkpoints', recipe, journal)
    restored = store.load('bootstrap.pt')
    if (store.accepted_payload['metadata'] != marker['source']
            or learning_state_sha256(store.accepted_payload) != marker['learning_state_sha256']):
        raise ValueError('failed recovery full learning-state proof differs')
    return restored

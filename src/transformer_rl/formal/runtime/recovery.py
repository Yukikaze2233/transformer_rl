"""Explicit private fork from a cached checkpoint behind a failed journal.

The old failure is never rewritten. Retained learning state and retired
executed work have separate counters; all reservations remain charged.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace

import torch

from ..learning.accounting import LearningJournal
from ..learning.checkpoint import LearningRecipe, StructuredCheckpointStore
from ..lab_env.sdk import (
    _SDK_CONTINUATION_FIELDS, _SDK_CONTINUATION_FORMAT, _has_inherited_sdk_gate,
    _sdk_evidence_identity,
)
from ...common.training_control import boot_id, file_pin, read_json, verify_pin, write_json
from .transition import _AcceptedSourceStore, _hold_resources, learning_state_sha256, source_table


_FORMAT = 'private_failed_learning_fork'
_SDK_TRANSITION_FORMAT = 'private_sdk_mechanical_transition_v1'
_MECHANICAL_PAIR_FORMAT = 'private_native_mechanical_pair_v1'
_SDK_BINDING_FIELDS = {
    'format', 'authorized_boot', 'source_checkpoint_sha256', 'source_recipe_sha256',
    'source_learner_source_sha256', 'target_learner_source_sha256',
    'source_sdk_pin', 'target_sdk_pin', 'physical_compatibility_pin',
}
_REVIEWABLE = {
    'common/training_cli.py',
    'formal/lab_env/collector.py', 'formal/lab_env/recovery.py', 'formal/lab_env/runtime.py',
    'formal/lab_env/terrain.py', 'formal/lab_env/train.py', 'formal/tasks/terrain_query.py',
    'formal/learning/accounting.py', 'formal/learning/loop.py',
    'formal/runtime/session.py', 'formal/runtime/process.py', 'formal/runtime/recovery.py',
}
_GA_REVIEWABLE = {
    'formal/lab_env/cfg.py', 'formal/lab_env/settings.py',
    'formal/lab_env/env.py', 'formal/lab_env/dynamics.py',
    'formal/lab_env/sdk.py', 'formal/lab_env/ga_isolation.py',
    'formal/lab_env/contact_measurement.py',
}
_TELEMETRY_REVIEWABLE = {'formal/runtime/telemetry.py', 'formal/learning/optimization.py'}


def _review_sources(previous, candidate, review, *, sdk_transition=False, same_sdk=False,
                    telemetry_review=None):
    before, after = source_table(previous), source_table(candidate)
    actual = {name: {'before': before.get(name), 'after': after.get(name)}
              for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    allowed = _REVIEWABLE | _GA_REVIEWABLE if sdk_transition or same_sdk else _REVIEWABLE
    telemetry = {name: item for name, item in actual.items() if name in _TELEMETRY_REVIEWABLE}
    if telemetry:
        if not isinstance(telemetry_review, dict) or telemetry_review != telemetry:
            raise ValueError('telemetry source differs from its exact reviewed and separately authorized changes')
        allowed = allowed | set(telemetry)
    elif telemetry_review not in (None, {}):
        raise ValueError('telemetry source authorization differs from the actual changes')
    if (not actual or actual != review or not set(actual).issubset(allowed)
            or any(item['after'] is None for item in actual.values())):
        raise ValueError('failed recovery source differs from the exact reviewed support/performance changes')
    return actual


def _pinned_bytes(pin, limit):
    """Read one bounded evidence file once, including its SHA verification."""
    if (not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}
            or not isinstance(pin['path'], str) or not Path(pin['path']).is_absolute()
            or type(pin['bytes']) is not int or not 0 < pin['bytes'] <= limit
            or not isinstance(pin['sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) is None):
        raise ValueError('invalid SDK transition evidence pin')
    path = Path(pin['path'])
    if path.resolve(strict=True) != path:
        raise ValueError('SDK transition evidence must not follow symlinks')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != pin['bytes']:
            raise ValueError('SDK transition evidence file identity differs')
        raw = stream.read(limit + 1)
    if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('SDK transition evidence bytes differ from their seal')
    return raw


def _pinned_document(pin):
    value = json.loads(_pinned_bytes(pin, 256 * 1024))
    if not isinstance(value, dict) or not value:
        raise ValueError('SDK transition evidence requires a nonempty document')
    json.dumps(value, allow_nan=False)
    return value


def _mechanical_pair(pin, *, num_envs, source_sha256):
    """Compare complete native tables, never a producer's claimed pass flag.

    The tensor API reports COM translation and inertia in the link/prim frame.
    Principal-axis quaternions may differ; each must still be a finite unit q.
    """
    value = _pinned_document(pin)
    expected = {'format', 'num_envs', 'body_order', 'byte_layout', 'inertia_frame',
                'source_raw_pin', 'target_raw_pin'}
    names = value.get('body_order')
    if (set(value) != expected or value['format'] != _MECHANICAL_PAIR_FORMAT
            or type(value['num_envs']) is not int or value['num_envs'] != num_envs
            or type(num_envs) is not int or not 1 <= num_envs <= 8192
            or not isinstance(names, list) or len(names) != 19
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 19 or names[0] != 'base_link'
            or value['byte_layout'] != 'float32_le_mass_com_xyzw_inertia'
            or value['inertia_frame'] != 'center_of_mass_in_rigid_body_prim'
            or sys.byteorder != 'little'):
        raise ValueError('SDK transition mechanical table contract differs')
    count = num_envs * 19
    size = count * 17 * 4
    tables = []
    for role in ('source', 'target'):
        raw_pin = value[role + '_raw_pin']
        if not isinstance(raw_pin, dict) or raw_pin.get('bytes') != size:
            raise ValueError('SDK transition requires all native mechanical rows')
        raw = _pinned_bytes(raw_pin, 32 * 1024 * 1024)
        flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        mass = flat[:count].reshape(num_envs, 19)
        com = flat[count:count * 8].reshape(num_envs, 19, 7)
        inertia = flat[count * 8:].reshape(num_envs, 19, 3, 3)
        if (not torch.isfinite(flat).all() or (mass <= 0).any()
                or not torch.allclose(inertia, inertia.transpose(-1, -2), rtol=1e-5, atol=1e-7)
                or (torch.linalg.eigvalsh(inertia) <= 0).any()
                or not torch.allclose(com[..., 3:].norm(dim=-1), torch.ones(num_envs, 19),
                                      rtol=0, atol=1e-4)):
            raise ValueError('SDK transition contains invalid native mechanics')
        tables.append((mass, com[..., :3], inertia))
    if value['source_raw_pin']['sha256'] != source_sha256:
        raise ValueError('SDK transition has not reconstructed the checkpoint mechanical identity')
    comparisons = {}
    for name, old, new, rtol, atol in zip(
            ('mass', 'com_xyz', 'inertia_link'), *tables,
            (1e-5, 0., 1e-4), (1e-7, 1e-6, 1e-7), strict=True):
        if not torch.allclose(old, new, rtol=rtol, atol=atol):
            raise ValueError('SDK transition changes physical ' + name)
        comparisons[name] = {'max_abs_difference': float((old - new).abs().max()),
                             'rtol': rtol, 'atol': atol}
    return {'source_startup_parameters_sha256': value['source_raw_pin']['sha256'],
            'target_startup_parameters_sha256': value['target_raw_pin']['sha256'],
            'physical_compatibility_pin': deepcopy(pin), 'num_envs': num_envs,
            'body_order': names, 'comparison': comparisons,
            'quaternion_comparison': 'unit norm; principal-axis representation is not byte identity'}


def _sdk_transition(pin, *, authorized_boot, checkpoint_sha256, recipe, target_source_sha256,
                    source_startup_sha256):
    value = _pinned_document(pin)
    if (set(value) != _SDK_BINDING_FIELDS or value['format'] != _SDK_TRANSITION_FORMAT
            or value['authorized_boot'] != authorized_boot
            or value['source_checkpoint_sha256'] != checkpoint_sha256
            or value['source_recipe_sha256'] != recipe.sha256
            or value['source_learner_source_sha256'] != recipe.learner_source_sha256
            or value['target_learner_source_sha256'] != target_source_sha256):
        raise ValueError('SDK transition boot, checkpoint, recipe or source binding differs')
    # These are the actual pinned inventories, not a fabricated common SDK SHA.
    # Runtime must independently match the target inventory before sampling.
    _pinned_document(value['source_sdk_pin'])
    _pinned_document(value['target_sdk_pin'])
    mechanics = _mechanical_pair(value['physical_compatibility_pin'],
                                 num_envs=recipe.num_envs, source_sha256=source_startup_sha256)
    return dict(value, evidence_pin=deepcopy(pin), **mechanics)


def _apply_sdk_identity(inherited, original, evidence):
    source = original['collection'].get('task_source')
    target = inherited['collection'].get('task_source')
    if (not isinstance(source, dict) or not isinstance(target, dict)
            or source.get('format') != 'managed_lab_task_source_v4'
            or source.get('startup_parameters_sha256') != evidence['source_startup_parameters_sha256']):
        raise ValueError('SDK transition requires the original complete managed task source')
    target['startup_parameters_sha256'] = evidence['target_startup_parameters_sha256']


def _require_preserved_learning(inherited, original, original_sha256, sdk_evidence=None):
    projected = deepcopy(inherited)
    projected['curriculum'] = original['curriculum']
    if sdk_evidence is not None:
        source = original['collection'].get('task_source')
        target = projected['collection'].get('task_source')
        if (not isinstance(source, dict) or not isinstance(target, dict)
                or source.get('startup_parameters_sha256') != sdk_evidence['source_startup_parameters_sha256']
                or target.get('startup_parameters_sha256') != sdk_evidence['target_startup_parameters_sha256']):
            raise ValueError('SDK transition mechanical identity patch differs')
        target['startup_parameters_sha256'] = source['startup_parameters_sha256']
    if learning_state_sha256(projected) != original_sha256:
        raise ValueError('failed recovery changed model, Adam, RNG, request or source state')


def _inherited_sdk_anchor(payload, recipe, *, authorized_boot):
    """Find the current gate in the already validated CP journal, not CP metadata."""
    boundary = payload['boundary']
    recovery = boundary.get('recovery')
    source = recovery.get('provenance') if isinstance(recovery, dict) else None
    fields = ('sdk_transition_pin', 'sdk_transition', 'sdk_continuation_pin', 'sdk_continuation')
    if not isinstance(source, dict) or not any(key in source for key in fields):
        if _has_inherited_sdk_gate(boundary):
            raise ValueError('checkpoint recovery discarded its inherited SDK gate')
        return None
    continuation = source.get('sdk_continuation')
    summary = source.get('sdk_transition') if continuation is None else continuation
    if not isinstance(summary, dict):
        raise ValueError('checkpoint SDK provenance lacks its verified summary')
    marker = {'source': source, 'boundary': boundary, 'authorized_boot': authorized_boot,
              'learning_state_sha256': summary.get('inherited_learning_state_sha256')}
    _require_sdk_metadata(marker, recipe)
    identity = _sdk_evidence_identity(summary['evidence_pin'], authorized_boot=authorized_boot)
    task_source = payload['collection'].get('task_source')
    startup = (None if not isinstance(task_source, dict)
               else task_source.get('startup_parameters_sha256'))
    summary_startup = summary.get('startup_parameters_sha256' if continuation is not None
                                  else 'target_startup_parameters_sha256')
    if (identity['target_learner_source_sha256'] != recipe.learner_source_sha256
            or identity['startup_parameters_sha256'] != startup or summary_startup != startup
            or identity['num_envs'] != recipe.num_envs
            or identity['target_sdk_pin'] != summary.get('target_sdk_pin')):
        raise ValueError('checkpoint SDK provenance differs from its source or native mechanics')
    return identity


def _publish_sdk_continuation(root, anchor, *, authorized_boot, checkpoint_sha256,
                              recipe, target_source_sha256, original_sha256, inherited_sha256):
    document = {
        'format': _SDK_CONTINUATION_FORMAT, 'authorized_boot': authorized_boot,
        'source_checkpoint_sha256': checkpoint_sha256, 'source_recipe_sha256': recipe.sha256,
        'source_learner_source_sha256': recipe.learner_source_sha256,
        'target_learner_source_sha256': target_source_sha256,
        'prior_sdk_evidence_pin': deepcopy(anchor['evidence_pin']),
        'target_sdk_pin': deepcopy(anchor['target_sdk_pin']),
        'startup_parameters_sha256': anchor['startup_parameters_sha256'],
    }
    directory = Path(root) / 'recovery'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'sdk_continuation.json'
    write_json(path, document)
    pin = file_pin(path)
    _sdk_evidence_identity(pin, authorized_boot=authorized_boot)
    return dict(document, evidence_pin=pin, checkpoint_learning_state_sha256=original_sha256,
                inherited_learning_state_sha256=inherited_sha256)


def prepare_failed_recovery(target_root, *, cache_review_pin, previous_recipe, previous_source,
                            learner_source, reviewed_changes, authorization_pin, authorized_boot,
                            runtime_factory, runtime_configuration, resources, python, name=None,
                            sdk_transition_pin=None):
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
    if approval.get('sdk_transition_pin') != sdk_transition_pin:
        raise ValueError('failed recovery SDK evidence differs from the explicit approval')
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
        anchor = _inherited_sdk_anchor(payload, recipe, authorized_boot=authorized_boot)
        changes = _review_sources(previous_source, learner_source, reviewed_changes,
                                  sdk_transition=sdk_transition_pin is not None,
                                  same_sdk=anchor is not None and sdk_transition_pin is None,
                                  telemetry_review=approval.get('telemetry_reviewed_changes'))
        before_hash = learning_state_sha256(payload)
        budget = recipe.make_curriculum()
        for _ in range(old.summary()['rollouts_reserved']):
            budget.charge_rollout()
        if budget.consumed != old.summary()['reserved_by_stage']:
            raise ValueError('failed reservations do not match the original continuous curriculum')
        candidate_recipe = replace(recipe, learner_source_sha256=StructuredTrainingSession.source_sha256(learner_source))
        sdk_evidence = None
        inherited = deepcopy(payload)
        inherited['curriculum'] = budget.state_dict()
        if sdk_transition_pin is not None:
            task_source = payload['collection'].get('task_source')
            if not isinstance(task_source, dict):
                raise ValueError('SDK transition requires the cached managed mechanical identity')
            sdk_evidence = _sdk_transition(sdk_transition_pin, authorized_boot=authorized_boot,
                checkpoint_sha256=approval['checkpoint_sha256'], recipe=recipe,
                target_source_sha256=candidate_recipe.learner_source_sha256,
                source_startup_sha256=task_source.get('startup_parameters_sha256'))
            _apply_sdk_identity(inherited, payload, sdk_evidence)
        _require_preserved_learning(inherited, payload, before_hash, sdk_evidence)
        fork_hash = learning_state_sha256(inherited)
        target = StructuredTrainingSession.create(
            target_root, candidate_recipe, runtime_factory=runtime_factory,
            runtime_configuration=runtime_configuration, resources=resources, python=python,
            learner_source=learner_source, name=name, failed_recovery_required=True)
        provenance = {'cache_review_pin': cache_review_pin, 'authorization_pin': authorization_pin,
                      'reviewed_changes': changes, 'checkpoint_learning_state_sha256': before_hash,
                      'physical_state_restored': False, 'lost_learning_updates':
                      old.summary()['ppo_updates'] - checkpoint['ppo_updates']}
        if sdk_evidence is not None:
            provenance['sdk_transition_pin'] = deepcopy(sdk_transition_pin)
            provenance['sdk_transition'] = dict(sdk_evidence,
                checkpoint_learning_state_sha256=before_hash,
                inherited_learning_state_sha256=fork_hash)
        elif anchor is not None:
            continuation = _publish_sdk_continuation(target.root, anchor,
                authorized_boot=authorized_boot, checkpoint_sha256=approval['checkpoint_sha256'],
                recipe=recipe, target_source_sha256=candidate_recipe.learner_source_sha256,
                original_sha256=before_hash, inherited_sha256=fork_hash)
            provenance['sdk_continuation_pin'] = deepcopy(continuation['evidence_pin'])
            provenance['sdk_continuation'] = continuation
        if approval.get('telemetry_reviewed_changes'):
            provenance['telemetry_reviewed_changes'] = deepcopy(approval['telemetry_reviewed_changes'])
        with LearningJournal.create(target.root / 'journal', candidate_recipe.journal_spec(),
                                    authorized_boot=authorized_boot) as journal:
            journal.import_failed_consumption(checkpoint, old.summary(), provenance=provenance)
            metadata = dict(provenance, purpose='explicit_failed_checkpoint_fork')
            inherited.update(recipe=candidate_recipe.to_dict(), boundary=journal.checkpoint_boundary(),
                             curriculum=budget.state_dict(), metadata=metadata)
            if learning_state_sha256(inherited) != fork_hash:
                raise ValueError('failed recovery publication changed its verified learning state')
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


def _require_sdk_metadata(marker, recipe, *, verify_mechanics=False):
    continuation_pin = marker['source'].get('sdk_continuation_pin')
    continuation = marker['source'].get('sdk_continuation')
    if continuation_pin is not None or continuation is not None:
        expected = _SDK_CONTINUATION_FIELDS | {
            'evidence_pin', 'checkpoint_learning_state_sha256', 'inherited_learning_state_sha256'}
        if (marker['source'].get('sdk_transition_pin') is not None
                or marker['source'].get('sdk_transition') is not None
                or not isinstance(continuation, dict) or set(continuation) != expected
                or continuation.get('evidence_pin') != continuation_pin):
            raise ValueError('failed recovery SDK continuation metadata differs')
        identity = _sdk_evidence_identity(continuation_pin, authorized_boot=marker['authorized_boot'])
        sealed = identity['document']
        approval = _pinned_document(marker['source']['authorization_pin'])
        source_recipe = replace(recipe, learner_source_sha256=sealed['source_learner_source_sha256'])
        if (sealed['format'] != _SDK_CONTINUATION_FORMAT
                or sealed != {key: continuation[key] for key in _SDK_CONTINUATION_FIELDS}
                or sealed['source_recipe_sha256'] != source_recipe.sha256
                or sealed['target_learner_source_sha256'] != recipe.learner_source_sha256
                or approval.get('authorization') != 'failed_checkpoint_fork_preserving_all_consumption'
                or approval.get('boot') != marker['authorized_boot']
                or approval.get('sdk_transition_pin') is not None
                or approval.get('checkpoint_sha256') != sealed['source_checkpoint_sha256']
                or continuation['checkpoint_learning_state_sha256']
                != marker['source']['checkpoint_learning_state_sha256']
                or continuation['inherited_learning_state_sha256'] != marker['learning_state_sha256']):
            raise ValueError('failed recovery SDK continuation is not bound to its source and target')
        return
    evidence = marker['source'].get('sdk_transition')
    if evidence is None:
        if marker['source'].get('sdk_transition_pin') is not None:
            raise ValueError('failed recovery SDK transition has no verified summary')
        if _has_inherited_sdk_gate(marker.get('boundary')):
            raise ValueError('failed recovery discarded its inherited SDK identity gate')
        return
    expected = _SDK_BINDING_FIELDS | {
        'evidence_pin', 'source_startup_parameters_sha256', 'target_startup_parameters_sha256',
        'num_envs', 'body_order', 'comparison', 'quaternion_comparison',
        'checkpoint_learning_state_sha256', 'inherited_learning_state_sha256',
    }
    if (not isinstance(evidence, dict) or set(evidence) != expected
            or marker['source'].get('sdk_transition_pin') != evidence['evidence_pin']):
        raise ValueError('failed recovery SDK transition metadata differs')
    sealed = _pinned_document(evidence['evidence_pin'])
    source_recipe = replace(recipe, learner_source_sha256=evidence['source_learner_source_sha256'])
    approval = _pinned_document(marker['source']['authorization_pin'])
    if (sealed != {key: evidence[key] for key in _SDK_BINDING_FIELDS}
            or evidence['format'] != _SDK_TRANSITION_FORMAT
            or evidence['authorized_boot'] != marker['authorized_boot']
            or evidence['source_recipe_sha256'] != source_recipe.sha256
            or evidence['target_learner_source_sha256'] != recipe.learner_source_sha256
            or approval.get('sdk_transition_pin') != evidence['evidence_pin']
            or approval.get('checkpoint_sha256') != evidence['source_checkpoint_sha256']
            or evidence['checkpoint_learning_state_sha256']
            != marker['source']['checkpoint_learning_state_sha256']
            or evidence['inherited_learning_state_sha256'] != marker['learning_state_sha256']):
        raise ValueError('failed recovery SDK evidence is not bound to its source and target')
    if verify_mechanics:
        verified = _sdk_transition(evidence['evidence_pin'], authorized_boot=marker['authorized_boot'],
            checkpoint_sha256=evidence['source_checkpoint_sha256'], recipe=source_recipe,
            target_source_sha256=recipe.learner_source_sha256,
            source_startup_sha256=evidence['source_startup_parameters_sha256'])
        if any(evidence[key] != value for key, value in verified.items()):
            raise ValueError('failed recovery SDK mechanical proof differs from its seal')


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
    _require_sdk_metadata(marker, recipe)
    return marker


def load_recovery(root, recipe, journal, binding):
    marker = recovery_metadata(root)
    _require_sdk_metadata(marker, recipe, verify_mechanics=True)
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
    evidence = marker['source'].get('sdk_transition')
    continuation = marker['source'].get('sdk_continuation')
    if continuation is not None:
        task_source = store.accepted_payload['collection'].get('task_source')
        if (not isinstance(task_source, dict) or task_source.get('startup_parameters_sha256')
                != continuation['startup_parameters_sha256']):
            raise ValueError('failed recovery continuation changed native mechanical identity')
    if evidence is not None:
        task_source = store.accepted_payload['collection'].get('task_source')
        if (not isinstance(task_source, dict)
                or task_source.get('startup_parameters_sha256')
                != evidence['target_startup_parameters_sha256']):
            raise ValueError('failed recovery bootstrap mechanical identity differs')
    return restored

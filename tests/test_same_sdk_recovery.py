"""Same-GA source repairs retain the original identity gate without raw rereads."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from test_failed_sdk_recovery import accepted_evidence, learning_payload, sdk_evidence
from test_lab_sdk_identity import identity_fixture
from transformer_rl.common.training_control import file_pin, write_json
from transformer_rl.formal.lab_env import sdk
from transformer_rl.formal.runtime import recovery
from transformer_rl.formal.runtime.transition import learning_state_sha256, source_table


def _seal(path, document):
    write_json(path, document, replace=path.exists())
    return file_pin(path)


def _anchor_fixture(tmp_path, monkeypatch):
    directory = tmp_path / 'identity'
    directory.mkdir()
    installed = identity_fixture(directory, monkeypatch)
    target_sdk_pin = _seal(installed.path, installed.document)
    recipe, mechanical, document, _ = sdk_evidence(tmp_path)
    document['target_sdk_pin'] = target_sdk_pin
    pin = _seal(tmp_path / 'original-migration.json', document)
    evidence = accepted_evidence(recipe, mechanical, pin)
    current_recipe = replace(recipe, learner_source_sha256='b' * 64)
    before_hash, after_hash = '1' * 64, '2' * 64
    approval = _seal(tmp_path / 'original-approval.json', {
        'sdk_transition_pin': pin, 'checkpoint_sha256': document['source_checkpoint_sha256']})
    source = {'authorization_pin': approval, 'checkpoint_learning_state_sha256': before_hash,
              'sdk_transition_pin': pin, 'sdk_transition': dict(evidence,
                  checkpoint_learning_state_sha256=before_hash,
                  inherited_learning_state_sha256=after_hash)}
    payload = learning_payload(evidence['target_startup_parameters_sha256'])
    payload['metadata'] = {'purpose': 'same_process_pause'}
    payload['boundary'] = {'recovery': {'checkpoint': {}, 'failed': {}, 'provenance': source}}
    # The current CP has already been strictly loaded by its real store. This
    # helper checks only its sealed ancestry; retired native tables are absent.
    for name in ('source.f32', 'target.f32', 'old-sdk.json'):
        (tmp_path / name).unlink()
    return current_recipe, payload, installed


def _continuation_fixture(tmp_path, monkeypatch):
    recipe, original, installed = _anchor_fixture(tmp_path, monkeypatch)
    anchor = recovery._inherited_sdk_anchor(original, recipe, authorized_boot='owned-boot')
    root = tmp_path / 'next-session'
    root.mkdir()
    before = learning_state_sha256(original)
    inherited = deepcopy(original)
    inherited['curriculum']['consumed'][0] = 330
    recovery._require_preserved_learning(inherited, original, before)
    after = learning_state_sha256(inherited)
    target_recipe = replace(recipe, learner_source_sha256='e' * 64)
    summary = recovery._publish_sdk_continuation(root, anchor,
        authorized_boot='owned-boot', checkpoint_sha256='d' * 64, recipe=recipe,
        target_source_sha256=target_recipe.learner_source_sha256,
        original_sha256=before, inherited_sha256=after)
    approval = _seal(tmp_path / 'next-approval.json', {
        'authorization': 'failed_checkpoint_fork_preserving_all_consumption',
        'boot': 'owned-boot', 'checkpoint_sha256': 'd' * 64,
        'human_objective': 'same SDK source repair without changing learning state'})
    source = {'authorization_pin': approval, 'checkpoint_learning_state_sha256': before,
              'sdk_continuation_pin': summary['evidence_pin'], 'sdk_continuation': summary}
    marker = {'format': 'private_failed_learning_fork', 'target_session': str(root),
              'authorized_boot': 'owned-boot', 'target_recipe_sha256': target_recipe.sha256,
              'source': source, 'learning_state_sha256': after,
              'boundary': {'recovery': {'checkpoint': original['boundary'], 'failed': {},
                                        'provenance': deepcopy(source)}}}
    return root, target_recipe, marker, original, inherited, installed


def _seal_marker(root, marker):
    marker_pin = _seal(root / 'recovery.json', marker)
    _seal(root / 'recovery.pin.json', marker_pin)


def _validate_sdk(root, recipe):
    return sdk.verify_recovery_sdk_identity(root, authorized_boot='owned-boot',
        recipe_sha256=recipe.sha256, learner_source_sha256=recipe.learner_source_sha256)


def test_anchor_comes_from_exact_cp_journal_even_when_pause_metadata_replaced_it(tmp_path, monkeypatch):
    recipe, payload, _ = _anchor_fixture(tmp_path, monkeypatch)
    assert set(payload['metadata']) == {'purpose'}
    anchor = recovery._inherited_sdk_anchor(payload, recipe, authorized_boot='owned-boot')
    assert anchor['target_learner_source_sha256'] == recipe.learner_source_sha256
    assert anchor['startup_parameters_sha256'] == payload['collection']['task_source'][
        'startup_parameters_sha256']
    assert not (tmp_path / 'source.f32').exists()
    assert not (tmp_path / 'target.f32').exists()


def test_same_sdk_fork_revalidates_actual_environment_without_old_raw_tables(tmp_path, monkeypatch):
    root, recipe, marker, original, inherited, installed = _continuation_fixture(tmp_path, monkeypatch)
    recovery._require_sdk_metadata(marker, recipe, verify_mechanics=True)
    assert inherited['collection']['task_source']['startup_parameters_sha256'] == original[
        'collection']['task_source']['startup_parameters_sha256']
    _seal_marker(root, marker)
    receipt = _validate_sdk(root, recipe)
    assert receipt['status'] == 'matched_before_sdk_launch'
    assert receipt['sdk_continuation_pin'] == marker['source']['sdk_continuation_pin']
    assert receipt['target_sdk_pin'] == marker['source']['sdk_continuation']['target_sdk_pin']
    assert len(installed.commands) == 3
    assert not (tmp_path / 'source.f32').exists() and not (tmp_path / 'target.f32').exists()


def test_third_source_repair_links_to_immediate_continuation_not_original_migration(tmp_path, monkeypatch):
    root, recipe, marker, _, payload, _ = _continuation_fixture(tmp_path, monkeypatch)
    payload['boundary'] = deepcopy(marker['boundary'])
    payload['metadata'] = {'purpose': 'periodic_learning_resume'}
    anchor = recovery._inherited_sdk_anchor(payload, recipe, authorized_boot='owned-boot')
    third = tmp_path / 'third-session'
    third.mkdir()
    summary = recovery._publish_sdk_continuation(third, anchor,
        authorized_boot='owned-boot', checkpoint_sha256='f' * 64, recipe=recipe,
        target_source_sha256='9' * 64, original_sha256='3' * 64, inherited_sha256='4' * 64)
    assert summary['prior_sdk_evidence_pin'] == marker['source']['sdk_continuation_pin']
    identity = sdk._sdk_evidence_identity(summary['evidence_pin'], authorized_boot='owned-boot')
    assert identity['target_learner_source_sha256'] == '9' * 64
    assert identity['target_sdk_pin'] == anchor['target_sdk_pin']
    assert identity['startup_parameters_sha256'] == anchor['startup_parameters_sha256']


@pytest.mark.parametrize('fault', [
    'source_chain', 'sdk_change', 'mechanical_change', 'boot', 'checkpoint',
    'target_source', 'extra_field', 'summary_field', 'missing_pin', 'mixed_branches',
])
def test_continuation_cannot_redirect_or_bypass_gate(tmp_path, monkeypatch, fault):
    root, recipe, marker, _, _, _ = _continuation_fixture(tmp_path, monkeypatch)
    summary = marker['source']['sdk_continuation']
    document = {key: summary[key] for key in sdk._SDK_CONTINUATION_FIELDS}
    if fault == 'source_chain':
        document['source_learner_source_sha256'] = '8' * 64
    elif fault == 'sdk_change':
        document['target_sdk_pin'] = _seal(tmp_path / 'different-sdk.json', {'different': True})
    elif fault == 'mechanical_change':
        document['startup_parameters_sha256'] = '0' * 64
    elif fault == 'boot':
        document['authorized_boot'] = 'different-boot'
    elif fault == 'checkpoint':
        document['source_checkpoint_sha256'] = '0' * 64
    elif fault == 'target_source':
        document['target_learner_source_sha256'] = '0' * 64
    elif fault == 'extra_field':
        document['unreviewed'] = True
    elif fault == 'summary_field':
        summary['unreviewed'] = True
    elif fault == 'missing_pin':
        marker['source'].pop('sdk_continuation_pin')
    else:
        marker['source']['sdk_transition_pin'] = summary['prior_sdk_evidence_pin']
    if fault not in ('summary_field', 'missing_pin', 'mixed_branches'):
        changed = _seal(tmp_path / 'changed-continuation.json', document)
        marker['source']['sdk_continuation_pin'] = changed
        summary.update(document, evidence_pin=changed)
    _seal_marker(root, marker)
    monkeypatch.setattr(sdk, 'verify_native_sdk_identity',
                        lambda *_: pytest.fail('invalid continuation must fail before checking environment'))
    with pytest.raises(ValueError):
        recovery._require_sdk_metadata(marker, recipe)
    with pytest.raises(ValueError):
        _validate_sdk(root, recipe)


@pytest.mark.parametrize('fault', ['startup', 'source', 'dropped_ancestor', 'forged_summary'])
def test_verified_cp_must_match_its_inherited_sdk_anchor(tmp_path, monkeypatch, fault):
    recipe, payload, _ = _anchor_fixture(tmp_path, monkeypatch)
    if fault == 'startup':
        payload['collection']['task_source']['startup_parameters_sha256'] = '0' * 64
    elif fault == 'source':
        recipe = replace(recipe, learner_source_sha256='9' * 64)
    elif fault == 'dropped_ancestor':
        payload['boundary'] = {'recovery': {'checkpoint': payload['boundary'], 'failed': {},
                                           'provenance': {}}}
    else:
        payload['boundary']['recovery']['provenance']['sdk_transition'][
            'target_startup_parameters_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        recovery._inherited_sdk_anchor(payload, recipe, authorized_boot='owned-boot')


def test_omitting_current_sdk_fields_cannot_turn_inherited_gate_into_ordinary_recovery(tmp_path, monkeypatch):
    root, recipe, marker, _, _, _ = _continuation_fixture(tmp_path, monkeypatch)
    marker['source'] = {}
    _seal_marker(root, marker)
    with pytest.raises(ValueError, match='discarded'):
        recovery._require_sdk_metadata(marker, recipe)
    with pytest.raises(ValueError, match='discarded'):
        _validate_sdk(root, recipe)


def test_actual_sdk_change_still_rejects_before_any_simulator_open(tmp_path, monkeypatch):
    root, recipe, marker, _, _, installed = _continuation_fixture(tmp_path, monkeypatch)
    _seal_marker(root, marker)
    installed.installed['isaacsim'] = 'different-version'
    with pytest.raises(ValueError, match='package versions'):
        _validate_sdk(root, recipe)


def test_in_memory_cyclic_continuation_resolver_is_bounded(tmp_path, monkeypatch):
    pin = {'path': '/synthetic/continuation.json', 'bytes': 1, 'sha256': '1' * 64}
    document = dict.fromkeys(sdk._SDK_CONTINUATION_FIELDS, '2' * 64)
    document.update(format=sdk._SDK_CONTINUATION_FORMAT, authorized_boot='owned-boot',
                    prior_sdk_evidence_pin=pin, target_sdk_pin=pin)
    monkeypatch.setattr(sdk, '_sdk_pin_document', lambda _: document)
    with pytest.raises(ValueError, match='cyclic'):
        sdk._sdk_evidence_identity(pin, authorized_boot='owned-boot')


def test_telemetry_files_need_exact_separate_review_and_do_not_open_policy_changes(tmp_path):
    before, after = tmp_path / 'old', tmp_path / 'new'
    before.mkdir()
    after.mkdir()
    for name in ('formal/learning/optimization.py', 'formal/runtime/telemetry.py'):
        for directory, text in ((before, '# Frozen fixture.\n'), (after, '# Reviewed telemetry fixture.\n')):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    old, new = source_table(before), source_table(after)
    changes = {name: {'before': old[name], 'after': new[name]} for name in old}
    with pytest.raises(ValueError, match='exact reviewed'):
        recovery._review_sources(before, after, changes, sdk_transition=True)
    assert recovery._review_sources(before, after, changes, same_sdk=True,
                                    telemetry_review=changes) == changes
    wrong = deepcopy(changes)
    wrong['formal/learning/optimization.py']['before'] = '0' * 64
    with pytest.raises(ValueError):
        recovery._review_sources(before, after, changes, same_sdk=True, telemetry_review=wrong)
    path = after / 'formal/policy.py'
    path.write_text('# Unreviewable policy change.\n')
    changed = dict(changes, **{'formal/policy.py': {'before': None,
                    'after': source_table(after)['formal/policy.py']}})
    with pytest.raises(ValueError, match='exact reviewed'):
        recovery._review_sources(before, after, changed, same_sdk=True, telemetry_review=changes)

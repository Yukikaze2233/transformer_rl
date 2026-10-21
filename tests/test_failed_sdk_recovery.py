"""Sealed SDK migration admits physical representation changes, not learning drift."""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json

import pytest
import torch

from transformer_rl.common.training_control import file_pin, write_json
from transformer_rl.formal.runtime.recovery import (
    _apply_sdk_identity, _mechanical_pair, _pinned_document, _require_preserved_learning,
    _require_sdk_metadata, _review_sources, _sdk_transition, prepare_failed_recovery,
)
from transformer_rl.formal.runtime.transition import learning_state_sha256, source_table


@dataclass(frozen=True)
class RecipeIdentity:
    num_envs: int = 2
    learner_source_sha256: str = 'a' * 64

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


def sealed_json(path, value):
    write_json(path, value)
    return file_pin(path)


def mechanical_rows(num_envs=2):
    masses = torch.ones(num_envs, 19, dtype=torch.float32)
    coms = torch.zeros(num_envs, 19, 7, dtype=torch.float32)
    coms[..., 6] = 1.
    inertias = torch.diag(torch.tensor([.1, .2, .3])).expand(num_envs, 19, 3, 3).clone()
    return masses, coms, inertias


def mechanical_pair(tmp_path, *, transform=None):
    before = mechanical_rows()
    after = tuple(value.clone() for value in before)
    # An actual native representation change; the link-frame mass/COM/I stay exact.
    after[1][..., 3:] *= -1
    if transform is not None:
        transform(after)
    pins = []
    for role, rows in [('source', before), ('target', after)]:
        raw = torch.cat([value.flatten() for value in rows]).numpy().tobytes()
        path = tmp_path / (role + '.f32')
        path.write_bytes(raw)
        pins.append(file_pin(path))
    value = {'format': 'private_native_mechanical_pair_v1', 'num_envs': 2,
             'body_order': ['base_link', *[f'body_{i}' for i in range(1, 19)]],
             'byte_layout': 'float32_le_mass_com_xyzw_inertia',
             'inertia_frame': 'center_of_mass_in_rigid_body_prim',
             'source_raw_pin': pins[0], 'target_raw_pin': pins[1]}
    return value, sealed_json(tmp_path / 'mechanics.json', value)


def sdk_evidence(tmp_path):
    mechanical, mechanical_pin = mechanical_pair(tmp_path)
    recipe = RecipeIdentity()
    source_sdk = sealed_json(tmp_path / 'old-sdk.json', {'isaaclab': 'old-pinned-sdk'})
    target_sdk = sealed_json(tmp_path / 'new-sdk.json', {'isaaclab': 'new-pinned-sdk'})
    value = {'format': 'private_sdk_mechanical_transition_v1', 'authorized_boot': 'owned-boot',
             'source_checkpoint_sha256': 'c' * 64, 'source_recipe_sha256': recipe.sha256,
             'source_learner_source_sha256': recipe.learner_source_sha256,
             'target_learner_source_sha256': 'b' * 64, 'source_sdk_pin': source_sdk,
             'target_sdk_pin': target_sdk, 'physical_compatibility_pin': mechanical_pin}
    return recipe, mechanical, value, sealed_json(tmp_path / 'sdk-transition.json', value)


def accepted_evidence(recipe, mechanical, pin):
    return _sdk_transition(pin, authorized_boot='owned-boot', checkpoint_sha256='c' * 64,
        recipe=recipe, target_source_sha256='b' * 64,
        source_startup_sha256=mechanical['source_raw_pin']['sha256'])


def learning_payload(startup_sha):
    return {'model': {'weight': torch.tensor([1., 2.])},
            'actor_adam': {'step': torch.tensor(64), 'exp_avg': torch.tensor([.1])},
            'critic_adam': {'step': torch.tensor(64), 'exp_avg': torch.tensor([.2])},
            'trainer': {'update_index': 200},
            'collection': {'returned_vector_steps': 9600, 'task_source': {
                'format': 'managed_lab_task_source_v4', 'identity': {'asset': 'unchanged'},
                'startup_parameters_sha256': startup_sha,
                'course': {'next_id': torch.tensor([23, 24]), 'rng': torch.tensor([9, 10])},
                'sensor_model': {'rng': torch.tensor([11, 12])},
                'cold_physical_state_restored': False}},
            'curriculum': {'consumed': [200, 0, 0, 0]},
            'sampler': {'rng': torch.tensor([13, 14])}, 'terrain': {'level': torch.tensor([2, 3])},
            'protocol': {'max_seen_id': torch.tensor([21, 22])},
            'global_rng': {'torch': torch.tensor([15, 16])}, 'normalization': {'enabled': False}}


def test_complete_native_table_accepts_only_physical_equivalence(tmp_path):
    value, pin = mechanical_pair(tmp_path)
    proof = _mechanical_pair(pin, num_envs=2, source_sha256=value['source_raw_pin']['sha256'])
    assert proof['source_startup_parameters_sha256'] != proof['target_startup_parameters_sha256']
    assert all(item['max_abs_difference'] == 0 for item in proof['comparison'].values())
    with pytest.raises(ValueError, match='reconstructed'):
        _mechanical_pair(pin, num_envs=2, source_sha256='f' * 64)
    with pytest.raises(ValueError, match='contract'):
        _mechanical_pair(pin, num_envs=3, source_sha256=value['source_raw_pin']['sha256'])


@pytest.mark.parametrize('change', ['mass', 'com', 'inertia', 'quaternion', 'nan', 'negative_mass', 'non_spd'])
def test_native_table_rejects_drift_or_invalid_rows(tmp_path, change):
    def transform(rows):
        mass, com, inertia = rows
        if change == 'mass':
            mass[1, 18] += .01
        elif change == 'com':
            com[1, 18, 0] += .01
        elif change == 'inertia':
            inertia[1, 18, 0, 0] += .01
        elif change == 'quaternion':
            com[1, 18, 3:] *= .5
        elif change == 'nan':
            com[1, 18, 0] = float('nan')
        elif change == 'negative_mass':
            mass[1, 18] = -1
        else:
            inertia[1, 18, 0, 0] = -1
    value, pin = mechanical_pair(tmp_path, transform=transform)
    with pytest.raises(ValueError, match='physical|invalid native'):
        _mechanical_pair(pin, num_envs=2, source_sha256=value['source_raw_pin']['sha256'])


@pytest.mark.parametrize('change', ['missing_row', 'frame', 'layout', 'producer_pass', 'body_alias'])
def test_native_descriptor_cannot_hide_missing_rows_or_supply_a_pass_flag(tmp_path, change):
    value, _ = mechanical_pair(tmp_path)
    if change == 'missing_row':
        value['target_raw_pin']['bytes'] -= 4
    elif change == 'frame':
        value['inertia_frame'] = 'unspecified'
    elif change == 'layout':
        value['byte_layout'] = 'per_environment_interleaved'
    elif change == 'producer_pass':
        value['passed'] = True
    else:
        value['body_order'][-1] = value['body_order'][0]
    pin = sealed_json(tmp_path / 'bad-descriptor.json', value)
    with pytest.raises(ValueError, match='contract|all native'):
        _mechanical_pair(pin, num_envs=2, source_sha256=value['source_raw_pin']['sha256'])


def test_evidence_files_are_sealed_and_cannot_follow_symlinks(tmp_path):
    path = tmp_path / 'inventory.json'
    pin = sealed_json(path, {'sdk': 'fixed'})
    path.write_bytes(path.read_bytes().replace(b'fixed', b'other'))
    with pytest.raises(ValueError, match='seal'):
        _pinned_document(pin)
    link = tmp_path / 'alias.json'
    link.symlink_to(path)
    with pytest.raises(ValueError, match='symlinks'):
        _pinned_document(dict(file_pin(path), path=str(link)))


@pytest.mark.parametrize('field', ['authorized_boot', 'source_checkpoint_sha256', 'source_recipe_sha256',
                                   'source_learner_source_sha256', 'target_learner_source_sha256'])
def test_sdk_transition_binds_both_sources_recipe_checkpoint_and_boot(tmp_path, field):
    recipe, mechanical, value, _ = sdk_evidence(tmp_path)
    value[field] = 'wrong'
    pin = sealed_json(tmp_path / 'bad-binding.json', value)
    with pytest.raises(ValueError, match='binding'):
        accepted_evidence(recipe, mechanical, pin)


def test_projection_preserves_both_adams_rng_course_and_protocol(tmp_path):
    recipe, mechanical, _, pin = sdk_evidence(tmp_path)
    evidence = accepted_evidence(recipe, mechanical, pin)
    original = learning_payload(evidence['source_startup_parameters_sha256'])
    before = learning_state_sha256(original)
    inherited = deepcopy(original)
    inherited['curriculum']['consumed'][0] = 241
    _apply_sdk_identity(inherited, original, evidence)
    _require_preserved_learning(inherited, original, before, evidence)
    assert original['curriculum']['consumed'][0] == 200
    assert inherited['collection']['task_source']['startup_parameters_sha256'] == evidence[
        'target_startup_parameters_sha256']
    assert learning_state_sha256(inherited) != before
    for name in ('model', 'actor_adam', 'critic_adam', 'trainer', 'sampler', 'terrain',
                 'protocol', 'global_rng', 'normalization'):
        bad = deepcopy(inherited)
        bad[name]['unauthorized_change'] = True
        with pytest.raises(ValueError, match='changed model'):
            _require_preserved_learning(bad, original, before, evidence)
    for name in ('course', 'sensor_model', 'identity'):
        bad = deepcopy(inherited)
        bad['collection']['task_source'][name]['unauthorized_change'] = True
        with pytest.raises(ValueError, match='changed model'):
            _require_preserved_learning(bad, original, before, evidence)
    with pytest.raises(ValueError, match='changed model'):
        _require_preserved_learning(inherited, original, before)


def test_sdk_branch_cannot_be_enabled_or_omitted_without_matching_approval(tmp_path, monkeypatch):
    from transformer_rl.formal.runtime import recovery

    monkeypatch.setattr(recovery, 'boot_id', lambda: 'owned-boot')
    _, _, _, sdk_pin = sdk_evidence(tmp_path)
    for declared, provided in [(None, sdk_pin), (sdk_pin, None)]:
        approval = {'authorization': 'failed_checkpoint_fork_preserving_all_consumption',
                    'boot': 'owned-boot', 'human_objective': 'preserve exact learning state'}
        if declared is not None:
            approval['sdk_transition_pin'] = declared
        approval_pin = sealed_json(tmp_path / ('approval-' + str(provided is None) + '.json'), approval)
        target = tmp_path / ('rejected-' + str(provided is None))
        with pytest.raises(ValueError, match='explicit approval'):
            prepare_failed_recovery(target, cache_review_pin=None, previous_recipe=None,
                previous_source=None, learner_source=None, reviewed_changes={},
                authorization_pin=approval_pin, authorized_boot='owned-boot',
                runtime_factory='', runtime_configuration={}, resources={}, python='',
                sdk_transition_pin=provided)
        assert not target.exists()


def test_sealed_recovery_metadata_rechecks_evidence_without_relaxing_runtime_identity(tmp_path):
    recipe, mechanical, _, pin = sdk_evidence(tmp_path)
    evidence = accepted_evidence(recipe, mechanical, pin)
    original = learning_payload(evidence['source_startup_parameters_sha256'])
    inherited = deepcopy(original)
    inherited['curriculum']['consumed'][0] = 241
    _apply_sdk_identity(inherited, original, evidence)
    old_hash, new_hash = learning_state_sha256(original), learning_state_sha256(inherited)
    approval = sealed_json(tmp_path / 'approval.json', {'sdk_transition_pin': pin,
                                                       'checkpoint_sha256': 'c' * 64})
    marker = {'authorized_boot': 'owned-boot', 'learning_state_sha256': new_hash,
              'source': {'authorization_pin': approval, 'checkpoint_learning_state_sha256': old_hash,
                         'sdk_transition_pin': pin,
                         'sdk_transition': dict(evidence, checkpoint_learning_state_sha256=old_hash,
                                                inherited_learning_state_sha256=new_hash)}}
    target_recipe = replace(recipe, learner_source_sha256='b' * 64)
    _require_sdk_metadata(marker, target_recipe)
    _require_sdk_metadata(marker, target_recipe, verify_mechanics=True)
    bad = deepcopy(marker)
    bad['source']['sdk_transition']['target_startup_parameters_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='proof differs'):
        _require_sdk_metadata(bad, target_recipe, verify_mechanics=True)
    with pytest.raises(ValueError, match='bound'):
        _require_sdk_metadata(marker, replace(recipe, learner_source_sha256='e' * 64))


def test_ga_source_review_is_explicit_and_cannot_admit_ppo_or_policy_changes(tmp_path):
    before, after = tmp_path / 'old', tmp_path / 'new'
    before.mkdir()
    after.mkdir()
    def reviewed_change(name):
        for directory, body in [(before, '# Old frozen source.\n'), (after, '# Reviewed source.\n')]:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        old, new = source_table(before), source_table(after)
        return {key: {'before': old.get(key), 'after': new.get(key)}
                for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    changes = reviewed_change('formal/lab_env/dynamics.py')
    with pytest.raises(ValueError, match='exact reviewed'):
        _review_sources(before, after, changes)
    assert _review_sources(before, after, changes, sdk_transition=True) == changes
    changes = reviewed_change('formal/lab_env/contact_measurement.py')
    assert _review_sources(before, after, changes, sdk_transition=True) == changes
    changes = reviewed_change('formal/learning/optimization.py')
    with pytest.raises(ValueError, match='exact reviewed'):
        _review_sources(before, after, changes, sdk_transition=True)

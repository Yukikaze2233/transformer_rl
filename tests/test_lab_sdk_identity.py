"""Sealed GA migration identity is checked before any simulator launch."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from transformer_rl.formal.lab_env import sdk


def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def publish(path, document):
    path.write_text(json.dumps(document))
    return pin(path)


def identity_fixture(tmp_path, monkeypatch):
    checkout = tmp_path / 'SDK'
    checkout.mkdir()
    lock = checkout / 'uv.lock'
    lock.write_text('version = 1\n')
    origins = {}
    for module in ('isaaclab', 'isaaclab_physx'):
        path = checkout / 'source' / module / module / '__init__.py'
        path.parent.mkdir(parents=True)
        path.write_text('# synthetic SDK module origin; never imported\n')
        origins[module] = path
    system_python = tmp_path / 'shared-python'
    system_python.write_text('synthetic executable identity')
    executable = tmp_path / 'private-venv' / 'bin' / 'python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(system_python)
    monkeypatch.setattr(sdk.sys, 'executable', str(executable))
    sysroot, driver = tmp_path / 'private-sysroot', tmp_path / 'driver'
    sysroot.mkdir()
    driver.mkdir()
    monkeypatch.setenv('LD_LIBRARY_PATH', f'{driver}:{sysroot}:{driver}:/additional/path')
    packages = {'isaaclab': '3.0.0', 'isaaclab-physx': '3.0.0', 'isaacsim': '6.2.0.0',
                'torch': '2.11.0+cu130', 'warp-lang': '1.17.0'}
    installed = dict(packages)
    monkeypatch.setattr(sdk.metadata, 'version', installed.__getitem__)
    monkeypatch.setattr(sdk.util, 'find_spec', lambda name: SimpleNamespace(origin=str(origins[name])))
    git_outputs = {'commit': 'a' * 40, 'tracked': '', 'source_untracked': ''}
    commands = []

    def run(command, *, check, capture_output, text, timeout):
        assert command[:4] == ['git', '--no-optional-locks', '-C', str(checkout)]
        assert check and capture_output and text and timeout == 15
        arguments = command[4:]
        commands.append(arguments)
        if arguments == ['rev-parse', '--verify', 'HEAD']:
            key = 'commit'
        elif arguments == ['status', '--porcelain', '--untracked-files=no']:
            key = 'tracked'
        else:
            assert arguments == ['status', '--porcelain', '--untracked-files=all', '--',
                                 'source/isaaclab', 'source/isaaclab_physx']
            key = 'source_untracked'
        return SimpleNamespace(stdout=git_outputs[key])

    monkeypatch.setattr(sdk.subprocess, 'run', run)
    document = {
        'format': 'private_native_sdk_identity_v1', 'interpreter': str(executable),
        'python': {'major': sys.version_info.major, 'minor': sys.version_info.minor},
        'checkout': {'directory': str(checkout), 'commit': 'a' * 40, 'uv_lock_pin': pin(lock)},
        'packages': packages, 'required_environment': {'LD_LIBRARY_PATH': [str(sysroot), str(driver)]},
        'contracts': {'force_physx': True, 'use_newton_actuators': False, 'quaternion_order': 'xyzw'}}
    path = tmp_path / 'target-sdk.json'
    return SimpleNamespace(document=document, path=path, installed=installed, git=git_outputs,
                           origins=origins, commands=commands, executable=executable,
                           system_python=system_python, lock=lock, driver=driver)


def test_target_identity_matches_the_actual_process_without_resolving_the_venv(tmp_path, monkeypatch):
    state = identity_fixture(tmp_path, monkeypatch)
    receipt = sdk.verify_native_sdk_identity(publish(state.path, state.document))
    assert receipt['interpreter'] == str(state.executable)
    assert receipt['interpreter'] != str(state.executable.resolve())
    assert receipt['packages'] == state.installed
    assert receipt['status'] == 'matched_before_sdk_launch'
    assert receipt['tracked_checkout_clean'] and receipt['source_untracked_clean']
    assert len(state.commands) == 3


@pytest.mark.parametrize('fault', [
    'shared_interpreter', 'other_venv', 'python', 'commit', 'tracked_source', 'new_source',
    'lock', 'package', 'package_missing', 'module_origin', 'environment', 'missing_path',
    'newton', 'non_boolean_physx', 'quaternion',
])
def test_changed_target_identity_is_rejected_before_launch(tmp_path, monkeypatch, fault):
    state = identity_fixture(tmp_path, monkeypatch)
    document = state.document
    if fault == 'shared_interpreter':
        monkeypatch.setattr(sdk.sys, 'executable', str(state.system_python))
    elif fault == 'other_venv':
        other = tmp_path / 'other-python'
        other.symlink_to(state.system_python)
        monkeypatch.setattr(sdk.sys, 'executable', str(other))
    elif fault == 'python':
        document['python']['minor'] += 1
    elif fault == 'commit':
        state.git['commit'] = 'b' * 40
    elif fault == 'tracked_source':
        state.git['tracked'] = ' M source/isaaclab/isaaclab/app/sim_launcher.py'
    elif fault == 'new_source':
        state.git['source_untracked'] = '?? source/isaaclab/isaaclab/new_module.py'
    elif fault == 'lock':
        state.lock.write_text('version = 2\n')
    elif fault == 'package':
        state.installed['torch'] = 'changed'
    elif fault == 'package_missing':
        document['packages'].pop('warp-lang')
    elif fault == 'module_origin':
        alternate = tmp_path / 'shadow.py'
        alternate.write_text('# never imported')
        state.origins['isaaclab'] = alternate
    elif fault == 'environment':
        monkeypatch.setenv('LD_LIBRARY_PATH', str(state.driver))
    elif fault == 'missing_path':
        state.driver.rmdir()
    elif fault == 'newton':
        document['contracts']['use_newton_actuators'] = True
    elif fault == 'non_boolean_physx':
        document['contracts']['force_physx'] = 1
    else:
        document['contracts']['quaternion_order'] = 'wxyz'
    with pytest.raises(ValueError):
        sdk.verify_native_sdk_identity(publish(state.path, document))


def test_identity_payload_change_or_symlink_cannot_keep_an_old_seal(tmp_path, monkeypatch):
    state = identity_fixture(tmp_path, monkeypatch)
    sealed = publish(state.path, state.document)
    state.path.write_text(state.path.read_text() + ' ')
    with pytest.raises(ValueError, match='file identity|bytes differ'):
        sdk.verify_native_sdk_identity(sealed)
    alias = tmp_path / 'target-alias.json'
    alias.symlink_to(state.path)
    with pytest.raises(ValueError, match='canonical absolute'):
        sdk.verify_native_sdk_identity({**pin(state.path), 'path': str(alias)})


def recovery_fixture(tmp_path, monkeypatch):
    state = identity_fixture(tmp_path, monkeypatch)
    target_pin = publish(state.path, state.document)
    root = tmp_path / 'session'
    root.mkdir()
    transition = {
        'format': 'private_sdk_mechanical_transition_v1', 'authorized_boot': 'test-boot',
        'source_checkpoint_sha256': 'b' * 64, 'source_recipe_sha256': 'c' * 64,
        'source_learner_source_sha256': 'd' * 64, 'target_learner_source_sha256': 'e' * 64,
        'source_sdk_pin': {'path': '/retired/source/provenance.json', 'bytes': 1, 'sha256': 'f' * 64},
        'target_sdk_pin': target_pin,
        'physical_compatibility_pin': {'path': '/retired/physical/proof.json', 'bytes': 1, 'sha256': 'f' * 64}}
    transition_path = tmp_path / 'sdk-transition.json'
    transition_pin = publish(transition_path, transition)
    marker = {'format': 'private_failed_learning_fork', 'target_session': str(root),
              'authorized_boot': 'test-boot', 'target_recipe_sha256': 'a' * 64,
              'source': {'sdk_transition_pin': transition_pin,
                         'sdk_transition': {'evidence_pin': transition_pin, 'target_sdk_pin': target_pin}}}
    return state, root, transition_path, transition, marker


def seal_marker(root, marker):
    marker_pin = publish(root / 'recovery.json', marker)
    publish(root / 'recovery.pin.json', marker_pin)


def validate_recovery(root):
    return sdk.verify_recovery_sdk_identity(root, authorized_boot='test-boot',
        recipe_sha256='a' * 64, learner_source_sha256='e' * 64)


def test_sealed_recovery_links_target_identity_without_rereading_retired_physics(tmp_path, monkeypatch):
    state, root, _, _, marker = recovery_fixture(tmp_path, monkeypatch)
    seal_marker(root, marker)
    receipt = validate_recovery(root)
    assert receipt['target_sdk_pin'] == marker['source']['sdk_transition']['target_sdk_pin']
    assert receipt['sdk_transition_pin'] == marker['source']['sdk_transition_pin']
    assert receipt['interpreter'] == str(state.executable)


@pytest.mark.parametrize('fault', ['unsealed', 'direct_pin_missing', 'summary_missing', 'summary_pin',
                                  'summary_target', 'boot', 'recipe', 'source', 'target_pin'])
def test_recovery_link_cannot_bypass_or_redirect_the_target_gate(tmp_path, monkeypatch, fault):
    _, root, transition_path, transition, marker = recovery_fixture(tmp_path, monkeypatch)
    if fault == 'direct_pin_missing':
        marker['source'].pop('sdk_transition_pin')
    elif fault == 'summary_missing':
        marker['source'].pop('sdk_transition')
    elif fault == 'summary_pin':
        marker['source']['sdk_transition']['evidence_pin'] = {'changed': True}
    elif fault == 'summary_target':
        marker['source']['sdk_transition']['target_sdk_pin'] = {'changed': True}
    elif fault == 'boot':
        marker['authorized_boot'] = 'different-boot'
    elif fault == 'recipe':
        marker['target_recipe_sha256'] = 'b' * 64
    elif fault == 'source':
        transition['target_learner_source_sha256'] = 'f' * 64
    elif fault == 'target_pin':
        transition['target_sdk_pin']['sha256'] = '0' * 64
    if fault in ('source', 'target_pin'):
        changed = publish(transition_path, transition)
        marker['source']['sdk_transition_pin'] = changed
        marker['source']['sdk_transition']['evidence_pin'] = changed
    seal_marker(root, marker)
    if fault == 'unsealed':
        (root / 'recovery.json').write_text('{}')
    with pytest.raises(ValueError):
        validate_recovery(root)


def test_fresh_and_ordinary_same_sdk_recovery_need_no_new_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk, 'verify_native_sdk_identity', lambda *_: pytest.fail('ordinary recovery has no SDK transition'))
    assert validate_recovery(tmp_path) is None
    seal_marker(tmp_path, {'source': {}})
    assert validate_recovery(tmp_path) is None


def test_runtime_identity_failure_is_published_before_the_sdk_can_open(tmp_path, monkeypatch):
    from transformer_rl.formal.lab_env import runtime as module

    monkeypatch.setattr(module, 'parse_configuration', lambda *_: (None,) * 5 + ({},) + (None,) * 3)
    monkeypatch.setattr(module, 'ManagedCurriculum', lambda *_: None)
    failure = ValueError('changed target SDK')

    def reject(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module, 'verify_recovery_sdk_identity', reject)
    monkeypatch.setattr(module, 'LabSimulation', lambda *_: pytest.fail('identity must reject before SDK launch'))
    errors = []
    control = SimpleNamespace(binding={'root': str(tmp_path), 'boot_id': 'test-boot'}, fail=errors.append)
    recipe = SimpleNamespace(device='cuda:0', sha256='a' * 64, learner_source_sha256='e' * 64)
    with pytest.raises(ValueError) as caught:
        with module.runtime(recipe, None, {}, control=control):
            pytest.fail('identity failure must not yield')
    assert caught.value is failure and errors == [failure]

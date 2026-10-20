"""Kaiser CPU checks for post-exec ownership and terminal PID retirement."""
from types import SimpleNamespace

import pytest

from transformer_rl.common import training_control as control
from transformer_rl.formal.runtime import session as module


ARGV = ['/usr/bin/python', '-m', 'owned_worker']
IDENTITY = {'pid': 12345, 'start_ticks': 91, 'boot_id': 'test-boot', 'argv': ARGV}


def test_child_binding_waits_for_two_full_post_exec_observations(monkeypatch):
    calls, sleeps = [], []
    samples = iter([dict(IDENTITY, argv=[]), None, IDENTITY.copy(), IDENTITY.copy()])
    monkeypatch.setattr(module, 'boot_id', lambda: 'test-boot')
    monkeypatch.setattr(module, 'process_identity', lambda pid: (calls.append(pid), next(samples))[1])
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    child = SimpleNamespace(pid=12345, poll=lambda: None)
    assert module.StructuredTrainingSession._bind_child_identity(child, ARGV, 'test-boot') == IDENTITY
    assert calls == [12345] * 4 and len(sleeps) == 3


def test_unexpected_argv_breaks_a_partially_matching_binding(monkeypatch):
    samples = iter([IDENTITY.copy(), dict(IDENTITY, argv=['another_program']),
                    IDENTITY.copy(), IDENTITY.copy()])
    reads = []
    monkeypatch.setattr(module, 'boot_id', lambda: 'test-boot')
    monkeypatch.setattr(module, 'process_identity', lambda pid: (reads.append(pid), next(samples))[1])
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    assert module.StructuredTrainingSession._bind_child_identity(
        SimpleNamespace(pid=12345, poll=lambda: None), ARGV, 'test-boot') == IDENTITY
    assert len(reads) == 4


@pytest.mark.parametrize('boot,returncode', [('different-boot', None), ('test-boot', 0)])
def test_changed_boot_or_reaped_child_is_never_polled(monkeypatch, boot, returncode):
    monkeypatch.setattr(module, 'boot_id', lambda: boot)
    monkeypatch.setattr(module, 'process_identity', lambda _: pytest.fail('unowned or retired PID read'))
    assert module.StructuredTrainingSession._bind_child_identity(
        SimpleNamespace(pid=12345, poll=lambda: returncode), ARGV, 'test-boot') is None


def test_incomplete_identity_timeout_is_not_replaced_with_expected_argv(monkeypatch):
    clock = iter([0., .1, .2, 1.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(module, 'boot_id', lambda: 'test-boot')
    monkeypatch.setattr(module, 'process_identity', lambda _: dict(IDENTITY, argv=[]))
    assert module.StructuredTrainingSession._bind_child_identity(
        SimpleNamespace(pid=12345, poll=lambda: None), ARGV, 'test-boot', timeout_s=.5) is None


@pytest.mark.parametrize('fault', ['empty_argv', 'wrong_start', 'missing_identity', 'nonzero', 'valid'])
def test_any_parent_exit_retires_pid_and_bad_binding_refuses_resume(tmp_path, monkeypatch, fault):
    value = control.ManagedSession.__new__(control.ManagedSession)
    value.root = tmp_path
    value.spec = dict(name='unit', updates=10, architecture='unit', device='cpu', seed=1)
    attempt = tmp_path/'attempts/0001-aaaaaaaaaaaa'
    (attempt/'control').mkdir(parents=True)
    control.write_json(tmp_path/'current.json', {'attempt': attempt.name})
    control.write_json(attempt/'request.json', {'token': 'unit-token', 'boot_id': 'test-boot', 'submitted_epoch': 0})
    control.write_json(attempt/'control/request.json', {'operation': 'stop', 'sequence': 1})
    status = {'token': 'unit-token', 'identity': IDENTITY, 'phase': 'stopped', 'update': 2}
    control.write_json(attempt/'control/status.json', status)
    control.write_json(attempt/'control/result.json', {'token': 'unit-token', 'summary': status})
    identity = IDENTITY
    if fault == 'empty_argv':
        identity = dict(IDENTITY, argv=[])
    elif fault == 'wrong_start':
        identity = dict(IDENTITY, start_ticks=90)
    elif fault == 'missing_identity':
        identity = None
    control.write_json(attempt/'exit.json', {'identity': identity, 'returncode': 1 if fault == 'nonzero' else 0})
    monkeypatch.setattr(control, 'boot_id', lambda: 'test-boot')
    monkeypatch.setattr(control, '_lease_is_held', lambda _: False)
    monkeypatch.setattr(control, 'process_identity', lambda _: pytest.fail('terminal PID must not be read'))
    state = value.status()
    assert not state['active']
    assert state['phase'] == ('stopped' if fault == 'valid' else 'failed')
    assert state['checkpoint_resume'] == (fault == 'valid')

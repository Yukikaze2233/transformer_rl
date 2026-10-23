"""A saved learner endpoint alone cannot acknowledge a completed cold stop."""
from pathlib import Path

from transformer_rl.common import training_control as control


def test_saved_endpoint_waits_for_the_matching_parent_exit(tmp_path, monkeypatch):
    root = tmp_path / 'session'
    attempt = root / 'attempts/0001-123456abcdef'
    identity = {'pid': 123, 'boot_id': 'test-boot', 'start_ticks': 7, 'argv': ['worker']}
    worker = {'token': 'token', 'identity': identity, 'phase': 'stopped', 'update': 4}
    documents = {
        str(attempt / 'request.json'): {'token': 'token', 'boot_id': 'test-boot'},
        str(attempt / 'control/status.json'): worker,
        str(attempt / 'control/request.json'): {'operation': 'stop', 'sequence': 2},
        str(attempt / 'control/result.json'): {'token': 'token', 'summary': worker},
    }
    session = object.__new__(control.ManagedSession)
    session.root = root
    session.requires_parent_exit = True
    session.spec = dict(name='test', updates=10, architecture='test', device='cpu', seed=1)
    monkeypatch.setattr(session, 'attempt', lambda: attempt)
    monkeypatch.setattr(control, 'read_json', lambda path, default=None: documents.get(str(path), default))
    monkeypatch.setattr(control, 'boot_id', lambda: 'test-boot')
    monkeypatch.setattr(control, 'process_identity', lambda pid: None)
    monkeypatch.setattr(control, '_lease_is_held', lambda path: False)
    pending = session.status()
    assert pending['phase'] == 'closing' and pending['active']
    assert not pending['checkpoint_resume']
    documents[str(attempt / 'exit.json')] = {'identity': identity, 'returncode': 0}
    stopped = session.status()
    assert stopped['phase'] == 'stopped' and not stopped['active']
    assert stopped['checkpoint_resume']
    documents[str(attempt / 'exit.json')]['returncode'] = 1
    failed = session.status()
    assert failed['phase'] == 'failed' and not failed['checkpoint_resume']

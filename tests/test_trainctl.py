"""Real-process identity checks and serialized start/pause using CPU-only fixtures."""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools import trainctl


@pytest.fixture
def registration(tmp_path):
    root = tmp_path / 'study with spaces'
    source = tmp_path / 'frozen source'
    root.mkdir()
    (source / 'tools').mkdir(parents=True)
    (source / 'tools/run_estimator_study.py').write_text('# fixture\n')
    runtime = tmp_path / 'runtime.sh'
    runtime.write_text('# fixture\n')
    trainctl.save_json(root / 'study.json', {'source': str(source), 'stage_plan_hashes': {}})
    path = tmp_path / 'registry.json'
    value = trainctl.register(path, 'test', root, Path(sys.executable), runtime)
    return path, value


def test_registration_is_idempotent_and_does_not_start_processes(registration):
    path, value = registration
    assert trainctl.status(value)['status'] == 'prepared'
    again = trainctl.register(path, 'test', Path(value['root']), Path(value['python']), Path(value['runtime']))
    assert again == value
    assert not (Path(value['root']) / 'control/current.json').exists()
    with pytest.raises(ValueError, match='name'):
        trainctl.register(path, 'bad;name', Path(value['root']), Path(value['python']), Path(value['runtime']))


def test_runtime_values_are_arguments_not_interpolated_shell(registration):
    _, value = registration
    changed = {**value, 'runtime': '/tmp/a; touch SHOULD_NOT_RUN', 'source': '/tmp/source with spaces'}
    command = trainctl.launch_command(changed, Path(value['root']) / 'attempt')
    assert 'SHOULD_NOT_RUN' not in command[4]
    assert changed['runtime'] in command and changed['source'] in command


@pytest.fixture
def sleeper():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    deadline = time.monotonic() + 2
    while trainctl.process_identity(process.pid) is None and time.monotonic() < deadline:
        time.sleep(0.01)
    yield process
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=5)


def test_stale_pid_identity_never_signals_a_reused_process(sleeper):
    identity = trainctl.process_identity(sleeper.pid)
    altered = copy.deepcopy(identity)
    altered['start_ticks'] = str(int(identity['start_ticks']) + 1)
    assert not trainctl.signal_checked(altered)
    assert sleeper.poll() is None
    altered = copy.deepcopy(identity)
    altered['boot_id'] = 'different-boot'
    assert not trainctl.signal_checked(altered)
    assert sleeper.poll() is None


def test_sdk_python_without_pidfd_uses_system_python(monkeypatch, sleeper):
    if not Path('/usr/bin/python3').exists():
        pytest.skip('Linux system Python required')
    identity = trainctl.process_identity(sleeper.pid)
    monkeypatch.delattr(os, 'pidfd_open', raising=False)
    monkeypatch.setattr(sys, 'executable', '/different/sdk/python')
    assert trainctl.signal_checked(identity)
    assert sleeper.wait(timeout=5) != 0


def test_start_is_serialized_pause_is_idempotent_and_resume_uses_new_attempt(monkeypatch, registration):
    _, value = registration
    spawned = []
    fixture = Path(__file__).parent / 'fixtures/control_process.py'
    tools = Path(trainctl.__file__).parent

    def command(registration, attempt):
        return [sys.executable, str(fixture), registration['root'], str(attempt), str(tools)]

    original_run = subprocess.run

    def fake_tmux(arguments, **kwargs):
        if arguments[0] != 'tmux':
            return original_run(arguments, **kwargs)
        pointer = trainctl.read_json(Path(value['root']) / 'control/current.json')
        attempt = Path(pointer['attempt'])
        request = trainctl.read_json(attempt / 'request.json')
        with (attempt / 'control.log').open('wb') as log:
            child = subprocess.Popen(request['argv'], stdout=log, stderr=subprocess.STDOUT)
        spawned.append(child)
        return subprocess.CompletedProcess(arguments, 0, '', '')

    monkeypatch.setattr(trainctl, 'launch_command', command)
    monkeypatch.setattr(subprocess, 'run', fake_tmux)
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: trainctl.start(value, 'test'), range(2)))
        assert len(spawned) == 1
        assert results[0]['attempt'] == results[1]['attempt']
        assert trainctl.status(value)['active']
        stopped = trainctl.pause(value, 5)
        assert not stopped['active'] and stopped['status'] == 'stopped'
        assert trainctl.pause(value, 0)['status'] == 'stopped'
        restarted = trainctl.start(value, 'test')
        assert restarted['active'] and restarted['attempt'] != stopped['attempt']
        assert len(spawned) == 2
        trainctl.pause(value, 5)
    finally:
        for child in spawned:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_review_gate_is_not_bypassed_by_start(registration):
    _, value = registration
    trainctl.save_json(Path(value['root']) / 'status.json', {'status': 'needs_review'})
    assert trainctl.start(value, 'test')['status'] == 'needs_review'
    assert not (Path(value['root']) / 'control/current.json').exists()


def test_pause_intent_before_supervisor_boot_never_imports_training_source(registration):
    _, value = registration
    root = Path(value['root'])
    attempt = root / 'control/attempts/pre-cancelled'
    trainctl.save_json(attempt / 'request.json', {'registration': value, 'generation': 1})
    trainctl.save_json(attempt / 'pause-request.json', {'requested_at': trainctl.now()})
    trainctl.save_json(root / 'control/current.json', {'attempt': str(attempt), 'generation': 1})
    result = subprocess.run([sys.executable, str(Path(trainctl.__file__)), '_run', '--root', str(root),
                             '--attempt', str(attempt)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert trainctl.read_json(attempt / 'exit.json')['status'] == 'stopped'
    assert not (attempt / 'binding.json').exists()


def test_orphan_detection_and_pause_are_scoped_to_registered_root(tmp_path, registration):
    _, value = registration
    module = tmp_path / 'fixture_worker.py'
    module.write_text('import time\ntime.sleep(30)\n')
    environment = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    ours = subprocess.Popen([sys.executable, '-m', 'fixture_worker', 'train', '--run-dir', value['root'] + '/train'], env=environment)
    other = subprocess.Popen([sys.executable, '-m', 'fixture_worker', 'train', '--run-dir', str(tmp_path / 'other/train')], env=environment)
    try:
        time.sleep(0.1)
        state = trainctl.status(value)
        assert state['status'] == 'orphaned_worker'
        assert [p['pid'] for p in state['orphaned_workers']] == [ours.pid]
        trainctl.pause(value, 5)
        assert ours.wait(timeout=5) != 0
        assert other.poll() is None
    finally:
        for process in (ours, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

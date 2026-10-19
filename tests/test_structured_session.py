"""Detached structured child/control integration in private CPU namespaces."""
from dataclasses import replace
import fcntl
import inspect
import os
from pathlib import Path
import shutil
import sys
import time

import pytest
import torch

from test_structured_collection import SyntheticLabeledEnv
from test_structured_learning import components
from transformer_rl.formal.runtime.session import StructuredTrainingSession
from transformer_rl.common.session_router import open_session


@pytest.fixture(autouse=True)
def seeded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


def session(tmp_path, monkeypatch, *, ready=True, implicit_source=False):
    import transformer_rl
    package = tmp_path / 'learner/transformer_rl'
    shutil.copytree(Path(transformer_rl.__file__).parent, package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    fixture = 'from contextlib import contextmanager\nimport torch\nfrom transformer_rl.formal.learning.collection import LabeledObservation, LabeledStep, PhysicalLabels\nfrom transformer_rl.common.training_control import read_json, write_json\n'
    fixture += inspect.getsource(SyntheticLabeledEnv)
    fixture += '''
@contextmanager
def runtime(recipe, components, configuration, *, control):
    env = SyntheticLabeledEnv()
    original = env.step
    def step(*args):
        value = original(*args)
        if env.counter == recipe.rollout_steps:
            request = control._request()
            write_json(control.directory / 'request.json', dict(request,
                operation='stop', sequence=request['sequence'] + 1), replace=True)
        return value
    env.step = step
    yield env
'''
    (package / '_owned_test_runtime.py').write_text(fixture)
    recipe, _ = components()
    recipe = replace(recipe, learner_source_sha256=StructuredTrainingSession.source_sha256(package))
    driver = tmp_path / 'driver'
    driver.mkdir()
    (driver / 'libcuda.so').write_bytes(b'')
    locks = []
    for index in range(4):
        path = tmp_path / f'resource_{index}.lock'
        path.touch()
        info = path.stat()
        locks.append({'path': str(path), 'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid})
    resources = {'gpu_uuid': 'GPU-' + 'a' * 36, 'nvidia_smi': sys.executable,
                 'driver_library_directory': str(driver), 'minimum_ram_bytes': 7 * 1024**3,
                 'minimum_vram_mib': 14336, 'minimum_disk_bytes': 10 * 1024**3, 'locks': locks}
    source_options = {'learner_source': package}
    if implicit_source:
        from transformer_rl.formal.runtime import session as session_module
        monkeypatch.setattr(session_module, '__file__', str(package / 'formal/runtime/session.py'))
        source_options = {}
    value = StructuredTrainingSession.create(tmp_path / 'session', recipe,
        runtime_factory='transformer_rl._owned_test_runtime:runtime', runtime_configuration={},
        resources=resources, python=sys.executable, **source_options)
    monkeypatch.setattr(value, '_resources', lambda: (ready, {
        'MemAvailable': 7 * 1024**3, 'minimum_ram_available_bytes': 7 * 1024**3,
        'gpu_free_mib': 14336, 'minimum_gpu_free_mib': 14336}))
    return value, resources


def wait(session, predicate, seconds=25.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = session.status()
        if predicate(state):
            return state
        time.sleep(.05)
    raise AssertionError('worker did not reach the expected owned boundary: ' + repr(session.status()))


@pytest.mark.parametrize('implicit_source', [False, True])
def test_structured_session_has_real_parent_receipts_and_cold_resume(tmp_path, monkeypatch, implicit_source):
    value, _ = session(tmp_path, monkeypatch, implicit_source=implicit_source)
    assert (value.root / 'source/transformer_rl/formal/runtime/process.py').is_file()
    assert (value.root / 'source/transformer_rl/common/training_control.py').is_file()
    assert isinstance(open_session(value.root), StructuredTrainingSession)
    assert not value.spec['automatic_evaluation']
    value.start(paused=True)
    paused = wait(value, lambda s: s['phase'] == 'paused')
    assert paused['active'] and paused['update'] == 0
    assert Path(paused['checkpoint']).is_file()
    value.command('resume')
    final = wait(value, lambda s: not s['active'])
    assert final['phase'] == 'stopped' and final['update'] == 1
    assert final['cumulative_transitions'] == 32
    first = value.attempt()
    assert (first / 'exit.json').is_file() and value.submitted_process.wait(timeout=5) == 0
    observed = value._resources()[1]
    monkeypatch.setattr(value, '_resources', lambda: (False, observed))
    waiting = value.start()
    assert not waiting['active'] and waiting['phase'] == 'waiting_resources' and waiting['update'] == 1
    assert value.attempt() == first and value.status()['phase'] == 'waiting_resources'
    monkeypatch.setattr(value, '_resources', lambda: (True, observed))
    value.start()
    resumed = wait(value, lambda s: not s['active'])
    assert resumed['phase'] == 'stopped' and resumed['update'] == 2
    assert resumed['cumulative_transitions'] == 64
    assert value.attempt() != first and value.submitted_process.wait(timeout=5) == 0


def test_resource_wait_creates_no_attempt_and_releases_original_leases(tmp_path, monkeypatch):
    from transformer_rl.common.training_cli import render
    value, resources = session(tmp_path, monkeypatch, ready=False)
    state = value.start()
    assert state['phase'] == 'waiting_resources' and not state['active']
    assert value.attempt() is None
    assert not (value.root / 'journal').exists()
    dashboard = render(state)
    assert '12 / 6' in dashboard and '尚未创建训练 attempt' in dashboard
    assert '达到原任务资源门槛后自动继续' not in dashboard
    for item in resources['locks']:
        with open(item['path'], 'r+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_structured_admission_cannot_lower_original_thresholds(tmp_path, monkeypatch):
    _, resources = session(tmp_path, monkeypatch)
    resources['minimum_ram_bytes'] -= 1
    with pytest.raises(ValueError, match='admission'):
        StructuredTrainingSession._resource_definition(resources)

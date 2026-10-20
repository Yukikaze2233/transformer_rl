"""Native file delivery must reject bad intervals without SDK thread callbacks."""
from pathlib import Path
import os

import pytest

from transformer_rl.formal.simulation.health import NativePhysicsHealth


def test_new_log_ranges_and_split_error_are_consumed_without_a_callback(tmp_path):
    path = tmp_path / 'native.log'
    path.write_bytes(b'[python] ordinary initialization\n')
    health = NativePhysicsHealth(log_path=path)
    try:
        health.check()
        health.check()
        assert health.messages_seen == 1
        with path.open('ab') as stream:
            stream.write(b'[omni.physx.plugin] PhysX er')
        health.check()
        with path.open('ab') as stream:
            stream.write(b'ror: capacity exhausted; simulation will miss interactions\n')
        with pytest.raises(RuntimeError, match='capacity exhausted'):
            health.check()
        assert health.messages_seen == 2
        assert health._logging is None and health._handle is None
        with pytest.raises(RuntimeError, match='capacity exhausted'):
            health.check()
    finally:
        health.close()
        health.close()


def test_unrelated_channel_cannot_masquerade_as_native_physics(tmp_path):
    path = tmp_path / 'native.log'
    path.write_text('[another.plugin] quoted PhysX error: unrelated text\n')
    health = NativePhysicsHealth(log_path=path)
    try:
        health.check()
        assert health.first_error is None
    finally:
        health.close()


@pytest.mark.parametrize('change', ('replace', 'truncate'))
def test_lost_log_history_refuses_qualification(tmp_path, change):
    path = tmp_path / 'native.log'
    path.write_bytes(b'[python] ordinary initialization\n')
    health = NativePhysicsHealth(log_path=path)
    try:
        health.check()
        if change == 'replace':
            path.rename(tmp_path / 'old.log')
        path.write_bytes(b'')
        with pytest.raises(RuntimeError, match='replaced or truncated'):
            health.check()
    finally:
        health.close()


def test_large_log_burst_is_not_silently_skipped(tmp_path):
    path = tmp_path / 'native.log'
    path.write_bytes(b'x' * (8 * 1024 * 1024 + 1))
    health = NativePhysicsHealth(log_path=path)
    try:
        with pytest.raises(RuntimeError, match='read limit'):
            health.check()
    finally:
        health.close()


@pytest.mark.parametrize('mode', ('separate', 'shared', 'foreign'))
def test_native_factory_binds_only_its_inherited_streams_and_probes_delivery(tmp_path, monkeypatch, mode):
    directory = tmp_path / 'run'
    directory.mkdir()
    paths = {number: directory / name for number, name in ((1, 'stdout'), (2, 'stderr'))}
    if mode == 'shared':
        paths[1] = paths[2]
    elif mode == 'foreign':
        paths[1] = tmp_path / 'outside_run.log'
    descriptors = {}
    writers = {}
    real_fstat, real_write = os.fstat, os.write
    for number, path in paths.items():
        if path not in writers:
            writers[path] = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        descriptors[number] = writers[path]
    monkeypatch.setattr(os, 'readlink', lambda name: str(paths[int(Path(name).name)]))
    monkeypatch.setattr(os, 'fstat', lambda number: real_fstat(descriptors.get(number, number)))
    monkeypatch.setattr(os, 'write', lambda number, raw: real_write(descriptors.get(number, number), raw))
    health = None
    try:
        if mode == 'foreign':
            with pytest.raises(ValueError, match='inside its run directory'):
                NativePhysicsHealth.for_native_process(directory)
            return
        health = NativePhysicsHealth.for_native_process(directory)
        assert health._probe_seen and health.messages_seen == 1
        real_write(descriptors[1], b'[omni.physx.plugin] PhysX error: lost contacts\n')
        with pytest.raises(RuntimeError, match='lost contacts'):
            health.check()
    finally:
        if health is not None:
            health.close()
        for descriptor in set(descriptors.values()):
            os.close(descriptor)

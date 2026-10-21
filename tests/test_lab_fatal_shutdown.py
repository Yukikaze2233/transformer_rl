"""A primary learning failure cannot enter native teardown or a GA finalizer."""
from contextlib import contextmanager
import gc
import weakref
from types import SimpleNamespace

import pytest

from test_lab_sdk import _modules
from transformer_rl.formal.lab_env import sdk
from transformer_rl.formal.lab_env.runtime import _close_runtime_resources


class _Launcher:
    def __init__(self, events):
        self.events = events
        self.retained = None

    def close(self):
        self.events.append('launcher.close')

    def defer_fatal_shutdown(self, environment):
        if self.retained is None:
            self.events.append('defer')
            self.retained = (self, environment)


@pytest.mark.parametrize('health_error', [None, RuntimeError('secondary health close error')])
def test_primary_error_skips_env_and_launcher_close_without_catching_native_exit0(health_error):
    events = []
    primary = RuntimeError('original terrain query failure')

    def native_close():
        events.append('env.close')
        raise SystemExit(0)  # Real C++ exit0 would never return to the Python handler.

    def health_close():
        events.append('health.close')
        if health_error is not None:
            raise health_error

    environment = SimpleNamespace(close=native_close)
    health = SimpleNamespace(close=health_close)
    launcher = _Launcher(events)
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            assert _close_runtime_resources(environment, health, launcher, primary) is None
    assert caught.value is primary
    assert events == ['defer', 'health.close']
    assert launcher.retained == (launcher, environment)
    if health_error is not None:
        assert any('secondary health close error' in note for note in primary.__notes__)


def test_first_cleanup_error_defers_the_remaining_native_owner():
    events = []
    failure = RuntimeError('native environment close raised')

    def env_close():
        events.append('env.close')
        raise failure

    environment = SimpleNamespace(close=env_close)
    health = SimpleNamespace(close=lambda: events.append('health.close'))
    launcher = _Launcher(events)
    assert _close_runtime_resources(environment, health, launcher, None) is failure
    assert events == ['env.close', 'defer', 'health.close']
    assert launcher.retained == (launcher, environment)


def test_success_closes_normally_without_retaining_native_resources():
    events = []
    environment = SimpleNamespace(close=lambda: events.append('env.close'))
    health = SimpleNamespace(close=lambda: events.append('health.close'))
    launcher = _Launcher(events)
    assert _close_runtime_resources(environment, health, launcher, None) is None
    assert events == ['env.close', 'health.close', 'launcher.close']
    assert launcher.retained is None


def test_fatal_strong_references_prevent_ga_generator_and_env_gc_during_unwinding(monkeypatch):
    modules = _modules(monkeypatch)
    events = []
    modules['isaaclab_physx.physics'].PhysxCfg = object

    @contextmanager
    def launch(*_args):
        try:
            yield object()
        finally:
            # GA's real launch_simulation finally calls launcher.close(0) on
            # GeneratorExit, since that BaseException misses except Exception.
            events.append('generator.close.exit0')

    modules['isaaclab.app'].launch_simulation = launch
    retained = []
    monkeypatch.setattr(sdk, '_FATAL_NATIVE_REFERENCES', retained)

    class Environment:
        def __del__(self):
            events.append('environment.finalized')

    simulation = sdk.LabSimulation('cuda:0')
    environment = Environment()
    simulation_ref, environment_ref = weakref.ref(simulation), weakref.ref(environment)
    simulation.defer_fatal_shutdown(environment)
    simulation.defer_fatal_shutdown(environment)
    assert len(retained) == 1
    del simulation, environment
    gc.collect()
    assert simulation_ref() is not None and environment_ref() is not None
    assert not events
    # Release only the synthetic owners for test hygiene. The real fatal
    # worker exits via os._exit(1), never through this native close path.
    retained[0][0].close()
    assert events == ['generator.close.exit0']
    retained.clear()
    gc.collect()
    assert simulation_ref() is None and environment_ref() is None
    assert events == ['generator.close.exit0', 'environment.finalized']

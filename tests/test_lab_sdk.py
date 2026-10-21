"""新版启动生命周期与设置接口；执行验证仅在 Kaiser。"""
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from transformer_rl.formal.lab_env.sdk import LabSimulation


def _modules(monkeypatch):
    names = ('isaaclab', 'isaaclab.app', 'isaaclab.app.settings_manager',
             'isaaclab_physx', 'isaaclab_physx.physics')
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    modules['isaaclab'].app = modules['isaaclab.app']
    modules['isaaclab.app'].settings_manager = modules['isaaclab.app.settings_manager']
    return modules


def test_ga_uses_explicit_physx_context_and_current_settings(monkeypatch):
    modules = _modules(monkeypatch)
    events, settings = [], {}
    physics = object()
    modules['isaaclab_physx.physics'].PhysxCfg = lambda: physics

    @contextmanager
    def launch(cfg, args):
        assert cfg is physics
        assert args == dict(headless=True, device='cuda:0', enable_cameras=False, livestream=0)
        assert len(sys.argv) == 1
        events.append('enter')
        try:
            yield cfg
        finally:
            events.append('exit')

    modules['isaaclab.app'].launch_simulation = launch
    modules['isaaclab.app.settings_manager'].get_settings_manager = lambda: SimpleNamespace(set=settings.__setitem__)
    original = sys.argv
    simulation = LabSimulation('cuda:0')
    assert sys.argv is original and simulation.declarative_scene
    assert simulation.physics is physics and events == ['enter']
    simulation.configure_settings()
    assert settings == {'/physics/cudaDevice': 0, '/physics/suppressReadback': True}
    simulation.close()
    simulation.close()
    assert events == ['enter', 'exit']


def test_gui_uses_kit_visualizer_and_standard_launcher(monkeypatch):
    modules = _modules(monkeypatch)
    received = []

    def launch(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(app=SimpleNamespace(close=lambda: None))

    modules['isaaclab.app'].AppLauncher = launch
    simulation = LabSimulation('cuda:0', gui=True)
    assert received == [dict(headless=False, device='cuda:0', enable_cameras=True,
                             livestream=0, visualizer='kit')]
    assert not simulation.declarative_scene
    simulation.close()


def test_beta_keeps_existing_launcher_and_settings(monkeypatch):
    modules = _modules(monkeypatch)
    events, settings = [], {}
    modules['isaaclab.app'].AppLauncher = lambda **kwargs: SimpleNamespace(
        app=SimpleNamespace(close=lambda: events.append('exit')))
    manager = SimpleNamespace(initialize_carb_settings=lambda: events.append('settings'),
                              set_int=settings.__setitem__, set_bool=settings.__setitem__)
    modules['isaaclab.app.settings_manager'].SettingsManager = SimpleNamespace(instance=lambda: manager)
    simulation = LabSimulation('cuda:0')
    assert not simulation.declarative_scene and simulation.physics is None
    simulation.configure_settings()
    simulation.close()
    assert events == ['settings', 'exit']
    assert settings == {'/physics/cudaDevice': 0, '/physics/suppressReadback': True}


def test_startup_failure_restores_callers_argv(monkeypatch):
    modules = _modules(monkeypatch)

    def fail(**kwargs):
        assert len(sys.argv) == 1
        raise RuntimeError('startup refused')

    modules['isaaclab.app'].AppLauncher = fail
    original = sys.argv
    with pytest.raises(RuntimeError, match='startup refused'):
        LabSimulation('cuda:0')
    assert sys.argv is original


@pytest.mark.parametrize('device,expected', [('cuda:1', 1), ('cpu', None)])
def test_settings_do_not_silently_retarget_a_device(monkeypatch, device, expected):
    modules = _modules(monkeypatch)
    settings = {}
    modules['isaaclab.app'].AppLauncher = lambda **_: SimpleNamespace(app=SimpleNamespace(close=lambda: None))
    modules['isaaclab.app.settings_manager'].SettingsManager = SimpleNamespace(
        instance=lambda: SimpleNamespace(initialize_carb_settings=lambda: None,
                                        set_int=settings.__setitem__, set_bool=settings.__setitem__))
    simulation = LabSimulation(device)
    simulation.configure_settings()
    assert settings == ({} if expected is None else {
        '/physics/cudaDevice': expected, '/physics/suppressReadback': True})
    simulation.close()

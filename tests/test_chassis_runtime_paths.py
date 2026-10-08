"""CPU SDK-interface mocks around real worker paths and application ownership.

These checks exercise the launcher contract, not an actual Isaac startup,
GPU execution, measured cache growth, or a system-wide filesystem quota.
"""
from copy import deepcopy
import hashlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from transformer_rl import chassis_adapter, frame_process, runtime_paths
from transformer_rl.continuation_process import _ApplicationRegistry


@pytest.fixture
def sdk_interface(tmp_path, monkeypatch):
    experience = tmp_path / "sdk" / "apps" / "interface_fixture.kit"
    experience.parent.mkdir(parents=True)
    experience.write_bytes(b'[settings]\nfixture = "CPU interface only"\n')
    experience.chmod(0o444)
    state = SimpleNamespace(experience=experience, events=[], launchers=[],
                            tokens={}, settings={})
    monkeypatch.setattr(frame_process, "_apps", [])
    monkeypatch.setattr(frame_process, "_active", False)
    registry = _ApplicationRegistry()
    registry.activate()
    state.registry = registry
    register = frame_process.register_app

    def register_app(app):
        register(app)
        state.events.append(("register", app))

    monkeypatch.setattr(frame_process, "register_app", register_app)

    class Application:
        def __init__(self):
            self.close_calls = []

        def close(self, *, wait_for_replicator, exit_code):
            self.close_calls.append((wait_for_replicator, exit_code))
            state.events.append(("close", self))

    class AppLauncher:
        def __init__(self, options):
            self.options = deepcopy(options)
            self.app = Application()
            self._sim_experience_file = str(experience)
            state.launchers.append(self)
            state.events.append(("construct", self))

    def registered_readback(event, value):
        # Exercise the existing registry rather than replacing its ownership.
        assert state.launchers[-1].app in frame_process._apps
        state.events.append((event, value))

    def resolve(query):
        registered_readback("resolve", query)
        assert query.startswith("${") and query.endswith("}")
        return state.tokens.get(query[2:-1], query)

    def get_setting(key):
        registered_readback("setting", key)
        return state.settings.get(key)

    def get_tokens_interface():
        registered_readback("tokens_provider", None)
        return SimpleNamespace(resolve=resolve)

    def get_settings():
        registered_readback("settings_provider", None)
        return SimpleNamespace(get=get_setting)

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = AppLauncher
    isaaclab.app = app_module
    carb = ModuleType("carb")
    carb.__path__ = []
    tokens_module = ModuleType("carb.tokens")
    tokens_module.get_tokens_interface = get_tokens_interface
    settings_module = ModuleType("carb.settings")
    settings_module.get_settings = get_settings
    carb.tokens, carb.settings = tokens_module, settings_module
    for name, module in (("isaaclab", isaaclab), ("isaaclab.app", app_module),
                         ("carb", carb), ("carb.tokens", tokens_module),
                         ("carb.settings", settings_module)):
        monkeypatch.setitem(sys.modules, name, module)
    try:
        yield state
    finally:
        assert registry.close(99) == []


@pytest.fixture
def owned_runtime(sdk_interface, tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    worker.mkdir(mode=0o700)
    environment = runtime_paths.prepare_worker_runtime(worker, dict(os.environ))
    profile = runtime_paths.validate_runtime_profile(worker, environment)
    for key in (*profile["environment"], runtime_paths.PROFILE_ENV,
                runtime_paths.PROFILE_SHA_ENV):
        monkeypatch.setenv(key, environment[key])
    sdk_interface.tokens = deepcopy(profile["tokens"])
    sdk_interface.settings = deepcopy(profile["settings"])
    sdk_interface.worker, sdk_interface.profile = worker, profile
    return sdk_interface


def test_owned_arguments_precede_constructor_and_actual_readback(owned_runtime):
    state = owned_runtime
    launcher, readback = chassis_adapter._launch_application("cpu")
    options, profile = launcher.options, state.profile
    assert {key: options[key] for key in ("headless", "enable_cameras", "device")} == {
        "headless": True, "enable_cameras": False, "device": "cpu"}
    arguments = options["kit_args"].split()
    assert arguments[:2] == ["--portable-root", profile["paths"]["kit"]]
    for key, value in profile["tokens"].items():
        assert f"--/app/tokens/{key}={value}" in arguments
    for key, value in profile["settings"].items():
        encoded = str(value).lower() if type(value) is bool else value
        assert f"--{key}={encoded}" in arguments
    assert state.events[:2] == [("construct", launcher), ("register", launcher.app)]
    assert [value for event, value in state.events if event == "resolve"] == [
        "${" + key + "}" for key in profile["tokens"]]
    assert [value for event, value in state.events if event == "setting"] == list(profile["settings"])
    assert [event for event, _ in state.events].count("tokens_provider") == 1
    assert [event for event, _ in state.events].count("settings_provider") == 1
    assert readback["tokens"] == profile["tokens"]
    assert readback["settings"] == profile["settings"]
    assert readback["profile"] == runtime_paths.profile_receipt(profile)
    raw = state.experience.read_bytes()
    assert readback["experience"] == {
        "path": str(state.experience), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert readback["hardware_verified"] is False
    assert state.registry.close(0) == []
    assert launcher.app.close_calls == [(False, 0)]
    assert frame_process._apps == [] and frame_process._active is False


@pytest.mark.parametrize("key", [runtime_paths.PROFILE_ENV, runtime_paths.PROFILE_SHA_ENV,
                                 "TMPDIR", "XDG_CACHE_HOME", "WARP_CACHE_PATH"])
def test_missing_inherited_binding_rejects_before_constructor(owned_runtime, monkeypatch, key):
    state = owned_runtime
    monkeypatch.delenv(key)
    with pytest.raises(ValueError):
        chassis_adapter._launch_application("cpu")
    assert state.launchers == [] and state.events == []
    assert frame_process._apps == []


@pytest.mark.parametrize("name", ["cache", "data", "omni_global_data"])
def test_wrong_actual_token_remains_owned_for_failure_shutdown(owned_runtime, name):
    state = owned_runtime
    state.tokens[name] = str(state.experience.parent)
    with pytest.raises(ValueError, match="active SDK output escapes"):
        chassis_adapter._launch_application("cpu")
    launcher = state.launchers[0]
    assert state.events[:2] == [("construct", launcher), ("register", launcher.app)]
    assert frame_process._apps == [launcher.app]
    assert state.registry.close(30) == []
    assert launcher.app.close_calls == [(False, 30)]
    assert frame_process._apps == [] and frame_process._active is False


@pytest.mark.parametrize("key,value", [
    ("/app/portableMode", False), ("/app/extensions/registryEnabled", True),
    ("/app/userConfigPath", "outside"),
])
def test_wrong_actual_setting_is_rejected_after_registration(owned_runtime, key, value):
    state = owned_runtime
    state.settings[key] = str(state.experience) if value == "outside" else value
    with pytest.raises(ValueError):
        chassis_adapter._launch_application("cpu")
    launcher = state.launchers[0]
    assert frame_process._apps == [launcher.app]
    assert state.registry.close(30) == []
    assert launcher.app.close_calls == [(False, 30)]


@pytest.mark.parametrize("provider,key", [
    ("tokens", "cache"), ("settings", "/app/portableMode"),
])
def test_missing_required_readback_is_rejected_and_application_closes(owned_runtime, provider, key):
    state = owned_runtime
    del getattr(state, provider)[key]
    with pytest.raises(ValueError, match="required Kit runtime"):
        chassis_adapter._launch_application("cpu")
    launcher = state.launchers[0]
    assert frame_process._apps == [launcher.app]
    assert state.registry.close(30) == []
    assert launcher.app.close_calls == [(False, 30)]


def test_legacy_profile_absent_keeps_telemetry_argument_and_ownership(sdk_interface, monkeypatch):
    state = sdk_interface
    monkeypatch.delenv(runtime_paths.PROFILE_ENV, raising=False)
    monkeypatch.delenv(runtime_paths.PROFILE_SHA_ENV, raising=False)
    launcher, readback = chassis_adapter._launch_application("cpu")
    assert launcher.options["kit_args"] == "--/exts/omni.kit.telemetry/skipDeferredStartup=true"
    assert readback is None
    assert state.events == [("construct", launcher), ("register", launcher.app)]
    assert state.registry.close(0) == []
    assert launcher.app.close_calls == [(False, 0)]

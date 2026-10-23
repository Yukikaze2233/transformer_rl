"""Formal entry validation without constructing Isaac or a robot."""
from dataclasses import replace
from pathlib import Path
import json

import pytest

from transformer_rl.formal.lab_env.action_transport import ActionTransportConfig
from transformer_rl.formal.lab_env.entry import _delay_steps, _validate_plan, _with_transport, _with_control_timing
from transformer_rl.formal.learning.settings import FormalTrainingSettings


SETTINGS = Path(__file__).resolve().parents[1] / 'configs' / 'formal_training.json'


def test_delay_steps_are_whole_step_unique_and_bounded():
    assert _delay_steps('0,1,2') == (0, 1, 2)
    with pytest.raises(Exception):
        _delay_steps('1,1')
    with pytest.raises(Exception):
        _delay_steps('-1')
    with pytest.raises(Exception):
        _delay_steps('17')


def test_transport_override_is_explicit_and_keeps_other_runtime_fields():
    runtime = {'format': 'managed_lab_runtime_v4', 'sensors': {'x': 1}}
    result = _with_transport(runtime, type('Args', (), {
        'no_action_transport': False, 'action_delay_steps': (0, 1),
        'action_delay_nominal_fraction': .5})())
    assert result['sensors'] == {'x': 1}
    assert result['action_transport'] == {'enabled': True, 'delays_steps': [0, 1],
                                          'nominal_fraction': .5}
    ActionTransportConfig.from_dict(result['action_transport'])
    original = {'action_transport': {'enabled': True, 'delays_steps': [2], 'nominal_fraction': 0.}}
    untouched = _with_transport(original, type('Args', (), {'no_action_transport': True,
        'action_delay_steps': (0,), 'action_delay_nominal_fraction': 0.})())
    assert untouched == original


def test_formal_plan_rejects_non_35k_or_unlisted_seed():
    settings = FormalTrainingSettings.load(SETTINGS)
    _validate_plan(settings, 2102)
    with pytest.raises(ValueError):
        _validate_plan(replace(settings, updates_per_seed=34999), 2102)
    with pytest.raises(ValueError):
        _validate_plan(settings, 9999)
    with pytest.raises(ValueError):
        _validate_plan(replace(settings, num_envs=2048), 2102)
    with pytest.raises(ValueError):
        _validate_plan(replace(settings, training_seeds=(9999,)), 9999)


def test_timing_preset_is_explicit_complete_and_does_not_mutate_input(tmp_path):
    path = SETTINGS.parent / 'control_timing.json'
    runtime = {'format': 'managed_lab_runtime_v4', 'sensors': {'example': True}}
    assert _with_control_timing(runtime, None) == runtime
    value = _with_control_timing(runtime, path)
    assert value['communication']['total_delay_range_s'] == [.0001, .005]
    assert value['episode_randomization']['full_at_update'] == 5000
    assert 'communication' not in runtime and runtime['sensors'] == {'example': True}
    bad = tmp_path / 'partial.json'
    bad.write_text(json.dumps({'communication': {'enabled': True}}))
    with pytest.raises(ValueError, match='preset'):
        _with_control_timing(runtime, bad)


def test_entry_binds_explicit_worker_interpreter_and_run_id(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from transformer_rl.formal.lab_env import entry

    runtime_path, resources_path = tmp_path / 'runtime.json', tmp_path / 'resources.json'
    worker = '/validated/isaaclab/python'
    calls = []
    monkeypatch.setattr(entry, 'read_json', lambda path: (
        {'asset_directory': '/frozen/robot'} if path == runtime_path else {}))
    monkeypatch.setattr(entry, '_asset_report', lambda _: {'scope': 'synthetic'})
    monkeypatch.setattr(entry, '_curriculum_report', lambda *args, **kwargs: {})

    def prepare(root, *args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(root=root, spec={'updates': 35000, 'recipe_sha256': 'a' * 64})

    monkeypatch.setattr(entry, 'prepare_session', prepare)
    assert entry.main([
        '--settings', str(SETTINGS), '--runtime-json', str(runtime_path),
        '--resources-json', str(resources_path), '--output', str(tmp_path / 'fresh'),
        '--python', worker, '--run-id', 'formal-2102',
    ]) == 0
    assert calls[0]['python'] == worker and calls[0]['run_id'] == 'formal-2102'
    assert json.loads(capsys.readouterr().out)['training_started'] is False


def test_start_resource_wait_is_not_reported_as_a_running_worker(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from transformer_rl.formal.lab_env import entry

    monkeypatch.setattr(entry, 'read_json', lambda _: {'asset_directory': '/frozen/robot'})
    monkeypatch.setattr(entry, '_asset_report', lambda _: {})
    monkeypatch.setattr(entry, '_curriculum_report', lambda *args, **kwargs: {})
    session = SimpleNamespace(
        root=tmp_path / 'fresh', spec={'updates': 35000, 'recipe_sha256': 'a' * 64},
        start=lambda **kwargs: {'phase': 'waiting_resources', 'active': False}, attempt=lambda: None)
    monkeypatch.setattr(entry, 'prepare_session', lambda *args, **kwargs: session)
    assert entry.main([
        '--settings', str(SETTINGS), '--runtime-json', '/runtime.json',
        '--resources-json', '/resources.json', '--output', str(session.root), '--start',
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['start_requested'] is True and result['training_started'] is False
    assert result['state']['phase'] == 'waiting_resources'

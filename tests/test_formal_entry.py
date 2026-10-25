"""Formal entry validation without constructing Isaac or a robot."""
from dataclasses import replace
from pathlib import Path
import json

import pytest

import torch

from transformer_rl.formal.lab_env.entry import _validate_plan, _with_control_timing
from transformer_rl.formal.learning.settings import FormalTrainingSettings


SETTINGS = Path(__file__).resolve().parents[1] / 'configs' / 'formal_training.json'
# A reviewed runtime always carries its termination limits; presets override fields.
LIMITS = {'minimum_normal_height_m': .08, 'minimum_recovery_height_m': .01, 'healthy_tilt_rad': .5,
          'failure_tilt_rad': 2.5, 'forbidden_body_names': ['base_link'], 'forbidden_body_load_n': 2.,
          'forbidden_contact_confirm_s': .02, 'passive_knee_overrun_rad': .02,
          'knee_pivot_residual_m': .01, 'preparation_timeout_s': 5.}


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
    from dataclasses import asdict
    from transformer_rl.formal.control.sensors import SensorModelConfig

    path = SETTINGS.parent / 'control_timing.json'
    sensors = asdict(SensorModelConfig(policy_dt_s=.02))
    runtime = {'format': 'managed_lab_runtime_v4', 'sensors': dict(sensors), 'termination_limits': dict(LIMITS)}
    assert _with_control_timing(runtime, None) == runtime
    value = _with_control_timing(runtime, path)
    assert value['communication']['total_delay_range_s'] == [.0001, .002]
    assert value['episode_randomization']['full_at_update'] == 5000
    # SCUT: one shared 5-15 ms observation lag in every episode; runtime clocks are kept.
    assert value['sensors']['shared_delay'] is True and value['sensors']['policy_dt_s'] == .02
    assert value['action_latch']['delay_substeps'] == [1, 2]
    assert value['birth_randomization'] == {'enabled': True}
    merged = value['termination_limits']
    # SCUT: 40 deg / 1 N, both persisting 0.4 s; other reviewed limits are kept.
    assert merged['failure_tilt_rad'] == pytest.approx(.6981317) and merged['forbidden_body_load_n'] == 1.
    assert merged['forbidden_contact_confirm_s'] == .4 and merged['pose_failure_confirm_s'] == .4
    assert merged['knee_pivot_residual_m'] == .01
    assert 'communication' not in runtime and runtime['sensors'] == sensors
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


def test_preset_freezes_one_target_delay_model_and_a_bounded_delay_contract():
    from dataclasses import asdict
    from transformer_rl.formal.control.actuation import ResponseParameters
    from transformer_rl.formal.control.sensors import SensorModelConfig
    from transformer_rl.formal.lab_env.entry import _single_target_delay
    from transformer_rl.formal.lab_env.runtime import control_delay_contract

    legacy = {'sensors': asdict(SensorModelConfig(policy_dt_s=.02)), 'termination_limits': dict(LIMITS),
              'action_transport': {'enabled': True, 'delays_steps': [0, 1], 'nominal_fraction': .5}}
    with pytest.raises(ValueError, match='retired'):
        _single_target_delay(legacy)
    runtime = _single_target_delay(_with_control_timing(legacy, SETTINGS.parent / 'control_timing.json'))
    assert runtime['action_transport']['enabled'] is False and runtime['action_latch']['enabled'] is True
    response = ResponseParameters((0.,) * 6, (0.,) * 6, (0.,) * 6,
                                  (.00386, .00386, .00441, .00441, .0, .0))  # policy order
    contract = control_delay_contract(runtime, response)
    # SCUT: 5-15 ms observation lag in every episode, 5-10 ms target delay.
    assert contract['observation_age_ms'] == [5.1, 17.0]
    assert contract['target_latch_ms'] == [5.0, 10.0]
    assert contract['torque_downlink_ms'] == [0.0, 0.0]
    assert contract['whole_step_transport'] is False
    torch.testing.assert_close(torch.tensor(contract['leg_drive_time_constant_ms']), torch.tensor([3.86, 11.025]))


def test_curriculum_report_covers_recovery_through_ground_births():
    import math
    from transformer_rl.formal.lab_env import entry
    from transformer_rl.formal.lab_env.birth import BirthRandomizationConfig

    settings = FormalTrainingSettings.load(SETTINGS)
    with pytest.raises(ValueError, match='recovery_coverage'):
        entry._curriculum_report(settings, seed=2102, sample_count=2000)
    report = entry._curriculum_report(settings, seed=2102, births=BirthRandomizationConfig(enabled=True),
                                      sample_count=2000)
    assert report['final']['recovery_source'] == 'ground_births'
    assert report['final']['declared_yaw_abs_max'] == pytest.approx(4.5 * math.pi)
    assert all(report['final']['checks'].values())

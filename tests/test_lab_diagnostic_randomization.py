"""Direct diagnostics must retain the explicit formal noise and dynamics draw."""
import copy
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.sensors import SensorModelConfig
from transformer_rl.formal.lab_env import train
from transformer_rl.formal.lab_env.train import _runtime_randomization
from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig


def configurations():
    # Nondefault values detect accidentally constructing the dataclass defaults.
    sensors = SensorModelConfig(nominal_episode_fraction=.31, extra_delay_probability=.73,
                               observation_delays_s=(.005, .015), gyro_halfwidth_rad_s=.12)
    dynamics = StartupDynamicsConfig(nominal_fraction=.37, base_mass_scale=(.85, 1.4),
                                     link_mass_scale=(.95, 1.05), base_com_offset_m=(.03, .01, .02))
    raw = json.loads(json.dumps({'sensors': asdict(sensors), 'dynamics': asdict(dynamics)}))
    return raw, sensors, dynamics


@pytest.mark.parametrize('layout', ('managed', 'native', 'matching_both'))
def test_runtime_values_survive_json_and_do_not_consume_random_draws(layout):
    raw, sensors, dynamics = configurations()
    runtime = ({'tuning': raw} if layout == 'native' else raw)
    if layout == 'matching_both':
        runtime = {**raw, 'tuning': copy.deepcopy(raw)}
    before = copy.deepcopy(runtime)
    rng = torch.get_rng_state().clone()
    actual_sensors, actual_dynamics = _runtime_randomization(runtime, required=True)
    assert json.loads(json.dumps(asdict(actual_sensors))) == raw['sensors']
    assert tuple(actual_sensors.observation_delays_s) == sensors.observation_delays_s
    assert actual_dynamics == dynamics
    assert runtime == before
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('missing', (('sensors',), ('dynamics',), ('sensors', 'dynamics')))
@pytest.mark.parametrize('nested', (False, True))
def test_formal_course_cannot_silently_disable_randomization(missing, nested):
    raw, _, _ = configurations()
    for name in missing:
        raw.pop(name)
    with pytest.raises(ValueError, match='requires explicit'):
        _runtime_randomization({'tuning': raw} if nested else raw, required=True)


def test_nonformal_legacy_diagnostic_keeps_its_unrandomized_scope():
    assert _runtime_randomization({}, required=False) == (None, None)


@pytest.mark.parametrize('missing', ('sensors', 'dynamics'))
def test_missing_formal_setting_does_not_open_sdk_or_create_output(tmp_path, monkeypatch, missing):
    raw, _, _ = configurations()
    raw.pop(missing)
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps(raw))
    output = tmp_path / 'must_not_exist'

    def unexpected_sdk(*_args, **_kwargs):
        raise AssertionError('invalid physical configuration reached the SDK')

    monkeypatch.setitem(sys.modules, 'transformer_rl.formal.lab_env.sdk',
                        SimpleNamespace(LabSimulation=unexpected_sdk))
    with pytest.raises(ValueError, match='requires explicit'):
        train.main(['--runtime-json', str(runtime), '--output', str(output),
                    '--formal-curriculum', '--device', 'cpu'])
    assert not output.exists()


@pytest.mark.parametrize('name', ('sensors', 'dynamics'))
def test_conflicting_runtime_sources_are_not_silently_overridden(name):
    raw, _, _ = configurations()
    tuning = copy.deepcopy(raw)
    field = 'extra_delay_probability' if name == 'sensors' else 'nominal_fraction'
    tuning[name][field] = .1
    with pytest.raises(ValueError, match='conflicting'):
        _runtime_randomization({**raw, 'tuning': tuning}, required=True)


@pytest.mark.parametrize('change', ('sensor_clock', 'sensor_field', 'dynamics_field', 'not_mapping'))
def test_invalid_runtime_fails_before_a_simulator_can_be_opened(change):
    raw, _, _ = configurations()
    if change == 'sensor_clock':
        raw['sensors']['physics_dt_s'] = .001
    elif change == 'sensor_field':
        raw['sensors']['unknown_option'] = True
    elif change == 'dynamics_field':
        raw['dynamics']['base_mass_scale'] = [-1., 1.]
    else:
        raw['dynamics'] = []
    with pytest.raises(ValueError):
        _runtime_randomization(raw, required=True)

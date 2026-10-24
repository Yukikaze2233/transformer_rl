"""Physical-time transport tests; no simulator, GPU or training budget."""
from dataclasses import replace
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.communication import (
    CommunicationConfig, EpisodeRandomization, EpisodeRandomizationConfig, PhysicalCommunication,
)
from transformer_rl.formal.control.actuation import EffectiveDriveResponse, ResponseParameters
from transformer_rl.formal.lab_env.sensors import LabSensorModel
from transformer_rl.formal.lab_env.action_transport import ActionTransportConfig, DelayedActionTransport


def transport(n=2, **kwargs):
    value = PhysicalCommunication(n, CommunicationConfig(enabled=True, **kwargs), seed=2102)
    value.reset(torch.ones(n, dtype=torch.bool), torch.zeros(n, 6), torch.zeros(n, 6),
                torch.zeros(n, dtype=torch.float64))
    return value


def test_single_round_trip_draw_and_partial_reset_preserve_continuing_rows():
    value = transport(8192)
    torch.testing.assert_close(value.total_s, value.uplink_s + value.downlink_s, rtol=0, atol=0)
    assert .0001 <= value.total_s.min() <= value.total_s.max() <= .005
    previous = value.total_s.clone()
    value.reset(torch.arange(8192) == 3, torch.zeros(8192, 6), torch.zeros(8192, 6),
                torch.ones(8192, dtype=torch.float64))
    assert torch.equal(previous[4:], value.total_s[4:])
    assert previous[3] != value.total_s[3]


def test_feedback_interpolates_only_observed_unwrapped_positions_and_shared_age():
    value = transport(1, total_delay_range_s=(.004, .004))
    q0 = torch.full((1, 6), 6.2)
    value.reset(torch.tensor([True]), q0, q0 * 0, torch.zeros(1, dtype=torch.float64))
    q, _, stamp = value.feedback(torch.zeros(1, dtype=torch.float64))
    assert q.eq(6.2).all() and stamp.eq(0).all()
    value.push(q0 + .2, torch.full((1, 6), 40.))
    q, dq, stamp = value.feedback(torch.tensor([.005], dtype=torch.float64))
    torch.testing.assert_close(q, torch.full_like(q, 6.32))
    torch.testing.assert_close(dq, torch.full_like(dq, 24.))
    torch.testing.assert_close(stamp, torch.tensor([.003], dtype=torch.float64))


def test_delayed_torque_impulse_and_origin_ages_are_separate_from_motor_response():
    value = transport(1, total_delay_range_s=(.004, .004))
    now = torch.zeros(1, dtype=torch.float64)
    value.held_effort.fill_(2.)
    command = torch.full((1, 6), 10.)
    bounds = torch.full_like(command, 40.)
    mean = value.deliver(command, bounds, now, now, now)
    # Old2 acts for2ms and new10 for3ms in the5ms interval.
    torch.testing.assert_close(mean, torch.full_like(mean, 6.8))
    torch.testing.assert_close(value.applied_at_s, torch.tensor([.005], dtype=torch.float64))
    assert value.action_age_steps(.005).eq(1).all()
    parameters = ResponseParameters((0.,) * 6, (0.,) * 6, (0.,) * 6, (.01,) * 6)
    drive = EffectiveDriveResponse(parameters, 1)
    value = transport(1, total_delay_range_s=(.004, .004))
    value.held_effort.fill_(2.)
    mean = value.deliver(command, bounds, now, now, now, response=drive)
    oracle = EffectiveDriveResponse(parameters, 1)
    a = oracle.advance(torch.full_like(command, 2.), torch.tensor([.002], dtype=torch.float64), bounds)
    b = oracle.advance(command, torch.tensor([.003], dtype=torch.float64), bounds)
    torch.testing.assert_close(mean.double() * .005, a.impulse + b.impulse)
    torch.testing.assert_close(drive.state, oracle.state)
    assert not torch.equal(mean, drive.state)


def test_ownership_handoff_clears_old_script_effort_and_partial_reset_is_local():
    value = transport()
    now = torch.ones(2, dtype=torch.float64)
    value.policy_allowed[:] = torch.tensor([False, True])
    value.held_effort[:] = 99.
    value.begin_interval(torch.tensor([True, True]), now)
    assert value.held_effort[0].eq(0).all() and value.held_effort[1].eq(99).all()
    value.reset(torch.tensor([True, False]), torch.zeros(2, 6), torch.zeros(2, 6), now)
    assert value.held_effort[1].eq(99).all()


def test_cold_restore_keeps_rng_but_never_replays_physical_packets():
    value = transport(32)
    value.held_effort.fill_(99)
    state = value.state_dict()
    cold = transport(32)
    cold.restore_after_physical_reset(state)
    assert cold.held_effort.eq(0).all() and not cold.has_previous.any()
    mask, q = torch.ones(32, dtype=torch.bool), torch.zeros(32, 6)
    value.reset(mask, q, q, torch.ones(32, dtype=torch.float64))
    cold.reset(mask, q, q, torch.zeros(32, dtype=torch.float64))
    assert torch.equal(value.total_s, cold.total_s)
    changed = PhysicalCommunication(32, CommunicationConfig(enabled=True, uplink_fraction=.2), seed=1)
    with pytest.raises(ValueError, match='contract'):
        changed.restore_after_physical_reset(state)


def test_reset_local_randomization_expands_without_changing_other_rows_or_network():
    config = EpisodeRandomizationConfig(enabled=True)
    assert config.scale_at(0) == 0 and config.scale_at(2500) == .5 and config.scale_at(5000) == 1
    value = EpisodeRandomization(8192, config, seed=2102)
    all_rows = torch.ones(8192, dtype=torch.bool)
    value.reset(all_rows, 0)
    assert value.strength.eq(1).all()
    value.reset(torch.arange(8192) < 4096, 5000)
    assert value.strength[:4096].min() < .901 and value.strength[:4096].max() > 1.099
    assert value.strength[4096:].eq(1).all()
    state = value.state_dict()
    cold = EpisodeRandomization(8192, config, seed=1)
    cold.restore_after_physical_reset(state)
    value.reset(all_rows, 6000)
    cold.reset(all_rows, 6000)
    assert torch.equal(value.strength, cold.strength)


def test_action_issuance_timestamp_survives_transport_and_handoff_flush():
    value = DelayedActionTransport(1, 6, ActionTransportConfig(True, (1,), 0.), seed=1)
    now = torch.tensor([1.], dtype=torch.float64)
    value.step(torch.ones(1, 6), issued_at_s=now)
    value.step(torch.full((1, 6), 2.), issued_at_s=now + .01)
    assert value.arrived_generated_at_s.eq(1).all()
    value.step(torch.full((1, 6), 99.), policy_allowed=torch.tensor([False]), issued_at_s=now + .02)
    value.step(torch.full((1, 6), 4.), policy_allowed=torch.tensor([True]), issued_at_s=now + .03)
    torch.testing.assert_close(value.arrived_generated_at_s, now + .03)


def test_actor_extra_lag_and_pd_share_one_uplink_without_fabricating_reset_history():
    from test_physical_observation import fixture
    from test_sensor_model import no_noise, push

    _, _, assembler, raw, _ = fixture(4)
    template = replace(assembler.measure(raw), timestamp_s=torch.zeros(4, dtype=torch.float64))
    value = transport(4, total_delay_range_s=(.004, .004))
    sensor = LabSensorModel(4, 2102, no_noise(nominal_episode_fraction=0.,
        extra_delay_probability=1., observation_delays_s=(.005,)),
        uplink_delay_s=value.uplink_s, maximum_uplink_s=.002)
    mask = torch.ones(4, dtype=torch.bool)
    sensor.reset(mask, template.timestamp_s)
    sensor.seed_after_reset(mask, template)
    assert sensor(template).sampled_at_s.eq(0).all()
    for _ in range(6):
        state = push(sensor, template)
        value.push(state.policy_position_rad, state.policy_velocity_rad_s)
    output = sensor(state)
    torch.testing.assert_close(output.policy_position_rad[:, :4], torch.full((4, 4), 4.6))
    torch.testing.assert_close(output.sampled_at_s, torch.full((4, 4), .023, dtype=torch.float64))
    pd_q, _, pd_at = value.feedback(state.timestamp_s)
    torch.testing.assert_close(pd_q, torch.full((4, 6), 5.6))
    torch.testing.assert_close(pd_at, torch.full((4,), .028, dtype=torch.float64))
    # Reset one row at the same boundary; its old history must not leak into
    # either PD or the Actor, and continuing endpoints remain byte-identical.
    reset = torch.tensor([True, False, False, False])
    value.reset(reset, state.policy_position_rad, state.policy_velocity_rad_s, state.timestamp_s)
    sensor.reset(reset, state.timestamp_s)
    sensor.seed_after_reset(reset, state)
    current = sensor(state)
    assert torch.equal(current.sampled_at_s[1:], output.sampled_at_s[1:])
    assert current.sampled_at_s[0].eq(.03).all()


def test_sensor_noise_and_target_lag_are_nominal_at_zero_exposure():
    from test_physical_observation import fixture
    from test_sensor_model import configuration, push

    _, _, assembler, raw, _ = fixture(8)
    template = replace(assembler.measure(raw), timestamp_s=torch.zeros(8, dtype=torch.float64))
    sensor = LabSensorModel(8, 2102, configuration(nominal_episode_fraction=0., extra_delay_probability=1.))
    mask = torch.ones(8, dtype=torch.bool)
    sensor.reset(mask, template.timestamp_s, difficulty=0.)
    sensor.seed_after_reset(mask, template)
    for _ in range(2):
        state = push(sensor, template)
    output = sensor(state)
    assert sensor.lags.eq(0).all()
    torch.testing.assert_close(output.policy_velocity_rad_s, state.policy_velocity_rad_s, rtol=0, atol=0)
    action = DelayedActionTransport(8, 6, ActionTransportConfig(True, (1,), 0.), seed=1)
    action.reset(mask, difficulty=0.)
    assert action.lags.eq(0).all()


def test_native_control_hook_applies_delayed_feedback_then_exact_response_once():
    from transformer_rl.formal.control.actuation import OutputFeedbackController

    path = Path(__file__).resolve().parents[1] / 'src/transformer_rl/formal/lab_env/env.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    method = next(node for node in definition.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_apply_action')
    from transformer_rl.formal.control.ownership import ControlSource
    namespace = {'torch': torch, 'ControlSource': ControlSource}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    comm = transport(1, total_delay_range_s=(.004, .004))
    comm.push(torch.full((1, 6), .2), torch.ones(1, 6))
    now = torch.full((1,), .005, dtype=torch.float64)
    controller = OutputFeedbackController((0.,) * 6, 1, pair_domain=None)
    targets = controller.decode(torch.zeros(1, 6), torch.full((1, 6), .2), now)
    parameters = ResponseParameters((0.,) * 6, (0.,) * 6, (0.,) * 6, (.01,) * 6)
    drive = EffectiveDriveResponse(parameters, 1)
    applied = []
    env = SimpleNamespace(_targets=targets, _communication=comm,
        _joint_positions=lambda: torch.full((1, 6), .2), _active_indices=torch.arange(6),
        _as_torch=lambda x: x, robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.ones(1, 6)),
            set_joint_effort_target=lambda value: applied.append(value.clone())),
        _clock=lambda: now, _policy_generated_at_s=now,
        _control=SimpleNamespace(controller=controller, drive=drive),
        _spring_force=None, num_envs=1, _device=torch.device('cpu'), _last_effort=torch.zeros(1, 6),
        _action_latch=None, _encoder_position=lambda q: q, _episode_randomization=None, _motor_envelope=None,
        _substep=0)
    namespace['_apply_action'](env)
    assert len(applied) == 1 and applied[0][:, 6:].eq(0).all()
    # Uplink2ms -> q=.12,dq=.6 -> requested leg torque=-20.7Nm.
    # Downlink2ms -> only3ms of the new request reach the response this step.
    impulse_weight = (.003 - .01 * (1 - math.exp(-.003 / .01))) / .005
    expected = -20.7 * impulse_weight
    torch.testing.assert_close(env._last_effort[:, :4], torch.full((1, 4), expected))
    torch.testing.assert_close(drive.state[:, :4], torch.full((1, 4), -20.7 * (1 - math.exp(-.3))))


@pytest.mark.parametrize('field,value', [('total_delay_range_s', (0., .01)), ('uplink_fraction', 2.),
                                       ('total_delay_range_s', (float('nan'), .005))])
def test_invalid_communication_prior_is_rejected(field, value):
    with pytest.raises(ValueError):
        CommunicationConfig(**{field: value})

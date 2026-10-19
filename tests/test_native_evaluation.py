"""First-episode physical evaluation, on Kaiser CPU synthetic interfaces only."""
from dataclasses import replace
import json
import math

import pytest
import torch

from test_native_task import fixture
from test_structured_collection import model
from transformer_rl._native_evaluation import (
    EvaluationPhysicalSample, FirstEpisodeStatistics, evaluate_first_episodes,
)
from transformer_rl._native_task import FailureReason
from transformer_rl._task_commands import RequestInputs


def sample(tick, dt=.005, *, count=2, **changes):
    zeros = lambda width=None: torch.zeros(count) if width is None else torch.zeros(count, width)
    booleans = lambda value: torch.full((count,), value, dtype=torch.bool)
    command = zeros(4)
    command[:, 3] = .305
    position = zeros(3)
    position[:, 0] = tick * dt * .01
    result = EvaluationPhysicalSample(
        tick, torch.full((count,), tick * dt, dtype=torch.float64), torch.zeros(count, dtype=torch.long),
        torch.ones(count, dtype=torch.long), booleans(True), booleans(True), booleans(False),
        command.clone(), command, torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long),
        zeros(3), torch.tensor([[.01, 0., 0.]]).repeat(count, 1), position,
        torch.full((count,), .305), zeros(2), zeros(3), zeros(), zeros(6), zeros(6),
        torch.zeros(count, 6, dtype=torch.bool), zeros(6), zeros(2), zeros(2), torch.ones(count, 2),
        zeros(), torch.ones(count, 2), zeros(2), booleans(False))
    return replace(result, **changes)


def statistics(horizon=.02, dt=.005, *, initial=True):
    return FirstEpisodeStatistics(torch.zeros(2, dtype=torch.long), torch.full((2,), horizon, dtype=torch.float64),
                                  physical_tick=0, physics_dt_s=dt,
                                  initial_world_position_m=torch.zeros(2, 3) if initial else None)


def close(stats, *, terminated=(False, False), truncated=(False, False), reasons=(0, 0)):
    stats.close_interval(torch.tensor(terminated), torch.tensor(truncated), torch.tensor(reasons))


@pytest.mark.parametrize('dt', [.005, .001])
def test_true_substep_moments_world_drift_and_centered_noise_are_not_body_velocity_or_mean_rms(dt):
    stats = statistics(dt=dt)
    n = round(.02 / dt)
    for tick in range(1, n + 1):
        body = torch.zeros(2, 3)
        body[:, 0] = 1 if tick <= n // 2 else 3
        torque = torch.full((2, 6), tick * dt * 2.)
        stats.observe(sample(tick, dt, velocity_body_m_s=body, active_effort_nm=torque,
                             output_velocity_rad_s=torch.full((2, 6), 3.)))
        if tick % stats.substeps == 0:
            close(stats, truncated=(tick == n, tick == n))
    row = stats.reports()[0]
    moments = row['moments']['full_task']
    assert moments['tracking_error']['mean'][0] == pytest.approx(2.)
    assert moments['tracking_error']['rms'][0] == pytest.approx(math.sqrt(5))
    assert moments['tracking_error']['centered_std'][0] == pytest.approx(1.)
    assert moments['world_xy_speed']['rms'][0] == pytest.approx(.01)
    assert row['world_xy_net_displacement_m'] == pytest.approx(.0002)
    assert moments['effort_slew']['physical_samples'] == n - 1
    assert moments['effort_slew']['mean'] == pytest.approx([2.] * 6, abs=1e-6)
    assert moments['absolute_mechanical_power']['mean'][0] == pytest.approx(3 * moments['active_effort']['mean'][0])
    assert moments['jump_world_z_error']['physical_samples'] == 0
    assert row['healthy_policy_owned_full_horizon']
    assert row['standing_10s_drift']['max_net_displacement_m'] is None
    json.dumps(row, allow_nan=False)


def test_first_failure_remains_failed_after_other_rows_continue_and_native_buffers_are_reused():
    stats = statistics(horizon=.03)
    for tick in range(1, 7):
        episode = torch.tensor([int(tick > 2), 0])
        bad = torch.tensor([tick == 1, False])
        packet = sample(tick, episode_id=episode, healthy=~bad, true_failure=bad)
        stats.observe(packet)
        # Reusing source positions/times after the callback cannot change its endpoint.
        packet.root_position_world_m.fill_(100.)
        packet.timestamp_s.fill_(100.)
        if tick % 2 == 0:
            close(stats, terminated=(tick == 2, False), truncated=(False, tick == 6),
                  reasons=(1 if tick == 2 else 0, 0))
    first, second = stats.reports()
    assert first['outcome'] == 'true_failure' and first['failure_reasons'] == 1
    assert first['observed_task_s'] == .01 and not first['healthy_policy_owned_full_horizon']
    assert first['moments']['full_task']['world_xy_speed']['physical_samples'] == 2
    assert second['observed_task_s'] == pytest.approx(.03) and second['healthy_policy_owned_full_horizon']
    assert second['world_xy_net_displacement_m'] < .001


def test_preparation_is_not_task_time_but_later_script_time_is_not_refunded():
    stats = statistics(horizon=.02, initial=False)
    for tick in range(1, 7):
        owned = torch.full((2,), tick in (3, 4), dtype=torch.bool)
        stats.observe(sample(tick, policy_owned=owned))
        if tick % 2 == 0:
            close(stats, truncated=(tick == 6, tick == 6))
    row = stats.reports()[0]
    assert row['observed_task_s'] == .02
    assert row['policy_owned_s'] == row['scripted_task_s'] == .01
    assert row['whole_task_healthy'] and not row['whole_task_policy_owned']
    assert not row['healthy_policy_owned_full_horizon']


def test_ten_second_drift_needs_a_full_unbroken_supported_standing_window():
    stats = statistics(horizon=10.01)
    for tick in range(1, 2003):
        stats.observe(sample(tick))
        if tick % 2 == 0:
            close(stats, truncated=(tick == 2002, tick == 2002))
    row = stats.reports()[0]
    assert row['standing_10s_drift']['windows'] == 2
    assert row['standing_10s_drift']['max_net_displacement_m'] == pytest.approx(.1, abs=1e-7)
    steady = row['moments']['steady_support']['world_xy_speed']
    assert steady['physical_samples'] == 1602 and steady['support_s'] == pytest.approx(8.01)


@pytest.mark.parametrize('fault', ['skipped_tick', 'nan', 'episode', 'half_handoff', 'no_position', 'missed_endpoint'])
def test_missing_physical_evidence_cannot_be_silently_reconstructed(fault):
    stats = statistics(initial=fault != 'no_position')
    if fault in ('half_handoff', 'missed_endpoint'):
        stats.observe(sample(1, policy_owned=torch.zeros(2, dtype=torch.bool)))
    if fault == 'missed_endpoint':
        stats.observe(sample(2, policy_owned=torch.zeros(2, dtype=torch.bool)))
    packet = sample(3 if fault == 'missed_endpoint' else 2 if fault in ('skipped_tick', 'half_handoff') else 1)
    if fault == 'nan':
        packet = replace(packet, wheel_clearance_m=torch.full((2, 2), float('nan')))
    if fault == 'episode':
        packet = replace(packet, episode_id=torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError):
        stats.observe(packet)


@pytest.mark.parametrize('dt', [.005, .001])
def test_native_runner_uses_fixed_means_all_physics_and_first_episodes_without_an_optimizer(dt):
    torch.manual_seed(2101)
    network = model()
    network.train()
    network.actor.policy.base_projection.eval()
    modes = [module.training for module in network.modules()]
    parameters = {name: value.detach().clone() for name, value in network.state_dict().items()}
    env, robot, sim, *_ = fixture(dt, horizons=(.02, .04))
    observed_actions = []
    original = env.lifecycle.plant.accept_issued_action

    def record(action):
        observed_actions.append(action.clone())
        return original(action)

    env.lifecycle.plant.accept_issued_action = record
    report = evaluate_first_episodes(network, env, max_vector_steps=15)
    assert report['all_first_episodes_finished'] and report['optimizer_steps'] == 0
    assert report['vector_steps'] == 9 and report['attempted_physical_transitions'] == 18
    assert sim.steps == 9 * round(.01 / dt)
    assert env.trace_sink is None and [module.training for module in network.modules()] == modes
    assert all(torch.equal(parameters[name], value) for name, value in network.state_dict().items())
    assert [row['observed_task_s'] for row in report['rows']] == pytest.approx([.02, .04])
    assert report['rows'][0]['moments']['full_task']['tracking_error']['physical_samples'] == round(.02 / dt)
    assert not report['physical_qualification'] and not report['scene_case_binding_verified']
    assert len(observed_actions) == 9
    assert all(torch.equal(action, torch.zeros_like(action)) for action in observed_actions[:5])
    json.dumps(report, allow_nan=False)


def test_native_runner_keeps_substep_fall_in_first_outcome_after_auto_reset():
    network = model()
    env, robot, sim, *_ = fixture(horizons=(.02, .04))
    step = sim.step

    def fall_then_recover(*, render):
        step(render=render)
        if sim.steps == 11:
            robot.data.root_link_pose_w.torch[0, 2] -= .25
        elif sim.steps == 12:
            robot.data.root_link_pose_w.torch[0, 2] += .25

    sim.step = fall_then_recover
    report = evaluate_first_episodes(network, env, max_vector_steps=15)
    first = report['rows'][0]
    assert first['outcome'] == 'true_failure' and first['failure_reasons'] & int(FailureReason.LOW_SUPPORT_HEIGHT)
    assert first['observed_task_s'] == .01 and not first['healthy_policy_owned_full_horizon']


def test_bounded_incomplete_run_never_fabricates_full_horizon_or_skill_success():
    env, *_ = fixture(jump=True)
    report = evaluate_first_episodes(model(), env, max_vector_steps=6)
    assert not report['all_first_episodes_finished']
    assert all(r['outcome'] == 'incomplete' and not r['healthy_policy_owned_full_horizon'] for r in report['rows'])
    assert all(not r['request_events'] for r in report['rows'])


def test_actual_request_acceptance_is_preserved_separately_from_physical_success():
    env, *_ = fixture(horizons=(.02, .02))
    reset_source = env.reset_source

    def stationary_reset(*args):
        plan = reset_source(*args)
        return replace(plan, root_com_velocity=torch.zeros_like(plan.root_com_velocity))

    env.reset_source = stationary_reset
    network = model()
    with torch.no_grad():
        network.actor.policy.state_head[-1].weight.zero_()
        network.actor.policy.state_head[-1].bias.zero_()
        network.actor.policy.state_head[-1].bias[6:8] = 5.

    def deliver(index, observation, finished):
        if index == 5:
            return replace(RequestInputs.empty(2, env.device), request_id=torch.tensor([5, -1]),
                           requested_height_m=torch.tensor([.1, 0.]))

    report = evaluate_first_episodes(network, env, max_vector_steps=10, boundary_inputs=deliver)
    events = report['rows'][0]['request_events']
    assert any(e['source'] == 'protocol_acceptance' and e['request_id'] == 5 for e in events)
    assert all(e.get('outcome') != 'PHYSICAL_SUCCESS' for e in events)
    assert any(e['source'] == 'episode_reset' and e['request_id'] == 5 for e in events)


def test_rotating_native_fixture_request_stays_not_ready_and_is_not_dropped_from_the_request_record():
    env, *_ = fixture(horizons=(.02, .02))
    network = model()
    with torch.no_grad():
        network.actor.policy.state_head[-1].weight.zero_()
        network.actor.policy.state_head[-1].bias.zero_()
        network.actor.policy.state_head[-1].bias[6:8] = 5.

    def deliver(index, observation, finished):
        if index == 5:
            return replace(RequestInputs.empty(2, env.device), request_id=torch.tensor([5, -1]),
                           requested_height_m=torch.tensor([.1, 0.]))

    report = evaluate_first_episodes(network, env, max_vector_steps=10, boundary_inputs=deliver)
    events = report['rows'][0]['request_events']
    assert any(e['source'] == 'protocol_request' and e['request_id'] == 5 and e['response'] == 'NOT_READY'
               for e in events)
    assert all(e['source'] != 'protocol_acceptance' for e in events)


def test_policy_parameter_mutation_or_trace_error_cannot_return_a_successful_report():
    network = model()
    env, *_ = fixture(horizons=(.02, .02))

    def change(index, observation, finished):
        network.actor.log_std.add_(1.)

    with pytest.raises(RuntimeError, match='changed'):
        evaluate_first_episodes(network, env, max_vector_steps=10, boundary_inputs=change)
    assert env.trace_sink is None


def test_policy_action_bounds_keep_six_axis_fractions_and_only_first_policy_owned_intervals():
    network = model()
    with torch.no_grad():
        network.actor.policy.leg_decoder[-1].weight.zero_()
        network.actor.policy.leg_decoder[-1].bias.copy_(torch.tensor([12., 0.]))
        network.actor.policy.wheel_decoder[-1].weight.zero_()
        network.actor.policy.wheel_decoder[-1].bias.fill_(13.)
    env, *_ = fixture(horizons=(.02, .04))
    report = evaluate_first_episodes(network, env, max_vector_steps=15)
    row = report['rows'][0]
    action = row['policy_action_moments_100hz']
    assert action['upper_bound_fraction']['mean'] == [1., 0., 1., 0., 1., 1.]
    assert action['lower_bound_fraction']['mean'] == [0.] * 6
    assert action['upper_bound_fraction']['policy_intervals'] == 2
    assert action['issued']['mean'] == [3., 0., 3., 0., 9., 9.]
    assert row['policy_action_joint_order'] == ['L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3']
    assert 'physical_samples' not in action['issued']
    json.dumps(report, allow_nan=False)

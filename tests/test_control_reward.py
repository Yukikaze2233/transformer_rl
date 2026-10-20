"""Synthetic physical reward/ownership checks; no robot or SDK is run."""
from dataclasses import replace

import pytest
import torch

from test_physical_observation import fixture as mechanics_fixture
from transformer_rl.formal.tasks.reward import (
    PhysicalRewardComposer, RewardPhase, RewardPhysics, RewardReference, RewardTuning,
)
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.research.reward_components import RewardComponentStatistics
from transformer_rl.formal.learning.contracts import BoundaryMasks, masked_gae


def fixture(period=.005, retired=None):
    binding = mechanics_fixture()[1]
    reward = PhysicalRewardComposer(binding, 2, RewardTuning(20.), physics_dt_s=period, retired_request_ids=retired)
    reward.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.float64),
                 torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    return reward


def reference(mode=CommandMode.NORMAL, phase=RewardPhase.SUPPORT):
    return RewardReference(torch.tensor([[0., 0., 0., .305]]).repeat(2, 1),
                            torch.full((2,), int(mode), dtype=torch.long),
                            torch.full((2,), int(phase), dtype=torch.long),
                            torch.full((2,), 2.305), torch.zeros(2), torch.zeros(2, 2))


def sample(reward, index):
    window = reward._window
    loads = torch.zeros(2, 19)
    loads[:, reward.binding.wheel_body_indices] = 100
    return RewardPhysics(window['start'] + index * reward.physics_dt_s, window['episode'].clone(),
                         window['control'].clone(), window['owned'].clone(), torch.zeros(2, 3),
                         torch.zeros(2, 3), torch.full((2,), 2.305), torch.full((2,), .305),
                         torch.zeros(2, 2), torch.tensor([[0., 0., -1.]]).repeat(2, 1), torch.zeros(2, 3),
                         window['dq'].clone(), torch.zeros(2, 6), loads, torch.zeros(2, 2), torch.zeros(2, 6))


def begin(reward, ref=None, issued=None, owned=None):
    ref = reference() if ref is None else ref
    issued = torch.zeros(2, 6) if issued is None else issued
    owned = torch.ones(2, dtype=torch.bool) if owned is None else owned
    initial = torch.where(reward._dq_valid[:, None], reward._last_dq, torch.zeros(2, 6))
    reward.begin(issued, initial.clone(), now_s=reward._time.clone(), episode_id=reward._episode.clone(),
                 control_epoch=reward._control.clone(), policy_owned=owned,
                 admitted_body_command=ref.admitted_body_command, command_mode=ref.mode)
    return ref


def finish(reward, *, failed=None, timeout=None, success=None, whole_owned=None):
    return reward.finish(terminated=torch.zeros(2, dtype=torch.bool) if failed is None else failed,
                          truncated=torch.zeros(2, dtype=torch.bool) if timeout is None else timeout,
                          physical_success_request_id=torch.full((2,), -1, dtype=torch.long) if success is None else success,
                          entire_skill_policy_owned=torch.ones(2, dtype=torch.bool) if whole_owned is None else whole_owned)


def step(reward, ref=None, *, issued=None, owned=None, change=None, **events):
    ref = begin(reward, ref, issued, owned)
    for index in range(1, reward.substeps + 1):
        value = sample(reward, index)
        reward.observe(value if change is None else change(value, index), ref)
    return finish(reward, **events)


@pytest.fixture(autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize('period', [.005, .001])
def test_perfect_support_reward_is_seven_per_second_with_exactly_one_dt(period):
    reward = fixture(period)
    result = step(reward)
    torch.testing.assert_close(result.learning_reward, torch.full((2,), .07))
    torch.testing.assert_close(torch.stack(tuple(result.raw_positive_densities.values())).sum(0),
                               torch.full((2,), 7., dtype=torch.float64))
    assert not torch.stack(tuple(result.raw_cost_densities.values())).any()
    assert result.timestamp_s.eq(.01).all()
    with pytest.raises(RuntimeError, match='every declared'):
        finish(reward)


def test_roll_pitch_rate_damping_is_mode_gated_and_separate_from_attitude():
    reward = fixture()

    def change(value, _):
        return replace(value, angular_velocity_body_rad_s=torch.tensor([[1., 2., 0.]]).repeat(2, 1))

    normal = step(reward, change=change)
    torch.testing.assert_close(normal.raw_cost_densities['roll_pitch_angular_velocity'],
                               torch.full((2,), .25, dtype=torch.float64))
    jump = step(reward, reference(CommandMode.JUMP, RewardPhase.FLIGHT), change=change)
    torch.testing.assert_close(jump.raw_cost_densities['roll_pitch_angular_velocity'],
                               torch.full((2,), .5, dtype=torch.float64))
    recovery = step(reward, reference(CommandMode.RECOVERY, RewardPhase.RECOVERY), change=change)
    assert recovery.raw_cost_densities['roll_pitch_angular_velocity'].eq(0).all()


def test_complete_jump_density_budget_is_eight_not_nine_and_does_not_change_support_weights():
    reward = fixture()
    result = step(reward, reference(CommandMode.JUMP, RewardPhase.FLIGHT))
    torch.testing.assert_close(result.learning_reward, torch.full((2,), .08))
    assert result.raw_positive_densities['forward'].eq(2).all()
    assert result.raw_positive_densities['jump_attitude'].eq(1).all()
    normal = step(reward)
    assert normal.raw_positive_densities['forward'].eq(3).all()


def test_recovery_has_a_broad_orientation_signal_and_explicit_height_eligibility():
    reward = fixture()
    ref = reference(CommandMode.RECOVERY, RewardPhase.RECOVERY)
    inverted = step(reward, ref, change=lambda value, _: replace(
        value, gravity_body=torch.tensor([[0., 0., 1.]]).repeat(2, 1),
        roll_pitch_rad=torch.tensor([[torch.pi, 0.]]).repeat(2, 1), height_above_ground_m=torch.full((2,), .1)))
    upright = step(reward, ref)
    assert inverted.raw_positive_densities['recovery_upright'].eq(0).all()
    assert inverted.raw_cost_densities['attitude'].gt(0).all()
    assert inverted.raw_positive_densities['height'].eq(0).all()
    assert upright.raw_positive_densities['recovery_upright'].eq(4).all()
    torch.testing.assert_close(upright.learning_reward, torch.full((2,), .06))
    assert upright.learning_reward.gt(inverted.learning_reward).all()


def test_effort_power_geometric_margin_and_harmful_contact_keep_separate_physical_units():
    reward = fixture()

    def change(value, _):
        loads = value.body_normal_load_n.clone()
        loads[:, reward._harmful[0]] = 4
        return replace(value, active_explicit_effort_nm=torch.tensor([[2., 2., 2., 2., 1., 1.]]).repeat(2, 1),
                       policy_velocity_rad_s=torch.tensor([[3., 3., 3., 3., 4., 4.]]).repeat(2, 1),
                       joint_boundary_shortfall_rad=torch.full((2, 6), .05), body_normal_load_n=loads)

    result = step(reward, change=change)
    torch.testing.assert_close(result.raw_cost_densities['effort'], torch.full((2,), .0036, dtype=torch.float64))
    torch.testing.assert_close(result.raw_cost_densities['mechanical_power_proxy'], torch.full((2,), .064, dtype=torch.float64))
    assert result.raw_cost_densities['joint_boundary'].eq(12).all()
    assert result.raw_cost_densities['harmful_contact'].eq(2).all()


def test_large_errors_remain_visible_while_bounded_density_keeps_the_safe_return_lower_bound():
    reward = fixture()

    def extreme(value, _):
        return replace(value, velocity_origin_body_m_s=torch.full((2, 3), 100.),
                       height_above_ground_m=torch.full((2,), 10.), roll_pitch_rad=torch.full((2, 2), 3.))

    result = step(reward, change=extreme)
    density = torch.stack(tuple(result.component_densities.values())).sum(0)
    assert density.ge(-10 - 1e-5).all() and density.le(8).all()
    assert torch.stack(tuple(result.raw_cost_densities.values())).sum(0).gt(100000).all()
    assert result.raw_error_means['forward_m_s'].eq(100).all()
    assert result.raw_error_mean_squares['forward_m_s'].eq(10000).all()


def test_exact_stationary_gate_uses_world_xy_and_has_no_hidden_velocity_deadzone():
    reward = fixture()
    world = torch.tensor([[.5, 0., 0.], [0., .5, 0.]])
    result = step(reward, change=lambda value, _: replace(value, velocity_origin_world_m_s=world))
    assert result.raw_cost_densities['stand_world_xy'].gt(0).all()
    assert result.raw_error_means['forward_m_s'].eq(0).all()
    ref = reference()
    ref.admitted_body_command[:, 0] = 1e-8
    moving = step(reward, ref, change=lambda value, _: replace(value, velocity_origin_world_m_s=world))
    assert moving.gate_fractions['exact_stationary'].eq(0).all()
    assert moving.raw_cost_densities['stand_world_xy'].eq(0).all()


@pytest.mark.parametrize('period', [.005, .001])
def test_acceleration_uses_each_physics_dt_and_endpoint_velocity_cannot_be_silently_rezeroed(period):
    reward = fixture(period)
    initial = torch.zeros(2, 6)
    result = step(reward, change=lambda value, index: replace(value, policy_velocity_rad_s=initial + 2 * index * period))
    torch.testing.assert_close(result.raw_cost_densities['leg_acceleration'], torch.full((2,), 1.6e-6, dtype=torch.float64),
                               rtol=1e-5, atol=1e-12)
    ref = reference()
    with pytest.raises(ValueError, match='PRE-reset'):
        reward.begin(torch.zeros(2, 6), torch.zeros(2, 6), now_s=reward._time.clone(),
                     episode_id=reward._episode.clone(), control_epoch=reward._control.clone(),
                     policy_owned=torch.ones(2, dtype=torch.bool), admitted_body_command=ref.admitted_body_command,
                     command_mode=ref.mode)


def test_issued_differences_are_at100Hz_once_and_selected_reset_does_not_clear_other_history():
    reward = fixture()
    first = step(reward)
    second = step(reward, issued=torch.ones(2, 6))
    third = step(reward, issued=torch.zeros(2, 6))
    assert first.raw_cost_densities['issued_first_difference'].eq(0).all()
    torch.testing.assert_close(second.raw_cost_densities['issued_first_difference'], torch.full((2,), .24, dtype=torch.float64))
    assert second.raw_cost_densities['issued_second_difference'].eq(0).all()
    torch.testing.assert_close(third.raw_cost_densities['issued_second_difference'], torch.full((2,), 14.08, dtype=torch.float64))
    reward.reset(torch.tensor([True, False]), reward._time.clone(), torch.tensor([1, 0]), torch.tensor([1, 0]))
    result = step(reward, issued=torch.ones(2, 6))
    assert result.gate_fractions['first_action_difference'].tolist() == [0., 1.]
    assert result.gate_fractions['second_action_difference'].tolist() == [0., 1.]


def test_slip_uses_named_contact_speed_and_force_gate_not_body_or_rolling_speed():
    reward = fixture()

    def change(value, _):
        loads = value.body_normal_load_n.clone()
        loads[:, reward._wheels[1]] = 4
        return replace(value, wheel_contact_tangent_speed_m_s=torch.full((2, 2), .2), body_normal_load_n=loads,
                       policy_velocity_rad_s=torch.tensor([[0., 0., 0., 0., 100., -100.]]).repeat(2, 1))

    result = step(reward, change=change)
    assert result.raw_error_means['forward_m_s'].eq(0).all()
    torch.testing.assert_close(result.raw_cost_densities['contact_slip'], torch.full((2,), .5, dtype=torch.float64))
    assert result.gate_fractions['slip_left'].eq(1).all() and result.gate_fractions['slip_right'].eq(0).all()


def test_true_failure_pays_absorbing_tail_once_but_time_limit_does_not():
    reward = fixture()
    result = step(reward, failed=torch.tensor([True, False]), timeout=torch.tensor([True, True]))
    expected = .07 - 4 + reward.tuning.gamma * reward.tuning.absorbing_value
    torch.testing.assert_close(result.learning_reward, torch.tensor([expected, .07]))
    assert result.event_rewards['failure_absorbing_value_tail'][1] == 0
    assert result.timestamp_s.eq(.01).all()  # No absorbing physics was appended.
    with pytest.raises(ValueError, match='initialized'):
        begin(reward)
    with pytest.raises(RuntimeError):
        finish(reward)


def test_safe_holding_and_recovery_outrank_immediate_failure_even_at_the_worst_density_bound():
    reward = fixture()
    failure = step(reward, failed=torch.ones(2, dtype=torch.bool))
    gamma = reward.tuning.gamma
    safe_2s = sum(gamma**i * -.0999 for i in range(200))
    recovery_2s = sum(gamma**i * (-.08 if i < 50 else .02) for i in range(200))
    infinite_safe_lower = -.1 / (1 - gamma)
    best_possible_failure = .08 - 4 + gamma * reward.tuning.absorbing_value
    assert -4 > safe_2s  # The old event alone could prefer immediate failure.
    assert recovery_2s > safe_2s > float(failure.learning_reward[0])
    assert infinite_safe_lower > best_possible_failure


@pytest.mark.parametrize('period', [.005, .001])
def test_each_subinterval_is_bounded_before_integration_rather_than_bounding_a_policy_rate_mean(period):
    reward = fixture(period)
    result = step(reward, change=lambda value, index: replace(value, joint_boundary_shortfall_rad=
                  torch.full((2, 6), .05 if index <= reward.substeps // 2 else 0.)))
    expected = 7 - 5 * torch.tanh(torch.tensor(1.2, dtype=torch.float64))
    total_density = torch.stack(tuple(result.component_densities.values())).sum(0)
    torch.testing.assert_close(total_density, expected.expand(2), atol=1e-6, rtol=1e-6)
    assert result.raw_cost_densities['joint_boundary'].eq(6).all()
    assert not torch.allclose(total_density, torch.full((2,), 7., dtype=torch.float64) - 10 * torch.tanh(torch.tensor(.6)))


def test_gae_does_not_bootstrap_an_already_included_failure_tail_but_time_limit_keeps_its_terminal_value():
    reward = fixture()
    failed, timeout = torch.tensor([[True, False]]), torch.tensor([[False, True]])
    result = step(reward, failed=failed[0], timeout=timeout[0])
    owned = torch.ones(1, 2, dtype=torch.bool)
    masks = BoundaryMasks.from_events(fully_policy_owned=owned, terminated=failed, truncated=timeout,
                                      final_value_valid=owned, handoff_after=~owned, rollout_end=owned,
                                      state_label_valid=owned, next_label_valid=owned, next_same_control_epoch=owned)
    _, returns = masked_gae(result.learning_reward[None], torch.zeros(1, 2), torch.tensor([[123., 4.]]), masks,
                            gamma=reward.tuning.gamma, gae_lambda=.9746794344808963)
    torch.testing.assert_close(returns[0, 0], result.learning_reward[0])
    torch.testing.assert_close(returns[0, 1], result.learning_reward[1] + reward.tuning.gamma * 4)


def test_physical_success_is_once_per_request_and_cold_retired_ids_cannot_be_rewarded_again():
    reward = fixture(retired=torch.tensor([7, 7]))
    retired = step(reward, success=torch.tensor([7, 7]))
    assert not retired.success_credited.any()
    first = step(reward, success=torch.tensor([8, 8]))
    assert first.success_credited.all() and first.event_rewards['physical_skill_success'].eq(1).all()
    again = step(reward, success=torch.tensor([8, 8]))
    assert not again.success_credited.any()
    reward.reset(torch.ones(2, dtype=torch.bool), reward._time.clone(), torch.ones(2, dtype=torch.long),
                 torch.ones(2, dtype=torch.long))
    assert not step(reward, success=torch.tensor([8, 8])).success_credited.any()


def test_scripted_or_partially_owned_skill_cannot_pay_policy_success_and_event_is_not_replayed_later():
    reward = fixture()
    ref = begin(reward)
    reward.observe(sample(reward, 1), ref)
    value = sample(reward, 2)
    reward.observe(replace(value, policy_owned=torch.tensor([False, True])), ref)
    result = finish(reward, success=torch.tensor([9, 9]), whole_owned=torch.tensor([True, False]))
    assert result.learning_reward[0] == 0 and not result.success_credited.any()
    assert result.physical_step_reward[0] > 0 and result.fully_policy_owned.tolist() == [False, True]
    assert not step(reward, success=torch.tensor([9, 9])).success_credited.any()


def test_physical_density_and_explicit_event_statistics_sum_without_an_unattributed_residual():
    reward = fixture()
    result = step(reward, success=torch.tensor([1, 1]), failed=torch.tensor([False, True]))
    statistics = RewardComponentStatistics(2, .01)
    statistics.observe(result.physical_step_reward, result.component_densities, result.event_rewards)
    report = statistics.drain()
    assert abs(report['residual_step_reward']['statistics']['sum']) < 1e-12
    assert not result.success_credited[1]


@pytest.mark.parametrize('change', ['timestamp', 'episode', 'source', 'negative_slip', 'negative_margin', 'gravity', 'phase', 'next_command', 'nan'])
def test_invalid_sources_or_next_commands_are_refused_before_consuming_the_physical_sample(change):
    reward = fixture()
    ref = begin(reward)
    value, wrong_ref = sample(reward, 1), ref
    if change == 'timestamp':
        value = replace(value, timestamp_s=value.timestamp_s + .001)
    elif change == 'episode':
        value = replace(value, episode_id=value.episode_id + 1)
    elif change == 'source':
        value = replace(value, slip_reference='wheel_radius_times_speed_minus_body_vx')
    elif change == 'negative_slip':
        value = replace(value, wheel_contact_tangent_speed_m_s=-torch.ones(2, 2))
    elif change == 'negative_margin':
        value = replace(value, joint_boundary_shortfall_rad=-torch.ones(2, 6))
    elif change == 'gravity':
        value = replace(value, gravity_body=torch.zeros(2, 3))
    elif change == 'phase':
        wrong_ref = replace(ref, phase=torch.full((2,), int(RewardPhase.FLIGHT), dtype=torch.long))
    elif change == 'next_command':
        command = ref.admitted_body_command.clone()
        command[:, 0] = 1
        wrong_ref = replace(ref, admitted_body_command=command)
    else:
        value = replace(value, root_world_z_m=torch.full((2,), float('nan')))
    with pytest.raises(ValueError):
        reward.observe(value, wrong_ref)
    assert reward._window['steps'] == 0
    reward.observe(sample(reward, 1), ref)
    reward.observe(sample(reward, 2), ref)
    finish(reward)


def test_incomplete_or_repeated_intervals_cannot_be_published_or_reset_early():
    reward = fixture()
    ref = begin(reward)
    reward.observe(sample(reward, 1), ref)
    with pytest.raises(RuntimeError, match='every declared'):
        finish(reward)
    with pytest.raises(RuntimeError, match='PRE-reset'):
        reward.reset(torch.ones(2, dtype=torch.bool), reward._time.clone(), reward._episode + 1, reward._control + 1)
    with pytest.raises(ValueError):
        reward.observe(sample(reward, 1), ref)
    reward.observe(sample(reward, 2), ref)
    with pytest.raises(RuntimeError, match='capacity'):
        reward.observe(sample(reward, 3), ref)
    finish(reward)


def test_reward_keeps_actual_native_tick_timestamp_instead_of_accumulating_a_different_float_clock():
    reward = fixture()
    for step_index in range(6):
        ref = begin(reward)
        for substep in (1, 2):
            value = sample(reward, substep)
            actual = torch.full((2,), (step_index * 2 + substep) * .005, dtype=torch.float64)
            reward.observe(replace(value, timestamp_s=actual), ref)
        result = finish(reward)
        assert torch.equal(result.timestamp_s, actual)
    now = torch.full((2,), 12 * .005, dtype=torch.float64)
    reward.reset(torch.ones(2, dtype=torch.bool), now, torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    assert torch.equal(reward._time, now)
    with pytest.raises(ValueError, match='rewinding'):
        reward.reset(torch.ones(2, dtype=torch.bool), now - .001, torch.full((2,), 2, dtype=torch.long),
                     torch.full((2,), 2, dtype=torch.long))


def test_reused_native_input_and_returned_statistics_do_not_alias_future_reward_state():
    reward = fixture()
    ref = begin(reward)
    for index in (1, 2):
        value = sample(reward, index)
        reward.observe(value, ref)
        value.velocity_origin_body_m_s.fill_(777)
    result = finish(reward)
    expected = result.physical_step_reward.clone()
    result.raw_cost_densities['effort'].fill_(1000)
    step(reward)
    torch.testing.assert_close(result.physical_step_reward, expected)
    assert result.raw_error_means['forward_m_s'].eq(0).all()

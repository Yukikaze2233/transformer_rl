"""Preparation/control ownership with synthetic buffers,never native physics."""
from dataclasses import replace
import math

import pytest
import torch

from test_control_reward import fixture as reward_fixture, sample, reference, finish
from test_native_task import fixture as task_fixture, advance
from test_physical_observation import fixture as physical_fixture
from test_structured_collection import model as small_model
from transformer_rl.formal.control.ownership import ControlSource, validate_issued_control
from transformer_rl.formal.control.actuation import OutputFeedbackController, PairTargetDomain
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.collection import StructuredCollector


def test_exact_script_hold_and_policy_target_admission_are_independent():
    controller = OutputFeedbackController((0.,)*6, 3, pair_domain=None)
    q = torch.zeros(3, 6)
    q[0, :4] = 2*math.pi
    q[1, :4] = math.pi
    q[2, 1] = 2*math.pi
    policy = controller.reset_hold_action(q, position_tolerance_rad=1e-6)
    hold = controller.preparation_hold(q, position_tolerance_rad=1e-6)
    assert policy.admitted.tolist() == [True, False, False]
    assert hold.admitted.tolist() == [True, True, False]
    assert torch.equal(hold.leg_position_rad, q[:, :4])
    assert hold.requested_action[1, :4].abs().min() > 12
    assert not hold.requested_action[:, 4:].any()
    assert controller.action_bounds.tolist() == [3., 3., 3., 3., 9., 9.]
    assert controller.effort_bounds.tolist() == [40., 40., 40., 40., 4.5, 4.5]
    limited = OutputFeedbackController((0.,)*6, 1,
                                       pair_domain=PairTargetDomain(((-.1, .1), (-.1, .1)), (1, 1)))
    q = torch.zeros(1, 6);q[0, 1] = .5
    rejected = limited.preparation_hold(q, position_tolerance_rad=1e-6)
    assert not rejected.admitted[0] and not rejected.pair_admitted[0, 0]
    assert rejected.leg_position_rad[0, 1] == .5


def test_source_metadata_cannot_expand_the_policy_action_domain_or_add_wheel_position_commands():
    action = torch.zeros(2, 6);action[0, :4] = 12.
    source = torch.tensor([int(ControlSource.PREPARATION_HOLD), int(ControlSource.POLICY)])
    validate_issued_control(action, source, num_envs=2, device='cpu', policy_owned=torch.tensor([False, True]))
    with pytest.raises(ValueError, match='ownership'):
        validate_issued_control(action, source, num_envs=2, device='cpu', policy_owned=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match='bounds'):
        validate_issued_control(action, torch.zeros(2, dtype=torch.long), num_envs=2, device='cpu')
    action[0, 4] = .1
    with pytest.raises(ValueError, match='hold'):
        validate_issued_control(action, source, num_envs=2, device='cpu')
    action[0, 4], action[0, 0] = 0, 6*math.pi + .01
    with pytest.raises(ValueError, match='hold'):
        validate_issued_control(action, source, num_envs=2, device='cpu')


def test_actor_and_clean_critic_keep_the_real_prior_script_command_without_an_extra_feature():
    _, _, assembler, physical, command = physical_fixture()
    state = assembler.measure(physical)
    previous = torch.zeros(2, 6);previous[0, :4] = 12.
    command = replace(command, previous_issued_action=previous,
                       previous_control_source=torch.tensor([2, 0]))
    actor = assembler.actor(assembler.clean_sensors(state), command, now_s=state.timestamp_s)
    critic = assembler.critic(command, state)
    assert actor.shape == (2, 35) and critic.shape == (2, 81)
    assert torch.equal(actor[:, 22:28], previous) and torch.equal(critic[:, :35], actor)
    assert not actor[:, 14:16].any()
    with pytest.raises(ValueError):
        assembler.actor(assembler.clean_sensors(state), replace(command, previous_control_source=None), now_s=state.timestamp_s)
    with pytest.raises(ValueError):
        assembler.actor(assembler.clean_sensors(state), replace(command, previous_control_source=torch.zeros(2, dtype=torch.long)),
                         now_s=state.timestamp_s)


def test_coaxial_observation_keeps_relative_phase_when_one_axis_crosses_the_angular_cut():
    _, binding, assembler, physical, command = physical_fixture()
    active = list(binding.policy_joint_indices)
    q = physical.joint_position_rad.clone()
    displacement = torch.tensor([[math.pi - .001, math.pi - .401, -math.pi + .001, -math.pi + .401],
                                 [math.pi + .001, math.pi - .399, -math.pi + .003, -math.pi + .403]])
    q[:, active[:4]] = torch.tensor(binding.nominal_policy_position_rad[:4]) + displacement
    state = assembler.measure(replace(physical, joint_position_rad=q))
    frame = assembler.actor(assembler.clean_sensors(state), command, now_s=state.timestamp_s)
    torch.testing.assert_close(frame[:, 10:14], displacement, atol=1e-6, rtol=0)
    torch.testing.assert_close(frame[1, 10:14] - frame[0, 10:14], torch.full((4,), .002), atol=1e-6, rtol=0)
    q[:, active[:2]] += 2*math.pi
    q[:, active[2:4]] -= 2*math.pi
    shifted = assembler.measure(replace(physical, joint_position_rad=q))
    other = assembler.actor(assembler.clean_sensors(shifted), command, now_s=shifted.timestamp_s)
    torch.testing.assert_close(other[:, 10:14], frame[:, 10:14], atol=2e-6, rtol=0)


def hold_task(period=.005):
    env, robot, sim, writes, resets = task_fixture(period)
    source = env.reset_source
    controller = env.lifecycle.plant.controller
    active = list(env.lifecycle.plant.scene.observation.binding.policy_joint_indices)

    def prepared(*args):
        plan = source(*args)
        q = plan.joint_position.clone()
        q[:, active[:4]] += 2.4
        hold = controller.preparation_hold(q[:, active], position_tolerance_rad=1e-6)
        assert hold.admitted.all()
        commands = replace(plan.commands, base_mode=torch.full((2,), 3, dtype=torch.long))
        return replace(plan, joint_position=q, commands=commands, preparation_hold=hold)

    env.reset_source = prepared
    return env, robot, sim, writes, resets


@pytest.mark.parametrize('period', [.005, .001])
def test_five_real_frames_hold_the_birth_outputs_and_handoff_to_the_same_bounded_policy(period):
    env, robot, sim, _, _ = hold_task(period)
    env.reset()
    life, plant = env.lifecycle, env.lifecycle.plant
    active = list(plant.scene.observation.binding.policy_joint_indices)
    born = robot.data.joint_pos.torch[:, active[:4]].clone()
    for index in range(5):
        result = advance(env)
        assert not result.fully_policy_owned.any() and not result.reward.any()
        assert torch.equal(plant.targets.admitted_leg_position_rad, born)
        assert plant.targets.control_source.eq(int(ControlSource.PREPARATION_HOLD)).all()
        assert plant.targets.issued_action[:, :4].abs().min() > 3
        assert torch.equal(result.observation.frame[:, 22:28], plant.targets.issued_action)
        assert not plant.targets.wheel_velocity_rad_s.any()
        assert sim.steps == (index + 1)*life.substeps
    assert result.observation.policy_allowed.all() and result.handoff_after.all()
    result = advance(env)
    assert result.fully_policy_owned.all() and plant.targets.control_source.eq(0).all()
    assert torch.equal(plant.targets.issued_action, torch.ones(2, 6))
    assert plant._last_tick_packet.feedback.effort_bounds[:, :4].eq(40).all()
    assert plant._last_tick_packet.feedback.effort_bounds[:, 4:].eq(4.5).all()


def test_partial_reset_preserves_the_other_rows_hold_mode_and_actual_control_history():
    env, _, _, _, _ = hold_task()
    env.reset()
    for _ in range(6):advance(env)
    life = env.lifecycle
    previous = life.previous_issued[1].clone()
    held = life._hold_position[1].clone()
    env._reset_rows(torch.tensor([True, False]))
    assert torch.equal(life.previous_issued[1], previous)
    assert torch.equal(life._hold_position[1], held)
    result = advance(env)
    assert result.fully_policy_owned.tolist() == [False, True]
    assert life.previous_control_source.tolist() == [2, 0]


def test_safety_transfer_latches_the_current_pose_and_waits_for_explicit_policy_return():
    env, robot, sim, _, _ = hold_task()
    env.reset()
    for _ in range(5):advance(env)
    active = list(env.lifecycle.plant.scene.observation.binding.policy_joint_indices)
    birth = env.lifecycle._hold_position.clone()
    step = sim.step

    def moved(*, render):
        robot.data.joint_pos.torch[:, active[:4]] += .01
        step(render=render)

    sim.step = moved
    advance(env)
    sim.step = step
    now = robot.data.joint_pos.torch[:, active[:4]].clone()
    mask = torch.tensor([True, False])
    env.lifecycle.enter_script(mask)
    env._return_allowed[0] = False
    assert torch.equal(env.lifecycle._hold_position[0], now[0])
    assert torch.equal(env.lifecycle._hold_position[1], birth[1])
    for _ in range(6):
        result = advance(env)
        assert result.observation.policy_allowed.tolist() == [False, True]
        assert torch.equal(env.lifecycle.plant.targets.admitted_leg_position_rad[0], now[0])
    env._return_allowed[0] = True
    result = advance(env)
    assert result.observation.policy_allowed.all() and result.handoff_after.tolist() == [True, False]


@pytest.mark.parametrize('fault', ['target_mutation', 'double_representation', 'single_axis_turn'])
def test_invalid_preparation_plans_fail_before_native_reset_writes(fault):
    env, _, sim, writes, _ = hold_task()
    source = env.reset_source

    def broken(*args):
        plan = source(*args)
        if fault == 'target_mutation':plan.preparation_hold.leg_position_rad[0, 0] += .01
        elif fault == 'double_representation':plan = replace(plan, preparation_issued_action=torch.zeros(2, 6))
        else:
            q = plan.joint_position.clone();active = list(env.lifecycle.plant.scene.observation.binding.policy_joint_indices)
            q[0, active[1]] += 2*math.pi
            hold = env.lifecycle.plant.controller.preparation_hold(q[:, active], position_tolerance_rad=1e-6)
            plan = replace(plan, joint_position=q, preparation_hold=hold)
        return plan

    env.reset_source = broken
    with pytest.raises(ValueError):env.reset()
    assert not writes and sim.steps == 0


def test_mutated_latched_hold_is_refused_without_replaying_or_advancing_physics():
    env, _, sim, _, _ = hold_task()
    env.reset()
    env.lifecycle._hold_position[0, 0] += .1
    with pytest.raises(RuntimeError, match='ownership'):
        advance(env)
    assert sim.steps == 0


def test_script_reward_cannot_be_credited_as_policy_reward_or_expand_policy_begin_bounds():
    composer = reward_fixture();ref = reference();issued = torch.zeros(2, 6);issued[:, :4] = 12.
    kwargs = dict(now_s=composer._time.clone(), episode_id=composer._episode.clone(),
                  control_epoch=composer._control.clone(), admitted_body_command=ref.admitted_body_command,
                  command_mode=ref.mode)
    with pytest.raises(ValueError):
        composer.begin(issued, torch.zeros(2, 6), policy_owned=torch.ones(2, dtype=torch.bool), **kwargs)
    with pytest.raises(ValueError):
        composer.begin(issued, torch.zeros(2, 6), policy_owned=torch.ones(2, dtype=torch.bool),
                         control_source=torch.full((2,), 2, dtype=torch.long), **kwargs)
    composer.begin(issued, torch.zeros(2, 6), policy_owned=torch.zeros(2, dtype=torch.bool),
                     control_source=torch.full((2,), 2, dtype=torch.long), **kwargs)
    for step in range(1, composer.substeps + 1):composer.observe(sample(composer, step), ref)
    result = finish(composer)
    assert not result.learning_reward.any() and not result.fully_policy_owned.any()


def test_collector_and_PPO_masks_keep_unapplied_samples_separate_from_the_real_preparation_command():
    env, _, sim, _, _ = hold_task()
    model = small_model()
    collector = StructuredCollector(env, model);collector.reset();batch = collector.collect(8)
    actor = batch.masks.actor.reshape(8, 2)
    assert not actor[:5].any() and actor[5:].all()
    assert (batch.issued_action.abs() <= batch.issued_action.new_tensor((3, 3, 3, 3, 9, 9))).all()
    assert sim.steps == 16 and collector.last_metrics['policy_owned_transitions'] == 6
    trainer = StructuredPPOTrainer(model, replace(StructuredPPOConfig(), epochs=1, minibatches=1,
                                                  probe_endpoints=16, evaluation_chunk=16, learning_rate=1e-6))
    trainer.update(batch)
    assert trainer.update_index == 1 and trainer.consumed['fresh_endpoints'] == 16

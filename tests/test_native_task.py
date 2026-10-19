"""Full native task composition with synthetic SDK buffers on Kaiser only."""
from dataclasses import replace

import pytest
import torch

from test_native_episode import fixture as episode_fixture
from test_native_reward import fixture as reward_fixture
from test_structured_collection import model as small_model
from transformer_rl._command_trajectory import CommandRates, CommandStream, CommandTrajectory, DriveEnvelope
from transformer_rl._jump_reference import JumpReferenceProgram
from transformer_rl._native_episode import NativeEpisodeLifecycle, TerrainRebuildRequired
from transformer_rl._native_ground import NativeGroundQuery, WheelCollisionCentres
from transformer_rl._native_task import (
    FailureReason, NativePreparationError, NativeTaskEnvironment, NativeTaskLimits, NativeTaskResetPlan,
)
from transformer_rl._policy_optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl._skill_evidence import PhysicalJumpEvidence, SkillEvidenceConfig
from transformer_rl._structured_collection import StructuredCollector
from transformer_rl._task_commands import RequestInputs, TaskCommandPlan, TaskCommandProgram
from transformer_rl._task_protocol import CompletedEstimate, RequestResponse


def fixture(period=.005, *, horizons=(.2, .2), jump=False, trace_sink=None):
    old, robot, sim, writes = episode_fixture(period)
    plant, binding = old.plant, old.plant.scene.observation.binding
    for name in ('L_spring_slide', 'R_spring_slide'):
        robot.data.joint_pos.torch[:, binding.native_joint_names.index(name)] = .0176
    reward, *_ = reward_fixture(period, native_plant=(plant, robot, sim, []), initialize=False)
    names = tuple(binding.native_body_names[i] for i in binding.wheel_body_indices)
    urdf = '<robot>' + ''.join('<link name="' + name + '"><collision><origin xyz="0 0 0"/>'
                               '<geometry><cylinder radius=".06" length=".025"/></geometry>'
                               '</collision></link>' for name in names) + '</robot>'
    wheels = WheelCollisionCentres.from_urdf(urdf, binding)
    # Synthetic interface evidence,not actual native cooking qualification.
    plant.scene.readback.update(selected_urdf_sha256=wheels.urdf_sha256,
                                native_wheel_collision_equivalence_verified=True)
    ground = NativeGroundQuery(plant.scene, old.ground.courses, wheels)
    plant._ground = ground
    life = NativeEpisodeLifecycle(plant, ground, old.protocol)
    initial = torch.tensor([[0., 0., 0., .305]]).repeat(2, 1)
    stream = CommandStream(initial, DriveEnvelope(.06, .4373), CommandRates(2., 3., .2))
    program = TaskCommandProgram(life.protocol, stream, JumpReferenceProgram())
    skills = PhysicalJumpEvidence(life.protocol, SkillEvidenceConfig(physics_dt_s=period))
    limits = NativeTaskLimits(.1, 0., .5, 2.5, ('base_link',), 1000., .015, .02, .001, .5)
    template = tuple(getattr(robot.data, name).torch.clone() for name in
                     ('root_link_pose_w', 'root_com_vel_w', 'joint_pos', 'joint_vel'))
    resets = []

    def reset_source(mask, episode, now):
        resets.append((mask.clone(), episode.clone(), now.clone()))
        target = initial.clone()
        target[:, 0], target[:, 3] = 1., .36
        plan = TaskCommandPlan(CommandTrajectory(initial, target, torch.ones(2), torch.ones(2, dtype=torch.long)),
                                torch.zeros(2, dtype=torch.long), torch.tensor(horizons, dtype=torch.float64),
                                torch.full((2,), .1 if jump else 0.), torch.full((2,), .1, dtype=torch.float64))
        return NativeTaskResetPlan(*(v.clone() for v in template), life.course_sha256, plan)

    env = NativeTaskEnvironment(life, reward, skills, program, limits, reset_source=reset_source,
                                 sensor_source=plant.scene.observation.assembler.clean_sensors,
                                 preparation_action=torch.zeros(2, 6), trace_sink=trace_sink)
    return env, robot, sim, writes, resets


def estimate(env, *, valid=True, contacts=1., available_at=None):
    now = env.lifecycle.plant.scene.observation.timestamp_s()
    return CompletedEstimate(torch.zeros(2), torch.full((2, 2), contacts), torch.zeros(2), torch.zeros(2),
                               now.clone() if available_at is None else available_at,
                               torch.full((2,), valid, dtype=torch.bool))


def advance(env, **kwargs):
    return env.step(torch.ones(2, 6), estimate(env, **kwargs))


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield


@pytest.mark.parametrize('period', [.005, .001])
def test_composed_native_task_advances_exact_physics_then_one_reward_and_five_frame_handoff(period):
    env, robot, sim, _, _ = fixture(period)
    observation = env.reset()
    assert not observation.policy_allowed.any() and observation.critic.isnan().all() and sim.steps == 0
    assert not observation.labels.valid.any()
    for i in range(5):
        result = advance(env)
        assert sim.steps == (i + 1) * env.lifecycle.substeps
        assert not result.fully_policy_owned.any() and not result.reward.any()
    assert result.observation.policy_allowed.all() and result.handoff_after.all()
    result = advance(env)
    assert result.fully_policy_owned.all() and torch.isfinite(result.reward).all()
    assert not result.handoff_after.any()
    assert env.last_outcome['physical_tick'] == 6 * env.lifecycle.substeps
    assert not env.last_outcome['formal_qualification']
    assert torch.equal(result.observation.frame[:, 22:28], torch.ones(2, 6))


def test_time_limit_uses_full_policy_owned_horizon_and_preserves_pre_reset_critic_and_labels():
    env, _, sim, _, resets = fixture(horizons=(.04, .2))
    env.reset()
    for _ in range(8):
        result = advance(env)
        assert not result.truncated.any()
    result = advance(env)
    assert result.truncated.tolist() == [True, False] and not result.terminated.any()
    assert result.observation.episode_id.tolist() == [1, 0]
    assert result.final_labels.episode_id.tolist() == [0, 0]
    assert result.final_critic_valid.all() and torch.isfinite(result.final_critic).all()
    assert result.observation.critic[0].isnan().all() and not result.observation.labels.valid[0]
    assert result.final_labels.valid.all() and result.final_labels.timestamp_s.eq(.09).all()
    assert sim.steps == 18 and resets[-1][0].tolist() == [True, False]


def test_current_command_is_held_for_all_reward_substeps_and_next_frame_gets_only_the_next_command():
    env, _, _, _, _ = fixture()
    env.reset()
    records = []
    observe = env.rewards.composer.observe

    def record(sample, reference):
        records.append(reference.admitted_body_command.clone())
        return observe(sample, reference)

    env.rewards.composer.observe = record
    old = env.commands.stream.current.clone()
    result = advance(env)
    assert len(records) == 2 and all(torch.equal(v, old) for v in records)
    assert result.observation.frame[:, 0].gt(old[:, 0]).all()
    assert env.commands.last_admission.requested[:, 0].gt(0).all()


def test_completed_estimate_changes_only_next_frame_request_and_visible_height_is_latched():
    env, _, _, _, _ = fixture(jump=True)
    env.reset()
    for _ in range(9):
        advance(env)
    before = env.commands.stream.current[:, 3].clone()
    before_mode = []
    original = env.rewards.composer.observe

    def record(sample, ref):
        before_mode.append(ref.mode.clone())
        return original(sample, ref)

    env.rewards.composer.observe = record
    result = advance(env)
    assert env.last_outcome['request_condition'].response.eq(int(RequestResponse.ACCEPTED)).all()
    assert all(v.eq(0).all() for v in before_mode)
    assert result.observation.frame[:, 32].eq(1).all()  # Jump one-hot.
    torch.testing.assert_close(env.commands.stream.current[:, 3], before)
    assert result.observation.frame[:, 34].eq(0).all()
    advance(env)
    torch.testing.assert_close(env.commands.stream.current[:, 3], before)


def test_substep_failure_stays_latched_even_if_endpoint_recovers_and_reset_keeps_old_reason():
    env, robot, sim, _, _ = fixture()
    env.reset()
    for _ in range(5):
        advance(env)
    step = sim.step

    def fall_then_recover(*, render):
        step(render=render)
        if sim.steps == 11:
            robot.data.root_link_pose_w.torch[0, 2] -= .25
        elif sim.steps == 12:
            robot.data.root_link_pose_w.torch[0, 2] += .25

    sim.step = fall_then_recover
    result = advance(env)
    assert result.terminated.tolist() == [True, False] and result.fully_policy_owned.all()
    assert env.last_outcome['failure_reasons'][0] & int(FailureReason.LOW_SUPPORT_HEIGHT)
    assert env.last_outcome['reward'].event_rewards['failure'][0] == -4
    assert result.observation.episode_id.tolist() == [1, 0]
    assert env.conditions.failure_reasons.eq(0).all()


def test_safety_transfer_stops_learning_credit_and_requires_explicit_release_with_new_sensor_window():
    env, _, sim, _, _ = fixture()
    env.reset()
    for _ in range(5):
        advance(env)
    incoming = replace(RequestInputs.empty(2, env.device), request_id=torch.tensor([5, -1]),
                        requested_height_m=torch.tensor([.1, 0.]))
    env.queue_requests(incoming)
    advance(env)
    assert env.lifecycle.protocol.active_id.tolist() == [5, -1]
    inputs = replace(RequestInputs.empty(2, env.device), safety_abort=torch.tensor([True, False]))
    env.queue_requests(inputs)
    first = advance(env)
    assert first.handoff_after.tolist() == [True, False]
    assert first.fully_policy_owned.tolist() == [False, True]
    assert env.lifecycle.previous_issued[0].eq(0).all()
    assert first.observation.policy_allowed.tolist() == [False, True]
    for _ in range(5):
        last = advance(env)
        assert not last.fully_policy_owned[0] and not last.observation.policy_allowed[0]
    env.queue_requests(replace(RequestInputs.empty(2, env.device), release_script=torch.tensor([True, False])))
    result = advance(env)
    assert result.handoff_after.tolist() == [True, False] and result.observation.policy_allowed.all()
    assert sim.steps == 26


def test_ordinary_safety_abort_takes_over_before_physics_without_an_active_jump():
    env, _, sim, _, _ = fixture()
    env.reset()
    for _ in range(5):
        advance(env)
    assert env.lifecycle.protocol.active_id.eq(-1).all()
    env.queue_requests(replace(RequestInputs.empty(2, env.device), safety_abort=torch.tensor([True, False])))
    result = advance(env)
    assert sim.steps == 12
    assert result.fully_policy_owned.tolist() == [False, True]
    assert result.handoff_after.tolist() == [True, False]
    assert result.reward[0] == 0 and result.observation.policy_allowed.tolist() == [False, True]
    assert env.lifecycle.previous_issued[0].eq(0).all() and env.lifecycle.previous_issued[1].eq(1).all()


def test_protocol_safety_after_physics_retains_actual_endpoint_before_script_projection():
    env, _, sim, _, _ = fixture()
    env.reset()
    for _ in range(5):
        advance(env)
    env.queue_requests(replace(RequestInputs.empty(2, env.device), request_id=torch.tensor([4, -1]),
                                requested_height_m=torch.tensor([.1, 0.])))
    advance(env)
    result = advance(env, valid=False)
    assert result.fully_policy_owned.all() and result.handoff_after.tolist() == [True, False]
    assert result.observation.critic[0].isnan().all()
    actual = env.last_outcome['pre_reset_observation']
    assert torch.isfinite(actual.critic[0]).all() and actual.labels.valid[0]
    assert result.final_labels.valid.all() and sim.steps == 14


def test_script_assistance_does_not_pause_or_refund_the_existing_episode_deadline():
    env, _, sim, _, _ = fixture(horizons=(.08, .4))
    env.reset()
    for _ in range(5):
        advance(env)
    env.queue_requests(replace(RequestInputs.empty(2, env.device), safety_abort=torch.tensor([True, False])))
    advance(env)
    for _ in range(6):
        result = advance(env)
        assert not result.truncated.any()
    result = advance(env)
    assert sim.steps == 26 and result.truncated.tolist() == [True, False]
    assert not result.fully_policy_owned[0] and result.observation.episode_id.tolist() == [1, 0]
    assert env.last_outcome['horizon_elapsed_s'][0] == pytest.approx(.08)


def test_policy_return_retains_the_original_deadline_instead_of_starting_a_new_horizon():
    env, _, _, _, _ = fixture(horizons=(.09, .4))
    env.reset()
    for _ in range(5):
        advance(env)
    env.queue_requests(replace(RequestInputs.empty(2, env.device), safety_abort=torch.tensor([True, False])))
    advance(env)
    for _ in range(5):
        advance(env)
    env.queue_requests(replace(RequestInputs.empty(2, env.device), release_script=torch.tensor([True, False])))
    result = advance(env)
    assert result.observation.policy_allowed.all()
    advance(env)
    result = advance(env)
    assert result.truncated.tolist() == [True, False]
    assert env.last_outcome['horizon_elapsed_s'][0] == pytest.approx(.09)


def test_changed_terrain_plan_fails_before_initial_physics_or_any_native_reset_write():
    env, _, sim, writes, _ = fixture()
    source = env.reset_source
    env.reset_source = lambda *args: replace(source(*args), course_sha256=('0' * 64, env.lifecycle.course_sha256[1]))
    with pytest.raises(TerrainRebuildRequired):
        env.reset()
    assert sim.steps == 0 and writes == []


def test_invalid_sensor_source_faults_the_consumed_interval_instead_of_replaying_or_changing_budget():
    env, _, sim, _, _ = fixture()
    env.reset()
    env.sensor_source = lambda state: None
    with pytest.raises(TypeError, match='sensor source'):
        advance(env)
    assert sim.steps == 2
    with pytest.raises(RuntimeError, match='faulted'):
        advance(env)
    assert sim.steps == 2


@pytest.mark.parametrize('period', [.005, .001])
def test_trace_sink_receives_every_actual_substep_with_current_commands_without_extra_physics(period):
    records = []

    def sink(tick, reward, condition, reference):
        records.append((tick.physical_tick, reward.physics.timestamp_s.clone(), condition.true_failure.clone(),
                         reference.admitted_body_command.clone(), reward.physics.policy_owned.clone()))

    env, _, sim, _, _ = fixture(period, trace_sink=sink)
    env.reset()
    current = env.commands.stream.current.clone()
    advance(env)
    assert len(records) == sim.steps == env.lifecycle.substeps
    assert [v[0] for v in records] == list(range(1, env.lifecycle.substeps + 1))
    assert all(torch.equal(v[3], current) and not v[4].any() for v in records)
    assert records[-1][1].eq(.01).all()


def test_trace_sink_failure_is_a_nonreplayable_partially_consumed_interval():
    def broken(*args):
        raise OSError('synthetic trace storage unavailable')

    env, _, sim, _, _ = fixture(trace_sink=broken)
    env.reset()
    with pytest.raises(OSError, match='trace storage'):
        advance(env)
    assert sim.steps == 1
    with pytest.raises(RuntimeError, match='faulted'):
        advance(env)
    assert sim.steps == 1


def test_preparation_timeout_is_a_data_readiness_error_instead_of_a_policy_fall_or_refunded_reset():
    env, _, sim, _, resets = fixture()
    env.reset()
    source = env.sensor_source
    env.sensor_source = lambda state: replace(source(state), sampled_at_s=torch.zeros(2, 4, dtype=torch.float64))
    for _ in range(49):
        result = advance(env)
        assert not result.fully_policy_owned.any() and not result.terminated.any()
    with pytest.raises(NativePreparationError):
        advance(env)
    assert sim.steps == 100 and len(resets) == 1
    with pytest.raises(RuntimeError, match='faulted'):
        advance(env)
    assert sim.steps == 100


def test_collector_and_ppo_use_the_composed_native_task_with_correct_reset_masks():
    env, _, sim, _, _ = fixture(horizons=(.04, .2))
    model = small_model()
    collector = StructuredCollector(env, model)
    collector.reset()
    batch = collector.collect(12)
    assert sim.steps == 24 and collector.returned_vector_steps == 12
    assert batch.masks.actor.reshape(12, 2)[:, 0].tolist() == [False] * 5 + [True] * 4 + [False] * 3
    assert batch.masks.actor.sum() == 11
    trainer = StructuredPPOTrainer(model, replace(StructuredPPOConfig(), epochs=1, minibatches=1,
                                                  probe_endpoints=16, evaluation_chunk=24, learning_rate=1e-6))
    trainer.update(batch)
    assert trainer.update_index == 1 and trainer.consumed['fresh_endpoints'] == 24


def test_collector_masks_an_immediate_safety_override_instead_of_crediting_the_unapplied_actor_action():
    env, _, sim, _, _ = fixture()
    model = small_model()
    collector = StructuredCollector(env, model)
    collector.reset()
    original, calls = env.step, []

    def scheduled(issued, estimate):
        if len(calls) == 5:
            env.queue_requests(replace(RequestInputs.empty(2, env.device), safety_abort=torch.tensor([True, False])))
        calls.append(True)
        return original(issued, estimate)

    env.step = scheduled
    batch = collector.collect(10)
    actor = batch.masks.actor.reshape(10, 2)
    assert actor[:, 0].tolist() == [False] * 10
    assert actor[:, 1].tolist() == [False] * 5 + [True] * 5
    assert sim.steps == 20 and collector.returned_vector_steps == 10


def test_bad_command_plan_is_rejected_without_reseeding_or_physical_progress():
    env, _, sim, writes, _ = fixture()
    with pytest.raises(ValueError, match='reseeds'):
        env.reset(seed=88)
    assert sim.steps == 0 and writes == []

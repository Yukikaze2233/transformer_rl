"""Native episode/reset ownership with synthetic buffers, not an SDK scene."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_plant import fixture as plant_fixture
from test_structured_collection import model as small_model
from transformer_rl._native_episode import NativeEpisodeLifecycle, TerrainRebuildRequired
from transformer_rl._native_ground import NativeGroundQuery, WheelCollisionCentres
from transformer_rl._physical_observation import CommandFeatures
from transformer_rl._policy_optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl._structured_collection import LabeledStep, StructuredCollector
from transformer_rl._task_protocol import CompletedEstimate, RequestProtocol, RequestResponse
from transformer_rl._terrain_geometry import TerrainKind
from transformer_rl._terrain_tasks import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning


def fixture(period=.005):
    plant, robot, sim, trace = plant_fixture(period)
    generator = ContinuousTerrainGenerator(2101, TerrainTuning(.8, (.2, 1.1), .25))
    request = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-3., 3.), (-3., 3.)))
    courses = tuple(generator.build(request) for _ in range(2))
    plant.scene = replace(plant.scene, terrain_meshes=tuple(course.surface.meshes for course in courses))
    bridge = plant.scene.observation
    poses = torch.zeros(2, 19, 7)
    poses[..., 6] = 1
    robot.data.body_link_pose_w = SimpleNamespace(torch=poses)
    update = robot.update

    def refresh(dt):
        update(dt)
        poses[:, 0].copy_(robot.data.root_link_pose_w.torch)
        for index, body in enumerate(bridge.binding.wheel_body_indices):
            poses[:, body, :3] = robot.data.root_link_pose_w.torch[:, :3]
            poses[:, body, 1] += .2 if index == 0 else -.2

    robot.update = refresh
    writes = []

    def setter(key, value, ids):
        writes.append((key, ids.clone()))
        getattr(robot.data, key).torch[ids] = value

    robot.write_root_link_pose_to_sim_index = lambda *, root_pose, env_ids: setter('root_link_pose_w', root_pose, env_ids)
    robot.write_root_com_velocity_to_sim_index = lambda *, root_velocity, env_ids: setter('root_com_vel_w', root_velocity, env_ids)
    robot.write_joint_position_to_sim_index = lambda *, position, env_ids: setter('joint_pos', position, env_ids)
    robot.write_joint_velocity_to_sim_index = lambda *, velocity, env_ids: setter('joint_vel', velocity, env_ids)
    robot.reset = lambda ids: writes.append(('reset', ids.clone()))
    names = tuple(bridge.binding.native_body_names[i] for i in bridge.binding.wheel_body_indices)
    wheels = WheelCollisionCentres(names, ((0., 0., .02035), (0., 0., -.02035)), (.06, .06), (.025, .025))
    ground = NativeGroundQuery(plant.scene, courses, wheels)
    plant._ground = ground
    lifecycle = NativeEpisodeLifecycle(plant, ground, RequestProtocol(2))
    return lifecycle, robot, sim, writes


def reset(lifecycle, robot, mask=None, **changes):
    mask = torch.ones(2, dtype=torch.bool) if mask is None else mask
    values = [getattr(robot.data, name).torch.clone() for name in
              ('root_link_pose_w', 'root_com_vel_w', 'joint_pos', 'joint_vel')]
    return lifecycle.reset(mask, *values, course_sha256=changes.get('course_sha256', lifecycle.course_sha256))


def observe(lifecycle, tick, *, delay_s=0.):
    assembler = lifecycle.plant.scene.observation.assembler
    sensors = assembler.clean_sensors(tick.measured)
    sensors = replace(sensors, sampled_at_s=sensors.sampled_at_s - delay_s)
    commands = CommandFeatures(torch.tensor([[0., 0., 0., .305]]).repeat(2, 1),
                               torch.zeros(2, dtype=torch.long), torch.zeros(2),
                               tick.measured.timestamp_s.float(), lifecycle.previous_issued.clone())
    return lifecycle.observe(tick, sensors, commands)


def advance(lifecycle, *, delay_s=0., policy=None):
    policy = torch.ones(2, 6) if policy is None else policy
    tick = lifecycle.advance(policy, script_issued=torch.zeros(2, 6))
    return observe(lifecycle, tick, delay_s=delay_s)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize('period', [.005, .001])
def test_initial_reset_has_no_fake_physics_or_sensor_truth_and_waits_five_real_frames(period):
    lifecycle, robot, sim, writes = fixture(period)
    packet = reset(lifecycle, robot)
    assert packet.pre_reset is None and not packet.projection_is_measurement
    assert sim.steps == packet.physical_tick == 0 and len(writes) == 5
    assert packet.observation.episode_id.tolist() == [0, 0]
    assert not packet.observation.policy_allowed.any() and not packet.observation.labels.valid.any()
    assert packet.observation.critic.isnan().all() and torch.isfinite(packet.observation.frame).all()
    with pytest.raises(RuntimeError, match='refresh'):
        lifecycle.plant.scene.observation.before_state_read()
    for index in range(5):
        endpoint = advance(lifecycle, delay_s=.001)
        assert endpoint.observation.policy_allowed.all().item() == (index == 4)
        assert endpoint.handoff_after.all().item() == (index == 4)
        assert not lifecycle.previous_issued.any()  # Warmup used the explicit script, not policy output.
    assert sim.steps == 5 * lifecycle.substeps and endpoint.observation.control_epoch.tolist() == [2, 2]
    endpoint = advance(lifecycle)
    assert lifecycle.previous_issued.eq(1).all() and not endpoint.handoff_after.any()


def test_selected_reset_retains_owned_pre_reset_endpoint_and_untouched_row_without_time_advance():
    lifecycle, robot, sim, writes = fixture()
    reset(lifecycle, robot)
    for _ in range(6):
        endpoint = advance(lifecycle)
    old = endpoint.observation.owned()
    before_steps, before_time = sim.steps, old.timestamp_s.clone()
    old_drive = lifecycle.plant.drive.state[1].clone()
    lifecycle.protocol.max_seen_id[:] = torch.tensor([9, 8])
    lifecycle.protocol.active_id[:] = torch.tensor([9, 8])
    lifecycle.protocol.height_m[:] = .1
    lifecycle.protocol.last_time_s[:] = before_time
    writes.clear()
    packet = reset(lifecycle, robot, torch.tensor([True, False]))
    assert sim.steps == before_steps and all(ids.tolist() == [0] for _, ids in writes)
    assert packet.aborted_request_ids.tolist() == [9, -1]
    assert lifecycle.protocol.max_seen_id.tolist() == [9, 8] and lifecycle.protocol.active_id.tolist() == [-1, 8]
    assert packet.pre_reset.labels.valid.all() and packet.pre_reset.episode_id.tolist() == [0, 0]
    assert packet.observation.episode_id.tolist() == [1, 0] and packet.observation.control_epoch.tolist() == [3, 2]
    assert packet.observation.policy_allowed.tolist() == [False, True]
    assert packet.observation.labels.valid.tolist() == [False, True]
    torch.testing.assert_close(packet.observation.timestamp_s, before_time)
    torch.testing.assert_close(packet.observation.frame[1], old.frame[1])
    torch.testing.assert_close(packet.observation.critic[1], old.critic[1])
    torch.testing.assert_close(lifecycle.plant.drive.state[1], old_drive)
    assert lifecycle.plant.contact_history.count.tolist() == [0, 3]
    for _ in range(5):
        endpoint = advance(lifecycle)
    assert endpoint.observation.policy_allowed.all() and endpoint.handoff_after.tolist() == [True, False]
    assert endpoint.observation.control_epoch.tolist() == [4, 2]
    # Returned old evidence does not alias the newly reset row or later views.
    torch.testing.assert_close(packet.pre_reset.frame, old.frame)
    torch.testing.assert_close(packet.pre_reset.labels.joint_velocity_rad_s, old.labels.joint_velocity_rad_s)


def test_new_terrain_is_rejected_before_any_native_write_or_epoch_change():
    lifecycle, robot, sim, writes = fixture()
    before = lifecycle.episode_id.clone()
    requested = ('0' * 64, lifecycle.course_sha256[1])
    with pytest.raises(TerrainRebuildRequired) as raised:
        reset(lifecycle, robot, course_sha256=requested)
    assert raised.value.rows == (0,) and raised.value.requested == requested
    assert writes == [] and sim.steps == 0
    torch.testing.assert_close(lifecycle.episode_id, before)


def test_reset_cannot_double_count_empty_episodes_or_initialize_only_half_the_rows():
    lifecycle, robot, sim, _ = fixture()
    with pytest.raises(RuntimeError, match='initialization'):
        reset(lifecycle, robot, torch.tensor([True, False]))
    reset(lifecycle, robot)
    with pytest.raises(RuntimeError, match='already reset'):
        reset(lifecycle, robot)
    assert lifecycle.episode_id.tolist() == [0, 0] and sim.steps == 0


def test_changed_mechanical_epoch_requires_full_reinitialization_and_new_sensor_window():
    lifecycle, robot, _, writes = fixture()
    reset(lifecycle, robot)
    for _ in range(5):
        advance(lifecycle)
    bridge = lifecycle.plant.scene.observation
    bridge.invalidate_model()
    bridge.refresh_model()
    writes.clear()
    with pytest.raises(RuntimeError, match='all episode'):
        reset(lifecycle, robot, torch.tensor([True, False]))
    assert writes == []
    reset(lifecycle, robot)
    assert lifecycle.episode_id.tolist() == [1, 1] and not lifecycle.policy_allowed.any()


def test_a_frame_cannot_be_published_twice_or_replaced_with_an_unowned_physical_tick():
    lifecycle, robot, sim, _ = fixture()
    reset(lifecycle, robot)
    tick = lifecycle.advance(torch.ones(2, 6), script_issued=torch.zeros(2, 6))
    with pytest.raises(ValueError, match='completed physical'):
        observe(lifecycle, replace(tick))
    observe(lifecycle, tick)
    with pytest.raises(ValueError, match='completed physical'):
        observe(lifecycle, tick)
    assert sim.steps == 2 and lifecycle._fresh_frames.tolist() == [1, 1]


def test_unchanged_old_sensor_timestamps_cannot_complete_warmup():
    lifecycle, robot, _, _ = fixture()
    reset(lifecycle, robot)
    for _ in range(6):
        tick = lifecycle.advance(torch.ones(2, 6), script_issued=torch.zeros(2, 6))
        observe(lifecycle, tick, delay_s=float(tick.measured.timestamp_s[0]))
    assert not lifecycle.policy_allowed.any() and lifecycle._fresh_frames.eq(0).all()


def test_preparation_estimates_cannot_accept_a_jump_or_replay_its_consumed_request_after_handoff():
    lifecycle, robot, _, _ = fixture()
    reset(lifecycle, robot)
    allow = torch.ones(2, dtype=torch.bool)
    cancel, abort = torch.full((2,), -1, dtype=torch.long), torch.zeros(2, dtype=torch.bool)
    for index in range(7):
        tick = lifecycle.advance(torch.zeros(2, 6), script_issued=torch.zeros(2, 6))
        estimate = CompletedEstimate(torch.zeros(2), torch.ones(2, 2), torch.zeros(2), torch.zeros(2),
                                     torch.full((2,), index * .01, dtype=torch.float64), allow.clone())
        # Both preparation and the first ready interval receive the SAME ID.
        ids = torch.full((2,), 1 if index < 6 else 2, dtype=torch.long)
        result = lifecycle.step_requests(estimate, request_id=ids, requested_height_m=torch.full((2,), .1),
                                          cancel_id=cancel, allow_new=allow, safety_abort=abort)
        if index == 0:
            assert result.response.eq(int(RequestResponse.NOT_READY)).all()
        if index < 6:
            assert lifecycle.protocol.active_id.eq(-1).all()
        else:
            assert result.response.eq(int(RequestResponse.ACCEPTED)).all()
        with pytest.raises(RuntimeError, match='one unpublished'):
            lifecycle.step_requests(estimate, request_id=ids, requested_height_m=torch.full((2,), .1),
                                     cancel_id=cancel, allow_new=allow, safety_abort=abort)
        observe(lifecycle, tick)
    assert lifecycle.protocol.max_seen_id.tolist() == [2, 2] and lifecycle.protocol.active_id.tolist() == [2, 2]


def test_partial_physical_failure_faults_the_episode_and_never_replays_that_interval():
    lifecycle, robot, sim, _ = fixture()
    reset(lifecycle, robot)
    sim.fail = True
    with pytest.raises(RuntimeError, match='synthetic'):
        advance(lifecycle)
    sim.fail = False
    with pytest.raises(RuntimeError, match='faulted'):
        advance(lifecycle)
    assert sim.steps == 0


class EpisodeEnv:
    num_envs = 2
    device = torch.device('cpu')

    def __init__(self):
        self.lifecycle, self.robot, self.sim, _ = fixture()
        self.counter = 0

    def reset(self, seed=None):
        return reset(self.lifecycle, self.robot).observation

    def step(self, issued, estimate):
        owned = self.lifecycle.policy_allowed.clone()
        tick = self.lifecycle.advance(issued, script_issued=torch.zeros(2, 6))
        endpoint = observe(self.lifecycle, tick)
        before = endpoint.observation.owned()
        self.counter += 1
        terminated = torch.tensor([self.counter == 6, False])
        observation = reset(self.lifecycle, self.robot, terminated).observation if terminated.any() else before
        return LabeledStep(observation, torch.ones(2), terminated, torch.zeros(2, dtype=torch.bool),
                            before.critic.clone(), before.labels.valid.clone(), before.labels.owned(),
                            owned, endpoint.handoff_after)


def test_reset_at_same_policy_timestamp_preserves_final_labels_and_runs_a_real_collector_ppo_transaction():
    network, env = small_model(), EpisodeEnv()
    collector = StructuredCollector(env, network)
    initial = collector.reset()
    assert not initial.policy_allowed.any() and initial.critic.isnan().all()
    batch = collector.collect(12)
    assert env.sim.steps == 24 and collector.returned_vector_steps == 12
    actor = batch.masks.actor.reshape(12, 2)
    assert actor[:, 0].tolist() == [False] * 5 + [True] + [False] * 5 + [True]
    assert actor[:, 1].tolist() == [False] * 5 + [True] * 7
    assert actor.sum() == 9 and batch.masks.next_auxiliary.sum() == 8
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=1,
                                   probe_endpoints=16, evaluation_chunk=24, learning_rate=1e-6))
    trainer.update(batch)
    assert trainer.update_index == 1 and trainer.consumed['fresh_endpoints'] == 24
    assert trainer.consumed['actor_optimizer_steps'] == trainer.consumed['critic_optimizer_steps'] == 1

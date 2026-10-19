"""Stream reward samples through synthetic native calls, never a real SDK."""
from dataclasses import replace

import pytest
import torch

from test_control_reward import reference
from test_native_episode import fixture as episode_fixture, observe, reset
from test_structured_collection import model as small_model
from transformer_rl.formal.tasks.reward import PhysicalRewardComposer, RewardPhysics, RewardTuning
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.collection import LabeledStep, StructuredCollector


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize('period', [.005, .001])
def test_read_only_stream_receives_every_substep_without_extra_physics_or_rng_consumption(period):
    lifecycle, robot, sim, _ = episode_fixture(period)
    reset(lifecycle, robot)
    seen = []
    before = torch.get_rng_state()

    def consumer(tick):
        seen.append((tick.physical_tick, tick.measured.timestamp_s.clone(), tick.explicit_actuator_buffer))

    tick = lifecycle.advance(torch.ones(2, 6), script_issued=torch.zeros(2, 6), physical_observer=consumer)
    assert [item[0] for item in seen] == list(range(1, lifecycle.substeps + 1))
    assert sim.steps == lifecycle.substeps and seen[-1][0] == tick.physical_tick
    assert torch.equal(before, torch.get_rng_state())
    for index, item in enumerate(seen):
        torch.testing.assert_close(item[1], torch.full((2,), (index + 1) * period, dtype=torch.float64))
    observe(lifecycle, tick)


@pytest.mark.parametrize('change', ['consumer_failure', 'model_change', 'clock_change'])
def test_failed_or_mutating_consumer_faults_after_its_irreversible_substep_and_cannot_replay(change):
    lifecycle, robot, sim, _ = episode_fixture()
    reset(lifecycle, robot)

    def consumer(_):
        if change == 'consumer_failure':
            raise RuntimeError('synthetic reward sample rejected')
        if change == 'model_change':
            lifecycle.plant.scene.observation.refresh_model()
        else:
            lifecycle.plant.step_physics()

    with pytest.raises(RuntimeError):
        lifecycle.advance(torch.zeros(2, 6), script_issued=torch.zeros(2, 6), physical_observer=consumer)
    assert sim.steps == (2 if change == 'clock_change' else 1)
    with pytest.raises(RuntimeError, match='faulted'):
        lifecycle.advance(torch.zeros(2, 6), script_issued=torch.zeros(2, 6))


def test_noncallable_consumer_is_rejected_before_native_writes_or_physics():
    lifecycle, robot, sim, _ = episode_fixture()
    reset(lifecycle, robot)
    with pytest.raises(ValueError, match='consumer'):
        lifecycle.advance(torch.zeros(2, 6), script_issued=torch.zeros(2, 6), physical_observer=object())
    assert sim.steps == 0


class RewardEpisodeEnv:
    num_envs = 2
    device = torch.device('cpu')

    def __init__(self):
        self.lifecycle, self.robot, self.sim, _ = episode_fixture()
        self.reward = PhysicalRewardComposer(self.lifecycle.plant.scene.observation.binding, 2, RewardTuning(20.),
                                             retired_request_ids=self.lifecycle.protocol.max_seen_id)
        self.counter, self.results, self.sample_count = 0, [], 0

    def reset(self, seed=None):
        packet = reset(self.lifecycle, self.robot)
        self.reward.reset(packet.reset_rows, packet.observation.timestamp_s,
                           packet.observation.episode_id, packet.observation.control_epoch)
        return packet.observation

    def step(self, issued, estimate):
        lifecycle, bridge = self.lifecycle, self.lifecycle.plant.scene.observation
        episode, control, owned = lifecycle.episode_id.clone(), lifecycle.control_epoch.clone(), lifecycle.policy_allowed.clone()
        ref = reference()
        actual = torch.where(owned[:, None], issued, 0)
        initial_dq = self.robot.data.joint_vel.torch[:, list(bridge.binding.policy_joint_indices)].clone()
        self.reward.begin(actual, initial_dq, now_s=bridge.timestamp_s(), episode_id=episode, control_epoch=control,
                           policy_owned=owned, admitted_body_command=ref.admitted_body_command, command_mode=ref.mode)

        def consumer(tick):
            state = tick.measured
            q = self.robot.data.root_link_pose_w.torch[:, 3:]
            velocity = state.velocity_origin_body_m_s
            cross = torch.linalg.cross(q[:, :3], velocity)
            world = velocity + 2 * (q[:, 3:] * cross + torch.linalg.cross(q[:, :3], cross))
            gravity = state.gravity_body
            pose = torch.stack((torch.atan2(-gravity[:, 1], -gravity[:, 2]), gravity[:, 0].clamp(-1, 1).asin()), -1)
            # Only synthetic fixture attachments: these are NOT an implemented
            # native contact-point slip or geometric-knee-margin producer.
            sample = RewardPhysics(state.timestamp_s, episode, control, owned,
                                   state.velocity_origin_body_m_s, world,
                                   self.robot.data.root_link_pose_w.torch[:, 2], state.height_above_ground_m,
                                   pose, gravity, state.angular_velocity_body_rad_s, state.policy_velocity_rad_s,
                                   tick.explicit_actuator_buffer['active_output_effort_nm'], tick.contacts.normal_load_n,
                                   torch.zeros(2, 2), torch.zeros(2, 6))
            self.reward.observe(sample, ref)
            self.sample_count += 1

        tick = lifecycle.advance(issued, script_issued=torch.zeros(2, 6), physical_observer=consumer)
        endpoint = observe(lifecycle, tick)
        before = endpoint.observation.owned()
        self.counter += 1
        terminated = torch.tensor([self.counter == 6, False])
        result = self.reward.finish(terminated=terminated, truncated=torch.zeros(2, dtype=torch.bool),
                                    physical_success_request_id=torch.full((2,), -1, dtype=torch.long),
                                    entire_skill_policy_owned=owned)
        self.results.append(result)
        if terminated.any():
            packet = reset(lifecycle, self.robot, terminated)
            observation = packet.observation
            self.reward.reset(terminated, observation.timestamp_s, observation.episode_id, observation.control_epoch)
        else:
            observation = before
        return LabeledStep(observation, result.learning_reward, terminated, torch.zeros(2, dtype=torch.bool),
                            before.critic.clone(), before.labels.valid.clone(), before.labels.owned(),
                            owned, endpoint.handoff_after)


def test_complete_physical_reward_reset_collector_and_ppo_path_keeps_failure_credit_and_masks_preparation():
    network, env = small_model(), RewardEpisodeEnv()
    collector = StructuredCollector(env, network)
    collector.reset()
    batch = collector.collect(12)
    assert env.sample_count == env.sim.steps == 24
    assert batch.masks.actor.sum() == 9
    assert env.results[5].event_rewards['failure_absorbing_value_tail'][0] < 0
    torch.testing.assert_close(batch.returns.reshape(12, 2)[5, 0], env.results[5].learning_reward[0])
    assert all(result.learning_reward.eq(0).all() for result in env.results[:5])
    assert all(result.learning_reward[0] == 0 for result in env.results[6:11])
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=1,
                                   probe_endpoints=16, evaluation_chunk=24, learning_rate=1e-6))
    trainer.update(batch)
    assert trainer.update_index == 1 and trainer.consumed['fresh_endpoints'] == 24
    assert trainer.consumed['actor_optimizer_steps'] == trainer.consumed['critic_optimizer_steps'] == 1

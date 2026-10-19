"""Label/reset/ownership collection checks with a synthetic environment."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.collection import (
    LabeledObservation, LabeledStep, PhysicalLabels, StructuredCollectionConfig, StructuredCollector,
)
from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig
from transformer_rl.formal.tasks.protocol import RequestProtocol


@pytest.fixture(autouse=True)
def seeded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(previous)


def model(*, full_width=False):
    if full_width:
        return StructuredActorCritic()
    return StructuredActorCritic(StructuredPolicyConfig(body_width=8, body_heads=2, body_ffn=16,
                                                       temporal_width=16, temporal_heads=4,
                                                       temporal_layers=1, temporal_ffn=32))


def test_request_vertical_velocity_projects_predicted_body_velocity_using_current_imu_gravity():
    env, network = SyntheticLabeledEnv(), model()
    original = env._observe

    def tilted(previous):
        value = original(previous)
        value.frame[:, 7:10] = torch.tensor([[.2, 0., -(1 - .2**2)**.5], [0., -.3, -(1 - .3**2)**.5]])
        return value

    env._observe = tilted
    output = network.actor.policy.outputs

    def known(history):
        mean, estimate, context = output(history)
        estimate = estimate.clone()
        estimate[:, :3] = torch.tensor([[1., 2., .1], [1., 2., .1]])
        return mean, estimate, context

    network.actor.policy.outputs = known
    collector = StructuredCollector(env, network)
    initial = collector.reset()
    collector.collect(1)
    expected = -(torch.tensor([[1., 2., .1]]).repeat(2, 1) * initial.frame[:, 7:10]).sum(-1)
    torch.testing.assert_close(env.estimates[0].velocity_z_m_s, expected)
    assert not torch.allclose(expected, torch.full((2,), .1))


class FirstColumnValue(nn.Module):
    def forward(self, state):
        return state[:, 0]


class SyntheticLabeledEnv:
    num_envs = 2
    device = torch.device('cpu')

    def __init__(self, schedule=None, *, protocol=False):
        self.schedule = schedule or {}
        self.protocol = RequestProtocol(2) if protocol else None
        self.estimates = []
        self.reward_modes = []
        self.last = None
        self.mutate_model = None
        self.source_fault = None

    def _labels(self):
        velocity = torch.stack((self.time.float(), self.time.float() * 0, self.time.float() * 0 + 99), -1)
        return PhysicalLabels(velocity, torch.ones(2, 2), self.dq.clone(), torch.ones(2, dtype=torch.bool),
                               self.time.clone(), self.episode.clone(), self.epoch.clone())

    def _observe(self, previous):
        frame = torch.zeros(2, 35)
        frame[:, 3] = 0.305 * 5
        frame[:, 9] = -1
        frame[:, 10] = self.counter
        frame[:, 16:22] = self.dq * 0.1 + 13.0
        frame[:, 22:28] = previous
        frame[:, 28] = 1
        critic = torch.zeros(2, 81)
        critic[:, 0] = torch.where(self.episode > 0, 1000.0, 2.0 + self.counter)
        return LabeledObservation(frame, critic, self.time.clone(), self.episode.clone(), self.epoch.clone(),
                                   self.allowed.clone(), self._labels())

    def reset(self, seed=None):
        self.counter = 0
        self.time = torch.zeros(2, dtype=torch.float64)
        self.episode = torch.zeros(2, dtype=torch.long)
        self.epoch = torch.zeros(2, dtype=torch.long)
        self.allowed = torch.ones(2, dtype=torch.bool)
        self.dq = torch.arange(1.0, 7).expand(2, -1).clone()
        self.last = self._observe(torch.zeros(2, 6))
        if self.source_fault == 'reference':
            self.last = replace(self.last, labels=replace(self.last.labels, velocity_reference='base_com_body'))
        elif self.source_fault == 'missing_labels':
            self.last = replace(self.last, labels=replace(self.last.labels, valid=torch.zeros(2, dtype=torch.bool)))
        return self.last

    def step(self, issued, estimate):
        self.estimates.append(estimate)
        self.reward_modes.append(self.last.frame[:, 32].clone())
        before_allowed = self.allowed.clone()
        event = self.schedule.get(self.counter, {})
        terminated = torch.tensor(event.get('terminated', [False, False]))
        truncated = torch.tensor(event.get('truncated', [False, False]))
        handoff = torch.tensor(event.get('handoff', [False, False]))
        owned = torch.tensor(event.get('owned', before_allowed.tolist()))
        self.counter += 1
        self.time += 0.01
        self.dq += torch.arange(1.0, 7)
        final_labels = self._labels().owned()
        final_critic = torch.full((2, 81), float(self.counter + 10))
        final_valid = ~terminated
        done = terminated | truncated
        self.episode += done.long()
        self.dq[done] = 9000
        new_allowed = torch.tensor(event.get('next_allowed', self.allowed.tolist()))
        changes = (new_allowed != self.allowed) | handoff
        self.epoch += changes.long()
        self.allowed = new_allowed
        previous = torch.where(owned[:, None], issued, -2.0)
        previous[done] = 0
        # Emulate native reuse of the old observation and supervision buffers.
        self.last.frame.fill_(777)
        self.last.labels.velocity_body_m_s.fill_(888)
        self.last = self._observe(previous)
        if self.protocol is not None:
            task = self.protocol.step(now_s=self.time, estimate=estimate,
                                      request_id=torch.full((2,), 1 if self.counter == 1 else -1, dtype=torch.long),
                                      requested_height_m=torch.full((2,), 0.12),
                                      cancel_id=torch.full((2,), -1, dtype=torch.long),
                                      allow_new=torch.ones(2, dtype=torch.bool), safety_abort=torch.zeros(2, dtype=torch.bool))
            self.last.frame[:, 28] = ~task.jump_mode
            self.last.frame[:, 32] = task.jump_mode
            self.last.frame[:, 33] = task.height_m * 5
            self.last.frame[:, 34] = task.clock_s
        if self.source_fault == 'reset_as_final':
            final_labels = self.last.labels.owned()
        if self.source_fault == 'missing_final':
            final_labels = replace(final_labels, valid=torch.zeros(2, dtype=torch.bool))
        if self.source_fault == 'wrong_previous':
            self.last.frame[:, 22:28] = 1e4
        if self.source_fault == 'wrong_clock':
            self.last.timestamp_s.add_(0.001)
            self.last.labels.timestamp_s.add_(0.001)
        if self.mutate_model is not None:
            with torch.no_grad():
                self.mutate_model.actor.log_std.add_(0.01)
        reward = torch.ones(2) + self.reward_modes[-1]
        reward[~owned] = 1000
        return LabeledStep(self.last, reward, terminated, truncated, final_critic, final_valid,
                            final_labels, owned, handoff)


def test_current_labels_are_owned_and_delta_uses_named_physical_next_sample():
    env = SyntheticLabeledEnv()
    collector = StructuredCollector(env, model())
    collector.reset()
    batch = collector.collect(3)
    torch.testing.assert_close(batch.labels.velocity[:2, 2], torch.full((2,), 99.0))
    torch.testing.assert_close(batch.labels.velocity_delta, torch.arange(1.0, 7).expand(6, -1))
    assert batch.masks.state_auxiliary.all() and batch.masks.next_auxiliary.all()
    # Noisy observed dq must not substitute for the physical supervision.
    assert not torch.allclose(batch.windows.gather(torch.arange(6))[:, -1, 16:22], batch.labels.velocity_delta)
    saved = batch.windows.gather(torch.arange(6))
    collector.collect(2)
    torch.testing.assert_close(batch.windows.gather(torch.arange(6)), saved, atol=0, rtol=0)


def test_time_limit_bootstraps_pre_reset_value_and_uses_pre_reset_delta():
    env = SyntheticLabeledEnv({0: {'truncated': [True, False], 'terminated': [False, True]}})
    network = model()
    network.critic = FirstColumnValue()
    collector = StructuredCollector(env, network)
    collector.reset()
    batch = collector.collect(2)
    gamma = collector.config.gamma
    torch.testing.assert_close(batch.returns[0], torch.tensor(1 + gamma * 11))
    torch.testing.assert_close(batch.returns[1], torch.tensor(1.0))
    torch.testing.assert_close(batch.labels.velocity_delta[0], torch.arange(1.0, 7))
    assert batch.masks.next_auxiliary[0] and not batch.masks.next_auxiliary[1]
    for endpoint in (2, 3):
        window = batch.windows.gather(torch.tensor([endpoint]))[0]
        torch.testing.assert_close(window, window[-1:].expand_as(window), atol=0, rtol=0)


def test_script_intervals_cannot_enter_actor_auxiliary_value_or_gae_targets():
    env = SyntheticLabeledEnv({
        0: {'handoff': [True, True], 'next_allowed': [False, False]},
        2: {'handoff': [True, True], 'next_allowed': [True, True]},
    })
    network = model()
    network.critic = FirstColumnValue()
    collector = StructuredCollector(env, network)
    collector.reset()
    batch = collector.collect(4)
    assert batch.masks.actor.reshape(4, 2)[:, 0].tolist() == [True, False, False, True]
    assert not batch.masks.state_auxiliary[2:6].any()
    assert not batch.masks.next_auxiliary[:6].any()
    torch.testing.assert_close(batch.returns[:2], torch.ones(2))
    torch.testing.assert_close(batch.returns[2:6], torch.zeros(4))
    windows = batch.windows.gather(torch.tensor([6, 7]))
    torch.testing.assert_close(windows, windows[:, -1:].expand_as(windows), atol=0, rtol=0)
    assert collector.last_metrics['policy_owned_transitions'] == 4


def test_estimate_is_previous_completed_tick_and_reward_keeps_current_command_reference():
    env = SyntheticLabeledEnv(protocol=True)
    network = model()
    with torch.no_grad():
        network.actor.policy.state_head[-1].weight.zero_()
        network.actor.policy.state_head[-1].bias.zero_()
        network.actor.policy.state_head[-1].bias[6:8] = 10
    collector = StructuredCollector(env, network)
    collector.reset()
    batch = collector.collect(2)
    assert env.estimates[0].available_at_s.tolist() == [0, 0]
    assert env.estimates[1].available_at_s.tolist() == [0.01, 0.01]
    assert env.estimates[0].velocity_z_m_s.tolist() == [0, 0]
    assert batch.labels.velocity[0, 2] == 99
    frames = batch.windows.gather(torch.arange(4))[:, -1]
    assert frames[:2, 32].tolist() == [0, 0]
    assert frames[2:, 32].tolist() == [1, 1]
    assert env.reward_modes[0].tolist() == [0, 0]
    assert env.reward_modes[1].tolist() == [1, 1]


def test_raw_behavior_likelihood_is_preserved_while_only_issued_action_is_clipped():
    network = model()
    with torch.no_grad():
        network.actor.policy.leg_decoder[-1].bias.fill_(12)
        network.actor.policy.wheel_decoder[-1].bias.fill_(13)
    collector = StructuredCollector(SyntheticLabeledEnv(), network)
    collector.reset()
    batch = collector.collect(2)
    assert (batch.raw_action[:, :4] > 3).all() and (batch.issued_action[:, :4] == 3).all()
    assert (batch.raw_action[:, 4:] > 9).all() and (batch.issued_action[:, 4:] == 9).all()
    expected = torch.distributions.Normal(batch.old_mean, batch.old_log_std.exp()).log_prob(batch.raw_action).sum(-1)
    torch.testing.assert_close(batch.old_log_prob, expected, atol=0, rtol=0)


@pytest.mark.parametrize('fault', ['reference', 'missing_labels'])
def test_missing_or_wrong_reference_supervision_is_rejected_before_first_action(fault):
    env = SyntheticLabeledEnv()
    env.source_fault = fault
    collector = StructuredCollector(env, model())
    with pytest.raises(ValueError):
        collector.reset()
    assert collector.attempted_vector_steps == 0


@pytest.mark.parametrize('fault', ['reset_as_final', 'missing_final', 'wrong_previous', 'wrong_clock'])
def test_corrupt_transition_invalidates_collection_without_automatic_restart(fault):
    schedule = {0: {'truncated': [True, False]}} if fault == 'reset_as_final' else None
    env = SyntheticLabeledEnv(schedule)
    collector = StructuredCollector(env, model())
    collector.reset()
    env.source_fault = fault
    with pytest.raises(ValueError):
        collector.collect(2)
    assert collector.attempted_vector_steps == collector.returned_vector_steps == 1
    with pytest.raises(RuntimeError, match='reset'):
        collector.collect(2)


def test_mid_interval_override_requires_a_new_control_epoch():
    env = SyntheticLabeledEnv({0: {'owned': [False, False]}})
    collector = StructuredCollector(env, model())
    collector.reset()
    with pytest.raises(ValueError, match='ownership'):
        collector.collect(1)


def test_changed_behavior_model_during_step_is_not_accepted_as_on_policy_data():
    network = model()
    env = SyntheticLabeledEnv()
    env.mutate_model = network
    collector = StructuredCollector(env, network)
    collector.reset()
    with pytest.raises(RuntimeError, match='changed'):
        collector.collect(2)
    with pytest.raises(RuntimeError, match='reset'):
        collector.collect(2)


def test_nonfinite_policy_output_is_rejected_before_actuation():
    network = model()
    collector = StructuredCollector(SyntheticLabeledEnv(), network)
    collector.reset()
    with torch.no_grad():
        network.actor.policy.leg_decoder[-1].bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='before submitting'):
        collector.collect(1)
    assert collector.attempted_vector_steps == collector.returned_vector_steps == 0


def test_full_width_collection_and_guarded_ppo_share_identical_raw_windows():
    network = model(full_width=True)
    collector = StructuredCollector(SyntheticLabeledEnv(), network)
    collector.reset()
    batch = collector.collect(4)
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2,
                                                    learning_rate=1e-6))
    metrics = trainer.update(batch)
    assert metrics['fresh_endpoints'] == 8
    assert metrics['actor_optimizer_steps'] == metrics['critic_optimizer_steps'] == 2
    assert metrics['full_rollout_kl_after_guard'] < 0.02
    next_batch = collector.collect(2)
    assert len(next_batch) == 4
    assert collector.returned_vector_steps == 6


def test_training_config_rejects_rate_change_and_invalid_bounds():
    with pytest.raises(ValueError):
        StructuredCollectionConfig(policy_dt_s=0.02)
    with pytest.raises(ValueError):
        StructuredCollectionConfig(action_bounds=(3, 3, 3, 3, 9, 0))


def test_rollout_storage_remains_owned_across_the_next_collection():
    collector = StructuredCollector(SyntheticLabeledEnv(), model())
    collector.reset()
    first = collector.collect(4)
    indices = torch.arange(len(first))
    frames, actions = first.windows.gather(indices).clone(), first.raw_action.clone()
    critic, masks = first.critic.clone(), first.masks.actor.clone()
    second = collector.collect(3)
    assert first.raw_action.data_ptr() != second.raw_action.data_ptr()
    assert torch.equal(first.windows.gather(indices), frames)
    assert torch.equal(first.raw_action, actions)
    assert torch.equal(first.critic, critic) and torch.equal(first.masks.actor, masks)
    assert collector.last_metrics['sampled_transitions'] == 6

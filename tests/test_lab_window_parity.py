"""Sampling/update window parity across genuine synthetic boundary metadata."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.contracts import LabeledObservation, LabeledStep, PhysicalLabels
from transformer_rl.formal.lab_env.adapter import LabLabeledEnv
from transformer_rl.formal.lab_env.collector import LabCollector
from transformer_rl.formal.learning.optimization import StructuredPPOTrainer
from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig


class BoundaryRows(LabLabeledEnv):
    """Three rows: time limit, failure and a complete script handoff/release."""

    def __init__(self):
        self.num_envs, self.device, self.step_dt = 3, torch.device('cpu'), .01
        self.last_timing, self.last_counts, self.terrain_stream = {}, {}, None
        self.histories = []

    def validate_collection_config(self, _config):
        pass

    def labels(self):
        return PhysicalLabels(torch.zeros(3, 3), torch.zeros(3, 2), torch.zeros(3, 6),
            torch.ones(3, dtype=torch.bool), torch.full((3,), self.tick * .01, dtype=torch.float64),
            self.episode.clone(), self.control.clone())

    def observation(self):
        frame = torch.full((3, 35), float(self.tick)) + torch.arange(3)[:, None] * 100.
        frame[:, 0] += (self.episode - 1) * 1000
        frame[:, 14:16] = 0
        frame[:, 22:28] = self.applied
        return LabeledObservation(frame, torch.zeros(3, 182), self.labels().timestamp_s,
            self.episode.clone(), self.control.clone(), self.allowed.clone(), self.labels())

    def reset(self, seed=None):
        self.tick = 0
        self.episode = torch.ones(3, dtype=torch.long)
        self.control = torch.ones(3, dtype=torch.long)
        self.allowed = torch.ones(3, dtype=torch.bool)
        self.applied = torch.zeros(3, 6)
        return self.observation()

    def step(self, issued_action, _estimate):
        self.histories.append(self.collector._history.clone())
        before_episode, before_control, owned = self.episode.clone(), self.control.clone(), self.allowed.clone()
        self.tick += 1
        self.applied = torch.where(owned[:, None], issued_action, 0)
        terminal_frame = self.observation().frame.clone()
        final_labels = replace(self.labels(), episode_id=before_episode, control_epoch=before_control)
        terminated, truncated, handoff = (torch.zeros(3, dtype=torch.bool) for _ in range(3))
        if self.tick == 2:
            truncated[0] = True
        if self.tick == 3:
            terminated[1] = True
        if self.tick in (4, 6):
            self.allowed[2] = self.tick == 6
            handoff[2] = True
        done = terminated | truncated
        self.episode += done.long()
        self.control += (done | handoff).long()
        self.applied[done | handoff] = 0
        return LabeledStep(observation=self.observation(), reward=torch.ones(3),
            terminated=terminated, truncated=truncated, final_critic=torch.ones(3, 182),
            final_critic_valid=torch.ones(3, dtype=torch.bool), final_labels=final_labels,
            fully_policy_owned=owned, handoff_after=handoff, terminal_frame=terminal_frame)


@pytest.mark.parametrize('first_rollout_steps', (2, 3, 4))
def test_behavior_reconstruction_survives_partial_boundaries_and_continuous_rollouts(first_rollout_steps):
    torch.manual_seed(2102)
    model = StructuredActorCritic(StructuredPolicyConfig())
    env = BoundaryRows()
    collector = LabCollector(env, model)
    env.collector = collector
    collector.reset()
    trainer = StructuredPPOTrainer(model)
    for steps in (first_rollout_steps, 8 - first_rollout_steps):
        begin = len(env.histories)
        batch = collector.collect(steps)
        # No collector.reset between rollouts: the prefix must survive as-is.
        expected = torch.stack(env.histories[begin:]).flatten(0, 1)
        endpoints = torch.arange(len(batch))
        torch.testing.assert_close(batch.windows.gather(endpoints), expected, atol=0, rtol=0)
        trainer._validate_behavior(batch, batch.masks.actor.nonzero().flatten())
        for local_step in range(steps):
            global_step = begin + local_step + 1
            if global_step in (5, 6):
                assert not batch.masks.actor[local_step * 3 + 2]
                assert not batch.masks.critic[local_step * 3 + 2]
            if global_step == 2:
                assert batch.masks.bootstrap[local_step * 3]
                assert not batch.masks.trace[local_step * 3]
            if global_step == 3:
                assert not batch.masks.bootstrap[local_step * 3 + 1]
                assert not batch.masks.trace[local_step * 3 + 1]
            if global_step in (4, 6):
                assert not batch.masks.bootstrap[local_step * 3 + 2]
                assert not batch.masks.next_auxiliary[local_step * 3 + 2]
    assert trainer.update_index == 0 and trainer.consumed['fresh_endpoints'] == 0

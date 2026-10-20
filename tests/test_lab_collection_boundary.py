"""Fast LabCollector boundary checks for the 182D -> 185D path."""
from __future__ import annotations

import torch

from transformer_rl.formal.contracts import (LabeledObservation, LabeledStep,
                                             PhysicalLabels, StructuredCollectionConfig)
from transformer_rl.formal.lab_env.adapter import LabLabeledEnv
from transformer_rl.formal.lab_env.collector import LabCollector
from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig


def _labels(timestamp, episode, control):
    return PhysicalLabels(
        velocity_body_m_s=torch.zeros(1, 3),
        wheel_contact=torch.zeros(1, 2),
        joint_velocity_rad_s=torch.zeros(1, 6),
        valid=torch.ones(1, dtype=torch.bool),
        timestamp_s=torch.full((1,), timestamp, dtype=torch.float64),
        episode_id=torch.full((1,), episode, dtype=torch.long),
        control_epoch=torch.full((1,), control, dtype=torch.long),
    )


def _observation(frame, critic, timestamp, episode, control):
    return LabeledObservation(
        frame=frame,
        critic=critic,
        timestamp_s=torch.full((1,), timestamp, dtype=torch.float64),
        episode_id=torch.full((1,), episode, dtype=torch.long),
        control_epoch=torch.full((1,), control, dtype=torch.long),
        policy_allowed=torch.ones(1, dtype=torch.bool),
        labels=_labels(timestamp, episode, control),
    )


class _TerminalEnv(LabLabeledEnv):
    def __init__(self):
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.step_dt = .01
        self.last_timing = {}
        self.last_counts = {}
        self.terrain_stream = None
        self.frame = torch.zeros(1, 35)
        self.critic = torch.zeros(1, 182)
        self.terminal_critic = torch.ones(1, 182)
        self.terminal_frame = torch.full((1, 35), 2.0)
        self.calls = 0

    def validate_collection_config(self, config):
        assert isinstance(config, StructuredCollectionConfig)

    def reset(self, seed=None):
        return _observation(self.frame, self.critic, 0.0, 1, 1)

    def step(self, issued_action, completed_estimate):
        assert issued_action.shape == (1, 6)
        self.calls += 1
        next_obs = _observation(self.frame + 1, self.critic + 3, .01, 2, 2)
        return LabeledStep(
            observation=next_obs,
            reward=torch.tensor([2.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.ones(1, dtype=torch.bool),
            final_critic=self.terminal_critic.clone(),
            final_critic_valid=torch.ones(1, dtype=torch.bool),
            final_labels=_labels(.01, 1, 1),
            fully_policy_owned=torch.ones(1, dtype=torch.bool),
            handoff_after=torch.zeros(1, dtype=torch.bool),
            terminal_frame=self.terminal_frame.clone(),
        )


def test_lab_collector_preserves_terminal_reward_and_uses_five_frame_critic():
    model = StructuredActorCritic(StructuredPolicyConfig())
    env = _TerminalEnv()
    collector = LabCollector(env, model)
    collector.reset()
    batch = collector.collect(steps=1)
    assert collector.last_metrics["reward_mean"] == 2.0
    assert batch.critic.shape == (1, 185)
    assert batch.masks.bootstrap.tolist() == [True]
    assert batch.masks.trace.tolist() == [False]
    assert torch.isfinite(batch.advantages).all()
    assert collector.last_metrics["terminated_rows"] == 0
    assert collector.last_metrics["truncated_rows"] == 1

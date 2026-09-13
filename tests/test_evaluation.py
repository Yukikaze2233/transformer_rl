"""Deterministic evaluation checks over scripted tensors, not physical simulation."""
from dataclasses import replace
import hashlib
import math

import pytest
import torch

from transformer_rl.checkpoint import save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.evaluation import evaluate_policy
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer
from transformer_rl.types import StepResult, VectorObservation


class ScriptedEvaluation:
    num_envs = 2
    device = torch.device("cpu")
    metadata = {"identity": {"task": "scripted_tensor_fixture"}}

    def __init__(self, config, failure=None):
        self.config = config
        self.failure = failure
        self.closed = False
        self.actions = []
        self.events = torch.zeros(2, dtype=torch.float64)

    def observation(self):
        frame = torch.zeros(2, self.config.frame_dim)
        frame[:, 0] = self.events.float()
        return VectorObservation(frame, self.events / 100,
                                 torch.zeros(2, self.config.command_dim),
                                 torch.zeros(2, self.config.critic_dim))

    def reset(self, seed=None):
        self.seed = seed
        self.events.zero_()
        return self.observation()

    def step(self, action):
        self.actions.append(action.clone())
        t = len(self.actions)
        self.events += 1
        term = torch.tensor([False, t == 3])
        trunc = torch.tensor([t == 2, False])
        self.events[term | trunc] = 0
        metric = torch.tensor([float(t), 2.0 * t])
        if self.failure == "nonfinite" and t == 2:
            metric[0] = float("nan")
        key = "different" if self.failure == "names" and t == 2 else "tracking_error"
        return StepResult(self.observation(), torch.tensor([1., 3.]), term, trunc,
                          torch.zeros(2, self.config.critic_dim), trunc.clone(),
                          {"evaluation_metrics": {key: metric}})

    def close(self):
        self.closed = True


@pytest.fixture
def checkpoint(tmp_path):
    config = ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                         critic_dim=2, d_model=8, num_heads=2, num_layers=1,
                         ffn_dim=16, history_length=3, critic_hidden=(8,))
    model = ActorCritic(config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, PPOTrainer(model, PPOConfig()), 0,
                    {"environment_provenance": ScriptedEvaluation.metadata})
    return path, config


def test_deterministic_evaluation_metrics_and_partial_reset(checkpoint, monkeypatch):
    path, config = checkpoint
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    env = ScriptedEvaluation(config)
    monkeypatch.setattr(PPOTrainer, "update", lambda *_: pytest.fail("evaluation optimized parameters"))
    result = evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert result["checkpoint_sha256"] == before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["policy"] == "deterministic_mean"
    assert result["transitions"] == 6
    assert result["reward_mean"] == 2.0
    assert result["terminated_count"] == result["truncated_count"] == 1
    assert result["done_count"] == 2
    assert result["metrics"]["tracking_error"] == pytest.approx(
        {"mean": 3.0, "rms": math.sqrt(70 / 6), "min": 1.0, "max": 6.0, "count": 6})
    torch.testing.assert_close(env.actions[2][0], env.actions[0][0])
    assert env.closed and env.seed == 101


@pytest.mark.parametrize("failure,error", [("nonfinite", FloatingPointError), ("names", ValueError)])
def test_invalid_metric_reports_fail_and_close(checkpoint, failure, error):
    path, config = checkpoint
    env = ScriptedEvaluation(config, failure)
    with pytest.raises(error):
        evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert env.closed


def test_no_physical_metrics_is_explicit(checkpoint):
    path, config = checkpoint
    env = ScriptedEvaluation(config)
    step = env.step
    env.step = lambda action: replace(step(action), info={})
    result = evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert result["physical_metrics_available"] is False
    assert result["metrics"] == {}


def test_task_identity_change_rejected_before_reset(checkpoint):
    path, config = checkpoint
    env = ScriptedEvaluation(config)
    env.metadata = {"identity": {"task": "different"}}
    with pytest.raises(ValueError, match="identity"):
        evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert env.closed and not env.actions

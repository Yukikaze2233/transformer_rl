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
    assert result["stability"]["available"] is False
    assert result["stability"]["signals"] == {}
    assert result["stability"]["protocol"]["settle_steps"] == 200
    assert result["stability"]["protocol"]["min_steady_samples"] == 200
    assert result["stability"]["protocol"]["centering"] == "per_environment_episode"
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


class SignalEvaluation(ScriptedEvaluation):
    def step(self, action):
        # The physical clock is independent of observation/actor-event time.
        physical_time = 100 + (self.events + 1) / 100
        result = super().step(action)
        t = len(self.actions)
        return replace(result, info={
            **result.info,
            "evaluation_signals": {"error": torch.tensor([float(t), 10.0 * t])},
            "evaluation_signal_time": physical_time,
        })


def test_stability_pre_reset_failure_and_partial_episode_integration(checkpoint):
    path, config = checkpoint
    env = SignalEvaluation(config)
    result = evaluate_policy(path, lambda **_: env, {}, 5, 101, "cpu",
                             settle_steps=1, min_steady_samples=2)
    signal = result["stability"]["signals"]["error"]
    # Usable segments are env1's terminated [20,30] and env0's partial [4,5].
    assert result["stability"]["available"] is True
    assert signal["mean"] == 14.75
    assert signal["within_episode_std"] == pytest.approx(math.sqrt(50.5 / 4))
    assert signal["derivative_rms"] == pytest.approx(math.sqrt((100**2 + 1000**2) / 2))
    assert signal["max_abs"] == 30
    assert signal["count"] == 4 and signal["segments"] == 2
    assert signal["completed_segments"] == signal["partial_segments"] == 1
    assert signal["short_segments"] == signal["short_count"] == 2
    assert signal["total_count"] == 10 and signal["settled_count"] == 4
    assert result["terminated_count"] == result["truncated_count"] == 1
    assert result["done_count"] == 2
    assert result["reward_mean"] == 2
    assert result["metrics"]["tracking_error"] == pytest.approx({
        "mean": 4.5, "rms": math.sqrt(275 / 10), "min": 1, "max": 10, "count": 10})
    assert env.closed


def test_all_short_stability_preserves_whole_interval_metrics(checkpoint):
    path, config = checkpoint
    env = SignalEvaluation(config)
    result = evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert result["stability"]["available"] is False
    signal = result["stability"]["signals"]["error"]
    assert signal["mean"] is signal["within_episode_std"] is signal["derivative_rms"] is None
    assert signal["max_abs"] is None
    assert signal["short_segments"] == 3 and signal["total_count"] == 6
    assert signal["count"] == signal["segments"] == 0
    assert result["metrics"]["tracking_error"]["count"] == 6
    assert result["done_count"] == 2
    assert env.closed


@pytest.mark.parametrize("failure,error", [
    ("names", ValueError), ("missing_signals", ValueError), ("missing_time", ValueError),
    ("nonfinite", FloatingPointError), ("time_nonfinite", FloatingPointError),
    ("time_dtype", TypeError), ("time_shape", ValueError), ("signal_shape", ValueError),
    ("signal_dtype", TypeError), ("signal_mapping", ValueError),
    ("duplicate_time", ValueError), ("reset_observation_time", ValueError),
])
def test_invalid_signal_reports_fail_and_close(checkpoint, failure, error):
    path, config = checkpoint
    env = SignalEvaluation(config)
    step = env.step
    previous_time = None

    def invalid_step(action):
        nonlocal previous_time
        result = step(action)
        info = result.info
        if len(env.actions) == 2:
            if failure == "names":
                info["evaluation_signals"] = {"changed": torch.zeros(2)}
            elif failure == "missing_signals":
                del info["evaluation_signals"]
                del info["evaluation_signal_time"]
            elif failure == "missing_time":
                del info["evaluation_signal_time"]
            elif failure == "nonfinite":
                info["evaluation_signals"]["error"][0] = float("nan")
            elif failure == "time_nonfinite":
                info["evaluation_signal_time"][0] = float("inf")
            elif failure == "time_dtype":
                info["evaluation_signal_time"] = info["evaluation_signal_time"].float()
            elif failure == "time_shape":
                info["evaluation_signal_time"] = torch.zeros(1, dtype=torch.float64)
            elif failure == "signal_shape":
                info["evaluation_signals"]["error"] = torch.zeros(2, 1)
            elif failure == "signal_dtype":
                info["evaluation_signals"]["error"] = torch.zeros(2, dtype=torch.int64)
            elif failure == "signal_mapping":
                info["evaluation_signals"] = []
            elif failure == "duplicate_time":
                info["evaluation_signal_time"] = previous_time
            elif failure == "reset_observation_time":
                info["evaluation_signal_time"] = result.observation.timestamp
        previous_time = info.get("evaluation_signal_time")
        return result

    env.step = invalid_step
    with pytest.raises(error):
        evaluate_policy(path, lambda **_: env, {}, 3, 101, "cpu")
    assert env.closed


@pytest.mark.parametrize("kwargs", [
    {"settle_steps": -1}, {"settle_steps": True}, {"settle_steps": 1.5},
    {"min_steady_samples": 0}, {"min_steady_samples": False}, {"min_steady_samples": 1.5},
])
def test_invalid_stability_options_fail_before_environment_creation(checkpoint, kwargs):
    path, _ = checkpoint
    with pytest.raises(ValueError):
        evaluate_policy(path, lambda **_: pytest.fail("invalid protocol created environment"),
                        {}, 3, 101, "cpu", **kwargs)

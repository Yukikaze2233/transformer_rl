"""Scripted transitions only: collection semantics, never environment learning."""
from dataclasses import fields, replace
import math

import pytest
import torch
from torch import nn

from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic, TimeAwareActor
from transformer_rl.ppo import PPOTrainer
from transformer_rl.runner import RolloutCollector
from transformer_rl.types import StepResult, VectorObservation


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seeded_torch():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


@pytest.fixture
def model_config():
    return ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                       critic_dim=2, history_length=4, d_model=8, num_heads=2,
                       num_layers=1, ffn_dim=16, critic_hidden=(8,))


def observation(config, times, values):
    values = torch.tensor(values, dtype=torch.float32)
    frame = torch.zeros(len(values), config.frame_dim)
    frame[:, 0] = values / 100.0
    frame[:, config.proprio_dim] = values / 100.0
    return VectorObservation(
        frame=frame, timestamp=torch.tensor(times, dtype=torch.float64),
        command=(values / 100.0).unsqueeze(-1),
        critic=torch.stack((values, torch.zeros_like(values)), dim=-1),
    )


def transition(config, times, values, rewards, terminated=None, truncated=None, final=None, valid=None):
    count = len(values)
    return StepResult(
        observation(config, times, values), torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(terminated or [False] * count), torch.tensor(truncated or [False] * count),
        torch.tensor(final or [float("nan")] * count)[:, None].expand(-1, config.critic_dim).clone(),
        torch.tensor(valid or [False] * count), {},
    )


class ScriptedEnv:
    """Replay predeclared outputs, reusing every returned observation/metadata tensor."""

    def __init__(self, initial, script, *, mutate_action=False):
        self.initial = initial
        self.script = script
        self.num_envs = initial.frame.shape[0]
        self.device = initial.frame.device
        self.reset_seeds = []
        self.actions = []
        self.mutate_action = mutate_action
        self.on_step = None
        self.position = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.position = 0
        self.current = VectorObservation(**{
            field.name: getattr(self.initial, field.name).clone() for field in fields(VectorObservation)
        })
        self.reward = torch.zeros(self.num_envs)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.final_critic = torch.zeros_like(self.current.critic)
        self.final_critic_valid = torch.zeros(self.num_envs, dtype=torch.bool)
        return self.current

    def step(self, action):
        self.actions.append(action.detach().clone())
        if self.mutate_action:
            action.fill_(999.0)
        result = self.script[self.position]
        self.position += 1
        for field in fields(VectorObservation):
            getattr(self.current, field.name).copy_(getattr(result.observation, field.name))
        metadata = {}
        for name in ("reward", "terminated", "truncated", "final_critic", "final_critic_valid"):
            value = getattr(result, name)
            target = getattr(self, name)
            if isinstance(value, torch.Tensor) and target is not None and value.shape == target.shape:
                target.copy_(value)
            else:
                setattr(self, name, value)
            metadata[name] = getattr(self, name)
        if self.on_step is not None:
            self.on_step()
        return StepResult(self.current, **metadata, info=result.info)


class RecordingCritic(nn.Module):
    """An exact value oracle with view outputs, to expose alias and row-selection errors."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, critic):
        assert torch.isfinite(critic).all(), "undefined rows reached the critic"
        self.inputs.append(critic.clone())
        return critic[:, 0]


def mixed_script(config):
    initial = observation(config, [1000.0] * 4, [1, 2, 3, 4])
    script = [
        transition(config, [0, 0, 0, 1000.01], [100, 200, 300, 5], [1, 2, 3, 4],
                   terminated=[True, False, True, False], truncated=[False, True, True, False],
                   final=[float("nan"), 20.0, float("nan"), float("nan")],
                   valid=[False, True, False, False]),
        transition(config, [0.01, 0.01, 0.01, 1000.02], [101, 201, 301, 6], [5, 6, 7, 8]),
        transition(config, [0.02, 0.02, 0.02, 1000.03], [102, 202, 302, 7], [1, 1, 1, 1]),
    ]
    return initial, script


def test_hand_bootstrap_gae_and_per_env_reset_with_real_actor(model_config):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    model = ActorCritic(model_config)
    model.critic = RecordingCritic()
    collector = RolloutCollector(env, model, PPOConfig(gamma=0.5, gae_lambda=0.5))
    collector.reset(seed=31)
    batch = collector.collect(2)
    assert isinstance(model.actor, TimeAwareActor)
    assert len(batch) == 8
    torch.testing.assert_close(batch.advantages, torch.tensor([0.0, 10.0, 0.0, 4.0, -44.5, -93.5, -142.5, 6.0]))
    torch.testing.assert_close(batch.returns, batch.old_value + batch.advantages)
    expected_values = ([1, 2, 3, 4], [5], [20], [100, 200, 300, 5], [101, 201, 301, 6])
    for actual, expected in zip(model.critic.inputs, expected_values, strict=True):
        torch.testing.assert_close(actual[:, 0], torch.tensor(expected, dtype=torch.float32))
    torch.testing.assert_close(batch.history.valid.sum(-1), torch.tensor([1, 1, 1, 1, 1, 1, 1, 2]))
    torch.testing.assert_close(batch.history.now[4:], torch.tensor([0, 0, 0, 1000.01], dtype=torch.float64))
    torch.testing.assert_close(batch.critic[:, 0], torch.tensor([1, 2, 3, 4, 100, 200, 300, 5], dtype=torch.float32))
    assert env.reset_seeds == [31]

    # Model updates between completed rollouts are legal; history is not a policy cache.
    with torch.no_grad():
        model.actor.mean_head.bias.add_(0.01)
    following = collector.collect(1)
    torch.testing.assert_close(following.history.valid.sum(-1), torch.tensor([2, 2, 2, 3]))
    torch.testing.assert_close(following.history.now, torch.tensor([0.01, 0.01, 0.01, 1000.02], dtype=torch.float64))
    assert collector.total_steps == 3 and collector.total_transitions == 12
    assert env.reset_seeds == [31]


@pytest.mark.parametrize("action_clip", [None, 0.25])
def test_step_owns_raw_issued_policy_statistics_and_old_observation(model_config, monkeypatch, action_clip):
    initial = observation(model_config, [1.0, 2.0], [3, 4])
    script = [transition(model_config, [1.01, 2.01], [5, 6], [1, 2]),
              transition(model_config, [1.02, 2.02], [7, 8], [3, 4])]
    env = ScriptedEnv(initial, script, mutate_action=True)
    model = ActorCritic(model_config)
    model.critic = RecordingCritic()
    with torch.no_grad():
        model.actor.mean_head.weight.zero_()
        model.actor.mean_head.bias.copy_(torch.tensor([2.0, -2.0]))
        model.actor.log_std.fill_(math.log(0.01))
    original_act = model.actor.act
    emitted = []
    expected = []

    def record_act(history):
        sample = original_act(history)
        emitted.append(sample)
        expected.append((sample.action.clone(), sample.evaluation.log_prob.clone(),
                         sample.evaluation.mean.clone(), sample.evaluation.std.clone()))
        return sample

    def overwrite_emitted_tensors():
        sample = emitted[-1]
        sample.action.fill_(123.0)
        for field in fields(sample.evaluation):
            getattr(sample.evaluation, field.name).fill_(123.0)

    monkeypatch.setattr(model.actor, "act", record_act)
    env.on_step = overwrite_emitted_tensors
    collector = RolloutCollector(env, model, PPOConfig(), action_clip=action_clip)
    returned = collector.reset()
    for field in fields(VectorObservation):
        getattr(returned, field.name).fill_(-100)
        getattr(env.current, field.name).fill_(-200)
    batch = collector.collect(2)
    torch.testing.assert_close(batch.critic[:, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(batch.old_value, batch.critic[:, 0])
    for position, field in enumerate(("raw_action", "old_log_prob", "old_mean", "old_std")):
        torch.testing.assert_close(getattr(batch, field), torch.cat([sample[position] for sample in expected]))
    assert (batch.raw_action.abs() > 1.0).all()
    expected_issued = batch.raw_action if action_clip is None else batch.raw_action.clamp(-action_clip, action_clip)
    torch.testing.assert_close(batch.issued_action, expected_issued)
    torch.testing.assert_close(torch.cat(env.actions), expected_issued)
    with torch.no_grad():
        evaluation = model.actor.evaluate(batch.history, batch.raw_action)
    torch.testing.assert_close(evaluation.log_prob, batch.old_log_prob, rtol=1e-5, atol=1e-5)
    assert not any(getattr(batch, field.name).requires_grad for field in fields(batch) if field.name != "history")


def test_real_model_collection_likelihood_and_one_finite_ppo_step(model_config):
    initial, script = mixed_script(model_config)
    model = ActorCritic(model_config)
    ppo_config = PPOConfig(epochs=1, num_minibatches=1)
    collector = RolloutCollector(ScriptedEnv(initial, script), model, ppo_config)
    collector.reset(seed=17)
    batch = collector.collect(2)
    with torch.no_grad():
        evaluation = model.actor.evaluate(batch.history, batch.raw_action)
    torch.testing.assert_close(evaluation.log_prob, batch.old_log_prob, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(evaluation.mean, batch.old_mean)
    torch.testing.assert_close(evaluation.std, batch.old_std)
    trainer = PPOTrainer(model, ppo_config)
    gradients = {}

    def capture(name):
        def hook(gradient):
            gradients[name] = gradient.detach().clone()
        return hook

    for name, parameter in model.named_parameters():
        parameter.register_hook(capture(name))
    metrics = trainer.update(batch)
    assert metrics["optimizer_steps"] == 1
    assert metrics["sample_count"] == len(batch)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients.values())
    assert any(gradient.abs().sum() > 0 for name, gradient in gradients.items() if name.startswith("actor."))
    assert any(gradient.abs().sum() > 0 for name, gradient in gradients.items() if name.startswith("critic."))
    assert all(math.isfinite(value) for value in metrics.values())


def test_callback_partial_rollout_counters_and_elapsed_time(model_config, monkeypatch):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    model = ActorCritic(model_config)
    model.critic = RecordingCritic()
    collector = RolloutCollector(env, model, PPOConfig(gamma=0.5, gae_lambda=0.5))
    collector.reset()
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("transformer_rl.runner.time.perf_counter", lambda: next(ticks))
    calls = []

    def stop():
        calls.append(env.position)
        return env.position == 2

    batch = collector.collect(10, should_stop=stop)
    assert len(batch) == 8 and calls == [0, 1, 2]
    assert collector.last_metrics == dict(
        reward_mean=4.5, terminated_count=2, truncated_count=2, done_count=3,
        vector_steps=2, transitions=8, elapsed_s=2.5, early_stopped=True,
        total_steps=2, total_transitions=8,
    )
    torch.testing.assert_close(batch.advantages[-4:], torch.tensor([-44.5, -93.5, -142.5, 6.0]))


def test_immediate_stop_zero_budget_and_explicit_reset(model_config):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    collector = RolloutCollector(env, ActorCritic(model_config), PPOConfig())
    assert collector.collect(5, should_stop=lambda: True) is None
    assert collector.last_metrics["early_stopped"]
    assert collector.last_metrics["vector_steps"] == collector.last_metrics["transitions"] == 0
    assert collector.last_metrics["reward_mean"] == 0.0
    assert env.reset_seeds == [] and env.actions == []
    assert collector.collect(0) is None
    assert not collector.last_metrics["early_stopped"]
    with pytest.raises(RuntimeError, match="reset"):
        collector.collect(1)
    assert env.reset_seeds == [] and env.actions == []


def test_explicit_reset_clears_history_but_preserves_lifetime_counters(model_config):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    collector = RolloutCollector(env, ActorCritic(model_config), PPOConfig())
    collector.reset(seed=1)
    collector.collect(2)
    collector.reset(seed=2)
    batch = collector.collect(1)
    assert env.reset_seeds == [1, 2]
    assert collector.total_steps == 3 and collector.total_transitions == 12
    assert (batch.history.valid.sum(-1) == 1).all()


def test_duplicate_observation_tick_is_not_appended_twice(model_config):
    initial = observation(model_config, [1000.0], [1])
    script = [transition(model_config, [1000.0], [1], [1]),
              transition(model_config, [1000.01], [2], [1])]
    collector = RolloutCollector(ScriptedEnv(initial, script), ActorCritic(model_config), PPOConfig())
    collector.reset()
    batch = collector.collect(2)
    torch.testing.assert_close(batch.history.valid.sum(-1), torch.tensor([1, 1]))


@pytest.mark.parametrize("both", [False, True])
def test_true_terminal_does_not_evaluate_any_final_or_reset_critic(model_config, both):
    initial = observation(model_config, [1000.0], [3])
    script = [transition(model_config, [0.0], [100], [2], terminated=[True], truncated=[both],
                         final=[float("nan")], valid=[True])]
    model = ActorCritic(model_config)
    model.critic = RecordingCritic()
    collector = RolloutCollector(ScriptedEnv(initial, script), model, PPOConfig())
    collector.reset()
    batch = collector.collect(1)
    assert len(model.critic.inputs) == 1
    torch.testing.assert_close(batch.returns, torch.tensor([2.0]))


@pytest.mark.parametrize("failure", ["missing", "invalid", "shape", "nonfinite"])
def test_timeout_requires_real_valid_final_state(model_config, failure):
    initial = observation(model_config, [1.0], [3])
    result = transition(model_config, [0.0], [100], [2], truncated=[True], final=[10.0], valid=[True])
    if failure == "missing":
        result = replace(result, final_critic=None, final_critic_valid=None)
    elif failure == "invalid":
        result.final_critic_valid.fill_(False)
    elif failure == "shape":
        result = replace(result, final_critic_valid=result.final_critic_valid[:, None])
    else:
        result.final_critic.fill_(float("nan"))
    env = ScriptedEnv(initial, [result])
    collector = RolloutCollector(env, ActorCritic(model_config), PPOConfig())
    collector.reset()
    with pytest.raises((ValueError, FloatingPointError), match="final_critic"):
        collector.collect(1)
    assert len(env.actions) == 1
    with pytest.raises(RuntimeError, match="reset"):
        collector.collect(1)
    assert len(env.actions) == 1


@pytest.mark.parametrize("field", ["frame", "timestamp", "command", "critic"])
def test_all_reset_next_observation_rows_must_be_finite(model_config, field):
    initial = observation(model_config, [1.0], [3])
    result = transition(model_config, [0.0], [100], [2], terminated=[True])
    getattr(result.observation, field).fill_(float("nan"))
    collector = RolloutCollector(ScriptedEnv(initial, [result]), ActorCritic(model_config), PPOConfig())
    collector.reset()
    with pytest.raises(FloatingPointError, match=f"observation.{field}"):
        collector.collect(1)


def test_backwards_ongoing_timestamp_requires_reset_after_rejection(model_config):
    initial = observation(model_config, [10.0], [3])
    result = transition(model_config, [0.0], [100], [2])
    collector = RolloutCollector(ScriptedEnv(initial, [result]), ActorCritic(model_config), PPOConfig())
    collector.reset()
    with pytest.raises(ValueError, match="timestamps"):
        collector.collect(1)
    with pytest.raises(RuntimeError, match="reset"):
        collector.collect(1)


@pytest.mark.parametrize("when", ["callback", "step"])
def test_rejects_policy_version_changes_within_collection(model_config, when):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    model = ActorCritic(model_config)
    collector = RolloutCollector(env, model, PPOConfig())
    collector.reset()

    def modify_model():
        with torch.no_grad():
            model.actor.mean_head.bias.add_(0.01)

    def stop():
        if env.position == 1:
            modify_model()
        return False

    if when == "step":
        env.on_step = modify_model
    with pytest.raises(RuntimeError, match="model changed"):
        collector.collect(2, should_stop=stop if when == "callback" else None)
    assert len(env.actions) == 1
    assert collector.total_steps == 1


@pytest.mark.parametrize("action_clip", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_action_clip_is_rejected_without_environment_start(model_config, action_clip):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    with pytest.raises(ValueError, match="action_clip"):
        RolloutCollector(env, ActorCritic(model_config), PPOConfig(), action_clip=action_clip)
    assert env.reset_seeds == [] and env.actions == []


@pytest.mark.parametrize("field", ["action", "log_prob", "mean", "std"])
def test_nonfinite_policy_output_cannot_reach_env(model_config, monkeypatch, field):
    initial, script = mixed_script(model_config)
    env = ScriptedEnv(initial, script)
    model = ActorCritic(model_config)
    collector = RolloutCollector(env, model, PPOConfig())
    collector.reset()
    act = model.actor.act

    def invalid_act(history):
        sample = act(history)
        tensor = sample.action if field == "action" else getattr(sample.evaluation, field)
        tensor.fill_(float("nan"))
        return sample

    monkeypatch.setattr(model.actor, "act", invalid_act)
    with pytest.raises(FloatingPointError):
        collector.collect(1)
    assert env.actions == []


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_step_budget(model_config, steps):
    initial, script = mixed_script(model_config)
    collector = RolloutCollector(ScriptedEnv(initial, script), ActorCritic(model_config), PPOConfig())
    with pytest.raises(ValueError, match="steps"):
        collector.collect(steps)

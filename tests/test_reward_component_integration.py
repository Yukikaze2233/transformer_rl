"""Reward telemetry boundaries on synthetic CPU environments, without an SDK."""
from copy import deepcopy
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.chassis_adapter import ChassisFrameAdapter
from transformer_rl.config import PPOConfig
from transformer_rl.frame_config import FrameModelConfig
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.frame_training import FrameActorCritic, FrameCollector
from transformer_rl.frame_workflow import _model_state_sha256, train_frame_policy
from transformer_rl.ppo import PPOTrainer
from transformer_rl.reward_components import RewardComponentStatistics

from test_chassis_adapter import TensorChassis
from test_frame_workflow import PackedFixture, configuration as packed_configuration


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RewardTensorChassis(TensorChassis):
    def __init__(self, *, components=True, events=False, invalid_tick=None):
        super().__init__()
        self.native_density = torch.zeros(3)
        self.native_events = torch.zeros(3)
        self.components, self.events, self.invalid_tick = components, events, invalid_tick

    def reset(self, rows):
        super().reset(rows)
        self.native_density[rows] = 0
        self.native_events[rows] = 0

    def step(self, issued):
        raw, _, done, extras = super().step(issued)
        self.native_density.copy_(torch.tensor([1., 2., 3.]) * self.tick)
        self.native_events.copy_(torch.tensor([0., 0., -5.]) * self.tick)
        # Total reward also includes an omitted continuous contribution. Its
        # storage is separate; only producer component buffers reset in place.
        reward = .01 * self.native_density + self.native_events + self.tick
        if self.components:
            extras["native_reward_components"] = {"height": self.native_density}
            if self.tick == self.invalid_tick:
                extras["native_reward_components"]["height"] = torch.full((3,), float("nan"))
        if self.events:
            extras["native_reward_events"] = {"terminal": self.native_events}
        extras["log"] = {"rewards/height": self.native_density.mean()}
        return raw, reward, done, extras


def chassis_collector(env):
    model_config = FrameModelConfig(FramePolicyConfig(architecture="mlp", history_length=1,
                                    frame_dim=35, action_dim=6, actor_hidden_dims=(8,)),
                                    critic_dim=81, critic_hidden=(8,), initial_std=.2)
    adapter = ChassisFrameAdapter(env, model_config, {})
    assert adapter.enable_reward_components is False
    collector = FrameCollector(adapter, FrameActorCritic(model_config),
                               PPOConfig(epochs=1, num_minibatches=1), [1.] * 6)
    assert adapter.enable_reward_components is True
    collector.reset()
    return adapter, collector


@pytest.mark.parametrize("components,events", [(True, False), (True, True), (False, False)])
def test_adapter_observes_aliasing_native_sources_before_manual_reset(components, events):
    env = RewardTensorChassis(components=components, events=events)
    config = FrameModelConfig(FramePolicyConfig(architecture="mlp", history_length=1,
                              actor_hidden_dims=(8,)), critic_hidden=(8,))
    adapter = ChassisFrameAdapter(env, config, {})
    adapter.enable_reward_components = True
    adapter.reset()
    for _ in range(2):
        result = adapter.step(torch.zeros(3, 6))
        assert result.truncated.all()
        assert not env.native_density.any() and not env.native_events.any()
    report = adapter.drain_reward_components()
    assert (report["steps"], report["samples"], report["observed_vector_steps"]) == (2, 6, 2)
    assert report["observation_error"] is None
    assert report["observer_host_elapsed_s"] >= 0
    assert report["actual_total_reward"]["mean"] == pytest.approx(-.97, abs=1e-6)
    assert adapter.training_diagnostics()["rewards/height"] == 4
    assert report["continuous_components"]["available"] is components
    assert report["event_components"]["available"] is events
    if components:
        term = report["continuous_components"]["terms"]["height"]
        assert term["density"]["mean"] == 3
        assert term["step_reward"]["mean"] == pytest.approx(.03)
        assert report["residual_step_reward"]["statistics"]["mean"] == pytest.approx(
            1.5 if events else -1, abs=1e-6)
    else:
        assert report["continuous_components"]["terms"] is None
        assert report["residual_step_reward"]["statistics"] is None
    if events:
        assert report["event_components"]["terms"]["terminal"]["mean"] == -2.5
    else:
        assert report["event_components"]["terms"] is None
    assert report["gate_coverage"]["available"] is False
    assert adapter.drain_reward_components()["samples"] == 0


def test_collector_capability_covers_all_vector_rows_and_independent_partial_zero_windows():
    env = RewardTensorChassis()
    adapter, collector = chassis_collector(env)
    calls = 0

    def stop_after_two():
        nonlocal calls
        calls += 1
        return calls > 2

    partial = collector.collect(7, stop_after_two)
    assert len(partial) == 6
    assert collector.last_metrics["early_stopped"] is True
    first = deepcopy(collector.last_metrics["reward_components"])
    assert (first["steps"], first["samples"]) == (2, 6)
    assert first["continuous_components"]["terms"]["height"]["density"]["mean"] == 3
    assert adapter.training_diagnostics()["rewards/height"] == 4
    assert collector.collect(0) is None
    zero = collector.last_metrics["reward_components"]
    assert (zero["steps"], zero["samples"], zero["observed_vector_steps"]) == (0, 0, 0)
    assert zero["actual_total_reward"]["mean"] is None
    assert zero["continuous_components"]["available"] is False
    assert len(collector.collect(2)) == 6
    fresh = collector.last_metrics["reward_components"]
    assert (fresh["steps"], fresh["samples"]) == (2, 6)
    assert fresh["continuous_components"]["terms"]["height"]["density"]["mean"] == 7
    assert collector.total_transitions == 12
    assert first["continuous_components"]["terms"]["height"]["density"]["mean"] == 3


def test_bad_native_source_is_rejected_at_collection_boundary_after_physical_samples_are_charged():
    env = RewardTensorChassis(invalid_tick=2)
    _, collector = chassis_collector(env)
    with pytest.raises(ValueError, match="invalid reward component telemetry.*nonfinite"):
        collector.collect(2)
    assert env.tick == 2
    assert collector.total_transitions == collector.last_metrics["transitions"] == 6
    report = collector.last_metrics["reward_components"]
    assert report["observed_vector_steps"] == 2
    assert report["steps"] == 1 and report["samples"] == 3
    assert "FloatingPointError" in report["observation_error"]
    assert collector._observation is None
    assert collector._reward_component_drain_failed is False
    collector.reset()
    assert len(collector.collect(1)) == 3
    assert collector.last_metrics["reward_components"]["samples"] == 3
    assert collector.last_metrics["reward_components"]["observation_error"] is None


def test_collection_exception_remains_primary_when_reward_drain_also_fails():
    env = RewardTensorChassis()
    adapter, collector = chassis_collector(env)
    step = adapter.step

    def broken_step(action):
        if env.tick == 2:
            raise RuntimeError("original physical collection failure")
        return step(action)

    def broken_drain():
        raise ValueError("secondary telemetry drain failure")

    adapter.step, adapter.drain_reward_components = broken_step, broken_drain
    with pytest.raises(RuntimeError, match="original physical collection failure") as caught:
        collector.collect(4)
    assert collector.total_transitions == 6
    assert collector.last_metrics["transitions"] == 6
    assert any("secondary telemetry drain failure" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("field", ["samples", "steps"])
def test_collector_rejects_telemetry_window_counts_that_differ_from_returned_samples(field):
    env = RewardTensorChassis()
    adapter, collector = chassis_collector(env)
    drain = adapter.drain_reward_components

    def mismatched_window():
        report = drain()
        report[field] += 1
        return report

    adapter.drain_reward_components = mismatched_window
    with pytest.raises(ValueError, match="window differs from returned collection samples"):
        collector.collect(2)
    assert collector.total_transitions == 6 and env.tick == 2
    assert collector.last_metrics["transitions"] == 6
    assert collector._observation is None
    assert collector._reward_component_drain_failed is False
    adapter.drain_reward_components = drain
    collector.reset()
    assert len(collector.collect(1)) == 3
    assert collector.last_metrics["reward_components"]["samples"] == 3


@pytest.mark.parametrize("after_batch", [False, True])
@pytest.mark.parametrize("steps,should_stop,error_type,pattern", [
    (-1, None, ValueError, "steps must be a nonnegative integer"),
    (True, None, ValueError, "steps must be a nonnegative integer"),
    (False, None, ValueError, "steps must be a nonnegative integer"),
    (1, 123, TypeError, "should_stop must be callable"),
])
def test_invalid_collect_arguments_preserve_original_error_and_do_not_drain_existing_window(
        after_batch, steps, should_stop, error_type, pattern):
    env = RewardTensorChassis()
    adapter, collector = chassis_collector(env)
    if after_batch:
        assert len(collector.collect(2)) == 6
    # Populate a pending producer window independently of collect. An invalid
    # collect call must not consume that window or replace the prior report.
    adapter.step(torch.zeros(3, 6))
    previous_metrics = collector.last_metrics
    previous_values = deepcopy(previous_metrics)
    previous_moments = adapter._reward_components._moments.clone()
    previous_steps = adapter._reward_components._steps
    previous_observed = adapter._reward_component_steps
    previous_elapsed = adapter._reward_component_elapsed_s
    previous_observation = collector._observation
    previous_tick, previous_transitions = env.tick, collector.total_transitions
    previous_rng = torch.get_rng_state().clone()
    drain, drain_calls = adapter.drain_reward_components, []

    def unexpected_drain():
        drain_calls.append(True)
        raise AssertionError("invalid collection arguments must not drain telemetry")

    adapter.drain_reward_components = unexpected_drain
    with pytest.raises(error_type, match=pattern):
        collector.collect(steps, should_stop)
    assert not drain_calls
    assert collector.last_metrics is previous_metrics
    assert collector.last_metrics == previous_values
    assert collector._observation is previous_observation
    assert collector._reward_component_drain_failed is False
    assert collector.total_transitions == previous_transitions and env.tick == previous_tick
    torch.testing.assert_close(torch.get_rng_state(), previous_rng, rtol=0, atol=0)
    torch.testing.assert_close(adapter._reward_components._moments, previous_moments, rtol=0, atol=0)
    assert adapter._reward_components._steps == previous_steps == 1
    assert adapter._reward_component_steps == previous_observed == 1
    assert adapter._reward_component_elapsed_s == previous_elapsed
    assert drain()["samples"] == 3


def test_drain_exception_latches_failure_and_reset_cannot_reuse_retained_window(monkeypatch):
    env = RewardTensorChassis()
    adapter, collector = chassis_collector(env)
    drain_calls, actor_calls, optimizer_calls = [], [], []
    original_act = collector.model.actor.act
    trainer = PPOTrainer(collector.model, collector.ppo_config)

    def counted_act(*args, **kwargs):
        actor_calls.append(True)
        return original_act(*args, **kwargs)

    def failed_drain():
        drain_calls.append(True)
        raise RuntimeError("native reward drain unavailable")

    def forbidden_update(*args, **kwargs):
        optimizer_calls.append(True)
        raise AssertionError("PPO must not consume a rollout whose reward drain failed")

    monkeypatch.setattr(collector.model.actor, "act", counted_act)
    monkeypatch.setattr(trainer, "update", forbidden_update)
    adapter.drain_reward_components = failed_drain
    with pytest.raises(RuntimeError, match="native reward drain unavailable"):
        trainer.update(collector.collect(2))
    assert env.tick == 2
    assert collector.total_transitions == collector.last_metrics["transitions"] == 6
    assert collector.last_metrics["vector_steps"] == 2
    assert collector._observation is None
    assert collector._reward_component_drain_failed is True
    assert adapter._reward_components._steps == 2
    assert len(drain_calls) == 1 and len(actor_calls) == 2 and not optimizer_calls
    previous_metrics = collector.last_metrics
    previous_values = deepcopy(previous_metrics)
    previous_moments = adapter._reward_components._moments.clone()
    for explicit_reset in (False, True):
        if explicit_reset:
            collector.reset()
            assert collector._observation is not None
        for steps in (2, 0):
            with pytest.raises(RuntimeError, match="reward component drain failed.*fresh environment"):
                trainer.update(collector.collect(steps))
            assert env.tick == 2 and collector.total_transitions == 6
            assert collector.last_metrics is previous_metrics
            assert collector.last_metrics == previous_values
            assert collector._reward_component_drain_failed is True
            assert len(drain_calls) == 1 and len(actor_calls) == 2 and not optimizer_calls
            torch.testing.assert_close(adapter._reward_components._moments, previous_moments, rtol=0, atol=0)


class RewardPackedFixture(PackedFixture):
    """Explicit telemetry trait; observation metadata remains the base fixture's."""

    def __init__(self, model_config, environment_config, device):
        super().__init__(model_config, environment_config, device)
        self.enable_reward_components = False
        self.reward_statistics = RewardComponentStatistics(self.num_envs, self.dt)
        self.reward_error = None
        self.observed_steps = 0
        self.latest_density = None

    def step(self, issued_action):
        result = super().step(issued_action)
        result = replace(result, reward=torch.full((self.num_envs,), float(self.tick), device=self.device))
        self.latest_density = 2. * self.tick
        if self.enable_reward_components:
            self.observed_steps += 1
            if self.reward_error is None:
                density = torch.full_like(result.reward, self.latest_density)
                if self.tick == self.options.get("invalid_reward_tick"):
                    density[0] = float("nan")
                events = ({"bonus": (result.terminated | result.truncated).float() * .5}
                          if self.options.get("explicit_events", False) else None)
                try:
                    self.reward_statistics.observe(result.reward, {"height": density}, events)
                except (ValueError, ArithmeticError) as error:
                    self.reward_error = f"{type(error).__name__}: {error}"
        return result

    def drain_reward_components(self):
        report = self.reward_statistics.drain()
        report.update(observation_error=self.reward_error, observed_vector_steps=self.observed_steps)
        self.reward_error, self.observed_steps = None, 0
        return report

    def training_diagnostics(self):
        return {"rewards/height": self.latest_density}


def packed_config(**environment_options):
    config = packed_configuration("mlp")
    return replace(config, ppo=replace(config.ppo, epochs=1, num_minibatches=1),
                   environment={**config.environment, **environment_options})


class RecordingWriter:
    instances = []

    def __init__(self, path):
        self.path, self.scalars, self.closed = path, [], False
        self.instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("events", [False, True])
def test_training_logs_rollout_means_with_unit_tags_and_preserves_legacy_last_step(tmp_path, monkeypatch, events):
    RecordingWriter.instances = []
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=RecordingWriter))
    config = packed_config(explicit_events=events)
    created = []

    def factory(**kwargs):
        env = RewardPackedFixture(**kwargs)
        created.append(env)
        return env

    report = train_frame_policy(config, factory, "packed_env:make_env", tmp_path / "run",
                                updates=2, rollout_steps=2, seed=71, tensorboard=True)
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and report["completed_updates"] == 2
    assert report["consumed_transitions"] == 12
    for index, record in enumerate(records):
        window = record["collection"]["reward_components"]
        assert (window["steps"], window["samples"]) == (2, 6)
        assert window["actual_total_reward"]["mean"] == (1.5, 3.5)[index]
        assert window["continuous_components"]["terms"]["height"]["density"]["mean"] == (3., 7.)[index]
        assert record["environment_diagnostics"]["rewards/height"] == (4., 8.)[index]
        assert window["event_components"]["available"] is events
    assert report["last_collection_reward_components"] == records[-1]["collection"]["reward_components"]
    persisted = json.loads((tmp_path / "run/completion.json").read_text())
    assert persisted["last_collection_reward_components"] == report["last_collection_reward_components"]
    env = created[0]
    assert env.closed and env.enable_reward_components
    assert env.metadata == {"identity": "synthetic_tensor_fixture", "control_sha256": env.metadata["control_sha256"]}
    writer = RecordingWriter.instances[0]
    assert writer.closed
    values = {(tag, step): value for tag, value, step in writer.scalars}
    assert values["reward_components/density_per_s/height", 1] == 3
    assert values["reward_components/step_contribution/height", 1] == pytest.approx(.03)
    assert values["reward_components/total_step_mean", 2] == 3.5
    assert values["environment/rewards/height", 1] == 4
    event_tags = [tag for tag, _, _ in writer.scalars if "event_step_contribution" in tag]
    assert bool(event_tags) is events
    assert not any("gate" in tag for tag, _, _ in writer.scalars)


def test_training_bad_reward_source_persists_failure_tail_without_starting_ppo(tmp_path, monkeypatch):
    calls, created = [], []

    def forbidden_update(*args, **kwargs):
        calls.append(True)
        raise AssertionError("PPO must not consume invalid telemetry")

    def factory(**kwargs):
        env = RewardPackedFixture(**kwargs)
        created.append(env)
        return env

    monkeypatch.setattr(PPOTrainer, "update", forbidden_update)
    with pytest.raises(ValueError, match="invalid reward component telemetry"):
        train_frame_policy(packed_config(invalid_reward_tick=2), factory, "packed_env:make_env",
                           tmp_path / "failed", updates=1, rollout_steps=2, seed=71, tensorboard=False)
    failure = json.loads((tmp_path / "failed/failure.json").read_text())
    assert not calls
    assert failure["attempted_updates"] == failure["completed_updates"] == 0
    assert failure["consumed_transitions"] == 6
    assert failure["last_collection"]["transitions"] == 6
    tail = failure["last_collection"]["reward_components"]
    assert tail["observed_vector_steps"] == 2
    assert tail["steps"] == 1 and tail["samples"] == 3
    assert "nonfinite" in tail["observation_error"]
    assert created[0].tick == 2 and created[0].closed


def test_continuation_discards_partial_tail_then_records_fresh_full_and_zero_windows():
    config, seed = packed_config(), 71
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected_sha = _model_state_sha256(FrameActorCritic(config.model))
    with FrameContinuation.start(config, RewardPackedFixture, "packed_env:make_env", rollout_steps=4,
        training_seed=seed, retention_seed=901, expected_initial_model_sha256=expected_sha) as session:
        before = deepcopy(session.model.state_dict())
        assert session.step(should_stop=lambda: session.env.tick >= 2) is None
        tail = deepcopy(session.collector.last_metrics["reward_components"])
        assert (tail["steps"], tail["samples"]) == (2, 6)
        assert tail["actual_total_reward"]["mean"] == 1.5
        assert tail["event_components"]["available"] is False
        assert session.discarded_transitions == 6
        assert session.attempted_updates == session.update == 0
        for key, value in session.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        record = session.step()
        fresh = record["collection"]["reward_components"]
        assert (fresh["steps"], fresh["samples"]) == (4, 12)
        assert fresh["actual_total_reward"]["mean"] == 4.5
        assert record["cumulative_transitions"] == 18
        assert record["update"] == 1
        assert tail["actual_total_reward"]["mean"] == 1.5
        assert session.step(should_stop=lambda: True) is None
        assert session.collector.last_metrics["reward_components"]["samples"] == 0
        assert session.collector.last_metrics["reward_components"]["actual_total_reward"]["mean"] is None
        assert session.discarded_transitions == 6
        assert session.update == 1
    assert session.env.closed
    assert not torch.cuda.is_initialized()

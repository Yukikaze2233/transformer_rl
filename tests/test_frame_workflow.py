"""End-to-end interface tests on synthetic tensors, not robot training."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from transformer_rl.config import PPOConfig
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest
from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig
from transformer_rl.frame_training import FrameActorCritic, FrameCollector, FrameHistory
from transformer_rl.frame_checkpoint import load_frame_checkpoint, restore_rng, save_frame_checkpoint
from transformer_rl.frame_export import export_frame_policy
from transformer_rl.frame_runtime import FrameRuntime
from transformer_rl.frame_workflow import _model_state_sha256, evaluate_frame_policy, train_frame_policy
from transformer_rl.ppo import PPOTrainer
from transformer_rl.retention import AnchorRegularizer, save_anchors

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from packed_env import make_env


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(architecture="transformer", **policy_overrides):
    values = {"architecture": architecture, "frame_dim": 5, "action_dim": 2,
              "history_length": 1 if architecture == "mlp" else 4, "actor_hidden_dims": (12, 8),
              "encoder_hidden_dims": (12,), "history_latent_dim": 3,
              "d_model": 8, "num_heads": 2, "num_layers": 1, "ffn_dim": 16, **policy_overrides}
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
               "feature_names": [f"feature_{i}" for i in range(5)], "action_names": ["position", "velocity"],
               "action_bounds": [.2, .5], "target_scale": [.25, 10.], "target_offset": [.1, -.2],
               "target_units": ["rad", "rad/s"]}
    return FrameTrainConfig(FrameModelConfig(FramePolicyConfig(**values), critic_dim=3, critic_hidden=(12, 8),
                                             command_indices=(0,), initial_std=.8),
                            PPOConfig(epochs=2, num_minibatches=2, learning_rate=1e-3, target_kl=.1),
                            control, {"control": control, "num_envs": 3})


@pytest.mark.parametrize("architecture,overrides", (
    ("mlp", {}), ("history_mlp", {}), ("transformer", {}),
    ("transformer", {"readout_type": "query"}), ("transformer", {"residual_type": "gated"}),
))
def test_all_architectures_learn_resume_and_deploy(tmp_path, architecture, overrides):
    config = configuration(architecture, **overrides)
    run = tmp_path / "first"
    report = train_frame_policy(config, make_env, "packed_env:make_env", run, updates=2,
                               rollout_steps=5, tensorboard=False, seed=71)
    assert report["status"] == "completed"
    assert report["consumed_transitions"] == 30
    model, trainer, restored, update, metadata, rng = load_frame_checkpoint(report["checkpoint"])
    assert restored.to_dict() == config.to_dict() and update == 2
    assert trainer.optimizer.state
    torch.manual_seed(71)
    initial = FrameActorCritic(config.model)
    assert any(not torch.equal(value, initial.state_dict()[name]) for name, value in model.state_dict().items())
    second = train_frame_policy(config, make_env, "packed_env:make_env", tmp_path / "resume", updates=1,
        rollout_steps=5, tensorboard=False, seed=71, resume=report["checkpoint"], consumed_update_offset=2)
    assert second["final_update"] == 3 and second["cumulative_transitions"] == 45
    evaluated = evaluate_frame_policy(second["checkpoint"], make_env, config.environment, steps=12, seed=801,
                                      settle_steps=1, min_steady_samples=1)
    assert evaluated["completed_episodes"] > 0 and evaluated["success_rate"] == 1.
    bundle = tmp_path / "bundle"
    exported = export_frame_policy(second["checkpoint"], bundle, onnx=True)
    assert exported["validation"]["torchscript_max_abs_error"] < 1e-6
    runtime = FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=.01)
    current, *_ = load_frame_checkpoint(second["checkpoint"])
    frames = []
    generator = np.random.default_rng(97)
    for index in range(9):
        frame = generator.standard_normal(5).astype(np.float32)
        frames = [frame] * config.model.history_length if index == 0 else frames[1:] + [frame]
        expected = current.actor.policy(torch.from_numpy(np.stack(frames)[None])).detach().numpy()[0]
        actual = runtime.step(frame, index * .01)
        np.testing.assert_allclose(actual["mean"], expected, rtol=1e-4, atol=1e-5)
        expected_issued = np.clip(expected, -runtime.bounds, runtime.bounds)
        np.testing.assert_array_equal(actual["issued"], np.clip(actual["mean"], -runtime.bounds, runtime.bounds))
        np.testing.assert_allclose(actual["targets"], runtime.offset + runtime.scale * expected_issued, rtol=1e-4, atol=1e-4)


def test_repeat_first_history_is_row_local_owned_and_idempotent():
    config = configuration()
    env = make_env(config.model, config.environment, "cpu")
    history = FrameHistory(config.model, 3, "cpu")
    first = env.reset()
    snapshot = history.append(first)
    assert torch.equal(snapshot.frames, first.frame[:, None].expand_as(snapshot.frames))
    assert torch.equal(history.append(first).frames, snapshot.frames)
    next_result = env.step(torch.full((3, 2), .1))
    history.append(next_result.observation)
    reset = torch.tensor([True, False, False])
    history.reset(reset)
    observation = next_result.observation
    observation.frame[0] += 3
    observation.command[0] = observation.frame[0, config.model.command_indices]
    before = history._frames.clone()
    after = history.append(observation)
    assert torch.equal(after.frames[0], observation.frame[0, None].expand_as(after.frames[0]))
    assert torch.equal(after.frames[1:], before[1:])
    after.frames.fill_(100)
    assert not (history.snapshot().frames == 100).any()


def test_collector_keeps_raw_likelihood_and_correct_timeout_state():
    config = configuration()
    env = make_env(config.model, config.environment, "cpu")
    model = FrameActorCritic(config.model)
    collector = FrameCollector(env, model, config.ppo, config.control["action_bounds"])
    collector.reset()
    batch = collector.collect(7)
    assert (batch.raw_action.abs() > torch.tensor(config.control["action_bounds"])).any()
    torch.testing.assert_close(batch.issued_action, batch.raw_action.clamp(-collector.bounds, collector.bounds))
    torch.testing.assert_close(model.actor.evaluate(batch.history, batch.raw_action).log_prob, batch.old_log_prob)
    # Reset endpoints have an owned repeat-first window with zero previous action.
    frames = batch.history.frames.reshape(7, 3, 4, 5)
    assert torch.equal(frames[3, 0], frames[3, 0, -1:].expand_as(frames[3, 0]))
    assert not torch.equal(frames[3, 2, :-1], frames[3, 2, -1:].expand_as(frames[3, 2, :-1]))


def test_checkpoint_rng_optimizer_and_contract_tampering(tmp_path):
    config = configuration()
    model = FrameActorCritic(config.model)
    trainer = PPOTrainer(model, config.ppo)
    path = tmp_path / "state.pt"
    save_frame_checkpoint(path, model, trainer, config, 0, {})
    expected = torch.rand(9)
    *_, rng = load_frame_checkpoint(path)
    restore_rng(rng)
    torch.testing.assert_close(torch.rand(9), expected)
    saved = torch.load(path, weights_only=True)
    saved["model"]["actor.policy.allowed"].logical_not_()
    torch.save(saved, tmp_path / "altered.pt")
    with pytest.raises(ValueError, match="fixed architecture buffer"):
        load_frame_checkpoint(tmp_path / "altered.pt")
    saved = torch.load(path, weights_only=True)
    saved["rng"]["numpy"][2] = -1
    torch.save(saved, tmp_path / "bad_rng.pt")
    with pytest.raises(ValueError, match="RNG"):
        load_frame_checkpoint(tmp_path / "bad_rng.pt")
    with pytest.raises(FileExistsError):
        save_frame_checkpoint(path, model, trainer, config, 0, {})


def test_independent_anchor_kl_has_gradients_without_stale_ppo_ratios(tmp_path):
    config = configuration()
    teacher = FrameActorCritic(config.model).actor
    frames = torch.randn(11, 4, 5)
    mean = teacher.policy(frames).detach()
    std = teacher.log_std.exp().expand_as(mean).detach()
    path = tmp_path / "anchors.pt"
    save_anchors(path, config, frames, mean, std, "0" * 64)
    student = FrameActorCritic(config.model).actor
    anchor = AnchorRegularizer(student, config, [path], .3)
    loss = anchor()
    loss.backward()
    assert loss > 0 and student.policy.output_layer.weight.grad.abs().sum() > 0
    assert teacher.policy.output_layer.weight.grad is None
    with pytest.raises(ValueError, match="contract mismatch"):
        AnchorRegularizer(student, replace(config, control={**config.control, "policy_dt_s": .02}), [path], .3)


def test_runtime_rejects_timing_and_corrupted_graph(tmp_path):
    config = configuration()
    model = FrameActorCritic(config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    save_frame_checkpoint(checkpoint, model, PPOTrainer(model, config.ppo), config, 0, {})
    bundle = tmp_path / "bundle"
    export_frame_policy(checkpoint, bundle, onnx=False)
    with pytest.raises(ValueError, match="interval"):
        FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=.02, backend="torchscript")
    runtime = FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=.01, backend="torchscript")
    runtime.step(np.zeros(5, dtype=np.float32), 0.)
    with pytest.raises(ValueError, match="sampling interval"):
        runtime.step(np.zeros(5, dtype=np.float32), .02)
    runtime.reset()
    runtime.step(np.zeros(5, dtype=np.float32), .02)
    (bundle / "policy.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=.01, backend="torchscript")


def test_history_mlp_latent_has_configured_width_and_window_is_stateless():
    config = configuration("history_mlp")
    actor = FramePolicy(config.model.policy)
    frames = torch.randn(3, 4, 5, requires_grad=True)
    tokens = actor.encode(frames)
    assert tokens.shape == (3, 1, 3)
    gradient, = torch.autograd.grad(tokens.square().sum(), frames)
    assert (gradient.abs().sum(-1) > 0).all()
    expected = actor(frames)
    actor(torch.randn_like(frames))
    torch.testing.assert_close(actor(frames), expected)


def test_frequency_difference_penalties_preserve_physical_scaling():
    from transformer_rl.chassis_adapter import rescale_action_differences
    baseline = {"action_rate": torch.tensor(4.), "leg_action_smoothness": torch.tensor(16.), "wheel_action_smoothness": torch.tensor(16.)}
    faster = {"action_rate": torch.tensor(1.), "leg_action_smoothness": torch.tensor(1.), "wheel_action_smoothness": torch.tensor(1.)}
    adjusted = rescale_action_differences(faster, .02, .01)
    for name in baseline:
        torch.testing.assert_close(adjusted[name], baseline[name])


@pytest.mark.parametrize("stale", (False, True))
def test_control_loop_stops_and_hands_off_without_rearming(monkeypatch, stale):
    from transformer_rl import frame_runtime
    class Clock:
        now = 0.
        def monotonic(self):
            return self.now
        def sleep(self, duration):
            self.now += duration
    class Runtime:
        dt = .01
        reset_count = 0
        calls = 0
        def step(self, frame, timestamp_s):
            self.calls += 1
            return {"targets": frame, "issued": frame}
        def reset(self):
            self.reset_count += 1
    clock, runtime = Clock(), Runtime()
    monkeypatch.setattr(frame_runtime.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(frame_runtime.time, "sleep", clock.sleep)
    sent, faults = [], []
    def read_frame():
        return np.zeros(2, dtype=np.float32), clock.now - (.02 if stale else 0.)
    status = frame_runtime.run_control_loop(runtime, read_frame, lambda *args: sent.append(args),
        should_stop=lambda: len(sent) == 2, on_fault=faults.append)
    assert runtime.reset_count == 1 and len(faults) == 1
    if stale:
        assert status == "fault" and not sent and runtime.calls == 0
        assert isinstance(faults[0], TimeoutError)
    else:
        assert status == "stopped" and len(sent) == 2
        assert isinstance(faults[0], InterruptedError)


def test_learning_rollback_keeps_adam_while_stage_initialization_resets_it(tmp_path):
    config = replace(configuration(), ppo=PPOConfig(epochs=1, num_minibatches=1, learning_rate=1e-3))
    original = train_frame_policy(config, make_env, "packed_env:make_env", tmp_path / "original", updates=2,
                                  rollout_steps=4, tensorboard=False)
    original_model, *_ = load_frame_checkpoint(original["checkpoint"])
    restore_start_sha256 = _model_state_sha256(original_model)
    stage = replace(config, environment={**config.environment, "stage": "new"})
    restored = train_frame_policy(stage, make_env, "packed_env:make_env", tmp_path / "restore", updates=1,
        rollout_steps=4, tensorboard=False, restore_learning_from=original["checkpoint"], consumed_update_offset=5)
    initialized = train_frame_policy(stage, make_env, "packed_env:make_env", tmp_path / "initialize", updates=1,
        rollout_steps=4, tensorboard=False, initialize_from=original["checkpoint"])
    assert restored["start_update"] == 2 and initialized["start_update"] == 0
    _, optimizer_restored, *_ = load_frame_checkpoint(restored["checkpoint"])
    _, optimizer_initialized, *_ = load_frame_checkpoint(initialized["checkpoint"])
    assert all(value["step"].item() == 3 for value in optimizer_restored.optimizer.state.values())
    assert all(value["step"].item() == 1 for value in optimizer_initialized.optimizer.state.values())
    original_run = json.loads((tmp_path / "original" / "run.json").read_text())
    assert restore_start_sha256 != original_run["initial_model_sha256"]
    for name, report in (("restore", restored), ("initialize", initialized)):
        run = json.loads((tmp_path / name / "run.json").read_text())
        _, _, _, _, metadata, _ = load_frame_checkpoint(report["checkpoint"])
        assert run["initial_model_sha256"] == metadata["initial_model_sha256"] == restore_start_sha256


def test_restored_curriculum_clock_precedes_reset_and_advances_with_rollouts(tmp_path):
    config = replace(configuration(), ppo=PPOConfig(epochs=1, num_minibatches=1, learning_rate=1e-3))
    original = train_frame_policy(config, make_env, "packed_env:make_env", tmp_path / "original",
                                  updates=2, rollout_steps=4, tensorboard=False)
    events = []

    def clocked_factory(model_config, environment_config, device):
        env = make_env(model_config, environment_config, device)
        reset = env.reset

        def progress(updates, transitions):
            events.append(("progress", updates, transitions))

        def clocked_reset(seed=None):
            events.append(("reset", seed))
            return reset(seed=seed)

        env.set_training_progress = progress
        env.reset = clocked_reset
        return env

    changed = replace(config, environment={**config.environment, "phase": "mixed"})
    report = train_frame_policy(changed, clocked_factory, "packed_env:make_env", tmp_path / "next",
        updates=2, rollout_steps=4, tensorboard=False, restore_learning_from=original["checkpoint"],
        consumed_update_offset=2)
    assert events[0] == ("progress", 2, 24)
    assert events[1][0] == "reset"
    assert [event for event in events if event[0] == "progress"] == [
        ("progress", 2, 24), ("progress", 2, 24), ("progress", 3, 36)]
    assert report["start_update"] == 2 and report["final_update"] == 4
    assert report["cumulative_transitions"] == 48


def test_fresh_initialization_is_independent_of_environment_cpu_rng_consumption(tmp_path, monkeypatch):
    config = replace(configuration(residual_type="gated"),
                     ppo=PPOConfig(epochs=1, num_minibatches=1, learning_rate=1e-3))
    states_before_update = []
    update = PPOTrainer.update

    def observed_update(trainer, batch, **kwargs):
        states_before_update.append(_model_state_sha256(trainer.model))
        return update(trainer, batch, **kwargs)

    monkeypatch.setattr(PPOTrainer, "update", observed_update)
    initialization = []
    for index, (seed, factory_draws) in enumerate(((71, 0), (71, 4096), (72, 4096))):
        def noisy_factory(model_config, environment_config, device):
            torch.rand(factory_draws, device="cpu")
            return make_env(model_config, environment_config, device)

        directory = tmp_path / str(index)
        report = train_frame_policy(config, noisy_factory, "packed_env:make_env", directory,
                                   updates=1, rollout_steps=4, tensorboard=False, seed=seed)
        run = json.loads((directory / "run.json").read_text())
        _, _, _, _, metadata, _ = load_frame_checkpoint(report["checkpoint"])
        assert run["initial_model_sha256"] == metadata["initial_model_sha256"] == states_before_update[index]
        assert run["initial_model_hash_format"] == metadata["initial_model_hash_format"] == "sorted_named_tensor_contents_v1"
        initialization.append(run["initial_model_sha256"])
    assert initialization[0] == initialization[1]
    assert initialization[0] != initialization[2]


def test_model_state_digest_uses_names_dtype_shape_and_contents_not_storage_or_registration_order():
    first, second = torch.nn.Module(), torch.nn.Module()
    first.register_buffer("b", torch.tensor([1., 2.]))
    first.register_buffer("a", torch.tensor([[True, False], [False, True]]))
    second.register_buffer("a", first.a.clone())
    second.register_buffer("b", first.b.clone())
    digest = _model_state_sha256(first)
    assert digest == _model_state_sha256(second)
    second.b = second.b.double()
    assert digest != _model_state_sha256(second)
    second.b = first.b.clone()
    second.a = second.a.flatten()
    assert digest != _model_state_sha256(second)
    second.a = first.a.clone()
    second.b[0] += 1
    assert digest != _model_state_sha256(second)

"""Small CPU learning-state checks, without robot training or simulator assets."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from transformer_rl.config import PPOConfig
from transformer_rl.frame_checkpoint import capture_rng, load_frame_checkpoint, restore_rng
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.frame_training import FrameCollector
from transformer_rl.frame_workflow import train_frame_policy
from transformer_rl.retention import save_anchors

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from packed_env import make_env


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(architecture="transformer", readout="last"):
    policy = FramePolicyConfig(architecture=architecture, frame_dim=5, action_dim=2,
        history_length=1 if architecture == "mlp" else 4, actor_hidden_dims=(8,),
        encoder_hidden_dims=(8,), history_latent_dim=3,
        d_model=8, num_heads=2, num_layers=1, ffn_dim=12,
        residual_type="gated" if architecture == "transformer" else "add", readout_type=readout)
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{index}" for index in range(5)],
        "action_names": ["position", "velocity"], "action_bounds": [.2, .5],
        "target_scale": [.25, 10.], "target_offset": [.1, -.2], "target_units": ["rad", "rad/s"]}
    return FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(8,),
        command_indices=(0,), initial_std=.8),
        PPOConfig(epochs=2, num_minibatches=2, learning_rate=1e-3, target_kl=.5),
        control, {"control": control, "num_envs": 3})


def parent(tmp_path, architecture="transformer", readout="last"):
    config = configuration(architecture, readout)
    result = train_frame_policy(config, make_env, "packed_env:make_env", tmp_path / "parent",
        updates=2, rollout_steps=4, seed=71, tensorboard=False)
    path = Path(result["checkpoint"])
    return config, path, hashlib.sha256(path.read_bytes()).hexdigest()


def open_branch(config, path, sha, **overrides):
    options = dict(checkpoint_sha256=sha, parent_update=2, cumulative_transitions=24,
        consumed_updates=2, rollout_steps=4, training_seed=71, retention_seed=901,
        evaluation_seeds=(801, 802))
    options.update(overrides)
    return FrameContinuation.open(config, make_env, "packed_env:make_env", path, **options)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for name in left:
            assert_same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_startup_error_survives_environment_shutdown_error(tmp_path):
    config, path, sha = parent(tmp_path)
    def broken_factory(**kwargs):
        env = make_env(**kwargs)
        def reset(*_, **__):
            raise ValueError("original reset failure")
        def close():
            raise RuntimeError("secondary shutdown failure")
        env.reset, env.close = reset, close
        return env
    with pytest.raises(ValueError, match="original reset failure") as caught:
        FrameContinuation.open(config, broken_factory, "packed_env:make_env", path,
            checkpoint_sha256=sha, parent_update=2, cumulative_transitions=24,
            consumed_updates=2, rollout_steps=4, training_seed=71, retention_seed=901)
    assert any("secondary shutdown failure" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("architecture,readout", [
    ("mlp", "last"), ("history_mlp", "last"),
    ("transformer", "last"), ("transformer", "query"),
])
def test_branch_matches_manual_full_state_restore_and_clock_precedes_reset(tmp_path, architecture, readout):
    config, path, sha = parent(tmp_path, architecture, readout)
    events = []

    def noisy_factory(**kwargs):
        torch.rand(19)
        random.random()
        np.random.rand(7)
        env = make_env(**kwargs)
        env.set_training_progress = lambda updates, transitions: events.append(("progress", updates, transitions))
        original = env.reset

        def noisy_reset(seed=None):
            events.append(("reset", seed))
            torch.rand(23)
            random.random()
            np.random.rand(11)
            return original(seed)

        env.reset = noisy_reset
        return env

    branch = FrameContinuation.open(config, noisy_factory, "packed_env:make_env", path,
        checkpoint_sha256=sha, parent_update=2, cumulative_transitions=24, consumed_updates=5,
        rollout_steps=4, training_seed=71, retention_seed=901)
    assert events[:2] == [("progress", 5, 24), ("reset", 71)]
    model, trainer, _, _, _, rng = load_frame_checkpoint(path)
    assert_same(branch.model.state_dict(), model.state_dict())
    assert_same(branch.trainer.optimizer.state_dict(), trainer.optimizer.state_dict())
    assert_same(capture_rng(), rng)
    result = branch.step()
    obtained_model = deepcopy(branch.model.state_dict())
    obtained_optimizer = deepcopy(branch.trainer.optimizer.state_dict())
    obtained_rng = capture_rng()
    assert result["update"] == 3 and result["cumulative_transitions"] == 36
    assert result["consumed_updates"] == 6
    assert events[2] == ("progress", 5, 24)
    branch.close()
    env = make_env(config.model, config.environment, "cpu")
    collector = FrameCollector(env, model, config.ppo, config.control["action_bounds"])
    collector.reset(seed=71)
    restore_rng(rng)
    trainer.update(collector.collect(4), diagnostics=True)
    assert_same(model.state_dict(), obtained_model)
    assert_same(trainer.optimizer.state_dict(), obtained_optimizer)
    assert_same(capture_rng(), obtained_rng)
    env.close()


def test_private_sampler_and_complete_checkpoint_resume(tmp_path):
    config, path, sha = parent(tmp_path)
    actor = load_frame_checkpoint(path)[0].actor
    frames = torch.randn(7, 4, 5, generator=torch.Generator().manual_seed(312))
    mean = actor.policy(frames).detach() + .2
    anchor = tmp_path / "anchors.pt"
    save_anchors(anchor, config, frames, mean, actor.log_std.exp().expand_as(mean), sha)
    with open_branch(config, path, sha, anchors=[anchor], retention_coef=.3) as branch:
        branch.step()
        state = branch.sampler.state_dict()
        saved = tmp_path / "continued.pt"
        receipt = branch.save(saved)
        expected_model = deepcopy(branch.model.state_dict())
        expected_optimizer = deepcopy(branch.trainer.optimizer.state_dict())
        expected_global_rng = capture_rng()
        expected_loss = branch.sampler().detach()
        expected_next_sampler = branch.sampler.state_dict()
    resumed = open_branch(config, saved, receipt["sha256"], resume=True,
        parent_update=3, cumulative_transitions=36, consumed_updates=3,
        anchors=[anchor], retention_coef=.3)
    assert_same(resumed.sampler.state_dict(), state)
    assert_same(resumed.model.state_dict(), expected_model)
    assert_same(resumed.trainer.optimizer.state_dict(), expected_optimizer)
    assert_same(capture_rng(), expected_global_rng)
    assert_same(resumed.sampler().detach(), expected_loss)
    assert_same(resumed.sampler.state_dict(), expected_next_sampler)
    assert_same(capture_rng(), expected_global_rng)
    assert resumed.metadata["episode_state_restored"] is False
    assert resumed.metadata["history_reset"] == "repeat_first"
    assert all(resumed.collector._history._ready)
    resumed.close()


@pytest.mark.parametrize("override,match", [
    ({"checkpoint_sha256": "0" * 64}, "SHA256"),
    ({"parent_update": 3}, "clock"),
    ({"cumulative_transitions": 25}, "clock"),
    ({"consumed_updates": 1}, "clock"),
    ({"training_seed": 72}, "seed"),
    ({"retention_seed": 71}, "independent"),
    ({"retention_seed": 801}, "independent"),
    ({"resume": True}, "state"),
    ({"device": "cuda"}, "explicit index"),
    ({"device": "cuda:0"}, "CUDA learning RNG"),
])
def test_invalid_parent_rejected_before_environment(tmp_path, monkeypatch, override, match):
    config, path, sha = parent(tmp_path)
    calls = []
    monkeypatch.setattr("transformer_rl.frame_continuation._seed", lambda *_: calls.append("seed"))
    with pytest.raises(ValueError, match=match):
        open_branch(config, path, sha, **override)
    assert calls == []


def test_ppo_recipe_change_rejected_before_environment(tmp_path):
    config, path, sha = parent(tmp_path)
    changed = replace(config, ppo=replace(config.ppo, learning_rate=3e-4))
    with pytest.raises(ValueError, match="PPO recipe"):
        open_branch(changed, path, sha)


@pytest.mark.parametrize("mutation,overrides,match", [
    (None, {"resume": False}, "resume=True"),
    (None, {"consumed_updates": 3}, "clock"),
    (None, {"rollout_steps": 5}, "clock"),
    (None, {"retention_seed": 902}, "seed"),
    ("source", {}, "source and device"),
    ("device", {}, "source and device"),
    ("boolean_clock", {}, "clock"),
])
def test_resume_rejects_implicit_reset_or_changed_identity(tmp_path, monkeypatch, mutation, overrides, match):
    config, path, sha = parent(tmp_path)
    saved = tmp_path / "continued.pt"
    with open_branch(config, path, sha) as branch:
        receipt = branch.save(saved)
    if mutation:
        payload = torch.load(saved, map_location="cpu", weights_only=True)
        if mutation == "source":
            payload["metadata"]["source"] = {"sha256": "0" * 64, "files": {}}
        elif mutation == "device":
            payload["metadata"]["continuation_device"] = "cuda:0"
        else:
            payload["metadata"]["continuation"]["clock"]["rollout_steps"] = True
        saved = tmp_path / "tampered.pt"
        torch.save(payload, saved)
        receipt = {"sha256": hashlib.sha256(saved.read_bytes()).hexdigest()}
    calls = []
    monkeypatch.setattr("transformer_rl.frame_continuation._seed", lambda *_: calls.append("seed"))
    with pytest.raises(ValueError, match=match):
        open_branch(config, saved, receipt["sha256"], **{"resume": True, **overrides})
    assert calls == []


def test_partial_tail_is_charged_but_never_optimized(tmp_path):
    config, path, sha = parent(tmp_path)
    with open_branch(config, path, sha) as branch:
        original_model = deepcopy(branch.model.state_dict())
        original_optimizer = deepcopy(branch.trainer.optimizer.state_dict())
        ticks = iter([False, True])
        assert branch.step(should_stop=lambda: next(ticks)) is None
        assert branch.update == 2 and branch.consumed_updates == 2
        assert branch.collected_transitions == 27 and branch.discarded_transitions == 3
        assert_same(branch.model.state_dict(), original_model)
        assert_same(branch.trainer.optimizer.state_dict(), original_optimizer)
        saved = tmp_path / "partial.pt"
        branch.save(saved)
        metadata = load_frame_checkpoint(saved)[4]
        state = metadata["continuation"]
        assert state["clock"] == {"consumed_updates": 2, "collected_transitions": 27, "rollout_steps": 4}
        assert metadata["continuation_segment"]["discarded_transitions"] == 3
        assert metadata["continuation_segment"]["fresh_transitions"] == 3


def test_failed_update_cannot_publish_or_continue(tmp_path, monkeypatch):
    config, path, sha = parent(tmp_path)
    with open_branch(config, path, sha) as branch:
        def failing_update(*_, **__):
            raise FloatingPointError("failed after some possible optimizer mutations")
        monkeypatch.setattr(branch.trainer, "update", failing_update)
        with pytest.raises(FloatingPointError):
            branch.step()
        assert branch.consumed_updates == 3 and branch.collected_transitions == 36
        with pytest.raises(RuntimeError, match="invalid"):
            branch.save(tmp_path / "unsafe.pt")
        with pytest.raises(RuntimeError, match="invalid"):
            branch.step()
        assert not (tmp_path / "unsafe.pt").exists()

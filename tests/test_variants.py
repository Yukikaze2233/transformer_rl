"""Architecture isolation, synthetic gradients, migration and ONNX equivalence."""
from dataclasses import replace
import json

import pytest
import torch

from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.export import _synthetic_history, export_policy
from transformer_rl.model import (
    ActorCritic, GaussianActor, HistoryMLPActor, TimeAwareActor, WindowGRUActor,
)
from transformer_rl.ppo import PPOTrainer


VARIANTS = [
    {}, {"time_encoding": "index"}, {"residual_type": "gated"},
    {"time_encoding": "index", "residual_type": "gated", "auxiliary_indices": (0, 2)},
    {"actor_type": "mlp"}, {"actor_type": "gru"},
]


@pytest.fixture(autouse=True)
def isolated_torch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        yield
    torch.set_num_threads(threads)


def make_config(variant):
    return ModelConfig(history_length=4, d_model=16, num_heads=2, num_layers=1,
                       ffn_dim=24, critic_hidden=(8,), baseline_hidden=(16, 8),
                       gru_hidden=8, **variant)


def test_shared_gaussian_base_registers_no_model_state():
    actor = GaussianActor(ModelConfig())
    assert list(actor.parameters()) == []
    assert list(actor.buffers()) == []
    assert list(actor.children()) == []


@pytest.mark.parametrize("actor_type", ["mlp", "gru"])
def test_baselines_do_not_expose_attention_representations(actor_type):
    actor = ActorCritic(make_config({"actor_type": actor_type})).actor
    assert isinstance(actor, GaussianActor)
    assert not isinstance(actor, TimeAwareActor)
    for name in ("encode", "encode_tensors", "time_features_tensors", "predict_auxiliary"):
        assert not hasattr(actor, name)


@pytest.mark.parametrize("actor_class,expected_type", [
    (TimeAwareActor, "transformer"), (HistoryMLPActor, "mlp"), (WindowGRUActor, "gru"),
])
@pytest.mark.parametrize("actor_type", ["transformer", "mlp", "gru"])
def test_direct_actor_construction_matches_described_architecture(
    actor_class, expected_type, actor_type,
):
    config = make_config({"actor_type": actor_type})
    if actor_type != expected_type:
        with pytest.raises(ValueError, match="requires actor_type"):
            actor_class(config)
    else:
        assert actor_class(config).describe()["actor_type"] == actor_type


@pytest.mark.parametrize("variant", VARIANTS)
def test_padding_gradients_precision_statelessness_and_gaussian(variant):
    config = make_config(variant)
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [0, 2, 4])
    expected = actor(history)
    poisoned = history.clone()
    poisoned.frames[~history.valid] = float("nan")
    poisoned.times[~history.valid] = float("inf")
    torch.testing.assert_close(actor(poisoned), expected, rtol=0, atol=0)
    shifted = replace(history, times=history.times + 2**30, now=history.now + 2**30)
    torch.testing.assert_close(actor(shifted), expected, rtol=0, atol=0)
    actor(replace(history, command=history.command + 1))
    torch.testing.assert_close(actor(history), expected, rtol=0, atol=0)
    assert not torch.allclose(actor(replace(history, command=history.command + 1)), expected)
    sample = actor.act(history)
    torch.testing.assert_close(actor.evaluate(history, sample.action).log_prob,
                               sample.evaluation.log_prob)
    poisoned.frames.requires_grad_()
    actor(poisoned).square().sum().backward()
    assert torch.isfinite(poisoned.frames.grad).all()
    assert torch.count_nonzero(poisoned.frames.grad[~history.valid]) == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    json.dumps(actor.describe(), allow_nan=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_and_export_all_variants(tmp_path, variant):
    pytest.importorskip("onnxruntime")
    config = make_config(variant)
    model = ActorCritic(config)
    trainer = PPOTrainer(model, PPOConfig(auxiliary_coef=0.2 if config.auxiliary_indices else 0))
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.001)
    trainer.optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 1, {"source": "synthetic"})
    restored, resumed, _, _ = load_checkpoint(path)
    assert restored.config == config and resumed.config == trainer.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    output = tmp_path / "policy.onnx"
    export_policy(path, output)
    sidecar = json.loads(output.with_suffix(".onnx.json").read_text())
    assert sidecar["actor"]["actor_type"] == config.actor_type
    assert sidecar["training_auxiliary"]["indices"] == list(config.auxiliary_indices)
    assert sidecar["training_auxiliary"]["coefficient"] == trainer.config.auxiliary_coef


def test_auxiliary_query_gradient_does_not_require_critic_inputs():
    config = make_config({"auxiliary_indices": (1, 3)})
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [2, 4])
    prediction = actor.predict_auxiliary(history)
    assert prediction.shape == (2, 2)
    prediction.square().mean().backward()
    assert actor.auxiliary_head.weight.grad.abs().sum() > 0
    assert actor.query_embedding.grad.abs().sum() > 0
    assert actor.mean_head.weight.grad is None
    with pytest.raises(ValueError, match="no configured"):
        ActorCritic(make_config({})).actor.predict_auxiliary(history)


def test_gru_skips_internal_padding_and_short_windows():
    config = make_config({"actor_type": "gru"})
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [2])
    short = replace(history, frames=history.frames[:, -2:], times=history.times[:, -2:],
                    valid=history.valid[:, -2:])
    torch.testing.assert_close(actor(short), actor(history), rtol=0, atol=0)
    holes = history.clone()
    holes.frames[:, 1] = holes.frames[:, 2]
    holes.times[:, 1] = holes.times[:, 2]
    holes.valid[:, 1] = True
    holes.valid[:, 2] = False
    holes.frames[:, 2] = float("nan")
    torch.testing.assert_close(actor(holes), actor(history), rtol=0, atol=0)


def test_index_only_ablates_position_time_encoding():
    config = make_config({"time_encoding": "index"})
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [4])
    older = replace(history, times=history.times - 0.5)
    torch.testing.assert_close(actor(older), actor(history), rtol=0, atol=0)
    changed = history.clone()
    changed.frames[..., -1] += 0.1
    assert not torch.allclose(actor(changed), actor(history))


@pytest.mark.parametrize("variant", VARIANTS)
def test_configured_preprocessing_cannot_drift(tmp_path, variant):
    model = ActorCritic(make_config(variant))
    trainer = PPOTrainer(model, PPOConfig())
    model.actor.frame_scale[0] = 2
    with pytest.raises(ValueError, match="configured preprocessing"):
        save_checkpoint(tmp_path / "invalid.pt", model, trainer, 0, {})


def test_mlp_short_window_matches_explicit_left_padding():
    config = make_config({"actor_type": "mlp"})
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [2])
    short = replace(history, frames=history.frames[:, -2:], times=history.times[:, -2:],
                    valid=history.valid[:, -2:])
    torch.testing.assert_close(actor(short), actor(history), rtol=0, atol=0)


@pytest.mark.parametrize("arguments", [
    {"actor_type": "unknown"}, {"time_encoding": "none"}, {"residual_type": "none"},
    {"auxiliary_indices": (1, 1)}, {"auxiliary_indices": (-1,)},
    {"auxiliary_indices": (29,)}, {"auxiliary_indices": (True,)},
    {"actor_type": "gru", "auxiliary_indices": (0,)}, {"baseline_hidden": ()},
    {"gru_hidden": 0},
])
def test_invalid_variant_config(arguments):
    with pytest.raises(ValueError):
        ModelConfig(**arguments)


@pytest.mark.parametrize("coefficient", [-1, float("nan"), True])
def test_invalid_auxiliary_coefficient(coefficient):
    with pytest.raises(ValueError):
        PPOConfig(auxiliary_coef=coefficient)


def test_schema_one_exact_default_migration_and_adam_order(tmp_path):
    model = ActorCritic(ModelConfig(history_length=4, critic_hidden=(8,)))
    trainer = PPOTrainer(model, PPOConfig())
    assert [name for name, _ in model.actor.named_parameters()][:2] == [
        "query_embedding", "log_std",
    ]
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.001)
    trainer.optimizer.step()
    path = tmp_path / "new.pt"
    save_checkpoint(path, model, trainer, 1, {"package_version": "original"})
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = 1
    payload.pop("source_schema_version")
    for name in ("actor_type", "time_encoding", "residual_type", "auxiliary_indices",
                 "baseline_hidden", "gru_hidden", "mean_init_scale"):
        payload["model_config"].pop(name)
    payload["ppo_config"].pop("auxiliary_coef")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    restored, resumed, _, metadata = load_checkpoint(legacy)
    assert restored.config == model.config
    assert restored.checkpoint_source_schema_version == 1
    assert metadata == {"package_version": "original"}
    for index, parameter in enumerate(restored.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.001)
    trainer.optimizer.step()
    resumed.optimizer.step()
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    migrated = tmp_path / "migrated.pt"
    save_checkpoint(migrated, restored, resumed, 2, metadata)
    assert load_checkpoint(migrated)[0].checkpoint_source_schema_version == 1
    payload["model_config"]["actor_type"] = "transformer"
    torch.save(payload, legacy)
    with pytest.raises(ValueError, match="original keys"):
        load_checkpoint(legacy)

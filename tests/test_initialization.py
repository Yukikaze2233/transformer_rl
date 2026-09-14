"""Initialization isolation, historical checkpoint compatibility and ONNX contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig, load_config
from transformer_rl.export import _synthetic_history, _tensor_args, export_policy
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer


VARIANTS = [
    {}, {"time_encoding": "index"}, {"residual_type": "gated"},
    {"time_encoding": "index", "residual_type": "gated", "auxiliary_indices": (0, 2)},
    {"actor_type": "mlp"}, {"actor_type": "gru"},
]
SCALES = [0.0, 0.01, 0.1, 1.0]


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        yield


def _config(variant):
    return ModelConfig(history_length=4, d_model=16, num_heads=2, num_layers=1,
                       ffn_dim=24, critic_hidden=(8,), baseline_hidden=(16, 8),
                       gru_hidden=8, **variant)


def _assert_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert list(actual) == list(expected)
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("seed", [0, 103, 2026])
def test_scale_changes_only_mean_head_with_identical_rng_and_parameter_order(variant, scale, seed):
    config = _config(variant)
    torch.manual_seed(seed)
    baseline = ActorCritic(config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    scaled = ActorCritic(replace(config, mean_init_scale=scale))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert list(dict(scaled.named_parameters())) == list(dict(baseline.named_parameters()))
    assert list(scaled.state_dict()) == list(baseline.state_dict())
    head = (f"actor.network.{len(scaled.actor.network) - 1}" if config.actor_type == "mlp"
            else "actor.mean_head")
    for name, actual in scaled.state_dict().items():
        expected = baseline.state_dict()[name]
        if name in (f"{head}.weight", f"{head}.bias"):
            expected = expected * scale
        assert torch.equal(actual, expected), name
    history = _synthetic_history(config, [0, 1, 4])
    baseline_evaluation = baseline.actor.act(history, deterministic=True).evaluation
    evaluation = scaled.actor.act(history, deterministic=True).evaluation
    torch.testing.assert_close(evaluation.mean, baseline_evaluation.mean * scale,
                               rtol=1e-5, atol=1e-7 * scale)
    assert torch.equal(evaluation.std, baseline_evaluation.std)
    if config.auxiliary_indices:
        assert torch.equal(scaled.actor.predict_auxiliary(history),
                           baseline.actor.predict_auxiliary(history))
    # Identical loaded weights must give identical means even when config says zero.
    scaled.load_state_dict(baseline.state_dict())
    assert torch.equal(scaled.actor(history), baseline.actor(history))
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("std", [0.1, 0.2, 0.4])
def test_initial_std_is_independent_of_mean_initialization(variant, std):
    config = _config(variant)
    rng = torch.get_rng_state().clone()
    baseline = ActorCritic(config)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    changed = ActorCritic(replace(config, initial_std=std))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, actual in changed.state_dict().items():
        if name != "actor.log_std":
            assert torch.equal(actual, baseline.state_dict()[name]), name
    history = _synthetic_history(config, [0, 1, 4])
    sample = changed.actor.act(history, deterministic=True)
    assert torch.equal(sample.action, baseline.actor(history))
    torch.testing.assert_close(sample.evaluation.std, torch.full_like(sample.action, std))


@pytest.mark.parametrize("scale", [-0.01, float("inf"), -float("inf"), float("nan"),
                                  True, False, None, "0.1"])
def test_mean_init_scale_rejects_invalid_values(scale):
    with pytest.raises(ValueError, match="mean_init_scale"):
        ModelConfig(mean_init_scale=scale)


@pytest.mark.parametrize("scale", [0, 0.01, 0.1, 1, 2.0])
def test_mean_init_scale_config_file_roundtrip(tmp_path, scale):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"mean_init_scale": scale}}))
    assert load_config(path)[0].mean_init_scale == scale
    assert ModelConfig().mean_init_scale == 1.0


def _synthetic_adam_step(model, optimizer, history):
    """Deterministic differentiable probe, with no rollout or PPO update."""
    optimizer.zero_grad(set_to_none=True)
    config = model.config
    action = torch.linspace(-0.2, 0.3, history.frames.shape[0] * config.action_dim)
    action = action.reshape(-1, config.action_dim)
    critic = torch.linspace(-0.3, 0.4, history.frames.shape[0] * config.critic_dim)
    critic = critic.reshape(-1, config.critic_dim)
    evaluation = model.actor.evaluate(history, action)
    loss = -evaluation.log_prob.mean() + model.critic(critic).square().mean()
    if config.auxiliary_indices:
        loss = loss + model.actor.predict_auxiliary(history).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scale", SCALES)
def test_scaled_checkpoint_adam_roundtrip_and_export(tmp_path, variant, scale):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = replace(_config(variant), mean_init_scale=scale)
    model = ActorCritic(config)
    trainer = PPOTrainer(model, PPOConfig())
    history = _synthetic_history(config, [0, 1, 4])
    _synthetic_adam_step(model, trainer.optimizer, history)
    # Zero initialization must still allow the output head to learn nonzero means.
    assert model.actor(history).abs().max() > 0
    path = tmp_path / "checkpoint.pt"
    rng = torch.get_rng_state().clone()
    metadata = {"source_schema": {"version": "user metadata", "nested": [1, None]}}
    save_checkpoint(path, model, trainer, 1, metadata)
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == payload["source_schema_version"] == 4
    assert payload["model_config"]["mean_init_scale"] == scale
    restored, resumed, update, restored_metadata = load_checkpoint(path)
    assert update == 1 and restored_metadata == metadata and restored.config == config
    _assert_tree_equal(restored.state_dict(), model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    assert torch.equal(restored.actor(history), model.actor(history))
    output = tmp_path / "policy.onnx"
    result = export_policy(path, output)
    assert torch.equal(torch.get_rng_state(), rng)
    sidecar = json.loads(Path(result["sidecar_path"]).read_text())
    assert sidecar["model_config"]["mean_init_scale"] == scale
    assert sidecar["checkpoint"]["source_schema_version"] == 4
    initialization = sidecar["actor"]["mean_initialization"]
    assert initialization == model.actor.describe()["mean_initialization"]
    assert initialization["scale"] == scale
    assert initialization["stored_in_weights"] is True
    assert initialization["runtime_gain"] is False
    assert "already incorporated in weights" in sidecar["output"]["semantics"]
    assert "not a runtime gain" in sidecar["output"]["semantics"]
    assert len(result["validation"]["cases"]) == 8
    # export_policy also validates all five ONNX input names/types/shapes and mean output.
    expected_loss = _synthetic_adam_step(model, trainer.optimizer, history)
    actual_loss = _synthetic_adam_step(restored, resumed.optimizer, history)
    assert torch.equal(actual_loss, expected_loss)
    _assert_tree_equal(restored.state_dict(), model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())


@pytest.fixture(scope="module", params=[
    ("79f67cf", 2, 1.0), ("e0f97a6", 3, 1.0), ("e0f97a6", 3, 0.1),
])
def schema_reference(request):
    """Use actual historical implementations, not relabeled current checkpoints.

    This regression requires the repository's historical Git objects. Modules live in
    an isolated namespace so current imports and concurrent working edits are untouched.
    """
    root = Path(__file__).resolve().parents[1]
    revision, schema, scale = request.param
    package = ModuleType(f"_initialization_reference_{revision}")
    package.schema = schema
    package.mean_init_scale = scale
    package.__path__ = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, package.__name__, package)
        for name in ("config", "types", "storage", "model", "ppo", "checkpoint"):
            source = subprocess.run(
                ["git", "show", f"{revision}:src/transformer_rl/{name}.py"],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout
            module = ModuleType(f"{package.__name__}.{name}")
            patch.setitem(sys.modules, module.__name__, module)
            setattr(package, name, module)
            exec(compile(source, f"{revision}/{name}.py", "exec"), module.__dict__)
        yield package


@pytest.mark.parametrize("variant", VARIANTS)
def test_historical_defaults_weights_mean_rng_and_next_adam_step_are_exact(
    tmp_path, schema_reference, variant,
):
    reference = schema_reference
    config = replace(_config(variant), mean_init_scale=reference.mean_init_scale)
    old_arguments = asdict(config)
    old_arguments.pop("readout_type")
    if reference.schema == 2:
        old_arguments.pop("mean_init_scale")
    rng = torch.get_rng_state().clone()
    old_model = reference.model.ActorCritic(reference.config.ModelConfig(**old_arguments))
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    model = ActorCritic(config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert list(dict(model.named_parameters())) == list(dict(old_model.named_parameters()))
    _assert_tree_equal(model.state_dict(), old_model.state_dict())
    history = _synthetic_history(config, [0, 1, 4])
    old_history = reference.types.HistoryBatch(*_tensor_args(history))
    assert torch.equal(model.actor(history), old_model.actor(old_history))
    trainer = PPOTrainer(model, PPOConfig())
    old_trainer = reference.ppo.PPOTrainer(old_model, reference.config.PPOConfig())
    _assert_tree_equal(trainer.optimizer.state_dict(), old_trainer.optimizer.state_dict())
    _synthetic_adam_step(old_model, old_trainer.optimizer, old_history)
    old_path = tmp_path / "legacy.pt"
    metadata = {"source_schema": {"original": 1}, "source_schema_version": "user-owned"}
    reference.checkpoint.save_checkpoint(old_path, old_model, old_trainer, 1, metadata)
    old_payload = torch.load(old_path, weights_only=True)
    assert old_payload["schema_version"] == old_payload["source_schema_version"] == reference.schema
    assert "readout_type" not in old_payload["model_config"]
    if reference.schema == 2:
        assert "mean_init_scale" not in old_payload["model_config"]
    restored, resumed, update, restored_metadata = load_checkpoint(old_path)
    assert update == 1 and restored_metadata == metadata
    assert restored.config == config and restored.checkpoint_source_schema_version == reference.schema
    _assert_tree_equal(restored.state_dict(), old_model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), old_trainer.optimizer.state_dict())
    old_evaluation = old_model.actor.act(old_history, deterministic=True).evaluation
    evaluation = restored.actor.act(history, deterministic=True).evaluation
    for name in ("mean", "std", "log_prob", "entropy"):
        assert torch.equal(getattr(evaluation, name), getattr(old_evaluation, name))
    expected_loss = _synthetic_adam_step(old_model, old_trainer.optimizer, old_history)
    actual_loss = _synthetic_adam_step(restored, resumed.optimizer, history)
    assert torch.equal(actual_loss, expected_loss)
    for actual, expected in zip(restored.parameters(), old_model.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            assert torch.equal(actual.grad, expected.grad)
    _assert_tree_equal(restored.state_dict(), old_model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), old_trainer.optimizer.state_dict())
    migrated = tmp_path / "migrated.pt"
    save_checkpoint(migrated, restored, resumed, 2, restored_metadata)
    payload = torch.load(migrated, weights_only=True)
    assert payload["schema_version"] == 4 and payload["source_schema_version"] == reference.schema
    assert payload["metadata"] == metadata
    _assert_tree_equal(load_checkpoint(migrated)[0].state_dict(), old_model.state_dict())
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    report = export_policy(old_path, tmp_path / "legacy.onnx")
    sidecar = json.loads(Path(report["sidecar_path"]).read_text())
    assert sidecar["checkpoint"]["source_schema_version"] == reference.schema
    assert sidecar["actor"]["mean_initialization"]["scale"] == config.mean_init_scale


def _legacy_payload(path, schema):
    model = ActorCritic(_config({}))
    trainer = PPOTrainer(model, PPOConfig())
    save_checkpoint(path, model, trainer, 0, {"source_schema": ["retained", 1]})
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = schema
    payload["source_schema_version"] = 1
    payload["model_config"].pop("readout_type")
    if schema <= 2:
        payload["model_config"].pop("mean_init_scale")
    if schema == 1:
        payload.pop("source_schema_version")
        for key in ("actor_type", "time_encoding", "residual_type", "auxiliary_indices",
                    "baseline_hidden", "gru_hidden"):
            payload["model_config"].pop(key)
        payload["ppo_config"].pop("auxiliary_coef")
    return payload


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_legacy_source_one_and_metadata_survive_migration_and_export(tmp_path, schema):
    path = tmp_path / "legacy.pt"
    payload = _legacy_payload(path, schema)
    torch.save(payload, path)
    model, trainer, update, metadata = load_checkpoint(path)
    assert model.config.mean_init_scale == 1.0
    assert model.config.readout_type == "query"
    assert model.checkpoint_source_schema_version == 1
    assert metadata == payload["metadata"]
    migrated = tmp_path / "migrated.pt"
    save_checkpoint(migrated, model, trainer, update, metadata)
    current = torch.load(migrated, weights_only=True)
    assert current["schema_version"] == 4 and current["source_schema_version"] == 1
    _assert_tree_equal(current["model_state"], payload["model_state"])
    assert load_checkpoint(migrated)[3] == metadata
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    report = export_policy(path, tmp_path / "legacy.onnx")
    sidecar = json.loads(Path(report["sidecar_path"]).read_text())
    assert sidecar["checkpoint"]["source_schema_version"] == 1


@pytest.mark.parametrize("schema", [1, 2, 3])
@pytest.mark.parametrize("mutation", [
    lambda p: p["model_config"].update(readout_type="query"),
    lambda p: p["model_config"].update(unknown=1),
    lambda p: p["model_config"].pop("initial_std"),
    lambda p: p["ppo_config"].update(unknown=1),
    lambda p: p["ppo_config"].pop("learning_rate"),
])
def test_legacy_migration_requires_exact_original_fields(tmp_path, schema, mutation):
    path = tmp_path / "legacy.pt"
    payload = _legacy_payload(path, schema)
    mutation(payload)
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=f"schema {schema} .*original keys"):
        load_checkpoint(path)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("source", [0, 5, True, "2", None])
def test_current_schema_requires_valid_source_schema(tmp_path, source):
    path = tmp_path / "checkpoint.pt"
    model = ActorCritic(_config({}))
    save_checkpoint(path, model, PPOTrainer(model, PPOConfig()), 0, {})
    payload = torch.load(path, weights_only=True)
    payload["source_schema_version"] = source
    torch.save(payload, path)
    with pytest.raises(ValueError, match="source_schema_version"):
        load_checkpoint(path)

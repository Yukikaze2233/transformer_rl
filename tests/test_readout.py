"""Auditable last-frame readout, causal gradients and synthetic deployment parity."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.export import (
    _synthetic_history, _tensor_args, _verification_cases, export_policy,
)
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer

from test_initialization import _assert_tree_equal, _synthetic_adam_step


VARIANTS = [
    {"time_encoding": position, "residual_type": residual}
    for position in ("elapsed", "index") for residual in ("add", "gated")
]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))]


@pytest.fixture(autouse=True)
def isolated_torch():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        yield
    torch.set_num_threads(threads)


def _config(**kwargs):
    return ModelConfig(readout_type="last", d_model=16, num_heads=2, num_layers=2,
                       ffn_dim=24, critic_hidden=(8,), **kwargs)


@pytest.mark.parametrize("value", ["", "first", None, True, 1])
def test_invalid_readout_rejected(value):
    with pytest.raises(ValueError, match="readout_type"):
        ModelConfig(readout_type=value)


@pytest.mark.parametrize("actor_type", ["mlp", "gru"])
def test_baselines_reject_unused_readout(actor_type):
    assert ModelConfig(actor_type=actor_type).readout_type == "query"
    with pytest.raises(ValueError, match="requires transformer"):
        ModelConfig(actor_type=actor_type, readout_type="last")


def test_stdlib_config_field_parse_and_metadata_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"readout_type": "last", "time_encoding": "index"}}))
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from transformer_rl.config import ModelConfig, load_config, config_dict; "
        "m, p, e = load_config(sys.argv[2]); "
        "assert m.readout_type == 'last' and m.time_encoding == 'index'; "
        "assert ModelConfig().readout_type == 'query'; "
        "assert config_dict(m, p, e)['model']['readout_type'] == 'last'; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-S", "-c", code, source, str(path)], check=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_count_and_description_have_no_query_state(variant):
    config = ModelConfig(readout_type="last", **variant)
    actor = ActorCritic(config).actor
    query = ActorCritic(replace(config, readout_type="query")).actor
    assert config.frame_dim == 30 and config.history_length == 16 and config.action_dim == 6
    assert not hasattr(actor, "query_embedding") and not hasattr(actor, "command_projection")
    assert not any("query_embedding" in key or "command_projection" in key
                   for key in actor.state_dict())
    count = sum(p.numel() for p in actor.parameters())
    removed = config.d_model * (config.command_dim + 2)
    assert query.describe()["parameter_count"] - count == removed == 320
    description = actor.describe()
    assert description["parameter_count"] == count
    assert description["readout_type"] == "last"
    assert "last current-frame" in description["history_encoder"]
    assert all(description["current_command"][key] for key in (
        "requires_valid_last_frame", "requires_last_time_equal_now",
        "requires_last_frame_command_equal_command",
    ))
    assert description["current_command"]["supports_empty_history"] is False
    json.dumps(description, allow_nan=False)


@pytest.mark.parametrize("method", ["forward", "act", "evaluate", "encode", "predict_auxiliary"])
@pytest.mark.parametrize("corruption,match", [
    ("empty", "valid last frame"), ("last_padding", "valid last frame"),
    ("old_current_frame", "time == now"), ("external_command_only", "command == command"),
    ("frame_command_only", "command == command"),
])
def test_checked_apis_reject_unsupported_inputs(method, corruption, match):
    config = _config(auxiliary_indices=(0, 2))
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [1, 16])
    if corruption == "empty":
        history.valid[0] = False
    elif corruption == "last_padding":
        history.valid[1, -1] = False
    elif corruption == "old_current_frame":
        history = replace(history, now=history.now + 0.1)
    elif corruption == "external_command_only":
        history.command[0, 0] += 0.1
    else:
        history.frames[0, -1, config.proprio_dim] += 0.1
    args = (history, torch.zeros(2, 6)) if method == "evaluate" else (history,)
    with pytest.raises(ValueError, match=match):
        getattr(actor, method)(*args)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("device", DEVICES)
def test_causal_padding_full_features_and_gaussian_gradients(variant, device):
    config = _config(**variant)
    actor = ActorCritic(config).actor.to(device)
    history = _synthetic_history(config, [1, 8, 16]).to(device)
    expected = actor(history)
    assert expected.shape == (3, 6)
    assert torch.equal(actor.forward_tensors(*_tensor_args(history)), expected)
    with torch.inference_mode():
        torch.testing.assert_close(actor(history), expected)
    poisoned = history.clone()
    poisoned.frames[~history.valid] = float("nan")
    poisoned.times[~history.valid] = float("inf")
    assert torch.equal(actor(poisoned), expected)
    shifted = replace(history, times=history.times + 2**30, now=history.now + 2**30)
    assert torch.equal(actor(shifted), expected)
    changed = history.clone()
    changed.command.add_(0.7)
    start = config.proprio_dim
    changed.frames[:, -1, start : start + config.command_dim] = changed.command
    tokens = actor.encode(history)
    changed_tokens = actor.encode(changed)
    assert tokens.shape == (3, 16, config.d_model)
    assert torch.equal(changed_tokens[:, :-1], tokens[:, :-1])
    assert torch.count_nonzero(tokens[~history.valid]) == 0
    assert not torch.allclose(actor(changed), expected)
    assert torch.equal(actor(history), expected)
    poisoned.frames.requires_grad_()
    poisoned.command.requires_grad_()
    action = torch.linspace(-0.2, 0.3, 18, device=device).reshape(3, 6)
    evaluation = actor.evaluate(poisoned, action)
    (-evaluation.log_prob.mean()).backward()
    assert torch.isfinite(poisoned.frames.grad).all()
    assert torch.count_nonzero(poisoned.frames.grad[~history.valid]) == 0
    assert (poisoned.frames.grad[:, -1].abs().sum(0) > 0).all()
    assert poisoned.frames.grad[2, 0].abs().sum() > 0
    assert poisoned.command.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    # A representation before the current frame cannot depend on future frames.
    probe = history.clone()
    probe.frames.requires_grad_()
    actor.encode(probe)[2, 5, 0].backward()
    assert torch.count_nonzero(probe.frames.grad[:, 6:]) == 0
    assert probe.frames.grad[2, :6].abs().sum() > 0
    sample = actor.act(history, deterministic=True)
    assert torch.equal(sample.action, expected)
    assert torch.equal(actor.evaluate(history, sample.action).log_prob, sample.evaluation.log_prob)


@pytest.mark.parametrize("position", ["elapsed", "index"])
@pytest.mark.parametrize("length", [1, 16])
def test_additive_last_matches_standard_pytorch_causal_transformer(position, length):
    config = _config(time_encoding=position, history_length=length)
    actor = ActorCritic(config).actor
    history = _synthetic_history(config, [length, length])
    tokens = actor.frame_projection(history.frames * actor.frame_scale)
    tokens = tokens + actor.time_features_tensors(history.times, history.valid, history.now)
    forbidden = torch.ones(length, length, dtype=torch.bool).triu(1)
    for block in actor.blocks:
        reference = nn.TransformerEncoderLayer(
            config.d_model, config.num_heads, config.ffn_dim, dropout=0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        with torch.no_grad():
            reference.self_attn.in_proj_weight.copy_(block.qkv.weight)
            reference.self_attn.in_proj_bias.copy_(block.qkv.bias)
        reference.self_attn.out_proj.load_state_dict(block.attention_output.state_dict())
        reference.norm1.load_state_dict(block.attention_norm.state_dict())
        reference.norm2.load_state_dict(block.ffn_norm.state_dict())
        reference.linear1.load_state_dict(block.ffn[0].state_dict())
        reference.linear2.load_state_dict(block.ffn[2].state_dict())
        tokens = reference(tokens, src_mask=forbidden)
    expected = actor.mean_head(actor.output_norm(tokens)[:, -1])
    torch.testing.assert_close(actor(history), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scale", [0.0, 0.1, 1.0])
def test_last_initialization_checkpoint_adam_and_onnx(tmp_path, variant, scale):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = _config(**variant, auxiliary_indices=(0, 2))
    rng = torch.get_rng_state().clone()
    baseline = ActorCritic(config)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    model = ActorCritic(replace(config, mean_init_scale=scale))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in model.state_dict().items():
        expected = baseline.state_dict()[name]
        if name in ("actor.mean_head.weight", "actor.mean_head.bias"):
            expected = expected * scale
        assert torch.equal(value, expected), name
    history = _synthetic_history(config, [1, 8, 16])
    trainer = PPOTrainer(model, PPOConfig(auxiliary_coef=0.1))
    _synthetic_adam_step(model, trainer.optimizer, history)
    assert model.actor(history).abs().max() > 0
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 1, {})
    restored, resumed, update, _ = load_checkpoint(path)
    assert restored.config == model.config and update == 1
    assert restored.checkpoint_source_schema_version == 5
    _assert_tree_equal(restored.state_dict(), model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    expected_loss = _synthetic_adam_step(model, trainer.optimizer, history)
    actual_loss = _synthetic_adam_step(restored, resumed.optimizer, history)
    assert torch.equal(actual_loss, expected_loss)
    _assert_tree_equal(restored.state_dict(), model.state_dict())
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    output = tmp_path / "policy.onnx"
    rng = torch.get_rng_state().clone()
    report = export_policy(path, output)
    assert torch.equal(torch.get_rng_state(), rng)
    assert len(report["validation"]["cases"]) == 8
    sidecar = json.loads(Path(report["sidecar_path"]).read_text())
    assert sidecar["model_config"]["readout_type"] == "last"
    assert sidecar["actor"]["readout_type"] == "last"
    assert sidecar["actor"]["auxiliary_source"] == "last current-frame representation"
    assert sidecar["checkpoint"]["source_schema_version"] == 5
    assert "empty history unsupported" in sidecar["history_contract"]["reset"]
    assert "equal now exactly" in sidecar["history_contract"]["valid_times"]
    assert "equal separate command exactly" in sidecar["history_contract"]["command"]
    assert "contract-only" in sidecar["inputs"][3]["semantics"]
    cases = dict(_verification_cases(config))
    assert cases["partial_reset"].valid.sum(1).tolist() == [1, 1, 16]
    changed, full = cases["changed_current_command"], cases["full_history"]
    assert torch.equal(changed.frames[:, :-1], full.frames[:, :-1])
    assert torch.equal(changed.frames[:, -1, 16:19], changed.command)
    for case in cases.values():
        model.actor(case)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_last_cuda_cpu_forward_and_gradient_parity(variant):
    config = _config(**variant)
    cpu = ActorCritic(config).actor
    cuda = ActorCritic(config).actor.cuda()
    cuda.load_state_dict(cpu.state_dict())
    history = _synthetic_history(config, [1, 8, 16])
    frames = history.frames.requires_grad_()
    gpu_history = history.to("cuda")
    gpu_history.frames.retain_grad()
    expected = cpu(history)
    actual = cuda(gpu_history)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    cpu_frame_grad = frames.grad.clone()
    frames.grad = None
    actual.square().sum().backward()
    torch.testing.assert_close(gpu_history.frames.grad.cpu(), cpu_frame_grad, rtol=1e-4, atol=1e-5)
    for name, parameter in cpu.named_parameters():
        other = dict(cuda.named_parameters())[name]
        assert (other.grad is None) == (parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad.cpu(), parameter.grad, rtol=1e-4, atol=1e-5)

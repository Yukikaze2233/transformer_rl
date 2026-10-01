"""Frame-only actor checks with synthetic tensors; no robot or simulator runs."""

from dataclasses import replace
import io
import json

import numpy as np
import pytest
import torch
from torch import nn

from transformer_rl.config import ModelConfig
from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig
from transformer_rl.model import TimeAwareActor
from transformer_rl.types import HistoryBatch


ARCHITECTURES = ("mlp", "frame_stack_mlp", "history_mlp", "transformer")


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seeded_torch():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        yield


def small_config(architecture, **overrides):
    fields = dict(
        architecture=architecture,
        history_length=1 if architecture == "mlp" else 5,
        actor_hidden_dims=(24, 12),
        d_model=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
    )
    fields.update(overrides)
    return FramePolicyConfig(**fields)


@pytest.mark.parametrize("config,expected", [
    (FramePolicyConfig(), 50_758),
    (FramePolicyConfig(actor_hidden_dims=(512, 256, 128)), 183_430),
    (FramePolicyConfig(architecture="frame_stack_mlp", history_length=16), 185_158),
    (FramePolicyConfig(architecture="transformer", history_length=16,
                       actor_hidden_dims=(128, 64)), 178_758),
])
def test_comparison_parameter_counts_exclude_gaussian_distribution(config, expected):
    policy = FramePolicy(config)
    assert sum(parameter.numel() for parameter in policy.parameters()) == expected
    assert policy.input_size == config.frame_dim * config.history_length
    assert not any("log_std" in name or "critic" in name for name, _ in policy.named_parameters())
    assert not any(isinstance(module, nn.Dropout) for module in policy.modules())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_backward_reaches_frames_and_every_parameter(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    frames = torch.randn(7, config.history_length, config.frame_dim, requires_grad=True)
    actions = policy(frames)
    assert actions.shape == (7, config.action_dim)
    assert torch.isfinite(actions).all()
    (actions - torch.randn_like(actions)).square().mean().backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all()
    assert frames.grad[:, -1].abs().sum() > 0
    if config.history_length > 1:
        assert frames.grad[:, :-1].abs().sum() > 0
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_flattened_interface_preserves_oldest_to_newest_frame_order(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    frames = torch.randn(4, config.history_length, config.frame_dim)
    torch.testing.assert_close(policy.forward_flat(frames.flatten(1)), policy(frames), atol=0, rtol=0)
    if architecture == "mlp":
        torch.testing.assert_close(policy.forward_features(frames), frames[:, -1], atol=0, rtol=0)
        torch.testing.assert_close(policy.encode(frames), frames, atol=0, rtol=0)
    elif architecture == "frame_stack_mlp":
        torch.testing.assert_close(policy.forward_features(frames), frames.flatten(1), atol=0, rtol=0)
        torch.testing.assert_close(policy.encode(frames), frames, atol=0, rtol=0)
    else:
        torch.testing.assert_close(policy.forward_features(frames), policy.encode(frames)[:, -1], atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_train_eval_and_shuffled_batches_are_stateless(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    frames = torch.randn(7, config.history_length, config.frame_dim)
    permutation = torch.tensor([5, 0, 6, 2, 1, 4, 3])
    with torch.no_grad():
        policy.train()
        before = policy(frames)
        torch.testing.assert_close(policy(frames), before, atol=0, rtol=0)
        policy.eval()
        torch.testing.assert_close(policy(frames), before, atol=0, rtol=0)
        policy(torch.randn(3, config.history_length, config.frame_dim))
        reordered = policy(frames[permutation])[torch.argsort(permutation)]
        singleton = torch.cat([policy(row[None]) for row in frames])
    torch.testing.assert_close(reordered, before, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(singleton, before, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("residual_type", ("add", "gated"))
def test_transformer_prefix_has_no_access_to_future_frames(residual_type):
    config = small_config("transformer", residual_type=residual_type)
    policy = FramePolicy(config)
    frames = torch.randn(3, config.history_length, config.frame_dim, requires_grad=True)
    original = policy.encode(frames)
    changed = frames.detach().clone()
    changed[:, 3:] += 10 * torch.randn_like(changed[:, 3:])
    encoded = policy.encode(changed)
    assert original.shape == (3, config.history_length, config.d_model)
    torch.testing.assert_close(encoded[:, :3], original[:, :3], atol=0, rtol=0)
    assert not torch.allclose(encoded[:, 3:], original[:, 3:])
    prefix = original[:, :3]
    gradient, = torch.autograd.grad((prefix * torch.randn_like(prefix)).sum(), frames)
    assert gradient[:, :3].abs().sum() > 0
    torch.testing.assert_close(gradient[:, 3:], torch.zeros_like(gradient[:, 3:]), atol=0, rtol=0)


@pytest.mark.parametrize("architecture,overrides", (
    ("history_mlp", {}), ("transformer", {}),
    ("transformer", {"readout_type": "query"}), ("transformer", {"residual_type": "gated"}),
))
def test_current_frame_bypass_works_with_history_branch_zeroed(architecture, overrides):
    config = small_config(architecture, **overrides)
    policy = FramePolicy(config)
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if not name.startswith("head."):
                parameter.zero_()
    frames = torch.randn(3, config.history_length, config.frame_dim, requires_grad=True)
    features = policy.forward_features(frames)
    torch.testing.assert_close(features, torch.zeros_like(features), atol=0, rtol=0)
    actions = policy(frames)
    gradient, = torch.autograd.grad(actions.square().sum(), frames)
    torch.testing.assert_close(gradient[:, :-1], torch.zeros_like(gradient[:, :-1]), atol=0, rtol=0)
    assert gradient[:, -1].abs().sum() > 0
    changed = frames.detach().clone()
    changed[:, -1] += 2
    assert not torch.allclose(policy(changed), actions)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_inputs_are_already_scaled_and_are_not_reinterpreted_as_sensor_ages(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    # These slots had time-related semantics in the separate old 30D interface.
    frames = torch.arange(config.input_size, dtype=torch.float32).reshape(1, config.history_length, 35) / 100
    seen = []
    handle = policy.head[0].register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach().clone()))
    try:
        policy(frames)
    finally:
        handle.remove()
    if architecture == "frame_stack_mlp":
        torch.testing.assert_close(seen[0], frames.flatten(1), atol=0, rtol=0)
    else:
        torch.testing.assert_close(seen[0][:, :35], frames[:, -1], atol=0, rtol=0)
    if architecture == "transformer":
        projected = []
        handle = policy.frame_projection.register_forward_pre_hook(
            lambda _module, inputs: projected.append(inputs[0].detach().clone()))
        try:
            policy.encode(frames)
        finally:
            handle.remove()
        torch.testing.assert_close(projected[0], frames, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_actions_remain_raw_unsquashed_means(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    with torch.no_grad():
        policy.output_layer.weight.zero_()
        policy.output_layer.bias.fill_(3.0)
    frames = torch.randn(2, config.history_length, config.frame_dim)
    torch.testing.assert_close(policy(frames), torch.full((2, 6), 3.0), atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_config_and_weights_roundtrip_to_a_fresh_policy(architecture):
    config = small_config(architecture)
    saved_config = json.loads(json.dumps(config.to_dict()))
    assert FramePolicyConfig.from_dict(saved_config) == config
    policy = FramePolicy(config)
    archive = io.BytesIO()
    torch.save({"config": saved_config, "state_dict": policy.state_dict()}, archive)
    archive.seek(0)
    saved = torch.load(archive, map_location="cpu", weights_only=True)
    restored = FramePolicy(FramePolicyConfig.from_dict(saved["config"]))
    restored.load_state_dict(saved["state_dict"], strict=True)
    frames = torch.randn(3, config.history_length, config.frame_dim)
    torch.testing.assert_close(restored(frames), policy(frames), atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_mean_initialization_scale_is_applied_once_to_weights(architecture):
    config = small_config(architecture)
    torch.manual_seed(56)
    standard = FramePolicy(config)
    torch.manual_seed(56)
    scaled = FramePolicy(replace(config, mean_init_scale=0.2))
    frames = torch.randn(2, config.history_length, config.frame_dim)
    torch.testing.assert_close(scaled(frames), standard(frames) * .2, atol=1e-6, rtol=1e-5)
    restored = FramePolicy(replace(config, mean_init_scale=.2))
    restored.load_state_dict(scaled.state_dict(), strict=True)
    torch.testing.assert_close(restored(frames), scaled(frames), atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_wrong_input_rank_or_fixed_window_dimensions_are_rejected(architecture):
    config = small_config(architecture)
    policy = FramePolicy(config)
    invalid_frames = (
        torch.zeros(config.history_length, config.frame_dim),
        torch.zeros(2, config.input_size),
        torch.zeros(2, config.history_length + 1, config.frame_dim),
        torch.zeros(2, config.history_length, config.frame_dim + 1),
    )
    for frames in invalid_frames:
        with pytest.raises((ValueError, RuntimeError)):
            policy(frames)
    for flat in (torch.zeros(config.input_size), torch.zeros(2, config.input_size + 1),
                 torch.zeros(2, config.history_length, config.frame_dim)):
        with pytest.raises((ValueError, RuntimeError)):
            policy.forward_flat(flat)


@pytest.mark.parametrize("fields", [
    {"architecture": "unknown"}, {"frame_dim": 0}, {"frame_dim": True},
    {"action_dim": -1}, {"history_length": 0}, {"history_length": 1.5},
    {"history_length": 2}, {"actor_hidden_dims": ()}, {"actor_hidden_dims": (24, 0)},
    {"d_model": 15}, {"d_model": 18, "num_heads": 4}, {"num_heads": 0},
    {"num_layers": 0}, {"ffn_dim": 0}, {"residual_type": "unknown"},
    {"residual_type": "gated"}, {"mean_init_scale": float("nan")}, {"mean_init_scale": -1},
    {"history_latent_dim": 0}, {"history_latent_dim": True}, {"encoder_hidden_dims": []},
    {"readout_type": "unknown"}, {"readout_type": "query"},
    {"architecture": "gru"}, {"architecture": "lstm"}, {"architecture": "tcn"},
])
def test_invalid_configuration_is_rejected(fields):
    with pytest.raises((ValueError, TypeError)):
        FramePolicyConfig(**fields)


def test_unknown_saved_configuration_field_is_rejected():
    data = FramePolicyConfig().to_dict()
    data["actor_hidden_dim_typo"] = 64
    with pytest.raises((ValueError, TypeError)):
        FramePolicyConfig.from_dict(data)


@pytest.mark.parametrize("architecture,residual_type", [
    ("mlp", "add"), ("frame_stack_mlp", "add"), ("history_mlp", "add"),
    ("transformer", "add"), ("transformer", "gated"),
])
def test_torchscript_save_load_matches_single_and_multiple_environment_outputs(architecture, residual_type):
    config = small_config(architecture, residual_type=residual_type)
    policy = FramePolicy(config).eval()
    compiled = torch.jit.script(policy)
    archive = io.BytesIO()
    torch.jit.save(compiled, archive)
    archive.seek(0)
    restored = torch.jit.load(archive)
    for batch in (1, 4):
        frames = torch.randn(batch, config.history_length, config.frame_dim)
        torch.testing.assert_close(restored(frames), policy(frames), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("architecture,residual_type", [
    ("mlp", "add"), ("frame_stack_mlp", "add"), ("history_mlp", "add"),
    ("transformer", "add"), ("transformer", "gated"),
])
def test_onnx_dynamic_batch_outputs_match_pytorch(architecture, residual_type, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    config = small_config(architecture, residual_type=residual_type)
    policy = FramePolicy(config).eval()
    output = tmp_path / "frame_policy.onnx"
    torch.onnx.export(
        policy, (torch.zeros(1, config.history_length, config.frame_dim),), str(output),
        input_names=["frames"], output_names=["actions"],
        dynamic_axes={"frames": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17, dynamo=False, external_data=False,
    )
    onnx.checker.check_model(onnx.load(output), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == ["batch", config.history_length, config.frame_dim]
    assert session.get_outputs()[0].shape == ["batch", config.action_dim]
    for batch in (1, 3, 7):
        frames = torch.randn(batch, config.history_length, config.frame_dim)
        with torch.no_grad():
            expected = policy(frames).numpy()
        actual, = session.run(["actions"], {"frames": frames.numpy()})
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_current_query_readout_uses_the_complete_observed_history_without_cross_call_state():
    policy = FramePolicy(small_config("transformer", readout_type="query"))
    frames = torch.randn(3, 5, 35, requires_grad=True)
    latent = policy.forward_features(frames)
    expected = policy.query_readout(frames[:, -1], policy.encode(frames))
    torch.testing.assert_close(latent, expected, atol=0, rtol=0)
    assert latent.shape == (3, 16)
    gradient, = torch.autograd.grad((latent * torch.randn_like(latent)).sum(), frames)
    assert (gradient.abs().sum(-1) > 0).all()
    actions = policy(frames)
    policy(torch.randn_like(frames))
    torch.testing.assert_close(policy(frames), actions, atol=0, rtol=0)
    scripted = torch.jit.script(policy)
    torch.testing.assert_close(scripted(frames), actions, atol=1e-6, rtol=1e-5)


def test_original_time_aware_30d_interface_and_preprocessing_remain_independent():
    FramePolicy(FramePolicyConfig(architecture="transformer", history_length=5, actor_hidden_dims=(128, 64)))
    config = ModelConfig()
    assert config.frame_dim == 30
    assert config.history_length == 16
    assert config.time_encoding == "elapsed" and config.readout_type == "query"
    assert config.d_model == 64 and config.critic_dim == 29
    actor = TimeAwareActor(config)
    expected_scale = torch.ones(30)
    expected_scale[25:27] = 10
    expected_scale[-1] = 10
    torch.testing.assert_close(actor.frame_scale, expected_scale, atol=0, rtol=0)
    assert actor.log_std.shape == (6,)
    length = config.history_length
    times = torch.arange(length, dtype=torch.float64)[None].repeat(2, 1) * .02
    history = HistoryBatch(
        frames=torch.randn(2, length, 30), times=times,
        valid=torch.ones(2, length, dtype=torch.bool),
        command=torch.randn(2, config.command_dim), now=times[:, -1].clone(),
    )
    actions = actor(history)
    assert actions.shape == (2, 6)
    assert torch.isfinite(actions).all()

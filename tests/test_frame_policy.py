"""Frame-only actor checks with synthetic tensors; no robot or simulator runs."""

from dataclasses import replace
import io
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from transformer_rl.config import ModelConfig
from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig
from transformer_rl.model import TimeAwareActor
from transformer_rl.types import HistoryBatch


ARCHITECTURES = ("mlp", "frame_stack_mlp", "history_mlp", "transformer")
EXPORT_VARIANTS = [
    ("mlp", "add", "last", "oldest"),
    ("frame_stack_mlp", "add", "last", "oldest"),
    ("history_mlp", "add", "last", "oldest"),
    ("transformer", "add", "last", "oldest"),
    ("transformer", "gated", "last", "oldest"),
    ("transformer", "add", "last", "current"),
    ("transformer", "gated", "last", "current"),
    ("transformer", "add", "query", "current"),
    ("transformer", "gated", "query", "current"),
]


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
    {"position_reference": "unknown"}, {"position_reference": None},
    {"position_reference": 0}, {"position_reference": "current"},
    {"architecture": "frame_stack_mlp", "history_length": 5, "position_reference": "current"},
    {"architecture": "history_mlp", "history_length": 5, "position_reference": "current"},
])
def test_invalid_configuration_is_rejected(fields):
    with pytest.raises((ValueError, TypeError)):
        FramePolicyConfig(**fields)


def test_unknown_saved_configuration_field_is_rejected():
    data = FramePolicyConfig().to_dict()
    data["actor_hidden_dim_typo"] = 64
    with pytest.raises((ValueError, TypeError)):
        FramePolicyConfig.from_dict(data)


@pytest.mark.parametrize("architecture,residual_type,readout_type,position_reference", EXPORT_VARIANTS)
def test_torchscript_save_load_matches_single_and_multiple_environment_outputs(architecture, residual_type, readout_type, position_reference):
    config = small_config(architecture, residual_type=residual_type, readout_type=readout_type,
                          position_reference=position_reference)
    policy = FramePolicy(config).eval()
    compiled = torch.jit.script(policy)
    archive = io.BytesIO()
    torch.jit.save(compiled, archive)
    archive.seek(0)
    restored = torch.jit.load(archive)
    for batch in (1, 4):
        frames = torch.randn(batch, config.history_length, config.frame_dim)
        torch.testing.assert_close(restored(frames), policy(frames), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("architecture,residual_type,readout_type,position_reference", EXPORT_VARIANTS)
def test_onnx_dynamic_batch_outputs_match_pytorch(architecture, residual_type, readout_type, position_reference, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    config = small_config(architecture, residual_type=residual_type, readout_type=readout_type,
                          position_reference=position_reference)
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


def test_legacy_policy_and_full_training_configuration_keep_canonical_identities():
    from transformer_rl.frame_config import FrameTrainConfig, digest
    from test_frame_workflow import configuration

    policy = small_config("transformer", history_length=31)
    assert "position_reference" not in policy.to_dict()
    assert replace(policy, position_reference="oldest").to_dict() == policy.to_dict()
    assert FramePolicyConfig.from_dict(policy.to_dict()).position_reference == "oldest"
    # Captured from the pre-feature classes, including the asdict(model) path.
    assert digest(policy.to_dict()) == "7a8769d87b4227d632d7ff7930f6f91312370639f69a125c324d436a9a58fa02"
    config = configuration()
    legacy = config.to_dict()
    assert "position_reference" not in legacy["model"]["policy"]
    assert digest(legacy) == "bf58b4032e6e448ae95be515772550cf8fe92cdf31716074ba53b6d0f1e76fae"
    assert FrameTrainConfig.from_dict(json.loads(json.dumps(legacy))).to_dict() == legacy
    current = replace(config, model=replace(config.model, policy=replace(config.model.policy, position_reference="current")))
    expected = json.loads(json.dumps(legacy))
    expected["model"]["policy"]["position_reference"] = "current"
    assert current.to_dict() == expected
    assert FrameTrainConfig.from_dict(current.to_dict()) == current
    assert digest(current.to_dict()) != digest(legacy)


@pytest.mark.parametrize("length", (1, 11, 31, 61))
def test_oldest_position_buffers_preserve_the_original_float32_formula(length):
    config = small_config("transformer", history_length=length)
    policy = FramePolicy(config)
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, config.d_model, 2, dtype=torch.float32) / config.d_model)
    phase = torch.arange(length, dtype=torch.float32)[:, None] * frequencies
    expected = torch.cat((phase.sin(), phase.cos()), dim=-1)[None]
    assert torch.equal(policy.position_encoding, expected)
    positions = torch.arange(length)
    assert torch.equal(policy.allowed, (positions[:, None] >= positions[None, :])[None])
    clone = FramePolicy(replace(config, position_reference="oldest"))
    clone.load_state_dict(policy.state_dict(), strict=True)
    frames = torch.randn(2, length, config.frame_dim)
    torch.testing.assert_close(clone(frames), policy(frames), atol=0, rtol=0)


@pytest.mark.parametrize("residual_type,readout_type", (("add", "last"), ("gated", "last"), ("add", "query"), ("gated", "query")))
def test_current_shared_frame_ages_match_across_eleven_thirty_one_and_sixty_one_frames(residual_type, readout_type):
    shared = torch.randn(2, 11, 35)
    policies = [FramePolicy(small_config("transformer", history_length=length, position_reference="current",
                    residual_type=residual_type, readout_type=readout_type)) for length in (11, 31, 61)]
    reference = dict(policies[0].named_parameters())
    for policy in policies[1:]:
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                parameter.copy_(reference[name])
    expected_tokens = policies[0].frame_projection(shared) + policies[0].position_encoding
    for policy in policies:
        assert torch.equal(policy.position_encoding[:, -11:], policies[0].position_encoding)
        torch.testing.assert_close(policy.frame_projection(shared) + policy.position_encoding[:, -11:],
                                   expected_tokens, atol=0, rtol=0)
        assert torch.equal(policy.position_encoding[0, -1, :8], torch.zeros(8))
        assert torch.equal(policy.position_encoding[0, -1, 8:], torch.ones(8))
        frames = torch.randn(2, policy.history_length, 35, requires_grad=True)
        actions = policy(frames)
        assert actions.shape == (2, 6) and torch.isfinite(actions).all()
        actions.square().sum().backward()
        assert frames.grad is not None and torch.isfinite(frames.grad).all()
        assert frames.grad[:, :-1].abs().sum() > 0
        torch.testing.assert_close(policy.forward_flat(frames.flatten(1)), actions, atol=0, rtol=0)
        with pytest.raises(ValueError, match="dimensions"):
            policy(frames[:, :-1])
        assert policy.config.to_dict()["position_reference"] == "current"
    # Compare encoded inputs, not encoder outputs: longer windows expose more
    # causal context even when the position encoding for common ages is equal.
    ages = torch.arange(-10, 1)
    frequencies = torch.exp(-math.log(10000.) * torch.arange(0, 16, 2, dtype=torch.float32) / 16)
    phase = ages[:, None] * frequencies
    assert torch.equal(policies[0].position_encoding, torch.cat((phase.sin(), phase.cos()), -1)[None])


@pytest.mark.parametrize("position_reference", ("oldest", "current"))
def test_checkpoint_sidecars_and_anchor_pools_roundtrip_without_relabeling_buffers(tmp_path, position_reference):
    from transformer_rl.frame_checkpoint import load_frame_checkpoint, save_frame_checkpoint
    from transformer_rl.frame_training import FrameActorCritic
    from transformer_rl.ppo import PPOTrainer
    from transformer_rl.retention import AnchorRegularizer, save_anchors
    from test_frame_workflow import configuration

    config = configuration(position_reference=position_reference)
    model = FrameActorCritic(config.model)
    checkpoint = tmp_path / "checkpoint.pt"
    save_frame_checkpoint(checkpoint, model, PPOTrainer(model, config.ppo), config, 0, {})
    sidecar = json.loads((tmp_path / "checkpoint.pt.json").read_text())
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["config"] == sidecar["config"] == config.to_dict()
    if position_reference == "oldest":
        assert "position_reference" not in payload["config"]["model"]["policy"]
    else:
        assert payload["config"]["model"]["policy"]["position_reference"] == "current"
    restored, _, loaded, _, _, _ = load_frame_checkpoint(checkpoint)
    assert loaded == config
    frames = torch.randn(9, config.model.history_length, config.model.frame_dim)
    torch.testing.assert_close(restored.actor.policy(frames), model.actor.policy(frames), atol=0, rtol=0)
    mean = model.actor.policy(frames).detach()
    std = model.actor.log_std.exp().expand_as(mean).detach()
    anchor_path = tmp_path / "anchors.pt"
    save_anchors(anchor_path, config, frames, mean, std, "0" * 64)
    anchors = torch.load(anchor_path, weights_only=True)
    assert anchors["policy_config"] == config.model.policy.to_dict()
    regularizer = AnchorRegularizer(restored.actor, loaded, [anchor_path], .3)
    torch.testing.assert_close(regularizer(), torch.tensor(0.), atol=1e-7, rtol=0)
    other = "current" if position_reference == "oldest" else "oldest"
    different = replace(config, model=replace(config.model, policy=replace(config.model.policy, position_reference=other)))
    with pytest.raises(ValueError, match="contract mismatch"):
        AnchorRegularizer(FrameActorCritic(different.model).actor, different, [anchor_path], .3)
    if other == "oldest":
        payload["config"]["model"]["policy"].pop("position_reference")
    else:
        payload["config"]["model"]["policy"]["position_reference"] = other
    altered = tmp_path / "relabeled.pt"
    torch.save(payload, altered)
    with pytest.raises(ValueError, match="fixed architecture buffer differs"):
        load_frame_checkpoint(altered)


def test_current_checkpoint_export_bundle_and_runtime_keep_the_position_contract(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from transformer_rl.frame_checkpoint import save_frame_checkpoint
    from transformer_rl.frame_export import export_frame_policy
    from transformer_rl.frame_runtime import FrameRuntime
    from transformer_rl.frame_training import FrameActorCritic
    from transformer_rl.ppo import PPOTrainer
    from test_frame_workflow import configuration

    config = configuration(position_reference="current", residual_type="gated", readout_type="query", history_length=11)
    model = FrameActorCritic(config.model)
    checkpoint = tmp_path / "current.pt"
    save_frame_checkpoint(checkpoint, model, PPOTrainer(model, config.ppo), config, 0, {})
    bundle = tmp_path / "bundle"
    exported = export_frame_policy(checkpoint, bundle, onnx=True)
    assert exported["model"]["policy"]["position_reference"] == "current"
    assert exported["validation"]["torchscript_max_abs_error"] < 1e-6
    assert exported["validation"]["onnx_max_abs_error"] < 1e-5
    for backend in ("torchscript", "onnx"):
        runtime = FrameRuntime(bundle, observation_schema="tensor_fixture", policy_dt_s=.01, backend=backend)
        frame = np.linspace(-.5, .5, 5, dtype=np.float32)
        frames = torch.from_numpy(frame).reshape(1, 1, 5).expand(1, 11, 5)
        expected = model.actor.policy(frames).detach().numpy()[0]
        np.testing.assert_allclose(runtime.step(frame, 0.)["mean"], expected, atol=1e-5, rtol=1e-4)


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

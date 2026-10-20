"""Synthetic policy-contract checks; run on the designated remote machine."""
from dataclasses import asdict, replace
import io
import json

import pytest
import torch
from torch import nn

from transformer_rl.formal.policy import (
    StructuredActorCritic, StructuredGaussianActor, StructuredPolicy, StructuredPolicyConfig,
)
from transformer_rl.research.types import HistoryBatch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(previous)


def observation(batch=4):
    frames = torch.randn(batch, 5, 35)
    frames[..., 14:16] = 0
    times = torch.arange(5, dtype=torch.float64).expand(batch, -1) * 0.01
    return HistoryBatch(frames, times, torch.ones(batch, 5, dtype=torch.bool),
                        frames[:, -1, :4].clone(), times[:, -1].clone())


def test_actual_parameter_counts_and_deployment_head_removal():
    model = StructuredActorCritic()
    deployed = model.actor.policy.for_deployment()
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(model.actor) == 144887
    assert count(model.critic) == 90369
    assert count(model) == 235256 < 450000
    assert count(deployed) == 142411
    assert deployed.prediction_head is None
    assert not any(isinstance(module, (nn.GRU, nn.LSTM, nn.Dropout)) for module in model.modules())
    assert not ({id(p) for p in model.actor.parameters()} & {id(p) for p in model.critic.parameters()})


def test_configured_widths_keep_spatial_temporal_gates_queries_and_local_decoders():
    policy = StructuredPolicy()
    assert policy.body_width == 32 and policy.time_width == 64
    assert len(policy.temporal_blocks) == 2
    assert policy.spatial_block.attention_gate.weight.shape == (32, 32)
    assert all(block.attention_gate.weight.shape == (64, 64) for block in policy.temporal_blocks)
    assert all(block.ffn_gate.weight.shape == (64, 64) for block in policy.temporal_blocks)
    assert policy.readout.query_identity.shape == (5, 64)
    assert policy.leg_decoder[0].in_features == 95
    assert policy.wheel_decoder[0].in_features == 91
    mean, state, context = policy.outputs(observation().frames)
    assert (mean.shape, state.shape, context.shape) == ((4, 6), (4, 8), (4, 64))


def test_json_roundtrip_preserves_all_hidden_widths_and_critic_contract():
    config = StructuredPolicyConfig()
    encoded = json.loads(json.dumps(asdict(config)))
    assert StructuredPolicyConfig(**encoded) == config
    assert config.normalization_contract == 'fixed_manual35_191_no_running_statistics'


def test_critic_feature_assembly_preserves_fields_and_detaches_actor_estimation():
    model = StructuredActorCritic()
    environment = torch.randn(4, 188, requires_grad=True)
    _, raw_state, _ = model.actor.policy.outputs(observation().frames)
    features = model.critic_features(environment, raw_state)
    assert features.shape == (4, 191) and not features.requires_grad
    torch.testing.assert_close(features[:, :188], environment)
    torch.testing.assert_close(features[:, 188:], raw_state[:, :3])
    model.critic(features).square().mean().backward()
    assert environment.grad is None
    assert all(p.grad is None for p in model.actor.parameters())
    assert any(p.grad is not None for p in model.critic.parameters())
    with pytest.raises(ValueError, match='physical inputs'):
        model.critic_features(environment[:, :81], raw_state)
    with pytest.raises(ValueError, match='physical inputs'):
        model.critic_features(environment, raw_state.double())


def test_explicit_old_dimensions_remain_representable_without_silent_weight_migration():
    old = StructuredActorCritic(StructuredPolicyConfig(
        body_width=64, body_ffn=128, temporal_width=96, temporal_ffn=192,
        state_hidden_dim=64, leg_hidden_dims=(96, 64), prediction_hidden_dim=64,
        critic_input_dim=81,
    ))
    assert sum(parameter.numel() for parameter in old.parameters()) == 408824
    assert old.config.normalization_contract == 'fixed_manual35_81_no_running_statistics'
    with pytest.raises(RuntimeError, match='size mismatch'):
        StructuredActorCritic().load_state_dict(old.state_dict(), strict=True)


def test_temporal_prefix_is_causal_in_values_and_gradients():
    policy = StructuredPolicy()
    frames = observation().frames.requires_grad_()
    prefix = policy.encode(frames)[:, :3]
    changed = frames.detach().clone()
    changed[:, 3:] += 10 * torch.randn_like(changed[:, 3:])
    torch.testing.assert_close(prefix, policy.encode(changed)[:, :3], atol=0, rtol=0)
    gradient, = torch.autograd.grad((prefix * torch.randn_like(prefix)).sum(), frames)
    assert gradient[:, :3].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 3:]) == 0


def test_wheel_angles_are_excluded_and_wheel_speeds_remain_observable():
    policy = StructuredPolicy()
    frames = observation().frames.requires_grad_()
    before = policy(frames)
    bad = frames.detach().clone()
    bad[..., 14:16] = 12345.0
    torch.testing.assert_close(policy(bad), before, atol=0, rtol=0)
    gradient, = torch.autograd.grad(before.square().sum(), frames)
    assert torch.count_nonzero(gradient[..., 14:16]) == 0
    assert gradient[..., 20:22].abs().sum() > 0
    with pytest.raises(ValueError, match="wheel-position"):
        policy.validate_frames(bad)
    policy.validate_frames(frames)


def test_mask_semantics_match_explicit_attention_reference():
    policy = StructuredPolicy()
    width = policy.time_width
    block = policy.temporal_blocks[0]
    tokens = torch.randn(2, 5, width)
    normalized = block.attention_norm(tokens)
    qkv = block.qkv(normalized).reshape(2, 5, 3, block.heads, block.head_width)
    q, k, v = [part.transpose(1, 2) for part in qkv.unbind(2)]
    scores = (q @ k.transpose(-2, -1)) * block.head_width**-0.5
    weights = scores.masked_fill(~policy.time_allowed, float('-inf')).softmax(-1)
    attended = (weights @ v).transpose(1, 2).reshape(2, 5, width)
    expected = tokens + block.attention_gate(normalized).sigmoid() * block.attention_output(attended)
    normalized = block.ffn_norm(expected)
    expected = expected + block.ffn_gate(normalized).sigmoid() * block.ffn(normalized)
    torch.testing.assert_close(block(tokens, policy.time_allowed), expected, atol=1e-6, rtol=1e-5)
    assert policy.body_allowed.diagonal().all()
    assert policy.body_allowed[:2].all() and policy.body_allowed[:, :2].all()
    assert not policy.body_allowed[2, 5]


def test_current_raw_state_bypass_survives_zeroed_history_branch():
    policy = StructuredPolicy()
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if not name.startswith(('leg_decoder.', 'wheel_decoder.')):
                parameter.zero_()
    frames = observation().frames.requires_grad_()
    gradient, = torch.autograd.grad(policy(frames).square().sum(), frames)
    assert torch.count_nonzero(gradient[:, :-1]) == 0
    assert gradient[:, -1].abs().sum() > 0


def test_shared_parameters_receive_ppo_and_auxiliary_gradients_without_critic_leakage():
    model = StructuredActorCritic()
    history = observation()
    action = torch.randn(4, 6)
    evaluation, state, prediction = model.actor.evaluate_training(history, action, action.clamp(-1, 1))
    loss = -evaluation.log_prob.mean() + state.square().mean() + prediction.square().mean()
    loss.backward()
    for name, parameter in model.actor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    model.zero_grad(set_to_none=True)
    model.critic(torch.randn(4, model.config.critic_input_dim)).square().mean().backward()
    assert all(parameter.grad is None for parameter in model.actor.parameters())


def test_auxiliary_gradient_does_not_update_action_decoders_or_exploration():
    actor = StructuredGaussianActor()
    history = observation()
    _, state, prediction = actor.evaluate_training(history, torch.zeros(4, 6), torch.ones(4, 6))
    (state.square().mean() + prediction.square().mean()).backward()
    assert actor.policy.command_projection.weight.grad.abs().sum() > 0
    assert actor.log_std.grad is None
    assert all(p.grad is None for p in actor.policy.leg_decoder.parameters())
    assert all(p.grad is None for p in actor.policy.wheel_decoder.parameters())


def test_old_gaussian_statistics_describe_raw_unsquashed_actions():
    actor = StructuredGaussianActor()
    history = observation()
    sampled = actor.act(history)
    evaluation = actor.evaluate(history, sampled.action)
    torch.testing.assert_close(sampled.evaluation.log_prob, evaluation.log_prob, atol=0, rtol=0)
    torch.testing.assert_close(evaluation.std[0], torch.tensor([0.3] * 4 + [0.5] * 2))
    direct = torch.distributions.Normal(evaluation.mean, evaluation.std).log_prob(sampled.action).sum(-1)
    torch.testing.assert_close(evaluation.log_prob, direct, atol=0, rtol=0)
    with torch.no_grad():
        actor.policy.leg_decoder[-1].bias.fill_(12)
        actor.policy.wheel_decoder[-1].bias.fill_(-13)
    mean = actor.act(history, deterministic=True).action
    assert (mean[:, :4] > 10).all() and (mean[:, 4:] < -10).all()


def test_stateless_training_and_deployment_with_shuffled_endpoints():
    policy = StructuredPolicy()
    frames = observation(7).frames
    before = policy(frames)
    policy.eval()
    torch.testing.assert_close(policy(frames), before, atol=0, rtol=0)
    order = torch.tensor([5, 0, 6, 2, 1, 4, 3])
    torch.testing.assert_close(policy(frames[order])[torch.argsort(order)], before, atol=1e-6, rtol=1e-5)
    policy(observation(2).frames)
    torch.testing.assert_close(policy(frames), before, atol=0, rtol=0)
    torch.testing.assert_close(policy.for_deployment()(frames), before, atol=0, rtol=0)


def test_prediction_conditions_on_issued_action_not_future_observation():
    actor = StructuredGaussianActor()
    history = observation()
    zero = torch.zeros(4, 6)
    first = actor.evaluate_training(history, zero, zero)
    second = actor.evaluate_training(history, zero, torch.ones_like(zero))
    torch.testing.assert_close(first[0].mean, second[0].mean, atol=0, rtol=0)
    torch.testing.assert_close(first[1], second[1], atol=0, rtol=0)
    assert not torch.allclose(first[2], second[2])


def test_script_export_keeps_dynamic_batch_and_identical_causal_mask():
    policy = StructuredPolicy().for_deployment()
    scripted = torch.jit.script(policy)
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    stream.seek(0)
    loaded = torch.jit.load(stream)
    for batch in (1, 7):
        frames = observation(batch).frames
        torch.testing.assert_close(loaded(frames), policy(frames), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(loaded.encode(frames), policy.encode(frames), atol=1e-6, rtol=1e-5)


def test_onnx_export_keeps_dynamic_batch_and_training_heads_out():
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    policy = StructuredPolicy().for_deployment()
    stream = io.BytesIO()
    torch.onnx.export(policy, (observation(1).frames,), stream, input_names=['frames'],
                      output_names=['raw_mean'], dynamic_axes={'frames': {0: 'batch'}, 'raw_mean': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(stream.getvalue(), providers=['CPUExecutionProvider'])
    for batch in (1, 7):
        frames = observation(batch).frames
        output = torch.from_numpy(session.run(None, {'frames': frames.numpy()})[0])
        torch.testing.assert_close(output, policy(frames), atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('changes', [
    {'history_length': 0}, {'body_width': 63}, {'temporal_width': 95},
    {'temporal_layers': True}, {'gate_bias': float('nan')},
    {'estimate_log_std_min': 2}, {'mean_init_scale': -1},
    {'critic_input_dim': 0}, {'critic_input_dim': True}, {'state_hidden_dim': 0},
    {'prediction_hidden_dim': False}, {'leg_hidden_dims': [64]},
    {'wheel_hidden_dims': [64, 0]}, {'critic_hidden_dims': [256, 128, True]},
])
def test_invalid_configuration_is_rejected_before_allocation(changes):
    with pytest.raises(ValueError):
        replace(StructuredPolicyConfig(), **changes)


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nan'])
def test_invalid_boundary_observations_are_rejected(kind):
    policy = StructuredPolicy()
    frames = observation().frames
    if kind == 'shape':
        frames = frames[:, :4]
    elif kind == 'dtype':
        frames = frames.double()
    else:
        frames[0, 0, 1] = float('nan')
    with pytest.raises(ValueError):
        policy.validate_frames(frames)

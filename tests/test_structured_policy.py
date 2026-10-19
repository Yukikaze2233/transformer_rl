"""Synthetic policy-contract checks; run on the designated remote machine."""
from dataclasses import replace
import io

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
    assert count(model.actor) == 346615
    assert count(model.critic) == 62209
    assert count(model) == 408824 < 450000
    assert count(deployed) == 339627
    assert deployed.prediction_head is None
    assert not any(isinstance(module, (nn.GRU, nn.LSTM, nn.Dropout)) for module in model.modules())
    assert not ({id(p) for p in model.actor.parameters()} & {id(p) for p in model.critic.parameters()})


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
    tokens = torch.randn(2, 5, 96)
    block = policy.temporal_blocks[0]
    normalized = block.attention_norm(tokens)
    qkv = block.qkv(normalized).reshape(2, 5, 3, 4, 24)
    q, k, v = [part.transpose(1, 2) for part in qkv.unbind(2)]
    scores = (q @ k.transpose(-2, -1)) * 24**-0.5
    weights = scores.masked_fill(~policy.time_allowed, float('-inf')).softmax(-1)
    attended = (weights @ v).transpose(1, 2).reshape(2, 5, 96)
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
    model.critic(torch.randn(4, 81)).square().mean().backward()
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

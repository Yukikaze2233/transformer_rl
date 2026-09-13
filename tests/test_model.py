"""Implementation invariants only; random-network checks do not measure control quality."""
from dataclasses import fields, replace
import io
import math
import warnings

import pytest
import torch
from torch import nn

from transformer_rl.config import ModelConfig
from transformer_rl.model import ActorCritic, TimeAwareActor, ValueCritic
from transformer_rl.types import ActionSample, HistoryBatch, PolicyEvaluation


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


def _history(config, batch=3, length=None, dtype=torch.float32, device="cpu"):
    length = config.history_length if length is None else length
    times = torch.arange(length, dtype=torch.float64, device=device)[None].expand(batch, -1)
    times = times / 128 + 1000
    return HistoryBatch(
        frames=torch.randn(batch, length, config.frame_dim, dtype=dtype, device=device),
        times=times.clone(),
        valid=torch.ones(batch, length, dtype=torch.bool, device=device),
        command=torch.randn(batch, config.command_dim, dtype=dtype, device=device),
        now=times[:, -1].clone(),
    )


def _tensor_args(history):
    return tuple(getattr(history, field.name) for field in fields(history))


def test_default_architecture_and_independent_critic():
    config = ModelConfig()
    model = ActorCritic(config)
    assert config.frame_dim == 30
    assert len(model.actor.blocks) == 2
    assert model.actor.frame_projection.out_features == 64
    for block in model.actor.blocks:
        assert block.num_heads == 4
        assert block.ffn[0].out_features == 128
    assert not any(isinstance(module, nn.Dropout) for module in model.actor.modules())
    assert model.actor.log_std.shape == (6,)
    assert not model.actor.time_frequencies.requires_grad
    actor_parameters = {id(p) for p in model.actor.parameters()}
    critic_parameters = {id(p) for p in model.critic.parameters()}
    assert not actor_parameters & critic_parameters
    value = model.critic(torch.randn(1, config.critic_dim))
    assert value.shape == (1,)
    value.sum().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is not None for p in model.critic.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_policy_contract_likelihood_and_gradients(dtype):
    config = ModelConfig(history_length=5)
    actor = TimeAwareActor(config).to(dtype=dtype)
    history = _history(config, dtype=dtype)
    history.frames.requires_grad_()
    history.command.requires_grad_()
    sample = actor.act(history)
    assert isinstance(sample, ActionSample)
    assert isinstance(sample.evaluation, PolicyEvaluation)
    assert sample.action.shape == (3, 6)
    evaluation = actor.evaluate(history, sample.action)
    assert evaluation.log_prob.shape == evaluation.entropy.shape == (3,)
    assert evaluation.mean.shape == evaluation.std.shape == (3, 6)
    torch.testing.assert_close(evaluation.std, torch.full_like(evaluation.std, config.initial_std))
    for field in fields(evaluation):
        torch.testing.assert_close(
            getattr(evaluation, field.name), getattr(sample.evaluation, field.name)
        )
    expected_log_prob = (
        -0.5 * ((sample.action - evaluation.mean) / evaluation.std).square()
        - evaluation.std.log()
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    expected_entropy = (evaluation.std.log() + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
    torch.testing.assert_close(evaluation.log_prob, expected_log_prob)
    torch.testing.assert_close(evaluation.entropy, expected_entropy)
    loss = -(evaluation.log_prob + 0.03 * evaluation.entropy).mean()
    loss.backward()
    for name, parameter in actor.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.isfinite(history.frames.grad).all()
    assert history.frames.grad[:, 0].abs().sum() > 0
    assert history.command.grad.abs().sum() > 0


def test_raw_actions_are_not_squashed_or_clipped():
    config = ModelConfig(history_length=2)
    actor = TimeAwareActor(config)
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.fill_(3.0)
        actor.log_std.copy_(torch.linspace(-2, -1, config.action_dim))
    history = _history(config)
    sample = actor.act(history, deterministic=True)
    torch.testing.assert_close(sample.action, torch.full((3, 6), 3.0))
    torch.testing.assert_close(sample.action, sample.evaluation.mean)
    raw_action = torch.full((3, 6), -4.0)
    result = actor.evaluate(history, raw_action)
    expected = torch.distributions.Normal(result.mean, result.std).log_prob(raw_action).sum(-1)
    torch.testing.assert_close(result.log_prob, expected)
    torch.testing.assert_close(result.std[0], actor.log_std.exp())
    torch.testing.assert_close(result.std[0], result.std[2])


def test_causal_prefix_cannot_see_future_frames_or_current_command():
    config = ModelConfig(history_length=6)
    actor = TimeAwareActor(config)
    history = _history(config)
    baseline = actor.encode(history)
    modified = history.clone()
    modified.frames[:, 3:] = torch.randn_like(modified.frames[:, 3:]) * 17
    modified.command.add_(11)
    encoded = actor.encode(modified)
    torch.testing.assert_close(encoded[:, :3], baseline[:, :3], rtol=0, atol=0)
    assert not torch.allclose(encoded[:, 3:], baseline[:, 3:])

    history.frames.requires_grad_()
    history.command.requires_grad_()
    prefix = actor.encode(history)[:, :3]
    (prefix * torch.randn_like(prefix)).sum().backward()
    assert history.frames.grad[:, :3].abs().sum() > 0
    assert torch.count_nonzero(history.frames.grad[:, 3:]) == 0
    assert torch.count_nonzero(history.command.grad) == 0


def test_current_command_changes_query_without_rewriting_past_commands():
    config = ModelConfig(history_length=5)
    actor = TimeAwareActor(config)
    history = _history(config)
    saved_frames = history.frames.clone()
    changed = replace(history, command=history.command + torch.tensor([1.0, -2.0, 0.5]))
    before, after = actor.encode(history), actor.encode(changed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1], rtol=0, atol=0)
    assert not torch.allclose(actor(history), actor(changed))
    torch.testing.assert_close(history.frames, saved_frames, rtol=0, atol=0)
    historical = history.clone()
    start = config.proprio_dim
    historical.frames[:, 0, start : start + config.command_dim].add_(2)
    assert not torch.allclose(actor(history), actor(historical))


def test_padding_and_empty_history_cannot_contaminate_outputs_or_gradients():
    config = ModelConfig(history_length=5)
    actor = TimeAwareActor(config)
    history = _history(config)
    history.valid[0] = False
    history.valid[1, :3] = False
    clean = history.clone()
    clean.frames[~clean.valid] = 0
    clean.times[~clean.valid] = 0
    poisoned = history.clone()
    poisoned.frames[~poisoned.valid] = float("nan")
    poisoned.frames[0, 0] = float("inf")
    poisoned.times[~poisoned.valid] = float("nan")
    poisoned.times[0, 0] = float("inf")
    poisoned.times[0, 1] = -1e300
    poisoned.frames.requires_grad_()
    mean = actor(poisoned)
    assert torch.isfinite(mean).all()
    torch.testing.assert_close(mean, actor(clean), rtol=0, atol=0)
    encoded = actor.encode(poisoned)
    assert torch.count_nonzero(encoded[:, :-1][~poisoned.valid]) == 0
    mean.square().sum().backward()
    assert torch.isfinite(poisoned.frames.grad).all()
    assert torch.count_nonzero(poisoned.frames.grad[~poisoned.valid]) == 0
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


def test_absolute_time_translation_and_uptime_precision():
    config = ModelConfig(history_length=5)
    actor = TimeAwareActor(config)
    history = _history(config)
    shifted = replace(history, times=history.times + 2**30, now=history.now + 2**30)
    original_features = actor.time_features_tensors(history.times, history.valid, history.now)
    shifted_features = actor.time_features_tensors(shifted.times, shifted.valid, shifted.now)
    torch.testing.assert_close(original_features, shifted_features, rtol=0, atol=0)
    torch.testing.assert_close(actor(history), actor(shifted), rtol=0, atol=0)

    times = torch.tensor([[1e9, 1e9 + 0.001]], dtype=torch.float64)
    now = times[:, -1]
    valid = torch.ones_like(times, dtype=torch.bool)
    features = actor.time_features_tensors(times, valid, now)
    assert not torch.allclose(features[:, 0], features[:, 1])
    relative = actor.time_features_tensors(times - now[:, None], valid, torch.zeros_like(now))
    torch.testing.assert_close(features, relative, rtol=0, atol=0)
    valid[:] = False
    assert torch.count_nonzero(actor.time_features_tensors(times * float("nan"), valid, now)) == 0


def test_real_time_spacing_matters_even_with_identical_token_order():
    config = ModelConfig(history_length=4)
    actor = TimeAwareActor(config)
    history = _history(config)
    slower = replace(
        history,
        times=history.now[:, None] - 5 * (history.now[:, None] - history.times),
    )
    assert not torch.allclose(actor(history), actor(slower))


def test_tensor_path_masks_future_events_and_public_path_rejects_them():
    config = ModelConfig(history_length=4)
    actor = TimeAwareActor(config)
    history = _history(config)
    history.times[:, -1] = history.now + 1
    with pytest.raises(ValueError, match="later than now"):
        actor(history)
    expected = history.clone()
    expected.valid[:, -1] = False
    torch.testing.assert_close(actor.forward_tensors(*_tensor_args(history)), actor(expected))


def test_train_eval_grad_and_inference_paths_are_identical_and_stateless():
    config = ModelConfig(history_length=4)
    actor = TimeAwareActor(config)
    history = _history(config)
    state = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    actor.train()
    baseline = actor(history)
    actor.eval()
    with torch.no_grad():
        no_grad = actor(history)
    with torch.inference_mode():
        inference = actor(history)
    with torch.enable_grad():
        autograd = actor(history)
    for actual in (no_grad, inference, autograd, actor.forward_tensors(*_tensor_args(history))):
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    actor(_history(config, batch=1))
    torch.testing.assert_close(actor(history), baseline, rtol=0, atol=0)
    for name, tensor in actor.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,mutation,error,match",
    [
        ("frames", lambda x: x[:, :, :-1], ValueError, "shape"),
        ("frames", lambda x: x.double(), TypeError, "dtype"),
        ("frames", lambda x: x.to("meta"), ValueError, "device"),
        ("frames", lambda x: x * float("nan"), ValueError, "nonfinite"),
        ("valid", lambda x: x.float(), TypeError, "bool"),
        ("times", lambda x: x.float().half(), TypeError, "float32 or float64"),
        ("times", lambda x: x * float("nan"), ValueError, "nonfinite"),
        ("times", lambda x: x.flip(1), ValueError, "strictly increasing"),
        ("times", lambda x: x[:, :1].expand_as(x), ValueError, "strictly increasing"),
        ("command", lambda x: x * float("inf"), ValueError, "nonfinite"),
        ("command", lambda x: x[:, :1], ValueError, "shape"),
        ("now", lambda x: x.float(), TypeError, "same dtype"),
        ("now", lambda x: x[:, None], ValueError, "shape"),
        ("now", lambda x: x * float("nan"), ValueError, "nonfinite"),
    ],
)
def test_invalid_histories_are_explicit(field, mutation, error, match):
    config = ModelConfig(history_length=4)
    actor = TimeAwareActor(config)
    history = _history(config)
    bad = replace(history, **{field: mutation(getattr(history, field))})
    with pytest.raises(error, match=match):
        actor(bad)


@pytest.mark.parametrize(
    "action,error,match",
    [
        (torch.zeros(3, 1), ValueError, "shape"),
        (torch.zeros(3, 6, dtype=torch.float64), TypeError, "dtype"),
        (torch.zeros(3, 6, device="meta"), ValueError, "device"),
        (torch.full((3, 6), float("nan")), ValueError, "finite"),
    ],
)
def test_invalid_actions_cannot_broadcast(action, error, match):
    config = ModelConfig(history_length=2)
    with pytest.raises(error, match=match):
        TimeAwareActor(config).evaluate(_history(config), action)


@pytest.mark.parametrize(
    "critic,error,match",
    [
        (torch.zeros(29), ValueError, "shape"),
        (torch.zeros(2, 29, dtype=torch.float64), TypeError, "dtype"),
        (torch.zeros(2, 29, device="meta"), ValueError, "device"),
        (torch.full((2, 29), float("nan")), ValueError, "finite"),
    ],
)
def test_invalid_critic_inputs(critic, error, match):
    with pytest.raises(error, match=match):
        ValueCritic(ModelConfig())(critic)


def test_onnx_tensor_path_matches_dynamic_cpu_execution_without_traced_validation():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    config = ModelConfig(history_length=5)
    actor = TimeAwareActor(config).eval()

    class TensorMean(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, frames, times, valid, command, now):
            return self.policy.forward_tensors(frames, times, valid, command, now)

    inputs = _history(config, batch=2)
    names = [field.name for field in fields(inputs)]
    stream = io.BytesIO()
    dynamic_axes = {name: {0: "batch"} for name in (*names, "mean")}
    for name in ("frames", "times", "valid"):
        dynamic_axes[name][1] = "history"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            TensorMean(actor),
            _tensor_args(inputs),
            stream,
            input_names=names,
            output_names=["mean"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
    assert not [
        warning for warning in caught
        if issubclass(warning.category, torch.jit.TracerWarning)
    ]
    model_bytes = stream.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(model_bytes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )
    for batch, length in ((2, 5), (3, 3), (1, 1)):
        history = _history(config, batch=batch, length=length)
        history = replace(
            history, times=history.times + 2**30, now=history.now + 2**30
        )
        history.valid[0] = False
        history.frames[~history.valid] = float("nan")
        history.times[~history.valid] = float("nan")
        feed = {name: tensor.numpy() for name, tensor in zip(names, _tensor_args(history))}
        actual = torch.from_numpy(session.run(["mean"], feed)[0])
        with torch.enable_grad():
            expected = actor(history)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable in CPU verification environment",
)
def test_cuda_actor_and_critic_gradients():
    config = ModelConfig(history_length=4)
    model = ActorCritic(config).cuda()
    history = _history(config, device="cuda")
    sample = model.actor.act(history)
    value = model.critic(torch.randn(3, config.critic_dim, device="cuda"))
    (-sample.evaluation.log_prob.mean() + value.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

"""Tensor API boundary checks using declared outputs, without simulator dependencies."""
from dataclasses import fields, replace

import pytest
import torch

from transformer_rl.adapters import TensorEnvAdapter
from transformer_rl.config import ModelConfig
from transformer_rl.types import VectorObservation


@pytest.fixture
def model_config():
    return ModelConfig(proprio_dim=2, command_dim=1, action_dim=2, sensor_groups=1,
                       critic_dim=2, history_length=3, d_model=8, num_heads=2,
                       num_layers=1, ffn_dim=16, critic_hidden=(8,))


class DeclaredTensorEnv:
    """Return tensor fixtures verbatim and intentionally mutate the input action."""

    num_envs = 3
    device = "cpu:0"

    def __init__(self, config, dtype=torch.float32):
        self.raw = dict(frame=torch.zeros(3, config.frame_dim, dtype=dtype),
                        timestamp=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
                        command=torch.zeros(3, config.command_dim, dtype=dtype),
                        critic=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype))
        self.reward = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        self.terminated = torch.tensor([True, False, True])
        self.truncated = torch.tensor([False, True, True])
        self.info = dict(phase="step", final_critic=torch.tensor([[float("nan")] * 2, [10, 11],
                                                               [float("nan")] * 2], dtype=dtype),
                         final_critic_valid=torch.tensor([False, True, False]))
        self.reset_seeds = []
        self.actions = []
        self.close_count = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return self.raw, {"phase": "reset"}

    def step(self, action):
        assert not action.requires_grad
        self.actions.append(action.clone())
        action.fill_(999.0)
        return self.raw, self.reward, self.terminated, self.truncated, self.info

    def close(self):
        self.close_count += 1


def encode(raw_observation, info):
    return VectorObservation(**raw_observation)


def test_tensor_api_encoder_seed_device_and_close_forwarding(model_config):
    env = DeclaredTensorEnv(model_config)
    seen = []

    def tracked_encoder(raw_observation, info):
        seen.append((raw_observation, info["phase"]))
        return encode(raw_observation, info)

    adapter = TensorEnvAdapter(env, model_config, tracked_encoder)
    assert adapter.num_envs == 3 and adapter.device == torch.device("cpu")
    assert env.reset_seeds == [] and env.actions == []
    observation = adapter.reset(seed=42)
    action = torch.tensor([[2.0, -2.0]]).expand(3, -1).clone().requires_grad_()
    result = adapter.step(action)
    assert env.reset_seeds == [42]
    assert seen[0][0] is env.raw and [phase for _, phase in seen] == ["reset", "step"]
    torch.testing.assert_close(action, torch.tensor([[2.0, -2.0]]).expand(3, -1))
    torch.testing.assert_close(env.actions[0], action)
    torch.testing.assert_close(result.final_critic[1], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(result.observation.critic, env.raw["critic"])
    assert result.final_critic[0].isnan().all() and result.final_critic[2].isnan().all()
    assert observation.timestamp.dtype == torch.float64
    adapter.close()
    assert env.close_count == 1


def test_reset_step_metadata_and_done_flags_are_owned_snapshots(model_config):
    env = DeclaredTensorEnv(model_config)
    for value in env.raw.values():
        value.requires_grad_()
    env.reward.requires_grad_()
    env.info["final_critic"].requires_grad_()
    adapter = TensorEnvAdapter(env, model_config, encode)
    initial = adapter.reset()
    result = adapter.step(torch.zeros(3, 2))
    expected_initial = {field.name: getattr(initial, field.name).clone() for field in fields(initial)}
    expected_reward = result.reward.clone()
    expected_term, expected_trunc = result.terminated.clone(), result.truncated.clone()
    expected_final = result.final_critic.clone()
    expected_valid = result.final_critic_valid.clone()
    with torch.no_grad():
        for value in env.raw.values():
            value.fill_(100.0)
        env.reward.fill_(100.0)
        env.terminated.logical_not_()
        env.truncated.logical_not_()
        env.info["final_critic"].fill_(100.0)
        env.info["final_critic_valid"].logical_not_()
    for field in fields(initial):
        torch.testing.assert_close(getattr(initial, field.name), expected_initial[field.name])
        torch.testing.assert_close(getattr(result.observation, field.name), expected_initial[field.name])
        assert not getattr(result.observation, field.name).requires_grad
    torch.testing.assert_close(result.reward, expected_reward)
    torch.testing.assert_close(result.terminated, expected_term)
    torch.testing.assert_close(result.truncated, expected_trunc)
    torch.testing.assert_close(result.final_critic, expected_final, equal_nan=True)
    torch.testing.assert_close(result.final_critic_valid, expected_valid)
    assert not result.reward.requires_grad and not result.final_critic.requires_grad


def test_encoder_cannot_overwrite_transition_metadata_before_snapshot(model_config):
    env = DeclaredTensorEnv(model_config)

    def scratch_encoder(raw_observation, info):
        if info["phase"] == "step":
            env.reward.fill_(999.0)
            env.terminated.logical_not_()
            env.truncated.zero_()
            env.info["final_critic"].zero_()
            env.info["final_critic_valid"].zero_()
        return encode(raw_observation, info)

    result = TensorEnvAdapter(env, model_config, scratch_encoder).step(torch.zeros(3, 2))
    torch.testing.assert_close(result.reward, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(result.terminated, torch.tensor([True, False, True]))
    torch.testing.assert_close(result.truncated, torch.tensor([False, True, True]))
    torch.testing.assert_close(result.final_critic[1], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(result.final_critic_valid, torch.tensor([False, True, False]))


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_missing_final_metadata_is_allowed_only_without_nonterminal_timeouts(model_config, both, dtype):
    env = DeclaredTensorEnv(model_config, dtype=dtype)
    env.terminated.fill_(True)
    env.truncated.fill_(both)
    env.info.clear()
    result = TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2, dtype=dtype))
    assert result.final_critic.shape == (3, 2) and result.final_critic.dtype == dtype
    assert not result.final_critic_valid.any()
    assert not torch.equal(result.final_critic, result.observation.critic)


@pytest.mark.parametrize("missing", ["final_critic", "final_critic_valid", "both"])
def test_timeout_final_metadata_is_required_and_never_inferred_from_reset_obs(model_config, missing):
    env = DeclaredTensorEnv(model_config)
    for key in (["final_critic", "final_critic_valid"] if missing == "both" else [missing]):
        del env.info[key]
    env.info["final_observation"] = env.raw
    adapter = TensorEnvAdapter(env, model_config, encode)
    with pytest.raises(ValueError, match="timeouts require"):
        adapter.step(torch.zeros(3, 2))


@pytest.mark.parametrize("failure", ["valid_shape", "valid_dtype", "invalid_timeout", "final_shape", "final_dtype", "nan_timeout"])
def test_final_critic_validation_is_mask_and_shape_explicit(model_config, failure):
    env = DeclaredTensorEnv(model_config)
    if failure == "valid_shape":
        env.info["final_critic_valid"] = torch.ones(3, 1, dtype=torch.bool)
    elif failure == "valid_dtype":
        env.info["final_critic_valid"] = torch.ones(3)
    elif failure == "invalid_timeout":
        env.info["final_critic_valid"].zero_()
    elif failure == "final_shape":
        env.info["final_critic"] = torch.zeros(3, 1)
    elif failure == "final_dtype":
        env.info["final_critic"] = env.info["final_critic"].double()
    else:
        env.info["final_critic"][1, 0] = float("nan")
    with pytest.raises((TypeError, ValueError, FloatingPointError), match="final_critic"):
        TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2))


def test_partial_final_metadata_is_rejected_even_without_timeout(model_config):
    env = DeclaredTensorEnv(model_config)
    env.terminated.fill_(True)
    del env.info["final_critic_valid"]
    with pytest.raises(ValueError, match="together"):
        TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2))


@pytest.mark.parametrize("field", ["frame", "timestamp", "command", "critic"])
@pytest.mark.parametrize("method", ["reset", "step"])
def test_every_encoded_observation_row_must_be_finite(model_config, field, method):
    env = DeclaredTensorEnv(model_config)
    env.raw[field].flatten()[0] = float("nan")
    adapter = TensorEnvAdapter(env, model_config, encode)
    with pytest.raises(FloatingPointError, match=f"observation.{field}"):
        adapter.reset() if method == "reset" else adapter.step(torch.zeros(3, 2))


@pytest.mark.parametrize("field", ["reward", "terminated", "truncated"])
def test_step_scalars_are_vectors_without_implicit_broadcasting(model_config, field):
    env = DeclaredTensorEnv(model_config)
    setattr(env, field, getattr(env, field).unsqueeze(-1))
    with pytest.raises(ValueError, match=field):
        TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2))


@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_done_flags_must_be_explicit_bool_tensors(model_config, field):
    env = DeclaredTensorEnv(model_config)
    setattr(env, field, getattr(env, field).long())
    with pytest.raises(TypeError, match="bool"):
        TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2))


@pytest.mark.parametrize("failure", ["four_tuple", "missing_terminated", "info"])
def test_rejects_ambiguous_raw_step_api(model_config, monkeypatch, failure):
    env = DeclaredTensorEnv(model_config)
    if failure == "four_tuple":
        result = (env.raw, env.reward, env.terminated, {"time_outs": env.truncated})
    elif failure == "missing_terminated":
        result = (env.raw, env.reward, None, env.truncated, env.info)
    else:
        result = (env.raw, env.reward, env.terminated, env.truncated, [])
    monkeypatch.setattr(env, "step", lambda action: result)
    with pytest.raises(TypeError):
        TensorEnvAdapter(env, model_config, encode).step(torch.zeros(3, 2))


@pytest.mark.parametrize("failure", ["raw_only", "info", "encoder_type", "encoder_shape", "encoder_dtype"])
def test_rejects_invalid_reset_and_encoder_contract(model_config, monkeypatch, failure):
    env = DeclaredTensorEnv(model_config)
    encoder = encode
    if failure == "raw_only":
        monkeypatch.setattr(env, "reset", lambda seed=None: env.raw)
    elif failure == "info":
        monkeypatch.setattr(env, "reset", lambda seed=None: (env.raw, []))
    elif failure == "encoder_type":
        encoder = lambda raw, info: raw
    elif failure == "encoder_shape":
        encoder = lambda raw, info: replace(encode(raw, info), critic=torch.zeros(3, 1))
    else:
        encoder = lambda raw, info: replace(encode(raw, info), command=torch.zeros(3, 1, dtype=torch.float64))
    with pytest.raises((TypeError, ValueError)):
        TensorEnvAdapter(env, model_config, encoder).reset()


@pytest.mark.parametrize("num_envs", [0, -1, True, 3.0])
def test_environment_count_must_be_a_positive_integer(model_config, num_envs):
    env = DeclaredTensorEnv(model_config)
    env.num_envs = num_envs
    with pytest.raises(ValueError, match="num_envs"):
        TensorEnvAdapter(env, model_config, encode)
    assert env.reset_seeds == [] and env.actions == []


@pytest.mark.parametrize("failure", ["nan_action", "shape_action", "device_reward", "nan_reward"])
def test_action_and_reward_boundaries(model_config, failure):
    env = DeclaredTensorEnv(model_config)
    action = torch.zeros(3, 2)
    if failure == "nan_action":
        action.fill_(float("nan"))
    elif failure == "shape_action":
        action = torch.zeros(3, 1)
    elif failure == "device_reward":
        env.reward = torch.empty(3, device="meta")
    else:
        env.reward.fill_(float("inf"))
    with pytest.raises((ValueError, FloatingPointError)):
        TensorEnvAdapter(env, model_config, encode).step(action)
    if failure.endswith("action"):
        assert env.actions == []

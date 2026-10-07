"""Small CPU fixtures for private anchor sampling, KL and strict JSON resume."""
import base64
import hashlib
import json

import pytest
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from transformer_rl.config import PPOConfig
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.private_retention import PrivateAnchorRegularizer
from transformer_rl.retention import AnchorRegularizer, save_anchors


class DeterministicPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(10, dtype=torch.float32).reshape(2, 5) / 50)
        self.bias = nn.Parameter(torch.tensor([.1, -.2]))
        self.last_frames = None
        self.fail = False

    def forward(self, frames):
        self.last_frames = frames.detach().clone()
        if self.fail:
            raise RuntimeError("synthetic actor failure after sampling")
        return torch.nn.functional.linear(frames[:, -1], self.weight, self.bias)


class DeterministicActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = DeterministicPolicy()
        self.log_std = nn.Parameter(torch.tensor([.9, 1.1]).log())


def configuration():
    policy = FramePolicyConfig(architecture="transformer", residual_type="gated",
        frame_dim=5, action_dim=2, history_length=4, actor_hidden_dims=(8,),
        d_model=8, num_heads=2, num_layers=1, ffn_dim=16)
    control = {"policy_dt_s": .01, "observation_schema": "small_private_retention_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)], "action_names": ["first", "second"],
        "action_bounds": [.2, .5], "target_scale": [1., 1.], "target_offset": [0., 0.],
        "target_units": ["rad", "rad/s"]}
    return FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(8,), command_indices=(0,)),
        PPOConfig(), control, {"num_envs": 1})


@pytest.fixture
def anchors(tmp_path):
    config, paths, pools = configuration(), [], []
    for index, rows in enumerate((3, 7)):
        frames = torch.arange(rows * 4 * 5, dtype=torch.float32).reshape(rows, 4, 5) / 100 + 1 + 5 * index
        mean = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2) / 10 + index + .4
        std = torch.tensor([.7, 1.2]).expand(rows, -1).clone()
        path = tmp_path / f"pool_{index}.pt"
        save_anchors(path, config, frames, mean, std, str(index) * 64)
        paths.append(path)
        pools.append((frames, mean, std))
    return config, paths, pools


def regularizer(anchors, *, actor=None, coefficient=.3, seed=17, batch_size=5, paths=None):
    config, original_paths, _ = anchors
    return PrivateAnchorRegularizer(actor or DeterministicActor(), config,
        original_paths if paths is None else paths, coefficient, seed=seed, batch_size=batch_size)


def reseal(state):
    state["sha256"] = digest({key: value for key, value in state.items() if key != "sha256"})
    return state


def set_generator(state, generator):
    raw = bytes(generator.get_state().tolist())
    state["generator"] = {"encoding": "base64_uint8", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode("ascii")}


def replay(seed, sizes, batch_size, calls):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    endpoints = 0
    for _ in range(calls):
        pool = torch.randint(len(sizes), (), generator=generator, device="cpu").item()
        count = min(batch_size, sizes[pool])
        torch.randint(sizes[pool], (count,), generator=generator, device="cpu")
        endpoints += count
    return generator, endpoints


def forbid_global_rng_and_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("private retention touched a global RNG or CUDA API")
    for name in ("manual_seed", "seed", "set_rng_state"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("is_available", "is_initialized", "device_count", "init", "_lazy_init", "_lazy_call",
                 "manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed",
                 "get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all"):
        if hasattr(torch.cuda, name):
            monkeypatch.setattr(torch.cuda, name, forbidden)


def sampled_state(anchors, calls=4):
    source = regularizer(anchors)
    for _ in range(calls):
        source()
    return source.state_dict()


def assert_rejected_without_mutation(target, state, **kwargs):
    before, global_before = target.state_dict(), torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        target.load_state_dict(state, **kwargs)
    assert target.state_dict() == before
    assert torch.equal(global_before, torch.get_rng_state())


def test_private_stream_exactly_matches_legacy_file_and_replacement_sampling(anchors):
    config, paths, _ = anchors
    actor, old_actor = DeterministicActor(), DeterministicActor()
    private = regularizer(anchors, actor=actor)
    old = AnchorRegularizer(old_actor, config, paths, .3, batch_size=5)
    sizes, duplicates = [], []
    global_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        for _ in range(24):
            old_loss = old()
            before_private = torch.get_rng_state().clone()
            private_loss = private()
            assert torch.equal(before_private, torch.get_rng_state())
            torch.testing.assert_close(private_loss, old_loss, rtol=0, atol=0)
            torch.testing.assert_close(actor.policy.last_frames, old_actor.policy.last_frames, rtol=0, atol=0)
            frames = actor.policy.last_frames
            sizes.append(len(frames))
            duplicates.append(torch.unique(frames[:, -1, 0]).numel() < len(frames))
        state = private.state_dict()
        assert base64.b64decode(state["generator"]["data"]) == bytes(torch.get_rng_state().tolist())
    assert torch.equal(global_before, torch.get_rng_state())
    assert set(sizes) == {3, 5} and any(duplicates)
    assert state["draw_count"] == 24 and state["endpoint_draw_count"] == sum(sizes)


def test_closed_form_teacher_to_student_kl_and_actor_gradients(anchors):
    _, _, pools = anchors
    actor, reference = DeterministicActor(), DeterministicActor()
    generator = torch.Generator(device="cpu").manual_seed(17)
    chosen = torch.randint(2, (), generator=generator, device="cpu").item()
    indices = torch.randint(len(pools[chosen][0]), (min(5, len(pools[chosen][0])),), generator=generator, device="cpu")
    frames, mean, std = (value[indices] for value in pools[chosen])
    predicted = reference.policy(frames)
    expected = .3 * kl_divergence(Normal(mean, std), Normal(predicted, reference.log_std.exp())).sum(-1).mean()
    actual = regularizer(anchors, actor=actor)()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == () and actual.device.type == "cpu" and actual.requires_grad
    actual.backward()
    expected.backward()
    for actual_parameter, expected_parameter in zip(actor.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-6, atol=1e-6)
    assert actor.policy.weight.grad.abs().sum() > 0 and actor.log_std.grad.abs().sum() > 0
    assert all(mean.grad is None and std.grad is None for _, mean, std in pools)


def test_zero_coefficient_needs_no_pools_forward_or_sampling_and_preserves_state(anchors, monkeypatch):
    actor = DeterministicActor()
    actor.policy.fail = True
    global_before = torch.get_rng_state().clone()
    forbid_global_rng_and_cuda(monkeypatch)
    zero = regularizer(anchors, actor=actor, paths=[], coefficient=0)
    def forbidden(*args, **kwargs):
        pytest.fail("zero retention evaluated the actor or drew samples")
    monkeypatch.setattr(zero, "_indices", forbidden)
    monkeypatch.setattr(actor.policy, "forward", forbidden)
    before = zero.state_dict()
    for _ in range(3):
        loss = zero()
        assert loss.shape == () and loss.dtype == next(actor.parameters()).dtype
        assert loss.device == next(actor.parameters()).device and loss.item() == 0
        assert zero.state_dict() == before
    zero.load_state_dict(json.loads(json.dumps(before)), max_replay_calls=0)
    assert zero.state_dict() == before and before["draw_count"] == before["endpoint_draw_count"] == 0
    assert before["identities"] == before["pool_sizes"] == []
    assert torch.equal(global_before, torch.get_rng_state())


def test_initialization_positive_loss_and_load_touch_no_global_rng_or_cuda(anchors, monkeypatch):
    global_before = torch.get_rng_state().clone()
    forbid_global_rng_and_cuda(monkeypatch)
    source = regularizer(anchors)
    assert torch.equal(global_before, torch.get_rng_state())
    for _ in range(3):
        source().backward()
    state = json.loads(json.dumps(source.state_dict(), allow_nan=False))
    target = regularizer(anchors)
    target()
    target.load_state_dict(state)
    assert target.state_dict() == state and torch.equal(global_before, torch.get_rng_state())


def test_global_draws_and_positive_coefficient_do_not_change_private_sequence(anchors):
    left_actor, right_actor = DeterministicActor(), DeterministicActor()
    left = regularizer(anchors, actor=left_actor, coefficient=.1)
    right = regularizer(anchors, actor=right_actor, coefficient=.7)
    for _ in range(10):
        before = torch.get_rng_state().clone()
        left_loss = left()
        assert torch.equal(before, torch.get_rng_state())
        torch.rand(11)
        before = torch.get_rng_state().clone()
        right_loss = right()
        assert torch.equal(before, torch.get_rng_state())
        torch.testing.assert_close(left_actor.policy.last_frames, right_actor.policy.last_frames, rtol=0, atol=0)
        torch.testing.assert_close(right_loss, left_loss * 7, rtol=1e-6, atol=1e-6)
    assert left.state_dict()["generator"] == right.state_dict()["generator"]


def test_nonzero_counter_json_resume_reproduces_every_future_sample_and_loss(anchors):
    actor, restored_actor = DeterministicActor(), DeterministicActor()
    source = regularizer(anchors, actor=actor)
    for _ in range(4):
        source()
    saved = json.loads(json.dumps(source.state_dict(), allow_nan=False))
    assert saved["pool_sizes"] == [3, 7] and saved["draw_count"] == 4
    expected = []
    for _ in range(6):
        loss = source()
        expected.append((loss.detach().clone(), actor.policy.last_frames.clone()))
    target = regularizer(anchors, actor=restored_actor)
    target()
    target.load_state_dict(saved, max_replay_calls=4)
    for loss, frames in expected:
        torch.testing.assert_close(target(), loss, rtol=0, atol=0)
        torch.testing.assert_close(restored_actor.policy.last_frames, frames, rtol=0, atol=0)
    assert source.state_dict() == target.state_dict()


@pytest.mark.parametrize("fault", ["seed", "algorithm", "torch_version", "coefficient", "batch_size",
    "identities", "identity_sha", "pool_sizes", "schema_bool", "seed_bool", "batch_bool",
    "coefficient_bool", "row_bool", "count_bool", "endpoints_bool", "negative_count", "extra_field"])
def test_resealed_metadata_mismatch_is_rejected_transactionally(anchors, fault):
    state = sampled_state(anchors)
    if fault == "seed":
        state["seed"] += 1
    elif fault == "algorithm":
        state["algorithm"] = "uniform_all_rows"
    elif fault == "torch_version":
        state["torch_version"] = "different_runtime"
    elif fault == "coefficient":
        state["coefficient"] = .4
    elif fault == "batch_size":
        state["batch_size"] = 4
    elif fault == "identities":
        state["identities"].reverse()
    elif fault == "identity_sha":
        state["identities"][0]["sha256"] = "0" * 64
    elif fault == "pool_sizes":
        state["pool_sizes"].reverse()
    elif fault == "schema_bool":
        state["schema_version"] = True
    elif fault == "seed_bool":
        state["seed"] = True
    elif fault == "batch_bool":
        state["batch_size"] = True
    elif fault == "coefficient_bool":
        state["coefficient"] = True
    elif fault == "row_bool":
        state["pool_sizes"][0] = True
    elif fault == "count_bool":
        state["draw_count"] = True
    elif fault == "endpoints_bool":
        state["endpoint_draw_count"] = True
    elif fault == "negative_count":
        state["draw_count"] = -1
    else:
        state["max_replay_calls"] = 10**12
    target = regularizer(anchors)
    target()
    assert_rejected_without_mutation(target, reseal(state))


@pytest.mark.parametrize("fault", ["counter", "generator_next_call", "generator_other_seed", "endpoint_counter"])
def test_legal_resealed_generator_bytes_and_counters_require_full_replay_match(anchors, fault):
    state = sampled_state(anchors)
    if fault == "counter":
        _, endpoints = replay(state["seed"], state["pool_sizes"], state["batch_size"], state["draw_count"] + 1)
        state["draw_count"] += 1
        state["endpoint_draw_count"] = endpoints
    elif fault == "generator_next_call":
        generator, _ = replay(state["seed"], state["pool_sizes"], state["batch_size"], state["draw_count"] + 1)
        set_generator(state, generator)
    elif fault == "generator_other_seed":
        generator, _ = replay(state["seed"] + 1, state["pool_sizes"], state["batch_size"], state["draw_count"])
        set_generator(state, generator)
    else:
        maximum = state["draw_count"] * state["batch_size"]
        state["endpoint_draw_count"] += -1 if state["endpoint_draw_count"] == maximum else 1
    target = regularizer(anchors)
    target()
    assert_rejected_without_mutation(target, reseal(state))


@pytest.mark.parametrize("fault", ["outer_sha", "byte_sha", "byte_count", "encoding", "invalid_base64", "extra_generator_field"])
def test_json_generator_encoding_and_checksums_are_strict(anchors, fault):
    state = sampled_state(anchors)
    if fault == "outer_sha":
        state["sha256"] = "0" * 64
    else:
        if fault == "byte_sha":
            state["generator"]["sha256"] = "0" * 64
        elif fault == "byte_count":
            state["generator"]["bytes"] += 1
        elif fault == "encoding":
            state["generator"]["encoding"] = "other"
        elif fault == "invalid_base64":
            state["generator"]["data"] = "!" + state["generator"]["data"][1:]
        else:
            state["generator"]["trusted"] = True
        reseal(state)
    assert_rejected_without_mutation(regularizer(anchors), state)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1., "17", None])
def test_seed_requires_exact_uint32_integer(anchors, seed):
    global_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="uint32"):
        regularizer(anchors, seed=seed)
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_uint32_seed_boundaries_are_valid(anchors, seed):
    target = regularizer(anchors, seed=seed)
    state = json.loads(json.dumps(target.state_dict()))
    target.load_state_dict(state, max_replay_calls=0)
    assert state["seed"] == seed and target.state_dict() == state


@pytest.mark.parametrize("limit", [-1, True, 3., "3", None])
def test_replay_limit_requires_a_trusted_nonnegative_integer(anchors, limit):
    assert_rejected_without_mutation(regularizer(anchors), sampled_state(anchors), max_replay_calls=limit)


def test_replay_budget_rejects_before_any_replay_draw(anchors, monkeypatch):
    target, state = regularizer(anchors), sampled_state(anchors)
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget state triggered replay work")
    monkeypatch.setattr(target, "_indices", forbidden)
    assert_rejected_without_mutation(target, state, max_replay_calls=state["draw_count"] - 1)


@pytest.mark.parametrize("fault", ["positive_calls", "advanced_generator"])
def test_zero_coefficient_only_accepts_initial_generator_and_zero_counts(anchors, fault):
    target = regularizer(anchors, coefficient=0, paths=[])
    state = target.state_dict()
    if fault == "positive_calls":
        state["draw_count"] = 1
    else:
        generator = torch.Generator(device="cpu").manual_seed(state["seed"])
        torch.randint(3, (1,), generator=generator, device="cpu")
        set_generator(state, generator)
    assert_rejected_without_mutation(target, reseal(state))


@pytest.mark.parametrize("fault", ["modified", "missing"])
def test_resume_rechecks_actual_anchor_files_without_changing_live_state(anchors, fault):
    _, paths, _ = anchors
    target, state = regularizer(anchors), sampled_state(anchors)
    if fault == "modified":
        paths[0].write_bytes(paths[0].read_bytes() + b"changed")
    else:
        paths[0].unlink()
    assert_rejected_without_mutation(target, state)


def test_actor_failure_after_sampling_keeps_counters_consistent_and_resumable(anchors):
    actor = DeterministicActor()
    target = regularizer(anchors, actor=actor)
    actor.policy.fail = True
    global_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="actor failure"):
        target()
    state = target.state_dict()
    assert state["draw_count"] == 1 and state["endpoint_draw_count"] == len(actor.policy.last_frames)
    assert torch.equal(global_before, torch.get_rng_state())
    restored = regularizer(anchors)
    restored.load_state_dict(json.loads(json.dumps(state)))
    assert restored.state_dict() == state
    actor.policy.fail = False
    torch.testing.assert_close(target(), restored(), rtol=0, atol=0)
    assert target.state_dict() == restored.state_dict()


def test_exported_state_and_identity_are_owned_json_values(anchors):
    target = regularizer(anchors)
    before = target.state_dict()
    exported = target.state_dict()
    exported["identities"][0]["path"] = "/unrelated"
    exported["pool_sizes"][0] = 999
    identities = target.identities
    identities[0]["sha256"] = "0" * 64
    assert target.state_dict() == before


@pytest.mark.parametrize("fault", ["dtype", "history", "std", "teacher", "gradient", "extra_field"])
def test_anchor_schema_is_validated_before_private_stream_creation(anchors, fault):
    _, paths, _ = anchors
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    if fault == "dtype":
        payload["frames"] = payload["frames"].double()
    elif fault == "history":
        payload["frames"] = payload["frames"][:, -3:]
    elif fault == "std":
        payload["std"][0, 0] = 0
    elif fault == "teacher":
        payload["teacher_checkpoint_sha256"] = "unqualified_identity"
    elif fault == "gradient":
        payload["mean"].requires_grad_(True)
    else:
        payload["eligible"] = True
    torch.save(payload, paths[0])
    global_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        regularizer(anchors)
    assert torch.equal(global_before, torch.get_rng_state())

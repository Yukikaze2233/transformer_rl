"""CPU-only frequency portability, with opt-in read-only recovered-state audits."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from transformer_rl.checkpoint import (
    _validate_time_frequencies,
    load_checkpoint,
    save_checkpoint,
)
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic, TimeAwareActor
from transformer_rl.ppo import PPOTrainer


_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
_FREQUENCIES = "actor.time_frequencies"
_STOP_HASHES = {
    789: "2510b96c79f7c9e935edcbe1dc86461a882b2c544dc24ae4e792be2e39b17f23",
    732: "2c6a28106dd88913d3a83e0fc2cb44d3d24f3c953fe6541fd8a1022fddc3f0fc",
}
_RECOVERED_JOBS = [
    (variant, seed, 977)
    for variant in (
        "last_token_attention", "time_attention", "index_attention", "gated_attention",
    )
    for seed in (1011, 1022, 1033)
] + [("supervised_attention", 1011, 789), ("supervised_attention", 1022, 732)]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.device.type == expected.device.type == "cpu"
        # Byte views also distinguish signed zero; numerical equality alone does not.
        assert torch.equal(
            actual.detach().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_exact(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def _recipe(d_model=64):
    return torch.exp(
        -math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    )


@pytest.fixture(params=_DTYPES)
def payload(request, tmp_path):
    model = ActorCritic(ModelConfig(history_length=2, critic_hidden=(8,)))
    _assert_exact(model.actor.time_frequencies, _recipe())
    model.to(dtype=request.param)
    _assert_exact(model.actor.time_frequencies, _recipe().to(request.param))
    trainer = PPOTrainer(model, PPOConfig())
    # Populate ordinary Adam using synthetic gradients, including half-safe eps.
    trainer.optimizer.param_groups[0]["eps"] = 1e-4
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.03125)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "original.pt"
    save_checkpoint(path, model, trainer, 1, {"source": "synthetic"})
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_float32_neighbors_cast_then_load_and_resave_exact(payload, direction, tmp_path):
    original = _recipe()
    neighbor = torch.nextafter(original, torch.full_like(original, direction))
    neighbor[0] = 1
    dtype = payload["model_state"][_FREQUENCIES].dtype
    payload["model_state"][_FREQUENCIES] = neighbor.to(dtype)
    before = deepcopy(payload)
    path = tmp_path / "neighbor.pt"
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    model, trainer, update, metadata = load_checkpoint(path, device="cpu")
    _assert_exact(torch.get_rng_state(), rng)
    _assert_exact(dict(model.state_dict()), payload["model_state"])
    _assert_exact(trainer.optimizer.state_dict(), payload["optimizer_state"])
    _assert_exact(payload, before)
    if dtype in (torch.float32, torch.float64):
        assert not torch.equal(model.actor.time_frequencies, original.to(dtype))
    saved_again = tmp_path / "resaved.pt"
    save_checkpoint(saved_again, model, trainer, update, metadata)
    _assert_exact(torch.load(saved_again, weights_only=True, map_location="cpu"), payload)
    _assert_exact(dict(model.state_dict()), payload["model_state"])
    _assert_exact(trainer.optimizer.state_dict(), payload["optimizer_state"])
    _assert_exact(torch.get_rng_state(), rng)


@pytest.mark.parametrize("steps", [2, 3, 100])
@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_outside_recipe_set_rejected(payload, steps, direction, tmp_path):
    saved = payload["model_state"][_FREQUENCIES]
    # Double inherits float32 spacing. Half tests use observable stored spacing:
    # multiple float32 ULPs can round to the same half value and leave no evidence.
    spacing = torch.float32 if saved.dtype == torch.float64 else saved.dtype
    changed = _recipe()[3].to(spacing)
    for _ in range(steps):
        changed = torch.nextafter(changed, torch.full_like(changed, direction))
    saved[3] = changed.to(saved.dtype)
    path = tmp_path / "outside.pt"
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="configured preprocessing"):
        load_checkpoint(path)
    _assert_exact(torch.get_rng_state(), rng)


@pytest.mark.parametrize("mutation", ["nan", "inf", "-inf", "zero", "negative", "swap", "anchor"])
def test_illegal_frequencies_rejected(payload, mutation, tmp_path):
    saved = payload["model_state"][_FREQUENCIES]
    if mutation == "swap":
        saved[[3, 4]] = saved[[4, 3]]
    elif mutation == "anchor":
        saved[0] = torch.nextafter(saved[0], torch.zeros_like(saved[0]))
    else:
        saved[3] = {"zero": 0.0, "negative": -0.1}.get(mutation, float("nan"))
        if mutation in ("nan", "inf", "-inf"):
            saved[3] = float(mutation)
    path = tmp_path / "illegal.pt"
    torch.save(payload, path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="configured preprocessing|nonfinite"):
        load_checkpoint(path)
    _assert_exact(torch.get_rng_state(), rng)


def test_frame_scale_one_stored_ulp_still_rejected(payload, tmp_path):
    scale = payload["model_state"]["actor.frame_scale"]
    # The derived time scaling, not just a constant-one feature, remains exact.
    scale[-1] = torch.nextafter(scale[-1], torch.full_like(scale[-1], math.inf))
    path = tmp_path / "frame-scale.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="actor.frame_scale.*configured preprocessing"):
        load_checkpoint(path)


@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_double_values_inside_bounds_but_off_float32_lattice_rejected(tmp_path, direction):
    model = ActorCritic(ModelConfig(history_length=2, critic_hidden=(8,))).double()
    frequencies = model.actor.time_frequencies
    frequencies[3] = torch.nextafter(frequencies[3], torch.full_like(frequencies[3], direction))
    trainer = PPOTrainer(model, PPOConfig())
    with pytest.raises(ValueError, match="configured preprocessing"):
        save_checkpoint(tmp_path / "off-lattice.pt", model, trainer, 0, {})
    assert not (tmp_path / "off-lattice.pt").exists()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_rounding_boundary_is_cast_from_float32(dtype):
    # Locate a local cast boundary rather than assume exp bits across machines.
    # Exercise the validator without allocating a large, unrelated policy.
    for d_model in range(4, 4097, 4):
        reference = _recipe(d_model)
        lower = torch.nextafter(reference, torch.full_like(reference, -math.inf)).to(dtype)
        upper = torch.nextafter(reference, torch.full_like(reference, math.inf)).to(dtype)
        lower[0] = upper[0] = 1
        if (lower != upper).any():
            break
    else:
        pytest.fail("no float32 recipe reached a low-precision rounding boundary")
    config = ModelConfig(d_model=d_model)
    _validate_time_frequencies(lower, config)
    _validate_time_frequencies(upper, config)
    # One stored ULP beyond either admitted endpoint is already invalid.
    index = (lower != upper).nonzero()[0].item()
    for endpoint, direction in ((lower, -math.inf), (upper, math.inf)):
        outside = endpoint.clone()
        outside[index] = torch.nextafter(outside[index], torch.full_like(outside[index], direction))
        with pytest.raises(ValueError, match="configured preprocessing"):
            _validate_time_frequencies(outside, config)


def test_double_reconstruction_is_not_the_declared_recipe(tmp_path):
    model = ActorCritic(ModelConfig(history_length=2, critic_hidden=(8,))).double()
    model.actor.time_frequencies.copy_(torch.exp(
        -math.log(10000.0) * torch.arange(0, model.config.d_model, 2, dtype=torch.float64)
        / model.config.d_model
    ))
    with pytest.raises(ValueError, match="configured preprocessing"):
        save_checkpoint(tmp_path / "double-recipe.pt", model, PPOTrainer(model, PPOConfig()), 0, {})


def test_synthetic_export_uses_saved_frequency_initializers(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from transformer_rl.export import export_policy

    model = ActorCritic(ModelConfig(history_length=2, critic_hidden=(8,)))
    frequencies = model.actor.time_frequencies
    frequencies[3] = torch.nextafter(frequencies[3], torch.full_like(frequencies[3], math.inf))
    expected_bytes = frequencies.detach().numpy().tobytes()
    assert expected_bytes != _recipe().numpy().tobytes()
    path = tmp_path / "synthetic.pt"
    save_checkpoint(path, model, PPOTrainer(model, PPOConfig()), 0, {})
    output = tmp_path / "synthetic.onnx"
    export_policy(path, output)
    graph = onnx.load(output)
    assert any(
        onnx.numpy_helper.to_array(value).tobytes() == expected_bytes
        for value in graph.graph.initializer
    )


@pytest.mark.skipif(
    not os.environ.get("TRANSFORMER_RL_RECOVERY_ROOT"), reason="recovered audit is opt-in",
)
@pytest.mark.parametrize("variant,seed,update", _RECOVERED_JOBS)
def test_recovered_checkpoint_state_only(variant, seed, update, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def forbidden(*args, **kwargs):
        raise AssertionError("real checkpoint audit is load/state only")

    for owner, name in (
        (ActorCritic, "forward"), (TimeAwareActor, "forward"),
        (TimeAwareActor, "forward_tensors"), (PPOTrainer, "update"),
        (torch.optim.Adam, "step"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    original_load = torch.load
    load_calls = []

    def weights_only_load(*args, **kwargs):
        assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"
        load_calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", weights_only_load)
    root = Path(os.environ["TRANSFORMER_RL_RECOVERY_ROOT"])
    relative = Path(f"run/jobs/{variant}/seed_{seed}/train/checkpoints/checkpoint_{update:06d}.pt")
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if update in _STOP_HASHES:
        assert digest == _STOP_HASHES[update]
    rng = torch.get_rng_state().clone()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, trainer, actual_update, metadata = load_checkpoint(path, device="cpu")
    assert actual_update == payload["update"] == update
    _assert_exact(metadata, payload["metadata"])
    _assert_exact(dict(model.state_dict()), payload["model_state"])
    _assert_exact(trainer.optimizer.state_dict(), payload["optimizer_state"])
    _assert_exact(torch.get_rng_state(), rng)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert len(load_calls) == 2 and not torch.cuda.is_initialized()
    frequencies = payload["model_state"][_FREQUENCIES]
    reference = _recipe(model.config.d_model)
    different = (frequencies != reference).nonzero().flatten().tolist()
    print("PORTABILITY_AUDIT " + json.dumps({
        "path": str(relative), "sha256": digest, "update": update,
        "model_tensors": len(payload["model_state"]),
        "optimizer_entries": len(payload["optimizer_state"]["state"]),
        "model_state_bitwise_equal": True, "optimizer_state_bitwise_equal": True,
        "rng_unchanged": True, "file_unchanged": True,
        "frequency_different_indices": different,
        "frequency_max_abs_difference": (frequencies - reference).abs().max().item(),
    }, sort_keys=True))

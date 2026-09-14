"""Checkpoint integrity and synthetic Adam continuation without environment steps."""
from dataclasses import asdict, replace
import hashlib

import pytest
import torch

import transformer_rl.checkpoint as checkpoint
from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer


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


def _components(dtype=torch.float32):
    model = ActorCritic(ModelConfig(history_length=4, critic_hidden=(16, 8))).to(dtype=dtype)
    trainer = PPOTrainer(model, PPOConfig(learning_rate=0.0003, epochs=2))
    return model, trainer


def _fake_step(trainer):
    for index, parameter in enumerate(trainer.model.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.0001)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def _assert_tree_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.fixture
def saved_checkpoint(tmp_path):
    model, trainer = _components()
    _fake_step(trainer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 7, {"run": "synthetic", "nested": [True, None, 0.5]})
    return path


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundtrip_restores_configs_weights_metadata_and_adam_next_step(tmp_path, dtype):
    model, trainer = _components(dtype)
    _fake_step(trainer)
    trainer.optimizer.param_groups[0]["lr"] = 0.00012
    model.train()
    metadata = {"experiment": "合成权重", "values": [1, 2.5, None, False]}
    path = tmp_path / "checkpoint.pt"
    rng = torch.get_rng_state().clone()
    result = save_checkpoint(path, model, trainer, 19, metadata)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    restored, resumed, update, restored_metadata = load_checkpoint(path)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert result == {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "update": 19,
    }
    assert update == 19 and restored_metadata == metadata
    assert asdict(restored.config) == asdict(model.config)
    assert asdict(resumed.config) == asdict(trainer.config)
    assert resumed.model is restored and not restored.training
    assert next(restored.parameters()).dtype == dtype
    _assert_tree_equal(dict(restored.state_dict()), dict(model.state_dict()))
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    assert all(
        a is b for a, b in zip(resumed.optimizer.param_groups[0]["params"], restored.parameters())
    )
    # This checks continuation of a known synthetic update, not bitwise training resume.
    _fake_step(trainer)
    _fake_step(resumed)
    _assert_tree_equal(dict(restored.state_dict()), dict(model.state_dict()))
    _assert_tree_equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    restored_metadata["values"].append("changed")
    assert load_checkpoint(path)[3] == metadata


def test_empty_optimizer_state_and_zero_update_are_valid(tmp_path):
    model, trainer = _components()
    path = tmp_path / "initial.pt"
    save_checkpoint(path, model, trainer, 0, {})
    _, resumed, update, metadata = load_checkpoint(path)
    assert update == 0 and metadata == {} and resumed.optimizer.state_dict()["state"] == {}
    _fake_step(resumed)


def test_parameter_conversion_cannot_leave_optimizer_bound_to_stale_parameters(saved_checkpoint):
    previous = torch.__future__.get_overwrite_module_params_on_conversion()
    torch.__future__.set_overwrite_module_params_on_conversion(True)
    try:
        model, trainer, _, _ = load_checkpoint(saved_checkpoint)
        assert all(
            a is b for a, b in zip(model.parameters(), trainer.optimizer.param_groups[0]["params"])
        )
        before = next(model.parameters()).detach().clone()
        _fake_step(trainer)
        assert not torch.equal(before, next(model.parameters()))
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(previous)


def test_loading_always_uses_weights_only(saved_checkpoint, monkeypatch):
    original = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    load_checkpoint(saved_checkpoint)
    assert len(calls) == 1


class _UnexpectedObject:
    pass


def test_weights_only_rejects_custom_pickled_objects(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save({"object": _UnexpectedObject()}, path)
    with pytest.raises(ValueError, match="weights-only"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(extra=1), "top-level keys"),
        (lambda p: p.pop("metadata"), "top-level keys"),
        (lambda p: p.update(format="other"), "format"),
        (lambda p: p.update(schema_version=5), "schema_version"),
        (lambda p: p.update(schema_version=True), "schema_version"),
        (lambda p: p.update(update=-1), "update"),
        (lambda p: p.update(update=True), "update"),
        (lambda p: p.update(model_dtype="int64"), "model_dtype"),
        (lambda p: p.update(optimizer_type="SGD"), "optimizer_type"),
        (lambda p: p["model_config"].pop("num_heads"), "configuration keys"),
        (lambda p: p["model_config"].update(extra=1), "configuration keys"),
        (lambda p: p["model_config"].update(d_model=63), "ModelConfig"),
        (lambda p: p["model_config"].update(time_scale_s=True), "ModelConfig"),
        (lambda p: p["model_config"].update(history_length=4.0), "ModelConfig"),
        (lambda p: p["model_config"].update(critic_hidden=[]), "ModelConfig"),
        (lambda p: p["ppo_config"].update(normalize_advantage=1), "PPOConfig"),
        (lambda p: p["ppo_config"].update(learning_rate=float("inf")), "PPOConfig"),
        (lambda p: p["ppo_config"].update(gamma=2.0), "PPOConfig"),
        (lambda p: p["model_state"].pop("actor.log_std"), "model_state keys"),
        (lambda p: p["model_state"].update(extra=torch.zeros(1)), "model_state keys"),
        (lambda p: p["model_state"].update({"actor.log_std": torch.zeros(1)}), "shape"),
        (lambda p: p["model_state"].update({"actor.log_std": torch.zeros(6).double()}), "dtype"),
        (lambda p: p["model_state"]["actor.log_std"].fill_(float("nan")), "nonfinite"),
        (lambda p: p["model_state"]["actor.frame_scale"].fill_(float("inf")), "nonfinite"),
        (lambda p: p.update(metadata={1: "bad key"}), "JSON"),
        (lambda p: p.update(metadata={"value": float("nan")}), "JSON"),
    ],
)
def test_invalid_payloads_are_rejected_without_rng_side_effects(saved_checkpoint, mutation, match):
    payload = torch.load(saved_checkpoint, weights_only=True)
    mutation(payload)
    torch.save(payload, saved_checkpoint)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=match):
        load_checkpoint(saved_checkpoint)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda s: s.update(extra=1), "optimizer_state"),
        (lambda s: s["param_groups"].append(dict(s["param_groups"][0])), "one parameter group"),
        (lambda s: s["param_groups"][0]["params"].reverse(), "IDs/order"),
        (lambda s: s["param_groups"][0].update(lr=float("nan")), "Adam lr"),
        (lambda s: s["param_groups"][0].update(eps=0), "Adam eps"),
        (lambda s: s["param_groups"][0].update(betas=(0.9, 1.0)), "betas"),
        (lambda s: s["param_groups"][0].update(capturable=True), "portability"),
        (lambda s: s["param_groups"][0].update(decoupled_weight_decay=True), "decoupled"),
        (lambda s: s["param_groups"][0].update(extra=1), "group keys"),
        (lambda s: s["state"].update({10000: {}}), "unknown parameter IDs"),
        (lambda s: s["state"][0].pop("exp_avg"), "state keys"),
        (lambda s: s["state"][0].update(extra=torch.tensor(0)), "state keys"),
        (lambda s: s["state"][0].update(step=torch.tensor(float("inf"))), "step"),
        (lambda s: s["state"][0].update(step=torch.tensor(0.5)), "step"),
        (lambda s: s["state"][0].update(step=torch.tensor([1.0])), "step"),
        (lambda s: s["state"][0].update(exp_avg=torch.zeros(1)), "shape"),
        (lambda s: s["state"][0].update(exp_avg=torch.zeros(6).double()), "dtype"),
        (lambda s: s["state"][0]["exp_avg"].fill_(float("nan")), "nonfinite"),
        (lambda s: s["state"][0]["exp_avg_sq"].fill_(-1), "nonnegative"),
    ],
)
def test_invalid_optimizer_state_cannot_be_silently_loaded(saved_checkpoint, mutation, match):
    payload = torch.load(saved_checkpoint, weights_only=True)
    mutation(payload["optimizer_state"])
    torch.save(payload, saved_checkpoint)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(saved_checkpoint)


@pytest.mark.parametrize(
    "metadata", [{"x": float("inf")}, {1: "value"}, {"x": (1, 2)}, {"x": torch.zeros(1)}, []]
)
def test_save_rejects_non_json_metadata_without_creating_files(tmp_path, metadata):
    model, trainer = _components()
    with pytest.raises(ValueError, match="JSON"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, trainer, 0, metadata)
    assert list(tmp_path.iterdir()) == []


def test_circular_metadata_is_explicit(tmp_path):
    model, trainer = _components()
    metadata = {}
    metadata["cycle"] = metadata
    with pytest.raises(ValueError, match="JSON"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, trainer, 0, metadata)


def test_save_rejects_nonfinite_weights_and_foreign_optimizer(tmp_path):
    model, trainer = _components()
    path = tmp_path / "checkpoint.pt"
    with torch.no_grad():
        model.actor.log_std[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_checkpoint(path, model, trainer, 0, {})
    other, _ = _components()
    with pytest.raises(ValueError, match="own this model"):
        save_checkpoint(path, other, trainer, 0, {})
    assert not path.exists()


def test_existing_file_and_dangling_symlink_are_not_overwritten(tmp_path):
    model, trainer = _components()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint(path, model, trainer, 0, {})
    assert path.read_bytes() == b"original"
    link = tmp_path / "link.pt"
    link.symlink_to(tmp_path / "absent.pt")
    with pytest.raises(FileExistsError):
        save_checkpoint(link, model, trainer, 0, {})
    assert link.is_symlink()


def test_atomic_publication_is_complete_and_losing_a_race_preserves_winner(tmp_path, monkeypatch):
    model, trainer = _components()
    path = tmp_path / "checkpoint.pt"

    def competing_link(source, destination):
        payload = torch.load(source, weights_only=True)
        assert payload["format"] == "transformer_rl.checkpoint"
        assert not path.exists()
        path.write_bytes(b"concurrent winner")
        raise FileExistsError("concurrent output")

    monkeypatch.setattr(checkpoint.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="concurrent"):
        save_checkpoint(path, model, trainer, 0, {})
    assert path.read_bytes() == b"concurrent winner"
    assert list(tmp_path.iterdir()) == [path]


def test_staging_failure_cleans_temporary_files(tmp_path, monkeypatch):
    model, trainer = _components()

    def failed_fsync(_):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(checkpoint.os, "fsync", failed_fsync)
    with pytest.raises(OSError, match="fsync"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, trainer, 0, {})
    assert list(tmp_path.iterdir()) == []


def test_interrupt_after_successful_link_removes_only_new_publication(tmp_path, monkeypatch):
    model, trainer = _components()
    original_link = checkpoint.os.link

    def interrupted_link(source, destination):
        original_link(source, destination)
        raise KeyboardInterrupt("synthetic publication interruption")

    monkeypatch.setattr(checkpoint.os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt, match="publication interruption"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, trainer, 0, {})
    assert list(tmp_path.iterdir()) == []


def test_source_submodule_configs_cannot_disagree_with_checkpoint_config(tmp_path):
    model, trainer = _components()
    model.config = replace(model.config, time_scale_s=0.2)
    with pytest.raises(ValueError, match="configurations must match"):
        save_checkpoint(tmp_path / "checkpoint.pt", model, trainer, 0, {})
    assert list(tmp_path.iterdir()) == []


def test_corrupted_bytes_are_rejected(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="weights-only"):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_optimizer_moments_follow_model_and_resume(saved_checkpoint):
    model, trainer, _, _ = load_checkpoint(saved_checkpoint, device="cuda")
    assert all(parameter.is_cuda for parameter in model.parameters())
    for state in trainer.optimizer.state.values():
        assert state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda
        assert state["step"].device.type == "cpu"
    _fake_step(trainer)

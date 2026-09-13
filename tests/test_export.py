"""Actor export contracts and failure isolation using synthetic checkpoints."""
import hashlib
import json

import pytest
import torch

import transformer_rl.checkpoint as checkpoint
import transformer_rl.export as policy_export
from transformer_rl.checkpoint import save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.export import export_policy
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


@pytest.fixture
def checkpoint_path(tmp_path):
    model = ActorCritic(ModelConfig(history_length=5, critic_hidden=(16,)))
    trainer = PPOTrainer(model, PPOConfig())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 3, {"source": "synthetic weights"})
    return path


def test_export_has_static_history_dynamic_batch_exact_types_and_bound_sidecar(
    checkpoint_path, tmp_path,
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    output = tmp_path / "policy.onnx"
    rng = torch.get_rng_state().clone()
    report = export_policy(checkpoint_path, output)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    sidecar_path = tmp_path / "policy.onnx.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert report["path"] == str(output) and report["sidecar_path"] == str(sidecar_path)
    assert sidecar["format"] == "transformer_rl.policy"
    assert sidecar["onnx_sha256"] == report["sha256"]
    assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert sidecar["checkpoint"]["sha256"] == checkpoint_digest
    assert report["sidecar_sha256"] == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    assert sidecar["checkpoint"]["update"] == 3
    assert sidecar["model_config"]["history_length"] == 5
    assert sidecar["model_config"]["action_dim"] == 6
    assert sidecar["opset"] == 17
    assert [entry["name"] for entry in sidecar["inputs"]] == [
        "frames", "times", "valid", "command", "now",
    ]
    assert [entry["dtype"] for entry in sidecar["inputs"]] == [
        "float32", "float64", "bool", "float32", "float64",
    ]
    layout = sidecar["frame_feature_layout"]
    assert [(entry["start"], entry["stop"]) for entry in layout] == [
        (0, 16), (16, 19), (19, 25), (25, 27), (27, 29), (29, 30),
    ]
    assert layout[2]["name"] == "previous_issued_action"
    assert "not raw sample or applied" in layout[2]["units"]
    assert "float64(now - times)" in sidecar["time_encoding"]["precision"]
    assert sidecar["time_encoding"]["units"] == "seconds"
    assert "no tanh, clipping" in sidecar["output"]["semantics"]
    assert sidecar["validation"]["provider"] == "CPUExecutionProvider"
    assert {case["name"] for case in report["validation"]["cases"]} == {
        "full_history", "single_batch", "left_padding", "partial_reset",
        "padding_sentinels", "large_uptime", "changed_current_command", "irregular_timing",
    }
    assert {case["batch"] for case in report["validation"]["cases"]} == {1, 2, 3}
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    assert not any(
        "critic" in item.name or "optimizer" in item.name or "log_std" in item.name
        for item in graph.graph.initializer
    )
    assert not any(node.op_type in {"Tanh", "Clip"} for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    assert [item.shape for item in session.get_inputs()] == [
        ["batch", 5, 30], ["batch", 5], ["batch", 5], ["batch", 3], ["batch"],
    ]
    assert session.get_outputs()[0].shape == ["batch", 6]


def test_exported_raw_mean_can_exceed_normalized_action_bounds(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    config = ModelConfig(history_length=1, critic_hidden=(8,))
    model = ActorCritic(config)
    with torch.no_grad():
        model.actor.mean_head.weight.zero_()
        model.actor.mean_head.bias.fill_(3)
    trainer = PPOTrainer(model, PPOConfig())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 0, {})
    output = tmp_path / "policy.onnx"
    export_policy(path, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    feed = {
        "frames": torch.full((4, 1, 30), float("nan")).numpy(),
        "times": torch.full((4, 1), float("nan"), dtype=torch.float64).numpy(),
        "valid": torch.zeros(4, 1, dtype=torch.bool).numpy(),
        "command": torch.ones(4, 3).numpy(),
        "now": torch.zeros(4, dtype=torch.float64).numpy(),
    }
    actual = torch.from_numpy(session.run(["mean"], feed)[0])
    torch.testing.assert_close(actual, torch.full((4, 6), 3.0), rtol=0, atol=0)


@pytest.mark.parametrize("existing", ["policy.onnx", "policy.onnx.json"])
def test_existing_output_or_sidecar_is_never_overwritten(checkpoint_path, tmp_path, existing):
    path = tmp_path / existing
    path.write_bytes(b"existing result")
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        export_policy(checkpoint_path, tmp_path / "policy.onnx")
    assert {item.name: item.read_bytes() for item in tmp_path.iterdir()} == before


def test_verification_failure_publishes_nothing(checkpoint_path, tmp_path, monkeypatch):
    def failed_verification(*_):
        raise RuntimeError("synthetic runtime mismatch")

    monkeypatch.setattr(policy_export, "_verify_runtime", failed_verification)
    with pytest.raises(RuntimeError, match="runtime mismatch"):
        export_policy(checkpoint_path, tmp_path / "policy.onnx")
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_actual_runtime_mismatch_is_rejected(checkpoint_path, tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    original = ort.InferenceSession.run

    def wrong_result(self, *args, **kwargs):
        values = original(self, *args, **kwargs)
        return [values[0] + 1]

    monkeypatch.setattr(ort.InferenceSession, "run", wrong_result)
    with pytest.raises(RuntimeError, match="mean mismatch"):
        export_policy(checkpoint_path, tmp_path / "policy.onnx")
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_sidecar_publication_race_rolls_back_only_our_onnx(checkpoint_path, tmp_path, monkeypatch):
    output = tmp_path / "policy.onnx"
    sidecar = tmp_path / "policy.onnx.json"
    original_link = checkpoint.os.link

    def competing_link(source, destination):
        if destination == sidecar:
            assert output.exists()
            sidecar.write_bytes(b"concurrent sidecar")
            raise FileExistsError("synthetic publication race")
        original_link(source, destination)

    monkeypatch.setattr(checkpoint.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="publication race"):
        export_policy(checkpoint_path, output)
    assert not output.exists()
    assert sidecar.read_bytes() == b"concurrent sidecar"
    assert set(tmp_path.iterdir()) == {checkpoint_path, sidecar}


def test_export_does_not_silently_downcast_checkpoint(tmp_path):
    model = ActorCritic(ModelConfig(history_length=2, critic_hidden=(8,))).double()
    trainer = PPOTrainer(model, PPOConfig())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 0, {})
    with pytest.raises(ValueError, match="float32 checkpoint"):
        export_policy(path, tmp_path / "policy.onnx")
    assert list(tmp_path.iterdir()) == [path]


def test_bad_output_suffix_is_explicit(checkpoint_path, tmp_path):
    with pytest.raises(ValueError, match=".onnx suffix"):
        export_policy(checkpoint_path, tmp_path / "policy.bin")


def test_export_binds_the_bytes_it_loaded_even_if_source_is_replaced(
    checkpoint_path, tmp_path, monkeypatch,
):
    original_bytes = checkpoint_path.read_bytes()
    original_verify = policy_export._verify_runtime

    def replace_checkpoint(*args):
        checkpoint_path.write_bytes(b"source changed after it was read")
        return original_verify(*args)

    monkeypatch.setattr(policy_export, "_verify_runtime", replace_checkpoint)
    report = export_policy(checkpoint_path, tmp_path / "policy.onnx")
    assert report["checkpoint_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert report["checkpoint_sha256"] != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

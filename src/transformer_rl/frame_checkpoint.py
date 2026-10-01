"""Portable packed-policy learning state, distinct from the time-aware format."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

from .checkpoint import (_cpu_snapshot, _json_metadata, _publish_new_files,
                         _validate_optimizer_state, _validate_tensor)
from .frame_config import FrameTrainConfig
from .frame_training import FrameActorCritic
from .ppo import PPOTrainer


_FORMAT = "transformer_rl.packed_checkpoint"


def _validate_weights(state, model):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("checkpoint model keys differ from the configured architecture")
    buffers = dict(model.named_buffers())
    for name, reference in expected.items():
        _validate_tensor(name, state[name], reference.shape, reference.dtype)
        if name in buffers and not torch.equal(state[name].cpu(), reference.cpu()):
            raise ValueError(f"fixed architecture buffer differs: {name}")


def capture_rng():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]]}


def restore_rng(state):
    _validate_rng(state)
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("exact RNG resume requires the checkpoint CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))


def _validate_rng(state):
    if not isinstance(state, dict) or set(state) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("checkpoint RNG fields differ")
    try:
        torch.Generator().set_state(state["torch"])
        if not isinstance(state["cuda"], list) or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.ndim != 1 or value.numel() == 0 for value in state["cuda"]):
            raise ValueError("invalid CUDA RNG state")
        random.Random().setstate(state["python"])
        name, values, position, gaussian, cached = state["numpy"]
        if (name != "MT19937" or not isinstance(values, list) or len(values) != 624
                or any(type(v) is not int or not 0 <= v < 2**32 for v in values)
                or type(position) is not int or not 0 <= position <= 624
                or type(gaussian) is not int or gaussian not in (0, 1)
                or type(cached) not in (float, int) or not np.isfinite(cached)):
            raise ValueError("invalid NumPy RNG state")
        np.random.RandomState().set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("invalid checkpoint RNG state") from error


def save_frame_checkpoint(path, model, trainer, config, update, metadata):
    if model.config != config.model or trainer.config != config.ppo:
        raise ValueError("checkpoint configuration differs from the live learner")
    if type(update) is not int or update < 0:
        raise ValueError("update must be nonnegative")
    state = _cpu_snapshot(dict(model.state_dict()))
    optimizer = _cpu_snapshot(trainer.optimizer.state_dict())
    _validate_weights(state, model)
    _validate_optimizer_state(optimizer, model)
    payload = {"format": _FORMAT, "schema_version": 1, "config": config.to_dict(),
               "model": state, "optimizer": optimizer, "update": update,
               "rng": _cpu_snapshot(capture_rng()), "metadata": _json_metadata(metadata)}
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    path = Path(path)
    receipt = {"format": _FORMAT, "schema_version": 1, "sha256": hashlib.sha256(data).hexdigest(),
               "update": update, "config": config.to_dict(), "metadata": payload["metadata"]}
    _publish_new_files({path: data, Path(str(path) + ".json"): json.dumps(receipt, indent=2, allow_nan=False).encode()})
    return receipt


def load_frame_checkpoint(path, device="cpu"):
    data = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    expected = {"format", "schema_version", "config", "model", "optimizer", "update", "rng", "metadata"}
    if (not isinstance(payload, dict) or set(payload) != expected or payload["format"] != _FORMAT
            or payload["schema_version"] != 1 or type(payload["update"]) is not int or payload["update"] < 0):
        raise ValueError("unsupported packed checkpoint")
    config = FrameTrainConfig.from_dict(payload["config"])
    with torch.random.fork_rng(devices=[]):
        model = FrameActorCritic(config.model)
    _validate_weights(payload["model"], model)
    _validate_optimizer_state(payload["optimizer"], model)
    metadata = _json_metadata(payload["metadata"])
    _validate_rng(payload["rng"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    trainer = PPOTrainer(model, config.ppo)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    return model, trainer, config, payload["update"], metadata, payload["rng"]

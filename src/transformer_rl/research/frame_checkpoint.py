"""Portable packed-policy learning state, distinct from the time-aware format."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..common.random_state import capture_rng, restore_rng, _validate_rng
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

"""独立保存与恢复 Python、NumPy、CPU/CUDA 随机流。"""
from __future__ import annotations

import random

import numpy as np
import torch


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

"""Validated, weights-only checkpoints with no-clobber atomic publication."""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import tempfile

import torch

from .config import ModelConfig, PPOConfig
from .model import ActorCritic
from .ppo import PPOTrainer


_FORMAT = "transformer_rl.checkpoint"
_SCHEMA_VERSION = 1
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_PAYLOAD_KEYS = {
    "format", "schema_version", "model_config", "ppo_config", "model_dtype",
    "model_state", "optimizer_type", "optimizer_state", "update", "metadata",
}


def _require_new_paths(paths: list[Path]) -> None:
    if len({path.absolute() for path in paths}) != len(paths):
        raise ValueError("output paths must be distinct")
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output parent directory does not exist: {path.parent}")


def _publish_new_files(contents: dict[Path, bytes]) -> None:
    """Stage/fsync then link exclusively; the last file is the completion marker.

    A hard link publishes a complete file without the overwrite race of rename.
    Multiple names cannot be crash-atomic; handled errors remove only links made
    by this call. Existing files, including dangling symlinks, are never replaced.
    """
    _require_new_paths(list(contents))
    temporary: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for destination, data in contents.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as stream:
                source = Path(stream.name)
                temporary.append((source, destination))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        for source, destination in temporary:
            # Track the attempt first: an interruption after a successful link
            # must still roll it back. Inode checks preserve competing outputs.
            published.append((source, destination))
            os.link(source, destination)
        for parent in {path.parent for path in contents}:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for source, destination in reversed(published):
            try:
                # Do not remove a replacement installed by another writer.
                staged = source.stat()
                current = destination.stat(follow_symlinks=False)
                if (staged.st_dev, staged.st_ino) == (current.st_dev, current.st_ino):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json_value(item)
        return
    raise ValueError("metadata requires JSON values, string object keys and finite numbers")


def _json_metadata(metadata: object) -> dict:
    if type(metadata) is not dict:
        raise ValueError("metadata must be a JSON object")
    try:
        # Detect circular references before recursively checking strict JSON types.
        encoded = json.dumps(metadata, allow_nan=False, ensure_ascii=False)
        _validate_json_value(metadata)
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"invalid JSON metadata: {error}") from error


def _config_payload(config: ModelConfig | PPOConfig) -> dict:
    data = asdict(config)
    if isinstance(config, ModelConfig):
        data["critic_hidden"] = list(config.critic_hidden)
    return data


def _parse_config(
    data: object, cls: type[ModelConfig] | type[PPOConfig]
) -> ModelConfig | PPOConfig:
    if type(data) is not dict or set(data) != {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__} requires exactly its declared configuration keys")
    defaults = cls()
    for field in fields(cls):
        value = data[field.name]
        default = getattr(defaults, field.name)
        if type(default) is tuple:
            valid = type(value) is list and all(type(item) is int for item in value)
        elif type(default) is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is type(default)
        if not valid:
            raise ValueError(f"invalid {cls.__name__}.{field.name} type or value")
    arguments = dict(data)
    if cls is ModelConfig:
        arguments["critic_hidden"] = tuple(arguments["critic_hidden"])
    try:
        return cls(**arguments)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {cls.__name__}: {error}") from error


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("checkpoint tensors must be dense strided tensors")
        return value.detach().cpu().clone()
    if type(value) is dict:
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if type(value) is list:
        return [_cpu_snapshot(item) for item in value]
    if type(value) is tuple:
        return tuple(_cpu_snapshot(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"unsupported checkpoint value type: {type(value).__name__}")


def _validate_tensor(name: str, value: object, shape: torch.Size, dtype: torch.dtype) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} requires dense shape {tuple(shape)} and dtype {dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _validate_model_state(state: object, model: ActorCritic) -> None:
    expected = model.state_dict()
    if type(state) is not dict or set(state) != set(expected):
        raise ValueError("model_state keys do not match the configured ActorCritic")
    for name, reference in expected.items():
        _validate_tensor(f"model_state.{name}", state[name], reference.shape, reference.dtype)
    # These buffers define preprocessing promised by ModelConfig and the ONNX
    # sidecar. They are not learned parameters and must not drift independently.
    for name in ("actor.time_frequencies", "actor.frame_scale"):
        if not torch.equal(state[name].cpu(), expected[name].cpu()):
            raise ValueError(f"model_state.{name} differs from configured preprocessing")


def _validate_optimizer_state(state: object, model: ActorCritic) -> None:
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("optimizer_state requires exactly state and param_groups")
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        raise ValueError("PPO Adam requires exactly one parameter group")
    group = groups[0]
    required = {
        "params", "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
        "foreach", "capturable", "differentiable", "fused",
    }
    optional = {"decoupled_weight_decay"}
    if not required <= set(group) or set(group) - required - optional:
        raise ValueError("unexpected or missing Adam parameter group keys")
    parameters = list(model.parameters())
    ids = group["params"]
    if (
        type(ids) is not list or any(type(index) is not int for index in ids)
        or ids != list(range(len(parameters)))
    ):
        raise ValueError("Adam parameter IDs/order must match all model parameters")
    for name in ("lr", "eps", "weight_decay"):
        value = group[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Adam {name} must be a finite nonnegative number")
    if group["eps"] == 0:
        raise ValueError("Adam eps must be positive")
    betas = group["betas"]
    if (
        type(betas) not in (list, tuple) or len(betas) != 2
        or any(type(v) not in (float, int) or not 0 <= v < 1 for v in betas)
    ):
        raise ValueError("Adam betas must contain two finite numbers in [0, 1)")
    for name in ("amsgrad", "maximize", "capturable", "differentiable"):
        if type(group[name]) is not bool:
            raise ValueError(f"Adam {name} must be bool")
    for name in ("foreach", "fused"):
        if group[name] is not None and type(group[name]) is not bool:
            raise ValueError(f"Adam {name} must be bool or None")
    if group["capturable"] or group["differentiable"] or group["fused"]:
        raise ValueError("checkpoint portability requires noncapturable, nonfused ordinary Adam")
    if group.get("decoupled_weight_decay", False) is not False:
        raise ValueError("PPO checkpoints require Adam, not decoupled weight decay")
    entries = state["state"]
    if type(entries) is not dict or any(type(key) is not int or key not in ids for key in entries):
        raise ValueError("Adam state has unknown parameter IDs")
    expected_keys = {"step", "exp_avg", "exp_avg_sq"}
    if group["amsgrad"]:
        expected_keys.add("max_exp_avg_sq")
    for index, entry in entries.items():
        if type(entry) is not dict or set(entry) != expected_keys:
            raise ValueError(f"Adam state[{index}] has invalid state keys")
        step = entry["step"]
        if (
            not isinstance(step, torch.Tensor) or step.layout != torch.strided
            or step.shape != torch.Size([])
            or step.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(step) or step < 0 or step != step.trunc()
        ):
            raise ValueError("Adam step must be a finite nonnegative integer scalar tensor")
        parameter = parameters[index]
        for name in expected_keys - {"step"}:
            _validate_tensor(
                f"Adam state[{index}].{name}", entry[name], parameter.shape, parameter.dtype
            )
            if name != "exp_avg" and (entry[name] < 0).any():
                raise ValueError(f"Adam {name} must be nonnegative")


def _new_model(config: ModelConfig, dtype: torch.dtype) -> ActorCritic:
    # Initialization consumes CPU RNG even though every parameter is overwritten.
    # Keep loading (including failed validation) invisible to caller RNG streams.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        return ActorCritic(config).to(dtype=dtype)


def _validated_components(
    payload: object, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, PPOTrainer, int, dict]:
    if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("checkpoint requires exactly the declared top-level keys")
    if payload["format"] != _FORMAT:
        raise ValueError("unrecognized checkpoint format")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    if type(payload["update"]) is not int or payload["update"] < 0:
        raise ValueError("checkpoint update must be a nonnegative integer")
    metadata = _json_metadata(payload["metadata"])
    config = _parse_config(payload["model_config"], ModelConfig)
    ppo_config = _parse_config(payload["ppo_config"], PPOConfig)
    dtype_name = payload["model_dtype"]
    if type(dtype_name) is not str or dtype_name not in _DTYPES:
        raise ValueError("unsupported checkpoint model_dtype")
    if payload["optimizer_type"] != "Adam":
        raise ValueError("unsupported checkpoint optimizer_type")
    model = _new_model(config, _DTYPES[dtype_name])
    _validate_model_state(payload["model_state"], model)
    _validate_optimizer_state(payload["optimizer_state"], model)
    model.load_state_dict(payload["model_state"], strict=True)
    # Device conversion may replace Parameter objects. Bind the optimizer only
    # after conversion, and let Adam place moments/counters by its own rules.
    model.to(device=device)
    trainer = PPOTrainer(model, ppo_config)
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    _validate_optimizer_state(trainer.optimizer.state_dict(), model)
    return model, trainer, payload["update"], metadata


def save_checkpoint(
    path: str | Path, model: ActorCritic, trainer: PPOTrainer,
    update: int, metadata: dict,
) -> dict:
    """Save a new checkpoint, returning path, SHA-256 and cumulative update.

    The parent directory must exist. Existing destinations are never replaced.
    Configs, dense tensor keys/shapes/dtypes/finiteness, JSON metadata and ordinary
    single-group Adam state are validated before publication. No environment,
    collector, controller or global RNG state is captured.
    """
    path = Path(path)
    _require_new_paths([path])
    if not isinstance(model, ActorCritic) or not isinstance(trainer, PPOTrainer):
        raise TypeError("save_checkpoint requires ActorCritic and PPOTrainer")
    if trainer.model is not model or type(trainer.optimizer) is not torch.optim.Adam:
        raise ValueError("trainer must own this model and an ordinary Adam optimizer")
    if model.actor.config != model.config or model.critic.config != model.config:
        raise ValueError("actor and critic configurations must match model.config")
    attached = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    parameters = list(model.parameters())
    if len(attached) != len(parameters) or any(a is not b for a, b in zip(attached, parameters)):
        raise ValueError("optimizer parameter identity/order must match the model")
    dtype = next(model.parameters()).dtype
    dtype_name = str(dtype).removeprefix("torch.")
    payload = {
        "format": _FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "model_config": _config_payload(model.config),
        "ppo_config": _config_payload(trainer.config),
        "model_dtype": dtype_name,
        "model_state": _cpu_snapshot(dict(model.state_dict())),
        "optimizer_type": "Adam",
        "optimizer_state": _cpu_snapshot(trainer.optimizer.state_dict()),
        "update": update,
        "metadata": _json_metadata(metadata),
    }
    _validated_components(payload)
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    _publish_new_files({path: data})
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "update": update}


def _load_checkpoint_bytes(
    data: bytes, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, PPOTrainer, int, dict]:
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, TypeError, ValueError) as error:
        raise ValueError(f"invalid weights-only checkpoint: {error}") from error
    return _validated_components(payload, device=device)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, PPOTrainer, int, dict]:
    """Return (model, trainer, update, metadata), using weights_only=True.

    Restores model dtype and Adam state, without changing global CPU RNG state.
    Resume collection from a new episode; exact external-state or bitwise
    training continuation is not part of this checkpoint contract.
    """
    return _load_checkpoint_bytes(Path(path).read_bytes(), device=device)

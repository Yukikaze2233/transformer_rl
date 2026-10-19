"""两条训练路径共用的制品发布与完整状态校验，不依赖具体网络。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import torch


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


def _validate_optimizer_state(state: object, model, *, estimator=False) -> None:
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
    parameters = model.estimator_parameters() if estimator else model.policy_parameters()
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

"""Actor-only ONNX publication after multi-condition CPU runtime verification."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import warnings

import torch
from torch import nn

from . import __version__
from .checkpoint import (
    _config_payload,
    _load_checkpoint_bytes,
    _publish_new_files,
    _require_new_paths,
)
from .config import ModelConfig
from .history import pack_frame
from .model import GaussianActor
from .types import HistoryBatch


_INPUT_NAMES = ("frames", "times", "valid", "command", "now")
_OPSET = 17
_RTOL = 1e-5
_ATOL = 1e-6


class _PolicyMean(nn.Module):
    def __init__(self, actor: GaussianActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(
        self, frames: torch.Tensor, times: torch.Tensor, valid: torch.Tensor,
        command: torch.Tensor, now: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor.forward_tensors(frames, times, valid, command, now)


def _tensor_args(history: HistoryBatch) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(history, name) for name in _INPUT_NAMES)


def _synthetic_history(config: ModelConfig, counts: list[int]) -> HistoryBatch:
    """Local deterministic tensors; no environment or global random sampling."""
    generator = torch.Generator(device="cpu").manual_seed(2026)
    batch, length = len(counts), config.history_length
    tokens = batch * length
    proprio = torch.randn(
        tokens, config.proprio_dim, generator=generator, dtype=torch.float32
    ) * 0.2
    commands = torch.randn(
        tokens, config.command_dim, generator=generator, dtype=torch.float32
    ) * 0.3
    issued = torch.randn(
        tokens, config.action_dim, generator=generator, dtype=torch.float32
    ) * 0.1
    ages = torch.rand(
        tokens, config.sensor_groups, generator=generator, dtype=torch.float32
    ) * 0.02
    known = torch.arange(tokens * config.sensor_groups).reshape(tokens, -1) % 3 != 0
    ages[~known] = float("nan")
    intervals = (1 + torch.arange(length) % 3).double() / 128
    elapsed = intervals.cumsum(0)
    now = torch.arange(batch, dtype=torch.float64) * 10 + 1
    times = now[:, None] - (elapsed[-1] - elapsed)[None]
    dt = intervals.float()[None].expand(batch, -1).clone()
    positions = torch.arange(length)[None]
    valid = positions >= length - torch.tensor(counts)[:, None]
    for row, count in enumerate(counts):
        if count:
            dt[row, length - count] = 0
    frames = pack_frame(
        config, proprio, commands, issued, ages, known, dt.reshape(tokens)
    ).reshape(batch, length, config.frame_dim)
    frames = torch.where(valid[..., None], frames, torch.zeros_like(frames))
    times = torch.where(valid, times, torch.zeros_like(times))
    current_command = commands.reshape(batch, length, config.command_dim)[:, -1].clone()
    return HistoryBatch(frames, times, valid, current_command, now)


def _verification_cases(config: ModelConfig) -> list[tuple[str, HistoryBatch]]:
    full = _synthetic_history(config, [config.history_length, config.history_length])
    padding = _synthetic_history(
        config, [1, max(1, config.history_length // 2), config.history_length]
    )
    reset = _synthetic_history(config, [0, 1, config.history_length])
    poisoned = reset.clone()
    poisoned.frames[~poisoned.valid] = float("nan")
    poisoned.times[~poisoned.valid] = float("inf")
    changed_command = replace(full, command=full.command + 0.7)
    irregular = full.clone()
    age = irregular.now[:, None] - irregular.times
    irregular.times.copy_(irregular.now[:, None] - 3 * age)
    irregular.frames[..., -1].mul_(3)
    return [
        ("full_history", full),
        ("single_batch", _synthetic_history(config, [config.history_length])),
        ("left_padding", padding),
        ("partial_reset", reset),
        ("padding_sentinels", poisoned),
        ("large_uptime", replace(full, times=full.times + 2**30, now=full.now + 2**30)),
        ("changed_current_command", changed_command),
        ("irregular_timing", irregular),
    ]


def _verify_runtime(
    data: bytes, actor: GaussianActor, cases: list[tuple[str, HistoryBatch]]
) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        data, sess_options=options, providers=["CPUExecutionProvider"]
    )
    config = actor.config
    expected_types = (
        "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(float)", "tensor(double)"
    )
    expected_shapes = (
        ["batch", config.history_length, config.frame_dim],
        ["batch", config.history_length], ["batch", config.history_length],
        ["batch", config.command_dim], ["batch"],
    )
    actual_inputs = session.get_inputs()
    if [item.name for item in actual_inputs] != list(_INPUT_NAMES):
        raise RuntimeError("exported ONNX inputs do not match the five-input contract")
    for item, dtype, shape in zip(actual_inputs, expected_types, expected_shapes):
        if item.type != dtype or item.shape != shape:
            raise RuntimeError(f"exported input {item.name} has unexpected type/shape")
    outputs = session.get_outputs()
    if (
        len(outputs) != 1 or outputs[0].name != "mean" or outputs[0].type != "tensor(float)"
        or outputs[0].shape != ["batch", config.action_dim]
    ):
        raise RuntimeError("exported mean has unexpected type/shape")
    results = []
    for name, history in cases:
        # Validation is intentionally outside the traced tensor-only wrapper.
        with torch.no_grad():
            expected = actor(history)
        feed = {key: tensor.numpy() for key, tensor in zip(_INPUT_NAMES, _tensor_args(history))}
        actual = torch.from_numpy(session.run(["mean"], feed)[0])
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise RuntimeError(f"nonfinite policy mean during ONNX verification: {name}")
        if actual.shape != expected.shape or not torch.allclose(
            actual, expected, rtol=_RTOL, atol=_ATOL
        ):
            raise RuntimeError(f"ONNX/PyTorch mean mismatch during verification: {name}")
        results.append({
            "name": name, "batch": history.frames.shape[0],
            "max_abs_error": (actual - expected).abs().max().item(),
        })
    return {
        "provider": "CPUExecutionProvider", "onnxruntime_version": ort.__version__,
        "rtol": _RTOL, "atol": _ATOL, "cases": results,
        "scope": "synthetic implementation equivalence, not control performance",
    }


def _input_contract(config: ModelConfig) -> list[dict]:
    length = config.history_length
    return [
        {"name": "frames", "dtype": "float32", "shape": ["batch", length, config.frame_dim],
         "units": "see frame_feature_layout; all time scalars are seconds"},
        {"name": "times", "dtype": "float64", "shape": ["batch", length],
         "units": "seconds; policy observation availability timestamps"},
        {"name": "valid", "dtype": "bool", "shape": ["batch", length],
         "units": "true for meaningful history tokens; false for padding"},
        {"name": "command", "dtype": "float32", "shape": ["batch", config.command_dim],
         "units": "current command in the same caller-defined scaling as historical commands"},
        {"name": "now", "dtype": "float64", "shape": ["batch"],
         "units": "seconds; current policy query time, same clock as times"},
    ]


def _frame_layout(config: ModelConfig) -> list[dict]:
    layout = []
    start = 0
    for name, width, units in (
        ("proprio", config.proprio_dim, "caller-defined scaled proprioception"),
        ("command", config.command_dim, "command at this historical event"),
        ("previous_issued_action", config.action_dim,
         "previous ISSUED command, not raw sample or applied target"),
        ("sensor_age_s", config.sensor_groups,
         "seconds; unknown age encodes zero with known=false"),
        ("sensor_age_known", config.sensor_groups, "float32 flags: 0=false, 1=true"),
        ("policy_dt_s", 1, "seconds since preceding policy event; zero permitted on reset"),
    ):
        layout.append({"name": name, "start": start, "stop": start + width, "units": units})
        start += width
    return layout


def export_policy(checkpoint_path: str | Path, output_path: str | Path) -> dict:
    """Export actor raw mean to ONNX plus ``<output_path>.json`` after CPU checks.

    Float32 checkpoints are required; no silent precision conversion is made.
    History length is static and batch is dynamic. Neither existing output may
    be overwritten. ONNX/ORT imports are optional until this function is called.
    """
    checkpoint_path, output_path = Path(checkpoint_path), Path(output_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("policy output_path must have an .onnx suffix")
    sidecar_path = Path(f"{output_path}.json")
    _require_new_paths([output_path, sidecar_path])
    checkpoint_bytes = checkpoint_path.read_bytes()
    model, trainer, update, _ = _load_checkpoint_bytes(checkpoint_bytes, device="cpu")
    if next(model.actor.parameters()).dtype != torch.float32:
        raise ValueError("ONNX policy export requires a float32 checkpoint")
    actor = model.actor.eval()
    config = actor.config
    cases = _verification_cases(config)
    stream = io.BytesIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            _PolicyMean(actor).eval(), _tensor_args(cases[0][1]), stream,
            input_names=list(_INPUT_NAMES), output_names=["mean"],
            dynamic_axes={name: {0: "batch"} for name in (*_INPUT_NAMES, "mean")},
            opset_version=_OPSET, dynamo=False, export_params=True,
        )
    if any(issubclass(item.category, torch.jit.TracerWarning) for item in caught):
        raise RuntimeError("ONNX export traced Python tensor-dependent validation")
    data = stream.getvalue()
    import onnx

    graph = onnx.load_model_from_string(data)
    onnx.checker.check_model(graph)
    if any(any(name in item.name for name in ("critic", "log_std", "auxiliary_head"))
           for item in graph.graph.initializer):
        raise RuntimeError("ONNX graph unexpectedly contains non-mean policy parameters")
    validation = _verify_runtime(data, actor, cases)
    digest = hashlib.sha256(data).hexdigest()
    checkpoint_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    sidecar = {
        "format": "transformer_rl.policy", "schema_version": 1,
        "package": "transformer_rl", "package_version": __version__,
        "torch_version": str(torch.__version__), "onnx_version": onnx.__version__,
        "opset": _OPSET, "onnx_file": output_path.name, "onnx_sha256": digest,
        "checkpoint": {
            "path": str(checkpoint_path), "sha256": checkpoint_digest, "update": update,
            "source_schema_version": model.checkpoint_source_schema_version,
        },
        "actor": actor.describe(),
        "training_auxiliary": {
            "indices": list(config.auxiliary_indices),
            "coefficient": trainer.config.auxiliary_coef,
            "exported": False,
        },
        "model_config": _config_payload(config), "model_dtype": "float32",
        "inputs": _input_contract(config),
        "output": {
            "name": "mean", "dtype": "float32", "shape": ["batch", config.action_dim],
            "semantics": (
                "deterministic raw Gaussian mean; no tanh, clipping or actuator conversion; "
                "mean_init_scale is initialization-only, already incorporated in weights, "
                "not a runtime gain"
            ),
        },
        "frame_feature_layout": _frame_layout(config),
        "history_contract": {
            "order": "oldest to newest, left padding, current complete frame last when present",
            "valid_times": "strictly increasing and <= now; times and now share a clock",
            "padding": "invalid frames/times are ignored, including NaN/Inf sentinels",
            "reset": "caller clears only reset environments; empty histories are valid",
            "command": "current query command is separate; never rewrite historical commands",
            "state": "no mutable KV cache or external controller state in graph",
            "finite": "all meaningful inputs must be finite; batch >= 1",
        },
        "time_encoding": {
            "units": "seconds", "time_scale_s": config.time_scale_s,
            "precision": (
                "float64(now - times), then cast to float32 before fixed Fourier encoding"
            ),
            "frequencies": "exp(-log(10000) * (2*k) / d_model), k=0..d_model/2-1",
            "features": "concat(sin(age / time_scale_s * frequencies), cos(...))",
            "frame_times": "sensor_age_s and policy_dt_s divided by time_scale_s inside actor",
        },
        "action_contract": {
            "history_action": "previous issued command, not measured/applied actuator state",
            "output_domain": (
                "raw policy action domain; external command limiting/scaling is caller-owned"
            ),
            "scope": "no transport FIFO, PID, actuator response or real-hardware timing encoded",
        },
        "validation": validation,
        "exporter_warnings": [str(item.message) for item in caught],
    }
    if config.actor_type != "transformer" or config.time_encoding != "elapsed":
        sidecar["time_encoding"] = actor.describe()["time_encoding"]
    sidecar_text = json.dumps(sidecar, indent=2, ensure_ascii=False, allow_nan=False)
    sidecar_bytes = (sidecar_text + "\n").encode("utf-8")
    _publish_new_files({output_path: data, sidecar_path: sidecar_bytes})
    return {
        "path": str(output_path), "sidecar_path": str(sidecar_path), "sha256": digest,
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "checkpoint_sha256": checkpoint_digest, "validation": validation,
    }

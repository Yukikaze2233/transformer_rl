"""Owned per-environment policy-event histories and explicit frame packing."""
from __future__ import annotations

import torch

from .config import ModelConfig
from .types import HistoryBatch, VectorObservation


def pack_frame(
    config: ModelConfig,
    proprio: torch.Tensor,
    command: torch.Tensor,
    previous_issued_action: torch.Tensor,
    sensor_age_s: torch.Tensor,
    sensor_age_known: torch.Tensor,
    policy_dt_s: torch.Tensor,
) -> torch.Tensor:
    """Pack [N, F] without clipping or changing observation/action scaling.

    Floating inputs share proprio's dtype/device. Shapes are [N, P], [N, C],
    [N, A], [N, G], bool [N, G], and [N] or [N, 1], respectively. Ages and
    intervals stay in seconds and must be nonnegative. Unknown ages may contain
    any sentinel (including NaN); they encode zero *with a false known flag*.
    Other meaningful values must be finite. Shape/device/value errors raise
    ValueError; non-tensor or incompatible dtype inputs raise TypeError.
    """
    if not isinstance(proprio, torch.Tensor):
        raise TypeError("proprio must be a tensor")
    if proprio.ndim != 2 or proprio.shape[0] < 1 or proprio.shape[1] != config.proprio_dim:
        raise ValueError(f"proprio must have shape [N, {config.proprio_dim}], N >= 1")
    if not proprio.is_floating_point():
        raise TypeError("proprio must have a floating dtype")
    count = proprio.shape[0]
    inputs = {
        "command": (command, (count, config.command_dim)),
        "previous_issued_action": (previous_issued_action, (count, config.action_dim)),
        "sensor_age_s": (sensor_age_s, (count, config.sensor_groups)),
        "sensor_age_known": (sensor_age_known, (count, config.sensor_groups)),
        "policy_dt_s": (policy_dt_s, (count,)),
    }
    for name, (tensor, shape) in inputs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != shape and not (
            name == "policy_dt_s" and tuple(tensor.shape) == (count, 1)
        ):
            suffix = " or [N, 1]" if name == "policy_dt_s" else ""
            raise ValueError(f"{name} must have shape {shape}{suffix}")
        expected_dtype = torch.bool if name == "sensor_age_known" else proprio.dtype
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}")
        if tensor.device != proprio.device:
            raise ValueError(f"{name} must be on proprio device {proprio.device}")
    ages = torch.where(sensor_age_known, sensor_age_s, torch.zeros_like(sensor_age_s))
    for name, tensor in (
        ("proprio", proprio),
        ("command", command),
        ("previous_issued_action", previous_issued_action),
        ("known sensor_age_s", ages),
        ("policy_dt_s", policy_dt_s),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
    if (ages < 0).any() or (policy_dt_s < 0).any():
        raise ValueError("known sensor ages and policy intervals must be nonnegative")
    return torch.cat(
        (
            proprio,
            command,
            previous_issued_action,
            ages,
            sensor_age_known.to(proprio.dtype),
            policy_dt_s.reshape(count, 1),
        ),
        dim=-1,
    )


class HistoryBuffer:
    """Fixed-length, left-padded history; reset is explicit and per environment.

    Feature dtype is inferred on first append and must remain consistent.
    Timestamps accept float32/float64, are stored as float64, and must strictly
    increase per environment unless the complete observation is identical at
    the same tick. Prefer float64 at the producer to preserve uptime precision.
    Shape/device/value errors raise ValueError; incompatible dtypes raise
    TypeError. Invalid appends are rejected before changing any environment.
    """

    def __init__(
        self, config: ModelConfig, num_envs: int, device: str | torch.device
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.config = config
        self.num_envs = num_envs
        self._frames = torch.zeros(
            num_envs, config.history_length, config.frame_dim, device=device
        )
        self.device = self._frames.device
        self._times = torch.zeros(
            num_envs, config.history_length, dtype=torch.float64, device=device
        )
        self._valid = torch.zeros(
            num_envs, config.history_length, dtype=torch.bool, device=device
        )
        self._command = torch.zeros(num_envs, config.command_dim, device=device)
        # Only retained for same-tick identity checks, never put in actor history.
        self._last_critic = torch.zeros(num_envs, config.critic_dim, device=device)
        self._initialized = False

    def reset(self, mask: torch.Tensor) -> None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("reset mask must be a bool tensor")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape [{self.num_envs}]")
        if mask.device != self.device:
            raise ValueError(f"reset mask must be on buffer device {self.device}")
        self._frames[mask] = 0
        self._times[mask] = 0
        self._valid[mask] = False
        self._command[mask] = 0
        self._last_critic[mask] = 0

    def _validate_observation(self, observation: VectorObservation) -> None:
        if not isinstance(observation, VectorObservation):
            raise TypeError("observation must be a VectorObservation")
        specs = (
            ("frame", observation.frame, (self.num_envs, self.config.frame_dim)),
            ("timestamp", observation.timestamp, (self.num_envs,)),
            ("command", observation.command, (self.num_envs, self.config.command_dim)),
            ("critic", observation.critic, (self.num_envs, self.config.critic_dim)),
        )
        for name, tensor, shape in specs:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"observation.{name} must be a tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"observation.{name} must have shape {shape}")
            if tensor.device != self.device:
                raise ValueError(f"observation.{name} must be on buffer device {self.device}")
            if name == "timestamp":
                if tensor.dtype not in (torch.float32, torch.float64):
                    raise TypeError("timestamp must be float32 or float64 (float64 recommended)")
            elif not tensor.is_floating_point() or tensor.dtype != observation.frame.dtype:
                raise TypeError("frame, command and critic must share a floating dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"observation.{name} must be finite")
        if self._initialized and observation.frame.dtype != self._frames.dtype:
            raise TypeError("observation feature dtype must match previous appends")

    def append(self, observation: VectorObservation) -> HistoryBatch:
        self._validate_observation(observation)
        times = observation.timestamp.detach().to(torch.float64)
        populated = self._valid[:, -1]
        if (populated & (times < self._times[:, -1])).any():
            raise ValueError("timestamps must not decrease without an explicit reset")
        repeated = populated & (times == self._times[:, -1])
        changed = (
            (observation.frame != self._frames[:, -1]).any(-1)
            | (observation.command != self._command).any(-1)
            | (observation.critic != self._last_critic).any(-1)
        )
        if (repeated & changed).any():
            raise ValueError("different observation data at the same timestamp")

        if not self._initialized:
            dtype = observation.frame.dtype
            self._frames = self._frames.to(dtype=dtype)
            self._command = self._command.to(dtype=dtype)
            self._last_critic = self._last_critic.to(dtype=dtype)
            self._initialized = True
        advance = ~repeated
        with torch.no_grad():
            # Boolean indexing copies the RHS, so overlapping shifts are safe.
            self._frames[advance, :-1] = self._frames[advance, 1:]
            self._times[advance, :-1] = self._times[advance, 1:]
            self._valid[advance, :-1] = self._valid[advance, 1:]
            self._frames[advance, -1] = observation.frame[advance]
            self._times[advance, -1] = times[advance]
            self._valid[advance, -1] = True
            self._command[advance] = observation.command[advance]
            self._last_critic[advance] = observation.critic[advance]
        return self.snapshot()

    def snapshot(self) -> HistoryBatch:
        """Return detached, independently owned storage, including command/now."""
        return HistoryBatch(
            frames=self._frames.clone(),
            times=self._times.clone(),
            valid=self._valid.clone(),
            command=self._command.clone(),
            now=self._times[:, -1].clone(),
        )

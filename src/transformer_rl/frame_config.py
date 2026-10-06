"""Configuration and deployment contracts for externally packed observations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from .config import PPOConfig
from .frame_policy import FramePolicyConfig


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def parse_dataclass(cls, values):
    if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    values = dict(values)
    defaults = cls()
    for f in fields(cls):
        if f.name in values and isinstance(getattr(defaults, f.name), tuple):
            if not isinstance(values[f.name], (list, tuple)):
                raise ValueError(f"{f.name} requires a list")
            values[f.name] = tuple(values[f.name])
    return cls(**values)


@dataclass(frozen=True)
class FrameModelConfig:
    policy: FramePolicyConfig = FramePolicyConfig()
    critic_dim: int = 81
    critic_hidden: tuple[int, ...] = (256, 128, 64)
    initial_std: float = 1.0
    command_indices: tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self):
        if not isinstance(self.policy, FramePolicyConfig):
            raise TypeError("policy must be a FramePolicyConfig")
        for name in ("critic_hidden", "command_indices"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} requires a list of integers")
            object.__setattr__(self, name, tuple(value))
        if type(self.critic_dim) is not int or self.critic_dim < 1:
            raise ValueError("critic_dim must be positive")
        if (not self.critic_hidden or any(type(n) is not int or n < 1 for n in self.critic_hidden)):
            raise ValueError("critic_hidden requires positive integer widths")
        if (not self.command_indices or len(set(self.command_indices)) != len(self.command_indices)
                or any(type(i) is not int or not 0 <= i < self.frame_dim for i in self.command_indices)):
            raise ValueError("command_indices must identify unique frame columns")
        if type(self.initial_std) not in (float, int) or not math.isfinite(self.initial_std) or self.initial_std <= 0:
            raise ValueError("initial_std must be finite and positive")

    @property
    def frame_dim(self):
        return self.policy.frame_dim

    @property
    def action_dim(self):
        return self.policy.action_dim

    @property
    def history_length(self):
        return self.policy.history_length

    @property
    def command_dim(self):
        return len(self.command_indices)

    @property
    def estimator_type(self):
        return "none"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("model requires an object")
        value = dict(value)
        value["policy"] = FramePolicyConfig.from_dict(value.get("policy", {}))
        return parse_dataclass(cls, value)


@dataclass(frozen=True)
class FrameTrainConfig:
    model: FrameModelConfig
    ppo: PPOConfig
    control: dict
    environment: dict

    def __post_init__(self):
        if not isinstance(self.model, FrameModelConfig) or not isinstance(self.ppo, PPOConfig):
            raise TypeError("configuration requires parsed model and PPO settings")
        if self.ppo.auxiliary_coef != 0:
            raise ValueError("packed policy studies do not implicitly add privileged supervision")
        required = {"policy_dt_s", "observation_schema", "feature_names", "action_names",
                    "action_bounds", "target_scale", "target_offset", "target_units"}
        if not isinstance(self.control, dict) or set(self.control) != required:
            raise ValueError("control requires the complete observation/action/time contract")
        if not isinstance(self.environment, dict):
            raise ValueError("environment requires an object")
        control = json.loads(json_bytes(self.control))
        if (type(control["policy_dt_s"]) not in (int, float)
                or not math.isfinite(control["policy_dt_s"]) or control["policy_dt_s"] <= 0):
            raise ValueError("policy_dt_s must be finite and positive")
        if not isinstance(control["observation_schema"], str) or not control["observation_schema"]:
            raise ValueError("observation_schema requires a nonempty identifier")
        for name, count in (("feature_names", self.model.frame_dim), ("action_names", self.model.action_dim),
                            ("target_units", self.model.action_dim)):
            values = control[name]
            if (not isinstance(values, list) or len(values) != count
                    or any(not isinstance(v, str) or not v for v in values)):
                raise ValueError(f"{name} must name each column")
            if name != "target_units" and len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate names")
        for name in ("action_bounds", "target_scale", "target_offset"):
            values = control[name]
            if (not isinstance(values, list) or len(values) != self.model.action_dim
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError(f"{name} requires one finite value per action")
        if any(v <= 0 for v in control["action_bounds"]) or any(v == 0 for v in control["target_scale"]):
            raise ValueError("action bounds must be positive and target scales nonzero")
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "environment", json.loads(json_bytes(self.environment)))

    def to_dict(self):
        model = asdict(self.model)
        model["policy"] = self.model.policy.to_dict()
        return json.loads(json_bytes({"model": model, "ppo": asdict(self.ppo),
                                      "control": self.control, "environment": self.environment}))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"model", "ppo", "control", "environment"}:
            raise ValueError("configuration requires model, ppo, control and environment")
        return cls(FrameModelConfig.from_dict(value["model"]), parse_dataclass(PPOConfig, value["ppo"]),
                   value["control"], value["environment"])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_policy(self, values):
        value = self.to_dict()
        value["model"]["policy"].update(values)
        return type(self).from_dict(value)

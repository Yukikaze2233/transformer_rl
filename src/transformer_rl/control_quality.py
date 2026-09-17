"""Episode-level physical failures, world displacement and evaluation traces."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch

from .checkpoint import _publish_new_files


class ControlQuality:
    def __init__(self, num_envs, settle_steps, min_steady_samples):
        self.num_envs = num_envs
        self.settle_steps, self.minimum = settle_steps, min_steady_samples
        self.active = [self._new() for _ in range(num_envs)]
        self.episodes = []
        self.available = None
        self.finished = False

    @staticmethod
    def _new():
        return dict(samples=0, retained=0, violation_s=0.0, failed=False,
                    origin=None, endpoint=0.0, maximum=0.0, previous_time=None)

    def update(self, state, times, terminated, truncated, step_dt):
        if self.finished:
            raise RuntimeError("quality statistics already finalized")
        present = bool(state)
        if self.available is not None and present != self.available:
            raise ValueError("evaluation_state availability changed")
        self.available = present
        if not present:
            return
        if type(step_dt) not in (float, int) or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("evaluation_step_dt must be finite and positive")
        if (times.shape != (self.num_envs,) or times.dtype != torch.float64
            or terminated.shape != (self.num_envs,) or truncated.shape != (self.num_envs,)
            or terminated.dtype != torch.bool or truncated.dtype != torch.bool):
            raise ValueError("invalid quality timestamps or termination masks")
        for name, width in (("height", None), ("tilt", None), ("world_position", 3)):
            value = state[name]
            shape = (self.num_envs,) if width is None else (self.num_envs, width)
            if value.shape != shape or not value.is_floating_point() or value.device != times.device:
                raise ValueError(f"invalid evaluation_state.{name}")
        packed = torch.cat((state["height"][:, None], state["tilt"][:, None],
                            state["world_position"][:, :2], times[:, None],
                            terminated[:, None], truncated[:, None]), dim=-1).double().cpu()
        if not torch.isfinite(packed).all():
            raise FloatingPointError("nonfinite control quality state")
        for index, row in enumerate(packed.tolist()):
            height, tilt, x, y, timestamp, term, trunc = row
            episode = self.active[index]
            dt = step_dt if episode["previous_time"] is None else timestamp - episode["previous_time"]
            if dt <= 0:
                raise ValueError("quality timestamps must increase within each episode")
            episode["previous_time"] = timestamp
            episode["samples"] += 1
            violation = height < 0.20 or tilt > 0.60
            episode["violation_s"] = episode["violation_s"] + dt if violation else 0.0
            episode["failed"] |= bool(term) or episode["violation_s"] >= 0.20 - 1e-9
            if episode["samples"] > self.settle_steps:
                episode["retained"] += 1
                if episode["origin"] is None:
                    episode["origin"] = (x, y)
                distance = math.hypot(x - episode["origin"][0], y - episode["origin"][1])
                episode["endpoint"] = distance
                episode["maximum"] = max(episode["maximum"], distance)
            if term or trunc:
                self._flush(index, bool(term), bool(trunc), complete=True)

    def _flush(self, index, terminated=False, truncated=False, complete=False):
        episode = self.active[index]
        if episode["samples"]:
            usable = episode["retained"] >= self.minimum
            self.episodes.append({
                "env_id": index, "samples": episode["samples"], "retained": episode["retained"],
                "complete": complete, "terminated": terminated, "truncated": truncated,
                "physical_failure": episode["failed"],
                "healthy_timeout": complete and truncated and not terminated and not episode["failed"],
                "world_xy_endpoint_drift": episode["endpoint"] if usable else None,
                "world_xy_max_excursion": episode["maximum"] if usable else None,
            })
        self.active[index] = self._new()

    def report(self):
        if not self.finished:
            for index in range(self.num_envs):
                self._flush(index)
            self.finished = True
        complete = [e for e in self.episodes if e["complete"]]
        total = sum(e["samples"] for e in self.episodes)
        censored = sum(e["samples"] for e in self.episodes if not e["complete"])
        excursions = [e["world_xy_max_excursion"] for e in self.episodes if e["world_xy_max_excursion"] is not None]
        return {
            "available": bool(self.available), "episodes": self.episodes,
            "completed_episodes": len(complete),
            "healthy_timeout_fraction": sum(e["healthy_timeout"] for e in complete) / len(complete) if complete else None,
            "physical_failure_episodes": sum(e["physical_failure"] for e in self.episodes),
            "censored_sample_fraction": censored / total if total else None,
            "world_xy_max_excursion_p95": float(np.percentile(excursions, 95)) if excursions else None,
            "protocol": {"height_below_m": 0.20, "tilt_above_rad": 0.60, "continuous_violation_s": 0.20,
                         "includes_warmup_failures": True, "censored_counts_as_success": False,
                         "drift_origin": "first retained world_xy sample per episode"},
        }


class EvaluationTrace:
    """One packed CPU transfer per step; preserve field shape, dtype and time role."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.episode_ids = np.zeros(num_envs, dtype=np.int64)
        self.layout = None
        self.rows = []

    def add(self, fields, done):
        values = dict(fields)
        values["env_id"] = torch.arange(self.num_envs, device=done.device)
        values["episode_id"] = torch.as_tensor(self.episode_ids.copy(), device=done.device)
        names = sorted(values)
        layout = [(name, tuple(values[name].shape[1:]), str(values[name].dtype).removeprefix("torch.")) for name in names]
        if self.layout is not None and layout != self.layout:
            raise ValueError("trace fields, shapes and dtypes must stay fixed")
        self.layout = layout
        packed = torch.cat([values[name].detach().reshape(self.num_envs, -1).double() for name in names], dim=-1).cpu().numpy().copy()
        if not np.isfinite(packed).all():
            raise FloatingPointError("nonfinite evaluation trace")
        self.rows.append(packed)
        self.episode_ids += done.detach().cpu().numpy().astype(np.int64)

    def save(self, path):
        data = np.stack(self.rows)
        arrays, offset = {}, 0
        for name, shape, dtype in self.layout:
            width = math.prod(shape)
            arrays[name] = data[..., offset:offset + width].reshape(len(data), self.num_envs, *shape).astype(dtype)
            offset += width
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        _publish_new_files({Path(path): payload})
        return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                "steps": len(self.rows), "fields": [name for name, _, _ in self.layout],
                "time_alignment": "observation_time/state estimates/targets precede action; signal_time/physical states are PRE-reset transition endpoints"}

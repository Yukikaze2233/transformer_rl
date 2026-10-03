"""Bounded disk-backed PRE-reset traces for declared evaluation replicas."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


class ControlTrace:
    """Store predetermined rows, never select trajectories by their outcome."""

    def __init__(self, path, *, steps, num_envs, replicas, groups, metadata):
        if type(replicas) is not int or replicas < 1:
            raise ValueError("trace_replicas must be a positive integer")
        self.path = Path(path).resolve()
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if groups is None:
            self.rows = np.arange(min(replicas, num_envs), dtype=np.int64)
        else:
            selected = []
            for name in sorted(set(groups)):
                selected.extend([i for i, group in enumerate(groups) if group == name][:replicas])
            self.rows = np.array(sorted(selected), dtype=np.int64)
        self.steps, self.count = steps, 0
        self.episode = np.zeros(num_envs, dtype=np.int64)
        self.maps = {}
        self.metadata = {**metadata, "row_indices": self.rows.tolist(),
                         "group_labels": [groups[i] for i in self.rows] if groups else None,
                         "selection": "first declared replicas per case; independent of outcomes",
                         "sample_scope": "policy-rate PRE-reset physical samples; not a high-frequency current measurement"}
        self.temporary = tempfile.TemporaryDirectory(prefix=".control-trace-", dir=self.path.parent)

    def add(self, packet, done):
        if self.count >= self.steps:
            raise ValueError("control trace exceeded its declared step budget")
        values = {name: value.detach().cpu().numpy()[self.rows] for name, value in packet.items()}
        values.update(done=done.detach().cpu().numpy()[self.rows], episode_id=self.episode[self.rows].copy())
        if not self.maps:
            for index, (name, value) in enumerate(values.items()):
                self.maps[name] = np.lib.format.open_memmap(
                    Path(self.temporary.name) / f"field_{index}.npy", mode="w+", dtype=value.dtype,
                    shape=(self.steps, *value.shape))
        if values.keys() != self.maps.keys():
            raise ValueError("control trace packet fields changed")
        for name, value in values.items():
            if value.shape != self.maps[name].shape[1:]:
                raise ValueError("control trace packet shape changed")
            self.maps[name][self.count] = value
        self.episode += done.detach().cpu().numpy().astype(np.int64)
        self.count += 1

    def publish(self):
        if self.count != self.steps:
            raise ValueError("cannot publish an incomplete control trace")
        self.metadata["steps"] = self.count
        staging = Path(self.temporary.name) / "trace.npz"
        for value in self.maps.values():
            value.flush()
        with staging.open("xb") as stream:
            np.savez_compressed(stream, row_indices=self.rows,
                                metadata_json=np.array(json.dumps(self.metadata, allow_nan=False)),
                                **self.maps)
        sha = hashlib.sha256()
        with staging.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        os.link(staging, self.path)
        return {"path": str(self.path), "sha256": sha.hexdigest(), "steps": self.count,
                "rows": self.rows.tolist(), "fields": list(self.maps), **self.metadata}

    def close(self):
        self.maps.clear()
        self.temporary.cleanup()

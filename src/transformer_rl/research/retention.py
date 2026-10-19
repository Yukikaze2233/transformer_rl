"""Frozen behavior anchors: a separate distillation loss, never stale PPO data."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from .checkpoint import _publish_new_files
from .frame_config import digest


def save_anchors(path, config, frames, mean, std, checkpoint_sha256):
    payload = {"format": "transformer_rl.behavior_anchors", "schema_version": 1,
               "control_sha256": digest(config.control), "policy_config": config.model.policy.to_dict(),
               "teacher_checkpoint_sha256": checkpoint_sha256,
               "frames": frames.detach().cpu(), "mean": mean.detach().cpu(), "std": std.detach().cpu()}
    import io
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    _publish_new_files({Path(path): data})
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(), "samples": len(frames)}


class AnchorRegularizer:
    """Mean KL(teacher || student), with uniform file/skill then endpoint sampling."""

    def __init__(self, actor, config, paths, coefficient, batch_size=256):
        if (type(coefficient) not in (int, float) or not math.isfinite(coefficient) or coefficient <= 0
                or type(batch_size) is not int or batch_size < 1 or not paths):
            raise ValueError("anchors require positive coefficient/batch size and nonempty paths")
        self.actor, self.coefficient, self.batch_size = actor, coefficient, batch_size
        self.device = next(actor.parameters()).device
        self.pools, self.identities = [], []
        for path in paths:
            path = Path(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("format") != "transformer_rl.behavior_anchors" or payload.get("schema_version") != 1
                    or payload.get("control_sha256") != digest(config.control)
                    or payload.get("policy_config") != config.model.policy.to_dict()):
                raise ValueError("anchor observation, timing or architecture contract mismatch")
            frames, mean, std = (payload[k] for k in ("frames", "mean", "std"))
            if (frames.ndim != 3 or frames.shape[1:] != (config.model.history_length, config.model.frame_dim)
                    or len(frames) < 1 or mean.shape != (len(frames), config.model.action_dim) or std.shape != mean.shape
                    or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (frames, mean, std))
                    or not (std > 0).all()):
                raise ValueError("invalid behavior anchor tensors")
            self.pools.append((frames, mean, std))
            self.identities.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def __call__(self):
        pool = self.pools[torch.randint(len(self.pools), ()).item()]
        indices = torch.randint(len(pool[0]), (min(self.batch_size, len(pool[0])),))
        frames, mean, std = (t[indices].to(self.device) for t in pool)
        predicted = self.actor.policy(frames)
        current_std = self.actor.log_std.exp().expand_as(predicted)
        kl = (current_std.log() - std.log()
              + 0.5 * ((std / current_std).square() + ((mean - predicted) / current_std).square() - 1.)).sum(-1)
        return self.coefficient * kl.mean()

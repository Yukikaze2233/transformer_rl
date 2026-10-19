"""Collection moments in actor coordinates, before transport and motor processing."""
from __future__ import annotations

import torch


@torch.no_grad()
def action_statistics(raw: torch.Tensor, issued: torch.Tensor, mean: torch.Tensor,
                      bounds: torch.Tensor) -> dict[str, int | float | None]:
    """Describe returned rollout endpoints, including terminal and partial rows.

    Outside counts use strict inequalities; a raw value exactly at a bound was
    not clipped. Issued at-bound counts include those exact-bound values. These
    are action-domain measurements, not motor torque saturation or centered
    episode jitter. Each channel has the same endpoint sample denominator.
    """
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] == 0
            or not raw.is_floating_point()):
        raise ValueError("raw actions require floating [samples, actions]")
    for name, value in (("issued", issued), ("mean", mean)):
        if (not isinstance(value, torch.Tensor) or value.shape != raw.shape
                or value.device != raw.device or not value.is_floating_point()):
            raise ValueError(f"{name} requires matching floating action rows and device")
    if (not isinstance(bounds, torch.Tensor) or bounds.shape != (raw.shape[1],)
            or bounds.device != raw.device or not bounds.is_floating_point()
            or not torch.isfinite(bounds).all() or not (bounds > 0).all()):
        raise ValueError("bounds require finite positive values per action on the same device")
    if any(not torch.isfinite(value).all() for value in (raw, issued, mean)):
        raise ValueError("action statistics require finite input moments and actions")
    if not torch.equal(issued, raw.clamp(-bounds, bounds)):
        raise ValueError("issued actions must equal the declared raw action clamp")
    n = raw.shape[0]
    result: dict[str, int | float | None] = {"action_sample_count": n}
    for channel in range(raw.shape[1]):
        suffix = str(channel)
        raw_column, issued_column = raw[:, channel].double(), issued[:, channel].double()
        bound = bounds[channel].double()
        outside = int((raw_column.abs() > bound).sum().item())
        mean_outside = int((mean[:, channel].double().abs() > bound).sum().item())
        at_bound = int((issued_column.abs() == bound).sum().item())
        result.update({f"raw_action_outside_count_{suffix}": outside,
                       f"raw_action_clip_fraction_{suffix}": outside / n if n else None,
                       f"action_mean_outside_count_{suffix}": mean_outside,
                       f"action_mean_outside_fraction_{suffix}": mean_outside / n if n else None,
                       f"issued_action_at_bound_count_{suffix}": at_bound,
                       f"issued_action_at_bound_fraction_{suffix}": at_bound / n if n else None})
        for name, column in (("raw_action", raw_column), ("issued_action", issued_column)):
            result[f"{name}_mean_{suffix}"] = column.mean().item() if n else None
            result[f"{name}_rms_{suffix}"] = column.square().mean().sqrt().item() if n else None
            result[f"{name}_std_{suffix}"] = column.std(unbiased=False).item() if n else None
    return result

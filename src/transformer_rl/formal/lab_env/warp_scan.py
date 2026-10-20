"""Warp-side validity check for RayCaster height samples."""
from __future__ import annotations

import torch


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def count_misses(hits: wp.array3d(dtype=wp.float32),
                         root_z: wp.array1d(dtype=wp.float32),
                         nominal_height: float,
                         misses: wp.array1d(dtype=wp.int32)):
            index = wp.tid()
            environment = index / 77
            point = index % 77
            height = root_z[environment] - hits[environment, point, 2] - nominal_height
            if not wp.isfinite(height):
                wp.atomic_add(misses, 0, 1)

        _KERNEL = count_misses
    return _KERNEL


def miss_count(hits: torch.Tensor, root_z: torch.Tensor, nominal_height: float) -> int:
    if (not isinstance(hits, torch.Tensor) or hits.ndim != 3 or hits.shape[1:] != (77, 3)
            or hits.dtype != torch.float32 or hits.device.type != 'cuda'
            or not hits.is_contiguous() or not isinstance(root_z, torch.Tensor)
            or root_z.shape != (hits.shape[0],) or root_z.dtype != torch.float32
            or root_z.device != hits.device):
        raise ValueError('Warp miss check requires contiguous CUDA [N,77,3] hits and [N] root_z')
    import warp as wp

    wp.init()
    wp_hits = wp.from_torch(hits)
    wp_root = wp.from_torch(root_z)
    misses = wp.zeros(1, dtype=wp.int32, device=wp_hits.device)
    wp.launch(_kernel(), dim=hits.shape[0] * 77,
              inputs=[wp_hits, wp_root, float(nominal_height), misses])
    wp.synchronize()
    return int(wp.to_torch(misses).item())


_STATS_KERNEL = None


def _stats_kernel():
    global _STATS_KERNEL
    if _STATS_KERNEL is None:
        import warp as wp

        @wp.kernel
        def count_miss_stats(hits: wp.array3d(dtype=wp.float32),
                             root_z: wp.array1d(dtype=wp.float32),
                             nominal_height: float,
                             per_env: wp.array1d(dtype=wp.int32),
                             per_point: wp.array1d(dtype=wp.int32)):
            index = wp.tid()
            environment = index / 77
            point = index % 77
            height = root_z[environment] - hits[environment, point, 2] - nominal_height
            if not wp.isfinite(height):
                wp.atomic_add(per_env, environment, 1)
                wp.atomic_add(per_point, point, 1)

        _STATS_KERNEL = count_miss_stats
    return _STATS_KERNEL


def miss_histograms(hits: torch.Tensor, root_z: torch.Tensor, nominal_height: float):
    if (not isinstance(hits, torch.Tensor) or hits.ndim != 3 or hits.shape[1:] != (77, 3)
            or hits.dtype != torch.float32 or hits.device.type != 'cuda'
            or not hits.is_contiguous() or root_z.shape != (hits.shape[0],)
            or root_z.dtype != torch.float32 or root_z.device != hits.device):
        raise ValueError('Warp miss stats require contiguous CUDA [N,77,3] hits and [N] root_z')
    import warp as wp

    wp.init()
    wp_hits, wp_root = wp.from_torch(hits), wp.from_torch(root_z)
    per_env = wp.zeros(hits.shape[0], dtype=wp.int32, device=wp_hits.device)
    per_point = wp.zeros(77, dtype=wp.int32, device=wp_hits.device)
    wp.launch(_stats_kernel(), dim=hits.shape[0] * 77,
              inputs=[wp_hits, wp_root, float(nominal_height), per_env, per_point])
    wp.synchronize()
    return wp.to_torch(per_env), wp.to_torch(per_point)

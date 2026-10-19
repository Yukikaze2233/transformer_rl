"""全局种子高度场：只生成请求的网格块，不预存完整二维噪声矩阵。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import torch

from .geometry import HeightPatch, TerrainQuery


@dataclass(frozen=True)
class SeededRoughGrid:
    """网格点由(seed,i,j)唯一确定，分块顺序不改变几何或消耗额外RNG。

    所有块共用全局float32坐标和三角对角线。完整域不缩短；本类只负责
    源几何与特权查询，不负责原生碰撞创建、替换或物理验收。
    """

    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    amplitude_m: float
    geometry_seed: int
    entry_m: float
    transition_m: float
    active_end_m: float
    friction: float
    direction: int = 1
    shape: tuple[int, int] = field(init=False)
    extreme_rows: tuple[int, ...] = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self):
        x, y = torch.tensor(self.x_m, dtype=torch.float32), torch.tensor(self.y_m, dtype=torch.float32)
        amplitude_m, geometry_seed = self.amplitude_m, self.geometry_seed
        entry_m, transition_m, active_end_m = self.entry_m, self.transition_m, self.active_end_m
        friction, direction = self.friction, self.direction
        if (x.ndim != 1 or y.ndim != 1 or min(x.numel(), y.numel()) < 2
                or not torch.isfinite(x).all() or not torch.isfinite(y).all()
                or not (x[1:] > x[:-1]).all() or not (y[1:] > y[:-1]).all()
                or type(geometry_seed) is not int or not 0 <= geometry_seed < 2**63
                or type(direction) is not int or direction not in (-1, 1)):
            raise ValueError('rough grid requires finite increasing axes,seed and travel direction')
        values = (amplitude_m, entry_m, transition_m, active_end_m, friction)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or amplitude_m < 0 or transition_m <= 0 or friction < 0
                or not float(x[0]) < entry_m < entry_m + transition_m < active_end_m
                or active_end_m + transition_m > float(x[-1])):
            raise ValueError('rough grid requires its complete entry,active region and runout')
        object.__setattr__(self, 'x_m', tuple(x.tolist()))
        object.__setattr__(self, 'y_m', tuple(y.tolist()))
        object.__setattr__(self, 'shape', (len(self.x_m), len(self.y_m)))
        core = [i for i, v in enumerate(self.x_m) if entry_m + transition_m <= v <= active_end_m]
        object.__setattr__(self, 'extreme_rows', tuple(core[:2]) if len(core) >= 2 else ())
        descriptor = {'algorithm': 'global_counter_splitmix64_v1', 'x_m': self.x_m, 'y_m': self.y_m,
                      'amplitude_m': amplitude_m, 'geometry_seed': geometry_seed, 'entry_m': entry_m,
                      'transition_m': transition_m, 'active_end_m': active_end_m,
                      'friction': friction, 'direction': direction}
        object.__setattr__(self, 'sha256', hashlib.sha256(json.dumps(descriptor, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest())

    @staticmethod
    def _logical_shift(value, bits):
        return (value >> bits) & ((1 << (64 - bits)) - 1)

    @classmethod
    def _mix(cls, counter):
        counter = counter - 7046029254386353131
        value = (counter ^ cls._logical_shift(counter, 30)) * -4658895280553007687
        value = (value ^ cls._logical_shift(value, 27)) * -7723592293110705685
        return value ^ cls._logical_shift(value, 31)

    def _heights(self, i, j, seeds):
        # int64按2^64回绕；先混合seed，避免相邻seed只是平移一格。
        seeds = torch.as_tensor(seeds, dtype=torch.long, device=i.device)
        value = self._mix(i * self.shape[1] + j + self._mix(seeds))
        # 高24位形成可精确表示的float32均匀噪声。
        noise = ((value >> 40) & ((1 << 24) - 1)).float() * (2. / (1 << 24)) - 1.
        if self.extreme_rows:
            middle = j == self.shape[1] // 2
            noise = torch.where((i == self.extreme_rows[0]) & middle, 1., noise)
            noise = torch.where((i == self.extreme_rows[1]) & middle, -1., noise)
        x = torch.tensor(self.x_m, dtype=torch.float64, device=i.device)[i]
        entry = ((x - self.entry_m) / self.transition_m).clamp(0, 1)
        exit = ((self.active_end_m + self.transition_m - x) / self.transition_m).clamp(0, 1)
        return (noise.double() * self.amplitude_m * entry * exit).float()

    def patch(self, i0, i1, j0, j1):
        """闭区间顶点索引；邻块共享边界顶点，不扩展或更改原网格。"""
        if (any(type(v) is not int for v in (i0, i1, j0, j1))
                or not 0 <= i0 < i1 < self.shape[0] or not 0 <= j0 < j1 < self.shape[1]):
            raise ValueError('patch requires a nonempty cell rectangle inside the complete global grid')
        i, j = torch.arange(i0, i1 + 1)[:, None], torch.arange(j0, j1 + 1)[None, :]
        z = self._heights(i, j, self.geometry_seed)
        x = self.x_m[i0:i1 + 1]
        if self.direction < 0:
            x, z = tuple(-v for v in reversed(x)), z.flip(0)
        return HeightPatch(x, self.y_m[j0:j1 + 1], tuple(map(tuple, z.tolist())), self.friction)

    def tile(self, row, column, *, cells=32):
        if (type(cells) is not int or cells < 1 or type(row) is not int or type(column) is not int
                or row < 0 or column < 0 or row * cells >= self.shape[0] - 1
                or column * cells >= self.shape[1] - 1):
            raise ValueError('tile index must contain cells inside the full global grid')
        return self.patch(row * cells, min((row + 1) * cells, self.shape[0] - 1),
                          column * cells, min((column + 1) * cells, self.shape[1] - 1))

    def tile_keys(self, x_bounds, y_bounds, *, cells=32, halo=1):
        """给出完整矩形足迹及外围块；域外请求报错，不截成较小任务。"""
        if (type(cells) is not int or cells < 1 or type(halo) is not int or halo < 0
                or len(x_bounds) != 2 or len(y_bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*x_bounds, *y_bounds))
                or x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]):
            raise ValueError('tile selection requires finite ordered physical bounds and integer cell counts')
        x_bounds = x_bounds if self.direction > 0 else (-x_bounds[1], -x_bounds[0])
        spans = []
        for axis, bounds in ((self.x_m, x_bounds), (self.y_m, y_bounds)):
            if bounds[0] < axis[0] or bounds[1] > axis[-1]:
                raise ValueError('requested footprint lies outside the full declared terrain domain')
            positions = torch.searchsorted(torch.tensor(axis, dtype=torch.float32),
                                           torch.tensor(bounds, dtype=torch.float32), right=True) - 1
            positions.clamp_(0, len(axis) - 2)
            lo, hi = (int(v) // cells for v in positions)
            spans.append(range(max(0, lo - halo), min((len(axis) - 2) // cells, hi + halo) + 1))
        return tuple((i, j) for i in spans[0] for j in spans[1])

    @torch.no_grad()
    def at(self, points, *, seeds=None):
        """同一布局可批量查询不同seed，只取每个点所需的四个顶点。"""
        if (not isinstance(points, torch.Tensor) or points.dtype != torch.float32
                or points.ndim != 3 or points.shape[-1] != 2 or min(points.shape[:2]) < 1
                or not torch.isfinite(points).all()):
            raise ValueError('rough grid query requires finite float32 [N,Q,2] physical coordinates')
        count = len(points)
        if seeds is None:
            seeds = torch.full((count,), self.geometry_seed, dtype=torch.long, device=points.device)
        if (not isinstance(seeds, torch.Tensor) or seeds.shape != (count,) or seeds.dtype != torch.long
                or seeds.device != points.device or (seeds < 0).any()):
            raise ValueError('each rough field query needs its nonnegative int64 geometry seed')
        x = torch.tensor(self.x_m, dtype=torch.float32, device=points.device)
        if self.direction < 0:
            x = -x.flip(0)
        y = torch.tensor(self.y_m, dtype=torch.float32, device=points.device)
        px, py = points[..., 0].contiguous(), points[..., 1].contiguous()
        i = (torch.searchsorted(x, px, right=True) - 1).clamp(0, len(x) - 2)
        j = (torch.searchsorted(y, py, right=True) - 1).clamp(0, len(y) - 2)
        u, v = (px - x[i]) / (x[i + 1] - x[i]), (py - y[j]) / (y[j + 1] - y[j])
        left = i if self.direction > 0 else len(x) - 1 - i
        right = left + self.direction
        a, b = self._heights(left, j, seeds[:, None]), self._heights(right, j, seeds[:, None])
        c, d = self._heights(right, j + 1, seeds[:, None]), self._heights(left, j + 1, seeds[:, None])
        z = torch.where(v <= u, a + u * (b - a) + v * (c - b), a + u * (c - d) + v * (d - a))
        valid = (px >= x[0]) & (px <= x[-1]) & (py >= y[0]) & (py <= y[-1])
        return TerrainQuery(torch.where(valid, z, float('nan')),
                            torch.where(valid, self.friction, float('nan')), valid,
                            torch.where(valid, 0, -1).long())

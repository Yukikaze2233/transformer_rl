"""Private authored terrain geometry and same-triangle privileged queries.

These coordinates describe collision geometry, not actor observations or
measured robot support. Native cooking/contact equivalence still needs SDK
acceptance. Points outside the authored domain are explicitly invalid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
import struct

import torch


class TerrainKind(IntEnum):
    FLAT = 0
    SLOPE = 1
    ROUGH = 2
    STEP_UP = 3
    STEP_DOWN = 4
    STAIRS = 5
    MATERIAL = 6
    TWO_PLATFORMS = 7


@dataclass(frozen=True)
class TerrainMesh:
    """Explicit static SI geometry in its environment's local coordinates."""

    vertices_m: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    friction: float

    def __post_init__(self):
        if (len(self.vertices_m) < 3 or not self.triangles
                or type(self.friction) not in (int, float) or not math.isfinite(self.friction)
                or self.friction < 0):
            raise ValueError('terrain requires finite explicit vertices, faces and nonnegative friction')
        for vertex in self.vertices_m:
            if (len(vertex) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vertex)):
                raise ValueError('terrain vertex coordinates must be finite SI triples')
        for face in self.triangles:
            if (len(face) != 3 or len(set(face)) != 3
                    or any(type(i) is not int or not 0 <= i < len(self.vertices_m) for i in face)):
                raise ValueError('terrain faces must name three distinct valid vertices')
            a, b, c = (self.vertices_m[i] for i in face)
            u, v = tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(a, c))
            cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            if not all(math.isfinite(x) for x in cross) or sum(x * x for x in cross) < 1e-20:
                raise ValueError('terrain triangles must have finite nonzero area')
        object.__setattr__(self, 'vertices_m', tuple(tuple(float(v) for v in p) for p in self.vertices_m))
        object.__setattr__(self, 'triangles', tuple(tuple(face) for face in self.triangles))


@dataclass(frozen=True)
class HeightPatch:
    """A rectangular grid with the exact diagonal used by its triangle mesh.

    Coordinates are rounded to float32 BEFORE both authoring and querying.
    Bilinear interpolation would describe a different surface inside a cell.
    Vertical risers are separate meshes and do not define a height function.
    """

    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    z_m: tuple[tuple[float, ...], ...]
    friction: float
    mesh: TerrainMesh = field(init=False, repr=False)

    def __post_init__(self):
        def coordinate(value):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('terrain grid coordinates must be finite')
            try:
                result = struct.unpack('<f', struct.pack('<f', value))[0]
            except (OverflowError, struct.error) as error:
                raise ValueError('terrain coordinates exceed float32 geometry') from error
            if not math.isfinite(result):
                raise ValueError('terrain coordinates exceed float32 geometry')
            return result

        x, y = tuple(map(coordinate, self.x_m)), tuple(map(coordinate, self.y_m))
        z = tuple(tuple(map(coordinate, row)) for row in self.z_m)
        if (min(len(x), len(y)) < 2 or len(z) != len(x) or any(len(row) != len(y) for row in z)
                or any(b <= a for axis in (x, y) for a, b in zip(axis, axis[1:]))):
            raise ValueError('terrain grid requires strictly increasing float32 axes and a complete height matrix')
        vertices = tuple((a, b, z[i][j]) for i, a in enumerate(x) for j, b in enumerate(y))
        triangles = []
        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                a, b = i * len(y) + j, (i + 1) * len(y) + j
                triangles.extend(((a, b, b + 1), (a, b + 1, a + 1)))
        mesh = TerrainMesh(vertices, tuple(triangles), self.friction)
        object.__setattr__(self, 'x_m', x)
        object.__setattr__(self, 'y_m', y)
        object.__setattr__(self, 'z_m', z)
        object.__setattr__(self, 'mesh', mesh)


@dataclass(frozen=True)
class TerrainSurface:
    patches: tuple[HeightPatch, ...]
    vertical_faces: tuple[TerrainMesh, ...] = ()
    meshes: tuple[TerrainMesh, ...] = field(init=False, repr=False)

    def __post_init__(self):
        patches, walls = tuple(self.patches), tuple(self.vertical_faces)
        if not patches or any(not isinstance(patch, HeightPatch) for patch in patches):
            raise ValueError('terrain requires at least one explicit height patch')
        if any(not isinstance(wall, TerrainMesh) for wall in walls):
            raise ValueError('terrain vertical faces require explicit mesh geometry')
        object.__setattr__(self, 'patches', patches)
        object.__setattr__(self, 'vertical_faces', walls)
        object.__setattr__(self, 'meshes', tuple(p.mesh for p in patches) + walls)


@dataclass(frozen=True)
class TerrainQuery:
    height_m: torch.Tensor
    friction: torch.Tensor
    valid: torch.Tensor
    patch_index: torch.Tensor


@dataclass(frozen=True)
class _PatchGroup:
    owners: torch.Tensor
    patch_ids: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    friction: torch.Tensor


class TerrainQueryBatch:
    """Prepack variable grid shapes once; vectorize all per-physical-tick work.

    At overlapping edges the highest top surface wins. Equal-height material
    seams select the first authored patch deterministically, never an average
    material absent from the simulator. No nearest-plane fallback fills holes.
    """

    def __init__(self, surfaces, *, device='cpu'):
        self.surfaces, self.device = tuple(surfaces), torch.device(device)
        self.num_envs = len(self.surfaces)
        if not self.surfaces or any(not isinstance(s, TerrainSurface) for s in self.surfaces):
            raise ValueError('terrain query needs one complete surface per native environment')
        grouped = {}
        patch_id = 0
        for owner, surface in enumerate(self.surfaces):
            for patch in surface.patches:
                grouped.setdefault((len(patch.x_m), len(patch.y_m)), []).append((owner, patch_id, patch))
                patch_id += 1
        self.total_patches = patch_id
        self.groups = tuple(_PatchGroup(
            torch.tensor([row[0] for row in rows], dtype=torch.long, device=self.device),
            torch.tensor([row[1] for row in rows], dtype=torch.long, device=self.device),
            torch.tensor([row[2].x_m for row in rows], dtype=torch.float32, device=self.device),
            torch.tensor([row[2].y_m for row in rows], dtype=torch.float32, device=self.device),
            torch.tensor([row[2].z_m for row in rows], dtype=torch.float32, device=self.device),
            torch.tensor([row[2].friction for row in rows], dtype=torch.float32, device=self.device)) for rows in grouped.values())

    @torch.no_grad()
    def at(self, local_xy_m):
        if (not isinstance(local_xy_m, torch.Tensor) or local_xy_m.ndim != 3
                or local_xy_m.shape[0] != self.num_envs or local_xy_m.shape[1] < 1 or local_xy_m.shape[2] != 2
                or local_xy_m.device != self.device or local_xy_m.dtype != torch.float32
                or not torch.isfinite(local_xy_m).all()):
            raise ValueError('terrain query requires finite native-order float32 [N,Q,2] local XY')
        count = local_xy_m.shape[1]
        height = torch.full((self.num_envs, count), -float('inf'), dtype=torch.float32, device=self.device)
        candidates = []
        for group in self.groups:
            xy = local_xy_m[group.owners]
            x, y = xy[..., 0].contiguous(), xy[..., 1].contiguous()
            i = (torch.searchsorted(group.x, x, right=True) - 1).clamp(0, group.x.shape[1] - 2)
            j = (torch.searchsorted(group.y, y, right=True) - 1).clamp(0, group.y.shape[1] - 2)
            x0, x1 = group.x.gather(1, i), group.x.gather(1, i + 1)
            y0, y1 = group.y.gather(1, j), group.y.gather(1, j + 1)
            u, v = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
            rows = torch.arange(len(group.owners), device=self.device)[:, None]
            a, b = group.z[rows, i, j], group.z[rows, i + 1, j]
            c, d = group.z[rows, i + 1, j + 1], group.z[rows, i, j + 1]
            z = torch.where(v <= u, a + u * (b - a) + v * (c - b),
                            a + u * (c - d) + v * (d - a))
            valid = ((x >= group.x[:, :1]) & (x <= group.x[:, -1:])
                     & (y >= group.y[:, :1]) & (y <= group.y[:, -1:]))
            z = torch.where(valid, z, -float('inf'))
            owners = group.owners[:, None].expand(-1, count)
            height.scatter_reduce_(0, owners, z, reduce='amax', include_self=True)
            candidates.append((group, owners, z, valid))
        selected = torch.full((self.num_envs, count), self.total_patches, dtype=torch.long, device=self.device)
        for group, owners, z, valid in candidates:
            eligible = valid & (z == height[group.owners])
            ids = torch.where(eligible, group.patch_ids[:, None], self.total_patches)
            selected.scatter_reduce_(0, owners, ids, reduce='amin', include_self=True)
        friction = torch.zeros_like(height)
        for group, owners, _, valid in candidates:
            chosen = valid & (selected[group.owners] == group.patch_ids[:, None])
            friction.scatter_add_(0, owners, torch.where(chosen, group.friction[:, None], 0.))
        valid = selected < self.total_patches
        if not torch.isfinite(height[valid]).all() or not torch.isfinite(friction[valid]).all():
            raise ValueError('terrain interpolation overflowed its physical representation')
        return TerrainQuery(torch.where(valid, height, float('nan')),
                            torch.where(valid, friction, float('nan')), valid,
                            torch.where(valid, selected, -1))

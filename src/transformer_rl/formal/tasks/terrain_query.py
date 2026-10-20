"""Exact downward triangle queries over an immutable, tile-indexed atlas.

The index accelerates the authored collision triangles. It does not replace
them with an interpolated height field or fill holes in the source geometry.
"""
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def wheel_lookahead_points(centres_world_m, yaw_rad, forward_m_s, *, near_m=.25, far_m=.8):
    """Current, near and far wheel columns for privileged diagnostics only.

    Direction follows the signed body-forward command and world yaw. These
    are exact geometry-query origins, not deployable ToF measurements.
    """
    if not all(isinstance(v, torch.Tensor) for v in (centres_world_m, yaw_rad, forward_m_s)):
        raise TypeError('wheel lookahead inputs must be tensors')
    n = len(centres_world_m)
    if (centres_world_m.shape != (n, 2, 3) or centres_world_m.dtype != torch.float32
            or yaw_rad.shape != (n,) or forward_m_s.shape != (n,)
            or yaw_rad.dtype != torch.float32 or forward_m_s.dtype != torch.float32
            or yaw_rad.device != centres_world_m.device or forward_m_s.device != centres_world_m.device
            or not all(torch.isfinite(v).all() for v in (centres_world_m, yaw_rad, forward_m_s))
            or type(near_m) not in (int, float) or type(far_m) not in (int, float)
            or not 0 < near_m < far_m < float('inf')):
        raise ValueError('wheel lookahead requires finite wheel centres, yaw, command and ordered metre offsets')
    distances = centres_world_m.new_tensor([0., near_m, far_m])
    direction = torch.where(forward_m_s < 0, -1., 1.)
    axis = torch.stack((yaw_rad.cos(), yaw_rad.sin(), torch.zeros_like(yaw_rad)), -1)
    points = centres_world_m[:, :, None, :] + (
        direction[:, None, None, None] * axis[:, None, None, :] * distances[None, None, :, None])
    points[..., 2] += 5.
    return points


def build_query_index(vertices, faces, face_tiles, tile_count, *, cell_size_m=.25):
    """Build a conservative XY cell -> triangle CSR table before simulation."""
    vertices = np.asarray(vertices, np.float32)
    faces, face_tiles = np.asarray(faces, np.int32), np.asarray(face_tiles, np.int32)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or face_tiles.shape != (len(faces),) or not np.isfinite(vertices).all()
            or (faces < 0).any() or (faces >= len(vertices)).any()
            or not np.isfinite(cell_size_m) or cell_size_m <= 0
            or (face_tiles < 0).any() or (face_tiles >= tile_count).any()):
        raise ValueError('triangle index requires finite geometry and one tile per face')
    triangles = vertices[faces]
    edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    projected_area = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    origins, shapes, bases, cells, candidates = [], [], [], [], []
    total = 0
    for tile in range(tile_count):
        indices = np.flatnonzero((face_tiles == tile) & (np.abs(projected_area) > 1e-14))
        if not len(indices):
            raise ValueError('terrain tile has no vertically queryable physical surface')
        points = triangles[indices, :, :2]
        origin = np.floor((points.min((0, 1)) - 1e-5) / cell_size_m) * cell_size_m
        shape = np.floor((points.max((0, 1)) + 1e-5 - origin) / cell_size_m).astype(np.int32) + 1
        origins.append(origin)
        shapes.append(shape)
        bases.append(total)
        for face, triangle in zip(indices, points):
            low = np.floor((triangle.min(0) - 1e-5 - origin) / cell_size_m).astype(int)
            high = np.floor((triangle.max(0) + 1e-5 - origin) / cell_size_m).astype(int)
            x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1), indexing='ij')
            cell = total + x.ravel() * shape[1] + y.ravel()
            cells.append(cell)
            candidates.append(np.full(len(cell), face, np.int32))
        total += int(shape.prod())
    cells, candidates = np.concatenate(cells), np.concatenate(candidates)
    order = np.argsort(cells, kind='stable')
    offsets = np.concatenate(([0], np.cumsum(np.bincount(cells, minlength=total))))
    if offsets[-1] > 2**31 - 1 or total > 2**31 - 1:
        raise ValueError('terrain query index exceeds int32 capacity')
    return {'origins': np.asarray(origins, np.float32), 'shapes': np.asarray(shapes, np.int32),
            'bases': np.asarray(bases, np.int32), 'offsets': offsets.astype(np.int32),
            'candidates': candidates[order], 'cell_size_m': np.asarray(cell_size_m, np.float32)}


def _kernel():
    # Import Warp only for runtime queries, keeping asset preparation SDK-free.
    global wp
    import warp as wp

    @wp.kernel
    def query(points: wp.array(dtype=wp.vec3), tile: wp.array(dtype=wp.int32),
              origins: wp.array(dtype=wp.vec2), shapes: wp.array(dtype=wp.vec2i),
              bases: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
              candidates: wp.array(dtype=wp.int32), vertices: wp.array(dtype=wp.vec3),
              faces: wp.array2d(dtype=wp.int32), materials: wp.array(dtype=wp.float32),
              cell_size: wp.float32, tile_count: wp.int32, height: wp.array(dtype=wp.float32),
              valid: wp.array(dtype=wp.int32), normal: wp.array(dtype=wp.vec3),
              friction: wp.array(dtype=wp.float32)):
        i = wp.tid()
        p, t = points[i], tile[i]
        best = float(-1.0e20)
        found = int(0)
        hit_normal = wp.vec3(0.0, 0.0, 0.0)
        hit_friction = float(0.0)
        begin = int(0)
        end = int(0)
        if t >= 0 and t < tile_count and wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
            origin, shape = origins[t], shapes[t]
            x = int(wp.floor((p[0] - origin[0]) / cell_size))
            y = int(wp.floor((p[1] - origin[1]) / cell_size))
            if x >= 0 and y >= 0 and x < shape[0] and y < shape[1]:
                cell = bases[t] + x * shape[1] + y
                begin = offsets[cell] + int(0)
                end = offsets[cell + 1] + int(0)
            for entry in range(begin, end):
                f = candidates[entry]
                a = vertices[faces[f, 0]]
                u = vertices[faces[f, 1]] - a
                v = vertices[faces[f, 2]] - a
                denominator = u[0] * v[1] - u[1] * v[0]
                s = ((p[0] - a[0]) * v[1] - (p[1] - a[1]) * v[0]) / denominator
                q = (u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0])) / denominator
                z = a[2] + s * u[2] + q * v[2]
                if s >= -1.0e-5 and q >= -1.0e-5 and s + q <= 1.00001 and z <= p[2] and z > best:
                    best = z
                    found = 1
                    n = wp.normalize(wp.cross(u, v))
                    if n[2] < 0.0:
                        n = -n
                    hit_normal = n
                    hit_friction = materials[f]
        # A miss stays invalid and visibly nonfinite; never synthesize a floor.
        height[i] = best
        valid[i] = found
        normal[i] = hit_normal
        friction[i] = hit_friction
    return query


class StaticTerrainQuery:
    """One GPU kernel per batch, on the caller's Torch stream; no host wait."""

    def __init__(self, geometry_path, index_path, *, device='cpu'):
        import warp as wp
        wp.init()
        self.device = torch.device(device)
        self._wp = wp
        with np.load(Path(geometry_path), allow_pickle=False) as source:
            vertices = source['vertices_m'].copy()
            faces = source['triangles'].astype(np.int32)
            friction = source['material_coefficients'][source['face_material']].astype(np.float32)
        with np.load(Path(index_path), allow_pickle=False) as source:
            data = {key: source[key].copy() for key in source.files}
        self.cell_size_m = float(data.pop('cell_size_m'))
        self.tile_count = len(data['origins'])
        data.update(vertices=vertices, faces=faces, materials=friction)
        self._buffers = {key: torch.as_tensor(value, device=self.device).contiguous() for key, value in data.items()}
        types = {'origins': wp.vec2, 'shapes': wp.vec2i, 'vertices': wp.vec3}
        self._arrays = {key: wp.from_torch(value, dtype=types.get(key)) for key, value in self._buffers.items()}
        self._query = _kernel()
        self.queries = 0
        self.misses = torch.zeros((), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def __call__(self, points, tiles):
        if points.shape[-1] != 3 or points.dtype != torch.float32 or points.device != self.device:
            raise ValueError('terrain query points require device-local float32 XYZ')
        if tiles.shape != points.shape[:-1] or tiles.device != self.device:
            raise ValueError('terrain query requires one assigned tile per point')
        wp = self._wp
        shape = points.shape[:-1]
        p = points.reshape(-1, 3).contiguous()
        t = tiles.reshape(-1).to(torch.int32).contiguous()
        height = torch.empty(len(p), dtype=torch.float32, device=self.device)
        valid = torch.empty(len(p), dtype=torch.int32, device=self.device)
        normal = torch.empty_like(p)
        friction = torch.empty_like(height)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.type == 'cuda' else None
        wp.launch(self._query, dim=len(p), inputs=[
            wp.from_torch(p, dtype=wp.vec3), wp.from_torch(t),
            *[self._arrays[key] for key in ('origins', 'shapes', 'bases', 'offsets', 'candidates', 'vertices', 'faces', 'materials')],
            self.cell_size_m, self.tile_count, wp.from_torch(height), wp.from_torch(valid),
            wp.from_torch(normal, dtype=wp.vec3), wp.from_torch(friction)],
            device=str(self.device), stream=stream)
        valid = valid.bool()
        height.masked_fill_(~valid, float('nan'))
        self.queries += len(p)
        self.misses.add_((~valid).sum())
        return (height.reshape(shape), valid.reshape(shape), normal.reshape(*shape, 3), friction.reshape(shape))

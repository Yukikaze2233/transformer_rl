"""Private finite-cylinder clearance against the authored triangle heights.

This is a signed WORLD-Z gap, not Euclidean distance or an EPA penetration
depth. The minimum includes the entire solid wheel footprint, so a higher
step under its rim cannot be hidden by querying only below the wheel centre.
Native collision-shape/cooking equivalence still requires SDK acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import math

import torch

from ..tasks.geometry import TerrainQueryBatch


@dataclass(frozen=True)
class CylinderGap:
    vertical_gap_m: torch.Tensor
    cylinder_point_local_m: torch.Tensor
    terrain_point_local_m: torch.Tensor
    triangle_index: torch.Tensor
    domain_covered: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class WheelClearanceSample:
    geometry: CylinderGap
    environment_origins_world_m: torch.Tensor
    timestamp_s: torch.Tensor
    physical_tick: int
    model_epoch: int
    body_names: tuple[str, str]
    urdf_sha256: str | None
    source_geometry_bound: bool
    native_collision_equivalence_verified: bool
    reference: str = 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint'


class CylinderTerrainGap:
    """Minimize each top-face plane over cylinder AND its triangular prism.

    A linear objective on this clipped convex solid reaches its minimum at
    an unconstrained support point, a vertical edge-plane intersection, or a
    vertical vertex line. Enumerating those active sets avoids rim sampling,
    iterative optimization and an infinite supporting-plane extrapolation.
    Only grid cells intersecting the exact cylinder AABB are expanded.
    """

    def __init__(self, terrain: TerrainQueryBatch, radii_m, lengths_m):
        radii, lengths = tuple(radii_m), tuple(lengths_m)
        if (not isinstance(terrain, TerrainQueryBatch) or not radii or len(radii) != len(lengths)
                or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
                       for v in (*radii, *lengths))):
            raise ValueError('cylinder query requires the authored terrain and positive complete dimensions')
        self.terrain, self.device, self.num_envs = terrain, terrain.device, terrain.num_envs
        self.count = len(radii)
        self._r = torch.tensor(radii, dtype=torch.float64, device=self.device)
        self._h = torch.tensor(lengths, dtype=torch.float64, device=self.device) * .5
        offsets, face_count, bounds, holes = [], 0, [], []
        for owner, surface in enumerate(terrain.surfaces):
            self._validate_risers(surface)
            rectangles = [(p.x_m[0], p.x_m[-1], p.y_m[0], p.y_m[-1]) for p in surface.patches]
            xs = sorted({v for r in rectangles for v in r[:2]})
            ys = sorted({v for r in rectangles for v in r[2:]})
            bounds.append((xs[0], xs[-1], ys[0], ys[-1]))
            for x0, x1 in zip(xs, xs[1:]):
                for y0, y1 in zip(ys, ys[1:]):
                    x, y = (x0 + x1) * .5, (y0 + y1) * .5
                    if not any(a <= x <= b and c <= y <= d for a, b, c, d in rectangles):
                        holes.append((owner, x0, x1, y0, y1))
            for patch in surface.patches:
                offsets.append(face_count)
                face_count += 2 * (len(patch.x_m) - 1) * (len(patch.y_m) - 1)
        self._offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        self._bounds = torch.tensor(bounds, dtype=torch.float64, device=self.device)
        self._holes = tuple(holes)
        self.total_triangles = face_count

    @staticmethod
    def _validate_risers(surface):
        # The signed vertical definition covers the filled terrain below its
        # top faces. An extra wall above those tops would be omitted geometry,
        # so reject it rather than silently treating it as a harmless riser.
        for wall in surface.vertical_faces:
            for indices in wall.triangles:
                a, b, c = (wall.vertices_m[i] for i in indices)
                if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-12:
                    raise ValueError('extra terrain surfaces must be represented by their actual height patches')
            for x, y, z in wall.vertices_m:
                tops = []
                for patch in surface.patches:
                    if not (patch.x_m[0] <= x <= patch.x_m[-1] and patch.y_m[0] <= y <= patch.y_m[-1]):
                        continue
                    i = min(max(bisect_right(patch.x_m, x) - 1, 0), len(patch.x_m) - 2)
                    j = min(max(bisect_right(patch.y_m, y) - 1, 0), len(patch.y_m) - 2)
                    u = (x - patch.x_m[i]) / (patch.x_m[i + 1] - patch.x_m[i])
                    v = (y - patch.y_m[j]) / (patch.y_m[j + 1] - patch.y_m[j])
                    a, b = patch.z_m[i][j], patch.z_m[i + 1][j]
                    c, d = patch.z_m[i + 1][j + 1], patch.z_m[i][j + 1]
                    tops.append(a + u * (b - a) + v * (c - b) if v <= u else a + u * (c - d) + v * (d - a))
                if not tops or z > max(tops) + 1e-7:
                    raise ValueError('a riser above or outside its authored top faces is unrepresented collision geometry')

    @staticmethod
    def _inside(point, triangle):
        edge = triangle[:, [1, 2, 0], :2] - triangle[:, :, :2]
        offset = point[:, None, :2] - triangle[:, :, :2]
        cross = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
        return (cross >= -1e-9 * edge.norm(dim=-1)).all(-1)

    @classmethod
    def _minimum(cls, centre, axis, radius, half, triangle):
        """All inputs are float64; return a feasible witness, never just a bound."""
        normal = torch.linalg.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        normal = normal / normal[:, 2:3]
        along = (normal * axis).sum(-1)
        radial = normal - along[:, None] * axis
        radial_norm = radial.norm(dim=-1)
        point = (centre - (along.sign() * half)[:, None] * axis
                 - radius[:, None] * radial / radial_norm.clamp_min(1e-30)[:, None])
        candidates = [point]
        eligible = [cls._inside(point, triangle)]
        for index in range(3):
            start, end = triangle[:, index], triangle[:, (index + 1) % 3]
            edge = end - start
            length = edge[:, :2].norm(dim=-1)
            plane = torch.stack((edge[:, 1], -edge[:, 0], torch.zeros_like(length)), -1) / length[:, None]
            k = (plane * (start - centre)).sum(-1)
            ma = (plane * axis).sum(-1)
            transverse = plane - ma[:, None] * axis
            b = transverse.norm(dim=-1)
            e = transverse / b.clamp_min(1e-30)[:, None]
            f = torch.linalg.cross(axis, e)
            ne, nf = (normal * e).sum(-1), (normal * f).sum(-1)
            nonparallel = ma.abs() > 1e-12
            safe_ma = torch.where(nonparallel, ma, torch.ones_like(ma))
            lower, upper = (k - b * radius) / safe_ma, (k + b * radius) / safe_ma
            lo = torch.where(nonparallel, torch.minimum(lower, upper).clamp_min(-half), -half)
            hi = torch.where(nonparallel, torch.maximum(lower, upper).clamp_max(half), half)
            reachable = torch.where(nonparallel, lo <= hi + 1e-10, k.abs() <= b * radius + 1e-10)
            linear = ne * ma - along * b
            denom = torch.sqrt(linear.square() + (nf * ma).square())
            optimum_t = -radius * linear * ma.sign() / denom.clamp_min(1e-30)
            s = (k - b * optimum_t) / safe_ma
            s = torch.where(denom > 1e-25, s, (lo + hi) * .5)
            s = torch.where(nonparallel, s, -along.sign() * half)
            s = torch.maximum(lo, torch.minimum(hi, s))
            t = (k - ma * s) / b.clamp_min(1e-30)
            w = -nf.sign() * (radius.square() - t.square()).clamp_min(0).sqrt()
            side_point = centre + s[:, None] * axis + t[:, None] * e + w[:, None] * f
            # A plane normal parallel to the axis fixes one axial section.
            cap_s = k / torch.where(ma.abs() > 1e-12, ma, torch.ones_like(ma))
            cap_point = (centre + cap_s[:, None] * axis
                         - radius[:, None] * radial / radial_norm.clamp_min(1e-30)[:, None])
            cap = b <= 1e-12
            side_point = torch.where(cap[:, None], cap_point, side_point)
            reachable = torch.where(cap, cap_s.abs() <= half + 1e-10, reachable)
            residual = (plane * (side_point - start)).sum(-1).abs() <= 1e-8
            candidates.append(side_point)
            eligible.append(reachable & residual & cls._inside(side_point, triangle))

            # Intersect the vertex's WORLD-Z line with both the radial tube
            # and the finite axial slab. A horizontal or vertical axis is
            # handled explicitly, without dividing by a near-zero component.
            offset = torch.cat((start[:, :2] - centre[:, :2], torch.zeros_like(k[:, None])), -1)
            s0 = (offset * axis).sum(-1)
            az = axis[:, 2]
            axial = az.abs() > 1e-12
            safe_az = torch.where(axial, az, torch.ones_like(az))
            z0, z1 = (-half - s0) / safe_az, (half - s0) / safe_az
            axial_lo = torch.where(axial, torch.minimum(z0, z1), -float('inf'))
            axial_hi = torch.where(axial, torch.maximum(z0, z1), float('inf'))
            aa = axis[:, :2].square().sum(-1)
            bb = -s0 * az
            cc = offset.square().sum(-1) - s0.square() - radius.square()
            radial_line = aa > 1e-24
            disc = bb.square() - aa * cc
            root = disc.clamp_min(0).sqrt()
            radial_lo = torch.where(radial_line, (-bb - root) / aa.clamp_min(1e-30), -float('inf'))
            radial_hi = torch.where(radial_line, (-bb + root) / aa.clamp_min(1e-30), float('inf'))
            low, high = torch.maximum(axial_lo, radial_lo), torch.minimum(axial_hi, radial_hi)
            vertex_point = start.clone()
            vertex_point[:, 2] = centre[:, 2] + low
            vertex_valid = ((axial | (s0.abs() <= half + 1e-10))
                            & torch.where(radial_line, disc >= -1e-14, cc <= 1e-14)
                            & (low <= high + 1e-10) & torch.isfinite(low))
            candidates.append(vertex_point)
            eligible.append(vertex_valid)
        points, valid = torch.stack(candidates, 1), torch.stack(eligible, 1)
        relative = points - centre[:, None]
        axial_distance = (relative * axis[:, None]).sum(-1)
        radial_distance = (relative - axial_distance[..., None] * axis[:, None]).norm(dim=-1)
        valid &= (torch.isfinite(points).all(-1) & (axial_distance.abs() <= half[:, None] + 1e-8)
                  & (radial_distance <= radius[:, None] + 1e-8))
        gap = ((points - triangle[:, :1]) * normal[:, None]).sum(-1)
        gap = torch.where(valid, gap, float('inf'))
        best, choice = gap.min(-1)
        witness = points[torch.arange(len(points), device=points.device), choice]
        return best, witness

    def _domain(self, centre, axis, extents):
        lo, hi = centre[..., :2] - extents[..., :2], centre[..., :2] + extents[..., :2]
        b = self._bounds
        covered = ((lo[..., 0] >= b[:, None, 0]) & (hi[..., 0] <= b[:, None, 1])
                   & (lo[..., 1] >= b[:, None, 2]) & (hi[..., 1] <= b[:, None, 3]))
        for owner, x0, x1, y0, y1 in self._holes:
            # Only the open missing rectangle is unknown. Shared authored
            # boundary lines are still part of the neighbouring terrain.
            dx, dy = min(1e-9, (x1 - x0) * .25), min(1e-9, (y1 - y0) * .25)
            v = centre.new_tensor(((x0 + dx, y0 + dy, 0.), (x1 - dx, y0 + dy, 0.),
                                   (x1 - dx, y1 - dy, 0.), (x0 + dx, y1 - dy, 0.)))
            for indices in ((0, 1, 2), (0, 2, 3)):
                value, _ = self._minimum(centre[owner], axis[owner], self._r, self._h,
                                         v[list(indices)][None].expand(self.count, -1, -1))
                covered[owner] &= ~torch.isfinite(value)
        return covered

    @torch.no_grad()
    def at(self, centre_local_m, axis_world):
        shape = (self.num_envs, self.count, 3)
        for value in (centre_local_m, axis_world):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device
                    or value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all()):
                raise ValueError('cylinder query requires finite same-device centres and world axes')
        if not torch.allclose(axis_world.norm(dim=-1), torch.ones(shape[:2], device=self.device,
                                                                  dtype=axis_world.dtype), atol=1e-5, rtol=0):
            raise ValueError('cylinder world axes must be normalized')
        centre, axis = centre_local_m.double(), axis_world.double()
        axis = axis / axis.norm(dim=-1, keepdim=True)
        extent = self._h[None, :, None] * axis.abs() + self._r[None, :, None] * (1 - axis.square()).clamp_min(0).sqrt()
        covered = self._domain(centre, axis, extent)
        size = self.num_envs * self.count
        gap = centre.new_full((size,), float('inf'))
        winner = torch.full((size,), self.total_triangles, dtype=torch.long, device=self.device)
        witness = centre.new_full((size, 3), float('nan'))
        for group in self.terrain.groups:
            c, a, ex = centre[group.owners], axis[group.owners], extent[group.owners]
            rows = len(group.owners)
            x, y = group.x.double(), group.y.double()
            lower, upper = c - ex, c + ex
            i0 = (torch.searchsorted(x, lower[..., 0].contiguous(), right=False) - 1).clamp(0, x.shape[1] - 2)
            i1 = (torch.searchsorted(x, upper[..., 0].contiguous(), right=True) - 1).clamp(0, x.shape[1] - 2)
            j0 = (torch.searchsorted(y, lower[..., 1].contiguous(), right=False) - 1).clamp(0, y.shape[1] - 2)
            j1 = (torch.searchsorted(y, upper[..., 1].contiguous(), right=True) - 1).clamp(0, y.shape[1] - 2)
            intersects = ((upper[..., 0] >= x[:, :1]) & (lower[..., 0] <= x[:, -1:])
                          & (upper[..., 1] >= y[:, :1]) & (lower[..., 1] <= y[:, -1:]))
            nj = j1 - j0 + 1
            counts = torch.where(intersects, (i1 - i0 + 1) * nj, 0).flatten()
            selected = torch.repeat_interleave(torch.arange(rows * self.count, device=self.device), counts)
            if selected.numel() == 0:
                continue
            rank = torch.arange(len(selected), device=self.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
            p, w = selected // self.count, selected % self.count
            i = i0.flatten()[selected] + rank // nj.flatten()[selected]
            j = j0.flatten()[selected] + rank % nj.flatten()[selected]
            vertices = torch.stack(tuple(torch.stack((x[p, ii], y[p, jj], group.z[p, ii, jj].double()), -1)
                                         for ii, jj in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))), 1)
            triangles = torch.stack((vertices[:, [0, 1, 2]], vertices[:, [0, 2, 3]]), 1).flatten(0, 1)
            cc, aa = c[p, w].repeat_interleave(2, 0), a[p, w].repeat_interleave(2, 0)
            values, points = self._minimum(cc, aa, self._r[w].repeat_interleave(2),
                                          self._h[w].repeat_interleave(2), triangles)
            target = (group.owners[p] * self.count + w).repeat_interleave(2)
            face = (self._offsets[group.patch_ids[p]] + 2 * (i * (y.shape[1] - 1) + j))[:, None]
            face = (face + torch.arange(2, device=self.device)).flatten()
            best = gap.clone()
            best.scatter_reduce_(0, target, values, reduce='amin', include_self=True)
            eligible = torch.isfinite(values) & (values == best[target])
            faces = torch.full_like(winner, self.total_triangles)
            faces.scatter_reduce_(0, target, torch.where(eligible, face, self.total_triangles), reduce='amin', include_self=True)
            chosen = eligible & (face == faces[target])
            improve = (best < gap) | ((best == gap) & (faces < winner))
            chosen &= improve[target]
            witness[target[chosen]] = points[chosen]
            winner = torch.where(improve, faces, winner)
            gap = best
        valid = covered.flatten() & torch.isfinite(gap)
        ground = witness.clone()
        ground[:, 2] -= gap
        gap = torch.where(valid, gap, float('nan')).reshape(self.num_envs, self.count)
        witness = torch.where(valid[:, None], witness, float('nan')).reshape(shape)
        ground = torch.where(valid[:, None], ground, float('nan')).reshape(shape)
        return CylinderGap(gap.float(), witness, ground,
                           torch.where(valid, winner, -1).reshape(self.num_envs, self.count), covered,
                           valid.reshape(self.num_envs, self.count))

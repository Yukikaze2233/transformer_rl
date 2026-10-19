"""Private source-collider placement on a proven flat part of an own course.

Every mesh face vertex and complete finite cylinder contributes to support.
This changes only the reset root pose,not links,axes,anchors or collision
geometry. Source equivalence to native cooking and physical settling remain
separate acceptance work; neither a plane fit nor closure grants qualification.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

import torch

from ._joint_boundaries import LegBoundaryGeometry
from ._reset_kinematics import AssetResetKinematics
from ._terrain_tasks import TerrainCourse


@dataclass(frozen=True)
class FlatSpawnPatch:
    bounds_world_xy_m: torch.Tensor  # [N,2,2]: coordinate,lower/upper
    height_world_z_m: torch.Tensor
    course_sha256: tuple[str, ...]
    source_surface_verified: bool = False
    _versions: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.bounds_world_xy_m, torch.Tensor) or not isinstance(self.height_world_z_m, torch.Tensor):
            raise ValueError('flat source proof requires owned numeric tensors')
        object.__setattr__(self, '_versions', (self.bounds_world_xy_m._version, self.height_world_z_m._version))

    @classmethod
    def from_courses(cls, courses, origins_world_m, bounds_local_xy_m, height_local_z_m):
        if (not isinstance(courses, tuple) or not courses or any(not isinstance(v, TerrainCourse) for v in courses)
                or not isinstance(origins_world_m, torch.Tensor) or origins_world_m.shape != (len(courses), 3)
                or origins_world_m.dtype != torch.float64 or not torch.isfinite(origins_world_m).all()):
            raise ValueError('flat spawn proof requires actual own courses and explicit clone world origins')
        n, device = len(courses), origins_world_m.device
        for name, value, shape in (('local bounds', bounds_local_xy_m, (n, 2, 2)),
                                    ('local plane', height_local_z_m, (n,))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float64
                    or value.device != device or not torch.isfinite(value).all()):
                raise ValueError('flat spawn ' + name + ' rows/device/float64 layout differs')
        if (bounds_local_xy_m[:, :, 0] >= bounds_local_xy_m[:, :, 1]).any():
            raise ValueError('flat spawn rectangles require positive extent')
        for row, course in enumerate(courses):
            (x0, x1), (y0, y1) = bounds_local_xy_m[row].tolist()
            height = height_local_z_m[row].item()
            rectangles = []
            for patch in course.surface.patches:
                # Inspect only cells intersecting this reset footprint. The
                # entire distant course is still authored and stored unchanged.
                xi = range(max(0, bisect_left(patch.x_m, x0) - 1), min(len(patch.x_m) - 1, bisect_right(patch.x_m, x1)))
                yi = range(max(0, bisect_left(patch.y_m, y0) - 1), min(len(patch.y_m) - 1, bisect_right(patch.y_m, y1)))
                for i in xi:
                    for j in yi:
                        a, b = max(x0, patch.x_m[i]), min(x1, patch.x_m[i + 1])
                        c, d = max(y0, patch.y_m[j]), min(y1, patch.y_m[j + 1])
                        if a > b or c > d:
                            continue
                        if a == b or c == d:
                            # Shared-edge ownership uses the highest source
                            # surface. Conservatively refuse a nonflat/higher
                            # neighbouring cell touching the spawn boundary.
                            if any(patch.z_m[u][v] > height + 1e-8 for u in (i, i + 1) for v in (j, j + 1)):
                                raise ValueError('spawn boundary touches a raised source surface')
                            continue
                        if any(abs(patch.z_m[u][v] - height) > 1e-8 for u in (i, i + 1) for v in (j, j + 1)):
                            raise ValueError('spawn rectangle crosses a nonflat or differently elevated source triangle')
                        rectangles.append((a, b, c, d))
            xs = sorted({x0, x1, *(a for a, _, _, _ in rectangles), *(b for _, b, _, _ in rectangles)})
            ys = sorted({y0, y1, *(c for _, _, c, _ in rectangles), *(d for _, _, _, d in rectangles)})
            for a, b in zip(xs, xs[1:]):
                for c, d in zip(ys, ys[1:]):
                    x, y = (a + b) / 2, (c + d) / 2
                    if not any(lo <= x <= hi and bottom <= y <= top for lo, hi, bottom, top in rectangles):
                        raise ValueError('flat spawn rectangle contains a hole or leaves its own course')
            # Extra source walls may not protrude into this otherwise flat box.
            for wall in course.surface.vertical_faces:
                vertices = wall.vertices_m
                if (max(v[0] for v in vertices) >= x0 and min(v[0] for v in vertices) <= x1
                        and max(v[1] for v in vertices) >= y0 and min(v[1] for v in vertices) <= y1
                        and max(v[2] for v in vertices) > height + 1e-8):
                    raise ValueError('spawn rectangle intersects a raised source wall')
        hashes = tuple(hashlib.sha256(json.dumps(asdict(course), sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest() for course in courses)
        return cls(bounds_local_xy_m.detach().clone() + origins_world_m[:, :2, None],
                    height_local_z_m.detach().clone() + origins_world_m[:, 2], hashes, True)


@dataclass(frozen=True)
class CollisionPlacement:
    root_pose_xyzw: torch.Tensor
    tree_position: torch.Tensor
    body_minimum_gap_m: torch.Tensor
    footprint_world_xy_m: torch.Tensor
    patch_covers_full_footprint: torch.Tensor
    wheel_gap_m: torch.Tensor
    bilateral_wheel_support: torch.Tensor
    maximum_loop_residual_m: torch.Tensor
    within_native_joint_limits: torch.Tensor
    geometry_ready: torch.Tensor
    course_sha256: tuple[str, ...]
    physical_qualification: bool = False


class AssetCollisionGeometry:
    """Bound source meshes/cylinders; per-reset support uses bounded point chunks."""

    def __init__(self, kinematics: AssetResetKinematics, manifest_bytes, model_spec_bytes, mesh_bytes,
                 *, vertex_chunk_size=256):
        if (not isinstance(kinematics, AssetResetKinematics)
                or hashlib.sha256(manifest_bytes).hexdigest() != kinematics.manifest_sha256
                or hashlib.sha256(model_spec_bytes).hexdigest() != kinematics.model_sha256
                or not isinstance(mesh_bytes, dict) or type(vertex_chunk_size) is not int or vertex_chunk_size < 1):
            raise ValueError('collision support requires the exact same selected kinematic source and bounded point chunks')
        manifest, spec = json.loads(manifest_bytes), json.loads(model_spec_bytes)
        self.kinematics, self.device, self.chunk_size = kinematics, kinematics.device, vertex_chunk_size
        self._mesh_points, self._cylinders = [], []
        names = kinematics.binding.native_body_names
        requested = set()
        body_map = {body['name']: body for body in spec['bodies']}
        for index, name in enumerate(names):
            collisions = body_map[name].get('collisions')
            if not isinstance(collisions, list) or not collisions:
                raise ValueError('every selected body needs its actual nonempty collision source')
            points = []
            for collision in collisions:
                origin = LegBoundaryGeometry._origin(collision).to(self.device)
                kind = collision['type']
                if kind == 'mesh':
                    filename = collision['file']
                    raw = mesh_bytes.get(filename)
                    if (not isinstance(filename, str) or not isinstance(raw, bytes)
                            or manifest['files_sha256'].get(filename) != hashlib.sha256(raw).hexdigest()
                            or 'scale' in collision and collision['scale'] != [1., 1., 1.]):
                        raise ValueError('collision mesh source hash/explicit unit scale differs')
                    requested.add(filename)
                    vertices = self._obj_face_vertices(raw).to(self.device)
                    points.append(vertices @ origin[:3, :3].T + origin[:3, 3])
                elif kind == 'cylinder':
                    radius, length = collision['radius'], collision['length']
                    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (radius, length)):
                        raise ValueError('finite cylinder support needs its exact positive source dimensions')
                    self._cylinders.append((index, origin[:3, 3], origin[:3, 2], radius, length))
                else:
                    raise ValueError('unimplemented source collider cannot be replaced with an estimated box')
            if points:
                self._mesh_points.append((index, torch.cat(points)))
        if set(mesh_bytes) != requested:
            raise ValueError('collision source bundle must contain exactly the selected referenced mesh files')
        self.source_mesh_pins = tuple({'name': name, 'bytes': len(mesh_bytes[name]),
                                       'sha256': hashlib.sha256(mesh_bytes[name]).hexdigest()}
                                      for name in sorted(mesh_bytes))

    @classmethod
    def from_asset_directory(cls, kinematics, asset_directory, *, maximum_file_bytes=2097152,
                             maximum_total_mesh_bytes=33554432, vertex_chunk_size=256):
        root = Path(asset_directory).resolve(strict=True)
        if (not root.is_dir() or any(type(v) is not int or v < 1 for v in (maximum_file_bytes, maximum_total_mesh_bytes))):
            raise ValueError('collision source reading requires a bounded selected asset directory')

        def read(name, cap):
            path = root / name
            if (Path(name).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(root)
                    or not path.is_file() or path.stat().st_size > cap):
                raise ValueError('selected collision source escaped its directory or read cap')
            data = path.read_bytes()
            if len(data) > cap:
                raise ValueError('collision source grew beyond the explicit read cap')
            return data

        manifest = read('manifest.json', 65536)
        spec = read('model_spec.json', 65536)
        if (hashlib.sha256(manifest).hexdigest() != kinematics.manifest_sha256
                or hashlib.sha256(spec).hexdigest() != kinematics.model_sha256):
            raise ValueError('selected collision source changed before dependency discovery')
        parsed = json.loads(spec)
        names = {collision['file'] for body in parsed['bodies'] for collision in body.get('collisions', ())
                 if collision['type'] == 'mesh'}
        files, total = {}, 0
        for name in sorted(names):
            data = read(name, min(maximum_file_bytes, maximum_total_mesh_bytes - total))
            files[name], total = data, total + len(data)
        return cls(kinematics, manifest, spec, files, vertex_chunk_size=vertex_chunk_size)

    @staticmethod
    def _obj_face_vertices(raw):
        vertices, referenced = [], set()
        for line in raw.decode('utf-8').splitlines():
            fields = line.partition('#')[0].split()
            if not fields:
                continue
            if fields[0] == 'v':
                if len(fields) != 4:
                    raise ValueError('collision OBJ vertices must be explicit xyz without a hidden homogeneous scale')
                point = tuple(float(v) for v in fields[1:])
                if any(not math.isfinite(v) for v in point):
                    raise ValueError('nonfinite collision mesh vertex')
                vertices.append(point)
            elif fields[0] == 'f':
                if len(fields) < 4:
                    raise ValueError('collision OBJ faces require at least three vertices')
                for field in fields[1:]:
                    value = int(field.split('/')[0])
                    index = value - 1 if value > 0 else len(vertices) + value
                    if value == 0 or not 0 <= index < len(vertices):
                        raise ValueError('collision OBJ face index is outside the actual source vertices')
                    referenced.add(index)
        if not referenced:
            raise ValueError('collision mesh has no actual face support')
        return torch.tensor([vertices[index] for index in sorted(referenced)], dtype=torch.float64)

    @staticmethod
    def _rotation(quaternion):
        x, y, z, w = quaternion.unbind(-1)
        return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)

    @torch.no_grad()
    def place(self, tree_position, root_orientation_xyzw, root_world_xy_m, patch: FlatSpawnPatch, *,
              clearance_m, loop_tolerance_m, wheel_support_tolerance_m):
        self.kinematics._positions(tree_position)
        n = len(tree_position)
        if not isinstance(patch, FlatSpawnPatch):
            raise TypeError('source collision birth needs a proven own flat patch')
        if (not patch.source_surface_verified
                or patch._versions != (patch.bounds_world_xy_m._version, patch.height_world_z_m._version)):
            raise ValueError('flat source proof is missing or its owned numeric bounds were modified')
        for value, shape in ((root_orientation_xyzw, (n, 4)), (root_world_xy_m, (n, 2)),
                              (patch.bounds_world_xy_m, (n, 2, 2)), (patch.height_world_z_m, (n,))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float64
                    or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError('source birth positions,orientations and patch require finite float64 native rows')
        if ((root_orientation_xyzw.norm(dim=-1) - 1).abs().gt(1e-8).any()
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (clearance_m, loop_tolerance_m, wheel_support_tolerance_m))
                or loop_tolerance_m == 0 or wheel_support_tolerance_m == 0 or len(patch.course_sha256) != n):
            raise ValueError('source birth needs unit xyzw and explicit plane/closure/support tolerances')
        state = self.kinematics.forward(tree_position)
        root = self._rotation(root_orientation_xyzw)
        rotations = root[:, None] @ state.body_rotation
        positions = (root[:, None] @ state.body_position_m[..., None]).squeeze(-1)
        positions[:, :, :2] += root_world_xy_m[:, None]
        low = torch.full((n, 19, 3), float('inf'), dtype=torch.float64, device=self.device)
        high = -low.clone()
        for body, points in self._mesh_points:
            for start in range(0, len(points), self.chunk_size):
                world = points[start:start + self.chunk_size][None] @ rotations[:, body].transpose(-1, -2)
                world += positions[:, body, None]
                low[:, body] = torch.minimum(low[:, body], world.amin(1))
                high[:, body] = torch.maximum(high[:, body], world.amax(1))
        for body, centre, axis, radius, length in self._cylinders:
            centre = positions[:, body] + (rotations[:, body] @ centre).squeeze(-1)
            axis = (rotations[:, body] @ axis).squeeze(-1)
            extent = axis.abs() * length / 2 + radius * (1 - axis.square()).clamp_min(0).sqrt()
            low[:, body] = torch.minimum(low[:, body], centre - extent)
            high[:, body] = torch.maximum(high[:, body], centre + extent)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError('every body must contribute finite complete collision support')
        height = patch.height_world_z_m + clearance_m - low[:, :, 2].amin(-1)
        gaps = low[:, :, 2] + height[:, None] - patch.height_world_z_m[:, None]
        footprint = torch.stack((low[:, :, :2].amin(1), high[:, :, :2].amax(1)), -1)
        covers = ((footprint[:, :, 0] >= patch.bounds_world_xy_m[:, :, 0] - 1e-9)
                   & (footprint[:, :, 1] <= patch.bounds_world_xy_m[:, :, 1] + 1e-9)).all(-1)
        wheel_gaps = gaps[:, list(self.kinematics.binding.wheel_body_indices)]
        supported = ((wheel_gaps >= -1e-9) & (wheel_gaps <= wheel_support_tolerance_m)).all(-1)
        residual = self.kinematics.closure(tree_position).norm(dim=-1).amax(-1)
        within = ((tree_position >= self.kinematics._lower - 1e-12)
                   & (tree_position <= self.kinematics._upper + 1e-12)).all(-1)
        ready = covers & within & (residual <= loop_tolerance_m)
        pose = torch.cat((root_world_xy_m, height[:, None], root_orientation_xyzw), -1)
        return CollisionPlacement(pose, tree_position.detach().clone(), gaps, footprint, covers, wheel_gaps,
                                   supported, residual, within, ready, patch.course_sha256)

    @torch.no_grad()
    def reset_plan(self, tree_position, root_orientation_xyzw, root_world_xy_m, patch, commands, *,
                   required_support, clearance_m, loop_tolerance_m, wheel_support_tolerance_m):
        from ._native_task import NativeTaskResetPlan
        from ._task_commands import TaskCommandPlan

        if not isinstance(commands, TaskCommandPlan) or required_support not in ('two_wheels', 'any_collider'):
            raise ValueError('native reset preparation requires explicit commands and standing/recovery support semantics')
        placement = self.place(tree_position, root_orientation_xyzw, root_world_xy_m, patch,
                               clearance_m=clearance_m, loop_tolerance_m=loop_tolerance_m,
                               wheel_support_tolerance_m=wheel_support_tolerance_m)
        ready = placement.geometry_ready.clone()
        if required_support == 'two_wheels':
            ready &= placement.bilateral_wheel_support
        if not ready.all():
            raise ValueError('source reset candidate has unresolved closure,limits,footprint or required support')
        if commands.task_horizon_s.shape != (len(tree_position),) or commands.task_horizon_s.device != self.device:
            raise ValueError('reset commands and source geometry rows differ')
        indices = [self.kinematics.binding.tree_joint_names.index(name)
                   for name in self.kinematics.binding.native_joint_names]
        native_q = tree_position[:, indices].float().detach().clone()
        native_pose = placement.root_pose_xyzw.float().detach().clone()
        # Native writes use float32. Verify the encoded coordinates,not only
        # their earlier float64 solve. Readback interprets xyzw as unit length.
        encoded_tree = native_q[:, list(self.kinematics.binding.tree_joint_indices)].double()
        encoded_orientation = native_pose[:, 3:].double()
        encoded_orientation /= encoded_orientation.norm(dim=-1, keepdim=True)
        encoded = self.place(encoded_tree, encoded_orientation, native_pose[:, :2].double(), patch,
                              clearance_m=clearance_m, loop_tolerance_m=loop_tolerance_m,
                              wheel_support_tolerance_m=wheel_support_tolerance_m)
        required_height = encoded.root_pose_xyzw[:, 2]
        native_height = required_height.float()
        native_height = torch.where(native_height.double() < required_height,
                                     torch.nextafter(native_height, torch.full_like(native_height, float('inf'))), native_height)
        native_pose[:, 2] = native_height
        upward = native_height.double() - required_height
        gaps = encoded.body_minimum_gap_m + upward[:, None]
        wheel_gap = encoded.wheel_gap_m + upward[:, None]
        supported = ((wheel_gap >= 0) & (wheel_gap <= wheel_support_tolerance_m)).all(-1)
        if (not encoded.geometry_ready.all() or not torch.isfinite(native_pose).all()
                or required_support == 'two_wheels' and not supported.all()):
            raise ValueError('float32 reset encoding lost source closure,limits,footprint or required wheel support')
        encoded_pose = encoded.root_pose_xyzw.clone()
        encoded_pose[:, 2] = native_height.double()
        placement = replace(encoded, root_pose_xyzw=encoded_pose, body_minimum_gap_m=gaps,
                            wheel_gap_m=wheel_gap, bilateral_wheel_support=supported)
        plan = NativeTaskResetPlan(native_pose,
                                    torch.zeros(len(tree_position), 6, device=self.device),
                                    native_q, torch.zeros_like(native_q), patch.course_sha256, commands)
        return plan, placement

    @torch.no_grad()
    def balance_two_wheels(self, tree_position, root_orientation_xyzw, root_world_xy_m, patch, *,
                           roll_correction_bounds_rad, clearance_m, loop_tolerance_m,
                           wheel_support_tolerance_m, iterations=32):
        """Propose an explicitly bounded intrinsic-X root correction,not an asset edit.

        Only the two finite-cylinder support heights enter the scalar solve.
        Final acceptance still checks ALL colliders/limits/loops/footprint.
        Unbracketed rows keep the exact requested root orientation and are
        reported separately; bounds and support tolerances are never expanded.
        """
        self.kinematics._positions(tree_position)
        n = len(tree_position)
        for value, shape in ((root_orientation_xyzw, (n, 4)), (roll_correction_bounds_rad, (n, 2))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float64
                    or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError('wheel support correction requires explicit finite float64 row bounds and xyzw')
        if ((root_orientation_xyzw.norm(dim=-1) - 1).abs().gt(1e-8).any()
                or (roll_correction_bounds_rad[:, 0] >= roll_correction_bounds_rad[:, 1]).any()
                or (roll_correction_bounds_rad[:, 0] > 0).any() or (roll_correction_bounds_rad[:, 1] < 0).any()
                or roll_correction_bounds_rad.abs().gt(math.pi / 4).any()
                or type(iterations) is not int or not 1 <= iterations <= 64):
            raise ValueError('bounded birth correction must contain the original pose and remain within45degrees')
        wheels = list(self.kinematics.binding.wheel_body_indices)
        if any(body in wheels for body, _ in self._mesh_points):
            raise ValueError('two-wheel balancing requires the selected complete cylindrical wheel sources')
        cylinders = [[value for value in self._cylinders if value[0] == body] for body in wheels]
        if any(len(values) != 1 for values in cylinders):
            raise ValueError('wheel balancing cannot ignore missing or additional collision primitives')
        state = self.kinematics.forward(tree_position)
        centres, axes, radii, lengths = [], [], [], []
        for values in cylinders:
            body, centre, axis, radius, length = values[0]
            centres.append(state.body_position_m[:, body] + (state.body_rotation[:, body] @ centre).squeeze(-1))
            axes.append((state.body_rotation[:, body] @ axis).squeeze(-1))
            radii.append(radius)
            lengths.append(length)
        centres, axes = torch.stack(centres, 1), torch.stack(axes, 1)
        radius, half_length = tree_position.new_tensor(radii), tree_position.new_tensor(lengths) / 2

        def orientation(correction):
            x, y, z, w = root_orientation_xyzw.unbind(-1)
            s, c = (correction / 2).sin(), (correction / 2).cos()
            return torch.stack((x*c + w*s, y*c + z*s, z*c - y*s, w*c - x*s), -1)

        def difference(correction):
            rotation = self._rotation(orientation(correction))
            centre = (rotation[:, None] @ centres[..., None]).squeeze(-1)
            axis = (rotation[:, None] @ axes[..., None]).squeeze(-1)
            vertical = centre[:, :, 2] - half_length * axis[:, :, 2].abs()
            vertical -= radius * (1 - axis[:, :, 2].square()).clamp_min(0).sqrt()
            return vertical[:, 0] - vertical[:, 1]

        lo, hi = roll_correction_bounds_rad[:, 0].clone(), roll_correction_bounds_rad[:, 1].clone()
        at_lo, at_hi = difference(lo), difference(hi)
        bracketed = (at_lo * at_hi <= 0)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            at_mid = difference(mid)
            left = at_lo * at_mid <= 0
            hi = torch.where(left, mid, hi)
            lo = torch.where(left, lo, mid)
            at_lo = torch.where(left, at_lo, at_mid)
        correction = torch.where(bracketed, (lo + hi) / 2, 0)
        placed_orientation = torch.where(bracketed[:, None], orientation(correction), root_orientation_xyzw)
        placement = self.place(tree_position, placed_orientation, root_world_xy_m, patch,
                                clearance_m=clearance_m, loop_tolerance_m=loop_tolerance_m,
                                wheel_support_tolerance_m=wheel_support_tolerance_m)
        return placement, correction, bracketed

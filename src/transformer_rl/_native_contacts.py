"""Private own-terrain native contact views and shared patch-buffer reduction.

Net force and positive normal contact load are different quantities. Opposing
contacts can cancel the first while the second remains positive. Native patch
starts index one shared buffer, never a lane-local subarray.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._physical_observation import MechanicalBinding, RigidBodyMotion


@dataclass(frozen=True)
class WheelContactMotion:
    """Positive-normal-force-weighted point motion, not wheel/body speed.

    Zero load publishes zero placeholders with loaded=False, not a claim of
    no slip. Only currently authored static own-terrain surfaces are supported.
    """

    tangent_speed_rms_m_s: torch.Tensor
    tangent_speed_max_m_s: torch.Tensor
    normal_speed_rms_m_s: torch.Tensor
    normal_load_n: torch.Tensor
    loaded_patch_count: torch.Tensor
    loaded: torch.Tensor
    timestamp_s: torch.Tensor
    physical_tick: int
    model_epoch: int
    terrain_motion_reference: str


@dataclass(frozen=True)
class TerrainContacts:
    force_world_n: torch.Tensor
    normal_load_n: torch.Tensor
    patch_count: torch.Tensor
    timestamp_s: torch.Tensor
    wheel_motion: WheelContactMotion | None = None


@dataclass(frozen=True)
class _ContactGroup:
    rows: torch.Tensor
    view: object
    sensor_paths: tuple[str, ...]
    filter_paths: tuple[tuple[str, ...], ...]
    surface_count: int
    capacity: int


class ContactCapacityError(RuntimeError):
    """Measurement capacity was reached; this is not a robot control failure."""


class NativeTerrainContacts:
    """Bind an already admitted native physics view to exact body/surface paths.

    Creating this object does not launch a simulator or authorize a GPU job.
    Native clone order must already have been resolved by the scene owner.
    Grouping by surface count preserves variable-size terrain layouts without
    padding nonexistent native filter paths or mixing neighboring environments.
    """

    def __init__(self, physics_view, binding: MechanicalBinding, env_paths, terrain_paths_by_env, *,
                 native_to_tensor, device='cpu', patches_per_body=64, terrain_motion_reference=None):
        if (not isinstance(binding, MechanicalBinding) or not callable(native_to_tensor)
                or type(patches_per_body) is not int or patches_per_body < 1):
            raise ValueError('contacts require a mechanical binding, converter and explicit positive capacity')
        if terrain_motion_reference not in (None, 'authored_static_own_terrain'):
            raise ValueError('contact motion requires a supported explicit terrain-motion binding')
        env_paths = tuple(env_paths)
        terrain_paths_by_env = tuple(tuple(paths) for paths in terrain_paths_by_env)
        if (not env_paths or len(set(env_paths)) != len(env_paths)
                or len(terrain_paths_by_env) != len(env_paths)
                or any(not paths or len(set(paths)) != len(paths) for paths in terrain_paths_by_env)):
            raise ValueError('each distinct environment requires nonempty distinct terrain collider paths')
        for path in (*env_paths, *(path for paths in terrain_paths_by_env for path in paths)):
            if (not isinstance(path, str) or not path.startswith('/') or '..' in path.split('/')
                    or any(character in path for character in '*?[]|')):
                raise ValueError('native contact paths must identify exact USD prims, not broad patterns')
        if any(not path.startswith(env + '/')
               for env, paths in zip(env_paths, terrain_paths_by_env) for path in paths):
            raise ValueError('terrain filters must belong to their corresponding environment')
        self.binding, self.num_envs, self.device = binding, len(env_paths), torch.device(device)
        self._convert = native_to_tensor
        self.terrain_motion_reference = terrain_motion_reference
        self.groups = []
        for size in sorted({len(paths) for paths in terrain_paths_by_env}):
            rows = [index for index, paths in enumerate(terrain_paths_by_env) if len(paths) == size]
            sensors = tuple(env_paths[row] + '/Robot/' + name
                            for row in rows for name in binding.native_body_names)
            filters = tuple(terrain_paths_by_env[row] for row in rows for _ in binding.native_body_names)
            capacity = patches_per_body * len(sensors)
            view = physics_view.create_rigid_contact_view(list(sensors), filter_patterns=[list(p) for p in filters],
                                                          max_contact_data_count=capacity)
            group = _ContactGroup(torch.tensor(rows, device=self.device), view, sensors, filters, size, capacity)
            self._identity(group)
            self.groups.append(group)

    @staticmethod
    def _identity(group):
        view = group.view
        if (view.sensor_count != len(group.sensor_paths) or view.filter_count != group.surface_count
                or tuple(view.sensor_paths) != group.sensor_paths
                or tuple(tuple(paths) for paths in view.filter_paths) != group.filter_paths
                or view.max_contact_data_count != group.capacity):
            raise RuntimeError('native contact sensor/filter identity, order or capacity changed')

    @staticmethod
    def _patch_layout(force, counts, starts, capacity):
        if (not isinstance(force, torch.Tensor) or force.dtype != torch.float32
                or force.ndim not in (1, 2) or force.ndim == 2 and force.shape[1] != 1
                or not isinstance(counts, torch.Tensor) or not isinstance(starts, torch.Tensor)
                or counts.ndim != 2 or starts.shape != counts.shape
                or counts.dtype not in (torch.int32, torch.int64, torch.uint32)
                or starts.dtype not in (torch.int32, torch.int64, torch.uint32)
                or counts.device != force.device or starts.device != force.device
                or type(capacity) is not int or capacity < 1):
            raise ValueError('native shared patch buffer layout or declared capacity differs')
        force, count, start = force.reshape(-1), counts.long(), starts.long()
        used = count > 0
        end = start + count
        if ((count < 0).any() or (used & ((start < 0) | (end < start) | (end > force.numel()))).any()
                or force.numel() > capacity or count.sum() > capacity):
            raise ValueError('native patch ranges exceed the actual shared buffer')
        if count.sum() >= capacity:
            raise ContactCapacityError('native contact patch capacity reached; qualification data is incomplete')
        # Invalid sentinel starts are allowed only for an empty pair. Difference
        # marks avoid reading uninitialized/NaN padding in the shared allocation.
        start = torch.where(used, start, 0)
        end = torch.where(used, end, 0)
        difference = torch.zeros(force.numel() + 1, dtype=torch.long, device=force.device)
        difference.scatter_add_(0, start.flatten(), used.long().flatten())
        difference.scatter_add_(0, end.flatten(), -used.long().flatten())
        occupancy = difference.cumsum(0)[:-1]
        if (occupancy > 1).any():
            raise ValueError('native contact pairs overlap shared patch ranges')
        occupied = occupancy > 0
        if (occupied & (~torch.isfinite(force) | (force < 0))).any():
            raise ValueError('native active normal patch force is invalid')
        return force, count, start, end, occupied

    @staticmethod
    def _loads_from_layout(layout):
        force, count, start, end, occupied = layout
        clean = torch.where(occupied, force, 0).double()
        prefix = torch.cat((clean.new_zeros(1), clean.cumsum(0)))
        loads = prefix[end] - prefix[start]
        total_load = loads.sum(-1).float()
        if not torch.isfinite(total_load).all():
            raise ValueError('native contact load exceeds the published tensor range')
        return total_load, count.sum(-1)

    @staticmethod
    def _patch_loads(force, counts, starts, capacity):
        return NativeTerrainContacts._loads_from_layout(NativeTerrainContacts._patch_layout(force, counts, starts, capacity))

    def _validate_motion(self, motion, physics_dt_s, timestamp_s):
        if (not isinstance(motion, RigidBodyMotion) or motion.body_names != self.binding.native_body_names
                or self.terrain_motion_reference != 'authored_static_own_terrain'
                or type(motion.physical_tick) is not int or motion.physical_tick < 1
                or type(motion.model_epoch) is not int or motion.model_epoch < 1):
            raise ValueError('contact motion requires exact native bodies, fresh tick and explicitly static own terrain')
        for name in ('com_position_world_m', 'com_velocity_world_m_s', 'angular_velocity_world_rad_s'):
            value = getattr(motion, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 19, 3)
                    or value.device != self.device or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('contact body motion needs finite world-SI tensors in native body order')
        if (not isinstance(motion.timestamp_s, torch.Tensor) or motion.timestamp_s.shape != timestamp_s.shape
                or motion.timestamp_s.dtype != torch.float64 or motion.timestamp_s.device != self.device
                or not torch.equal(motion.timestamp_s, timestamp_s)
                or (timestamp_s - motion.physical_tick * physics_dt_s).abs().gt(1e-8).any()):
            raise ValueError('contact body motion and force measurement clocks differ')

    def _wheel_motion(self, group, layout, points, normals, motion):
        force, counts, starts, _, _ = layout
        for name, values in (('points', points), ('normals', normals)):
            if (not isinstance(values, torch.Tensor) or values.shape != (force.numel(), 3)
                    or values.device != self.device or values.dtype != torch.float32):
                raise ValueError('native patch ' + name + ' do not match the actual shared force buffer')
        bodies = len(self.binding.native_body_names)
        wheels = torch.tensor(self.binding.wheel_body_indices, device=self.device)
        sensors = torch.arange(len(group.rows), device=self.device)[:, None] * bodies + wheels
        wheel_counts = counts[sensors].flatten()
        wheel_starts = starts[sensors].flatten()
        # Expand only real wheel ranges, without an individual-pair capacity
        # limit or another dense scan over the maximum native allocation.
        pair = torch.repeat_interleave(torch.arange(wheel_counts.numel(), device=self.device), wheel_counts)
        begins = torch.cat((wheel_counts.new_zeros(1), wheel_counts.cumsum(0)[:-1]))
        indices = wheel_starts[pair] + torch.arange(pair.numel(), device=self.device) - begins[pair]
        positive = force[indices] > 0
        indices, pair = indices[positive], pair[positive]
        bucket = pair // group.surface_count
        selected = (bucket // 2) * bodies + wheels[bucket.remainder(2)]
        p, n = points[indices].double(), normals[indices].double()
        norm = n.norm(dim=-1)
        if (not torch.isfinite(p).all() or not torch.isfinite(n).all()
                or (norm - 1).abs().gt(1e-3).any()):
            raise ValueError('loaded wheel patch needs finite world points and unit world normals')
        n = n / norm[:, None]
        com = motion.com_position_world_m[group.rows].flatten(0, 1)[selected].double()
        velocity = motion.com_velocity_world_m_s[group.rows].flatten(0, 1)[selected].double()
        omega = motion.angular_velocity_world_rad_s[group.rows].flatten(0, 1)[selected].double()
        # Terrain velocity is zero only because this view is explicitly bound
        # to authored static colliders. Moving terrain needs another contract.
        point_velocity = velocity + torch.linalg.cross(omega, p - com)
        normal_velocity = (point_velocity * n).sum(-1)
        tangent_square = (point_velocity - normal_velocity[:, None] * n).square().sum(-1)
        weight = force[indices].double()
        size = len(group.rows) * 2
        total = torch.zeros(size, dtype=torch.float64, device=self.device).scatter_add_(0, bucket, weight)
        tangential = torch.zeros_like(total).scatter_add_(0, bucket, weight * tangent_square)
        normal = torch.zeros_like(total).scatter_add_(0, bucket, weight * normal_velocity.square())
        maximum = torch.zeros_like(total).scatter_reduce_(0, bucket, tangent_square, reduce='amax', include_self=True)
        number = torch.zeros(size, dtype=torch.long, device=self.device).scatter_add_(0, bucket, torch.ones_like(bucket))
        denominator = torch.where(total > 0, total, 1.)
        rms = (tangential / denominator).sqrt()
        normal_rms = (normal / denominator).sqrt()
        result = tuple(v.reshape(len(group.rows), 2) for v in (rms.float(), maximum.sqrt().float(), normal_rms.float(),
                                                             total.float(), number))
        if not all(torch.isfinite(v).all() for v in result[:4]):
            raise ValueError('native contact point motion exceeds its finite physical tensor range')
        return result

    @torch.no_grad()
    def read(self, physics_dt_s, timestamp_s, *, body_motion=None):
        if (type(physics_dt_s) not in (int, float) or physics_dt_s not in (.001, .005)
                or not isinstance(timestamp_s, torch.Tensor) or timestamp_s.shape != (self.num_envs,)
                or timestamp_s.device != self.device or timestamp_s.dtype != torch.float64
                or not torch.isfinite(timestamp_s).all() or (timestamp_s < 0).any()):
            raise ValueError('native contact read requires the selected physical interval and same-batch float64 clock')
        if body_motion is not None:
            self._validate_motion(body_motion, physics_dt_s, timestamp_s)
        bodies = len(self.binding.native_body_names)
        force = torch.zeros(self.num_envs, bodies, 3, device=self.device)
        load = torch.zeros(self.num_envs, bodies, device=self.device)
        patches = torch.zeros(self.num_envs, bodies, dtype=torch.long, device=self.device)
        if body_motion is not None:
            wheel_values = [torch.zeros(self.num_envs, 2, device=self.device) for _ in range(4)]
            wheel_counts = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        for group in self.groups:
            self._identity(group)
            matrix = self._convert(group.view.get_contact_force_matrix(dt=physics_dt_s))
            expected = (len(group.sensor_paths), group.surface_count, 3)
            if (not isinstance(matrix, torch.Tensor) or matrix.shape != expected
                    or matrix.dtype != torch.float32 or matrix.device != self.device or not torch.isfinite(matrix).all()):
                raise ValueError('native own-terrain force matrix shape/device/dtype or values differ')
            packet = group.view.get_contact_data(dt=physics_dt_s)
            if len(packet) != 6:
                raise ValueError('native contact patch tuple layout differs')
            normal_force, counts, starts = (self._convert(packet[index]) for index in (0, 4, 5))
            if (not all(isinstance(value, torch.Tensor) for value in (counts, starts, normal_force))
                    or counts.shape != expected[:2] or starts.shape != expected[:2]
                    or normal_force.device != self.device):
                raise ValueError('native patch ranges do not match the bound sensor/filter matrix')
            layout = self._patch_layout(normal_force, counts, starts, group.capacity)
            normal, count = self._loads_from_layout(layout)
            force[group.rows] = matrix.reshape(len(group.rows), bodies, group.surface_count, 3).sum(2)
            load[group.rows] = normal.reshape(len(group.rows), bodies)
            patches[group.rows] = count.reshape(len(group.rows), bodies)
            if body_motion is not None:
                point, direction = (self._convert(packet[index]) for index in (1, 2))
                values = self._wheel_motion(group, layout, point, direction, body_motion)
                for output, value in zip((*wheel_values, wheel_counts), values):
                    output[group.rows] = value
            self._identity(group)
        if not torch.isfinite(force).all():
            raise ValueError('native net force exceeds the published tensor range')
        wheel_motion = None
        if body_motion is not None:
            wheel_motion = WheelContactMotion(*wheel_values, wheel_counts, wheel_counts > 0, timestamp_s.clone(),
                                               body_motion.physical_tick, body_motion.model_epoch, self.terrain_motion_reference)
        return TerrainContacts(force, load, patches, timestamp_s.clone(), wheel_motion)

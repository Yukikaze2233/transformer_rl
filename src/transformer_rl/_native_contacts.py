"""Private own-terrain native contact views and shared patch-buffer reduction.

Net force and positive normal contact load are different quantities. Opposing
contacts can cancel the first while the second remains positive. Native patch
starts index one shared buffer, never a lane-local subarray.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._physical_observation import MechanicalBinding


@dataclass(frozen=True)
class TerrainContacts:
    force_world_n: torch.Tensor
    normal_load_n: torch.Tensor
    patch_count: torch.Tensor
    timestamp_s: torch.Tensor


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
                 native_to_tensor, device='cpu', patches_per_body=64):
        if (not isinstance(binding, MechanicalBinding) or not callable(native_to_tensor)
                or type(patches_per_body) is not int or patches_per_body < 1):
            raise ValueError('contacts require a mechanical binding, converter and explicit positive capacity')
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
    def _patch_loads(force, counts, starts, capacity):
        if (not isinstance(force, torch.Tensor) or force.dtype != torch.float32
                or force.ndim not in (1, 2) or force.ndim == 2 and force.shape[1] != 1
                or not isinstance(counts, torch.Tensor) or not isinstance(starts, torch.Tensor)
                or counts.ndim != 2 or starts.shape != counts.shape
                or counts.dtype not in (torch.int32, torch.int64) or starts.dtype not in (torch.int32, torch.int64)
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
        clean = torch.where(occupied, force, 0).double()
        prefix = torch.cat((clean.new_zeros(1), clean.cumsum(0)))
        loads = prefix[end] - prefix[start]
        total_load = loads.sum(-1).float()
        if not torch.isfinite(total_load).all():
            raise ValueError('native contact load exceeds the published tensor range')
        return total_load, count.sum(-1)

    @torch.no_grad()
    def read(self, physics_dt_s, timestamp_s):
        if (type(physics_dt_s) not in (int, float) or physics_dt_s not in (.001, .005)
                or not isinstance(timestamp_s, torch.Tensor) or timestamp_s.shape != (self.num_envs,)
                or timestamp_s.device != self.device or timestamp_s.dtype != torch.float64
                or not torch.isfinite(timestamp_s).all() or (timestamp_s < 0).any()):
            raise ValueError('native contact read requires the selected physical interval and same-batch float64 clock')
        bodies = len(self.binding.native_body_names)
        force = torch.zeros(self.num_envs, bodies, 3, device=self.device)
        load = torch.zeros(self.num_envs, bodies, device=self.device)
        patches = torch.zeros(self.num_envs, bodies, dtype=torch.long, device=self.device)
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
            normal, count = self._patch_loads(normal_force, counts, starts, group.capacity)
            force[group.rows] = matrix.reshape(len(group.rows), bodies, group.surface_count, 3).sum(2)
            load[group.rows] = normal.reshape(len(group.rows), bodies)
            patches[group.rows] = count.reshape(len(group.rows), bodies)
            self._identity(group)
        if not torch.isfinite(force).all():
            raise ValueError('native net force exceeds the published tensor range')
        return TerrainContacts(force, load, patches, timestamp_s.clone())

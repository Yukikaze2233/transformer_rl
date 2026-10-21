"""Contact patch loads for a static shared terrain, without an SDK dependency.

Scalar normal loads add even when opposing normals cancel in net force. Only
upward components support a wheel; a horizontal step impact is not a landing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

import torch

from ..control.observation import MeasuredState, NativePhysicalSample, ObservationAssembler
from ..simulation.contacts import NativeTerrainContacts


@dataclass(frozen=True)
class LabPhysicalSample(NativePhysicalSample):
    body_normal_load_n: torch.Tensor
    wheel_support_load_n: torch.Tensor
    contact_measurement_valid: torch.Tensor


@dataclass(frozen=True)
class LabMeasuredState(MeasuredState):
    contact_measurement_valid: torch.Tensor


class LabObservationAssembler(ObservationAssembler):
    def measure(self, sample):
        state = super().measure(sample)
        if not isinstance(sample, LabPhysicalSample):
            return state
        support = sample.wheel_support_load_n
        if (support.shape != (self.num_envs, 2) or support.device != self.device
                or support.dtype != torch.float32 or not torch.isfinite(support).all()
                or (support < 0).any()):
            raise ValueError('Lab wheel support requires finite positive patch loads')
        valid = sample.contact_measurement_valid
        if valid.shape != (self.num_envs,) or valid.dtype != torch.bool or valid.device != self.device:
            raise ValueError('Lab contact validity must follow its physical sample')
        state = replace(state, wheel_ground_force_n=support.clone())
        return LabMeasuredState(**state.__dict__, contact_measurement_valid=valid.clone())


class LabContactLoads:
    """One pinned body/filter view; one patch reduction per physical tick."""

    def __init__(self, sensor, binding, environment_paths, *, collision_path, device):
        self.view = sensor.contact_view
        self.device, self.num_envs = torch.device(device), len(environment_paths)
        environment_paths = tuple(environment_paths)
        names = tuple(sensor.body_names)
        expected = tuple(sensor.body_physx_view.prim_paths)
        native_names = tuple(binding.native_body_names)
        environment_ids = {path: index for index, path in enumerate(environment_paths)}
        native_ids = {name: index for index, name in enumerate(native_names)}
        target_rows, relative_paths = [], {}
        owned = (bool(names) and bool(environment_paths)
                 and len(environment_ids) == self.num_envs
                 and len(native_ids) == len(native_names)
                 and len(set(names)) == len(names) and set(names) == set(native_names))
        for path in expected:
            if not owned or not isinstance(path, str):
                owned = False
                break
            environment, separator, relative = path.partition('/Robot/')
            name = relative.rsplit('/', 1)[-1]
            if (not separator or environment not in environment_ids or name not in native_ids
                    or any(part in ('', '.', '..') for part in relative.split('/'))
                    or relative_paths.setdefault(name, relative) != relative):
                owned = False
                break
            target_rows.append(environment_ids[environment] * len(native_names) + native_ids[name])
        complete = (owned and len(target_rows) == self.num_envs * len(native_names)
                    and len(set(target_rows)) == len(target_rows))
        collision_paths = ((collision_path,) if isinstance(collision_path, str) else tuple(collision_path))
        actual_filters = tuple(tuple(paths) for paths in self.view.filter_paths)
        if (not complete or len(expected) != self.num_envs * len(names)
                or tuple(self.view.sensor_paths) != expected or self.view.sensor_count != len(expected)
                or not collision_paths or len(set(collision_paths)) != len(collision_paths)
                or self.view.filter_count != len(collision_paths)
                or len(actual_filters) != len(expected)
                or any(len(paths) != len(collision_paths) or set(paths) != set(collision_paths)
                       for paths in actual_filters)
                or set(names) != set(binding.native_body_names)):
            details = {
                'body_names': names, 'native_body_names': binding.native_body_names,
                'owned_body_rows': owned, 'complete_body_rows': complete,
                'body_path_count': len(expected),
                'expected_body_path_count': self.num_envs * len(names),
                'sensor_count': self.view.sensor_count, 'filter_count': self.view.filter_count,
                'expected_filter_count': len(collision_paths),
                'filter_row_count': len(actual_filters),
                'body_path_samples': expected[:2] + expected[-2:],
                'sensor_path_samples': tuple(self.view.sensor_paths[:2]) + tuple(self.view.sensor_paths[-2:]),
                'filter_path_samples': actual_filters[:2],
                'expected_filter_paths': collision_paths,
            }
            raise ValueError('Lab patch loads require exact robot bodies and every static terrain partition: '
                             + json.dumps(details, separators=(',', ':')))
        self.surface_count = len(collision_paths)
        # Compile once from actual paths: GA raw views are body-major, whereas
        # the public sensor data and older raw views may be environment-major.
        self.sensor_to_native = torch.tensor(target_rows, dtype=torch.long, device=self.device)
        self.wheels = torch.tensor(binding.wheel_body_indices, device=self.device)
        self.normal = torch.zeros(self.num_envs, len(native_names), device=self.device)
        self.upward = torch.zeros_like(self.normal)
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tick = None
        self.capacity = self.view.max_contact_data_count
        self.receipt = {'source': 'native_contact_patch_normal_forces', 'filters': actual_filters[0],
                        'sensor_bodies': names, 'native_bodies': binding.native_body_names,
                        'max_contact_data_count': self.capacity,
                        'sensor_row_binding': 'actual_paths_to_environment_and_native_body',
                        'wheel_support': 'sum_positive_world_z_components_per_patch',
                        'slip_available': False}

    @staticmethod
    def _tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        tensor = getattr(value, 'torch', None)
        if isinstance(tensor, torch.Tensor):
            return tensor
        import warp as wp
        return wp.to_torch(value)

    @staticmethod
    def reduce(force, normals, counts, starts, capacity):
        layout = NativeTerrainContacts._patch_layout(force, counts, starts, capacity)
        normal, _ = NativeTerrainContacts._loads_from_layout(layout)
        if normals.shape != (force.numel(), 3) or normals.device != force.device or normals.dtype != torch.float32:
            raise ValueError('native normal directions differ from the shared force allocation')
        direction = normals[layout.indices]
        if (not torch.isfinite(direction).all() or (direction.norm(dim=-1) - 1).abs().gt(1e-3).any()):
            raise ValueError('active native patch normals must be finite unit world vectors')
        sensor = layout.pairs // counts.shape[1]
        upward = torch.zeros_like(normal).scatter_add_(
            0, sensor, layout.force[layout.indices] * direction[:, 2].clamp_min(0))
        return normal, upward

    def reset_rows(self, ids, physical_tick):
        # A cold/partial reset has no new force report. Publish no support for
        # those reset rows until their first real physical substep.
        self.normal[ids] = 0
        self.upward[ids] = 0
        self.valid[ids] = False
        self._tick = physical_tick

    @torch.no_grad()
    def read(self, physical_tick, physics_dt_s):
        if physical_tick != self._tick:
            packet = self.view.get_contact_data(dt=physics_dt_s)
            if len(packet) != 6:
                raise ValueError('native patch packet layout differs')
            force, normals, counts, starts = (self._tensor(packet[i]) for i in (0, 2, 4, 5))
            if counts.shape != (self.view.sensor_count, self.surface_count):
                raise ValueError('native patch pairs differ from the bound shared terrain')
            normal, upward = self.reduce(force, normals, counts, starts, self.capacity)
            self.normal.view(-1).index_copy_(0, self.sensor_to_native, normal)
            self.upward.view(-1).index_copy_(0, self.sensor_to_native, upward)
            self.valid.fill_(True)
            self._tick = physical_tick
        return self.normal, self.upward.index_select(1, self.wheels)

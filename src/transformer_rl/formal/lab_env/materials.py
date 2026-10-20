"""Read wheel-body materials from the live PhysX articulation, never terrain."""
from __future__ import annotations

from copy import deepcopy

import torch


class WheelMaterialReader:
    """Cache [mu_s_L, mu_d_L, e_L, mu_s_R, mu_d_R, e_R] for the critic.

    The current Lab scene disables event-based material randomization. Read once
    at setup; a future material writer must call ``refresh`` after its write.
    A wheel with unlike shape materials cannot be represented by one triple.
    """

    def __init__(self, robot, wheel_body_indices, *, num_envs, device, native_to_tensor=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('wheel materials require a positive environment count')
        self.robot, self._view = robot, robot.root_view
        self.num_envs, self.device = num_envs, torch.device(device)
        names = tuple(robot.body_names)
        wheels = tuple(wheel_body_indices)
        if (len(wheels) != 2 or len(set(wheels)) != 2
                or any(type(index) is not int or not 0 <= index < len(names) for index in wheels)
                or len(set(names)) != len(names) or self._view.count != num_envs):
            raise ValueError('wheel material binding differs from the articulation rows or bodies')
        paths = self._view.link_paths
        if len(paths) != num_envs or tuple(path.rsplit('/', 1)[-1] for path in paths[0]) != names:
            raise ValueError('native link order differs from the wheel material binding')
        # Match IsaacLab's PhysX material adapter: articulation properties are
        # flattened by link, while each link may own several collision shapes.
        counts = tuple(int(robot._physics_sim_view.create_rigid_body_view(path).max_shapes)
                       for path in paths[0])
        if any(count < 0 for count in counts) or sum(counts) != self._view.max_shapes:
            raise ValueError('native link shape counts differ from the articulation material table')
        self._ranges = tuple((sum(counts[:index]), sum(counts[:index + 1])) for index in wheels)
        if any(start == end for start, end in self._ranges):
            raise ValueError('a bound wheel has no native collision shape material')
        if native_to_tensor is None:
            import warp as wp

            native_to_tensor = wp.to_torch
        if not callable(native_to_tensor):
            raise TypeError('native material conversion must be callable')
        self._convert = native_to_tensor
        self._values = torch.empty(num_envs, 6, dtype=torch.float32, device=self.device)
        self._receipt = {
            'source': 'PhysX ArticulationView.get_material_properties',
            'wheel_body_names': [names[index] for index in wheels],
            'wheel_shape_ranges': [list(interval) for interval in self._ranges],
            'shape_counts_by_link': list(counts),
            'parameter_order': ['static_friction', 'dynamic_friction', 'restitution'],
            'uniform_within_each_wheel': True,
            'refresh_policy': 'setup_and_after_explicit_material_writes',
            'terrain_material_substitution': False,
        }
        self.refresh()

    @torch.no_grad()
    def refresh(self, ids=None):
        """Publish selected rows only after their full source values validate."""
        if self.robot.root_view is not self._view:
            raise RuntimeError('wheel material articulation view changed; rebuild the scene binding')
        values = self._convert(self._view.get_material_properties())
        if (not isinstance(values, torch.Tensor) or not values.dtype.is_floating_point
                or values.shape != (self.num_envs, self._view.max_shapes, 3)):
            raise ValueError('native material buffer must be floating [environment, shape, 3]')
        if ids is None:
            rows = torch.arange(self.num_envs, device=values.device)
        else:
            rows = torch.as_tensor(ids, device=values.device)
            if (rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64)
                    or (rows < 0).any() or (rows >= self.num_envs).any()
                    or rows.unique().numel() != rows.numel()):
                raise ValueError('material refresh requires unique valid environment rows')
            rows = rows.long()
        if not rows.numel():
            return
        selected = values.index_select(0, rows)
        wheels = []
        for start, end in self._ranges:
            shapes = selected[:, start:end]
            if (not torch.isfinite(shapes).all() or (shapes < 0).any()
                    or (shapes[..., 2] > 1).any()):
                raise ValueError('wheel shape material contains invalid physical parameters')
            triple = shapes[:, 0]
            if not torch.equal(shapes, triple[:, None].expand_as(shapes)):
                raise ValueError('one wheel has different shape materials; no single critic triple exists')
            wheels.append(triple)
        self._values.index_copy_(0, rows.to(self.device),
                                 torch.cat(wheels, -1).to(device=self.device, dtype=torch.float32))

    def features(self):
        """Return the cached critic features without another native read."""
        return self._values

    @property
    def receipt(self):
        return deepcopy(self._receipt)

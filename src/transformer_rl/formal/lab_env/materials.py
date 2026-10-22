"""Cache wheel-shape materials and optionally randomize them once at startup."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import math

import torch


@dataclass(frozen=True)
class WheelMaterialConfig:
    static_friction: tuple[float, float] = (.5, 1.2)
    dynamic_friction: tuple[float, float] = (.4, 1.)
    restitution: tuple[float, float] = (.02, .2)
    num_buckets: int = 64
    anchor_lower_endpoint: bool = False

    def __post_init__(self):
        for name in ('static_friction', 'dynamic_friction', 'restitution'):
            bounds = tuple(getattr(self, name))
            if (len(bounds) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
                    or not 0 <= bounds[0] <= bounds[1]
                    or name == 'restitution' and bounds[1] > 1):
                raise ValueError('wheel material bounds must be finite nonnegative physical coefficients')
            object.__setattr__(self, name, bounds)
        if (self.static_friction[0] < self.dynamic_friction[0]
                or type(self.num_buckets) is not int or not 1 <= self.num_buckets <= 256
                or type(self.anchor_lower_endpoint) is not bool):
            raise ValueError('wheel buckets require dynamic<=static and an explicit bounded palette')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError('wheel material configuration fields differ')
        return cls(**value)


class WheelMaterialReader:
    """Cache [mu_s_L, mu_d_L, e_L, mu_s_R, mu_d_R, e_R] for the critic.

    Startup writes use a bounded shared palette and verify the whole native
    table before publishing new critic features. Terrain and non-wheel shapes
    retain their original materials. No material sampling occurs on reset.
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
        self._startup_applied = False
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
    def randomize_startup(self, configuration, *, seed, tensor_to_native=None):
        """Write once, then verify every shape; cold setup repeats a private draw."""
        if (not isinstance(configuration, WheelMaterialConfig)
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError('startup wheel materials require parsed bounds and an independent seed')
        if self._startup_applied:
            raise RuntimeError('startup wheel material write was already attempted')
        if self.robot.root_view is not self._view:
            raise RuntimeError('wheel material articulation view changed; rebuild the scene binding')
        if not callable(getattr(self._view, 'set_material_properties', None)):
            raise ValueError('native wheel material setter is unavailable')
        if tensor_to_native is None:
            import warp as wp

            tensor_to_native = lambda value: wp.from_torch(
                value, dtype=wp.int32 if value.dtype == torch.int32 else wp.float32)
        if not callable(tensor_to_native):
            raise TypeError('native material conversion must be callable')
        source = self._read_native()
        if source.dtype != torch.float32 or not torch.isfinite(source).all():
            raise ValueError('startup materials require a finite native float32 table')
        source = source.detach().cpu().clone()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        bounds = (configuration.static_friction, configuration.dynamic_friction, configuration.restitution)
        low = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device='cpu')
        high = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device='cpu')
        palette = low + (high - low) * torch.rand(
            configuration.num_buckets, 3, generator=generator, dtype=torch.float32, device='cpu')
        palette[:, 1] = torch.minimum(palette[:, 0], palette[:, 1])
        if configuration.anchor_lower_endpoint:
            palette[0] = low
        bucket_ids = torch.randint(configuration.num_buckets, (self.num_envs, 2),
                                   generator=generator, device='cpu')
        desired = palette[bucket_ids]
        target = source.clone()
        for wheel, (start, end) in enumerate(self._ranges):
            target[:, start:end] = desired[:, wheel:wheel + 1]
        indices = torch.arange(self.num_envs, dtype=torch.int32, device='cpu')
        # A failed native write is not retried in this scene: its outcome may
        # be partial, so the caller must abort before collecting transitions.
        self._startup_applied = True
        self._view.set_material_properties(tensor_to_native(target.contiguous()), tensor_to_native(indices))
        actual = self._read_native().detach().cpu()
        if not torch.equal(actual, target):
            raise RuntimeError('native material readback differs from the requested full shape table')
        self._publish(actual)
        self._receipt['startup_randomization'] = {
            'status': 'native_readback_matched', 'seed': seed, 'configuration': asdict(configuration),
            'native_parameters_sha256': hashlib.sha256(desired.contiguous().numpy().tobytes()).hexdigest(),
            'palette_sha256': hashlib.sha256(palette.numpy().tobytes()).hexdigest(),
            'bucket_histogram_by_wheel': [torch.bincount(bucket_ids[:, wheel], minlength=configuration.num_buckets).tolist()
                                         for wheel in range(2)],
            'actual_min': desired.flatten(0, 1).amin(0).tolist(),
            'actual_max': desired.flatten(0, 1).amax(0).tolist(),
            'non_wheel_shapes_unchanged': True, 'terrain_materials_changed': False,
            'reset_resampling': False, 'effective_contact_coefficient': False,
            'distribution_evidence': 'engineering prior; contact pair combine modes remain unchanged',
        }

    def _read_native(self):
        values = self._convert(self._view.get_material_properties())
        if (not isinstance(values, torch.Tensor) or not values.dtype.is_floating_point
                or values.shape != (self.num_envs, self._view.max_shapes, 3)):
            raise ValueError('native material buffer must be floating [environment, shape, 3]')
        return values

    @torch.no_grad()
    def refresh(self, ids=None):
        """Publish selected rows only after their full source values validate."""
        if self.robot.root_view is not self._view:
            raise RuntimeError('wheel material articulation view changed; rebuild the scene binding')
        self._publish(self._read_native(), ids)

    def _publish(self, values, ids=None):
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

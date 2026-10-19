"""Private continuous course construction, independent of SDK and policy code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct

import torch

from ._terrain_geometry import HeightPatch, TerrainKind, TerrainMesh, TerrainSurface


@dataclass(frozen=True)
class TerrainExtent:
    x_m: tuple[float, float]
    y_m: tuple[float, float]

    def __post_init__(self):
        for name in ('x_m', 'y_m'):
            bounds = tuple(getattr(self, name))
            if (len(bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                    or not bounds[0] < 0 < bounds[1]):
                raise ValueError('terrain extents must explicitly surround the spawn origin')
            object.__setattr__(self, name, bounds)

    @classmethod
    def for_horizon(cls, duration_s, speed_bound_m_s, margin_m):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (duration_s, margin_m)) or type(speed_bound_m_s) not in (int, float) or not math.isfinite(speed_bound_m_s) or speed_bound_m_s < 0:
            raise ValueError('terrain horizon requires finite duration, speed bound and explicit positive margin')
        half = duration_s * speed_bound_m_s + margin_m
        return cls((-half, half), (-half, half))


@dataclass(frozen=True)
class TerrainTuning:
    nominal_friction: float
    material_friction_bounds: tuple[float, float]
    rough_grid_spacing_m: float
    entry_m: float = 1.5
    transition_m: float = .5
    runout_m: float = 1.
    stair_count: int = 4
    stair_tread_m: float = .3
    platform_tread_m: float = .6

    def __post_init__(self):
        values = (self.nominal_friction, *self.material_friction_bounds, self.rough_grid_spacing_m,
                  self.entry_m, self.transition_m, self.runout_m, self.stair_tread_m, self.platform_tread_m)
        if (len(self.material_friction_bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or self.nominal_friction < 0 or not 0 <= self.material_friction_bounds[0] < self.material_friction_bounds[1]
                or min(values[3:]) <= 0 or type(self.stair_count) is not int or self.stair_count < 2):
            raise ValueError('terrain tuning requires explicit material and positive geometric dimensions')
        object.__setattr__(self, 'material_friction_bounds', tuple(self.material_friction_bounds))


@dataclass(frozen=True)
class TerrainRequest:
    kind: TerrainKind
    extent: TerrainExtent
    direction: int = 1
    slope_rad: float = 0.
    slope_bearing_rad: float = 0.
    roughness_m: float = 0.
    step_height_m: float = 0.
    platform_tops_m: tuple[float, float] = (.20, .25)
    material_friction: tuple[float, float] = (.8, .8)

    def __post_init__(self):
        if (not isinstance(self.kind, TerrainKind) or not isinstance(self.extent, TerrainExtent)
                or type(self.direction) is not int or self.direction not in (-1, 1)):
            raise ValueError('terrain request needs a named kind, explicit extent and forward/reverse direction')
        values = (self.slope_rad, self.slope_bearing_rad, self.roughness_m, self.step_height_m,
                  *self.platform_tops_m, *self.material_friction)
        if (len(self.platform_tops_m) != 2 or len(self.material_friction) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or abs(self.slope_rad) >= math.pi / 2 or self.roughness_m < 0
                or not 0 <= self.platform_tops_m[0] <= self.platform_tops_m[1]
                or min(self.material_friction) < 0):
            raise ValueError('terrain request dimensions and material values are invalid')
        object.__setattr__(self, 'platform_tops_m', tuple(self.platform_tops_m))
        object.__setattr__(self, 'material_friction', tuple(self.material_friction))


@dataclass(frozen=True)
class TerrainCourse:
    request: TerrainRequest
    geometry_seed: int
    surface: TerrainSurface


class ContinuousTerrainGenerator:
    """Geometry RNG and parameters are checkpointed, never global torch RNG.

    Construction happens on episode/scene preparation boundaries, not each
    policy tick. Extents are caller supplied and never silently shortened to
    fit memory. A resource owner must measure the complete authored artifacts.
    """

    def __init__(self, seed, tuning: TerrainTuning):
        if type(seed) is not int or not 0 <= seed < 2**63 or not isinstance(tuning, TerrainTuning):
            raise ValueError('terrain generator requires a nonnegative seed and complete tuning')
        self.tuning = tuning
        self.generator = torch.Generator(device='cpu').manual_seed(seed)

    def _uniform(self):
        return torch.rand((), generator=self.generator).item()

    @staticmethod
    def _axis(low, high, spacing):
        count = max(1, math.ceil((high - low) / spacing))
        return tuple(low + (high - low) * i / count for i in range(count + 1))

    @staticmethod
    def _patch(request, s, y, heights, friction):
        # Transform the entire geometry for reverse travel, including risers.
        if request.direction < 0:
            return HeightPatch(tuple(-v for v in reversed(s)), y, tuple(reversed(heights)), friction)
        return HeightPatch(s, y, tuple(heights), friction)

    def _platforms(self, request, levels, breaks):
        low, high = (request.extent.x_m if request.direction > 0
                     else (-request.extent.x_m[1], -request.extent.x_m[0]))
        points = (low, *breaks, high)
        if any(b <= a for a, b in zip(points, points[1:])) or len(levels) != len(points) - 1:
            raise ValueError('declared terrain extent does not contain the complete requested platforms and runout')
        if high - breaks[-1] < self.tuning.runout_m:
            raise ValueError('platform course requires its full declared runout')
        materials = tuple(request.material_friction[i % 2] if request.kind == TerrainKind.MATERIAL
                          else self.tuning.nominal_friction for i in range(len(levels)))
        patches = [self._patch(request, (a, b), request.extent.y_m, ((h, h), (h, h)), mu)
                   for a, b, h, mu in zip(points, points[1:], levels, materials)]
        walls = []
        for index, s in enumerate(breaks):
            before, after = levels[index], levels[index + 1]
            if before == after:
                continue
            # Reuse rounded patch edges so vertical faces do not open a seam.
            x = patches[index].x_m[-1 if request.direction > 0 else 0]
            y0, y1 = patches[index].y_m
            vertices = ((x, y0, before), (x, y1, before), (x, y1, after), (x, y0, after))
            walls.append(TerrainMesh(vertices, ((0, 1, 2), (0, 2, 3)), self.tuning.nominal_friction))
        return TerrainSurface(tuple(patches), tuple(walls))

    def build(self, request: TerrainRequest, *, geometry_seed=None):
        if not isinstance(request, TerrainRequest):
            raise ValueError('course construction requires a complete named request')
        if geometry_seed is None:
            geometry_seed = torch.randint(2**63 - 1, (), generator=self.generator).item()
        if type(geometry_seed) is not int or not 0 <= geometry_seed < 2**63:
            raise ValueError('geometry seed is outside its declared domain')
        t, kind = self.tuning, request.kind
        lo, hi = request.extent.x_m if request.direction > 0 else (-request.extent.x_m[1], -request.extent.x_m[0])
        mu = t.nominal_friction
        if kind == TerrainKind.FLAT:
            surface = TerrainSurface((HeightPatch(request.extent.x_m, request.extent.y_m, ((0., 0.), (0., 0.)), mu),))
        elif kind in (TerrainKind.STEP_UP, TerrainKind.STEP_DOWN):
            h = abs(request.step_height_m) * (1 if kind == TerrainKind.STEP_UP else -1)
            surface = self._platforms(request, (0., h), (t.entry_m,))
        elif kind == TerrainKind.STAIRS:
            levels = tuple(i * request.step_height_m for i in range(t.stair_count + 1))
            breaks = tuple(t.entry_m + i * t.stair_tread_m for i in range(t.stair_count))
            if hi - breaks[-1] < t.runout_m:
                raise ValueError('staircase needs the full declared final runout')
            surface = self._platforms(request, levels, breaks)
        elif kind == TerrainKind.TWO_PLATFORMS:
            if hi - (t.entry_m + 2 * t.platform_tread_m) < t.runout_m:
                raise ValueError('two platforms require both complete treads and the declared final runout')
            surface = self._platforms(request, (0., *request.platform_tops_m),
                                      (t.entry_m, t.entry_m + t.platform_tread_m))
        elif kind == TerrainKind.MATERIAL:
            middle = (t.entry_m + hi - t.runout_m) / 2
            surface = self._platforms(request, (0., 0., 0.), (t.entry_m, middle))
        else:
            end = hi - t.runout_m - t.transition_m
            if end <= t.entry_m + 2 * t.transition_m:
                raise ValueError('slopes and roughness require complete entry, active terrain and runout')
            if kind == TerrainKind.SLOPE:
                s = (lo, t.entry_m, t.entry_m + t.transition_m, end, end + t.transition_m, hi)
                y = request.extent.y_m
                along = math.tan(request.slope_rad) * math.cos(request.slope_bearing_rad)
                across = math.tan(request.slope_rad) * math.sin(request.slope_bearing_rad)
                z = tuple(tuple(along * min(max(x - t.entry_m, 0), end - t.entry_m)
                                + across * b * min(max((x - t.entry_m) / t.transition_m, 0), 1)
                                * min(max((end + t.transition_m - x) / t.transition_m, 0), 1) for b in y) for x in s)
            else:
                raw_s = (lo, t.entry_m, t.entry_m + t.transition_m, end, end + t.transition_m, hi,
                         *self._axis(t.entry_m, end + t.transition_m, t.rough_grid_spacing_m))
                s = tuple(sorted({struct.unpack('<f', struct.pack('<f', v))[0] for v in raw_s}))
                y = self._axis(*request.extent.y_m, t.rough_grid_spacing_m)
                rng = torch.Generator(device='cpu').manual_seed(geometry_seed)
                noise = torch.rand((len(s), len(y)), generator=rng) * 2 - 1
                core = [i for i, x in enumerate(s) if t.entry_m + t.transition_m <= x <= end]
                if len(core) >= 2:
                    noise[core[0], len(y) // 2], noise[core[1], len(y) // 2] = 1., -1.
                z = tuple(tuple(float(noise[i, j]) * request.roughness_m
                                * min(max((x - t.entry_m) / t.transition_m, 0), 1)
                                * min(max((end + t.transition_m - x) / t.transition_m, 0), 1)
                                for j in range(len(y))) for i, x in enumerate(s))
            surface = TerrainSurface((self._patch(request, s, y, z, mu),))
        return TerrainCourse(request, geometry_seed, surface)

    def from_draw(self, draw, extents):
        from ._task_curriculum import TaskDraw

        if not isinstance(draw, TaskDraw) or len(extents) != len(draw.family):
            raise ValueError('terrain reset adapter requires the exact task rows and explicit per-row extents')
        rows = {name: getattr(draw, name).detach().cpu() for name in
                ('terrain_kind', 'requested_command', 'slope_rad', 'roughness_m', 'step_up_m', 'step_down_m')}
        courses = []
        for index, extent in enumerate(extents):
            kind = TerrainKind(int(rows['terrain_kind'][index]))
            bearing = (0. if self._uniform() < .5 else math.pi / 2) if self._uniform() < .5 else self._uniform() * 2 * math.pi
            ascending = self._uniform() < .5
            step = float(rows['step_up_m'][index] if ascending else -rows['step_down_m'][index])
            if kind == TerrainKind.STEP_UP:
                step = float(rows['step_up_m'][index])
            elif kind == TerrainKind.STEP_DOWN:
                step = -float(rows['step_down_m'][index])
            upper = float(rows['step_up_m'][index])
            fraction = .8 if self._uniform() < .5 else self._uniform()
            low_mu, high_mu = self.tuning.material_friction_bounds
            materials = tuple(low_mu + (high_mu - low_mu) * self._uniform() for _ in range(2))
            request = TerrainRequest(kind, extent, -1 if rows['requested_command'][index, 0] < 0 else 1,
                                     float(rows['slope_rad'][index]), bearing, float(rows['roughness_m'][index]),
                                     step, (upper * fraction, upper), materials)
            courses.append(self.build(request))
        return tuple(courses)

    def from_fixed_case(self, case, extent):
        """Resolve terrain only; this does not implement the case's skill/success."""
        if not isinstance(case, dict) or case.get('family') != 'terrain' or not isinstance(case.get('parameters'), dict):
            raise ValueError('fixed terrain adapter needs an explicit terrain-family case')
        p = case['parameters']
        name = p['terrain']
        direction = -1 if p['vx_m_s'] < 0 else 1
        base = {'extent': extent, 'direction': direction}
        if name in ('slope_up', 'slope_down', 'cross_slope'):
            request = TerrainRequest(TerrainKind.SLOPE, slope_rad=math.radians(p['angle_deg']),
                                     slope_bearing_rad=math.pi / 2 if name == 'cross_slope' else 0., **base)
        elif name == 'rough':
            request = TerrainRequest(TerrainKind.ROUGH, roughness_m=p['amplitude_m'], **base)
        elif name in ('step_up', 'step_down'):
            request = TerrainRequest(TerrainKind.STEP_UP if name == 'step_up' else TerrainKind.STEP_DOWN,
                                     step_height_m=p['rise_m'] if name == 'step_up' else -p['drop_m'], **base)
        elif name == 'stairs':
            if p['tread_m'] != self.tuning.stair_tread_m:
                raise ValueError('fixed staircase tread differs from the frozen generator tuning')
            request = TerrainRequest(TerrainKind.STAIRS, step_height_m=p['rise_m'], **base)
        elif name == 'two_platforms':
            a, b = p['absolute_top_heights_m']
            if (p['tread_m'] != self.tuning.platform_tread_m
                    or not math.isclose(b - a, p['second_rise_m'], rel_tol=0, abs_tol=1e-12)):
                raise ValueError('fixed platform heights/rise/tread are inconsistent')
            request = TerrainRequest(TerrainKind.TWO_PLATFORMS, platform_tops_m=(a, b), **base)
        else:
            raise ValueError('fixed terrain case has no authored geometry: ' + str(name))
        return self.build(request)

    def state_dict(self):
        return {'tuning': asdict(self.tuning), 'rng': self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.keys() != {'tuning', 'rng'} or state['tuning'] != asdict(self.tuning):
            raise ValueError('terrain generator checkpoint tuning differs')
        candidate = torch.Generator(device='cpu')
        candidate.set_state(state['rng'].detach().cpu().clone())
        self.generator = candidate

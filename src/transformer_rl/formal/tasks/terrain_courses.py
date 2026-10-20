"""Immutable terrain atlas and reset-only difficulty routing.

Geometry is built before simulation. Runtime changes episode origins and level
indices only; it never recooks USD, rebuilds a BVH, or edits nearby geometry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .geometry import TerrainKind
from .terrain import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning
from .terrain_assets import file_pin, write_collision_usda
from .terrain_query import build_query_index


def surface_height(vertices, faces, xy):
    """Finite vertical intersections against authored triangles, for offline QA."""
    tri = np.asarray(vertices, float)[np.asarray(faces)]
    a, u, v = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    denominator = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    answer = []
    for point in np.asarray(xy):
        dx, dy = (point - a[:, :2]).T
        safe = np.where(np.abs(denominator) > 1e-15, denominator, np.nan)
        s = (dx * v[:, 1] - dy * v[:, 0]) / safe
        t = (u[:, 0] * dy - u[:, 1] * dx) / safe
        valid = (s >= -1e-7) & (t >= -1e-7) & (s + t <= 1 + 1e-7)
        heights = a[:, 2] + s * u[:, 2] + t * v[:, 2]
        answer.append(np.max(heights[valid]) if valid.any() else np.nan)
    return np.asarray(answer)


def _rectangle(x0, x1, y0, y1, z0, z1):
    return (np.asarray(((x0, y0, z0), (x1, y0, z1), (x1, y1, z1), (x0, y1, z0)), np.float32),
            np.asarray(((0, 1, 2), (0, 2, 3)), np.int64))


def _asset_geometry(root, name):
    record = json.loads((root / 'manifest.json').read_text())['assets'][name]
    material = record['material']
    if material['combine_mode'] != 'multiply':
        raise ValueError('prepared terrain material combine mode is unsupported')
    path = root / name / 'collision.npz'
    if file_pin(path) != record['files']['collision.npz']:
        raise ValueError('prepared terrain geometry identity differs')
    with np.load(path, allow_pickle=False) as source:
        vertices, faces = source['vertices_m'].copy(), source['triangles'].copy()
    if record['specification']['approach_axis'] == '+Y':
        vertices = vertices[:, [1, 0, 2]] * np.asarray((1, -1, 1), np.float32)
    low, high = vertices[:, :2].min(0), vertices[:, :2].max(0)
    # Padding is outside the ROI only. The original underpanel gap is untouched.
    epsilon = .002
    ends = surface_height(vertices, faces, ((low[0] + epsilon, 0), (high[0] - epsilon, 0)))
    if not np.isfinite(ends).all():
        raise ValueError('asset approach/runout has no source ground support')
    parts = [(vertices, faces), _rectangle(-2, low[0], low[1], high[1], ends[0], ends[0]),
             _rectangle(high[0], 6, low[1], high[1], ends[1], ends[1]),
             _rectangle(-2, 6, -4, low[1], ends[0], ends[1]),
             _rectangle(-2, 6, high[1], 4, ends[0], ends[1])]
    return parts, {
        'asset': name, 'source_manifest_sha256': file_pin(root / 'manifest.json')['sha256'],
        'source_roi_m': [low.tolist(), high.tolist()],
        'training_padding_only': True, 'valid_half_width_m': float(min(abs(low[1]), high[1]) - .15),
        'spawn_z_m': float(ends[0]),
        'physical_material': material,
    }


def build_atlas(prepared_assets, output, *, seed=2102, levels=10):
    """Freeze a seeded continuous draw within each difficulty bin, plus exact maxima.

    Requested task draws and the selected atlas parameters remain separate. The
    finite atlas is not described as a new continuously recooked per-episode mesh.
    """
    prepared_assets, output = Path(prepared_assets), Path(output)
    if type(levels) is not int or not 2 <= levels <= 32 or type(seed) is not int or seed < 0:
        raise ValueError('atlas requires 2..32 levels and a nonnegative geometry seed')
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    tuning = TerrainTuning(.7, (.2, 1.1), .25, platform_tread_m=.8, stair_tread_m=.8)
    generator = ContinuousTerrainGenerator(seed, tuning)
    vertices, faces, materials, face_tiles, tiles = [], [], [], [], []
    vertex_count = 0

    def append(parts, material_values, tile):
        nonlocal vertex_count
        origin = np.asarray(tile['origin_m'], np.float32)
        for (points, triangles), material in zip(parts, material_values):
            points = np.asarray(points, np.float32) + origin
            triangles = np.asarray(triangles, np.int64)
            vertices.append(points)
            faces.append(triangles + vertex_count)
            vertex_count += len(points)
            materials.extend([material] * len(triangles))
            face_tiles.extend([len(tiles)] * len(triangles))
        tiles.append(tile)

    append([_rectangle(-90, 90, -90, 90, 0, 0)], [(.7, .7, 0.)], {
        'name': 'flat', 'kind': 0, 'level': 0, 'origin_m': [-200., 0., 0.],
        'spawn_z_m': 0., 'valid_half_width_m': 88., 'finish_x_m': 88., 'start_x_m': -88.,
        'parameters': {'friction': .7},
    })
    for kind in list(TerrainKind)[1:]:
        for level in range(levels):
            # Keep endpoints and sample continuous values inside the other bins.
            fraction = 1. if level == levels - 1 else (level + rng.uniform(.05, .95)) / levels
            sign = -1 if level % 2 else 1
            request = TerrainRequest(
                kind, TerrainExtent((-2., 6.), (-4., 4.)),
                slope_rad=sign * math.pi / 12 * fraction,
                slope_bearing_rad=(level % 4) * math.pi / 2,
                roughness_m=.02 * fraction,
                step_height_m=(-.15 if kind == TerrainKind.STEP_DOWN else .20) * fraction,
                platform_tops_m=(.20 * fraction, .40 * fraction),
                material_friction=(.7 - .5 * fraction, .7 + .4 * fraction),
            )
            surface = generator.build(request, geometry_seed=seed * 1000 + int(kind) * 32 + level).surface
            tile_index = len(tiles) - 1
            origin = [float(tile_index % 8 * 12), float(tile_index // 8 * 12), 0.]
            append([(mesh.vertices_m, mesh.triangles) for mesh in surface.meshes],
                   [(mesh.friction, mesh.friction, 0.) for mesh in surface.meshes], {
                       'name': kind.name.lower(), 'kind': int(kind), 'level': level, 'origin_m': origin,
                       'spawn_z_m': 0., 'valid_half_width_m': 3., 'finish_x_m': 5., 'start_x_m': -1.5,
                       'parameters': asdict(request),
                   })
    for name, kind in (('road_step', TerrainKind.STEP_UP), ('central_steps', TerrainKind.TWO_PLATFORMS)):
        parts, metadata = _asset_geometry(prepared_assets, name)
        index = len(tiles) - 1
        material = metadata['physical_material']
        parameters = tuple(material[key] for key in ('static_friction', 'dynamic_friction', 'restitution'))
        append(parts, [parameters] * len(parts), {
            'name': name, 'kind': int(kind), 'level': levels - 1,
            'origin_m': [float(index % 8 * 12), float(index // 8 * 12), 0.],
            'finish_x_m': 5., 'start_x_m': -1.5, 'parameters': metadata, **metadata,
        })
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    # Use one physical/scanning mesh with per-face material subsets.
    parameters, material_ids = np.unique(np.asarray(materials, dtype=float), axis=0, return_inverse=True)
    write_collision_usda(output / 'terrain.usda', vertices, faces, static_friction=.7,
                         dynamic_friction=.7, face_material=material_ids, material_parameters=parameters)
    np.savez_compressed(output / 'geometry.npz', vertices_m=vertices, triangles=faces,
                        face_material=material_ids, material_coefficients=parameters[:, 0],
                        material_parameters=parameters,
                        face_tile=np.asarray(face_tiles, np.int32))
    np.savez_compressed(output / 'query_index.npz',
                        **build_query_index(vertices, faces, face_tiles, len(tiles)))
    manifest = {'format': 'terrain_course_atlas_v1', 'seed': seed, 'levels': levels,
                'source_assets': file_pin(prepared_assets / 'manifest.json'), 'tuning': asdict(tuning),
                'tiles': tiles, 'files': {p.name: file_pin(p) for p in sorted(output.iterdir())},
                'collision_mesh': '/Terrain/Collision', 'robot_qualified': False,
                'geometry_sampling': 'seeded_continuous_bins_with_exact_upper_endpoints',
                'material_parameter_order': ['static_friction', 'dynamic_friction', 'restitution'],
                'material_combine_mode': 'multiply',
                'material_coefficients_semantics': 'static_friction_only',
                'material_policy': {'authored': 'source_manifest_values',
                                    'procedural': 'static_equals_dynamic_zero_restitution'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


class TerrainCourseRouter:
    """GPU row state. Difficulty changes only on reset from completed episodes."""

    def __init__(self, manifest_path, num_envs, *, seed=2102, device='cpu'):
        path = Path(manifest_path)
        raw = path.read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest['format'] != 'terrain_course_atlas_v1':
            raise ValueError('unsupported terrain atlas format')
        for name, expected in self.manifest['files'].items():
            if file_pin(path.parent / name) != expected:
                raise ValueError('terrain atlas payload identity differs')
        self.num_envs, self.device = num_envs, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        tiles = self.manifest['tiles']
        self.origins_table = torch.tensor([t['origin_m'] for t in tiles], device=self.device)
        self.spawn_z = torch.tensor([t['spawn_z_m'] for t in tiles], device=self.device)
        self.kinds = torch.tensor([t['kind'] for t in tiles], device=self.device)
        self.tile_levels = torch.tensor([t['level'] for t in tiles], device=self.device)
        self.half_widths = torch.tensor([t['valid_half_width_m'] for t in tiles], device=self.device)
        self.finishes = torch.tensor([t['finish_x_m'] for t in tiles], device=self.device)
        self.starts = torch.tensor([t['start_x_m'] for t in tiles], device=self.device)
        self.level = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.tile = torch.zeros_like(self.level)
        self.origins = self.origins_table[0].expand(num_envs, -1).clone()
        self.started = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.direction = torch.ones(num_envs, device=self.device)
        self.last_success = torch.zeros_like(self.started)
        self.last_failure = torch.zeros_like(self.started)
        self.course_exits = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def observe_end(self, position, terminated):
        local = position - self.origins
        progress = local[:, 0] * self.direction
        outside = ((local[:, 1].abs() > self.half_widths[self.tile])
                   | (local[:, 0] > self.finishes[self.tile]) | (local[:, 0] < self.starts[self.tile]))
        nonflat = self.kinds[self.tile] != 0
        self.last_success.copy_(nonflat & ~terminated & (progress > 4.5))
        self.last_failure.copy_(nonflat & terminated)
        exits = nonflat & outside & ~terminated
        self.course_exits.add_(exits.long())
        return exits

    def reset(self, ids, kind, *, stage, command_forward):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        kind = torch.as_tensor(kind, device=self.device, dtype=torch.long)
        if kind.shape != ids.shape or ((kind < 0) | (kind > 7)).any():
            raise ValueError('terrain reset requires one valid kind per environment')
        delta = self.last_success[ids].long() - self.last_failure[ids].long()
        cap = (0 if stage == 0 else min(6, self.manifest['levels'] - 1)
               if stage == 1 else self.manifest['levels'] - 1)
        self.level[ids] = (self.level[ids] + delta).clamp(0, cap)
        # A full support sample avoids silently excluding hard terrain from
        # long-lived low-level rows; it is exposure, never an evaluation gate.
        explore = torch.rand(len(ids), generator=self.generator, device=self.device) < .15
        sampled = torch.randint(cap + 1, (len(ids),), generator=self.generator, device=self.device)
        level = torch.where(explore, sampled, self.level[ids])
        scores = torch.rand((len(ids), len(self.kinds)), generator=self.generator, device=self.device)
        candidates = ((kind[:, None] == self.kinds) & (level[:, None] == self.tile_levels))
        candidates[kind == 0] = False
        candidates[kind == 0, 0] = True
        if not candidates.any(-1).all():
            raise ValueError('atlas lacks a requested terrain kind/level')
        selected = scores.masked_fill(~candidates, -1).argmax(-1)
        self.tile[ids] = selected
        self.origins[ids] = self.origins_table[selected]
        self.direction[ids] = 1.0
        self.started[ids] = True
        self.last_success[ids] = False
        self.last_failure[ids] = False
        return self.origins[ids], self.spawn_z[selected]

    def state_dict(self):
        return {'format': 'terrain_course_router_v1', 'atlas_sha256': self.identity,
                'level': self.level.clone(), 'tile': self.tile.clone(), 'direction': self.direction.clone(),
                'course_exits': self.course_exits.clone(), 'rng': self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if (state.get('format') != 'terrain_course_router_v1' or state.get('atlas_sha256') != self.identity
                or state['level'].shape != self.level.shape or state['tile'].shape != self.tile.shape
                or ((state['level'] < 0) | (state['level'] >= self.manifest['levels'])).any()
                or ((state['tile'] < 0) | (state['tile'] >= len(self.kinds))).any()):
            raise ValueError('terrain router checkpoint identity or row state differs')
        for name in ('level', 'tile', 'direction', 'course_exits'):
            getattr(self, name).copy_(state[name])
        self.origins.copy_(self.origins_table[self.tile])
        self.generator.set_state(state['rng'].cpu())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build a static training terrain atlas without a simulator')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2102)
    parser.add_argument('--levels', type=int, default=10)
    args = parser.parse_args(argv)
    result = build_atlas(args.assets, args.output, seed=args.seed, levels=args.levels)
    print(json.dumps({'tiles': len(result['tiles']), 'files': result['files']}, indent=2))


if __name__ == '__main__':
    main()

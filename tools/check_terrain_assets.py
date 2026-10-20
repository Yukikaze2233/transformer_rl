#!/usr/bin/env python3
"""Probe prepared collision meshes in an admitted Kaiser PhysX context.

This is a bounded asset check, not robot, material-calibration or PPO acceptance.
The caller owns resource admission and keeps its leases until this process exits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np


class TerrainAssetCheck:
    """Keep source identity, native paths and geometric probes in one owner."""

    def __init__(self, manifest, *, tolerance_m=5e-5, disable_welding=False):
        from transformer_rl.formal.tasks.terrain_assets import file_pin

        self.path = Path(manifest).resolve()
        self.manifest_pin = file_pin(self.path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get('format') != 'prepared_terrain_assets_v1':
            raise ValueError('expected a prepared terrain asset manifest')
        if not 0 < tolerance_m < .0005:
            raise ValueError('probe tolerance must resolve the source millimetre features')
        if type(disable_welding) is not bool:
            raise TypeError('mesh welding diagnostic requires an explicit bool')
        self.tolerance = tolerance_m
        self.disable_welding = disable_welding
        self.assets = {}
        for name, record in self.manifest['assets'].items():
            if Path(name).name != name:
                raise ValueError('invalid prepared asset name')
            folder = self.path.parent / name
            for filename, pin in record['files'].items():
                path = folder / filename
                if path.is_symlink() or not path.is_file() or file_pin(path) != pin:
                    raise ValueError('prepared asset identity differs: ' + str(path))
            with np.load(folder / 'collision.npz', allow_pickle=False) as source:
                geometry = {key: source[key].copy() for key in source.files}
            self.assets[name] = (folder, record, geometry)

    def _probes(self, record, geometry):
        vertices, faces = geometry['vertices_m'], geometry['triangles']
        triangles = vertices[faces].astype(np.float64)
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        norms = np.linalg.norm(normals, axis=-1)
        if not np.isfinite(triangles).all() or not (norms > 0).all():
            raise ValueError('prepared geometry contains nonfinite or degenerate faces')
        for index, (point, normal) in enumerate(zip(triangles.mean(1), normals / norms[:, None])):
            yield 'triangle_centroid', index, point + .005 * normal, -normal, .01, point

        additions = record['processing'].get('training_additions', ())
        if additions:
            from transformer_rl.formal.tasks.terrain_courses import surface_height

            addition, = additions
            source_faces = np.isin(geometry['output_to_input_face'], addition['source_face_indices'])
            generated = geometry['derived_face_kind'] == 1
            # Both the generated strip and the original upper panel must be
            # present in the cooked collider, not just in the exported USD.
            x, z = np.meshgrid(np.linspace(-.5, .5, 9),
                               np.array([.001, .01, .025, .049, .051, .10, .15, .199]))
            points = np.column_stack((x.ravel(), z.ravel()))
            y = surface_height(vertices[:, [0, 2, 1]], faces[source_faces | generated], points)
            if not np.isfinite(y).all():
                raise ValueError('approved front grid is not supported by prepared geometry')
            for index, (point, hit_y) in enumerate(zip(points, y)):
                expected = np.array([point[0], hit_y, point[1]])
                yield 'continuous_front', index, expected - [0, .02, 0], np.array([0., 1., 0.]), .04, expected

    @staticmethod
    def _first_intersection(triangles, start, direction, distance):
        """PhysX may report only the closest triangle of each mesh shape."""
        first = triangles[:, 1] - triangles[:, 0]
        second = triangles[:, 2] - triangles[:, 0]
        cross = np.cross(direction, second)
        determinant = (first * cross).sum(-1)
        safe = np.where(np.abs(determinant) > 1e-15, determinant, np.nan)
        offset = start - triangles[:, 0]
        u = (offset * cross).sum(-1) / safe
        q = np.cross(offset, first)
        v = (q * direction).sum(-1) / safe
        length = (q * second).sum(-1) / safe
        valid = ((u >= -1e-7) & (v >= -1e-7) & (u + v <= 1.0000001)
                 & (length >= 0) & (length <= distance))
        if not valid.any():
            raise ValueError('reference ray has no intersection with the prepared mesh')
        return start + length[valid].min() * direction

    def run(self, sim, query):
        from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        stage = sim.stage
        authored = []
        for ordinal, (name, (folder, record, geometry)) in enumerate(self.assets.items()):
            root = stage.DefinePrim('/World/Assets/' + name, 'Xform')
            if not root.GetReferences().AddReference(str(folder / 'terrain.usda')):
                raise RuntimeError('asset reference was not authored')
            origin = (10. * ordinal, 0., 0.)
            UsdGeom.XformCommonAPI(root).SetTranslate(origin)
            collision = stage.GetPrimAtPath(str(root.GetPath()) + '/Collision')
            if not collision.IsA(UsdGeom.Mesh) or not collision.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError('prepared collision mesh is absent')
            if self.disable_welding:
                # Diagnostic stage override only: neither the source asset nor
                # a running training scene is edited by this comparison.
                PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(collision).CreateWeldToleranceAttr(0.)
            mesh = UsdGeom.Mesh(collision)
            np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), geometry['vertices_m'])
            np.testing.assert_array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                                          geometry['triangles'])
            material, _relationship = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial('physics')
            values = {}
            for key, attribute in (('static_friction', 'physics:staticFriction'),
                                   ('dynamic_friction', 'physics:dynamicFriction'),
                                   ('restitution', 'physics:restitution')):
                value = material.GetPrim().GetAttribute(attribute).Get()
                if value is None or abs(float(value) - record['material'][key]) > 1e-6:
                    raise RuntimeError('composed physics material differs from the prepared source')
                values[key] = float(value)
            cooking = {}
            for api_name in ('PhysxTriangleMeshCollisionAPI', 'PhysxMeshCollisionAPI', 'PhysxCollisionAPI'):
                schema = getattr(PhysxSchema, api_name, None)
                if schema is not None:
                    cooking[api_name] = {str(key): str(collision.GetAttribute(key).Get())
                                         for key in schema.GetSchemaAttributeNames()}
            authored.append({'asset': name, 'collision_path': str(collision.GetPath()), 'origin_m': origin,
                             'material': values, 'material_calibrated': False, 'cooking_attributes': cooking})
        sim.reset()
        sim.forward()
        results = []
        for entry in authored:
            name, path = entry['asset'], entry['collision_path']
            _folder, record, geometry = self.assets[name]
            counts, maximum, failed, diagnostics, occluded = {}, 0., 0, [], 0
            triangles = geometry['vertices_m'][geometry['triangles']].astype(np.float64)
            origin = np.asarray(entry['origin_m'])
            for kind, index, start, direction, distance, centroid in self._probes(record, geometry):
                expected = self._first_intersection(triangles, start, direction, distance)
                occluded += int(np.linalg.norm(expected - centroid) > self.tolerance)
                expected, start = expected + origin, start + origin
                errors = []
                positions = []

                def report(hit):
                    if str(hit.collision) == path:
                        actual = np.array(tuple(float(value) for value in hit.position))
                        if np.isfinite(actual).all():
                            errors.append(float(np.linalg.norm(actual - expected)))
                            positions.append(actual.tolist())
                    return True

                query.raycast_all(tuple(start), tuple(direction), float(distance), report, True)
                error = min(errors, default=None)
                if error is None or error > self.tolerance:
                    failed += 1
                    if len(diagnostics) < 64:
                        diagnostics.append({'kind': kind, 'probe': index, 'error_m': error,
                                            'expected_m': expected.tolist(), 'actual_positions_m': positions})
                if error is not None:
                    maximum = max(maximum, error)
                counts[kind] = counts.get(kind, 0) + 1
            results.append({**entry, 'probes': counts, 'maximum_error_m': maximum})
            results[-1].update(failed_probes=failed, first_failures=diagnostics,
                               occluded_source_centroids=occluded)
        return {'format': 'native_terrain_asset_probes_v1', 'completed': True,
                'passed': all(entry['failed_probes'] == 0 for entry in results),
                'prepared_manifest_pin': self.manifest_pin, 'results': results,
                'stage_overrides': {'mesh_weld_tolerance_m': 0. if self.disable_welding else None},
                'tolerance_m': self.tolerance, 'robot_or_PPO_test': False,
                'dynamic_contact_qualified': False, 'hardware_qualified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--disable-mesh-welding', action='store_true',
                        help='Diagnostic stage override; preserve and compare a baseline result separately')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, device=args.device, enable_cameras=False, livestream=0)
        import omni.kit.app
        import omni.physx
        import torch
        from isaaclab.app.settings_manager import SettingsManager
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        settings = SettingsManager.instance()
        settings.initialize_carb_settings()
        if args.device.startswith('cuda'):
            settings.set_int('/physics/cudaDevice', int(args.device.split(':')[-1]))
            settings.set_bool('/physics/suppressReadback', True)
            extensions = omni.kit.app.get_app().get_extension_manager()
            if not extensions.is_extension_enabled('omni.physx.gpu'):
                extensions.set_extension_enabled_immediate('omni.physx.gpu', True)
            torch.cuda.set_device(args.device)
            torch.cuda.init()
        sim = SimulationContext(SimulationCfg(
            dt=.005, device=args.device, enable_scene_query_support=True,
            physics=PhysxCfg(enable_scene_query_support=True)))
        checker = TerrainAssetCheck(args.manifest, disable_welding=args.disable_mesh_welding)
        result = checker.run(sim, omni.physx.get_physx_scene_query_interface())
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        marker = 'TERRAIN_ASSET_CHECK_COMPLETE ' if result['passed'] else 'TERRAIN_ASSET_CHECK_FAILED '
        print(marker + json.dumps(result), flush=True)
        if not result['passed']:
            raise RuntimeError('native collision/query parity failed; see result.json')
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()

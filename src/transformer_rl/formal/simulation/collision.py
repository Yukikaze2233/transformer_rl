"""Startup native wheel-cylinder probes against named owned colliders.

These bounded ray probes check the selected cook at sampled radial/cap points;
they are not a settling, task-health or hardware qualification. They never run
inside a PPO update and never replace wheel clearance with contact separation.
"""
from __future__ import annotations

import math

import torch

from .ground import WheelCollisionCentres
from .scene import NativeSceneState


@torch.no_grad()
def verify_wheel_cylinders(scene, wheels, query, *, native_to_tensor,
                           tolerance_m, ray_offset_m=.01, radial_samples=32):
    if (not isinstance(scene, NativeSceneState) or not isinstance(wheels, WheelCollisionCentres)
            or not callable(native_to_tensor) or not callable(getattr(query, 'raycast_all', None))
            or type(radial_samples) is not int or radial_samples < 8
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (tolerance_m, ray_offset_m))
            or not 0 < tolerance_m < ray_offset_m or wheels.urdf_sha256 is None
            or scene.readback.get('selected_urdf_sha256') != wheels.urdf_sha256):
        raise ValueError('native cylinder probes require selected sources and explicit sample/error bounds')
    bridge = scene.observation
    bridge.before_physics_step()
    poses = native_to_tensor(scene.robot.root_view.get_link_transforms()).detach().cpu().double()
    if (poses.shape != (bridge.num_envs, 19, 7) or not torch.isfinite(poses).all()
            or not torch.allclose(poses[..., 3:].norm(dim=-1), torch.ones_like(poses[..., 6]), atol=1e-5, rtol=0)):
        raise ValueError('wheel probe requires current named native link poses')
    indices = bridge.binding.wheel_body_indices
    if tuple(bridge.binding.native_body_names[i] for i in indices) != wheels.body_names:
        raise ValueError('native wheel names differ from the selected cylinders')
    records = []
    # All clones share the pinned source/authoring path. Probe both ends of
    # the native permutation rather than ray-testing every identical clone.
    rows = sorted({0, bridge.num_envs - 1})
    for row in rows:
        for side, body in enumerate(wheels.body_names):
            pose = poses[row, indices[side]]
            quaternion = pose[3:] / pose[3:].norm()

            def rotate(v):
                cross = torch.linalg.cross(quaternion[:3], v)
                return v + 2 * (quaternion[3] * cross + torch.linalg.cross(quaternion[:3], cross))

            axis = torch.tensor(wheels.local_axis_xyz[side], dtype=torch.float64)
            seed = torch.eye(3, dtype=torch.float64)[axis.abs().argmin()]
            u = torch.linalg.cross(axis, seed)
            u /= u.norm()
            v = torch.linalg.cross(axis, u)
            directions = [(math.cos(2 * math.pi * i / radial_samples) * u
                           + math.sin(2 * math.pi * i / radial_samples) * v, wheels.radius_m[side])
                          for i in range(radial_samples)]
            directions += [(axis, wheels.length_m[side] / 2), (-axis, wheels.length_m[side] / 2)]
            centre = torch.tensor(wheels.local_xyz_m[side], dtype=torch.float64)
            prefix = scene.environment_paths[row] + '/Robot/' + body + '/'
            for index, (normal, distance) in enumerate(directions):
                point, normal_world = pose[:3] + rotate(centre + normal * distance), rotate(normal)
                errors = []

                def report(hit):
                    if str(hit.collision).startswith(prefix):
                        actual = torch.tensor(tuple(float(x) for x in hit.position), dtype=torch.float64)
                        if torch.isfinite(actual).all():
                            errors.append(float((actual - point).norm()))
                    return True

                start = point + ray_offset_m * normal_world
                query.raycast_all(tuple(start.tolist()), tuple((-normal_world).tolist()),
                                  2 * ray_offset_m, report, True)
                error = min(errors, default=None)
                records.append({'native_row': row, 'body': body, 'sample': index,
                                'kind': 'radial' if index < radial_samples else 'cap', 'error_m': error})
                if error is None or error > tolerance_m:
                    raise RuntimeError('selected wheel native collision differs at ' + repr(records[-1]))
    return {'scope': 'selected_cylinder_radial_and_cap_native_ray_probes',
            'urdf_sha256': wheels.urdf_sha256, 'native_rows': rows, 'radial_samples': radial_samples,
            'tolerance_m': tolerance_m, 'maximum_error_m': max(item['error_m'] for item in records),
            'records': records, 'physical_task_qualified': False, 'hardware_qualified': False}

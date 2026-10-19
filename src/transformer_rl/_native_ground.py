"""Private privileged ground query at root and actual wheel collider centres."""
from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import torch

from ._native_plant import GroundReference
from ._native_scene import NativeSceneState
from ._physical_observation import MechanicalBinding
from ._terrain_geometry import TerrainQuery, TerrainQueryBatch
from ._terrain_tasks import TerrainCourse


@dataclass(frozen=True)
class WheelCollisionCentres:
    body_names: tuple[str, str]
    local_xyz_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    radius_m: tuple[float, float]
    length_m: tuple[float, float]

    @classmethod
    def from_urdf(cls, urdf, binding: MechanicalBinding):
        if not isinstance(binding, MechanicalBinding):
            raise ValueError('wheel centres require the complete mechanical binding')
        root = ET.fromstring(urdf)
        bodies = tuple(binding.native_body_names[i] for i in binding.wheel_body_indices)
        offsets, radii, lengths = [], [], []
        for name in bodies:
            links = [link for link in root.findall('link') if link.attrib.get('name') == name]
            if len(links) != 1 or len(links[0].findall('collision')) != 1:
                raise ValueError('each named wheel must bind exactly one collision cylinder')
            collision = links[0].find('collision')
            cylinder = collision.find('geometry/cylinder')
            origin = collision.find('origin')
            if cylinder is None or origin is None or 'xyz' not in origin.attrib:
                raise ValueError('wheel cylinder origin and dimensions must be explicit')
            xyz = tuple(float(v) for v in origin.attrib['xyz'].split())
            radius, length = float(cylinder.attrib['radius']), float(cylinder.attrib['length'])
            if len(xyz) != 3 or not all(math.isfinite(v) for v in (*xyz, radius, length)) or min(radius, length) <= 0:
                raise ValueError('wheel collider geometry is invalid')
            offsets.append(xyz)
            radii.append(radius)
            lengths.append(length)
        return cls(bodies, tuple(offsets), tuple(radii), tuple(lengths))


@dataclass(frozen=True)
class GroundQuerySample:
    terrain: TerrainQuery
    world_xyz_m: torch.Tensor
    timestamp_s: torch.Tensor
    model_epoch: int


class TerrainCoverageError(RuntimeError):
    """A requested reference point has no authored ground, not infinite plane."""

    def __init__(self, sample):
        super().__init__('root or wheel collider centre left the authored terrain domain')
        self.sample = sample


class NativeGroundQuery:
    """Reorder design courses to actual clones and query fresh link poses.

    This is a privileged geometry reference. It is not a terrain observation
    for the actor, a measured contact point, or proof that native cooking and
    analytic geometry agree. That equivalence needs the SDK acceptance trace.
    """

    def __init__(self, scene: NativeSceneState, design_courses, wheels: WheelCollisionCentres):
        if not isinstance(scene, NativeSceneState) or not isinstance(wheels, WheelCollisionCentres):
            raise ValueError('ground query requires its actual scene and named collider geometry')
        courses = tuple(design_courses)
        count = scene.observation.num_envs
        if (len(courses) != count or any(not isinstance(c, TerrainCourse) for c in courses)
                or sorted(scene.clone_indices) != list(range(count))):
            raise ValueError('ground query requires all original courses and an exact native clone permutation')
        courses = tuple(courses[i] for i in scene.clone_indices)
        if tuple(c.surface.meshes for c in courses) != scene.terrain_meshes:
            raise ValueError('ground query geometry/material differs from the authored native scene')
        binding = scene.observation.binding
        if tuple(binding.native_body_names[i] for i in binding.wheel_body_indices) != wheels.body_names:
            raise ValueError('wheel geometry does not match the native named body order')
        self.scene, self._view = scene, scene.robot.root_view
        self.courses = courses  # Native order, bound to exactly the authored meshes.
        self.device, self.num_envs = scene.observation.device, count
        self.batch = TerrainQueryBatch(tuple(c.surface for c in courses), device=self.device)
        self._origins = scene.environment_origins_m.detach().clone()
        self._meshes = scene.terrain_meshes
        self._wheels = list(binding.wheel_body_indices)
        self._offsets = torch.tensor(wheels.local_xyz_m, dtype=torch.float32, device=self.device)
        self.last_sample = None

    @torch.no_grad()
    def __call__(self, robot, scene):
        bridge = self.scene.observation
        bridge.before_state_read()
        if (scene is not self.scene or robot is not self.scene.robot or robot.root_view is not self._view
                or scene.terrain_meshes is not self._meshes
                or not torch.equal(scene.environment_origins_m, self._origins)):
            raise RuntimeError('native ground owner, terrain or environment origins changed')
        poses = robot.data.body_link_pose_w.torch
        root = robot.data.root_link_pose_w.torch
        if (not isinstance(poses, torch.Tensor) or poses.shape != (self.num_envs, 19, 7)
                or poses.device != self.device or poses.dtype != torch.float32 or not torch.isfinite(poses).all()
                or not isinstance(root, torch.Tensor) or root.shape != (self.num_envs, 7)
                or root.device != self.device or root.dtype != torch.float32 or not torch.isfinite(root).all()
                or not torch.allclose(poses[:, 0, :3], root[:, :3], atol=1e-5, rtol=0)):
            raise ValueError('fresh native root/body link pose shape, values or reference frames differ')
        selected = poses[:, self._wheels].clone()
        q = selected[..., 3:7]
        norm = q.norm(dim=-1, keepdim=True)
        if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0):
            raise ValueError('native wheel link orientation is not a normalized xyzw quaternion')
        q = q / norm
        cross = torch.linalg.cross(q[..., :3], self._offsets.expand(self.num_envs, -1, -1))
        rotated = self._offsets + 2 * (q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross))
        points = torch.cat((root[:, None, :3], selected[..., :3] + rotated), 1)
        local_xy = points[..., :2] - self._origins[:, None, :2]
        result = self.batch.at(local_xy)
        sample = GroundQuerySample(result, points.clone(), bridge.timestamp_s(), bridge.model_epoch)
        self.last_sample = sample
        if not result.valid.all():
            raise TerrainCoverageError(sample)
        bridge.before_state_read()
        if bridge.model_epoch != sample.model_epoch or not torch.equal(bridge.timestamp_s(), sample.timestamp_s):
            raise RuntimeError('ground query crossed a reported physical/model epoch')
        return GroundReference(result.height_m[:, 0] + self._origins[:, 2],
                               result.height_m[:, 1:] + self._origins[:, 2:3], result.friction[:, 1:])

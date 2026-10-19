"""Private selected-asset pin geometry and output-admission diagnostics.

Knee angles use real body poses and pin-centre vectors projected normal to the
hinge, not motor differences or inferred gear coordinates. Parsing a source
model does not certify native stops, closure, self-contact or reachability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .actuation import OutputTargets
from .observation import MechanicalBinding


@dataclass(frozen=True)
class _LegPins:
    hip_parent_body: str
    upper_body: str
    lower_body: str
    knee_joint: str
    hip_pin_parent_m: tuple[float, ...]
    knee_pin_upper_m: tuple[float, ...]
    wheel_pin_lower_m: tuple[float, ...]
    axis_upper: tuple[float, ...]
    native_limits_rad: tuple[float, float]


@dataclass(frozen=True)
class JointBoundaryState:
    knee_inner_angle_rad: torch.Tensor
    knee_signed_clearance_rad: torch.Tensor
    native_knee_signed_clearance_rad: torch.Tensor
    knee_pivot_residual_m: torch.Tensor
    output_admission_correction_rad: torch.Tensor
    shortfall_rad: torch.Tensor


@dataclass(frozen=True)
class LegBoundaryGeometry:
    binding: MechanicalBinding
    manifest_sha256: str
    model_spec_sha256: str
    nominal_mass_kg: float
    inner_limits_rad: tuple[float, float]
    sides: tuple[_LegPins, _LegPins]

    @staticmethod
    def _origin(joint):
        value = torch.as_tensor(joint['origin'], dtype=torch.float64)
        if (value.shape != (4, 4) or not torch.isfinite(value).all()
                or not torch.allclose(value[3], value.new_tensor([0., 0., 0., 1.]), atol=1e-8, rtol=0)
                or not torch.allclose(value[:3, :3].T @ value[:3, :3], torch.eye(3, dtype=torch.float64), atol=1e-6, rtol=0)
                or abs(torch.linalg.det(value[:3, :3]).item() - 1) > 1e-6):
            raise ValueError('pin geometry requires a finite rigid source joint origin')
        return value

    @classmethod
    @torch.no_grad()
    def from_asset_sources(cls, manifest_bytes, model_spec_bytes, binding: MechanicalBinding):
        if (not isinstance(manifest_bytes, bytes) or not isinstance(model_spec_bytes, bytes)
                or not isinstance(binding, MechanicalBinding)):
            raise ValueError('pin geometry requires exact asset source bytes and native named binding')
        manifest, spec = json.loads(manifest_bytes), json.loads(model_spec_bytes)
        sha = hashlib.sha256(model_spec_bytes).hexdigest()
        declared = MechanicalBinding.from_manifest(manifest, binding.native_joint_names, binding.native_body_names)
        if (declared != binding or manifest['files_sha256'].get('model_spec.json') != sha
                or manifest.get('knee_angle_definition') != 'pin_center_vectors_projected_normal_to_common_hinge_axis'):
            raise ValueError('selected asset pin-angle definition, source hash or native binding differs')
        joints = {j['name']: j for j in spec['joints']}
        if len(joints) != 18 or len(spec['joints']) != 18 or set(joints) != set(binding.native_joint_names):
            raise ValueError('pin geometry must bind the complete18-joint selected model')
        raw = manifest['knee_inner_limits_deg']
        mass = manifest['total_mass_kg']
        if (not isinstance(raw, (tuple, list)) or len(raw) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*raw, mass))
                or not 0 < raw[0] < raw[1] < 180 or mass <= 0):
            raise ValueError('selected mass and geometric inner-angle interval must be explicit and finite')
        limits = tuple(math.radians(v) for v in raw)
        active = tuple(binding.native_joint_names[i] for i in binding.policy_joint_indices)
        sides = []
        for side in range(2):
            hip, wheel = joints[active[2 * side]], joints[active[4 + side]]
            candidates = [j for j in joints.values() if j['parent'] == hip['child'] and j['type'] == 'revolute']
            if (hip['parent'] != 'base_link' or hip['type'] != 'continuous' or len(candidates) != 1
                    or wheel['type'] != 'continuous'
                    or wheel['child'] != binding.native_body_names[binding.wheel_body_indices[side]]):
                raise ValueError('selected hip, passive knee and wheel topology is ambiguous')
            knee = candidates[0]
            if wheel['parent'] != knee['child'] or knee['name'] in active:
                raise ValueError('wheel must be attached to its own lower link beyond a passive knee')
            origins = [cls._origin(j) for j in (hip, knee, wheel)]
            axis = torch.as_tensor(knee['axis'], dtype=torch.float64)
            if axis.shape != (3,) or not torch.isfinite(axis).all() or abs(axis.norm().item() - 1) > 1e-5:
                raise ValueError('selected knee axis must be finite and unit length')
            axis /= axis.norm()
            axis_upper = origins[1][:3, :3] @ axis
            qlimits = tuple(float(knee['limit'][key]) for key in ('lower', 'upper'))
            if (not all(math.isfinite(v) for v in qlimits) or qlimits[0] >= qlimits[1]
                    or abs(qlimits[1] - qlimits[0] - (limits[1] - limits[0])) > 1e-6):
                raise ValueError('native passive stops and geometric angle span differ')
            angles = []
            upper_vector = -origins[1][:3, 3]
            upper_vector -= (upper_vector @ axis_upper) * axis_upper
            for q in qlimits:
                point = origins[2][:3, 3]
                rotated = point * math.cos(q) + torch.linalg.cross(axis, point) * math.sin(q)
                rotated += axis * (axis @ point) * (1 - math.cos(q))
                lower_vector = origins[1][:3, :3] @ rotated
                lower_vector -= (lower_vector @ axis_upper) * axis_upper
                if min(upper_vector.norm(), lower_vector.norm()) <= 1e-8:
                    raise ValueError('selected pin vectors collapse onto the hinge axis')
                angles.append(torch.atan2(torch.linalg.cross(upper_vector, lower_vector).norm(),
                                           upper_vector @ lower_vector).item())
            if any(abs(a - b) > 1e-6 for a, b in zip(sorted(angles), limits)):
                raise ValueError('native stop endpoints do not reproduce the selected pin-centre inner angles')
            sides.append(_LegPins(hip['parent'], hip['child'], knee['child'], knee['name'],
                                   tuple(origins[0][:3, 3].tolist()), tuple(origins[1][:3, 3].tolist()),
                                   tuple(origins[2][:3, 3].tolist()), tuple(axis_upper.tolist()), qlimits))
        return cls(binding, hashlib.sha256(manifest_bytes).hexdigest(), sha, float(mass), limits, tuple(sides))

    @staticmethod
    def _rotate(quaternion, value):
        cross = torch.linalg.cross(quaternion[..., :3], value)
        return value + 2 * (quaternion[..., 3:] * cross + torch.linalg.cross(quaternion[..., :3], cross))

    @torch.no_grad()
    def evaluate(self, poses_world_xyzw, tree_position_rad, targets: OutputTargets, *, knee_margin_rad):
        if (not isinstance(poses_world_xyzw, torch.Tensor) or poses_world_xyzw.ndim != 3
                or poses_world_xyzw.shape[0] < 1 or poses_world_xyzw.shape[1:] != (19, 7)
                or poses_world_xyzw.dtype != torch.float32
                or type(knee_margin_rad) not in (int, float) or not math.isfinite(knee_margin_rad)
                or not 0 < knee_margin_rad < (self.inner_limits_rad[1] - self.inner_limits_rad[0]) / 2
                or not isinstance(targets, OutputTargets)):
            raise ValueError('boundary evaluation requires complete native poses, targets and explicit geometric margin')
        count, device = poses_world_xyzw.shape[0], poses_world_xyzw.device
        for name, value, shape, dtype in (('tree positions', tree_position_rad, (count, 18), torch.float32),
                                         ('raw targets', targets.raw_leg_position_rad, (count, 4), torch.float32),
                                         ('admitted targets', targets.admitted_leg_position_rad, (count, 4), torch.float32),
                                         ('common winding', targets.common_turns, (count, 2), torch.float64)):
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device
                    or value.dtype != dtype or not torch.isfinite(value).all()):
                raise ValueError('boundary ' + name + ' shape, device, units or finite values differ')
        if (not torch.isfinite(poses_world_xyzw).all()
                or (poses_world_xyzw[:, :, 3:].norm(dim=-1) - 1).abs().gt(1e-4).any()
                or (targets.common_turns != targets.common_turns.round()).any()):
            raise ValueError('boundary source poses or shared winding are invalid')
        angles, residuals, native_distances = [], [], []
        body_names = self.binding.native_body_names
        poses = poses_world_xyzw.double()

        def point(body, local):
            pose = poses[:, body_names.index(body)]
            q = pose[:, 3:] / pose[:, 3:].norm(dim=-1, keepdim=True)
            local = poses.new_tensor(local).expand(count, -1)
            return pose[:, :3] + self._rotate(q, local)

        for side in self.sides:
            hip = point(side.hip_parent_body, side.hip_pin_parent_m)
            knee = point(side.upper_body, side.knee_pin_upper_m)
            wheel = point(side.lower_body, side.wheel_pin_lower_m)
            pose = poses[:, body_names.index(side.upper_body)]
            q = pose[:, 3:] / pose[:, 3:].norm(dim=-1, keepdim=True)
            axis = self._rotate(q, poses.new_tensor(side.axis_upper).expand(count, -1))
            u, v = hip - knee, wheel - knee
            u -= (u * axis).sum(-1, keepdim=True) * axis
            v -= (v * axis).sum(-1, keepdim=True) * axis
            if (u.norm(dim=-1).le(1e-8) | v.norm(dim=-1).le(1e-8)).any():
                raise ValueError('physical pin vectors collapsed; no valid geometric angle may be invented')
            angles.append(torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u * v).sum(-1)))
            residuals.append((knee - poses[:, body_names.index(side.lower_body), :3]).norm(dim=-1))
            native = tree_position_rad[:, self.binding.tree_joint_names.index(side.knee_joint)].double()
            native_distances.append(torch.minimum(native - side.native_limits_rad[0], side.native_limits_rad[1] - native))
        angle = torch.stack(angles, -1)
        clearance = torch.minimum(angle - self.inner_limits_rad[0], self.inner_limits_rad[1] - angle)
        # Both outputs of each side share the recorded winding. Independent
        # per-axis wraps would hide or reverse a real admission correction.
        admitted = targets.admitted_leg_position_rad.double().unflatten(-1, (2, 2))
        admitted -= targets.common_turns[:, :, None] * (2 * math.pi)
        correction = (targets.raw_leg_position_rad.double() - admitted.flatten(-2)).abs()
        shortfall = torch.cat(((knee_margin_rad - clearance).clamp_min(0), correction), -1).float()
        native_clearance = torch.stack(native_distances, -1).float()
        pivot_residual = torch.stack(residuals, -1).float()
        if not all(torch.isfinite(v).all() for v in (shortfall, native_clearance, pivot_residual, correction.float())):
            raise ValueError('boundary shortfall exceeded its finite physical tensor range')
        return JointBoundaryState(angle.float(), clearance.float(), native_clearance, pivot_residual, correction.float(), shortfall)

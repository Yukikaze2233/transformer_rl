"""Private selected-asset tree kinematics and passive closed-chain solving.

Source joint frames,axes and loop anchors stay unchanged. Four coaxial output
coordinates and both wheel coordinates are prescribed,not optimized away to
make an infeasible request appear closed. This is reset geometry,not native
physics,collision acceptance or a direct-knee controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from ._joint_boundaries import LegBoundaryGeometry
from ._physical_observation import MechanicalBinding


@dataclass(frozen=True)
class TreeKinematics:
    body_rotation: torch.Tensor
    body_position_m: torch.Tensor
    joint_axis_world: torch.Tensor
    joint_origin_world_m: torch.Tensor


@dataclass(frozen=True)
class PassiveClosure:
    tree_position: torch.Tensor
    residual_by_constraint_m: torch.Tensor
    maximum_residual_m: torch.Tensor
    within_native_limits: torch.Tensor
    converged: torch.Tensor
    iterations: int
    prescribed_policy_positions: torch.Tensor
    source_manifest_sha256: str
    source_model_sha256: str
    physical_qualification: bool = False


class AssetResetKinematics:
    """Batched full-3D tree/Jacobian, with bounded damped passive-only updates.

    Units are metres and radians. The slider increment scale is explicit and
    differs from angle increments; all original passive limits remain enforced.
    Each row keeps its own residual/damping and may fail independently.
    """

    def __init__(self, manifest_bytes, model_spec_bytes, binding: MechanicalBinding, *, device='cpu'):
        geometry = LegBoundaryGeometry.from_asset_sources(manifest_bytes, model_spec_bytes, binding)
        manifest, spec = json.loads(manifest_bytes), json.loads(model_spec_bytes)
        bodies = spec.get('bodies')
        constraints = spec.get('constraints')
        if (not isinstance(bodies, list) or len(bodies) != 19
                or {body.get('name') for body in bodies} != set(binding.native_body_names)
                or not isinstance(constraints, list) or len(constraints) != 6
                or len({value.get('name') for value in constraints}) != 6):
            raise ValueError('reset kinematics requires every selected body and all six named loop constraints')
        self.binding, self.device, self.geometry = binding, torch.device(device), geometry
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.model_sha256 = hashlib.sha256(model_spec_bytes).hexdigest()
        joints = {value['name']: value for value in spec['joints']}
        if (set(joints) != set(binding.tree_joint_names)
                or set(spec.get('nominal_joint_pos', {})) != set(binding.tree_joint_names)
                or any(spec['nominal_joint_pos'][name] != manifest['nominal_joint_pos'][name]
                       for name in binding.tree_joint_names)):
            raise ValueError('selected tree order and complete nominal source coordinates differ')
        self._joints = tuple(joints[name] for name in binding.tree_joint_names)
        self._kinds, self._parents, self._children = [], [], []
        origin, axes, lower, upper = [], [], [], []
        for joint in self._joints:
            kind = joint['type']
            axis = torch.as_tensor(joint['axis'], dtype=torch.float64)
            if (kind not in ('continuous', 'revolute', 'prismatic') or axis.shape != (3,)
                    or not torch.isfinite(axis).all() or abs(axis.norm().item() - 1) > 1e-5
                    or joint['parent'] not in binding.native_body_names
                    or joint['child'] not in binding.native_body_names or joint['child'] == 'base_link'):
                raise ValueError('reset tree joint type,axis or parent/child differs from selected source')
            origin.append(LegBoundaryGeometry._origin(joint))
            axes.append(axis / axis.norm())
            self._kinds.append(kind)
            self._parents.append(binding.native_body_names.index(joint['parent']))
            self._children.append(binding.native_body_names.index(joint['child']))
            limits = joint.get('limit', {})
            lo, hi = (-math.inf, math.inf) if kind == 'continuous' else (float(limits['lower']), float(limits['upper']))
            if kind != 'continuous' and (not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi):
                raise ValueError('bounded passive joints require explicit ordered source limits')
            lower.append(lo)
            upper.append(hi)
        root = binding.native_body_names.index('base_link')
        if len(set(self._children)) != 18:
            raise ValueError('reset source must be a single connected19-body tree')
        remaining, visited, order = set(range(18)), {root}, []
        ancestors = torch.zeros(19, 18, dtype=torch.bool)
        while remaining:
            ready = sorted(i for i in remaining if self._parents[i] in visited)
            if not ready:
                raise ValueError('reset source tree has an unrooted component or cycle')
            for i in ready:
                if self._children[i] in visited:
                    raise ValueError('source body has more than one parent')
                ancestors[self._children[i]] = ancestors[self._parents[i]]
                ancestors[self._children[i], i] = True
                visited.add(self._children[i])
                remaining.remove(i)
                order.append(i)
        self._root, self._order = root, tuple(order)
        self._origin = torch.stack(origin).to(self.device)
        self._axes = torch.stack(axes).to(self.device)
        self._lower = torch.tensor(lower, dtype=torch.float64, device=self.device)
        self._upper = torch.tensor(upper, dtype=torch.float64, device=self.device)
        self._prismatic = torch.tensor([kind == 'prismatic' for kind in self._kinds], device=self.device)
        self._ancestors = ancestors.to(self.device)
        policy_names = tuple(binding.native_joint_names[i] for i in binding.policy_joint_indices)
        self.policy_tree_indices = tuple(binding.tree_joint_names.index(name) for name in policy_names)
        self.passive_tree_indices = tuple(i for i in range(18) if i not in self.policy_tree_indices)
        loop_bodies, loop_points = [], []
        for constraint in constraints:
            if (constraint.get('type') != 'spherical' or constraint.get('physics:excludeFromArticulation') is not True
                    or any(constraint.get(name) not in binding.native_body_names for name in ('body0', 'body1'))
                    or constraint['body0'] == constraint['body1']):
                raise ValueError('reset solver needs the selected six explicit spherical loop anchors')
            points = torch.as_tensor([constraint['local_pos0_m'], constraint['local_pos1_m']], dtype=torch.float64)
            if points.shape != (2, 3) or not torch.isfinite(points).all():
                raise ValueError('source loop anchors must be finite local3D positions')
            loop_bodies.append([binding.native_body_names.index(constraint[name]) for name in ('body0', 'body1')])
            loop_points.append(points)
        self.constraint_names = tuple(value['name'] for value in constraints)
        self._loop_bodies = torch.tensor(loop_bodies, dtype=torch.long, device=self.device)
        self._loop_points = torch.stack(loop_points).to(self.device)
        self.nominal_tree_position = torch.tensor([manifest['nominal_joint_pos'][name]
                                                    for name in binding.tree_joint_names],
                                                   dtype=torch.float64, device=self.device)
        if not torch.isfinite(self.nominal_tree_position).all():
            raise ValueError('selected nominal reset coordinates must be finite')

    def _positions(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 18
                or value.shape[0] < 1 or value.device != self.device or value.dtype != torch.float64
                or not torch.isfinite(value).all()):
            raise ValueError('reset kinematics expects finite float64[N,18] in selected tree order')

    @staticmethod
    def _skew(axis):
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)

    @torch.no_grad()
    def forward(self, tree_position):
        self._positions(tree_position)
        count = len(tree_position)
        identity = torch.eye(3, dtype=torch.float64, device=self.device).expand(count, -1, -1)
        zero = torch.zeros(count, 3, dtype=torch.float64, device=self.device)
        rotations, positions, axes, pivots = [None] * 19, [None] * 19, [None] * 18, [None] * 18
        rotations[self._root], positions[self._root] = identity, zero
        for index in self._order:
            parent, child = self._parents[index], self._children[index]
            origin = self._origin[index]
            rotation = rotations[parent] @ origin[:3, :3]
            position = positions[parent] + (rotations[parent] @ origin[:3, 3]).squeeze(-1)
            axis, q = self._axes[index], tree_position[:, index]
            axes[index] = (rotation @ axis).squeeze(-1)
            pivots[index] = position
            if self._kinds[index] == 'prismatic':
                rotations[child], positions[child] = rotation, position + axes[index] * q[:, None]
            else:
                cross = self._skew(axis)
                motion = identity + q.sin()[:, None, None] * cross + (1 - q.cos())[:, None, None] * (cross @ cross)
                rotations[child], positions[child] = rotation @ motion, position
        return TreeKinematics(torch.stack(rotations, 1), torch.stack(positions, 1),
                              torch.stack(axes, 1), torch.stack(pivots, 1))

    @torch.no_grad()
    def closure(self, tree_position, *, jacobian=False):
        state = self.forward(tree_position)
        rotation = state.body_rotation[:, self._loop_bodies]
        anchors = state.body_position_m[:, self._loop_bodies] + (rotation @ self._loop_points[..., None]).squeeze(-1)
        residual = anchors[:, :, 0] - anchors[:, :, 1]
        if not jacobian:
            return residual
        offsets = anchors[:, :, :, None] - state.joint_origin_world_m[:, None, None]
        axis = state.joint_axis_world[:, None, None].expand_as(offsets)
        derivative = torch.linalg.cross(axis, offsets)
        derivative = torch.where(self._prismatic[None, None, None, :, None], axis, derivative)
        derivative *= self._ancestors[self._loop_bodies][None, :, :, :, None]
        jac = (derivative[:, :, 0] - derivative[:, :, 1]).permute(0, 1, 3, 2)
        return residual, jac.reshape(len(tree_position), 18, 18)

    @torch.no_grad()
    def solve_passive(self, seed_tree_position, prescribed_policy_positions, *, closure_tolerance_m,
                      maximum_iterations=48, angular_step_rad=.25, slider_step_m=.01,
                      slider_scale_m=.05, initial_damping_m2=1e-6):
        self._positions(seed_tree_position)
        count = len(seed_tree_position)
        if (not isinstance(prescribed_policy_positions, torch.Tensor)
                or prescribed_policy_positions.shape != (count, 6)
                or prescribed_policy_positions.dtype != torch.float64
                or prescribed_policy_positions.device != self.device
                or not torch.isfinite(prescribed_policy_positions).all()
                or type(maximum_iterations) is not int or not 1 <= maximum_iterations <= 256
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (closure_tolerance_m, angular_step_rad, slider_step_m,
                                 slider_scale_m, initial_damping_m2))):
            raise ValueError('passive solve requires complete output targets and explicit finite convergence/scaling bounds')
        q = seed_tree_position.detach().clone()
        q[:, self.policy_tree_indices] = prescribed_policy_positions
        passive = list(self.passive_tree_indices)
        low, high = self._lower[passive], self._upper[passive]
        q[:, passive] = q[:, passive].clamp(low, high)
        scale = torch.where(self._prismatic[passive], q.new_tensor(slider_scale_m), q.new_tensor(1.))
        step_bound = torch.where(self._prismatic[passive], q.new_tensor(slider_step_m), q.new_tensor(angular_step_rad))
        damping = torch.full((count,), initial_damping_m2, dtype=torch.float64, device=self.device)
        identity = torch.eye(len(passive), dtype=torch.float64, device=self.device)
        iterations = 0
        for iteration in range(maximum_iterations):
            residual, jac = self.closure(q, jacobian=True)
            ready = residual.norm(dim=-1).amax(-1) <= closure_tolerance_m
            if ready.all():
                break
            jac = jac[:, :, passive] * scale[None, None]
            flat = residual.flatten(1)
            normal = jac.transpose(-1, -2) @ jac + damping[:, None, None] * identity
            rhs = -(jac.transpose(-1, -2) @ flat[:, :, None])
            delta = torch.linalg.solve(normal, rhs).squeeze(-1) * scale
            fraction = (step_bound / delta.abs().clamp_min(1e-30)).amin(-1).clamp_max(1)
            delta *= fraction[:, None]
            candidate = q.clone()
            candidate[:, passive] = (q[:, passive] + delta).clamp(low, high)
            candidate_residual = self.closure(candidate)
            improved = (candidate_residual.square().sum((-1, -2)) < residual.square().sum((-1, -2))) & ~ready
            q = torch.where(improved[:, None], candidate, q)
            damping = torch.where(improved, damping * .3, damping * 10).clamp(1e-14, 1e4)
            iterations = iteration + 1
        residual = self.closure(q).norm(dim=-1)
        within = ((q >= self._lower - 1e-12) & (q <= self._upper + 1e-12)).all(-1)
        if not torch.equal(q[:, self.policy_tree_indices], prescribed_policy_positions):
            raise RuntimeError('passive solver altered prescribed motor/wheel coordinates')
        maximum = residual.amax(-1)
        return PassiveClosure(q, residual, maximum, within, within & (maximum <= closure_tolerance_m), iterations,
                               prescribed_policy_positions.detach().clone(), self.manifest_sha256, self.model_sha256)

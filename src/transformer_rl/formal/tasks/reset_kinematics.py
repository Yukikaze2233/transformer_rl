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

from ..control.joints import LegBoundaryGeometry
from ..control.observation import MechanicalBinding


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


@dataclass(frozen=True)
class PostureClosure:
    tree_position: torch.Tensor
    residual_by_constraint_m: torch.Tensor
    maximum_residual_m: torch.Tensor
    within_native_limits: torch.Tensor
    converged: torch.Tensor
    iterations: int
    requested_inner_knee_angle_rad: torch.Tensor
    prescribed_hip_positions: torch.Tensor
    policy_positions: torch.Tensor
    source_manifest_sha256: str
    source_model_sha256: str
    physical_qualification: bool = False
    solver_rounds_by_row: torch.Tensor | None = None


@dataclass(frozen=True)
class PostureContinuation:
    closure: PostureClosure
    requested_segments: torch.Tensor
    completed_segments: torch.Tensor
    attempted_segments: torch.Tensor
    last_attempted_fraction: torch.Tensor
    last_attempted_tree_position: torch.Tensor
    last_attempted_residual_m: torch.Tensor
    budget_exhausted: torch.Tensor
    maximum_segment_angle_rad: float
    maximum_segments: int
    physical_qualification: bool = False


class AssetResetKinematics:
    """Batched full-3D tree/Jacobian, with bounded reduced-QR passive updates.

    Units are metres and radians. The slider increment scale is explicit and
    differs from angle increments; all original passive limits remain enforced.
    Full-rank closure rows use the source mapper's reduced-QR kernel. A
    rank-deficient row (a compliant slide outside the six loop equations) uses
    minimum-norm Levenberg fallback and keeps its own residual/damping budget.
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
        # Reduced QR treats the six three-row loop residuals as the overdetermined
        # 18-row closure system. The same cutoff discovers slide-only null
        # directions that need the explicit fallback below.
        self._rank_rtol = 18 * torch.finfo(torch.float64).eps
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
        self.knee_tree_indices = tuple(binding.tree_joint_names.index(side.knee_joint) for side in geometry.sides)
        endpoints = self.nominal_tree_position[None].repeat(2, 1)
        for column, index in enumerate(self.knee_tree_indices):
            endpoints[:, index] = endpoints.new_tensor(geometry.sides[column].native_limits_rad)
        self._knee_endpoint_angles = self.inner_knee_angles(endpoints)
        if (self._knee_endpoint_angles[1] - self._knee_endpoint_angles[0]).abs().lt(.1).any():
            raise ValueError('selected geometric knee mapping is not invertible over its source stops')

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
    def closure_from_poses(self, body_link_pose_world_xyzw):
        """Measure the six source loop anchors from actual native body poses.

        Joint-coordinate closure proves the source model was solved. This
        second check is the native link-frame readback that detects an exporter
        or frame mismatch between the selected URDF/model_spec and the USD.
        """
        if (not isinstance(body_link_pose_world_xyzw, torch.Tensor)
                or body_link_pose_world_xyzw.ndim != 3
                or body_link_pose_world_xyzw.shape[1:] != (len(self.binding.native_body_names), 7)
                or not torch.isfinite(body_link_pose_world_xyzw).all()):
            raise ValueError('native closure requires finite [N,19,7] body-link poses')
        poses = body_link_pose_world_xyzw.to(device=self.device, dtype=torch.float64)
        quaternion = poses[..., 3:7]
        norm = quaternion.norm(dim=-1, keepdim=True)
        if (norm <= 0).any():
            raise ValueError('native body quaternion collapsed')
        q = quaternion / norm
        local = self._loop_points.unsqueeze(0).expand(len(poses), -1, -1, -1)
        body = poses[:, self._loop_bodies, :3]
        rotation = self._quaternion_rotation(q[:, self._loop_bodies])
        anchors = body + (rotation @ local.unsqueeze(-1)).squeeze(-1)
        return anchors[:, :, 0] - anchors[:, :, 1]

    @staticmethod
    def _quaternion_rotation(quaternion):
        x, y, z, w = quaternion.unbind(-1)
        return torch.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), -1).reshape(*quaternion.shape[:-1], 3, 3)

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
    def inner_knee_angles(self, tree_position):
        state = self.forward(tree_position)
        names, result = self.binding.native_body_names, []
        for side in self.geometry.sides:
            def point(body, local):
                i = names.index(body)
                return state.body_position_m[:, i] + (state.body_rotation[:, i] @ tree_position.new_tensor(local)).squeeze(-1)
            hip = point(side.hip_parent_body, side.hip_pin_parent_m)
            knee = point(side.upper_body, side.knee_pin_upper_m)
            wheel = point(side.lower_body, side.wheel_pin_lower_m)
            axis = (state.body_rotation[:, names.index(side.upper_body)] @ tree_position.new_tensor(side.axis_upper)).squeeze(-1)
            u, v = hip - knee, wheel - knee
            u -= (u * axis).sum(-1, keepdim=True) * axis
            v -= (v * axis).sum(-1, keepdim=True) * axis
            result.append(torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u * v).sum(-1)))
        return torch.stack(result, -1)

    def native_knee_positions(self, inner_angle_rad):
        if (not isinstance(inner_angle_rad, torch.Tensor) or inner_angle_rad.ndim != 2
                or inner_angle_rad.shape[1] != 2 or inner_angle_rad.dtype != torch.float64
                or inner_angle_rad.device != self.device or not torch.isfinite(inner_angle_rad).all()
                or ((inner_angle_rad < self.geometry.inner_limits_rad[0])
                    | (inner_angle_rad > self.geometry.inner_limits_rad[1])).any()):
            raise ValueError('reset knee request must stay in the explicit selected geometric angle domain')
        limits = inner_angle_rad.new_tensor([side.native_limits_rad for side in self.geometry.sides])
        fraction = ((inner_angle_rad - self._knee_endpoint_angles[0])
                    / (self._knee_endpoint_angles[1] - self._knee_endpoint_angles[0]))
        native = limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])
        return native.clamp(limits[:, 0], limits[:, 1])

    @torch.no_grad()
    def _solve_fixed(self, seed_tree_position, fixed_indices, prescribed_positions, *, closure_tolerance_m,
                     maximum_iterations=48, angular_step_rad=.25, slider_step_m=.01,
                     slider_scale_m=.05, initial_damping_m2=1e-6, iteration_budget_by_row=None):
        self._positions(seed_tree_position)
        count = len(seed_tree_position)
        if (not isinstance(fixed_indices, tuple) or len(set(fixed_indices)) != len(fixed_indices)
                or not 1 <= len(fixed_indices) < 18 or any(type(i) is not int or not 0 <= i < 18 for i in fixed_indices)
                or not isinstance(prescribed_positions, torch.Tensor)
                or prescribed_positions.shape != (count, len(fixed_indices))
                or prescribed_positions.dtype != torch.float64 or prescribed_positions.device != self.device
                or not torch.isfinite(prescribed_positions).all()
                or type(maximum_iterations) is not int or not 1 <= maximum_iterations <= 256
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (closure_tolerance_m, angular_step_rad, slider_step_m,
                                 slider_scale_m, initial_damping_m2))):
            raise ValueError('passive solve requires complete output targets and explicit finite convergence/scaling bounds')
        q = seed_tree_position.detach().clone()
        if iteration_budget_by_row is None:
            iteration_budget_by_row = torch.full((count,), maximum_iterations, dtype=torch.long, device=self.device)
        if (not isinstance(iteration_budget_by_row, torch.Tensor) or iteration_budget_by_row.shape != (count,)
                or iteration_budget_by_row.dtype != torch.long or iteration_budget_by_row.device != self.device
                or ((iteration_budget_by_row < 0) | (iteration_budget_by_row > maximum_iterations)).any()):
            raise ValueError('passive solve row budgets must stay within the explicit iteration cap')
        q[:, fixed_indices] = prescribed_positions
        passive = [i for i in range(18) if i not in fixed_indices]
        low, high = self._lower[passive], self._upper[passive]
        q[:, passive] = q[:, passive].clamp(low, high)
        scale = torch.where(self._prismatic[passive], q.new_tensor(slider_scale_m), q.new_tensor(1.))
        step_bound = torch.where(self._prismatic[passive], q.new_tensor(slider_step_m), q.new_tensor(angular_step_rad))
        # Reduced QR solves the full-rank passive system. Rank-deficient
        # closure uses the explicit regularized fallback below; this is not
        # the virtual-leg length/angle birth-mapping problem.
        iterations = 0
        rounds = torch.zeros(count, dtype=torch.long, device=self.device)
        damping = torch.full((count,), initial_damping_m2, dtype=torch.float64, device=self.device)
        for iteration in range(maximum_iterations):
            residual, jac = self.closure(q, jacobian=True)
            ready = residual.norm(dim=-1).amax(-1) <= closure_tolerance_m
            active = ~ready & (rounds < iteration_budget_by_row)
            if not active.any():
                break
            rounds += active.long()
            matrix = jac[:, :, passive] * scale[None, None]
            rhs = -residual.flatten(1)
            finite = torch.isfinite(matrix).all(dim=(-2, -1)) & torch.isfinite(rhs).all(-1)
            eye = torch.eye(matrix.shape[-2], matrix.shape[-1], dtype=torch.float64,
                            device=self.device).expand(count, -1, -1)
            # Decompositions reject NaN before a later mask can take effect.
            # Only internal algebra sees this placeholder; finite=False keeps
            # the invalid lane's update zero and its unresolved pose rejected.
            safe_matrix = torch.where(finite[:, None, None], matrix, eye)
            singular = torch.linalg.svdvals(safe_matrix)
            full_rank = finite & (singular[:, -1] > self._rank_rtol * singular[:, 0])
            # A spring slide can be a null direction of the six loop
            # constraints. It remains at its seed coordinate instead of making
            # the whole row fail or receiving an arbitrary damped displacement.
            # Full-rank rows use reduced QR; rank-deficient rows use the
            # explicit Levenberg fallback with the same QR-space scaling.
            qr_matrix = torch.where(full_rank[:, None, None], matrix, eye)
            qr_rhs = torch.where(full_rank[:, None], rhs, 0.)
            orthogonal, triangular = torch.linalg.qr(qr_matrix, mode='reduced')
            qr_solution = torch.linalg.solve_triangular(
                triangular, orthogonal.transpose(-1, -2) @ qr_rhs[:, :, None],
                upper=True).squeeze(-1)
            # Rank-deficient closure is possible when a compliant slide is a
            # null direction of the loop equations. A small explicit
            # Levenberg term keeps the minimum-norm fallback finite without
            # changing the direction used by the full-rank QR path.
            eye_k = torch.eye(matrix.shape[-1], dtype=torch.float64,
                              device=self.device).expand(count, -1, -1)
            safe_rhs = torch.where(finite[:, None], rhs, 0.)
            normal = safe_matrix.transpose(-1, -2) @ safe_matrix
            normal += eye_k * damping[:, None, None]
            damped = torch.linalg.solve(
                normal, (safe_matrix.transpose(-1, -2) @ safe_rhs[:, :, None])).squeeze(-1)
            solved = torch.where(full_rank[:, None], qr_solution, damped)
            solved = torch.where(finite[:, None], solved, 0.)
            delta = solved * scale
            fraction = (step_bound / delta.abs().clamp_min(1e-30)).amin(-1).clamp_max(1)
            delta *= fraction[:, None]
            candidate = q.clone()
            candidate[:, passive] = (q[:, passive] + delta).clamp(low, high)
            candidate_residual = self.closure(candidate)
            improved = (candidate_residual.square().sum((-1, -2))
                        < residual.square().sum((-1, -2))) & active
            q = torch.where(improved[:, None], candidate, q)
            # The QR lane has no damping term; only the rank-deficient
            # fallback adapts its explicit Levenberg regularization.
            damping = torch.where(full_rank, damping,
                                  torch.where(improved, damping * .3, damping * 10)
                                  .clamp(1e-14, 1e4))
            iterations = iteration + 1
        residual = self.closure(q).norm(dim=-1)
        within = ((q >= self._lower - 1e-12) & (q <= self._upper + 1e-12)).all(-1)
        if not torch.equal(q[:, fixed_indices], prescribed_positions):
            raise RuntimeError('closed-chain solver altered its explicitly prescribed coordinates')
        maximum = residual.amax(-1)
        return q, residual, maximum, within, within & (maximum <= closure_tolerance_m), iterations, rounds

    @torch.no_grad()
    def solve_passive(self, seed_tree_position, prescribed_policy_positions, **options):
        q, residual, maximum, within, converged, iterations, _ = self._solve_fixed(
            seed_tree_position, self.policy_tree_indices, prescribed_policy_positions, **options)
        return PassiveClosure(q, residual, maximum, within, converged, iterations,
                               prescribed_policy_positions.detach().clone(), self.manifest_sha256, self.model_sha256)

    @torch.no_grad()
    def solve_posture(self, seed_tree_position, hip_positions, inner_knee_angle_rad, **options):
        """Geometric reset IK; the actor still controls only coaxial outputs."""
        self._positions(seed_tree_position)
        count = len(seed_tree_position)
        if (not isinstance(hip_positions, torch.Tensor) or hip_positions.shape != (count, 2)
                or hip_positions.dtype != torch.float64 or hip_positions.device != self.device
                or not torch.isfinite(hip_positions).all() or not isinstance(inner_knee_angle_rad, torch.Tensor)
                or inner_knee_angle_rad.shape != (count, 2)):
            raise ValueError('posture reset needs prescribed left/right output hips and geometric knee angles')
        hip_indices = (self.policy_tree_indices[0], self.policy_tree_indices[2])
        wheels = self.policy_tree_indices[4:]
        fixed = hip_indices + self.knee_tree_indices + wheels
        values = torch.cat((hip_positions, self.native_knee_positions(inner_knee_angle_rad),
                             seed_tree_position[:, wheels]), -1)
        q, residual, maximum, within, converged, iterations, rounds = self._solve_fixed(
            seed_tree_position, fixed, values, **options)
        return PostureClosure(q, residual, maximum, within, converged, iterations,
                               inner_knee_angle_rad.detach().clone(), hip_positions.detach().clone(),
                               q[:, self.policy_tree_indices].clone(), self.manifest_sha256, self.model_sha256,
                               solver_rounds_by_row=rounds)

    @torch.no_grad()
    def continue_posture(self, seed_tree_position, hip_positions, inner_knee_angle_rad, *,
                         maximum_segment_angle_rad, maximum_segments, closure_tolerance_m, maximum_iterations):
        """Trace one prescribed request from a closed seed, without retries.

        Each row has its own total nonlinear-update budget across all segments.
        A failed segment ends that row; other rows retain their own budget. The
        returned endpoint always keeps the complete original requested hips,
        knees and wheel coordinates, even when the attempted prefix failed.
        """
        self._positions(seed_tree_position)
        n = len(seed_tree_position)
        if (not isinstance(hip_positions, torch.Tensor) or hip_positions.shape != (n, 2)
                or hip_positions.dtype != torch.float64 or hip_positions.device != self.device
                or not torch.isfinite(hip_positions).all()
                or not isinstance(inner_knee_angle_rad, torch.Tensor) or inner_knee_angle_rad.shape != (n, 2)
                or type(maximum_segment_angle_rad) not in (int, float)
                or not math.isfinite(maximum_segment_angle_rad) or not 0 < maximum_segment_angle_rad <= math.pi/4
                or type(maximum_segments) is not int or not 1 <= maximum_segments <= 256
                or type(maximum_iterations) is not int or not 1 <= maximum_iterations <= 256
                or type(closure_tolerance_m) not in (int, float)
                or not math.isfinite(closure_tolerance_m) or closure_tolerance_m <= 0):
            raise ValueError('posture continuation needs explicit bounded segments,targets and total per-row budgets')
        final_knee = self.native_knee_positions(inner_knee_angle_rad)
        initial_residual = self.closure(seed_tree_position).norm(dim=-1).amax(-1)
        if (((seed_tree_position < self._lower) | (seed_tree_position > self._upper)).any()
                or (initial_residual > closure_tolerance_m).any()):
            raise ValueError('posture continuation starts only from an already closed,source-limited seed')
        hip_indices = (self.policy_tree_indices[0], self.policy_tree_indices[2])
        initial_hip = seed_tree_position[:, hip_indices]
        initial_knee = self.inner_knee_angles(seed_tree_position)
        delta = torch.cat((hip_positions - initial_hip, inner_knee_angle_rad - initial_knee), -1).abs().amax(-1)
        requested = (delta / maximum_segment_angle_rad).ceil().clamp_min(1)
        if (requested > maximum_segments).any():
            raise ValueError('requested posture path exceeds its explicit segment cap;do not shorten the target')
        requested = requested.long()
        completed = torch.zeros(n, dtype=torch.long, device=self.device)
        attempted, rounds = completed.clone(), completed.clone()
        failed = torch.zeros(n, dtype=torch.bool, device=self.device)
        fraction = torch.zeros(n, dtype=torch.float64, device=self.device)
        current = seed_tree_position.detach().clone()
        last_attempt, last_residual = current.clone(), initial_residual.clone()
        kernel_rounds = 0
        for index in range(1, int(requested.max()) + 1):
            active = ~failed & (completed < requested) & (rounds < maximum_iterations)
            if not active.any():
                break
            ids = active.nonzero().flatten()
            phase = index / requested[ids].double()
            final = index == requested[ids]
            hip = torch.where(final[:, None], hip_positions[ids],
                              initial_hip[ids] + phase[:, None] * (hip_positions[ids] - initial_hip[ids]))
            knee = torch.where(final[:, None], inner_knee_angle_rad[ids],
                               initial_knee[ids] + phase[:, None] * (inner_knee_angle_rad[ids] - initial_knee[ids]))
            result = self.solve_posture(current[ids], hip, knee, closure_tolerance_m=closure_tolerance_m,
                                         maximum_iterations=maximum_iterations,
                                         iteration_budget_by_row=maximum_iterations - rounds[ids])
            rounds[ids] += result.solver_rounds_by_row
            attempted[ids] += 1
            fraction[ids], last_attempt[ids], last_residual[ids] = phase, result.tree_position, result.maximum_residual_m
            kernel_rounds += result.iterations
            good = ids[result.converged]
            current[good] = result.tree_position[result.converged]
            completed[good] = index
            failed[ids[~result.converged]] = True
        endpoint = last_attempt.clone()
        endpoint[:, hip_indices] = hip_positions
        endpoint[:, self.knee_tree_indices] = final_knee
        endpoint[:, self.policy_tree_indices[4:]] = seed_tree_position[:, self.policy_tree_indices[4:]]
        residual = self.closure(endpoint).norm(dim=-1)
        maximum = residual.amax(-1)
        within = ((endpoint >= self._lower - 1e-12) & (endpoint <= self._upper + 1e-12)).all(-1)
        converged = (completed == requested) & within & (maximum <= closure_tolerance_m)
        closure = PostureClosure(endpoint, residual, maximum, within, converged, kernel_rounds,
                                  inner_knee_angle_rad.detach().clone(), hip_positions.detach().clone(),
                                  endpoint[:, self.policy_tree_indices].clone(), self.manifest_sha256, self.model_sha256,
                                  solver_rounds_by_row=rounds)
        exhausted = (rounds >= maximum_iterations) & ~converged
        return PostureContinuation(closure, requested, completed, attempted, fraction, last_attempt, last_residual,
                                    exhausted, maximum_segment_angle_rad, maximum_segments)

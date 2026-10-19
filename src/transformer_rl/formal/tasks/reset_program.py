"""Private full-domain reset preparation before native scene admission.

Sampling, source geometry and native writes have separate boundaries. A failed
candidate remains in its requested row; there is no retry, standing fallback,
shortened course or physical qualification from a successful source solve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import math

import torch

from .trajectory import CommandTrajectory
from .terrain import TerrainRebuildRequired
from ..control.actuation import OutputFeedbackController
from .recovery import RecoveryCandidates, RecoveryPoseCompiler
from .reset_collision import AssetCollisionGeometry, CollisionPlacement, FlatSpawnPatch
from .commands import TaskCommandPlan
from .curriculum import ContinuousTaskSampler, TaskDraw, TaskFamily
from .terrain import ContinuousTerrainGenerator, TerrainCourse, TerrainExtent


@dataclass(frozen=True)
class ResetProgramTuning:
    ordinary_horizon_s: float
    extended_horizon_s: float
    trajectory_duration_s: tuple[float, float]
    request_interval_s: tuple[float, float]
    terrain_margin_m: float
    spawn_patch_half_xy_m: tuple[float, float]
    xy_jitter_m: float
    roll_pitch_jitter_rad: float
    inner_knee_jitter_rad: float
    leg_pitch_jitter_rad: float
    standing_clearance_m: float
    birth_roll_correction_rad: float
    wheel_support_tolerance_m: float

    def __post_init__(self):
        for name in ('trajectory_duration_s', 'request_interval_s', 'spawn_patch_half_xy_m'):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value)
                    or name != 'spawn_patch_half_xy_m' and value[0] >= value[1]):
                raise ValueError('reset program requires explicit finite positive range and footprint inputs')
            object.__setattr__(self, name, tuple(value))
        scalar = {k: v for k, v in vars(self).items() if not isinstance(v, tuple)}
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in scalar.values())
                or min(self.ordinary_horizon_s, self.extended_horizon_s, self.terrain_margin_m,
                       self.birth_roll_correction_rad, self.wheel_support_tolerance_m) <= 0
                or self.extended_horizon_s < self.ordinary_horizon_s
                or any(abs(v / .01 - round(v / .01)) > 1e-7
                       for v in (self.ordinary_horizon_s, self.extended_horizon_s))
                or self.birth_roll_correction_rad > math.pi / 4
                or self.standing_clearance_m >= self.wheel_support_tolerance_m
                or self.xy_jitter_m >= min(self.spawn_patch_half_xy_m)):
            raise ValueError('reset horizons, perturbations and bounded support recipe are inconsistent')


def _tensor_versions(value):
    if isinstance(value, torch.Tensor):
        return ((id(value), value._version),)
    if isinstance(value, CommandTrajectory):
        return tuple(pair for name in ('start', 'end', 'duration_s', 'kind',
                                        'frequency_start_hz', 'frequency_end_hz')
                     for pair in _tensor_versions(getattr(value, name)))
    if isinstance(value, TerrainCourse):
        # Source courses contain only validated immutable numeric tuples. Do
        # not walk millions of mesh coordinates to check tensor ownership.
        return ()
    if is_dataclass(value):
        return tuple(pair for f in fields(value) if not f.name.startswith('_')
                     for pair in _tensor_versions(getattr(value, f.name)))
    if isinstance(value, (tuple, list)):
        return tuple(pair for item in value for pair in _tensor_versions(item))
    return ()


@dataclass(frozen=True)
class PreparedTaskReset:
    draw: TaskDraw
    courses: tuple
    patch: FlatSpawnPatch
    commands: TaskCommandPlan
    recovery: RecoveryCandidates
    placement: CollisionPlacement
    required_two_wheels: torch.Tensor
    applied_birth_roll_rad: torch.Tensor
    source_ready: torch.Tensor
    tuning: ResetProgramTuning
    physical_qualification: bool = False
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_versions', _tensor_versions(self))

    @torch.no_grad()
    def native_plan(self, geometry, controller, *, actual_course_sha256, actual_base_com_body_m):
        actual = tuple(actual_course_sha256)
        requested = self.patch.course_sha256
        if len(actual) != len(requested):
            raise ValueError('native reset needs every actual authored course identity')
        changed = [i for i, (a, b) in enumerate(zip(actual, requested)) if a != b]
        if changed:
            raise TerrainRebuildRequired(changed, actual, requested)
        return self.encode_plan(geometry, controller, base_com_body_m=actual_base_com_body_m)

    @torch.no_grad()
    def encode_plan(self, geometry, controller, *, base_com_body_m):
        """Recheck encoded coordinates and use the actual randomized base COM.

        This returns data only. Scene construction, native collider equivalence
        and physical acceptance are still required by the owning environment.
        """
        if self._versions != _tensor_versions(self):
            raise ValueError('prepared reset numeric inputs changed after source evaluation')
        if not isinstance(geometry, AssetCollisionGeometry) or not isinstance(controller, OutputFeedbackController):
            raise TypeError('native plan binding requires actual selected geometry and output controller')
        n, device = len(self.draw.family), geometry.device
        kinematics = geometry.kinematics
        if (controller.num_envs != n or controller.device != device
                or not torch.equal(controller.nominal, torch.tensor(kinematics.binding.nominal_policy_position_rad,
                                                                    dtype=torch.float32, device=device))
                or kinematics.manifest_sha256 != self.recovery.closure.source_manifest_sha256
                or kinematics.model_sha256 != self.recovery.closure.source_model_sha256):
            raise ValueError('native reset geometry or actual controller differs from the prepared source')
        if not self.source_ready.all():
            raise ValueError('requested reset contains unresolved source closure, output hold or collision support')
        if (not isinstance(base_com_body_m, torch.Tensor) or base_com_body_m.shape != (n, 3)
                or base_com_body_m.device != device or base_com_body_m.dtype != torch.float64
                or not torch.isfinite(base_com_body_m).all()):
            raise ValueError('native reset requires finite float64 actual per-row base COM coordinates')
        plan, placement = geometry.reset_plan(
            self.recovery.closure.tree_position, self.placement.root_pose_xyzw[:, 3:],
            self.placement.root_pose_xyzw[:, :2], self.patch, self.commands,
            required_support=self.required_two_wheels, clearance_m=self._clearance(),
            loop_tolerance_m=self.recovery.tuning.loop_tolerance_m,
            wheel_support_tolerance_m=self.tuning.wheel_support_tolerance_m)
        q = plan.joint_position[:, list(kinematics.binding.policy_joint_indices)]
        hold = controller.preparation_hold(q, position_tolerance_rad=self.recovery.tuning.output_hold_tolerance_rad)
        if not hold.admitted.all():
            raise ValueError('encoded native reset pose is outside the actual preparation hold domain')
        # Body origin velocity and PhysX root COM velocity are different when
        # a randomized COM has an angular velocity. Use encoded orientation.
        orientation = plan.root_pose_xyzw[:, 3:].double()
        orientation /= orientation.norm(dim=-1, keepdim=True)
        rotation = geometry._rotation(orientation)
        omega = (rotation @ self.recovery.angular_velocity_body_rad_s[..., None]).squeeze(-1)
        offset = (rotation @ base_com_body_m[..., None]).squeeze(-1)
        linear = (rotation @ self.recovery.origin_velocity_body_m_s[..., None]).squeeze(-1)
        velocity = torch.cat((linear + torch.linalg.cross(omega, offset), omega), -1).float()
        if not torch.isfinite(velocity).all():
            raise ValueError('encoded native reset COM velocity is nonfinite')
        return replace(plan, root_com_velocity=velocity, preparation_hold=hold, source_courses=self.courses), placement

    def _clearance(self):
        return self.recovery.release_clearance_m + self.tuning.standing_clearance_m


class TaskResetPreparer:
    """Checkpoint all three RNG streams; advance once for every requested batch."""

    def __init__(self, tasks, terrain, recovery, geometry, tuning, *, pose_seed):
        if (not isinstance(tasks, ContinuousTaskSampler) or not isinstance(terrain, ContinuousTerrainGenerator)
                or not isinstance(recovery, RecoveryPoseCompiler) or not isinstance(geometry, AssetCollisionGeometry)
                or geometry.kinematics is not recovery.kinematics or tasks.device != geometry.device
                or not isinstance(tuning, ResetProgramTuning)
                or type(pose_seed) is not int or not 0 <= pose_seed < 2**63
                or tuning.spawn_patch_half_xy_m[0] >= terrain.tuning.entry_m):
            raise ValueError('reset preparation needs one selected asset, dedicated samplers and an explicit spawn recipe')
        self.tasks, self.terrain, self.recovery, self.geometry, self.tuning = tasks, terrain, recovery, geometry, tuning
        self.device = geometry.device
        self.generator = torch.Generator(device=self.device).manual_seed(pose_seed)
        self.attempted_batches, self.requested_rows, self.completed_batches = 0, 0, 0

    def _uniform(self, n, width=None):
        shape = (n,) if width is None else (n, width)
        return torch.rand(shape, dtype=torch.float64, device=self.device, generator=self.generator)

    def _inputs(self, n, origins, controller):
        if (not isinstance(origins, torch.Tensor) or origins.shape != (n, 3) or origins.dtype != torch.float64
                or origins.device != self.device or not torch.isfinite(origins).all()
                or not isinstance(controller, OutputFeedbackController) or controller.num_envs != n
                or controller.device != self.device
                or not torch.equal(controller.nominal,
                                   torch.tensor(self.geometry.kinematics.binding.nominal_policy_position_rad,
                                                dtype=torch.float32, device=self.device))):
            raise ValueError('reset preparation requires finite actual clone origins and a matching output-controller batch')

    def _draw(self, draw):
        if (not isinstance(draw, TaskDraw) or not isinstance(draw.family, torch.Tensor)
                or draw.family.ndim != 1 or draw.family.numel() < 1):
            raise ValueError('reset preparation requires nonempty original continuous task rows')
        n = len(draw.family)
        integer = {'family', 'stratum', 'terrain_kind', 'recovery_posture', 'trajectory_kind'}
        for f in fields(draw):
            value = getattr(draw, f.name)
            dtype = torch.long if f.name in integer else torch.bool if f.name == 'nominal' else torch.float32
            shape = (n, 4) if f.name == 'requested_command' else (n,)
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device
                    or value.dtype != dtype or dtype == torch.float32 and not torch.isfinite(value).all()):
                raise ValueError('continuous task draw layout, dtype or finite source rows differ')
        bounds = {'family': len(TaskFamily), 'stratum': 3, 'terrain_kind': 8,
                  'recovery_posture': 16, 'trajectory_kind': 7}
        if any(((getattr(draw, k) < 0) | (getattr(draw, k) >= high)).any() for k, high in bounds.items()):
            raise ValueError('continuous task draw contains an undeclared task or recovery identity')
        d, command = self.tasks.domain, draw.requested_command
        if (command[:, 1].abs().gt(1e-8).any()
                or any(((command[:, column] < low) | (command[:, column] > high)).any()
                       for column, (low, high) in ((0, d.forward_m_s), (2, d.yaw_rad_s), (3, d.height_m)))
                or any(((getattr(draw, name) < low) | (getattr(draw, name) > high)).any()
                       for name, (low, high) in (('slope_rad', d.slope_rad), ('roughness_m', d.roughness_m),
                                                  ('step_up_m', d.step_up_m), ('step_down_m', d.step_down_m)))
                or ((draw.jump_height_m != 0) & ((draw.jump_height_m < d.jump_m[0])
                                                 | (draw.jump_height_m > d.jump_m[1]))).any()):
            raise ValueError('continuous task draw leaves its declared unprojected request domain')
        return n

    def sample(self, count, stage, origins_world_m, controller):
        if type(count) is not int or count < 1:
            raise ValueError('reset sampling requires positive explicit batch size')
        self._inputs(count, origins_world_m, controller)
        return self.prepare_draw(self.tasks.sample(count, stage), origins_world_m, controller)

    @torch.no_grad()
    def prepare_draw(self, draw, origins_world_m, controller):
        n = self._draw(draw)
        self._inputs(n, origins_world_m, controller)
        draw = TaskDraw(*(getattr(draw, f.name).detach().clone() for f in fields(draw)))
        self.attempted_batches += 1
        self.requested_rows += n
        t, domain = self.tuning, self.tasks.domain
        extended = (draw.family == int(TaskFamily.REQUEST)) | (draw.family == int(TaskFamily.RECOVERY))
        horizon = torch.where(extended, t.extended_horizon_s, t.ordinary_horizon_s).double()
        speed = max(abs(v) for v in domain.forward_m_s)
        # 共用最长回合的完整几何域，避免30s/60s平地回合反复拆建同一地面。
        # 各任务的实际时限仍由horizon保留，既不缩域也不增加训练转移预算。
        extent = TerrainExtent.for_horizon(max(t.ordinary_horizon_s, t.extended_horizon_s),
                                           speed, t.terrain_margin_m)
        extents = (extent,) * n
        courses = self.terrain.from_draw(draw, extents)
        bounds = torch.tensor([[[-t.spawn_patch_half_xy_m[0], t.spawn_patch_half_xy_m[0]],
                                 [-t.spawn_patch_half_xy_m[1], t.spawn_patch_half_xy_m[1]]]],
                              dtype=torch.float64, device=self.device).repeat(n, 1, 1)
        patch = FlatSpawnPatch.from_courses(courses, origins_world_m, bounds,
                                           torch.zeros(n, dtype=torch.float64, device=self.device))
        active = (~draw.nominal).double()
        yaw = (self._uniform(n) * 2 - 1) * math.pi * active
        xy = origins_world_m[:, :2] + (self._uniform(n, 2) * 2 - 1) * t.xy_jitter_m * active[:, None]
        offsets = tuple((self._uniform(n, 2) * 2 - 1) * size * active[:, None] for size in
                        (t.roll_pitch_jitter_rad, t.inner_knee_jitter_rad, t.leg_pitch_jitter_rad))
        is_recovery = draw.family == int(TaskFamily.RECOVERY)
        posture = torch.where(is_recovery, draw.recovery_posture, 0)
        candidate = self.recovery.compile(posture, yaw, controller, roll_pitch_offset_rad=offsets[0],
                                           inner_knee_offset_rad=offsets[1], leg_pitch_offset_rad=offsets[2])
        orientation = candidate.requested_root_orientation_xyzw.clone()
        two_wheels = draw.nominal & (posture <= 1)
        correction = torch.zeros(n, dtype=torch.float64, device=self.device)
        clearance = candidate.release_clearance_m + t.standing_clearance_m
        options = {'clearance_m': clearance, 'loop_tolerance_m': self.recovery.tuning.loop_tolerance_m,
                   'wheel_support_tolerance_m': t.wheel_support_tolerance_m}
        if two_wheels.any():
            balanced, proposed, bracketed = self.geometry.balance_two_wheels(
                candidate.closure.tree_position, orientation, xy, patch,
                roll_correction_bounds_rad=orientation.new_tensor((-t.birth_roll_correction_rad,
                                                                    t.birth_roll_correction_rad)).repeat(n, 1), **options)
            selected = two_wheels & bracketed
            orientation[selected] = balanced.root_pose_xyzw[selected, 3:]
            correction[selected] = proposed[selected]
        placement = self.geometry.place(candidate.closure.tree_position, orientation, xy, patch, **options)
        initial = draw.requested_command.new_tensor((0., 0., 0., domain.nominal_height_m)).repeat(n, 1)
        duration = t.trajectory_duration_s[0] + self._uniform(n) * (t.trajectory_duration_s[1] - t.trajectory_duration_s[0])
        period = t.request_interval_s[0] + self._uniform(n) * (t.request_interval_s[1] - t.request_interval_s[0])
        commands = TaskCommandPlan.from_draw(draw, initial, duration.float(), horizon, period, domain=domain)
        ready = (candidate.closure.converged & candidate.preparation_hold.admitted & placement.geometry_ready
                 & (~two_wheels | placement.bilateral_wheel_support))
        self.completed_batches += 1
        return PreparedTaskReset(draw, courses, patch, commands, candidate, placement, two_wheels,
                                   correction, ready, t)

    def state_dict(self):
        return {'tuning': asdict(self.tuning), 'recovery_tuning': asdict(self.recovery.tuning),
                'source_manifest_sha256': self.geometry.kinematics.manifest_sha256,
                'source_model_sha256': self.geometry.kinematics.model_sha256,
                'tasks': self.tasks.state_dict(), 'terrain': self.terrain.state_dict(),
                'pose_rng': self.generator.get_state().clone(), 'attempted_batches': self.attempted_batches,
                'requested_rows': self.requested_rows, 'completed_batches': self.completed_batches}

    def load_state_dict(self, state):
        expected = self.state_dict()
        stable = ('tuning', 'recovery_tuning', 'source_manifest_sha256', 'source_model_sha256')
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[k] != expected[k] for k in stable)
                or any(type(state[k]) is not int or state[k] < 0
                       for k in ('attempted_batches', 'requested_rows', 'completed_batches'))
                or not state['completed_batches'] <= state['attempted_batches'] <= state['requested_rows']):
            raise ValueError('reset program checkpoint source, recipe or attempt accounting differs')
        tasks = ContinuousTaskSampler(0, self.device, self.tasks.domain)
        terrain = ContinuousTerrainGenerator(0, self.terrain.tuning)
        generator = torch.Generator(device=self.device)
        tasks.load_state_dict(state['tasks'])
        terrain.load_state_dict(state['terrain'])
        generator.set_state(state['pose_rng'].detach().cpu().clone())
        # Learner checkpoints and the reset source share these sampler objects.
        # Keep their identity after validating the complete candidate state.
        self.tasks.load_state_dict(tasks.state_dict())
        self.terrain.load_state_dict(terrain.state_dict())
        self.generator = generator
        self.attempted_batches, self.requested_rows, self.completed_batches = (
            state[k] for k in ('attempted_batches', 'requested_rows', 'completed_batches'))

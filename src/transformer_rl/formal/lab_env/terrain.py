"""Lab terrain-course composition; policy and PPO remain independent."""
from __future__ import annotations

from pathlib import Path

import torch

from ..control.observation import CommandMode
from ..simulation.ground import WheelCollisionCentres
from ..tasks.reward import RewardPhase
from ..tasks.terrain_courses import TerrainCourseRouter
from ..tasks.terrain_query import StaticTerrainQuery
from ..tasks.trajectory import DriveEnvelope
from .course_tasks import LabCourseTasks
from .skill_fsm import StairState


class LabTerrainCourse:
    def __init__(self, env, manifest):
        self.env = env
        self.router = TerrainCourseRouter(manifest, env.num_envs, seed=int(env.cfg.seed or 0), device=env._device)
        directory = Path(manifest).parent
        self.query = StaticTerrainQuery(directory / 'geometry.npz', directory / 'query_index.npz', device=env._device)
        kin = env._recovery_reset.kinematics
        wheels = WheelCollisionCentres.from_urdf(
            (env._asset.directory / env._asset.manifest['urdf']).read_bytes(), env._binding)
        nominal = kin.forward(kin.nominal_tree_position[None])
        indices = list(kin.binding.wheel_body_indices)
        centers = nominal.body_position_m[0, indices] + torch.einsum(
            'nij,nj->ni', nominal.body_rotation[0, indices], nominal.body_position_m.new_tensor(wheels.local_xyz_m))
        envelope = DriveEnvelope(wheels.radius_m[0], abs(float(centers[0, 1] - centers[1, 1])))
        nominal_spawn = env._recovery_reset.pose(torch.tensor([0]), torch.tensor([0]),
                                                torch.zeros(1, dtype=torch.float64),
                                                torch.zeros(1, 3, dtype=torch.float64))
        self.nominal_spawn_height_m = float(nominal_spawn.root_pose_xyzw[0, 2])
        self.tasks = LabCourseTasks(env.num_envs, envelope, seed=int(env.cfg.seed or 0), device=env._device)
        self.body_contact_allowed = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
        self.wheel_ground = torch.zeros(env.num_envs, 2, device=env._device)
        self.wheel_material = torch.zeros_like(self.wheel_ground)
        self._wheel_offsets = torch.tensor(wheels.local_xyz_m, dtype=torch.float32, device=env._device)
        self._ground_token = None
        # Match IsaacLab's default GridPatternCfg ordering: X changes fastest.
        x, y = torch.meshgrid(torch.linspace(-.5, .5, 11, device=env._device),
                              torch.linspace(-.3, .3, 7, device=env._device), indexing='xy')
        self._grid_offsets = torch.stack((x.flatten(), y.flatten()), -1)
        self._lookahead = torch.tensor([0., .25, -.25, .8, -.8], device=env._device)

    def _wheel_centres(self):
        env = self.env
        poses = env._as_torch(env.robot.data.body_link_pose_w).index_select(1, env._wheel_body_indices)
        quaternion = poses[..., 3:]
        offset = self._wheel_offsets[None].expand(env.num_envs, -1, -1)
        vector, scalar = quaternion[..., :3], quaternion[..., 3:]
        cross = 2 * torch.cross(vector, offset, dim=-1)
        return poses[..., :3] + offset + scalar * cross + torch.cross(vector, cross, dim=-1)

    def _yaw_offsets(self, center, offsets):
        quaternion = self.env._root_link_quat()
        x, y, z, w = quaternion.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        delta = torch.stack((cosine * offsets[..., 0] - sine * offsets[..., 1],
                             sine * offsets[..., 0] + cosine * offsets[..., 1]), -1)
        points = center[:, None, :].expand(-1, len(offsets), -1).clone()
        points[..., :2] += delta
        points[..., 2] += 5
        return points

    def ground(self):
        env = self.env
        if self._ground_token == env._physical_tick:
            return env._ground_at_root, self.wheel_ground
        points = torch.cat((env._root_link_position()[:, None], self._wheel_centres()), 1)
        points[..., 2] += 5
        heights, valid, _, friction = self.query(points, self.router.tile[:, None].expand(-1, 3))
        if not valid.all():
            raise RuntimeError('course root/wheel query has no physical terrain support')
        env._ground_at_root.copy_(heights[:, 0])
        self.wheel_ground.copy_(heights[:, 1:])
        self.wheel_material.copy_(friction[:, 1:])
        self._ground_token = env._physical_tick
        return env._ground_at_root, self.wheel_ground

    def scan_77(self):
        env = self.env
        points = self._yaw_offsets(env._root_link_position(), self._grid_offsets)
        heights, valid, _, _ = self.query(points, self.router.tile[:, None].expand(-1, 77))
        if not valid.all():
            raise RuntimeError('privileged terrain scan left the authored collision geometry')
        relative = env._root_link_position()[:, 2, None] - heights - env._task.nominal_height_m
        return (5 * relative).clamp(-1, 1), heights[:, 38]

    def wheel_lookahead(self):
        """Current, near +/-0.25m, and far +/-0.8m samples for both wheels."""
        env = self.env
        centres = self._wheel_centres()
        offsets = torch.stack((self._lookahead, torch.zeros_like(self._lookahead)), -1)
        points = torch.stack([self._yaw_offsets(centres[:, side], offsets) for side in range(2)], 1)
        heights, valid, _, _ = self.query(points, self.router.tile[:, None, None].expand(-1, 2, 5))
        if not valid.all():
            raise RuntimeError('wheel lookahead left the authored collision geometry')
        reverse = env._command_base[:, 0] < 0
        current = heights[:, :, 0].mean(-1)
        near = torch.where(reverse[:, None], heights[:, :, 2], heights[:, :, 1]).mean(-1)
        far = torch.where(reverse[:, None], heights[:, :, 4], heights[:, :, 3]).mean(-1)
        return current, near, far

    def place_reset(self, ids, task, pose):
        if task is None:
            raise ValueError('terrain course requires the formal TaskDraw sampler')
        env = self.env
        origins, ground = self.router.reset(ids, task.terrain_kind, stage=env._task_sampler.stage,
                                             command_forward=task.command[:, 0])
        pose[:, :3] -= env._as_torch(env.scene.env_origins).index_select(0, ids)
        pose[:, :3] += origins
        pose[:, 2] = origins[:, 2] + ground + self.nominal_spawn_height_m
        reverse = (task.command[:, 0] < 0) & (task.terrain_kind != 0)
        pose[reverse, 5] = 1
        pose[reverse, 6] = 0
        self._ground_token = None

    def reset_commands(self, ids, task):
        env = self.env
        self.tasks.reset(ids, task, env._clock())
        env._command[ids] = self.tasks.stream.current[ids]
        env._command_base[ids] = env._command[ids]
        env._command_mode[ids] = self.tasks.base_mode[ids]
        env._base_command_mode[ids] = self.tasks.base_mode[ids]
        env._request_height[ids] = 0
        self.body_contact_allowed[ids] = False

    def advance_skill(self):
        env = self.env
        admission, mode, rise, _ = self.tasks.step(env._clock(), env._completed_estimate, env._ground_at_root)
        env._command_base.copy_(admission.admitted)
        env._base_command_mode.copy_(mode)
        current, near, far = self.wheel_lookahead()
        step = (near - current).clamp_min(0)
        state = env._last_state
        load = env._last_sample.terrain_contact_force_world_n[:, env._base_body_index].norm(dim=-1)
        # Base support is intentional only during the bounded BODY_BUMP state.
        load = torch.where(env._skill_fsm.stair.state == int(StairState.BODY_BUMP), 0, load)
        stair = env._skill_fsm.stair.step(
            step_height_m=step, wheel_force_n=state.wheel_ground_force_n,
            body_height_m=state.height_above_ground_m, tilt_rad=env._last_tilt,
            base_contact_force_n=load, command_height_m=admission.admitted[:, 3],
            auto_enabled=(mode != int(CommandMode.RECOVERY)) & (mode != int(CommandMode.JUMP)))
        active = stair.mode != int(CommandMode.NORMAL)
        env._command.copy_(admission.admitted)
        env._command[:, 0] *= torch.where(active, stair.speed_scale, 1)
        # Far samples slow the approach; near samples classify the actual rise.
        # A two-step aggregate is never misclassified as one 400mm wall.
        edge = ((far - current).abs() > .04) | ((near - current) < -.04)
        env._command[:, 0] *= torch.where(edge & ~active, .6, 1.)
        env._command[:, 3] = torch.where(active, stair.height_command_m, env._command[:, 3])
        env._command_mode.copy_(torch.where(active, stair.mode, mode))
        env._request_height.copy_(rise)
        self.body_contact_allowed.copy_(stair.body_contact_allowed)

    def observe_reward(self, sample, state, boundary):
        env = self.env
        self.tasks.observe_physics(state.wheel_ground_force_n, sample.root_com_velocity_world_m_s[:, 2],
                                   env._last_tilt, sample.root_link_position_world_m[:, 2])
        env._reward_phase.copy_(self.tasks.phase)
        curve = self.tasks.jump_reference()
        landing = ((self.tasks.phase == int(RewardPhase.LANDING))
                   | (self.tasks.phase == int(RewardPhase.SETTLING)))
        target = torch.where(landing, env._ground_at_root + env._command[:, 3], curve.height_world_z_m)
        speed = torch.where(landing, 0, curve.velocity_world_z_m_s)
        env._formal_reward.observe(
            sample, state, effort_nm=env._last_effort, episode_id=env._episode_id,
            control_epoch=env._control_epoch, command=env._command, mode=env._command_mode,
            phase=self.tasks.phase, jump_height_world_z_m=target, jump_velocity_world_z_m_s=speed,
            joint_boundary_shortfall_rad=boundary, body_contact_allowed=self.body_contact_allowed)

    def termination(self, terminated, state, tilt, base_contact):
        # Retain pose limits; only the intentional base-contact criterion differs.
        safe = ((state.height_above_ground_m >= self.env._task.minimum_height_m)
                & (tilt <= self.env._task.failure_tilt_rad))
        return torch.where(self.body_contact_allowed & base_contact & safe, False, terminated)

    def state_dict(self):
        return {'format': 'lab_terrain_course_v1', 'router': self.router.state_dict(), 'tasks': self.tasks.state_dict()}

    def restore_after_physical_reset(self, state):
        if state.get('format') != 'lab_terrain_course_v1':
            raise ValueError('terrain course checkpoint format differs')
        self.router.load_state_dict(state['router'])
        self.tasks.restore_after_physical_reset(state['tasks'], self.env._clock())

    def metrics(self):
        return {'atlas_sha256': self.router.identity,
                'level_histogram': torch.bincount(self.router.level, minlength=self.router.manifest['levels']).tolist(),
                'tile_histogram': torch.bincount(self.router.tile, minlength=len(self.router.kinds)).tolist(),
                'course_exits': int(self.router.course_exits.sum()),
                'scan_queries': self.query.queries, 'scan_misses': int(self.query.misses),
                'request_responses': self.tasks.request_counters.tolist()}

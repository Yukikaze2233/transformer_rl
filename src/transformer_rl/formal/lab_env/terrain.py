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
from .course_tasks import LabCourseTasks, RequestResponse, StairRequestProtocol
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
        self.stair_requests = StairRequestProtocol(env.num_envs, device=env._device)
        self.body_contact_allowed = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
        self.wheel_ground = torch.zeros(env.num_envs, 2, device=env._device)
        self.wheel_material = torch.zeros_like(self.wheel_ground)
        self._wheel_offsets = torch.tensor(wheels.local_xyz_m, dtype=torch.float32, device=env._device)
        self._ground_token = None
        # Match IsaacLab's default GridPatternCfg ordering: X changes fastest.
        x, y = torch.meshgrid(torch.linspace(-.5, .5, 11, device=env._device),
                              torch.linspace(-.3, .3, 7, device=env._device), indexing='xy')
        self._grid_offsets = torch.stack((x.flatten(), y.flatten()), -1)
        # Stair control has no privileged terrain sensor. These are the only
        # stair request sources; a future ToF driver fills the same tensors.
        self.manual_stair_request = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
        self.manual_stair_height_m = torch.zeros(env.num_envs, dtype=torch.float32, device=env._device)
        self.manual_stair_request_id = torch.full((env.num_envs,), -1, dtype=torch.long, device=env._device)
        self.stair_cancel_id = torch.full((env.num_envs,), -1, dtype=torch.long, device=env._device)
        self.stair_next_id = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)
        self.tof_step_height_m = torch.zeros(env.num_envs, dtype=torch.float32, device=env._device)
        self.tof_request_id = torch.full((env.num_envs,), -1, dtype=torch.long, device=env._device)

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

    def submit_stair_request(self, ids, height_m, request_id=None):
        """Submit one manual stair ID; it is consumed even if BUSY/invalid."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env._device)
        value = torch.as_tensor(height_m, dtype=torch.float32, device=self.env._device)
        if (ids.ndim != 1 or value.shape != ids.shape or (ids < 0).any()
                or (ids >= self.env.num_envs).any() or not torch.isfinite(value).all()):
            raise ValueError('stair request requires finite row heights')
        if request_id is None:
            request_id = self.stair_next_id[ids].clone()
            self.stair_next_id[ids] += 1
        request_id = torch.as_tensor(request_id, dtype=torch.long, device=self.env._device)
        if request_id.shape != ids.shape or (request_id < 0).any():
            raise ValueError('stair request ID must be nonnegative per row')
        self.manual_stair_request[ids] = True
        self.manual_stair_height_m[ids] = value
        self.manual_stair_request_id[ids] = request_id
        self.stair_cancel_id[ids] = -1

    def cancel_stair_request(self, ids, request_id):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env._device)
        request_id = torch.as_tensor(request_id, dtype=torch.long, device=self.env._device)
        if ids.shape != request_id.shape or (ids < 0).any() or (ids >= self.env.num_envs).any():
            raise ValueError('stair cancel requires matching row IDs')
        self.stair_cancel_id[ids] = request_id

    def submit_tof_step_height(self, height_m, request_id):
        value = torch.as_tensor(height_m, dtype=torch.float32, device=self.env._device)
        ids = torch.as_tensor(request_id, dtype=torch.long, device=self.env._device)
        if value.shape != (self.env.num_envs,) or ids.shape != value.shape or not torch.isfinite(value).all():
            raise ValueError('ToF step height and request ID must match all environment rows')
        self.tof_step_height_m.copy_(value)
        self.tof_request_id.copy_(ids)

    def place_reset(self, ids, task, pose):
        if task is None:
            raise ValueError('terrain course requires the formal TaskDraw sampler')
        env = self.env
        origins, ground = self.router.reset(ids, task.terrain_kind, stage=env._task_sampler.stage,
                                             command_forward=task.command[:, 0])
        pose[:, :3] -= env._as_torch(env.scene.env_origins).index_select(0, ids)
        pose[:, :3] += origins
        # Spawn on the declared approach, rather than at the tile's first rise.
        nonflat = task.terrain_kind != 0
        approach_x = self.router.starts[self.router.tile[ids]] + .5
        pose[:, 0] += torch.where(nonflat, approach_x, 0.)
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
        self.manual_stair_request[ids] = False
        self.manual_stair_height_m[ids] = 0
        self.manual_stair_request_id[ids] = -1
        self.stair_cancel_id[ids] = -1
        self.tof_step_height_m[ids] = 0
        self.tof_request_id[ids] = -1
        self.stair_requests.reset(ids, env._clock())

    def advance_skill(self):
        # Deployment-visible request decision: env._completed_estimate comes
        # from the preceding actor observation. Terrain truth is not a stair
        # sensor; manual/ToF input is latched in the fields below.
        env = self.env
        blocked = self.stair_requests.active_id >= 0
        admission, mode, rise, _ = self.tasks.step(
            env._clock(), env._completed_estimate, env._ground_at_root, blocked_by_stair=blocked)
        env._command_base.copy_(admission.admitted)
        env._base_command_mode.copy_(mode)
        # One-shot manual/ToF stair requests; jump has arbitration priority.
        request_id = torch.where(self.manual_stair_request, self.manual_stair_request_id,
                                 self.tof_request_id)
        height_request = torch.where(self.manual_stair_request, self.manual_stair_height_m,
                                     self.tof_step_height_m)
        estimate = env._completed_estimate
        if estimate is None:
            ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
            contact_probability = torch.zeros(env.num_envs, 2, device=env._device)
            tilt = torch.zeros(env.num_envs, device=env._device)
        else:
            age = env._clock() - estimate.available_at_s
            contact_probability, tilt = estimate.wheel_contact_probability, estimate.tilt_rad
            ready = (estimate.valid & torch.isfinite(age) & (age > 0)
                     & (age <= self.tasks.protocol.config.estimate_max_age_s + 1e-9)
                     & torch.isfinite(contact_probability).all(-1)
                     & ((contact_probability >= 0) & (contact_probability <= 1)).all(-1)
                     & torch.isfinite(tilt) & (tilt >= 0))
            # These inactive placeholders never become observations or labels.
            contact_probability = torch.where(ready[:, None], contact_probability, 0.)
            tilt = torch.where(ready, tilt, 0.)
        condition = self.stair_requests.step(
            now_s=env._clock(), request_id=request_id, height_m=height_request,
            cancel_id=self.stair_cancel_id, jump_active=self.tasks.active_id >= 0, ready=ready)
        # Each submitted packet is consumed once, including BUSY/invalid IDs.
        self.manual_stair_request.zero_()
        self.manual_stair_request_id.fill_(-1)
        self.tof_request_id.fill_(-1)
        self.stair_next_id.copy_(torch.maximum(
            self.stair_next_id, self.stair_requests.max_seen_id + 1))
        finished = condition.finished_id >= 0
        self.stair_cancel_id.masked_fill_(finished, -1)
        if finished.any():
            env._skill_fsm.stair.reset(finished.nonzero().flatten())
        stair = env._skill_fsm.stair.deployment_step(
            step_height_m=torch.where(condition.active, condition.height_m,
                                      torch.zeros_like(condition.height_m)),
            wheel_contact_probability=contact_probability, tilt_rad=tilt,
            command_height_m=admission.admitted[:, 3],
            manual_request=condition.active, manual_height_m=condition.height_m)
        self.stair_requests.clear(stair.exited)
        active = stair.mode != int(CommandMode.NORMAL)
        env._command.copy_(admission.admitted)
        env._command[:, 0] *= torch.where(active, stair.speed_scale, 1)
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
        # BODY_BUMP exempts only the intentional base-contact criterion. Height
        # and tilt failures remain terminal, and an already latched failure in
        # the caller is never cleared here.
        height_ok = state.height_above_ground_m >= self.env._task.minimum_height_m
        tilt_ok = tilt <= self.env._task.failure_tilt_rad
        allowed_contact = self.body_contact_allowed & base_contact & height_ok & tilt_ok
        return terminated & ~allowed_contact

    def state_dict(self):
        return {'format': 'lab_terrain_course_v2', 'router': self.router.state_dict(),
                'tasks': self.tasks.state_dict(), 'stair_requests': self.stair_requests.state_dict(),
                'stair_next_id': self.stair_next_id.clone()}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict)
                or set(state) != {'format', 'router', 'tasks', 'stair_requests', 'stair_next_id'}
                or state.get('format') != 'lab_terrain_course_v2'):
            raise ValueError('terrain course checkpoint format differs')
        next_id = state['stair_next_id']
        if (not isinstance(next_id, torch.Tensor) or next_id.shape != self.stair_next_id.shape
                or next_id.dtype != torch.long or (next_id < 0).any()
                or (next_id.to(self.env._device)
                    <= state['stair_requests']['max_seen_id'].to(self.env._device)).any()):
            raise ValueError('terrain course stair request watermarks differ')
        self.router.load_state_dict(state['router'])
        self.tasks.restore_after_physical_reset(state['tasks'], self.env._clock())
        self.stair_requests.restore_after_physical_reset(state['stair_requests'], self.env._clock())
        self.stair_next_id.copy_(next_id)

    def metrics(self):
        return {'atlas_sha256': self.router.identity,
                'task_kind_histogram': self.tasks.kind_counts.tolist(),
                'task_mode_histogram': self.tasks.mode_counts.tolist(),
                'level_histogram': torch.bincount(self.router.level, minlength=self.router.manifest['levels']).tolist(),
                'tile_histogram': torch.bincount(self.router.tile, minlength=len(self.router.kinds)).tolist(),
                'course_exits': int(self.router.course_exits.sum()),
                'scan_queries': self.query.queries, 'scan_misses': int(self.query.misses),
                'request_responses': self.tasks.request_counters.tolist(),
                'external_request_responses': self.tasks.external_request_counters.tolist()}

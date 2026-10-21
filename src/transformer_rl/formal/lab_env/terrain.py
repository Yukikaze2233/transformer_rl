"""Lab terrain-course composition; policy and PPO remain independent."""
from __future__ import annotations

from pathlib import Path
import json

import torch

from ..control.observation import CommandMode
from ..simulation.ground import WheelCollisionCentres
from ..tasks.reward import RewardPhase
from ..tasks.geometry import TerrainKind
from ..tasks.terrain_courses import TerrainCourseRouter
from ..tasks.terrain_query import StaticTerrainQuery, wheel_lookahead_points
from ..tasks.trajectory import DriveEnvelope
from .course_tasks import LabCourseTasks, RequestCommand, StairRequestProtocol


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
        self._wheel_radius = torch.tensor(wheels.radius_m, dtype=torch.float32, device=env._device)
        # Root-link-relative wheel centres from the selected closed chain. Do
        # not reuse the drive-envelope track width as a vertical datum.
        root_index = kin.binding.native_body_names.index('base_link')
        self._wheel_centres_root = (centers - nominal.body_position_m[0, root_index, None]).float().to(env._device)
        self._spawn_y_offset = self._measure_spawn_y_offsets()
        self._reset_clearance_m = float(env._recovery_reset.config.clearance_m)
        self._last_support_clearance = torch.full((env.num_envs,), float('nan'),
                                                  dtype=torch.float32, device=env._device)
        self._last_support_clearance_lr = torch.full((env.num_envs, 2), float('nan'),
                                                     dtype=torch.float32, device=env._device)
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
        self._pending_requests = {'jump': None, 'stair': None}
        # Training commands are sampled task intent, never inferred from scan
        # or contact truth. A new physical episode supplies at most one request.
        self._course_stair_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
        self._course_stair_height = torch.zeros(env.num_envs, device=env._device)
        self._last_stair_response = torch.zeros(env.num_envs, dtype=torch.long, device=env._device)

    @staticmethod
    def _rotate_xyzw(quaternion, vector):
        imaginary, scalar = quaternion[..., :3], quaternion[..., 3]
        cross = torch.linalg.cross(imaginary, vector)
        return vector + 2 * (scalar[..., None] * cross
                             + torch.linalg.cross(imaginary, cross))

    @torch.no_grad()
    def _measure_spawn_y_offsets(self):
        """Choose an approach-line lateral offset with two supported wheels.

        Terrain tiles are intentionally not all laterally symmetric (the
        authored central-step asset has a lower side platform). A root at the
        tile's nominal y=0 would therefore spawn one wheel above the surface.
        The offset is selected from exact authored triangle heights and is a
        reset placement choice, not a terrain observation or an actor input.
        """
        count = len(self.router.origins_table)
        offsets = torch.linspace(-1.0, 1.0, 41, dtype=torch.float32,
                                 device=self.env._device)
        local = self._wheel_centres_root
        approach = self.router.starts + .5
        best = torch.zeros(count, dtype=torch.float32, device=self.env._device)
        for tile in range(count):
            if int(self.router.kinds[tile]) == int(TerrainKind.FLAT):
                continue
            origin = self.router.origins_table[tile]
            # (candidate, wheel) -> world query point.
            points = torch.stack((
                (origin[0] + approach[tile] + local[:, 0]).expand(len(offsets), 2),
                origin[1] + offsets[:, None] + local[None, :, 1],
                torch.full((len(offsets), 2), self.query._ceiling_m,
                           dtype=torch.float32, device=self.env._device),
            ), -1).reshape(-1, 3)
            heights, valid, _, _ = self.query(
                points, torch.full((len(points),), tile, dtype=torch.long,
                                   device=self.env._device))
            heights = heights.reshape(len(offsets), 2)
            valid = valid.reshape(len(offsets), 2)
            score = (heights[:, 0] - heights[:, 1]).abs()
            score = torch.where(valid.all(-1) & torch.isfinite(score), score,
                                torch.full_like(score, float('inf')))
            finite = torch.isfinite(score)
            if not bool(finite.any()):
                raise RuntimeError(
                    f'terrain tile {tile} has no two-wheel-supported approach line')
            best_score = score[finite].min()
            tied = finite & (score <= best_score + 1e-8)
            # Prefer the nominal centre line whenever it has equal support;
            # do not spend downward runout on an arbitrary first tie.
            choice = offsets.masked_fill(~tied, float('inf')).abs().argmin()
            best[tile] = offsets[choice]
        return best

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
        points[..., 2] = self.query._ceiling_m
        return points

    def _query_failure(self, message, details, points, tiles, valid, physical_points):
        """Attach bounded source-geometry evidence without replacing the failure."""
        error = RuntimeError(message + ': ' + json.dumps(
            details, allow_nan=True, separators=(',', ':')))
        diagnose = getattr(self.query, '_diagnose_misses', None)
        if not callable(diagnose):
            return error
        try:
            report = diagnose(points, tiles, valid, physical_points=physical_points)
        except BaseException as secondary:
            error.add_note('Terrain query diagnostics raised '
                           f'{type(secondary).__name__}: {secondary}')
            return error
        return RuntimeError(message + ': ' + json.dumps(
            {**details, 'query_diagnostics': report}, allow_nan=True, separators=(',', ':')))

    def ground(self):
        env = self.env
        if self._ground_token == env._physical_tick:
            return env._ground_at_root, self.wheel_ground
        points = torch.cat((env._root_link_position()[:, None], self._wheel_centres()), 1)
        finite_state = torch.isfinite(points).all(-1)
        points[..., 2] = self.query._ceiling_m
        heights, valid, _, friction = self.query(points, self.router.tile[:, None].expand(-1, 3))
        valid = valid & finite_state
        if not valid.all():
            ids = (~valid.all(-1)).nonzero().flatten()[:8]
            details = {'reason': 'nonfinite_state' if not finite_state[ids].all() else 'no_terrain_hit',
                       'physical_tick': env._physical_tick, 'rows': ids.tolist(),
                       'tile': self.router.tile[ids].tolist(), 'ray_origins': points[ids].tolist(),
                       'valid_root_left_right': valid[ids].tolist(),
                       'root_local_m': (env._root_link_position() - self.router.origins)[ids].tolist()}
            physical = torch.cat((env._root_link_position()[:, None], self._wheel_centres()), 1)
            raise self._query_failure(
                'course root/wheel query has no physical terrain support', details,
                points, self.router.tile[:, None].expand(-1, 3), valid, physical)
        env._ground_at_root.copy_(heights[:, 0])
        self.wheel_ground.copy_(heights[:, 1:])
        self.wheel_material.copy_(friction[:, 1:])
        self._ground_token = env._physical_tick
        return env._ground_at_root, self.wheel_ground

    def scan_77(self):
        env = self.env
        root = env._root_link_position()
        points = self._yaw_offsets(root, self._grid_offsets)
        tiles = self.router.tile[:, None].expand(-1, 77)
        heights, valid, _, _ = self.query(points, tiles)
        valid = valid & torch.isfinite(root).all(-1, keepdim=True)
        if not valid.all():
            ids = (~valid.all(-1)).nonzero().flatten()[:8]
            details = {'physical_tick': env._physical_tick, 'rows': ids.tolist(),
                       'tile': self.router.tile[ids].tolist(),
                       'root_local_m': (root - self.router.origins)[ids].tolist()}
            raise self._query_failure(
                'privileged terrain scan left the authored collision geometry',
                details, points, tiles, valid, root[:, None].expand_as(points))
        relative = root[:, 2, None] - heights - env._task.nominal_height_m
        return (5 * relative).clamp(-1, 1), heights[:, 38]

    def wheel_lookahead(self):
        """Privileged near/far supervision; never called by command/FSM code."""
        quaternion = self.env._root_link_quat()
        x, y, z, w = quaternion.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        points = wheel_lookahead_points(self._wheel_centres(), yaw, self.env._command[:, 0])
        points[..., 2] = self.query._ceiling_m
        height, valid, _, _ = self.query(points, self.router.tile[:, None, None].expand(-1, 2, 3))
        # Invalid heights stay NaN. Missing columns cannot be labelled as a
        # downward edge or silently replaced with the wheel's current ground.
        delta = height[:, :, 1:] - height[:, :, :1]
        paired_valid = valid[:, :, 1:] & valid[:, :, :1]
        return height, valid, delta, paired_valid

    def lookahead_summary(self):
        height, valid, delta, paired_valid = self.wheel_lookahead()
        measured = delta[paired_valid]
        return {'scope': 'privileged_training_diagnostic_only',
                'policy_or_request_consumption': False,
                'distances_m': [0., .25, .8], 'direction': 'signed_body_forward_command_and_world_yaw',
                'query_points': height.numel(), 'misses': int((~valid).sum()),
                'valid_relative_columns': int(paired_valid.sum()),
                'upward_columns_over_1mm': int((paired_valid & (delta > .001)).sum()),
                'downward_columns_over_1mm': int((paired_valid & (delta < -.001)).sum()),
                'relative_height_range_m': ([float(measured.min()), float(measured.max())]
                                            if measured.numel() else None)}

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

    def submit_request(self, skill, command):
        """One boundary packet per skill; manual and future ToF use the same port."""
        if skill not in self._pending_requests or not isinstance(command, RequestCommand):
            raise ValueError('external request requires jump/stair and a RequestCommand')
        command.validate(self.env.num_envs, self.env._device)
        if self._pending_requests[skill] is not None:
            raise RuntimeError('unconsumed external request already owns this skill boundary')
        self._pending_requests[skill] = command.owned()

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
        tile = self.router.tile[ids]
        approach_x = self.router.starts[tile] + .5
        pose[:, 0] += torch.where(nonflat, approach_x, 0.)
        pose[:, 1] += torch.where(nonflat, self._spawn_y_offset[tile], 0.)
        pose[:, 2] = origins[:, 2] + ground + self.nominal_spawn_height_m
        # Query the wheel locations in the orientation actually written to
        # physics. Rotating after this query certifies the opposite stance.
        reverse = (task.command[:, 0] < 0) & nonflat
        pose[reverse, 5] = 1
        pose[reverse, 6] = 0
        # The authored approach can contain a lower side platform. Correct the
        # root height from the selected wheel colliders before writing it; a
        # nominal flat spawn height alone would insert one wheel into geometry.
        ordinary = task.mode != int(CommandMode.RECOVERY)
        if ordinary.any():
            rows = ordinary.nonzero(as_tuple=False).flatten()
            wheel_offsets = self._rotate_xyzw(
                pose[rows, 3:7][:, None, :], self._wheel_centres_root[None].to(pose.dtype))
            wheel_points = pose[rows, :3][:, None, :] + wheel_offsets
            wheel_points = wheel_points.to(torch.float32)
            wheel_points[..., 2] = self.query._ceiling_m
            query_tiles = tile.index_select(0, rows)[:, None].expand(-1, 2).contiguous()
            heights, valid, _, _ = self.query(wheel_points, query_tiles)
            if not valid.all():
                raise RuntimeError('reset approach has no authored support under both wheel colliders')
            # wheel bottom = root_z + local_z - radius. Raise only; never sink.
            required = heights - wheel_offsets[..., 2] + self._wheel_radius[None]
            required = required + self._reset_clearance_m
            adjusted = torch.maximum(pose[rows, 2:3], required.amax(-1, keepdim=True))
            pose[rows, 2] = adjusted[:, 0]
            gap = (pose[rows, 2, None] + wheel_offsets[..., 2]
                   - self._wheel_radius[None] - heights)
            # Keep both wheels separately: the old minimum cannot establish
            # that the second wheel was supported by authored terrain.
            self._last_support_clearance[ids[rows]] = gap.amin(-1)
            self._last_support_clearance_lr[ids[rows]] = gap
        self._last_support_clearance[ids[~ordinary]] = float('nan')
        self._last_support_clearance_lr[ids[~ordinary]] = float('nan')
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
        upward_course = ((task.mode == int(CommandMode.STAIR))
                         & (task.terrain_kind != int(TerrainKind.STEP_DOWN)))
        height = task.draw.step_up_m
        self._course_stair_pending[ids] = (upward_course
            & (height >= self.stair_requests.config.min_height_m)
            & (height <= self.stair_requests.config.max_height_m))
        self._course_stair_height[ids] = torch.where(upward_course, height, 0.)
        self._last_stair_response[ids] = 0
        # A physical reset cancels only these rows, not another row's packet.
        for packet in self._pending_requests.values():
            if packet is not None:
                packet.request_id[ids] = -1
                packet.height_m[ids] = 0
                if packet.cancel_id is not None:
                    packet.cancel_id[ids] = -1
                if packet.safety_abort is not None:
                    packet.safety_abort[ids] = False

    def _course_stair_command(self, request_id, height_m, cancel_id, safety_abort, ready):
        """Fill vacant command rows from the sampled episode, exactly once.

        An external request/cancel owns its row even if invalid or BUSY. Wait
        for a completed estimate before issuing a new course request; do not
        consume an ID merely because the reset endpoint has no valid estimate.
        """
        external = (request_id != -1) | (cancel_id != -1) | safety_abort
        self._course_stair_pending &= ~external
        issue = (self._course_stair_pending & ready & (self.tasks.active_id < 0)
                 & (self.stair_requests.active_id < 0))
        next_id = torch.maximum(self.stair_next_id, self.stair_requests.max_seen_id + 1)
        request_id = torch.where(issue, next_id, request_id)
        height_m = torch.where(issue, self._course_stair_height, height_m)
        self._course_stair_pending &= ~issue
        return request_id, height_m

    def advance_skill(self):
        # Deployment-visible request decision: env._completed_estimate comes
        # from the preceding actor observation. Terrain truth is not a stair
        # sensor; manual/ToF input is latched in the fields below.
        env = self.env
        blocked = self.stair_requests.active_id >= 0
        jump_packet = self._pending_requests['jump']
        self._pending_requests['jump'] = None
        admission, mode, rise, _ = self.tasks.step(
            env._clock(), env._completed_estimate, env._ground_at_root,
            command=jump_packet, blocked_by_stair=blocked)
        env._command_base.copy_(admission.admitted)
        env._base_command_mode.copy_(mode)
        # One-shot manual/ToF stair requests; jump has arbitration priority.
        request_id = torch.where(self.manual_stair_request, self.manual_stair_request_id,
                                 self.tof_request_id)
        height_request = torch.where(self.manual_stair_request, self.manual_stair_height_m,
                                     self.tof_step_height_m)
        stair_packet = self._pending_requests['stair']
        self._pending_requests['stair'] = None
        cancel_id = self.stair_cancel_id
        safety_abort = torch.zeros(env.num_envs, dtype=torch.bool, device=env._device)
        if stair_packet is not None:
            request_id, height_request = stair_packet.request_id, stair_packet.height_m
            cancel_id = (torch.full_like(request_id, -1) if stair_packet.cancel_id is None
                         else stair_packet.cancel_id)
            if stair_packet.safety_abort is not None:
                safety_abort = stair_packet.safety_abort
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
        request_id, height_request = self._course_stair_command(
            request_id, height_request, cancel_id, safety_abort, ready)
        condition = self.stair_requests.step(
            now_s=env._clock(), request_id=request_id, height_m=height_request,
            cancel_id=cancel_id, jump_active=self.tasks.active_id >= 0,
            ready=ready & ~safety_abort)
        self._last_stair_response.copy_(condition.response)
        # Each submitted packet is consumed once, including BUSY/invalid IDs.
        self.manual_stair_request.zero_()
        self.manual_stair_request_id.fill_(-1)
        self.tof_request_id.fill_(-1)
        self.stair_next_id.copy_(torch.maximum(
            self.stair_next_id, self.stair_requests.max_seen_id + 1))
        finished = condition.finished_id >= 0
        self.stair_cancel_id.fill_(-1)
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
                                   env._last_tilt, sample.root_link_position_world_m[:, 2],
                                   angular_speed_rad_s=state.angular_velocity_body_rad_s.norm(dim=-1),
                                   dt=float(env.physics_dt))
        env._reward_phase.copy_(self.tasks.phase)
        curve = self.tasks.jump_reference()
        landing = ((self.tasks.phase == int(RewardPhase.LANDING))
                   | (self.tasks.phase == int(RewardPhase.SETTLING)))
        target = torch.where(landing, env._ground_at_root + env._command[:, 3], curve.height_world_z_m)
        speed = torch.where(landing, 0, curve.velocity_world_z_m_s)
        env._formal_reward.observe(
            sample, state, effort_nm=env._last_effort, episode_id=env._episode_id,
            control_epoch=env._control_epoch, command=env._command, mode=env._command_mode,
            policy_owned=getattr(env, '_interval_policy_owned', None),
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
                'spawn_lateral_offsets_m': self._spawn_y_offset.tolist(),
                'spawn_lateral_offset_scope': 'exact_authored_two_wheel_approach_line',
                'task_kind_histogram': self.tasks.kind_counts.tolist(),
                'task_mode_histogram': self.tasks.mode_counts.tolist(),
                'level_histogram': torch.bincount(self.router.level, minlength=self.router.manifest['levels']).tolist(),
                'tile_histogram': torch.bincount(self.router.tile, minlength=len(self.router.kinds)).tolist(),
                'course_exits': int(self.router.course_exits.sum()),
                'scan_queries': self.query.queries, 'scan_misses': int(self.query.misses),
                'stair_response_last_interval': torch.bincount(
                    self._last_stair_response, minlength=9).tolist(),
                'stair_command_source': 'external packet or sampled episode command; no terrain sensing',
                'request_responses': self.tasks.request_counters.tolist(),
                'external_request_responses': self.tasks.external_request_counters.tolist(),
                'request_end_counts': dict(zip(
                    ('none', 'estimated_complete', 'cancelled', 'safety_abort', 'timeout', 'reset', 'cold_restore'),
                    self.tasks.request_end_counts.tolist())),
                'request_end_count_scope': 'live protocol boundaries; cold setup is reported separately',
                'physical_request_outcomes': dict(zip(
                    ('verified_success', 'reset_before_verified_success', 'closed_before_verified_success'),
                    self.tasks.physical_outcome_counts.tolist()))}

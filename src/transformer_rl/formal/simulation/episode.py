"""Private episode/control epochs around one immutable, admitted native scene.

Reset packets are explicitly script-owned projections, not fresh measurements.
Only subsequent real physical samples can return a row to policy ownership.
Changing terrain still needs a separately qualified native rebuild mechanism;
this component never relabels stale collision geometry as a new task.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..control.ownership import ControlSource, validate_issued_control
from .ground import NativeGroundQuery
from .plant import NativePhysicsPlant, PhysicalTick
from ..control.observation import CommandFeatures, MeasuredState, SensorSample
from ..contracts import LabeledObservation, PhysicalLabels
from ..tasks.protocol import CompletedEstimate, RequestProtocol
from ..tasks.terrain import TerrainRebuildRequired




@dataclass(frozen=True)
class EpisodeReset:
    observation: LabeledObservation
    pre_reset: LabeledObservation | None
    aborted_request_ids: torch.Tensor
    reset_rows: torch.Tensor
    physical_tick: int
    projection_is_measurement: bool = False


@dataclass(frozen=True)
class EpisodeObservation:
    observation: LabeledObservation
    handoff_after: torch.Tensor
    physical_tick: int


class NativeEpisodeLifecycle:
    """Keep clocks, first episodes, reset histories and ownership explicit.

    The caller computes rewards/termination from the owned PRE-reset endpoint.
    An immediate reset returns a finite, fully masked packet at the SAME clock,
    so the collector does not need an extra or fictitious physics step. A row
    remains script-owned until five new 100Hz sensor frames and a complete
    contact-label window exist. No reset projection is a PPO/auxiliary target.
    """

    def __init__(self, plant: NativePhysicsPlant, ground: NativeGroundQuery, protocol: RequestProtocol):
        if (not isinstance(plant, NativePhysicsPlant) or not isinstance(ground, NativeGroundQuery)
                or ground.scene is not plant.scene or plant._ground is not ground
                or not isinstance(protocol, RequestProtocol) or protocol.num_envs != plant.num_envs
                or protocol.device != plant.device or protocol.config.policy_dt_s != .01):
            raise ValueError('episode lifecycle requires one native plant, exact ground owner and request protocol')
        self.plant, self.ground, self.protocol = plant, ground, protocol
        self.num_envs, self.device = plant.num_envs, plant.device
        self.substeps = round(.01 / plant.dt)
        if self.substeps not in (2, 10) or abs(self.substeps * plant.dt - .01) > 1e-12:
            raise ValueError('episode lifecycle requires the declared100Hz/200Hz or1kHz clocks')
        self._scene, self._view = plant.scene, plant.scene.robot.root_view
        self._meshes, self._courses = plant.scene.terrain_meshes, ground.courses
        self._model_epoch = plant.scene.observation.model_epoch
        self._course_sha256 = tuple(course.sha256 for course in self._courses)
        self.episode_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.control_epoch = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.policy_allowed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fresh_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_at_s = torch.full((self.num_envs,), -float('inf'), dtype=torch.float64, device=self.device)
        self._last_sensor_times = torch.full((self.num_envs, 4), -float('inf'), dtype=torch.float64, device=self.device)
        self._last_reset_tick = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.previous_issued = torch.zeros(self.num_envs, 6, device=self.device)
        self.previous_control_source = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hold_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_position = torch.zeros(self.num_envs, 4, device=self.device)
        self._hold_tolerance_rad = None
        self._seal_hold()
        self._last, self._stepped, self._faulted = None, False, False
        self._pending_tick = None
        self._last_tick = plant.scene.observation.physical_tick
        self._request_tick = -1

    @property
    def course_sha256(self):
        return self._course_sha256

    def _identity(self):
        if (self._faulted or self.plant.scene is not self._scene or self.ground.scene is not self._scene
                or self._scene.robot.root_view is not self._view or self._scene.terrain_meshes is not self._meshes
                or self.ground.courses is not self._courses or self._hold_signature != self._hold_versions()):
            raise RuntimeError('episode scene, ground or native ownership changed or faulted')
        self.plant._identity()

    def _hold_versions(self):
        return (id(self._hold_position), self._hold_position._version,
                id(self._hold_enabled), self._hold_enabled._version, self._hold_tolerance_rad)

    def _seal_hold(self):
        self._hold_signature = self._hold_versions()

    def _mask(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                or value.dtype != torch.bool or value.device != self.device):
            raise ValueError('episode reset requires an exact native-order bool mask')

    def _actions(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 6)
                or value.dtype != torch.float32 or value.device != self.device
                or not torch.isfinite(value).all()
                or (value.abs() > value.new_tensor((3, 3, 3, 3, 9, 9)) + 1e-6).any()):
            raise ValueError('issued policy/script actions require finite clipped native rows')

    def _projection(self, mask, now_s):
        # Neutral finite placeholders are recognizable as invalid supervision
        # and script ownership. They are never called sensor/critic truth.
        if self._last is None:
            frame = torch.zeros(self.num_envs, 35, device=self.device)
            frame[:, 9], frame[:, 28] = -1, 1
            critic = torch.full((self.num_envs, 81), float('nan'), device=self.device)
            labels = PhysicalLabels(torch.full((self.num_envs, 3), float('nan'), device=self.device),
                                    torch.full((self.num_envs, 2), float('nan'), device=self.device),
                                    torch.full((self.num_envs, 6), float('nan'), device=self.device),
                                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                                    now_s.clone(), self.episode_id.clone(), self.control_epoch.clone())
        else:
            retained = self._last.owned()
            frame, critic, labels = retained.frame, retained.critic, retained.labels
            frame[mask] = 0
            frame[mask, 9], frame[mask, 28] = -1, 1
            critic[mask] = float('nan')
            labels.valid[mask] = False
            for value in (labels.velocity_body_m_s, labels.wheel_contact, labels.joint_velocity_rad_s):
                value[mask] = float('nan')
            labels.timestamp_s[mask] = now_s[mask]
            labels.episode_id[mask] = self.episode_id[mask]
            labels.control_epoch[mask] = self.control_epoch[mask]
        return LabeledObservation(frame, critic, now_s.clone(), self.episode_id.clone(), self.control_epoch.clone(),
                                   self.policy_allowed.clone(), labels)

    @torch.no_grad()
    def reset(self, mask, root_pose_xyzw, root_com_velocity, joint_position, joint_velocity, *,
              course_sha256, source_courses=None, preparation_hold=None):
        self._identity()
        self._mask(mask)
        bridge = self._scene.observation
        bridge.before_physics_step()
        if (self._stepped or bridge.physical_tick != self._last_tick or self._last_tick % self.substeps
                or ((self.episode_id < 0) & ~mask).any()):
            raise RuntimeError('reset needs a published policy boundary and complete initial row initialization')
        if (mask & (self._last_reset_tick == self._last_tick)).any():
            raise RuntimeError('an episode row was already reset at this same physical boundary')
        requested = tuple(course_sha256)
        if len(requested) != self.num_envs:
            raise ValueError('reset course identities must include every native row')
        changed = [i for i in mask.nonzero().flatten().tolist() if requested[i] != self.course_sha256[i]]
        courses = None
        if source_courses is not None:
            courses = tuple(source_courses)
            if (len(courses) != self.num_envs
                    or any(not isinstance(c, type(self._courses[i])) for i, c in enumerate(courses))):
                raise ValueError('reset source courses require every actual native-order course')
            hashes = tuple(self._course_sha256[i] if c is self._courses[i] else c.sha256
                           for i, c in enumerate(courses))
            if hashes != requested:
                raise ValueError('reset course request hashes differ from their exact source descriptions')
            geometry_changed = [i for i in mask.nonzero().flatten().tolist()
                                if courses[i].surface != self._courses[i].surface]
            if geometry_changed:
                raise TerrainRebuildRequired(geometry_changed, self.course_sha256, requested)
            self.ground.validate_same_surface_courses(courses, mask)
        elif changed:
            raise TerrainRebuildRequired(changed, self.course_sha256, requested)
        model_changed = bridge.model_epoch != self._model_epoch
        if model_changed and not mask.all():
            raise RuntimeError('a changed native mechanical model requires all episode rows to restart')
        if not mask.any():
            if self._last is None:
                raise RuntimeError('initial observation requires an explicit physical reset')
            return EpisodeReset(self._last.owned(), self._last.owned(), torch.full_like(self.episode_id, -1),
                                mask.clone(), self._last_tick)
        pre_reset = self._last.owned() if self._last is not None else None
        now_s = bridge.timestamp_s()
        if preparation_hold is not None:
            self.plant._matrix('prepared native reset coordinates', joint_position, 18)
            self.plant.controller.validate_preparation_hold(
                preparation_hold, joint_position[:, list(bridge.binding.policy_joint_indices)], mask)
            if (self._hold_enabled[~mask].any() and self._hold_tolerance_rad != preparation_hold.position_tolerance_rad):
                raise ValueError('partial reset cannot change untouched preparation hold comparison semantics')
        try:
            self.plant.reset(mask, root_pose_xyzw, root_com_velocity, joint_position, joint_velocity)
            aborted = self.protocol.reset(mask, now_s)
            if courses is not None:
                self.ground.replace_same_surface_courses(courses, mask)
                self._courses, self._course_sha256 = self.ground.courses, requested
        except BaseException:
            self._faulted = True
            raise
        self.episode_id[mask] += 1
        self.control_epoch[mask] += 1
        self.policy_allowed[mask] = False
        self._fresh_frames[mask] = 0
        self._reset_at_s[mask] = now_s[mask]
        self._last_sensor_times[mask] = -float('inf')
        self._last_reset_tick[mask] = self._last_tick
        self.previous_issued[mask] = 0
        self.previous_control_source[mask] = int(ControlSource.BOUNDED_SCRIPT)
        self._hold_enabled[mask] = preparation_hold is not None
        if preparation_hold is not None:
            self._hold_position[mask] = preparation_hold.leg_position_rad[mask]
            self._hold_tolerance_rad = preparation_hold.position_tolerance_rad
        self._seal_hold()
        self._model_epoch = bridge.model_epoch
        self._last = self._projection(mask, now_s)
        return EpisodeReset(self._last.owned(), pre_reset, aborted.clone(), mask.clone(), self._last_tick)

    def issued_controls(self, policy_issued, script_issued):
        self._identity()
        self._actions(policy_issued)
        self._actions(script_issued)
        actual = torch.where(self.policy_allowed[:, None], policy_issued, script_issued).clone()
        source = torch.where(self.policy_allowed, int(ControlSource.POLICY), int(ControlSource.BOUNDED_SCRIPT))
        selected = ~self.policy_allowed & self._hold_enabled
        if selected.any():
            coordinates = torch.cat((self._hold_position, torch.zeros(self.num_envs, 2, device=self.device)), -1)
            hold = self.plant.controller.preparation_hold(coordinates, position_tolerance_rad=self._hold_tolerance_rad)
            if not hold.admitted[selected].all():
                raise ValueError('latched preparation coordinates no longer pass the actual controller')
            actual[selected] = hold.requested_action[selected]
            source[selected] = int(ControlSource.PREPARATION_HOLD)
        validate_issued_control(actual, source, num_envs=self.num_envs, device=self.device,
                                policy_owned=self.policy_allowed)
        return actual, source

    @torch.no_grad()
    def advance(self, policy_issued, *, script_issued, motor_strength=None, pressure_strength=None,
                physical_observer=None):
        self._identity()
        self._actions(policy_issued)
        self._actions(script_issued)
        if physical_observer is not None and not callable(physical_observer):
            raise ValueError('physical observer must be a synchronous read-only sample consumer')
        bridge = self._scene.observation
        if (self._last is None or self._stepped or (self.episode_id < 0).any()
                or bridge.physical_tick != self._last_tick or bridge.model_epoch != self._model_epoch):
            raise RuntimeError('policy interval needs its initialized, published episode/model boundary')
        actual, source = self.issued_controls(policy_issued, script_issued)
        try:
            self.plant.accept_issued_action(actual, control_source=source, hold_leg_position_rad=self._hold_position,
                                             hold_position_tolerance_rad=self._hold_tolerance_rad)
            for _ in range(self.substeps):
                tick = self.plant.step_physics(motor_strength=motor_strength, pressure_strength=pressure_strength)
                if physical_observer is not None:
                    physical_observer(tick)
                    self._identity()
                    if (bridge.physical_tick != tick.physical_tick or bridge.model_epoch != self._model_epoch):
                        raise RuntimeError('physical observer changed the native model or advanced the clock')
        except BaseException:
            self._faulted = True
            raise
        self.previous_issued.copy_(actual)
        self.previous_control_source.copy_(source)
        self._stepped = True
        self._pending_tick = tick
        return tick

    @torch.no_grad()
    def step_requests(self, estimate, *, request_id, requested_height_m, cancel_id, allow_new, safety_abort):
        """Use the preceding completed estimate, never reset/warmup outputs.

        Call after the physical interval and before publishing its observation.
        Request IDs received during preparation are consumed as NOT_READY;
        keeping the same ID until warmup ends must not create a delayed jump.
        """
        self._identity()
        bridge = self._scene.observation
        if (not self._stepped or bridge.physical_tick != self._last_tick + self.substeps
                or self._request_tick == bridge.physical_tick):
            raise RuntimeError('request processing requires one unpublished completed policy interval')
        if (not isinstance(estimate, CompletedEstimate) or not isinstance(estimate.valid, torch.Tensor)
                or estimate.valid.shape != (self.num_envs,) or estimate.valid.dtype != torch.bool
                or estimate.valid.device != self.device):
            raise ValueError('request processing requires the named completed estimate rows')
        self._mask(allow_new)
        accepted_estimate = replace(estimate, valid=estimate.valid & self.policy_allowed)
        result = self.protocol.step(now_s=bridge.timestamp_s(), estimate=accepted_estimate,
                                    request_id=request_id, requested_height_m=requested_height_m,
                                    cancel_id=cancel_id, allow_new=allow_new & self.policy_allowed,
                                    safety_abort=safety_abort)
        self._request_tick = bridge.physical_tick
        return result

    @torch.no_grad()
    def observe(self, tick: PhysicalTick, sensors: SensorSample, commands: CommandFeatures, *, policy_return_allowed=None):
        self._identity()
        bridge = self._scene.observation
        bridge.before_state_read()
        if (not self._stepped or tick is not self._pending_tick or not isinstance(tick, PhysicalTick)
                or not isinstance(tick.measured, MeasuredState) or not isinstance(sensors, SensorSample)
                or not isinstance(commands, CommandFeatures)
                or tick.physical_tick != self._last_tick + self.substeps
                or bridge.physical_tick != tick.physical_tick or bridge.model_epoch != self._model_epoch
                or not torch.equal(tick.measured.timestamp_s, bridge.timestamp_s())
                or not torch.equal(commands.previous_issued_action, self.previous_issued)):
            raise ValueError('episode observation needs its one completed physical interval and actual issued actions')
        if commands.previous_control_source is None:
            if (self.previous_control_source == int(ControlSource.PREPARATION_HOLD)).any():
                raise ValueError('preparation holds require their actual deployable command source metadata')
        elif not torch.equal(commands.previous_control_source, self.previous_control_source):
            raise ValueError('observation command provenance differs from actual preceding control')
        assembler = bridge.assembler
        release = (torch.ones_like(self.policy_allowed) if policy_return_allowed is None else policy_return_allowed)
        self._mask(release)
        frame = assembler.actor(sensors, commands, now_s=tick.measured.timestamp_s)
        critic = assembler.critic(commands, tick.measured)
        labels = assembler.labels(tick.measured, self.plant.contact_history, self.episode_id, self.control_epoch)
        fresh = ((sensors.sampled_at_s > self._reset_at_s[:, None] + 1e-9)
                 & (sensors.sampled_at_s > self._last_sensor_times + 1e-9)).all(-1)
        proposed = torch.where(fresh, self._fresh_frames + 1, torch.zeros_like(self._fresh_frames))
        handoff = ~self.policy_allowed & release & (proposed >= 5) & labels.valid
        epoch = self.control_epoch + handoff.long()
        allowed = self.policy_allowed | handoff
        # Only commit counters after all assembler validation succeeds. Reads
        # cannot increment physical time or count the same frame twice.
        labels = replace(labels, control_epoch=epoch.clone())
        result = LabeledObservation(frame, critic, tick.measured.timestamp_s.clone(), self.episode_id.clone(),
                                    epoch.clone(), allowed.clone(), labels)
        self._fresh_frames, self.control_epoch, self.policy_allowed = proposed, epoch, allowed
        self._last_sensor_times = sensors.sampled_at_s.detach().clone()
        self._last_tick, self._stepped, self._last = tick.physical_tick, False, result.owned()
        self._pending_tick = None
        return EpisodeObservation(result.owned(), handoff.clone(), tick.physical_tick)

    @torch.no_grad()
    def enter_script(self, mask):
        """Transfer at a published boundary without resetting physical state.

        No expert policy is selected. Direct-hold rows latch the current native
        pose;legacy rows use the caller's bounded script. The caller must
        explicitly allow policy return through its controller release signal.
        """
        self._identity()
        self._mask(mask)
        bridge = self._scene.observation
        if (self._last is None or self._stepped or bridge.physical_tick != self._last_tick
                or self._last_tick % self.substeps or (mask & ~self.policy_allowed).any()):
            raise RuntimeError('script transfer requires currently policy-owned rows at a published boundary')
        now = bridge.timestamp_s()
        selected = mask & self._hold_enabled
        if selected.any():
            q, _ = self.plant._feedback()
            hold = self.plant.controller.preparation_hold(q[:, list(bridge.binding.policy_joint_indices)],
                                                          position_tolerance_rad=self._hold_tolerance_rad)
            if not hold.admitted[selected].all():
                raise ValueError('script transfer cannot latch an unadmitted current physical hold')
            self._hold_position[selected] = hold.leg_position_rad[selected]
            self._seal_hold()
        self.control_epoch[mask] += 1
        self.policy_allowed[mask] = False
        self._fresh_frames[mask] = 0
        self._reset_at_s[mask] = now[mask]
        self._last_sensor_times[mask] = -float('inf')
        self.plant.contact_history.reset(mask, now)
        self._last = self._projection(mask, now)
        return EpisodeObservation(self._last.owned(), mask.clone(), self._last_tick)

"""Private composition of the native100Hz task and labelled collector boundary.

The caller owns the admitted scene, reset plans, sensor/timing implementation,
and independently frozen physical limits. No SDK is launched here, no new
terrain is relabelled over old shapes, and no production training is admitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math

import torch

from ._jump_reference import reward_phases, reward_reference
from ._native_episode import NativeEpisodeLifecycle, TerrainRebuildRequired
from ._native_reward import NativeRewardBridge, NativeRewardSample
from ._physical_observation import CommandMode, SensorSample
from ._skill_evidence import PhysicalJumpEvidence
from ._structured_collection import LabeledStep
from ._task_commands import RequestInputs, TaskCommandPlan, TaskCommandProgram
from ._task_protocol import RequestResponse


@dataclass(frozen=True)
class NativeTaskResetPlan:
    root_pose_xyzw: torch.Tensor
    root_com_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    course_sha256: tuple[str, ...]
    commands: TaskCommandPlan
    preparation_issued_action: torch.Tensor | None = None
    source_courses: tuple | None = None  # Native order, not proof of native replacement.


class FailureReason(IntFlag):
    LOW_SUPPORT_HEIGHT = 1
    EXCESSIVE_TILT = 2
    FORBIDDEN_BODY_CONTACT = 4
    PASSIVE_KNEE_OVERRUN = 8
    KNEE_PIVOT_RESIDUAL = 16
    PRESSURE_STROKE = 32


@dataclass(frozen=True)
class NativeTaskLimits:
    """Explicit recipe inputs; there are no guessed hardware-safe defaults."""

    minimum_normal_height_m: float
    minimum_recovery_height_m: float
    healthy_tilt_rad: float
    failure_tilt_rad: float
    forbidden_body_names: tuple[str, ...]
    forbidden_body_load_n: float
    forbidden_contact_confirm_s: float
    passive_knee_overrun_rad: float
    knee_pivot_residual_m: float
    preparation_timeout_s: float

    def __post_init__(self):
        positive = ('healthy_tilt_rad', 'failure_tilt_rad', 'forbidden_body_load_n',
                    'forbidden_contact_confirm_s', 'passive_knee_overrun_rad', 'knee_pivot_residual_m',
                    'preparation_timeout_s')
        if (any(type(getattr(self, k)) not in (int, float) or not math.isfinite(getattr(self, k))
                or getattr(self, k) <= 0 for k in positive)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (self.minimum_normal_height_m, self.minimum_recovery_height_m))
                or not self.minimum_recovery_height_m <= self.minimum_normal_height_m
                or not self.healthy_tilt_rad < self.failure_tilt_rad <= math.pi
                or self.preparation_timeout_s < .05
                or not isinstance(self.forbidden_body_names, tuple) or not self.forbidden_body_names
                or len(set(self.forbidden_body_names)) != len(self.forbidden_body_names)
                or any(not isinstance(v, str) or not v for v in self.forbidden_body_names)):
            raise ValueError('native task limits require explicit finite,ordered physical/contact/preparation bounds')


@dataclass(frozen=True)
class PhysicalConditions:
    healthy: torch.Tensor
    true_failure: torch.Tensor
    failure_reasons: torch.Tensor


class NativeTaskConditions:
    """Latch true failure over all physical substeps, not just the endpoint."""

    def __init__(self, plant, limits: NativeTaskLimits):
        if not isinstance(limits, NativeTaskLimits):
            raise ValueError('native task conditions require an explicit limits recipe')
        binding = plant.scene.observation.binding
        if not set(limits.forbidden_body_names).issubset(binding.native_body_names):
            raise ValueError('forbidden contact bodies differ from the actual native topology')
        self.plant, self.limits = plant, limits
        self.num_envs, self.device = plant.num_envs, plant.device
        self._bodies = tuple(binding.native_body_names.index(v) for v in limits.forbidden_body_names)
        self._tick = plant.scene.observation.physical_tick
        self._contact_time = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self.failure_reasons = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, mask):
        if (self.plant.scene.observation.physical_tick != self._tick or mask.shape != self.failure_reasons.shape
                or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError('condition reset requires its current published native boundary')
        self._contact_time[mask], self.failure_reasons[mask] = 0, 0

    @torch.no_grad()
    def observe(self, tick, sample: NativeRewardSample, commands):
        self.plant._identity()
        if (tick is not self.plant._last_tick_packet or not isinstance(sample, NativeRewardSample)
                or tick.physical_tick != self._tick + 1 or sample.physical_tick != tick.physical_tick):
            raise ValueError('conditions require every next actual native/reward sample')
        p, c = sample.physics, self.limits
        recovery = commands.mode == int(CommandMode.RECOVERY)
        tilt = torch.atan2(p.gravity_body[:, :2].norm(dim=-1), -p.gravity_body[:, 2])
        contact = p.body_normal_load_n[:, list(self._bodies)].gt(c.forbidden_body_load_n).any(-1) & ~recovery
        self._contact_time = torch.where(contact, self._contact_time + self.plant.dt, 0)
        minimum = torch.where(recovery, c.minimum_recovery_height_m, c.minimum_normal_height_m)
        events = (
            (p.height_above_ground_m < minimum, FailureReason.LOW_SUPPORT_HEIGHT),
            ((tilt > c.failure_tilt_rad) & ~recovery, FailureReason.EXCESSIVE_TILT),
            (self._contact_time >= c.forbidden_contact_confirm_s - 1e-9, FailureReason.FORBIDDEN_BODY_CONTACT),
            ((sample.boundary.knee_signed_clearance_rad < -c.passive_knee_overrun_rad).any(-1),
             FailureReason.PASSIVE_KNEE_OVERRUN),
            ((sample.boundary.knee_pivot_residual_m > c.knee_pivot_residual_m).any(-1), FailureReason.KNEE_PIVOT_RESIDUAL),
            (~tick.pressure.within_mechanical_stroke.all(-1), FailureReason.PRESSURE_STROKE),
        )
        for event, reason in events:
            self.failure_reasons |= event.long() * int(reason)
        failed = self.failure_reasons != 0
        healthy = (~failed & ~contact & (tilt <= c.healthy_tilt_rad)
                   & sample.boundary.knee_signed_clearance_rad.ge(0).all(-1))
        self._tick = tick.physical_tick
        return PhysicalConditions(healthy, failed.clone(), self.failure_reasons.clone())


class NativePreparationError(RuntimeError):
    """Preparation did not yield valid control observations; not a policy fall."""


class NativeTaskEnvironment:
    """One policy interval, one reward endpoint, then optional same-clock reset.

    Native readiness remains external: this composition does not qualify the
    effective-response probe, wheel cook, output domain or physical task limits.
    Invalid geometry is a fault, never a fabricated valid training sample.
    """

    def __init__(self, lifecycle: NativeEpisodeLifecycle, rewards: NativeRewardBridge,
                 skills: PhysicalJumpEvidence, commands: TaskCommandProgram, limits: NativeTaskLimits,
                 *, reset_source, sensor_source, preparation_action, trace_sink=None):
        if (not isinstance(lifecycle, NativeEpisodeLifecycle) or not isinstance(rewards, NativeRewardBridge)
                or rewards.plant is not lifecycle.plant or not isinstance(skills, PhysicalJumpEvidence)
                or not isinstance(commands, TaskCommandProgram)
                or skills.protocol is not lifecycle.protocol or commands.protocol is not lifecycle.protocol
                or skills.config.physics_dt_s != lifecycle.plant.dt
                or commands.jump.config.gravity_m_s2 != rewards.gravity_m_s2
                or not callable(reset_source) or not callable(sensor_source)
                or trace_sink is not None and not callable(trace_sink)):
            raise ValueError('native task requires the actual shared plant,protocol,model gravity and explicit sources')
        lifecycle._actions(preparation_action)
        self.lifecycle, self.rewards, self.skills, self.commands = lifecycle, rewards, skills, commands
        self.num_envs, self.device = lifecycle.num_envs, lifecycle.device
        self.reset_source, self.sensor_source = reset_source, sensor_source
        self.trace_sink = trace_sink
        self.preparation_action = preparation_action.detach().clone()
        self.conditions = NativeTaskConditions(lifecycle.plant, limits)
        self._scene, self._view = lifecycle.plant.scene, lifecycle.plant.scene.robot.root_view
        self._current, self._pending_requests = None, None
        self._faulted, self._in_step = False, False
        self._preparation_at = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self._policy_at = torch.full_like(self._preparation_at, float('inf'))
        self._return_allowed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward_ground_anchor = torch.zeros(self.num_envs, device=self.device)
        self._reward_anchor_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_outcome = None

    def validate_collection_config(self, config):
        self.rewards.validate_collection_config(config)

    def _identity(self):
        if (self._faulted or self.lifecycle.plant.scene is not self._scene
                or self._scene.robot.root_view is not self._view):
            raise RuntimeError('native task is faulted or its owned scene changed;never replay a partially consumed interval')
        self.lifecycle._identity()
        self.rewards._identity()

    def queue_requests(self, value: RequestInputs):
        self._identity()
        if self._in_step or self._pending_requests is not None:
            raise RuntimeError('request inputs may be delivered once at a published boundary')
        self.commands.validate_requests(value)
        self._pending_requests = value.owned()

    def _reset_rows(self, mask):
        life, plant = self.lifecycle, self.lifecycle.plant
        now = plant.scene.observation.timestamp_s()
        plan = self.reset_source(mask.clone(), (life.episode_id + mask.long()).clone(), now.clone())
        if not isinstance(plan, NativeTaskResetPlan):
            raise TypeError('native reset source must return its complete bound plan')
        self.commands.validate_plan(plan.commands)
        if plan.preparation_issued_action is not None:
            life._actions(plan.preparation_issued_action)
        if len(plan.course_sha256) != self.num_envs:
            raise ValueError('reset plan requires every actual native course identity')
        if tuple(plan.course_sha256) != life.course_sha256:
            changed = [i for i in mask.nonzero().flatten().tolist() if plan.course_sha256[i] != life.course_sha256[i]]
            if changed and plan.source_courses is None:
                raise TerrainRebuildRequired(changed, life.course_sha256, tuple(plan.course_sha256))
        result = life.reset(mask, plan.root_pose_xyzw, plan.root_com_velocity, plan.joint_position,
                             plan.joint_velocity, course_sha256=plan.course_sha256, source_courses=plan.source_courses)
        if plan.preparation_issued_action is not None:
            self.preparation_action[mask] = plan.preparation_issued_action[mask]
        if plant.scene.observation.model_epoch != self.rewards._epoch:
            self.rewards.refresh_model()
        self.commands.reset(mask, plan.commands, now)
        self.rewards.composer.reset(mask, now, life.episode_id, life.control_epoch)
        self.skills.reset(mask, now, life.episode_id, life.control_epoch, physical_tick=plant.scene.observation.physical_tick)
        self.conditions.reset(mask)
        self._preparation_at[mask], self._policy_at[mask], self._return_allowed[mask] = now[mask], float('inf'), True
        self._reward_anchor_valid[mask] = False
        self._current = result.observation.owned()
        return result

    @torch.no_grad()
    def reset(self, seed=None):
        self._identity()
        if seed is not None or self._in_step:
            raise ValueError('native task reset never reseeds or resets an open physical interval')
        try:
            result = self._reset_rows(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
            return result.observation.owned()
        except BaseException:
            self._faulted = True
            raise

    @torch.no_grad()
    def step(self, issued_action, completed_estimate):
        self._identity()
        self.lifecycle._actions(issued_action)
        if self._current is None or self._in_step:
            raise RuntimeError('native task step needs its initialized published observation')
        life, plant, composer = self.lifecycle, self.lifecycle.plant, self.rewards.composer
        bridge = plant.scene.observation
        inputs = self._pending_requests or RequestInputs.empty(self.num_envs, self.device)
        self._in_step = True
        try:
            # A controller abort already delivered at this boundary must not
            # execute one more policy interval while waiting for FSM updates.
            immediate = inputs.safety_abort & life.policy_allowed
            if immediate.any():
                life.enter_script(immediate)
            self._return_allowed[inputs.safety_abort] = False
            now, episode, control, owned = (bridge.timestamp_s(), life.episode_id.clone(),
                                             life.control_epoch.clone(), life.policy_allowed.clone())
            current_commands = self.commands.commands(now, life.previous_issued)
            if ((current_commands.mode == int(CommandMode.JUMP)) & ~self._reward_anchor_valid).any():
                raise RuntimeError('active jump has no actual accepted-ground reward anchor')
            actual = torch.where(owned[:, None], issued_action, self.preparation_action)
            _, dq = plant._feedback()
            composer.begin(actual, dq[:, list(bridge.binding.policy_joint_indices)], now_s=now,
                             episode_id=episode, control_epoch=control, policy_owned=owned,
                             admitted_body_command=current_commands.admitted_body_command,
                             command_mode=current_commands.mode)

            def physical(tick):
                sample = self.rewards.sample(tick, episode_id=episode, control_epoch=control, policy_owned=owned)
                condition = self.conditions.observe(tick, sample, current_commands)
                truth = self.rewards.skill_measurement(tick, sample, healthy=condition.healthy,
                                                        true_failure=condition.true_failure)
                self.skills.observe(truth)
                phase = reward_phases(current_commands, self.skills.phase)
                ground = sample.physics.root_world_z_m - sample.physics.height_above_ground_m
                curve = self.commands.curve(current_commands, truth.timestamp_s,
                                              reward_ground_anchor_world_z_m=self._reward_ground_anchor)
                reference = reward_reference(current_commands, curve,
                                               phase, ground)
                composer.observe(sample.physics, reference)
                if self.trace_sink is not None:
                    # Evaluation owns storage/aggregation. Its read-only sink
                    # observes actual substeps,not reconstructed100Hz endpoints.
                    self.trace_sink(tick, sample, condition, reference)

            tick = life.advance(issued_action, script_issued=self.preparation_action, physical_observer=physical)
            final_time = bridge.timestamp_s()
            evidence = self.skills.drain_policy_evidence()
            terminated = self.conditions.failure_reasons != 0
            truncated = ~terminated & (final_time - self._policy_at >= self.commands.task_horizon_s - 1e-9)
            reward = composer.finish_from_evidence(evidence, terminated=terminated, truncated=truncated)
            labels = bridge.assembler.labels(tick.measured, plant.contact_history, episode, control).owned()
            done = terminated | truncated
            request = self.commands.requests(final_time, owned, inputs)
            result = life.step_requests(completed_estimate, request_id=request.request_id,
                                          requested_height_m=request.requested_height_m, cancel_id=request.cancel_id,
                                          allow_new=request.allow_new & ~done,
                                          safety_abort=request.safety_abort | terminated)
            ground = tick.body_motion.link_pose_world_xyzw[:, 0, 2] - tick.measured.height_above_ground_m
            self.commands.advance(final_time, result)
            accepted = result.response == int(RequestResponse.ACCEPTED)
            self._reward_ground_anchor[accepted] = ground[accepted]
            self._reward_anchor_valid[accepted] = True
            skill_receipt = self.skills.accept_protocol_condition(result, self.commands._latched_height)
            next_commands = self.commands.commands(final_time, life.previous_issued)
            sensors = self.sensor_source(tick.measured)
            if not isinstance(sensors, SensorSample):
                raise TypeError('task sensor source must provide explicit deployed sensor fields and timestamps')
            release = request.release_script & ~self._return_allowed
            self._preparation_at[release] = final_time[release]
            self._return_allowed |= request.release_script
            safety = (result.safety_required | request.safety_abort) & ~done
            self._return_allowed[safety] = False
            endpoint = life.observe(tick, sensors, next_commands, policy_return_allowed=self._return_allowed)
            physical_endpoint = endpoint.observation.owned()
            final_critic = endpoint.observation.critic.clone()
            final_valid = torch.isfinite(final_critic).all(-1) & labels.valid & ~terminated
            handoff = immediate | endpoint.handoff_after
            first_policy = endpoint.handoff_after & ~torch.isfinite(self._policy_at)
            self._policy_at[first_policy] = final_time[first_policy]
            transfer = safety & life.policy_allowed
            if transfer.any():
                endpoint = life.enter_script(transfer)
                handoff |= transfer
            waiting = ~life.policy_allowed & self._return_allowed & ~done
            if (waiting & (final_time - self._preparation_at >= self.conditions.limits.preparation_timeout_s)).any():
                raise NativePreparationError('sensor/contact preparation failed;not a policy failure or successful episode')
            self._current = endpoint.observation.owned()
            final_reasons = self.conditions.failure_reasons.clone()
            horizon_elapsed = torch.where(torch.isfinite(self._policy_at), final_time - self._policy_at, 0)
            reset = self._reset_rows(done) if done.any() else None
            self.last_outcome = {'reward': reward, 'physical_evidence': evidence, 'skill_receipt': skill_receipt,
                                 'request_condition': result, 'request_input': request.owned(),
                                 'terminated': terminated.clone(), 'truncated': truncated.clone(),
                                 'failure_reasons': final_reasons,
                                 'horizon_elapsed_s': horizon_elapsed.clone(),
                                 'pre_reset_observation': physical_endpoint, 'reset': reset,
                                 'physical_tick': tick.physical_tick, 'formal_qualification': False}
            self._pending_requests, self._in_step = None, False
            return LabeledStep(self._current.owned(), reward.learning_reward.clone(), terminated.clone(),
                                truncated.clone(), final_critic, final_valid, labels,
                                reward.fully_policy_owned.clone(), handoff)
        except BaseException:
            self._faulted = True
            raise

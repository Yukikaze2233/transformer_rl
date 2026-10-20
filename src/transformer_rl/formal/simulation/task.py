"""Private composition of the native100Hz task and labelled collector boundary.

The caller owns the admitted scene, reset plans, sensor/timing implementation,
and independently frozen physical limits. No SDK is launched here, no new
terrain is relabelled over old shapes, and no production training is admitted.
"""
from __future__ import annotations

import time

import torch

from ..tasks.jump import reward_phases, reward_reference
from .episode import NativeEpisodeLifecycle, TerrainRebuildRequired
from .reward import NativeRewardBridge, NativeRewardSample
from ..control.observation import CommandMode, SensorSample
from ..control.actuation import PreparationHold
from ..tasks.skills import PhysicalJumpEvidence
from ..contracts import LabeledStep
from ..tasks.termination import FailureReason, NativeTaskLimits, PhysicalConditions, PhysicalFailureLatch
from ..tasks.commands import RequestInputs, TaskCommandPlan, TaskCommandProgram
from ..tasks.protocol import RequestResponse
from ..tasks.reset_plan import NativeTaskResetPlan




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
        self._latch = PhysicalFailureLatch(limits, self.num_envs, physics_dt_s=plant.dt, device=self.device)
        self.failure_reasons = self._latch.failure_reasons

    def reset(self, mask):
        if (self.plant.scene.observation.physical_tick != self._tick or mask.shape != self.failure_reasons.shape
                or mask.dtype != torch.bool or mask.device != self.device):
            raise ValueError('condition reset requires its current published native boundary')
        self._latch.reset(mask)

    @torch.no_grad()
    def observe(self, tick, sample: NativeRewardSample, commands):
        self.plant._identity()
        if (tick is not self.plant._last_tick_packet or not isinstance(sample, NativeRewardSample)
                or tick.physical_tick != self._tick + 1 or sample.physical_tick != tick.physical_tick):
            raise ValueError('conditions require every next actual native/reward sample')
        p = sample.physics
        result = self._latch.observe(
            height_m=p.height_above_ground_m, gravity_body=p.gravity_body,
            forbidden_body_load_n=p.body_normal_load_n[:, list(self._bodies)],
            knee_signed_clearance_rad=sample.boundary.knee_signed_clearance_rad,
            knee_pivot_residual_m=sample.boundary.knee_pivot_residual_m,
            within_pressure_stroke=tick.pressure.within_mechanical_stroke,
            recovery=commands.mode == int(CommandMode.RECOVERY))
        self._tick = tick.physical_tick
        return result


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
                or skills._tick != lifecycle.plant.scene.observation.physical_tick
                or skills._drained_tick != skills._tick
                or commands.jump.config.gravity_m_s2 != rewards.gravity_m_s2
                or not callable(reset_source) or not callable(sensor_source)
                or trace_sink is not None and not callable(trace_sink)):
            raise ValueError('native task requires the actual shared plant,protocol,model gravity and explicit sources')
        lifecycle._actions(preparation_action)
        self.lifecycle, self.rewards, self.skills, self.commands = lifecycle, rewards, skills, commands
        self.num_envs, self.device = lifecycle.num_envs, lifecycle.device
        self.reset_source, self.sensor_source = reset_source, sensor_source
        self.trace_sink = trace_sink
        self.terrain_stream = None
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
        self.last_timing = {}
        self.last_counts = {}

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
        if plan.preparation_hold is not None and plan.preparation_issued_action is not None:
            raise ValueError('reset preparation must have one unambiguous control representation')
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
                             plan.joint_velocity, course_sha256=plan.course_sha256, source_courses=plan.source_courses,
                             preparation_hold=plan.preparation_hold)
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
        started = time.perf_counter()
        self._identity()
        self.lifecycle._actions(issued_action)
        if self._current is None or self._in_step:
            raise RuntimeError('native task step needs its initialized published observation')
        if self.terrain_stream is not None:
            self.terrain_stream()
        streamed = time.perf_counter()
        physical_timing = {}
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
            current_commands = self.commands.commands(now, life.previous_issued, life.previous_control_source)
            if ((current_commands.mode == int(CommandMode.JUMP)) & ~self._reward_anchor_valid).any():
                raise RuntimeError('active jump has no actual accepted-ground reward anchor')
            actual, source = life.issued_controls(issued_action, self.preparation_action)
            _, dq = plant._feedback()
            composer.begin(actual, dq[:, list(bridge.binding.policy_joint_indices)], now_s=now,
                             episode_id=episode, control_epoch=control, policy_owned=owned,
                             admitted_body_command=current_commands.admitted_body_command,
                             command_mode=current_commands.mode, control_source=source)

            def physical(tick):
                for name, elapsed in getattr(plant, 'last_timing', {}).items():
                    physical_timing[name] = physical_timing.get(name, 0.) + elapsed
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

            advance_started = time.perf_counter()
            tick = life.advance(issued_action, script_issued=self.preparation_action, physical_observer=physical)
            advanced = time.perf_counter()
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
            next_commands = self.commands.commands(final_time, life.previous_issued, life.previous_control_source)
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
            reset_started = time.perf_counter()
            reset = self._reset_rows(done) if done.any() else None
            reset_finished = time.perf_counter()
            # 主机墙钟不插入逐步CUDA同步；嵌套physical项不能再加到总耗时。
            self.last_timing = {
                'terrain_stream_s': streamed - started,
                'policy_preparation_s': advance_started - streamed,
                'advance_and_physical_reward_s': advanced - advance_started,
                'endpoint_protocol_observation_s': reset_started - advanced,
                'reset_s': reset_finished - reset_started,
                **{'physical_' + name: value for name, value in physical_timing.items()},
            }
            self.last_counts = {
                'reset_rows': getattr(self.reset_source, 'last_reset_rows', 0) if reset is not None else 0,
                'terrain_stream_changed_rows': getattr(self.terrain_stream, 'last_changed_rows', 0),
            }
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

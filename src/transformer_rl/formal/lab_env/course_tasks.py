"""Episode command trajectories and deployable jump-request scheduling.

Actor-visible request decisions use only the preceding completed estimate.
Physical flight/landing labels and success are a separate reward-only channel.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ..control.observation import CommandMode
from ..tasks.curriculum import TaskFamily
from ..tasks.jump import JumpReferenceProgram
from ..tasks.protocol import CompletedEstimate, RequestProtocol, RequestResponse
from ..tasks.reward import RewardPhase
from ..tasks.trajectory import CommandRates, CommandStream, CommandTrajectory, DriveEnvelope


class LabCourseTasks:
    def __init__(self, num_envs, envelope: DriveEnvelope, *, seed=2102, device='cpu'):
        self.num_envs, self.device = num_envs, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        initial = torch.zeros(num_envs, 4, device=self.device)
        initial[:, 3] = .32
        self.stream = CommandStream(initial, envelope, CommandRates(2., 3., .2))
        self.trajectory = CommandTrajectory(initial, initial, torch.ones(num_envs, device=self.device),
                                             torch.zeros(num_envs, dtype=torch.long, device=self.device))
        self.protocol = RequestProtocol(num_envs, self.device)
        self.reference = JumpReferenceProgram()
        self.family = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.base_mode = self.family.clone()
        self.elapsed = torch.zeros(num_envs, device=self.device)
        self.next_request_s = torch.zeros_like(self.elapsed)
        self.next_id = torch.zeros_like(self.family)
        self.rise = torch.zeros_like(self.elapsed)
        self.latched_height = torch.full_like(self.elapsed, .32)
        self.ground_anchor = torch.zeros_like(self.elapsed)
        self.request_clock = torch.zeros_like(self.elapsed)
        self.active_id = torch.full_like(self.family, -1)
        self.phase = torch.zeros_like(self.family)
        self.flight_seen = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.air_ticks = torch.zeros_like(self.family)
        self.ascent_seen = torch.zeros_like(self.flight_seen)
        self.peak_rise = torch.zeros_like(self.elapsed)
        self.landed_s = torch.zeros_like(self.elapsed)
        self.success_id = torch.full_like(self.family, -1)
        self.paid_id = torch.full_like(self.family, -1)
        self.request_counters = torch.zeros(9, dtype=torch.long, device=self.device)

    def reset(self, ids, task, now):
        self.family[ids] = task.draw.family
        self.base_mode[ids] = torch.where(task.mode == int(CommandMode.JUMP),
                                         int(CommandMode.NORMAL), task.mode)
        self.rise[ids] = task.request_height_m
        self.elapsed[ids] = 0
        self.next_request_s[ids] = 3 + 5 * torch.rand(len(ids), generator=self.generator, device=self.device)
        target = task.command.clone()
        target[task.draw.nominal] = target.new_tensor((0., 0., 0., .32))
        start = torch.zeros_like(target)
        start[:, 3] = .32
        hold = task.draw.trajectory_kind == 0
        start[hold] = target[hold]
        self.trajectory.start[ids], self.trajectory.end[ids] = start, target
        duration = .5 + 2.5 * torch.rand(len(ids), generator=self.generator, device=self.device)
        self.trajectory.duration_s[ids] = duration
        self.trajectory.frequency_start_hz[ids] = 1 / duration
        self.trajectory.frequency_end_hz[ids] = 2 / duration
        self.trajectory.kind[ids] = task.draw.trajectory_kind
        self.stream.current[ids] = self.stream.envelope.project(start)[0]
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.protocol.reset(mask, now)
        self.active_id[ids] = -1
        self.request_clock[ids] = 0
        self.flight_seen[ids] = False
        self.air_ticks[ids] = 0
        self.ascent_seen[ids] = False
        self.peak_rise[ids] = 0
        self.landed_s[ids] = 0
        self.phase[ids] = 0
        self.success_id[ids] = -1

    def step(self, now, estimate, ground):
        self.success_id.fill_(-1)
        self.elapsed.add_(.01)
        cyclic = (self.trajectory.kind == 2) | (self.trajectory.kind == 3) | (self.trajectory.kind == 5)
        clock = torch.where(cyclic, self.elapsed.remainder(self.trajectory.duration_s), self.elapsed)
        admission = self.stream.step(self.trajectory.at(clock))
        if estimate is None:
            estimate = CompletedEstimate(
                torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, 2, device=self.device),
                torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device),
                now.clone(), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        due = ((self.family == int(TaskFamily.REQUEST)) & (self.elapsed >= self.next_request_s)
               & (self.protocol.active_id < 0))
        request_id = torch.where(due, self.next_id, -1)
        self.next_id.add_(due.long())
        interval = 3 + 5 * torch.rand(self.num_envs, generator=self.generator, device=self.device)
        self.next_request_s.copy_(torch.where(due, self.elapsed + interval, self.next_request_s))
        condition = self.protocol.step(
            now_s=now, estimate=estimate, request_id=request_id, requested_height_m=self.rise,
            cancel_id=torch.full_like(request_id, -1), allow_new=self.family == int(TaskFamily.REQUEST),
            safety_abort=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        accepted = condition.response == int(RequestResponse.ACCEPTED)
        self.request_counters.add_(torch.bincount(condition.response, minlength=9))
        self.latched_height.copy_(torch.where(accepted, admission.admitted[:, 3], self.latched_height))
        self.ground_anchor.copy_(torch.where(accepted, ground, self.ground_anchor))
        self.flight_seen.masked_fill_(accepted, False)
        self.air_ticks.masked_fill_(accepted, 0)
        self.ascent_seen.masked_fill_(accepted, False)
        self.peak_rise.masked_fill_(accepted, 0)
        self.landed_s.masked_fill_(accepted, 0)
        self.active_id.copy_(condition.active_id)
        self.request_clock.copy_(condition.clock_s)
        mode = torch.where(condition.jump_mode, int(CommandMode.JUMP), self.base_mode)
        # The reference baseline is actor-visible throughout an active request.
        # A continuing sine/ramp must not hide a latched baseline outside H5.
        self.stream.current[:, 3] = torch.where(condition.jump_mode, self.latched_height,
                                                self.stream.current[:, 3])
        admission = replace(admission, admitted=self.stream.current.clone())
        return admission, mode, condition.height_m, condition.clock_s

    def observe_physics(self, force, velocity_z, tilt, world_height, dt=.005):
        active = self.active_id >= 0
        air = (force < 2).all(-1)
        self.air_ticks.copy_(torch.where(active & air, self.air_ticks + 1, 0))
        self.flight_seen |= active & (self.air_ticks >= 2)
        actual_rise = world_height - self.ground_anchor - self.latched_height
        self.ascent_seen |= active & (velocity_z > .15) & (actual_rise > .01)
        self.peak_rise.copy_(torch.where(active, torch.maximum(self.peak_rise, actual_rise), self.peak_rise))
        contact = (force > 2).all(-1)
        achieved = self.peak_rise >= (self.rise - .02).clamp_min(.01)
        stable = (active & self.flight_seen & self.ascent_seen & achieved & contact
                  & (velocity_z.abs() <= .15) & (tilt <= .25)
                  & (actual_rise.abs() <= .05))
        self.landed_s.copy_(torch.where(stable, self.landed_s + dt, 0))
        self.phase.copy_(torch.where(
            active, torch.where(air & self.flight_seen, int(RewardPhase.FLIGHT),
                                torch.where(self.flight_seen, int(RewardPhase.LANDING), int(RewardPhase.TAKEOFF))),
            torch.where(self.base_mode == int(CommandMode.RECOVERY), int(RewardPhase.RECOVERY), int(RewardPhase.SUPPORT))))
        success = stable & (self.landed_s >= 2.) & (self.active_id > self.paid_id)
        self.success_id.copy_(torch.where(success, self.active_id, self.success_id))
        self.paid_id.copy_(torch.where(success, self.active_id, self.paid_id))

    def jump_reference(self):
        return self.reference.sample(self.latched_height, self.rise, self.request_clock.double(),
                                     self.ground_anchor, active=self.active_id >= 0)

    def state_dict(self):
        # Cold restoration aborts physical requests, preserving IDs and RNG.
        return {'format': 'lab_course_tasks_v1', 'protocol': self.protocol.state_dict(),
                'next_id': self.next_id.clone(), 'paid_id': self.paid_id.clone(),
                'rng': self.generator.get_state().clone(), 'request_counters': self.request_counters.clone()}

    def restore_after_physical_reset(self, state, now):
        if state.get('format') != 'lab_course_tasks_v1' or state['next_id'].shape != self.next_id.shape:
            raise ValueError('course request checkpoint differs')
        self.protocol.restore_after_physical_reset(state['protocol'], now)
        self.next_id.copy_(state['next_id'])
        self.paid_id.copy_(state['paid_id'])
        self.request_counters.copy_(state['request_counters'])
        self.generator.set_state(state['rng'].cpu())


def completed_policy_estimate(frame, raw_state, timestamp):
    """Deployment-reproducible estimate: no critic, terrain truth, or future state."""
    return CompletedEstimate._from_actor_observation(frame, raw_state, timestamp)

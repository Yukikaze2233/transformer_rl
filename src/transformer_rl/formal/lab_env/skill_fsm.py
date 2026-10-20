"""台阶与跳跃的批量状态机契约。

状态机只产生可观察的 mode、高度目标和速度缩放；不读取策略权重，也不
直接写执行器。前向感知和手动请求进入同一套状态，便于训练与部署共用。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from ..control.observation import CommandMode


class StairState(IntEnum):
    NORMAL = 0
    STEP_UP = 1
    STAIR = 2
    BODY_BUMP = 3


class JumpState(IntEnum):
    NORMAL = 0
    PRELOAD = 1
    TAKEOFF = 2
    FLIGHT = 3
    LANDING = 4
    SETTLE = 5


@dataclass(frozen=True)
class StairFSMConfig:
    policy_dt_s: float = .01
    curb_max_m: float = .04
    step_up_max_m: float = .10
    stair_max_m: float = .14
    ledge_max_m: float = .20
    step_up_bias_m: float = .04
    step_up_hold_s: float = 1.5
    stair_height_m: float = .34
    body_bump_height_m: float = .34
    max_height_m: float = .35
    success_height_error_m: float = .05
    min_progress_m: float = .02
    success_duration_s: float = .2
    wheel_contact_force_n: float = 20.0
    failure_drop_m: float = .10
    failure_tilt_rad: float = math.radians(40.0)
    base_contact_force_n: float = 1.0
    timeout_s: float = 5.0
    body_bump_timeout_s: float = 6.0
    step_up_speed_scale: float = .6
    stair_speed_scale: float = .5
    body_bump_speed_scale: float = .35

    def __post_init__(self):
        for name in ("policy_dt_s", "curb_max_m", "step_up_max_m", "stair_max_m",
                     "ledge_max_m", "step_up_bias_m", "step_up_hold_s", "stair_height_m",
                     "body_bump_height_m", "success_height_error_m", "min_progress_m", "success_duration_s",
                     "wheel_contact_force_n", "failure_drop_m", "failure_tilt_rad",
                     "base_contact_force_n", "timeout_s", "body_bump_timeout_s",
                     "step_up_speed_scale", "stair_speed_scale", "body_bump_speed_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值")
        if (not 0 <= self.curb_max_m < self.step_up_max_m < self.stair_max_m < self.ledge_max_m
                or self.policy_dt_s <= 0 or self.step_up_hold_s < 0 or self.timeout_s <= 0
                or self.body_bump_timeout_s <= 0 or self.success_duration_s < 0
                or self.min_progress_m < 0 or self.failure_drop_m < 0 or not 0 < self.failure_tilt_rad < math.pi
                or self.base_contact_force_n < 0 or self.wheel_contact_force_n < 0
                or self.stair_height_m <= 0 or self.body_bump_height_m <= 0
                or self.max_height_m <= 0
                or self.step_up_speed_scale <= 0 or self.stair_speed_scale <= 0
                or self.body_bump_speed_scale <= 0):
            raise ValueError("台阶状态机阈值不一致")


@dataclass(frozen=True)
class StairOutput:
    state: torch.Tensor
    mode: torch.Tensor
    height_command_m: torch.Tensor
    speed_scale: torch.Tensor
    body_contact_allowed: torch.Tensor
    leg_tuck: torch.Tensor
    stage: torch.Tensor
    entered: torch.Tensor
    exited: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    wall: torch.Tensor


def _tensor(name, value, num_envs, device, dtype=torch.float32):
    value = value.to(device=device, dtype=dtype)
    if value.shape != (num_envs,) or not torch.isfinite(value).all():
        raise ValueError(f"{name} 需要有限 [{num_envs}]")
    return value


class StairFSM:
    """踏板、台阶与车体上台沿的批量状态机。"""

    def __init__(self, num_envs: int, config: StairFSMConfig = StairFSMConfig(), *, device="cpu"):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, StairFSMConfig):
            raise ValueError("台阶状态机需要正整数环境数和配置")
        self.num_envs, self.config, self.device = num_envs, config, torch.device(device)
        self.state = torch.full((num_envs,), int(StairState.NORMAL), dtype=torch.long, device=self.device)
        self.state_time = torch.zeros(num_envs, device=self.device)
        self.success_time = torch.zeros(num_envs, device=self.device)
        self.reference_height = torch.zeros(num_envs, device=self.device)
        self.target_height = torch.zeros(num_envs, device=self.device)
        self._mode = torch.full((num_envs,), int(CommandMode.NORMAL), dtype=torch.long, device=self.device)
        self._height = torch.zeros(num_envs, device=self.device)
        self._speed = torch.ones(num_envs, device=self.device)
        self._body_allowed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._tuck = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stage = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def active(self):
        return self.state != int(StairState.NORMAL)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("台阶 reset ids 超出范围")
        self.state[ids] = int(StairState.NORMAL)
        self.state_time[ids] = 0
        self.success_time[ids] = 0
        self.reference_height[ids] = 0
        self.target_height[ids] = 0
        self._mode[ids] = int(CommandMode.NORMAL)
        self._height[ids] = 0
        self._speed[ids] = 1
        self._body_allowed[ids] = False
        self._tuck[ids] = False
        self._stage[ids] = 0

    def _classify(self, step_height):
        state = torch.full_like(step_height, int(StairState.STAIR), dtype=torch.long)
        state = torch.where(step_height <= self.config.step_up_max_m,
                            torch.full_like(state, int(StairState.STEP_UP)), state)
        state = torch.where(step_height > self.config.stair_max_m,
                            torch.full_like(state, int(StairState.BODY_BUMP)), state)
        return state

    def step(self, *, step_height_m, wheel_force_n=None, body_height_m, tilt_rad,
             base_contact_force_n=None, command_height_m, auto_enabled=True,
             manual_request=None, manual_height_m=None,
             wheel_contact_probability=None, base_contact_probability=None,
             success_from_height=True):
        n, device = self.num_envs, self.device
        step = _tensor("step height", step_height_m, n, device)
        body = _tensor("body height", body_height_m, n, device)
        tilt = _tensor("tilt", tilt_rad, n, device)
        command_height = _tensor("command height", command_height_m, n, device)
        if wheel_contact_probability is not None:
            contact = torch.as_tensor(wheel_contact_probability, dtype=torch.float32, device=device)
            if (contact.shape != (n, 2) or not torch.isfinite(contact).all()
                    or ((contact < 0) | (contact > 1)).any()):
                raise ValueError("wheel contact probability must be in [0,1]")
            contact = (contact >= .5).all(-1)
        else:
            if wheel_force_n is None:
                raise ValueError("stair FSM requires wheel force or contact probability")
            force = torch.as_tensor(wheel_force_n, dtype=torch.float32, device=device)
            if force.shape != (n, 2) or not torch.isfinite(force).all() or (force < 0).any():
                raise ValueError("wheel force 需要有限非负 [N,2]")
            contact = (force > self.config.wheel_contact_force_n).all(-1)
        if base_contact_probability is not None:
            base_prob = torch.as_tensor(base_contact_probability, dtype=torch.float32, device=device)
            if (base_prob.shape != (n,) or not torch.isfinite(base_prob).all()
                    or ((base_prob < 0) | (base_prob > 1)).any()):
                raise ValueError("base contact probability must be in [0,1]")
            base = base_prob >= .5
        elif base_contact_force_n is None:
            base = torch.zeros(n, dtype=torch.bool, device=device)
        else:
            base = _tensor("base contact", base_contact_force_n, n, device) > self.config.base_contact_force_n
        auto = torch.as_tensor(auto_enabled, dtype=torch.bool, device=device).expand(n)
        manual = (torch.zeros(n, dtype=torch.bool, device=device) if manual_request is None
                  else torch.as_tensor(manual_request, dtype=torch.bool, device=device))
        if manual.shape != (n,):
            raise ValueError("manual request 需要 [N] bool")
        manual_height = step if manual_height_m is None else _tensor("manual height", manual_height_m, n, device)

        normal = self.state == int(StairState.NORMAL)
        valid = (step > self.config.curb_max_m) & (step <= self.config.ledge_max_m)
        auto_trigger = normal & auto & valid
        manual_trigger = normal & manual & (manual_height > self.config.curb_max_m)
        trigger = auto_trigger | manual_trigger
        chosen_height = torch.where(manual_trigger, manual_height, step)
        chosen_state = self._classify(chosen_height)
        wall = normal & ((step > self.config.ledge_max_m) | (manual_height > self.config.ledge_max_m))
        entered = trigger & ~wall
        self.state = torch.where(entered, chosen_state, self.state)
        self.state_time = torch.where(entered, torch.zeros_like(self.state_time), self.state_time)
        self.success_time = torch.where(entered, torch.zeros_like(self.success_time), self.success_time)
        self.reference_height = torch.where(entered, body, self.reference_height)
        self.target_height = torch.where(
            entered,
            torch.where(chosen_state == int(StairState.STEP_UP),
                        (command_height + self.config.step_up_bias_m).clamp_max(self.config.max_height_m),
                        torch.where(chosen_state == int(StairState.STAIR),
                                    torch.full_like(command_height, self.config.stair_height_m),
                                    torch.full_like(command_height, self.config.body_bump_height_m))),
            self.target_height,
        )

        active = self.state != int(StairState.NORMAL)
        self.state_time = torch.where(active, self.state_time + self.config.policy_dt_s, self.state_time)
        is_step = self.state == int(StairState.STEP_UP)
        is_stair = self.state == int(StairState.STAIR)
        is_bump = self.state == int(StairState.BODY_BUMP)
        height_ok = body >= self.target_height - self.config.success_height_error_m
        progress_ok = body >= self.reference_height + self.config.min_progress_m
        success_cond = ((is_stair | is_bump) & contact & height_ok & progress_ok
                        if success_from_height else torch.zeros_like(is_stair))
        self.success_time = torch.where(success_cond, self.success_time + self.config.policy_dt_s,
                                        torch.where(active, torch.zeros_like(self.success_time), self.success_time))
        drop = body < self.reference_height - self.config.failure_drop_m
        wall_active = active & (step > self.config.ledge_max_m)
        hard_failure = active & (drop | (tilt > self.config.failure_tilt_rad)
                                 | base | wall_active)
        timeout = active & (self.state_time > torch.where(
            is_bump, torch.full_like(self.state_time, self.config.body_bump_timeout_s),
            torch.full_like(self.state_time, self.config.timeout_s)))
        step_done = is_step & (self.state_time >= self.config.step_up_hold_s)
        success = (is_stair | is_bump) & (self.success_time >= self.config.success_duration_s)
        failure = hard_failure | timeout
        exited = step_done | success | failure
        self.state = torch.where(exited, torch.full_like(self.state, int(StairState.NORMAL)), self.state)
        self.state_time = torch.where(exited, torch.zeros_like(self.state_time), self.state_time)
        self.success_time = torch.where(exited, torch.zeros_like(self.success_time), self.success_time)

        active_out = self.state != int(StairState.NORMAL)
        self._mode = torch.where(active_out, torch.full_like(self._mode, int(CommandMode.STAIR)),
                                 torch.full_like(self._mode, int(CommandMode.NORMAL)))
        self._height = torch.where(
            active_out,
            torch.where(self.state == int(StairState.STEP_UP),
                        (command_height + self.config.step_up_bias_m).clamp_max(self.config.max_height_m),
                        torch.where(self.state == int(StairState.STAIR),
                                    torch.full_like(command_height, self.config.stair_height_m),
                                    torch.full_like(command_height, self.config.body_bump_height_m))),
            command_height,
        )
        self._speed = torch.where(
            active_out,
            torch.where(self.state == int(StairState.STEP_UP), torch.full_like(self._speed, self.config.step_up_speed_scale),
                        torch.where(self.state == int(StairState.STAIR),
                                    torch.full_like(self._speed, self.config.stair_speed_scale),
                                    torch.full_like(self._speed, self.config.body_bump_speed_scale))),
            torch.ones_like(self._speed),
        )
        self._body_allowed = self.state == int(StairState.BODY_BUMP)
        self._tuck = self._body_allowed & (self.state_time >= .2)
        self._stage = torch.where(
            self.state == int(StairState.BODY_BUMP),
            (self.state_time / .5).long().clamp_max(3),
            torch.zeros_like(self._stage),
        )
        return StairOutput(
            self.state.clone(), self._mode.clone(), self._height.clone(), self._speed.clone(),
            self._body_allowed.clone(), self._tuck.clone(), self._stage.clone(),
            entered, exited, success, failure, wall,
        )

    def deployment_step(self, *, step_height_m, wheel_contact_probability, tilt_rad,
                        command_height_m, manual_request, manual_height_m=None):
        """Command-process stair step without force or body-height success truth.

        The body height is only a target baseline here. ``success`` remains
        false; real landing/step success is measured by the training-side
        physical evidence path.
        """
        return self.step(
            step_height_m=step_height_m, wheel_force_n=None,
            wheel_contact_probability=wheel_contact_probability,
            body_height_m=command_height_m, tilt_rad=tilt_rad,
            base_contact_force_n=None, command_height_m=command_height_m,
            auto_enabled=False, manual_request=manual_request,
            manual_height_m=manual_height_m, success_from_height=False,
        )


@dataclass(frozen=True)
class JumpFSMConfig:
    policy_dt_s: float = .01
    preload_s: float = .2
    takeoff_s: float = .2
    flight_timeout_s: float = 1.5
    landing_timeout_s: float = .5
    settle_s: float = .3
    success_height_error_m: float = .05
    wheel_contact_force_n: float = 20.0
    failure_tilt_rad: float = math.radians(40.0)
    base_contact_force_n: float = 1.0
    min_rise_m: float = .03
    max_rise_m: float = .20
    preload_speed_scale: float = .6
    flight_speed_scale: float = 1.0
    landing_speed_scale: float = .5

    def __post_init__(self):
        for name in ("policy_dt_s", "preload_s", "takeoff_s", "flight_timeout_s",
                     "landing_timeout_s", "settle_s", "success_height_error_m",
                     "wheel_contact_force_n", "failure_tilt_rad", "base_contact_force_n",
                     "min_rise_m", "max_rise_m", "preload_speed_scale",
                     "flight_speed_scale", "landing_speed_scale"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值")
        if (self.policy_dt_s <= 0 or min(self.preload_s, self.takeoff_s, self.flight_timeout_s,
                                         self.landing_timeout_s, self.settle_s) < 0
                or self.success_height_error_m < 0 or self.wheel_contact_force_n < 0
                or self.base_contact_force_n < 0 or not 0 < self.failure_tilt_rad < math.pi
                or not 0 < self.min_rise_m <= self.max_rise_m
                or min(self.preload_speed_scale, self.flight_speed_scale,
                       self.landing_speed_scale) <= 0):
            raise ValueError("跳跃状态机阈值不一致")


@dataclass(frozen=True)
class JumpOutput:
    state: torch.Tensor
    mode: torch.Tensor
    height_command_m: torch.Tensor
    request_height_m: torch.Tensor
    speed_scale: torch.Tensor
    phase: torch.Tensor
    request_id: torch.Tensor
    entered: torch.Tensor
    exited: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor


class JumpFSM:
    """五阶段跳跃状态机；请求高度是相对当前高度命令的增量。"""

    def __init__(self, num_envs: int, config: JumpFSMConfig = JumpFSMConfig(), *, device="cpu"):
        if type(num_envs) is not int or num_envs < 1 or not isinstance(config, JumpFSMConfig):
            raise ValueError("跳跃状态机需要正整数环境数和配置")
        self.num_envs, self.config, self.device = num_envs, config, torch.device(device)
        self.state = torch.full((num_envs,), int(JumpState.NORMAL), dtype=torch.long, device=self.device)
        self.state_time = torch.zeros(num_envs, device=self.device)
        self.landing_time = torch.zeros(num_envs, device=self.device)
        self.settle_time = torch.zeros(num_envs, device=self.device)
        self.target_height = torch.zeros(num_envs, device=self.device)
        self.request_height = torch.zeros(num_envs, device=self.device)
        self._height = torch.zeros(num_envs, device=self.device)
        self._speed = torch.ones(num_envs, device=self.device)
        self.request_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.request_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def active(self):
        return self.state != int(JumpState.NORMAL)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("跳跃 reset ids 超出范围")
        self.state[ids] = int(JumpState.NORMAL)
        self.state_time[ids] = 0
        self.landing_time[ids] = 0
        self.settle_time[ids] = 0
        self.target_height[ids] = 0
        self.request_height[ids] = 0
        self._height[ids] = 0
        self._speed[ids] = 1
        self.request_id[ids] = -1

    def step(self, *, request, request_height_m, command_height_m, body_height_m,
             wheel_force_n, tilt_rad, base_contact_force_n, vertical_velocity_m_s=None):
        n, device = self.num_envs, self.device
        request = torch.as_tensor(request, dtype=torch.bool, device=device)
        rise = _tensor("jump rise", request_height_m, n, device)
        command = _tensor("command height", command_height_m, n, device)
        body = _tensor("body height", body_height_m, n, device)
        force = torch.as_tensor(wheel_force_n, dtype=torch.float32, device=device)
        tilt = _tensor("tilt", tilt_rad, n, device)
        base = _tensor("base contact", base_contact_force_n, n, device)
        velocity = (torch.zeros(n, device=device) if vertical_velocity_m_s is None
                    else _tensor("vertical velocity", vertical_velocity_m_s, n, device))
        if request.shape != (n,) or force.shape != (n, 2) or not torch.isfinite(force).all():
            raise ValueError("跳跃请求或轮力布局不一致")
        invalid_rise = request & ((rise < self.config.min_rise_m) | (rise > self.config.max_rise_m))
        if invalid_rise.any():
            raise ValueError("跳跃请求高度超出配置范围")

        normal = self.state == int(JumpState.NORMAL)
        entered = normal & request
        self.state = torch.where(entered, torch.full_like(self.state, int(JumpState.PRELOAD)), self.state)
        self.state_time = torch.where(entered, torch.zeros_like(self.state_time), self.state_time)
        self.landing_time = torch.where(entered, torch.zeros_like(self.landing_time), self.landing_time)
        self.settle_time = torch.where(entered, torch.zeros_like(self.settle_time), self.settle_time)
        self.request_height = torch.where(entered, rise, self.request_height)
        self.target_height = torch.where(entered, command + rise, self.target_height)
        next_request_id = self.request_counter + 1
        self.request_id = torch.where(entered, next_request_id, self.request_id)
        self.request_counter = torch.where(entered, next_request_id, self.request_counter)
        active = self.state != int(JumpState.NORMAL)
        self.state_time = torch.where(active, self.state_time + self.config.policy_dt_s, self.state_time)

        preload = active & (self.state == int(JumpState.PRELOAD)) & (self.state_time >= self.config.preload_s)
        self.state = torch.where(preload, torch.full_like(self.state, int(JumpState.TAKEOFF)), self.state)
        self.state_time = torch.where(preload, torch.zeros_like(self.state_time), self.state_time)
        takeoff = (self.state == int(JumpState.TAKEOFF)) & (self.state_time >= self.config.takeoff_s)
        self.state = torch.where(takeoff, torch.full_like(self.state, int(JumpState.FLIGHT)), self.state)
        self.state_time = torch.where(takeoff, torch.zeros_like(self.state_time), self.state_time)
        flight = self.state == int(JumpState.FLIGHT)
        landed = flight & ((body >= self.target_height - self.config.success_height_error_m)
                           | ((velocity <= 0) & (body > command + self.config.min_rise_m)))
        self.state = torch.where(landed, torch.full_like(self.state, int(JumpState.LANDING)), self.state)
        self.state_time = torch.where(landed, torch.zeros_like(self.state_time), self.state_time)
        landing = self.state == int(JumpState.LANDING)
        contact = (force > self.config.wheel_contact_force_n).all(-1)
        self.landing_time = torch.where(landing & contact,
                                        self.landing_time + self.config.policy_dt_s,
                                        torch.where(landing, torch.zeros_like(self.landing_time), self.landing_time))
        settled = landing & (self.landing_time >= self.config.settle_s)
        self.state = torch.where(settled, torch.full_like(self.state, int(JumpState.SETTLE)), self.state)
        self.state_time = torch.where(settled, torch.zeros_like(self.state_time), self.state_time)
        self.settle_time = torch.where(settled, torch.zeros_like(self.settle_time), self.settle_time)
        settle = self.state == int(JumpState.SETTLE)
        self.settle_time = torch.where(settle & contact,
                                       self.settle_time + self.config.policy_dt_s,
                                       torch.where(settle, torch.zeros_like(self.settle_time), self.settle_time))
        success = settle & contact & (self.settle_time >= self.config.settle_s) & (tilt <= self.config.failure_tilt_rad)
        failure = active & ((tilt > self.config.failure_tilt_rad)
                            | (base > self.config.base_contact_force_n)
                            | ((self.state == int(JumpState.FLIGHT)) & (self.state_time > self.config.flight_timeout_s))
                            | ((self.state == int(JumpState.LANDING)) & (self.state_time > self.config.landing_timeout_s)))
        exited = success | failure
        self.state = torch.where(exited, torch.full_like(self.state, int(JumpState.NORMAL)), self.state)
        self.state_time = torch.where(exited, torch.zeros_like(self.state_time), self.state_time)
        self.landing_time = torch.where(exited, torch.zeros_like(self.landing_time), self.landing_time)
        active_out = self.state != int(JumpState.NORMAL)
        self._height = command
        self._speed = torch.where(
            self.state == int(JumpState.PRELOAD), torch.full_like(self._speed, self.config.preload_speed_scale),
            torch.where(self.state == int(JumpState.LANDING),
                        torch.full_like(self._speed, self.config.landing_speed_scale),
                        torch.ones_like(self._speed)),
        )
        mode = torch.where(active_out, torch.full_like(self.state, int(CommandMode.JUMP)),
                           torch.full_like(self.state, int(CommandMode.NORMAL)))
        request_out = torch.where(active_out, self.request_height, torch.zeros_like(self.request_height))
        return JumpOutput(self.state.clone(), mode, self._height.clone(), request_out,
                          self._speed.clone(), self.state.clone(), self.request_id.clone(),
                          entered, exited, success, failure)

    def reset_after_success(self, ids=None):
        self.reset(ids)


@dataclass(frozen=True)
class SkillOutput:
    mode: torch.Tensor
    height_command_m: torch.Tensor
    request_height_m: torch.Tensor
    speed_scale: torch.Tensor
    stair_state: torch.Tensor
    jump_state: torch.Tensor
    jump_success: torch.Tensor
    jump_request_id: torch.Tensor
    entered: torch.Tensor
    exited: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    jump_ignored: torch.Tensor


class SkillFSM:
    """台阶与跳跃的统一仲裁器；两套状态机互斥。"""

    def __init__(self, num_envs: int, *, stair_config: StairFSMConfig = StairFSMConfig(),
                 jump_config: JumpFSMConfig = JumpFSMConfig(), device="cpu"):
        self.num_envs, self.device = num_envs, torch.device(device)
        self.stair = StairFSM(num_envs, stair_config, device=device)
        self.jump = JumpFSM(num_envs, jump_config, device=device)

    @property
    def active(self):
        return self.stair.active | self.jump.active

    def reset(self, ids=None):
        self.stair.reset(ids)
        self.jump.reset(ids)

    def step(self, *, step_height_m, wheel_force_n, body_height_m, tilt_rad,
             base_contact_force_n, command_height_m, auto_enabled=True,
             manual_stair_request=None, manual_height_m=None, jump_request=None,
             jump_request_height_m=None, vertical_velocity_m_s=None):
        n = self.num_envs
        manual = (torch.zeros(n, dtype=torch.bool, device=self.device)
                  if manual_stair_request is None else torch.as_tensor(manual_stair_request, dtype=torch.bool, device=self.device))
        request = (torch.zeros(n, dtype=torch.bool, device=self.device)
                   if jump_request is None else torch.as_tensor(jump_request, dtype=torch.bool, device=self.device))
        if manual.shape != (n,) or request.shape != (n,):
            raise ValueError("manual/jump request 需要 [N] bool")
        stair_active = self.stair.active
        jump_active = self.jump.active
        jump_start = request & ~stair_active & ~jump_active
        stair_inputs = dict(
            step_height_m=step_height_m, wheel_force_n=wheel_force_n,
            body_height_m=body_height_m, tilt_rad=tilt_rad,
            base_contact_force_n=base_contact_force_n,
            command_height_m=command_height_m,
            auto_enabled=auto_enabled & ~jump_start & ~jump_active,
            manual_request=manual & ~jump_start & ~jump_active,
            manual_height_m=manual_height_m,
        )
        stair_out = self.stair.step(**stair_inputs)
        rise = (torch.zeros(n, device=self.device) if jump_request_height_m is None
                else torch.as_tensor(jump_request_height_m, dtype=torch.float32, device=self.device))
        jump_out = self.jump.step(
            request=jump_start, request_height_m=rise, command_height_m=command_height_m,
            body_height_m=body_height_m, wheel_force_n=wheel_force_n, tilt_rad=tilt_rad,
            base_contact_force_n=base_contact_force_n, vertical_velocity_m_s=vertical_velocity_m_s,
        )
        jump_live = self.jump.active
        mode = torch.where(jump_live, jump_out.mode, stair_out.mode)
        height = torch.where(jump_live, jump_out.height_command_m, stair_out.height_command_m)
        request_height = torch.where(jump_live, jump_out.request_height_m, torch.zeros_like(jump_out.request_height_m))
        speed = torch.where(jump_live, jump_out.speed_scale, stair_out.speed_scale)
        return SkillOutput(
            mode=mode, height_command_m=height, request_height_m=request_height,
            speed_scale=speed, stair_state=stair_out.state, jump_state=jump_out.state,
            jump_success=jump_out.success,
            jump_request_id=jump_out.request_id,
            entered=stair_out.entered | jump_out.entered,
            exited=stair_out.exited | jump_out.exited,
            success=stair_out.success | jump_out.success,
            failure=stair_out.failure | jump_out.failure,
            jump_ignored=request & (stair_active | jump_active),
        )


__all__ = [
    "JumpFSM", "JumpFSMConfig", "JumpOutput", "JumpState",
    "SkillFSM", "SkillOutput", "StairFSM", "StairFSMConfig", "StairOutput", "StairState",
]

"""平地 DirectRLEnv 编排层。

只负责 IsaacLab 场景/传感器/物理接线；35D/81D/6D、历史、控制核心和
任务公式分别来自 lab_env.contract / control / observations。默认走
M1 flat_reward；配置 ``formal_reward_spec`` 时改走正式
PhysicalRewardComposer。100Hz 策略/200Hz 反馈的边界保持不变。
"""
from __future__ import annotations

from dataclasses import replace
import json
import time

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils

from ..control.joints import LegBoundaryGeometry
from ..control.observation import CommandFeatures, CommandMode, MechanicalBinding, NativePhysicalSample
from ..control.pressure import NativeSliderPressure, PressureParameters
from .actuator import WheeledControlCore
from .cfg import WheeledLabEnvCfg
from .contract import ACTION_DIM, FlatTaskSpec, flat_reward, flat_termination
from .formal_curriculum import LabTaskSample
from .formal_reward import FormalRewardConfig, LabFormalReward
from ..tasks.reward import RewardPhase
from .recover import LabRecoverTask, RecoverTaskSpec
from .recovery import LabRecoveryReset
from .observations import LabObservationBuilder
from .settings import LabEnvSettings, load_asset_description


class WheeledLabEnv(DirectRLEnv):
    """平地 M1；逐 physics substep 反馈，逐 policy step 观测/奖励。"""

    cfg: WheeledLabEnvCfg

    def __init__(self, cfg: WheeledLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._task = getattr(cfg, "task_spec", FlatTaskSpec())
        if not isinstance(self._task, FlatTaskSpec):
            raise TypeError("cfg.task_spec 必须是 FlatTaskSpec")
        self._asset = load_asset_description(cfg.asset_directory)
        self._binding = MechanicalBinding.from_manifest(
            self._asset.manifest, self.robot.joint_names, self.robot.body_names,
        )
        self._settings = LabEnvSettings(
            asset_directory=str(self._asset.directory),
            num_envs=self.num_envs,
            env_spacing_m=float(self.cfg.scene.env_spacing),
            sim_dt_s=float(self.cfg.sim.dt),
            decimation=int(self.cfg.decimation),
            episode_length_s=float(self.cfg.episode_length_s),
            device=str(self.device),
            history_length=int(self.cfg.history_length),
            task=self._task,
        )
        self._device = torch.device(self.device)
        self._active_indices = torch.tensor(self._binding.policy_joint_indices,
                                           dtype=torch.long, device=self._device)
        self._spring_indices = torch.tensor(
            [self._binding.native_joint_names.index(name) for name in self._asset.manifest["spring_joint_names"]],
            dtype=torch.long, device=self._device,
        )
        self._base_body_index = self._binding.native_body_names.index("base_link")
        self._action_bounds = torch.tensor(self._task.action_bounds, dtype=torch.float32,
                                           device=self._device)
        self._physics_dt_tensor = torch.full((self.num_envs,), float(self.physics_dt),
                                             dtype=torch.float64, device=self._device)

        response = getattr(cfg, "response_spec", None)
        pair_domain = getattr(cfg, "pair_domain_spec", None)
        self._control = WheeledControlCore(
            self._binding.nominal_policy_position_rad, self.num_envs,
            response=response, pair_domain=pair_domain, device=str(self._device),
        )
        self._pressure = self._build_pressure()
        self._observations = LabObservationBuilder(self._binding, self.num_envs,
                                                   device=self._device,
                                                   history_length=self.cfg.history_length)
        formal_config = getattr(cfg, "formal_reward_spec", None)
        self._formal_reward = None
        self._boundary_geometry = None
        if formal_config is not None:
            if not isinstance(formal_config, FormalRewardConfig):
                raise TypeError("cfg.formal_reward_spec 必须是 FormalRewardConfig")
            raw = (self._asset.directory / "manifest.json").read_bytes()
            spec = (self._asset.directory / "model_spec.json").read_bytes()
            self._boundary_geometry = LegBoundaryGeometry.from_asset_sources(raw, spec, self._binding)
            if "joint_boundary" not in formal_config.available_terms:
                formal_config = replace(
                    formal_config,
                    available_terms=tuple(dict.fromkeys((*formal_config.available_terms, "joint_boundary"))),
                )
            self._formal_reward = LabFormalReward(
                self._binding, self.num_envs, formal_config, device=str(self._device),
                boundary_geometry=self._boundary_geometry,
            )
            self._episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
            self._control_epoch = torch.zeros_like(self._episode_id)
            self._formal_initialized = torch.zeros(self.num_envs, dtype=torch.bool,
                                                   device=self._device)
            self._formal_result = None

        self._contact_force = torch.zeros(self.num_envs, len(self._binding.native_body_names), 3,
                                          dtype=torch.float32, device=self._device)
        self._contact_source, self._contact_target = self._contact_mapping()
        self._command = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self._command[:, 3] = self._task.nominal_height_m
        self._command_mode = torch.full((self.num_envs,), int(CommandMode.NORMAL),
                                        dtype=torch.long, device=self._device)
        self._reward_phase = torch.full((self.num_envs,), 0, dtype=torch.long, device=self._device)
        self._recovery_posture = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._command_sampler = None
        self._task_sampler = None
        self._recovery_reset = None
        recover_spec = getattr(cfg, "recover_spec", None)
        self._recover_task = None
        self._recover_initial_leg = None
        if recover_spec is not None:
            if not isinstance(recover_spec, RecoverTaskSpec):
                raise TypeError("cfg.recover_spec 必须是 RecoverTaskSpec")
            if self._formal_reward is None:
                raise ValueError("recover task 必须同时启用 formal reward")
            self._recover_task = LabRecoverTask(
                self.num_envs, recover_spec, device=str(self._device), seed=int(getattr(cfg, "seed", 0) or 0))
            self._recover_initial_leg = torch.zeros(self.num_envs, 4, dtype=torch.float32,
                                                    device=self._device)
        self._issued_action = torch.zeros(self.num_envs, ACTION_DIM, dtype=torch.float32,
                                          device=self._device)
        self._previous_action = torch.zeros_like(self._issued_action)
        self._before_action = torch.zeros_like(self._issued_action)
        self._last_effort = torch.zeros(self.num_envs, ACTION_DIM, dtype=torch.float32,
                                        device=self._device)
        self._reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._last_terminated = torch.zeros_like(self._terminated)
        self._last_truncated = torch.zeros_like(self._terminated)
        self._last_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._reward_terms: dict[str, torch.Tensor] = {}
        self._last_sample: NativePhysicalSample | None = None
        self._last_state = None
        self._last_tilt = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._last_base_contact = torch.zeros_like(self._terminated)
        self._targets = None
        self._physical_tick = 0
        self._timing: dict[str, float] = {}
        if self.cfg.events or self.cfg.action_noise_model or self.cfg.observation_noise_model:
            raise RuntimeError("M1 暂不接入事件/噪声管理器；请关闭 cfg.events/noise_model")
        if self._physics_handles_decimation:
            raise RuntimeError("M1 需要逐物理步 PD；当前后端折叠 decimation，请关闭该优化")

    def set_command_sampler(self, sampler):
        """安装 reset 行命令采样器；None 表示保持固定 nominal command。"""
        if sampler is not None and not callable(sampler):
            raise TypeError("command sampler 必须可调用")
        self._command_sampler = sampler

    def set_task_sampler(self, sampler):
        """安装正式 TaskDraw 采样器，并构建 recovery 出生姿态几何。"""
        if sampler is not None and not callable(sampler):
            raise TypeError("task sampler 必须可调用")
        self._task_sampler = sampler
        self._recovery_reset = None
        if sampler is not None:
            # 资产/碰撞源只读一次；8192 行共享同一 flat source。
            self._recovery_reset = LabRecoveryReset(self._asset, self.num_envs, device=str(self._device))

    def _build_pressure(self):
        asset = self._asset.directory
        model = json.loads((asset / "model_spec.json").read_text())
        fit = json.loads((asset / "fit_10mpa.json").read_text())
        parameters = PressureParameters.from_asset(self._asset.manifest, model, fit)
        return NativeSliderPressure(parameters, self.num_envs, device=str(self._device))

    def _contact_mapping(self):
        names = tuple(self.contact_sensor.body_names)
        source, target = [], []
        for index, name in enumerate(self._binding.native_body_names):
            if name in names:
                target.append(index)
                source.append(names.index(name))
        if not target:
            raise RuntimeError("接触传感器未覆盖任何 native body")
        return (torch.tensor(source, dtype=torch.long, device=self._device),
                torch.tensor(target, dtype=torch.long, device=self._device))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _as_torch(value):
        """IsaacLab 3.0 的 ArticulationData 是 ProxyArray；显式取零拷贝 torch 视图。"""
        if isinstance(value, torch.Tensor):
            return value
        tensor = getattr(value, "torch", None)
        if isinstance(tensor, torch.Tensor):
            return tensor
        raise TypeError("IsaacLab 数据字段不是 torch/ProxyArray")

    def _clock(self):
        return torch.full((self.num_envs,), self._physical_tick * float(self.physics_dt),
                          dtype=torch.float64, device=self._device)

    def _root_link_position(self):
        data = self.robot.data
        return self._as_torch(getattr(data, "root_link_pos_w", data.root_pos_w))

    def _root_link_quat(self):
        data = self.robot.data
        return self._as_torch(getattr(data, "root_link_quat_w", data.root_quat_w))

    def _sample(self) -> NativePhysicalSample:
        data = self.robot.data
        net = None
        if self.contact_sensor.data.net_forces_w is not None:
            net = self._as_torch(self.contact_sensor.data.net_forces_w)
        self._contact_force.zero_()
        if net is not None:
            self._contact_force.index_copy_(
                1, self._contact_target,
                net.index_select(1, self._contact_source),
            )
        position = self._root_link_position()
        quaternion_xyzw = self._root_link_quat()
        com_position = self._as_torch(getattr(data, "root_com_pos_w", position))
        com_velocity = self._as_torch(getattr(data, "root_com_vel_w", data.root_lin_vel_w))
        if com_velocity.shape[-1] == 6:
            com_velocity = com_velocity[..., :3]
        angular_velocity = self._as_torch(getattr(data, "root_ang_vel_w", data.root_ang_vel_b))
        ground = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        return NativePhysicalSample(
            self._as_torch(data.joint_pos),
            self._as_torch(data.joint_vel),
            position,
            quaternion_xyzw,
            com_position,
            com_velocity,
            angular_velocity,
            self._contact_force,
            ground,
            ground[:, None].expand(-1, 2),
            self._as_torch(self.scene.env_origins),
            torch.ones(self.num_envs, 2, dtype=torch.float32, device=self._device),
            self._clock(),
        )

    def _commands(self) -> CommandFeatures:
        return CommandFeatures(
            admitted_body_command=self._command,
            mode=self._command_mode.clone(),
            request_height_m=torch.zeros(self.num_envs, dtype=torch.float32, device=self._device),
            command_clock_s=self._clock().to(torch.float32),
            previous_issued_action=self._previous_action,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if (not isinstance(actions, torch.Tensor) or actions.shape != (self.num_envs, ACTION_DIM)
                or actions.device != self._device or not torch.isfinite(actions).all()):
            raise ValueError("actions 需要有限 [N,6] 张量")
        self._before_action.copy_(self._previous_action)
        self._issued_action.copy_(actions.clamp(-self._action_bounds, self._action_bounds))
        q = self._as_torch(self.robot.data.joint_pos).index_select(1, self._active_indices)
        self._targets = self._control.decode(self._issued_action, q, self._clock())
        if self._recover_task is not None:
            hold = self._recover_task.hold_mask
            self._command_mode.copy_(self._recover_task.task_mode(
                int(CommandMode.RECOVERY), int(CommandMode.NORMAL)))
            self._reward_phase.copy_(torch.where(
                hold, torch.full_like(hold, int(RewardPhase.SUPPORT), dtype=torch.long),
                torch.full_like(hold, int(RewardPhase.RECOVERY), dtype=torch.long)))
            if hold.any():
                zeros_wheel = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self._device)
                self._targets = replace(
                    self._targets,
                    raw_leg_position_rad=torch.where(
                        hold[:, None], self._recover_initial_leg, self._targets.raw_leg_position_rad),
                    admitted_leg_position_rad=torch.where(
                        hold[:, None], self._recover_initial_leg,
                        self._targets.admitted_leg_position_rad),
                    wheel_velocity_rad_s=torch.where(
                        hold[:, None], zeros_wheel, self._targets.wheel_velocity_rad_s),
                )
        if self._formal_reward is not None:
            dq = self._as_torch(self.robot.data.joint_vel).index_select(1, self._active_indices)
            self._formal_reward.begin(
                self._issued_action, dq, now_s=self._clock(), episode_id=self._episode_id,
                control_epoch=self._control_epoch, command=self._command,
                mode=self._command_mode,
            )

    def _apply_action(self) -> None:
        if self._targets is None:
            raise RuntimeError("必须先由 _pre_physics_step 生成输出目标")
        q = self._as_torch(self.robot.data.joint_pos).index_select(1, self._active_indices)
        dq = self._as_torch(self.robot.data.joint_vel).index_select(1, self._active_indices)
        now = self._clock()
        feedback = self._control.controller.effort(self._targets, q, dq, now)
        if self._control.drive is not None:
            effort = self._control.drive.advance(
                feedback.capped_effort, self._physics_dt_tensor, feedback.effort_bounds,
            ).mean_effort
        else:
            effort = feedback.capped_effort
        pressure = self._pressure.evaluate(
            self._as_torch(self.robot.data.joint_pos).index_select(1, self._spring_indices),
        )
        command = torch.zeros(self.num_envs, 18, dtype=torch.float32, device=self._device)
        command.index_copy_(1, self._active_indices, effort)
        command.index_copy_(1, self._spring_indices, pressure.slider_effort_n)
        self.robot.set_joint_effort_target(command)
        self._last_effort.copy_(effort)

    def _observe_substep(self) -> None:
        sample = self._sample()
        state = self._observations.assembler.measure(sample)
        terminated, tilt, base_contact = flat_termination(
            height_m=state.height_above_ground_m,
            gravity_body=state.gravity_body,
            base_contact_force_n=sample.terrain_contact_force_world_n[:, self._base_body_index].norm(dim=-1),
            spec=self._task,
        )
        recovery = self._command_mode == int(CommandMode.RECOVERY)
        terminated = torch.where(
            recovery,
            state.height_above_ground_m < self._task.minimum_recovery_height_m,
            terminated,
        )
        self._terminated |= terminated
        self._last_sample = sample
        self._last_state = state
        self._last_tilt = tilt
        self._last_base_contact = base_contact
        if self._formal_reward is not None:
            boundary = None
            if self._boundary_geometry is not None and self._targets is not None:
                boundary = self._boundary_geometry.evaluate(
                    self._as_torch(self.robot.data.body_link_pose_w),
                    self._as_torch(self.robot.data.joint_pos),
                    self._targets, knee_margin_rad=.05,
                ).shortfall_rad
            self._formal_reward.observe(
                sample, state, effort_nm=self._last_effort, episode_id=self._episode_id,
                control_epoch=self._control_epoch, command=self._command,
                mode=self._command_mode, phase=self._reward_phase,
                joint_boundary_shortfall_rad=boundary,
            )
        if self._recover_task is not None:
            base_load = sample.terrain_contact_force_world_n[:, self._base_body_index].norm(dim=-1)
            self._recover_task.success_mask(
                wheel_force_n=state.wheel_ground_force_n, base_load_n=base_load,
                height_m=state.height_above_ground_m, gravity_body=state.gravity_body,
            )

    def _reward_policy_step(self) -> torch.Tensor:
        if self._last_state is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        state = self._last_state
        velocity = torch.cat((state.velocity_origin_body_m_s,
                              state.angular_velocity_body_rad_s[:, 2:3]), -1)
        reward, terms = flat_reward(
            command=self._command[:, [0, 2, 3]],
            velocity_body=velocity,
            height_m=state.height_above_ground_m,
            gravity_body=state.gravity_body,
            action=self._issued_action,
            previous_action=self._before_action,
            effort=self._last_effort,
            spec=self._task,
            dt=self.step_dt,
        )
        self._reward_terms = terms
        return reward

    def step(self, action: torch.Tensor):
        started = time.perf_counter()
        action = action.to(self._device)
        self._pre_physics_step(action)
        self._terminated.zero_()
        is_rendering = self.sim.is_rendering
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)
            self._physical_tick += 1
            self._observe_substep()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        if self._formal_reward is not None:
            self._formal_result = self._formal_reward.finish(
                terminated=self._terminated, truncated=truncated & ~self._terminated,
            )
            self._last_reward.copy_(self._formal_result.learning_reward)
            self._reward_terms = self._formal_result.component_densities
            if self._recover_task is not None:
                self._last_reward.add_(self._recover_task.reward_bonus())
                if self._recover_task.spec.hold_reward_zero:
                    self._last_reward[self._recover_task.hold_mask] = 0.0
                self._recover_task.advance()
        else:
            self._last_reward.copy_(self._reward_policy_step())
        self._last_truncated.copy_(truncated & ~self._terminated)
        self._last_terminated.copy_(self._terminated)
        self.reset_terminated.copy_(self._last_terminated)
        self.reset_time_outs.copy_(self._last_truncated)
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._previous_action.copy_(self._issued_action)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
        self.obs_buf = self._get_observations()
        self._timing = {"step_s": time.perf_counter() - started}
        self.extras["timing"] = self._timing
        self.extras["reward_terms"] = self._reward_terms
        return (self.obs_buf, self._last_reward, self._last_terminated,
                self._last_truncated, self.extras)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        if ids.numel() == 0:
            return
        super()._reset_idx(ids)
        count = ids.numel()
        task_sample = None
        self._command_mode[ids] = int(CommandMode.NORMAL)
        self._reward_phase[ids] = 0
        self._recovery_posture[ids] = 0
        recover_plan = None
        if self._recover_task is not None:
            self._command[ids].zero_()
            self._command[ids, 3] = self._recover_task.spec.height_command_m
            recover_plan = self._recover_task.reset(
                ids, self.scene.env_origins.index_select(0, ids),
                self._as_torch(self.robot.data.default_joint_pos).index_select(0, ids))
        elif self._task_sampler is not None:
            task_sample = self._task_sampler(ids)
            if (not isinstance(task_sample, LabTaskSample) or task_sample.command.shape != (count, 4)
                    or task_sample.mode.shape != (count,) or task_sample.phase.shape != (count,)
                    or task_sample.command.dtype != torch.float32 or task_sample.command.device != self._device
                    or not torch.isfinite(task_sample.command).all()):
                raise ValueError("task sampler 必须返回有限 LabTaskSample")
            self._command[ids] = task_sample.command
            self._command_mode[ids] = task_sample.mode
            self._reward_phase[ids] = task_sample.phase
            self._recovery_posture[ids] = task_sample.draw.recovery_posture
        elif self._command_sampler is not None:
            sampled = self._command_sampler(ids)
            if (not isinstance(sampled, torch.Tensor) or sampled.shape != (count, 4)
                    or sampled.dtype != torch.float32 or sampled.device != self._device
                    or not torch.isfinite(sampled).all()):
                raise ValueError("command sampler 必须返回有限 [M,4] float32")
            self._command[ids] = sampled

        root_pose = torch.zeros(count, 7, dtype=torch.float32, device=self._device)
        root_pose[:, 2] = self._task.nominal_height_m
        root_pose[:, 6] = 1.0
        root_pose[:, :3] += self.scene.env_origins.index_select(0, ids)
        root_velocity = torch.zeros(count, 6, dtype=torch.float32, device=self._device)
        joint_position = self._as_torch(self.robot.data.default_joint_pos).index_select(0, ids).clone()
        joint_velocity = torch.zeros_like(joint_position)
        if recover_plan is not None:
            root_pose = recover_plan.root_pose_xyzw
            root_velocity = recover_plan.root_velocity
            joint_position = recover_plan.joint_position
            joint_velocity = recover_plan.joint_velocity
        if task_sample is not None:
            recovery_rows = task_sample.mode == int(CommandMode.RECOVERY)
            if recovery_rows.any():
                if self._recovery_reset is None:
                    raise RuntimeError("recovery task 需要已安装的 recovery reset")
                local = recovery_rows.nonzero(as_tuple=False).flatten()
                recovery_ids = ids.index_select(0, local)
                origins = self.scene.env_origins.index_select(0, recovery_ids).detach().cpu().to(torch.float64)
                pose = self._recovery_reset.pose(
                    recovery_ids.cpu(), task_sample.draw.recovery_posture.index_select(0, local).cpu(),
                    torch.zeros(local.numel(), dtype=torch.float64), origins)
                root_pose[local] = pose.root_pose_xyzw
                joint_position[local] = pose.joint_position
                root_velocity[local] = pose.root_com_velocity
                joint_velocity[local] = pose.joint_velocity
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        if self._control.drive is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            mask[ids] = True
            self._control.drive.reset(mask)
        self._terminated[ids] = False
        self._last_terminated[ids] = False
        self._last_truncated[ids] = False
        self._last_reward[ids] = 0.0
        self._last_effort[ids] = 0.0
        self._issued_action[ids] = 0.0
        self._previous_action[ids] = 0.0
        self._before_action[ids] = 0.0
        self._reset_mask[ids] = True
        if self._formal_reward is not None:
            fresh = ~self._formal_initialized[ids]
            self._episode_id[ids] += 1
            self._control_epoch[ids] += 1
            self._episode_id[ids[fresh]] = 1
            self._control_epoch[ids[fresh]] = 1
            self._formal_initialized[ids] = True
            self._formal_reward.reset(
                torch.zeros(self.num_envs, dtype=torch.bool, device=self._device).index_put_(
                    (ids,), torch.ones(ids.numel(), dtype=torch.bool, device=self._device)),
                self._clock(), self._episode_id, self._control_epoch,
            )
        if self._recover_task is not None and recover_plan is not None:
            self._recover_initial_leg[ids] = joint_position[
                :, self._active_indices[:4]]
        q = self._as_torch(self.robot.data.joint_pos).index_select(1, self._active_indices)
        self._targets = self._control.decode(
            torch.zeros(self.num_envs, ACTION_DIM, dtype=torch.float32, device=self._device),
            q, self._clock(),
        )

    def _get_observations(self):
        sample = self._sample()
        state = self._observations.assembler.measure(sample)
        commands = self._commands()
        sensors = self._observations.assembler.clean_sensors(state)
        actor, critic = self._observations.append(
            sample, commands, sensors, self._reset_mask, measured=state,
        )
        self._reset_mask.zero_()
        self.obs_buf = {"policy": actor.flatten(1).contiguous(),
                        "critic": critic.contiguous()}
        return self.obs_buf

    def _get_rewards(self):
        return self._last_reward

    def _get_dones(self):
        return self._last_terminated, self._last_truncated

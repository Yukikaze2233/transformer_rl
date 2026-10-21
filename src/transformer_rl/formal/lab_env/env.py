"""平地 DirectRLEnv 编排层。

只负责 IsaacLab 场景/传感器/物理接线；35D Actor/185D Critic/6D 动作、历史、控制核心和
任务公式分别来自 lab_env.contract / control / observations。默认走
M1 flat_reward；配置 ``formal_reward_spec`` 时改走正式
PhysicalRewardComposer。100Hz 策略/200Hz 反馈的边界保持不变。
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import time

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.sim as sim_utils

from ..control.joints import LegBoundaryGeometry
from ..control.observation import CommandFeatures, CommandMode, MechanicalBinding, NativePhysicalSample
from ..control.pressure import NativeSliderPressure, PressureParameters
from .actuator import WheeledControlCore
from .cfg import WheeledLabEnvCfg
from .contact_mapping import contact_body_mapping
from .contact_measurement import LabContactLoads, LabPhysicalSample
from .contract import ACTION_DIM, CRITIC182_DIM, FlatTaskSpec, flat_reward, flat_termination
from .formal_curriculum import LabTaskSample
from .formal_reward import FormalRewardConfig, LabFormalReward
from .critic_features import TERRAIN_SCAN_DIM, environment_features_182
from .warp_scan import miss_histograms
from .forward_scan import ForwardStepDetector
from .skill_fsm import JumpState, SkillFSM
from ..tasks.reward import RewardPhase
from ..tasks.termination import NativeTaskLimits, PhysicalFailureLatch
from .recover import LabRecoverTask, RecoverTaskSpec
from .recovery import LabRecoveryReset
from .materials import WheelMaterialReader
from .isolation import author_shared_ground_groups, isolate_scene
from .observations import LabObservationBuilder
from .settings import LabEnvSettings, load_asset_description
from .terrain_collision import partition_static_collision
from .dynamics import LabStartupDynamics
from .sensors import LabSensorModel


class WheeledLabEnv(DirectRLEnv):
    """平地 M1；逐 physics substep 反馈，逐 policy step 观测/奖励。"""

    cfg: WheeledLabEnvCfg

    def __init__(self, cfg: WheeledLabEnvCfg, render_mode: str | None = None, **kwargs):
        self._scan_debug = os.environ.get("LAB_SCAN_DEBUG") == "1"
        super().__init__(cfg, render_mode, **kwargs)
        if self._collision_isolation is not None:
            self._collision_isolation.finish_warmup()
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
        self._terrain_course = None
        self._completed_estimate = None
        self._active_indices = torch.tensor(self._binding.policy_joint_indices,
                                           dtype=torch.long, device=self._device)
        self._spring_indices = torch.tensor(
            [self._binding.native_joint_names.index(name) for name in self._asset.manifest["spring_joint_names"]],
            dtype=torch.long, device=self._device,
        )
        self._base_body_index = self._binding.native_body_names.index("base_link")
        self._wheel_body_indices = torch.tensor(
            self._binding.wheel_body_indices, dtype=torch.long, device=self._device)
        self._wheel_materials = WheelMaterialReader(
            self.robot, self._binding.wheel_body_indices,
            num_envs=self.num_envs, device=self._device)
        self._startup_dynamics = LabStartupDynamics(
            self.robot, self._binding, self.num_envs, self._asset.manifest['total_mass_kg'],
            int(getattr(cfg, 'seed', 0) or 0) + 15485863,
            getattr(cfg, 'startup_dynamics_spec', None), device=self._device,
            use_public_setters=cfg.declarative_scene)
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
        self._stiffness = self._control.controller.stiffness
        self._damping = self._control.controller.damping
        self._last_wheel_velocity_body = torch.zeros(self.num_envs, 6,
                                                     dtype=torch.float32, device=self._device)
        self._last_joint_acceleration = torch.zeros(self.num_envs, 6,
                                                    dtype=torch.float32, device=self._device)
        self._last_policy_velocity = torch.zeros(self.num_envs, 6,
                                                 dtype=torch.float32, device=self._device)
        self._terminal_state = None
        self._terminal_sample = None
        self._terminal_frame = None
        self._terminal_critic_182 = None
        self._ground_at_root = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._pressure = self._build_pressure()
        self._observations = LabObservationBuilder(self._binding, self.num_envs,
                                                   device=self._device,
                                                   history_length=self.cfg.history_length)
        sensor_configuration = getattr(cfg, 'sensor_model_spec', None)
        self._sensor_model = (None if sensor_configuration is None else LabSensorModel(
            self.num_envs, int(getattr(cfg, 'seed', 0) or 0) + 1299709,
            sensor_configuration, device=self._device))
        formal_config = getattr(cfg, "formal_reward_spec", None)
        self._formal_reward = None
        self._boundary_geometry = None
        self._failure_latch = None
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
            self._formal_reward.composer.update_native_mass(self._startup_dynamics.mass_kg_by_environment)
            limits = getattr(cfg, 'termination_limits', None)
            if not isinstance(limits, NativeTaskLimits):
                raise ValueError('formal reward cannot use the simplified M1 termination contract')
            self._forbidden_bodies = [self._binding.native_body_names.index(name)
                                      for name in limits.forbidden_body_names]
            self._failure_latch = PhysicalFailureLatch(
                limits, self.num_envs, physics_dt_s=float(self.physics_dt), device=self._device)
            self._episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
            self._control_epoch = torch.zeros_like(self._episode_id)
            self._formal_initialized = torch.zeros(self.num_envs, dtype=torch.bool,
                                                   device=self._device)
            self._formal_result = None

        self._contact_force = torch.zeros(self.num_envs, len(self._binding.native_body_names), 3,
                                          dtype=torch.float32, device=self._device)
        self._contact_source, self._contact_target = self._contact_mapping()
        self._contact_loads = None
        if (formal_config is not None
                and formal_config.normal_load_source == 'native_contact_patch_normal_forces'):
            self._contact_loads = LabContactLoads(
                self.contact_sensor, self._binding, self.scene.env_prim_paths,
                collision_path=tuple(self._course_collision_receipt['collider_paths']), device=self._device)
        self._command = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self._command[:, 3] = self._task.nominal_height_m
        self._command_base = self._command.clone()
        self._request_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
        self._command_mode = torch.full((self.num_envs,), int(CommandMode.NORMAL),
                                        dtype=torch.long, device=self._device)
        self._base_command_mode = self._command_mode.clone()
        self._skill_fsm = (SkillFSM(self.num_envs, device=str(self._device))
                           if (len(cfg.forward_scanners) or self._formal_reward is not None)
                           else None)
        self._forward_detector = (ForwardStepDetector(self.num_envs, device=str(self._device))
                                  if self._skill_fsm is not None else None)
        self._reward_phase = torch.full((self.num_envs,), 0, dtype=torch.long, device=self._device)
        self._pending_success_id = torch.full((self.num_envs,), -1, dtype=torch.long,
                                              device=self._device)
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
            if getattr(self.cfg, 'course_manifest', None):
                from .terrain import LabTerrainCourse
                self._terrain_course = LabTerrainCourse(self, self.cfg.course_manifest)

    @property
    def wheel_material_receipt(self):
        return self._wheel_materials.receipt

    @property
    def collision_isolation_receipt(self):
        return {'groups': dict(self._collision_group_receipt),
                'native': (self._collision_isolation.receipt
                           if self._collision_isolation is not None else None)}

    @property
    def terrain_collision_receipt(self):
        return self._course_collision_receipt

    def _build_pressure(self):
        asset = self._asset.directory
        model = json.loads((asset / "model_spec.json").read_text())
        fit = json.loads((asset / "fit_10mpa.json").read_text())
        parameters = PressureParameters.from_asset(self._asset.manifest, model, fit)
        return NativeSliderPressure(parameters, self.num_envs, device=str(self._device))

    @property
    def contact_mapping_receipt(self):
        return dict(self._contact_mapping_receipt)

    def _contact_mapping(self):
        source, target, required = contact_body_mapping(
            self._binding.native_body_names, self.contact_sensor.body_names,
            self._wheel_body_indices.tolist(), require_all=self._formal_reward is not None)
        self._contact_required_bodies = required
        self._contact_mapping_receipt = {
            'native_bodies': tuple(self._binding.native_body_names),
            'sensor_bodies': tuple(self.contact_sensor.body_names),
            'required_bodies': tuple(required),
            'source_indices': tuple(source), 'target_indices': tuple(target),
        }
        return (torch.tensor(source, dtype=torch.long, device=self._device),
                torch.tensor(target, dtype=torch.long, device=self._device))

    def _setup_scene(self):
        if self.cfg.declarative_scene:
            self.robot = self.scene.articulations['robot']
        else:
            self.robot = Articulation(self.cfg.robot_cfg)
            self.scene.articulations["robot"] = self.robot
        self.height_scanner = None
        if self.cfg.height_scanner is not None:
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner
        self.terrain_scanner = None
        if self.cfg.terrain_scanner is not None:
            self.terrain_scanner = RayCaster(self.cfg.terrain_scanner)
            self.scene.sensors["terrain_scanner"] = self.terrain_scanner
        self._forward_scanners = tuple(RayCaster(cfg) for cfg in self.cfg.forward_scanners)
        for index, scanner in enumerate(self._forward_scanners):
            self.scene.sensors[f"forward_scanner_{index}"] = scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        if self.cfg.declarative_scene:
            from isaaclab.utils import instantiate

            self.terrain = instantiate(self.cfg.terrain)
        else:
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._course_collision_receipt = (partition_static_collision(self.scene.stage)
            if getattr(self.cfg, 'course_manifest', None) else None)
        if self._course_collision_receipt is not None:
            self.cfg.contact_sensor.filter_prim_paths_expr = list(self._course_collision_receipt['collider_paths'])
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors['contact_sensor'] = self.contact_sensor
        if self._scan_debug:
            print("TERRAIN cfg", self.cfg.terrain.terrain_type,
                  getattr(self.cfg.terrain, "use_terrain_origins", None), flush=True)
            to = getattr(self.terrain, "terrain_origins", None)
            print("TERRAIN origins", None if to is None else tuple(to.shape),
                  None if to is None else (to.reshape(-1, 3).min(0).values.tolist(),
                                           to.reshape(-1, 3).max(0).values.tolist()), flush=True)
            eo = self.terrain.env_origins
            print("TERRAIN env origins", eo.min(0).values.tolist(), eo.max(0).values.tolist(), flush=True)
        if self.cfg.declarative_scene:
            self._collision_isolation = self.cfg._native_clone_binding
        else:
            self._collision_isolation = isolate_scene(self.scene)
            self.scene.clone_environments(copy_from_source=False)
        self._collision_group_receipt = author_shared_ground_groups(self.scene)
        if self.cfg.declarative_scene:
            self._collision_isolation.confirm_shared_ground_groups(self._collision_group_receipt)
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
    def _scan_height(self, scanner):
        hits = self._as_torch(scanner.data.ray_hits_w)
        if (hits.ndim != 3 or hits.shape[0] != self.num_envs or hits.shape[1] < 1
                or hits.shape[-1] != 3):
            raise ValueError("forward scanner 未返回 [N,Q,3] hits")
        height = hits[:, 0, 2] + 0.0
        return height, torch.isfinite(height)

    def _scan_ground(self):
        if not self._forward_scanners:
            floor = torch.zeros(self.num_envs, device=self._device)
            return floor, floor, torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        left_now, left_now_valid = self._scan_height(self._forward_scanners[0])
        left_ahead, left_ahead_valid = self._scan_height(self._forward_scanners[1])
        right_now, right_now_valid = self._scan_height(self._forward_scanners[2])
        right_ahead, right_ahead_valid = self._scan_height(self._forward_scanners[3])
        valid = left_now_valid & left_ahead_valid & right_now_valid & right_ahead_valid
        current = (left_now + right_now) * .5
        ahead = (left_ahead + right_ahead) * .5
        return current, ahead, valid

    def _terrain_scan_77(self):
        if self._terrain_course is not None:
            return self._terrain_course.scan_77()
        plane = getattr(self.cfg, 'static_plane_height_m', None)
        if plane is not None:
            ground = torch.full_like(self._ground_at_root, plane)
            height = self._root_link_position()[:, 2] - ground - self._task.nominal_height_m
            if not torch.isfinite(height).all():
                raise ValueError('nonfinite root position in the fixed-plane scan')
            return (5 * height[:, None]).clamp(-1, 1).expand(-1, TERRAIN_SCAN_DIM), ground
        if self.terrain_scanner is None:
            raise RuntimeError("182D critic 需要 terrain scanner；请启用 --terrain-scan")
        hits = self._as_torch(self.terrain_scanner.data.ray_hits_w)
        if (hits.ndim != 3 or hits.shape[0] != self.num_envs
                or hits.shape[1] != TERRAIN_SCAN_DIM or hits.shape[-1] != 3):
            raise ValueError(f"terrain scanner 需要 [N,{TERRAIN_SCAN_DIM},3] hits")
        root_z = self._root_link_position()[:, 2]
        if self._scan_debug:
            print("SCAN before warp miss_count", flush=True)
        per_env, per_point = miss_histograms(hits, root_z, self._task.nominal_height_m)
        misses = int(per_env.sum().item())
        if self._scan_debug:
            bad = (per_env > 0).nonzero().flatten()
            print("SCAN miss envs", bad.numel(), bad[:20].tolist(),
                  "env_counts", per_env[bad[:20]].tolist(), flush=True)
            print("SCAN miss points", per_point.tolist(), flush=True)
        if misses:
            raise ValueError(f"terrain scanner 有 {misses} 个未命中")
        center = hits[:, hits.shape[1] // 2, 2]
        height = root_z[:, None] - hits[..., 2] - self._task.nominal_height_m
        return (5 * height).clamp(-1, 1), center

    def _wheel_velocity_body(self):
        body_velocity = getattr(self.robot.data, "body_lin_vel_w", None)
        if body_velocity is None:
            raise RuntimeError("182D critic 需要 body_lin_vel_w 产生轮体速度")
        velocity = self._as_torch(body_velocity)
        if velocity.ndim != 3 or velocity.shape[:2] != (self.num_envs, len(self._binding.native_body_names)) or velocity.shape[-1] != 3:
            raise ValueError("body_lin_vel_w 布局不一致")
        wheels = velocity.index_select(1, self._wheel_body_indices).reshape(-1, 3)
        quaternion = self._root_link_quat()
        rotated = self._observations.assembler._inverse_rotate(
            quaternion[:, None, :].expand(-1, 2, -1).reshape(-1, 4), wheels).reshape(self.num_envs, 6)
        return rotated

    def _critic_182(self, state, sample, *, joint_acceleration=None):
        commands = self._commands()
        actor_frame = self._observations.assembler.actor(
            self._observations.assembler.clean_sensors(state), commands, now_s=state.timestamp_s)
        if joint_acceleration is None:
            joint_acceleration = self._last_joint_acceleration
        wheel_material = self._wheel_materials.features()
        com_offset = self._observations.assembler._inverse_rotate(
            sample.root_link_orientation_xyzw,
            sample.root_com_position_world_m - sample.root_link_position_world_m)
        sensor_packet = self._sensor_packet(state)
        observation_delay = ((state.timestamp_s[:, None] - sensor_packet.sampled_at_s) / self.step_dt).float()
        zeros2 = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self._device)
        terrain, ground = self._terrain_scan_77()
        self._ground_at_root.copy_(ground)
        height = (self._root_link_position()[:, 2] - ground).clamp_min(0.0)
        return environment_features_182(
            actor_frame=actor_frame,
            root_velocity_body_m_s=state.velocity_origin_body_m_s,
            height_above_ground_m=height[:, None],
            stiffness=self._stiffness[None].expand(self.num_envs, -1),
            damping=self._damping[None].expand(self.num_envs, -1),
            active_torque=self._last_effort,
            observation_delay_steps=observation_delay,
            action_delay_steps=zeros2,
            wheel_velocity_body_m_s=self._last_wheel_velocity_body,
            wheel_contact=(state.wheel_ground_force_n > 2.0).to(torch.float32),
            mass_scale=self._startup_dynamics.mass_scale,
            wheel_material=wheel_material,
            terrain_scan=terrain,
            previous_actions=torch.cat((self._before_action, self._previous_action), -1),
            joint_acceleration_rad_s2=joint_acceleration,
            com_offset_body_m=com_offset,
            nominal_pose_delta_rad=state.policy_position_rad - self._observations.assembler._nominal,
        )

    def _capture_terminal(self):
        if self._last_state is None or self._last_sample is None:
            return
        state, sample = self._last_state, self._last_sample
        self._terminal_state, self._terminal_sample = state, sample
        self._terminal_frame = self._observations.assembler.actor(
            self._sensor_packet(state), self._commands(),
            now_s=state.timestamp_s).detach().clone()
        terminal_acceleration = (state.policy_velocity_rad_s - self._last_policy_velocity) / self.step_dt
        self._last_policy_velocity.copy_(state.policy_velocity_rad_s)
        self._last_joint_acceleration.copy_(terminal_acceleration)
        self._terminal_critic_182 = self._critic_182(
            state, sample, joint_acceleration=terminal_acceleration).detach().clone()

    def _update_skill(self):
        if self._terrain_course is not None:
            self._terrain_course.advance_skill()
            return
        if self._skill_fsm is None or self._last_state is None or self._last_sample is None:
            return
        current, ahead, valid = self._scan_ground()
        step_height = torch.where(valid, (ahead - current).clamp_min(0),
                                  torch.zeros_like(current))
        state = self._last_state
        base_force = self._last_sample.terrain_contact_force_world_n[
            :, self._base_body_index].norm(dim=-1)
        recovery = self._base_command_mode == int(CommandMode.RECOVERY)
        jump_request = (self._request_height > 0) & ~recovery
        out = self._skill_fsm.step(
            step_height_m=step_height,
            wheel_force_n=state.wheel_ground_force_n,
            body_height_m=state.height_above_ground_m,
            tilt_rad=self._last_tilt,
            base_contact_force_n=base_force,
            command_height_m=self._command_base[:, 3],
            auto_enabled=~recovery,
            jump_request=jump_request,
            jump_request_height_m=self._request_height,
            vertical_velocity_m_s=self._last_sample.root_com_velocity_world_m_s[:, 2],
        )
        active = out.mode != int(CommandMode.NORMAL)
        self._command[:, 3] = torch.where(active, out.height_command_m, self._command_base[:, 3])
        self._command[:, 0] = self._command_base[:, 0] * torch.where(
            active, out.speed_scale, torch.ones_like(out.speed_scale))
        self._command_mode.copy_(torch.where(active, out.mode, self._base_command_mode))
        self._request_height.copy_(torch.where(
            out.mode == int(CommandMode.JUMP), out.request_height_m,
            torch.zeros_like(out.request_height_m)))
        jump_phase = torch.where(
            (out.jump_state == int(JumpState.PRELOAD)) | (out.jump_state == int(JumpState.TAKEOFF)),
            torch.full_like(out.jump_state, int(RewardPhase.TAKEOFF)),
            torch.where(out.jump_state == int(JumpState.FLIGHT),
                        torch.full_like(out.jump_state, int(RewardPhase.FLIGHT)),
                        torch.where(out.jump_state == int(JumpState.LANDING),
                                    torch.full_like(out.jump_state, int(RewardPhase.LANDING)),
                                    torch.full_like(out.jump_state, int(RewardPhase.SETTLING)))))
        self._reward_phase.copy_(torch.where(
            out.mode == int(CommandMode.JUMP), jump_phase,
            torch.where(self._base_command_mode == int(CommandMode.RECOVERY),
                        torch.full_like(out.mode, int(RewardPhase.RECOVERY)),
                        torch.full_like(out.mode, int(RewardPhase.SUPPORT)))))
        self._pending_success_id.copy_(torch.where(
            out.jump_success, out.jump_request_id,
            torch.full_like(out.jump_request_id, -1)))


    def _root_link_position(self):
        return self._as_torch(self.robot.data.root_link_pos_w)

    def _root_link_quat(self):
        # beta2 与 GA 均为 xyzw；不能再按旧 wxyz 约定二次重排。
        return self._as_torch(self.robot.data.root_link_quat_w)

    def _sample(self) -> NativePhysicalSample:
        data = self.robot.data
        net = None
        forces = (self.contact_sensor.data.net_normal_forces_w if self.cfg.declarative_scene
                  else self.contact_sensor.data.net_forces_w)
        if forces is not None:
            net = self._as_torch(forces)
        self._contact_force.zero_()
        if net is not None:
            self._contact_force.index_copy_(
                1, self._contact_target,
                net.index_select(1, self._contact_source),
            )
        position = self._root_link_position()
        quaternion_xyzw = self._root_link_quat()
        com_position = self._startup_dynamics.base_com_position(position, quaternion_xyzw)
        com_twist_world = self._as_torch(data.root_com_vel_w)
        com_velocity, angular_velocity = com_twist_world[..., :3], com_twist_world[..., 3:]
        # The 11x7 terrain scanner supplies the base ground reference once per
        # observation. _sample stays free of GPU ray-hit reductions, which can
        # stall on large rough-terrain batches.
        ground = self._ground_at_root
        wheels_ground = ground[:, None].expand(-1, 2)
        origins = self._as_torch(self.scene.env_origins)
        if self._terrain_course is not None:
            ground, wheels_ground = self._terrain_course.ground()
            origins = self._terrain_course.router.origins
        result = NativePhysicalSample(
            self._as_torch(data.joint_pos),
            self._as_torch(data.joint_vel),
            position,
            quaternion_xyzw,
            com_position,
            com_velocity,
            angular_velocity,
            self._contact_force,
            ground,
            wheels_ground,
            origins,
            torch.ones(self.num_envs, 2, dtype=torch.float32, device=self._device),
            self._clock(),
        )
        if self._contact_loads is not None:
            normal, support = self._contact_loads.read(self._physical_tick, float(self.physics_dt))
            return LabPhysicalSample(**result.__dict__, body_normal_load_n=normal,
                                     wheel_support_load_n=support,
                                     contact_measurement_valid=self._contact_loads.valid)
        return result

    def _commands(self) -> CommandFeatures:
        return CommandFeatures(
            admitted_body_command=self._command,
            mode=self._command_mode.clone(),
            request_height_m=self._request_height.clone(),
            command_clock_s=(self._terrain_course.tasks.request_clock.clone()
                             if self._terrain_course is not None else self._clock().to(torch.float32)),
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
        if self._sensor_model is not None:
            self._sensor_model.push(state, self._physical_tick)
        boundary = None
        if self._failure_latch is not None:
            boundary = self._boundary_geometry.evaluate(
                self._as_torch(self.robot.data.body_link_pose_w),
                self._as_torch(self.robot.data.joint_pos), self._targets, knee_margin_rad=.05)
            pressure = self._pressure.evaluate(
                self._as_torch(self.robot.data.joint_pos).index_select(1, self._spring_indices))
            allowed = (None if self._terrain_course is None
                       else self._terrain_course.body_contact_allowed)
            conditions = self._failure_latch.observe(
                height_m=state.height_above_ground_m, gravity_body=state.gravity_body,
                forbidden_body_load_n=(sample.body_normal_load_n[:, self._forbidden_bodies]
                    if isinstance(sample, LabPhysicalSample) else sample.terrain_contact_force_world_n[
                        :, self._forbidden_bodies].norm(dim=-1)),
                knee_signed_clearance_rad=boundary.knee_signed_clearance_rad,
                knee_pivot_residual_m=boundary.knee_pivot_residual_m,
                within_pressure_stroke=pressure.within_mechanical_stroke,
                recovery=self._command_mode == int(CommandMode.RECOVERY),
                body_contact_allowed=allowed)
            terminated = conditions.true_failure
            tilt = torch.atan2(state.gravity_body[:, :2].norm(dim=-1), -state.gravity_body[:, 2])
            base_contact = sample.terrain_contact_force_world_n[
                :, self._base_body_index].norm(dim=-1) > self._failure_latch.limits.forbidden_body_load_n
        else:
            terminated, tilt, base_contact = flat_termination(
                height_m=state.height_above_ground_m, gravity_body=state.gravity_body,
                base_contact_force_n=sample.terrain_contact_force_world_n[:, self._base_body_index].norm(dim=-1),
                spec=self._task)
            recovery = self._command_mode == int(CommandMode.RECOVERY)
            terminated = torch.where(recovery, state.height_above_ground_m < self._task.minimum_recovery_height_m,
                                     terminated)
            if self._terrain_course is not None:
                terminated = self._terrain_course.termination(terminated, state, tilt, base_contact)
        self._terminated |= terminated
        self._last_sample = sample
        self._last_state = state
        self._last_wheel_velocity_body.copy_(self._wheel_velocity_body())
        self._last_tilt = tilt
        self._last_base_contact = base_contact
        if self._formal_reward is not None:
            boundary = boundary.shortfall_rad
            jump = self._command_mode == int(CommandMode.JUMP)
            zero = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
            jump_height_world = self._ground_at_root + self._command[:, 3] + self._request_height
            if self._terrain_course is not None:
                self._terrain_course.observe_reward(sample, state, boundary)
            else:
                self._formal_reward.observe(
                    sample, state, effort_nm=self._last_effort, episode_id=self._episode_id,
                    control_epoch=self._control_epoch, command=self._command,
                    mode=self._command_mode, phase=self._reward_phase,
                    jump_height_world_z_m=torch.where(jump, jump_height_world, zero),
                    jump_velocity_world_z_m_s=zero,
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
        time_limit = self.episode_length_buf >= self.max_episode_length - 1
        spatial_exit = torch.zeros_like(time_limit)
        if self._terrain_course is not None:
            spatial_exit = self._terrain_course.router.observe_end(self._root_link_position(), self._terminated)
        truncated = time_limit | spatial_exit
        # Keep causes separate even when both share bootstrap/trace masks.
        # The collector transfers these small counters once per rollout.
        self.extras['termination_counts'] = torch.stack((
            self._terminated.sum(), (time_limit & ~self._terminated).sum(),
            (spatial_exit & ~self._terminated).sum(),
            (time_limit & spatial_exit & ~self._terminated).sum()))
        if self._formal_reward is not None:
            self._formal_result = self._formal_reward.finish(
                terminated=self._terminated, truncated=truncated & ~self._terminated,
                physical_success_request_id=(self._terrain_course.tasks.success_id
                                             if self._terrain_course is not None else self._pending_success_id),
            )
            self._pending_success_id.fill_(-1)
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
        # Preserve the interval's done flags before _reset_idx clears the
        # internal row state for the next step. GAE must cut on these flags.
        terminated_out = self._last_terminated.clone()
        truncated_out = self._last_truncated.clone()
        self._previous_action.copy_(self._issued_action)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        self._update_skill()
        # Reward used this interval's command. Bootstrap describes the next
        # decision endpoint, including its newly advanced command, before any
        # physical row reset changes the terminal state or sensor packet.
        self._capture_terminal()
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
        self.obs_buf = self._get_observations()
        self._timing = {"step_s": time.perf_counter() - started}
        self.extras["timing"] = self._timing
        self.extras["reward_terms"] = self._reward_terms
        return (self.obs_buf, self._last_reward, terminated_out,
                truncated_out, self.extras)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        if ids.numel() == 0:
            return
        if self._scan_debug:
            print("RESET enter", ids.numel(), flush=True)
        super()._reset_idx(ids)
        if self._scan_debug:
            print("RESET after super", flush=True)
        count = ids.numel()
        task_sample = None
        self._command_mode[ids] = int(CommandMode.NORMAL)
        self._reward_phase[ids] = 0
        self._recovery_posture[ids] = 0
        self._request_height[ids] = 0
        self._pending_success_id[ids] = -1
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
            if self._terrain_course is None and (task_sample.terrain_kind != 0).any():
                raise RuntimeError("LabEnv 尚未安装 terrain FSM 路由")
            self._command[ids] = task_sample.command
            self._request_height[ids] = task_sample.request_height_m
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

        self._command_base[ids] = self._command[ids]
        self._base_command_mode[ids] = self._command_mode[ids]
        if self._skill_fsm is not None:
            self._skill_fsm.reset(ids)
            self._forward_detector.reset(ids)
        root_pose = torch.zeros(count, 7, dtype=torch.float32, device=self._device)
        root_pose[:, 2] = self._task.nominal_height_m
        root_pose[:, 6] = 1.0
        root_pose[:, :3] += self.scene.env_origins.index_select(0, ids)
        if self._terrain_course is not None:
            self._terrain_course.place_reset(ids, task_sample, root_pose)
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
                all_origins = (self._terrain_course.router.origins if self._terrain_course is not None
                               else self.scene.env_origins)
                origins = all_origins.index_select(0, recovery_ids).detach().cpu().to(torch.float64)
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
        if self._contact_loads is not None:
            self._contact_loads.reset_rows(ids, self._physical_tick)
        if self._scan_debug:
            print("RESET after scene.update", flush=True)
        if self._control.drive is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            mask[ids] = True
            self._control.drive.reset(mask)
        self._terminated[ids] = False
        if self._failure_latch is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            mask[ids] = True
            self._failure_latch.reset(mask)
        self._last_terminated[ids] = False
        self._last_truncated[ids] = False
        # Keep the terminal reward in _last_reward until the next policy step;
        # the collector must return the PRE-reset reward for done rows.
        self._last_effort[ids] = 0.0
        self._last_wheel_velocity_body[ids] = 0.0
        self._last_joint_acceleration[ids] = 0.0
        self._last_policy_velocity[ids] = 0.0
        self._issued_action[ids] = 0.0
        self._previous_action[ids] = 0.0
        self._before_action[ids] = 0.0
        self._reset_mask[ids] = True
        if self._terrain_course is not None:
            self._terrain_course.reset_commands(ids, task_sample)
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
        if self._sensor_model is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            mask[ids] = True
            self._sensor_model.reset(mask, self._clock())
            current = self._observations.assembler.measure(self._sample())
            self._sensor_model.seed_after_reset(mask, current)

    def _sensor_packet(self, state):
        return (self._observations.assembler.clean_sensors(state) if self._sensor_model is None
                else self._sensor_model(state))

    def _get_observations(self):
        if self._scan_debug:
            print("OBS enter", flush=True)
        sample = self._sample()
        if self._scan_debug:
            print("OBS after sample", flush=True)
        state = self._observations.assembler.measure(sample)
        if self._scan_debug:
            print("OBS after measure", flush=True)
        commands = self._commands()
        sensors = self._sensor_packet(state)
        actor, _ = self._observations.append(
            sample, commands, sensors, self._reset_mask, measured=state,
        )
        if self._scan_debug:
            print("OBS after append", flush=True)
        reset = self._reset_mask
        acceleration = torch.where(reset[:, None], torch.zeros_like(self._last_joint_acceleration),
                                  self._last_joint_acceleration)
        self._last_joint_acceleration.copy_(acceleration)
        self._last_policy_velocity.copy_(state.policy_velocity_rad_s)
        if reset.any():
            self._last_wheel_velocity_body[reset] = 0.0
        critic = self._critic_182(state, sample, joint_acceleration=acceleration)
        self._reset_mask.zero_()
        self.obs_buf = {"policy": actor.flatten(1).contiguous(),
                        "critic": critic.contiguous()}
        return self.obs_buf

    def _get_rewards(self):
        return self._last_reward

    def _get_dones(self):
        return self._last_terminated, self._last_truncated

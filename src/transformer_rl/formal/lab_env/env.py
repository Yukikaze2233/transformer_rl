"""平地 DirectRLEnv 编排层。

只负责 IsaacLab 场景/传感器/物理接线；35D Actor/185D Critic/6D 动作、历史、控制核心和
任务公式分别来自 lab_env.contract / control / observations。默认走
M1 flat_reward；配置 ``formal_reward_spec`` 时改走正式
PhysicalRewardComposer。50Hz 策略/200Hz 反馈的边界保持不变。
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
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
from ..control.ownership import ControlSource
from .actuator import WheeledControlCore
from .cfg import WheeledLabEnvCfg
from .contact_mapping import contact_body_mapping
from .contact_measurement import LabContactLoads, LabPhysicalSample
from .contract import ACTION_DIM, CRITIC182_DIM, FlatTaskSpec, flat_reward, flat_termination
from .formal_curriculum import LabTaskSample
from .formal_reward import FormalRewardConfig, LabFormalReward
from .critic_features import CriticActionHistory, TERRAIN_SCAN_DIM, environment_features_182
from .warp_scan import miss_histograms
from .forward_scan import ForwardStepDetector
from .skill_fsm import JumpFSMConfig, JumpState, SkillFSM, StairFSMConfig
from ..tasks.reward import RewardPhase, wheel_positions_in_base
from ..tasks.termination import NativeTaskLimits, PhysicalFailureLatch
from .recover import LabRecoverTask, RecoverTaskSpec
from .recovery import LabRecoveryReset, _RecoveryPreparation
from .materials import WheelMaterialReader
from .spring_force_element import SpringForceElement
from .isolation import author_shared_ground_groups, isolate_scene
from .observations import LabObservationBuilder
from .settings import LabEnvSettings, load_asset_description
from .terrain_collision import partition_static_collision
from .dynamics import LabStartupDynamics
from .sensors import LabSensorModel
from .action_transport import DelayedActionTransport, SubstepActionLatch
from .birth import BirthRandomization
from ..control.invariants import DeferredInvariants, require
from ..simulation.ground import WheelCollisionCentres
from ..control.communication import EpisodeRandomization, PhysicalCommunication
from .joint_coordinates import ContinuousJointCoordinates
from .mechanics import NativeLoopTelemetry


def _quat_mul(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), -1)


def _yaw(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _select_targets(mask, held, current):
    """Row-wise choice between two decoded target packets of the same layout."""
    values = []
    for name in ('issued_action', 'raw_leg_position_rad', 'admitted_leg_position_rad', 'wheel_velocity_rad_s',
                 'pair_projected', 'common_turns', 'controller_issued_at_s', 'control_source'):
        a, b = getattr(held, name), getattr(current, name)
        values.append(torch.where(mask.view(-1, *([1] * (b.ndim - 1))), a, b))
    return type(current)(*values)


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
        self._joint_coordinates = ContinuousJointCoordinates(
            self._as_torch(self.robot.data.joint_pos), self._binding.native_joint_names,
            self._asset.continuous_joint_names(), float(self.physics_dt))
        model_spec = json.loads(self._asset.bound_bytes('model_spec.json', maximum_bytes=256 * 1024))
        self._loop_telemetry = NativeLoopTelemetry(
            model_spec['constraints'], self.robot.body_names, self.num_envs, self._device,
            enabled=getattr(cfg, 'loop_telemetry_enabled', False))
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
        wheel_material = getattr(cfg, 'wheel_material_spec', None)
        if wheel_material is not None:
            self._wheel_materials.randomize_startup(
                wheel_material, seed=int(getattr(cfg, 'seed', 0) or 0) + 32452843)
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
        # Per-substep value checks are read once per policy interval (no host sync each).
        self._invariants = DeferredInvariants(self._device)
        self._control.controller.invariants = self._invariants
        if self._control.drive is not None:
            self._control.drive.invariants = self._invariants
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
        self._spring_force = self._build_spring_force()
        self._spring_strength = torch.ones(self.num_envs, 1, dtype=torch.float32, device=self._device)
        episode_configuration = getattr(cfg, 'episode_randomization_spec', None)
        self._episode_randomization = (None if episode_configuration is None or not episode_configuration.enabled
            else EpisodeRandomization(self.num_envs, episode_configuration,
                seed=int(getattr(cfg, 'seed', 0) or 0) + 49979687, device=self._device))
        if self._episode_randomization is not None:
            self._spring_strength = self._episode_randomization.strength
        self._spring_source_sha256 = getattr(self, '_spring_source_sha256', {})
        # Rollout-level mechanical telemetry. Accumulate device tensors here;
        # read them once at the collector boundary, not per physics substep.
        self._spring_mount_min = torch.full((), float('inf'), dtype=torch.float32, device=self._device)
        self._spring_mount_max = torch.full((), -float('inf'), dtype=torch.float32, device=self._device)
        self._spring_compression_min = torch.full((), float('inf'), dtype=torch.float32, device=self._device)
        self._spring_compression_max = torch.full((), -float('inf'), dtype=torch.float32, device=self._device)
        self._spring_slider_compression_min = torch.full((), float('inf'), dtype=torch.float32, device=self._device)
        self._spring_slider_compression_max = torch.full((), -float('inf'), dtype=torch.float32, device=self._device)
        self._spring_force_max = torch.full((), 0., dtype=torch.float32, device=self._device)
        self._spring_compression_mismatch_max = torch.full((), 0., dtype=torch.float32, device=self._device)
        # Last physical-substep values, kept per side for the two mounts.
        self._spring_mount_compression_last = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self._device)
        self._spring_slider_compression_last = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self._device)
        self._spring_mount_stroke_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_slider_stroke_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_mount_fit_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_slider_fit_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_mount_valid_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_slider_valid_last = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self._device)
        self._spring_telemetry_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_mount_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_slider_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_mount_stroke_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_slider_stroke_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_mount_fit_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._spring_slider_fit_invalid_samples = torch.zeros((), dtype=torch.long, device=self._device)
        self._last_reset_closure_max = None
        self._last_reset_q_closure_max = None
        self._last_reset_pose_closure_max = None
        self._last_reset_native_closure_max = None
        self._last_reset_support_clearance = None
        self._last_reset_support_clearance_lr = None
        self._last_reset_support_rows = 0
        self._last_reset_support_bilateral_rows = 0
        self._reset_support_left_min = None
        self._reset_support_right_min = None
        self._reset_support_rows = 0
        self._reset_support_bilateral_rows = 0
        self._observations = LabObservationBuilder(self._binding, self.num_envs,
                                                   device=self._device,
                                                   history_length=self.cfg.history_length)
        sensor_configuration = getattr(cfg, 'sensor_model_spec', None)
        communication_configuration = getattr(cfg, 'communication_spec', None)
        self._communication = (None if communication_configuration is None or not communication_configuration.enabled
            else PhysicalCommunication(self.num_envs, communication_configuration,
                seed=int(getattr(cfg, 'seed', 0) or 0) + 67867967, device=self._device))
        uplink = ({} if self._communication is None or communication_configuration.uplink_fraction == 0
                  or communication_configuration.total_delay_range_s[1] == 0 else {
            'uplink_delay_s': self._communication.uplink_s,
            'maximum_uplink_s': (communication_configuration.total_delay_range_s[1]
                                  * communication_configuration.uplink_fraction)})
        self._sensor_model = (None if sensor_configuration is None else LabSensorModel(
            self.num_envs, int(getattr(cfg, 'seed', 0) or 0) + 1299709,
            sensor_configuration, device=self._device, **uplink))
        transport_configuration = getattr(cfg, 'action_transport_spec', None)
        self._action_transport = (None if transport_configuration is None else DelayedActionTransport(
            self.num_envs, ACTION_DIM, transport_configuration,
            seed=int(getattr(cfg, 'seed', 0) or 0) + 32452843, device=self._device))
        latch_configuration = getattr(cfg, 'action_latch_spec', None)
        self._action_latch = (None if latch_configuration is None or not latch_configuration.enabled
            else SubstepActionLatch(self.num_envs, latch_configuration,
                                    seed=int(getattr(cfg, 'seed', 0) or 0) + 15485863, device=self._device))
        if self._action_latch is not None and transport_configuration is not None and transport_configuration.enabled:
            raise ValueError('policy-target delay needs one model: whole-step transport and the substep latch both enabled')
        self._latch_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._held_targets = None
        self._held_generated_at_s = torch.zeros(self.num_envs, dtype=torch.float64, device=self._device)
        self._substep = 0
        self._motor_envelope = getattr(cfg, 'motor_envelope_spec', None)
        self._birth_randomization = None
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
        self._skill_fsm = (SkillFSM(
            self.num_envs,
            stair_config=StairFSMConfig(policy_dt_s=float(self.step_dt)),
            jump_config=JumpFSMConfig(policy_dt_s=float(self.step_dt)),
            device=str(self._device))
            if (len(cfg.forward_scanners) or self._formal_reward is not None) else None)
        self._forward_detector = (ForwardStepDetector(self.num_envs, device=str(self._device))
                                  if self._skill_fsm is not None else None)
        self._reward_phase = torch.full((self.num_envs,), 0, dtype=torch.long, device=self._device)
        self._pending_success_id = torch.full((self.num_envs,), -1, dtype=torch.long,
                                              device=self._device)
        self._recovery_posture = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._command_sampler = None
        self._task_sampler = None
        self._recovery_reset = None
        self._recovery_preparation = None
        self._preparation_handoff = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
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
        self._previous_control_source = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._before_action = torch.zeros_like(self._issued_action)
        self._critic_action_history = CriticActionHistory(self.num_envs, device=self._device)
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
        self._recovery_preparation = None
        if sampler is not None:
            # 资产/碰撞源只读一次；8192 行共享同一 flat source。
            self._recovery_reset = LabRecoveryReset(self._asset, self.num_envs, device=str(self._device))
            self._recovery_preparation = _RecoveryPreparation(
                self._control.controller, policy_dt_s=float(self.step_dt),
                timeout_s=self._failure_latch.limits.preparation_timeout_s)
            birth = getattr(self.cfg, 'birth_randomization_spec', None)
            if birth is not None and birth.enabled:
                wheels = WheelCollisionCentres.from_urdf(
                    (self._asset.directory / self._asset.manifest['urdf']).read_bytes(), self._binding)
                self._birth_randomization = BirthRandomization(
                    self._recovery_reset, birth, wheels, seed=int(getattr(self.cfg, 'seed', 0) or 0) + 86028121,
                    device=self._device, pair_domain=self._control.controller.pair_domain)
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

    def reset_mechanical_metrics(self):
        self._loop_telemetry.clear_window()
        self._spring_mount_min.fill_(float('inf'))
        self._spring_mount_max.fill_(-float('inf'))
        self._spring_compression_min.fill_(float('inf'))
        self._spring_compression_max.fill_(-float('inf'))
        self._spring_slider_compression_min.fill_(float('inf'))
        self._spring_slider_compression_max.fill_(-float('inf'))
        self._spring_force_max.zero_()
        self._spring_compression_mismatch_max.zero_()
        self._spring_mount_compression_last.zero_()
        self._spring_slider_compression_last.zero_()
        self._spring_mount_stroke_last.zero_()
        self._spring_slider_stroke_last.zero_()
        self._spring_mount_fit_last.zero_()
        self._spring_slider_fit_last.zero_()
        self._spring_mount_valid_last.zero_()
        self._spring_slider_valid_last.zero_()
        self._spring_telemetry_samples.zero_()
        self._spring_mount_invalid_samples.zero_()
        self._spring_slider_invalid_samples.zero_()
        self._spring_mount_stroke_invalid_samples.zero_()
        self._spring_slider_stroke_invalid_samples.zero_()
        self._spring_mount_fit_invalid_samples.zero_()
        self._spring_slider_fit_invalid_samples.zero_()
        self._last_reset_closure_max = None
        self._last_reset_q_closure_max = None
        self._last_reset_pose_closure_max = None
        self._last_reset_native_closure_max = None
        self._last_reset_support_clearance = None
        self._last_reset_support_clearance_lr = None
        self._last_reset_support_rows = 0
        self._last_reset_support_bilateral_rows = 0
        self._reset_support_left_min = None
        self._reset_support_right_min = None
        self._reset_support_rows = 0
        self._reset_support_bilateral_rows = 0

    @property
    def mechanical_metrics(self):
        """Rollout mechanical evidence, read only at the collection boundary.

        ``last_reset_*`` describes only the rows reset by the most recent
        partial or ordinary reset call. Reset-window fields aggregate every
        reset observed since ``reset_mechanical_metrics``. ``six_loop_*``
        separates forward(dt=0), the first returned physical substep and the
        full physical window.
        """
        def scalar(value):
            if value is None:
                return None
            value = float(value.detach().cpu())
            return value if math.isfinite(value) else None

        def side_min(value, side):
            if value is None:
                return None
            selected = value[:, side]
            finite = selected[torch.isfinite(selected)]
            return float(finite.amin().detach().cpu()) if finite.numel() else None

        def side_last(value, side):
            if value is None:
                return None
            return scalar(value[:, side].amin())

        support_lr = self._last_reset_support_clearance_lr
        return {
            'last_reset_closure_before_m': scalar(self._last_reset_closure_max),
            'last_reset_q_closure_after_physics_m': scalar(self._last_reset_q_closure_max),
            'last_reset_body_pose_closure_after_physics_m': scalar(self._last_reset_pose_closure_max),
            'last_reset_closure_after_physics_m': scalar(self._last_reset_native_closure_max),
            'reset_closure_before_m': scalar(self._last_reset_closure_max),
            'reset_q_closure_after_physics_m': scalar(self._last_reset_q_closure_max),
            'reset_body_pose_closure_after_physics_m': scalar(self._last_reset_pose_closure_max),
            'reset_closure_after_physics_m': scalar(self._last_reset_native_closure_max),
            'reset_support_clearance_m': scalar(self._last_reset_support_clearance),
            'last_reset_support_clearance_left_m': side_min(support_lr, 0),
            'last_reset_support_clearance_right_m': side_min(support_lr, 1),
            'last_reset_support_rows': int(self._last_reset_support_rows),
            'last_reset_both_clearance_nonnegative_rows': int(
                self._last_reset_support_bilateral_rows),
            'reset_support_window_left_min_m': scalar(self._reset_support_left_min),
            'reset_support_window_right_min_m': scalar(self._reset_support_right_min),
            'reset_support_window_rows': int(self._reset_support_rows),
            'reset_support_window_both_clearance_nonnegative_rows': int(
                self._reset_support_bilateral_rows),
            'reset_scope': 'last_reset_* = most recent reset call; reset_support_window_* = all resets since collection start',
            'spring_mount_distance_min_m': scalar(self._spring_mount_min),
            'spring_mount_distance_max_m': scalar(self._spring_mount_max),
            'spring_mount_compression_min_m': scalar(self._spring_compression_min),
            'spring_mount_compression_max_m': scalar(self._spring_compression_max),
            'spring_compression_min_m': scalar(self._spring_compression_min),
            'spring_compression_max_m': scalar(self._spring_compression_max),
            'spring_slider_compression_min_m': scalar(self._spring_slider_compression_min),
            'spring_slider_compression_max_m': scalar(self._spring_slider_compression_max),
            'spring_compression_mismatch_max_m': scalar(self._spring_compression_mismatch_max),
            'spring_mount_compression_last_left_m': side_last(self._spring_mount_compression_last, 0),
            'spring_mount_compression_last_right_m': side_last(self._spring_mount_compression_last, 1),
            'spring_slider_compression_last_left_m': side_last(self._spring_slider_compression_last, 0),
            'spring_slider_compression_last_right_m': side_last(self._spring_slider_compression_last, 1),
            'spring_mount_stroke_valid_last': bool(self._spring_mount_stroke_last.all()),
            'spring_slider_stroke_valid_last': bool(self._spring_slider_stroke_last.all()),
            'spring_mount_fit_valid_last': bool(self._spring_mount_fit_last.all()),
            'spring_slider_fit_valid_last': bool(self._spring_slider_fit_last.all()),
            'spring_mount_valid_last': bool(self._spring_mount_valid_last.all()),
            'spring_slider_valid_last': bool(self._spring_slider_valid_last.all()),
            'spring_telemetry_samples': int(self._spring_telemetry_samples),
            'spring_mount_invalid_samples': int(self._spring_mount_invalid_samples),
            'spring_slider_invalid_samples': int(self._spring_slider_invalid_samples),
            'spring_mount_stroke_invalid_samples': int(self._spring_mount_stroke_invalid_samples),
            'spring_slider_stroke_invalid_samples': int(self._spring_slider_stroke_invalid_samples),
            'spring_mount_fit_invalid_samples': int(self._spring_mount_fit_invalid_samples),
            'spring_slider_fit_invalid_samples': int(self._spring_slider_fit_invalid_samples),
            'spring_pressure_force_max_n': scalar(self._spring_force_max),
            'spring_telemetry_scope': 'post-physics substep state; mount compression drives the wrenches, slider compression is the native guide coordinate',
            'spring_source_sha256': dict(self._spring_source_sha256),
            'reset_readback_scope': 'forward(dt=0), before the next physical substep',
            **self._loop_telemetry.summary(),
        }

    def _build_spring_force(self):
        """Use the authenticated v7 main-link gas-spring mount path."""
        if (len(self._binding.native_body_names) != 19
                or len(self._binding.tree_joint_names) != 18
                or len(self._asset.manifest.get('spring_joint_names', [])) != 2):
            return None
        model_spec = json.loads(self._asset.bound_bytes('model_spec.json', maximum_bytes=256 * 1024))
        fit = json.loads(self._asset.bound_bytes('fit_10mpa.json', maximum_bytes=64 * 1024))
        element = SpringForceElement(model_spec, self._asset.manifest, fit, self.robot.body_names,
                                     self._device, dtype=torch.float32)
        # Keep the source byte identities beside the applied force path so a
        # native report can distinguish this mechanism from a later asset.
        self._spring_source_sha256 = {
            'model_spec.json': self._asset.manifest['files_sha256']['model_spec.json'],
            'fit_10mpa.json': self._asset.manifest['files_sha256']['fit_10mpa.json'],
        }
        return element

    def _build_pressure(self):
        model = json.loads(self._asset.bound_bytes('model_spec.json', maximum_bytes=256 * 1024))
        fit = json.loads(self._asset.bound_bytes('fit_10mpa.json', maximum_bytes=64 * 1024))
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

    def _wheel_position_base(self):
        """Wheel link origins in the base_link frame for the leg-fork reward."""
        return wheel_positions_in_base(self._as_torch(self.robot.data.body_link_pose_w),
                                       self._base_body_index, self._binding.wheel_body_indices)

    def _wheel_velocity_body(self):
        # Relative COM velocities expressed in the base frame. This is neither
        # contact-point slip nor the derivative of rotating-frame coordinates.
        body_velocity = getattr(self.robot.data, "body_lin_vel_w", None)
        if body_velocity is None:
            raise RuntimeError("182D critic 需要 body_lin_vel_w 产生轮体速度")
        velocity = self._as_torch(body_velocity)
        if velocity.ndim != 3 or velocity.shape[:2] != (self.num_envs, len(self._binding.native_body_names)) or velocity.shape[-1] != 3:
            raise ValueError("body_lin_vel_w 布局不一致")
        root_twist = self._as_torch(self.robot.data.root_com_vel_w)
        if root_twist.shape != (self.num_envs, 6):
            raise ValueError("root_com_vel_w 布局不一致")
        wheels = (velocity.index_select(1, self._wheel_body_indices)
                  - root_twist[:, None, :3]).reshape(-1, 3)
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
        com_offset = self._startup_dynamics.base_com_delta_local_m
        sensor_packet = self._sensor_packet(state)
        observation_delay = ((state.timestamp_s[:, None] - sensor_packet.sampled_at_s) / self.physics_dt).float()
        action_delay = (torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self._device)
                        if self._action_transport is None else
                        self._action_transport.critic_steps() * (self.step_dt / self.physics_dt))
        communication = getattr(self, '_communication', None)
        if communication is not None:
            action_delay = communication.action_age_steps(self.physics_dt)
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
            action_delay_steps=action_delay,
            wheel_velocity_body_m_s=self._wheel_velocity_body(),
            wheel_contact=(state.wheel_ground_force_n > 2.0).to(torch.float32),
            mass_scale=self._startup_dynamics.mass_scale,
            wheel_material=wheel_material,
            terrain_scan=terrain,
            previous_actions=self._critic_action_history.older_actions(),
            joint_acceleration_rad_s2=joint_acceleration,
            com_offset_body_m=com_offset,
            # Encoder zero offset of each output axis (privileged), plus any
            # controller/observation nominal mismatch.
            nominal_pose_delta_rad=((self._control.controller.nominal - self._observations.assembler._nominal
                                     )[None].expand(self.num_envs, -1)
                                    + (0 if self._episode_randomization is None
                                       else self._episode_randomization.encoder_bias_rad)),
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

    def _joint_positions(self):
        return self._joint_coordinates.read(
            self._as_torch(self.robot.data.joint_pos),
            self._as_torch(self.robot.data.joint_vel), self._physical_tick)

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
            self._joint_positions(),
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
            previous_control_source=self._previous_control_source,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_width = self._issued_action.shape[-1]
        if (not isinstance(actions, torch.Tensor)
                or actions.shape != (self.num_envs, action_width)
                or actions.device != self._device or not torch.isfinite(actions).all()):
            raise ValueError(f"actions 需要有限 [N,{action_width}] 张量")
        self._apply_push()
        self._before_action.copy_(self._previous_action)
        self._issued_action.copy_(actions.clamp(-self._action_bounds, self._action_bounds))
        self._held_targets = self._targets
        self._held_generated_at_s = (self._policy_generated_at_s.clone()
                                     if isinstance(getattr(self, '_policy_generated_at_s', None), torch.Tensor)
                                     else self._clock().clone())
        q = self._encoder_position(self._joint_positions().index_select(1, self._active_indices))
        communication = getattr(self, '_communication', None)
        if communication is not None:
            q, _, _ = communication.feedback(self._clock())
        transport_owned = torch.ones(self.num_envs, dtype=torch.bool, device=self._device)
        if self._recover_task is not None:
            transport_owned &= ~self._recover_task.hold_mask
        if self._recovery_preparation is not None:
            transport_owned &= ~self._recovery_preparation.active
        applied_action = (self._issued_action if self._action_transport is None
                          else self._action_transport.step(
                              self._issued_action, policy_allowed=transport_owned, issued_at_s=self._clock()))
        self._policy_generated_at_s = (self._clock() if self._action_transport is None
                                       else self._action_transport.arrived_generated_at_s.clone())
        self._targets = self._control.decode(applied_action, q, self._clock())
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
            if self._recovery_preparation is not None:
                self._targets = self._recovery_preparation.targets(self._targets, self._clock())
                script = self._targets.control_source != int(ControlSource.POLICY)
                # Actor history and action-rate costs describe issuance, not
                # the older packet arriving at the controller this interval.
                self._issued_action.copy_(torch.where(
                    script[:, None], self._targets.issued_action, self._issued_action))
            self._interval_policy_owned = self._targets.control_source == int(ControlSource.POLICY)
            self._formal_reward.composer.begin(
                self._issued_action, dq, now_s=self._clock(), episode_id=self._episode_id,
                control_epoch=self._control_epoch, admitted_body_command=self._command,
                command_mode=self._command_mode, policy_owned=self._interval_policy_owned,
                control_source=self._targets.control_source,
            )
        if communication is not None:
            communication.begin_interval(transport_owned &
                (self._targets.control_source == int(ControlSource.POLICY)), self._clock())
        if self._action_latch is not None:
            # Lag only between two consecutive policy-owned targets; resets,
            # scripts and handoffs latch their first target immediately.
            owned = self._targets.control_source == int(ControlSource.POLICY)
            self._latch_active = self._action_latch.held & owned
            self._action_latch.held.copy_(owned)
        self._substep = 0

    def _reset_targets(self, mask, q):
        """Neutral targets for reset rows only; other rows keep their latched targets."""
        neutral = self._control.decode(
            torch.zeros(self.num_envs, ACTION_DIM, dtype=torch.float32, device=self._device),
            q, self._clock())
        self._targets = neutral if self._targets is None else _select_targets(mask, neutral, self._targets)

    def _apply_action(self) -> None:
        if self._targets is None:
            raise RuntimeError("必须先由 _pre_physics_step 生成输出目标")
        targets, generated_at = self._targets, self._policy_generated_at_s
        if self._action_latch is not None and self._held_targets is not None:
            lagging = self._latch_active & (self._substep < self._action_latch.lags)
            targets = _select_targets(lagging, self._held_targets, self._targets)
            generated_at = torch.where(lagging, self._held_generated_at_s, self._policy_generated_at_s)
        self._substep += 1
        q = self._encoder_position(self._joint_positions().index_select(1, self._active_indices))
        dq = self._as_torch(self.robot.data.joint_vel).index_select(1, self._active_indices)
        shaft_velocity = dq
        now = self._clock()
        communication = getattr(self, '_communication', None)
        if communication is not None:
            q, dq, feedback_sample_s = communication.feedback(now)
        randomization = self._episode_randomization
        feedback = self._control.controller.effort(
            targets, q, dq, now, strength=None if randomization is None else randomization.motor_strength,
            envelope=self._motor_envelope, motor_velocity_rad_s=shaft_velocity,
            no_load_scale=None if randomization is None else randomization.no_load_scale)
        requested = feedback.capped_effort
        preparation = getattr(self, '_recovery_preparation', None)
        if preparation is not None:
            requested = preparation.release_effort(requested)
        if communication is not None:
            generated = torch.where(targets.control_source == int(ControlSource.POLICY), generated_at, now)
            effort = communication.deliver(requested, feedback.effort_bounds,
                now, generated, feedback_sample_s, response=self._control.drive,
                target_arrived_at_s=targets.controller_issued_at_s)
        elif self._control.drive is not None:
            effort = self._control.drive.advance(
                requested, self._physics_dt_tensor, feedback.effort_bounds,
            ).mean_effort
        else:
            effort = requested
        # Gas pressure is a main-link mount force/torque, not a second
        # prismatic slider effort. Slider damping remains in the actuator cfg.
        if self._spring_force is not None:
            frame = self._spring_force.evaluate(
                self._as_torch(self.robot.data.body_link_pose_w),
                self._as_torch(self.robot.data.body_com_pose_w)[..., :3],
                self._spring_strength,
            )
            require(self._invariants, 'invalid gas-spring mount geometry before physics', ~frame.valid)
            self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=frame.forces_world.contiguous(),
                torques=frame.torques_about_com_world.contiguous(),
                body_ids=list(self._spring_force.body_ids), is_global=True,
            )
        randomization = self._episode_randomization
        if randomization is not None and (randomization.config.external_force_n
                                          or randomization.config.external_torque_nm):
            wrench = randomization.advance_wrench(float(self.physics_dt))
            self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=wrench[:, None, :3].contiguous(), torques=wrench[:, None, 3:].contiguous(),
                body_ids=[self._base_body_index], is_global=False)
        command = torch.zeros(self.num_envs, 18, dtype=torch.float32, device=self._device)
        command.index_copy_(1, self._active_indices, effort)
        self.robot.set_joint_effort_target(command)
        self._last_effort.copy_(effort)

    def _observe_spring_state(self):
        """Measure both spring coordinates from the *returned* physical state.

        The mount wrenches were already applied once in ``_apply_action`` from
        the pre-substep geometry. This method only evaluates the same pure
        element again for telemetry; it never adds a second wrench.
        """
        if self._spring_force is None:
            return None, None
        mount = self._spring_force.evaluate(
            self._as_torch(self.robot.data.body_link_pose_w),
            self._as_torch(self.robot.data.body_com_pose_w)[..., :3],
            self._spring_strength,
        )
        slider = self._pressure.evaluate(
            self._as_torch(self.robot.data.joint_pos).index_select(1, self._spring_indices))
        mount_valid = mount.valid
        mount_valid_lr = mount_valid[:, None]
        slider_compression = slider.compression_m.to(torch.float32)
        slider_valid = torch.isfinite(slider_compression)
        finite_mount = torch.isfinite(mount.compression_m) & mount_valid_lr
        finite_slider = slider_valid
        self._spring_telemetry_samples.add_(1)
        self._spring_mount_invalid_samples.add_((~mount_valid).sum())
        self._spring_slider_invalid_samples.add_((~slider_valid).sum())
        self._spring_mount_stroke_invalid_samples.add_(
            (~mount.within_mechanical_stroke).sum())
        self._spring_slider_stroke_invalid_samples.add_(
            (~slider.within_mechanical_stroke).sum())
        self._spring_mount_fit_invalid_samples.add_(
            (~mount.within_fitted_domain).sum())
        self._spring_slider_fit_invalid_samples.add_(
            (~slider.within_fitted_curve).sum())
        self._spring_mount_min = torch.minimum(
            self._spring_mount_min,
            torch.where(finite_mount, mount.mount_distance_m, float('inf')).amin())
        self._spring_mount_max = torch.maximum(
            self._spring_mount_max,
            torch.where(finite_mount, mount.mount_distance_m, float('-inf')).amax())
        self._spring_compression_min = torch.minimum(
            self._spring_compression_min,
            torch.where(finite_mount, mount.compression_m, float('inf')).amin())
        self._spring_compression_max = torch.maximum(
            self._spring_compression_max,
            torch.where(finite_mount, mount.compression_m, float('-inf')).amax())
        self._spring_slider_compression_min = torch.minimum(
            self._spring_slider_compression_min,
            torch.where(finite_slider, slider_compression, float('inf')).amin())
        self._spring_slider_compression_max = torch.maximum(
            self._spring_slider_compression_max,
            torch.where(finite_slider, slider_compression, float('-inf')).amax())
        self._spring_force_max = torch.maximum(
            self._spring_force_max,
            torch.where(mount_valid_lr, mount.pressure_force_n.abs(), 0.).amax())
        mismatch = (mount.compression_m-slider_compression).abs()
        mismatch = torch.where(finite_mount & finite_slider, mismatch, 0.)
        self._spring_compression_mismatch_max = torch.maximum(
            self._spring_compression_mismatch_max, mismatch.amax())
        self._spring_mount_compression_last.copy_(mount.compression_m)
        self._spring_slider_compression_last.copy_(slider_compression)
        self._spring_mount_stroke_last.copy_(mount.within_mechanical_stroke)
        self._spring_slider_stroke_last.copy_(slider.within_mechanical_stroke)
        self._spring_mount_fit_last.copy_(mount.within_fitted_domain)
        self._spring_slider_fit_last.copy_(slider.within_fitted_curve)
        self._spring_mount_valid_last.copy_(mount_valid_lr.expand(-1, 2))
        self._spring_slider_valid_last.copy_(slider_valid)
        return mount, slider

    def _observe_substep(self) -> None:
        if self._loop_telemetry.enabled:
            self._loop_telemetry.observe_physics(
                self._as_torch(self.robot.data.body_link_pose_w),
                self._as_torch(self.robot.data.joint_vel))
        sample = self._sample()
        state = self._observations.assembler.measure(sample)
        encoder = self._encoder_state(state)
        if self._communication is not None:
            self._communication.push(encoder.policy_position_rad, encoder.policy_velocity_rad_s)
        if self._sensor_model is not None:
            self._sensor_model.push(encoder, self._physical_tick, heading_rad=_yaw(sample.root_link_orientation_xyzw))
        boundary = None
        mount, pressure = self._observe_spring_state()
        if self._failure_latch is not None:
            boundary = self._boundary_geometry.evaluate(
                self._as_torch(self.robot.data.body_link_pose_w),
                self._as_torch(self.robot.data.joint_pos), self._targets, knee_margin_rad=.05)
            if pressure is None:
                pressure = self._pressure.evaluate(
                    self._as_torch(self.robot.data.joint_pos).index_select(1, self._spring_indices))
            allowed = (None if self._terrain_course is None
                       else self._terrain_course.body_contact_allowed)
            # Mount compression is the actual pressure path. Keep the native
            # slider stop as an independent gate: either coordinate leaving
            # its authored travel invalidates the physical spring state.
            within_stroke = pressure.within_mechanical_stroke
            if mount is not None:
                within_stroke = within_stroke & mount.within_mechanical_stroke
            conditions = self._failure_latch.observe(
                height_m=state.height_above_ground_m, gravity_body=state.gravity_body,
                forbidden_body_load_n=(sample.body_normal_load_n[:, self._forbidden_bodies]
                    if isinstance(sample, LabPhysicalSample) else sample.terrain_contact_force_world_n[
                        :, self._forbidden_bodies].norm(dim=-1)),
                knee_signed_clearance_rad=boundary.knee_signed_clearance_rad,
                knee_pivot_residual_m=boundary.knee_pivot_residual_m,
                within_pressure_stroke=within_stroke,
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
                    policy_owned=self._interval_policy_owned,
                    jump_height_world_z_m=torch.where(jump, jump_height_world, zero),
                    jump_velocity_world_z_m_s=zero,
                    joint_boundary_shortfall_rad=boundary,
                    wheel_position_base_m=self._wheel_position_base(),
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
        invariants = getattr(self, '_invariants', None)
        if invariants is not None:
            invariants.raise_if_violated()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        birth = getattr(self, '_birth_randomization', None)
        if birth is not None and self._last_state is not None:
            birth.observe(
                float(self.step_dt), height_m=self._last_state.height_above_ground_m,
                height_command_m=self._command[:, 3], tilt_rad=self._last_tilt,
                wheel_load_n=self._last_state.wheel_ground_force_n, terminated=self._terminated,
                failure_reasons=(None if getattr(self, '_failure_latch', None) is None
                                 else self._failure_latch.failure_reasons))
        preparation = getattr(self, '_recovery_preparation', None)
        if preparation is not None:
            q = self._joint_positions().index_select(1, self._active_indices)
            handoff, preparation_timeout = preparation.finish_interval(q)
            self._terminated |= preparation_timeout
            self._preparation_handoff.copy_(handoff & ~self._terminated)
            self.extras['recovery_preparation_timeouts'] = preparation_timeout.sum()
        time_limit = self.episode_length_buf >= self.max_episode_length - 1
        query_invalid = torch.zeros_like(time_limit)
        if self._terrain_course is not None:
            query_invalid = self._terrain_course.finish_queries().clone()
            self._terminated |= query_invalid
            recovery_timeout = self._terrain_course.finish_recovery_interval()
            time_limit |= recovery_timeout
            self.extras['recovery_timeouts'] = recovery_timeout.sum()
        spatial_exit = torch.zeros_like(time_limit)
        if self._terrain_course is not None:
            spatial_exit = self._terrain_course.router.observe_end(self._root_link_position(), self._terminated)
        truncated = time_limit | spatial_exit
        # Keep causes separate even when both share bootstrap/trace masks.
        # The collector transfers these small counters once per rollout.
        self.extras['termination_counts'] = torch.stack((
            (self._terminated & ~query_invalid).sum(), (time_limit & ~self._terminated).sum(),
            (spatial_exit & ~self._terminated).sum(),
            (time_limit & spatial_exit & ~self._terminated).sum()))
        self.extras['interval_observation_valid'] = ~query_invalid
        self.extras['query_rejected_intervals'] = query_invalid.sum()
        if self._formal_reward is not None:
            self._formal_result = self._formal_reward.finish(
                terminated=self._terminated, truncated=truncated & ~self._terminated,
                physical_success_request_id=(self._terrain_course.tasks.success_id.masked_fill(query_invalid, -1)
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
        # Missing terrain is not an invented physical reward/terminal value.
        # Its finite placeholders are only for vectorized row bookkeeping.
        self._last_reward.masked_fill_(query_invalid, 0.)
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
        self._critic_action_history.record(
            self._issued_action, policy_owned=self._targets.control_source == int(ControlSource.POLICY))
        previous_source = getattr(self, '_previous_control_source', None)
        if previous_source is not None:
            previous_source.copy_(self._targets.control_source)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        self._update_skill()
        # Reward used this interval's command. Bootstrap describes the next
        # decision endpoint, including its newly advanced command, before any
        # physical row reset changes the terminal state or sensor packet.
        self._capture_terminal()
        if preparation is not None:
            self._preparation_handoff &= ~self.reset_buf
            self._control_epoch.add_(self._preparation_handoff.long())
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
        if self._recovery_reset is not None and task_sample is not None:
            # Ordinary birth is the selected asset's nominal closed chain, not
            # an SDK default that may be reordered or changed by an exporter.
            tree_order = [self._binding.native_joint_names.index(name)
                          for name in self._binding.tree_joint_names]
            nominal_native = self._recovery_reset.kinematics.nominal_tree_position.new_zeros(18)
            nominal_native[tree_order] = self._recovery_reset.kinematics.nominal_tree_position
            joint_position.copy_(nominal_native.to(self._device).float().expand(count, -1))
            joint_velocity.zero_()
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
                # LabRecoveryReset returns the asset's manifest-native joint
                # order. The Articulation tensor follows the SDK body/joint
                # order, so map by name instead of assuming both exporters kept
                # the same ordering.
                asset_joint_order = [self._recovery_reset.kinematics.binding.native_joint_names.index(name)
                                     for name in self._binding.native_joint_names]
                joint_position[local] = pose.joint_position.index_select(
                    1, torch.tensor(asset_joint_order, device=self._device))
                root_velocity[local] = pose.root_com_velocity
                joint_velocity[local] = pose.joint_velocity
        ground = self._randomize_births(ids, task_sample, root_pose, joint_position, joint_velocity)
        if self._terrain_course is not None:
            self._record_reset_support(ids)
        if self._recovery_reset is not None:
            # Validate the exact q18 written to physics against the selected
            # closed-chain source. Nominal/recovery solver output must already
            # satisfy this; a bad branch must not enter the native solver.
            tree_order = [self._binding.native_joint_names.index(name)
                          for name in self._binding.tree_joint_names]
            q_tree = joint_position.index_select(1, torch.tensor(tree_order, device=self._device)).double().cpu()
            closure = self._recovery_reset.kinematics.closure(q_tree).norm(dim=-1).amax(-1)
            if not torch.isfinite(closure).all() or (closure > 2e-6).any():
                raise RuntimeError('reset q18 violates the selected closed-chain residual bound')
            self._last_reset_closure_max = closure.max()
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        self._joint_coordinates.reset_rows(ids, joint_position)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        if self._loop_telemetry.enabled:
            self._loop_telemetry.observe_reset(
                self._as_torch(self.robot.data.body_link_pose_w), ids)
        if self._recovery_reset is not None:
            # Readback admission after the native write. The source residual
            # above is a solver claim; this is what PhysX actually accepted.
            native_q = self._as_torch(self.robot.data.joint_pos).index_select(0, ids)
            native_q = native_q.index_select(1, torch.tensor(tree_order, device=self._device))
            native_closure = self._recovery_reset.kinematics.closure(
                native_q.double().cpu()).norm(dim=-1).amax(-1)
            if not torch.isfinite(native_closure).all():
                raise RuntimeError('native reset readback contains nonfinite closed-chain state')
            native_poses = self._as_torch(self.robot.data.body_link_pose_w).index_select(0, ids)
            asset_body_order = [
                self._binding.native_body_names.index(name)
                for name in self._recovery_reset.kinematics.binding.native_body_names
            ]
            native_poses_asset = native_poses.index_select(
                1, torch.tensor(asset_body_order, device=self._device))
            native_pose_closure = self._recovery_reset.kinematics.closure_from_poses(
                native_poses_asset).norm(dim=-1).amax(-1)
            self._last_reset_q_closure_max = native_closure.max()
            self._last_reset_pose_closure_max = native_pose_closure.cpu().max()
            self._last_reset_native_closure_max = torch.maximum(
                native_closure, native_pose_closure.cpu()).max()
            if not torch.isfinite(native_pose_closure).all():
                raise RuntimeError('native body-pose readback contains nonfinite closure')
            if bool((native_pose_closure > 0.01).any()):
                bad = (native_pose_closure > 0.01).nonzero().flatten()
                raise RuntimeError(
                    'native body-pose readback violates the 10mm closed-chain evidence bound: '
                    f'max={float(native_pose_closure.max()):.6g} rows={bad[:8].tolist()} '
                    f'per_constraint_max={native_pose_closure.max(0).values.tolist()}')
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
        self._last_joint_acceleration[ids] = 0.0
        self._last_policy_velocity[ids] = 0.0
        self._issued_action[ids] = 0.0
        self._previous_action[ids] = 0.0
        self._previous_control_source[ids] = int(ControlSource.POLICY)
        self._before_action[ids] = 0.0
        self._reset_mask[ids] = True
        if self._terrain_course is not None:
            self._terrain_course.reset_commands(ids, task_sample)
            if ground is not None and ground.any():
                # SCUT ground reset: vx, yaw in U(-1, 1) for 1.5 s, then the pool.
                rows = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
                rows[ids[ground]] = True
                self._terrain_course.tasks.begin_tracking(rows)
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
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        mask[ids] = True
        difficulty = 1.
        if self._episode_randomization is not None:
            sampler = getattr(self, '_task_sampler', None)
            update = 0 if sampler is None else sampler.curriculum.budget.total_updates
            difficulty = self._episode_randomization.reset(mask, update)
            if self._control.drive is not None:
                self._control.drive.set_time_constant_scale(mask, self._episode_randomization.time_constant_scale)
        # Controller and preparation work in encoder coordinates.
        q = self._encoder_position(self._joint_positions().index_select(1, self._active_indices))
        self._reset_targets(mask, q)
        if self._recovery_preparation is not None:
            self._recovery_preparation.reset_rows(
                ids, q, self._base_command_mode == int(CommandMode.RECOVERY), self._clock())
            if getattr(self, '_pending_ground_rows', None) is not None and self._pending_ground_rows.numel():
                self._recovery_preparation.release_rows(
                    self._pending_ground_rows, self._birth_randomization.release_frames(float(self.step_dt)))
            self._pending_ground_rows = None
        if self._communication is not None:
            dq = self._as_torch(self.robot.data.joint_vel).index_select(1, self._active_indices)
            self._communication.reset(mask, q, dq, self._clock(), difficulty=difficulty)
        if self._sensor_model is not None:
            self._sensor_model.reset(mask, self._clock(), difficulty=difficulty)
            current = self._observations.assembler.measure(self._sample())
            self._sensor_model.seed_after_reset(mask, self._encoder_state(current),
                                                heading_rad=_yaw(self._root_link_quat()))
        if self._action_transport is not None:
            self._action_transport.reset(mask, difficulty=difficulty)
        if self._action_latch is not None:
            self._action_latch.reset(mask, difficulty=difficulty)
            self._latch_active[mask] = False

    def control_timing_state_dict(self):
        if self._communication is None and self._episode_randomization is None and self._action_latch is None:
            return None
        state = {'format': 'lab_control_timing_v1',
                 'communication': (None if self._communication is None else self._communication.state_dict()),
                 'episode_randomization': (None if self._episode_randomization is None
                     else self._episode_randomization.state_dict())}
        if self._action_latch is not None:
            state['action_latch'] = self._action_latch.state_dict()
        if self._birth_randomization is not None:
            state['birth_randomization'] = self._birth_randomization.state_dict()
        return state

    def restore_control_timing(self, state):
        if self._physical_tick != 0:
            raise ValueError('control timing restores before new physical episodes')
        expected = self.control_timing_state_dict()
        if expected is None:
            if state is not None:
                raise ValueError('control timing differs from the frozen runtime')
            return
        if (not isinstance(state, dict) or set(state) != set(expected)
                or state['format'] != expected['format']
                or any((state[key] is None) != (expected[key] is None)
                       for key in ('communication', 'episode_randomization'))):
            raise ValueError('control timing requires its complete cold-reset state')
        if self._communication is not None:
            self._communication.restore_after_physical_reset(state['communication'])
        if self._episode_randomization is not None:
            self._episode_randomization.restore_after_physical_reset(state['episode_randomization'])
        if self._action_latch is not None:
            self._action_latch.restore_after_physical_reset(state['action_latch'])
        if self._birth_randomization is not None:
            self._birth_randomization.restore_after_physical_reset(state['birth_randomization'])

    def _randomize_births(self, ids, task_sample, root_pose, joint_position, joint_velocity):
        """SCUT births inside this asset's travel: ground or landing posture on every ordinary row.

        Writes the reset tensors in place and returns the [M] ground-birth mask
        (None without birth randomization). Recovery rows keep their posture.
        """
        birth = self._birth_randomization
        self._pending_ground_rows = None
        if birth is None or task_sample is None:
            return None
        ordinary = task_sample.mode != int(CommandMode.RECOVERY)
        ground, index, tilt, wheels = birth.draw(ordinary, birth.ground_eligible(task_sample.terrain_kind))
        landing = ordinary & ~ground
        # Close the previous episodes before tracking the new births.
        birth.settle(ids)
        birth.begin(ids[ground], 'ground')
        birth.begin(ids[landing], 'landing')
        # The course placed the nominal stance on its local ground; keep that datum.
        spawn = (self._terrain_course.nominal_spawn_height_m if self._terrain_course is not None
                 else self._task.nominal_height_m)
        datum = root_pose[:, 2] - spawn
        root_pose[ground, 2] = datum[ground] + birth.ground_root_height_m[index[ground]]
        root_pose[landing, 2] = datum[landing] + birth.landing_root_height_m + birth.config.tilt_clearance_m
        root_pose[landing, 3:7] = _quat_mul(root_pose[landing, 3:7], birth.tilt_quaternion(tilt[landing]))
        rows = ordinary.nonzero(as_tuple=False).flatten()
        order = [self._recovery_reset.kinematics.binding.native_joint_names.index(name)
                 for name in self._binding.native_joint_names]
        joint_position[rows] = birth.joint_position[index[rows]].index_select(
            1, torch.tensor(order, device=self._device))
        if self._terrain_course is not None and rows.numel():
            lifted = root_pose[rows]
            self._terrain_course.raise_to_support(ids[rows], lifted, birth.wheel_centre_root_m[index[rows]])
            root_pose[rows] = lifted
        wheel_columns = [self._binding.native_joint_names.index(name) for name in ('L_joint3', 'R_joint3')]
        joint_velocity[:, wheel_columns] = torch.where(ordinary[:, None], wheels, joint_velocity[:, wheel_columns])
        self._pending_ground_rows = ids[ground]
        return ground

    def _record_reset_support(self, ids):
        """Wheel-bottom support gaps of the reset rows as finally written."""
        support_lr = self._terrain_course._last_support_clearance_lr.index_select(0, ids)
        self._last_reset_support_clearance_lr = support_lr
        finite_rows = torch.isfinite(support_lr).all(-1)
        last_ordinary = int(finite_rows.sum())
        last_bilateral = int((finite_rows & (support_lr[:, 0] >= 0) & (support_lr[:, 1] >= 0)).sum())
        self._last_reset_support_rows = last_ordinary
        self._last_reset_support_bilateral_rows = last_bilateral
        support = support_lr[finite_rows]
        self._last_reset_support_clearance = support.amin() if support.numel() else None
        self._reset_support_rows += last_ordinary
        self._reset_support_bilateral_rows += last_bilateral
        for side, name in ((0, '_reset_support_left_min'), (1, '_reset_support_right_min')):
            values = support_lr[:, side]
            values = values[torch.isfinite(values)]
            if values.numel():
                value = values.amin()
                previous = getattr(self, name)
                setattr(self, name, value if previous is None else torch.minimum(previous, value))

    def _encoder_position(self, q):
        """True output positions to encoder coordinates (randomized zero bias)."""
        if self._episode_randomization is None:
            return q
        return q + self._episode_randomization.encoder_bias_rad

    def _encoder_state(self, state):
        if self._episode_randomization is None:
            return state
        return replace(state, policy_position_rad=self._encoder_position(state.policy_position_rad))

    def _apply_push(self):
        """World-frame velocity steps on ordinary policy-owned rows (SCUT push events)."""
        randomization = self._episode_randomization
        if randomization is None or not (randomization.config.push_linear_m_s or randomization.config.push_yaw_rad_s):
            return
        eligible = ((self._command_mode == int(CommandMode.NORMAL)) | (self._command_mode == int(CommandMode.SLOPE))
                    | (self._command_mode == int(CommandMode.STAIR)))
        if self._recovery_preparation is not None:
            eligible &= ~self._recovery_preparation.active
        rows, delta = randomization.push(self.step_dt, eligible)
        if not rows.numel():
            return
        velocity = self._as_torch(self.robot.data.root_com_vel_w).index_select(0, rows).clone()
        velocity[:, :2] += delta[:, :2]
        velocity[:, 5] += delta[:, 2]
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=rows)

    def episode_diagnostics(self):
        """Birth randomization and physical-failure counters owned by the environment."""
        birth, latch = self._birth_randomization, self._failure_latch
        return {'births': None if birth is None else birth.metrics(),
                'terminations': None if latch is None else latch.metrics()}

    def control_timing_metrics(self):
        values = {} if self._communication is None else self._communication.metrics()
        if self._sensor_model is not None:
            values['sensor_noise_scale'] = self._sensor_model.noise_scale
            values['sensor_extra_delay_s'] = self._sensor_model.lags.float() * self.physics_dt
        if self._action_transport is not None:
            values['policy_target_extra_delay_s'] = self._action_transport.lags.float() * self.step_dt
        if self._episode_randomization is not None:
            values['episode_exposure_scale'] = self._episode_randomization.scale
            values['spring_force_scale'] = self._spring_strength
        if self._communication is not None:
            values['native_interval_effort_abs_nm'] = self._last_effort.abs()
            if self._control.drive is not None:
                values['motor_endpoint_effort_abs_nm'] = self._control.drive.state.abs()
        if not values:
            return {}
        keys = tuple(values)
        summaries = torch.stack([torch.stack((values[key].amin(), values[key].mean(), values[key].amax()))
                                 for key in keys]).cpu().tolist()
        return {'scope': 'current_episode_configuration_and_last_physical_interval',
                'pooled_rollout_statistics': False,
                'values': {key: dict(zip(('min', 'mean', 'max'), value))
                           for key, value in zip(keys, summaries)}}

    def heading_estimate(self):
        """Deployable heading for the command outer loop (delayed like the IMU)."""
        if self._sensor_model is None:
            return _yaw(self._root_link_quat())
        return self._sensor_model.delayed_heading()

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
        boundary = self._reset_mask | self._preparation_handoff
        self._critic_action_history.clear(boundary)
        actor, _ = self._observations.append(
            sample, commands, sensors, boundary, measured=state,
        )
        if self._scan_debug:
            print("OBS after append", flush=True)
        reset = self._reset_mask
        acceleration = torch.where(reset[:, None], torch.zeros_like(self._last_joint_acceleration),
                                  self._last_joint_acceleration)
        self._last_joint_acceleration.copy_(acceleration)
        self._last_policy_velocity.copy_(state.policy_velocity_rad_s)
        critic = self._critic_182(state, sample, joint_acceleration=acceleration)
        self._reset_mask.zero_()
        self._preparation_handoff.zero_()
        self.obs_buf = {"policy": actor.flatten(1).contiguous(),
                        "critic": critic.contiguous()}
        return self.obs_buf

    def _get_rewards(self):
        return self._last_reward

    def _get_dones(self):
        return self._last_terminated, self._last_truncated

"""Assemble the native task graph inside an externally admitted SDK scene.

This module starts no application, allocates no resource lease and schedules
no evaluation. The launch owner supplies the complete physical recipe and
sensor implementation, then owns reset, collection and learning boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path

import torch

from ..tasks.trajectory import CommandRates, CommandStream, DriveEnvelope
from ..tasks.jump import JumpReferenceConfig, JumpReferenceProgram
from ..control.joints import LegBoundaryGeometry
from ..simulation.episode import NativeEpisodeLifecycle
from ..simulation.ground import NativeGroundQuery, WheelCollisionCentres
from ..simulation.plant import NativePhysicsPlant
from ..simulation.reward import NativeRewardBridge
from ..simulation.task import NativeTaskEnvironment, NativeTaskLimits
from ..simulation.dynamics import StartupDynamicsConfig
from ..simulation.terrain import NativeTerrainEditor, NativeTerrainInstaller, TerrainEditTuning
from ..simulation.terrain_stream import NativeTerrainStream
from ..control.actuation import EffectiveDriveResponse, OutputFeedbackController, PairTargetDomain, ResponseParameters
from ..control.pressure import NativeSliderPressure, PressureParameters
from ..control.sensors import SensorModel, SensorModelConfig
from ..tasks.recovery import RecoveryPoseCompiler, RecoveryTuning
from ..tasks.reset_collision import AssetCollisionGeometry
from .reset_coordinator import ResetBatchCoordinator
from ..tasks.reset_kinematics import AssetResetKinematics
from ..tasks.reset_program import ResetProgramTuning, TaskResetPreparer
from ..tasks.skills import PhysicalJumpEvidence, SkillEvidenceConfig
from ..tasks.commands import TaskCommandProgram


@dataclass(frozen=True)
class NativeRuntimeTuning:
    reset: ResetProgramTuning
    recovery: RecoveryTuning
    limits: NativeTaskLimits
    command_rates: CommandRates
    terrain_edit: TerrainEditTuning
    pair_domain: PairTargetDomain | None
    knee_margin_rad: float = .05
    sensors: SensorModelConfig = field(default_factory=SensorModelConfig)
    dynamics: StartupDynamicsConfig = field(default_factory=StartupDynamicsConfig)

    def __post_init__(self):
        if (any(not isinstance(getattr(self, name), cls) for name, cls in
                (('reset', ResetProgramTuning), ('recovery', RecoveryTuning), ('limits', NativeTaskLimits),
                 ('command_rates', CommandRates), ('terrain_edit', TerrainEditTuning)))
                or self.pair_domain is not None and not isinstance(self.pair_domain, PairTargetDomain)
                or not isinstance(self.sensors, SensorModelConfig)
                or not isinstance(self.dynamics, StartupDynamicsConfig)
                or type(self.knee_margin_rad) not in (int, float) or not math.isfinite(self.knee_margin_rad)
                or self.knee_margin_rad <= 0):
            raise ValueError('native runtime requires parsed physical recipe components and a positive knee margin')


@dataclass
class NativeTaskRuntime:
    environment: NativeTaskEnvironment
    reset_source: ResetBatchCoordinator
    terrain_installer: NativeTerrainInstaller

    @staticmethod
    def _drive_envelope(kinematics, wheels):
        """Use nominal collider centres in the base frame, not a copied track."""
        if abs(wheels.radius_m[0] - wheels.radius_m[1]) > 1e-9:
            raise ValueError('the symmetric wheel-target envelope needs equal selected wheel radii')
        state = kinematics.forward(kinematics.nominal_tree_position[None])
        indices = list(kinematics.binding.wheel_body_indices)
        offsets = state.body_position_m.new_tensor(wheels.local_xyz_m)
        centres = state.body_position_m[0, indices] + torch.einsum('nij,nj->ni',
                    state.body_rotation[0, indices], offsets)
        return DriveEnvelope(wheels.radius_m[0], abs(float(centres[0, 1] - centres[1, 1])))

    @classmethod
    def on_admitted_scene(cls, sim, scene, recipe, components, tuning: NativeRuntimeTuning,
                          *, asset_directory, response: ResponseParameters, design_courses,
                          native_to_tensor, sensor_source, trace_sink=None):
        if (not isinstance(tuning, NativeRuntimeTuning) or not isinstance(response, ResponseParameters)
                or not callable(native_to_tensor) or not callable(sensor_source)
                or components.task_source is not None):
            raise ValueError('native runtime requires one fresh source graph and explicit physical/sensor adapters')
        bridge, asset = scene.observation, Path(asset_directory).resolve(strict=True)
        if (bridge.num_envs != recipe.num_envs or str(bridge.device) != recipe.device
                or components.protocol.num_envs != bridge.num_envs or sim.cfg.dt != .005
                or scene.readback.get('solver_position_iterations') != 12
                or scene.readback.get('solver_velocity_iterations') != 6):
            raise ValueError('native scene differs from the fixed learner batch,device,200Hz or12/6 solver')
        manifest_raw, model_raw = (asset / 'manifest.json').read_bytes(), (asset / 'model_spec.json').read_bytes()
        manifest, model = json.loads(manifest_raw), json.loads(model_raw)
        if (hashlib.sha256(manifest_raw).hexdigest() != recipe.asset_manifest_sha256
                or scene.readback.get('selected_asset_manifest_sha256') != recipe.asset_manifest_sha256
                or hashlib.sha256(model_raw).hexdigest() != scene.readback.get('selected_model_spec_sha256')):
            raise ValueError('native task sources differ from the selected scene and learner recipe')
        kin = AssetResetKinematics(manifest_raw, model_raw, bridge.binding, device=bridge.device)
        collision = AssetCollisionGeometry.from_asset_directory(kin, asset)
        pins = LegBoundaryGeometry.from_asset_sources(manifest_raw, model_raw, bridge.binding)
        wheels = WheelCollisionCentres.from_urdf((asset / manifest['urdf']).read_bytes(), bridge.binding)
        ground = NativeGroundQuery(scene, design_courses, wheels)
        controller = OutputFeedbackController(bridge.binding.nominal_policy_position_rad, recipe.num_envs,
                                               pair_domain=tuning.pair_domain, device=bridge.device)
        drive = EffectiveDriveResponse(response, recipe.num_envs, device=bridge.device)
        pressure = NativeSliderPressure(PressureParameters.from_asset(manifest, model,
            json.loads((asset / 'fit_10mpa.json').read_text())), recipe.num_envs, bridge.device)
        plant = NativePhysicsPlant(sim, scene, controller, drive, pressure, native_to_tensor=native_to_tensor,
                                    ground_query=ground, timing_path='selected_effective_response_no_extra_transport',
                                    sensor_model=sensor_source if isinstance(sensor_source, SensorModel) else None)
        life = NativeEpisodeLifecycle(plant, ground, components.protocol)
        reward = NativeRewardBridge(plant, pins, recipe.collection, knee_margin_rad=tuning.knee_margin_rad,
                                     retired_request_ids=components.protocol.max_seen_id.clone())
        initial = torch.tensor((0., 0., 0., recipe.domain.nominal_height_m),
                               device=bridge.device).repeat(recipe.num_envs, 1)
        stream = CommandStream(initial, cls._drive_envelope(kin, wheels), tuning.command_rates)
        reference = JumpReferenceProgram(replace(JumpReferenceConfig(), gravity_m_s2=reward.gravity_m_s2))
        commands = TaskCommandProgram(components.protocol, stream, reference, domain=recipe.domain)
        skills = PhysicalJumpEvidence(components.protocol, SkillEvidenceConfig(physics_dt_s=plant.dt),
                                       initial_physical_tick=bridge.physical_tick)
        preparer = TaskResetPreparer(components.sampler, components.terrain,
            RecoveryPoseCompiler(kin, tuning.recovery), collision, tuning.reset,
            pose_seed=recipe.training_seed + 104729)
        source = ResetBatchCoordinator(life, preparer, stage_source=lambda: components.curriculum.stage)
        deferred = components.deferred_task_source_state
        if deferred is not None:
            source.load_after_explicit_cold_setup(deferred)
            components.deferred_task_source_state = None
        env = NativeTaskEnvironment(life, reward, skills, commands, tuning.limits, reset_source=source,
            sensor_source=sensor_source, preparation_action=torch.zeros(recipe.num_envs, 6, device=bridge.device),
            trace_sink=trace_sink)
        editor = NativeTerrainEditor.on_admitted_context(sim, scene, tuning.terrain_edit)
        installer = NativeTerrainInstaller(env, source, editor)
        source.bind_native_installer(installer)
        # 4 policy steps at <=5m/s move 0.2m, inside the stream's 0.25m pad.
        env.terrain_stream = NativeTerrainStream(installer, collision, interval_steps=4)
        components.task_source = source
        return cls(env, source, installer)

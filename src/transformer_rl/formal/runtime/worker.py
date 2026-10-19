"""Native SDK lifetime for the private structured learner's runtime factory.

The launch parent supplies fixed-GPU admission and original resource leases.
This factory creates one owned context, checks its selected collision binding
once and then exposes the continuous task. It schedules no evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch

from ..tasks.trajectory import CommandRates
from ..simulation.collision import verify_wheel_cylinders
from ..simulation.dynamics import StartupDynamicsConfig, apply_startup_dynamics
from ..simulation.ground import WheelCollisionCentres
from ..simulation.health import NativePhysicsHealth
from .interop import native_proxy_launch
from .assembly import NativeRuntimeTuning, NativeTaskRuntime
from ..simulation.scene import NativeSceneRecipe
from ..simulation.task import NativeTaskLimits
from ..simulation.terrain import NativeShapeQueries, TerrainEditTuning
from ..control.actuation import PairTargetDomain, ResponseParameters
from ..control.sensors import SensorModel, SensorModelConfig
from ..tasks.recovery import RecoveryTuning
from ..tasks.reset_program import ResetProgramTuning
from ..tasks.geometry import TerrainKind
from ..tasks.terrain import TerrainExtent, TerrainRequest
from ...common.training_control import boot_id, verify_pin, write_json


def parse_tuning(value):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(NativeRuntimeTuning)}:
        raise ValueError('native runtime tuning fields differ')
    parsed = dict(value)
    for name, config_type in (('reset', ResetProgramTuning), ('recovery', RecoveryTuning),
                              ('limits', NativeTaskLimits), ('command_rates', CommandRates),
                              ('terrain_edit', TerrainEditTuning), ('sensors', SensorModelConfig),
                              ('dynamics', StartupDynamicsConfig)):
        raw = dict(parsed[name])
        if name == 'limits':
            raw['forbidden_body_names'] = tuple(raw['forbidden_body_names'])
        parsed[name] = config_type(**raw)
    parsed['pair_domain'] = PairTargetDomain(**parsed['pair_domain']) if parsed['pair_domain'] is not None else None
    return NativeRuntimeTuning(**parsed)


def _initial_policy_boundary(sim, scene):
    """完成一个真实100Hz准备区间，避免在奇数200Hz tick创建生命周期。"""
    bridge = scene.observation
    if sim.cfg.dt != .005 or bridge.physical_tick != 0:
        raise ValueError('initial refresh requires a fresh200Hz scene before its first physical step')
    for _ in range(2):
        sim.step(render=False)
        if scene.physics_health is not None:
            scene.physics_health.check()
        bridge.after_physics_step(bridge.physical_tick + 1)


def _physics_configuration(config_type):
    # 地面已改为近身驻留块；不再为旧的重叠大AABB常驻分配亿级配对缓冲。
    return config_type(enable_external_forces_every_iteration=True, enable_scene_query_support=True,
                       gpu_found_lost_pairs_capacity=2**22, gpu_found_lost_aggregate_pairs_capacity=2**25,
                       gpu_total_aggregate_pairs_capacity=2**22)


def _read_pair_buffers(prim, configuration):
    fields = {'gpuFoundLostPairsCapacity': 'gpu_found_lost_pairs_capacity',
              'gpuFoundLostAggregatePairsCapacity': 'gpu_found_lost_aggregate_pairs_capacity',
              'gpuTotalAggregatePairsCapacity': 'gpu_total_aggregate_pairs_capacity'}
    actual = {name: prim.GetAttribute('physxScene:' + name).Get() for name in fields}
    if actual != {name: getattr(configuration, field) for name, field in fields.items()}:
        raise RuntimeError('native GPU pair-buffer capacity differs from the full-scale configuration')
    return actual


@contextmanager
def runtime(recipe, components, configuration, *, control):
    """Return a single native task after explicit cold physics setup.

    Initial flat geometry is temporary, not an alternate curriculum draw.
    The reset coordinator consumes and applies the real first task request.
    """
    expected = {'asset_directory', 'response_source_pin', 'tuning', 'environment_spacing_m',
                'wheel_probe_tolerance_m', 'authorized_boot'}
    if (not isinstance(configuration, dict) or set(configuration) != expected
            or configuration['authorized_boot'] != boot_id()
            or configuration['authorized_boot'] != control.binding['boot_id']
            or recipe.device != 'cuda:0' or recipe.collection.policy_dt_s != .01
            or type(configuration['environment_spacing_m']) not in (int, float)
            or configuration['environment_spacing_m'] <= 0):
        raise ValueError('native worker requires its exact current-boot recipe and spatial configuration')
    tuning = parse_tuning(configuration['tuning'])
    verify_pin(configuration['response_source_pin'])
    response_document = json.loads(Path(configuration['response_source_pin']['path']).read_text())
    response = ResponseParameters.from_recorded_response(response_document['actuator_response'])
    launcher = None
    compatibility = None
    physics_health = None
    try:
        from isaaclab.app import AppLauncher

        previous_argv = sys.argv
        try:
            sys.argv = [sys.argv[0]]
            launcher = AppLauncher(headless=True, device=recipe.device, enable_cameras=False, livestream=0)
        finally:
            sys.argv = previous_argv
        import omni.kit.app
        import omni.physx
        import carb.logging
        import warp as wp
        from isaaclab.app.settings_manager import SettingsManager
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.warp.proxy_array import ProxyArray
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        settings = SettingsManager.instance()
        settings.initialize_carb_settings()
        settings.set_int('/physics/cudaDevice', 0)
        settings.set_bool('/physics/suppressReadback', True)
        extensions = omni.kit.app.get_app().get_extension_manager()
        if not extensions.is_extension_enabled('omni.physx.gpu'):
            extensions.set_extension_enabled_immediate('omni.physx.gpu', True)
        torch.cuda.set_device(0)
        torch.cuda.init()
        compatibility = native_proxy_launch(wp, ProxyArray)
        statistics = compatibility.__enter__()
        physics_health = NativePhysicsHealth(carb.logging.acquire_logging())
        carb.log_warn('native physics error monitor attached')
        if physics_health.messages_seen == 0:
            raise RuntimeError('native physics logging callback was not invoked by the installed SDK')
        physics_configuration = _physics_configuration(PhysxCfg)
        sim = SimulationContext(SimulationCfg(dt=.005, device=recipe.device, enable_scene_query_support=True,
                                physics=physics_configuration))
        maximum_horizon = max(tuning.reset.ordinary_horizon_s, tuning.reset.extended_horizon_s)
        extent = TerrainExtent.for_horizon(maximum_horizon, max(abs(v) for v in recipe.domain.forward_m_s),
                                           tuning.reset.terrain_margin_m)
        initial = components.terrain.build(TerrainRequest(TerrainKind.FLAT, extent), geometry_seed=0,
                                           resident_tiles=True)
        courses = (initial,) * recipe.num_envs
        columns = max(1, int(recipe.num_envs**.5))
        spacing = configuration['environment_spacing_m']
        origins = tuple((spacing * (i % columns), spacing * (i // columns), 0.) for i in range(recipe.num_envs))
        scene_recipe = NativeSceneRecipe(configuration['asset_directory'], recipe.asset_manifest_sha256,
                                         response, origins, tuple(c.surface.meshes for c in courses))
        scene = scene_recipe.build_on_admitted_context(sim, physics_health=physics_health)
        physics_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
        scene.readback['GPU_pair_buffers'] = _read_pair_buffers(physics_prim, physics_configuration)
        dynamics_receipt = apply_startup_dynamics(
            scene, tuning.dynamics, recipe.training_seed + 15485863, native_to_tensor=wp.to_torch,
            tensor_to_native=lambda value: wp.from_torch(value, dtype=wp.uint32 if value.dtype == torch.int32
                                                       else wp.float32))
        wheels = WheelCollisionCentres.from_urdf(
            (scene_recipe.asset / scene_recipe.manifest['urdf']).read_bytes(), scene.observation.binding)
        query = omni.physx.get_physx_scene_query_interface()
        wheel_receipt = verify_wheel_cylinders(scene, wheels, query, native_to_tensor=wp.to_torch,
                                               tolerance_m=configuration['wheel_probe_tolerance_m'])
        scene.readback['native_wheel_collision_equivalence_verified'] = True
        _initial_policy_boundary(sim, scene)
        terrain_presence = NativeShapeQueries(query, tuning.terrain_edit.native_query_offset_m)(
            scene.contacts.terrain_paths_by_environment, scene.terrain_meshes,
            scene.environment_origins_m, required_present=True)
        sensors = SensorModel(recipe.num_envs, recipe.training_seed + 1299709, tuning.sensors,
                              device=recipe.device, initial_physical_tick=scene.observation.physical_tick)
        value = NativeTaskRuntime.on_admitted_scene(
            sim, scene, recipe, components, tuning, asset_directory=scene_recipe.asset, response=response,
            design_courses=courses, native_to_tensor=wp.to_torch,
            sensor_source=sensors)
        write_json(Path(control.binding['attempt']) / 'native_setup.json', {
            'boot': boot_id(), 'scene_readback': scene.readback, 'wheel_probe': wheel_receipt,
            'terrain_presence': terrain_presence, 'solver_position_iterations': 12,
            'solver_velocity_iterations': 6, 'automatic_evaluation': False,
            'timing': 'selected_effective_response_no_additional_USB_transport_delay',
            'sensor_noise_or_delay_randomization_active': True, 'sensor_configuration': configuration['tuning']['sensors'],
            'sensor_scope': 'actor only; physical-unit uniform noise after past-only observation lag',
            'sensor_delay_scope': 'engineering stress separate from recorded drive response; no extra USB delay',
            'startup_dynamics': dynamics_receipt,
            'hardware_qualified': False,
            'full_task_health_qualified': False})
        yield value
        control.publish(control.state['phase'], SDK_zero_copy_compatibility=dict(statistics))
    except BaseException as error:
        # Kit may exit during close. Publish an actual functional error first,
        # so a zero waitpid code can never fabricate a learning endpoint.
        control.fail(error)
        raise
    finally:
        if physics_health is not None:
            physics_health.close()
        if launcher is not None:
            launcher.app.close()
        if compatibility is not None:
            compatibility.__exit__(None, None, None)

"""Admitted IsaacLab lifetime for the managed fast-collector training path.

The existing session parent owns GPU/RAM admission, leases and waitpid. This
factory builds one static course scene; it never launches another worker,
changes a registry alias, migrates a checkpoint or schedules evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

from ...common.training_control import boot_id, verify_pin, write_json
from ..control.actuation import MotorEnvelope, PairTargetDomain, ResponseParameters
from ..simulation.health import NativePhysicsHealth
from ..tasks.termination import NativeTaskLimits
from ..control.sensors import SensorModelConfig
from ..simulation.dynamics import StartupDynamicsConfig
from .contract import FlatTaskSpec
from .critic_features import CRITIC_INPUT_CONTRACT
from .action_transport import ActionLatchConfig, ActionTransportConfig
from ..tasks.command_pool import CommandPoolConfig
from .birth import BirthRandomizationConfig
from ..tasks.reward import REWARD_VERSION
from ..control.communication import CommunicationConfig, EpisodeRandomizationConfig
from .materials import WheelMaterialConfig
from .formal_curriculum import FormalCurriculumCapabilities
from .managed import ManagedCurriculum, ManagedLabRuntime
from .settings import load_asset_description
from .sdk import LabSimulation, verify_recovery_sdk_identity


_PHYSICS_FIELDS = {'gpu_found_lost_pairs_capacity', 'gpu_found_lost_aggregate_pairs_capacity',
                   'gpu_total_aggregate_pairs_capacity', 'gpu_max_rigid_contact_count',
                   'gpu_max_rigid_patch_count', 'gpu_collision_stack_size'}


def _close_runtime_resources(env, health, launcher, failure):
    """Leave fatal native teardown to the worker's explicit nonzero exit."""
    cleanup_failure = None
    for resource in (env, health, launcher):
        if resource is None:
            continue
        if failure is not None or cleanup_failure is not None:
            if launcher is not None:
                launcher.defer_fatal_shutdown(env)
            if resource is env or resource is launcher:
                # env.close() and GA context-manager finalization can both
                # terminate in C++ without returning to Python handlers.
                continue
        try:
            resource.close()
        except BaseException as error:
            if failure is not None:
                failure.add_note(f'Runtime cleanup raised {type(error).__name__}: {error}')
            elif isinstance(error, SystemExit) and error.code in (None, 0):
                continue
            elif cleanup_failure is None:
                cleanup_failure = error
            else:
                cleanup_failure.add_note(f'Additional cleanup raised {type(error).__name__}: {error}')
    if cleanup_failure is not None and launcher is not None:
        launcher.defer_fatal_shutdown(env)
    return cleanup_failure


def control_delay_contract(configuration, response):
    """Full-exposure delay range of each control path, in ms (no simulation).

    observation = shared sensor lag + PD-feedback uplink; target latch =
    held-previous-target substeps; torque = downlink, then the identified
    first-order leg drive lag tau scaled by its episode randomization.
    """
    def ms(low, high):
        return [round(1000 * low, 6), round(1000 * high, 6)]

    sensors = configuration['sensors']
    communication = configuration.get('communication')
    total = (0., 0.) if communication is None else tuple(communication['total_delay_range_s'])
    up = 0. if communication is None else communication['uplink_fraction']
    lags = sorted(sensors['observation_delays_s'])
    always = sensors['extra_delay_probability'] == 1 and sensors['nominal_episode_fraction'] == 0
    latch = configuration.get('action_latch')
    latch_steps = (sorted(latch['delay_substeps']) if latch is not None and latch['enabled'] else [0])
    zero_latch = latch is None or not latch['enabled'] or latch['nominal_fraction'] > 0
    transport = configuration.get('action_transport') or {'enabled': False}
    randomization = configuration.get('episode_randomization') or {}
    tau = max(response.effective_time_constant_s[:4])
    scale = randomization.get('leg_time_constant_scale', (1., 1.))
    return {
        'observation_age_ms': ms((lags[0] if always else 0.) + total[0] * up, lags[-1] + total[1] * up),
        'target_latch_ms': ms(0. if zero_latch else latch_steps[0] * .005, latch_steps[-1] * .005),
        'whole_step_transport': bool(transport['enabled']),
        'pd_feedback_uplink_ms': ms(total[0] * up, total[1] * up),
        'torque_downlink_ms': ms(total[0] * (1 - up), total[1] * (1 - up)),
        'leg_drive_time_constant_ms': ms(min(response.effective_time_constant_s[:4]) * scale[0], tau * scale[1]),
        'exposure': 'ranges at full exposure; reset-local draws scale toward zero before full_at_update',
    }


def parse_configuration(recipe, configuration, authorized_boot):
    """Validate the frozen private runtime before opening a simulator."""
    expected = {'format', 'authorized_boot', 'asset_directory', 'response_source_pin',
                'terrain_atlas_pin', 'environment_spacing_m', 'task', 'pair_domain',
                'physics', 'profile_collection', 'termination_limits', 'sensors', 'dynamics'}
    allowed = expected | {'action_transport', 'wheel_material', 'communication', 'episode_randomization',
                          'action_latch', 'motor_envelope', 'command_pool', 'birth_randomization'}
    if (not isinstance(configuration, dict) or not expected <= set(configuration) <= allowed
            or configuration['format'] != 'managed_lab_runtime_v4'
            or configuration['authorized_boot'] != authorized_boot or authorized_boot != boot_id()
            or recipe.device != 'cuda:0' or recipe.policy.critic_input_dim != 185
            or recipe.policy.history_length != 5 or recipe.collection.policy_dt_s not in (.01, .02)
            or type(configuration['profile_collection']) is not bool
            or type(configuration['environment_spacing_m']) not in (int, float)
            or not math.isfinite(configuration['environment_spacing_m'])
            or configuration['environment_spacing_m'] <= 0
            or not isinstance(configuration['task'], dict)
            or set(configuration['task']) != {field.name for field in fields(FlatTaskSpec)}
            or not isinstance(configuration['termination_limits'], dict)
            or not ({field.name for field in fields(NativeTaskLimits)} - {'pose_failure_confirm_s'}
                    <= set(configuration['termination_limits']) <= {field.name for field in fields(NativeTaskLimits)})
            or not isinstance(configuration['sensors'], dict)
            or not ({field.name for field in fields(SensorModelConfig)} - {'shared_delay'}
                    <= set(configuration['sensors']) <= {field.name for field in fields(SensorModelConfig)})
            or not isinstance(configuration['dynamics'], dict)
            or set(configuration['dynamics']) != {field.name for field in fields(StartupDynamicsConfig)}
            or not isinstance(configuration['physics'], dict)
            or set(configuration['physics']) != _PHYSICS_FIELDS
            or any(type(value) is not int or value < 1 for value in configuration['physics'].values())):
        raise ValueError('managed Lab runtime requires its exact current-boot physical configuration')
    task = FlatTaskSpec(**configuration['task'])
    limits = dict(configuration['termination_limits'])
    limits['forbidden_body_names'] = tuple(limits['forbidden_body_names'])
    limits = NativeTaskLimits(**limits)
    sensors = SensorModelConfig(**configuration['sensors'])
    dynamics = StartupDynamicsConfig(**configuration['dynamics'])
    if sensors.physics_dt_s != .005 or sensors.policy_dt_s != recipe.collection.policy_dt_s:
        raise ValueError('managed Lab sensor delay must share the actual physical/policy clock')
    transport_raw = configuration.get('action_transport')
    transport = (ActionTransportConfig() if transport_raw is None
                 else ActionTransportConfig.from_dict(transport_raw))
    if configuration.get('wheel_material') is not None:
        WheelMaterialConfig.from_dict(configuration['wheel_material'])
    if configuration.get('communication') is not None:
        CommunicationConfig.from_dict(configuration['communication'])
    if configuration.get('episode_randomization') is not None:
        EpisodeRandomizationConfig.from_dict(configuration['episode_randomization'])
    for key, parser in (('action_latch', ActionLatchConfig), ('motor_envelope', MotorEnvelope),
                        ('command_pool', CommandPoolConfig), ('birth_randomization', BirthRandomizationConfig)):
        if configuration.get(key) is not None:
            parser.from_dict(configuration[key])
    latch_raw = configuration.get('action_latch')
    if transport.enabled and latch_raw is not None and ActionLatchConfig.from_dict(latch_raw).enabled:
        raise ValueError('policy-target delay needs one model: whole-step transport and the substep latch both enabled')
    if (tuple(task.action_bounds) != tuple(recipe.collection.action_bounds)
            or task.nominal_height_m != recipe.domain.nominal_height_m):
        raise ValueError('managed Lab task differs from the collector action bounds or nominal height')
    asset = load_asset_description(configuration['asset_directory'],
                                   expected_manifest_sha256=recipe.asset_manifest_sha256)
    response_path = verify_pin(configuration['response_source_pin'])
    response = ResponseParameters.from_recorded_response(json.loads(response_path.read_text())['actuator_response'])
    atlas = verify_pin(configuration['terrain_atlas_pin'])
    atlas_document = json.loads(atlas.read_text())
    payloads = atlas_document.get('files')
    if (atlas_document.get('format') != 'terrain_course_atlas_v1'
            or not isinstance(payloads, dict)
            or not {'terrain.usda', 'geometry.npz', 'query_index.npz'} <= payloads.keys()):
        raise ValueError('managed Lab requires the prepared static course atlas')
    for name, pin in payloads.items():
        # Prepared atlases deliberately store basename pins, so their whole
        # directory can move without rewriting the source manifest identity.
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..')
                or not isinstance(pin, dict) or set(pin) != {'file', 'bytes', 'sha256'}
                or pin['file'] != name):
            raise ValueError('managed Lab atlas payload escapes its frozen directory')
        verify_pin({'path': str(atlas.parent / name), 'bytes': pin['bytes'], 'sha256': pin['sha256']})
    pair = configuration['pair_domain']
    pair = None if pair is None else PairTargetDomain(**pair)
    identity = {'runtime_sha256': hashlib.sha256(json.dumps(configuration, sort_keys=True,
                 separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                'asset_manifest_sha256': asset.manifest_sha256,
                'atlas_sha256': configuration['terrain_atlas_pin']['sha256'],
                'response_sha256': configuration['response_source_pin']['sha256'],
                'critic_input_contract': CRITIC_INPUT_CONTRACT}
    return asset, response, pair, task, atlas, identity, limits, sensors, dynamics, transport


@contextmanager
def runtime(recipe, components, configuration, *, control):
    """Expose the actual Lab task to the complete managed learning loop."""
    parsed = parse_configuration(recipe, configuration, control.binding['boot_id'])
    # Accept the old nine-value return used by existing synthetic fixtures;
    # current parser returns the explicit action transport as the tenth value.
    if len(parsed) == 9:
        asset, response, pair, task, atlas, identity, limits, sensors, dynamics = parsed
        transport = ActionTransportConfig()
    elif len(parsed) == 10:
        asset, response, pair, task, atlas, identity, limits, sensors, dynamics, transport = parsed
    else:
        raise ValueError('runtime parser returned an unexpected configuration tuple')
    # Check every stage before scene setup or any reserved production rollout.
    ManagedCurriculum(recipe, components, FormalCurriculumCapabilities(True, True, True))
    launcher = env = health = None
    failure = None
    try:
        sdk_identity = verify_recovery_sdk_identity(
            control.binding['root'], authorized_boot=control.binding['boot_id'],
            recipe_sha256=recipe.sha256, learner_source_sha256=recipe.learner_source_sha256)
        launcher = LabSimulation(recipe.device)
        import omni.kit.app
        import torch
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        from .cfg import build_env_cfg
        from .env import WheeledLabEnv
        from .formal_reward import FormalRewardConfig

        launcher.configure_settings()
        extensions = omni.kit.app.get_app().get_extension_manager()
        if not extensions.is_extension_enabled('omni.physx.gpu'):
            extensions.set_extension_enabled_immediate('omni.physx.gpu', True)
        torch.cuda.set_device(0)
        torch.cuda.init()
        health = NativePhysicsHealth.for_native_process(control.binding['attempt'])
        cfg = build_env_cfg(
            str(asset.directory), num_envs=recipe.num_envs,
            env_spacing_m=configuration['environment_spacing_m'], device=recipe.device,
            response=response, pair_domain=pair, task=task,
            formal_reward=FormalRewardConfig.from_manifest(
                asset.manifest, normal_load_source='native_contact_patch_normal_forces'),
            termination_limits=limits,
            sensors=sensors, dynamics=dynamics, action_transport=transport,
            communication=(None if configuration.get('communication') is None
                           else CommunicationConfig.from_dict(configuration['communication'])),
            episode_randomization=(None if configuration.get('episode_randomization') is None
                                   else EpisodeRandomizationConfig.from_dict(configuration['episode_randomization'])),
            wheel_material=(None if configuration.get('wheel_material') is None
                            else WheelMaterialConfig.from_dict(configuration['wheel_material'])),
            action_latch=(None if configuration.get('action_latch') is None
                          else ActionLatchConfig.from_dict(configuration['action_latch'])),
            motor_envelope=(None if configuration.get('motor_envelope') is None
                            else MotorEnvelope.from_dict(configuration['motor_envelope'])),
            command_pool=(None if configuration.get('command_pool') is None
                          else CommandPoolConfig.from_dict(configuration['command_pool'])),
            birth_randomization=(None if configuration.get('birth_randomization') is None
                                 else BirthRandomizationConfig.from_dict(configuration['birth_randomization'])),
            declarative_scene=launcher.declarative_scene,
            terrain_scan=True, forward_scan=True, terrain_profile='course',
            terrain_atlas_manifest=str(atlas), expected_manifest_sha256=asset.manifest_sha256)
        cfg.seed = recipe.training_seed
        # The course queries its immutable triangles on Warp's Torch stream.
        # Updating a second PhysX raycast/overlap BVH for every robot shape is
        # unused work; contact reporting and collision solving stay enabled.
        cfg.sim.enable_scene_query_support = False
        cfg.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True,
                                  enable_scene_query_support=False, **configuration['physics'])
        if sdk_identity is not None:
            from isaaclab.utils.string import string_to_callable
            from isaaclab_physx.physics import PhysxManager

            selected = cfg.sim.physics.class_type
            selected = string_to_callable(selected) if isinstance(selected, str) else selected
            if (not launcher.declarative_scene or cfg.sim.use_newton_actuators is not False
                    or selected is not PhysxManager):
                raise ValueError('migrated SDK launch differs from its PhysX actuator contract')
            sdk_identity['launch_contract'] = {
                'physics_manager': 'isaaclab_physx.physics.physx_manager:PhysxManager',
                'use_newton_actuators': False, 'quaternion_order': 'xyzw'}
        if launcher.declarative_scene:
            from .ga_isolation import configure_ga_isolation

            object.__setattr__(cfg, '_native_clone_binding', configure_ga_isolation(
                cfg.sim.physics, cfg.scene.robot, recipe.num_envs, recipe.device))
        env = WheeledLabEnv(cfg)
        health.check()
        if configuration.get('wheel_material') is not None:
            identity['wheel_material_parameters_sha256'] = (
                env.wheel_material_receipt['startup_randomization']['native_parameters_sha256'])
        value = ManagedLabRuntime(env, recipe, components, identity,
                                  profile=configuration['profile_collection'], physics_health=health)
        health.check()
        write_json(Path(control.binding['attempt']) / 'lab_setup.json', {
            'boot': boot_id(), 'identity': identity, 'recipe_sha256': recipe.sha256,
            'num_envs': env.num_envs, 'rollout_steps': recipe.rollout_steps,
            'physics_dt_s': cfg.sim.dt, 'policy_dt_s': env.step_dt,
            'solver_position_iterations': cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count,
            'solver_velocity_iterations': cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count,
            'physics_capacity': configuration['physics'],
            'reward_contract': value.checkpoint_state_dict()['reward_contract'],
            'termination_contract': {'limits': configuration['termination_limits'],
                                     'contact_source': 'native_contact_patch_normal_forces',
                                     'failure_latch_clock': 'each_physical_substep'},
            'contact_mapping': getattr(env, 'contact_mapping_receipt', None),
            'contact_measurement': env._contact_loads.receipt,
            'terrain_collision': env.terrain_collision_receipt,
            'startup_dynamics': env._startup_dynamics.receipt,
            'mechanical_asset': (None if env._recovery_reset is None
                                 else env._recovery_reset.receipt),
            'gas_spring': (None if getattr(env, '_spring_source_sha256', None) is None
                           else {'source_sha256': dict(env._spring_source_sha256),
                                 'application': 'main_link_mount_force_and_com_torque',
                                 'slider_effort': 'zero_target_with_manifest_damping_only'}),
            'sensors': {'configuration': configuration['sensors'],
                        'scope': 'actor encoder and IMU only; critic and supervision remain clean',
                        'timebase': 'past-only physical-clock samples; fresh reset projection'},
            'action_transport': {'configuration': configuration.get('action_transport'),
                                 'scope': 'policy-action to controller-target latch; whole policy steps; not USB one-way latency'},
            'communication': {'configuration': configuration.get('communication'),
                              'allocation': 'one round-trip draw; uplink shared by PD and Actor; downlink before motor response',
                              'substep_model': 'duration-weighted torque impulse, not intermediate robot dynamics'},
            'episode_randomization': {'configuration': configuration.get('episode_randomization'),
                                     'sampling': 'episode reset; startup mass and wheel materials remain fixed'},
            'action_latch': {'configuration': configuration.get('action_latch'),
                             'scope': 'previous policy target held for whole 5 ms substeps at the controller latch'},
            'motor_envelope': None if configuration.get('motor_envelope') is None else {
                'constants': asdict(MotorEnvelope()),
                'scope': 'datasheet-derived linear back-EMF limit on true shaft speed; not identified under load'},
            'command_pool': None if configuration.get('command_pool') is None else asdict(CommandPoolConfig()),
            'birth_randomization': (None if configuration.get('birth_randomization') is None
                                    else asdict(BirthRandomizationConfig.from_dict(configuration['birth_randomization']))),
            'reward_version': REWARD_VERSION,
            'control_delay_contract': control_delay_contract(configuration, response),
            'wheel_material': env.wheel_material_receipt,
            'collision_isolation': env.collision_isolation_receipt,
            'cold_physical_state_restored': False, 'automatic_evaluation': False,
            'sdk_scene_api': 'declarative' if launcher.declarative_scene else 'manual_clone',
            'sdk_identity': sdk_identity,
            'quaternion_order': 'xyzw', 'velocity_frame': 'world_COM_twist',
            'hardware_qualified': False, 'full_task_health_qualified': False})
        yield value
        health.check()
    except BaseException as error:
        # Preserve a real failure before Kit shutdown can obscure the exit code.
        failure = error
        try:
            control.fail(error)
        except BaseException as publication_error:
            error.add_note('Runtime failure publication raised: ' + repr(publication_error))
        raise
    finally:
        cleanup_failure = _close_runtime_resources(env, health, launcher, failure)
        if failure is None and cleanup_failure is not None:
            control.fail(cleanup_failure)
            raise cleanup_failure

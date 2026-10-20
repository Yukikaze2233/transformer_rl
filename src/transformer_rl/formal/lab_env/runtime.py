"""Admitted IsaacLab lifetime for the managed fast-collector training path.

The existing session parent owns GPU/RAM admission, leases and waitpid. This
factory builds one static course scene; it never launches another worker,
changes a registry alias, migrates a checkpoint or schedules evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys

from ...common.training_control import boot_id, verify_pin, write_json
from ..control.actuation import PairTargetDomain, ResponseParameters
from ..simulation.health import NativePhysicsHealth
from .contract import FlatTaskSpec
from .formal_curriculum import FormalCurriculumCapabilities
from .managed import ManagedCurriculum, ManagedLabRuntime
from .settings import load_asset_description


_PHYSICS_FIELDS = {'gpu_found_lost_pairs_capacity', 'gpu_found_lost_aggregate_pairs_capacity',
                   'gpu_total_aggregate_pairs_capacity', 'gpu_max_rigid_contact_count',
                   'gpu_max_rigid_patch_count'}


def parse_configuration(recipe, configuration, authorized_boot):
    """Validate the frozen private runtime before opening a simulator."""
    expected = {'format', 'authorized_boot', 'asset_directory', 'response_source_pin',
                'terrain_atlas_pin', 'environment_spacing_m', 'task', 'pair_domain',
                'physics', 'profile_collection'}
    if (not isinstance(configuration, dict) or set(configuration) != expected
            or configuration['format'] != 'managed_lab_runtime_v1'
            or configuration['authorized_boot'] != authorized_boot or authorized_boot != boot_id()
            or recipe.device != 'cuda:0' or recipe.policy.critic_input_dim != 185
            or recipe.policy.history_length != 5 or recipe.collection.policy_dt_s != .01
            or type(configuration['profile_collection']) is not bool
            or type(configuration['environment_spacing_m']) not in (int, float)
            or not math.isfinite(configuration['environment_spacing_m'])
            or configuration['environment_spacing_m'] <= 0
            or not isinstance(configuration['task'], dict)
            or set(configuration['task']) != {field.name for field in fields(FlatTaskSpec)}
            or not isinstance(configuration['physics'], dict)
            or set(configuration['physics']) != _PHYSICS_FIELDS
            or any(type(value) is not int or value < 1 for value in configuration['physics'].values())):
        raise ValueError('managed Lab runtime requires its exact current-boot physical configuration')
    task = FlatTaskSpec(**configuration['task'])
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
                'response_sha256': configuration['response_source_pin']['sha256']}
    return asset, response, pair, task, atlas, identity


@contextmanager
def runtime(recipe, components, configuration, *, control):
    """Expose the actual Lab task to the complete managed learning loop."""
    parsed = parse_configuration(recipe, configuration, control.binding['boot_id'])
    asset, response, pair, task, atlas, identity = parsed
    # Check every stage before scene setup or any reserved production rollout.
    ManagedCurriculum(recipe, components, FormalCurriculumCapabilities(True, True, True))
    launcher = env = health = None
    try:
        from isaaclab.app import AppLauncher

        argv = sys.argv
        try:
            sys.argv = [argv[0]]
            launcher = AppLauncher(headless=True, device=recipe.device, enable_cameras=False, livestream=0)
        finally:
            sys.argv = argv
        import omni.kit.app
        import torch
        from isaaclab.app.settings_manager import SettingsManager
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        from .cfg import build_env_cfg
        from .env import WheeledLabEnv
        from .formal_reward import FormalRewardConfig

        settings = SettingsManager.instance()
        settings.initialize_carb_settings()
        settings.set_int('/physics/cudaDevice', 0)
        settings.set_bool('/physics/suppressReadback', True)
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
            formal_reward=FormalRewardConfig.from_manifest(asset.manifest),
            terrain_scan=True, forward_scan=True, terrain_profile='course',
            terrain_atlas_manifest=str(atlas), expected_manifest_sha256=asset.manifest_sha256)
        cfg.seed = recipe.training_seed
        cfg.sim.enable_scene_query_support = True
        cfg.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True,
                                  enable_scene_query_support=True, **configuration['physics'])
        env = WheeledLabEnv(cfg)
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
            'contact_mapping': getattr(env, 'contact_mapping_receipt', None),
            'cold_physical_state_restored': False, 'automatic_evaluation': False,
            'hardware_qualified': False, 'full_task_health_qualified': False})
        yield value
        health.check()
    except BaseException as error:
        # Preserve a real failure before Kit shutdown can obscure the exit code.
        control.fail(error)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except BaseException as error:
            control.fail(error)
            raise
        finally:
            if health is not None:
                health.close()
            if launcher is not None:
                launcher.app.close()

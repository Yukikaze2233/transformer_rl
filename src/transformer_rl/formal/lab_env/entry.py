"""Complete fresh formal-training entry point.

This command composes the frozen 35k settings, the reviewed runtime/resources
and a fresh session root. It performs CPU-side validation and session creation;
it starts Isaac only when ``--start`` is explicitly supplied.

Examples
--------
Prepare only (recommended before reviewing the frozen recipe):

    python -m transformer_rl.formal.lab_env.entry \
        --settings configs/formal_training.json \
        --runtime-json /path/to/runtime.json \
        --resources-json /path/to/resources.json \
        --output /path/to/new-session --seed 2102

Prepare and start at the first update boundary:

    python -m transformer_rl.formal.lab_env.entry ... --start
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from ...common.training_control import read_json
from ..learning.settings import FormalTrainingSettings
from ..tasks.curriculum import (
    COMPLETE_STAGE_BUDGETS, COMPLETE_STAGE_NAMES, COMPLETE_STAGE_WEIGHTS,
    STAGE_BUDGETS, STAGE_NAMES, STAGE_WEIGHTS, ContinuousTaskSampler, RotationMode, TaskFamily,
)
from .action_transport import ActionTransportConfig
from ..control.communication import CommunicationConfig, EpisodeRandomizationConfig
from .prepare import prepare_session
from .settings import load_asset_description


def _delay_steps(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(',') if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError('delay steps must be comma-separated integers') from error
    if not values or any(step < 0 or step > 16 for step in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('delay steps must be unique nonnegative integers <= 16')
    return values


def _parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--runtime-json', type=Path, required=True)
    parser.add_argument('--resources-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2102)
    parser.add_argument('--run-id')
    parser.add_argument('--python', type=Path, default=Path(sys.executable),
                        help='Worker interpreter; use the exact Isaac Lab environment validated by preflight')
    parser.add_argument('--action-delay-steps', type=_delay_steps, default=(0, 1),
                        help='Whole policy-step transport delays; 0,1 means 0 or 10ms at 100Hz')
    parser.add_argument('--action-delay-nominal-fraction', type=float, default=.5,
                        help='Fraction of reset episodes forced to zero transport delay')
    parser.add_argument('--no-action-transport', action='store_true',
                        help='Do not override/replace an existing action_transport field')
    parser.add_argument('--control-timing-json', type=Path,
                        help='Freeze an explicit communication and reset-local randomization preset')
    parser.add_argument('--start', action='store_true',
                        help='Start the worker after creating and validating the session')
    parser.add_argument('--paused', action='store_true',
                        help='With --start, initialize and pause instead of running PPO updates')
    return parser


def _with_transport(runtime, args):
    if not isinstance(runtime, dict):
        raise ValueError('runtime JSON must be an object')
    value = dict(runtime)
    if args.no_action_transport:
        raw = value.get('action_transport')
        if raw is not None:
            ActionTransportConfig.from_dict(raw)
        return value
    transport = ActionTransportConfig(
        enabled=True,
        delays_steps=tuple(args.action_delay_steps),
        nominal_fraction=float(args.action_delay_nominal_fraction),
    )
    value['action_transport'] = {'enabled': transport.enabled,
                                 'delays_steps': list(transport.delays_steps),
                                 'nominal_fraction': transport.nominal_fraction}
    return value


def _with_control_timing(runtime, path):
    value = dict(runtime)
    if path is None:
        return value
    preset = read_json(path)
    if not isinstance(preset, dict) or set(preset) != {'communication', 'episode_randomization'}:
        raise ValueError('control timing preset must bind communication and episode randomization')
    CommunicationConfig.from_dict(preset['communication'])
    EpisodeRandomizationConfig.from_dict(preset['episode_randomization'])
    value.update(preset)
    return value


def _validate_plan(settings, seed):
    if (settings.updates_per_seed != 35000 or settings.num_envs != 8192
            or settings.rollout_steps != 48):
        raise ValueError('formal entry requires the frozen 8192x48, 35k-update plan')
    if (tuple(settings.training_seeds) != (2101, 2102, 2103)
            or type(seed) is not int or seed not in settings.training_seeds):
        raise ValueError('formal entry seed is not in the frozen training plan')
    if sum(COMPLETE_STAGE_BUDGETS) != 35000:
        raise ValueError('formal complete entry must use the four-stage 35k plan')
    if settings.collection.policy_dt_s != .01:
        raise ValueError('formal entry requires the 100Hz policy clock')
    if settings.checkpoint_interval < 1:
        raise ValueError('formal entry requires a positive checkpoint interval')


def _curriculum_report(settings, *, seed, sample_count=12000):
    """CPU-only coverage check over every declared stage and the final bucket."""
    names, budgets, weights = (COMPLETE_STAGE_NAMES, COMPLETE_STAGE_BUDGETS,
                               COMPLETE_STAGE_WEIGHTS)
    if tuple(budgets) != (5000, 20000, 5000, 5000) or sum(budgets) != 35000:
        raise ValueError('complete curriculum budgets differ from 35k')
    report = {'stages': {}, 'final': {}}
    for stage, (name, budget, stage_weights) in enumerate(zip(names, budgets, weights)):
        sampler = ContinuousTaskSampler(seed, domain=settings.domain, continuous=True,
                                        stage_weights=weights)
        draw = sampler.sample(sample_count, stage)
        counts = torch.bincount(draw.family, minlength=len(TaskFamily)).tolist()
        expected = [int(round(sample_count * value)) for value in stage_weights]
        if any(abs(actual - want) > 1 for actual, want in zip(counts, expected)):
            raise ValueError(f'{name} family allocation differs from declared weights')
        report['stages'][name] = {'budget': budget, 'weights': list(stage_weights),
                                   'family_counts': counts}
    sampler = ContinuousTaskSampler(seed, domain=settings.domain, continuous=True,
                                    stage_weights=weights)
    final = sampler.sample(max(sample_count, 20000), len(COMPLETE_STAGE_NAMES) - 1)
    stage_recovery = set()
    for stage in range(len(COMPLETE_STAGE_NAMES)):
        stage_sampler = ContinuousTaskSampler(seed, domain=settings.domain, continuous=True,
                                              stage_weights=weights)
        stage_draw = stage_sampler.sample(sample_count, stage)
        stage_recovery.update(stage_draw.recovery_posture[
            stage_draw.family == TaskFamily.RECOVERY].tolist())
    families = torch.bincount(final.family, minlength=len(TaskFamily))
    expected = [int(round(len(final.family) * value)) for value in COMPLETE_STAGE_WEIGHTS[-1]]
    if any(abs(actual - want) > 1 for actual, want in zip(families.tolist(), expected)):
        raise ValueError('final family allocation differs from declared weights')
    rotation_mask = final.family == TaskFamily.ROTATION
    rotation = final.requested_command[rotation_mask]
    rotation_modes = final.rotation_mode[rotation_mask]
    modes = torch.bincount(rotation_modes, minlength=len(RotationMode))
    normal = rotation_modes == int(RotationMode.NORMAL)
    spin = (rotation_modes == int(RotationMode.SPIN_LOW)) | (rotation_modes == int(RotationMode.SPIN_MID))
    dash = rotation_modes == int(RotationMode.DASH)
    terrain_mask = final.family == TaskFamily.TERRAIN
    request_mask = final.family == TaskFamily.REQUEST
    checks = {
        'forward_within_contract': bool((final.requested_command[:, 0].abs() <= 2.7 + 1e-6).all()),
        'height_within_contract': bool((final.requested_command[:, 3] >= settings.domain.height_m[0] - 1e-6).all()
                                       and (final.requested_command[:, 3] <= settings.domain.height_m[1] + 1e-6).all()),
        'height_endpoint_coverage': bool(final.requested_command[:, 3].min() <= settings.domain.height_m[0] + 1e-6
                                         and final.requested_command[:, 3].max() >= settings.domain.height_m[1] - 1e-6),
        'lateral_always_zero': bool((final.requested_command[:, 1] == 0).all()),
        'yaw_within_declared_domain': bool((final.requested_command[:, 2].abs() <= 6 * torch.pi + 1e-6).all()),
        'normal_yaw_bucket': bool((rotation[normal, 2].abs() <= 2 * torch.pi + 1e-6).all()),
        'spin_low_present': int(modes[1]) > 0,
        'spin_mid_present': int(modes[2]) > 0,
        'dash_present': int(modes[3]) > 0,
        'spin_forward_zero': bool((rotation[spin, 0] == 0).all()),
        'dash_forward_bucket': bool((rotation[dash, 0].abs() >= 2.0 - 1e-6).all()
                                    and (rotation[dash, 0].abs() <= 2.7 + 1e-6).all()),
        'request_jump_range': bool((final.jump_height_m[request_mask] >= .03 - 1e-6).all()
                                   and (final.jump_height_m[request_mask] <= .40 + 1e-6).all()),
        'terrain_kinds': set(final.terrain_kind[terrain_mask].tolist()) >= set(range(1, 8)),
        'flat_rehearsal': bool((final.terrain_kind[terrain_mask] == 0).any()),
        'recovery_postures': stage_recovery == set(range(16)),
        'trajectory_kinds': set(final.trajectory_kind.tolist()) == set(range(7)),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError('complete curriculum checks failed: ' + ', '.join(failed))
    report['final'] = {
        'samples': int(len(final.family)),
        'family_counts': families.tolist(),
        'rotation_modes': modes.tolist(),
        'rotation_sampled_yaw_abs_max': float(rotation[:, 2].abs().max()),
        'declared_yaw_abs_max': float(6 * torch.pi),
        'covered_yaw_abs_max': float(rotation[:, 2].abs().max()),
        'checks': checks,
        'notice': 'declared yaw is +/-6pi; normal +/-2pi, spin low 2pi-3.25pi, spin mid 3.25pi-6pi',
    }
    return report


def _asset_report(asset_directory):
    asset = load_asset_description(asset_directory)
    manifest = asset.manifest
    constraints = manifest.get('constraints', [])
    # Constraints are stored in model_spec, not always duplicated in manifest.
    spec = json.loads(asset.bound_bytes('model_spec.json'))
    constraints = spec.get('constraints', [])
    files = manifest.get('files_sha256', {})
    return {
        'directory': str(asset.directory),
        'manifest_sha256': asset.manifest_sha256,
        'model_kind': manifest.get('model_kind'),
        'asset_revision': manifest.get('asset_revision'),
        'total_mass_kg': manifest.get('total_mass_kg'),
        'nominal_base_height_m': manifest.get('nominal_base_height_m'),
        'body_count': len(manifest.get('rigid_body_names', [])),
        'tree_joint_count': len(manifest.get('tree_joint_names', [])),
        'loop_constraint_count': len(constraints),
        'constraints': [item.get('name') for item in constraints],
        'active_joint_names': manifest.get('control_joint_names'),
        'spring_joint_names': manifest.get('spring_joint_names'),
        'wheel_gear_ratio': manifest.get('wheel_gear_ratio'),
        'knee_inner_limits_deg': manifest.get('knee_inner_limits_deg'),
        'source_inertia_invalid_count': manifest.get('source_inertia_invalid_count'),
        'inertia_rebuilt_count': manifest.get('inertia_rebuilt_count'),
        'hardware_deployment_ready': manifest.get('hardware_deployment_ready'),
        'training_ready': manifest.get('training_ready'),
        'static_validation_passed': manifest.get('static_validation_passed'),
        'dynamics_validation': manifest.get('dynamics_validation'),
        'physx_validation': manifest.get('physx_validation'),
        'file_count': len(files),
        'file_bytes': sum((asset.directory / name).stat().st_size for name in files),
        'urdf_is_tree_only': True,
        'closed_chain_source': 'model_spec.json constraints / robot.usda',
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.paused and not args.start:
        raise ValueError('--paused requires --start')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('formal entry requires a new session directory')
    settings = FormalTrainingSettings.load(args.settings)
    _validate_plan(settings, args.seed)
    runtime = _with_control_timing(_with_transport(read_json(args.runtime_json), args),
                                   args.control_timing_json)
    curriculum = _curriculum_report(settings, seed=args.seed)
    asset_report = _asset_report(runtime['asset_directory'])
    resources = read_json(args.resources_json)
    session = prepare_session(
        args.output, settings, runtime, resources, training_seed=args.seed,
        include_jump=True, run_id=args.run_id, python=str(args.python),
    )
    result = {
        'root': str(session.root),
        'target_updates': session.spec['updates'],
        'recipe_sha256': session.spec['recipe_sha256'],
        'action_transport': runtime.get('action_transport'),
        'communication': runtime.get('communication'),
        'episode_randomization': runtime.get('episode_randomization'),
        'curriculum': curriculum,
        'asset': asset_report,
        'tensorboard': {'enabled': True,
                        'pattern': str(session.root / 'attempts' / '*' / 'train/tensorboard/formal')},
        'training_started': False,
        'next_command': ['trainctl', 'start', str(session.root)],
    }
    if args.start:
        state = session.start(paused=args.paused)
        result['start_requested'] = True
        result['training_started'] = bool(state.get('active', False))
        attempt = session.attempt()
        if attempt is not None:
            result['tensorboard'] = {
                'enabled': True,
                'path': str(attempt / 'train/tensorboard/formal'),
            }
        result['state'] = state
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Byte-bound fixed exposure definitions for a complete architecture/history grid.

This module freezes and reconstructs input identities. It does not launch a
learner, acquire resources, evaluate policies or select a winning architecture.
The operating-system controller must consume this definition without reducing
its denominator and use one isolated process per stage.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

import numpy as np
import torch

from . import retention_campaign as predecessors
from .experiments import source_identity
from .exposure_training import initial_model_sha256
from .frame_config import FrameTrainConfig, digest, json_bytes
from .history_study import validate_history_study
from .frame_study import validate_study


FORMAT = 'transformer_rl.fixed_exposure_protocol'
PACKAGE = Path(__file__).resolve().parent


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _path(value):
    path = Path(value)
    _require(path.is_absolute() and path.resolve() == path
             and not any(p.is_symlink() for p in (path, *path.parents)),
             'absolute canonical nonsymlink path required')
    return path


def _receipt(value):
    path = _path(value)
    before = path.stat()
    _require(stat.S_ISREG(before.st_mode), 'regular input file required')
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             'input changed during receipt construction')
    return {'path': str(path), 'sha256': sha, 'bytes': before.st_size}


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate input JSON key')
            result[key] = value
        return result
    value = json.loads(_path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite input JSON')))
    json_bytes(value)
    _require(type(value) is dict, 'input JSON object required')
    return value


def _tree(root, *, mutable_subdirectories=()):
    """Actual declared input bytes, excluding Git and rebuildable Python caches."""
    root = _path(root)
    _require(root.is_dir(), 'runtime/input tree must exist')
    mutable = [Path(value) for value in mutable_subdirectories]
    _require(all(not value.is_absolute() and value.parts
                 and '..' not in value.parts and str(value) == original
                 for value, original in zip(mutable, mutable_subdirectories)),
             'exact relative SDK mutable subdirectories required')
    mutable_device = root.lstat().st_dev

    def is_mutable(path):
        return any(path.relative_to(root).is_relative_to(value) for value in mutable)

    files = {}
    for base, directories, names in os.walk(root, followlinks=False):
        # Exclusions waive cache bytes, never path/type checks. A symlink with
        # an excluded name must not silently escape the declared input tree.
        for name in sorted(directories):
            path = _path(Path(base) / name)
            observed = path.lstat()
            _require(stat.S_ISDIR(observed.st_mode), 'input tree contains a special directory')
            if is_mutable(path):
                _require(observed.st_uid == os.getuid() and observed.st_dev == mutable_device,
                         'SDK mutable directory is foreign-owned or crosses its filesystem')
        # Still traverse excluded directories to reject unsafe descendant
        # types; only their regular file bytes are omitted from the receipt.
        directories[:] = sorted(directories)
        for name in sorted(names):
            path = _path(Path(base) / name)
            observed = path.lstat()
            _require(stat.S_ISREG(observed.st_mode), 'regular input file required')
            if is_mutable(path):
                _require(observed.st_uid == os.getuid() and observed.st_dev == mutable_device,
                         'SDK mutable file is foreign-owned or crosses its filesystem')
                continue
            if name.endswith('.pyc') or any(
                    part in ('.git', '__pycache__') for part in path.relative_to(root).parts[:-1]):
                continue
            files[str(path.relative_to(root))] = _receipt(path)
    _require(bool(files), 'declared runtime/input tree must not be empty')
    value = {'root': str(root), 'files': files, 'sha256': digest(files),
             'exclusions': ['.git directories', '__pycache__ directories', '*.pyc']}
    if mutable:
        value['mutable_subdirectories'] = sorted(map(str, mutable))
    return value


def _sdk_mutable_subdirectories(roots, mutable_paths):
    """Recognize exact writable Kit directories without importing the SDK.

    A directory called cache is not sufficient evidence. The inspected Isaac
    bootstrap, SimulationApp source and native app plugin must share the same
    declared installation root. Extension caches and shipped shaders are
    executable inputs and can never be declared writable exclusions.
    """
    _require(type(mutable_paths) in (list, tuple), 'explicit SDK mutable path list required')
    paths = [_path(path) for path in mutable_paths]
    _require(len(paths) == len(set(paths)), 'SDK mutable paths must be distinct')
    result = {root: [] for root in roots}
    recognized = set()
    for path in sorted(paths):
        owners = [root for root in roots if path.is_relative_to(root)]
        _require(len(owners) == 1, 'SDK mutable path must belong to one declared SDK root')
        root = owners[0]
        relative = path.relative_to(root)
        _require(str(relative) in ('kit/cache', 'kit/data', 'kit/logs'),
                 'only exact Isaac Kit cache, data and logs directories may be mutable')
        observed = path.lstat()
        _require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid()
                 and observed.st_dev == root.lstat().st_dev,
                 'SDK mutable path must be an owned directory on its installation filesystem')
        if root not in recognized:
            initializers = (root / '__init__.py', root / 'python_packages/isaacsim/__init__.py')
            actual_initializers = [candidate for candidate in initializers if candidate.is_file()]
            _require(len(actual_initializers) == 1, 'recognized Isaac SDK bootstrap layout required')
            bootstrap = _receipt(actual_initializers[0])
            bootstrap_source = _path(bootstrap['path']).read_text(encoding='utf-8')
            application_path = root / 'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py'
            application = _receipt(application_path)
            application_source = application_path.read_text(encoding='utf-8')
            plugin_path = root / 'kit/kernel/plugins/libomni.kit.app.plugin.so'
            plugin = _receipt(plugin_path)
            with plugin_path.open('rb') as stream:
                native_header = stream.read(4)
            _require(all(marker in bootstrap_source for marker in ('bootstrap_kernel', 'CARB_APP_PATH', 'ISAAC_PATH'))
                     and all(marker in application_source for marker in ('class SimulationApp', 'CARB_APP_PATH',
                                                                           'kernel/plugins', 'load_plugins'))
                     and native_header == b'\x7fELF', 'recognized Isaac SimulationApp and native Kit plugin required')
            _require(_receipt(actual_initializers[0]) == bootstrap and _receipt(application_path) == application
                     and _receipt(plugin_path) == plugin, 'Isaac SDK layout changed during recognition')
            recognized.add(root)
        result[root].append(str(relative))
    return result


def runtime_identity(runtime_roots, mutable_paths=None):
    """Read CPU runtime identity and all explicitly declared external SDK trees.

    Torch/NumPy import and native extension bytes are pinned. This does not
    inventory every operating-system library, GPU driver or sensor dependency.
    No CUDA context or simulator is constructed by this function.
    """
    _require(type(runtime_roots) is list and bool(runtime_roots), 'explicit external runtime roots required')
    roots = [_path(value) for value in runtime_roots]
    _require(len(roots) == len(set(roots)), 'runtime roots must be distinct')
    _require(not any(a.is_relative_to(b) for i, a in enumerate(roots)
                     for j, b in enumerate(roots) if i != j), 'runtime roots must not overlap')
    mutable = _sdk_mutable_subdirectories(roots, [] if mutable_paths is None else mutable_paths)
    # Both modules have already been imported on CPU; reading these origins
    # does not invoke Isaac AppLauncher or a GPU availability probe.
    from numpy._core import _multiarray_umath
    files = {'python': _receipt(Path(sys.executable).resolve()),
             'torch_import': _receipt(Path(torch.__file__).resolve()),
             'torch_native': _receipt(Path(torch._C.__file__).resolve()),
             'numpy_import': _receipt(Path(np.__file__).resolve()),
             'numpy_native': _receipt(Path(_multiarray_umath.__file__).resolve())}
    return {'python_invocation': sys.executable, 'python_version': sys.version,
            'python_prefix': sys.prefix, 'python_base_prefix': sys.base_prefix,
            'platform': platform.platform(), 'torch_version': str(torch.__version__),
            'numpy_version': np.__version__, 'files': files,
            'declared_sdk_trees': [_tree(path, mutable_subdirectories=mutable[path]) for path in sorted(roots)],
            'coverage': 'interpreter/import/native origins and explicitly declared SDK trees',
            'hardware_and_complete_system_runtime_verified': False}


def _environment_inputs(configs):
    snapshots, contracts = {}, {}
    for config in configs:
        environment = config.environment
        _require(isinstance(environment.get('snapshot'), str)
                 and isinstance(environment.get('snapshot_sha256'), str),
                 'each environment requires its actual frozen snapshot identity')
        root = _path(environment['snapshot'])
        if str(root) not in snapshots:
            manifest = _read(root / 'snapshot.json')
            _require(type(manifest.get('files')) is dict and bool(manifest['files'])
                     and digest(manifest['files']) == manifest.get('sha256'), 'environment snapshot identity differs')
            tree = _tree(root)
            expected = {'snapshot.json', *manifest['files']}
            _require(set(tree['files']) == expected, 'snapshot inventory has missing or extra files')
            for route, sha in manifest['files'].items():
                _require(isinstance(route, str) and not Path(route).is_absolute()
                         and tree['files'][route]['sha256'] == sha,
                         'snapshot member bytes differ from its identity')
            snapshots[str(root)] = {'identity': manifest['sha256'], 'tree': tree,
                                    'manifest': _receipt(root / 'snapshot.json')}
        snapshot = snapshots[str(root)]
        _require(snapshot['identity'] == environment['snapshot_sha256'], 'configuration snapshot SHA differs')
        _require(isinstance(environment.get('contract'), str)
                 and not Path(environment['contract']).is_absolute(), 'relative frozen environment contract required')
        path = _path(root / environment['contract'])
        _require(path.is_relative_to(root) and path != root, 'environment contract escapes its snapshot')
        receipt = _receipt(path)
        _require(receipt['sha256'] == environment.get('contract_sha256'), 'effective contract bytes differ')
        contracts[str(path)] = receipt
    return {'snapshots': [snapshots[key] for key in sorted(snapshots)],
            'contracts': [contracts[key] for key in sorted(contracts)]}


def _positive(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             f'{name} must be positive and finite')


def _protected_roots(history, descriptor, dependencies, runtime_roots, environment):
    result = {str(PACKAGE), str(history)}
    result.update(str(_path(item['path']).parent) for item in descriptor['inputs'].values())
    result.update(str(_path(p)) for p in runtime_roots)
    result.update(item['tree']['root'] for item in environment['snapshots'])
    for dependency in dependencies:
        result.add(str(_path(dependency['summary_path']).parent))
        result.add(str(_path(dependency['definition']['path']).parent))
        result.update(str(_path(p)) for p in dependency['worker_roots'])
        for item in dependency.get('input_receipts', []):
            if isinstance(item, dict) and 'path' in item:
                result.add(str(_path(item['path']).parent))
    return sorted(result)


def freeze(history_root, *, output_root, retention_seed, device, curriculum_summary,
           diagnostic_summary, learning_summary, resource_lock, runtime_roots,
           runtime_mutable_paths=None,
           worker_timeout_seconds=86700., max_wait_seconds=1814400., poll_seconds=20.,
           _allow_existing_output=False):
    """Reconstruct the full grid and all file bindings; create no output or process."""
    history = _path(history_root)
    descriptor = validate_history_study(history)
    _require(descriptor['execution_state'] == 'unobserved', 'use a new unexecuted history preparation')
    plan = validate_study(history / 'study', source=True)
    _require(source_identity() == plan['source'], 'current learner differs from the packed history source')
    history_receipt = _receipt(history / 'history_plan.json')
    spec = plan['spec']
    _require(type(retention_seed) is int and 0 <= retention_seed < 2**32, 'private seed must be uint32')
    eval_seeds = [*spec['evaluation']['validation_seeds'], *spec['evaluation']['seeds']]
    _require(retention_seed not in [*spec['seeds'], *eval_seeds, *spec['training']['anchor_seeds']],
             'private seed must differ from training, evaluation and capture seeds')
    _require(spec['training']['retention_coef'] == 0., 'architecture exposure requires explicit lambda zero')
    _require(isinstance(device, str) and re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', device)
             and device in spec['execution']['devices'], 'explicit declared execution device required')
    for name, value in (('worker_timeout_seconds', worker_timeout_seconds),
                        ('max_wait_seconds', max_wait_seconds), ('poll_seconds', poll_seconds)):
        _positive(value, name)
    _require(worker_timeout_seconds > spec['training']['max_seconds'], 'hard process deadline must exceed the soft stage deadline')
    _require(poll_seconds <= 60 and poll_seconds <= max_wait_seconds, 'bounded observation interval required')
    dependencies, locks = predecessors._dependencies(curriculum_summary, diagnostic_summary,
                                                       learning_summary, str(_path(resource_lock)))
    _require([item['role'] for item in dependencies] == ['curriculum', 'diagnostics', 'learning']
             and len({item['summary_path'] for item in dependencies}) == 3,
             'three distinct original queue definitions required')
    _require(len(locks) == 2 and len({item['path'] for item in locks}) == 2,
             'two distinct original resource locks required')
    for pin in locks:
        _require(predecessors._lock_signature(pin['path']) == pin, 'original lock inode changed')
    _require(runtime_mutable_paths is None or type(runtime_mutable_paths) in (list, tuple),
             'explicit SDK mutable path list required')
    mutable_paths = sorted(str(_path(path)) for path in (runtime_mutable_paths or []))
    _require(len(mutable_paths) == len(set(mutable_paths)), 'SDK mutable paths must be distinct')
    runtime = (runtime_identity(runtime_roots, mutable_paths=mutable_paths) if mutable_paths
               else runtime_identity(runtime_roots))
    configs = {}
    for route in plan['configs']:
        path = _path(history / 'study' / route)
        configs[route] = FrameTrainConfig.from_dict(_read(path))
    environment = _environment_inputs(list(configs.values()))
    destination = _path(output_root)
    _require(destination.parent.is_dir() and (_allow_existing_output or not os.path.lexists(destination)),
             'new output with existing parent required; no directory reuse')
    _require(not os.path.lexists(destination) or destination.is_dir(),
             'existing campaign output must be a directory')
    protected = _protected_roots(history, descriptor, dependencies, runtime_roots, environment)
    input_roots = [history, PACKAGE, *(_path(item['path']).parent for item in descriptor['inputs'].values()),
                   *(_path(item['tree']['root']) for item in environment['snapshots'])]
    for runtime_root in map(_path, runtime_roots):
        _require(not any(runtime_root.is_relative_to(root) or root.is_relative_to(runtime_root)
                         for root in input_roots), 'external SDK tree overlaps history, source or environment inputs')
    for root in protected:
        path = _path(root)
        _require(not destination.is_relative_to(path) and not path.is_relative_to(destination),
                 'campaign output overlaps original inputs, runtime or source')
    jobs, cells = [], []
    for variant in spec['variants']:
        for seed in spec['seeds']:
            identity = {'candidate': variant['name'], 'training_seed': seed}
            job_id = 'job_' + digest(identity)[:24]
            stages, updates, transitions = [], 0, 0
            for index, stage in enumerate(spec['stages']):
                route = f'configs/{variant["name"]}.train.{stage["name"]}.json'
                config = configs[route]
                stage_samples = stage['updates'] * spec['training']['rollout_steps'] * config.environment['num_envs']
                updates += stage['updates']
                transitions += stage_samples
                stages.append({'name': stage['name'], 'index': index, 'config': config.to_dict(),
                    'config_receipt': _receipt(history / 'study' / route), 'updates': stage['updates'],
                    'fresh_transitions': stage_samples, 'expected_cumulative_updates': updates,
                    'expected_cumulative_transitions': transitions})
            guard = initial_model_sha256(configs[f'configs/{variant["name"]}.train.{spec["stages"][0]["name"]}.json'], seed)
            jobs.append({**identity, 'id': job_id, 'retention_seed': retention_seed,
                         'initial_model_sha256': guard, 'stages': stages,
                         'reserved_updates': updates, 'reserved_fresh_transitions': transitions})
            for index in range(len(stages)):
                for role, seeds in (('validation', spec['evaluation']['validation_seeds']),
                                    ('held_out', spec['evaluation']['seeds'])):
                    for evaluation_seed in seeds:
                        for case in spec['scenarios']:
                            route = f'configs/{variant["name"]}.eval.{case["name"]}.json'
                            config = configs[route]
                            cell = {'job_id': job_id, 'stage_index': index, 'role': role,
                                    'seed': evaluation_seed, 'scenario': case['name']}
                            cells.append({**cell, 'id': 'cell_' + digest(cell)[:24],
                                'config_receipt': _receipt(history / 'study' / route),
                                'num_envs': config.environment['num_envs'],
                                'expected_policy_samples': spec['evaluation']['steps'] * config.environment['num_envs']})
    _require(len({job['id'] for job in jobs}) == len(jobs)
             and len({cell['id'] for cell in cells}) == len(cells), 'grid identity collision')
    budget = {'jobs': len(jobs), 'training_updates': sum(j['reserved_updates'] for j in jobs),
              'fresh_transitions': sum(j['reserved_fresh_transitions'] for j in jobs),
              'validation_cells': sum(c['role'] == 'validation' for c in cells),
              'held_out_cells': sum(c['role'] == 'held_out' for c in cells),
              'evaluation_transition_upper': sum(c['expected_policy_samples'] for c in cells)}
    _require(budget['jobs'] == descriptor['budget']['job_count']
             and budget['training_updates'] == descriptor['budget']['updates_all_jobs_upper']
             and budget['fresh_transitions'] == descriptor['budget']['fresh_transitions_all_jobs_upper'],
             'full history job/sample denominator differs')
    _require(validate_history_study(history) == descriptor
             and _receipt(history / 'history_plan.json') == history_receipt
             and source_identity() == plan['source'], 'history inputs or learner changed while freezing')
    final_runtime = (runtime_identity(runtime_roots, mutable_paths=mutable_paths) if mutable_paths
                     else runtime_identity(runtime_roots))
    _require(_environment_inputs(list(configs.values())) == environment
             and final_runtime == runtime, 'environment or runtime bytes changed while freezing')
    final_dependencies, final_locks = predecessors._dependencies(
        curriculum_summary, diagnostic_summary, learning_summary, str(_path(resource_lock)))
    _require(final_dependencies == dependencies and final_locks == locks,
             'original queue definitions or lock bindings changed while freezing')
    for pin in locks:
        _require(predecessors._lock_signature(pin['path']) == pin,
                 'original lock inode changed while freezing')
    definition = {'format': FORMAT, 'schema_version': 1, 'history_root': str(history),
        'history_manifest': history_receipt, 'history_sha256': descriptor['sha256'],
        'packed_plan_sha256': plan['sha256'], 'source': source_identity(), 'output_root': str(destination),
        'runtime_roots': list(map(str, map(_path, runtime_roots))), 'runtime': runtime,
        'runtime_mutable_paths': mutable_paths,
        'environment_inputs': environment, 'environment_factory': spec['environment_factory'],
        'retention_seed': retention_seed, 'jobs': jobs, 'evaluation_cells': cells, 'budget': budget,
        'evaluation': deepcopy(spec['evaluation']), 'scenarios': deepcopy(spec['scenarios']),
        'selection': deepcopy(spec['selection']), 'protected_roots': protected,
        'execution': {'device': device, 'rollout_steps': spec['training']['rollout_steps'],
            'max_seconds': spec['training']['max_seconds'], 'worker_timeout_seconds': worker_timeout_seconds,
            'max_wait_seconds': max_wait_seconds, 'poll_seconds': poll_seconds,
            'resource_locks': locks, 'dependencies': dependencies},
        'policy': {'stage_execution': 'one_OS_process_per_stage_full_learning_state_resume',
            'training': 'all_declared_stages_no_promotion_rollback_score_rejection_or_retry',
            'reservation': 'charge_full_job_before_first_child_no_refund',
            'evaluation': 'every_stage_all_declared_scenarios_with_missing_cells_preserved',
            'selection': 'validation_only_then_immutable_exact_checkpoint_choice_then_held_out_confirmation',
            'failure': 'numerical_failure_preserves_missing_job;integrity_disk_or_manual_interrupt_stops_campaign'},
        'execution_status': 'unexecuted', 'needs_fixed_authorization_OS_controller': True,
        'independent_evaluation_performed': False, 'hardware_verified': False,
        'formal_architecture_selection': False}
    return {**definition, 'sha256': digest(definition)}


def validate_protocol(protocol):
    """Rebuild all inputs and the complete denominator; a self-signature is insufficient."""
    _require(type(protocol) is dict and protocol.get('format') == FORMAT
             and type(protocol.get('schema_version')) is int and protocol['schema_version'] == 1
             and digest({k: v for k, v in protocol.items() if k != 'sha256'}) == protocol.get('sha256'),
             'invalid exposure protocol identity')
    dependencies = protocol['execution']['dependencies']
    _require([d['role'] for d in dependencies] == ['curriculum', 'diagnostics', 'learning'],
             'original queue membership changed')
    locks = protocol['execution']['resource_locks']
    _require(len(locks) == 2, 'original lock coverage changed')
    # The learning definition identifies the common resource path; the fixed
    # predecessor helper rebinds both original lock inodes on reconstruction.
    shared = set(item['path'] for item in dependencies[0]['locks'])
    for dependency in dependencies[1:]:
        shared.intersection_update(item['path'] for item in dependency['locks'])
    _require(len(shared) == 1, 'exactly one original shared resource lock required')
    execution = protocol['execution']
    expected = freeze(protocol['history_root'], output_root=protocol['output_root'],
        retention_seed=protocol['retention_seed'], device=execution['device'],
        curriculum_summary=dependencies[0]['summary_path'], diagnostic_summary=dependencies[1]['summary_path'],
        learning_summary=dependencies[2]['summary_path'], resource_lock=next(iter(shared)),
        runtime_roots=protocol['runtime_roots'], runtime_mutable_paths=protocol['runtime_mutable_paths'],
        worker_timeout_seconds=execution['worker_timeout_seconds'],
        max_wait_seconds=execution['max_wait_seconds'], poll_seconds=execution['poll_seconds'],
        _allow_existing_output=True)
    _require(protocol == expected, 'exposure source, inputs, runtime, jobs, cells or budget changed')
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    prepare = commands.add_parser('freeze')
    for name in ('history-root', 'output-root', 'protocol-output', 'curriculum-summary',
                 'diagnostic-summary', 'learning-summary', 'resource-lock'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--runtime-roots', type=Path, nargs='+', required=True)
    prepare.add_argument('--runtime-mutable-path', type=Path, action='append', default=[],
                         help='Exact recognized Isaac kit/cache, kit/data or kit/logs directory; repeat as needed')
    prepare.add_argument('--retention-seed', type=int, required=True)
    prepare.add_argument('--device', required=True)
    prepare.add_argument('--worker-timeout-seconds', type=float, default=86700.)
    prepare.add_argument('--max-wait-seconds', type=float, default=1814400.)
    prepare.add_argument('--poll-seconds', type=float, default=20.)
    validate = commands.add_parser('validate')
    validate.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'freeze':
            protocol = freeze(args.history_root, output_root=args.output_root, retention_seed=args.retention_seed,
                device=args.device, curriculum_summary=args.curriculum_summary, diagnostic_summary=args.diagnostic_summary,
                learning_summary=args.learning_summary, resource_lock=args.resource_lock, runtime_roots=args.runtime_roots,
                runtime_mutable_paths=args.runtime_mutable_path,
                worker_timeout_seconds=args.worker_timeout_seconds, max_wait_seconds=args.max_wait_seconds,
                poll_seconds=args.poll_seconds)
            destination = _path(args.protocol_output)
            for root in [protocol['output_root'], *protocol['protected_roots']]:
                path = _path(root)
                _require(not destination.is_relative_to(path) and not path.is_relative_to(destination),
                         'protocol file overlaps output or protected inputs')
            _require(destination.parent.is_dir(), 'protocol output parent must exist')
            predecessors._write_new(destination, protocol)
        else:
            protocol = validate_protocol(_read(args.protocol))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'exposure protocol: {error}\n')
    print(json.dumps({'sha256': protocol['sha256'], 'output_root': protocol['output_root'],
                      'budget': protocol['budget'], 'execution_status': protocol['execution_status'],
                      'needs_fixed_authorization_OS_controller': True}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

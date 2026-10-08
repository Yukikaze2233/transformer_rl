"""Fixed stage exposure without score-controlled promotion or learning resets.

This job layer owns learning and local records. An external controller must own
queue closure, OS workers, resource leases, disk reservations and evaluations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import torch

from .checkpoint import _publish_new_files
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest, json_bytes
from .frame_continuation import FrameContinuation
from .frame_checkpoint import load_frame_checkpoint
from .frame_training import FrameActorCritic
from .frame_workflow import _model_state_sha256
from .ppo import PPOTrainer
from .private_retention import PrivateAnchorRegularizer


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _integer(value, name, minimum=1):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


def _checkpoint_updates(stage):
    """Validate explicit local successful-update boundaries without adding stages."""
    updates = stage['updates']
    values = stage.get('checkpoint_updates', [updates])
    _require(type(values) is list and bool(values)
             and all(type(value) is int and 0 < value <= updates for value in values)
             and values == sorted(set(values)) and values[-1] == updates,
             'checkpoint updates must be increasing local integers including the final update')
    return values


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _new(path, value):
    _publish_new_files({Path(path): json_bytes(value) + b'\n'})


def initial_model_sha256(config, seed):
    """CPU-only named-tensor guard; leaves Python/NumPy/CUDA streams untouched."""
    _require(isinstance(config, FrameTrainConfig), 'parsed training configuration required')
    config = FrameTrainConfig.from_dict(config.to_dict())
    _integer(seed, 'training seed', 0)
    _require(seed < 2**32, 'training seed must be uint32')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(seed)
        return _model_state_sha256(FrameActorCritic(config.model))


def _definition(stages, *, job_id, rollout_steps, training_seed, retention_seed,
                evaluation_seeds, device, expected_initial_model_sha256,
                max_seconds, environment_reference):
    _require(isinstance(job_id, str) and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,127}', job_id),
             'explicit job identity required')
    _require(type(stages) is list and bool(stages), 'nonempty explicit stages required')
    _integer(rollout_steps, 'rollout_steps')
    for name, seed in (('training seed', training_seed), ('retention seed', retention_seed)):
        _integer(seed, name, 0)
        _require(seed < 2**32, f'{name} must be uint32')
    _require(type(evaluation_seeds) in (list, tuple) and len(evaluation_seeds) == len(set(evaluation_seeds))
             and all(type(seed) is int and 0 <= seed < 2**32 for seed in evaluation_seeds)
             and training_seed not in evaluation_seeds
             and retention_seed not in (training_seed, *evaluation_seeds),
             'private seed must be independent of explicit training/evaluation seeds')
    _require(isinstance(device, str) and re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', device),
             'device must be cpu or explicit cuda index')
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and max_seconds > 0,
             'positive finite deadline required')
    _require(isinstance(environment_reference, str) and re.fullmatch(
             r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', environment_reference),
             'environment factory reference required')
    _require(isinstance(expected_initial_model_sha256, str)
             and re.fullmatch(r'[0-9a-f]{64}', expected_initial_model_sha256), 'initial model SHA required')
    configs, declared, names = [], [], set()
    for stage in stages:
        _require(isinstance(stage, dict) and set(stage) in (
            {'name', 'config', 'updates'}, {'name', 'config', 'updates', 'checkpoint_updates'}),
                 'stage fields must be name/config/updates and optional checkpoint_updates')
        name = stage['name']
        _require(isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,63}', name)
                 and name not in names, 'stage names must be unique')
        names.add(name)
        _integer(stage['updates'], 'stage updates')
        _checkpoint_updates(stage)
        config = stage['config']
        config = FrameTrainConfig.from_dict(config.to_dict() if isinstance(config, FrameTrainConfig) else config)
        num_envs = config.environment.get('num_envs')
        _integer(num_envs, 'stage num_envs')
        if configs:
            _require(all(config.to_dict()[key] == configs[0].to_dict()[key]
                         for key in ('model', 'ppo', 'control')), 'only stage environment may change')
        configs.append(config)
        declared.append({'name': name, 'config': config.to_dict(), 'updates': stage['updates'],
                         'transitions_per_update': rollout_steps * num_envs,
                         'fresh_transition_budget': stage['updates'] * rollout_steps * num_envs})
        if 'checkpoint_updates' in stage:
            declared[-1]['checkpoint_updates'] = list(stage['checkpoint_updates'])
    _require(initial_model_sha256(configs[0], training_seed) == expected_initial_model_sha256,
             'initial model SHA mismatch before output/environment construction')
    value = {'format': 'transformer_rl.fixed_exposure_job',
             'schema_version': 2 if any('checkpoint_updates' in stage for stage in declared) else 1,
             'job_id': job_id, 'source': source_identity(), 'stages': declared,
             'training_seed': training_seed, 'retention_seed': retention_seed,
             'evaluation_seeds': list(evaluation_seeds), 'rollout_steps': rollout_steps,
             'device': device, 'max_seconds': max_seconds, 'environment_factory': environment_reference,
             'expected_initial_model_sha256': expected_initial_model_sha256,
             'reserved_updates': sum(s['updates'] for s in declared),
             'reserved_fresh_transitions': sum(s['fresh_transition_budget'] for s in declared),
             'retention_coefficient': 0., 'retry_policy': 'no_retry_no_refund',
             'stage_rule': 'fixed_exposure_no_promotion_rollback_or_score_rejection',
             'evaluation_rule': 'independent_workers_after_learning_only'}
    return {**value, 'sha256': digest(value)}, configs


def train_exposure_job(stages, env_factory, environment_reference, output_root, *,
                       job_id, rollout_steps, training_seed, retention_seed,
                       evaluation_seeds, device, expected_initial_model_sha256,
                       max_seconds, should_stop=None, protected_paths=()):
    """Run every declared stage with complete rollouts and inherited learner state.

    A failed job keeps its full reservation. Previously completed stage endpoints
    remain available for independent evaluation; incomplete endpoints are missing.
    Recorded failures are never retried or silently replaced by a seed. A failed
    artifact publication can leave only a partial directory/reservation; its
    external controller must still retain the charge and refuse directory reuse.
    """
    _require(callable(env_factory) and (should_stop is None or callable(should_stop)),
             'environment and optional stop callbacks must be callable')
    plan, configs = _definition(stages, job_id=job_id, rollout_steps=rollout_steps,
        training_seed=training_seed, retention_seed=retention_seed, evaluation_seeds=evaluation_seeds,
        device=device, expected_initial_model_sha256=expected_initial_model_sha256,
        max_seconds=max_seconds, environment_reference=environment_reference)
    return _run_exposure_plan(plan, configs, env_factory, environment_reference, output_root,
        job_id=job_id, rollout_steps=rollout_steps, training_seed=training_seed,
        retention_seed=retention_seed, evaluation_seeds=evaluation_seeds, device=device,
        expected_initial_model_sha256=expected_initial_model_sha256, max_seconds=max_seconds,
        should_stop=should_stop, protected_paths=protected_paths)


def _run_exposure_plan(plan, configs, env_factory, environment_reference, output_root, *,
                       job_id, rollout_steps, training_seed, retention_seed,
                       evaluation_seeds, device, expected_initial_model_sha256,
                       max_seconds, should_stop, protected_paths, parent=None, segment=False):
    base = parent['endpoint'] if parent else None
    offset = base['stage_index'] + 1 if base else 0
    prior_success = base['cumulative_successful_updates'] if base else 0
    prior_attempts = base['cumulative_attempted_updates'] if base else 0
    prior_samples = base['cumulative_collected_transitions'] if base else 0
    root = Path(output_root)
    _require(root.is_absolute() and root.resolve() == root
             and not any(p.is_symlink() for p in (root, *root.parents)), 'canonical output path required')
    _require(not os.path.lexists(root) and root.parent.is_dir(), 'new output and existing parent required')
    for protected in (Path(__file__).parent.resolve(), *map(lambda p: Path(p).resolve(), protected_paths)):
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                 'output overlaps a protected source/input tree')
    root.mkdir()
    _new(root / 'request.json', plan)
    reservation = {'job_plan_sha256': plan['sha256'],
        'charged_updates': plan['reserved_updates'], 'charged_fresh_transition_budget': plan['reserved_fresh_transitions'],
        'actual_samples': 'recorded separately; reservation is not actual collection', 'refund': False}
    if segment:
        reservation.update(format='transformer_rl.exposure_segment_reservation', schema_version=1,
            charge_scope='segment_itemization_of_external_whole_job_reservation',
            whole_job_reservation_created=False)
    _new(root / 'reservation.json', reservation)
    started = time.monotonic()
    session, endpoints, successful, attempts, collected = None, [], 0, 0, 0
    full_samples, recorded_samples, recorded_updates = 0, 0, 0
    shutdown_errors, stop_reason = [], None
    active_stage, failure_phase = None, 'before_environment'
    optimizer_steps, optimization_sample_uses, status, error = 0, 0, 'completed', None
    sealed_checkpoints, metric_sha, metric_bytes = [], hashlib.sha256(), 0
    scheduled = plan['schema_version'] == 2

    def stop():
        nonlocal stop_reason
        if time.monotonic() - started >= max_seconds:
            stop_reason = 'deadline'
            return True
        if should_stop is not None and should_stop():
            stop_reason = 'caller_stop'
            return True
        return False

    try:
        _require(source_identity() == plan['source'], 'learner source changed before environment construction')
        with (root / 'metrics.jsonl').open('x') as metrics:
            for local_index, (stage, config) in enumerate(zip(plan['stages'], configs)):
                index = offset + local_index
                active_stage = {'name': stage['name'], 'index': index}
                if stop():
                    status = 'interrupted'
                    break
                failure_phase = 'verify_source'
                _require(source_identity() == plan['source'], 'learner source changed at stage boundary')
                directory = root / f'stage_{index:04d}_{stage["name"]}'
                directory.mkdir()
                if session is None and base is None:
                    failure_phase = 'construct_fresh_environment'
                    session = FrameContinuation.start(config, env_factory, environment_reference,
                        expected_initial_model_sha256=expected_initial_model_sha256,
                        training_seed=training_seed, retention_seed=retention_seed,
                        evaluation_seeds=evaluation_seeds, rollout_steps=rollout_steps, device=device)
                else:
                    previous_endpoint = endpoints[-1] if endpoints else base
                    if session is not None:
                        failure_phase = 'close_previous_environment'
                        session.close()
                    failure_phase = 'open_stage_environment'
                    session = FrameContinuation.open(config, env_factory, environment_reference,
                        previous_endpoint['checkpoint']['path'], checkpoint_sha256=previous_endpoint['checkpoint']['sha256'],
                        parent_update=previous_endpoint['cumulative_successful_updates'], cumulative_transitions=previous_endpoint['cumulative_collected_transitions'],
                        consumed_updates=previous_endpoint['cumulative_attempted_updates'], rollout_steps=rollout_steps,
                        training_seed=training_seed, retention_seed=retention_seed,
                        evaluation_seeds=evaluation_seeds, device=device, resume=True, environment_transition=True)
                failure_phase = 'verify_runtime_budget'
                _require(session.config.to_dict() == stage['config'], 'runtime configuration differs from frozen stage')
                _require(session.metadata.get('source') == plan['source'], 'runtime producer source differs from job')
                _require(session.metadata.get('seed') == training_seed
                         and session.metadata.get('environment_factory') == environment_reference,
                         'runtime producer seed or environment factory differs from job')
                _require(session.collector.num_envs * rollout_steps == stage['transitions_per_update'],
                         'actual environment count differs from stage budget')
                session.metadata['fixed_exposure_job'] = {'job_plan_sha256': plan['sha256'],
                    'job_id': job_id, 'stage': stage['name'], 'stage_index': index,
                    'initial_model_sha256': expected_initial_model_sha256}
                for _ in range(stage['updates']):
                    if stop():
                        status = 'interrupted'
                        break
                    failure_phase = 'collect_optimize'
                    _require(session.config.to_dict() == stage['config'], 'runtime configuration changed during stage')
                    _require(session.metadata.get('source') == plan['source'], 'runtime producer source changed during stage')
                    row = session.step(should_stop=stop)
                    if row is None:
                        status = 'partial_rollout'
                        break
                    _require(row['batch_samples'] == stage['transitions_per_update'], 'successful rollout is incomplete')
                    # Learning already happened. Publication failure must not
                    # erase measured successful updates or optimizer accounting.
                    successful += 1
                    full_samples += row['batch_samples']
                    optimizer_steps += row['optimization']['optimizer_steps']
                    optimization_sample_uses += row['optimization']['sample_count']
                    item = {'stage': stage['name'], 'stage_index': index, **row}
                    failure_phase = 'publish_metrics'
                    line = json_bytes(item).decode() + '\n'
                    if metrics.write(line) != len(line):
                        raise OSError('incomplete metrics write')
                    metrics.flush()
                    recorded_updates += 1
                    recorded_samples += row['batch_samples']
                    metric_sha.update(line.encode())
                    metric_bytes += len(line.encode())
                    if scheduled and successful - sum(s['stage_updates'] for s in endpoints) in _checkpoint_updates(stage):
                        local_update = session.update - (prior_success + sum(s['stage_updates'] for s in endpoints))
                        if local_update != stage['updates']:
                            failure_phase = 'fsync_checkpoint_metrics'
                            os.fsync(metrics.fileno())
                            failure_phase = 'verify_source'
                            _require(source_identity() == plan['source'], 'learner source changed before checkpoint')
                            _require(session.config.to_dict() == stage['config'], 'runtime configuration changed before checkpoint')
                            _require(session.metadata.get('source') == plan['source'], 'runtime producer source changed before checkpoint')
                            path = directory / f'checkpoint_{local_update:08d}.pt'
                            failure_phase = 'publish_checkpoint'
                            session.save(path)
                            checkpoint_record = {'format': 'transformer_rl.exposure_learning_checkpoint',
                                'schema_version': 1, 'job_id': job_id, 'job_plan_sha256': plan['sha256'],
                                'stage': stage['name'], 'stage_index': index, 'stage_updates': local_update,
                                'stage_fresh_transitions': local_update * stage['transitions_per_update'],
                                'cumulative_successful_updates': session.update,
                                'cumulative_attempted_updates': session.consumed_updates,
                                'cumulative_collected_transitions': session.collected_transitions,
                                'checkpoint': {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size},
                                'sidecar': {'path': str(path) + '.json', 'sha256': _sha(str(path) + '.json')},
                                'learning_state': 'full_actor_critic_Adam_global_private_RNG_clock',
                                'episode_state_restored': False, 'history_reset': 'repeat_first'}
                            record_path = directory / f'checkpoint_{local_update:08d}.json'
                            failure_phase = 'publish_checkpoint_record'
                            _new(record_path, checkpoint_record)
                            sealed_checkpoints.append({'stage_index': index, 'local_update': local_update,
                                'checkpoint_update': session.update, 'kind': 'intermediate',
                                'record': {'path': str(record_path), 'sha256': _sha(record_path), 'bytes': record_path.stat().st_size},
                                'metrics_prefix': {'path': str(root / 'metrics.jsonl'), 'sha256': metric_sha.hexdigest(),
                                    'bytes': metric_bytes, 'rows': recorded_updates, 'optimizer_steps': optimizer_steps,
                                    'optimization_sample_uses': optimization_sample_uses}})
                if status != 'completed':
                    break
                failure_phase = 'verify_source'
                _require(source_identity() == plan['source'], 'learner source changed during stage')
                _require(session.config.to_dict() == stage['config'], 'runtime configuration changed before endpoint')
                _require(session.metadata.get('source') == plan['source'], 'runtime producer source changed before endpoint')
                path = directory / 'endpoint.pt'
                if scheduled:
                    failure_phase = 'fsync_checkpoint_metrics'
                    os.fsync(metrics.fileno())
                failure_phase = 'publish_checkpoint'
                session.save(path)
                endpoint = {'stage': stage['name'], 'stage_index': index,
                    'stage_updates': stage['updates'], 'stage_fresh_transitions': stage['fresh_transition_budget'],
                    'cumulative_successful_updates': session.update,
                    'cumulative_attempted_updates': session.consumed_updates,
                    'cumulative_collected_transitions': session.collected_transitions,
                    'checkpoint': {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size},
                    'sidecar': {'path': str(path) + '.json', 'sha256': _sha(str(path) + '.json')},
                    'learning_state': 'full_actor_critic_Adam_global_private_RNG_clock',
                    'episode_state_restored': False, 'history_reset': 'repeat_first'}
                if segment:
                    endpoint.update(format='transformer_rl.exposure_segment_endpoint', schema_version=1,
                        job_id=job_id, segment_plan_sha256=plan['sha256'])
                failure_phase = 'publish_endpoint'
                _new(directory / 'endpoint.json', endpoint)
                endpoints.append(endpoint)
                if scheduled:
                    record_path = directory / 'endpoint.json'
                    sealed_checkpoints.append({'stage_index': index, 'local_update': stage['updates'],
                        'checkpoint_update': session.update, 'kind': 'endpoint',
                        'record': {'path': str(record_path), 'sha256': _sha(record_path), 'bytes': record_path.stat().st_size},
                        'metrics_prefix': {'path': str(root / 'metrics.jsonl'), 'sha256': metric_sha.hexdigest(),
                            'bytes': metric_bytes, 'rows': recorded_updates, 'optimizer_steps': optimizer_steps,
                            'optimization_sample_uses': optimization_sample_uses}})
            failure_phase = 'fsync_metrics'
            os.fsync(metrics.fileno())
    except BaseException as failure:
        status = 'user_interrupted' if isinstance(failure, (KeyboardInterrupt, SystemExit)) else 'failed'
        error = {'type': type(failure).__name__, 'message': str(failure), 'phase': failure_phase}
    finally:
        if session is not None:
            attempts = session.consumed_updates - prior_attempts
            collected = session.collected_transitions - prior_samples
            try:
                session.close()
            except BaseException as failure:
                shutdown_errors.append({'type': type(failure).__name__, 'message': str(failure), 'during': 'close'})
                if error is None:
                    status, error = 'failed', shutdown_errors[-1]
    unverified = max(0, collected - full_samples)
    if status == 'completed' and not (len(endpoints) == len(plan['stages'])
            and successful == attempts == plan['reserved_updates']
            and recorded_updates == successful and recorded_samples == full_samples
            and collected == full_samples == plan['reserved_fresh_transitions']):
        status, error = 'failed', {'type': 'BudgetMismatch', 'message': 'actual learning endpoint differs from fixed exposure'}
    completion = {'format': 'transformer_rl.fixed_exposure_completion', 'schema_version': 1,
        'job_plan_sha256': plan['sha256'], 'job_id': job_id, 'status': status, 'error': error,
        'shutdown_errors': shutdown_errors, 'stop_reason': stop_reason,
        'active_stage': active_stage,
        'last_sealed_checkpoint': deepcopy(endpoints[-1]['checkpoint']) if endpoints else None,
        'reserved_updates': plan['reserved_updates'], 'charged_updates': plan['reserved_updates'],
        'charged_fresh_transition_budget': plan['reserved_fresh_transitions'],
        'successful_updates': successful, 'attempted_updates': attempts,
        'unsealed_successful_updates': successful - sum(s['stage_updates'] for s in endpoints),
        'optimizer_update_may_be_partial': attempts > successful,
        'actual_collected_transitions': collected, 'successful_full_rollout_samples': full_samples,
        'recorded_metric_updates': recorded_updates, 'recorded_full_rollout_samples': recorded_samples,
        'unverified_or_unoptimized_samples': unverified, 'optimizer_steps': optimizer_steps,
        'optimizer_accounting_scope': 'successfully returned PPO updates only',
        'failed_update_optimizer_steps': None if attempts > successful else 0,
        'optimization_sample_uses': optimization_sample_uses, 'endpoints': deepcopy(endpoints),
        'missing_stage_endpoints': [s['name'] for s in plan['stages'][len(endpoints):]],
        'unsealed_checkpoint_paths': [str(root / f'stage_{offset+i:04d}_{s["name"]}' / 'endpoint.pt')
            for i, s in enumerate(plan['stages']) if i >= len(endpoints)
            and (root / f'stage_{offset+i:04d}_{s["name"]}' / 'endpoint.pt').is_file()],
        'elapsed_s': time.monotonic() - started, 'refund': False, 'automatic_retries': 0,
        'independent_evaluation_performed': False, 'source': plan['source']}
    if segment:
        completion.update(format='transformer_rl.exposure_segment_completion', stage_index=offset,
            charge_scope='segment_itemization_of_external_whole_job_reservation',
            whole_job_reservation_created=False,
            parent_endpoint=deepcopy(parent['receipt']) if parent else None,
            cumulative_successful_updates=prior_success+successful,
            cumulative_attempted_updates=prior_attempts+attempts,
            cumulative_collected_transitions=prior_samples+collected)
    if scheduled:
        expected_checkpoints = [(offset+i, value) for i, stage in enumerate(plan['stages'])
                                for value in _checkpoint_updates(stage)]
        sealed_keys = {(item['stage_index'], item['local_update']) for item in sealed_checkpoints}
        completion.update(schema_version=2, sealed_checkpoints=deepcopy(sealed_checkpoints),
            missing_checkpoints=[{'stage_index': index, 'local_update': value}
                                 for index, value in expected_checkpoints if (index, value) not in sealed_keys],
            last_sealed_learning_checkpoint=deepcopy(sealed_checkpoints[-1]) if sealed_checkpoints else None)
        declared_paths = []
        for i, stage in enumerate(plan['stages']):
            directory = root / f'stage_{offset+i:04d}_{stage["name"]}'
            for value in _checkpoint_updates(stage):
                path = directory / ('endpoint.pt' if value == stage['updates'] else f'checkpoint_{value:08d}.pt')
                if (offset+i, value) not in sealed_keys and path.is_file():
                    declared_paths.append(str(path))
        completion['unsealed_checkpoint_paths'] = declared_paths
    _new(root / 'completion.json', completion)
    return completion


def _input_path(value):
    _require(isinstance(value, str), 'input receipt path must be a string')
    path = Path(value)
    _require(path.is_absolute() and path.resolve() == path
             and not any(p.is_symlink() for p in (path, *path.parents))
             and path.is_file(), 'canonical existing input file required')
    return path


def _input_receipt(value, *, require_bytes=True):
    fields = {'path', 'sha256', 'bytes'} if require_bytes else {'path', 'sha256'}
    _require(isinstance(value, dict) and set(value) == fields, 'input receipt fields differ')
    path = _input_path(value['path'])
    _require(isinstance(value['sha256'], str)
             and re.fullmatch(r'[0-9a-f]{64}', value['sha256']), 'input receipt SHA required')
    if require_bytes:
        _integer(value['bytes'], 'input receipt bytes')
        _require(path.stat().st_size == value['bytes'], 'input receipt size differs')
    _require(_sha(path) == value['sha256'], 'input receipt SHA differs')
    return path


def _read_input(path, *, canonical=True):
    path = _input_path(str(path))
    before = path.stat()
    raw = path.read_bytes()
    value = json.loads(raw)
    if canonical:
        _require(raw == json_bytes(value) + b'\n', 'input JSON is not canonical')
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'input changed while reading')
    receipt = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    _require(_sha(path) == receipt['sha256'], 'input changed after reading')
    return value, receipt


def _checkpoint_learning_proof(endpoint, endpoint_path, plan, request, config, *, intermediate=False):
    """Bind an actual stage checkpoint to its declared learning producer."""
    fields = {'stage', 'stage_index', 'stage_updates', 'stage_fresh_transitions',
        'cumulative_successful_updates', 'cumulative_attempted_updates',
        'cumulative_collected_transitions', 'checkpoint', 'sidecar', 'learning_state',
        'episode_state_restored', 'history_reset'}
    segment_fields = {'format', 'schema_version', 'job_id', 'segment_plan_sha256'}
    is_segment = request['format'] == 'transformer_rl.exposure_segment_request'
    intermediate_fields = {'format', 'schema_version', 'job_id', 'job_plan_sha256'}
    expected_fields = fields | intermediate_fields if intermediate else fields | segment_fields if is_segment else fields
    _require(isinstance(endpoint, dict) and set(endpoint) == expected_fields,
             'parent stage endpoint schema differs from request')
    if intermediate:
        _require(endpoint['format'] == 'transformer_rl.exposure_learning_checkpoint'
                 and type(endpoint['schema_version']) is int and endpoint['schema_version'] == 1
                 and endpoint['job_id'] == plan['job_id']
                 and endpoint['job_plan_sha256'] == request['sha256'], 'learning checkpoint identity differs')
    elif is_segment:
        _require(endpoint['format'] == 'transformer_rl.exposure_segment_endpoint'
                 and type(endpoint['schema_version']) is int and endpoint['schema_version'] == 1
                 and endpoint['job_id'] == plan['job_id']
                 and endpoint['segment_plan_sha256'] == request['sha256'],
                 'parent stage endpoint identity differs')
    _require(isinstance(endpoint['stage'], str)
             and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,63}', endpoint['stage']), 'parent stage identity differs')
    for key in ('stage_index', 'cumulative_successful_updates', 'cumulative_attempted_updates',
                'cumulative_collected_transitions'):
        _integer(endpoint[key], 'parent stage '+key, 0)
    for key in ('stage_updates', 'stage_fresh_transitions'):
        _integer(endpoint[key], 'parent stage '+key)
    _require(endpoint['cumulative_attempted_updates'] == endpoint['cumulative_successful_updates']
             and endpoint['learning_state'] == 'full_actor_critic_Adam_global_private_RNG_clock'
             and endpoint['episode_state_restored'] is False and endpoint['history_reset'] == 'repeat_first',
             'parent stage must have a complete successful learning boundary')
    _require(endpoint_path.parent.name == f"stage_{endpoint['stage_index']:04d}_{endpoint['stage']}",
             'parent stage endpoint directory differs')
    actual_endpoint, endpoint_receipt = _read_input(endpoint_path)
    _require(actual_endpoint == endpoint, 'actual sealed stage endpoint differs from completion')
    checkpoint_path = _input_receipt(endpoint['checkpoint'])
    sidecar_path = _input_receipt(endpoint['sidecar'], require_bytes=False)
    checkpoint_name = f"checkpoint_{endpoint['stage_updates']:08d}.pt" if intermediate else 'endpoint.pt'
    _require(checkpoint_path == endpoint_path.parent/checkpoint_name
             and sidecar_path == Path(str(checkpoint_path)+'.json'), 'parent stage checkpoint/sidecar paths differ')
    with torch.device('cpu'):
        model, trainer, parent_config, update, metadata, rng = load_frame_checkpoint(checkpoint_path)
        expected_group = PPOTrainer(model, parent_config.ppo).optimizer.state_dict()['param_groups']
    optimizer = trainer.optimizer.state_dict()
    _require(optimizer['param_groups'] == expected_group,
             'parent actual Adam options differ from the frozen PPO recipe')
    _input_receipt(endpoint['checkpoint'])
    sidecar, actual_sidecar = _read_input(sidecar_path, canonical=False)
    _require(actual_sidecar['sha256'] == endpoint['sidecar']['sha256']
             and sidecar.get('format') == 'transformer_rl.packed_checkpoint'
             and type(sidecar.get('schema_version')) is int and sidecar['schema_version'] == 1
             and sidecar.get('sha256') == endpoint['checkpoint']['sha256']
             and sidecar.get('config') == parent_config.to_dict()
             and type(sidecar.get('update')) is int and sidecar['update'] == update
             and sidecar.get('metadata') == metadata,
             'parent sidecar differs from actual CPU learning checkpoint')
    _require(update == endpoint['cumulative_successful_updates'], 'parent actual checkpoint update differs')
    _require(plan['device'] == 'cpu' or bool(rng['cuda']), 'CUDA parent learning RNG state missing')
    _require(all(config.to_dict()[key] == parent_config.to_dict()[key] for key in ('model','ppo','control')),
             'only stage environment may change')
    fixed = metadata.get('fixed_exposure_job')
    _require(isinstance(fixed, dict) and fixed.get('job_id') == plan['job_id']
             and fixed.get('job_plan_sha256') == request['sha256']
             and fixed.get('stage') == endpoint['stage'] and fixed.get('stage_index') == endpoint['stage_index']
             and type(fixed.get('stage_index')) is int
             and fixed.get('initial_model_sha256') == plan['expected_initial_model_sha256'],
             'parent fixed exposure identity or original initial model guard differs')
    _require(metadata.get('initialization_guard') == {
        'expected_sha256':plan['expected_initial_model_sha256'],
        'actual_sha256':plan['expected_initial_model_sha256'],'verified':True},
        'parent original initialization guard differs')
    _require(metadata['initialization_guard']['verified'] is True
             and type(metadata.get('seed')) is int, 'parent original guard or seed types differ')
    _require(metadata.get('seed') == plan['training_seed']
             and metadata.get('environment_factory') == plan['environment_factory']
             and metadata.get('evaluation_seeds') == plan['evaluation_seeds']
             and metadata.get('source') == plan['source']
             and metadata.get('continuation_device') == plan['device'], 'parent learning producer metadata differs')
    state = metadata.get('continuation')
    _require(isinstance(state, dict) and set(state) == {'format','schema_version','clock','retention'}
             and state['format'] == 'transformer_rl.frame_continuation'
             and type(state['schema_version']) is int and state['schema_version'] == 1,
             'parent full continuation state required')
    clock = {'consumed_updates':endpoint['cumulative_attempted_updates'],
        'collected_transitions':endpoint['cumulative_collected_transitions'], 'rollout_steps':plan['rollout_steps']}
    _require(state.get('clock') == clock and all(type(v) is int for v in state['clock'].values())
             and type(metadata.get('collected_transitions')) is int
             and metadata['collected_transitions'] == clock['collected_transitions'], 'parent full clocks differ')
    _require(metadata.get('anchors') == [] and type(metadata.get('retention_coef')) in (int,float)
             and metadata['retention_coef'] == 0., 'parent must be a zero-retention exposure learner')
    sampler = PrivateAnchorRegularizer(model.actor, parent_config, (), 0., seed=plan['retention_seed'])
    sampler.load_state_dict(state['retention'])
    expected_segment = {'start_update':update-endpoint['stage_updates'],
        'successful_updates':endpoint['stage_updates'],'attempted_updates':endpoint['stage_updates'],
        'fresh_transitions':endpoint['stage_fresh_transitions'],'discarded_transitions':0}
    _require(metadata.get('continuation_segment') == expected_segment
             and all(type(v) is int for v in metadata['continuation_segment'].values()),
             'parent actual continuation segment differs')
    return {'config':parent_config, 'metadata':metadata, 'optimizer':optimizer,
        'proof_inputs':[(endpoint_path,endpoint_receipt), (checkpoint_path,deepcopy(endpoint['checkpoint'])),
                        (sidecar_path,actual_sidecar)]}


def _learning_transition(metadata, config, previous_endpoint=None, previous_config=None):
    if previous_endpoint is None:
        _require(metadata.get('continuation_parent') is None and 'stage_transition' not in metadata,
                 'fresh parent checkpoint has an unexpected continuation origin')
        return
    _require(metadata.get('continuation_parent') == {
        'path':previous_endpoint['checkpoint']['path'], 'sha256':previous_endpoint['checkpoint']['sha256'],
        'update':previous_endpoint['cumulative_successful_updates'], 'resume':True}
        and metadata['continuation_parent']['resume'] is True
        and type(metadata['continuation_parent']['update']) is int,
        'parent learning state does not resume its sealed ancestor')
    _require(metadata.get('stage_transition') == {
        'environment_transition':True, 'parent_environment_sha256':digest(previous_config.environment),
        'environment_sha256':digest(config.environment)}
        and metadata['stage_transition']['environment_transition'] is True,
        'parent environment transition does not match sealed ancestor')


def _adam_step_proof(optimizer, expected_steps):
    """Every packed actor/critic parameter participates in each applied PPO step."""
    _integer(expected_steps, 'verified cumulative PPO optimizer steps', 0)
    states, ids = optimizer['state'], optimizer['param_groups'][0]['params']
    _require(set(states) == (set(ids) if expected_steps else set()),
             'parent actual Adam parameter state does not match verified PPO steps')
    _require(all(int(state['step'].item()) == expected_steps for state in states.values()),
             'parent actual Adam step differs from verified cumulative PPO optimizer steps')


def _sealed_checkpoint_records(request, completion, root, stage_offset, first_update, *,
                               first_samples=0, require_complete=False):
    """Check the complete declared snapshot inventory without loading its model payloads."""
    expected, update, samples = [], first_update, first_samples
    for index, stage in enumerate(request['stages']):
        for local in _checkpoint_updates(stage):
            expected.append((stage_offset+index, local, update+local,
                             samples+local*stage['transitions_per_update'], stage))
        update += stage['updates']
        samples += stage['fresh_transition_budget']
    records = completion.get('sealed_checkpoints')
    _require(type(records) is list and len(records) <= len(expected), 'sealed checkpoint inventory required')
    result = []
    for item, (index, local, cumulative, cumulative_samples, stage) in zip(records, expected):
        _require(type(item) is dict and set(item) == {
            'stage_index','local_update','checkpoint_update','kind','record','metrics_prefix'},
            'sealed checkpoint inventory fields differ')
        _require(all(type(item.get(k)) is int for k in ('stage_index','local_update','checkpoint_update'))
                 and (item['stage_index'], item['local_update'], item['checkpoint_update']) == (index,local,cumulative),
                 'sealed checkpoint sequence differs from the declared schedule')
        final = local == stage['updates']
        _require(item['kind'] == ('endpoint' if final else 'intermediate'), 'checkpoint kind differs')
        name = 'endpoint.json' if final else f'checkpoint_{local:08d}.json'
        path = _input_receipt(item['record'])
        _require(path == root/f'stage_{index:04d}_{stage["name"]}'/name, 'sealed checkpoint path differs')
        record, actual = _read_input(path)
        _require(actual == item['record'] and record['stage_index'] == index and record['stage'] == stage['name']
                 and record['stage_updates'] == local and record['cumulative_successful_updates'] == cumulative
                 and record['cumulative_attempted_updates'] == cumulative
                 and record['stage_fresh_transitions'] == local*stage['transitions_per_update']
                 and record['cumulative_collected_transitions'] == cumulative_samples,
                 'sealed checkpoint record differs from inventory')
        checkpoint_path = _input_receipt(record['checkpoint'])
        sidecar_path = _input_receipt(record['sidecar'], require_bytes=False)
        expected_checkpoint = path.with_suffix('.pt') if not final else path.parent/'endpoint.pt'
        _require(checkpoint_path == expected_checkpoint and sidecar_path == Path(str(expected_checkpoint)+'.json'),
                 'sealed checkpoint payload or sidecar path differs')
        prefix = item['metrics_prefix']
        _require(type(prefix) is dict and set(prefix) == {
            'path','sha256','bytes','rows','optimizer_steps','optimization_sample_uses'}
            and prefix['path'] == str(root/'metrics.jsonl')
            and isinstance(prefix['sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',prefix['sha256']),
            'sealed checkpoint metric prefix fields differ')
        for key in ('bytes','rows','optimizer_steps','optimization_sample_uses'):
            _integer(prefix[key], 'sealed checkpoint metric prefix '+key, 0 if key in ('optimizer_steps','optimization_sample_uses') else 1)
        _require(prefix['rows'] == cumulative-first_update, 'sealed metric prefix update count differs')
        result.append((item, record, path))
    missing = [{'stage_index': index, 'local_update': local} for index,local,_,_,_ in expected[len(records):]]
    _require(completion.get('missing_checkpoints') == missing
             and completion.get('last_sealed_learning_checkpoint') == (records[-1] if records else None),
             'checkpoint completion inventory or missing denominator differs')
    if require_complete:
        _require(not missing, 'completed stage is missing a declared learning checkpoint')
    return result


def _verify_metric_prefixes(path, records):
    """Bind all inventory prefixes in one streamed pass over immutable worker metrics."""
    pending = {item['metrics_prefix']['bytes']: item['metrics_prefix'] for item,_,_ in records}
    _require(len(pending) == len(records), 'checkpoint metric prefixes repeat a boundary')
    sha, count, length, steps, uses = hashlib.sha256(), 0, 0, 0, 0
    with path.open('rb') as stream:
        for raw in stream:
            if not pending:
                break
            row = json.loads(raw)
            _require(raw == json_bytes(row)+b'\n', 'checkpoint prefix metric row is incomplete')
            sha.update(raw)
            length += len(raw)
            count += 1
            steps += row['optimization']['optimizer_steps']
            uses += row['optimization']['sample_count']
            if length in pending:
                prefix = pending.pop(length)
                _require(prefix['sha256'] == sha.hexdigest() and prefix['rows'] == count
                         and prefix['optimizer_steps'] == steps and prefix['optimization_sample_uses'] == uses,
                         'checkpoint metric prefix differs from actual rows')
            _require(not pending or length < min(pending), 'checkpoint prefix is not a complete metric-row boundary')
    _require(not pending, 'checkpoint metric prefix exceeds the actual recorded metrics')


def _segment_parent(receipt, plan, config):
    """CPU-only proof of a completed, logged parent before any new output/env."""
    endpoint_path = _input_receipt(receipt)
    _require(endpoint_path.name == 'endpoint.json', 'sealed endpoint receipt required')
    endpoint, actual_receipt = _read_input(endpoint_path)
    _require(actual_receipt == receipt, 'endpoint actual receipt differs')
    fields = {'stage', 'stage_index', 'stage_updates', 'stage_fresh_transitions',
        'cumulative_successful_updates', 'cumulative_attempted_updates',
        'cumulative_collected_transitions', 'checkpoint', 'sidecar', 'learning_state',
        'episode_state_restored', 'history_reset'}
    segment_fields = {'format', 'schema_version', 'job_id', 'segment_plan_sha256'}
    _require(isinstance(endpoint, dict) and set(endpoint) in (fields, fields | segment_fields),
             'unsupported exposure endpoint schema')
    if 'format' in endpoint:
        _require(endpoint['format'] == 'transformer_rl.exposure_segment_endpoint'
                 and type(endpoint['schema_version']) is int and endpoint['schema_version'] == 1
                 and endpoint['job_id'] == plan['job_id'], 'segment endpoint identity differs')
    _require(isinstance(endpoint['stage'], str)
             and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,63}', endpoint['stage']), 'parent stage identity differs')
    for key in ('stage_index', 'cumulative_successful_updates', 'cumulative_attempted_updates',
                'cumulative_collected_transitions'):
        _integer(endpoint[key], 'parent '+key, 0)
    for key in ('stage_updates', 'stage_fresh_transitions'):
        _integer(endpoint[key], 'parent '+key)
    _require(endpoint['cumulative_successful_updates'] >= endpoint['stage_updates']
             and endpoint['cumulative_attempted_updates'] == endpoint['cumulative_successful_updates']
             and endpoint['cumulative_collected_transitions'] >= endpoint['stage_fresh_transitions'],
             'parent successful boundary clocks differ')
    _require(endpoint['learning_state'] == 'full_actor_critic_Adam_global_private_RNG_clock'
             and endpoint['episode_state_restored'] is False and endpoint['history_reset'] == 'repeat_first',
             'complete parent learning state required')
    _require(endpoint_path.parent.name == f"stage_{endpoint['stage_index']:04d}_{endpoint['stage']}",
             'parent stage directory differs')
    checkpoint_path = _input_receipt(endpoint['checkpoint'])
    sidecar_path = _input_receipt(endpoint['sidecar'], require_bytes=False)
    _require(checkpoint_path == endpoint_path.parent/'endpoint.pt'
             and sidecar_path == Path(str(checkpoint_path)+'.json'), 'parent checkpoint/sidecar paths differ')
    parent_root = endpoint_path.parent.parent
    request, request_receipt = _read_input(parent_root/'request.json')
    reservation, reservation_receipt = _read_input(parent_root/'reservation.json')
    completion, completion_receipt = _read_input(parent_root/'completion.json')
    _require(request.get('format') in ('transformer_rl.fixed_exposure_job', 'transformer_rl.exposure_segment_request')
             and type(request.get('schema_version')) is int and request['schema_version'] in (1, 2),
             'parent request schema differs')
    _require(completion.get('format') in ('transformer_rl.fixed_exposure_completion', 'transformer_rl.exposure_segment_completion')
             and type(completion.get('schema_version')) is int and completion['schema_version'] == request['schema_version'],
             'parent completion schema differs')
    expected_completion_format = ('transformer_rl.exposure_segment_completion'
        if request['format'] == 'transformer_rl.exposure_segment_request' else 'transformer_rl.fixed_exposure_completion')
    _require(completion['format'] == expected_completion_format, 'parent request/completion formats differ')
    request_fields = set(plan)
    if request['format'] == 'transformer_rl.exposure_segment_request':
        request_fields |= {'stage_index','parent_endpoint','parent_request','parent_reservation',
                           'parent_completion','parent_metrics','charge_scope','whole_job_reservation_created'}
    _require(set(request) == request_fields, 'parent request fields differ from schema')
    for key in ('reserved_updates','reserved_fresh_transitions','training_seed','retention_seed','rollout_steps'):
        _integer(request.get(key), 'parent request '+key, 0 if key.endswith('seed') else 1)
    _require(request.get('sha256') == digest({k:v for k,v in request.items() if k != 'sha256'})
             and completion.get('job_plan_sha256') == request['sha256'], 'parent request/completion binding differs')
    _require(completion.get('status') == 'completed' and completion.get('error') is None
             and completion.get('shutdown_errors') == [] and completion.get('stop_reason') is None
             and completion.get('missing_stage_endpoints') == []
             and completion.get('unsealed_checkpoint_paths') == [], 'parent completion is not a clean successful boundary')
    endpoints = completion.get('endpoints')
    _require(isinstance(endpoints, list) and endpoints and endpoints[-1] == endpoint
             and sum(item == endpoint for item in endpoints) == 1
             and completion.get('last_sealed_checkpoint') == endpoint['checkpoint'],
             'parent endpoint is not the unique last sealed completion endpoint')
    for key in ('job_id', 'training_seed', 'retention_seed', 'evaluation_seeds', 'rollout_steps',
                'device', 'environment_factory', 'expected_initial_model_sha256', 'source',
                'retention_coefficient','retry_policy','stage_rule','evaluation_rule'):
        _require(request.get(key) == plan[key], 'parent '+key+' differs from segment')
    _require(completion.get('job_id') == plan['job_id'] and completion.get('source') == plan['source'],
             'parent completion producer differs')
    _require(reservation.get('job_plan_sha256') == request['sha256']
             and reservation.get('charged_updates') == request['reserved_updates']
             and reservation.get('charged_fresh_transition_budget') == request['reserved_fresh_transitions']
             and reservation.get('refund') is False and completion.get('refund') is False
             and completion.get('automatic_retries') == 0
             and completion.get('independent_evaluation_performed') is False,
             'parent actual reservation or no-retry policy differs')
    if request['format'] == 'transformer_rl.exposure_segment_request':
        _require(reservation.get('format') == 'transformer_rl.exposure_segment_reservation'
                 and type(reservation.get('schema_version')) is int and reservation['schema_version'] == 1
                 and reservation.get('charge_scope') == request.get('charge_scope')
                 == completion.get('charge_scope') == 'segment_itemization_of_external_whole_job_reservation'
                 and reservation.get('whole_job_reservation_created') is False
                 and request.get('whole_job_reservation_created') is False
                 and completion.get('whole_job_reservation_created') is False,
                 'parent segment reservation scope differs')
    for key in ('reserved_updates', 'charged_updates', 'successful_updates', 'attempted_updates',
                'recorded_metric_updates'):
        _integer(completion.get(key), 'parent completion '+key)
        _require(completion[key] == request['reserved_updates'], 'parent full successful update budget differs')
    for key in ('charged_fresh_transition_budget', 'actual_collected_transitions',
                'successful_full_rollout_samples', 'recorded_full_rollout_samples'):
        _integer(completion.get(key), 'parent completion '+key)
        _require(completion[key] == request['reserved_fresh_transitions'], 'parent full sample budget differs')
    _require(completion.get('failed_update_optimizer_steps') == 0
             and completion.get('optimizer_update_may_be_partial') is False
             and completion.get('unverified_or_unoptimized_samples') == 0
             and completion.get('unsealed_successful_updates') == 0, 'parent incomplete learning accounting')
    for key in ('failed_update_optimizer_steps','unverified_or_unoptimized_samples','unsealed_successful_updates'):
        _integer(completion.get(key), 'parent completion '+key, 0)
    learning = _checkpoint_learning_proof(endpoint, endpoint_path, plan, request, config)
    parent_config, metadata = learning['config'], learning['metadata']
    update = endpoint['cumulative_successful_updates']
    proof_inputs = list(learning['proof_inputs'])
    parent_stages = request.get('stages')
    _require(isinstance(parent_stages, list) and parent_stages
             and all(isinstance(stage,dict) for stage in parent_stages), 'parent stage request missing')
    _require(len(endpoints) == len(parent_stages), 'parent completion stage endpoint count differs')
    declared = parent_stages[-1]
    _require(declared.get('name') == endpoint['stage'] and declared.get('config') == parent_config.to_dict()
             and declared.get('updates') == endpoint['stage_updates']
             and declared.get('fresh_transition_budget') == endpoint['stage_fresh_transitions']
             and declared.get('transitions_per_update') == plan['rollout_steps']*parent_config.environment['num_envs'],
             'parent endpoint stage budget or configuration differs')
    if request['format'] == 'transformer_rl.exposure_segment_request':
        _require(type(request.get('stage_index')) is int
                 and request['stage_index'] == endpoint['stage_index']
                 and type(completion.get('stage_index')) is int
                 and completion['stage_index'] == endpoint['stage_index']
                 and len(parent_stages) == 1 and endpoint.get('segment_plan_sha256') == request['sha256'],
                 'parent segment stage binding differs')
        for endpoint_key, completion_key in (
            ('cumulative_successful_updates','cumulative_successful_updates'),
            ('cumulative_attempted_updates','cumulative_attempted_updates'),
            ('cumulative_collected_transitions','cumulative_collected_transitions')):
            _integer(completion.get(completion_key), 'parent segment '+completion_key)
            _require(completion[completion_key] == endpoint[endpoint_key], 'parent segment cumulative completion differs')
    else:
        _require(len(parent_stages)-1 == endpoint['stage_index'], 'parent job last stage index differs')
    stage_offset = endpoint['stage_index']-len(parent_stages)+1
    first_update = update-request['reserved_updates']
    first_samples = endpoint['cumulative_collected_transitions']-request['reserved_fresh_transitions']
    _require(stage_offset >= 0 and first_update >= 0 and first_samples >= 0,
             'parent stage or cumulative budget baseline differs')
    if request['format'] == 'transformer_rl.fixed_exposure_job':
        _require(first_update == first_samples == stage_offset == 0, 'fresh parent job has nonzero baseline')
    protected_trees = [str(parent_root)]
    prior_optimizer_steps = 0
    if request['format'] == 'transformer_rl.exposure_segment_request':
        previous_receipt = request.get('parent_endpoint')
        _require(completion.get('parent_endpoint') == previous_receipt,
                 'parent segment request/completion input binding differs')
        if endpoint['stage_index'] == 0:
            _require(all(request.get(key) is None for key in (
                'parent_endpoint','parent_request','parent_reservation','parent_completion','parent_metrics'))
                and first_update == first_samples == 0, 'fresh segment has a continuation baseline')
            _learning_transition(metadata, parent_config)
        else:
            previous_path = _input_receipt(previous_receipt)
            previous_endpoint, actual_previous = _read_input(previous_path)
            _require(actual_previous == previous_receipt
                     and type(previous_endpoint.get('stage_index')) is int
                     and previous_endpoint['stage_index'] == endpoint['stage_index']-1,
                     'parent stage chain must decrease by exactly one')
            previous = _segment_parent(previous_receipt, plan, parent_config)
            for key in ('request','reservation','completion','metrics'):
                _require(request.get('parent_'+key) == previous[key],
                         'parent segment ancestor '+key+' receipt differs')
            previous_endpoint = previous['endpoint']
            _require(first_update == previous_endpoint['cumulative_successful_updates']
                     and first_update == previous_endpoint['cumulative_attempted_updates']
                     and first_samples == previous_endpoint['cumulative_collected_transitions'],
                     'parent segment baseline differs from sealed ancestor')
            _learning_transition(metadata, parent_config, previous_endpoint, previous['config'])
            prior_optimizer_steps = previous['cumulative_optimizer_steps']
            proof_inputs.extend(previous['proof_inputs'])
            protected_trees.extend(previous['protected_trees'])
    schedule, names, stage_learning = [], set(), {}
    next_update, next_samples = first_update, first_samples
    for local_index, stage in enumerate(parent_stages):
        _require(isinstance(stage,dict) and set(stage) in ({
            'name','config','updates','transitions_per_update','fresh_transition_budget'}, {
            'name','config','updates','transitions_per_update','fresh_transition_budget','checkpoint_updates'}),
            'parent declared stage fields differ')
        _checkpoint_updates(stage)
        _require(isinstance(stage['name'],str)
                 and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,63}',stage['name'])
                 and stage['name'] not in names, 'parent declared stage names differ')
        names.add(stage['name'])
        previous_config = FrameTrainConfig.from_dict(stage['config'])
        _integer(stage.get('updates'), 'parent declared stage updates')
        _integer(stage.get('transitions_per_update'), 'parent declared transitions_per_update')
        _integer(stage.get('fresh_transition_budget'), 'parent declared fresh_transition_budget')
        _integer(previous_config.environment.get('num_envs'), 'parent declared num_envs')
        per_update = plan['rollout_steps']*previous_config.environment['num_envs']
        _require(all(previous_config.to_dict()[key] == parent_config.to_dict()[key]
                     for key in ('model','ppo','control'))
                 and stage.get('transitions_per_update') == per_update
                 and stage.get('fresh_transition_budget') == stage['updates']*per_update,
                 'parent declared stage configuration or budget differs')
        schedule.append((stage['name'],stage_offset+local_index,next_update,next_samples,per_update,stage['updates']))
        next_update += stage['updates']
        next_samples += stage['updates']*per_update
        current_endpoint = endpoints[local_index]
        _require(isinstance(current_endpoint,dict)
                 and current_endpoint.get('stage') == stage['name']
                 and type(current_endpoint.get('stage_index')) is int
                 and current_endpoint['stage_index'] == stage_offset+local_index
                 and type(current_endpoint.get('stage_updates')) is int
                 and current_endpoint['stage_updates'] == stage['updates']
                 and type(current_endpoint.get('stage_fresh_transitions')) is int
                 and current_endpoint['stage_fresh_transitions'] == stage['fresh_transition_budget']
                 and current_endpoint.get('cumulative_successful_updates') == next_update
                 and current_endpoint.get('cumulative_attempted_updates') == next_update
                 and current_endpoint.get('cumulative_collected_transitions') == next_samples,
                 'parent actual stage endpoint sequence differs from declared exposure')
        if request['format'] == 'transformer_rl.fixed_exposure_job':
            path = parent_root/f'stage_{local_index:04d}_{stage["name"]}'/'endpoint.json'
            current_learning = (learning if local_index == len(parent_stages)-1 else
                _checkpoint_learning_proof(current_endpoint,path,plan,request,previous_config))
            _require(current_learning['config'].to_dict() == stage['config'],
                     'parent actual stage checkpoint configuration differs from request')
            before_endpoint = endpoints[local_index-1] if local_index else None
            before_config = stage_learning[local_index-1]['config'] if local_index else None
            _learning_transition(current_learning['metadata'],current_learning['config'],before_endpoint,before_config)
            proof_inputs.extend(current_learning['proof_inputs'])
            stage_learning[local_index] = current_learning
        else:
            stage_learning[stage_offset+local_index] = learning
    expected_count = sum(item[-1] for item in schedule)
    expected_rows = ((name,index,start_update+count,start_samples+count*per_update,per_update)
        for name,index,start_update,start_samples,per_update,updates in schedule
        for count in range(1,updates+1))
    _require(expected_count == request['reserved_updates']
             and next_update == update and next_samples == endpoint['cumulative_collected_transitions'],
             'parent requested full budget differs from cumulative endpoint')
    metrics_path = _input_path(str(parent_root/'metrics.jsonl'))
    metrics_before = metrics_path.stat()
    metric_sha, all_rows, matching_count, all_samples, all_steps, all_uses = hashlib.sha256(), 0, 0, 0, 0, 0
    with metrics_path.open('rb') as stream:
        for raw in stream:
            metric_sha.update(raw)
            row = json.loads(raw)
            _require(raw == json_bytes(row)+b'\n', 'parent metrics row is not canonical or complete')
            for key in ('batch_samples','update','consumed_updates','cumulative_transitions','stage_index'):
                _integer(row.get(key), 'parent metric '+key, 0 if key == 'stage_index' else 1)
            _require(all_rows < expected_count, 'parent unexpected extra metrics row')
            stage_name, stage_index, row_update, row_samples, per_update = next(expected_rows)
            _require(row.get('stage') == stage_name and row['stage_index'] == stage_index
                     and row['update'] == row['consumed_updates'] == row_update
                     and row['cumulative_transitions'] == row_samples
                     and row['batch_samples'] == row['collection']['transitions'] == per_update
                     and type(row['collection']['transitions']) is int
                     and type(row['collection']['vector_steps']) is int
                     and row['collection']['vector_steps'] == plan['rollout_steps']
                     and row['collection'].get('early_stopped') is False,
                     'parent complete rollout sequence or stage clock differs')
            for key in ('optimizer_steps','planned_optimizer_steps','sample_count'):
                _integer(row['optimization'].get(key), 'parent optimization '+key, 0)
            chunks = min(parent_config.ppo.num_minibatches,per_update)
            steps = row['optimization']['optimizer_steps']
            full_epochs, remainder = divmod(steps,chunks)
            expected_uses = (full_epochs*per_update + remainder*(per_update//chunks)
                             + min(remainder,per_update%chunks))
            _require(row['optimization']['planned_optimizer_steps'] == parent_config.ppo.epochs*chunks
                     and steps <= row['optimization']['planned_optimizer_steps']
                     and row['optimization']['sample_count'] == expected_uses
                     and type(row['optimization'].get('early_stopped')) is bool
                     and row['optimization']['early_stopped'] == (steps < row['optimization']['planned_optimizer_steps']),
                     'parent optimization accounting differs from actual frozen minibatch prefix')
            all_rows += 1
            all_samples += row['batch_samples']
            all_steps += row['optimization']['optimizer_steps']
            all_uses += row['optimization']['sample_count']
            if row['stage_index'] == endpoint['stage_index']:
                matching_count += 1
            if row['update'] == endpoints[stage_index-stage_offset]['cumulative_successful_updates']:
                _adam_step_proof(stage_learning[stage_index]['optimizer'],prior_optimizer_steps+all_steps)
    metric_receipt = {'path':str(metrics_path),'sha256':metric_sha.hexdigest(),'bytes':metrics_before.st_size}
    metrics_after = metrics_path.stat()
    _require(_sha(metrics_path) == metric_receipt['sha256']
             and (metrics_before.st_dev,metrics_before.st_ino,metrics_before.st_size,metrics_before.st_mtime_ns)
             == (metrics_after.st_dev,metrics_after.st_ino,metrics_after.st_size,metrics_after.st_mtime_ns),
             'parent metrics changed while verifying')
    _require(all_rows == completion['recorded_metric_updates'] and all_samples == completion['recorded_full_rollout_samples']
             and all_steps == completion.get('optimizer_steps') and all_uses == completion.get('optimization_sample_uses'),
             'parent actual metrics differ from completion accounting')
    for key in ('optimizer_steps','optimization_sample_uses','automatic_retries'):
        _integer(completion.get(key), 'parent completion '+key, 0)
    _require(matching_count == endpoint['stage_updates'], 'parent stage complete metrics count differs')
    if request['schema_version'] == 2:
        checkpoint_records = _sealed_checkpoint_records(request, completion, parent_root, stage_offset,
                                                        first_update, first_samples=first_samples, require_complete=True)
        _verify_metric_prefixes(metrics_path, checkpoint_records)
    proof_inputs.extend(((parent_root/'request.json',request_receipt),
                         (parent_root/'reservation.json',reservation_receipt),
                         (parent_root/'completion.json',completion_receipt), (metrics_path,metric_receipt)))
    for path, identity in proof_inputs:
        _require(_input_path(str(path)).stat().st_size == identity['bytes'], 'parent inputs size changed after preflight')
        _require(_sha(path) == identity['sha256'], 'parent inputs changed after preflight')
    return {'endpoint':endpoint,'receipt':actual_receipt,'request':request_receipt,'reservation':reservation_receipt,
        'completion':completion_receipt,'metrics':metric_receipt,'config':parent_config,
        'protected_trees':tuple(protected_trees), 'proof_inputs':proof_inputs,
        'cumulative_optimizer_steps':prior_optimizer_steps+all_steps}


def verify_exposure_checkpoint(receipt, plan, config):
    """CPU proof of a saved observation point, including a subsequently failed stage.

    Intermediate observation points authorize offline evaluation only. They do
    not authorize a restarted learner or turn partial exposure into completion.
    """
    path = _input_receipt(receipt)
    record, actual_receipt = _read_input(path)
    _require(actual_receipt == receipt, 'learning checkpoint receipt differs')
    if path.name == 'endpoint.json':
        proof = _segment_parent(receipt, plan, config)
        completion, _ = _read_input(Path(proof['completion']['path']))
        item = next((item for item in completion.get('sealed_checkpoints', []) if item['record'] == receipt), None)
        return {**proof, 'checkpoint': deepcopy(proof['endpoint']['checkpoint']),
                'metrics_prefix': deepcopy(item['metrics_prefix']) if item is not None else None}
    _require(record.get('format') == 'transformer_rl.exposure_learning_checkpoint',
             'intermediate learning checkpoint record required')
    root = path.parent.parent
    request, request_receipt = _read_input(root/'request.json')
    reservation, reservation_receipt = _read_input(root/'reservation.json')
    completion, completion_receipt = _read_input(root/'completion.json')
    _require(request.get('format') in ('transformer_rl.fixed_exposure_job','transformer_rl.exposure_segment_request')
             and request.get('schema_version') == 2 and type(request['schema_version']) is int
             and request.get('sha256') == digest({k:v for k,v in request.items() if k != 'sha256'}),
             'scheduled learning checkpoint request required')
    extra = {'stage_index','parent_endpoint','parent_request','parent_reservation','parent_completion',
             'parent_metrics','charge_scope','whole_job_reservation_created'} if request['format'].endswith('segment_request') else set()
    _require(set(request) == set(plan)|extra and all(request.get(k) == v for k,v in plan.items()
             if k not in ('format','sha256')), 'learning checkpoint request differs from authorized plan')
    _require(completion.get('format') == ('transformer_rl.exposure_segment_completion' if extra
             else 'transformer_rl.fixed_exposure_completion') and completion.get('schema_version') == 2
             and completion.get('job_plan_sha256') == request['sha256']
             and completion.get('job_id') == plan['job_id'] and completion.get('source') == plan['source']
             and completion.get('status') in ('completed','failed','interrupted','partial_rollout','user_interrupted'),
             'learning checkpoint completion producer or state differs')
    _require(reservation.get('job_plan_sha256') == request['sha256']
             and reservation.get('charged_updates') == request['reserved_updates']
             and reservation.get('charged_fresh_transition_budget') == request['reserved_fresh_transitions']
             and reservation.get('refund') is False and completion.get('refund') is False
             and completion.get('automatic_retries') == 0 and completion.get('independent_evaluation_performed') is False,
             'learning checkpoint reservation differs')
    if extra:
        _require(reservation.get('format') == 'transformer_rl.exposure_segment_reservation'
                 and reservation.get('schema_version') == 1
                 and reservation.get('charge_scope') == request.get('charge_scope') == completion.get('charge_scope')
                 == 'segment_itemization_of_external_whole_job_reservation'
                 and all(value.get('whole_job_reservation_created') is False for value in (request,reservation,completion)),
                 'learning checkpoint segment charge differs')
    prior, prior_steps, first_update, first_samples = None, 0, 0, 0
    offset = request.get('stage_index', 0)
    if extra:
        _integer(offset, 'learning checkpoint stage index', 0)
        _require(len(request['stages']) == 1 and completion.get('stage_index') == offset,
                 'learning checkpoint segment stage membership differs')
        previous_receipt = request.get('parent_endpoint')
        _require(completion.get('parent_endpoint') == previous_receipt, 'learning checkpoint parent differs')
        if offset:
            prior = _segment_parent(previous_receipt, plan, config)
            _require(prior['endpoint']['stage_index'] == offset-1, 'learning checkpoint parent skips a stage')
            for key in ('request','reservation','completion','metrics'):
                _require(request.get('parent_'+key) == prior[key], 'learning checkpoint ancestor receipt differs')
            first_update = prior['endpoint']['cumulative_successful_updates']
            first_samples = prior['endpoint']['cumulative_collected_transitions']
            prior_steps = prior['cumulative_optimizer_steps']
        else:
            _require(all(request.get(key) is None for key in ('parent_endpoint','parent_request','parent_reservation',
                     'parent_completion','parent_metrics')), 'fresh checkpoint has an unexpected parent')
    records = _sealed_checkpoint_records(request, completion, root, offset, first_update,
                                        first_samples=first_samples,
                                        require_complete=completion['status'] == 'completed')
    selected = next((item for item in records if item[0]['record'] == receipt), None)
    _require(selected is not None and selected[0]['kind'] == 'intermediate', 'checkpoint is not uniquely sealed')
    item = selected[0]
    local_index = item['stage_index']-offset
    stage = request['stages'][local_index]
    _require(config.to_dict() == stage['config'] and item['local_update'] < stage['updates'],
             'intermediate checkpoint changes the frozen stage config or endpoint')
    learning = _checkpoint_learning_proof(record, path, plan, request, config, intermediate=True)
    _require(learning['config'].to_dict() == stage['config'], 'checkpoint actual config differs from stage')
    previous_endpoint, previous_config = (prior['endpoint'],prior['config']) if prior else (None,None)
    earlier_inputs = []
    for index in range(local_index):
        earlier = completion['endpoints'][index]
        earlier_path = root/f'stage_{index:04d}_{request["stages"][index]["name"]}'/'endpoint.json'
        earlier_config = FrameTrainConfig.from_dict(request['stages'][index]['config'])
        earlier_learning = _checkpoint_learning_proof(earlier, earlier_path, plan, request, earlier_config)
        _learning_transition(earlier_learning['metadata'], earlier_config, previous_endpoint, previous_config)
        earlier_item = next(item for item,_,_ in records if item['stage_index'] == index and item['kind'] == 'endpoint')
        _adam_step_proof(earlier_learning['optimizer'],prior_steps+earlier_item['metrics_prefix']['optimizer_steps'])
        earlier_inputs.extend(earlier_learning['proof_inputs'])
        previous_endpoint, previous_config = earlier, earlier_config
    _learning_transition(learning['metadata'], learning['config'], previous_endpoint, previous_config)
    metrics_path = _input_path(str(root/'metrics.jsonl'))
    before = metrics_path.stat()
    metrics_receipt = {'path':str(metrics_path),'sha256':_sha(metrics_path),'bytes':before.st_size}
    _verify_metric_prefixes(metrics_path, records)
    rows, samples, steps, uses = 0, first_samples, 0, 0
    expected_rows = ((stage['name'], offset+index, start_update+local, start_samples+local*stage['transitions_per_update'],
                      stage['transitions_per_update'])
        for index,stage,start_update,start_samples in _stage_prefix_schedule(request, first_update, first_samples)
        for local in range(1,stage['updates']+1))
    with metrics_path.open('rb') as stream:
        while rows < item['metrics_prefix']['rows']:
            raw = stream.readline()
            row = json.loads(raw)
            name,index,update,expected_samples,per_update = next(expected_rows)
            _require(raw == json_bytes(row)+b'\n' and row.get('stage') == name
                     and type(row.get('stage_index')) is int and row['stage_index'] == index
                     and all(type(row.get(k)) is int for k in ('update','consumed_updates','cumulative_transitions','batch_samples'))
                     and row['update'] == row['consumed_updates'] == update
                     and row['cumulative_transitions'] == expected_samples and row['batch_samples'] == per_update
                     and row['collection'].get('transitions') == per_update
                     and type(row['collection'].get('transitions')) is int
                     and row['collection'].get('vector_steps') == plan['rollout_steps']
                     and type(row['collection'].get('vector_steps')) is int
                     and row['collection'].get('early_stopped') is False, 'checkpoint complete rollout prefix differs')
            optimization = row['optimization']
            for key in ('optimizer_steps','planned_optimizer_steps','sample_count'):
                _integer(optimization.get(key), 'checkpoint optimizer '+key, 0)
            chunks = min(config.ppo.num_minibatches,per_update)
            epochs,remainder = divmod(optimization['optimizer_steps'],chunks)
            sample_uses = epochs*per_update + remainder*(per_update//chunks) + min(remainder,per_update%chunks)
            _require(optimization['planned_optimizer_steps'] == config.ppo.epochs*chunks
                     and optimization['optimizer_steps'] <= optimization['planned_optimizer_steps']
                     and optimization['sample_count'] == sample_uses
                     and type(optimization.get('early_stopped')) is bool
                     and optimization['early_stopped'] == (optimization['optimizer_steps'] < optimization['planned_optimizer_steps']),
                     'checkpoint frozen optimizer minibatch prefix differs')
            rows += 1
            samples += per_update
            steps += optimization['optimizer_steps']
            uses += optimization['sample_count']
    _require(record['cumulative_successful_updates'] == first_update+rows
             and record['cumulative_attempted_updates'] == first_update+rows
             and record['cumulative_collected_transitions'] == samples
             and item['metrics_prefix']['optimizer_steps'] == steps
             and item['metrics_prefix']['optimization_sample_uses'] == uses, 'checkpoint measured clocks differ')
    _adam_step_proof(learning['optimizer'], prior_steps+steps)
    for key, lower, upper in (('successful_updates',rows,request['reserved_updates']),
                             ('attempted_updates',rows,request['reserved_updates']),
                             ('recorded_metric_updates',rows,request['reserved_updates']),
                             ('actual_collected_transitions',samples-first_samples,request['reserved_fresh_transitions'])):
        _integer(completion.get(key), 'checkpoint completion '+key, 0)
        _require(lower <= completion[key] <= upper, 'checkpoint completion contradicts its sealed prefix')
    inputs = [*learning['proof_inputs'], *earlier_inputs, (root/'request.json',request_receipt),
              (root/'reservation.json',reservation_receipt), (root/'completion.json',completion_receipt),
              (metrics_path,metrics_receipt)]
    if prior:
        inputs.extend(prior['proof_inputs'])
    for checked, pin in inputs:
        _require(checked.stat().st_size == pin['bytes'] and _sha(checked) == pin['sha256'],
                 'learning checkpoint proof input changed')
    after = metrics_path.stat()
    _require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
             == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'learning checkpoint metrics changed')
    return {'endpoint':record,'receipt':actual_receipt,'checkpoint':deepcopy(record['checkpoint']),
        'config':learning['config'],'completion':completion_receipt,'request':request_receipt,
        'reservation':reservation_receipt,'metrics':metrics_receipt,'metrics_prefix':deepcopy(item['metrics_prefix']),
        'cumulative_optimizer_steps':prior_steps+steps,'proof_inputs':inputs}


def _stage_prefix_schedule(request, first_update, first_samples):
    update, samples = first_update, first_samples
    for index, stage in enumerate(request['stages']):
        yield index, stage, update, samples
        update += stage['updates']
        samples += stage['fresh_transition_budget']


def train_exposure_segment(stage, env_factory, environment_reference, output_root, *,
                           job_id, rollout_steps, training_seed, retention_seed,
                           evaluation_seeds, device, expected_initial_model_sha256,
                           max_seconds, parent_endpoint=None, should_stop=None, protected_paths=()):
    """Train one stage in a fresh OS worker; resume a sealed completed parent.

    An external controller reserves the whole job once. This layer writes only
    a segment itemization, with local and cumulative actual learning counters.
    Parent verification is CPU-only and precedes output/environment construction.
    """
    _require(callable(env_factory) and (should_stop is None or callable(should_stop)),
             'environment and optional stop callbacks must be callable')
    plan, configs = _definition([stage], job_id=job_id, rollout_steps=rollout_steps,
        training_seed=training_seed, retention_seed=retention_seed, evaluation_seeds=evaluation_seeds,
        device=device, expected_initial_model_sha256=expected_initial_model_sha256,
        max_seconds=max_seconds, environment_reference=environment_reference)
    parent = _segment_parent(parent_endpoint, plan, configs[0]) if parent_endpoint is not None else None
    value = {k:v for k,v in plan.items() if k != 'sha256'}
    value.update(format='transformer_rl.exposure_segment_request',
        stage_index=parent['endpoint']['stage_index']+1 if parent else 0,
        parent_endpoint=deepcopy(parent['receipt']) if parent else None,
        parent_completion=deepcopy(parent['completion']) if parent else None,
        parent_request=deepcopy(parent['request']) if parent else None,
        parent_reservation=deepcopy(parent['reservation']) if parent else None,
        parent_metrics=deepcopy(parent['metrics']) if parent else None,
        charge_scope='segment_itemization_of_external_whole_job_reservation',
        whole_job_reservation_created=False)
    plan = {**value,'sha256':digest(value)}
    return _run_exposure_plan(plan, configs, env_factory, environment_reference, output_root,
        job_id=job_id, rollout_steps=rollout_steps, training_seed=training_seed,
        retention_seed=retention_seed, evaluation_seeds=evaluation_seeds, device=device,
        expected_initial_model_sha256=expected_initial_model_sha256, max_seconds=max_seconds,
        should_stop=should_stop, protected_paths=(*protected_paths, *parent['protected_trees']) if parent else protected_paths,
        parent=parent, segment=True)

"""Fixed stage exposure without score-controlled promotion or learning resets.

This job layer owns learning and local records. An external controller must own
queue closure, OS workers, resource leases, disk reservations and evaluations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
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
from .frame_training import FrameActorCritic
from .frame_workflow import _model_state_sha256


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _integer(value, name, minimum=1):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


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
        _require(isinstance(stage, dict) and set(stage) == {'name', 'config', 'updates'},
                 'stage fields must be name/config/updates')
        name = stage['name']
        _require(isinstance(name, str) and re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,63}', name)
                 and name not in names, 'stage names must be unique')
        names.add(name)
        _integer(stage['updates'], 'stage updates')
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
    _require(initial_model_sha256(configs[0], training_seed) == expected_initial_model_sha256,
             'initial model SHA mismatch before output/environment construction')
    value = {'format': 'transformer_rl.fixed_exposure_job', 'schema_version': 1,
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
    root = Path(output_root)
    _require(root.is_absolute() and root.resolve() == root
             and not any(p.is_symlink() for p in (root, *root.parents)), 'canonical output path required')
    _require(not os.path.lexists(root) and root.parent.is_dir(), 'new output and existing parent required')
    for protected in (Path(__file__).parent.resolve(), *map(lambda p: Path(p).resolve(), protected_paths)):
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                 'output overlaps a protected source/input tree')
    root.mkdir()
    _new(root / 'request.json', plan)
    _new(root / 'reservation.json', {'job_plan_sha256': plan['sha256'],
        'charged_updates': plan['reserved_updates'], 'charged_fresh_transition_budget': plan['reserved_fresh_transitions'],
        'actual_samples': 'recorded separately; reservation is not actual collection', 'refund': False})
    started = time.monotonic()
    session, endpoints, successful, attempts, collected = None, [], 0, 0, 0
    full_samples, recorded_samples, recorded_updates = 0, 0, 0
    shutdown_errors, stop_reason = [], None
    active_stage, failure_phase = None, 'before_environment'
    optimizer_steps, optimization_sample_uses, status, error = 0, 0, 'completed', None

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
            for index, (stage, config) in enumerate(zip(plan['stages'], configs)):
                active_stage = {'name': stage['name'], 'index': index}
                if stop():
                    status = 'interrupted'
                    break
                failure_phase = 'verify_source'
                _require(source_identity() == plan['source'], 'learner source changed at stage boundary')
                directory = root / f'stage_{index:04d}_{stage["name"]}'
                directory.mkdir()
                if session is None:
                    failure_phase = 'construct_fresh_environment'
                    session = FrameContinuation.start(config, env_factory, environment_reference,
                        expected_initial_model_sha256=expected_initial_model_sha256,
                        training_seed=training_seed, retention_seed=retention_seed,
                        evaluation_seeds=evaluation_seeds, rollout_steps=rollout_steps, device=device)
                else:
                    parent = endpoints[-1]
                    failure_phase = 'close_previous_environment'
                    session.close()
                    failure_phase = 'open_stage_environment'
                    session = FrameContinuation.open(config, env_factory, environment_reference,
                        parent['checkpoint']['path'], checkpoint_sha256=parent['checkpoint']['sha256'],
                        parent_update=parent['cumulative_successful_updates'], cumulative_transitions=parent['cumulative_collected_transitions'],
                        consumed_updates=parent['cumulative_attempted_updates'], rollout_steps=rollout_steps,
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
                if status != 'completed':
                    break
                failure_phase = 'verify_source'
                _require(source_identity() == plan['source'], 'learner source changed during stage')
                _require(session.config.to_dict() == stage['config'], 'runtime configuration changed before endpoint')
                _require(session.metadata.get('source') == plan['source'], 'runtime producer source changed before endpoint')
                path = directory / 'endpoint.pt'
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
                failure_phase = 'publish_endpoint'
                _new(directory / 'endpoint.json', endpoint)
                endpoints.append(endpoint)
            failure_phase = 'fsync_metrics'
            os.fsync(metrics.fileno())
    except BaseException as failure:
        status = 'user_interrupted' if isinstance(failure, (KeyboardInterrupt, SystemExit)) else 'failed'
        error = {'type': type(failure).__name__, 'message': str(failure), 'phase': failure_phase}
    finally:
        if session is not None:
            attempts, collected = session.consumed_updates, session.collected_transitions
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
        'unsealed_checkpoint_paths': [str(root / f'stage_{i:04d}_{s["name"]}' / 'endpoint.pt')
            for i, s in enumerate(plan['stages']) if i >= len(endpoints)
            and (root / f'stage_{i:04d}_{s["name"]}' / 'endpoint.pt').is_file()],
        'elapsed_s': time.monotonic() - started, 'refund': False, 'automatic_retries': 0,
        'independent_evaluation_performed': False, 'source': plan['source']}
    _new(root / 'completion.json', completion)
    return completion

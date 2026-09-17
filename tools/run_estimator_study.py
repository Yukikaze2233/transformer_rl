"""Bounded wiring -> oracle feasibility -> paired estimator study orchestration.

The feasibility gate uses the fixed 16M checkpoint. A failed gate stops for
review rather than silently consuming the optional 64M oracle extension.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from transformer_rl.checkpoint import load_checkpoint
from transformer_rl.config import load_config
from transformer_rl.experiments import _report, plan, source_identity, validate_plan
from transformer_rl.export import export_policy
from transformer_rl.model import ActorCritic


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args):
    source = Path(__file__).resolve().parents[1]
    protocol = read(source / 'docs/protocols/estimator_comparison.json')
    task = args.task_root.resolve(strict=True)
    contract = (task / args.contract).resolve(strict=True)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = read(source / 'configs/estimator_comparison.json')
    if (spec['seeds'] != protocol['stages']['screening']['training_seeds']
        or spec['training']['updates'] != protocol['stages']['screening']['target_updates']
        or {v['name'] for v in spec['variants']} != {v['name'] for v in protocol['variants']}):
        raise ValueError('compiled study differs from the research protocol')
    base = read(source / 'configs/optimized_control.json')
    base['environment'] = dict(task_root=str(task), contract=str(contract), num_envs=512,
                               mode='train', stage='locomotion')
    spec['environment_factory'] = 'examples.isaaclab_task:make_env'
    manifests = {}
    for stage in ('wiring', 'feasibility', 'screening'):
        directory = root / stage
        directory.mkdir()
        stage_spec, stage_base = copy.deepcopy(spec), copy.deepcopy(base)
        stage_spec['base_config'] = 'base.json'
        if stage == 'wiring':
            stage_spec['seeds'] = [2003]
            stage_spec['training'].update(updates=80, max_seconds=1800, checkpoint_interval=80)
            stage_spec['execution']['job_timeout_seconds'] = 3600
            stage_spec['evaluation'].update(steps=1000, seeds=[3003])
            stage_spec['evaluation']['scenarios'] = [spec['evaluation']['scenarios'][1]]
        elif stage == 'feasibility':
            stage_base['model'].update(proprio_dim=20, actor_type='mlp', history_length=1,
                                       baseline_hidden=[128, 64, 32])
            stage_base['environment']['privileged_actor'] = True
            stage_spec['seeds'] = [2309, 2311]
            stage_spec['variants'] = [dict(name='privileged_diagnostic', model={}, ppo={}, group='architecture')]
            stage_spec['training'].update(updates=977, max_seconds=7200, checkpoint_interval=977)
            stage_spec['execution']['job_timeout_seconds'] = 14400
        write(directory / 'base.json', stage_base)
        write(directory / 'spec.json', stage_spec)
        manifests[stage] = plan(directory / 'spec.json', directory / 'plan')
    capacities = {}
    for variant in spec['variants']:
        config, _, _ = load_config(root / 'screening/plan/configs' / f'{variant["name"]}.json')
        actor = ActorCritic(config).actor
        count = sum(p.numel() for p in actor.parameters()) - actor.log_std.numel()
        if count != protocol['model_capacity']['expected_deployment_parameters']['standard'][variant['name']] or count > 100000:
            raise ValueError(f'capacity contract mismatch: {variant["name"]}')
        capacities[variant['name']] = count
    frozen = [contract, source / 'examples/isaaclab_task.py', source / 'examples/_isaaclab_process.py',
              source / 'docs/protocols/estimator_comparison.json', Path(__file__).resolve()]
    frozen += sorted((task / 'src').rglob('*.py'))
    for stage in manifests:
        frozen += [root / stage / 'base.json', root / stage / 'spec.json']
    budget = sum(len(m['jobs']) * m['spec']['execution']['job_timeout_seconds'] for m in manifests.values()) + 600
    manifest = dict(created_at=now(), source=str(source), package=source_identity(), task_root=str(task),
                    protocol=protocol, stage_plan_hashes={k: v['plan_sha256'] for k, v in manifests.items()},
                    frozen_files={str(p): sha(p) for p in frozen}, max_parallel=1,
                    deployment_parameter_counts=capacities,
                    queue_max_seconds=budget, child_termination_grace_seconds=120,
                    oracle_contract='Diagnostic history-MLP with one frame: frame34 + age/valid + duplicate current command.',
                    confirmation_and_compact_not_automatically_scheduled=True)
    write(root / 'study.json', manifest)
    print(json.dumps({'root': str(root), 'stage_jobs': {k: len(v['jobs']) for k, v in manifests.items()},
                      'queue_max_seconds': budget}, indent=2))


def validate(root):
    manifest = read(root / 'study.json')
    if manifest['package'] != source_identity():
        raise ValueError('package changed after study preparation')
    for path, expected in manifest['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError(f'frozen source/config changed: {path}')
    for stage, expected in manifest['stage_plan_hashes'].items():
        if validate_plan(root / stage / 'plan', check_source=True)['plan_sha256'] != expected:
            raise ValueError(f'plan changed: {stage}')
    return manifest


def task_gate(report, scenario, thresholds):
    failures = []
    quality = report.get('control_quality', {})
    if not quality.get('available'):
        return ['physical_quality_unavailable']
    healthy, censored = quality.get('healthy_timeout_fraction'), quality.get('censored_sample_fraction')
    if healthy is None or healthy < thresholds['minimum_healthy_timeout_episode_fraction_per_report']:
        failures.append('healthy_episode_fraction')
    if censored is None or censored > thresholds['maximum_censored_sample_fraction']:
        failures.append('censored_sample_fraction')
    prefix = 'stand' if scenario.startswith('stand_') else 'motion'
    for signal, suffix in (('height_error', 'height_mae_m'), ('vx_error', 'vx_mae_m_s'),
                           ('wz_error', 'wz_mae_rad_s')):
        value = report.get('stability', {}).get('signals', {}).get(signal, {})
        mae = value.get('mean_abs')
        if mae is None or mae > thresholds[f'{prefix}_steady_{suffix}']:
            failures.append(signal + '_mae')
        if value.get('count', 0) / report['transitions'] < thresholds['minimum_steady_coverage_per_report']:
            failures.append(signal + '_coverage')
    return failures


def process(argv, log, deadline, stop, state):
    if stop.is_set():
        raise InterruptedError('study stop requested')
    with log.open('xb') as stream:
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state['worker_pid'] = child.pid
        try:
            while child.poll() is None:
                if stop.wait(0.2):
                    raise InterruptedError('study stop requested')
                if time.monotonic() >= deadline:
                    raise TimeoutError('job deadline exceeded')
            if child.returncode:
                raise RuntimeError(f'worker exit {child.returncode}: {log}')
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                grace = time.monotonic() + 120
                while time.monotonic() < grace:
                    child.poll()
                    os.killpg(child.pid, 0)
                    time.sleep(0.1)
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            state['worker_pid'] = None


def job_run(root, stage, manifest, job, stop, state):
    plan_root = root / stage / 'plan'
    directory = plan_root / job['directory']
    directory.mkdir(parents=True, exist_ok=False)
    spec = manifest['spec']
    deadline = time.monotonic() + spec['execution']['job_timeout_seconds']
    config_path = plan_root / job['config']
    common = ['--env-factory', spec['environment_factory'], '--device', 'cuda:0', '--action-clip', '100.0']
    worker = [sys.executable, '-m', 'examples._isaaclab_process']
    command = worker + ['train', '--config', str(config_path), '--seed', str(job['seed']),
                        '--run-dir', str(directory / 'train'), *common]
    for key in ('updates', 'rollout_steps', 'max_seconds', 'checkpoint_interval'):
        command += ['--' + key.replace('_', '-'), str(spec['training'][key])]
    write(directory / 'train-command.json', {'argv': command})
    state.update(stage=stage, job=job['id'], phase='train')
    process(command, directory / 'train.log', deadline, stop, state)
    completion = read(directory / 'train/completion.json')
    if completion['status'] != 'completed' or completion['cumulative_update'] != spec['training']['updates']:
        raise RuntimeError(f'training did not finish its budget: {job["id"]}')
    expected_samples = spec['training']['updates'] * 512 * spec['training']['rollout_steps']
    if completion['collected_transitions'] != expected_samples:
        raise RuntimeError('training sample budget mismatch')
    checkpoints = {item['update']: item for item in completion['checkpoints']}
    updates = [977, 1954, 3908] if stage == 'screening' else [spec['training']['updates']]
    results = []
    for update in updates:
        checkpoint = checkpoints[update]
        path = Path(checkpoint['path']).resolve()
        if not path.is_relative_to((directory / 'train/checkpoints').resolve()) or sha(path) != checkpoint['sha256']:
            raise ValueError('checkpoint path or SHA mismatch')
        model, _, actual_update, _ = load_checkpoint(path, device='cpu')
        if actual_update != update:
            raise ValueError('checkpoint update mismatch')
        deployed = sum(p.numel() for p in model.actor.parameters()) - model.actor.log_std.numel()
        if stage != 'feasibility' and deployed > 100000:
            raise ValueError('deployment parameter budget exceeded')
        evaluation_dir = directory / f'evaluation_{update:06d}'
        evaluation_dir.mkdir()
        for scenario, config in job['eval_configs'].items():
            environment = read(plan_root / config)['environment']
            for seed in spec['evaluation']['seeds']:
                state.update(phase=f'evaluate_{update}_{scenario}_{seed}')
                output = evaluation_dir / f'{scenario}_{seed}.json'
                trace = output.with_suffix('.npz')
                command = worker + ['evaluate', '--checkpoint', str(path), '--config', str(plan_root / config),
                                    '--steps', str(spec['evaluation']['steps']), '--seed', str(seed),
                                    '--output', str(output), '--trace-output', str(trace), *common,
                                    '--settle-steps', '200', '--min-steady-samples', '200']
                process(command, output.with_suffix('.log'), deadline, stop, state)
                report = _report(output, checkpoint['sha256'], seed, spec, environment)
                if report.get('checkpoint_update') != update or sha(trace) != report['trajectory']['sha256']:
                    raise ValueError('evaluation checkpoint or trace mismatch')
                if not report.get('control_quality', {}).get('available'):
                    raise ValueError('physical quality capture unavailable')
                if model.config.estimator_type != 'none' and 'state_estimation' not in report:
                    raise ValueError('estimator error report unavailable')
                failures = task_gate(report, scenario, read(root / 'study.json')['protocol']['evaluation']['research_acceptance'])
                results.append(dict(update=update, scenario=scenario, seed=seed, report=str(output),
                                    sha256=sha(output), task_gate_failures=failures))
        if stage == 'wiring':
            state['phase'] = 'cpu_export_verification'
            export_policy(path, directory / 'policy.onnx')
    validate(root)
    result = dict(status='completed', stage=stage, job=job['id'], finished_at=now(),
                  training_transitions=completion['collected_transitions'], final_update=completion['cumulative_update'],
                  deployment_parameters=deployed, evaluations=results)
    write(directory / 'result.json', result)
    return result


def summarize_stage(root, stage, results):
    rows = []
    for job in results:
        for evaluation in job['evaluations']:
            report = read(evaluation['report'])
            row = dict(job=job['job'], update=evaluation['update'], scenario=evaluation['scenario'],
                       evaluation_seed=evaluation['seed'], task_pass=not evaluation['task_gate_failures'],
                       task_gate_failures=';'.join(evaluation['task_gate_failures']))
            for name in ('height_error', 'vx_error', 'wz_error'):
                values = report['stability']['signals'][name]
                for metric in ('mean', 'mean_abs', 'within_episode_std', 'derivative_rms'):
                    row[f'{name}_{metric}'] = values[metric]
            row.update(healthy_timeout_fraction=report['control_quality']['healthy_timeout_fraction'],
                       world_xy_max_excursion_p95=report['control_quality']['world_xy_max_excursion_p95'])
            rows.append(row)
    write(root / stage / 'summary.json', {'rows': rows, 'scope': 'per training seed/scenario/checkpoint; no automatic winner'})
    with (root / stage / 'summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = args.root.resolve(strict=True)
    frozen = validate(root)
    stop, finished = threading.Event(), threading.Event()
    state = dict(started_at=now(), pid=os.getpid(), status='running', stage=None, job=None,
                 phase=None, worker_pid=None, completed=[], abort_reason=None)
    write(root / 'launch.json', {**state, 'script': str(Path(__file__).resolve()),
                                'process_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]})
    def interrupt(signum, frame):
        state['abort_reason'] = signal.Signals(signum).name
        stop.set()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    started = time.monotonic()
    def monitor():
        low_since = None
        with (root / 'resources.jsonl').open('x') as stream:
            while not finished.is_set():
                memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                available = int(memory['MemAvailable'].split()[0]) * 1024
                low_since = (low_since or time.monotonic()) if available < 1024**3 else None
                if low_since is not None and time.monotonic() - low_since >= 30:
                    state['abort_reason'] = 'low_memory_30s'
                    stop.set()
                if time.monotonic() - started >= frozen['queue_max_seconds']:
                    state['abort_reason'] = 'queue_deadline'
                    stop.set()
                snapshot = dict(state, observed_at=now(), available_memory_bytes=available)
                if state['stage'] and state['job']:
                    metric_path = root / state['stage'] / 'plan/jobs' / state['job'] / 'train/metrics.jsonl'
                    if metric_path.is_file():
                        with metric_path.open('rb') as metric_stream:
                            size = os.fstat(metric_stream.fileno()).st_size
                            metric_stream.seek(max(0, size - 65536))
                            lines = metric_stream.read().splitlines()
                        for line in reversed(lines):
                            try:
                                latest = json.loads(line)
                                snapshot.update(latest_update=latest['update'],
                                                collected_transitions=latest['collection']['total_transitions'],
                                                optimizer_steps=latest['optimization']['optimizer_steps'],
                                                estimator_accepted_steps=latest['optimization'].get('estimator_accepted_steps'))
                                break
                            except (ValueError, KeyError):
                                continue
                stream.write(json.dumps(snapshot) + '\n')
                stream.flush()
                write(root / 'status.json', snapshot, replace=True)
                finished.wait(10)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for stage in ('wiring', 'feasibility', 'screening'):
            manifest = validate_plan(root / stage / 'plan', check_source=True)
            stage_results = []
            for job in manifest['jobs']:
                if stop.is_set():
                    raise InterruptedError('study stop requested')
                validate(root)
                result = job_run(root, stage, manifest, job, stop, state)
                stage_results.append(result)
                state['completed'].append(f'{stage}/{job["id"]}')
            write(root / stage / 'results.json', {'jobs': stage_results, 'finished_at': now()})
            summarize_stage(root, stage, stage_results)
            if stage == 'feasibility':
                failures = [dict(job=r['job'], **e) for r in stage_results for e in r['evaluations'] if e['task_gate_failures']]
                write(root / 'feasibility-gate.json', {'passed': not failures, 'failures': failures,
                                                      'scope': '16M diagnostic, not architecture ranking'})
                if failures:
                    state.update(status='needs_review', phase='feasibility_gate_failed')
                    return
        state.update(status='completed', job=None, phase=None)
    except BaseException as error:
        state.update(status='stopped' if stop.is_set() else 'failed', error=str(error))
        raise
    finally:
        finished.set()
        thread.join(timeout=15)
        state['finished_at'] = now()
        write(root / 'execution-receipt.json', state)
        write(root / 'status.json', state, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run', 'validate', 'stop'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task-root', type=Path)
    parser.add_argument('--contract', default='contracts/own_v40_v2.json')
    args = parser.parse_args()
    if args.operation == 'prepare':
        if args.task_root is None:
            parser.error('prepare requires --task-root')
        prepare(args)
    elif args.operation == 'run':
        run(args)
    elif args.operation == 'validate':
        result = validate(args.root.resolve(strict=True))
        print(json.dumps({'valid': True, 'stage_plan_hashes': result['stage_plan_hashes']}))
    else:
        root = args.root.resolve(strict=True)
        if (root / 'execution-receipt.json').exists():
            print('Study already exited; see execution-receipt.json')
            return
        launch = read(root / 'launch.json')
        pid = launch['pid']
        if __package__:
            from . import trainctl
        else:
            import trainctl
        identity = trainctl.process_identity(pid)
        if identity is None:
            print('Study process already exited; inspect its receipt and checkpoints')
            return
        if (str(Path(__file__).resolve()) not in identity['argv'] or str(root) not in identity['argv']
            or 'run' not in identity['argv'] or identity['start_ticks'] != launch['process_start_ticks']):
            raise ValueError('PID identity differs from the study launcher')
        trainctl.signal_checked(identity)
        print('Study stop requested; wait for its exit receipt and saved checkpoint')


if __name__ == '__main__':
    main()

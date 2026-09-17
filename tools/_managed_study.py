"""Resumable orchestration over an immutable learning-source snapshot.

Each invocation owns a new attempt. Completed models/reports are reused only
after validation; partial optimizer state is restored by the frozen training CLI.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading
import time

if __package__:
    from . import trainctl as control
else:
    import trainctl as control


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def latest_metric(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - 65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
            if 'update' in value and 'collection' in value:
                return value
        except ValueError:
            continue
    return None


class ManagedStudy:
    def __init__(self, root, attempt=None, *, helpers=None):
        self.root = root.resolve(strict=True)
        self.attempt = control.inside(self.root, attempt) if attempt is not None else None
        definition = control.read_json(self.root / 'study.json')
        source = Path(definition['source']).resolve(strict=True)
        self.helpers = helpers or runpy.run_path(str(source / 'tools/run_estimator_study.py'))
        if helpers is None:
            module = sys.modules[self.helpers['load_checkpoint'].__module__]
            if not Path(module.__file__).resolve().is_relative_to(source / 'src'):
                raise ValueError('training package was imported from another source; use the registered runtime')
        self.definition = self.helpers['validate'](self.root)
        stages = ('wiring', 'feasibility', 'screening')
        if set(definition['stage_plan_hashes']) != set(stages):
            raise ValueError('unsupported study stages')
        self.plans = {stage: self.helpers['validate_plan'](self.root / stage / 'plan', check_source=True)
                      for stage in stages}
        self.cache = {}

    def directories(self, stage, job):
        directories = [self.root / stage / 'plan' / job['directory']]
        attempts = list((self.root / 'control/attempts').glob('*'))
        attempts.sort(key=lambda p: control.read_json(p / 'request.json', {}).get('generation', 0))
        directories.extend(p / stage / job['directory'] for p in attempts)
        return [control.inside(self.root, p) for p in directories if p.is_dir()]

    def checkpoint(self, path, stage, job):
        path = control.inside(self.root, path)
        key = (path, stage, job['id'])
        if key not in self.cache:
            model, trainer, update, metadata = self.helpers['load_checkpoint'](path, device='cpu')
            spec = self.plans[stage]['spec']
            config_path = self.root / stage / 'plan' / job['config']
            config, ppo, environment = self.helpers['load_config'](config_path)
            if model.config != config or trainer.config != ppo:
                raise ValueError(f'checkpoint model/PPO configuration mismatch: {path}')
            expected = dict(environment=environment, environment_factory=spec['environment_factory'],
                            action_clip=spec['training']['action_clip'], seed=job['seed'])
            if any(metadata.get(k) != value for k, value in expected.items()):
                raise ValueError(f'checkpoint environment/seed mismatch: {path}')
            if type(update) is not int or not 0 <= update <= spec['training']['updates']:
                raise ValueError(f'checkpoint update outside this job budget: {path}')
            count = metadata.get('collected_transitions')
            if type(count) is not int or count < 0:
                raise ValueError(f'checkpoint lacks valid segment sample count: {path}')
            self.cache[key] = dict(path=str(path), sha256=sha(path), update=update,
                                   segment_collected_transitions=count,
                                   resume_source=metadata.get('resume_source'),
                                   num_envs=environment.get('num_envs'),
                                   environment_identity=metadata.get('environment_provenance', {}).get('identity'),
                                   deployment_parameters=sum(p.numel() for p in model.actor.parameters()) - model.actor.log_std.numel())
        return self.cache[key]

    def checkpoints(self, stage, job):
        records = []
        for directory in self.directories(stage, job):
            completion = control.read_json(directory / 'train/completion.json', {})
            declared = {Path(c['path']).resolve(): c for c in completion.get('checkpoints', [])}
            for path in sorted((directory / 'train/checkpoints').glob('checkpoint_*.pt')):
                record = self.checkpoint(path, stage, job)
                if path.resolve() in declared and record['sha256'] != declared[path.resolve()]['sha256']:
                    raise ValueError(f'checkpoint digest differs from completion: {path}')
                if path.resolve() in declared and record['update'] != declared[path.resolve()]['update']:
                    raise ValueError(f'checkpoint update differs from completion: {path}')
                records.append(record)
        by_path = {Path(c['path']): c for c in records}
        for record in records:
            parent = control.inside(self.root, record['resume_source']) if record['resume_source'] else None
            if parent is not None and parent not in by_path:
                raise ValueError('checkpoint parent is unavailable in this study job')
            start = by_path[parent]['update'] if parent is not None else 0
            if start > record['update']:
                raise ValueError('checkpoint lineage moves backwards in updates')
            if record['num_envs'] is not None:
                minimum = (record['update'] - start) * record['num_envs'] * self.plans[stage]['spec']['training']['rollout_steps']
                if record['segment_collected_transitions'] < minimum:
                    raise ValueError('checkpoint sample count is smaller than its completed rollout budget')
            seen = {Path(record['path'])}
            while parent is not None:
                if parent in seen:
                    raise ValueError('checkpoint lineage cycle')
                seen.add(parent)
                ancestor = by_path[parent]
                parent = control.inside(self.root, ancestor['resume_source']) if ancestor['resume_source'] else None
                if parent is not None and parent not in by_path:
                    raise ValueError('checkpoint ancestor is unavailable')
        return records

    def milestones(self, stage):
        if stage == 'screening':
            return self.definition['protocol']['stages']['screening']['evaluation_checkpoint_updates']
        return [self.plans[stage]['spec']['training']['updates']]

    def validate_report(self, path, checkpoint, stage, job, scenario, seed, expected_sha=None):
        path = control.inside(self.root, path)
        if expected_sha is not None and sha(path) != expected_sha:
            raise ValueError(f'evaluation report digest mismatch: {path}')
        config_path = self.root / stage / 'plan' / job['eval_configs'][scenario]
        environment = self.helpers['read'](config_path)['environment']
        report = self.helpers['_report'](path, checkpoint['sha256'], seed,
                                          self.plans[stage]['spec'], environment)
        if report['checkpoint_update'] != checkpoint['update']:
            raise ValueError('evaluation checkpoint update mismatch')
        if report.get('environment_provenance', {}).get('identity') != checkpoint['environment_identity']:
            raise ValueError('evaluation environment identity mismatch')
        trace = report.get('trajectory', {})
        trace_path = control.inside(self.root, trace['path'])
        if not trace_path.is_file() or sha(trace_path) != trace['sha256']:
            raise ValueError('evaluation trace missing or changed')
        if not report.get('control_quality', {}).get('available'):
            raise ValueError('physical quality unavailable')
        failures = self.helpers['task_gate'](report, scenario,
            self.definition['protocol']['evaluation']['research_acceptance'])
        return dict(update=checkpoint['update'], scenario=scenario, seed=seed,
                    report=str(path), sha256=sha(path), task_gate_failures=failures)

    def completed(self, stage, job):
        candidates = [d / 'result.json' for d in self.directories(stage, job) if (d / 'result.json').is_file()]
        if not candidates:
            return None
        by_hash = {c['sha256']: c for c in self.checkpoints(stage, job)}
        expected = {(update, scenario, seed) for update in self.milestones(stage)
                    for scenario in job['eval_configs'] for seed in self.plans[stage]['spec']['evaluation']['seeds']}
        for path in reversed(candidates):
            result = control.read_json(path)
            if result.get('status') != 'completed':
                continue
            if result.get('job') != job['id'] or result.get('final_update') != self.plans[stage]['spec']['training']['updates']:
                raise ValueError(f'invalid completed job identity or budget: {path}')
            evaluations = result['evaluations']
            keys = {(e['update'], e['scenario'], e['seed']) for e in evaluations}
            if keys != expected or len(keys) != len(evaluations):
                raise ValueError(f'incomplete or duplicate evaluations in result: {path}')
            validated = []
            for item in evaluations:
                report_path = control.inside(self.root, item['report'])
                report = control.read_json(report_path)
                checkpoint = by_hash.get(report['checkpoint_sha256'])
                if checkpoint is None:
                    raise ValueError('completed evaluation refers to an unavailable checkpoint')
                validated.append(self.validate_report(report_path, checkpoint, stage, job,
                    item['scenario'], item['seed'], item['sha256']))
            if stage == 'wiring':
                export_path = control.inside(self.root, result.get('policy_onnx', path.parent / 'policy.onnx'))
                sidecar = control.read_json(Path(str(export_path) + '.json'))
                if sidecar is None or not export_path.is_file():
                    continue
                if sha(export_path) != sidecar['onnx_sha256']:
                    raise ValueError('completed ONNX export digest mismatch')
                exported_checkpoint = by_hash.get(sidecar['checkpoint']['sha256'])
                if exported_checkpoint is None or exported_checkpoint['update'] != result['final_update']:
                    raise ValueError('ONNX export belongs to another checkpoint')
            return {**result, 'evaluations': validated, 'reused_result': str(path)}
        return None

    def accounting(self, stage, job):
        segments = []
        for directory in self.directories(stage, job):
            training = directory / 'train'
            if not training.is_dir():
                continue
            completion = control.read_json(training / 'completion.json')
            failure = control.read_json(training / 'failure.json', {})
            metric = latest_metric(training / 'metrics.jsonl')
            candidates = [0, failure.get('collected_transitions', 0)]
            if completion:
                candidates.append(completion['collected_transitions'])
            if metric:
                candidates.append(metric['collection']['total_transitions'])
            candidates += [c['segment_collected_transitions'] for c in self.checkpoints(stage, job)
                           if Path(c['path']).is_relative_to(training)]
            segments.append(dict(directory=str(training), collected_transitions=max(candidates),
                                 exact=completion is not None))
        return dict(segments=segments, observed_collected_transitions=sum(s['collected_transitions'] for s in segments),
                    collection_count_exact=all(s['exact'] for s in segments),
                    scope='All segment observations; counts may be lower bounds after a crash. Rollout/update counters are not optimizer sample reuse.')

    def execute(self, argv, log, deadline, stop, state):
        state.update(worker_log=str(log), command=argv)
        control.save_json(self.attempt / 'status.json', dict(state), replace=True)
        return self.helpers['process'](argv, log, deadline, stop, state)

    def run_job(self, stage, job, stop, state):
        completed = self.completed(stage, job)
        if completed:
            return completed
        manifest = self.plans[stage]
        spec = manifest['spec']
        records = self.checkpoints(stage, job)
        # Stable sort keeps the latest serialized segment when updates tie.
        latest = max(enumerate(records), key=lambda x: (x[1]['update'], x[0]))[1] if records else None
        start_update = latest['update'] if latest else 0
        target_update = spec['training']['updates']
        remaining = target_update - start_update
        directory = self.attempt / stage / job['directory']
        directory.mkdir(parents=True, exist_ok=False)
        state.update(stage=stage, job=job['id'], phase='train' if remaining else 'evaluation_resume',
                     latest_update=start_update, resumed_from_update=start_update,
                     target_update=target_update, metric_path=None)
        control.save_json(directory / 'resume.json', dict(resume_checkpoint=latest, additional_updates=remaining,
                          target_update=target_update, source_plan_sha256=manifest['plan_sha256'],
                          continuation='model and optimizer(s) restored; environment, history and random stream reset'))
        deadline = time.monotonic() + spec['execution']['job_timeout_seconds']
        plan_root = self.root / stage / 'plan'
        device = spec['execution']['devices'][0]
        common = ['--env-factory', spec['environment_factory'], '--device', device]
        if spec['training']['action_clip'] is not None:
            common += ['--action-clip', str(spec['training']['action_clip'])]
        worker = [sys.executable, '-m', spec['execution'].get('worker_module', 'transformer_rl')]
        if remaining:
            command = worker + ['train', '--config', str(plan_root / job['config']), '--seed', str(job['seed']),
                                '--run-dir', str(directory / 'train'), '--updates', str(remaining), *common]
            for key in ('rollout_steps', 'max_seconds', 'checkpoint_interval'):
                command += ['--' + key.replace('_', '-'), str(spec['training'][key])]
            if latest:
                if sha(latest['path']) != latest['sha256']:
                    raise ValueError('resume checkpoint changed after validation')
                command += ['--resume', latest['path']]
            if spec['training'].get('diagnostics'):
                command += ['--diagnostics']
            state['metric_path'] = str(directory / 'train/metrics.jsonl')
            self.execute(command, directory / 'train.log', deadline, stop, state)
            completion = control.read_json(directory / 'train/completion.json')
            if (completion['status'] != 'completed' or completion['updates_completed'] != remaining
                or completion['cumulative_update'] != target_update):
                raise RuntimeError('training segment stopped before the remaining budget completed')
            records = self.checkpoints(stage, job)
        by_update = {record['update']: record for record in records}
        evaluations = []
        for update in self.milestones(stage):
            if update not in by_update:
                raise ValueError(f'required checkpoint {update} unavailable; cannot replace it with a later policy')
            checkpoint = by_update[update]
            for scenario in job['eval_configs']:
                for seed in spec['evaluation']['seeds']:
                    relative = f'evaluation_{update:06d}/{scenario}_{seed}.json'
                    reuse = None
                    for previous in reversed(self.directories(stage, job)):
                        path = previous / relative
                        if path.is_file():
                            report = control.read_json(path)
                            if report.get('checkpoint_sha256') == checkpoint['sha256']:
                                reuse = self.validate_report(path, checkpoint, stage, job, scenario, seed)
                                break
                    if reuse is None:
                        output = directory / relative
                        output.parent.mkdir(parents=True, exist_ok=True)
                        trace = output.with_suffix('.npz')
                        command = worker + ['evaluate', '--checkpoint', checkpoint['path'],
                                            '--config', str(plan_root / job['eval_configs'][scenario]),
                                            '--steps', str(spec['evaluation']['steps']), '--seed', str(seed),
                                            '--output', str(output), '--trace-output', str(trace), *common]
                        for key in ('settle_steps', 'min_steady_samples'):
                            command += ['--' + key.replace('_', '-'), str(spec['evaluation'].get(key, 200))]
                        state['phase'] = f'evaluate_{update}_{scenario}_{seed}'
                        self.execute(command, output.with_suffix('.log'), deadline, stop, state)
                        reuse = self.validate_report(output, checkpoint, stage, job, scenario, seed)
                    evaluations.append(reuse)
        final = by_update[target_update]
        policy = None
        if stage == 'wiring':
            state['phase'] = 'cpu_export_verification'
            policy = directory / 'policy.onnx'
            self.helpers['export_policy'](final['path'], policy)
        self.helpers['validate'](self.root)
        accounting = self.accounting(stage, job)
        result = dict(status='completed', stage=stage, job=job['id'], finished_at=control.now(),
                      final_update=target_update, checkpoint=final, checkpoint_records=list(by_update.values()),
                      deployment_parameters=final['deployment_parameters'], evaluations=evaluations,
                      training_transitions=accounting['observed_collected_transitions'], accounting=accounting,
                      policy_onnx=str(policy) if policy else None, control_attempt=str(self.attempt))
        control.save_json(directory / 'result.json', result)
        return result


def inspect_study(root):
    study = ManagedStudy(root)
    jobs = []
    for stage, plan in study.plans.items():
        for job in plan['jobs']:
            complete = study.completed(stage, job)
            checkpoints = study.checkpoints(stage, job)
            update = max((c['update'] for c in checkpoints), default=0)
            jobs.append(dict(stage=stage, job=job['id'], completed=bool(complete),
                             latest_saved_update=update, remaining_updates=plan['spec']['training']['updates'] - update))
    return dict(valid=True, source=study.definition['source'], jobs=jobs,
                scope='CPU validation and resume planning only; no environment or worker started')


class WorkerState(dict):
    def __init__(self, attempt, *args, **kwargs):
        self.attempt = attempt
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == 'worker_pid':
            super().__setitem__('worker_identity', control.process_identity(value) if value else None)
            control.save_json(self.attempt / 'status.json', dict(self), replace=True)


def _run_managed(root, attempt):
    root = root.resolve(strict=True)
    attempt = control.inside(root, attempt)
    request = control.read_json(attempt / 'request.json')
    if request['registration']['root'] != str(root):
        raise ValueError('attempt belongs to another study')
    with control.locked(root / 'control/run.lock', blocking=False):
        current = control.read_json(root / 'control/current.json')
        if current['attempt'] != str(attempt):
            raise ValueError('attempt is no longer the current generation')
        legacy = control.legacy_identity(root)
        if legacy or control.orphan_workers(root):
            raise RuntimeError('an existing study supervisor/worker is still active')
        stop, finished = threading.Event(), threading.Event()
        state = WorkerState(attempt, status='running', started_at=control.now(), stage=None, job=None,
                            phase='validating', worker_pid=None, worker_identity=None,
                            worker_log=None, completed=[], abort_reason=None)
        def interrupt(signum, frame):
            state['abort_reason'] = signal.Signals(signum).name
            stop.set()
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        control.save_json(attempt / 'launch.json', dict(identity=control.process_identity(os.getpid()),
                          started_at=control.now(), controller_files={str(Path(__file__).resolve()): sha(__file__),
                          str(Path(control.__file__).resolve()): sha(control.__file__)}))
        control.save_json(attempt / 'status.json', dict(state), replace=True)
        started = time.monotonic()
        thread = None
        try:
            if (attempt / 'pause-request.json').exists():
                state.update(status='stopped', abort_reason='pause_requested_before_start')
                return 0
            study = ManagedStudy(root, attempt)
            if Path(request['registration']['source']).resolve() != Path(study.definition['source']).resolve():
                raise ValueError('registered source differs from the frozen study')
            control.save_json(attempt / 'binding.json', dict(source=study.definition['source'],
                              package=study.definition['package'], stage_plan_hashes=study.definition['stage_plan_hashes']))
            def monitor():
                low_since = None
                with (attempt / 'resources.jsonl').open('x') as stream:
                    while not finished.is_set():
                        try:
                            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
                            available = int(memory['MemAvailable'].split()[0]) * 1024
                            low_since = (low_since or time.monotonic()) if available < 1024**3 else None
                            if low_since is not None and time.monotonic() - low_since >= 30:
                                state['abort_reason'] = 'low_memory_30s'
                                stop.set()
                            if time.monotonic() - started >= study.definition['queue_max_seconds']:
                                state['abort_reason'] = 'queue_deadline'
                                stop.set()
                            snapshot = dict(state, observed_at=control.now(), available_memory_bytes=available)
                            if state.get('metric_path'):
                                metric = latest_metric(Path(state['metric_path']))
                                if metric:
                                    snapshot.update(latest_update=metric['update'],
                                                    collected_transitions=metric['collection']['total_transitions'])
                            stream.write(json.dumps(snapshot) + '\n')
                            stream.flush()
                            control.save_json(attempt / 'status.json', snapshot, replace=True)
                        except Exception as error:
                            state['abort_reason'] = 'monitor_error: ' + str(error)
                            stop.set()
                        finished.wait(5)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            for stage, plan in study.plans.items():
                results = []
                for job in plan['jobs']:
                    if stop.is_set():
                        raise InterruptedError('pause requested')
                    study.helpers['validate'](root)
                    result = study.run_job(stage, job, stop, state)
                    results.append(result)
                    state['completed'].append(f'{stage}/{job["id"]}')
                directory = attempt / stage
                directory.mkdir(parents=True, exist_ok=True)
                control.save_json(directory / 'results.json', {'jobs': results, 'finished_at': control.now()})
                study.helpers['summarize_stage'](attempt, stage, results)
                if stage == 'feasibility':
                    failures = [dict(job=r['job'], **e) for r in results for e in r['evaluations'] if e['task_gate_failures']]
                    control.save_json(attempt / 'feasibility-gate.json', dict(passed=not failures, failures=failures))
                    if failures:
                        state.update(status='needs_review', phase='feasibility_gate_failed')
                        return 2
            state.update(status='completed', phase=None, job=None)
            return 0
        except BaseException as error:
            state.update(status='stopped' if stop.is_set() else 'failed', error=str(error))
            return 0 if stop.is_set() else 1
        finally:
            finished.set()
            if thread:
                thread.join(timeout=15)
            state['finished_at'] = control.now()
            control.save_json(attempt / 'exit.json', dict(state))
            control.save_json(attempt / 'status.json', dict(state), replace=True)


def run_managed(root, attempt):
    try:
        return _run_managed(root, attempt)
    except Exception as error:
        attempt = control.inside(root.resolve(strict=True), attempt)
        if not (attempt / 'exit.json').exists():
            control.save_json(attempt / 'exit.json', dict(status='failed', error=str(error), finished_at=control.now()))
        raise

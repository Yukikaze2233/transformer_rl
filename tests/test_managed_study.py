"""Resume accounting and artifact reuse, with no neural or simulator execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import signal

import pytest

from tools import trainctl
from tools._managed_study import ManagedStudy, sha
from tools import _managed_study


class Harness:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.environment = {'num_envs': 2}
        self.job = {'id': 'model/seed_1', 'directory': 'jobs/model/seed_1', 'variant': 'model', 'seed': 1,
                    'config': 'train.json', 'eval_configs': {'stand_mid': 'eval.json'}}
        self.spec = {'environment_factory': 'fixtures:factory',
                     'training': {'updates': 2, 'rollout_steps': 3, 'max_seconds': 10,
                                  'checkpoint_interval': 1, 'action_clip': 100.0},
                     'execution': {'devices': ['cpu'], 'worker_module': 'fixture_worker', 'job_timeout_seconds': 30},
                     'evaluation': {'steps': 4, 'seeds': [3], 'settle_steps': 0, 'min_steady_samples': 1}}
        source = root / 'frozen-source'
        source.mkdir()
        self.definition = {'source': str(source), 'stage_plan_hashes': dict.fromkeys(('wiring', 'feasibility', 'screening'), 'hash'),
                           'protocol': {'stages': {'screening': {'evaluation_checkpoint_updates': [1, 2]}},
                                        'evaluation': {'research_acceptance': {}}}}
        trainctl.save_json(root / 'study.json', self.definition)
        for stage in self.definition['stage_plan_hashes']:
            trainctl.save_json(root / stage / 'plan/eval.json', {'environment': self.environment})

    def checkpoint(self, directory, update, count, parent=None):
        path = directory / f'train/checkpoints/checkpoint_{update:06d}.pt'
        metadata = {'environment': self.environment, 'environment_factory': 'fixtures:factory',
                    'action_clip': 100.0, 'seed': 1, 'collected_transitions': count,
                    'resume_source': str(parent) if parent else None,
                    'environment_provenance': {'identity': {'test': True}}}
        trainctl.save_json(path, {'update': update, 'metadata': metadata})
        return {'path': str(path), 'update': update, 'sha256': sha(path)}

    def completion(self, directory, update, count, checkpoint, status='completed'):
        trainctl.save_json(directory / 'train/completion.json', {'status': status, 'cumulative_update': update,
                          'updates_completed': update, 'collected_transitions': count, 'checkpoints': [checkpoint]})

    def load(self, path, device='cpu'):
        payload = trainctl.read_json(path)
        parameter = SimpleNamespace(numel=lambda: 16)
        actor = SimpleNamespace(parameters=lambda: [parameter], log_std=SimpleNamespace(numel=lambda: 6))
        return SimpleNamespace(config='model-config', actor=actor), SimpleNamespace(config='ppo-config'), payload['update'], payload['metadata']

    def report(self, path, checkpoint, seed=3):
        trace = path.with_suffix('.npz')
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(b'fixture trace; not simulated data')
        trainctl.save_json(path, {'checkpoint_sha256': checkpoint['sha256'], 'checkpoint_update': checkpoint['update'],
                          'environment_provenance': {'identity': {'test': True}}, 'seed': seed,
                          'trajectory': {'path': str(trace), 'sha256': sha(trace)}, 'control_quality': {'available': True}})

    def validate_report(self, path, digest, seed, spec, environment):
        value = trainctl.read_json(path)
        assert value['checkpoint_sha256'] == digest and value['seed'] == seed
        return value

    def export(self, checkpoint, output):
        output.write_bytes(b'fixture model')
        trainctl.save_json(Path(str(output) + '.json'), {'onnx_sha256': sha(output), 'checkpoint': {'sha256': sha(checkpoint)}})

    def process(self, argv, log, deadline, stop, state):
        self.calls.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('fixture worker\n')
        arg = lambda name: argv[argv.index(name) + 1]
        if 'train' in argv:
            directory = Path(arg('--run-dir')).parent
            remaining = int(arg('--updates'))
            parent = Path(arg('--resume')) if '--resume' in argv else None
            initial = trainctl.read_json(parent)['update'] if parent else 0
            checkpoint = self.checkpoint(directory, initial + remaining, remaining * 6, parent)
            trainctl.save_json(directory / 'train/completion.json', {'status': 'completed',
                'cumulative_update': initial + remaining, 'updates_completed': remaining,
                'collected_transitions': remaining * 6, 'checkpoints': [checkpoint]})
        else:
            path = Path(arg('--checkpoint'))
            self.report(Path(arg('--output')), {'sha256': sha(path), 'update': trainctl.read_json(path)['update']})

    def helpers(self):
        return {'validate': lambda root: self.definition,
                'validate_plan': lambda root, check_source: {'spec': self.spec, 'jobs': [self.job], 'plan_sha256': 'hash'},
                'load_checkpoint': self.load, 'load_config': lambda path: ('model-config', 'ppo-config', self.environment),
                'read': trainctl.read_json, '_report': self.validate_report, 'task_gate': lambda *args: [],
                'process': self.process, 'export_policy': self.export}

    def attempt(self, generation):
        attempt = self.root / f'control/attempts/{generation}'
        trainctl.save_json(attempt / 'request.json', {'generation': generation})
        return ManagedStudy(self.root, attempt, helpers=self.helpers())


def test_partial_training_resumes_only_remaining_updates_and_counts_discarded_samples(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'wiring/plan' / fixture.job['directory']
    checkpoint = fixture.checkpoint(original, 1, 8)
    fixture.completion(original, 1, 8, checkpoint, status='stopped')
    original_sha = sha(checkpoint['path'])
    managed = fixture.attempt(1)
    result = managed.run_job('wiring', fixture.job, threading.Event(), {})
    command = fixture.calls[0]
    assert command[command.index('--updates') + 1] == '1'
    assert command[command.index('--resume') + 1] == checkpoint['path']
    assert result['final_update'] == 2 and result['training_transitions'] == 14
    assert result['accounting']['collection_count_exact']
    assert sha(checkpoint['path']) == original_sha
    fixture.calls.clear()
    repeated = fixture.attempt(2).run_job('wiring', fixture.job, threading.Event(), {})
    assert repeated['reused_result'] and not fixture.calls


def test_pause_during_evaluation_does_not_retrain_finished_model(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'wiring/plan' / fixture.job['directory']
    checkpoint = fixture.checkpoint(original, 2, 12)
    fixture.completion(original, 2, 12, checkpoint)
    managed = fixture.attempt(1)
    result = managed.run_job('wiring', fixture.job, threading.Event(), {})
    assert len(fixture.calls) == 1 and 'evaluate' in fixture.calls[0]
    assert result['checkpoint']['sha256'] == checkpoint['sha256']


def test_existing_valid_evaluation_is_reused_while_export_is_repaired(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'wiring/plan' / fixture.job['directory']
    checkpoint = fixture.checkpoint(original, 2, 12)
    fixture.completion(original, 2, 12, checkpoint)
    report = original / 'evaluation_000002/stand_mid_3.json'
    fixture.report(report, checkpoint)
    result = fixture.attempt(1).run_job('wiring', fixture.job, threading.Event(), {})
    assert not fixture.calls
    assert result['evaluations'][0]['report'] == str(report)


def test_missing_earlier_checkpoint_is_not_replaced_with_later_policy(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'screening/plan' / fixture.job['directory']
    checkpoint = fixture.checkpoint(original, 2, 12)
    fixture.completion(original, 2, 12, checkpoint)
    with pytest.raises(ValueError, match='required checkpoint 1'):
        fixture.attempt(1).run_job('screening', fixture.job, threading.Event(), {})
    assert not fixture.calls


def test_completed_report_corruption_is_rejected_before_any_worker(tmp_path):
    fixture = Harness(tmp_path)
    result = fixture.attempt(1).run_job('wiring', fixture.job, threading.Event(), {})
    report = Path(result['evaluations'][0]['report'])
    report.write_text(report.read_text() + ' ')
    fixture.calls.clear()
    with pytest.raises(ValueError, match='digest mismatch'):
        fixture.attempt(2).run_job('wiring', fixture.job, threading.Event(), {})
    assert not fixture.calls


def test_underreported_checkpoint_samples_are_rejected(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'wiring/plan' / fixture.job['directory']
    fixture.checkpoint(original, 2, 1)
    with pytest.raises(ValueError, match='sample count'):
        fixture.attempt(1).run_job('wiring', fixture.job, threading.Event(), {})
    assert not fixture.calls


def test_crashed_segment_counts_remain_a_lower_bound_after_checkpoint_resume(tmp_path):
    fixture = Harness(tmp_path)
    original = tmp_path / 'wiring/plan' / fixture.job['directory']
    fixture.checkpoint(original, 1, 6)
    trainctl.save_json(original / 'train/failure.json', {'collected_transitions': 18})
    result = fixture.attempt(1).run_job('wiring', fixture.job, threading.Event(), {})
    assert result['training_transitions'] == 24
    assert not result['accounting']['collection_count_exact']


def test_managed_supervisor_runs_stages_and_reuses_completed_work_without_workers(tmp_path, monkeypatch):
    fixture = Harness(tmp_path)
    fixture.definition.update(package={'test': True}, queue_max_seconds=120)
    fixture.definition['protocol']['stages']['screening']['evaluation_checkpoint_updates'] = [2]
    original_class = ManagedStudy
    helpers = fixture.helpers()
    helpers['summarize_stage'] = lambda root, stage, results: trainctl.save_json(root / stage / 'summary.json', {'jobs': len(results)})
    monkeypatch.setattr(_managed_study, 'ManagedStudy',
                        lambda root, attempt: original_class(root, attempt, helpers=helpers))
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    for generation in (1, 2):
        attempt = tmp_path / f'control/attempts/{generation}'
        trainctl.save_json(attempt / 'request.json', {'generation': generation,
            'registration': {'root': str(tmp_path), 'source': fixture.definition['source']}})
        trainctl.save_json(tmp_path / 'control/current.json', {'attempt': str(attempt), 'generation': generation}, replace=True)
        fixture.calls.clear()
        assert _managed_study.run_managed(tmp_path, attempt) == 0
        receipt = trainctl.read_json(attempt / 'exit.json')
        assert receipt['status'] == 'completed'
        assert len(receipt['completed']) == 3
        assert len(fixture.calls) == (6 if generation == 1 else 0)

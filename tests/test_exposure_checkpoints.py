"""Continuous CPU checkpoint observations and complete learning-prefix proofs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from transformer_rl import exposure_training as exposure
from transformer_rl.frame_checkpoint import capture_rng
from transformer_rl.frame_config import json_bytes
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.ppo import PPOTrainer
from test_exposure_training import config, run_job, same
from test_exposure_segments import options, pin, stage
from packed_env import make_env


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def declared(configuration=None, updates=6, points=None):
    return [{**stage(configuration, updates=updates),
             'checkpoint_updates': list(points if points is not None else [2,4,6])}]


def plan_for(root):
    request = json.loads((root/'request.json').read_bytes())
    values = {key: request[key] for key in ('job_id','rollout_steps','training_seed','retention_seed',
        'evaluation_seeds','device','expected_initial_model_sha256','max_seconds')}
    stages = [{key: value[key] for key in ('name','config','updates','checkpoint_updates')}
              for value in request['stages']]
    return exposure._definition(stages, environment_reference=request['environment_factory'], **values)[0]


@pytest.mark.parametrize('architecture,history,readout', [
    ('mlp',1,'last'), ('history_mlp',5,'last'),
    ('transformer',3,'last'), ('transformer',5,'query')])
def test_intermediate_saves_leave_one_environment_history_Adam_and_RNG_unchanged(
        tmp_path, monkeypatch, architecture, history, readout):
    configuration = config(architecture,history,readout)
    plain_root = tmp_path/'plain'
    plain = run_job(plain_root, [stage(configuration,updates=6)])
    original_save = FrameContinuation.save
    observed, environments = [], []

    def factory(**kwargs):
        env = make_env(**kwargs)
        environments.append(env)
        return env

    def inspected_save(self, path):
        before = {'RNG':capture_rng(), 'model':deepcopy(self.model.state_dict()),
            'Adam':deepcopy(self.trainer.optimizer.state_dict()),
            'history':self.collector._history.snapshot().frames.clone(),
            'environment':self.env.state.clone(), 'age':self.env.age.clone(), 'tick':self.env.tick,
            'clock':(self.update,self.consumed_updates,self.collected_transitions)}
        result = original_save(self,path)
        after = {'RNG':capture_rng(), 'model':self.model.state_dict(),
            'Adam':self.trainer.optimizer.state_dict(), 'history':self.collector._history.snapshot().frames,
            'environment':self.env.state, 'age':self.env.age, 'tick':self.env.tick,
            'clock':(self.update,self.consumed_updates,self.collected_transitions)}
        same(before,after)
        assert not self.env.closed
        observed.append(self.update)
        return result

    monkeypatch.setattr(FrameContinuation,'save',inspected_save)
    output = tmp_path/'continuous'
    result = run_job(output, declared(configuration), factory)
    assert result['status'] == 'completed' and result['schema_version'] == 2
    assert observed == [2,4,6] and len(environments) == 1 and environments[0].closed
    assert len(result['endpoints']) == 1 and result['successful_updates'] == 6
    assert result['missing_checkpoints'] == [] and result['unsealed_checkpoint_paths'] == []
    assert [item['local_update'] for item in result['sealed_checkpoints']] == [2,4,6]
    assert [item['kind'] for item in result['sealed_checkpoints']] == ['intermediate','intermediate','endpoint']
    assert len(list(output.rglob('*.pt'))) == 3
    saved = torch.load(result['endpoints'][0]['checkpoint']['path'],map_location='cpu',weights_only=True)
    original = torch.load(plain['endpoints'][0]['checkpoint']['path'],map_location='cpu',weights_only=True)
    for key in ('model','optimizer','rng','update'):
        same(saved[key],original[key])
    for item in result['sealed_checkpoints']:
        proof = exposure.verify_exposure_checkpoint(item['record'],plan_for(output),configuration)
        assert proof['endpoint']['cumulative_successful_updates'] == item['checkpoint_update']
        assert proof['checkpoint']['sha256'] == proof['endpoint']['checkpoint']['sha256']
        assert proof['metrics_prefix'] == item['metrics_prefix']


@pytest.mark.parametrize('points', [[],[1,1,6],[4,2,6],[2,4],[True,6],[0,6],[2,7],[2.,6]])
def test_invalid_local_schedule_is_rejected_before_output_or_factory(tmp_path, points):
    calls = []
    with pytest.raises(ValueError,match='checkpoint updates'):
        run_job(tmp_path/'invalid',declared(points=points),lambda **kwargs: calls.append(True))
    assert calls == [] and not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('failure', ['numerical','interrupted','partial_rollout'])
def test_saved_prefix_is_auditable_after_later_failure_without_final_completion(tmp_path,monkeypatch,failure):
    original_step = FrameContinuation.step
    calls = 0

    def failed_step(self,**kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            if failure == 'numerical':
                original_update = self.trainer.update
                def broken(*args,**options):
                    original_update(*args,**options)
                    raise FloatingPointError('synthetic failure after actual Adam')
                self.trainer.update = broken
            elif failure == 'interrupted':
                raise KeyboardInterrupt('synthetic interrupt')
            else:
                return original_step(self,should_stop=lambda: True)
        return original_step(self,**kwargs)

    monkeypatch.setattr(FrameContinuation,'step',failed_step)
    output = tmp_path/failure
    configuration = config('transformer',5,'query')
    result = run_job(output,declared(configuration))
    assert result['status'] != 'completed' and result['endpoints'] == []
    assert [item['local_update'] for item in result['sealed_checkpoints']] == [2]
    assert result['missing_checkpoints'] == [{'stage_index':0,'local_update':4},{'stage_index':0,'local_update':6}]
    item = result['sealed_checkpoints'][0]
    proof = exposure.verify_exposure_checkpoint(item['record'],plan_for(output),configuration)
    assert proof['endpoint']['cumulative_successful_updates'] == 2
    assert proof['endpoint']['cumulative_collected_transitions'] == 8
    assert result['charged_updates'] == 6 and result['charged_fresh_transition_budget'] == 24
    assert result['refund'] is False and result['automatic_retries'] == 0
    with pytest.raises(ValueError,match='sealed endpoint'):
        exposure._segment_parent(item['record'],plan_for(output),configuration)


def test_scheduled_final_can_resume_and_midpoint_proves_original_full_parent_chain(tmp_path):
    configuration = config()
    first_root = tmp_path/'first'
    first = exposure.train_exposure_segment(declared(configuration,updates=2,points=[1,2])[0],
        make_env,'packed_env:make_env',first_root,**options(configuration))
    final_receipt = first['sealed_checkpoints'][-1]['record']
    second_root = tmp_path/'second'
    second = exposure.train_exposure_segment(declared(configuration,updates=4,points=[2,4])[0],
        make_env,'packed_env:make_env',second_root,parent_endpoint=final_receipt,**options(configuration))
    assert second['status'] == 'completed'
    assert [item['checkpoint_update'] for item in second['sealed_checkpoints']] == [4,6]
    midpoint = exposure.verify_exposure_checkpoint(second['sealed_checkpoints'][0]['record'],
                                                   plan_for(second_root),configuration)
    assert midpoint['endpoint']['cumulative_successful_updates'] == 4
    assert midpoint['cumulative_optimizer_steps'] > 0
    assert any(str(first_root) in str(path) for path,_ in midpoint['proof_inputs'])


@pytest.mark.parametrize('mutation', ['missing','prefix_hash','prefix_optimizer','payload_Adam'])
def test_self_signed_snapshot_inventory_cannot_replace_actual_schedule_metrics_or_Adam(tmp_path,mutation):
    configuration = config()
    output = tmp_path/'original'
    result = run_job(output,declared(configuration))
    completion_path = output/'completion.json'
    completion = json.loads(completion_path.read_bytes())
    item = completion['sealed_checkpoints'][0]
    if mutation == 'missing':
        completion['sealed_checkpoints'].pop(0)
    elif mutation == 'prefix_hash':
        item['metrics_prefix']['sha256'] = '0'*64
    elif mutation == 'prefix_optimizer':
        item['metrics_prefix']['optimizer_steps'] += 1
    else:
        record_path = Path(item['record']['path'])
        record = json.loads(record_path.read_bytes())
        checkpoint = Path(record['checkpoint']['path'])
        payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
        for state in payload['optimizer']['state'].values():
            state['step'] += 1
        torch.save(payload,checkpoint)
        record['checkpoint'] = pin(checkpoint)
        sidecar_path = Path(record['sidecar']['path'])
        sidecar = json.loads(sidecar_path.read_bytes())
        sidecar['sha256'] = record['checkpoint']['sha256']
        sidecar_path.write_bytes(json_bytes(sidecar)+b'\n')
        record['sidecar']['sha256'] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        record_path.write_bytes(json_bytes(record)+b'\n')
        item['record'] = pin(record_path)
    completion_path.write_bytes(json_bytes(completion)+b'\n')
    receipt = item['record']
    with pytest.raises(ValueError):
        exposure.verify_exposure_checkpoint(receipt,plan_for(output),configuration)
    if mutation == 'missing':
        with pytest.raises(ValueError):
            exposure._segment_parent(completion['sealed_checkpoints'][-1]['record'],plan_for(output),configuration)

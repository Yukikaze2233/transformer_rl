"""Cold CPU stage resume and actual parent-proof rejection; no simulator runs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import transformer_rl.exposure_training as exposure
from transformer_rl.frame_checkpoint import capture_rng
from transformer_rl.frame_config import digest, json_bytes
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.ppo import PPOTrainer
from test_exposure_training import config, same
from packed_env import make_env


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(configuration):
    return dict(job_id="cold_cpu_job", rollout_steps=2, training_seed=71,
        retention_seed=901, evaluation_seeds=(801, 802), device="cpu", max_seconds=60.,
        expected_initial_model_sha256=exposure.initial_model_sha256(configuration, 71))


def stage(configuration=None, name="stage_a", updates=2):
    return {"name": name, "config": configuration or config(), "updates": updates}


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size}


def canonical(path, value):
    path.write_bytes(json_bytes(value)+b"\n")


def endpoint_path(result):
    return Path(result["endpoints"][0]["checkpoint"]["path"]).parent/"endpoint.json"


def parent(tmp_path, configuration=None, factory=make_env):
    configuration = configuration or config()
    output = tmp_path/"parent"
    result = exposure.train_exposure_segment(stage(configuration), factory,
        "packed_env:make_env", output, **options(configuration))
    return configuration, result, pin(endpoint_path(result))


def next_stage(configuration, num_envs=3, name="stage_b"):
    return stage(replace(configuration, environment={**deepcopy(configuration.environment),
        "num_envs": num_envs}), name)


def rejected(output, configuration, parent_receipt, *, match=None, **overrides):
    calls = []
    values = options(configuration)
    values.update(overrides)
    before = capture_rng()
    with pytest.raises((ValueError, TypeError, KeyError, FileNotFoundError, RuntimeError), match=match):
        exposure.train_exposure_segment(next_stage(configuration),
            lambda **kwargs: calls.append(True), "packed_env:make_env", output,
            parent_endpoint=parent_receipt, **values)
    same(capture_rng(), before)
    assert calls == [] and not output.exists()


def rebind_endpoint(path, value):
    canonical(path, value)
    completion_path = path.parent.parent/"completion.json"
    completion = json.loads(completion_path.read_bytes())
    index = next(index for index, item in enumerate(completion['endpoints'])
                 if item['stage_index'] == value['stage_index'])
    completion['endpoints'][index] = value
    if index == len(completion['endpoints'])-1:
        completion["last_sealed_checkpoint"] = value["checkpoint"]
    canonical(completion_path, completion)
    return pin(path)


def rebind_payload(path, payload, request=None):
    endpoint = json.loads(path.read_bytes())
    checkpoint = Path(endpoint['checkpoint']['path'])
    torch.save(payload, checkpoint)
    endpoint['checkpoint'] = pin(checkpoint)
    sidecar_path = Path(endpoint['sidecar']['path'])
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar.update(sha256=endpoint['checkpoint']['sha256'], metadata=payload['metadata'])
    canonical(sidecar_path, sidecar)
    endpoint['sidecar']['sha256'] = pin(sidecar_path)['sha256']
    if request is not None:
        endpoint['segment_plan_sha256'] = request['sha256']
    return rebind_endpoint(path, endpoint)


def rebind_request(path, request):
    root = path.parent.parent
    request['sha256'] = digest({key:value for key,value in request.items() if key != 'sha256'})
    canonical(root/'request.json', request)
    reservation = json.loads((root/'reservation.json').read_bytes())
    reservation['job_plan_sha256'] = request['sha256']
    canonical(root/'reservation.json', reservation)
    completion = json.loads((root/'completion.json').read_bytes())
    completion.update(job_plan_sha256=request['sha256'], parent_endpoint=request['parent_endpoint'])
    canonical(root/'completion.json', completion)
    endpoint = json.loads(path.read_bytes())
    payload = torch.load(endpoint['checkpoint']['path'], map_location='cpu', weights_only=True)
    payload['metadata']['fixed_exposure_job']['job_plan_sha256'] = request['sha256']
    return rebind_payload(path, payload, request)


CHILD = r'''
import json,os,random,sys
from pathlib import Path
import numpy as np
import torch
from dataclasses import replace
import transformer_rl.exposure_training as exposure
from transformer_rl.frame_config import FrameTrainConfig
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.frame_checkpoint import capture_rng
from packed_env import make_env
torch.set_num_threads(1)
data=json.loads(sys.stdin.read())
data['stage']['config']=FrameTrainConfig.from_dict(data['stage']['config'])
original=FrameContinuation.open
report={}
def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
def noisy_factory(**kwargs):
    torch.rand(17);random.random();np.random.rand(5)
    env=make_env(**kwargs)
    reset=env.reset
    def noisy_reset(seed=None):
        torch.rand(11);random.random();np.random.rand(7)
        return reset(seed)
    env.reset=noisy_reset
    return env
def checked_open(cls,configuration,factory,reference,checkpoint,**kwargs):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    session=original(configuration,factory,reference,checkpoint,**kwargs)
    same(session.model.state_dict(),payload['model'])
    same(session.trainer.optimizer.state_dict(),payload['optimizer'])
    same(capture_rng(),payload['rng'])
    same(session.sampler.state_dict(),payload['metadata']['continuation']['retention'])
    assert session.update==kwargs['parent_update']
    assert session.consumed_updates==kwargs['consumed_updates']
    assert session.collected_transitions==kwargs['cumulative_transitions']
    history=session.collector._history.snapshot().frames
    assert torch.equal(history,history[:,:1].expand_as(history))
    report.update(pid=os.getpid(),full_learning_state_equal=True,
        episode_reset=True,history_repeat_first=True,
        inherited_update=session.update,inherited_samples=session.collected_transitions,
        Adam_state_entries=len(payload['optimizer']['state']))
    return session
FrameContinuation.open=classmethod(checked_open)
result=exposure.train_exposure_segment(data['stage'],noisy_factory,'packed_env:make_env',
    Path(data['output']),parent_endpoint=data['parent'],**data['options'])
print(json.dumps({'result':result,'report':report}))
'''


@pytest.mark.parametrize("architecture,history,readout", [
    ("mlp",1,"last"), ("history_mlp",3,"last"),
    ("transformer",3,"last"), ("transformer",5,"query")])
def test_cold_OS_stage_restores_actual_Adam_all_RNG_and_local_cumulative_clocks(
        tmp_path, architecture, history, readout):
    configuration, first, receipt = parent(tmp_path, config(architecture,history,readout))
    assert first["status"] == "completed" and first["stage_index"] == 0
    assert first["successful_updates"] == first["cumulative_successful_updates"] == 2
    assert first["actual_collected_transitions"] == first["cumulative_collected_transitions"] == 8
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES":"", "PYTHONDONTWRITEBYTECODE":"1",
        "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
        "PYTHONPATH":os.pathsep.join((str(root/"src"),str(root/"tests/fixtures")))}
    middle = next_stage(configuration)
    data = {"stage":{**middle,"config":middle["config"].to_dict()},
        "parent":receipt,"output":str(tmp_path/"cold_stage"),"options":options(configuration)}
    child = subprocess.run([sys.executable,"-B","-c",CHILD],input=json.dumps(data),
        capture_output=True,text=True,env=environment,timeout=60)
    assert child.returncode == 0, child.stderr+child.stdout
    response = json.loads(child.stdout)
    second, report = response["result"], response["report"]
    assert report["pid"] != os.getpid() and report["full_learning_state_equal"]
    assert report["Adam_state_entries"] > 0 and report["inherited_update"] == 2
    assert report["inherited_samples"] == 8 and report["history_repeat_first"]
    assert second["status"] == "completed" and second["stage_index"] == 1
    assert second["successful_updates"] == second["attempted_updates"] == 2
    assert second["actual_collected_transitions"] == second["successful_full_rollout_samples"] == 12
    assert second["cumulative_successful_updates"] == second["cumulative_attempted_updates"] == 4
    assert second["cumulative_collected_transitions"] == 20
    assert second["charged_updates"] == 2 and second["charged_fresh_transition_budget"] == 12
    assert second["whole_job_reservation_created"] is False
    assert second["charge_scope"] == "segment_itemization_of_external_whole_job_reservation"
    assert second["parent_endpoint"] == receipt
    third = exposure.train_exposure_segment(next_stage(configuration,4,"stage_c"), make_env,
        "packed_env:make_env", tmp_path/"third", parent_endpoint=pin(endpoint_path(second)), **options(configuration))
    assert third["status"] == "completed" and third["stage_index"] == 2
    assert third["actual_collected_transitions"] == 16 and third["successful_updates"] == 2
    assert third["cumulative_collected_transitions"] == 36 and third["cumulative_successful_updates"] == 6
    request = json.loads((tmp_path/"third/request.json").read_bytes())
    assert request["format"] == "transformer_rl.exposure_segment_request"
    assert request["parent_metrics"]["sha256"] == hashlib.sha256((tmp_path/"cold_stage/metrics.jsonl").read_bytes()).hexdigest()
    reservation = json.loads((tmp_path/"third/reservation.json").read_bytes())
    assert reservation["format"] == "transformer_rl.exposure_segment_reservation"
    assert reservation["charged_fresh_transition_budget"] == 16


def test_completed_legacy_fresh_job_last_stage_is_supported(tmp_path):
    configuration = config()
    first = exposure.train_exposure_job([stage(configuration)], make_env,
        "packed_env:make_env", tmp_path/"old_job", **options(configuration))
    result = exposure.train_exposure_segment(next_stage(configuration), make_env,
        "packed_env:make_env", tmp_path/"next", parent_endpoint=pin(endpoint_path(first)), **options(configuration))
    assert result["status"] == "completed" and result["stage_index"] == 1
    assert result["cumulative_successful_updates"] == 4 and result["actual_collected_transitions"] == 12


def test_completed_legacy_multistage_last_endpoint_keeps_cumulative_baseline(tmp_path):
    configuration = config()
    first = exposure.train_exposure_job([stage(configuration),next_stage(configuration)],make_env,
        'packed_env:make_env',tmp_path/'old_multistage',**options(configuration))
    last = Path(first['endpoints'][-1]['checkpoint']['path']).parent/'endpoint.json'
    result = exposure.train_exposure_segment(next_stage(configuration,4,'stage_c'),make_env,
        'packed_env:make_env',tmp_path/'new_stage',parent_endpoint=pin(last),**options(configuration))
    assert result['status'] == 'completed' and result['stage_index'] == 2
    assert result['successful_updates'] == 2 and result['actual_collected_transitions'] == 16
    assert result['cumulative_successful_updates'] == 6 and result['cumulative_collected_transitions'] == 36


@pytest.mark.parametrize("failure", ["close", "fsync"])
def test_real_checkpoint_from_failed_shutdown_or_sync_is_not_an_eligible_parent(tmp_path, monkeypatch, failure):
    def factory(**kwargs):
        env = make_env(**kwargs)
        if failure == "close":
            env.close = lambda: (_ for _ in ()).throw(RuntimeError("shutdown failed"))
        return env
    if failure == "fsync":
        copied_os = SimpleNamespace(**vars(exposure.os))
        copied_os.fsync = lambda *_: (_ for _ in ()).throw(OSError("sync failed"))
        monkeypatch.setattr(exposure,"os",copied_os)
    configuration, result, receipt = parent(tmp_path,factory=factory)
    assert result["status"] == "failed" and Path(result["endpoints"][0]["checkpoint"]["path"]).is_file()
    rejected(tmp_path/"forbidden",configuration,receipt)


@pytest.mark.parametrize("override", [
    {"job_id":"another_job"},{"training_seed":72},{"retention_seed":902},
    {"evaluation_seeds":(803,804)},{"rollout_steps":3},{"device":"cuda:0"}])
def test_cross_job_seed_recipe_and_runtime_rejected_before_output(tmp_path, override):
    configuration, _, receipt = parent(tmp_path)
    rejected(tmp_path/"other",configuration,receipt,**override)


@pytest.mark.parametrize("mutation", ["sha","bytes","bool_bytes","alias","symlink","fields"])
def test_parent_external_receipt_requires_exact_canonical_actual_file(tmp_path,mutation):
    configuration, _, receipt = parent(tmp_path)
    if mutation == "sha":receipt["sha256"] = "0"*64
    elif mutation == "bytes":receipt["bytes"] += 1
    elif mutation == "bool_bytes":receipt["bytes"] = True
    elif mutation == "alias":receipt["path"] = str(Path(receipt["path"]).parent/".."/Path(receipt["path"]).parent.name/"endpoint.json")
    elif mutation == "symlink":
        link=tmp_path/"endpoint.json";link.symlink_to(receipt["path"]);receipt["path"]=str(link)
    else:receipt["eligible"] = True
    rejected(tmp_path/"bad_receipt",configuration,receipt)


@pytest.mark.parametrize("mutation", ["status","shutdown","unverified","unsealed","duplicate","last","metrics_count","charged","reservation","schema"])
def test_parent_completion_and_reservation_must_prove_clean_full_budget(tmp_path,mutation):
    configuration, _, receipt = parent(tmp_path)
    path=tmp_path/"parent/completion.json";value=json.loads(path.read_bytes())
    if mutation=="status":value["status"]="partial_rollout"
    elif mutation=="shutdown":value["shutdown_errors"]=[{"type":"RuntimeError"}]
    elif mutation=="unverified":value["unverified_or_unoptimized_samples"]=1
    elif mutation=="unsealed":value["unsealed_checkpoint_paths"]=["actual_cp_exists"]
    elif mutation=="duplicate":value["endpoints"].append(deepcopy(value["endpoints"][0]))
    elif mutation=="last":value["last_sealed_checkpoint"]["sha256"]="0"*64
    elif mutation=="metrics_count":value["recorded_metric_updates"]=1
    elif mutation=="charged":value["charged_updates"]=1
    elif mutation=="schema":value["schema_version"]=True
    else:
        reservation=tmp_path/"parent/reservation.json";v=json.loads(reservation.read_bytes());v["charged_updates"]=99;canonical(reservation,v)
    canonical(path,value)
    rejected(tmp_path/"bad_completion",configuration,receipt)


@pytest.mark.parametrize("mutation", ["missing","duplicate","batch","clock","partial","optimization"])
def test_actual_parent_metrics_are_verified_not_only_completion_labels(tmp_path,mutation):
    configuration, _, receipt = parent(tmp_path)
    path=tmp_path/"parent/metrics.jsonl";rows=[json.loads(line) for line in path.read_bytes().splitlines()]
    if mutation=="missing":rows.pop()
    elif mutation=="duplicate":rows[1]=deepcopy(rows[0])
    elif mutation=="batch":rows[1]["batch_samples"]=3
    elif mutation=="clock":rows[1]["cumulative_transitions"]=7
    elif mutation=="optimization":rows[1]["optimization"]["sample_count"]+=1
    raw=b"".join(json_bytes(row)+b"\n" for row in rows)
    path.write_bytes(raw[:-1] if mutation=="partial" else raw)
    rejected(tmp_path/"bad_metrics",configuration,receipt)


@pytest.mark.parametrize("mutation", ["legacy","guard","source","private_seed","segment_partial","missing_rng","Adam"])
def test_actual_CPU_checkpoint_metadata_and_state_are_bound_to_exposure_origin(tmp_path,mutation):
    configuration, _, receipt = parent(tmp_path)
    ep=Path(receipt["path"]);value=json.loads(ep.read_bytes());cp=Path(value["checkpoint"]["path"])
    payload=torch.load(cp,map_location="cpu",weights_only=True)
    if mutation=="legacy":payload["metadata"].pop("fixed_exposure_job")
    elif mutation=="guard":payload["metadata"]["fixed_exposure_job"]["initial_model_sha256"]="0"*64
    elif mutation=="source":payload["metadata"]["source"]={"sha256":"0"*64,"files":{}}
    elif mutation=="private_seed":payload["metadata"]["continuation"]["retention"]["seed"]=902
    elif mutation=="segment_partial":payload["metadata"]["continuation_segment"]["discarded_transitions"]=1
    elif mutation=="missing_rng":payload.pop("rng")
    else:payload["optimizer"]["state"]={}
    torch.save(payload,cp);value["checkpoint"]=pin(cp)
    sidecar=Path(value["sidecar"]["path"]);s=json.loads(sidecar.read_bytes())
    s["sha256"]=value["checkpoint"]["sha256"]
    if "metadata" in payload:s["metadata"]=payload["metadata"]
    canonical(sidecar,s);value["sidecar"]["sha256"]=pin(sidecar)["sha256"]
    receipt=rebind_endpoint(ep,value)
    rejected(tmp_path/"bad_CPU_state",configuration,receipt)


def test_parent_sidecar_with_valid_actual_hash_still_must_equal_CPU_payload(tmp_path):
    configuration, _, receipt = parent(tmp_path)
    path=Path(receipt["path"]);endpoint=json.loads(path.read_bytes());sidecar=Path(endpoint["sidecar"]["path"])
    value=json.loads(sidecar.read_bytes());value["metadata"]["seed"]=999;canonical(sidecar,value)
    endpoint["sidecar"]["sha256"]=pin(sidecar)["sha256"]
    receipt=rebind_endpoint(path,endpoint)
    rejected(tmp_path/"bad_sidecar",configuration,receipt)


@pytest.mark.parametrize('mutation', ['endpoint_sha','request_pin','missing_chain','index',
    'ancestor_failed','continuation_parent','stage_transition'])
def test_actual_ancestor_is_bound_to_parent_request_and_learning_transition(tmp_path, mutation):
    configuration, first, first_receipt = parent(tmp_path)
    second = exposure.train_exposure_segment(next_stage(configuration),make_env,
        'packed_env:make_env',tmp_path/'middle',parent_endpoint=first_receipt,**options(configuration))
    path = endpoint_path(second)
    request = json.loads((tmp_path/'middle/request.json').read_bytes())
    if mutation in ('continuation_parent','stage_transition'):
        payload = torch.load(second['endpoints'][0]['checkpoint']['path'],map_location='cpu',weights_only=True)
        if mutation == 'continuation_parent':payload['metadata']['continuation_parent']['sha256'] = '0'*64
        else:payload['metadata']['stage_transition']['parent_environment_sha256'] = '0'*64
        receipt = rebind_payload(path,payload)
    else:
        if mutation == 'endpoint_sha':request['parent_endpoint']['sha256'] = '0'*64
        elif mutation == 'request_pin':request['parent_request']['sha256'] = '0'*64
        elif mutation == 'missing_chain':request['parent_endpoint'] = None
        elif mutation == 'index':request['stage_index'] = True
        else:
            ancestor = tmp_path/'parent/completion.json'
            value = json.loads(ancestor.read_bytes());value['status'] = 'failed';canonical(ancestor,value)
            request['parent_completion'] = pin(ancestor)
        receipt = rebind_request(path,request)
    rejected(tmp_path/'invalid_ancestor',configuration,receipt)


def test_segment_cannot_write_inside_its_parent_source_tree(tmp_path):
    configuration, _, receipt = parent(tmp_path)
    rejected(tmp_path/"parent/new_segment",configuration,receipt)


def test_earlier_endpoint_from_completed_multistage_job_cannot_silently_roll_back(tmp_path):
    configuration=config()
    result=exposure.train_exposure_job([stage(configuration),next_stage(configuration)],make_env,
        "packed_env:make_env",tmp_path/"full_job",**options(configuration))
    first=Path(result["endpoints"][0]["checkpoint"]["path"]).parent/"endpoint.json"
    rejected(tmp_path/"rollback",configuration,pin(first))


def test_failed_PPO_has_local_attempts_samples_and_cumulative_parent_clocks(tmp_path,monkeypatch):
    configuration, first, receipt=parent(tmp_path)
    original=PPOTrainer.update
    observed=[]
    def partially_failed(self,batch,**kwargs):
        value=original(self,batch,**kwargs);observed.append(value)
        raise RuntimeError("Adam changed before stage failure")
    monkeypatch.setattr(PPOTrainer,"update",partially_failed)
    result=exposure.train_exposure_segment(next_stage(configuration),make_env,
        "packed_env:make_env",tmp_path/"failed_stage",parent_endpoint=receipt,**options(configuration))
    assert result["status"]=="failed" and len(observed)==1
    assert result["successful_updates"]==0 and result["attempted_updates"]==1
    assert result["actual_collected_transitions"]==result["unverified_or_unoptimized_samples"]==6
    assert result["successful_full_rollout_samples"]==result["optimizer_steps"]==0
    assert result["failed_update_optimizer_steps"] is None and result["optimizer_update_may_be_partial"]
    assert result["cumulative_successful_updates"]==2 and result["cumulative_attempted_updates"]==3
    assert result["cumulative_collected_transitions"]==14
    assert result["charged_updates"]==2 and result["charged_fresh_transition_budget"]==12
    assert result["endpoints"]==[] and result["parent_endpoint"]==receipt
    rejected(tmp_path/"retry_from_failed",configuration,
        {"path":str(tmp_path/"failed_stage/stage_0001_stage_b/endpoint.json"),"sha256":"0"*64,"bytes":1})


@pytest.mark.parametrize('option', ['lr','betas','eps','weight_decay','maximize','foreach','amsgrad'])
def test_actual_Adam_options_cannot_override_the_frozen_learning_recipe(tmp_path, option):
    configuration, result, receipt = parent(tmp_path)
    payload = torch.load(result['endpoints'][0]['checkpoint']['path'],map_location='cpu',weights_only=True)
    group = payload['optimizer']['param_groups'][0]
    alternatives = {'lr':0., 'betas':(.8,.99), 'eps':1e-3, 'weight_decay':.01,
                    'maximize':True, 'foreach':True, 'amsgrad':True}
    assert group[option] != alternatives[option]
    group[option] = alternatives[option]
    if option == 'amsgrad':
        for state in payload['optimizer']['state'].values():
            state['max_exp_avg_sq'] = state['exp_avg_sq'].clone()
    receipt = rebind_payload(Path(receipt['path']),payload)
    rejected(tmp_path/'different_Adam_recipe',configuration,receipt,
             match='actual Adam options differ')


@pytest.mark.parametrize('mutation', ['one_step','all_zero_steps','missing_parameter'])
def test_actual_Adam_each_parameter_step_matches_all_verified_PPO_steps(tmp_path, mutation):
    configuration, result, receipt = parent(tmp_path)
    payload = torch.load(result['endpoints'][0]['checkpoint']['path'],map_location='cpu',weights_only=True)
    states = payload['optimizer']['state']
    assert len(states) > 1 and all(state['step'].item() > 0 for state in states.values())
    if mutation == 'one_step':
        next(iter(states.values()))['step'].add_(1)
    elif mutation == 'all_zero_steps':
        for state in states.values():
            state['step'].zero_()
    else:
        states.pop(next(iter(states)))
    receipt = rebind_payload(Path(receipt['path']),payload)
    rejected(tmp_path/'wrong_Adam_steps',configuration,receipt,match='actual Adam')


@pytest.mark.parametrize('mutation', ['zero_steps','one_fewer_step'])
def test_self_consistent_metrics_and_completion_cannot_replace_actual_Adam_learning(tmp_path, mutation):
    configuration, _, receipt = parent(tmp_path)
    metrics = tmp_path/'parent/metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_bytes().splitlines()]
    changed = rows if mutation == 'zero_steps' else rows[:1]
    for row in changed:
        optimization = row['optimization']
        assert optimization['optimizer_steps'] > 0
        steps = 0 if mutation == 'zero_steps' else optimization['optimizer_steps']-1
        chunks = min(configuration.ppo.num_minibatches,row['batch_samples'])
        epochs, prefix = divmod(steps,chunks)
        optimization['optimizer_steps'] = steps
        optimization['sample_count'] = (epochs*row['batch_samples']
            + prefix*(row['batch_samples']//chunks) + min(prefix,row['batch_samples']%chunks))
        optimization['early_stopped'] = steps < optimization['planned_optimizer_steps']
    metrics.write_bytes(b''.join(json_bytes(row)+b'\n' for row in rows))
    path = tmp_path/'parent/completion.json'
    completion = json.loads(path.read_bytes())
    completion['optimizer_steps'] = sum(row['optimization']['optimizer_steps'] for row in rows)
    completion['optimization_sample_uses'] = sum(row['optimization']['sample_count'] for row in rows)
    canonical(path,completion)
    # The actual checkpoint and its external endpoint receipt remain unchanged.
    rejected(tmp_path/'fabricated_log',configuration,receipt,match='actual Adam')


def test_self_consistent_sample_uses_must_equal_the_actual_uneven_minibatch_prefix(tmp_path):
    configuration = config()
    configuration = replace(configuration,ppo=replace(configuration.ppo,num_minibatches=3))
    configuration, _, receipt = parent(tmp_path,configuration)
    path = tmp_path/'parent/metrics.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert rows[0]['batch_samples'] == 4
    assert rows[0]['optimization']['sample_count'] > 0
    rows[0]['optimization']['sample_count'] -= 1
    path.write_bytes(b''.join(json_bytes(row)+b'\n' for row in rows))
    completion_path = tmp_path/'parent/completion.json'
    completion = json.loads(completion_path.read_bytes())
    completion['optimization_sample_uses'] = sum(row['optimization']['sample_count'] for row in rows)
    canonical(completion_path,completion)
    rejected(tmp_path/'invalid_prefix',configuration,receipt,match='minibatch prefix')


@pytest.mark.parametrize('mutation', ['fresh_parent','fresh_transition','last_parent','last_transition',
    'earlier_sidecar','missing_earlier_endpoint','earlier_Adam_steps','earlier_config'])
def test_legacy_multistage_actual_checkpoints_and_every_stage_origin_are_verified(tmp_path,mutation):
    configuration = config()
    result = exposure.train_exposure_job([stage(configuration),next_stage(configuration)],make_env,
        'packed_env:make_env',tmp_path/'legacy',**options(configuration))
    assert result['status'] == 'completed'
    first_path = Path(result['endpoints'][0]['checkpoint']['path']).parent/'endpoint.json'
    last_path = Path(result['endpoints'][-1]['checkpoint']['path']).parent/'endpoint.json'
    if mutation == 'missing_earlier_endpoint':
        first_path.unlink()
    elif mutation == 'earlier_sidecar':
        endpoint = json.loads(first_path.read_bytes())
        path = Path(endpoint['sidecar']['path'])
        value = json.loads(path.read_bytes())
        value['metadata']['seed'] += 1
        canonical(path,value)
        endpoint['sidecar']['sha256'] = pin(path)['sha256']
        rebind_endpoint(first_path,endpoint)
    else:
        selected_path = first_path if mutation.startswith('fresh_') or mutation.startswith('earlier_') else last_path
        endpoint = json.loads(selected_path.read_bytes())
        payload = torch.load(endpoint['checkpoint']['path'],map_location='cpu',weights_only=True)
        if mutation == 'fresh_parent':
            payload['metadata']['continuation_parent'] = {
                'path':endpoint['checkpoint']['path'],'sha256':endpoint['checkpoint']['sha256'],
                'update':0,'resume':True}
        elif mutation == 'fresh_transition':
            payload['metadata']['stage_transition'] = {'environment_transition':True,
                'parent_environment_sha256':'0'*64,'environment_sha256':'0'*64}
        elif mutation == 'last_parent':
            payload['metadata']['continuation_parent']['sha256'] = '0'*64
        elif mutation == 'last_transition':
            payload['metadata']['stage_transition']['parent_environment_sha256'] = '0'*64
        elif mutation == 'earlier_Adam_steps':
            next(iter(payload['optimizer']['state'].values()))['step'].add_(1)
        else:
            payload['config']['environment']['num_envs'] = 9
        rebind_payload(selected_path,payload)
        if selected_path == first_path:
            # Keep the later actual checkpoint chain bound to the modified
            # earlier SHA, so rejection must inspect the earlier payload.
            first_endpoint = json.loads(first_path.read_bytes())
            last_endpoint = json.loads(last_path.read_bytes())
            last_payload = torch.load(last_endpoint['checkpoint']['path'],map_location='cpu',weights_only=True)
            last_payload['metadata']['continuation_parent']['sha256'] = first_endpoint['checkpoint']['sha256']
            rebind_payload(last_path,last_payload)
    rejected(tmp_path/'invalid_legacy_origin',configuration,pin(last_path))


def test_legacy_single_stage_checkpoint_must_have_a_fresh_origin(tmp_path):
    configuration = config()
    result = exposure.train_exposure_job([stage(configuration)],make_env,
        'packed_env:make_env',tmp_path/'legacy',**options(configuration))
    path = endpoint_path(result)
    endpoint = json.loads(path.read_bytes())
    payload = torch.load(endpoint['checkpoint']['path'],map_location='cpu',weights_only=True)
    payload['metadata']['continuation_parent'] = {
        'path':endpoint['checkpoint']['path'],'sha256':endpoint['checkpoint']['sha256'],
        'update':0,'resume':True}
    rejected(tmp_path/'nonfresh_legacy',configuration,rebind_payload(path,payload),
             match='fresh parent checkpoint')

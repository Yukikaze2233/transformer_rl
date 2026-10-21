"""Exact learning-state comparisons with diagnostic accumulation on and off."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from test_policy_optimization import assert_identical, batch_for, model
from transformer_rl.formal.learning.optimization import (
    StructuredPPOConfig, StructuredPPOTrainer, _LossTelemetry,
)
from transformer_rl.formal.learning.contracts import ObservationWindows


@pytest.fixture(autouse=True)
def owned_rng_and_cpu_threads():
    previous = torch.get_num_threads()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch.set_num_threads(1)
    devices = ([0] if os.environ.get('RUN_FORMAL_CUDA_PARITY') == '1'
               and torch.cuda.is_available() else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2102)
        yield
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_num_threads(previous)


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(**{field.name: _to_device(getattr(value, field.name), device)
                              for field in fields(value)})
    if isinstance(value, ObservationWindows):
        windows = deepcopy(value)
        for name in ('_raw', '_indices', '_environments'):
            setattr(windows, name, getattr(windows, name).to(device))
        return windows
    return value


def _reference_trainer(network, config):
    source = os.environ.get('FORMAL_PPO_BASELINE_SOURCE')
    if source is None:
        return StructuredPPOTrainer(network, config, loss_telemetry=False)
    # The coordinator can compare against the pinned pre-change module rather
    # than the disabled path, without copying optimizer algebra into a test.
    expected = os.environ.get('FORMAL_PPO_BASELINE_SHA256')
    raw = Path(source).read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('original PPO source pin is required for baseline execution')
    name = 'transformer_rl.formal.learning._telemetry_baseline'
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module.StructuredPPOTrainer(network, config)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('training_chunk', [3, 3072])
@pytest.mark.parametrize('rollback', ['none', 'minibatch', 'full'])
def test_loss_telemetry_preserves_model_gradients_two_adams_actions_rng_and_consumption(
        device, training_chunk, rollback):
    if device == 'cuda' and os.environ.get('RUN_FORMAL_CUDA_PARITY') != '1':
        pytest.skip('CUDA parity requires explicit Kaiser resource-owner coordination')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    base = model()
    batch = batch_for(base)
    base = base.to(device)
    batch = _to_device(batch, device)
    config = replace(StructuredPPOConfig(), epochs=1, minibatches=2,
                     learning_rate=1e-7, diagnostics_interval=1,
                     training_chunk=training_chunk, probe_endpoints=8)
    left = _reference_trainer(deepcopy(base), config)
    right = StructuredPPOTrainer(deepcopy(base), config, loss_telemetry=True)
    for trainer in (left, right):
        original = trainer._kl
        calls = {'count': 0}

        def guarded(batch, endpoints, *, original=original, calls=calls):
            calls['count'] += 1
            reject = ((rollback == 'minibatch' and calls['count'] == 1)
                      or (rollback == 'full' and endpoints.numel() == len(batch)
                          and calls['count'] == 3))
            return batch.old_log_prob.new_tensor(.03) if reject else original(batch, endpoints)

        trainer._kl = guarded
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    snapshots, reports = [], []
    for trainer in (left, right):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        reports.append(trainer.update(batch))
        with torch.no_grad():
            means = trainer.model.actor.policy(batch.windows.gather(torch.arange(len(batch), device=device)))
            action = means + torch.randn_like(means) * trainer.model.actor.log_std.exp()
        snapshots.append({
            'model': deepcopy(trainer.model.state_dict()),
            'gradients': {name: None if parameter.grad is None else parameter.grad.detach().clone()
                          for name, parameter in trainer.model.named_parameters()},
            'actor_adam': deepcopy(trainer.actor_optimizer.state_dict()),
            'critic_adam': deepcopy(trainer.critic_optimizer.state_dict()),
            'actions': action, 'means': means,
            'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
            'consumed': dict(trainer.consumed),
            'python_rng': random.getstate(),
        })
        actual_numpy_rng = np.random.get_state()
        assert actual_numpy_rng[0] == numpy_rng[0]
        assert np.array_equal(actual_numpy_rng[1], numpy_rng[1])
        assert actual_numpy_rng[2:] == numpy_rng[2:]
    assert_identical(snapshots[0], snapshots[1])
    counters = ('fresh_endpoints', 'actor_optimizer_steps', 'critic_optimizer_steps',
                'actor_endpoint_uses', 'critic_endpoint_uses', 'retained_actor_steps',
                'rolled_back_actor_steps', 'whole_actor_update_rolled_back', 'actor_stop_reason')
    for name in counters:
        assert reports[0][name] == reports[1][name]
    assert 'loss_statistics' not in reports[0]
    statistics = reports[1]['loss_statistics']
    assert statistics['actor_attempted']['surrogate']['sample_uses'] == reports[1]['actor_endpoint_uses']
    if rollback != 'none':
        assert statistics['actor_retained']['surrogate']['sample_uses'] == 0
        assert statistics['actor_retained']['surrogate']['mean'] is None


def test_absent_auxiliary_labels_are_not_zero_loss_curves():
    network = model()
    batch = batch_for(network)
    absent = torch.zeros_like(batch.masks.state_auxiliary)
    batch = replace(batch, masks=replace(batch.masks, state_auxiliary=absent, next_auxiliary=absent))
    trainer = StructuredPPOTrainer(network, replace(StructuredPPOConfig(), epochs=1, minibatches=2,
                                                   learning_rate=1e-7, training_chunk=3))
    statistics = trainer.update(batch)['loss_statistics']
    for scope in ('actor_attempted', 'actor_retained'):
        for name in ('velocity', 'contact', 'prediction'):
            assert statistics[scope][name] == {'mean': None, 'sample_uses': 0}


def test_loss_summaries_have_explicit_weighting_and_one_small_host_transfer(monkeypatch):
    observer = _LossTelemetry(torch.zeros(()))
    observer.begin_actor()
    observer.observe_actor(torch.tensor([1., 3.]), torch.tensor([2., 2.]),
                           (torch.tensor(4.), torch.tensor(6.), torch.tensor(0.)),
                           torch.tensor(10.), torch.tensor([True, False]),
                           torch.tensor([False, False]), None)
    observer.finish_actor(retained=False)
    observer.observe_critic(torch.tensor(3.), 2, 4.)
    original_cpu, calls = torch.Tensor.cpu, []

    def counted_cpu(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', counted_cpu)
    summary = observer.summary()
    assert calls == [28]
    assert summary['actor_attempted']['surrogate'] == {'mean': 2., 'sample_uses': 2}
    assert summary['actor_attempted']['velocity'] == {'mean': 4., 'sample_uses': 1}
    assert summary['actor_attempted']['prediction'] == {'mean': None, 'sample_uses': 0}
    assert summary['actor_retained']['surrogate'] == {'mean': None, 'sample_uses': 0}
    assert summary['critic_attempted']['clipped_value_mse'] == {'mean': 3., 'sample_uses': 2}
    assert summary['critic_attempted']['weighted_objective'] == {'mean': 6., 'sample_uses': 2}

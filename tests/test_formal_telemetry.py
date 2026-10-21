"""Telemetry has no budget authority and never needs an SDK or old log reader."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from transformer_rl.formal.runtime.telemetry import RolloutTelemetry, TensorBoardTelemetry


@pytest.fixture(autouse=True)
def owned_rng_and_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(previous)


class Writer:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.scalars, self.texts, self.closed = [], [], False

    def add_scalar(self, name, value, *, global_step):
        self.scalars.append((name, value, global_step))

    def add_text(self, name, value, *, global_step):
        self.texts.append((name, value, global_step))

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_formal_steps_use_actual_budget_and_resume_never_purges(tmp_path):
    row = {'update': 288, 'budget_update': 330, 'collection': {'reward_mean': 1.2},
           'optimization': {'full_rollout_kl_after_guard': .001}}
    before = deepcopy(row)
    with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=Writer) as telemetry:
        writer = telemetry.writer
        telemetry.record(row)
        assert ('reward/learning_step_mean', 1.2, 330) in writer.scalars
        assert ('progress/update', 288, 330) in writer.scalars
        with pytest.raises(ValueError, match='actual budget_update'):
            telemetry.record(row)
    assert row == before and writer.closed
    assert Path(writer.arguments['log_dir']) == tmp_path / 'formal'
    assert 'purge_step' not in writer.arguments
    with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=Writer) as resumed:
        resumed.record({**row, 'budget_update': 337})
        assert all(step == 337 for _, _, step in resumed.writer.scalars)


def test_test_scope_has_local_steps_and_no_formal_progress_curve(tmp_path):
    row = {'update': 289, 'budget_update': 331, 'consumption': {'ppo_updates': 289},
           'total_consumption': {'reserved': 331}, 'collection': {'reward_mean': .2}}
    with TensorBoardTelemetry(tmp_path, scope='test', writer_factory=Writer) as telemetry:
        telemetry.record(row)
        telemetry.record(row)
        curves = telemetry.writer.scalars
        assert ('reward/learning_step_mean', .2, 1) in curves
        assert ('reward/learning_step_mean', .2, 2) in curves
        assert ('telemetry/formal_budget_credit', 0, 2) in curves
        assert not any('budget_update' in name or name.startswith(('consumption/', 'total_consumption/'))
                       for name, _, _ in curves)
        assert Path(telemetry.writer.arguments['log_dir']) == tmp_path / 'test'
    assert row['budget_update'] == 331


def test_none_loss_and_missing_reward_reasons_are_text_not_zero(tmp_path):
    row = {'budget_update': 1, 'collection': {'telemetry': {
        'missing_reward_terms': {'contact_slip': 'No contact-point velocity producer'},
        'reward_density_per_s': {'height': 2.}}},
        'optimization': {'loss_statistics': {'actor_attempted': {'velocity': {'mean': None, 'sample_uses': 0}}}}}
    with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=Writer) as telemetry:
        telemetry.record(row)
        assert any('contact_slip' in name and 'producer' in value
                   for name, value, _ in telemetry.writer.texts)
        assert not any('contact_slip' in name for name, _, _ in telemetry.writer.scalars)
        assert not any(name.endswith('velocity/mean') for name, _, _ in telemetry.writer.scalars)


def test_writer_failure_does_not_mask_committed_update(tmp_path, capsys):
    class BrokenWriter(Writer):
        def add_scalar(self, *args, **kwargs):
            raise OSError('event disk unavailable')
    with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=BrokenWriter) as telemetry:
        writer = telemetry.writer
        telemetry.record({'budget_update': 1, 'collection': {'reward_mean': 1.}})
        assert telemetry.writer is None and writer.closed
    assert 'TENSORBOARD_DISABLED OSError' in capsys.readouterr().err


def test_rollout_density_averages_exclude_missing_placeholders_and_own_aliases():
    observer = RolloutTelemetry(2, 'cpu', missing_terms={'contact_slip': 'unavailable'})
    height = torch.tensor([1., 3.])
    observer.observe_rewards({'height': height, 'bounded_cost.contact_slip': torch.zeros(2)})
    height.fill_(5.)
    observer.observe_rewards({'height': height, 'bounded_cost.contact_slip': torch.zeros(2)})
    report = observer.summary()
    assert report['reward_density_per_s'] == {'height': 3.5}
    assert report['reward_observed_samples'] == 4
    assert report['missing_reward_terms'] == {'contact_slip': 'unavailable'}


def test_rollout_occupancy_is_actual_pre_reset_exposure_and_unknown_is_explicit():
    observer = RolloutTelemetry(2, 'cpu')
    family, mode = torch.tensor([0, 5]), torch.tensor([0, 4])
    observer.observe_occupancy(family, mode)
    family.fill_(3)
    mode[:] = torch.tensor([1, 99])
    observer.observe_occupancy(family, mode)
    report = observer.summary()
    assert report['task_environment_step_fraction']['terrain'] == .5
    assert report['task_environment_step_fraction']['recovery'] == .25
    assert report['command_mode_before_step_fraction']['stair'] == .25
    assert report['command_mode_before_step_fraction']['jump'] == .25
    assert report['command_mode_before_step_fraction']['unknown'] == .25


def test_nonfinite_density_is_unavailable_and_does_not_turn_into_zero():
    observer = RolloutTelemetry(2, 'cpu')
    observer.observe_rewards({'height': torch.tensor([1., float('nan')])})
    report = observer.summary()
    assert report['reward_density_per_s']['height'] is None
    assert report['reward_nonfinite_samples']['reward_density_per_s/height'] == 1


def test_reward_events_raw_costs_and_gates_keep_their_units_and_one_host_transfer(monkeypatch):
    observer = RolloutTelemetry(2, 'cpu', missing_terms={'contact_slip': 'unavailable'})
    observer.observe_rewards({'bounded_cost.effort': torch.tensor([-2., -4.])}, series={
        'raw_cost_density_per_s': {'effort': torch.tensor([20., 40.]), 'contact_slip': torch.zeros(2)},
        'event_step_reward': {'failure': torch.tensor([-4., 0.])},
        'gate_fraction': {'jump': torch.tensor([1., 0.])},
    })
    original_cpu, copies = torch.Tensor.cpu, []

    def counted_cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', counted_cpu)
    report = observer.summary()
    assert len(copies) == 1
    assert report['reward_density_per_s']['bounded_cost.effort'] == -3.
    assert report['raw_cost_density_per_s'] == {'effort': 30.}
    assert report['event_step_reward']['failure'] == -2.
    assert report['gate_fraction']['jump'] == .5


def test_worker_flushes_current_rows_before_the_sdk_exit_boundary(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import fcntl
    import os

    from test_structured_collection import SyntheticLabeledEnv
    from test_structured_learning import attempt, command, components
    from transformer_rl.formal.runtime import process

    events, writers = [], []

    class OrderedWriter(Writer):
        def close(self):
            events.append('telemetry_closed')
            super().close()

    def writer_factory(**kwargs):
        writer = OrderedWriter(**kwargs)
        writers.append(writer)
        return writer

    monkeypatch.setattr(process, 'TensorBoardTelemetry',
                        lambda *args, **kwargs: TensorBoardTelemetry(
                            *args, **kwargs, writer_factory=writer_factory))
    root = tmp_path / 'owned_test_run'
    root.mkdir()
    descriptor = os.open(root / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe, _ = components()
    directory = attempt(root, 'attempt_1', descriptor)
    (directory / 'train/checkpoints').rmdir()
    env = SyntheticLabeledEnv()
    original_step = env.step

    def step(*args):
        returned = original_step(*args)
        if env.counter == recipe.rollout_steps:
            command(directory, 'stop', 1)
        return returned

    env.step = step

    @contextmanager
    def runtime(*_args, **_kwargs):
        try:
            yield env
        finally:
            events.append('sdk_closed')

    try:
        report = process.run_worker(recipe, directory / 'control', runtime, {},
                                    tensorboard=True, telemetry_scope='test')
        assert report['final_update'] == 1
        assert events == ['telemetry_closed', 'sdk_closed']
        assert writers[0].scalars and all(value[2] == 1 for value in writers[0].scalars)
        assert Path(writers[0].arguments['log_dir']).name == 'test'
    finally:
        os.close(descriptor)

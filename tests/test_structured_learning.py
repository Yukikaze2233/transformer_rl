"""Owned CPU worker/control tests; no native robot or production namespace."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest
import torch

from test_policy_optimization import assert_identical
from test_structured_collection import SyntheticLabeledEnv, model as small_model
from transformer_rl.formal.learning.accounting import LearningJournal, RecoveryConsumptionMismatch
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.checkpoint import LearningComponents, LearningRecipe, StructuredCheckpointStore
from transformer_rl.formal.learning.collection import StructuredCollectionConfig, StructuredCollector
from transformer_rl.formal.learning.loop import StageValidationRequired, StructuredLearningLoop, load_stopped_attempt
from transformer_rl.formal.tasks.curriculum import ContinuousTaskSampler, CurriculumBudget, TaskDomain
from transformer_rl.formal.tasks.protocol import RequestProtocol, RequestProtocolConfig
from transformer_rl.formal.tasks.terrain import ContinuousTerrainGenerator, TerrainTuning
from transformer_rl.research.frame_checkpoint import capture_rng, restore_rng
from transformer_rl.common.training_control import PROTOCOL, TrainingControl, boot_id, file_pin, process_identity, read_json, write_json


class StopBudget:
    def __init__(self):
        self.deadline = time.monotonic() + 30
        self.reason = None

    def stopped(self):
        if self.reason is None and time.monotonic() >= self.deadline:
            self.reason = 'time_budget'
        return self.reason is not None


def components(promotion='exposure', checkpoint_interval=25):
    network = small_model()
    ppo = replace(StructuredPPOConfig(), epochs=1, minibatches=1, probe_endpoints=16,
                  evaluation_chunk=32, learning_rate=1e-6)
    terrain = TerrainTuning(.7, (.2, 1.1), .25)
    recipe = LearningRecipe('control_fixture', 2101, 2, 16, 'a' * 64, 'b' * 64, 'c' * 64,
                            StructuredCheckpointStore.model_sha256(network), network.config, ppo,
                            StructuredCollectionConfig(), TaskDomain(), RequestProtocolConfig(), terrain,
                            curriculum_promotion=promotion, checkpoint_interval=checkpoint_interval)
    state = LearningComponents(network, StructuredPPOTrainer(network, ppo), CurriculumBudget(promotion=promotion),
                               ContinuousTaskSampler(2101), ContinuousTerrainGenerator(2101, terrain), RequestProtocol(2))
    return recipe, state


def attempt(root, name, descriptor, *, operation='resume', start=0, spent=0., target=30000):
    directory = root / name
    (directory / 'control').mkdir(parents=True)
    (directory / 'train' / 'checkpoints').mkdir(parents=True)
    token = uuid.uuid4().hex
    binding = {'protocol': PROTOCOL, 'token': token, 'boot_id': boot_id(), 'root': str(root),
               'attempt': str(directory), 'lease_fd': descriptor, 'start_update': start,
               'target_update': target, 'prior_transitions': start * 32, 'prior_active_elapsed_s': spent}
    write_json(directory / 'request.json', binding)
    command(directory, operation, 0)
    return directory


def command(directory, operation, sequence):
    token = read_json(directory / 'request.json')['token']
    write_json(directory / 'control' / 'request.json', {'token': token, 'sequence': sequence,
               'operation': operation}, replace=True)


@contextmanager
def fixture(tmp_path, promotion='exposure', checkpoint_interval=25):
    root = tmp_path / 'session'
    root.mkdir()
    descriptor = os.open(root / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe, state = components(promotion, checkpoint_interval)
    directory = attempt(root, 'attempt_1', descriptor, target=recipe.journal_spec().target_rollouts)
    control = TrainingControl(directory / 'control')
    with LearningJournal.create(root / 'journal', recipe.journal_spec(), authorized_boot=boot_id()) as journal:
        store = StructuredCheckpointStore(directory / 'train' / 'checkpoints', recipe, journal)
        env = SyntheticLabeledEnv()
        collector = StructuredCollector(env, state.model, recipe.collection)
        collector.reset()
        loop = StructuredLearningLoop(recipe, state, collector, journal, store, control, StopBudget())
        try:
            yield loop, env, directory
        finally:
            os.close(descriptor)


@pytest.fixture(autouse=True)
def rng_and_threads():
    before, threads = capture_rng(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(2101)
    yield
    restore_rng(before)
    torch.set_num_threads(threads)


def test_reservation_is_published_before_first_real_collection_step(tmp_path):
    with fixture(tmp_path) as (loop, env, directory):
        step = env.step

        def checked_step(action, estimate):
            assert loop.journal.summary()['pending'] == {'index': 1, 'stage': 0}
            assert loop.components.curriculum.consumed[0] == 1
            assert (loop.journal.directory / 'events' / '00000000.json').is_file()
            return step(action, estimate)

        env.step = checked_step
        loop.run_update()
        state = loop.journal.checkpoint_boundary()
        assert state['rollouts_reserved'] == state['ppo_updates'] == 1
        assert state['reserved_transitions'] == state['returned_transitions'] == 32
        assert state['actor_optimizer_steps'] == state['critic_optimizer_steps'] == 1
        assert loop.components.returned_vector_steps == 16
        assert not (directory / 'control' / 'result.json').exists()


def test_partial_collection_failure_remains_fully_charged_and_not_resumable(tmp_path):
    with fixture(tmp_path) as (loop, env, directory):
        step = env.step

        def failing_step(action, estimate):
            if env.counter == 4:
                raise RuntimeError('synthetic physical write failed')
            return step(action, estimate)

        env.step = failing_step
        with pytest.raises(RuntimeError, match='physical write'):
            loop.run_update()
        state = loop.journal.summary()
        assert state['failed'] and state['pending'] is None
        assert state['reserved_transitions'] == 32
        assert state['attempted_transitions'] == 10 and state['returned_transitions'] == 8
        assert state['ppo_updates'] == state['actor_optimizer_steps'] == 0
        with pytest.raises(RecoveryConsumptionMismatch):
            loop.journal.checkpoint_boundary()
        with pytest.raises(RuntimeError, match='faulted'):
            loop.run_update()
        assert read_json(directory / 'control' / 'status.json')['phase'] == 'failed'
        assert not (directory / 'control' / 'result.json').exists()
        assert list(loop.store.directory.iterdir()) == []


def test_failed_ppo_retains_executed_work_without_publishing_a_resume_endpoint(tmp_path):
    with fixture(tmp_path) as (loop, _, directory):
        update = loop.components.trainer.update

        def fail_after_executed_work(batch):
            update(batch)
            raise RuntimeError('synthetic post-update failure')

        loop.components.trainer.update = fail_after_executed_work
        with pytest.raises(RuntimeError, match='post-update'):
            loop.run_update()
        state = loop.journal.summary()
        assert state['failed'] and state['ppo_fresh_endpoints'] == 32
        assert state['actor_optimizer_steps'] == state['critic_optimizer_steps'] == 1
        assert not (directory / 'control' / 'result.json').exists()


def test_pause_snapshot_does_not_reset_in_memory_history_rng_or_learning(tmp_path):
    with fixture(tmp_path) as (loop, _, _):
        loop.run_update()
        weights = deepcopy(loop.components.model.state_dict())
        rng, history = capture_rng(), loop.collector._history.clone()
        current = loop.collector._current
        path = loop._pause_checkpoint(17)
        assert path.is_file() and loop.collector._current is current
        assert_identical(loop.components.model.state_dict(), weights)
        assert_identical(capture_rng(), rng)
        assert_identical(loop.collector._history, history)
        assert loop.journal.summary()['returned_transitions'] == 32


def test_validation_wait_is_not_a_failure_or_another_training_reservation(tmp_path):
    with fixture(tmp_path, promotion='audited') as (loop, _, directory):
        # A guard fixture, not a forged qualifying curriculum checkpoint.
        loop.components.curriculum.status = 'stage_budget_complete_validation_pending'
        with pytest.raises(StageValidationRequired):
            loop.run_update()
        assert loop.journal.summary()['rollouts_reserved'] == 0 and not loop._faulted
        assert read_json(directory / 'control' / 'status.json')['phase'] == 'validation_pending'


def test_periodic_complete_snapshot_does_not_change_live_history_or_rng(tmp_path, monkeypatch):
    with fixture(tmp_path) as (loop, _, _):
        for _ in range(24):
            loop.run_update()
        assert not (loop.store.directory / 'latest.json').exists()
        original = loop.store.save

        def checked_save(*args, **kwargs):
            before, history = capture_rng(), loop.collector._history.clone()
            current = loop.collector._current
            result = original(*args, **kwargs)
            assert_identical(before, capture_rng())
            assert_identical(history, loop.collector._history)
            assert loop.collector._current is current
            return result

        monkeypatch.setattr(loop.store, 'save', checked_save)
        loop.run_update()
        marker = read_json(loop.store.directory / 'latest.json')
        assert marker['update'] == 25 and marker['purpose'] == 'periodic_learning_resume'
        assert marker['checkpoint'] == file_pin(loop.store.directory / 'update_00025.pt')
        assert marker['sidecar'] == file_pin(loop.store.directory / 'update_00025.pt.json')
        loaded = loop.store.load('update_00025.pt')
        assert loaded.components.curriculum.promotion == 'exposure'
        assert loaded.components.trainer.update_index == 25
        assert loop.journal.summary()['returned_transitions'] == 800
        assert not (loop.control.directory / 'result.json').exists()


def test_snapshot_io_failure_preserves_spent_work_and_stops_further_sampling(tmp_path, monkeypatch):
    with fixture(tmp_path) as (loop, env, directory):
        for _ in range(24):
            loop.run_update()

        def fail(*args, **kwargs):
            raise OSError('synthetic snapshot storage failure')

        monkeypatch.setattr(loop.store, 'save', fail)
        with pytest.raises(OSError, match='storage'):
            loop.run_update()
        assert loop.journal.summary()['returned_transitions'] == 800
        assert loop.components.trainer.update_index == 25 and loop._faulted
        assert not (loop.store.directory / 'latest.json').exists()
        assert not (directory / 'control' / 'result.json').exists()
        before = env.counter
        with pytest.raises(RuntimeError, match='faulted'):
            loop.run_update()
        assert env.counter == before


def test_selected_checkpoint_interval_is_consumed_by_the_complete_learning_loop(tmp_path):
    with fixture(tmp_path, checkpoint_interval=3) as (loop, _, _):
        loop.run_update()
        loop.run_update()
        assert not (loop.store.directory / 'latest.json').exists()
        loop.run_update()
        marker = read_json(loop.store.directory / 'latest.json')
        assert marker['update'] == 3 and marker['purpose'] == 'periodic_learning_resume'
        loaded = loop.store.load('update_00003.pt')
        assert loaded.recipe.checkpoint_interval == 3
        assert loaded.components.trainer.update_index == 3


def test_exposure_training_crosses_every_stage_and_finishes_without_any_evaluation(tmp_path, monkeypatch):
    import transformer_rl.formal.tasks.curriculum as curriculum_module

    # Three real synthetic updates exercise the complete control state machine;
    # the separate schedule test covers the unchanged 30000 formal budget.
    monkeypatch.setattr(curriculum_module, 'STAGE_BUDGETS', (1, 1, 1))
    original = LearningRecipe.journal_spec
    monkeypatch.setattr(LearningRecipe, 'journal_spec',
                        lambda recipe: replace(original(recipe), target_rollouts=3))
    with fixture(tmp_path) as (loop, env, directory):
        for stage in range(3):
            loop.run_update()
            assert loop.components.curriculum.consumed[stage] == 1
        result = read_json(directory / 'control' / 'result.json')
        assert result['summary']['phase'] == 'completed'
        assert result['summary']['stop_reason'] == 'training_budget_complete'
        assert result['summary']['update'] == 3 and loop._finished
        assert loop.components.curriculum.decisions == [] and env.counter == 48
        assert loop.journal.summary()['reserved_by_stage'] == [1] * 3
        marker = read_json(loop.store.directory / 'latest.json')
        assert marker['purpose'] == 'training_budget_complete'
        assert loop.store.load('final.pt').components.trainer.update_index == 3


def test_stop_at_zero_preserves_initialization_and_is_acknowledged_only_after_complete_snapshot(tmp_path):
    with fixture(tmp_path) as (loop, _, directory):
        command(directory, 'stop', 1)
        assert loop.run_update() is None
        result = read_json(directory / 'control' / 'result.json')
        assert result['summary']['phase'] == 'stopped' and result['summary']['update'] == 0
        assert result['summary']['ack_sequence'] == 1
        assert all(Path(pin['path']).is_file() for pin in result['checkpoint_pins'])
        assert loop.store.load('final.pt').components.trainer.update_index == 0
        with pytest.raises(RuntimeError, match='finished'):
            loop.run_update()


@pytest.mark.parametrize('change', ['exit_code', 'boot', 'token', 'phase', 'active_time', 'checkpoint_pin'])
def test_cold_resume_refuses_invalid_parent_or_final_endpoint_before_loading_weights(tmp_path, monkeypatch, change):
    with fixture(tmp_path) as (loop, _, directory):
        command(directory, 'stop', 1)
        loop.run_update()
        result = read_json(directory / 'control' / 'result.json')
        receipt = {'format': 'private_structured_child_exit', 'token': loop.control.binding['token'],
                   'boot': boot_id(), 'attempt': str(directory), 'identity': result['summary']['identity'],
                   'returncode': 0}
        if change == 'exit_code':
            receipt['returncode'] = 1
        elif change == 'boot':
            receipt['boot'] = '00000000-0000-0000-0000-000000000000'
        elif change == 'token':
            receipt['token'] = 'foreign'
        elif change == 'phase':
            result['summary']['phase'] = 'paused'
        elif change == 'active_time':
            result['summary']['session_active_elapsed_s'] = -1
        else:
            result['checkpoint_pins'][0]['sha256'] = '0' * 64
        write_json(directory / 'control' / 'result.json', result, replace=True)
        write_json(directory / 'exit.json', receipt)

        def no_weight_load(_):
            raise AssertionError('invalid endpoint must be refused before model deserialization')

        monkeypatch.setattr(loop.store, 'load', no_weight_load)
        with pytest.raises(ValueError):
            load_stopped_attempt(loop.store, directory / 'control', parent_exit_pin=file_pin(directory / 'exit.json'))


def probe(loop, env):
    rng = capture_rng()
    assert not rng['cuda']
    digest = hashlib.sha256(rng['torch'].numpy().tobytes() + json.dumps(
        [rng['python'], rng['numpy']], separators=(',', ':')).encode()).hexdigest()
    return {'model': StructuredCheckpointStore.model_sha256(loop.components.model), 'rng': digest,
            'history': hashlib.sha256(loop.collector._history.numpy().tobytes()).hexdigest(),
            'physical_fixture_step': env.counter, 'attempted_vector_steps': loop.collector.attempted_vector_steps,
            'returned_vector_steps': loop.collector.returned_vector_steps,
            'consumed': dict(loop.components.trainer.consumed), 'update': loop.components.trainer.update_index}


def worker(directory, previous=None):
    torch.set_num_threads(1)
    torch.manual_seed(2101)
    root = directory.parent
    recipe, state = components()
    control = TrainingControl(directory / 'control')
    journal = (LearningJournal(root / 'journal', writable=True, authorized_boot=boot_id()) if previous else
               LearningJournal.create(root / 'journal', recipe.journal_spec(), authorized_boot=boot_id()))
    with journal:
        if previous:
            old_store = StructuredCheckpointStore(previous / 'train' / 'checkpoints', recipe, journal)
            loaded, spent = load_stopped_attempt(old_store, previous / 'control',
                                                parent_exit_pin=read_json(previous / 'exit.pin.json'))
            assert spent == control.binding['prior_active_elapsed_s']
            state = loaded.components
            assert state.protocol.max_seen_id[0] == 7 and state.protocol.active_id.eq(-1).all()
        env = SyntheticLabeledEnv()
        collector = StructuredCollector(env, state.model, recipe.collection)
        collector.attempted_vector_steps = state.attempted_vector_steps
        collector.returned_vector_steps = state.returned_vector_steps
        collector.reset(seed=recipe.training_seed)
        if previous:
            loaded.restore_global_rng_after_cold_setup()
        store = StructuredCheckpointStore(directory / 'train' / 'checkpoints', recipe, journal)
        loop = StructuredLearningLoop(recipe, state, collector, journal, store, control, StopBudget())
        loop.run_update()
        state.protocol.max_seen_id[0] = state.protocol.active_id[0] = 7
        state.protocol.height_m[0] = .1
        state.protocol.accepted_at_s[0] = .1
        state.protocol.last_time_s[:] = .16
        write_json(directory / 'before_pause.json', probe(loop, env))
        deadline = time.monotonic() + 10
        while not (directory / 'proceed.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('parent did not permit the next controlled boundary')
            time.sleep(.02)
        assert loop.run_update() is None
        write_json(directory / 'after_stop.json', probe(loop, env))


def wait_for(path, predicate=lambda value: bool(value), timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read_json(path)
        if value is not None and predicate(value):
            return value
        time.sleep(.02)
    raise TimeoutError('owned test worker did not publish: ' + str(path))


def test_real_child_pause_stop_waitpid_and_cold_resume_preserve_learning_budget(tmp_path):
    root = tmp_path / 'owned_workers'
    root.mkdir()
    descriptor = os.open(root / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe, _ = components()
    previous, spent = None, 0.
    try:
        for start in (0, 1):
            directory = attempt(root, f'attempt_{start + 1}', descriptor, operation='pause', start=start, spent=spent)
            argv = [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory)]
            if previous:
                argv += [str(previous)]
            with (directory / 'worker.log').open('wb') as log:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(descriptor,))
                try:
                    initial = wait_for(directory / 'control' / 'status.json', lambda value: value['phase'] == 'paused')
                    assert initial['update'] == start and initial['cumulative_transitions'] == start * 32
                    command(directory, 'resume', 1)
                    before = wait_for(directory / 'before_pause.json')
                    command(directory, 'pause', 2)
                    write_json(directory / 'proceed.json', {})
                    paused = wait_for(directory / 'control' / 'status.json',
                                      lambda value: value['phase'] == 'paused' and value['ack_sequence'] == 2)
                    identity = process_identity(process.pid)
                    assert identity == paused['identity']
                    assert paused['update'] == start + 1 and before['physical_fixture_step'] == 16
                    assert before['returned_vector_steps'] == (start + 1) * 16
                    command(directory, 'stop', 3)
                    assert process.wait(timeout=10) == 0
                finally:
                    if process.poll() is None:
                        command(directory, 'stop', 99)
                        process.wait(timeout=3)
            after = read_json(directory / 'after_stop.json')
            assert before == after  # Includes history, global RNG, weights and actual optimizer consumption.
            write_json(directory / 'exit.json', {'format': 'private_structured_child_exit',
                       'token': read_json(directory / 'request.json')['token'], 'boot': boot_id(),
                       'attempt': str(directory), 'identity': identity, 'returncode': process.returncode})
            pin = file_pin(directory / 'exit.json')
            write_json(directory / 'exit.pin.json', pin)
            journal = LearningJournal(root / 'journal')
            store = StructuredCheckpointStore(directory / 'train' / 'checkpoints', recipe, journal)
            loaded, spent = load_stopped_attempt(store, directory / 'control', parent_exit_pin=pin)
            assert loaded.components.trainer.update_index == start + 1
            assert loaded.components.curriculum.total_updates == start + 1
            assert loaded.components.trainer.consumed['actor_optimizer_steps'] == start + 1
            assert loaded.components.trainer.consumed['critic_optimizer_steps'] == start + 1
            assert loaded.components.protocol.max_seen_id[0] == 7
            with pytest.raises(ValueError, match='pin'):
                load_stopped_attempt(store, directory / 'control', parent_exit_pin={**pin, 'sha256': '0' * 64})
            bad = read_json(directory / 'exit.json')
            bad['returncode'] = 1
            write_json(directory / 'failed_parent.json', bad)
            with pytest.raises(ValueError, match='escapes'):
                load_stopped_attempt(store, directory / 'control', parent_exit_pin=file_pin(directory / 'failed_parent.json'))
            previous = directory
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    assert sys.argv[1] == '--worker'
    worker(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) == 4 else None)

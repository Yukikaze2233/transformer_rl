"""Worker orchestration on owned synthetic CPU environments, without an SDK."""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os

import pytest
import torch

from test_structured_learning import attempt, command, components
from test_structured_collection import SyntheticLabeledEnv
from transformer_rl._structured_process import fresh_components, run_worker
from transformer_rl.training_control import read_json


@pytest.fixture(autouse=True)
def seeded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


def test_worker_collects_updates_then_commits_stop_without_evaluation(tmp_path):
    root = tmp_path / 'session'
    root.mkdir()
    descriptor = os.open(root / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recipe, _ = components()
    directory = attempt(root, 'attempt_1', descriptor)
    # The production worker owns creation of the new checkpoint directory.
    (directory / 'train/checkpoints').rmdir()
    env = SyntheticLabeledEnv()
    original_step = env.step
    events = []

    def step(*args):
        value = original_step(*args)
        if env.counter == recipe.rollout_steps:
            command(directory, 'stop', 1)
        return value

    env.step = step

    @contextmanager
    def runtime(parsed, state, config, *, control):
        assert parsed == recipe and config == {'test': 'owned'}
        assert state.trainer.update_index == 0
        assert control.state['automatic_evaluation'] is False
        assert control.state['solver_position_iterations'] == 12
        assert control.state['solver_velocity_iterations'] == 6
        events.append('opened')
        try:
            yield env
        finally:
            events.append('closed')

    try:
        report = run_worker(recipe, directory / 'control', runtime, {'test': 'owned'})
        assert report['status'] == 'stopped' and report['final_update'] == 1
        assert report['cumulative_transitions'] == 32
        assert events == ['opened', 'closed']
        assert read_json(directory / 'control/result.json')['summary']['ack_sequence'] == 1
        rows = [json.loads(row) for row in (directory / 'train/metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['consumption']['ppo_updates'] == 1
        assert (directory / 'train/checkpoints/final.pt').is_file()
    finally:
        os.close(descriptor)


def test_fresh_worker_refuses_a_different_frozen_initialization():
    recipe, _ = components()
    with pytest.raises(ValueError, match='frozen initial model'):
        fresh_components(replace(recipe, initial_model_sha256='f' * 64))

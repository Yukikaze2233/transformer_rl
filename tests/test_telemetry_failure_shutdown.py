"""Optional event-writer teardown cannot replace the learner's primary error."""
import pytest

from transformer_rl.formal.runtime.telemetry import TensorBoardTelemetry
from test_formal_telemetry import Writer


@pytest.mark.parametrize('flush_error,close_error', [
    (SystemExit(0), None),
    (KeyboardInterrupt(), None),
    (OSError('flush failed'), SystemExit(0)),
    (RuntimeError('flush failed'), RuntimeError('close failed')),
])
def test_writer_shutdown_retains_the_original_learning_exception(tmp_path, flush_error, close_error):
    class BrokenShutdownWriter(Writer):
        def flush(self):
            raise flush_error

        def close(self):
            if close_error is not None:
                raise close_error
            super().close()

    primary = RuntimeError('original terrain query failure')
    observed = None
    try:
        with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=BrokenShutdownWriter):
            raise primary
    except BaseException as error:
        observed = error
    assert observed is primary
    assert any('TensorBoard' in note for note in primary.__notes__)


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(0)])
def test_writer_shutdown_without_a_primary_failure_keeps_interrupt_semantics(tmp_path, interrupt):
    class InterruptedWriter(Writer):
        def flush(self):
            raise interrupt

    observed = None
    try:
        with TensorBoardTelemetry(tmp_path, scope='formal', writer_factory=InterruptedWriter):
            pass
    except BaseException as error:
        observed = error
    assert observed is interrupt


def test_managed_worker_publishes_primary_error_and_keeps_failed_reservation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import fcntl
    import os
    import torch

    from test_structured_collection import SyntheticLabeledEnv
    from test_structured_learning import attempt, components
    from transformer_rl.common.training_control import read_json
    from transformer_rl.formal.learning.accounting import LearningJournal
    from transformer_rl.formal.runtime import process

    class BrokenWriter(Writer):
        def flush(self):
            raise SystemExit(0)

    monkeypatch.setattr(process, 'TensorBoardTelemetry',
                        lambda *args, **kwargs: TensorBoardTelemetry(
                            *args, **kwargs, writer_factory=BrokenWriter))
    root = tmp_path / 'owned_failure_run'
    root.mkdir()
    descriptor = os.open(root / 'run.lock', os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    primary = RuntimeError('original terrain geometry gap')
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2101)
            recipe, _ = components()
            directory = attempt(root, 'attempt_1', descriptor)
            (directory / 'train/checkpoints').rmdir()
            environment = SyntheticLabeledEnv()

            def failed_step(*_args):
                raise primary

            environment.step = failed_step

            @contextmanager
            def runtime(*_args, **_kwargs):
                yield environment

            with pytest.raises(RuntimeError) as caught:
                process.run_worker(recipe, directory / 'control', runtime, {}, tensorboard=True)
            assert caught.value is primary
            status = read_json(directory / 'control/status.json')
            assert status['phase'] == 'failed'
            assert status['error'] == 'RuntimeError: original terrain geometry gap'
            assert not (directory / 'control/result.json').exists()
            journal = LearningJournal(root / 'journal')
            try:
                summary = journal.summary()
                assert summary['failed'] and summary['rollouts_reserved'] == 1
                assert summary['reserved_transitions'] == recipe.num_envs * recipe.rollout_steps
                assert summary['ppo_updates'] == 0
            finally:
                journal.close()
    finally:
        os.close(descriptor)

"""Durable reservation/recovery checks confined to new temporary journals."""
from copy import deepcopy
import json

import pytest

from transformer_rl.formal.learning import accounting
from transformer_rl.formal.learning.accounting import LearningJournal, LearningJournalSpec, RecoveryConsumptionMismatch
from transformer_rl.common.training_control import boot_id


def spec():
    return LearningJournalSpec('fixture', 'a' * 64, 4, 8, 1, 2)


def actual(journal, *, steps=2):
    state = journal.summary()
    return {key: state[key] + delta for key, delta in zip(LearningJournal._ACTUAL,
             (32, 32, 1, 32, steps, 32, steps, 32))}


def test_reservation_is_durable_before_any_work_and_is_never_refunded(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as writer:
        empty = writer.checkpoint_boundary()
        writer.reserve(0, active_elapsed_s=.1)
        observed = LearningJournal(writer.directory).summary()
        assert observed['rollouts_reserved'] == 1 and observed['reserved_transitions'] == 32
        assert observed['returned_transitions'] == 0 and observed['pending']['index'] == 1
        with pytest.raises(RecoveryConsumptionMismatch):
            writer.require_exact_boundary(empty)
        with pytest.raises(ValueError):
            writer.reserve(0, active_elapsed_s=.2)


def test_completed_boundary_replays_every_counter_and_stage_without_mutating_returned_snapshots(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as writer:
        writer.reserve(0, active_elapsed_s=.1)
        writer.finish(actual(writer), active_elapsed_s=.2)
        writer.record_boundary(active_elapsed_s=.25)
        boundary = writer.checkpoint_boundary()
        assert boundary['reserved_by_stage'] == [1, 0, 0, 0, 0, 0]
        assert boundary['actor_optimizer_steps'] == boundary['critic_optimizer_steps'] == 2
        reader = LearningJournal(writer.directory)
        reader.require_exact_boundary(boundary)
        boundary['reserved_by_stage'][0] = 99
        assert writer.summary()['reserved_by_stage'][0] == 1


def test_checkpoint_behind_another_complete_update_cannot_replay_old_weights(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        initial = journal.checkpoint_boundary()
        journal.reserve(0, active_elapsed_s=.1)
        journal.finish(actual(journal), active_elapsed_s=.2)
        with pytest.raises(RecoveryConsumptionMismatch, match='behind'):
            journal.require_exact_boundary(initial)


def test_failed_partial_collection_keeps_full_reservation_and_is_not_resumable(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        journal.reserve(0, active_elapsed_s=.1)
        counters = {key: 0 for key in LearningJournal._ACTUAL}
        counters.update(attempted_transitions=12, returned_transitions=8)
        journal.finish(counters, failed=True, active_elapsed_s=.2)
        assert journal.summary()['reserved_transitions'] == 32
        assert journal.summary()['returned_transitions'] == 8
        with pytest.raises(RecoveryConsumptionMismatch):
            journal.checkpoint_boundary()
        with pytest.raises(ValueError):
            journal.reserve(0, active_elapsed_s=.3)


@pytest.mark.parametrize('change', ['returned', 'optimizer', 'uses', 'ppo_size', 'decrease', 'bool'])
def test_counter_inconsistencies_are_rejected_without_publishing_a_finish(tmp_path, change):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        journal.reserve(0, active_elapsed_s=.1)
        values = actual(journal)
        key, value = {'returned': ('returned_transitions', 33), 'optimizer': ('actor_optimizer_steps', 3),
                      'uses': ('actor_endpoint_uses', 33), 'ppo_size': ('ppo_fresh_endpoints', 31),
                      'decrease': ('attempted_transitions', -1), 'bool': ('ppo_updates', True)}[change]
        values[key] = value
        before = journal.summary()
        with pytest.raises(ValueError):
            journal.finish(values, active_elapsed_s=.2)
        assert journal.summary() == before
        assert len(list((journal.directory / 'events').iterdir())) == 1


def test_only_one_writer_can_hold_this_journals_lease(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as writer:
        with pytest.raises(BlockingIOError):
            LearningJournal(writer.directory, writable=True, authorized_boot=boot_id())
    with LearningJournal(tmp_path / 'journal', writable=True, authorized_boot=boot_id()) as replacement:
        replacement.reserve(0, active_elapsed_s=.1)


def test_post_publication_error_cannot_cause_a_duplicate_reservation(monkeypatch, tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        original = accounting.write_json

        def fail_after_publish(path, value):
            original(path, value)
            raise OSError('synthetic interruption after durable publication')

        monkeypatch.setattr(accounting, 'write_json', fail_after_publish)
        with pytest.raises(OSError):
            journal.reserve(0, active_elapsed_s=.1)
        with pytest.raises(RuntimeError, match='faulted'):
            journal.reserve(0, active_elapsed_s=.2)
        reader = LearningJournal(journal.directory)
        assert reader.summary()['rollouts_reserved'] == 1 and reader.summary()['pending'] is not None


def test_modified_chain_or_deleted_commit_is_not_accepted_as_the_original_checkpoint(tmp_path):
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        journal.reserve(0, active_elapsed_s=.1)
        journal.finish(actual(journal), active_elapsed_s=.2)
        reference = journal.checkpoint_boundary()
    path = tmp_path / 'journal/events/00000000.json'
    event = json.loads(path.read_text())
    event['active_elapsed_s'] = .05
    path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match='chain'):
        LearningJournal(tmp_path / 'journal')


def test_different_actual_boot_cannot_create_or_mutate_a_journal(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        LearningJournal.create(tmp_path / 'wrong', spec(), authorized_boot='0' * 36)
    with LearningJournal.create(tmp_path / 'journal', spec(), authorized_boot=boot_id()) as journal:
        monkeypatch.setattr(accounting, 'boot_id', lambda: '0' * 36)
        with pytest.raises(RuntimeError, match='boot'):
            journal.reserve(0, active_elapsed_s=.1)

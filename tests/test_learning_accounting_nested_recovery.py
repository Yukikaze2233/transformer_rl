"""Repeated failed forks retain every spent reservation and prior loss."""
from copy import deepcopy

import pytest

from transformer_rl.common.training_control import boot_id
from transformer_rl.formal.learning.accounting import (
    LearningJournal, LearningJournalSpec, RecoveryConsumptionMismatch,
)


def _spec():
    return LearningJournalSpec('nested_fixture', 'a' * 64, 4, 8, 1, 2)


def _boundary(updates, reserved, *, failed=False, attempted_extra=0, returned_extra=0):
    size = 32
    return {
        'identity_sha256': 'b' * 64,
        'head': {'sequence': reserved * 2 - 1, 'bytes': 512, 'sha256': 'c' * 64},
        'rollouts_reserved': reserved, 'reserved_transitions': reserved * size,
        'reserved_by_stage': [reserved, 0, 0],
        'attempted_transitions': updates * size + attempted_extra,
        'returned_transitions': updates * size + returned_extra,
        'ppo_updates': updates, 'ppo_fresh_endpoints': updates * size,
        'actor_optimizer_steps': updates * 2, 'actor_endpoint_uses': updates * size,
        'critic_optimizer_steps': updates * 2, 'critic_endpoint_uses': updates * size,
        'active_elapsed_s': float(reserved * 2), 'pending': None, 'failed': failed,
    }


def _first_pair():
    return (_boundary(200, 200),
            _boundary(240, 241, failed=True, attempted_extra=12, returned_extra=8))


def _advance(journal, *, partial=False):
    state = journal.summary()
    elapsed = state['active_elapsed_s']
    journal.reserve(0, active_elapsed_s=elapsed + 1)
    delta = ((12, 8, 0, 0, 0, 0, 0, 0) if partial
             else (32, 32, 1, 32, 2, 32, 2, 32))
    counts = {key: state[key] + value for key, value in zip(journal._ACTUAL, delta)}
    journal.finish(counts, failed=partial, active_elapsed_s=elapsed + 2)


def _nested_pair(tmp_path):
    checkpoint, failed = _first_pair()
    with LearningJournal.create(tmp_path / 'first', _spec(), authorized_boot=boot_id()) as first:
        first.import_failed_consumption(checkpoint, failed, provenance={'first_authorization': 'sealed'})
        checkpoint = first.checkpoint_boundary()
    failed = deepcopy(checkpoint)
    # A later failure after 88 full live updates plus one partial reservation.
    for key, value in zip(LearningJournal._ACTUAL, (88 * 32 + 12, 88 * 32 + 8,
                         88, 88 * 32, 176, 88 * 32, 176, 88 * 32)):
        failed[key] += value
    failed.update(rollouts_reserved=330, reserved_transitions=330 * 32,
                  reserved_by_stage=[330, 0, 0], active_elapsed_s=660., failed=True,
                  head={'sequence': 178, 'bytes': 512, 'sha256': 'd' * 64})
    return checkpoint, failed


def test_second_durable_fork_preserves_40_plus_88_and_both_partial_reservations(tmp_path):
    original_checkpoint, original_failed = _first_pair()
    with LearningJournal.create(tmp_path / 'first', _spec(), authorized_boot=boot_id()) as first:
        first.import_failed_consumption(original_checkpoint, original_failed,
                                        provenance={'first_authorization': 'sealed'})
        surviving = first.checkpoint_boundary()
        for _ in range(88):
            _advance(first)
        _advance(first, partial=True)
        failed = first.summary()
        assert failed['ppo_updates'] == 288
        assert failed['rollouts_reserved'] == 330
        assert first.total_consumption()['ppo_updates'] == 328
        old_events = {p.name: p.read_bytes() for p in (first.directory / 'events').iterdir()}
    original_snapshots = deepcopy((surviving, failed))
    with LearningJournal.create(tmp_path / 'second', _spec(), authorized_boot=boot_id()) as second:
        second.import_failed_consumption(surviving, failed,
                                         provenance={'second_authorization': 'sealed'})
        state = second.checkpoint_boundary()
        assert state['ppo_updates'] == 200
        assert state['rollouts_reserved'] == 330
        assert state['retired_consumption']['ppo_updates'] == 128
        assert state['retired_consumption']['attempted_transitions'] == 128 * 32 + 24
        assert state['retired_consumption']['returned_transitions'] == 128 * 32 + 16
        assert state['retired_consumption']['actor_optimizer_steps'] == 256
        assert second.total_consumption()['ppo_updates'] == 328
        assert second.total_consumption()['attempted_transitions'] == 328 * 32 + 24
        assert second.total_consumption()['returned_transitions'] == 328 * 32 + 16
        assert second.total_consumption()['actor_optimizer_steps'] == 656
        # The next live model update is 201; spent reservation 331 is not replayed.
        _advance(second)
        final = second.checkpoint_boundary()
        assert final['ppo_updates'] == 201 and final['rollouts_reserved'] == 331
        assert second.total_consumption()['ppo_updates'] == 329
    with LearningJournal(tmp_path / 'second') as reader:
        reader.require_exact_boundary(final)
        changed = deepcopy(final)
        changed['retired_consumption']['ppo_updates'] = 88
        with pytest.raises(RecoveryConsumptionMismatch, match='differs'):
            reader.require_exact_boundary(changed)
    assert (surviving, failed) == original_snapshots
    assert {p.name: p.read_bytes() for p in (tmp_path / 'first/events').iterdir()} == old_events
    assert LearningJournal(tmp_path / 'first').summary() == failed


def test_first_generation_summary_shape_and_consumption_remain_compatible(tmp_path):
    checkpoint, failed = _first_pair()
    with LearningJournal.create(tmp_path / 'new', _spec(), authorized_boot=boot_id()) as new:
        new.import_failed_consumption(checkpoint, failed, provenance={'legacy_fixture': True})
        assert new.summary()['retired_consumption']['ppo_updates'] == 40
        assert new.total_consumption()['returned_transitions'] == failed['returned_transitions']
        assert new.summary()['recovery'] == {
            'checkpoint': checkpoint, 'failed': failed, 'provenance': {'legacy_fixture': True}}


def test_third_fork_can_keep_a_later_checkpoint_without_retiring_prior_loss_twice(tmp_path):
    checkpoint, failed = _nested_pair(tmp_path)
    with LearningJournal.create(tmp_path / 'second', _spec(), authorized_boot=boot_id()) as second:
        second.import_failed_consumption(checkpoint, failed, provenance={'second_fixture': True})
        _advance(second)
        surviving = second.checkpoint_boundary()
        _advance(second, partial=True)
        failed_again = second.summary()
    with LearningJournal.create(tmp_path / 'third', _spec(), authorized_boot=boot_id()) as third:
        third.import_failed_consumption(surviving, failed_again, provenance={'third_fixture': True})
        assert third.summary()['ppo_updates'] == 201
        assert third.summary()['rollouts_reserved'] == 332
        assert third.summary()['retired_consumption']['ppo_updates'] == 128
        assert third.summary()['retired_consumption']['attempted_transitions'] == 128 * 32 + 36
        assert third.total_consumption()['ppo_updates'] == 329
        assert third.summary()['recovery']['checkpoint']['recovery'] == surviving['recovery']


@pytest.mark.parametrize('change', [
    'different_provenance', 'different_prior_head', 'forged_retired', 'retired_extra_key',
    'missing_retired', 'missing_recovery', 'unknown_field', 'boolean_counter',
    'negative_retired', 'invalid_head', 'invalid_optimizer', 'inherited_stage_refund',
])
def test_nested_fork_rejects_incompatible_or_forged_ancestry_before_publication(tmp_path, change):
    checkpoint, failed = _nested_pair(tmp_path)
    if change == 'different_provenance':
        failed['recovery']['provenance']['first_authorization'] = 'different'
    elif change == 'different_prior_head':
        failed['recovery']['failed']['head']['sha256'] = 'e' * 64
    elif change == 'forged_retired':
        for boundary in (checkpoint, failed):
            boundary['retired_consumption']['ppo_updates'] = 39
            boundary['retired_consumption']['ppo_fresh_endpoints'] = 39 * 32
    elif change == 'retired_extra_key':
        checkpoint['retired_consumption']['refunded'] = 0
    elif change == 'missing_retired':
        del checkpoint['retired_consumption']
    elif change == 'missing_recovery':
        del failed['recovery']
    elif change == 'unknown_field':
        failed['accepted_without_review'] = True
    elif change == 'boolean_counter':
        failed['attempted_transitions'] = True
    elif change == 'negative_retired':
        failed['retired_consumption']['returned_transitions'] = -1
    elif change == 'invalid_head':
        checkpoint['head']['sequence'] = True
    elif change == 'invalid_optimizer':
        failed['critic_optimizer_steps'] = failed['ppo_updates'] * 2 + 1
    else:
        for boundary in (checkpoint, failed):
            boundary['recovery']['failed']['reserved_by_stage'] = [240, 1, 0]
    with LearningJournal.create(tmp_path / 'rejected', _spec(), authorized_boot=boot_id()) as new:
        before = new.summary()
        with pytest.raises(ValueError):
            new.import_failed_consumption(checkpoint, failed, provenance={'second_fixture': True})
        assert new.summary() == before
        assert list((new.directory / 'events').iterdir()) == []


@pytest.mark.parametrize('change', ['counter', 'head', 'fields', 'provenance'])
def test_invalid_old_generation_cannot_hide_inside_an_equal_prior_payload(tmp_path, change):
    checkpoint, failed = _nested_pair(tmp_path)
    for boundary in (checkpoint, failed):
        prior = boundary['recovery']
        if change == 'counter':
            prior['failed']['attempted_transitions'] = -1
        elif change == 'head':
            prior['checkpoint']['head']['unexpected'] = 0
        elif change == 'fields':
            prior['checkpoint']['unknown'] = None
        else:
            prior['provenance'] = {}
    with LearningJournal.create(tmp_path / 'rejected', _spec(), authorized_boot=boot_id()) as new:
        with pytest.raises(ValueError):
            new.import_failed_consumption(checkpoint, failed, provenance={'second_fixture': True})
        assert not list((new.directory / 'events').iterdir())


def test_ancestry_depth_is_bounded_before_deepcopy_or_publication(tmp_path):
    checkpoint, failed = _first_pair()
    payload = {'checkpoint': checkpoint, 'failed': failed, 'provenance': {'fixture': True}}
    # An in-memory caller must not bypass the ancestry limit with a cycle.
    for boundary in (checkpoint, failed):
        boundary['recovery'] = payload
        boundary['retired_consumption'] = {key: 0 for key in LearningJournal._ACTUAL}
    with LearningJournal.create(tmp_path / 'rejected', _spec(), authorized_boot=boot_id()) as new:
        with pytest.raises(ValueError, match='bounded private depth'):
            new.import_failed_consumption(checkpoint, failed, provenance={'second_fixture': True})
        assert not list((new.directory / 'events').iterdir())

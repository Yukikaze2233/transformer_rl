"""Recovery forks retain both lost work and exact surviving learning state."""
import shutil

import pytest
import torch

from test_learning_accounting import actual, spec
from test_structured_session import session, wait
from transformer_rl.common.training_control import boot_id, file_pin, read_json, write_json
from transformer_rl.formal.learning.accounting import LearningJournal, RecoveryConsumptionMismatch
from transformer_rl.formal.learning.checkpoint import LearningRecipe
from transformer_rl.formal.runtime.recovery import (
    load_recovery, prepare_failed_recovery, recovery_metadata,
)
from transformer_rl.formal.runtime.session import StructuredTrainingSession
from transformer_rl.formal.runtime.transition import source_table


@pytest.fixture(autouse=True)
def cpu_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


def boundaries(tmp_path):
    with LearningJournal.create(tmp_path / 'old', spec(), authorized_boot=boot_id()) as old:
        old.reserve(0, active_elapsed_s=1)
        old.finish(actual(old), active_elapsed_s=2)
        checkpoint = old.checkpoint_boundary()
        old.reserve(0, active_elapsed_s=3)
        old.finish(actual(old), active_elapsed_s=4)
        old.reserve(0, active_elapsed_s=5)
        partial = {key: old.summary()[key] for key in old._ACTUAL}
        partial.update(attempted_transitions=76, returned_transitions=72)
        old.finish(partial, failed=True, active_elapsed_s=6)
        return checkpoint, old.summary()


def test_fork_retains_model_counter_and_all_spent_budget_separately(tmp_path):
    checkpoint, failed = boundaries(tmp_path)
    with LearningJournal.create(tmp_path / 'new', spec(), authorized_boot=boot_id()) as new:
        new.import_failed_consumption(checkpoint, failed, provenance={'approved_fixture': True})
        assert new.summary()['ppo_updates'] == 1
        assert new.summary()['rollouts_reserved'] == 3
        assert new.total_consumption()['ppo_updates'] == 2
        assert new.total_consumption()['returned_transitions'] == 72
        assert new.summary()['reserved_transitions'] == 96
        new.reserve(0, active_elapsed_s=7)
        new.finish(actual(new), active_elapsed_s=8)
        assert new.summary()['ppo_updates'] == 2 and new.summary()['rollouts_reserved'] == 4
        assert new.total_consumption()['returned_transitions'] == 104
        assert new.total_consumption()['ppo_updates'] == 3
        with pytest.raises(ValueError, match='first'):
            new.import_failed_consumption(checkpoint, failed, provenance={'again': True})
        boundary = new.checkpoint_boundary()
    with LearningJournal(tmp_path / 'new') as reader:
        reader.require_exact_boundary(boundary)
        assert reader.total_consumption()['returned_transitions'] == 104
    with LearningJournal(tmp_path / 'old') as old:
        assert old.summary() == failed
        with pytest.raises(RecoveryConsumptionMismatch):
            old.checkpoint_boundary()


@pytest.mark.parametrize('change', ['refund', 'reseed_identity', 'clean_source', 'zero_model', 'bad_stage'])
def test_fork_rejects_refunds_and_inconsistent_source_boundaries(tmp_path, change):
    checkpoint, failed = boundaries(tmp_path)
    if change == 'refund':
        failed['attempted_transitions'] = 0
    elif change == 'reseed_identity':
        checkpoint['identity_sha256'] = 'b' * 64
    elif change == 'clean_source':
        failed['failed'] = False
    elif change == 'zero_model':
        checkpoint['ppo_updates'] = 0
    else:
        failed['reserved_by_stage'][1] = 1
    with LearningJournal.create(tmp_path / 'new', spec(), authorized_boot=boot_id()) as new:
        with pytest.raises(ValueError):
            new.import_failed_consumption(checkpoint, failed, provenance={'fixture': True})
        assert new.summary()['rollouts_reserved'] == 0


def test_cached_failed_fork_validates_then_pauses_stops_and_cold_resumes(tmp_path, monkeypatch):
    value, resources = session(tmp_path, monkeypatch)
    value.start()
    assert wait(value, lambda state: not state['active'])['update'] == 1
    assert value.submitted_process.wait(timeout=5) == 0
    previous = value.attempt()
    checkpoint = read_json(previous / 'train/checkpoints/final.pt.json')['boundary']
    with LearningJournal(value.root / 'journal', writable=True, authorized_boot=boot_id()) as old:
        old.reserve(0, active_elapsed_s=checkpoint['active_elapsed_s'] + 1)
        old.finish(actual(old, steps=1), active_elapsed_s=checkpoint['active_elapsed_s'] + 2)
        old.reserve(0, active_elapsed_s=checkpoint['active_elapsed_s'] + 3)
        partial = {key: old.summary()[key] for key in old._ACTUAL}
        partial.update(attempted_transitions=76, returned_transitions=72)
        old.finish(partial, failed=True, active_elapsed_s=checkpoint['active_elapsed_s'] + 4)
        failed = old.summary()
    cache = tmp_path / 'cache'
    cache.mkdir()
    shutil.copytree(value.root / 'journal', cache / 'journal')
    for source, target in [('final.pt', 'checkpoint.pt'), ('final.pt.json', 'checkpoint.pt.json')]:
        shutil.copyfile(previous / 'train/checkpoints' / source, cache / target)
    records = {str(p): dict(file_pin(p), cached_path=str(p))
               for p in [cache / 'checkpoint.pt', cache / 'checkpoint.pt.json',
                         cache / 'journal/identity.json', cache / 'journal/identity.pin.json',
                         *sorted((cache / 'journal/events').glob('*.json'))]}
    recipe = LearningRecipe.from_dict(read_json(value.root / 'recipe.json'))
    write_json(cache / 'review.json', {'boot': boot_id(), 'checkpoint_recipe_sha256': recipe.sha256,
                                     'checkpoint_boundary': checkpoint, 'final_journal': failed,
                                     'records': records})
    write_json(tmp_path / 'approval.json', {
        'authorization': 'failed_checkpoint_fork_preserving_all_consumption', 'boot': boot_id(),
        'human_objective': 'retain surviving weights and every consumed budget slot',
        'checkpoint_sha256': records[str(cache / 'checkpoint.pt')]['sha256'],
        'failed_head_sha256': failed['head']['sha256']})
    package = tmp_path / 'candidate/transformer_rl'
    shutil.copytree(value.root / 'source/transformer_rl', package)
    path = package / 'formal/lab_env/recovery.py'
    path.write_text(path.read_text() + '\n# Equivalent source fixture.\n')
    before, after = source_table(value.root / 'source/transformer_rl'), source_table(package)
    changes = {name: {'before': before.get(name), 'after': after.get(name)}
               for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    monkeypatch.setattr(StructuredTrainingSession, '_resources', lambda self: (True, {'fixture': True}))
    old_events = {p.name: p.read_bytes() for p in (value.root / 'journal/events').iterdir()}
    rng = torch.get_rng_state().clone()
    target = prepare_failed_recovery(tmp_path / 'fork', cache_review_pin=file_pin(cache / 'review.json'),
        previous_recipe=recipe, previous_source=value.root / 'source/transformer_rl',
        learner_source=package, reviewed_changes=changes, authorization_pin=file_pin(tmp_path / 'approval.json'),
        authorized_boot=boot_id(), runtime_factory=value.spec['runtime_factory'],
        runtime_configuration={}, resources=resources, python=value.spec['python'])
    assert torch.equal(rng, torch.get_rng_state())
    marker = recovery_metadata(target.root)
    assert marker['update'] == 1 and marker['budget_update'] == 3
    with LearningJournal(target.root / 'journal') as journal:
        binding = {'start_update': 1, 'prior_transitions': 32,
                   'prior_active_elapsed_s': marker['active_elapsed_s']}
        restored = load_recovery(target.root, LearningRecipe.from_dict(read_json(target.root / 'recipe.json')),
                                 journal, binding)
        assert restored.components.trainer.update_index == 1
        assert restored.components.curriculum.total_updates == 3
        with pytest.raises(ValueError, match='rebased'):
            load_recovery(target.root, recipe, journal, dict(binding, start_update=2))
    target.start(paused=True)
    paused = wait(target, lambda state: state['phase'] == 'paused')
    assert paused['update'] == 1 and paused['budget_update'] == 3
    target.command('resume')
    final = wait(target, lambda state: not state['active'])
    assert target.submitted_process.wait(timeout=5) == 0
    assert final['update'] == 2 and final['budget_update'] == 4
    assert final['total_consumption']['ppo_updates'] == 3
    target.start()
    final = wait(target, lambda state: not state['active'])
    assert target.submitted_process.wait(timeout=5) == 0
    assert final['update'] == 3 and final['budget_update'] == 5
    assert {p.name: p.read_bytes() for p in (value.root / 'journal/events').iterdir()} == old_events
    assert LearningJournal(value.root / 'journal').summary() == failed


def test_dashboard_distinguishes_model_updates_and_nonrefundable_budget():
    from transformer_rl.common.training_cli import render

    state = dict(name='fixture', root='/fixture', phase='between_updates', architecture='structured_transformer',
                 seed=2102, device='cuda:0', update=200, target_update=35000, budget_update=241,
                 retired_consumption={'ppo_updates': 40}, cumulative_transitions=78643200,
                 total_consumption={'returned_transitions': 94715904})
    text = render(state, width=108)
    assert '241 / 35,000' in text and '保留模型更新  200' in text
    assert '历史丢失更新仍计费  40' in text and '94,715,904' in text

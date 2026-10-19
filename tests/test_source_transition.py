"""Source epochs preserve actual parent receipts, full state and paid history."""
from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import torch

from test_structured_session import session, wait
from transformer_rl.common.training_control import boot_id, file_pin, read_json, write_json
from transformer_rl.formal.learning.accounting import LearningJournal
from transformer_rl.formal.learning.checkpoint import LearningRecipe
from transformer_rl.formal.runtime.session import StructuredTrainingSession
from transformer_rl.formal.runtime.transition import (
    _review_sources, learning_state_sha256, load_transition, migrate_stopped_session,
    source_table, transition_metadata,
)


@pytest.fixture(autouse=True)
def cpu_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


def candidate(value, root):
    package = root / 'candidate/transformer_rl'
    shutil.copytree(value.root / 'source/transformer_rl', package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = package / 'formal/tasks/terrain.py'
    path.write_text(path.read_text() + '\n# Equivalent source fixture.\n')
    before, after = source_table(value.root / 'source/transformer_rl'), source_table(package)
    review = {name: {'before': before.get(name), 'after': after.get(name)}
              for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    return package, review


def approve(root, *, boot=None):
    path = root / 'approval.json'
    write_json(path, {'authorization': 'approved_implementation_validation_then_switch',
                     'boot': boot or boot_id(), 'human_answer': 'approve validated state-preserving switch'})
    return file_pin(path)


def test_transition_preserves_nonzero_state_then_runs_only_the_next_update(tmp_path, monkeypatch):
    value, _ = session(tmp_path, monkeypatch)
    value.start()
    assert wait(value, lambda s: not s['active'])['update'] == 1
    assert value.submitted_process.wait(timeout=5) == 0
    previous = value.attempt()
    old_files = {str(p.relative_to(value.root)): p.read_bytes()
                 for p in [value.root / 'recipe.json', previous / 'train/checkpoints/final.pt',
                           previous / 'exit.json', *sorted((value.root / 'journal/events').glob('*.json'))]}
    package, review = candidate(value, tmp_path)
    monkeypatch.setattr(StructuredTrainingSession, '_resources', lambda self: (True, {'test_admitted': True}))
    rng = torch.get_rng_state().clone()
    target = migrate_stopped_session(value.root, tmp_path / 'continued', learner_source=package,
                                    reviewed_changes=review, approval_pin=approve(tmp_path),
                                    authorized_boot=boot_id())
    assert torch.equal(rng, torch.get_rng_state())
    assert target.attempt() is None
    marker = transition_metadata(target.root)
    old_recipe = LearningRecipe.from_dict(read_json(value.root / 'recipe.json'))
    recipe = LearningRecipe.from_dict(read_json(target.root / 'recipe.json'))
    assert replace(old_recipe, learner_source_sha256=recipe.learner_source_sha256) == recipe
    assert old_recipe.sha256 != recipe.sha256
    assert marker['update'] == 1 and marker['returned_transitions'] == 32
    assert marker['source']['source_checkpoint_pin']['path'] == str(previous / 'train/checkpoints/final.pt')
    with LearningJournal(target.root / 'journal', authorized_boot=boot_id()) as journal:
        binding = {'start_update': marker['update'], 'prior_transitions': marker['returned_transitions'],
                   'prior_active_elapsed_s': marker['active_elapsed_s']}
        restored = load_transition(target.root, recipe, journal, binding)
        assert restored.components.trainer.update_index == 1
        assert restored.components.curriculum.consumed == marker['boundary']['reserved_by_stage']
        assert restored.components.trainer.consumed['actor_optimizer_steps'] == 1
        assert restored.components.trainer.actor_optimizer.state
        bad_binding = dict(binding, start_update=0)
        with pytest.raises(ValueError, match='rebased'):
            load_transition(target.root, recipe, journal, bad_binding)
    for relative, raw in old_files.items():
        assert (value.root / relative).read_bytes() == raw
    assert read_json(value.root / 'structured.json').get('source_transition_required') is None
    target.start()
    final = wait(target, lambda s: not s['active'])
    assert target.submitted_process.wait(timeout=5) == 0
    assert final['phase'] == 'stopped' and final['update'] == 2
    assert final['cumulative_transitions'] == 64
    row = (target.attempt() / 'train/metrics.jsonl').read_text().splitlines()
    assert len(row) == 1


def test_unsealed_transition_never_becomes_a_fresh_run(tmp_path, monkeypatch):
    value, _ = session(tmp_path, monkeypatch)
    value.spec['source_transition_required'] = True
    with pytest.raises(ValueError, match='unsealed'):
        value.start()
    assert value.attempt() is None
    assert not (value.root / 'journal').exists()


def test_source_review_rejects_unlisted_and_policy_changes(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for package in (old, new):
        (package / 'formal/tasks').mkdir(parents=True)
        (package / 'formal/tasks/terrain.py').write_text('# before\n')
    (new / 'formal/tasks/terrain.py').write_text('# after\n')
    with pytest.raises(ValueError, match='reviewed'):
        _review_sources(old, new, {})
    (new / 'formal/policy.py').write_text('# unapproved policy\n')
    before, after = source_table(old), source_table(new)
    review = {name: {'before': before.get(name), 'after': after.get(name)}
              for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    with pytest.raises(ValueError, match='reviewed'):
        _review_sources(old, new, review)


def test_learning_state_proof_detects_tensor_rng_and_counter_changes():
    fields = ('model', 'actor_adam', 'critic_adam', 'trainer', 'collection', 'curriculum',
              'sampler', 'terrain', 'protocol', 'global_rng', 'normalization')
    payload = {name: {'value': torch.tensor([1., 2.]), 'count': 7} for name in fields}
    original = learning_state_sha256(payload)
    assert learning_state_sha256(dict(payload, recipe='new', boundary='new', metadata='new')) == original
    for name in fields:
        changed = dict(payload, **{name: {'value': torch.tensor([1., 3.]), 'count': 7}})
        assert learning_state_sha256(changed) != original


def test_wrong_boot_approval_refuses_before_creating_target(tmp_path, monkeypatch):
    value, _ = session(tmp_path, monkeypatch)
    package, review = candidate(value, tmp_path)
    with pytest.raises(ValueError, match='approval'):
        migrate_stopped_session(value.root, tmp_path / 'denied', learner_source=package,
                               reviewed_changes=review, approval_pin=approve(tmp_path, boot='0' * 36),
                               authorized_boot=boot_id())
    assert not (tmp_path / 'denied').exists()

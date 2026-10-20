"""Stage smoke offsets must never become resumed learning consumption."""
from types import SimpleNamespace

import pytest

from transformer_rl.formal.lab_env import train
from transformer_rl.formal.lab_env.formal_curriculum import FormalTaskCurriculum


def invocation(**changes):
    values = dict(resume=None, init_checkpoint=None, formal_curriculum=True,
                  include_jump=True, diagnostic_only=False, diagnostic_stage=None)
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize('stage', range(4))
def test_stage_validation_has_explicit_non_learning_provenance(stage):
    scope = train._invocation_scope(invocation(diagnostic_only=True, diagnostic_stage=stage))
    assert scope == {'validation_only': True, 'diagnostic_stage': stage,
                     'budget_scope': 'diagnostic_stage_offset', 'initialization': 'fresh'}
    with pytest.raises(ValueError, match='validation-only'):
        train._require_resume_scope({'format': 'lab_ppo_v1', 'update': 1, **scope})


@pytest.mark.parametrize('changes', (
    {'diagnostic_only': True},
    {'diagnostic_stage': 0},
    {'diagnostic_only': True, 'diagnostic_stage': 3, 'include_jump': False},
    {'diagnostic_only': True, 'diagnostic_stage': 0, 'formal_curriculum': False},
    {'diagnostic_only': True, 'diagnostic_stage': 0, 'resume': 'checkpoint.pt'},
    {'diagnostic_only': True, 'diagnostic_stage': True},
    {'diagnostic_only': 1, 'diagnostic_stage': 0},
    {'resume': 'checkpoint.pt', 'init_checkpoint': 'other.pt'},
))
def test_invalid_validation_invocation_fails_before_sdk_or_output(
        changes, monkeypatch, tmp_path):
    args = invocation(**changes)
    args.output = tmp_path / 'must_not_be_created'
    monkeypatch.setattr(train, '_parse_args', lambda _argv: args)
    with pytest.raises(ValueError):
        train.main([])
    assert not args.output.exists()


@pytest.mark.parametrize('source,key', (('resume', 'resume'), ('warm_start', 'init_checkpoint')))
def test_resume_and_warm_start_have_distinct_origins(source, key):
    scope = train._invocation_scope(invocation(**{key: 'checkpoint.pt'}))
    assert scope['initialization'] == source
    assert scope['validation_only'] is False
    assert scope['budget_scope'] == 'lab_learning_updates'


@pytest.mark.parametrize('include_jump', (False, True))
def test_real_curriculum_checkpoint_preserves_count_without_stage_offset(include_jump):
    curriculum = FormalTaskCurriculum(2, include_jump=include_jump)
    curriculum.budget.charge_rollout()
    payload = {'format': 'lab_ppo_v1', 'update': 1, 'formal_curriculum': True,
               'curriculum': curriculum.state_dict()}
    train._require_resume_scope(payload)

    # Earlier stage probes did not persist a validation-only marker. Their
    # prepaid stage offsets must still be rejected when presented as resume.
    payload['curriculum']['budget']['consumed'][0] = 5000
    payload['curriculum']['budget']['consumed'][1] = 1
    with pytest.raises(ValueError, match='actual completed PPO updates'):
        train._require_resume_scope(payload)


@pytest.mark.parametrize('marker', (
    {'validation_only': True},
    {'diagnostic_stage': 0},
    {'budget_scope': 'diagnostic_stage_offset'},
))
def test_partial_validation_marker_still_forbids_resume(marker):
    with pytest.raises(ValueError, match='validation-only'):
        train._require_resume_scope({'format': 'lab_ppo_v1', 'update': 0, **marker})


@pytest.mark.parametrize('payload', (
    None,
    {'format': 'structured_checkpoint_v1', 'update': 1},
    {'format': 'lab_ppo_v1', 'update': -1},
    {'format': 'lab_ppo_v1', 'update': True},
    {'format': 'lab_ppo_v1', 'update': 1, 'formal_curriculum': True},
    {'format': 'lab_ppo_v1', 'update': 1, 'formal_curriculum': True,
     'curriculum': {'budget': {'consumed': [True, 0, 0]}}},
))
def test_invalid_or_different_checkpoint_contract_is_rejected(payload):
    with pytest.raises(ValueError):
        train._require_resume_scope(payload)

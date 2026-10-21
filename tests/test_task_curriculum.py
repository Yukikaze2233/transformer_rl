"""Continuous-domain and budget algebra; no capability qualification is implied."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib

import pytest
import torch

from transformer_rl.formal.tasks.curriculum import (
    AuditedStageDecision, ContinuousTaskSampler, CurriculumBudget, STAGE_BUDGETS,
    STAGE_NAMES, STAGE_WEIGHTS, TaskDomain, TaskDraw, TaskFamily,
)


def assert_draw_equal(left, right):
    for field in fields(TaskDraw):
        assert torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name


def decision(budget, **changes):
    sha = hashlib.sha256(f'{budget.stage}/{budget.total_updates}'.encode()).hexdigest()
    value = AuditedStageDecision(STAGE_NAMES[budget.stage], budget.total_updates, sha, 'a' * 64,
                                 (44101, 44102), True, True, True)
    return replace(value, **changes)


def charge_to(budget, total):
    while budget.total_updates < total:
        budget.charge_rollout()


def test_final_mixture_contains_full_continuous_domains_and_both_endpoints():
    sampler = ContinuousTaskSampler(2101)
    draw = sampler.sample(10000, 2)
    assert torch.bincount(draw.family, minlength=6).tolist() == [1500, 1500, 1500, 3500, 0, 2000]
    assert torch.count_nonzero(draw.requested_command[:, 1]) == 0
    height = draw.requested_command[:, 3]
    assert (height >= 0.20).all() and (height <= 0.35).all()
    assert height.min() < 0.201 and height.max() > 0.349
    motion = draw.requested_command[draw.family == TaskFamily.MOTION]
    assert motion[:, 0].min() == -2.7 and motion[:, 0].max() == 2.7
    assert torch.count_nonzero(motion[:, 2]) == 0
    rotation = draw.requested_command[draw.family == TaskFamily.ROTATION]
    torch.testing.assert_close(rotation[:, 2].min(), torch.tensor(-6 * torch.pi))
    torch.testing.assert_close(rotation[:, 2].max(), torch.tensor(6 * torch.pi))
    assert (rotation[:, 0] == 0).sum() > 400
    assert set(draw.terrain_kind[draw.family == TaskFamily.TERRAIN].tolist()) == set(range(1, 8))
    assert set(draw.recovery_posture[draw.family == TaskFamily.RECOVERY].tolist()) == set(range(16))
    assert set(draw.trajectory_kind.tolist()) == set(range(7))
    assert 0.15 < draw.nominal.float().mean() < 0.25


def test_jump_stage_samples_full_confirmed_rise_range_without_changing_quota_or_budget():
    from transformer_rl.formal.tasks.curriculum import (
        COMPLETE_STAGE_BUDGETS, COMPLETE_STAGE_WEIGHTS,
    )

    sampler = ContinuousTaskSampler(2102, stage_weights=COMPLETE_STAGE_WEIGHTS)
    draw = sampler.sample(10000, 3)
    assert torch.bincount(draw.family, minlength=6).tolist() == [2000, 1000, 1000, 1000, 5000, 0]
    rise = draw.jump_height_m[draw.family == TaskFamily.REQUEST]
    torch.testing.assert_close(rise.min(), torch.tensor(.03))
    torch.testing.assert_close(rise.max(), torch.tensor(.40))
    assert (rise > .30).any()
    assert sum(COMPLETE_STAGE_BUDGETS) == 35000
    saved = sampler.state_dict()
    resumed = ContinuousTaskSampler(7, stage_weights=COMPLETE_STAGE_WEIGHTS)
    resumed.load_state_dict(saved)
    assert_draw_equal(sampler.sample(193, 3), resumed.sample(193, 3))


def test_strata_are_70_15_15_and_not_a_discrete_command_grid():
    draw = ContinuousTaskSampler(2102).sample(10000, 2)
    for family, count in enumerate((1500, 1500, 1500, 3500, 0, 2000)):
        if count == 0:
            continue
        selected = draw.family == family
        assert torch.bincount(draw.stratum[selected], minlength=3).tolist() == [int(count * 0.7),
                                                                               int(count * 0.15),
                                                                               int(count * 0.15)]
        interior_height = draw.requested_command[selected & (draw.stratum == 0), 3]
        assert interior_height.unique().numel() > interior_height.numel() * 0.95
    near_motion = (draw.family == TaskFamily.MOTION) & (draw.stratum == 2)
    assert (draw.requested_command[near_motion, 0] == 0).any()


def test_one_row_reset_batches_do_not_starve_rare_families_or_strata():
    sampler = ContinuousTaskSampler(2103)
    family_counts = torch.zeros(6, dtype=torch.long)
    strata_counts = torch.zeros(6, 3, dtype=torch.long)
    for _ in range(1000):
        draw = sampler.sample(1, 2)
        family_counts[draw.family[0]] += 1
        strata_counts[draw.family[0], draw.stratum[0]] += 1
    torch.testing.assert_close(family_counts, torch.tensor([150, 150, 150, 350, 0, 200]))
    for family in range(6):
        if family_counts[family] == 0:
            continue
        assert (strata_counts[family] > 0).all()
        expected = family_counts[family] * torch.tensor([0.7, 0.15, 0.15])
        assert (strata_counts[family] - expected).abs().max() <= 1


@pytest.mark.parametrize('stage', range(3))
def test_stage_introduction_preserves_rehearsal_floors(stage):
    draw = ContinuousTaskSampler(2101).sample(10000, stage)
    actual = torch.bincount(draw.family, minlength=6).float() / 10000
    torch.testing.assert_close(actual, torch.tensor(STAGE_WEIGHTS[stage]))
    assert actual[TaskFamily.HEIGHT] >= 0.1
    if stage >= 1:
        assert actual[TaskFamily.MOTION] >= 0.1
    if stage >= 2:
        assert actual[TaskFamily.ROTATION] >= 0.1
    assert not draw.jump_height_m[draw.family != TaskFamily.REQUEST].any()


def test_sampler_resume_restores_rng_and_all_fractional_quota_deficits():
    original = ContinuousTaskSampler(2101)
    original.sample(13, 0)
    original.sample(19, 1)
    original.sample(37, 2)
    saved = original.state_dict()
    restored = ContinuousTaskSampler(999)
    restored.load_state_dict(saved)
    for stage, count in ((2, 107), (1, 23), (0, 1), (2, 71)):
        assert_draw_equal(original.sample(count, stage), restored.sample(count, stage))
    saved['family_credit'][2][0] = 100
    assert restored.state_dict()['family_credit'][2][0] != 100


def test_sampler_uses_its_own_rng_and_rejects_corrupt_resume_atomically():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    sampler = ContinuousTaskSampler(2101)
    sampler.sample(1024, 2)
    assert torch.equal(before, torch.random.get_rng_state())
    state = sampler.state_dict()
    bad = deepcopy(state)
    bad['stratum_credit'][0][0][0] = float('nan')
    with pytest.raises(ValueError):
        sampler.load_state_dict(bad)
    valid = ContinuousTaskSampler(0)
    valid.load_state_dict(state)
    assert_draw_equal(sampler.sample(127, 2), valid.sample(127, 2))


def test_successful_curriculum_spends_exactly_30000_without_refunding_early_stage_budget():
    budget = CurriculumBudget(promotion='audited')
    for stage in range(2):
        assert budget.stage == stage
        charge_to(budget, budget.total_updates + (STAGE_BUDGETS[stage] + 3) // 4)
        budget.submit_stage_validation(decision(budget))
        budget.charge_rollout()
        budget.submit_stage_validation(decision(budget))
    assert budget.stage == 2 and budget.transferred_to_consolidation > 0
    charge_to(budget, 30000)
    assert budget.status == 'training_budget_complete_evaluation_pending'
    assert budget.total_updates == 30000
    assert budget.total_updates * 1024 * 48 == 1474560000
    assert budget.total_updates * 1024 * 48 * 3 == 4423680000
    with pytest.raises(RuntimeError):
        budget.charge_rollout()


def test_full_stage_validation_before_minimum_exposure_cannot_promote():
    budget = CurriculumBudget(promotion='audited')
    charge_to(budget, 1249)
    original = budget.state_dict()
    with pytest.raises(ValueError, match='25 percent'):
        budget.submit_stage_validation(decision(budget))
    assert budget.state_dict() == original


def test_duplicate_checkpoint_and_heldout_selection_are_rejected():
    budget = CurriculumBudget(promotion='audited')
    charge_to(budget, 1250)
    with pytest.raises(ValueError):
        budget.submit_stage_validation(decision(budget, heldout_used=True))
    with pytest.raises(ValueError):
        budget.submit_stage_validation(decision(budget, validation_seeds=(44101, 44901)))
    first = decision(budget)
    budget.submit_stage_validation(first)
    with pytest.raises(ValueError, match='distinct'):
        budget.submit_stage_validation(first)
    budget.charge_rollout()
    with pytest.raises(ValueError, match='distinct'):
        budget.submit_stage_validation(decision(budget, checkpoint_sha256=first.checkpoint_sha256))


def test_validation_failures_reset_pass_streak_and_four_candidates_are_bounded():
    budget = CurriculumBudget(promotion='audited')
    charge_to(budget, 1250)
    for index, passed in enumerate((True, False, True, True)):
        if index:
            budget.charge_rollout()
        budget.submit_stage_validation(decision(budget, passed=passed))
    assert budget.stage == 1
    failed = CurriculumBudget(promotion='audited')
    charge_to(failed, 1250)
    for index in range(4):
        if index:
            failed.charge_rollout()
        failed.submit_stage_validation(decision(failed, coverage_passed=False))
    assert failed.status == 'failed_qualification'
    assert failed.total_updates == 1253
    with pytest.raises(RuntimeError):
        failed.charge_rollout()


def test_stage_cap_requires_qualification_and_does_not_silently_skip_forward():
    budget = CurriculumBudget(promotion='audited')
    charge_to(budget, 5000)
    assert budget.stage == 0 and budget.status == 'stage_budget_complete_validation_pending'
    with pytest.raises(RuntimeError):
        budget.charge_rollout()
    budget.submit_stage_validation(decision(budget, passed=False))
    assert budget.stage == 0 and budget.status == 'failed_qualification'
    assert budget.total_updates == 5000


def test_budget_checkpoint_is_replayed_and_tampering_rejected_without_mutation():
    budget = CurriculumBudget(promotion='audited')
    charge_to(budget, 1250)
    budget.submit_stage_validation(decision(budget))
    budget.charge_rollout()
    budget.submit_stage_validation(decision(budget))
    budget.charge_rollout()
    saved = budget.state_dict()
    restored = CurriculumBudget(promotion='audited')
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    bad = deepcopy(saved)
    bad['consumed'][0] -= 1
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
    assert restored.state_dict() == saved


def test_default_exposure_schedule_covers_all_three_phases_without_evaluation_or_refund():
    budget = CurriculumBudget()
    stages = []
    for _ in range(30000):
        stages.append(budget.prepare_rollout())
        budget.charge_rollout()
    assert [stages.count(stage) for stage in range(3)] == list(STAGE_BUDGETS)
    assert budget.consumed == list(STAGE_BUDGETS) and budget.total_updates == 30000
    assert budget.status == 'training_budget_complete'
    assert budget.decisions == [] and budget.transferred_to_consolidation == 0
    assert budget.total_updates * 1024 * 48 * 3 == 4423680000
    with pytest.raises(RuntimeError):
        budget.charge_rollout()


def test_exposure_schedule_resumes_on_both_sides_of_the_boundary_without_promoting_mid_rollout():
    budget = CurriculumBudget()
    charge_to(budget, 5000)
    assert budget.stage == 0 and budget.status == 'running'
    for stage in (0, 1):
        if stage == 1:
            budget.prepare_rollout()
        saved = budget.state_dict()
        restored = CurriculumBudget()
        restored.load_state_dict(saved)
        assert restored.state_dict() == saved
        restored.charge_rollout()
        assert restored.stage == 1 and restored.consumed[:2] == [5000, 1]
    with pytest.raises(RuntimeError, match='manual evaluation'):
        budget.submit_stage_validation(decision(budget))
    with pytest.raises(ValueError):
        CurriculumBudget(promotion='audited').load_state_dict(budget.state_dict())


@pytest.mark.parametrize('changes', [
    {'height_m': (0.4, 0.17)}, {'nominal_height_m': 0.5}, {'forward_m_s': (0, 5)},
    {'jump_m': (0, 0.3)}, {'nominal_fraction': float('nan')},
])
def test_invalid_domain_does_not_become_silent_sampling_clipping(changes):
    with pytest.raises(ValueError):
        replace(TaskDomain(), **changes)


def test_dynamic_disable_retires_fractional_credit_and_roundtrips():
    from copy import deepcopy
    from transformer_rl.formal.tasks.curriculum import COMPLETE_STAGE_WEIGHTS

    sampler = ContinuousTaskSampler(2102, stage_weights=COMPLETE_STAGE_WEIGHTS, continuous=True)
    for _ in range(12):
        sampler.sample(1000, 2, weights=COMPLETE_STAGE_WEIGHTS[2])
    # The jump-stage target disables recovery at the boundary.
    sampler.sample(1000, 3, weights=COMPLETE_STAGE_WEIGHTS[3])
    state = sampler.state_dict()
    assert state['dynamic_family_credit'][5] == 0
    assert abs(sum(state['dynamic_family_credit'])) < 1e-6
    restored = ContinuousTaskSampler(7, stage_weights=COMPLETE_STAGE_WEIGHTS, continuous=True)
    restored.load_state_dict(state)
    assert_draw_equal(sampler.sample(97, 3, weights=COMPLETE_STAGE_WEIGHTS[3]),
                      restored.sample(97, 3, weights=COMPLETE_STAGE_WEIGHTS[3]))

    # A pre-fix checkpoint may carry a residual deficit for a now-disabled
    # family. Loading it must retire that credit rather than fail or carry it.
    old = deepcopy(state)
    old['dynamic_family_credit'][5] = .25
    restored.load_state_dict(old)
    assert restored._dynamic_family.credit[5] == 0
    assert abs(sum(restored._dynamic_family.credit)) < 1e-6

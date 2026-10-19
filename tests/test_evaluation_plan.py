"""Private full evaluation catalogue/cost/selection checks; no simulator."""
import json
import math
from pathlib import Path

import pytest

from transformer_rl.formal.evaluation.plan import (
    EvaluationSelection, FAMILIES, INTRODUCED_AT, PROBE_CASE_NAMES,
    evaluation_cost_bound, evaluation_plan, formal_cases,
)


def test_all_original_case_parameters_and_denominators_are_retained():
    original = json.loads((Path(__file__).parent / 'formal_evaluation_design.json').read_text())
    cases = formal_cases()
    assert [case.request() for case in cases] == original['cases']
    assert len(cases) == len({case.name for case in cases}) == 144
    assert {family: sum(c.family == family for c in cases) for family in FAMILIES} == original['counts']
    assert sum(c.duration_s for c in cases) == 5280
    # Necessary wheel limits do not silently delete difficult requested pairs.
    assert max(c.parameters.get('wz_rad_s', 0) for c in cases) == 6 * math.pi
    assert max(c.parameters.get('vx_m_s', 0) for c in cases) == 5
    assert all(c.request()['geometry_and_runtime_resolution_required'] for c in cases)
    changed = cases[0].parameters
    changed['height_m'] = 99
    assert cases[0].parameters['height_m'] == .17


def test_final_seed_repetitions_and_stage_protection_are_exact():
    selection = EvaluationSelection(2101, 'a' * 64, 'b' * 64, 'c' * 64)
    validation = evaluation_plan('final_validation', 2101, 'a' * 64)
    heldout = evaluation_plan('heldout', 2101, 'a' * 64, selection=selection)
    assert len(validation.trials) == len(heldout.trials) == 5760
    assert {t.evaluation_seed for t in validation.trials} == {44101, 44102}
    assert {t.evaluation_seed for t in heldout.trials} == {44901, 44902}
    assert len({(t.case_name, t.evaluation_seed, t.repetition) for t in validation.trials}) == 5760
    assert validation.sha256 != heldout.sha256
    assert evaluation_plan('final_validation', 2102, 'a' * 64).sha256 != validation.sha256
    for stage, count in enumerate((40, 72, 104, 128, 144, 144)):
        plan = evaluation_plan('stage', 2101, 'a' * 64, stage=stage)
        assert len(plan.trials) == count * 2 * 4
        assert all(INTRODUCED_AT[t.family] <= stage for t in plan.trials)
        assert sum(t.family == 'recovery' for t in plan.trials) == 16 * 2 * 4
    assert len(PROBE_CASE_NAMES) == len(set(PROBE_CASE_NAMES)) == 12
    assert len(evaluation_plan('probe', 2101, 'a' * 64).trials) == 12 * 2 * 2


@pytest.mark.parametrize('tier', ['heldout', 'challenges'])
def test_selection_and_challenge_manifest_cannot_be_omitted_or_rebound(tier):
    with pytest.raises(ValueError, match='selected'):
        evaluation_plan(tier, 2101, 'a' * 64)
    wrong = EvaluationSelection(2101, 'd' * 64, 'b' * 64, 'c' * 64)
    with pytest.raises(ValueError, match='selected'):
        evaluation_plan(tier, 2101, 'a' * 64, selection=wrong)
    right = EvaluationSelection(2101, 'a' * 64, 'b' * 64, 'c' * 64)
    if tier == 'challenges':
        with pytest.raises(ValueError, match='manifest'):
            evaluation_plan(tier, 2101, 'a' * 64, selection=right)
        plan = evaluation_plan(tier, 2101, 'a' * 64, selection=right, challenge_manifest_sha256='f' * 64)
        assert len(plan.trials) == 6 * 4 * 20 and all(t.duration_s == 120 for t in plan.trials)


def test_costs_use_complete_horizons_instead_of_early_failure_or_probe_substitution():
    assert evaluation_cost_bound() == {
        'probes': 19200000, 'stage_candidates': 72960000, 'final_validation': 21120000,
        'heldout': 21120000, 'challenges': 5760000, 'total_task_transitions': 140160000,
        'policy_dt_s': .01, 'preparation_and_discarded_vector_rows_counted_separately': True,
        'physical_qualification': False,
    }

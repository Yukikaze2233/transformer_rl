"""Independent probability/moment oracles and complete request denominators."""
from dataclasses import replace
import json
import math

import pytest
from scipy.stats import beta

from transformer_rl._evaluation_plan import evaluation_plan
from transformer_rl._evaluation_results import (
    RecordedTrial, exact_binomial_limits, pool_moments, summarize_evaluation,
)


@pytest.mark.parametrize('n', [1, 2, 20, 40, 200])
def test_binomial_inversion_matches_independent_beta_quantiles_and_keeps_onesided_distinct(n):
    for k in {0, 1, n // 2, n - 1, n}:
        value = exact_binomial_limits(k, n)
        expected = [0. if k == 0 else beta.ppf(.025, k, n - k + 1),
                    1. if k == n else beta.ppf(.975, k + 1, n - k)]
        assert value['two_sided'] == pytest.approx(expected, abs=2e-12)
        assert value['one_sided_lower'] == pytest.approx(0. if not k else beta.ppf(.05, k, n - k + 1), abs=2e-12)
    if n == 40:
        assert exact_binomial_limits(40, 40)['one_sided_lower'] == pytest.approx(.05 ** (1 / 40))
        assert exact_binomial_limits(40, 40)['one_sided_lower'] < .95


def moment(center, seconds):
    return {'physical_samples': round(seconds * 200), 'support_s': seconds, 'mean': [center],
            'rms': [abs(center)], 'centered_std': [0.], 'min': [center], 'max': [center], 'units': ['m_s']}


def row(healthy=True, center=1.):
    seconds = 30. if healthy else 10.
    return {'started': True, 'finished': True, 'declared_horizon_s': 30, 'observed_task_s': seconds,
            'whole_task_healthy': healthy, 'whole_task_policy_owned': True,
            'healthy_policy_owned_full_horizon': healthy, 'outcome': 'time_limit' if healthy else 'true_failure',
            'moments': {'full_task': {'error': moment(center, seconds)}}}


def record(plan, index, value, domain='feasible_verified'):
    return RecordedTrial(plan.sha256, index, 'b' * 64, domain,
                         'c' * 64 if domain != 'uncharacterized' else None, json.dumps(value))


def test_pooled_moments_use_all_physical_time_and_between_episode_variation():
    value = pool_moments([moment(1., 30.), moment(3., 10.)])
    assert value['mean'] == [1.5] and value['rms'] == pytest.approx([math.sqrt(3)])
    assert value['centered_std'] == pytest.approx([math.sqrt(.75)])
    assert value['physical_samples'] == 8000 and value['support_s'] == 40


def test_plan_denominator_missing_trials_and_failed_prefixes_remain_visible_without_cross_seed_pooling():
    plan = evaluation_plan('probe', 2101, 'a' * 64, stage=0)
    records = [record(plan, 0, row()), record(plan, 1, row(False, 3.))]
    report = summarize_evaluation(plan, iter(records))
    assert report['requested_denominator'] == 16 and report['recorded_first_episodes'] == 2
    assert len(report['unobserved_trial_indices']) == 14
    assert not report['complete_first_episode_report'] and not report['stage_decision_available']
    group = report['groups'][0]
    assert group['health_ratio_interval']['observed_ratio'] == .5
    assert group['true_failures'] == 1 and group['healthy_first_episode_count'] == 1
    assert group['pooled_physical_moments']['full_task']['error']['rms'] == pytest.approx([math.sqrt(3)])
    assert report['domain_counts']['feasible_verified'] == 2
    other = evaluation_plan('probe', 2102, 'a' * 64, stage=0)
    with pytest.raises(ValueError, match='another'):
        summarize_evaluation(other, records)
    with pytest.raises(ValueError, match='replace'):
        summarize_evaluation(plan, [records[0], records[0]])


def test_uncharacterized_and_infeasible_requests_keep_their_own_denominator_and_no_premature_interval():
    plan = evaluation_plan('probe', 2101, 'a' * 64, stage=0)
    missing = summarize_evaluation(plan, [record(plan, 0, row(), 'uncharacterized')])
    assert missing['domain_counts']['uncharacterized'] == 1
    assert missing['groups'][0]['health_ratio_interval'] is None
    incomplete = row()
    incomplete.update(finished=False, observed_task_s=2., outcome='incomplete', healthy_policy_owned_full_horizon=False)
    incomplete['moments']['full_task']['error'] = moment(1., 2.)
    report = summarize_evaluation(plan, [record(plan, 0, incomplete, 'infeasible_verified'),
                                         record(plan, 1, row(), 'uncharacterized')])
    assert report['requested_denominator'] == 16 and report['domain_counts']['infeasible_verified'] == 1
    assert all(group['health_ratio_interval'] is None for group in report['groups'])
    with pytest.raises(ValueError, match='reachability'):
        replace(record(plan, 0, row()), domain_evidence_sha256=None)


def test_successful_short_prefix_and_missing_vector_metric_are_not_complete_results():
    plan = evaluation_plan('probe', 2101, 'a' * 64, stage=0)
    short = row()
    short['observed_task_s'] = 1.
    with pytest.raises(ValueError, match='full-horizon'):
        summarize_evaluation(plan, [record(plan, 0, short)])
    unsupported = moment(1., 1.)
    unsupported.update(physical_samples=0, support_s=0.)
    with pytest.raises(ValueError, match='null'):
        pool_moments([unsupported])
    other = row()
    other['moments']['full_task'] = {}
    with pytest.raises(ValueError, match='drop'):
        summarize_evaluation(plan, [record(plan, 0, row()), record(plan, 1, other)])

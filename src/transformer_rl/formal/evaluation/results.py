"""Private per-training-seed evaluation accounting and statistical projections.

Missing, unsupported and unknown trials keep their requested denominators.
Observed health is not a terrain/request skill pass or native qualification.
The exact binomial inversion follows the binomial-tail equations documented by
NIST Dataplot; its confidence interpretation assumes independent common-p trials
within the reported case/seed/domain group,not arbitrary pooled training runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from .plan import EvaluationPlan, _sha


def exact_binomial_limits(successes, count, *, confidence=.95):
    """Two-sided Clopper-Pearson interval and a separate one-sided lower limit."""
    if (type(successes) is not int or type(count) is not int or not 0 <= successes <= count
            or type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError('exact binomial limits require integer success/trial counts and finite confidence')
    if not count:
        return {'observed_ratio': None, 'two_sided': [0., 1.], 'one_sided_lower': 0.,
                'confidence': confidence, 'successes': 0, 'trials': 0}
    alpha = 1 - confidence

    def tail(p, indices):
        if p <= 0:
            return float(0 in indices)
        if p >= 1:
            return float(count in indices)
        return min(1., math.fsum(math.exp(math.lgamma(count + 1) - math.lgamma(i + 1)
                                        - math.lgamma(count - i + 1) + i * math.log(p)
                                        + (count - i) * math.log1p(-p)) for i in indices))

    def lower(target):
        if successes == 0:
            return 0.
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo + hi) / 2
            if tail(mid, range(successes, count + 1)) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def upper(target):
        if successes == count:
            return 1.
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo + hi) / 2
            if tail(mid, range(0, successes + 1)) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return {'observed_ratio': successes / count, 'two_sided': [lower(alpha / 2), upper(alpha / 2)],
            'one_sided_lower': lower(alpha), 'confidence': confidence, 'successes': successes, 'trials': count}


def pool_moments(records):
    """Pool physical-time moments including between-episode variation,not RMS means."""
    if not records:
        return None
    units = records[0]['units']
    if not isinstance(units, list) or not units or any(record.get('units') != units for record in records):
        raise ValueError('pooled physical moments require matching explicit channel units')
    width, weight, samples = len(units), 0., 0
    mean, m2 = [0.] * width, [0.] * width
    minimum, maximum = [float('inf')] * width, [-float('inf')] * width
    for record in records:
        seconds, count = record['support_s'], record['physical_samples']
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or type(count) is not int or count < 0 or (count == 0) != (seconds == 0)):
            raise ValueError('physical support cannot be omitted,negative or replaced with endpoint count')
        if not count:
            if any(record.get(name) is not None for name in ('mean', 'rms', 'centered_std', 'min', 'max')):
                raise ValueError('unsupported physical channels must remain explicitly null')
            continue
        for name in ('mean', 'rms', 'centered_std', 'min', 'max'):
            values = record[name]
            if (not isinstance(values, list) or len(values) != width
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError('physical moments need complete finite vectors')
        total = weight + seconds
        for index in range(width):
            center, std, rms = (record[name][index] for name in ('mean', 'centered_std', 'rms'))
            if (std < 0 or rms < 0 or record['min'][index] > center + 1e-8
                    or record['max'][index] < center - 1e-8
                    or not math.isclose(rms * rms, std * std + center * center, rel_tol=1e-8, abs_tol=1e-10)):
                raise ValueError('physical moment arithmetic or extrema disagree')
            delta = center - mean[index]
            m2[index] += seconds * std * std + delta * delta * weight * seconds / total
            mean[index] += delta * seconds / total
            minimum[index] = min(minimum[index], record['min'][index])
            maximum[index] = max(maximum[index], record['max'][index])
        weight, samples = total, samples + count
    if not samples:
        return {'physical_samples': 0, 'support_s': 0., 'mean': None, 'rms': None,
                'centered_std': None, 'min': None, 'max': None, 'units': list(units)}
    variance = [max(0., value / weight) for value in m2]
    return {'physical_samples': samples, 'support_s': weight, 'mean': mean,
            'rms': [math.sqrt(v + m * m) for v, m in zip(variance, mean)],
            'centered_std': [math.sqrt(v) for v in variance], 'min': minimum, 'max': maximum,
            'units': list(units)}


@dataclass(frozen=True)
class RecordedTrial:
    plan_sha256: str
    trial_index: int
    observation_sha256: str
    reachability: str
    domain_evidence_sha256: str | None
    statistics_json: str

    def __post_init__(self):
        if (not _sha(self.plan_sha256) or not _sha(self.observation_sha256)
                or type(self.trial_index) is not int or self.trial_index < 0
                or self.reachability not in ('feasible_verified', 'infeasible_verified', 'uncharacterized')
                or self.reachability != 'uncharacterized' and not _sha(self.domain_evidence_sha256)
                or self.reachability == 'uncharacterized' and self.domain_evidence_sha256 is not None):
            raise ValueError('trial content identities and independent reachability evidence must be explicit')
        value = json.loads(self.statistics_json)
        if not isinstance(value, dict):
            raise ValueError('trial statistics must be a complete JSON object')
        object.__setattr__(self, 'statistics_json', json.dumps(value, sort_keys=True, allow_nan=False))

    @property
    def statistics(self):
        return json.loads(self.statistics_json)


def summarize_evaluation(plan: EvaluationPlan, records):
    """One selected checkpoint and training seed per plan; no automatic promotion.

    Completed prefixes of failures contribute real physical error samples.
    Missing episodes contribute no fabricated zeros,CI successes or RMS values.
    Their planned entries stay visible and prevent a complete report claim.
    """
    if not isinstance(plan, EvaluationPlan):
        raise TypeError('evaluation summary requires its frozen complete request plan')
    records = tuple(records)
    plan_sha256 = plan.sha256
    by_index, grouped = {}, {}
    for record in records:
        if (not isinstance(record, RecordedTrial) or record.plan_sha256 != plan_sha256
                or record.trial_index >= len(plan.trials) or record.trial_index in by_index):
            raise ValueError('trial belongs to another plan/model/split or would replace a consumed first outcome')
        trial, row = plan.trials[record.trial_index], record.statistics
        for name in ('started', 'finished', 'whole_task_healthy', 'whole_task_policy_owned',
                     'healthy_policy_owned_full_horizon'):
            if type(row.get(name)) is not bool:
                raise ValueError('trial health/completion flags must be explicit booleans')
        horizon, elapsed = row.get('declared_horizon_s'), row.get('observed_task_s')
        if (type(horizon) not in (int, float) or horizon != trial.duration_s
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                or row.get('outcome') not in ('true_failure', 'time_limit', 'incomplete')
                or row['finished'] != (row['outcome'] != 'incomplete')):
            raise ValueError('trial cannot replace its full catalogue horizon with a successful prefix')
        expected_health = (row['started'] and row['finished'] and row['outcome'] == 'time_limit'
                           and elapsed + 1e-8 >= horizon and row['whole_task_healthy'] and row['whole_task_policy_owned'])
        if row['healthy_policy_owned_full_horizon'] != expected_health:
            raise ValueError('reported full-horizon health disagrees with its physical interval')
        moments = row.get('moments')
        if not isinstance(moments, dict) or not moments:
            raise ValueError('physical moment availability must be explicit')
        for scope in moments.values():
            if not isinstance(scope, dict):
                raise ValueError('physical moment scopes must contain named channels')
            for value in scope.values():
                if (not isinstance(value, dict) or type(value.get('support_s')) not in (int, float)
                        or not math.isfinite(value['support_s']) or not 0 <= value['support_s'] <= elapsed + 1e-8):
                    raise ValueError('physical moment support cannot outlive the observed first episode')
        by_index[record.trial_index] = row
        key = (trial.case_name, trial.family, trial.evaluation_seed, record.reachability)
        grouped.setdefault(key, []).append(row)
    groups = []
    complete_groups = {}
    for index, trial in enumerate(plan.trials):
        key = (trial.case_name, trial.evaluation_seed)
        complete_groups[key] = (complete_groups.get(key, True) and index in by_index and by_index[index]['finished'])
    for (case, family, seed, domain), rows in sorted(grouped.items()):
        healthy = sum(row['healthy_policy_owned_full_horizon'] for row in rows)
        keys = {(scope, name) for row in rows for scope, metrics in row.get('moments', {}).items() for name in metrics}
        moments = {}
        for scope, name in sorted(keys):
            values = [row.get('moments', {}).get(scope, {}).get(name) for row in rows]
            if any(value is None for value in values):
                raise ValueError('different trial metric sets cannot silently drop unsupported fields')
            moments.setdefault(scope, {})[name] = pool_moments(values)
        complete = sum(row['finished'] for row in rows)
        groups.append({'case_name': case, 'family': family, 'evaluation_seed': seed, 'reachability': domain,
                       'recorded_first_episodes': len(rows), 'finished_first_episodes': complete,
                       'true_failures': sum(row['outcome'] == 'true_failure' for row in rows),
                       'incomplete_first_episodes': len(rows) - complete,
                       'healthy_first_episode_count': healthy,
                       'case_seed_first_outcomes_complete': complete_groups[(case, seed)],
                       'health_ratio_interval': (exact_binomial_limits(healthy, len(rows))
                                                 if complete_groups[(case, seed)] else None),
                       'pooled_physical_moments': moments,
                       'support_includes_failed_first_episode_prefixes': True})
    missing = [index for index in range(len(plan.trials)) if index not in by_index]
    return {'plan_sha256': plan_sha256, 'training_seed': plan.training_seed, 'tier': plan.tier,
            'checkpoint_sha256': plan.checkpoint_sha256, 'requested_denominator': len(plan.trials),
            'recorded_first_episodes': len(by_index), 'unobserved_trial_indices': missing,
            'complete_first_episode_report': not missing and all(row['finished'] for row in by_index.values()),
            'domain_counts': {name: sum(record.reachability == name for record in records)
                              for name in ('feasible_verified', 'infeasible_verified', 'uncharacterized')},
            'groups': groups, 'observed_proportions_are_not_success_probability_guarantees': True,
            'interval_assumption': 'independent common-p episodes within each case/evaluation-seed/domain group',
            'intervals_are_not_training_seed_confidence_intervals': True,
            'native_recipe_and_skill_acceptance_required': True,
            'stage_decision_available': False, 'physical_qualification': False}

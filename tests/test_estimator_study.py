"""The study gate must reject collapsed, cancelling-error and censored policies."""
import copy
from pathlib import Path
import runpy

import pytest


GATE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/run_estimator_study.py'))['task_gate']
THRESHOLDS = {
    'minimum_healthy_timeout_episode_fraction_per_report': 0.9,
    'maximum_censored_sample_fraction': 0.1,
    'minimum_steady_coverage_per_report': 0.8,
    'stand_steady_height_mae_m': 0.015,
    'stand_steady_vx_mae_m_s': 0.03,
    'stand_steady_wz_mae_rad_s': 0.1,
}


def healthy():
    return {'transitions': 1000, 'control_quality': {'available': True,
            'healthy_timeout_fraction': 1.0, 'censored_sample_fraction': 0.0},
            'stability': {'signals': {name: {'mean': 0, 'mean_abs': 0.001, 'count': 900}
                                     for name in ('height_error', 'vx_error', 'wz_error')}}}


@pytest.mark.parametrize('case', ['collapsed', 'cancelling_error', 'censored', 'missing_mae', 'short'])
def test_task_gate_never_accepts_low_jitter_alone(case):
    report = copy.deepcopy(healthy())
    if case == 'collapsed':
        report['control_quality']['healthy_timeout_fraction'] = 0
    elif case == 'cancelling_error':
        report['stability']['signals']['height_error']['mean_abs'] = 0.08
    elif case == 'censored':
        report['control_quality']['censored_sample_fraction'] = 1.0
    elif case == 'missing_mae':
        report['stability']['signals']['height_error'].pop('mean_abs')
    else:
        report['stability']['signals']['height_error']['count'] = 300
    assert GATE(report, 'stand_mid', THRESHOLDS)
    assert not GATE(healthy(), 'stand_mid', THRESHOLDS)

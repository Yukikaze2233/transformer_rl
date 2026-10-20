"""Substep latch, mechanical failures and narrow BODY_BUMP exceptions."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.tasks.termination import FailureReason, NativeTaskLimits, PhysicalFailureLatch
from transformer_rl.formal.simulation.task import NativeTaskLimits as HistoricalLimits


def limits():
    return NativeTaskLimits(.08, .01, .5, 2.5, ('base_link',), 2., .02, .02, .01, 5.)


def sample(n=6, bodies=1):
    return dict(height_m=torch.full((n,), .32),
                gravity_body=torch.tensor([[0., 0., -1.]]).expand(n, -1).clone(),
                forbidden_body_load_n=torch.zeros(n, bodies),
                knee_signed_clearance_rad=torch.full((n, 2), .1),
                knee_pivot_residual_m=torch.zeros(n, 2),
                within_pressure_stroke=torch.ones(n, 2, dtype=torch.bool),
                recovery=torch.zeros(n, dtype=torch.bool))


@pytest.mark.parametrize('dt', (.005, .001))
def test_contact_confirmation_uses_physical_time_and_reset_is_partial(dt):
    latch = PhysicalFailureLatch(limits(), 2, physics_dt_s=dt)
    value = sample(2)
    value['forbidden_body_load_n'].fill_(3)
    for _ in range(round(.02 / dt) - 1):
        assert not latch.observe(**value).true_failure.any()
    result = latch.observe(**value)
    assert result.failure_reasons.tolist() == [4, 4]
    latch.reset(torch.tensor([True, False]))
    assert latch.failure_reasons.tolist() == [0, 4]
    value['forbidden_body_load_n'].zero_()
    result = latch.observe(**value)
    assert result.true_failure.tolist() == [False, True]
    assert latch.contact_time_s.eq(0).all()


def test_six_failure_sources_are_independent_and_remain_latched():
    latch = PhysicalFailureLatch(limits(), 6, physics_dt_s=.005)
    value = sample()
    value['height_m'][0] = .07
    value['gravity_body'][1] = torch.tensor([0., 0., 1.])
    value['forbidden_body_load_n'][2] = 3
    value['knee_signed_clearance_rad'][3, 0] = -.021
    value['knee_pivot_residual_m'][4, 1] = .011
    value['within_pressure_stroke'][5, 0] = False
    for _ in range(4):
        result = latch.observe(**value)
    assert result.failure_reasons.tolist() == [1, 2, 4, 8, 16, 32]
    assert latch.observe(**sample()).failure_reasons.tolist() == [1, 2, 4, 8, 16, 32]


def test_body_bump_only_exempts_base_contact_not_mechanical_or_pose_failures():
    latch = PhysicalFailureLatch(limits(), 4, physics_dt_s=.005)
    value = sample(4)
    value['forbidden_body_load_n'].fill_(30)
    value['body_contact_allowed'] = torch.ones(4, dtype=torch.bool)
    value['height_m'][1] = .07
    value['knee_pivot_residual_m'][2, 0] = .02
    value['gravity_body'][3] = torch.tensor([0., 0., 1.])
    for _ in range(5):
        result = latch.observe(**value)
    assert result.failure_reasons.tolist() == [0, 1, 16, 2]
    assert result.healthy.tolist() == [True, False, False, False]


def test_recovery_relaxes_pose_and_contact_but_not_mechanical_stroke():
    latch = PhysicalFailureLatch(limits(), 3, physics_dt_s=.005)
    value = sample(3)
    value['recovery'].fill_(True)
    value['height_m'].fill_(.02)
    value['gravity_body'][:, 2] = 1
    value['forbidden_body_load_n'].fill_(30)
    value['height_m'][1] = .009
    value['within_pressure_stroke'][2, 0] = False
    for _ in range(4):
        result = latch.observe(**value)
    assert result.failure_reasons.tolist() == [0, 1, 32]
    assert not result.healthy.any()


def test_allowed_base_cannot_exempt_another_forbidden_body():
    specification = replace(limits(), forbidden_body_names=('base_link', 'leg'))
    latch = PhysicalFailureLatch(specification, 1, physics_dt_s=.005)
    value = sample(1, bodies=2)
    value['body_contact_allowed'] = torch.ones(1, dtype=torch.bool)
    value['forbidden_body_load_n'].fill_(30)
    for _ in range(4):
        result = latch.observe(**value)
    assert result.failure_reasons.item() == int(FailureReason.FORBIDDEN_BODY_CONTACT)


def test_historical_import_and_frozen_configuration_remain_compatible():
    assert HistoricalLimits is NativeTaskLimits
    with pytest.raises(ValueError):
        PhysicalFailureLatch(limits(), 1, physics_dt_s=0)

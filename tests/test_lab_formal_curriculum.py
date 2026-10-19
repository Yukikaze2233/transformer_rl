"""正式 6-stage 课程调度单测；不启动 Isaac。"""
from copy import deepcopy

import pytest
import torch

from transformer_rl.formal.lab_env.formal_curriculum import (
    FormalCurriculumCapabilities, FormalPPOPlan, FormalTaskCurriculum,
    UnsupportedCurriculumTask,
)
from transformer_rl.formal.tasks.curriculum import (
    STAGE_BUDGETS, STAGE_NAMES, STAGE_WEIGHTS, TaskFamily,
)


def all_capabilities():
    return FormalCurriculumCapabilities(terrain=True, requests=True, recovery=True)


def test_stage_weights_and_budgets_are_exactly_the_frozen_course():
    plan = FormalPPOPlan()
    assert plan.stage_budgets == STAGE_BUDGETS
    assert plan.stage_weights == STAGE_WEIGHTS
    assert sum(plan.stage_budgets) == 20000
    assert plan.stage(0)["weights"] == STAGE_WEIGHTS[0]
    assert plan.stage(5)["name"] == "consolidation"


def test_default_capabilities_refuse_recovery_stage_zero_instead_of_faking_it():
    curriculum = FormalTaskCurriculum(8)
    assert curriculum.missing_capabilities(0) == ("recovery",)
    with pytest.raises(UnsupportedCurriculumTask, match="recovery"):
        curriculum.sample(torch.arange(8))


def test_full_capabilities_sample_every_stage_family_exactly():
    curriculum = FormalTaskCurriculum(16, capabilities=all_capabilities(), seed=7)
    for stage in range(6):
        curriculum.budget.stage = stage
        draw = curriculum.sample(torch.arange(16))
        counts = torch.bincount(draw.family, minlength=6).float() / 16
        assert draw.family.shape == (16,)
        assert draw.requested_command.shape == (16, 4)
        assert abs(float(counts[TaskFamily.HEIGHT]) - STAGE_WEIGHTS[stage][0]) < 0.13


def test_budget_advances_only_on_update_boundary_and_is_not_refunded():
    curriculum = FormalTaskCurriculum(4, capabilities=all_capabilities())
    curriculum.budget.consumed = [STAGE_BUDGETS[0], 0, 0, 0, 0, 0]
    assert curriculum.stage == 0
    assert curriculum.begin_update() == 1
    assert curriculum.stage == 1
    assert curriculum.budget.consumed == [STAGE_BUDGETS[0], 1, 0, 0, 0, 0]
    assert curriculum.budget.total_updates == STAGE_BUDGETS[0] + 1


def test_state_roundtrip_restores_sampler_rng_quota_and_budget():
    curriculum = FormalTaskCurriculum(8, capabilities=all_capabilities(), seed=11)
    curriculum.sample(torch.arange(5))
    curriculum.budget.consumed[0] = 17
    state = curriculum.state_dict()
    restored = FormalTaskCurriculum(8, capabilities=all_capabilities(), seed=999)
    restored.load_state_dict(state)
    assert restored.state_dict()["budget"] == state["budget"]
    torch.testing.assert_close(restored.sample(torch.arange(6)).requested_command,
                               curriculum.sample(torch.arange(6)).requested_command)


def test_checkpoint_rejects_capability_or_domain_mismatch():
    curriculum = FormalTaskCurriculum(4, capabilities=all_capabilities(), seed=3)
    state = curriculum.state_dict()
    bad = deepcopy(state)
    bad["capabilities"]["terrain"] = False
    with pytest.raises(ValueError, match="身份"):
        curriculum.load_state_dict(bad)


def test_no_silent_capability_shortcut_for_terrain_or_requests():
    only_recovery = FormalCurriculumCapabilities(recovery=True)
    terrain = FormalTaskCurriculum(4, capabilities=only_recovery)
    assert terrain.missing_capabilities(3) == ("terrain",)
    both = FormalTaskCurriculum(4, capabilities=FormalCurriculumCapabilities(terrain=True, recovery=True))
    assert both.missing_capabilities(4) == ("request",)

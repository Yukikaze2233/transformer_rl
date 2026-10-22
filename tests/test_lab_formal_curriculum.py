"""正式 3-phase 课程调度单测；不启动 Isaac。"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.lab_env.formal_curriculum import (
    FormalCurriculumCapabilities, FormalPPOPlan, FormalTaskCurriculum,
    LabTaskDrawSampler, UnsupportedCurriculumTask,
)
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.curriculum import (
    COMPLETE_STAGE_BUDGETS, COMPLETE_STAGE_WEIGHTS, JUMP_BUDGET, JUMP_WEIGHTS,
    STAGE_BUDGETS, STAGE_NAMES, STAGE_WEIGHTS, TaskFamily,
)


def all_capabilities():
    return FormalCurriculumCapabilities(terrain=True, requests=True, recovery=True)


def test_requested_height_envelope_and_rough_terrain_cap_expand_before_hard():
    curriculum = FormalTaskCurriculum(128, include_jump=True, continuous=True,
                                     capabilities=all_capabilities())
    initial = curriculum._effective_domain()
    assert initial.height_m == (.28, .32)
    assert initial.forward_m_s == (-.5, .5)
    assert curriculum.terrain_level_cap(10) == 0
    curriculum.budget.stage = 1
    curriculum.budget.consumed = [5000, 0, 0, 0]
    assert curriculum.terrain_level_cap(10) == 6
    curriculum.budget.consumed[1] = 16000
    assert curriculum.terrain_level_cap(10) == 9
    curriculum.budget.consumed[1] = 20000
    assert curriculum._effective_domain().height_m == (.17, .40)
    assert curriculum.weights[int(TaskFamily.REQUEST)] == 0


def test_stage_weights_and_budgets_are_exactly_the_frozen_course():
    plan = FormalPPOPlan()
    assert plan.stage_budgets == COMPLETE_STAGE_BUDGETS
    assert plan.stage_weights == COMPLETE_STAGE_WEIGHTS
    assert sum(plan.stage_budgets) == 35000
    assert plan.stage(0)["weights"] == STAGE_WEIGHTS[0]
    assert plan.stage(2)["name"] == "hard"
    assert plan.jump()["name"] == "jump"
    assert plan.jump()["budget"] == JUMP_BUDGET


def test_default_capabilities_refuse_recovery_stage_zero_instead_of_faking_it():
    curriculum = FormalTaskCurriculum(8)
    assert curriculum.missing_capabilities(0) == ("recovery",)
    with pytest.raises(UnsupportedCurriculumTask, match="recovery"):
        curriculum.sample(torch.arange(8))


def test_full_capabilities_sample_every_stage_family_exactly():
    curriculum = FormalTaskCurriculum(16, capabilities=all_capabilities(), seed=7)
    for stage in range(3):
        curriculum.budget.stage = stage
        draw = curriculum.sample(torch.arange(16))
        counts = torch.bincount(draw.family, minlength=6).float() / 16
        assert draw.family.shape == (16,)
        assert draw.requested_command.shape == (16, 4)
        assert abs(float(counts[TaskFamily.HEIGHT]) - STAGE_WEIGHTS[stage][0]) < 0.13


def test_budget_advances_only_on_update_boundary_and_is_not_refunded():
    curriculum = FormalTaskCurriculum(4, capabilities=all_capabilities())
    curriculum.budget.consumed = [STAGE_BUDGETS[0], 0, 0]
    assert curriculum.stage == 0
    assert curriculum.begin_update() == 1
    assert curriculum.stage == 1
    assert curriculum.budget.consumed == [STAGE_BUDGETS[0], 1, 0]
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


def test_task_draw_sampler_maps_recovery_to_mode_and_phase():
    curriculum = FormalTaskCurriculum(64, capabilities=FormalCurriculumCapabilities(recovery=True), seed=5)
    sampler = LabTaskDrawSampler(curriculum)
    sample = sampler(torch.arange(64))
    recovery = sample.draw.family == 5
    assert recovery.any()
    assert (sample.mode[recovery] == int(CommandMode.RECOVERY)).all()
    assert (sample.mode[~recovery] == int(CommandMode.NORMAL)).all()
    assert (sample.command[recovery, 0] == 0).all() and (sample.command[recovery, 2] == 0).all()


def test_task_draw_sampler_maps_terrain_and_request_metadata():
    curriculum = FormalTaskCurriculum(32, capabilities=FormalCurriculumCapabilities(
        terrain=True, requests=True, recovery=True), seed=5)
    sampler = LabTaskDrawSampler(curriculum)
    curriculum.budget.stage = 1
    sample = sampler(torch.arange(32))
    terrain = sample.draw.family == int(TaskFamily.TERRAIN)
    assert terrain.any()
    assert set(sample.terrain_kind[terrain].tolist()) <= set(range(1, 8))
    assert set(sample.mode[terrain].tolist()) <= {
        int(CommandMode.NORMAL), int(CommandMode.STAIR), int(CommandMode.SLOPE)}
    assert (sample.request_height_m[~terrain] == 0).all()

    draw = replace(sample.draw, family=torch.full((32,), int(TaskFamily.REQUEST)),
                   jump_height_m=torch.full((32,), .1))
    jump = sampler.map_draw(draw)
    assert (jump.mode == int(CommandMode.JUMP)).all()
    torch.testing.assert_close(jump.request_height_m, torch.full((32,), .1))


def test_request_course_is_separate_and_does_not_change_main_phase_weights():
    assert JUMP_BUDGET == 5000
    assert JUMP_WEIGHTS[TaskFamily.REQUEST] == .5
    only_recovery = FormalCurriculumCapabilities(recovery=True)
    terrain = FormalTaskCurriculum(4, capabilities=only_recovery)
    assert terrain.missing_capabilities(1) == ("terrain",)


def test_continuous_course_ramps_mixture_domain_and_flat_rehearsal():
    curriculum = FormalTaskCurriculum(256, capabilities=all_capabilities(), seed=2102,
                                      include_jump=True, continuous=True)
    curriculum.budget.consumed = [COMPLETE_STAGE_BUDGETS[0], 0, 0, 0]
    curriculum.budget.stage = 1
    curriculum.budget.consumed[1] = 1
    early = curriculum.weights
    assert 0 < early[TaskFamily.TERRAIN] < STAGE_WEIGHTS[1][TaskFamily.TERRAIN]
    assert early[TaskFamily.REQUEST] == 0
    curriculum.budget.consumed[1] = COMPLETE_STAGE_BUDGETS[1]
    torch.testing.assert_close(torch.tensor(curriculum.weights), torch.tensor(STAGE_WEIGHTS[1]))
    domain = curriculum._effective_domain()
    assert domain.jump_m[0] >= .03 and domain.jump_m[1] <= .40
    draw = curriculum.sample(torch.arange(256))
    terrain = draw.family == TaskFamily.TERRAIN
    assert terrain.any()
    assert 0 in set(draw.terrain_kind[terrain].tolist())

    curriculum.budget.consumed = [5000, 20000, 5000, 1]
    curriculum.budget.stage = 3
    start = curriculum.weights[TaskFamily.REQUEST]
    curriculum.budget.consumed[3] = COMPLETE_STAGE_BUDGETS[3]
    end = curriculum.weights[TaskFamily.REQUEST]
    assert start < end == JUMP_WEIGHTS[TaskFamily.REQUEST]
    assert curriculum._effective_domain().jump_m == (0.03, 0.40)


def test_jump_envelope_follows_jump_stage_exposure_not_total_progress():
    curriculum = FormalTaskCurriculum(32, capabilities=all_capabilities(), seed=2102,
                                      include_jump=True, continuous=True)
    curriculum.budget.consumed = [5000, 20000, 5000, 0]
    curriculum.budget.stage = 3
    early = curriculum._effective_domain().jump_m
    assert early[0] == .03 and early[1] < .11
    curriculum.budget.consumed[3] = 2500
    middle = curriculum._effective_domain().jump_m
    assert early[1] < middle[1] < .40
    curriculum.budget.consumed[3] = 5000
    final = curriculum._effective_domain().jump_m
    assert final == (.03, .40)


def test_continuous_recovery_starts_easy_and_keeps_full_catalogue_before_rough():
    curriculum = FormalTaskCurriculum(4096, capabilities=all_capabilities(), seed=12,
                                      include_jump=True, continuous=True)
    early = curriculum.sample(torch.arange(4096))
    rows = early.family == int(TaskFamily.RECOVERY)
    assert set(early.recovery_posture[rows].tolist()) == {0, 1}
    assert int(rows.sum()) in (409, 410)
    curriculum.budget.consumed[0] = 4000
    later = curriculum.sample(torch.arange(4096))
    assert set(later.recovery_posture[later.family == int(TaskFamily.RECOVERY)].tolist()) == set(range(16))
    assert curriculum.stage_weights == COMPLETE_STAGE_WEIGHTS


def test_full_domain_has_rehearsal_before_budget_exhaustion():
    curriculum = FormalTaskCurriculum(512, capabilities=all_capabilities(), seed=12,
                                      include_jump=True, continuous=True)
    curriculum.budget.stage = 2
    curriculum.budget.consumed = [5000, 20000, 0, 0]
    domain = curriculum._effective_domain()
    assert domain.forward_m_s == curriculum.domain.forward_m_s
    assert domain.yaw_rad_s == curriculum.domain.yaw_rad_s
    assert domain.step_up_m == curriculum.domain.step_up_m
    curriculum.budget.stage = 3
    early = curriculum.sample(torch.arange(512))
    request = early.family == int(TaskFamily.REQUEST)
    # At zero exposure the mixture still has no jump rows; after transition
    # starts, any request rows learn a stationary nominal-height jump first.
    curriculum.budget.consumed[3] = 1000
    early = curriculum.sample(torch.arange(512))
    request = early.family == int(TaskFamily.REQUEST)
    assert request.any()
    assert early.requested_command[request, :3].eq(0).all()
    assert early.requested_command[request, 3].eq(curriculum.domain.nominal_height_m).all()
    curriculum.budget.consumed[3] = 4000
    late = curriculum.sample(torch.arange(512))
    request = late.family == int(TaskFamily.REQUEST)
    assert (late.requested_command[request, 0].abs() > 1.).any()
    assert curriculum._effective_domain().jump_m == (.03, .40)
    assert curriculum.weights == JUMP_WEIGHTS
    assert not (late.family == int(TaskFamily.RECOVERY)).any()


def test_conditioned_difficulty_rng_roundtrip_preserves_draw_and_quota():
    a = FormalTaskCurriculum(128, capabilities=all_capabilities(), seed=12,
                             include_jump=True, continuous=True)
    a.budget.consumed[0] = 1234
    a.sample(torch.arange(23))
    b = FormalTaskCurriculum(128, capabilities=all_capabilities(), seed=99,
                             include_jump=True, continuous=True)
    b.load_state_dict(a.state_dict())
    first, second = a.sample(torch.arange(128)), b.sample(torch.arange(128))
    torch.testing.assert_close(first.requested_command, second.requested_command)
    torch.testing.assert_close(first.recovery_posture, second.recovery_posture)
    torch.testing.assert_close(first.family, second.family)

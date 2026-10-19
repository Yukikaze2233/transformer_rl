"""Physical receipt-to-reward ownership checks with synthetic measurements."""
from dataclasses import replace

import pytest
import torch

from test_control_reward import begin, reference, sample
from test_physical_observation import fixture as mechanics_fixture
from test_skill_evidence import SkillHarness
from transformer_rl._control_reward import PhysicalRewardComposer, RewardPhase, RewardTuning
from transformer_rl._physical_observation import CommandMode
from transformer_rl._skill_evidence import SkillOutcome


@pytest.fixture(scope='module')
def successful_evidence():
    h = SkillHarness(stable=2.)
    h.accept()
    h.airborne()
    return h.settle()


def reward_window(evidence, *, retired=None):
    reward = PhysicalRewardComposer(mechanics_fixture()[1], 2, RewardTuning(20.), retired_request_ids=retired)
    start = evidence.timestamp_s - .01
    reward.reset(torch.ones(2, dtype=torch.bool), start, evidence.episode_id, evidence.control_epoch)
    ref = reference(CommandMode.JUMP, RewardPhase.SETTLING)
    ref.admitted_body_command[:, 3] = .3
    ref.jump_height_world_z_m.fill_(.3)
    begin(reward, ref)
    for index in range(1, reward.substeps + 1):
        value = replace(sample(reward, index), root_world_z_m=torch.full((2,), .3), height_above_ground_m=torch.full((2,), .3),
                         timestamp_s=evidence.timestamp_s - (reward.substeps - index) * reward.physics_dt_s)
        reward.observe(value, ref)
    return reward


def finish(reward, evidence):
    return reward.finish_from_evidence(evidence, terminated=torch.zeros(2, dtype=torch.bool),
                                       truncated=torch.zeros(2, dtype=torch.bool))


def test_complete_physical_receipt_pays_once_with_matching_pre_reset_endpoint(successful_evidence):
    reward = reward_window(successful_evidence)
    result = finish(reward, successful_evidence)
    assert result.success_credited.all()
    assert result.event_rewards['physical_skill_success'].eq(1).all()
    assert torch.equal(result.timestamp_s, successful_evidence.timestamp_s)
    with pytest.raises(RuntimeError, match='complete reward interval'):
        finish(reward, successful_evidence)
    cold = reward_window(successful_evidence, retired=torch.tensor([1, 1]))
    assert finish(cold, successful_evidence).event_rewards['physical_skill_success'].eq(0).all()


@pytest.mark.parametrize('changed', ['timestamp_s', 'episode_id', 'control_epoch', 'physical_tick',
                                    'stable_landing_s', 'supported_start', 'entire_skill_healthy',
                                    'entire_skill_policy_owned', 'flight_seen', 'apex_rise_m',
                                    'height_tolerance_m', 'outcome'])
def test_invalid_or_mismatched_physical_success_is_rejected_before_paying_and_keeps_interval(successful_evidence, changed):
    evidence = successful_evidence
    value = getattr(evidence, changed)
    if changed == 'physical_tick':
        wrong_value = value + 1
    elif changed in ('supported_start', 'entire_skill_healthy', 'entire_skill_policy_owned', 'flight_seen'):
        wrong_value = torch.zeros_like(value)
    elif changed == 'stable_landing_s':
        wrong_value = torch.full_like(value, 1.99)
    elif changed == 'height_tolerance_m':
        wrong_value = torch.full_like(value, .1)
    elif changed == 'outcome':
        wrong_value = torch.full_like(value, int(SkillOutcome.NONE))
    else:
        wrong_value = value + 1
    wrong = replace(evidence, **{changed: wrong_value})
    reward = reward_window(evidence)
    with pytest.raises(ValueError, match='physical evidence'):
        finish(reward, wrong)
    assert reward._window is not None and reward._success_seen.eq(-1).all()
    assert finish(reward, evidence).success_credited.all()


def test_protocol_completion_type_and_script_assistance_do_not_enter_success_reward(successful_evidence):
    reward = reward_window(successful_evidence)
    h = SkillHarness()
    h.accept()
    with pytest.raises(TypeError, match='independent physical evidence'):
        finish(reward, h.last_condition)
    assisted = replace(successful_evidence, physical_success_request_id=torch.full((2,), -1, dtype=torch.long),
                        outcome=torch.full((2,), int(SkillOutcome.SCRIPT_ASSISTED), dtype=torch.long),
                        entire_skill_policy_owned=torch.zeros(2, dtype=torch.bool))
    assert finish(reward, assisted).event_rewards['physical_skill_success'].eq(0).all()

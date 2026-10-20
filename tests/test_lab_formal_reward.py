"""LabEnv 正式 reward 适配器单测；不启动 Isaac。"""
from dataclasses import replace

import torch

from test_physical_observation import fixture
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.formal_reward import (
    FormalRewardConfig, LAB_AVAILABLE_TERMS, LabFormalReward, reward_physics_from_lab,
)
from transformer_rl.formal.tasks.reward import RewardPhase


def enabled_fixture():
    _, binding, assembler, sample, command = fixture()
    forces = torch.zeros_like(sample.terrain_contact_force_world_n)
    forces[:, binding.wheel_body_indices[0], 2] = 10
    forces[:, binding.wheel_body_indices[1], 2] = 20
    sample = replace(sample, terrain_contact_force_world_n=forces)
    state = assembler.measure(sample)
    return binding, state, sample, command


def test_lab_formal_reward_keeps_formal_terms_and_omits_unavailable_ones():
    binding, state, sample, command = enabled_fixture()
    reward = LabFormalReward(binding, 2, FormalRewardConfig(13.886))
    episode = torch.zeros(2, dtype=torch.long)
    control = torch.zeros(2, dtype=torch.long)
    reward.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.float64),
                 episode, control)
    issued = torch.zeros(2, 6)
    reward.begin(issued, torch.zeros(2, 6), now_s=torch.zeros(2, dtype=torch.float64),
                 episode_id=episode, control_epoch=control,
                 command=command.admitted_body_command, mode=command.mode)
    for step in (1, 2):
        value = replace(sample, timestamp_s=torch.full((2,), step * .005, dtype=torch.float64))
        current = replace(state, timestamp_s=value.timestamp_s.clone())
        reward.observe(value, current, effort_nm=torch.zeros(2, 6), episode_id=episode,
                       control_epoch=control, command=command.admitted_body_command,
                       mode=command.mode, phase=torch.full((2,), int(RewardPhase.SUPPORT),
                                                          dtype=torch.long),
                       wheel_contact_tangent_speed_m_s=torch.zeros(2, 2))
    result = reward.finish(terminated=torch.zeros(2, dtype=torch.bool),
                           truncated=torch.zeros(2, dtype=torch.bool))
    assert torch.isfinite(result.learning_reward).all()
    assert result.raw_cost_densities["contact_slip"].eq(0).all()
    assert result.raw_cost_densities["joint_boundary"].eq(0).all()
    assert "contact_slip" in reward.missing_terms
    assert "contact_slip" not in LAB_AVAILABLE_TERMS
    assert result.gate_fractions["ordinary"].all()
    assert "bounded_cost.contact_slip" in result.component_densities


def test_unavailable_terms_are_removed_from_composition_not_just_logged():
    binding, state, sample, command = enabled_fixture()
    terms = tuple(name for name in LAB_AVAILABLE_TERMS if name != "harmful_contact")
    physics = reward_physics_from_lab(
        sample, state, effort_nm=torch.zeros(2, 6),
        episode_id=torch.zeros(2, dtype=torch.long),
        control_epoch=torch.zeros(2, dtype=torch.long),
        policy_owned=torch.ones(2, dtype=torch.bool), available_terms=terms,
        wheel_contact_tangent_speed_m_s=torch.zeros(2, 2))
    assert physics.available_terms == terms
    assert physics.body_normal_load_n.shape == (2, 19)
    assert physics.wheel_contact_tangent_speed_m_s.eq(0).all()


def test_reward_reference_is_normal_mode_for_flat_commands():
    _, _, _, command = enabled_fixture()
    assert command.mode.eq(int(CommandMode.NORMAL)).all()
    assert command.admitted_body_command.shape == (2, 4)

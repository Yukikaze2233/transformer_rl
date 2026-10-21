"""Visible-clock task references only; these checks do not simulate a robot."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.tasks.reward import RewardPhase
from transformer_rl.formal.tasks.jump import JumpReferenceConfig, JumpReferenceProgram, ReferenceStage, reward_phases, reward_reference
from transformer_rl.formal.control.observation import CommandFeatures, CommandMode
from transformer_rl.formal.tasks.skills import SkillPhase


def sample(height, rise, clock, *, ground=None, active=None):
    height, rise = torch.as_tensor(height, dtype=torch.float32), torch.as_tensor(rise, dtype=torch.float32)
    clock = torch.broadcast_to(torch.as_tensor(clock, dtype=torch.float64), height.shape).clone()
    ground = torch.zeros_like(height) if ground is None else torch.as_tensor(ground, dtype=torch.float32)
    active = torch.ones_like(height, dtype=torch.bool) if active is None else active
    return JumpReferenceProgram().sample(height, rise, clock, ground, active=active)


def commands(modes):
    n = len(modes)
    return CommandFeatures(torch.tensor([[0., 0., 0., .3]]).repeat(n, 1), torch.tensor(modes, dtype=torch.long),
                            torch.full((n,), .1), torch.zeros(n), torch.zeros(n, 6))


def test_continuous_height_and_rise_grid_keeps_total_apex_without_shrinking_request_range():
    height, rise = torch.meshgrid(torch.linspace(.17, .40, 13), torch.linspace(.03, .30, 17), indexing='ij')
    height, rise = height.flatten(), rise.flatten()
    value = sample(height, rise, 0)
    assert value.launch_velocity_m_s.dtype == torch.float64  # Same precision as its analytic event clocks.
    apex_time = value.expected_launch_time_s + value.launch_velocity_m_s.double() / 9.81
    apex = sample(height, rise, apex_time)
    torch.testing.assert_close(apex.height_world_z_m, height + rise, atol=1e-7, rtol=0)
    torch.testing.assert_close(apex.velocity_world_z_m_s, torch.zeros_like(height), atol=1e-7, rtol=0)
    assert value.extension_time_limited_by_support_range.any()
    assert (~value.extension_time_limited_by_support_range).any()
    assert value.extension_duration_s.gt(0).all()
    at_launch = sample(height, rise, value.expected_launch_time_s)
    assert at_launch.height_world_z_m.le(.40 + 1e-7).all()
    assert at_launch.height_world_z_m.ge(.17 - 1e-7).all()


@pytest.mark.parametrize('height,rise', [(.17, .03), (.30, .15), (.40, .30)])
def test_compression_extension_and_launch_are_c1_but_landing_hold_is_explicit(height, rise):
    h, j = [height], [rise]
    value = sample(h, j, 0)
    assert value.height_world_z_m.item() == torch.tensor(height).item()
    assert value.velocity_world_z_m_s.item() == 0
    for boundary in (.20, value.expected_launch_time_s.item()):
        before, after = sample(h, j, boundary - 1e-8), sample(h, j, boundary + 1e-8)
        torch.testing.assert_close(before.height_world_z_m, after.height_world_z_m, atol=1e-7, rtol=0)
        torch.testing.assert_close(before.velocity_world_z_m_s, after.velocity_world_z_m_s, atol=1e-6, rtol=0)
    low = sample(h, j, .20)
    assert low.height_world_z_m.ge(.17 - 1e-7).all()
    returned = sample(h, j, value.expected_return_time_s + .01)
    assert returned.clock_stage.item() == int(ReferenceStage.HOLD)
    torch.testing.assert_close(returned.height_world_z_m, torch.tensor(h))
    assert returned.velocity_world_z_m_s.item() == 0


def test_world_translation_row_permutation_and_inactive_rows_do_not_change_clock_or_rng():
    rng = torch.get_rng_state().clone()
    h, j, clock = [.17, .30, .40], [.03, .15, .30], [.1, .25, .45]
    plain = sample(h, j, clock)
    shifted = sample(h, j, clock, ground=[2., -1., .5])
    torch.testing.assert_close(shifted.height_world_z_m - plain.height_world_z_m, torch.tensor([2., -1., .5]))
    assert torch.equal(shifted.velocity_world_z_m_s, plain.velocity_world_z_m_s)
    order = torch.tensor([2, 0, 1])
    permuted = sample(torch.tensor(h)[order], torch.tensor(j)[order], torch.tensor(clock)[order].double())
    torch.testing.assert_close(permuted.height_world_z_m, plain.height_world_z_m[order], atol=1e-7, rtol=0)
    inactive = sample([.3, .3], [0., .1], [.3, .3], active=torch.tensor([False, True]))
    assert inactive.clock_stage[0] == int(ReferenceStage.HOLD)
    assert inactive.height_world_z_m[0] == torch.tensor(.3)
    assert inactive.velocity_world_z_m_s[0] == 0
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('height,rise,clock', [(.16, .1, 0), (.41, .1, 0), (.3, .02, 0),
                                            (.3, .41, 0), (.3, .1, -.01), (.3, float('nan'), 0)])
def test_invalid_visible_target_or_clock_is_rejected(height, rise, clock):
    with pytest.raises(ValueError, match='invalid visible'):
        sample([height], [rise], clock)


def test_physical_phase_is_reward_only_and_never_rewrites_actor_command_or_task_curve():
    cmd = commands([CommandMode.JUMP, CommandMode.JUMP, CommandMode.RECOVERY, CommandMode.NORMAL])
    before = {name: value.clone() if isinstance(value, torch.Tensor) else value
              for name, value in vars(cmd).items()}
    curve = sample([.3] * 4, [.1] * 4, .35, ground=[2.] * 4)
    phase = reward_phases(cmd, torch.tensor([int(SkillPhase.FLIGHT), int(SkillPhase.SETTLING),
                                            int(SkillPhase.FLIGHT), int(SkillPhase.FLIGHT)]))
    assert phase.tolist() == [int(RewardPhase.FLIGHT), int(RewardPhase.SETTLING),
                              int(RewardPhase.RECOVERY), int(RewardPhase.SUPPORT)]
    ref = reward_reference(cmd, curve, phase, torch.full((4,), 3.))
    assert ref.jump_height_world_z_m[0] == curve.height_world_z_m[0]
    assert ref.jump_height_world_z_m[1] == torch.tensor(3.3)
    assert ref.jump_velocity_world_z_m_s[1] == 0
    for name, value in vars(cmd).items():
        old = before[name]
        assert torch.equal(value, old) if isinstance(value, torch.Tensor) else value == old
    ref.admitted_body_command.zero_()
    assert torch.equal(cmd.admitted_body_command, before['admitted_body_command'])


def test_reference_configuration_and_reward_packet_shape_fail_before_use():
    with pytest.raises(ValueError):
        JumpReferenceConfig(compression_duration_s=0)
    cmd, curve = commands([CommandMode.JUMP]), sample([.3], [.1], .1)
    with pytest.raises(ValueError, match='phase'):
        reward_phases(cmd, torch.tensor([9]))
    with pytest.raises(ValueError, match='reference differs'):
        reward_reference(cmd, replace(curve, height_world_z_m=torch.zeros(2)), torch.tensor([1]), torch.zeros(1))

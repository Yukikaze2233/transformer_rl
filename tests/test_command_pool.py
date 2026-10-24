"""Command pool and within-episode segments; synthetic CPU tensors only."""
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks
from transformer_rl.formal.tasks.command_pool import (
    CommandContext, CommandPool, CommandPoolConfig, SegmentKind,
)
from transformer_rl.formal.tasks.curriculum import TaskDomain, TaskFamily
from transformer_rl.formal.tasks.geometry import TerrainKind
from transformer_rl.formal.tasks.trajectory import DriveEnvelope

DOMAIN = TaskDomain(height_m=(.23, .40), forward_m_s=(-2.7, 2.7), yaw_rad_s=(-15., 15.))


def draw(family, terrain=TerrainKind.FLAT, *, elapsed=6., update=10000, count=4000, sign=1., scale=1., seed=0):
    pool = CommandPool()
    generator = torch.Generator().manual_seed(seed)
    return pool.sample(torch.full((count,), int(family)), torch.full((count,), int(terrain)),
                       torch.full((count,), elapsed), torch.full((count,), sign),
                       CommandContext(DOMAIN, update, scale), generator)


def test_special_modes_unlock_by_update_and_only_after_the_episode_settles():
    special = torch.tensor([int(SegmentKind.SPIN_LOW), int(SegmentKind.SPIN_MID), int(SegmentKind.DASH)])
    assert not torch.isin(draw(TaskFamily.MOTION, update=1000).kind, special).any()
    assert not torch.isin(draw(TaskFamily.MOTION, elapsed=2.).kind, special).any()
    dash_only = draw(TaskFamily.MOTION, update=2500)
    assert (dash_only.kind == SegmentKind.DASH).any()
    assert not torch.isin(dash_only.kind, special[:2]).any()
    late = draw(TaskFamily.MOTION)
    shares = torch.bincount(late.kind, minlength=5).float() / len(late.kind)
    torch.testing.assert_close(shares, torch.tensor([.136, .504, .09, .09, .18]), atol=.025, rtol=0)
    mid = late.kind == SegmentKind.SPIN_MID
    yaw = late.command[mid, 2].abs()
    assert yaw.ge(3.25 * math.pi - 1e-5).all() and yaw.le(4.5 * math.pi + 1e-5).all()
    assert late.command[mid, 0].abs().le(.1).all()  # inside the reward deadband
    fast = late.kind == SegmentKind.DASH
    assert late.command[fast, 0].abs().ge(2. - 1e-6).all() and late.command[fast, 0].abs().le(2.7 + 1e-6).all()
    assert late.command[fast, 2].abs().le(2 * math.pi + 1e-6).all()


def test_normal_segments_cover_both_directions_heights_and_heading_hold():
    result = draw(TaskFamily.MOTION)
    normal = result.kind == SegmentKind.NORMAL
    forward = result.command[normal, 0]
    assert forward.gt(1.).float().mean() > .2 and forward.lt(-1.).float().mean() > .2
    assert result.command[normal, 2].abs().le(2 * math.pi + 1e-6).all()
    assert result.command[:, 1].eq(0).all()
    assert result.command[:, 3].ge(.23).all() and result.command[:, 3].le(.40).all()
    assert .4 < result.heading_enabled[normal].float().mean() < .6
    assert not result.heading_enabled[~normal].any()
    stand = result.kind == SegmentKind.STAND
    assert result.command[stand, :3].eq(0).all()


def test_approach_courses_keep_direction_scut_stair_band_and_course_heading():
    result = draw(TaskFamily.TERRAIN, TerrainKind.STAIRS, sign=-1.)
    assert torch.isin(result.kind, torch.tensor([0, 1])).all()
    moving = result.kind != SegmentKind.STAND
    assert result.command[moving, 0].le(-.5 + 1e-6).all() and result.command[moving, 0].ge(-2. - 1e-6).all()
    assert result.command[:, 2].abs().le(1. + 1e-6).all()
    assert result.heading_enabled[moving].all() and result.heading_offset_rad.eq(0).all()
    assert result.command[:, 3].ge(.25 - 1e-6).all()
    slope = draw(TaskFamily.TERRAIN, TerrainKind.SLOPE)
    assert slope.command[:, 0].abs().max() > 2.  # SCUT binds only the height on slopes
    assert slope.command[:, 3].ge(.25 - 1e-6).all()


def test_jump_episodes_scale_planar_motion_but_not_height():
    full, still = draw(TaskFamily.REQUEST, seed=3), draw(TaskFamily.REQUEST, scale=0., seed=3)
    assert full.command[:, 0].abs().max() > .5
    assert still.command[:, [0, 2]].eq(0).all()
    torch.testing.assert_close(still.command[:, 3], full.command[:, 3])


def test_pool_configuration_is_frozen_in_code():
    assert CommandPoolConfig.from_dict({'enabled': True}) == CommandPoolConfig()
    with pytest.raises(ValueError):
        CommandPoolConfig.from_dict({'enabled': True, 'segment_s': [1, 2]})
    with pytest.raises(ValueError):
        CommandPoolConfig(segment_weights=(1., 0., 0., 0.))


def course(count=6, family=TaskFamily.MOTION, terrain=TerrainKind.FLAT, command=None):
    tasks = LabCourseTasks(count, DriveEnvelope(.06, .437), policy_dt_s=.02,
                           command_pool=CommandPool(), command_context=lambda: CommandContext(DOMAIN, 10000))
    draw_command = torch.tensor([[.5, 0., 0., .32]]).repeat(count, 1) if command is None else command
    task = SimpleNamespace(
        draw=SimpleNamespace(family=torch.full((count,), int(family)), nominal=torch.zeros(count, dtype=torch.bool),
                             trajectory_kind=torch.zeros(count, dtype=torch.long),
                             rotation_mode=torch.zeros(count, dtype=torch.long)),
        command=draw_command, terrain_kind=torch.full((count,), int(terrain)),
        mode=torch.full((count,), int(CommandMode.NORMAL)), request_height_m=torch.zeros(count))
    tasks.reset(torch.arange(count), task, torch.zeros(count, dtype=torch.float64))
    return tasks


def run(tasks, seconds, yaw=None):
    now = torch.zeros(tasks.num_envs, dtype=torch.float64)
    admitted = []
    for step in range(round(seconds / tasks.policy_dt_s)):
        now = now + tasks.policy_dt_s
        admission, *_ = tasks.step(now, None, torch.zeros(tasks.num_envs),
                                   yaw_rad=torch.zeros(tasks.num_envs) if yaw is None else yaw)
        admitted.append(admission.admitted.clone())
    return torch.stack(admitted)


def test_segments_resample_within_an_episode_through_the_deployment_slew_limits():
    tasks = course()
    commands = run(tasks, 30.)
    assert tasks.segment_counts.sum().item() >= tasks.num_envs  # U(5, 15) s segments in 30 s
    # RMCS command_rate_limits [2.0, 3.0, 0.2] apply to every admitted change.
    rates = (commands[1:] - commands[:-1]).abs().amax((0, 1)) / tasks.policy_dt_s
    assert rates[0] <= 2. + 1e-3 and rates[2] <= 3. + 1e-3 and rates[3] <= .2 + 1e-3


def test_recovery_handoff_uses_the_scut_ground_reset_command_before_segments():
    tasks = course(count=4)
    born = tasks.segment_counts.sum().item()  # each birth draws its first segment
    tasks.base_mode.fill_(int(CommandMode.RECOVERY))
    run(tasks, 16.)
    assert tasks.segment_counts.sum() == born  # no segment while recovering
    tasks.base_mode.fill_(int(CommandMode.NORMAL))
    tasks.begin_tracking(torch.ones(4, dtype=torch.bool))
    target = tasks.trajectory.end.clone()
    assert target[:, [0, 2]].abs().le(1.).all() and target[:, 1].eq(0).all()
    run(tasks, 1.4)
    assert tasks.segment_counts.sum() == born
    run(tasks, .2)
    assert tasks.segment_counts.sum() == born + 4


def test_approach_course_heading_target_survives_segment_changes():
    tasks = course(count=3, family=TaskFamily.TERRAIN, terrain=TerrainKind.STEP_UP)
    run(tasks, .1, yaw=torch.full((3,), .3))
    target = tasks.heading_target_rad.clone()
    assert tasks.heading_enabled.all()
    torch.testing.assert_close(target, torch.full((3,), .3))
    run(tasks, 25., yaw=torch.full((3,), .3))
    torch.testing.assert_close(tasks.heading_target_rad, target)


def test_births_take_the_first_segment_from_the_pool_with_special_modes_locked():
    count = 3000
    # The legacy birth sampler may request a 15 rad/s spin or a 2.6 m/s dash.
    tasks = course(count, family=TaskFamily.ROTATION,
                   command=torch.tensor([[2.6, 0., 15., .32]]).repeat(count, 1))
    assert torch.isin(tasks.segment_kind, torch.tensor([int(SegmentKind.STAND), int(SegmentKind.NORMAL)])).all()
    assert tasks.trajectory.end[:, 2].abs().le(2 * math.pi + 1e-6).all()
    assert tasks.trajectory.start.eq(tasks.trajectory.end).all()  # one step target per segment
    assert tasks.stream.current[:, :3].eq(0).all()  # born at rest, slewing from standing
    commands = run(tasks, 4.9)
    assert commands[..., 2].abs().max() <= 2 * math.pi + 1e-4

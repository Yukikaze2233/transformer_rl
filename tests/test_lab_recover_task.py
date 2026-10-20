"""recover task 的纯逻辑单测；不启动 Isaac。"""
import math

import torch

from transformer_rl.formal.lab_env.recover import LabRecoverTask, RecoverTaskSpec


def fixture(n=4):
    spec = RecoverTaskSpec(action_hold_s=.05, policy_dt_s=.01)
    task = LabRecoverTask(n, spec, seed=7)
    origins = torch.tensor([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.], [4., 4., 0.]])[:n]
    default = torch.zeros(n, 18)
    plan = task.reset(torch.arange(n), origins, default)
    return task, plan


def test_recover_reset_has_random_fallen_state_and_exact_hold():
    task, plan = fixture()
    assert plan.root_pose_xyzw.shape == (4, 7)
    assert plan.root_pose_xyzw[:, 3:].norm(dim=-1).allclose(torch.ones(4))
    assert plan.root_velocity.shape == (4, 6)
    assert plan.joint_position.shape == (4, 18)
    assert task.hold_mask.all()
    for _ in range(task.hold_steps):
        task.advance()
    assert not task.hold_mask.any()


def test_recover_success_requires_both_wheels_low_base_height_and_upright():
    task, _ = fixture()
    for _ in range(task.hold_steps):
        task.advance()
    wheel = torch.full((4, 2), 10.)
    base = torch.zeros(4)
    height = torch.full((4,), .25)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    assert task.success_mask(wheel_force_n=wheel, base_load_n=base, height_m=height,
                             gravity_body=gravity).all()
    bad = wheel.clone(); bad[:, 1] = 0
    assert not task.success_mask(wheel_force_n=bad, base_load_n=base, height_m=height,
                                 gravity_body=gravity).any()
    low = height.clone(); low[:] = .1
    assert not task.success_mask(wheel_force_n=wheel, base_load_n=base, height_m=low,
                                 gravity_body=gravity).any()
    tilted = gravity.clone(); tilted[:, 0] = math.sin(math.radians(20.))
    tilted[:, 2] = -math.cos(math.radians(20.))
    assert not task.success_mask(wheel_force_n=wheel, base_load_n=base, height_m=height,
                                 gravity_body=tilted).any()


def test_recover_hold_is_excluded_from_success_and_can_be_zeroed():
    task, _ = fixture()
    wheel = torch.full((4, 2), 10.)
    base = torch.zeros(4)
    height = torch.full((4,), .25)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    assert not task.success_mask(wheel_force_n=wheel, base_load_n=base, height_m=height,
                                 gravity_body=gravity).any()
    assert task.reward_bonus().eq(0).all()

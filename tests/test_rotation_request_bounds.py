"""Rotation buckets must cover the confirmed box without bypassing its curriculum."""
from dataclasses import replace
import math

import pytest
import torch

from transformer_rl.formal.tasks.curriculum import (
    ContinuousTaskSampler, RotationMode, TaskDomain, TaskFamily,
)


def rotation_draw(domain):
    draw = ContinuousTaskSampler(2102, domain=domain).sample(25000, 2)
    rows = draw.family == int(TaskFamily.ROTATION)
    return draw.requested_command[rows], draw.rotation_mode[rows]


def test_final_rotation_bucket_reaches_beyond_four_point_five_pi():
    command, mode = rotation_draw(TaskDomain())
    spin = command[mode == int(RotationMode.SPIN_MID), 2]
    assert (spin > 5.9 * math.pi).any()
    assert (spin < -5.9 * math.pi).any()
    assert spin.abs().max() <= 6 * math.pi + 1e-5
    dash = command[mode == int(RotationMode.DASH), 0]
    assert (dash >= 2.).any() and (dash <= -2.).any()
    assert dash.abs().max() <= 2.7 + 1e-6


@pytest.mark.parametrize('yaw,forward', ((6.4, 2.05), (10.4, 2.3), (18., 2.65)))
def test_partially_introduced_buckets_stay_inside_current_domain(yaw, forward):
    domain = replace(TaskDomain(), yaw_rad_s=(-yaw, yaw),
                     forward_m_s=(-forward, forward))
    command, _ = rotation_draw(domain)
    assert command[:, 2].abs().max() <= yaw + 1e-5
    assert command[:, 0].abs().max() <= forward + 1e-6

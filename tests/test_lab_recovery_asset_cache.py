"""Optional source-asset checks; run on Kaiser without starting a simulator."""
from dataclasses import fields
from copy import deepcopy
import os

import pytest
import torch

from transformer_rl.formal.lab_env.recovery import LabRecoveryReset
from transformer_rl.formal.lab_env.settings import load_asset_description


@pytest.fixture(scope='module')
def selected_reset():
    directory = os.environ.get('SELECTED_RESET_ASSET')
    identity = os.environ.get('SELECTED_RESET_ASSET_SHA256')
    if directory is None and identity is None:
        pytest.skip('requires an explicit selected source-asset directory and manifest SHA256')
    if not directory or not identity:
        pytest.fail('source-asset cache validation requires both the directory and manifest SHA256')
    asset = load_asset_description(directory, expected_manifest_sha256=identity)
    reset = LabRecoveryReset(asset, 32)
    assert reset._zero_yaw_pose is not None
    assert reset._zero_yaw_pose.posture_id.tolist() == list(range(16))
    return reset


def _assert_same_pose(actual, expected):
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name),
                                   atol=0, rtol=0)


def test_all_selected_asset_postures_match_uncached_solves_and_leave_rng_unchanged(selected_reset):
    reset = selected_reset
    table = reset._zero_yaw_pose
    ids = torch.arange(32)
    posture = (ids + 7) % 16
    yaw = torch.zeros(32, dtype=torch.float64)
    origins = torch.zeros(32, 3, dtype=torch.float64)
    origins[:, 0] = ids.double() * .125 - 2.
    origins[:, 1] = ids.double() * -.0625
    before = torch.get_rng_state().clone()
    actual = reset.pose(ids, posture, yaw, origins)
    try:
        reset._zero_yaw_pose = None
        expected = reset.pose(ids, posture, yaw, origins)
    finally:
        reset._zero_yaw_pose = table
    _assert_same_pose(actual, expected)
    assert torch.equal(torch.get_rng_state(), before)


def test_partial_selected_asset_lookup_never_aliases_or_changes_the_pose_table(selected_reset):
    reset = selected_reset
    table = deepcopy(reset._zero_yaw_pose)
    ids = torch.tensor([31, 0, 19, 7])
    posture = torch.tensor([15, 2, 3, 9])
    yaw = torch.zeros(4, dtype=torch.float64)
    origins = torch.zeros(4, 3, dtype=torch.float64)
    actual = reset.pose(ids, posture, yaw, origins)
    expected = deepcopy(actual)
    for field in fields(actual):
        getattr(actual, field.name).zero_()
    _assert_same_pose(reset._zero_yaw_pose, table)
    _assert_same_pose(reset.pose(ids, posture, yaw, origins), expected)


@pytest.mark.parametrize('yaw,z', ((.317, 0.), (0., 2.)))
def test_selected_asset_noncanonical_pose_uses_fresh_geometry(selected_reset, yaw, z):
    reset = selected_reset
    table = reset._zero_yaw_pose
    args = (torch.tensor([5, 29]), torch.tensor([2, 15]),
            torch.full((2,), yaw, dtype=torch.float64),
            torch.tensor([[.5, -.25, z], [-.5, .25, z]], dtype=torch.float64))
    before = torch.get_rng_state().clone()
    expected = reset.pose(*args)
    try:
        # An unusable table proves these calls use the full geometric path.
        reset._zero_yaw_pose = object()
        actual = reset.pose(*args)
    finally:
        reset._zero_yaw_pose = table
    _assert_same_pose(actual, expected)
    assert torch.equal(torch.get_rng_state(), before)

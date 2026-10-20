"""Recovery compiler outputs survive the Lab collision-placement boundary."""
import json
import math

import pytest
import torch

from test_joint_boundaries import encode
from test_reset_collision import fixture as collision_fixture
from transformer_rl.formal.control.observation import MechanicalBinding
from transformer_rl.formal.lab_env.recovery import LabRecoveryReset, RecoveryResetConfig
from transformer_rl.formal.tasks.recovery import RecoveryPoseCompiler
from transformer_rl.formal.tasks.reset_collision import AssetCollisionGeometry
from transformer_rl.formal.tasks.reset_kinematics import AssetResetKinematics


def source_reset():
    original, raw, model, mesh = collision_fixture()
    manifest, specification = json.loads(raw), json.loads(model)
    manifest['nominal_joint_pos'] = dict.fromkeys(manifest['nominal_joint_pos'], 0.)
    specification['nominal_joint_pos'] = dict(manifest['nominal_joint_pos'])
    raw, model = encode(manifest, specification)
    binding = MechanicalBinding.from_manifest(
        manifest, original.kinematics.binding.native_joint_names,
        original.kinematics.binding.native_body_names)
    value = object.__new__(LabRecoveryReset)
    value.num_envs, value.device, value.config = 16, torch.device('cpu'), RecoveryResetConfig()
    value.kinematics = AssetResetKinematics(raw, model, binding)
    value.geometry = AssetCollisionGeometry(value.kinematics, raw, model,
                                           {'collisions/body.obj': mesh})
    value.compiler = RecoveryPoseCompiler(value.kinematics, value.config.recovery_tuning())
    value._flat_courses = value._build_flat_courses()
    value._local_patch = value._build_local_patch()
    return value


@pytest.mark.parametrize('yaw', [0., .317, -math.pi + .02])
def test_release_poses_keep_extra_clearance_and_world_down_velocity(yaw):
    value = source_reset()
    ids = torch.tensor([8, 2])
    origins = torch.tensor([[10., -3., 2.]] * 2, dtype=torch.float64)
    result = value.pose(ids, torch.tensor([0, 2]),
                        torch.full((2,), yaw, dtype=torch.float64), origins)
    assert result.geometry_ready.all()
    # This synthetic asset proves the upright pair, not crouched feasibility.
    torch.testing.assert_close(result.root_pose_xyzw[1, 2] - result.root_pose_xyzw[0, 2],
                               torch.tensor(.04), atol=3e-7, rtol=0)
    expected = torch.zeros(2, 6)
    expected[1, 2] = -.15
    torch.testing.assert_close(result.root_com_velocity, expected, atol=1e-7, rtol=0)
    assert not result.joint_velocity.any()
    torch.testing.assert_close(result.joint_position[0], result.joint_position[1])


def test_nonrelease_poses_keep_zero_initial_velocity_and_continuous_yaw():
    value = source_reset()
    result = value.pose(torch.tensor([0, 1]), torch.tensor([0, 0]),
                        torch.tensor([.317, .318], dtype=torch.float64),
                        torch.zeros(2, 3, dtype=torch.float64))
    assert result.geometry_ready.all()
    assert not result.root_com_velocity.any()
    assert not torch.equal(result.root_pose_xyzw[0, 3:], result.root_pose_xyzw[1, 3:])

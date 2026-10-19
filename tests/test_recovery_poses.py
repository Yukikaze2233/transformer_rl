"""Recovery request semantics and source-coordinate admission checks."""
from dataclasses import replace
import math

import pytest
import torch

from test_reset_kinematics import fixture
from transformer_rl._output_actuation import OutputFeedbackController, PairTargetDomain
from transformer_rl._recovery_poses import RECOVERY_NAMES, RecoveryPoseCompiler, RecoveryTuning


def tuning():
    return RecoveryTuning(math.radians(50), .25, math.pi, .04, .15, 1e-6, 1e-6, 80)


def test_selected_geometric_knee_mapping_handles_both_mirrored_native_coordinates():
    model, *_ = fixture()
    expected = torch.deg2rad(torch.tensor([[40., 107.], [50., 80.], [107., 40.]], dtype=torch.float64))
    q = model.nominal_tree_position[None].repeat(3, 1)
    q[:, model.knee_tree_indices] = model.native_knee_positions(expected)
    torch.testing.assert_close(model.inner_knee_angles(q), expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match='domain'):
        model.native_knee_positions(expected + math.pi)
    seed = torch.zeros(2, 18, dtype=torch.float64)
    hip = seed[:, [model.policy_tree_indices[0], model.policy_tree_indices[2]]]
    theta = model.inner_knee_angles(seed)
    result = model.solve_posture(seed, hip, theta, closure_tolerance_m=1e-7)
    assert result.converged.all() and result.physical_qualification is False
    torch.testing.assert_close(model.inner_knee_angles(result.tree_position), theta, atol=1e-12, rtol=0)


def test_all16_requests_remain_distinct_in_orientation_joint_shape_or_release_data():
    model, *_ = fixture()
    compiler = RecoveryPoseCompiler(model, tuning())
    controller = OutputFeedbackController(model.binding.nominal_policy_position_rad, 16, pair_domain=None)
    value = compiler.compile(torch.arange(16), torch.zeros(16, dtype=torch.float64), controller)
    assert len(RECOVERY_NAMES) == len(set(RECOVERY_NAMES)) == 16
    assert value.requested_posture_id.tolist() == list(range(16))
    descriptors = torch.cat((value.requested_root_orientation_xyzw, value.requested_inner_knee_rad,
                             value.requested_leg_forward_pitch_rad, value.release_clearance_m[:, None]), -1)
    assert len(torch.unique(descriptors, dim=0)) == 16
    assert value.release_clearance_m.nonzero().flatten().tolist() == [2, 3]
    assert value.origin_velocity_body_m_s[2, 2] == -.15
    assert not value.current_output_hold_admitted[14]  # half-turn exceeds the existing action box
    assert value.required_preparation_action[14, :4].abs().max() > 3
    assert value.requested_root_orientation_xyzw[4, 0] < 0 and value.requested_root_orientation_xyzw[5, 0] > 0
    assert not value.physical_qualification


def test_hold_encoder_preserves_common_winding_and_rejects_halfturn_or_pair_projection():
    nominal = (0.,) * 6
    controller = OutputFeedbackController(nominal, 3, pair_domain=None)
    position = torch.zeros(3, 6)
    position[0, :4] = 2 * math.pi
    position[1, :4] = math.pi
    position[2, 1] = 2 * math.pi
    value = controller.reset_hold_action(position, position_tolerance_rad=1e-6)
    assert value.admitted.tolist() == [True, False, False]
    decoded = controller.decode(value.requested_action, position, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(decoded.admitted_leg_position_rad[0], position[0, :4], atol=1e-6, rtol=0)
    limited = OutputFeedbackController(nominal, 1, pair_domain=PairTargetDomain(((-.1, .1), (-.1, .1)), (1, 1)))
    q = torch.zeros(1, 6)
    q[0, 1] = .5
    clipped = limited.reset_hold_action(q, position_tolerance_rad=1e-6)
    assert not clipped.admitted[0] and clipped.pair_projected[0, 0]


def test_continuous_perturbations_do_not_get_quantized_or_clipped_into_a_recovery_bucket():
    model, *_ = fixture()
    compiler = RecoveryPoseCompiler(model, tuning())
    controller = OutputFeedbackController(model.binding.nominal_policy_position_rad, 2, pair_domain=None)
    offsets = torch.tensor([[.001, -.003], [.002, -.004]], dtype=torch.float64)
    value = compiler.compile(torch.tensor([10, 10]), torch.tensor([.3, .4], dtype=torch.float64), controller,
                              leg_pitch_offset_rad=offsets)
    torch.testing.assert_close(value.requested_leg_forward_pitch_rad - offsets,
                               torch.tensor([[.25, -.25], [.25, -.25]], dtype=torch.float64), atol=0, rtol=0)
    assert not torch.equal(value.requested_root_orientation_xyzw[0], value.requested_root_orientation_xyzw[1])

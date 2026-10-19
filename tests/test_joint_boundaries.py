"""Pin-angle and target-admission checks using an explicit synthetic asset."""
from copy import deepcopy
import hashlib
import json
import math

import pytest
import torch

from test_physical_observation import BODIES, TREE, fixture as physical_fixture
from transformer_rl.formal.control.joints import LegBoundaryGeometry
from transformer_rl.formal.control.actuation import OutputFeedbackController, PairTargetDomain


def asset_sources(binding):
    parents = (('base_link', 'L_link1'), ('base_link', 'R_link1'), ('base_link', 'LL_link1'), ('base_link', 'RR_link1'),
               ('L_link1', 'L_link2'), ('R_link1', 'R_link2'), ('LL_link1', 'LL_link2'), ('RR_link1', 'RR_link2'),
               ('L_link2', 'L_link3'), ('L_link2', 'LLL_link2'), ('R_link2', 'R_link3'), ('R_link2', 'RRR_link2'),
               ('LL_link2', 'LL_link3'), ('RR_link2', 'RR_link3'), ('LLL_link2', 'LLL_link1'),
               ('RRR_link2', 'RRR_link1'), ('LL_link3', 'LL_link4'), ('RR_link3', 'RR_link4'))
    joints = []
    for name, (parent, child) in zip(TREE, parents):
        kind = 'prismatic' if 'spring_slide' in name else 'continuous'
        joints.append({'name': name, 'parent': parent, 'child': child, 'type': kind,
                        'origin': torch.eye(4).tolist(), 'axis': [0., 0., 1.], 'limit': {}})
    indexed = {j['name']: j for j in joints}
    for side, hip, knee, wheel in ((0, 'L_joint1', 'L_joint2', 'L_joint3'), (1, 'R_joint1', 'R_jonit2', 'R_joint3')):
        indexed[hip]['origin'][1][3] = .2 if side == 0 else -.2
        indexed[knee]['type'] = 'revolute'
        indexed[knee]['origin'] = [[0., 1., 0., 0.], [-1., 0., 0., .2], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        indexed[knee]['axis'] = [0., 0., 1. if side == 0 else -1.]
        limits = (-50., 17.) if side == 0 else (-17., 50.)
        indexed[knee]['limit'] = {key: math.radians(v) for key, v in zip(('lower', 'upper'), limits)}
        indexed[wheel]['origin'][1][3], indexed[wheel]['origin'][2][3] = .25, .02
    nominal = dict(zip(TREE, (.01 * i for i in range(18))))
    manifest = {'control_frame': 'Xforward_Yleft_Zup', 'tree_joint_names': list(TREE), 'rigid_body_names': list(BODIES),
                 'control_joint_names': ['L_joint1', 'LL_joint1', 'L_joint3', 'R_joint1', 'RR_joint1', 'R_joint3'],
                 'nominal_joint_pos': nominal, 'knee_angle_definition': 'pin_center_vectors_projected_normal_to_common_hinge_axis',
                 'knee_inner_limits_deg': [40., 107.], 'total_mass_kg': 13.886, 'files_sha256': {}}
    return manifest, {'joints': joints}


def encode(manifest, spec):
    manifest = deepcopy(manifest)
    model = (json.dumps(spec, sort_keys=True) + '\n').encode()
    manifest['files_sha256']['model_spec.json'] = hashlib.sha256(model).hexdigest()
    return (json.dumps(manifest, sort_keys=True) + '\n').encode(), model


def quaternion(rotation):
    """Independent test FK conversion with stable diagonal branches."""
    m = rotation.tolist()
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        value = ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, .25 * s)
    else:
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + m[i][i] - m[j][j] - m[k][k]) * 2
        value = [0.] * 4
        value[i], value[j], value[k], value[3] = .25 * s, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s, (m[k][j] - m[j][k]) / s
    result = torch.tensor(value, dtype=torch.float64)
    return result / result.norm()


def forward_poses(spec, binding, qtree, root_pose=None):
    rows = []
    root_pose = torch.tensor([[0., 0., 2.305, 0., 0., 0., 1.]]).repeat(qtree.shape[0], 1) if root_pose is None else root_pose
    for row, qvalues in enumerate(qtree.double()):
        x, y, z, w = root_pose[row, 3:].double()
        root = torch.eye(4, dtype=torch.float64)
        root[:3, :3] = torch.stack((torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
                                     torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
                                     torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))))
        root[:3, 3] = root_pose[row, :3]
        transforms, remaining = {'base_link': root}, list(spec['joints'])
        while remaining:
            progressed = False
            for joint in tuple(remaining):
                if joint['parent'] not in transforms:
                    continue
                q = qvalues[binding.tree_joint_names.index(joint['name'])]
                axis = torch.tensor(joint['axis'], dtype=torch.float64)
                axis /= axis.norm()
                motion = torch.eye(4, dtype=torch.float64)
                if joint['type'] == 'prismatic':
                    motion[:3, 3] = axis * q
                else:
                    x, y, z = axis
                    cross = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
                    motion[:3, :3] = torch.eye(3, dtype=torch.float64) + q.sin()*cross + (1-q.cos())*(cross@cross)
                transforms[joint['child']] = transforms[joint['parent']] @ torch.tensor(joint['origin'], dtype=torch.float64) @ motion
                remaining.remove(joint)
                progressed = True
            assert progressed
        rows.append(torch.stack([torch.cat((transforms[name][:3, 3], quaternion(transforms[name][:3, :3])))
                                  for name in binding.native_body_names]))
    return torch.stack(rows).float()


def fixture():
    binding = physical_fixture()[1]
    manifest, spec = asset_sources(binding)
    raw, model = encode(manifest, spec)
    geometry = LegBoundaryGeometry.from_asset_sources(raw, model, binding)
    qtree = torch.zeros(2, 18)
    controller = OutputFeedbackController(binding.nominal_policy_position_rad, 2, pair_domain=None)
    targets = controller.decode(torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2, dtype=torch.float64))
    return geometry, manifest, spec, qtree, targets


def test_pin_angles_reproduce_both_mirrored_stops_and_preserve_a_negative_true_violation():
    geometry, _, spec, q, targets = fixture()
    for side in geometry.sides:
        index = geometry.binding.tree_joint_names.index(side.knee_joint)
        q[:, index] = torch.tensor(side.native_limits_rad)
    result = geometry.evaluate(forward_poses(spec, geometry.binding, q), q, targets, knee_margin_rad=.05)
    expected = torch.deg2rad(torch.tensor([[40., 107.], [107., 40.]]))
    torch.testing.assert_close(result.knee_inner_angle_rad, expected, atol=5e-7, rtol=0)
    torch.testing.assert_close(result.shortfall_rad[:, :2], torch.full((2, 2), .05), atol=5e-7, rtol=0)
    assert result.knee_pivot_residual_m.lt(1e-6).all()
    q[:, geometry.binding.tree_joint_names.index(geometry.sides[0].knee_joint)] -= .1
    outside = geometry.evaluate(forward_poses(spec, geometry.binding, q), q, targets, knee_margin_rad=.05)
    assert outside.knee_signed_clearance_rad[0, 0] < -.09
    assert outside.shortfall_rad[0, 0] > .14


def test_geometric_measurement_is_not_replaced_by_inconsistent_joint_coordinates():
    geometry, _, spec, q, targets = fixture()
    poses = forward_poses(spec, geometry.binding, q)
    wrong_q = q.clone()
    wrong_q[:, geometry.binding.tree_joint_names.index(geometry.sides[0].knee_joint)] = geometry.sides[0].native_limits_rad[0]
    result = geometry.evaluate(poses, wrong_q, targets, knee_margin_rad=.05)
    torch.testing.assert_close(result.knee_inner_angle_rad, torch.full((2, 2), math.pi/2))
    assert result.knee_signed_clearance_rad[:, 0].gt(.2).all()
    assert result.native_knee_signed_clearance_rad[:, 0].abs().lt(1e-7).all()


def test_common_winding_and_pair_projection_remain_distinct_from_real_knee_margin():
    geometry, _, spec, q, _ = fixture()
    controller = OutputFeedbackController(geometry.binding.nominal_policy_position_rad, 2,
                                           pair_domain=PairTargetDomain(((-.1, .1), (-.1, .1)), (1, -1)))
    issued = torch.tensor([[1., -1., -1., 1., 0., 0.]]).repeat(2, 1)
    zero = controller.decode(issued, torch.zeros(2, 6), torch.zeros(2, dtype=torch.float64))
    wound = controller.decode(issued, torch.tensor([[6*math.pi, 6*math.pi, -4*math.pi, -4*math.pi, 0., 0.]]).repeat(2, 1),
                              torch.zeros(2, dtype=torch.float64))
    first = geometry.evaluate(forward_poses(spec, geometry.binding, q), q, zero, knee_margin_rad=.05)
    second = geometry.evaluate(forward_poses(spec, geometry.binding, q), q, wound, knee_margin_rad=.05)
    assert not first.shortfall_rad[:, :2].any()
    assert first.output_admission_correction_rad.gt(.18).all()
    torch.testing.assert_close(first.shortfall_rad, second.shortfall_rad, atol=2e-6, rtol=0)


def test_three_dimensional_world_rotation_translation_and_native_body_permutation_do_not_change_angles():
    geometry, _, spec, q, targets = fixture()
    baseline = geometry.evaluate(forward_poses(spec, geometry.binding, q), q, targets, knee_margin_rad=.05)
    root = torch.tensor([[4., -2., 5., .5, .5, .5, .5]]).repeat(2, 1)
    rotated = geometry.evaluate(forward_poses(spec, geometry.binding, q, root), q, targets, knee_margin_rad=.05)
    torch.testing.assert_close(rotated.knee_inner_angle_rad, baseline.knee_inner_angle_rad, atol=2e-6, rtol=0)
    assert rotated.knee_pivot_residual_m.lt(1e-6).all()


@pytest.mark.parametrize('fault', ['hash', 'definition', 'geometric_span', 'origin_shear', 'axis_zero', 'wheel_parent', 'mass', 'pin_collapse'])
def test_invalid_selected_geometry_is_rejected_before_any_native_read(fault):
    geometry, manifest, spec, _, _ = fixture()
    knee = next(j for j in spec['joints'] if j['name'] == geometry.sides[0].knee_joint)
    if fault == 'definition':manifest['knee_angle_definition'] = 'motor_difference'
    elif fault == 'geometric_span':manifest['knee_inner_limits_deg'][1] = 106
    elif fault == 'origin_shear':knee['origin'][0][0] = .2
    elif fault == 'axis_zero':knee['axis'] = [0., 0., 0.]
    elif fault == 'wheel_parent':next(j for j in spec['joints'] if j['name'] == 'L_joint3')['parent'] = 'base_link'
    elif fault == 'mass':manifest['total_mass_kg'] = -1
    elif fault == 'pin_collapse':knee['origin'][1][3] = 0
    raw, model = encode(manifest, spec)
    if fault == 'hash':model += b' '
    with pytest.raises(ValueError):
        LegBoundaryGeometry.from_asset_sources(raw, model, geometry.binding)


@pytest.mark.parametrize('fault', ['quaternion', 'nan', 'collapsed_physical_pin', 'fractional_winding'])
def test_bad_physical_geometry_or_winding_cannot_be_filled_with_safe_zero_shortfalls(fault):
    geometry, _, spec, q, targets = fixture()
    poses = forward_poses(spec, geometry.binding, q)
    if fault == 'quaternion':poses[0, 0, 6] = 2
    elif fault == 'nan':poses[0, 0, 0] = float('nan')
    elif fault == 'fractional_winding':targets.common_turns[0, 0] = .5
    else:
        upper = geometry.binding.native_body_names.index(geometry.sides[0].upper_body)
        base = geometry.binding.native_body_names.index('base_link')
        # Move the actual parent hip pin onto its actual knee pin.
        poses[:, base, :3] = poses[:, upper, :3]
        poses[:, base, 1] += .2 - geometry.sides[0].hip_pin_parent_m[1]
    with pytest.raises(ValueError):
        geometry.evaluate(poses, q, targets, knee_margin_rad=.05)

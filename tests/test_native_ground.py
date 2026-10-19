"""Fresh native pose/terrain binding checks with synthetic SDK buffers only."""
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from test_native_observation import native_fixture
from transformer_rl.formal.simulation.ground import NativeGroundQuery, TerrainCoverageError, WheelCollisionCentres
from transformer_rl.formal.simulation.scene import NativeSceneState
from transformer_rl.formal.tasks.geometry import HeightPatch, TerrainSurface
from transformer_rl.formal.tasks.terrain import TerrainCourse, TerrainExtent, TerrainRequest, TerrainKind


def fixture():
    robot, bridge, _ = native_fixture()
    poses = torch.zeros(2, 19, 7)
    poses[..., 6] = 1
    poses[:, 0] = robot.data.root_link_pose_w.torch
    wheels = list(bridge.binding.wheel_body_indices)
    poses[:, wheels, :3] = torch.tensor([[[.1, .2, 2.06], [.1, -.2, 2.06]]]).repeat(2, 1, 1)
    robot.data.body_link_pose_w = SimpleNamespace(torch=poses)
    req = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-3., 3.), (-3., 3.)))
    # Two independent materials and heights, in design order. Native order is reversed.
    a = TerrainCourse(req, 1, TerrainSurface((HeightPatch((-3., 3.), (-3., 3.), ((0., 0.), (0., 0.)), .3),)))
    b = TerrainCourse(req, 2, TerrainSurface((HeightPatch((-3., 3.), (-3., 3.), ((.4, .4), (.4, .4)), .9),)))
    courses = (a, b)
    meshes = tuple(courses[i].surface.meshes for i in (1, 0))
    scene = NativeSceneState(robot, bridge, object(), object(), ('env_b', 'env_a'),
                             torch.tensor([[0., 0., 2.], [0., 0., 2.]]), meshes, (1, 0), {})
    names = tuple(bridge.binding.native_body_names[i] for i in wheels)
    centres = WheelCollisionCentres(names, ((0., 0., .02035), (0., 0., -.02035)), (.06, .06), (.025, .025))
    return robot, bridge, scene, courses, centres


def test_query_reorders_actual_clones_and_returns_world_height_and_actual_materials():
    robot, bridge, scene, courses, centres = fixture()
    query = NativeGroundQuery(scene, courses, centres)
    with pytest.raises(RuntimeError, match='refresh'):
        query(robot, scene)
    bridge.after_physics_step(1)
    result = query(robot, scene)
    torch.testing.assert_close(result.root_world_z_m, torch.tensor([2.4, 2.]))
    torch.testing.assert_close(result.wheel_world_z_m, torch.tensor([[2.4, 2.4], [2., 2.]]))
    torch.testing.assert_close(result.wheel_friction, torch.tensor([[.9, .9], [.3, .3]]))
    assert query.last_sample.model_epoch == bridge.model_epoch and query.last_sample.terrain.valid.all()
    assert bridge.physical_tick == 1 and robot.updates == [.005]


def test_collider_centre_offset_is_rotated_before_looking_up_cross_slope_height():
    robot, bridge, scene, courses, centres = fixture()
    patch = HeightPatch((-3., 3.), (-3., 3.), ((-.3, .3), (-.3, .3)), .8)
    surface = TerrainSurface((patch,))
    courses = tuple(replace(c, surface=surface) for c in courses)
    scene = replace(scene, terrain_meshes=(surface.meshes, surface.meshes))
    wheel_ids = list(bridge.binding.wheel_body_indices)
    # +90 degrees around X rotates local +Z to world -Y.
    robot.data.body_link_pose_w.torch[:, wheel_ids, 3:7] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    result = query(robot, scene)
    expected_y = torch.tensor([[.2 - .02035, -.2 + .02035]]).repeat(2, 1)
    torch.testing.assert_close(query.last_sample.world_xyz_m[:, 1:, 1], expected_y)
    torch.testing.assert_close(result.wheel_world_z_m, expected_y * .1 + 2., atol=3e-7, rtol=0)
    assert not torch.allclose(result.wheel_world_z_m, torch.tensor([[2.02, 1.98]]).repeat(2, 1))


def test_reused_native_pose_buffers_do_not_mutate_previous_ground_evidence():
    robot, bridge, scene, courses, centres = fixture()
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    query(robot, scene)
    first = query.last_sample
    expected = first.world_xyz_m.clone()
    robot.data.body_link_pose_w.torch[:, 1:, 0] += .5
    query(robot, scene)
    torch.testing.assert_close(first.world_xyz_m, expected)


def test_reset_requires_an_actual_new_refresh_before_ground_reading():
    robot, bridge, scene, courses, centres = fixture()
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    query(robot, scene)
    bridge.invalidate_state()
    with pytest.raises(RuntimeError, match='refresh'):
        query(robot, scene)
    bridge.after_physics_step(2)
    query(robot, scene)
    assert bridge.physical_tick == 2


def test_outside_ground_remains_typed_invalid_evidence_instead_of_zero_height():
    robot, bridge, scene, courses, centres = fixture()
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    robot.data.body_link_pose_w.torch[0, bridge.binding.wheel_body_indices[0], 0] = 10
    with pytest.raises(TerrainCoverageError) as raised:
        query(robot, scene)
    sample = raised.value.sample
    assert sample.terrain.valid.tolist() == [[True, False, True], [True, True, True]]
    assert torch.isnan(sample.terrain.height_m[0, 1])
    assert sample.world_xyz_m[0, 1, 0] == 10


@pytest.mark.parametrize('change', ['geometry', 'clone_permutation', 'wheel_order'])
def test_geometry_or_clone_binding_mismatch_is_rejected(change):
    _, _, scene, courses, centres = fixture()
    if change == 'geometry':
        scene = replace(scene, terrain_meshes=tuple(reversed(scene.terrain_meshes)))
    elif change == 'clone_permutation':
        scene = replace(scene, clone_indices=(0, 0))
    else:
        centres = replace(centres, body_names=tuple(reversed(centres.body_names)))
    with pytest.raises(ValueError):
        NativeGroundQuery(scene, courses, centres)


@pytest.mark.parametrize('change', ['owner', 'origin', 'root_frame', 'orientation'])
def test_live_owner_origin_and_pose_reference_changes_are_not_silently_reinterpreted(change):
    robot, bridge, scene, courses, centres = fixture()
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    if change == 'owner':
        robot.root_view = object()
    elif change == 'origin':
        scene.environment_origins_m[0, 0] += 1
    elif change == 'root_frame':
        robot.data.body_link_pose_w.torch[0, 0, 0] += .01
    else:
        robot.data.body_link_pose_w.torch[0, bridge.binding.wheel_body_indices[0], 6] = 2
    with pytest.raises((ValueError, RuntimeError)):
        query(robot, scene)


def test_root_quaternion_sign_does_not_change_the_geometric_reference_point():
    robot, bridge, scene, courses, centres = fixture()
    bridge.after_physics_step(1)
    query = NativeGroundQuery(scene, courses, centres)
    first = query(robot, scene)
    robot.data.body_link_pose_w.torch[:, 0, 3:7].neg_()
    second = query(robot, scene)
    torch.testing.assert_close(first.root_world_z_m, second.root_world_z_m)


def test_named_urdf_collision_origins_are_used_instead_of_wheel_joint_or_visual_origins():
    _, bridge, _, _, _ = fixture()
    names = tuple(bridge.binding.native_body_names[i] for i in bridge.binding.wheel_body_indices)
    urdf = '<robot name="fixture">' + ''.join(
        '<link name="' + name + '"><visual><origin xyz="9 9 9"/></visual>'
        '<collision><origin xyz="0 0 ' + str(offset) + '" rpy="0 0 0"/>'
        '<geometry><cylinder radius="0.06" length="0.025"/></geometry></collision></link>'
        for name, offset in zip(names, (.02035, -.02035))) + '</robot>'
    descriptor = WheelCollisionCentres.from_urdf(urdf, bridge.binding)
    assert descriptor.body_names == names and descriptor.local_xyz_m == ((0., 0., .02035), (0., 0., -.02035))
    assert descriptor.radius_m == (.06, .06)
    with pytest.raises(ValueError):
        WheelCollisionCentres.from_urdf(urdf.replace('cylinder', 'sphere'), bridge.binding)


def test_recorded_collider_precision_is_not_replaced_by_a_rounded_nominal_offset():
    _, bridge, _, _, _ = fixture()
    names = tuple(bridge.binding.native_body_names[i] for i in bridge.binding.wheel_body_indices)
    offset = .020350000821053982
    text = '<robot>' + ''.join(
        '<link name="' + name + '"><collision><origin xyz="0 0 ' + str(value) + '"/>'
        '<geometry><cylinder radius="0.06" length="0.02500000037252903"/></geometry>'
        '</collision></link>' for name, value in zip(names, (offset, -offset))) + '</robot>'
    centres = WheelCollisionCentres.from_urdf(text, bridge.binding)
    assert centres.local_xyz_m == ((0., 0., offset), (0., 0., -offset))
    assert centres.local_xyz_m[0][2] != .02035
    assert centres.length_m == (.02500000037252903, .02500000037252903)


def test_collision_rpy_is_applied_to_the_cylinder_axis_and_urdf_bytes_are_bound():
    _, bridge, _, _, _ = fixture()
    names = tuple(bridge.binding.native_body_names[i] for i in bridge.binding.wheel_body_indices)
    urdf = '<robot>' + ''.join('<link name="' + name + '"><collision><origin xyz="0 0 0" '
                               'rpy="1.5707963267948966 0 0"/><geometry><cylinder radius=".06" '
                               'length=".025"/></geometry></collision></link>' for name in names) + '</robot>'
    value = WheelCollisionCentres.from_urdf(urdf, bridge.binding)
    assert value.local_axis_xyz[0] == pytest.approx((0., -1., 0.), abs=1e-12)
    assert len(value.urdf_sha256) == 64
    assert WheelCollisionCentres.from_urdf(urdf + ' ', bridge.binding).urdf_sha256 != value.urdf_sha256
    with pytest.raises(ValueError):
        WheelCollisionCentres.from_urdf(urdf.replace('1.5707963267948966', 'nan'), bridge.binding)


def test_full_collision_gap_is_owned_same_tick_and_explicitly_unqualified_without_native_evidence():
    robot, bridge, scene, courses, centres = fixture()
    ids = list(bridge.binding.wheel_body_indices)
    # Rotate the actual link and its offset/axis; neither link-Z minus radius
    # nor a cylinder assumed to remain vertical describes this packet.
    robot.data.body_link_pose_w.torch[:, ids, 3:7] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    value = query(robot, scene).wheel_clearance
    assert value is query.last_sample.wheel_clearance
    assert value.physical_tick == 1 and value.model_epoch == bridge.model_epoch
    assert value.geometry.valid.all() and value.geometry.domain_covered.all()
    torch.testing.assert_close(value.geometry.vertical_gap_m, torch.tensor([[-.4, -.4], [0., 0.]]), atol=2e-7, rtol=0)
    assert not value.source_geometry_bound and not value.native_collision_equivalence_verified
    before = value.geometry.cylinder_point_local_m.clone()
    robot.data.body_link_pose_w.torch[:, ids, 2] += .1
    query(robot, scene)
    assert torch.equal(value.geometry.cylinder_point_local_m, before)


def test_complete_footprint_can_be_invalid_while_centre_height_remains_valid():
    robot, bridge, scene, courses, centres = fixture()
    robot.data.body_link_pose_w.torch[:, list(bridge.binding.wheel_body_indices), 0] = 2.99
    query = NativeGroundQuery(scene, courses, centres)
    bridge.after_physics_step(1)
    result = query(robot, scene)
    assert query.last_sample.terrain.valid.all()
    assert not result.wheel_clearance.geometry.valid.any()
    assert torch.isnan(result.wheel_clearance.geometry.vertical_gap_m).all()


def test_actual_scene_urdf_source_mismatch_is_rejected_before_any_geometry_read():
    robot, bridge, scene, courses, centres = fixture()
    scene.readback['selected_urdf_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='URDF bytes'):
        NativeGroundQuery(scene, courses, centres)

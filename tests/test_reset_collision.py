"""Full-source collision support and geometric reset preparation on Kaiser CPU."""
from dataclasses import replace
import hashlib
import math

import pytest
import torch

from test_joint_boundaries import encode
from test_reset_kinematics import fixture as kinematic_fixture
from transformer_rl._command_trajectory import CommandTrajectory
from transformer_rl._reset_collision import AssetCollisionGeometry, FlatSpawnPatch
from transformer_rl._reset_kinematics import AssetResetKinematics
from transformer_rl._task_commands import TaskCommandPlan
from transformer_rl._terrain_geometry import HeightPatch, TerrainKind, TerrainSurface
from transformer_rl._terrain_tasks import (
    ContinuousTerrainGenerator, TerrainCourse, TerrainExtent, TerrainRequest, TerrainTuning,
)


def fixture(*, lowered_body=False, raised_non_wheel=False, wheel_offsets=(0., 0.), chunk_size=3):
    original, manifest, spec = kinematic_fixture()
    mesh = (b'v -.02 -.03 -.04\nv .02 -.03 -.04\nv .02 .03 -.04\nv -.02 .03 -.04\n'
            b'v -.02 -.03 .04\nv .02 -.03 .04\nv .02 .03 .04\nv -.02 .03 .04\n'
            b'f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\nf 3 4 8 7\n')
    manifest['files_sha256']['collisions/body.obj'] = hashlib.sha256(mesh).hexdigest()
    wheels = {original.binding.native_body_names[i] for i in original.binding.wheel_body_indices}
    for body in spec['bodies']:
        origin = torch.eye(4).tolist()
        if body['name'] not in wheels and raised_non_wheel:
            origin[2][3] = .5
        if lowered_body and body['name'] == 'base_link':
            origin[2][3] = -.9
        if body['name'] in wheels:
            which = list(original.binding.wheel_body_indices).index(original.binding.native_body_names.index(body['name']))
            origin[2][3] = wheel_offsets[which]
        body['collisions'] = ([{'type': 'cylinder', 'radius': .06, 'length': .025, 'origin': origin}]
                              if body['name'] in wheels else
                              [{'type': 'mesh', 'file': 'collisions/body.obj', 'origin': origin}])
    raw, model = encode(manifest, spec)
    kin = AssetResetKinematics(raw, model, original.binding)
    geometry = AssetCollisionGeometry(kin, raw, model, {'collisions/body.obj': mesh}, vertex_chunk_size=chunk_size)
    return geometry, raw, model, mesh


def patch(count=2, *, origins=None, bounds=None, courses=None):
    origins = torch.zeros(count, 3, dtype=torch.float64) if origins is None else origins
    if courses is None:
        generator = ContinuousTerrainGenerator(44101, TerrainTuning(.8, (.2, 1.1), .25))
        request = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-3., 3.), (-3., 3.)))
        courses = tuple(generator.build(request) for _ in range(count))
    bounds = torch.tensor([[[-2., 2.], [-2., 2.]]], dtype=torch.float64).repeat(count, 1, 1) if bounds is None else bounds
    return FlatSpawnPatch.from_courses(courses, origins, bounds, torch.zeros(count, dtype=torch.float64))


def unit_orientation(count=2):
    return torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64).repeat(count, 1)


def commands(count=2):
    initial = torch.tensor([[0., 0., 0., .305]]).repeat(count, 1)
    return TaskCommandPlan(CommandTrajectory(initial, initial, torch.ones(count), torch.zeros(count, dtype=torch.long)),
                           torch.full((count,), 4, dtype=torch.long), torch.full((count,), 30., dtype=torch.float64),
                           torch.zeros(count), torch.zeros(count, dtype=torch.float64))


def test_all_body_meshes_not_only_wheels_determine_the_birth_height_and_native_joint_order():
    geometry, *_ = fixture(lowered_body=True)
    q = torch.zeros(2, 18, dtype=torch.float64)
    q[:, geometry.kinematics.policy_tree_indices[4:]] = torch.tensor([7., -11.], dtype=torch.float64)
    origins = torch.tensor([[100., -200., 5.], [300., 0., -2.]], dtype=torch.float64)
    own = patch(origins=origins)
    plan, result = geometry.reset_plan(q, unit_orientation(), origins[:, :2], own, commands(),
                                       required_support='any_collider', clearance_m=.002,
                                       loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    body = geometry.kinematics.binding.native_body_names.index('base_link')
    assert result.geometry_ready.all() and result.patch_covers_full_footprint.all()
    assert not result.bilateral_wheel_support.any() and result.wheel_gap_m.min() > .8
    assert (result.body_minimum_gap_m[:, body] >= .002).all()
    assert result.body_minimum_gap_m[:, body].tolist() == pytest.approx([.002, .002], abs=5e-7)
    assert result.root_pose_xyzw[:, 2].tolist() == pytest.approx([5.942, -1.058])
    assert plan.root_pose_xyzw.dtype == torch.float32 and plan.joint_position.shape == (2, 18)
    native_wheels = list(geometry.kinematics.binding.policy_joint_indices[4:])
    assert plan.joint_position[:, native_wheels].tolist() == [[7., -11.], [7., -11.]]
    assert not plan.root_com_velocity.any() and not plan.joint_velocity.any()
    assert plan.course_sha256 == own.course_sha256 and not result.physical_qualification
    with pytest.raises(ValueError, match='support'):
        geometry.reset_plan(q, unit_orientation(), origins[:, :2], own, commands(),
                             required_support='two_wheels', clearance_m=.002,
                             loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)


def test_finite_cylinder_support_is_exact_for_rotated_postures_and_point_chunking_does_not_change_it():
    first, *_ = fixture(chunk_size=1)
    second, *_ = fixture(chunk_size=1024)
    orientation = unit_orientation()
    orientation[0] = torch.tensor([math.sin(.35), 0., 0., math.cos(.35)], dtype=torch.float64)
    orientation[1] = torch.tensor([0., math.sin(math.pi / 4), 0., math.cos(math.pi / 4)], dtype=torch.float64)
    torch.testing.assert_close(first._rotation(orientation)[1] @ torch.tensor([1., 0., 0.], dtype=torch.float64),
                               torch.tensor([0., 0., -1.], dtype=torch.float64), atol=1e-14, rtol=0)
    q = torch.zeros(2, 18, dtype=torch.float64)
    own = patch()
    a = first.place(q, orientation, torch.zeros(2, 2, dtype=torch.float64), own, clearance_m=.001,
                     loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    b = second.place(q, orientation, torch.zeros(2, 2, dtype=torch.float64), own, clearance_m=.001,
                      loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    torch.testing.assert_close(a.root_pose_xyzw, b.root_pose_xyzw, atol=1e-14, rtol=0)
    torch.testing.assert_close(a.body_minimum_gap_m, b.body_minimum_gap_m, atol=1e-14, rtol=0)
    assert a.body_minimum_gap_m.amin(-1).tolist() == pytest.approx([.001, .001])
    # Independent dense cylinder surface samples approach its analytic support.
    state = first.kinematics.forward(q)
    root = first._rotation(orientation)
    angle = torch.arange(4096, dtype=torch.float64) * (2 * math.pi / 4096)
    for body, centre, axis, radius, length in first._cylinders:
        rotation = root @ state.body_rotation[:, body]
        centre_world = (root @ state.body_position_m[:, body, :, None]).squeeze(-1) + (rotation @ centre).squeeze(-1)
        circle = torch.stack((radius * angle.cos(), radius * angle.sin(), torch.zeros_like(angle)), -1)
        points = torch.cat((circle + torch.tensor([0., 0., length / 2]),
                            circle - torch.tensor([0., 0., length / 2])))
        world = points[None] @ rotation.transpose(-1, -2) + centre_world[:, None]
        expected = world[:, :, 2].amin(-1) + a.root_pose_xyzw[:, 2]
        assert ((expected - a.body_minimum_gap_m[:, body]).abs() < 3e-8).all()


@pytest.mark.parametrize('fault', ['raised_edge', 'hole', 'slope'])
def test_flat_birth_proof_cannot_ignore_a_highest_shared_edge_hole_or_sloping_triangle(fault):
    request = TerrainRequest(TerrainKind.FLAT, TerrainExtent((-3., 3.), (-3., 3.)))
    if fault == 'raised_edge':
        surface = TerrainSurface((HeightPatch((-3., 1.), (-3., 3.), ((0., 0.), (0., 0.)), .8),
                                  HeightPatch((1., 3.), (-3., 3.), ((.2, .2), (.2, .2)), .8)))
        bounds = torch.tensor([[[-1., 1.], [-1., 1.]]], dtype=torch.float64)
    elif fault == 'hole':
        surface = TerrainSurface((HeightPatch((-3., -.5), (-3., 3.), ((0., 0.), (0., 0.)), .8),
                                  HeightPatch((.5, 3.), (-3., 3.), ((0., 0.), (0., 0.)), .8)))
        bounds = torch.tensor([[[-1., 1.], [-1., 1.]]], dtype=torch.float64)
    else:
        surface = TerrainSurface((HeightPatch((-3., 3.), (-3., 3.), ((0., 0.), (.1, .1)), .8),))
        bounds = torch.tensor([[[-1., 1.], [-1., 1.]]], dtype=torch.float64)
    with pytest.raises(ValueError):
        patch(1, bounds=bounds, courses=(TerrainCourse(request, 1, surface),))


def test_invalid_source_changed_numeric_proof_and_small_spawn_rectangle_are_not_accepted():
    geometry, raw, model, mesh = fixture()
    with pytest.raises(ValueError, match='hash'):
        AssetCollisionGeometry(geometry.kinematics, raw, model, {'collisions/body.obj': mesh + b'#changed'})
    own = patch()
    own.bounds_world_xy_m.add_(1)
    with pytest.raises(ValueError, match='modified'):
        geometry.place(torch.zeros(2, 18, dtype=torch.float64), unit_orientation(),
                        torch.zeros(2, 2, dtype=torch.float64), own, clearance_m=.001,
                        loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    small = patch(bounds=torch.tensor([[[-.01, .01], [-.01, .01]]], dtype=torch.float64).repeat(2, 1, 1))
    result = geometry.place(torch.zeros(2, 18, dtype=torch.float64), unit_orientation(),
                             torch.zeros(2, 2, dtype=torch.float64), small, clearance_m=.001,
                             loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    assert not result.geometry_ready.any() and not result.patch_covers_full_footprint.any()


def test_obj_support_uses_actual_faces_not_unused_vertices_and_rejects_invalid_indices():
    raw = b'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 -100\nf -4/-1 -3/-1 -2/-1\n'
    value = AssetCollisionGeometry._obj_face_vertices(raw)
    assert value.shape == (3, 3) and value[:, 2].eq(0).all()
    with pytest.raises(ValueError, match='index'):
        AssetCollisionGeometry._obj_face_vertices(b'v 0 0 0\nf 1 2 3\n')


def test_bounded_root_roll_correction_can_align_both_wheels_without_touching_joint_targets():
    geometry, *_ = fixture(raised_non_wheel=True, wheel_offsets=(0., .002))
    q = torch.zeros(2, 18, dtype=torch.float64)
    orientation, xy, own = unit_orientation(), torch.zeros(2, 2, dtype=torch.float64), patch()
    original = geometry.place(q, orientation, xy, own, clearance_m=0., loop_tolerance_m=1e-7,
                               wheel_support_tolerance_m=.001)
    assert not original.bilateral_wheel_support.any()
    bounds = torch.tensor([[-.02, .02], [-.0001, .0001]], dtype=torch.float64)
    placed, correction, bracketed = geometry.balance_two_wheels(q, orientation, xy, own,
                                                               roll_correction_bounds_rad=bounds,
                                                               clearance_m=0., loop_tolerance_m=1e-7,
                                                               wheel_support_tolerance_m=.001)
    assert bracketed.tolist() == [True, False] and placed.bilateral_wheel_support.tolist() == [True, False]
    assert correction[0].abs() < .02 and correction[1] == 0
    assert torch.equal(placed.tree_position, q) and torch.equal(placed.root_pose_xyzw[1, 3:], orientation[1])
    assert placed.wheel_gap_m[0].abs().max() < 1e-10 and placed.geometry_ready.all()
    assert not placed.physical_qualification


def test_float32_root_height_rounds_up_and_loss_of_joint_precision_cannot_pass_reset_preparation():
    geometry, *_ = fixture(lowered_body=True)
    q = torch.zeros(2, 18, dtype=torch.float64)
    own = patch()
    plan, result = geometry.reset_plan(q, unit_orientation(), torch.zeros(2, 2, dtype=torch.float64), own,
                                       commands(), required_support='any_collider', clearance_m=0.,
                                       loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)
    assert result.body_minimum_gap_m.min() >= 0
    torch.testing.assert_close(plan.root_pose_xyzw[:, :3].double(), result.root_pose_xyzw[:, :3], atol=0, rtol=0)
    # A whole-turn tree rotation has the original geometry in float64,
    # while float32 loses its phase by milliradians.
    first = geometry.kinematics.binding.tree_joint_names.index('LL_joint2')
    q[:, first] = 100000 * 2 * math.pi
    assert geometry.kinematics.closure(q).norm(dim=-1).max() < 1e-7
    with pytest.raises(ValueError, match='float32'):
        geometry.reset_plan(q, unit_orientation(), torch.zeros(2, 2, dtype=torch.float64), own,
                             commands(), required_support='any_collider', clearance_m=0.,
                             loop_tolerance_m=1e-7, wheel_support_tolerance_m=.001)

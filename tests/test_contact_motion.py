"""Contact-point motion fixtures only; no native SDK or robot simulation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_native_contacts import fixture as contact_fixture
from test_native_observation import native_fixture
from transformer_rl.formal.simulation.contacts import NativeTerrainContacts
from transformer_rl.formal.simulation.scene import NativeSceneRecipe
from transformer_rl.formal.control.observation import RigidBodyMotion


def fixture():
    binding, contacts, views, *_ = contact_fixture()
    contacts.terrain_motion_reference = 'authored_static_own_terrain'
    for view in views:
        view.points = torch.full((view.force.numel(), 3), float('nan'))
        view.normals = torch.full_like(view.points, float('nan'))
        active = int(view.counts.sum())
        view.points[:active], view.normals[:active] = 0., torch.tensor([0., 0., 1.])
        view.get_contact_data = lambda *, dt, view=view: (view.force, view.points, view.normals, None,
                                                          view.counts, view.starts)
    shape = (contacts.num_envs, 19, 3)
    poses = torch.zeros(contacts.num_envs, 19, 7)
    poses[:, :, 6] = 1
    motion = RigidBodyMotion(poses, torch.zeros(shape), torch.zeros(shape), torch.zeros(shape),
                              torch.full((contacts.num_envs,), .005, dtype=torch.float64),
                              binding.native_body_names, 1, 1)
    return binding, contacts, views, motion


def read(contacts, motion):
    return contacts.read(.005, motion.timestamp_s, body_motion=motion)


def test_installed_uint32_indices_and_unused_max_unsigned_sentinel_are_accepted():
    force = torch.tensor([float('nan'), 2., 3., float('nan'), 7.])
    counts = torch.tensor([[2, 0], [0, 1]], dtype=torch.uint32)
    starts = torch.tensor([[1, 4294967295], [4294967295, 4]], dtype=torch.uint32)
    load, number = NativeTerrainContacts._patch_loads(force, counts, starts, 10)
    torch.testing.assert_close(load, torch.tensor([5., 7.]))
    assert number.tolist() == [2, 1]
    bad = torch.tensor([[4294967295, 0], [0, 4]], dtype=torch.uint32)
    with pytest.raises(ValueError, match='shared buffer'):
        NativeTerrainContacts._patch_loads(force, counts, bad, 10)


def test_native_clone_group_order_and_zero_load_validity_are_preserved():
    binding, contacts, views, motion = fixture()
    motion.com_velocity_world_m_s[:, :, 0] = torch.tensor([1., 2., 3.])[:, None]
    result = read(contacts, motion)
    assert result.wheel_motion is not None
    expected = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    torch.testing.assert_close(result.wheel_motion.tangent_speed_rms_m_s, expected)
    torch.testing.assert_close(result.wheel_motion.tangent_speed_max_m_s, expected)
    torch.testing.assert_close(result.wheel_motion.normal_load_n, result.normal_load_n[:, list(binding.wheel_body_indices)])
    assert result.wheel_motion.loaded.all() and result.wheel_motion.loaded_patch_count.eq(1).all()
    for view in views:
        view.force.zero_()
        view.points.fill_(float('nan'))  # Zero-force rows cannot create loaded slip evidence.
    empty = read(contacts, motion).wheel_motion
    assert not empty.loaded.any() and not empty.tangent_speed_rms_m_s.any()
    assert not empty.loaded_patch_count.any()
    assert result.wheel_motion.tangent_speed_rms_m_s.any()  # Own output, not a reused native allocation.


@pytest.mark.parametrize('linear,spin,expected', [(.6, 10., 0.), (0., 10., .6), (-.6, 10., 1.2)])
def test_actual_contact_point_velocity_distinguishes_rolling_spin_and_counter_motion(linear, spin, expected):
    binding, contacts, views, motion = fixture()
    wheels = list(binding.wheel_body_indices)
    motion.com_velocity_world_m_s[:, wheels, 0] = linear
    motion.angular_velocity_world_rad_s[:, wheels, 1] = spin
    for view in views:
        view.points[:int(view.counts.sum()), 2] = -.06
    result = read(contacts, motion).wheel_motion
    torch.testing.assert_close(result.tangent_speed_rms_m_s, torch.full((3, 2), expected), atol=1e-7, rtol=0)
    assert not result.normal_speed_rms_m_s.any()


def test_tangential_projection_uses_world_contact_normal_not_fixed_world_xy():
    _, contacts, views, motion = fixture()
    normal = torch.tensor([.6, 0., .8])
    motion.com_velocity_world_m_s[:] = 2 * normal
    for view in views:
        view.normals[:int(view.counts.sum())] = normal
    result = read(contacts, motion).wheel_motion
    torch.testing.assert_close(result.tangent_speed_rms_m_s, torch.zeros(3, 2), atol=1e-7, rtol=0)
    torch.testing.assert_close(result.normal_speed_rms_m_s, torch.full((3, 2), 2.))


@pytest.mark.parametrize('forces,points,expected', [([1., 1.], [-1., 1.], 1.), ([1., 3.], [-1., 3.], 7.**.5)])
def test_multicontact_uses_force_weighted_mean_square_and_cannot_cancel_opposing_slip(forces, points, expected):
    binding, contacts, views, motion = fixture()
    wheel = binding.wheel_body_indices[0]
    motion.angular_velocity_world_rad_s[:, wheel, 2] = 1.
    for view in views:
        view.counts.zero_()
        view.starts.fill_(-1)
        view.force.fill_(float('nan'))
        for lane in range(len(view.sensor_paths) // 19):
            index = lane * 2
            view.counts[lane * 19 + wheel, 0] = 2
            view.starts[lane * 19 + wheel, 0] = index
            view.force[index:index + 2, 0] = torch.tensor(forces)
            view.points[index:index + 2] = torch.tensor([[points[0], 0., 0.], [points[1], 0., 0.]])
            view.normals[index:index + 2] = torch.tensor([0., 0., 1.])
    result = read(contacts, motion).wheel_motion
    torch.testing.assert_close(result.tangent_speed_rms_m_s[:, 0], torch.full((3,), expected))
    assert result.tangent_speed_max_m_s[:, 0].eq(max(abs(p) for p in points)).all()
    assert result.loaded_patch_count[:, 0].eq(2).all()
    assert not result.loaded[:, 1].any()


def test_world_translation_does_not_change_contact_point_angular_velocity_offset():
    _, contacts, views, motion = fixture()
    motion.angular_velocity_world_rad_s[:] = torch.tensor([0., 0., 1.])
    for view in views:
        view.points[:int(view.counts.sum()), 0] = 1.
    first = read(contacts, motion).wheel_motion
    delta = torch.tensor([1000., -500., 20.])
    motion.com_position_world_m[:] += delta
    for view in views:
        view.points += delta
    shifted = read(contacts, motion).wheel_motion
    torch.testing.assert_close(first.tangent_speed_rms_m_s, shifted.tangent_speed_rms_m_s)


def test_sparse_wheel_ranges_do_not_clip_a_valid_pair_to_the_average_capacity_setting():
    binding, contacts, views, motion = fixture()
    wheel = binding.wheel_body_indices[0]
    for view in views:
        view.counts.zero_()
        view.starts.fill_(-1)
        view.force.fill_(float('nan'))
    view = views[0]  # The one-surface group owns native environment row1.
    view.counts[wheel, 0], view.starts[wheel, 0] = 70, 0
    weight, position = torch.arange(1., 71.), torch.linspace(0., 1., 70)
    view.force[:70, 0] = weight
    view.points[:70] = 0.
    view.points[:70, 0] = position
    view.normals[:70] = torch.tensor([0., 0., 1.])
    motion.angular_velocity_world_rad_s[:, wheel, 2] = 1.
    result = read(contacts, motion).wheel_motion
    expected = ((weight.double() * position.double().square()).sum() / weight.double().sum()).sqrt().float()
    torch.testing.assert_close(result.tangent_speed_rms_m_s[1, 0], expected)
    assert result.loaded_patch_count[1, 0] == 70
    assert not result.loaded[[0, 2]].any() and not result.loaded[:, 1].any()


@pytest.mark.parametrize('fault', ['unknown_terrain', 'body_order', 'future_time', 'wrong_tick', 'bad_position',
                                  'missing_points', 'point_nan', 'bad_normal', 'float_overflow'])
def test_missing_stale_or_invalid_loaded_data_is_rejected_instead_of_substituted(fault):
    binding, contacts, views, motion = fixture()
    if fault == 'unknown_terrain':
        contacts.terrain_motion_reference = None
    elif fault == 'body_order':
        motion = replace(motion, body_names=tuple(reversed(binding.native_body_names)))
    elif fault == 'future_time':
        motion = replace(motion, timestamp_s=motion.timestamp_s + .005)
    elif fault == 'wrong_tick':
        motion = replace(motion, physical_tick=2)
    elif fault == 'bad_position':
        motion.com_position_world_m[0, 0, 0] = float('nan')
    elif fault == 'missing_points':
        views[0].points = None
    else:
        wheel = binding.wheel_body_indices[0]
        index = views[0].starts[wheel].max().item()
        if fault == 'point_nan':
            views[0].points[index, 0] = float('nan')
        elif fault == 'bad_normal':
            views[0].normals[index] = 0.
        else:
            motion.angular_velocity_world_rad_s[:, wheel, 2] = 3e38
            views[0].points[index, 0] = 3e38
    with pytest.raises(ValueError):
        contacts.read(.005, torch.full((3,), .005, dtype=torch.float64), body_motion=motion)


def test_body_motion_uses_rotated_final_local_com_and_world_com_velocity_with_owned_buffers():
    robot, bridge, _ = native_fixture()
    with pytest.raises(RuntimeError, match='refresh'):
        bridge.body_motion()
    wheel = bridge.binding.wheel_body_indices[0]
    robot.root_view.local_com[:, wheel, :3] = torch.tensor([.1, 0., 0.])
    bridge.invalidate_model()
    bridge.refresh_model()
    half = 2.**-.5
    robot.data.body_link_pose_w.torch[:, wheel, 3:] = torch.tensor([0., 0., half, half])
    robot.data.body_com_vel_w.torch[:, wheel] = torch.tensor([1., 2., 3., 4., 5., 6.])
    bridge.after_physics_step(1)
    result = bridge.body_motion()
    expected = robot.data.body_link_pose_w.torch[:, wheel, :3] + torch.tensor([0., .1, 0.])
    torch.testing.assert_close(result.com_position_world_m[:, wheel], expected)
    assert result.com_velocity_world_m_s[:, wheel].tolist() == [[1., 2., 3.]] * 2
    assert result.angular_velocity_world_rad_s[:, wheel].tolist() == [[4., 5., 6.]] * 2
    robot.data.body_com_vel_w.torch.zero_()
    robot.root_view.local_com.zero_()
    assert result.com_velocity_world_m_s[:, wheel].eq(torch.tensor([1., 2., 3.])).all()
    assert result.model_epoch == bridge.model_epoch == 2 and result.physical_tick == 1


@pytest.mark.parametrize('fault', ['non_root_COM_nan', 'body_pose_shape', 'unnormalized', 'wrong_root_pose',
                                  'wrong_root_velocity', 'model_invalidated', 'reset_invalidated'])
def test_body_motion_requires_all_model_readbacks_and_fresh_consistent_native_buffers(fault):
    robot, bridge, _ = native_fixture()
    if fault == 'non_root_COM_nan':
        bridge.invalidate_model()
        robot.root_view.local_com[0, 1, 0] = float('nan')
        with pytest.raises(ValueError, match='every named body'):
            bridge.refresh_model()
        return
    bridge.after_physics_step(1)
    if fault == 'body_pose_shape':
        robot.data.body_link_pose_w.torch = torch.zeros(2, 18, 7)
    elif fault == 'unnormalized':
        robot.data.body_link_pose_w.torch[0, 1, 6] = 2
    elif fault == 'wrong_root_pose':
        robot.data.body_link_pose_w.torch[0, 0, 0] += .1
    elif fault == 'wrong_root_velocity':
        robot.data.body_com_vel_w.torch[0, 0, 0] += .1
    elif fault == 'model_invalidated':
        bridge.invalidate_model()
    else:
        bridge.invalidate_state()
    with pytest.raises((ValueError, RuntimeError)):
        bridge.body_motion()


@pytest.mark.parametrize('fault', ['static', 'rigid_ancestor', 'animated_ancestor', 'missing_mesh'])
def test_static_terrain_source_binding_checks_collider_ancestry(fault):
    sentinel = object()

    class Prim:
        def __init__(self, path, parent=None, rigid=False, times=0, valid=True):
            self.path, self.parent, self.rigid, self.times, self.valid = path, parent, rigid, times, valid

        def IsValid(self):
            return self.valid

        def GetPath(self):
            return self.path

        def HasAPI(self, api):
            assert api is sentinel
            return self.rigid

        def GetAttributes(self):
            return [SimpleNamespace(GetNumTimeSamples=lambda: self.times)]

        def GetParent(self):
            return self.parent

    invalid = Prim('', valid=False)
    root = Prim('/World', invalid, rigid=fault == 'rigid_ancestor', times=int(fault == 'animated_ancestor'))
    mesh = Prim('/World/mesh', root, valid=fault != 'missing_mesh')
    stage = SimpleNamespace(GetPrimAtPath=lambda path: mesh)
    if fault == 'static':
        result = NativeSceneRecipe._static_terrain_readback(stage, (('/World/mesh',),), sentinel)
        assert result['terrain_motion_reference'] == 'authored_static_own_terrain'
        assert result['static_terrain_ancestry_prims_checked'] == 2
        assert not result['native_static_collision_equivalence_verified']
    else:
        with pytest.raises(RuntimeError):
            NativeSceneRecipe._static_terrain_readback(stage, (('/World/mesh',),), sentinel)

"""Synthetic native-buffer lifecycle tests, without an SDK import or context."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_physical_observation import fixture
from transformer_rl._native_observation import NativeObservationBridge, SurfaceSample


class NativeView:
    def __init__(self, local_com):
        self.local_com = local_com
        self.reads = 0

    def get_coms(self):
        self.reads += 1
        return self.local_com


class NativeRobot:
    def __init__(self, binding, sample):
        self.joint_names = binding.native_joint_names
        self.body_names = ('base_link', *(name for name in binding.native_body_names if name != 'base_link'))
        count = len(sample.timestamp_s)
        local_com = torch.zeros(count, 19, 7)
        local_com[:, :, 6] = 1
        local_com[:, 0, :3] = sample.root_com_position_world_m - sample.root_link_position_world_m
        self.root_view = NativeView(local_com)
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=sample.joint_position_rad.clone()),
            joint_vel=SimpleNamespace(torch=sample.joint_velocity_rad_s.clone()),
            root_link_pose_w=SimpleNamespace(torch=torch.cat((sample.root_link_position_world_m,
                                                             sample.root_link_orientation_xyzw), -1)),
            root_com_vel_w=SimpleNamespace(torch=torch.cat((sample.root_com_velocity_world_m_s,
                                                           sample.angular_velocity_world_rad_s), -1)),
        )
        self.updates, self.fail_update = [], False

    def update(self, dt):
        self.updates.append(dt)
        if self.fail_update:
            raise RuntimeError('synthetic refresh failure')


def native_fixture(period=.005):
    manifest, binding, _, sample, _ = fixture()
    robot = NativeRobot(binding, sample)
    bridge = NativeObservationBridge(robot, manifest, 2, native_to_tensor=lambda value: value,
                                     physics_dt_s=period)
    order = [binding.native_body_names.index(name) for name in robot.body_names]
    surface = SurfaceSample(sample.terrain_contact_force_world_n[:, order], sample.ground_at_root_world_z_m,
                             sample.ground_at_wheel_world_z_m, sample.environment_origin_world_m,
                             sample.wheel_friction, torch.full((2,), period, dtype=torch.float64), robot.body_names)
    return robot, bridge, surface


@pytest.mark.parametrize('period', [.005, .001])
def test_bridge_reads_the_bound_native_order_after_one_reported_physical_refresh(period):
    robot, bridge, surface = native_fixture(period)
    with pytest.raises(RuntimeError, match='refresh'):
        bridge.measure(surface)
    bridge.after_physics_step(1)
    measured = bridge.measure(surface)
    torch.testing.assert_close(measured.velocity_origin_body_m_s, torch.tensor([[.4, -.3, .2]]).repeat(2, 1))
    torch.testing.assert_close(measured.wheel_ground_force_n, torch.tensor([[10., 20.]]).repeat(2, 1))
    assert robot.updates == [period] and bridge.refresh_attempts == 1
    again = bridge.measure(surface)
    assert robot.updates == [period]
    torch.testing.assert_close(measured.velocity_origin_body_m_s, again.velocity_origin_body_m_s)
    torch.testing.assert_close(measured.timestamp_s, torch.full((2,), period, dtype=torch.float64))


def test_mechanical_writes_require_final_native_com_readback_and_a_new_physical_refresh():
    robot, bridge, surface = native_fixture()
    bridge.after_physics_step(1)
    bridge.measure(surface)
    bridge.invalidate_model()
    robot.root_view.local_com[:, 0, 0] += .1
    with pytest.raises(RuntimeError, match='refresh'):
        bridge.measure(surface)
    with pytest.raises(ValueError, match='readback'):
        bridge.after_physics_step(2)
    bridge.refresh_model()
    assert robot.root_view.reads == 2 and bridge.model_epoch == 2
    with pytest.raises(RuntimeError, match='refresh'):
        bridge.measure(surface)
    omega = robot.data.root_com_vel_w.torch[:, 3:]
    velocity = torch.tensor([[.4, -.3, .2]]).repeat(2, 1)
    robot.data.root_com_vel_w.torch[:, :3] = velocity + torch.linalg.cross(omega, robot.root_view.local_com[:, 0, :3])
    bridge.after_physics_step(2)
    result = bridge.measure(replace(surface, timestamp_s=torch.full((2,), .01, dtype=torch.float64)))
    torch.testing.assert_close(result.velocity_origin_body_m_s, velocity)


def test_root_com_snapshot_and_returned_measurements_do_not_alias_native_reuse_buffers():
    robot, bridge, surface = native_fixture()
    robot.root_view.local_com[:, 0, 0] += .1
    # Simulates reuse of a native getter buffer. Real mechanical setters must
    # additionally use invalidate_model; this checks snapshot ownership only.
    bridge.after_physics_step(1)
    result = bridge.measure(surface)
    expected = result.velocity_origin_body_m_s.clone()
    robot.data.root_com_vel_w.torch.zero_()
    robot.data.joint_vel.torch.zero_()
    surface.terrain_contact_force_world_n.zero_()
    torch.testing.assert_close(result.velocity_origin_body_m_s, expected)
    torch.testing.assert_close(expected, torch.tensor([[.4, -.3, .2]]).repeat(2, 1))
    assert result.policy_velocity_rad_s.any() and result.wheel_ground_force_n.any()


def test_reset_state_writes_do_not_reuse_an_old_sample_or_reset_the_monotonic_tick():
    robot, bridge, surface = native_fixture()
    bridge.after_physics_step(1)
    bridge.invalidate_state()
    with pytest.raises(RuntimeError):
        bridge.measure(surface)
    bridge.after_physics_step(2)
    with pytest.raises(ValueError, match='time'):
        bridge.measure(surface)
    result = bridge.measure(replace(surface, timestamp_s=torch.full((2,), .01, dtype=torch.float64)))
    assert bridge.physical_tick == 2 and robot.updates == [.005, .005]
    torch.testing.assert_close(result.timestamp_s, torch.full((2,), .01, dtype=torch.float64))


@pytest.mark.parametrize('change', ['view', 'joint_order', 'body_order'])
def test_view_or_native_identity_changes_require_a_new_bridge(change):
    robot, bridge, surface = native_fixture()
    bridge.after_physics_step(1)
    if change == 'view':
        robot.root_view = NativeView(robot.root_view.local_com.clone())
    elif change == 'joint_order':
        robot.joint_names = tuple(reversed(robot.joint_names))
    else:
        robot.body_names = tuple(reversed(robot.body_names))
    with pytest.raises(RuntimeError, match='identity'):
        bridge.measure(surface)


def test_failed_native_update_is_not_counted_as_a_successful_tick_or_automatically_retried():
    robot, bridge, _ = native_fixture()
    robot.fail_update = True
    with pytest.raises(RuntimeError, match='synthetic'):
        bridge.after_physics_step(1)
    assert bridge.physical_tick == 0 and bridge.refresh_attempts == 1
    robot.fail_update = False
    with pytest.raises(RuntimeError, match='new bridge'):
        bridge.after_physics_step(1)
    assert len(robot.updates) == 1


def test_repeated_physical_tick_or_wrong_surface_order_is_rejected_without_another_native_update():
    robot, bridge, surface = native_fixture()
    bridge.after_physics_step(1)
    with pytest.raises(ValueError):
        bridge.after_physics_step(1)
    assert len(robot.updates) == 1
    with pytest.raises(ValueError, match='identity'):
        bridge.measure(replace(surface, body_names=tuple(reversed(surface.body_names))))

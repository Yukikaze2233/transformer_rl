"""Coupled control/physics API ordering fixtures; no real simulator is run."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from test_native_observation import NativeRobot
from test_output_actuation import parameters
from test_passive_pressure import fixture as pressure_fixture
from test_physical_observation import fixture as physical_fixture
from transformer_rl._native_actuation import NativeEffortWriter
from transformer_rl._native_contacts import NativeTerrainContacts
from transformer_rl._native_observation import NativeObservationBridge
from transformer_rl._native_plant import GroundReference, NativePhysicsPlant
from transformer_rl._native_scene import NativeSceneState
from transformer_rl._output_actuation import EffectiveDriveResponse, OutputFeedbackController


def fixture(period=.005):
    trace = []
    manifest, binding, _, sample, _ = physical_fixture()
    robot = NativeRobot(binding, sample)
    robot.data.joint_vel.torch.zero_()
    view = robot.root_view
    view.get_dof_positions = lambda: robot.data.joint_pos.torch
    view.get_dof_velocities = lambda: robot.data.joint_vel.torch
    robot.data.applied_torque = SimpleNamespace(torch=torch.zeros(2, 18))

    def set_target(value):
        trace.append('set_target')
        robot.target = value.clone()

    def write():
        trace.append('write')
        robot.data.applied_torque.torch.copy_(robot.target)

    def update(dt):
        trace.append('refresh')
        robot.updates.append(dt)

    robot.set_joint_effort_target, robot.write_data_to_sim, robot.update = set_target, write, update
    spec = {'joints': [{'name': name, 'type': 'prismatic' if 'spring_slide' in name else 'continuous'}
                       for name in robot.joint_names]}
    bridge = NativeObservationBridge(robot, manifest, 2, native_to_tensor=lambda value: value, physics_dt_s=period)
    writer = NativeEffortWriter(robot, bridge.binding, spec, 2)

    def create(sensors, *, filter_patterns, max_contact_data_count):
        n = len(sensors)
        matrix = torch.zeros(n, 2, 3)
        matrix[:, 0, 2], matrix[:, 1, 2] = 10., -10.
        force = torch.full((max_contact_data_count, 1), float('nan'))
        force[:n] = 20
        counts = torch.zeros(n, 2, dtype=torch.int32)
        counts[:, 0] = 1
        starts = torch.full_like(counts, -1)
        starts[:, 0] = torch.arange(n)
        points = torch.full((max_contact_data_count, 3), float('nan'))
        normals = torch.full_like(points, float('nan'))
        points[:n], normals[:n] = 0., torch.tensor([0., 0., 1.])
        view = SimpleNamespace(sensor_paths=sensors, filter_paths=filter_patterns, sensor_count=n,
                               filter_count=2, max_contact_data_count=max_contact_data_count,
                               get_contact_force_matrix=lambda *, dt: matrix,
                               get_contact_data=lambda *, dt: (force, points, normals, None, counts, starts))
        return view

    envs = ('/World/envs/env_0', '/World/envs/env_1')
    contacts = NativeTerrainContacts(SimpleNamespace(create_rigid_contact_view=create), bridge.binding, envs,
                                     tuple((p + '/Terrain/a', p + '/Terrain/b') for p in envs),
                                     native_to_tensor=lambda value: value,
                                     terrain_motion_reference='authored_static_own_terrain')
    read = contacts.read

    def traced_read(*args, **kwargs):
        trace.append('contacts')
        return read(*args, **kwargs)

    contacts.read = traced_read
    scene = NativeSceneState(robot, bridge, writer, contacts, envs, sample.environment_origin_world_m.clone(),
                             ((), ()), (0, 1), {'physics_qualified': False})
    control = OutputFeedbackController(bridge.binding.nominal_policy_position_rad, 2, pair_domain=None)
    drive = EffectiveDriveResponse(parameters(), 2)
    pressure = pressure_fixture(2)[-1]
    sim = SimpleNamespace(cfg=SimpleNamespace(dt=period), steps=0, fail=False)

    def step(*, render):
        trace.append('step')
        if sim.fail:
            raise RuntimeError('synthetic simulation failure')
        assert render is False
        sim.steps += 1
        robot.data.root_link_pose_w.torch[:, 2] += .001
        poses = robot.data.body_link_pose_w.torch
        poses[:, :, :3] += (robot.data.root_link_pose_w.torch[:, :3] - poses[:, 0, :3])[:, None]
        poses[:, :, 3:7] = robot.data.root_link_pose_w.torch[:, None, 3:7]
        robot.data.body_com_vel_w.torch[:, 0] = robot.data.root_com_vel_w.torch

    sim.step = step

    def ground(*_):
        trace.append('ground')
        return GroundReference(torch.full((2,), 2.), torch.full((2, 2), 2.), torch.full((2, 2), .8))

    plant = NativePhysicsPlant(sim, scene, control, drive, pressure, native_to_tensor=lambda value: value,
                                ground_query=ground, timing_path='effective_response_probe_no_extra_transport')
    return plant, robot, sim, trace


@pytest.mark.parametrize('period', [.005, .001])
def test_one_real_step_precedes_native_refresh_and_same_tick_contact_read(period):
    plant, robot, sim, trace = fixture(period)
    actions = torch.tensor([[.2, .3, -.2, -.3, .4, -.5]]).repeat(2, 1)
    plant.accept_issued_action(actions)
    result = plant.step_physics()
    assert trace == ['set_target', 'write', 'step', 'refresh', 'contacts', 'ground']
    assert sim.steps == result.physical_tick == plant.scene.observation.physical_tick == 1
    assert robot.updates == [period]
    torch.testing.assert_close(result.measured.timestamp_s, torch.full((2,), period, dtype=torch.float64))
    torch.testing.assert_close(result.feedback.controller_issued_at_s, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(result.feedback.pd_computed_at_s, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(result.drive.impulse, period * result.drive.mean_effort.double(), atol=1e-8, rtol=1e-6)
    assert result.drive.endpoint_drive_effort[:, :4].abs().gt(result.drive.mean_effort[:, :4].abs()).all()
    assert result.submitted.commanded_by_native_joint[:, plant._springs].gt(0).all()
    assert not result.contact_labels_valid.any()


def test_contact_labels_use_positive_patch_load_even_when_critic_force_vector_cancels():
    plant, _, sim, _ = fixture()
    plant.accept_issued_action(torch.zeros(2, 6))
    for index in range(3):
        result = plant.step_physics()
        assert not result.measured.wheel_ground_force_n.any()
        assert result.contacts.normal_load_n.gt(0).all()
        assert result.contact_labels_valid.all().item() == (index == 2)
    labels, valid = plant.contact_history.contacts()
    assert labels.all() and valid.all() and sim.steps == 3
    before = plant.contact_history.history.clone()
    plant.contact_history.contacts()
    torch.testing.assert_close(plant.contact_history.history, before)
    assert sim.steps == 3


def test_actual_step_publishes_owned_contact_motion_with_the_same_physical_and_model_identity():
    plant, robot, sim, _ = fixture()
    plant.accept_issued_action(torch.zeros(2, 6))
    result = plant.step_physics()
    motion = result.contacts.wheel_motion
    assert motion is not None and motion.loaded.all()
    assert motion.physical_tick == result.physical_tick == sim.steps == 1
    assert motion.model_epoch == plant.scene.observation.model_epoch
    assert torch.equal(motion.timestamp_s, result.measured.timestamp_s)
    assert motion.tangent_speed_rms_m_s.gt(0).all()
    before = motion.tangent_speed_rms_m_s.clone()
    robot.data.body_com_vel_w.torch.zero_()
    torch.testing.assert_close(motion.tangent_speed_rms_m_s, before)


def test_missing_loaded_contact_points_faults_the_advanced_step_and_cannot_be_replayed():
    plant, _, sim, _ = fixture()
    plant.accept_issued_action(torch.zeros(2, 6))
    view = plant.scene.contacts.groups[0].view
    force, _, _, separation, counts, starts = view.get_contact_data(dt=.005)
    view.get_contact_data = lambda *, dt: (force, None, None, separation, counts, starts)
    with pytest.raises(ValueError, match='patch points'):
        plant.step_physics()
    assert sim.steps == plant.scene.observation.physical_tick == 1
    with pytest.raises(RuntimeError, match='faulted'):
        plant.step_physics()
    assert sim.steps == 1


def test_changed_static_terrain_motion_binding_is_rejected_before_another_physics_step():
    plant, _, sim, _ = fixture()
    plant.accept_issued_action(torch.zeros(2, 6))
    plant.scene.contacts.terrain_motion_reference = None
    with pytest.raises(RuntimeError, match='terrain-motion binding'):
        plant.step_physics()
    assert sim.steps == 0


def test_feedback_is_recomputed_from_fresh_output_positions_without_extra_physics():
    plant, robot, sim, _ = fixture()
    plant.accept_issued_action(torch.full((2, 6), .2))
    first = plant.step_physics()
    robot.data.joint_pos.torch[:, plant._active[0]] += .01
    second = plant.step_physics()
    torch.testing.assert_close(second.feedback.requested_effort[:, 0], first.feedback.requested_effort[:, 0] - 1.6,
                               rtol=1e-5, atol=1e-5)
    assert sim.steps == 2 and robot.updates == [.005, .005]


def test_reset_writes_only_selected_rows_without_a_fake_physics_refresh_or_budget_transition():
    plant, robot, sim, trace = fixture()
    plant.accept_issued_action(torch.full((2, 6), .2))
    for _ in range(3):
        plant.step_physics()
    before = {f.name: getattr(plant.targets, f.name).clone() for f in fields(plant.targets)}
    retained = plant.drive.state[1].clone()
    calls = []

    def setter(key, tensor, ids):
        calls.append((key, ids.clone()))
        getattr(robot.data, key).torch[ids] = tensor

    robot.write_root_link_pose_to_sim_index = lambda *, root_pose, env_ids: setter('root_link_pose_w', root_pose, env_ids)
    robot.write_root_com_velocity_to_sim_index = lambda *, root_velocity, env_ids: setter('root_com_vel_w', root_velocity, env_ids)
    robot.write_joint_position_to_sim_index = lambda *, position, env_ids: setter('joint_pos', position, env_ids)
    robot.write_joint_velocity_to_sim_index = lambda *, velocity, env_ids: setter('joint_vel', velocity, env_ids)
    robot.reset = lambda ids: calls.append(('reset', ids.clone()))
    mask = torch.tensor([True, False])
    pose, vel = robot.data.root_link_pose_w.torch.clone(), robot.data.root_com_vel_w.torch.clone()
    q, dq = robot.data.joint_pos.torch.clone(), robot.data.joint_vel.torch.clone()
    trace.clear()
    plant.reset(mask, pose, vel, q, dq)
    assert trace == [] and sim.steps == 3 and len(robot.updates) == 3
    assert len(calls) == 5 and all(ids.tolist() == [0] for _, ids in calls)
    assert plant.scene.observation.physical_tick == 3
    assert plant.contact_history.count.tolist() == [0, 3]
    assert not plant.drive.state[0].any()
    torch.testing.assert_close(plant.drive.state[1], retained)
    for f in fields(plant.targets):
        torch.testing.assert_close(getattr(plant.targets, f.name)[1], before[f.name][1])
    assert not plant.targets.issued_action[0].any()
    for index in range(3):
        result = plant.step_physics()
        assert result.contact_labels_valid.tolist() == [index == 2, True]
    assert sim.steps == 6


def test_a_partial_step_faults_the_plant_and_cannot_be_replayed():
    plant, _, sim, trace = fixture()
    plant.accept_issued_action(torch.ones(2, 6))
    sim.fail = True
    with pytest.raises(RuntimeError, match='simulation failure'):
        plant.step_physics()
    assert trace == ['set_target', 'write', 'step'] and plant.scene.observation.physical_tick == 0
    with pytest.raises(RuntimeError, match='faulted'):
        plant.step_physics()
    with pytest.raises(RuntimeError, match='faulted'):
        plant.accept_issued_action(torch.zeros(2, 6))
    assert sim.steps == 0


def test_missing_model_refresh_is_rejected_before_any_effort_write_or_physics():
    plant, _, sim, trace = fixture()
    plant.accept_issued_action(torch.zeros(2, 6))
    plant.scene.observation.invalidate_model()
    with pytest.raises(RuntimeError, match='model readback'):
        plant.step_physics()
    assert not trace and sim.steps == 0


def test_no_physical_step_can_run_without_an_explicit_target():
    plant, _, sim, trace = fixture()
    with pytest.raises(RuntimeError, match='target'):
        plant.step_physics()
    assert not trace and sim.steps == 0


def test_changing_the_native_view_rejects_actions_and_physical_steps():
    plant, robot, sim, trace = fixture()
    robot.root_view = object()
    with pytest.raises(RuntimeError, match='owner'):
        plant.accept_issued_action(torch.zeros(2, 6))
    assert not trace and sim.steps == 0


def test_probe_does_not_allow_an_undeclared_or_duplicate_transport_path():
    plant, _, sim, _ = fixture()
    with pytest.raises(ValueError, match='timing path'):
        NativePhysicsPlant(sim, plant.scene, plant.controller, plant.drive, plant.pressure,
                           native_to_tensor=lambda x: x, ground_query=lambda *_: None,
                           timing_path='add_usb_rtt_and_can_to_effective_tau')

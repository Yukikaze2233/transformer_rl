"""Synthetic physical-reference checks; these do not launch or certify an SDK."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from transformer_rl._physical_observation import (
    CommandFeatures, CommandMode, MechanicalBinding, NativePhysicalSample,
    ObservationAssembler, WheelContactHistory,
)
from transformer_rl._structured_collection import ACTION_ORDER


TREE = ('L_joint1', 'R_joint1', 'LL_joint1', 'RR_joint1', 'L_joint2', 'R_jonit2',
        'LL_joint2', 'RR_joint2', 'L_joint3', 'LLL_joint2', 'R_joint3', 'RRR_joint2',
        'LL_joint3', 'RR_joint3', 'L_spring_slide', 'R_spring_slide', 'LL_link4', 'RR_joint4')
BODIES = ('base_link', 'L_link1', 'L_link2', 'L_link3', 'LLL_link2', 'LLL_link1',
          'R_link1', 'R_link2', 'R_link3', 'RRR_link2', 'RRR_link1', 'LL_link1', 'LL_link2',
          'LL_link3', 'LL_link4', 'RR_link1', 'RR_link2', 'RR_link3', 'RR_link4')


def fixture(count=2):
    manifest = {'control_frame': 'Xforward_Yleft_Zup', 'tree_joint_names': list(TREE),
                'rigid_body_names': list(BODIES),
                'control_joint_names': ['L_joint1', 'LL_joint1', 'L_joint3',
                                        'R_joint1', 'RR_joint1', 'R_joint3'],
                'nominal_joint_pos': {name: i / 100 for i, name in enumerate(TREE)}}
    binding = MechanicalBinding.from_manifest(manifest, tuple(reversed(TREE)), tuple(reversed(BODIES)))
    assembler = ObservationAssembler(binding, count)
    position = torch.tensor([[manifest['nominal_joint_pos'][name]
                              for name in binding.native_joint_names]]).repeat(count, 1)
    velocity = torch.arange(1, 19, dtype=torch.float32)[None].repeat(count, 1)
    root = torch.tensor([[.1, -.2, 2.305]]).repeat(count, 1)
    com = root + torch.tensor([[.02, -.01, -.05]])
    omega = torch.tensor([[1., 2., 3.]]).repeat(count, 1)
    origin_velocity = torch.tensor([[.4, -.3, .2]]).repeat(count, 1)
    com_velocity = origin_velocity + torch.linalg.cross(omega, com - root)
    forces = torch.zeros(count, 19, 3)
    forces[:, binding.native_body_names.index('base_link'), 2] = 100
    forces[:, binding.wheel_body_indices[0], 2] = 10
    forces[:, binding.wheel_body_indices[1], 2] = 20
    sample = NativePhysicalSample(
        position, velocity, root, torch.tensor([[0., 0., 0., 1.]]).repeat(count, 1), com,
        com_velocity, omega, forces, torch.full((count,), 2.),
        torch.tensor([[2.1, 2.2]]).repeat(count, 1), torch.tensor([[0., 0., 2.]]).repeat(count, 1),
        torch.tensor([[.7, .8]]).repeat(count, 1), torch.full((count,), 1., dtype=torch.float64),
    )
    command = CommandFeatures(torch.tensor([[.5, 0., .2, .305]]).repeat(count, 1),
                              torch.zeros(count, dtype=torch.long), torch.zeros(count),
                              torch.full((count,), .3),
                              torch.tensor([[.1, .2, .3, .4, .5, .6]]).repeat(count, 1))
    return manifest, binding, assembler, sample, command


def test_native_orders_are_resolved_by_name_not_by_control_array_position():
    manifest, binding, assembler, sample, _ = fixture()
    measured = assembler.measure(sample)
    assert tuple(binding.native_joint_names[index] for index in binding.policy_joint_indices) == ACTION_ORDER
    assert tuple(binding.native_joint_names[index] for index in binding.tree_joint_indices) == TREE
    assert manifest['control_joint_names'] != list(ACTION_ORDER)
    expected = torch.tensor([[18. - TREE.index(name) for name in ACTION_ORDER]]).repeat(2, 1)
    torch.testing.assert_close(measured.policy_velocity_rad_s, expected)
    torch.testing.assert_close(measured.wheel_ground_force_n, torch.tensor([[10., 20.]]).repeat(2, 1))


@pytest.mark.parametrize('change', ['joint_missing', 'joint_duplicate', 'body_missing', 'body_duplicate',
                                    'active_duplicate', 'control_frame', 'nominal_nan'])
def test_incomplete_or_ambiguous_asset_binding_is_rejected(change):
    manifest, binding, *_ = fixture()
    manifest = deepcopy(manifest)
    joints, bodies = list(binding.native_joint_names), list(binding.native_body_names)
    if change == 'joint_missing':
        joints.pop()
    elif change == 'joint_duplicate':
        joints[0] = joints[1]
    elif change == 'body_missing':
        bodies.pop()
    elif change == 'body_duplicate':
        bodies[0] = bodies[1]
    elif change == 'active_duplicate':
        manifest['control_joint_names'][0] = manifest['control_joint_names'][1]
    elif change == 'control_frame':
        manifest['control_frame'] = 'Yforward_Xright_Zup'
    else:
        manifest['nominal_joint_pos']['L_joint1'] = float('nan')
    with pytest.raises(ValueError):
        MechanicalBinding.from_manifest(manifest, joints, bodies)


def test_rotating_body_com_velocity_is_converted_to_link_origin_before_body_rotation():
    _, _, assembler, sample, _ = fixture()
    measured = assembler.measure(sample)
    expected = torch.tensor([[.4, -.3, .2]]).repeat(2, 1)
    assert not torch.allclose(sample.root_com_velocity_world_m_s, expected)
    torch.testing.assert_close(measured.velocity_origin_body_m_s, expected)
    quaternion = torch.tensor([[0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]]).repeat(2, 1)
    rotated = assembler.measure(replace(sample, root_link_orientation_xyzw=quaternion))
    torch.testing.assert_close(rotated.velocity_origin_body_m_s,
                               torch.tensor([[-.3, -.4, .2]]).repeat(2, 1))
    torch.testing.assert_close(rotated.angular_velocity_body_rad_s,
                               torch.tensor([[2., -1., 3.]]).repeat(2, 1))
    torch.testing.assert_close(rotated.gravity_body, torch.tensor([[0., 0., -1.]]).repeat(2, 1))
    negated = assembler.measure(replace(sample, root_link_orientation_xyzw=-quaternion))
    torch.testing.assert_close(rotated.velocity_origin_body_m_s, negated.velocity_origin_body_m_s)


def test_height_is_at_the_link_origin_ground_datum_not_world_z_or_mean_wheel_support():
    _, _, assembler, sample, _ = fixture()
    measured = assembler.measure(sample)
    torch.testing.assert_close(measured.height_above_ground_m, torch.full((2,), .305))
    torch.testing.assert_close(measured.wheel_support_above_environment_m,
                               torch.tensor([[.1, .2]]).repeat(2, 1))


def test_actor_encodes_exactly35_fields_without_privileged_inputs_or_wheel_angles():
    _, binding, assembler, sample, command = fixture()
    measured = assembler.measure(sample)
    frame = assembler.actor(assembler.clean_sensors(measured), command, now_s=sample.timestamp_s)
    assert frame.shape == (2, 35)
    torch.testing.assert_close(frame[:, :4], torch.tensor([[.5, 0., .2, 1.525]]).repeat(2, 1))
    torch.testing.assert_close(frame[:, 4:7], torch.tensor([[.5, 1., 1.5]]).repeat(2, 1))
    torch.testing.assert_close(frame[:, 7:10], torch.tensor([[0., 0., -1.]]).repeat(2, 1))
    assert not frame[:, 10:16].any()
    torch.testing.assert_close(frame[:, 16:22], measured.policy_velocity_rad_s * .1)
    torch.testing.assert_close(frame[:, 22:28], command.previous_issued_action)
    torch.testing.assert_close(frame[:, 28:33], torch.tensor([[1., 0., 0., 0., 0.]]).repeat(2, 1))
    assert not frame[:, 33].any()
    torch.testing.assert_close(frame[:, 34], torch.full((2,), .3))
    q = sample.joint_position_rad.clone()
    q[:, binding.native_joint_names.index('L_joint3')] = 9e6
    q[:, binding.native_joint_names.index('R_joint3')] = -8e6
    other = assembler.measure(replace(sample, joint_position_rad=q))
    other_frame = assembler.actor(assembler.clean_sensors(other), command, now_s=sample.timestamp_s)
    torch.testing.assert_close(frame, other_frame)


def test_leg_position_uses_wrapped_relative_coaxial_coordinates_without_a_differential_transform():
    _, binding, assembler, sample, command = fixture()
    q = sample.joint_position_rad.clone()
    for name in ACTION_ORDER[:4]:
        q[:, binding.native_joint_names.index(name)] += 2 * math.pi + .2
    state = assembler.measure(replace(sample, joint_position_rad=q))
    frame = assembler.actor(assembler.clean_sensors(state), command, now_s=sample.timestamp_s)
    torch.testing.assert_close(frame[:, 10:14], torch.full((2, 4), .2), atol=1e-6, rtol=0)


def test_critic81_has_explicit_blocks_and_rebuilds_its_own_clean_prefix():
    _, binding, assembler, sample, command = fixture()
    state = assembler.measure(sample)
    noisy = assembler.clean_sensors(state)
    noisy.angular_velocity_body_rad_s.add_(5)
    actor = assembler.actor(noisy, command, now_s=sample.timestamp_s)
    critic = assembler.critic(command, state)
    assert critic.shape == (2, 81)
    torch.testing.assert_close(critic[:, 4:7], sample.angular_velocity_world_rad_s * .5)
    assert not torch.equal(critic[:, :35], actor)
    expected_q = sample.joint_position_rad[:, list(binding.tree_joint_indices)].clone()
    expected_q[:, list(binding.wheel_tree_indices)] = 0
    expected = {
        'velocity_origin_body_m_s': torch.tensor([[.4, -.3, .2]]).repeat(2, 1),
        'height_above_ground_m': torch.full((2, 1), .305),
        'wheel_ground_force_div125': torch.tensor([[.08, .16]]).repeat(2, 1),
        'tree_position_rad': expected_q,
        'tree_velocity_times0p1': sample.joint_velocity_rad_s[:, list(binding.tree_joint_indices)] * .1,
        'wheel_friction': torch.tensor([[.7, .8]]).repeat(2, 1),
        'wheel_support_above_environment_m': torch.tensor([[.1, .2]]).repeat(2, 1),
    }
    for name, value in expected.items():
        torch.testing.assert_close(critic[:, assembler.CRITIC_SLICES[name]], value)
    with pytest.raises(TypeError):
        assembler.critic(actor, state)


@pytest.mark.parametrize('mode', list(CommandMode))
def test_five_modes_are_deployable_commands_and_jump_context_is_not_a_hidden_physical_phase(mode):
    _, _, assembler, sample, command = fixture()
    command = replace(command, mode=torch.full((2,), int(mode), dtype=torch.long),
                      request_height_m=torch.full((2,), .2 if mode == CommandMode.JUMP else 0.),
                      command_clock_s=torch.full((2,), 6.))
    state = assembler.measure(sample)
    frame = assembler.actor(assembler.clean_sensors(state), command, now_s=sample.timestamp_s)
    assert torch.equal(frame[:, 28:33].argmax(-1), command.mode)
    torch.testing.assert_close(frame[:, 33], torch.full((2,), 1. if mode == CommandMode.JUMP else 0.))
    torch.testing.assert_close(frame[:, 34], torch.full((2,), 5.))


def test_sensor_ages_remain_metadata_and_future_values_are_rejected():
    _, _, assembler, sample, command = fixture()
    sensors = assembler.clean_sensors(assembler.measure(sample))
    current = assembler.actor(sensors, command, now_s=sample.timestamp_s)
    past = replace(sensors, sampled_at_s=sensors.sampled_at_s - .005)
    torch.testing.assert_close(current, assembler.actor(past, command, now_s=sample.timestamp_s))
    future = replace(sensors, sampled_at_s=sensors.sampled_at_s + .001)
    with pytest.raises(ValueError, match='time'):
        assembler.actor(future, command, now_s=sample.timestamp_s)


@pytest.mark.parametrize('change', ['quaternion', 'position', 'friction', 'clock', 'force_overflow'])
def test_invalid_physical_inputs_are_rejected_instead_of_clipped_or_zero_filled(change):
    _, _, assembler, sample, _ = fixture()
    if change == 'quaternion':
        sample.root_link_orientation_xyzw.zero_()
    elif change == 'position':
        sample.root_link_position_world_m[0, 0] = float('nan')
    elif change == 'friction':
        sample.wheel_friction[0, 0] = -1
    elif change == 'clock':
        sample.timestamp_s[0] = -1
    else:
        sample.terrain_contact_force_world_n.fill_(1e38)
    with pytest.raises(ValueError):
        assembler.measure(sample)


@pytest.mark.parametrize('change', ['nan_sensor', 'issued_action', 'jump_height', 'lateral', 'scaling_overflow'])
def test_actor_rejects_invalid_values_before_scaling_and_clipping(change):
    _, _, assembler, sample, command = fixture()
    sensors = assembler.clean_sensors(assembler.measure(sample))
    if change == 'nan_sensor':
        sensors.gravity_body[0, 0] = float('nan')
    elif change == 'issued_action':
        command.previous_issued_action[0, 0] = 3.01
    elif change == 'jump_height':
        command.request_height_m[0] = .1
    elif change == 'lateral':
        command.admitted_body_command[0, 1] = .2
    else:
        command.admitted_body_command[0, 3] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError):
        assembler.actor(sensors, command, now_s=sample.timestamp_s)


def test_measurements_and_encoded_outputs_own_storage_before_native_buffers_are_reused():
    _, _, assembler, sample, command = fixture()
    measured = assembler.measure(sample)
    critic = assembler.critic(command, measured)
    expected = critic.clone()
    sample.joint_position_rad.fill_(100)
    sample.root_com_velocity_world_m_s.fill_(100)
    sample.terrain_contact_force_world_n.zero_()
    command.previous_issued_action.zero_()
    torch.testing.assert_close(critic, expected)
    torch.testing.assert_close(measured.velocity_origin_body_m_s, expected[:, 35:38])


def test_contact_history_needs_three_fresh_physical_samples_and_clears_only_reset_rows():
    history = WheelContactHistory(2)
    assert not history.contacts()[1].any()
    for tick, value in enumerate((3., 0., 0.), 1):
        history.push(torch.tensor([[value, 0.], [0., value]]), torch.full((2,), tick * .005, dtype=torch.float64))
    contact, valid = history.contacts()
    assert valid.all() and torch.equal(contact, torch.tensor([[True, False], [False, True]]))
    history.reset(torch.tensor([True, False]), torch.full((2,), .015, dtype=torch.float64))
    assert torch.equal(history.contacts()[1], torch.tensor([False, True]))
    assert not history.contacts()[0][0].any()
    for tick in range(4, 7):
        history.push(torch.zeros(2, 2), torch.full((2,), tick * .005, dtype=torch.float64))
    assert history.contacts()[1].all() and not history.contacts()[0].any()


@pytest.mark.parametrize('time', [.005, .004, .011, float('nan')])
def test_repeated_or_missing_physical_ticks_do_not_mutate_contact_history(time):
    history = WheelContactHistory(2)
    history.push(torch.ones(2, 2), torch.full((2,), .005, dtype=torch.float64))
    previous = (history.history.clone(), history.count.clone(), history.last_time_s.clone())
    with pytest.raises(ValueError):
        history.push(torch.full((2, 2), 5.), torch.full((2,), time, dtype=torch.float64))
    for actual, expected in zip((history.history, history.count, history.last_time_s), previous):
        torch.testing.assert_close(actual, expected)


def test_reference_and_candidate_contact_windows_keep_the_same15ms_physical_duration():
    for period, samples in ((.005, 3), (.001, 15)):
        history = WheelContactHistory(1, physics_dt_s=period)
        assert history.samples == samples
        for tick in range(1, samples + 1):
            history.push(torch.tensor([[3., 0.]]), torch.tensor([tick * period], dtype=torch.float64))
        contact, valid = history.contacts()
        assert valid.all() and contact[0, 0] and not contact[0, 1]
        for tick in range(samples + 1, 2 * samples + 1):
            history.push(torch.zeros(1, 2), torch.tensor([tick * period], dtype=torch.float64))
        assert history.contacts()[1].all() and not history.contacts()[0].any()


def test_contact_history_rejects_an_unvalidated_physics_clock():
    with pytest.raises(ValueError):
        WheelContactHistory(1, physics_dt_s=.003)


def test_named_physical_labels_match_the_collector_order_epoch_and_pre_reset_sample():
    _, _, assembler, sample, _ = fixture()
    state = assembler.measure(sample)
    history = WheelContactHistory(2)
    for time in (.99, .995, 1.):
        history.push(state.wheel_ground_force_n, torch.full((2,), time, dtype=torch.float64))
    episode, epoch = torch.tensor([1, 3]), torch.tensor([2, 4])
    labels = assembler.labels(state, history, episode, epoch)
    assert labels.valid.all() and labels.velocity_reference == 'base_link_origin_body'
    assert labels.joint_order == ACTION_ORDER and labels.wheel_contact.all()
    torch.testing.assert_close(labels.joint_velocity_rad_s, state.policy_velocity_rad_s)
    torch.testing.assert_close(labels.velocity_body_m_s, state.velocity_origin_body_m_s)
    state.policy_velocity_rad_s.zero_()
    episode.zero_()
    epoch.zero_()
    assert labels.joint_velocity_rad_s.any()
    assert torch.equal(labels.episode_id, torch.tensor([1, 3]))
    assert torch.equal(labels.control_epoch, torch.tensor([2, 4]))


def test_stale_contact_history_is_not_attached_to_a_fresh_physical_label():
    _, _, assembler, sample, _ = fixture()
    state = assembler.measure(sample)
    history = WheelContactHistory(2)
    for time in (.985, .99, .995):
        history.push(state.wheel_ground_force_n, torch.full((2,), time, dtype=torch.float64))
    with pytest.raises(ValueError, match='same-tick'):
        assembler.labels(state, history, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))

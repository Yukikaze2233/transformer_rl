"""Pressure-wrench conservation and virtual work, without native physics."""
from copy import deepcopy
import math

import pytest
import torch

from test_passive_pressure import fixture as pressure_fixture
from test_reset_kinematics import fixture as kinematic_fixture
from transformer_rl.formal.lab_env.spring_force_element import SpringForceElement


def fixture():
    kinematics, manifest, spec = kinematic_fixture()
    pressure_manifest, pressure_spec, fit, *_ = pressure_fixture()
    manifest.update(pressure_manifest)
    spec['spring_binding'] = pressure_spec['spring_binding']
    for side in ('L', 'R'):
        constraint = next(c for c in spec['constraints']
                          if c['body1'] == side * 3 + '_link1')
        constraint['name'] = side + '_spring_upper_mount'
        constraint['local_pos0_m'] = [.01, .02, 0.]
    names = kinematics.binding.native_body_names
    pose = torch.zeros(1, 19, 7, dtype=torch.float64)
    pose[..., 6] = 1.
    for side, lateral in (('L', .15), ('R', -.15)):
        pose[:, names.index(side + '_link1'), :3] = torch.tensor([0., lateral, .18])
        pose[:, names.index(side + '_link2'), :3] = torch.tensor([0., lateral, 0.])
    com = pose[..., :3] + torch.tensor([.012, -.013, .009], dtype=torch.float64)
    element = SpringForceElement(spec, manifest, fit, names, 'cpu', dtype=torch.float64)
    return element, pose, com, spec, manifest, fit, names


def potential(element, pose, com, fit):
    frame = element.evaluate(pose, com, torch.ones(len(pose), 2, dtype=torch.float64))
    u = frame.compression_m / fit['stroke_m']
    assert bool(((u > 0) & (u < 1)).all())
    a, b, c, d = fit['monomial_coefficients_n']
    return (fit['stroke_m'] * (a*u + b*u**2/2 + c*u**3/3 + d*u**4/4)).sum()


def test_pressure_pair_conserves_world_force_and_moment_with_offset_coms():
    element, pose, com, *_ = fixture()
    saved_pose, saved_com = pose.clone(), com.clone()
    frame = element.evaluate(pose, com, torch.ones(1, 2, dtype=torch.float64))
    assert frame.valid.all()
    torch.testing.assert_close(frame.forces_world.sum(1), torch.zeros(1, 3, dtype=torch.float64))
    arms = com[:, element.body_ids] - com[:, 0:1]
    moment = frame.torques_about_com_world + torch.linalg.cross(arms, frame.forces_world)
    torch.testing.assert_close(moment.sum(1), torch.zeros(1, 3, dtype=torch.float64),
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(pose, saved_pose, atol=0, rtol=0)
    torch.testing.assert_close(com, saved_com, atol=0, rtol=0)


def test_body_translation_and_rotation_work_match_the_pressure_potential():
    element, pose, com, _, _, fit, _ = fixture()
    frame = element.evaluate(pose, com, torch.ones(1, 2, dtype=torch.float64))
    body = element.body_ids[0]
    epsilon = 1e-6
    for axis in range(3):
        plus, minus = pose.clone(), pose.clone()
        plus[:, body, axis] += epsilon
        minus[:, body, axis] -= epsilon
        force = -(potential(element, plus, com, fit) - potential(element, minus, com, fit)) / (2*epsilon)
        torch.testing.assert_close(force, frame.forces_world[0, 0, axis], atol=2e-7, rtol=1e-7)

        direction = torch.eye(3, dtype=torch.float64)[axis]
        arm = pose[:, body, :3] - com[:, body]
        for target, angle in ((plus, epsilon), (minus, -epsilon)):
            rotated = (arm * math.cos(angle)
                       + torch.linalg.cross(direction.expand_as(arm), arm) * math.sin(angle)
                       + direction * (arm * direction).sum(-1, keepdim=True) * (1-math.cos(angle)))
            target[:, body, :3] = com[:, body] + rotated
            target[:, body, 3:6] = direction * math.sin(angle/2)
            target[:, body, 6] = math.cos(angle/2)
        torque = -(potential(element, plus, com, fit) - potential(element, minus, com, fit)) / (2*epsilon)
        torch.testing.assert_close(torque, frame.torques_about_com_world[0, 0, axis],
                                   atol=2e-7, rtol=1e-7)


def test_native_body_permutation_preserves_named_mount_wrenches():
    element, pose, com, spec, manifest, fit, names = fixture()
    strength = torch.tensor([[.9, 1.1]], dtype=torch.float64)
    original = element.evaluate(pose, com, strength)
    order = list(reversed(range(19)))
    permuted = SpringForceElement(spec, manifest, fit, [names[i] for i in order],
                                 'cpu', dtype=torch.float64)
    actual = permuted.evaluate(pose[:, order], com[:, order], strength)
    for result, expected in zip(actual, original):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_invalid_mount_geometry_is_explicit_and_does_not_invent_a_nominal_pose():
    element, pose, com, spec, manifest, fit, names = fixture()
    pose[:, element.body_ids[0], 3:] = 0.
    frame = element.evaluate(pose, com, torch.ones(1, 2, dtype=torch.float64))
    assert not frame.valid.any() and frame.forces_world.isnan().all()
    changed = deepcopy(spec)
    next(c for c in changed['constraints'] if c['name'] == 'L_spring_upper_mount')['body0'] = 'base_link'
    with pytest.raises(ValueError, match='mounts'):
        SpringForceElement(changed, manifest, fit, names, 'cpu', dtype=torch.float64)

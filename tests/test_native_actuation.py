"""Synthetic articulation effort API checks; no native simulation is launched."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_passive_pressure import fixture as pressure_fixture
from test_physical_observation import fixture as observation_fixture
from transformer_rl.formal.simulation.actuation import NativeEffortWriter


def fixture():
    _, binding, *_ = observation_fixture()
    spec = {'joints': [{'name': name, 'type': 'prismatic' if 'spring_slide' in name else 'continuous'}
                       for name in binding.native_joint_names]}
    submitted = []
    robot = SimpleNamespace(root_view=object(), joint_names=binding.native_joint_names,
                            body_names=binding.native_body_names,
                            data=SimpleNamespace(applied_torque=SimpleNamespace(
                                torch=torch.arange(18, dtype=torch.float32)[None].repeat(2, 1))),
                            set_joint_effort_target=lambda values: submitted.append(values))
    writer = NativeEffortWriter(robot, binding, spec, 2)
    *_, pressure_model = pressure_fixture(2)
    pressure = pressure_model.evaluate(torch.full((2, 2), .02))
    active = torch.tensor([[1., 2., 3., 4., .5, -.6]]).repeat(2, 1)
    return robot, binding, spec, submitted, writer, pressure_model, pressure, active


def test_writer_separates_six_rotary_nm_from_two_slider_n_and_keeps_passive_command_zero():
    _, binding, _, submitted, writer, _, pressure, active = fixture()
    receipt = writer.submit(active, pressure)
    assert writer.submissions == 1 and len(submitted) == 1
    expected = torch.zeros(2, 18)
    expected[:, list(binding.policy_joint_indices)] = active
    springs = [binding.native_joint_names.index(name) for name in ('L_spring_slide', 'R_spring_slide')]
    expected[:, springs] = pressure.slider_effort_n
    torch.testing.assert_close(submitted[0], expected)
    torch.testing.assert_close(receipt.commanded_by_native_joint, expected)
    assert all(receipt.native_joint_units[index] == 'Nm' for index in binding.policy_joint_indices)
    assert all(receipt.native_joint_units[index] == 'N' for index in springs)
    submitted[0].zero_()
    active.zero_()
    pressure.slider_effort_n.zero_()
    torch.testing.assert_close(receipt.commanded_by_native_joint, expected)


def test_explicit_native_actuator_buffer_is_named_and_owned_not_a_force_sensor_claim():
    robot, binding, _, _, writer, *_ = fixture()
    before = robot.data.applied_torque.torch.clone()
    actual = writer.explicit_actuator_buffer()
    torch.testing.assert_close(actual['active_output_effort_nm'], before[:, list(binding.policy_joint_indices)])
    springs = [binding.native_joint_names.index(name) for name in ('L_spring_slide', 'R_spring_slide')]
    torch.testing.assert_close(actual['slider_actuator_effort_n'], before[:, springs])
    robot.data.applied_torque.torch.zero_()
    assert actual['active_output_effort_nm'].any() and actual['slider_actuator_effort_n'].any()


def test_pressure_extrapolation_and_overtravel_are_published_without_being_relabelled_as_healthy():
    _, _, _, _, writer, pressure_model, _, active = fixture()
    pressure = pressure_model.evaluate(torch.full((2, 2), -.03))
    result = writer.submit(active, pressure)
    assert not result.pressure_within_fitted_curve.any()
    assert not result.pressure_within_mechanical_stroke.any()
    assert result.pressure_endpoint_clamped.all()


@pytest.mark.parametrize('bad', ['active_nan', 'active_over_cap', 'negative_pressure', 'pressure_over_cap', 'missing_flags'])
def test_bad_effort_or_missing_domain_evidence_is_rejected_before_native_submission(bad):
    _, _, _, submitted, writer, _, pressure, active = fixture()
    if bad == 'active_nan':
        active[0, 0] = float('nan')
    elif bad == 'active_over_cap':
        active[0, 4] = 4.51
    elif bad == 'negative_pressure':
        pressure.slider_effort_n[0, 0] = -1
    elif bad == 'pressure_over_cap':
        pressure.slider_effort_n[0, 0] = 1001
    else:
        pressure = replace(pressure, within_fitted_curve=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        writer.submit(active, pressure)
    assert writer.submissions == 0 and not submitted


def test_a_changed_native_owner_view_cannot_receive_an_effort():
    robot, _, _, submitted, writer, _, pressure, active = fixture()
    robot.root_view = object()
    with pytest.raises(RuntimeError):
        writer.submit(active, pressure)
    with pytest.raises(RuntimeError):
        writer.explicit_actuator_buffer()
    assert not submitted


def test_native_joint_units_must_match_the_model_not_an_array_position_assumption():
    robot, binding, spec, *_ = fixture()
    next(joint for joint in spec['joints'] if joint['name'] == 'L_spring_slide')['type'] = 'revolute'
    with pytest.raises(ValueError, match='prismatic'):
        NativeEffortWriter(robot, binding, spec, 2)

"""Artificial physical traces verify motion metrics without a simulator."""
import json
import math

import pytest
import torch

from transformer_rl.control_metrics import ControlMetrics


def packet(num_envs=1, *, time=0., reference=(0., 0., .3), actual=None,
           leg_target=None, wheel_target=None, effort=None, requested=None,
           velocity=None, position=(0., 0.), failure=False, success=False):
    def tensor(value, shape):
        return torch.as_tensor(value, dtype=torch.float64).expand((num_envs, *shape)).clone()
    actual = reference if actual is None else actual
    effort = (0.,) * 6 if effort is None else effort
    return {"time_s": tensor(time, ()), "command_reference": tensor(reference, (3,)),
            "actual": tensor(actual, (3,)), "position_xy": tensor(position, (2,)), "tilt": tensor(0., ()),
            "leg_target": tensor((0.,) * 4 if leg_target is None else leg_target, (4,)),
            "wheel_target": tensor((0.,) * 2 if wheel_target is None else wheel_target, (2,)),
            "motor_position": tensor((0.,) * 6, (6,)),
            "motor_velocity": tensor((0.,) * 6 if velocity is None else velocity, (6,)),
            "motor_effort": tensor(effort, (6,)),
            "requested_motor_effort": tensor(effort if requested is None else requested, (6,)),
            "effort_bounds": tensor(((-10., 10.),) * 6, (6, 2)),
            "failure": torch.as_tensor(failure, dtype=torch.bool).expand(num_envs).clone(),
            "success": torch.as_tensor(success, dtype=torch.bool).expand(num_envs).clone()}


def add(metrics, time, *, done=False, **kwargs):
    metrics.update(packet(metrics.num_envs, time=time, **kwargs),
                   torch.as_tensor(done, dtype=torch.bool).expand(metrics.num_envs).clone())


def test_false_survival_with_constant_height_bias_is_not_successful_tracking():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=2)
    for index in range(3):
        add(metrics, index * .01, actual=(0., 0., .1), done=index == 2, success=index == 2)
    report = metrics.report()
    full = report["full_interval"]["axes"]["height"]
    steady = report["steady"]["axes"]["height"]
    assert full["bias"] == pytest.approx(-.2)
    assert full["mae"] == full["rmse"] == pytest.approx(.2)
    assert full["in_band_fraction"] == 0
    assert full["in_band_time_fraction"] == 0
    assert full["iae"] == pytest.approx(.004)
    assert steady["within_group_std"] == 0
    assert report["episodes"]["success_flags"] == 1
    assert report["episodes"]["completed_all_samples_tracking_in_band"] == 0


def test_episode_bias_differences_are_not_counted_as_within_episode_jitter():
    metrics = ControlMetrics(2, .01, settle_steps=0, min_steady_samples=2)
    for time in (0., .01):
        add(metrics, time, actual=((1., 0., .3), (3., 0., .3)))
    values = metrics.report()["steady"]["axes"]["vx"]
    assert values["bias"] == 2
    assert values["rmse"] == pytest.approx(math.sqrt(5))
    assert values["within_group_std"] == 0
    assert values["group_mean_std"] == 1


def test_oscillation_is_separate_from_bias():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=3)
    for index, velocity in enumerate((-1., 1., -1., 1.)):
        add(metrics, index * .01, actual=(velocity, 0., .3))
    values = metrics.report()["steady"]["axes"]["vx"]
    assert values["bias"] == 0
    assert values["mae"] == values["rmse"] == values["within_group_std"] == 1


def test_reset_never_contributes_a_target_or_effort_derivative():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=1)
    for offset in (0., 100.):
        for time in (0., .01):
            add(metrics, time, leg_target=(offset,) * 4, wheel_target=(offset,) * 2,
                effort=(offset,) * 6, done=time == .01)
    report = metrics.report()
    for name in ("leg_target_rate", "wheel_target_acceleration", "effort_rate"):
        for channel in report["actuation"][name]["channels"]:
            assert channel["count"] == 2
            assert channel["rms"] == 0
    assert report["episodes"]["completed"] == 2
    assert report["full_interval"]["observed_duration_s"] == .02


def test_only_reset_rows_restart_derivative_and_time():
    metrics = ControlMetrics(2, .01, settle_steps=0, min_steady_samples=1)
    add(metrics, (0., 0.))
    add(metrics, (.01, .01), leg_target=((0.,) * 4, (.01,) * 4), done=(True, False))
    add(metrics, (0., .02), leg_target=((50.,) * 4, (.02,) * 4))
    report = metrics.report()
    assert report["actuation"]["leg_target_rate"]["channels"][0]["count"] == 3
    assert report["actuation"]["leg_target_rate"]["channels"][0]["rms"] == pytest.approx(math.sqrt(2 / 3))
    assert report["episodes"]["completed"] == 1
    assert report["episodes"]["partial"] == 2


def test_leg_and_wheel_derivatives_use_physical_units_and_actual_dt():
    metrics = ControlMetrics(1, .01)
    add(metrics, 0., leg_target=(1., 1., 1., 1.), wheel_target=(2., 2.), effort=(1.,) * 6)
    add(metrics, .02, leg_target=(1.02,) * 4, wheel_target=(2.2,) * 2, effort=(1.4,) * 6)
    actuation = metrics.report()["actuation"]
    assert actuation["leg_target_rate"]["unit"] == "rad/s"
    assert actuation["wheel_target_acceleration"]["unit"] == "rad/s^2"
    assert actuation["leg_target_rate"]["channels"][0]["rms"] == pytest.approx(1)
    assert actuation["wheel_target_acceleration"]["channels"][0]["rms"] == pytest.approx(10)
    assert actuation["effort_rate"]["channels"][0]["rms"] == pytest.approx(20)


def test_effort_bounds_and_mechanical_abs_power_are_not_action_smoothness():
    metrics = ControlMetrics(1, .01)
    add(metrics, 0., effort=(10., -10., 2., 3., 4., 5.), requested=(11., -11., 2., 3., 4., 5.),
        velocity=(2., -3., 4., -5., 6., -7.))
    actuation = metrics.report()["actuation"]
    assert actuation["actual_bound_fraction"] == [1., 1., 0., 0., 0., 0.]
    assert actuation["requested_outside_bounds_fraction"] == [1., 1., 0., 0., 0., 0.]
    assert actuation["sampled_abs_mechanical_power"]["mean"] == 132
    assert actuation["actual_effort"]["channels"][0]["rms"] == 10


def test_disabled_actuator_has_null_saturation_fraction():
    metrics = ControlMetrics(1, .01)
    data = packet()
    data["effort_bounds"][0, 0] = 0
    metrics.update(data, torch.tensor([False]))
    report = metrics.report()["actuation"]
    assert report["active_bound_samples"][0] == 0
    assert report["actual_bound_fraction"][0] is None


def with_scaled_nominal_envelope(data, scale):
    scale = torch.as_tensor(scale, dtype=torch.float64).expand_as(data["requested_motor_effort"])
    return {**data,
            "scaled_nominal_requested_motor_effort": data["requested_motor_effort"] * scale,
            "scaled_nominal_effort_bounds": data["effort_bounds"] * scale[:, :, None]}


@pytest.mark.parametrize("strength", (1., .85))
def test_scaled_nominal_envelope_scales_request_and_bounds_and_preserves_legacy(strength):
    metrics, legacy = ControlMetrics(1, .01), ControlMetrics(1, .01)
    requested = torch.tensor([[41., -41., 39., -39., 40., -40.]], dtype=torch.float64)
    data = packet()
    data["effort_bounds"] *= 4.
    data["requested_motor_effort"] = requested
    data["motor_effort"] = requested.clamp(-40., 40.) * strength
    done = torch.tensor([False])
    legacy.update(data, done)
    metrics.update(with_scaled_nominal_envelope(data, strength), done)
    report = metrics.report()["actuation"]
    scaled = report["scaled_nominal_envelope"]
    actual = {key: value for key, value in report.items() if key != "scaled_nominal_envelope"}
    old = {key: value for key, value in legacy.report()["actuation"].items() if key != "scaled_nominal_envelope"}
    assert actual == old
    assert actual["actual_bound_fraction"] == ([1., 1., 0., 0., 1., 1.] if strength == 1. else [0.] * 6)
    assert scaled["available"] is True and scaled["sample_count"] == 1
    assert scaled["active_bound_samples"] == [1] * 6
    assert scaled["applied_at_bound_fraction"] == [1., 1., 0., 0., 1., 1.]
    assert scaled["requested_outside_bounds_fraction"] == [1., 1., 0., 0., 0., 0.]
    assert scaled["applied_outside_bounds_fraction"] == [0.] * 6
    assert "not final physical wheel output limits" in scaled["semantics"]


def test_scaled_nominal_envelope_counts_all_environment_rows_with_independent_strengths():
    metrics = ControlMetrics(2, .01)
    data = packet(2, effort=((40.,) * 6, (34.,) * 6), requested=(41.,) * 6)
    data["effort_bounds"] *= 4.
    scale = torch.tensor([[1.] * 6, [.85] * 6], dtype=torch.float64)
    metrics.update(with_scaled_nominal_envelope(data, scale), torch.tensor([False, False]))
    scaled = metrics.report()["actuation"]["scaled_nominal_envelope"]
    assert scaled["sample_count"] == 2
    assert scaled["active_bound_samples"] == [2] * 6
    assert scaled["applied_at_bound_fraction"] == scaled["requested_outside_bounds_fraction"] == [1.] * 6


def test_dynamic_wheel_output_outside_scaled_nominal_envelope_is_reported_separately():
    metrics = ControlMetrics(1, .01)
    data = packet(effort=(0., 0., 0., 0., 2.8, -2.8), requested=(0., 0., 0., 0., 1., -1.))
    data["effort_bounds"][0, 4:] = torch.tensor([[-3., 3.], [-3., 3.]])
    metrics.update(with_scaled_nominal_envelope(data, .85), torch.tensor([False]))
    actuation = metrics.report()["actuation"]
    scaled = actuation["scaled_nominal_envelope"]
    assert actuation["actual_bound_fraction"] == [0.] * 6
    assert scaled["applied_at_bound_fraction"] == scaled["requested_outside_bounds_fraction"] == [0.] * 6
    assert scaled["applied_outside_bounds_fraction"] == [0., 0., 0., 0., 1., 1.]


def test_scaled_nominal_zero_width_bounds_have_no_valid_fraction():
    metrics = ControlMetrics(1, .01)
    data = with_scaled_nominal_envelope(packet(effort=(10.,) * 6), .85)
    data["scaled_nominal_effort_bounds"][0, 0] = 0.
    metrics.update(data, torch.tensor([False]))
    scaled = metrics.report()["actuation"]["scaled_nominal_envelope"]
    assert scaled["sample_count"] == 1 and scaled["active_bound_samples"][0] == 0
    for key in ("applied_at_bound_fraction", "requested_outside_bounds_fraction", "applied_outside_bounds_fraction"):
        assert scaled[key][0] is None


def test_missing_scaled_nominal_fields_report_unavailable_without_fabricated_fractions():
    metrics = ControlMetrics(1, .01)
    add(metrics, 0.)
    scaled = metrics.report()["actuation"]["scaled_nominal_envelope"]
    assert scaled["available"] is False and scaled["sample_count"] == 0
    assert scaled["active_bound_samples"] == [0] * 6
    for key in ("applied_at_bound_fraction", "requested_outside_bounds_fraction", "applied_outside_bounds_fraction"):
        assert scaled[key] == [None] * 6


@pytest.mark.parametrize("missing", ("scaled_nominal_requested_motor_effort", "scaled_nominal_effort_bounds"))
def test_partial_scaled_nominal_field_group_is_rejected_before_mutation(missing):
    metrics = ControlMetrics(1, .01)
    data = with_scaled_nominal_envelope(packet(), .85)
    del data[missing]
    with pytest.raises(ValueError, match="provided together"):
        metrics.update(data, torch.tensor([False]))
    add(metrics, 0.)
    assert metrics.report()["full_interval"]["samples"] == 1
    assert metrics.report()["actuation"]["scaled_nominal_envelope"]["available"] is False


@pytest.mark.parametrize("first_available", (False, True))
def test_scaled_nominal_availability_cannot_change_during_evaluation(first_available):
    metrics = ControlMetrics(1, .01)
    data = packet()
    first = with_scaled_nominal_envelope(data, .85) if first_available else data
    metrics.update(first, torch.tensor([False]))
    data = packet(time=.01)
    second = data if first_available else with_scaled_nominal_envelope(data, .85)
    with pytest.raises(ValueError, match="availability must remain constant"):
        metrics.update(second, torch.tensor([False]))
    report = metrics.report()
    assert report["full_interval"]["samples"] == 1
    assert report["actuation"]["scaled_nominal_envelope"]["available"] is first_available


@pytest.mark.parametrize("field,value", [
    ("scaled_nominal_effort_bounds", torch.zeros(1, 6)),
    ("scaled_nominal_effort_bounds", torch.zeros(1, 6, 2, dtype=torch.int64)),
    ("scaled_nominal_effort_bounds", torch.full((1, 6, 2), float("inf"))),
    ("scaled_nominal_effort_bounds", torch.full((1, 6, 2), float("nan"))),
    ("scaled_nominal_effort_bounds", torch.tensor([[[1., -1.]] * 6])),
    ("scaled_nominal_requested_motor_effort", torch.full((1, 6), float("nan"))),
    ("scaled_nominal_requested_motor_effort", torch.zeros(1, 5))])
def test_invalid_scaled_nominal_packet_does_not_mutate_statistics_or_availability(field, value):
    metrics = ControlMetrics(1, .01)
    data = with_scaled_nominal_envelope(packet(), .85)
    data[field] = value
    with pytest.raises(ValueError):
        metrics.update(data, torch.tensor([False]))
    report = metrics.report()
    assert report["available"] is False
    assert report["actuation"]["sample_count"] == 0
    assert report["actuation"]["scaled_nominal_envelope"]["available"] is False


def test_scaled_nominal_bounds_validation_checks_every_row_before_mutation():
    metrics = ControlMetrics(2, .01)
    data = with_scaled_nominal_envelope(packet(2), .85)
    data["scaled_nominal_effort_bounds"][1, 5] = torch.tensor([1., -1.])
    with pytest.raises(ValueError, match="lower bound exceeds"):
        metrics.update(data, torch.tensor([False, False]))
    assert metrics.report()["full_interval"]["samples"] == 0


def test_failed_short_and_final_partial_segments_remain_in_full_scaled_nominal_actuation():
    metrics = ControlMetrics(1, .01, settle_steps=2, min_steady_samples=2)
    for failed in (True, False):
        data = packet(effort=(8.5,) * 6, requested=(11.,) * 6, failure=failed)
        metrics.update(with_scaled_nominal_envelope(data, .85), torch.tensor([failed]))
    report = metrics.report()
    assert report["steady"]["available"] is False
    assert report["episodes"]["failed"] == report["episodes"]["partial"] == 1
    scaled = report["actuation"]["scaled_nominal_envelope"]
    assert scaled["sample_count"] == report["full_interval"]["samples"] == 2
    assert scaled["active_bound_samples"] == [2] * 6
    assert scaled["applied_at_bound_fraction"] == scaled["requested_outside_bounds_fraction"] == [1.] * 6
    assert "no steady filtering" in scaled["scope"]
    json.dumps(report, allow_nan=False)


def test_continuous_leg_joint_error_uses_shortest_arc_at_wrap_boundary():
    metrics = ControlMetrics(1, .01)
    data = packet(leg_target=(math.pi - .01,) * 4)
    data["motor_position"][0, :4] = -math.pi + .01
    metrics.update(data, torch.tensor([False]))
    channels = metrics.report()["actuation"]["leg_position_error"]["channels"]
    assert all(channel["mean"] == pytest.approx(.02) for channel in channels)


def test_step_response_never_reaching_target_is_censored_not_zero_settling():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=1, settling_hold_s=.5)
    add(metrics, 0.)
    for index in range(101):
        add(metrics, .01 + index * .01, reference=(1., 0., .3), actual=(0., 0., .3))
    report = metrics.report()
    response = report["response"]["axes"]["vx"]
    assert response["eligible_steps"] == 1
    assert response["rise_censored"] == response["settling_censored"] == 1
    assert response["metrics"]["settling_time_s"]["mean"] is None
    event = report["response"]["recent_events"][0]
    assert event["complete_iae"] is None
    assert event["observed_iae"] == pytest.approx(1.)


def test_sampled_step_rise_overshoot_and_final_held_settling():
    metrics = ControlMetrics(1, .1, settle_steps=0, min_steady_samples=1, settling_hold_s=.3,
                             tracking_tolerance=(.05, .25, .03))
    add(metrics, 0.)
    values = (0., .1, .5, .9, 1.2, 1., 1., 1., 1.)
    for index, value in enumerate(values):
        add(metrics, .1 + index * .1, reference=(1., 0., .3), actual=(value, 0., .3), done=index == 8,
            success=index == 8)
    event = metrics.report()["response"]["recent_events"][0]
    assert event["rise_10_90_s"] == pytest.approx(.2)
    assert event["absolute_overshoot"] == pytest.approx(.2)
    assert event["settling_time_s"] == pytest.approx(.5)
    assert event["complete_iae"] == pytest.approx(.22)


def test_losing_settled_band_censors_settling_at_endpoint():
    metrics = ControlMetrics(1, .1, settling_hold_s=.2)
    add(metrics, 0.)
    for index, value in enumerate((1., 1., 1., 0.)):
        add(metrics, .1 + index * .1, reference=(1., 0., .3), actual=(value, 0., .3))
    event = metrics.report()["response"]["recent_events"][0]
    assert event["settling_time_s"] is None
    assert event["settling_censored"] is True


def test_terminal_failure_censors_previously_observed_step_response():
    metrics = ControlMetrics(1, .1, settling_hold_s=.2)
    add(metrics, 0.)
    for index in range(4):
        add(metrics, .1 + index * .1, reference=(1., 0., .3), done=index == 3, failure=index == 3)
    event = metrics.report()["response"]["recent_events"][0]
    assert event["failure"] is True
    assert event["rise_10_90_s"] is event["settling_time_s"] is event["complete_iae"] is None
    assert metrics.report()["episodes"]["failed"] == 1


def test_negative_step_uses_sign_correct_rise_and_overshoot():
    metrics = ControlMetrics(1, .1, settling_hold_s=.2)
    add(metrics, 0., reference=(1., 0., .3))
    for index, value in enumerate((1., .9, .5, .1, -.2, 0., 0., 0.)):
        add(metrics, .1 + index * .1, reference=(0., 0., .3), actual=(value, 0., .3))
    event = metrics.report()["response"]["recent_events"][0]
    assert event["rise_10_90_s"] == pytest.approx(.2)
    assert event["absolute_overshoot"] == pytest.approx(.2)


def test_badly_tracking_previous_command_is_not_a_standard_step_response():
    metrics = ControlMetrics(1, .1, settling_hold_s=.2)
    add(metrics, 0., reference=(1., 0., .3), actual=(0., 0., .3))
    for index in range(4):
        add(metrics, .1 + index * .1, reference=(2., 0., .3), actual=(2., 0., .3))
    report = metrics.report()
    response = report["response"]["axes"]["vx"]
    event = report["response"]["recent_events"][0]
    assert response["nonsteady_start"] == 1
    assert response["eligible_steps"] == 0
    assert event["initial_tracking_in_band"] is False
    assert event["rise_10_90_s"] is event["settling_time_s"] is None
    assert report["full_interval"]["axes"]["vx"]["mae"] == pytest.approx(.2)


def test_command_change_segments_remove_between_command_mean_offsets():
    metrics = ControlMetrics(1, .01, settle_steps=1, min_steady_samples=2)
    for index in range(3):
        add(metrics, index * .01, actual=(0., 0., .4))
    for index in range(3, 6):
        add(metrics, index * .01, reference=(0., 0., .5), actual=(0., 0., .7))
    report = metrics.report()
    assert report["steady"]["groups"] == 2
    assert report["steady"]["axes"]["height"]["within_group_std"] == pytest.approx(0.)
    assert report["steady"]["axes"]["height"]["bias"] == pytest.approx(.15)
    assert report["steady"]["discarded_settle_samples"] == 2
    assert report["steady"]["command_changes"] == 1


def test_continuous_small_reference_slew_is_not_a_steady_or_step_response():
    metrics = ControlMetrics(1, .01, settle_steps=2, min_steady_samples=2)
    for index in range(10):
        add(metrics, index * .01, reference=(0., 0., .3 + .001 * index), actual=(0., 0., .3))
    report = metrics.report()
    assert report["steady"]["available"] is False
    assert report["steady"]["short"] == 10
    assert report["steady"]["command_changes"] == 9
    assert report["response"]["axes"]["height"]["events"] == 0
    assert report["response"]["axes"]["height"]["small_reference_changes"] == 9
    assert report["full_interval"]["axes"]["height"]["iae"] == pytest.approx(.000405)


def test_rapid_large_changes_are_short_candidates_not_valid_step_metrics():
    metrics = ControlMetrics(1, .01)
    for index in range(4):
        add(metrics, index * .01, reference=(float(index % 2), 0., .3))
    values = metrics.report()["response"]["axes"]["vx"]
    assert values["events"] == values["short_hold_candidates"] == 3
    assert values["eligible_steps"] == 0
    assert values["metrics"]["rise_10_90_s"]["mean"] is None


def test_failed_and_partial_short_segments_still_count_in_full_interval():
    metrics = ControlMetrics(1, .01, settle_steps=2, min_steady_samples=2)
    add(metrics, 0., actual=(1., 0., .3), done=True, failure=True)
    add(metrics, 0., actual=(2., 0., .3))
    report = metrics.report()
    assert report["full_interval"]["samples"] == 2
    assert report["steady"]["available"] is False
    assert report["steady"]["short"] == 2
    assert report["steady"]["failed"] == report["steady"]["partial"] == 1
    assert report["episodes"]["failed"] == report["episodes"]["partial"] == 1
    assert report["full_interval"]["axes"]["vx"]["bias"] == 1.5


def test_time_fraction_uses_observed_duration_without_reset_bridge():
    metrics = ControlMetrics(1, .01)
    add(metrics, 0., actual=(0., 0., .3))
    add(metrics, .1, actual=(1., 0., .3))
    add(metrics, .4, actual=(1., 0., .3), done=True, failure=True)
    add(metrics, 0., actual=(0., 0., .3))
    values = metrics.report()["full_interval"]
    assert values["observed_duration_s"] == .4
    assert values["axes"]["vx"]["in_band_time_fraction"] == pytest.approx(.25)
    assert values["axes"]["vx"]["iae"] == pytest.approx(.35)


def test_bounded_recent_events_and_idempotent_finalization():
    metrics = ControlMetrics(1, .1, recent_response_limit=2, settling_hold_s=.1)
    for index in range(100):
        add(metrics, index * .1, reference=(float(index % 2), 0., .3))
    report = metrics.report()
    assert len(report["response"]["recent_events"]) == 2
    assert report["response"]["axes"]["vx"]["events"] == 99
    assert report is metrics.report()
    json.dumps(report, allow_nan=False)
    with pytest.raises(RuntimeError, match="after report"):
        add(metrics, 10.)


def test_optional_request_is_distinct_from_effective_tracking_reference():
    metrics = ControlMetrics(1, .01)
    data = packet(reference=(1., 0., .3))
    data["request"] = torch.tensor([[2., 0., .4]], dtype=torch.float64)
    metrics.update(data, torch.tensor([False]))
    report = metrics.report()
    assert report["full_interval"]["axes"]["vx"]["bias"] == 0
    assert report["request_reference_difference"]["axes"]["vx"]["mean"] == -1


@pytest.mark.parametrize("field,value", [("time_s", torch.tensor([0.], dtype=torch.float32)),
    ("actual", torch.tensor([[float("nan"), 0., .3]], dtype=torch.float64)),
    ("failure", torch.tensor([1.])), ("actual", torch.zeros(1, 2)),
    ("effort_bounds", torch.tensor([[[1., -1.]] * 6], dtype=torch.float64))])
def test_invalid_packet_does_not_mutate_statistics(field, value):
    metrics = ControlMetrics(1, .01)
    data = packet()
    data[field] = value
    with pytest.raises(ValueError):
        metrics.update(data, torch.tensor([False]))
    assert metrics.report()["available"] is False


def test_time_and_terminal_contract_is_checked_before_any_row_mutation():
    metrics = ControlMetrics(2, .01)
    add(metrics, (0., 0.))
    with pytest.raises(ValueError, match="strictly increasing"):
        add(metrics, (.01, 0.))
    assert metrics.report()["full_interval"]["samples"] == 2
    metrics = ControlMetrics(1, .01)
    with pytest.raises(ValueError, match="require done"):
        add(metrics, 0., success=True)
    assert metrics.report()["available"] is False


@pytest.mark.parametrize("arguments", [{"num_envs": 0}, {"policy_dt_s": 0}, {"settle_steps": -1},
    {"tracking_tolerance": (1., -1., 1.)}, {"command_tolerance": (1., 1.)}, {"settling_hold_s": 0},
    {"response_min_step": (float("inf"), 1., 1.)}, {"recent_response_limit": -1}])
def test_constructor_rejects_invalid_protocol(arguments):
    kwargs = {"num_envs": 1, "policy_dt_s": .01, **arguments}
    with pytest.raises(ValueError):
        ControlMetrics(**kwargs)


@pytest.mark.parametrize("num_envs", (8, 16))
def test_planar_motion_covers_every_declared_environment_and_both_world_axes(num_envs):
    metrics = ControlMetrics(num_envs, .01, settle_steps=0, min_steady_samples=2)
    velocity = torch.arange(1, num_envs + 1, dtype=torch.float64)[:, None] * torch.tensor([[.03, -.04]])
    for index in range(3):
        add(metrics, index * .01, position=velocity * index * .01)
    report = metrics.report()["planar_motion"]
    full = report["full_interval"]
    assert report["num_envs"] == num_envs
    assert report["physical_samples"] == num_envs * 3
    assert full["intervals"] == num_envs * 2
    assert full["observed_duration_s"] == pytest.approx(num_envs * .02)
    assert full["mean_speed_m_s"] == pytest.approx(.05 * (num_envs + 1) / 2)
    assert full["rms_speed_m_s"] == pytest.approx(.05 * math.sqrt((num_envs + 1) * (2 * num_envs + 1) / 6))
    assert full["velocity_world"]["vx"]["mean_m_s"] == pytest.approx(.03 * (num_envs + 1) / 2)
    assert full["velocity_world"]["vy"]["mean_m_s"] == pytest.approx(-.04 * (num_envs + 1) / 2)
    assert report["stationary"]["runs"] == num_envs
    assert report["stationary_steady"]["mean_speed_m_s"] == pytest.approx(full["mean_speed_m_s"])


def test_planar_speed_uses_actual_duration_and_distinguishes_path_from_net_drift():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=1)
    add(metrics, 0., position=(0., 0.))
    add(metrics, .1, position=(.1, 0.))
    add(metrics, .3, position=(0., 0.))
    report = metrics.report()["planar_motion"]
    full, stationary = report["full_interval"], report["stationary"]
    assert full["path_length_m"] == pytest.approx(.2)
    assert full["mean_speed_m_s"] == pytest.approx(2 / 3)
    assert full["rms_speed_m_s"] == pytest.approx(math.sqrt(.5))
    assert full["velocity_world"]["vx"]["mean_m_s"] == pytest.approx(0.)
    assert stationary["endpoint_displacement_m"]["mean"] == 0.
    assert stationary["max_excursion_m"]["mean"] == pytest.approx(.1)
    assert full["p95_bin_m_s"] == [1., 1.001]
    assert full["p95_speed_m_s"] == pytest.approx(1.0005)


def test_planar_motion_excludes_reset_teleports_and_reset_rows_only():
    metrics = ControlMetrics(2, .01, settle_steps=0, min_steady_samples=2)
    add(metrics, (0., 0.), position=((0., 0.), (0., 0.)))
    add(metrics, (.01, .01), position=((.01, 0.), (0., .02)), done=(True, False), failure=(True, False))
    add(metrics, (0., .02), position=((1000., -1000.), (0., .04)))
    add(metrics, (.01, .03), position=((1000.01, -1000.), (0., .06)))
    report = metrics.report()["planar_motion"]
    assert report["full_interval"]["intervals"] == 5
    assert report["full_interval"]["path_length_m"] == pytest.approx(.08)
    assert report["full_interval"]["max_speed_m_s"] == pytest.approx(2.)
    assert report["stationary"]["runs"] == 3
    assert report["stationary"]["endpoint_displacement_m"]["mean"] == pytest.approx(.08 / 3)


def test_stationary_drift_starts_when_effective_command_becomes_zero():
    metrics = ControlMetrics(1, .01, settle_steps=0, min_steady_samples=2)
    add(metrics, 0., reference=(1., 0., .3), position=(0., 0.))
    add(metrics, .01, position=(10., 0.))
    add(metrics, .02, position=(10., .003))
    # Height changes preserve the stationary origin; motion changes end it.
    add(metrics, .03, reference=(0., 0., .4), position=(10., .006))
    add(metrics, .04, reference=(.5, 0., .4), position=(20., .006))
    add(metrics, .05, reference=(0., 0., .4), position=(30., .006))
    add(metrics, .06, reference=(0., 0., .4), position=(30., .009))
    stationary = metrics.report()["planar_motion"]["stationary"]
    assert stationary["samples"] == 5
    assert stationary["runs"] == 2
    assert stationary["intervals"] == 3
    assert stationary["path_length_m"] == pytest.approx(.009)
    assert stationary["mean_speed_m_s"] == pytest.approx(.3)
    assert stationary["endpoint_displacement_m"]["mean"] == pytest.approx(.0045)
    assert stationary["max_excursion_m"]["max_abs"] == pytest.approx(.006)


def test_planar_steady_omits_settle_boundary_short_segments_and_command_change_edges():
    metrics = ControlMetrics(1, .01, settle_steps=2, min_steady_samples=2)
    for index in range(4):
        add(metrics, index * .01, position=(float(index), 0.))
    # New segment has only one retained point and is ineligible, despite motion.
    for index in range(4, 7):
        add(metrics, index * .01, reference=(.5, 0., .3), position=(float(index), 0.))
    report = metrics.report()["planar_motion"]
    assert report["full_interval"]["intervals"] == 6
    assert report["steady"]["intervals"] == 1
    assert report["steady"]["observed_duration_s"] == pytest.approx(.01)
    assert report["steady"]["path_length_m"] == 1.
    assert report["stationary_steady"]["intervals"] == 1


@pytest.mark.parametrize("done,failure", ((False, False), (True, True)))
def test_planar_steady_retains_eligible_final_partial_and_failure_segments(done, failure):
    metrics = ControlMetrics(1, .01, settle_steps=1, min_steady_samples=2)
    for index in range(3):
        add(metrics, index * .01, position=(0., .005 * index), done=done and index == 2,
            failure=failure and index == 2)
    steady = metrics.report()["planar_motion"]["stationary_steady"]
    assert steady["available"] is True
    assert steady["intervals"] == 1
    assert steady["mean_speed_m_s"] == pytest.approx(.5)


def test_planar_speed_quantile_reports_overflow_without_clipping_mean_or_rms():
    metrics = ControlMetrics(1, .01)
    add(metrics, 0.)
    add(metrics, .01, position=(.12, 0.))
    full = metrics.report()["planar_motion"]["full_interval"]
    assert full["mean_speed_m_s"] == full["rms_speed_m_s"] == 12.
    assert full["p95_speed_m_s"] is None
    assert full["p95_bin_m_s"] == [10., None]
    assert full["overflow_duration_s"] == .01


def test_isolated_planar_samples_do_not_fabricate_velocity_or_quantiles():
    metrics = ControlMetrics(2, .01)
    add(metrics, (0., 0.), position=((.1, .2), (.3, .4)), done=(True, False))
    report = metrics.report()["planar_motion"]
    assert report["physical_samples"] == 2
    assert report["full_interval"]["available"] is False
    assert report["full_interval"]["mean_speed_m_s"] is None
    assert report["full_interval"]["p95_bin_m_s"] == [None, None]
    assert report["stationary"]["runs"] == 2
    json.dumps(report, allow_nan=False)

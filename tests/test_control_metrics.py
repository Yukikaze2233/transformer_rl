"""Artificial physical traces verify motion metrics without a simulator."""
import json
import math

import pytest
import torch

from transformer_rl.control_metrics import ControlMetrics


def packet(num_envs=1, *, time=0., reference=(0., 0., .3), actual=None,
           leg_target=None, wheel_target=None, effort=None, requested=None,
           velocity=None, failure=False, success=False):
    def tensor(value, shape):
        return torch.as_tensor(value, dtype=torch.float64).expand((num_envs, *shape)).clone()
    actual = reference if actual is None else actual
    effort = (0.,) * 6 if effort is None else effort
    return {"time_s": tensor(time, ()), "command_reference": tensor(reference, (3,)),
            "actual": tensor(actual, (3,)), "position_xy": tensor((0., 0.), (2,)), "tilt": tensor(0., ()),
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

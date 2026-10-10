"""Scenario-specific control grading with explicit synthetic report fixtures."""
from copy import deepcopy
import json
import math

import pytest

from transformer_rl import exposure_selection, frame_study
from transformer_rl.exposure_protocol import _receipt
from test_frame_study import specification
from test_frame_workflow import configuration


def objective(path, *, scenarios=None, scale=1., weight=1.):
    value = {"path": path, "direction": "minimize", "scale": scale, "weight": weight}
    if scenarios is not None:
        value["scenarios"] = scenarios
    return value


@pytest.mark.parametrize("routes", ([], "normal", ["normal", "normal"], ["unknown"], [True], [None], [["normal"]]))
def test_objective_scope_rejects_ambiguous_or_unknown_scenarios(tmp_path, routes):
    spec = json.loads(specification(tmp_path).read_bytes())
    spec["selection"]["objectives"][0]["scenarios"] = routes
    with pytest.raises(ValueError, match="objective scenarios"):
        frame_study._validate_spec(spec)


def test_explicit_scope_must_cover_every_declared_scenario(tmp_path):
    spec = json.loads(specification(tmp_path).read_bytes())
    original = deepcopy(spec["selection"]["objectives"][0])
    spec["selection"]["objectives"][0]["scenarios"] = ["normal"]
    with pytest.raises(ValueError, match="every scenario"):
        frame_study._validate_spec(spec)
    spec["selection"]["objectives"].append({**original, "scenarios": ["new_skill"]})
    frame_study._validate_spec(spec)
    # A common objective may cover all cases alongside a stationary-only term.
    spec["selection"]["objectives"][1].pop("scenarios")
    frame_study._validate_spec(spec)


@pytest.mark.parametrize("sample_window", (None, True, "full", [], {"window": "post_settle"}))
def test_history_sample_window_rejects_implicit_or_unknown_choices(tmp_path, sample_window):
    spec = json.loads(specification(tmp_path).read_bytes())
    spec["scenarios"][0].update(require_steady=True, history_sample_window=sample_window)
    with pytest.raises(ValueError, match="history_sample_window"):
        frame_study._validate_spec(spec)


@pytest.mark.parametrize("sample_window", ("constant_reference", "post_settle"))
def test_history_sample_window_requires_the_declared_sample_gate(tmp_path, sample_window):
    spec = json.loads(specification(tmp_path).read_bytes())
    scenario = spec["scenarios"][0]
    scenario.update(require_steady=True, history_sample_window=sample_window)
    frame_study._validate_spec(spec)
    scenario["require_steady"] = False
    with pytest.raises(ValueError, match="history_sample_window"):
        frame_study._validate_spec(spec)


def test_unscoped_objectives_keep_the_original_grade():
    report = {"completed_episodes": 8, "success_rate": 1., "metrics": {"height": .03}}
    scenario = {"gates": [{"path": "success_rate", "operator": "min", "value": .95}]}
    objectives = [objective("metrics.height", scale=.03, weight=2.),
                  {"path": "success_rate", "direction": "maximize", "scale": 1., "weight": 4.}]
    assert frame_study.grade_report(report, scenario, {"min_completed_episodes": 8}, objectives) == {
        "passed": True, "reasons": [], "score": -2.}


def test_stationary_drift_is_required_for_standing_and_not_for_motion():
    report = {"completed_episodes": 8, "control": {"height": .03}}
    objectives = [objective("control.height", scale=.03),
                  objective("control.planar_motion.stationary_steady.rms_speed_m_s", scenarios=["stand"], scale=.1)]
    evaluation = {"min_completed_episodes": 8}
    moving = {"name": "forward", "gates": []}
    standing = {"name": "stand", "gates": []}
    assert frame_study.grade_report(report, moving, evaluation, objectives) == {
        "passed": True, "reasons": [], "score": 1.}
    missing = frame_study.grade_report(report, standing, evaluation, objectives)
    assert not missing["passed"] and missing["score"] is None
    report["control"]["planar_motion"] = {"stationary_steady": {"rms_speed_m_s": .2}}
    assert frame_study.grade_report(report, standing, evaluation, objectives)["score"] == pytest.approx(3.)
    assert frame_study.grade_report(report, moving, evaluation, objectives)["score"] == 1.


def test_no_applicable_objective_cannot_produce_a_zero_score():
    grade = frame_study.grade_report({"completed_episodes": 8}, {"name": "forward", "gates": []},
        {"min_completed_episodes": 8}, [objective("drift", scenarios=["stand"])])
    assert grade == {"passed": False, "reasons": ["missing applicable objectives"], "score": None}


def test_physical_channel_index_applies_to_both_gates_and_scores():
    path = "control.actuation.effort_rate.channels.1.rms"
    report = {"completed_episodes": 8, "control": {"actuation": {
        "effort_rate": {"channels": [{"rms": 0.}, {"rms": 120.}]}}}}
    scenario = {"gates": [{"path": path, "operator": "max", "value": 100.}]}
    grade = frame_study.grade_report(report, scenario, {"min_completed_episodes": 8}, [objective(path, scale=100.)])
    assert grade == {"passed": False, "reasons": [path], "score": 1.2}
    report["control"]["actuation"]["effort_rate"]["channels"][1]["rms"] = 80.
    assert frame_study.grade_report(report, scenario, {"min_completed_episodes": 8},
                                   [objective(path, scale=100.)])["passed"]


@pytest.mark.parametrize("index", ("-1", "+1", "01", "2", "x", "١", "", "9" * 5000))
def test_list_paths_reject_invalid_or_unavailable_channels(index):
    path = f"channels.{index}.rms"
    report = {"completed_episodes": 8, "channels": [{"rms": 10.}, {"rms": 20.}]}
    grade = frame_study.grade_report(report, {"gates": []}, {"min_completed_episodes": 8}, [objective(path)])
    assert not grade["passed"] and grade["score"] is None


@pytest.mark.parametrize("value", (None, True, math.nan, math.inf, -math.inf, "1.0", {"rms": 1.}))
def test_unavailable_or_nonfinite_physical_metric_cannot_be_scored(value):
    grade = frame_study.grade_report({"completed_episodes": 8, "channels": [value]},
        {"gates": []}, {"min_completed_episodes": 8}, [objective("channels.0")])
    assert not grade["passed"] and grade["score"] is None


def test_smooth_height_with_large_bias_cannot_beat_correct_tracking():
    objectives = [objective("control.steady.axes.height.rmse", scale=.03),
                  objective("control.steady.axes.height.within_group_std", scale=.005, weight=.1)]
    def measured(rmse, jitter):
        return frame_study.grade_report({"completed_episodes": 8, "control": {"steady": {"axes": {
            "height": {"rmse": rmse, "within_group_std": jitter}}}}},
            {"gates": []}, {"min_completed_episodes": 8}, objectives)
    biased = measured(.09, .0001)
    correct = measured(.01, .002)
    assert biased["score"] > correct["score"]


@pytest.mark.parametrize("scenario_name,expected", (("stand", .27), ("forward", .02)))
def test_exposure_provider_and_independent_selector_share_scoped_physical_grade(tmp_path, scenario_name, expected):
    config = configuration("transformer", history_length=4)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    evaluation = {"min_completed_episodes": 8, "min_steady_samples": 2, "settle_steps": 1}
    objectives = [objective("control.actuation.leg_target_rate.channels.0.rms", scale=10., weight=.1),
                  objective("control.planar_motion.stationary_steady.rms_speed_m_s", scenarios=["stand"], scale=.1)]
    metrics = {"completed_episodes": 8, "control": {"actuation": {"leg_target_rate": {"channels": [{"rms": 2.}]}}},
        "history_control": {"history_length": 4, "minimum_full_age": 3, "policy_dt_s": .01,
            "settle_steps": 1, "min_steady_samples": 2,
            "windows": {"full_history": {"samples": 4, "steady_tracking": {"samples": 4}}}}}
    if scenario_name == "stand":
        metrics["control"]["planar_motion"] = {"stationary_steady": {"rms_speed_m_s": .025}}
    scenarios = [{"name": name, "gates": []} for name in ("stand", "forward")]
    protocol = {"evaluation": evaluation, "scenarios": scenarios, "selection": {"objectives": objectives}}
    current = next(scenario for scenario in scenarios if scenario["name"] == scenario_name)
    grade = frame_study.grade_report(metrics, current, evaluation, objectives)
    cell = {"config_receipt": _receipt(path), "scenario": scenario_name, "num_envs": 2}
    result, full_history = exposure_selection._physical_grade(protocol, cell, {"metrics": metrics, "grade": grade})
    assert result["passed"] and full_history["passed"] and result["score"] == pytest.approx(expected)
    substituted = {**grade, "score": grade["score"] + .01}
    with pytest.raises(ValueError, match="provider grade differs"):
        exposure_selection._physical_grade(protocol, cell, {"metrics": metrics, "grade": substituted})

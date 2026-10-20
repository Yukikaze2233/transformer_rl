"""V6/V7 shared real2sim manifest consistency checks."""
import json
from pathlib import Path


def load():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/real2sim_v6_v1.json").read_text())


def test_shared_asset_and_platform_contract():
    data = load()
    assert data["schema"] == "real2sim-identification-manifest-v1"
    assert data["applies_to"] == ["v6", "v7", "current"]
    asset = data["asset"]
    assert asset["revision"] == "v6_232mm_stop105_nominal_v1"
    assert asset["wheel_gear_ratio"] == 15.8
    assert asset["wheel_radius_m"] == .06
    assert asset["spring_free_length_mm"] == 232
    assert asset["spring_stroke_mm"] == 80
    assert data["terrain"]["double_step_platform_depth_m"] == .80
    assert data["terrain"]["depth_interpretation"] == "platform_depth_not_step_height"
    assert data["terrain"]["single_step_rise_m"] == data["terrain"]["double_step_rise_m"] == .20


def test_bag_catalog_and_identification_boundaries_are_explicit():
    data = load()
    assert len(data["bags"]) == 11
    assert all({"run", "path", "mcap_sha256", "status"} <= set(bag) for bag in data["bags"])
    assert data["identification"]["wheel"]["status"] == "diagnostic_estimates_not_released_to_training"
    assert data["identification"]["leg"]["status"] == "coupled_model_not_released_to_training"
    assert not data["boundaries"]["online_identification"]
    assert not data["boundaries"]["bag_replay_as_ppo_experience"]
    assert not data["boundaries"]["six_axis_plant_model_released"]

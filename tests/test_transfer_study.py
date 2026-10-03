"""File-only preparation, provenance rejection and scratch-clock invariants."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from transformer_rl.frame_config import FrameTrainConfig
from transformer_rl.frame_study import plan_study, validate_study
from transformer_rl.transfer_study import prepare_transfer_study


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    package = root / "src/wheeled_tasks"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Frozen fixture; never imported.\n")
    (package / "chassis").mkdir()
    (package / "chassis/task.py").write_text(
        "def choose_scene_groups(groups, count):\n"
        "    result = []\n"
        "    for i, group in enumerate(groups):\n"
        "        size = count-len(result) if i == len(groups)-1 else int(count*group['fraction'])\n"
        "        result.extend((group['name'], group['terrain'][j % len(group['terrain'])]) for j in range(size))\n"
        "    return result\n")
    asset = root / "model/v6"
    files = {}
    for name in ("model_spec.json", "fit_10mpa.json", "gas_spring_binding.json"):
        files[name] = write_json(asset / name, {"source": name})
    names = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]
    manifest_sha = write_json(asset / "manifest.json", {"files_sha256": files,
                              "nominal_joint_pos": dict.fromkeys(names, 0.)})
    control_sha = write_json(root / "contracts/control.json", {
        "timing": {"physics_dt": .005, "policy_dt": .02, "pc_control_dt": .005}})
    response = {"enabled": True, "armature": [0., 0., .0038, 0., 0., .0038]}
    measurement_sha = write_json(root / "reports/measurement.json", {"measured": True})
    evidence_sha = write_json(root / "contracts/evidence.json", {
        "status": "validated_for_research", "hardware_release": False,
        "adopted_joint_names": ["L_joint3", "R_joint3"], "actuator_response": response,
        "files_sha256": {"reports/measurement.json": measurement_sha}})
    lookup_sha = write_json(root / "contracts/lookup.json", {"geometry": True})
    profiles_sha = write_json(root / "contracts/profiles.json", {"file": "contracts/lookup.json", "sha256": lookup_sha})
    parent = {
        "contract_id": "v6-flat-fixture", "asset_directory": "model/v6",
        "asset_manifest_sha256": manifest_sha, "control_math_source": "contracts/control.json",
        "control_math_sha256": control_sha, "physics_dt": .005, "policy_dt": .02,
        "history_length": 1, "actor_frame_dim": 35, "actor_dim": 35, "critic_dim": 81, "action_dim": 6,
        "policy_action_order": names, "v5_control": {"action_clip": 3., "wheel_action_clip": 9.,
            "leg_position_scale": .25, "wheel_velocity_scale": 10.},
        "scene_groups": [{"name": "stand_305mm", "fraction": 1., "terrain": ["flat"]}],
        "skill_specs": {"stand_305mm": {"kind": "stand"}},
        "actuator_response": response, "actuator_response_evidence": {
            "file": "contracts/evidence.json", "sha256": evidence_sha},
        "dynamics_randomization": {"enabled": True}, "contact_domain": {"enabled": True},
        "signal_perturbations": {"enabled": True, "max_delay_steps": 0, "enabled_fraction": .5},
        "command_transport": {"enabled": True}, "usb_transport": {"enabled": False},
        "signal_delay": {"enabled": True, "version": "physical_signal_delay_v1", "clock": "physics_steps",
            "jitter_clock": "policy_steps", "enabled_fraction": .75, "jitter_ms": 5.,
            "feedback_pd": "fresh_physics_feedback", "profiles": [
                {"probability": .6, "observation_ms": [0., 20.], "action_ms": [0., 15.]},
                {"probability": .35, "observation_ms": [20., 40.], "action_ms": [15., 30.]},
                {"probability": .05, "observation_ms": [40., 80.], "action_ms": [30., 60.]}],
            "schedule": {"clock": "domain_actor_updates", "start": 1500, "end": 2500}},
        "new_asset_curriculum": {"enabled": True, "height_sampling_schedule": {
            "actor_update_boundary": 200, "initial_fixed_fraction": .8, "final_fixed_fraction": .5}},
        "recovery_training": {"enabled": True, "profiles_file": "contracts/profiles.json", "profiles_sha256": profiles_sha},
        "jump_assist": {"enabled": False}, "step_assist": {"enabled": False},
        "evaluation": {"cases": [{"name": "stand_305mm", "task": "survive", "terrain": "flat"}]},
        "task_campaign": {"domain": "flat"}, "global_actor_update_offset": 13000, "task_actor_update": 13000,
    }
    task = root / "parent.json"
    write_json(task, parent)
    module = ModuleType("transformer_rl.transfer_profiles")
    def cases(base):
        result = []
        for profile in ("nominal", "delay40_30"):
            case = deepcopy(base[0])
            case.update(name=case["name"] + "__" + profile,
                        transfer_profile={"name": profile, "base_case": base[0]["name"]})
            result.append(case)
        return result
    def evaluation(base, entries):
        result = deepcopy(base)
        result["evaluation_exact_cases"] = True
        result["evaluation"]["cases"] = entries
        return result
    module.build_transfer_cases = cases
    module.configure_transfer_evaluation = evaluation
    monkeypatch.setitem(sys.modules, "transformer_rl.transfer_profiles", module)
    return root, task, parent


def test_frozen_transfer_plan_preserves_response_and_resets_scratch_clock(source, tmp_path):
    root, task, parent = source
    parent_bytes = task.read_bytes()
    directory = tmp_path / "prepared"
    receipt = prepare_transfer_study(root, task, directory)
    snapshot = directory / "snapshot"
    train = json.loads((snapshot / "contracts/transfer.train.json").read_text())
    assert task.read_bytes() == parent_bytes == (snapshot / "parent_task.json").read_bytes()
    assert (train["policy_dt"], train["physics_dt"]) == (.01, .005)
    assert train["actuator_response"] == parent["actuator_response"]
    assert "task_campaign" not in train
    assert train["global_actor_update_offset"] == train["stage_actor_update"] == train["task_actor_update"] == 0
    assert train["signal_delay"]["schedule"] == {"clock": "domain_actor_updates", "start": 200, "end": 600}
    assert train["new_asset_curriculum"]["height_sampling_schedule"] == parent["new_asset_curriculum"]["height_sampling_schedule"]
    assert train["recovery_training"]["enabled"] is False
    assert (snapshot / "contracts/profiles.json").exists() and (snapshot / "contracts/lookup.json").exists()
    assert (snapshot / "reports/measurement.json").exists()
    base = FrameTrainConfig.load(directory / "base.json")
    assert base.ppo.learning_rate == 3e-5
    assert base.model.initial_std == base.model.policy.mean_init_scale == 1.
    spec = json.loads((directory / "study.json").read_text())
    assert len(spec["variants"]) == 10 and len(spec["stages"]) == 1
    assert spec["training"]["checkpoint_interval"] == spec["stages"][0]["updates"] == 1200
    assert spec["training"]["max_seconds"] == 86400.
    assert all(case["environment"]["evaluation_batch"] == "transfer_all" for case in spec["scenarios"])
    assert receipt["training_started"] is False
    packed = tmp_path / "packed"
    planned = plan_study(directory / "study.json", packed)
    validated = validate_study(packed)
    assert planned["jobs"] == 10 and len(validated["spec"]["scenarios"]) == 2


@pytest.mark.parametrize("route", ["model/v6/fit_10mpa.json", "contracts/control.json", "contracts/lookup.json",
                                   "reports/measurement.json", "contracts/evidence.json"])
def test_mutated_asset_control_or_recursive_evidence_is_rejected(source, tmp_path, route):
    root, task, _ = source
    (root / route).write_text("{}")
    with pytest.raises(ValueError, match="source dependency changed"):
        prepare_transfer_study(root, task, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("changes", [{"physics_dt": .001}, {"policy_dt": .01}, {"actor_dim": 36},
                                     {"usb_transport": {"enabled": True}}])
def test_parent_contract_must_match_v6_timing_abi_and_pipeline(source, tmp_path, changes):
    root, task, parent = source
    parent.update(changes)
    write_json(task, parent)
    with pytest.raises(ValueError):
        prepare_transfer_study(root, task, tmp_path / "prepared")


def test_derived_evidence_cannot_substitute_different_wheel_response(source, tmp_path):
    root, task, parent = source
    parent["actuator_response"] = {"enabled": True, "armature": [0.] * 6}
    write_json(task, parent)
    with pytest.raises(ValueError, match="wheel response does not match"):
        prepare_transfer_study(root, task, tmp_path / "prepared")


@pytest.mark.parametrize("kwargs", [{"updates": 200}, {"seeds": [701]}, {"seeds": [1, 1]},
                                    {"num_envs": 1}, {"evaluation_replicas": 2}])
def test_budget_and_seeds_cannot_hide_uncovered_delay_training(source, tmp_path, kwargs):
    root, task, _ = source
    with pytest.raises(ValueError):
        prepare_transfer_study(root, task, tmp_path / "prepared", **kwargs)


def test_dependency_path_cannot_escape_source_even_with_valid_hash(source, tmp_path):
    root, task, parent = source
    outside = root.parent / "outside.json"
    sha = write_json(outside, {"secret": "fixture"})
    parent["extra_dependency"] = {"file": "../outside.json", "sha256": sha}
    write_json(task, parent)
    with pytest.raises(ValueError, match="escapes"):
        prepare_transfer_study(root, task, tmp_path / "prepared")


def test_integer_allocation_is_verified_by_the_frozen_source_helper(source, tmp_path):
    root, task, parent = source
    parent["scene_groups"] = [{"name": f"group_{i}", "fraction": fraction, "terrain": ["flat"]}
                               for i, fraction in enumerate((.333, .333, .334))]
    write_json(task, parent)
    directory = tmp_path / "prepared"
    prepare_transfer_study(root, task, directory, num_envs=1026)
    design = json.loads((directory / "transfer_design.json").read_text())
    allocation = design["scene_allocation"]
    assert allocation["requested_counts"] == {"group_0": 342, "group_1": 341, "group_2": 343}
    assert allocation["frozen_allocator_verification"]["verified_counts"] == allocation["requested_counts"]


def test_incompatible_frozen_allocator_cannot_be_hidden_by_config_rounding(source, tmp_path):
    root, task, _ = source
    (root / "src/wheeled_tasks/chassis/task.py").write_text("def choose_scene_groups(groups, count):\n    return []\n")
    with pytest.raises(ValueError, match="frozen scene allocator"):
        prepare_transfer_study(root, task, tmp_path / "prepared")

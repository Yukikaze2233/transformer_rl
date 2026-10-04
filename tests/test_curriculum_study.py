"""File-only curriculum preparation and immutable experimental controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from transformer_rl.curriculum_study import (
    ARM_DOMAINS, CLOCK_FIELDS, TASK_FIELDS, TRANSITIONS_PER_UPDATE,
    prepare_curriculum_study, validate_curriculum_study,
)
from transformer_rl.frame_config import FrameTrainConfig, digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def seal_manifest(directory, manifest):
    manifest["sha256"] = digest({key: value for key, value in manifest.items() if key != "sha256"})
    write_json(directory / "manifest.json", manifest)


@pytest.fixture
def source(tmp_path):
    """Minimal already-frozen transfer study, without simulator imports."""
    directory = tmp_path / "original"
    snapshot = directory / "snapshot"
    task_path = snapshot / "src/wheeled_tasks/chassis/task.py"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        "def choose_scene_groups(groups, count):\n"
        "    result = []\n"
        "    for i, group in enumerate(groups):\n"
        "        size = count-len(result) if i == len(groups)-1 else int(count*group['fraction'])\n"
        "        result.extend((group['name'], group['terrain'][j % len(group['terrain'])]) for j in range(size))\n"
        "    return result\n")
    (task_path.parent / "rewards.py").write_text("REWARD_VERSION = 1\n")
    # A nested file with the same name as the root receipt must remain sealed.
    write_json(snapshot / "model/metadata/snapshot.json", {"asset": "immutable"})
    parent = {
        "contract_id": "packed-transfer-study", "physics_dt": .005, "policy_dt": .01,
        "target_num_envs": 1024, "num_steps_per_env": 48,
        "history_length": 1, "actor_frame_dim": 35, "actor_dim": 35, "critic_dim": 81, "action_dim": 6,
        "scene_groups": [{"name": "stand_305mm", "fraction": .25, "terrain": ["flat"]},
                         {"name": "forward_05", "fraction": .75, "terrain": ["flat"]}],
        "skill_specs": {
            "stand_305mm": {"kind": "stand", "command": [0., 0., .305], "mode": 0,
                            "height_sampling": {"range": [.23, .43]}},
            "forward_05": {"kind": "translation", "command": [.5, 0., .305], "mode": 1}},
        "terrain_limits": {"max_slope_deg": 15.},
        "behavior_pool_membership": {"stand_305mm": "standing", "forward_05": "translation"},
        "behavior_pool_fractions": {"standing": .25, "translation": .75},
        "dense_tracking": {"enabled": True, "velocity_weight": 4., "height_weight": 30.},
        "reward_profile": {"stationary_velocity_weight": 3., "yaw_weight": 1.},
        "actuator_response": {"enabled": True, "armature": [0., 0., .0038, 0., 0., .0038]},
        "dynamics_randomization": {"enabled": True, "mass_range": [.8, 1.2]},
        "contact_domain": {"enabled": True, "friction_range": [.5, 1.1]},
        "signal_perturbations": {"enabled": True, "observation_noise": .01},
        "signal_delay": {"enabled": True, "schedule": {"clock": "domain_actor_updates", "start": 200, "end": 600}},
        "new_asset_curriculum": {"enabled": True, "stage": "flat", "stage_start_actor_update": 0,
                                 "height_sampling_schedule": {"actor_update_boundary": 200},
                                 "start_stop_start_delay_seconds": [0., 8.],
                                 "command_resampling_seconds": [1., 8.]},
        "evaluation_exact_cases": False,
        **dict.fromkeys(CLOCK_FIELDS, 0), "total_updates": 1200,
    }
    train_route = "contracts/transfer.train.json"
    train_sha = write_json(snapshot / train_route, parent)
    scenarios = []
    for index in range(50):
        route = f"contracts/transfer.eval.case_{index:02d}.json"
        contract = deepcopy(parent)
        contract.update(target_num_envs=8, evaluation_exact_cases=True,
                        evaluation={"cases": [{"name": f"case_{index:02d}", "target_height_m": .305}]})
        scenarios.append({"name": f"case_{index:02d}", "environment": {
            "contract": route, "contract_sha256": write_json(snapshot / route, contract),
            "num_envs": 8, "evaluation_batch": "transfer_all"}, "gates": [{"irrelevant_to_curriculum": True}]})
    control = {
        "policy_dt_s": .01, "observation_schema": "scaled35_fixture",
        "feature_names": [f"feature_{index}" for index in range(35)],
        "action_names": [f"joint_{index}" for index in range(6)],
        "action_bounds": [3.] * 4 + [9.] * 2,
        "target_scale": [.25] * 4 + [10.] * 2, "target_offset": [0.] * 6,
        "target_units": ["rad"] * 4 + ["rad/s"] * 2,
    }
    files = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    identity = {"files": files, "sha256": digest(files), "control": control,
                "parent_task_sha256": "a" * 64, "contract_validator": "packed_transfer_study"}
    write_json(snapshot / "snapshot.json", identity)
    base = FrameTrainConfig.from_dict({
        "model": {"policy": {"architecture": "mlp", "frame_dim": 35, "action_dim": 6,
                              "history_length": 1}, "critic_dim": 81, "initial_std": 1.},
        "ppo": {"learning_rate": 3e-5, "num_minibatches": 32}, "control": control,
        "environment": {"snapshot": str(snapshot), "snapshot_sha256": identity["sha256"],
                        "contract": train_route, "contract_sha256": train_sha, "num_envs": 1024}})
    write_json(directory / "base.json", base.to_dict())
    write_json(directory / "study.json", {"base_config": "base.json", "scenarios": scenarios,
               "environment_factory": "transformer_rl.chassis_adapter:make_env"})
    return directory, parent


def phase_config(directory, manifest, arm, phase):
    entry = next(value for value in manifest["arms"] if value["name"] == arm)["phases"][phase - 1]
    config = FrameTrainConfig.load(directory / entry["config"])
    contract = read(directory / "snapshot" / config.environment["contract"])
    return entry, config, contract


def test_three_arms_freeze_shared_gated_recipe_and_leave_original_untouched(source, tmp_path):
    original, parent = source
    before = {str(path.relative_to(original)): path.read_bytes()
              for path in original.rglob("*") if path.is_file()}
    directory = tmp_path / "prepared"
    receipt = prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(receipt["manifest"])
    assert receipt["training_started"] is False
    assert receipt["arms"] == 3 and receipt["training_seeds"] == [1101, 1102, 1103]
    assert [arm["name"] for arm in manifest["arms"]] == list(ARM_DOMAINS)
    assert manifest["protocol"]["planned_transitions_per_arm_seed"] == 58_982_400
    assert manifest["protocol"]["planned_total_transitions"] == 530_841_600
    assert manifest["protocol"]["allocation_counts"] == {
        "mixed": {"stand_305mm": 256, "forward_05": 768}, "stationary": {"stand_305mm": 1024}}
    assert manifest["protocol"]["task_dose_matched"] is False
    assert manifest["protocol"]["phase_evaluation_does_not_gate_progress"] is True
    assert len(manifest["configs"]) == 56 and len(manifest["scenarios"]) == 50
    assert not any("gates" in case for case in manifest["scenarios"])
    assert {str(path.relative_to(original)): path.read_bytes()
            for path in original.rglob("*") if path.is_file()} == before
    config = phase_config(directory, manifest, "mixed", 1)[1]
    assert config.model.policy.architecture == "transformer"
    assert config.model.policy.residual_type == "gated"
    assert (config.model.policy.history_length, config.model.policy.d_model,
            config.model.policy.num_layers, config.model.policy.num_heads) == (31, 128, 2, 4)
    assert config.ppo == FrameTrainConfig.load(original / "base.json").ppo
    assert config.control == FrameTrainConfig.load(original / "base.json").control
    assert config.model.initial_std == 1.
    assert read(directory / "snapshot/model/metadata/snapshot.json") == {"asset": "immutable"}
    assert parent == read(original / "snapshot/contracts/transfer.train.json")


def test_phase_clocks_are_cumulative_with_identical_resets_and_preserved_delay(source, tmp_path):
    original, parent = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(directory)
    protected = {key: value for key, value in parent.items()
                 if key not in TASK_FIELDS | CLOCK_FIELDS | {"total_updates"}}
    protected = deepcopy(protected)
    protected["new_asset_curriculum"].pop("start_stop_start_delay_seconds")
    identities = set()
    for arm in manifest["arms"]:
        for index in (1, 2):
            entry, config, contract = phase_config(directory, manifest, arm["name"], index)
            start = 0 if index == 1 else 400
            assert (entry["start_update"], entry["updates"]) == (start, 400 if index == 1 else 800)
            assert contract["global_actor_update_offset"] == 0
            assert contract["training_transitions"] == start * TRANSITIONS_PER_UPDATE
            assert all(contract[name] == start for name in CLOCK_FIELDS - {
                "global_actor_update_offset", "training_transitions"})
            assert contract["new_asset_curriculum"]["stage_start_actor_update"] == 0
            assert contract["signal_delay"] == parent["signal_delay"]
            actual = {key: deepcopy(value) for key, value in contract.items() if key not in
                      TASK_FIELDS | CLOCK_FIELDS | {"total_updates"}}
            actual["new_asset_curriculum"].pop("start_stop_start_delay_seconds", None)
            assert actual == protected
            assert config.environment["num_envs"] * manifest["training"]["rollout_steps"] == 49152
            identities.add(config.environment["snapshot_sha256"])
    assert identities == {manifest["snapshot_identity"]["sha256"]}
    assert manifest["protocol"]["same_phase_boundary_reset_all_arms"] is True


def test_mixed_keeps_original_pool_while_stationary_has_only_fixed_ordinary_standing(source, tmp_path):
    original, parent = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(directory)
    for arm, domains in ARM_DOMAINS.items():
        for index, domain in enumerate(domains, 1):
            contract = phase_config(directory, manifest, arm, index)[2]
            if domain == "mixed":
                assert all(contract[key] == parent[key] for key in TASK_FIELDS)
                assert contract["new_asset_curriculum"] == parent["new_asset_curriculum"]
            else:
                assert contract["scene_groups"] == [{"name": "stand_305mm", "fraction": 1., "terrain": ["flat"]}]
                assert list(contract["skill_specs"]) == ["stand_305mm"]
                skill = contract["skill_specs"]["stand_305mm"]
                assert skill["kind"] == "stand" and skill["command"] == [0., 0., .305]
                assert skill["mode"] == 0 and skill["push_m_s"] == 0.
                assert skill["sample_amplitude"] is False and skill["sample_yaw_sign"] is False
                assert "height_sampling" not in skill
                assert skill["terrain_limits"] == parent["terrain_limits"]
                assert "start_stop_start_delay_seconds" not in contract["new_asset_curriculum"]
                assert {key: value for key, value in parent["new_asset_curriculum"].items()
                        if key != "start_stop_start_delay_seconds"} == contract["new_asset_curriculum"]
    original_cases = {case["name"]: case["environment"] for case in read(original / "study.json")["scenarios"]}
    for case in manifest["scenarios"]:
        environment = FrameTrainConfig.load(directory / case["config"]).environment
        assert {key: value for key, value in environment.items() if key not in {
            "snapshot", "snapshot_sha256"}} == original_cases[case["name"]]
        assert environment["snapshot_sha256"] == manifest["snapshot_identity"]["sha256"]
    assert manifest["evaluation"]["seeds"] == [8701, 9701]


def test_independent_prepared_snapshot_remains_valid_after_original_is_removed(source, tmp_path):
    original, _ = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    shutil.rmtree(original)
    assert validate_curriculum_study(directory)["protocol"]["total_updates"] == 1200


@pytest.mark.parametrize("route", ["src/wheeled_tasks/chassis/rewards.py",
                                   "contracts/transfer.train.json", "model/metadata/snapshot.json"])
def test_changed_source_bytes_rejected_before_output_is_created(source, tmp_path, route):
    original, _ = source
    (original / "snapshot" / route).write_text("{}")
    directory = tmp_path / "prepared"
    with pytest.raises(ValueError, match="dependency changed"):
        prepare_curriculum_study(original / "base.json", directory)
    assert not directory.exists()


@pytest.mark.parametrize("kwargs", [
    {"seeds": [1101, 1101, 1103]}, {"seeds": [1101, 1102]}, {"seeds": [8701, 1102, 1103]},
    {"seeds": [True, 1102, 1103]}, {"seeds": [-1, 1102, 1103]},
    {"warmup_updates": 0}, {"total_updates": 599}, {"warmup_updates": 1200},
    {"warmup_updates": True},
])
def test_insufficient_budget_or_nonindependent_seeds_are_rejected(source, tmp_path, kwargs):
    original, _ = source
    directory = tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_curriculum_study(original / "base.json", directory, **kwargs)
    assert not directory.exists()


def test_output_cannot_modify_source_or_overwrite_existing_directory(source, tmp_path):
    original, _ = source
    with pytest.raises(ValueError, match="immutable source snapshot"):
        prepare_curriculum_study(original / "base.json", original / "snapshot/derived")
    directory = tmp_path / "prepared"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("existing result")
    with pytest.raises(FileExistsError):
        prepare_curriculum_study(original / "base.json", directory)
    assert marker.read_text() == "existing result"


def test_contract_path_cannot_escape_snapshot_with_valid_external_sha(source, tmp_path):
    original, _ = source
    base_path = original / "base.json"
    base = read(base_path)
    base["environment"]["contract"] = "../outside.json"
    base["environment"]["contract_sha256"] = write_json(original / "outside.json", {"fixture": True})
    write_json(base_path, base)
    with pytest.raises(ValueError, match="escapes"):
        prepare_curriculum_study(base_path, tmp_path / "prepared")


def test_evaluation_coverage_cannot_silently_shrink(source, tmp_path):
    original, _ = source
    spec = read(original / "study.json")
    spec["scenarios"].pop()
    write_json(original / "study.json", spec)
    with pytest.raises(ValueError, match="all 50"):
        prepare_curriculum_study(original / "base.json", tmp_path / "prepared")


def test_unsealed_source_files_are_rejected(source, tmp_path):
    original, _ = source
    (original / "snapshot/unsealed.py").write_text("UNREVIEWED = True\n")
    with pytest.raises(ValueError, match="unsealed"):
        prepare_curriculum_study(original / "base.json", tmp_path / "prepared")


@pytest.mark.parametrize("mutation", ["config_whitespace", "phase_contract", "snapshot_extra", "manifest"])
def test_prepared_identity_catches_artifact_mutation(source, tmp_path, mutation):
    original, _ = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(directory)
    if mutation == "config_whitespace":
        route = manifest["arms"][0]["phases"][0]["config"]
        path = directory / route
        path.write_text(path.read_text() + "\n")
    elif mutation == "phase_contract":
        config = phase_config(directory, manifest, "stationary", 1)[1]
        path = directory / "snapshot" / config.environment["contract"]
        contract = read(path)
        contract["dense_tracking"]["height_weight"] = 0.
        write_json(path, contract)
    elif mutation == "snapshot_extra":
        (directory / "snapshot/unsealed.py").write_text("UNREVIEWED = True\n")
    else:
        manifest["protocol"]["total_updates"] = 1300
        write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        validate_curriculum_study(directory)


@pytest.mark.parametrize("mutation", ["phase_clock", "arm_domain", "budget", "reset", "factory", "evaluation"])
def test_resealing_manifest_does_not_authorize_semantic_protocol_changes(source, tmp_path, mutation):
    original, _ = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(directory)
    if mutation == "phase_clock":
        manifest["arms"][2]["phases"][1]["start_update"] = 0
    elif mutation == "arm_domain":
        manifest["arms"][1]["phases"][1]["domain"] = "mixed"
    elif mutation == "budget":
        manifest["protocol"]["planned_total_transitions"] -= 1
    elif mutation == "reset":
        manifest["protocol"]["same_phase_boundary_reset_all_arms"] = False
    elif mutation == "factory":
        manifest["environment_factory"] = "untrusted.module:environment"
    else:
        manifest["evaluation"]["steps"] = 2000
    seal_manifest(directory, manifest)
    with pytest.raises(ValueError):
        validate_curriculum_study(directory)


def test_resealed_config_cannot_change_learner_or_shared_environment_identity(source, tmp_path):
    original, _ = source
    directory = tmp_path / "prepared"
    prepare_curriculum_study(original / "base.json", directory)
    manifest = validate_curriculum_study(directory)
    route = manifest["arms"][1]["phases"][1]["config"]
    config = read(directory / route)
    config["ppo"]["learning_rate"] = 1e-4
    manifest["artifacts"][route] = write_json(directory / route, config)
    manifest["configs"][route] = digest(FrameTrainConfig.from_dict(config).to_dict())
    seal_manifest(directory, manifest)
    with pytest.raises(ValueError, match="share model/PPO/control"):
        validate_curriculum_study(directory)

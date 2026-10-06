"""Prepare and validate a paired learning-rate study; never execute workers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from .chassis_adapter import control_contract, network_variants
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest
from .frame_study import _configs, _validate_spec, plan_study, validate_study
from .transfer_study import _dependencies, _scene_quotas, _training_contract, _validate_parent


FORMAT = "transformer_rl.learning_rate_study"
HASH_FORMAT = "sorted_named_tensor_contents_v1"
DEFAULT_RATES = (1e-5, 3e-5, 1e-4)
DEFAULT_TRAINING_SEEDS = (1101, 1102, 1103)
DEFAULT_CONFIRMATORY_NOISE_SEEDS = (11701, 12701)
DEVELOPMENT_VALIDATION_SEEDS = (701, 1701)
DEVELOPMENT_FINAL_SEEDS = (2701, 3701)
UPDATES = 1200
ENVIRONMENTS = 1024
ROLLOUT_STEPS = 48


def _read(path):
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def _write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink paths are not permitted")
    return path.resolve()


def _inside(root, route):
    if (not isinstance(route, str) or not route or Path(route).is_absolute()
            or any(part in (".", "..") for part in route.split("/"))):
        raise ValueError("artifact route must be a plain relative path")
    path = _plain(root / route)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("artifact path escapes its root")
    return path


def _inventory(root):
    root = _plain(root)
    if not root.is_dir():
        raise ValueError("missing snapshot directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("snapshot/source symlinks are not permitted")
        if path.is_file():
            result[str(path.relative_to(root))] = _sha(path)
    return result


def _receipt(root, route, *, canonical=False):
    path = _inside(root, route)
    value = {"path": route, "sha256": _sha(path)}
    if canonical:
        value["canonical_sha256"] = digest(_read(path))
    return value


def _check_receipt(root, item, *, canonical=False):
    keys = {"path", "sha256", "canonical_sha256"} if canonical else {"path", "sha256"}
    if not isinstance(item, dict) or set(item) != keys:
        raise ValueError("invalid artifact receipt")
    actual = _receipt(root, item["path"], canonical=canonical)
    if item != actual:
        raise ValueError("artifact SHA mismatch")
    return _inside(root, item["path"])


def _seeds(values, name, *, size=None, minimum=1):
    if (not isinstance(values, (list, tuple)) or len(values) < minimum
            or (size is not None and len(values) != size)
            or any(type(value) is not int or not 0 <= value < 2**32 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} requires distinct unsigned integer seeds")
    return list(values)


def _rates(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)
            or len(set(values)) != 3):
        raise ValueError("learning rates require three distinct finite positive numbers")
    return [float(value) for value in values]


def _splits(training_seeds, confirmatory_noise_seeds):
    training_seeds = _seeds(training_seeds, "training", size=3)
    confirmation = _seeds(confirmatory_noise_seeds, "confirmatory noise", minimum=2)
    groups = [training_seeds, list(DEVELOPMENT_VALIDATION_SEEDS), list(DEVELOPMENT_FINAL_SEEDS),
              [4101, 5101], confirmation]
    if any(set(first) & set(second) for i, first in enumerate(groups) for second in groups[i + 1:]):
        raise ValueError("training, development, anchor and confirmation seed pools must be disjoint")
    return training_seeds, confirmation


def _gates(case):
    result = [{"path": "success_rate", "operator": "min", "value": .95},
              {"path": "metrics.height_abs_error.mean", "operator": "max", "value": .03},
              {"path": "metrics.vx_abs_error.mean", "operator": "max", "value": .15},
              {"path": "metrics.wz_abs_error.mean", "operator": "max", "value": .25},
              {"path": "metrics.tilt_angle.mean", "operator": "max", "value": .25}]
    if case["transfer_profile"]["base_case"].startswith("stand"):
        result.append({"path": "metrics.drift_m.max", "operator": "max", "value": .20})
    return result


def _inspect_parent(base, spec, snapshot, snapshot_reference):
    """Authenticate the materialized template using file-only transfer helpers."""
    from .transfer_profiles import build_transfer_cases, configure_transfer_evaluation

    _validate_spec(spec)
    config = FrameTrainConfig.from_dict(base)
    if config.to_dict() != base or spec["base_config"] != "base.json":
        raise ValueError("parent base must be the canonical materialized base.json")
    if (spec["environment_factory"] != "transformer_rl.chassis_adapter:make_env"
            or spec["variants"] != network_variants() or len(spec["stages"]) != 1
            or spec["stages"][0]["name"] != "transfer" or spec["stages"][0]["updates"] != UPDATES):
        raise ValueError("parent must contain the ten-network single-stage transfer recipe")
    identity = _read(_inside(snapshot, "snapshot.json"))
    if set(identity) != {"files", "sha256", "control", "parent_task_sha256", "contract_validator"}:
        raise ValueError("invalid parent snapshot identity")
    if (not isinstance(identity["files"], dict) or not identity["files"] or "snapshot.json" in identity["files"]
            or any(not isinstance(route, str) or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                   for route, sha in identity["files"].items())):
        raise ValueError("snapshot file manifest must exclude itself and contain file/SHA pairs")
    if identity["contract_validator"] != "packed_transfer_study" or digest(identity["files"]) != identity["sha256"]:
        raise ValueError("parent snapshot identity SHA differs")
    inventory = _inventory(snapshot)
    expected_inventory = {**identity["files"], "snapshot.json": _sha(snapshot / "snapshot.json")}
    if inventory != expected_inventory:
        raise ValueError("snapshot complete file inventory differs")
    for route in identity["files"]:
        _inside(snapshot, route)
    parent = _read(_inside(snapshot, "parent_task.json"))
    if _sha(snapshot / "parent_task.json") != identity["parent_task_sha256"]:
        raise ValueError("parent task SHA differs")
    dependencies, asset, _ = _dependencies(snapshot, parent)
    _validate_parent(parent, _read(_inside(snapshot, parent["control_math_source"])))
    calibration = {route: _sha(snapshot / route) for route in (
        "scripts/build_v6_height_lookup.py", "scripts/build_v6_recovery_profiles.py",
        "scripts/identify_pair_dynamics.py", "scripts/identify_wheel_response.py")
        if (snapshot / route).is_file()}
    parent_identity = {"task_contract_sha256": identity["parent_task_sha256"], "task_contract_id": parent["contract_id"],
        "asset_manifest_sha256": parent["asset_manifest_sha256"], "control_math_sha256": parent["control_math_sha256"],
        "authenticated_dependencies": dependencies, "calibration_source_sha256": calibration,
        "initialization": "scratch_no_parent_weights_optimizer_or_progress", "hardware_deployment_ready": False}
    if _read(_inside(snapshot, "parent_identity.json")) != parent_identity:
        raise ValueError("snapshot parent provenance identity differs")
    expected_train = _training_contract(parent, ENVIRONMENTS, UPDATES)
    expected_train["scene_groups"], counts = _scene_quotas(parent["scene_groups"], ENVIRONMENTS)
    training = _read(_inside(snapshot, "contracts/transfer.train.json"))
    if training != expected_train:
        raise ValueError("parent materialized training contract/clock/recipe differs")
    if control_contract(training, asset) != config.control or config.control != identity["control"]:
        raise ValueError("parent control contract differs")
    environment = {"snapshot": snapshot_reference, "snapshot_sha256": identity["sha256"],
                   "contract": "contracts/transfer.train.json", "contract_sha256": _sha(snapshot / "contracts/transfer.train.json"),
                   "num_envs": ENVIRONMENTS}
    if config.environment != environment or spec["stages"][0]["environment"] != {
            key: value for key, value in environment.items() if key not in ("snapshot", "snapshot_sha256")}:
        raise ValueError("parent environment path/SHA/count differs")
    expected_base = FrameTrainConfig.from_dict({
        "model": {"policy": {"architecture": "mlp", "history_length": 1, "mean_init_scale": 1.},
                  "critic_dim": 81, "critic_hidden": [256, 128, 64], "initial_std": 1.},
        "ppo": {"learning_rate": 3e-5, "gamma": math.sqrt(.99), "gae_lambda": math.sqrt(.95),
                "epochs": 5, "num_minibatches": 32, "value_coef": 4., "entropy_coef": .005,
                "target_kl": .01, "clip_ratio": .2},
        "control": config.control, "environment": environment}).to_dict()
    if base != expected_base:
        raise ValueError("parent optimization/exploration/critic recipe differs")
    if (spec["training"] != {"rollout_steps": ROLLOUT_STEPS, "checkpoint_interval": UPDATES,
            "max_seconds": 86400., "retention_coef": 0., "anchor_seeds": [4101, 5101]}
            or spec["evaluation"] != {"validation_seeds": list(DEVELOPMENT_VALIDATION_SEEDS),
            "seeds": list(DEVELOPMENT_FINAL_SEEDS), "steps": 4001, "settle_steps": 200,
            "min_steady_samples": 200, "min_completed_episodes": 8}):
        raise ValueError("parent rollout/evaluation protocol differs")
    expected_selection = {"min_training_seeds": 3, "std_penalty": 1., "latency_p99_ms": 8., "latency_max_ms": 10.,
        "max_deadline_misses": 0, "retention_score_tolerance": .25, "rollback_limit": 0,
        "objectives": [{"path": "success_rate", "direction": "maximize", "scale": 1., "weight": 4.},
                       {"path": "metrics.height_abs_error.mean", "direction": "minimize", "scale": .03, "weight": 1.},
                       {"path": "metrics.vx_abs_error.mean", "direction": "minimize", "scale": .15, "weight": 1.},
                       {"path": "metrics.issued_action_rate_rms.mean", "direction": "minimize", "scale": 100., "weight": .1}]}
    if (spec["selection"] != expected_selection or spec["execution"] != {"devices": ["cuda:0"],
            "worker_module": "transformer_rl.frame_process", "job_timeout_seconds": 86700.}):
        raise ValueError("parent execution/development selection recipe differs")
    cases = build_transfer_cases(parent["evaluation"]["cases"])
    if len(cases) != 50 or len(spec["scenarios"]) != 50:
        raise ValueError("parent must cover all 50 transfer cases")
    evaluation = configure_transfer_evaluation(training, cases)
    evaluation["auto_reset"] = False
    evaluation["record_diagnostics"] = evaluation["diagnostic_trace"] = True
    evaluation.pop("design_preflight", None)
    for case, scenario in zip(cases, spec["scenarios"]):
        expected = deepcopy(evaluation)
        expected["evaluation"].update(cases=[deepcopy(case)], stable_case_layout=False, episodes_per_case=8)
        expected["target_num_envs"] = 8
        expected["scene_groups"] = [{"name": case["name"], "fraction": 1., "terrain": [case.get("terrain", "flat")]}]
        route = f"contracts/eval.{case['name']}.json"
        if _read(_inside(snapshot, route)) != expected:
            raise ValueError("parent fixed evaluation contract differs")
        expected_scenario = {"name": case["name"], "environment": {"contract": route,
            "contract_sha256": _sha(snapshot / route), "num_envs": 8, "evaluation_batch": "transfer_all"},
            "gates": _gates(case), "require_steady": case.get("task") == "survive"}
        if scenario != expected_scenario:
            raise ValueError("parent task gates/evaluation scenario differs")
    if spec["stages"][0]["scenarios"] != [case["name"] for case in cases]:
        raise ValueError("parent stage evaluation coverage differs")
    _configs(config, spec)
    return {"snapshot_identity_sha256": identity["sha256"], "snapshot_files": inventory,
            "authenticated_dependencies": dependencies, "scene_counts": counts,
            "case_names": [case["name"] for case in cases], "cases": cases,
            "parent_identity": parent_identity, "parent_task": parent, "training_contract": training}


def _initializations(base, variants, seeds):
    """CPU-only initialization: preserve CPU RNG and never seed/query CUDA."""
    import torch
    from .frame_training import FrameActorCritic
    from .frame_workflow import _model_state_sha256

    runtime = {"torch_version": str(torch.__version__), "default_dtype": str(torch.get_default_dtype()),
               "device": "cpu", "hash_format": HASH_FORMAT,
               "seeding": "torch.random.default_generator.manual_seed; CPU fork_rng devices=[]"}
    values = []
    config = FrameTrainConfig.from_dict(base)
    with torch.random.fork_rng(devices=[]):
        with torch.device("cpu"):
            for variant in variants:
                configured = config.with_policy(variant["policy"])
                model_config = configured.model
                for seed in seeds:
                    torch.random.default_generator.manual_seed(seed)
                    model = FrameActorCritic(model_config)
                    values.append({"variant": variant["name"], "training_seed": seed,
                                   "model_sha256": digest(configured.to_dict()["model"]),
                                   "initial_model_sha256": _model_state_sha256(model)})
                    del model
    return runtime, values


def _inspect_design(design, snapshot, inspected):
    parent, training = inspected["parent_task"], inspected["training_contract"]
    expected = {"schema": "transformer_rl.transfer_study_design.v1", "policy_hz": 100,
        "physics_feedback_pd_hz": 200, "hardware_equivalence_verified": False,
        "updates_per_variant_seed": UPDATES, "rollout_samples": ENVIRONMENTS * ROLLOUT_STEPS,
        "planned_transitions_per_variant_seed": UPDATES * ENVIRONMENTS * ROLLOUT_STEPS,
        "single_stage_no_intermediate_gate": True, "all_cases_evaluated_including_failed_models": True,
        "optimization": {"learning_rate": 3e-5, "mean_init_scale": 1., "initial_std": 1.},
        "parent": inspected["parent_identity"],
        "delay_schedule": {"parent": parent["signal_delay"]["schedule"],
            "research": training["signal_delay"]["schedule"],
            "clock_units": "research_completed_actor_updates_not_parent_consumption"},
        "height_schedule": {"parent": parent["new_asset_curriculum"]["height_sampling_schedule"],
            "research": training["new_asset_curriculum"]["height_sampling_schedule"],
            "clock_units": "research_completed_actor_updates_from_zero"},
        "training_population": {"noise_effective_fraction_expectation": .25, "signal_delay_fraction": .75,
            "actual_population_required_in_runtime_receipt": True},
        "scene_allocation": {"method": "minimum_one_then_largest_remainder_integer_quotas",
            "parent_fractions": {group["name"]: group["fraction"] for group in parent["scene_groups"]},
            "requested_counts": inspected["scene_counts"],
            "frozen_allocator_verification": {"source_file": "src/wheeled_tasks/chassis/task.py",
                "source_sha256": _sha(_inside(snapshot, "src/wheeled_tasks/chassis/task.py")),
                "verified_counts": inspected["scene_counts"]}, "runtime_scene_group_counts_required": True},
        "recovery": "disabled_no_recovery_poses_in_flat_pool_50hz_FSM_not_adapted",
        "cases": [{"name": case["name"], "transfer_profile": case["transfer_profile"]} for case in inspected["cases"]]}
    if not isinstance(design, dict) or design != expected:
        raise ValueError("parent transfer design recipe/budget differs")


def _without_rate(config):
    config = deepcopy(config)
    del config["ppo"]["learning_rate"]
    return config


def _protocol(rates, training_seeds, confirmation, cases):
    samples = UPDATES * ENVIRONMENTS * ROLLOUT_STEPS
    return {"learning_rates": rates, "training_seeds": training_seeds,
            "variants": [variant["name"] for variant in network_variants()],
            "updates_per_job": UPDATES, "num_envs": ENVIRONMENTS, "rollout_steps": ROLLOUT_STEPS,
            "planned_transitions_per_job": samples, "expected_training_jobs": 90,
            "planned_total_transitions": samples * 90, "retention_coef": 0.,
            "initialization": "fresh_actor_critic_new_Adam_zero_progress_no_parent_checkpoint",
            "pairing": "same_variant_and_training_seed_identical_initial_model; only ppo.learning_rate changes",
            "rollout_matching": "equal_fresh_sample_budget_and_sampling_rules; trajectories_not_identical",
            "history": "MLP_1_history_models_31_repeat_first; Transformer_oldest",
            "environment_learning_rate": "fixed_legacy_3e-5_non_authoritative; FrameTrainConfig.ppo_controls_Adam",
            "development": {"validation_seeds": list(DEVELOPMENT_VALIDATION_SEEDS),
                "packed_final_seeds": list(DEVELOPMENT_FINAL_SEEDS),
                "both_pools_use": "development_only; generic_selector_has_no_confirmatory_qualification"},
            "confirmation": {"noise_seeds": confirmation, "scope": "heldout_noise_stream_only",
                "deterministic_cases": 38, "stochastic_noise_or_combined_cases": 12,
                "new_initial_conditions": False, "new_perturbation_domains": False,
                "requires": "seal_per_architecture_development_LR_choice_SHA_before_independent_test_controller",
                "no_reselection_on_confirmation": True},
            "evaluation": {"case_names": cases, "steps": 4001, "replicas_per_case": 8,
                "settle_steps": 200, "min_steady_samples": 200, "weighting": "predeclared_equal_case_eval_seed_training_seed"},
            "future_selection": {"implemented": False, "architecture_neutral": True,
                "eligibility": "complete_paired_three_training_seed_grid_and_all_original_joint_task_gates",
                "multi_dimensional_control": ["success_rate", "height_error", "vx_error", "wz_error", "tilt", "stationary_drift"],
                "control_diagnostics": ["full_interval", "steady_coverage", "response_censoring", "contact", "actuation_envelopes"],
                "ranking_rule_implemented": False, "no_eligible_candidate": "no_winner",
                "missing_evidence": "not_ready; never_drop_cases_or_seeds"},
            "future_runtime_checks": {"implemented": False,
                "required": ["actual_initial_model_SHA_matches_expected_and_all_three_LRs",
                    "run_config_and_sealed_learner_source_and_environment_factory",
                    "fresh_first_attempt_no_resume_initialize_restore_or_anchors",
                    "continuous_exact_recipe_resume_chain_if_interrupted",
                    "exact_1200_successful_updates_58982400_fresh_samples_no_budget_refund_or_reseed",
                    "actual_optimizer_steps_sample_count_first_final_KL_early_stops"]},
            "execution_implemented": False, "confirmation_controller_implemented": False,
            "independent_initial_condition_protocol_implemented": False,
            "formal_architecture_selection": False, "hardware_deployment_ready": False}


def prepare_learning_study(parent, directory, *, learning_rates=DEFAULT_RATES,
                           training_seeds=DEFAULT_TRAINING_SEEDS,
                           confirmatory_noise_seeds=DEFAULT_CONFIRMATORY_NOISE_SEEDS):
    parent, root = _plain(parent), _plain(directory)
    if root.exists():
        raise FileExistsError(root)
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("output must be independent of the immutable parent")
    rates = _rates(learning_rates)
    seeds, confirmation = _splits(training_seeds, confirmatory_noise_seeds)
    parent_files = {name: _receipt(parent, name, canonical=True)
                    for name in ("base.json", "study.json", "transfer_design.json", "snapshot/snapshot.json")}
    base, spec = _read(parent / "base.json"), _read(parent / "study.json")
    inspected = _inspect_parent(base, spec, parent / "snapshot", str(parent / "snapshot"))
    _inspect_design(_read(parent / "transfer_design.json"), parent / "snapshot", inspected)
    runtime, initializations = _initializations(base, spec["variants"], seeds)
    source = source_identity()
    root.mkdir(parents=True, exist_ok=False)
    (root / "parent").mkdir()
    for name, receipt in parent_files.items():
        target = root / "parent" / ("snapshot.json" if name == "snapshot/snapshot.json" else name)
        shutil.copyfile(_check_receipt(parent, receipt, canonical=True), target)
    shutil.copytree(parent / "snapshot", root / "snapshot")
    if _inventory(root / "snapshot") != inspected["snapshot_files"]:
        raise ValueError("snapshot changed while copying")
    common_base = deepcopy(base)
    common_base["environment"]["snapshot"] = str(root / "snapshot")
    child_spec = deepcopy(spec)
    child_spec["seeds"] = seeds
    child_spec["evaluation"]["validation_seeds"] = list(DEVELOPMENT_VALIDATION_SEEDS)
    child_spec["evaluation"]["seeds"] = list(DEVELOPMENT_FINAL_SEEDS)
    child_spec["selection"]["min_training_seeds"] = 3
    children, cells = [], []
    for index, rate in enumerate(rates):
        name = f"rate_{index:03d}"
        inputs = root / name
        inputs.mkdir()
        candidate = deepcopy(common_base)
        candidate["ppo"]["learning_rate"] = rate
        _write(inputs / "base.json", candidate)
        _write(inputs / "study.json", child_spec)
        packed = inputs / "study"
        plan_study(inputs / "study.json", packed)
        plan = validate_study(packed, source=True)
        children.append({"id": name, "learning_rate": rate, "root": f"{name}/study",
            "base": _receipt(root, f"{name}/base.json", canonical=True),
            "spec": _receipt(root, f"{name}/study.json", canonical=True),
            "plan": _receipt(root, f"{name}/study/plan.json", canonical=True),
            "configurations": [_receipt(root, f"{name}/study/{route}", canonical=True) for route in sorted(plan["configs"])]})
        for variant in spec["variants"]:
            for seed in seeds:
                initialization = next(item for item in initializations if item["variant"] == variant["name"] and item["training_seed"] == seed)
                cells.append({"rate_id": name, "learning_rate": rate, "variant": variant["name"], "training_seed": seed,
                    "job": f"{name}/study/jobs/{variant['name']}/seed_{seed}",
                    "training_config": _receipt(root, f"{name}/study/configs/{variant['name']}.train.transfer.json", canonical=True),
                    "initial_model_sha256": initialization["initial_model_sha256"],
                    "updates": UPDATES, "rollout_steps": ROLLOUT_STEPS, "num_envs": ENVIRONMENTS,
                    "planned_transitions": UPDATES * ROLLOUT_STEPS * ENVIRONMENTS})
    parent_receipts = [_receipt(root, f"parent/{name}", canonical=True)
                       for name in ("base.json", "study.json", "transfer_design.json", "snapshot.json")]
    manifest = {"format": FORMAT, "schema_version": 1,
        "parent": {"origin": str(parent), "snapshot_reference": str(parent / "snapshot"),
                   "files": parent_receipts, "authenticated_dependencies": inspected["authenticated_dependencies"]},
        "snapshot": {"path": "snapshot", "identity_sha256": inspected["snapshot_identity_sha256"],
                     "files": inspected["snapshot_files"]},
        "source": source, "initialization_runtime": runtime, "initializations": initializations,
        "protocol": _protocol(rates, seeds, confirmation, inspected["case_names"]),
        "children": children, "cells": cells}
    manifest["sha256"] = digest(manifest)
    _write(root / "manifest.json", manifest)
    return validate_learning_study(root)


def validate_learning_study(directory):
    root = _plain(directory)
    manifest = _read(_inside(root, "manifest.json"))
    expected_keys = {"format", "schema_version", "parent", "snapshot", "source", "initialization_runtime",
                     "initializations", "protocol", "children", "cells", "sha256"}
    if (not isinstance(manifest, dict) or set(manifest) != expected_keys or manifest["format"] != FORMAT
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or digest({key: value for key, value in manifest.items() if key != "sha256"}) != manifest["sha256"]):
        raise ValueError("learning study manifest format/SHA differs")
    parent = manifest["parent"]
    if not isinstance(parent, dict) or set(parent) != {"origin", "snapshot_reference", "files", "authenticated_dependencies"}:
        raise ValueError("invalid parent receipt schema")
    if not isinstance(parent["origin"], str) or parent["snapshot_reference"] != str(Path(parent["origin"]) / "snapshot"):
        raise ValueError("invalid parent snapshot reference")
    routes = [f"parent/{name}" for name in ("base.json", "study.json", "transfer_design.json", "snapshot.json")]
    if not isinstance(parent["files"], list) or [item.get("path") for item in parent["files"]] != routes:
        raise ValueError("parent receipt coverage differs")
    for item in parent["files"]:
        _check_receipt(root, item, canonical=True)
    base, spec = _read(root / "parent/base.json"), _read(root / "parent/study.json")
    inspected = _inspect_parent(base, spec, root / "snapshot", parent["snapshot_reference"])
    _inspect_design(_read(root / "parent/transfer_design.json"), root / "snapshot", inspected)
    if (_read(root / "parent/snapshot.json") != _read(root / "snapshot/snapshot.json")
            or parent["authenticated_dependencies"] != inspected["authenticated_dependencies"]
            or manifest["snapshot"] != {"path": "snapshot", "identity_sha256": inspected["snapshot_identity_sha256"],
                                      "files": inspected["snapshot_files"]}):
        raise ValueError("snapshot/parent dependency binding differs")
    protocol = manifest["protocol"]
    rates = _rates(protocol["learning_rates"])
    seeds, confirmation = _splits(protocol["training_seeds"], protocol["confirmation"]["noise_seeds"])
    if protocol != _protocol(rates, seeds, confirmation, inspected["case_names"]):
        raise ValueError("learning study protocol/budget/split differs")
    runtime, initializations = _initializations(base, spec["variants"], seeds)
    if manifest["initialization_runtime"] != runtime or manifest["initializations"] != initializations:
        raise ValueError("CPU initialization runtime/model SHA differs; do not reseed or refund")
    current_source = source_identity()
    if manifest["source"] != current_source:
        raise ValueError("current learner source differs; prepare a new frozen study")
    common_base = deepcopy(base)
    common_base["environment"]["snapshot"] = str(root / "snapshot")
    child_spec = deepcopy(spec)
    child_spec["seeds"] = seeds
    child_spec["evaluation"]["validation_seeds"] = list(DEVELOPMENT_VALIDATION_SEEDS)
    child_spec["evaluation"]["seeds"] = list(DEVELOPMENT_FINAL_SEEDS)
    child_spec["selection"]["min_training_seeds"] = 3
    if not isinstance(manifest["children"], list) or len(manifest["children"]) != 3:
        raise ValueError("three child studies are required")
    expected_cells, paired = [], {}
    for index, (rate, child) in enumerate(zip(rates, manifest["children"])):
        name = f"rate_{index:03d}"
        if (not isinstance(child, dict) or set(child) != {"id", "learning_rate", "root", "base", "spec", "plan", "configurations"}
                or child["id"] != name or child["learning_rate"] != rate or child["root"] != f"{name}/study"):
            raise ValueError("child study identity/rate differs")
        for field, route in (("base", f"{name}/base.json"), ("spec", f"{name}/study.json"), ("plan", f"{name}/study/plan.json")):
            if child[field].get("path") != route:
                raise ValueError("child artifact path differs")
            _check_receipt(root, child[field], canonical=True)
        expected_base = deepcopy(common_base)
        expected_base["ppo"]["learning_rate"] = rate
        if _read(root / f"{name}/base.json") != expected_base or _read(root / f"{name}/study.json") != child_spec:
            raise ValueError("child base/spec differs from the LR-only recipe")
        packed = _inside(root, child["root"])
        if _inventory(packed / "jobs"):
            raise ValueError("preparation-only validation refuses runtime job outputs")
        if _inventory(packed / "policy_source") != {f"transformer_rl/{route}": sha
                for route, sha in manifest["source"]["files"].items()}:
            raise ValueError("child frozen source complete inventory differs")
        plan = validate_study(packed, source=True)
        if plan["base"] != expected_base or plan["spec"] != child_spec or plan["source"] != manifest["source"]:
            raise ValueError("child plan/source binding differs")
        expected_receipts = [_receipt(root, f"{name}/study/{route}", canonical=True) for route in sorted(plan["configs"])]
        if child["configurations"] != expected_receipts:
            raise ValueError("child configuration raw/canonical SHA coverage differs")
        for route in plan["configs"]:
            _inside(root, f"{name}/study/{route}")
            configured = _read(packed / route)
            normalized = _without_rate(configured)
            if configured["ppo"]["learning_rate"] != rate or (route in paired and paired[route] != normalized):
                raise ValueError("paired configs differ in fields other than PPO learning_rate")
            paired[route] = normalized
        for variant in spec["variants"]:
            for seed in seeds:
                initialization = next(item for item in initializations if item["variant"] == variant["name"] and item["training_seed"] == seed)
                job_route = f"{name}/study/jobs/{variant['name']}/seed_{seed}"
                _inside(root, job_route)
                expected_cells.append({"rate_id": name, "learning_rate": rate, "variant": variant["name"], "training_seed": seed,
                    "job": job_route, "training_config": _receipt(root, f"{name}/study/configs/{variant['name']}.train.transfer.json", canonical=True),
                    "initial_model_sha256": initialization["initial_model_sha256"],
                    "updates": UPDATES, "rollout_steps": ROLLOUT_STEPS, "num_envs": ENVIRONMENTS,
                    "planned_transitions": UPDATES * ROLLOUT_STEPS * ENVIRONMENTS})
    if manifest["cells"] != expected_cells or len(expected_cells) != 90:
        raise ValueError("complete paired 90-cell training grid differs")
    return {"root": str(root), "manifest": str(root / "manifest.json"), "sha256": manifest["sha256"],
            "status": "prepared", "expected_training_jobs": 90, "initial_models": 30,
            "planned_total_transitions": protocol["planned_total_transitions"],
            "training_started": False, "queued": False, "execution_implemented": False,
            "selection_implemented": False, "confirmation_scope": "heldout_noise_stream_only",
            "formal_architecture_selection": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    prepare = operations.add_parser("prepare", help="write a new independent preparation; never train or queue")
    prepare.add_argument("--parent", required=True, type=Path)
    prepare.add_argument("--directory", required=True, type=Path)
    prepare.add_argument("--learning-rates", type=float, nargs=3, default=list(DEFAULT_RATES))
    prepare.add_argument("--training-seeds", type=int, nargs=3, default=list(DEFAULT_TRAINING_SEEDS))
    prepare.add_argument("--confirmatory-noise-seeds", type=int, nargs="+", default=list(DEFAULT_CONFIRMATORY_NOISE_SEEDS))
    validate = operations.add_parser("validate", help="authenticate preparation using files and CPU initial models")
    validate.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        result = prepare_learning_study(args.parent, args.directory, learning_rates=args.learning_rates,
            training_seeds=args.training_seeds, confirmatory_noise_seeds=args.confirmatory_noise_seeds)
    else:
        result = validate_learning_study(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

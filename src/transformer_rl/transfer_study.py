"""Prepare an isolated study from an already materialized V6 flat contract.

Preparation performs file and configuration checks only. It neither imports the
simulator nor changes the source checkout, its checkpoints or its campaign.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys

from .chassis_adapter import control_contract, merge_evaluation_contracts, network_variants
from .frame_config import FrameTrainConfig, digest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    # Reject nonfinite JSON constants before using a training contract.
    json.dumps(value, allow_nan=False)
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _inside(root, route):
    if not isinstance(route, str) or Path(route).is_absolute():
        raise ValueError("source dependencies require relative paths")
    result = (root / route).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError("source dependency escapes its root")
    return result


def _checked(files, root, route, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"invalid dependency SHA: {route}")
    path = _inside(root, route)
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"source dependency changed: {route}")
    previous = files.setdefault(route, expected)
    if previous != expected:
        raise ValueError(f"conflicting dependency identity: {route}")
    return path


def _dependencies(root, base):
    """Authenticate explicit file/SHA pairs recursively, including disabled modules."""
    files = {}
    asset = _inside(root, base["asset_directory"])
    asset_manifest = _checked(files, root, base["asset_directory"] + "/manifest.json",
                              base["asset_manifest_sha256"])
    manifest = _read(asset_manifest)
    if not isinstance(manifest.get("files_sha256"), dict) or not manifest["files_sha256"]:
        raise ValueError("asset requires a sealed file manifest")
    for route, expected in manifest["files_sha256"].items():
        _checked(files, root, str(Path(base["asset_directory"]) / route), expected)
    for required in ("model_spec.json", "fit_10mpa.json", "gas_spring_binding.json"):
        if required not in manifest["files_sha256"]:
            raise ValueError(f"asset does not seal its spring/mechanical dependency: {required}")
    queue, seen = [base], set()
    while queue:
        value = queue.pop()
        if isinstance(value, list):
            queue.extend(value)
            continue
        if not isinstance(value, dict):
            continue
        queue.extend(v for v in value.values() if isinstance(v, (dict, list)))
        pairs = []
        if isinstance(value.get("file"), str) and "sha256" in value:
            pairs.append((value["file"], value["sha256"]))
        for key, route in value.items():
            if not isinstance(route, str):
                continue
            stem = key[:-5] if key.endswith("_file") else key[:-7] if key.endswith("_source") else None
            if stem is not None and stem + "_sha256" in value:
                pairs.append((route, value[stem + "_sha256"]))
        # Evidence bundles explicitly seal paths relative to the source root.
        for route, expected in value.get("files_sha256", {}).items():
            pairs.append((route, expected))
        for route, expected in pairs:
            path = _checked(files, root, route, expected)
            if route not in seen and path.suffix == ".json":
                seen.add(route)
                queue.append(_read(path))
    return files, manifest, asset


def _validate_parent(base, control):
    fields = ("physics_dt", "policy_dt", "history_length", "actor_frame_dim", "actor_dim", "critic_dim", "action_dim")
    if tuple(base.get(key) for key in fields) != (.005, .02, 1, 35, 35, 81, 6):
        raise ValueError("transfer study requires materialized V6 50 Hz/200 Hz 35D/81D/6D parent")
    if base.get("evaluation_exact_cases") or not base.get("scene_groups") or not base.get("skill_specs"):
        raise ValueError("parent must be a materialized flat training worker")
    if any(spec.get("recovery_pose") or spec.get("kind") in ("recovery_handoff", "jump", "body_climb")
           for spec in base["skill_specs"].values()):
        raise ValueError("flat study cannot import recovery, jump or terrain task curricula")
    if not base.get("actuator_response", {}).get("enabled"):
        raise ValueError("transfer study requires the adopted wheel response")
    if not base.get("actuator_response_evidence"):
        raise ValueError("wheel response requires authenticated evidence")
    for name in ("dynamics_randomization", "contact_domain", "signal_perturbations", "signal_delay", "command_transport"):
        if not base.get(name, {}).get("enabled"):
            raise ValueError(f"parent does not enable V6 {name}")
    if base.get("usb_transport", {}).get("enabled") or base["signal_perturbations"].get("max_delay_steps") != 0:
        raise ValueError("V6 transfer contract cannot add a second delayed-feedback pipeline")
    delay = base["signal_delay"]
    expected_profiles = [{"probability": .6, "observation_ms": [0., 20.], "action_ms": [0., 15.]},
                         {"probability": .35, "observation_ms": [20., 40.], "action_ms": [15., 30.]},
                         {"probability": .05, "observation_ms": [40., 80.], "action_ms": [30., 60.]}]
    if (delay.get("version") != "physical_signal_delay_v1" or delay.get("profiles") != expected_profiles
            or delay.get("enabled_fraction") != .75 or delay.get("jitter_ms") != 5.
            or delay.get("feedback_pd") != "fresh_physics_feedback"):
        raise ValueError("parent does not contain the audited V6 delay distribution")
    timing = control.get("timing", {})
    if (timing.get("physics_dt"), timing.get("policy_dt"), timing.get("pc_control_dt")) != (.005, .02, .005):
        raise ValueError("parent control timing differs from materialized worker")


def _training_contract(parent, num_envs, updates):
    config = deepcopy(parent)
    # Legacy task campaign identities and budget references describe a different
    # learner and its consumed history. They do not enter scratch research clocks.
    for key in ("task_campaign", "budget_spec_file", "budget_spec_sha256", "budget_stage_id", "design_preflight"):
        config.pop(key, None)
    for key in ("global_actor_update", "global_actor_update_offset", "stage_actor_update", "task_actor_update",
                "actor_updates_consumed", "parent_updates", "training_transitions", "consumed_updates"):
        config[key] = 0
    config.update(contract_id="packed-transfer-study", policy_dt=.01, physics_dt=.005,
                  actor_dim=35, actor_frame_dim=35, critic_dim=81, action_dim=6, history_length=1,
                  target_num_envs=num_envs, num_steps_per_env=48, num_mini_batches=32,
                  ppo_minibatch_samples=num_envs * 48 // 32, curriculum_reference_batch=num_envs * 48,
                  total_updates=updates, checkpoint_interval=updates, checkpoint_first_update=updates,
                  save_interval=updates, learning_rate=3e-5, initial_noise_std=1.,
                  critic_warmup_updates=0, resume_critic_warmup_updates=0,
                  record_diagnostics=True, diagnostic_trace=True, auto_reset=False,
                  action_difference_reference_dt=.02, requires_design_preflight=True,
                  hardware_deployment_ready=False, research_only=True)
    config["signal_delay"]["schedule"] = {"clock": "domain_actor_updates", "start": 200, "end": 600}
    curriculum = config["new_asset_curriculum"]
    curriculum["stage_start_actor_update"] = 0
    # The frozen flat pool has no recovery poses; the legacy recovery FSM itself
    # requires 50 Hz. Its sealed files remain in the snapshot for provenance.
    if config.get("recovery_training"):
        config["recovery_training"]["enabled"] = False
    config["jump_assist"]["enabled"] = False
    config["step_assist"]["enabled"] = False
    return config


def _scene_quotas(groups, num_envs):
    """Encode fair integer quotas for the frozen source's floor-and-tail allocator."""
    if (not groups or num_envs < len(groups)
            or not math.isclose(sum(group["fraction"] for group in groups), 1., abs_tol=1e-9)
            or any(group["fraction"] <= 0 for group in groups)):
        raise ValueError("flat pool requires valid fractions and at least one environment per skill")
    ideals = [num_envs * group["fraction"] for group in groups]
    counts = [max(1, math.floor(value)) for value in ideals]
    while sum(counts) < num_envs:
        index = max(range(len(groups)), key=lambda i: (ideals[i] - counts[i], -i))
        counts[index] += 1
    while sum(counts) > num_envs:
        eligible = [i for i, count in enumerate(counts) if count > 1]
        index = min(eligible, key=lambda i: (ideals[i] - counts[i], i))
        counts[index] -= 1
    encoded = deepcopy(groups)
    for group, count in zip(encoded[:-1], counts[:-1]):
        fraction = count / num_envs
        if math.floor(num_envs * fraction) < count:
            fraction = math.nextafter(fraction, math.inf)
        group["fraction"] = fraction
    encoded[-1]["fraction"] = 1. - sum(group["fraction"] for group in encoded[:-1])
    if [math.floor(num_envs * group["fraction"]) for group in encoded[:-1]] != counts[:-1]:
        raise ValueError("integer pool quota cannot be represented by frozen allocator")
    return encoded, dict(zip((group["name"] for group in groups), counts))


def _check_source_scene_counts(source_root, groups, count, expected):
    """Execute the frozen pure task helper, never its simulator environment."""
    path = source_root / "src/wheeled_tasks/chassis/task.py"
    module_name = "_transformer_rl_transfer_task_" + _sha(path)[:16]
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        actual = Counter(name for name, _ in module.choose_scene_groups(groups, count))
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    if dict(actual) != expected or sum(actual.values()) != count:
        raise ValueError("frozen scene allocator differs from requested integer quotas")
    return {"source_file": "src/wheeled_tasks/chassis/task.py", "source_sha256": _sha(path),
            "verified_counts": dict(actual)}


def prepare_transfer_study(source_root, task_contract_path, directory, *, num_envs=1024,
                           evaluation_replicas=8, updates=1200, seeds=None):
    """Freeze V6 mechanics and prepare all architectures; never start training."""
    from .transfer_profiles import build_transfer_cases, configure_transfer_evaluation

    source_root, directory = Path(source_root).resolve(), Path(directory).resolve()
    task_path = Path(task_contract_path)
    if not task_path.is_absolute():
        task_path = _inside(source_root, str(task_path))
    if (type(num_envs) is not int or num_envs < 32 or num_envs * 48 % 32
            or type(evaluation_replicas) is not int or evaluation_replicas < 4
            or type(updates) is not int or updates < 600):
        raise ValueError("study requires complete minibatches, >=4 replicas and >=600 updates")
    seeds = [1101] if seeds is None else list(seeds)
    reserved = {4101, 5101, 701, 1701, 2701, 3701}
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 or seed in reserved for seed in seeds)):
        raise ValueError("training seeds must be unique and separate from evaluation seeds")
    parent = _read(task_path)
    dependencies, manifest, source_asset = _dependencies(source_root, parent)
    control_path = _inside(source_root, parent["control_math_source"])
    _validate_parent(parent, _read(control_path))
    evidence = _read(_inside(source_root, parent["actuator_response_evidence"]["file"]))
    if (evidence.get("status") != "validated_for_research" or evidence.get("hardware_release") is not False
            or evidence.get("actuator_response") != parent["actuator_response"]
            or evidence.get("adopted_joint_names") != ["L_joint3", "R_joint3"]):
        raise ValueError("adopted wheel response does not match its research evidence")
    train = _training_contract(parent, num_envs, updates)
    train["scene_groups"], requested_scene_counts = _scene_quotas(parent["scene_groups"], num_envs)
    scene_verification = _check_source_scene_counts(source_root, train["scene_groups"], num_envs, requested_scene_counts)
    cases = build_transfer_cases(parent["evaluation"]["cases"])
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("transfer cases require unique names")
    eval_base = configure_transfer_evaluation(train, cases)
    # Exact evaluation starts at full delay strength via the profile callback;
    # neither checkpoint training history nor baseline offsets determine it.
    eval_base["auto_reset"] = False
    eval_base["record_diagnostics"] = eval_base["diagnostic_trace"] = True
    eval_base.pop("design_preflight", None)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "snapshot"
    snapshot.mkdir()
    shutil.copytree(source_root / "src/wheeled_tasks", snapshot / "src/wheeled_tasks",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    asset_target = snapshot / parent["asset_directory"]
    asset_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_asset, asset_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for route, expected in dependencies.items():
        target = snapshot / route
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_inside(source_root, route), target)
        if _sha(target) != expected:
            raise ValueError(f"copied dependency identity changed: {route}")
    # Freeze available calibration source bytes without implying historical
    # generator hashes or hardware calibration have been revalidated.
    calibration_sources = {}
    for pattern in ("scripts/build_v6_height_lookup.py", "scripts/build_v6_recovery_profiles.py",
                    "scripts/identify_pair_dynamics.py", "scripts/identify_wheel_response.py"):
        source = source_root / pattern
        if source.is_file():
            target = snapshot / pattern
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            calibration_sources[pattern] = _sha(target)
    shutil.copy2(task_path, snapshot / "parent_task.json")
    parent_identity = {"task_contract_sha256": _sha(task_path), "task_contract_id": parent["contract_id"],
                       "asset_manifest_sha256": parent["asset_manifest_sha256"],
                       "control_math_sha256": parent["control_math_sha256"],
                       "authenticated_dependencies": dependencies, "calibration_source_sha256": calibration_sources,
                       "initialization": "scratch_no_parent_weights_optimizer_or_progress",
                       "hardware_deployment_ready": False}
    _write(snapshot / "parent_identity.json", parent_identity)
    train_route = "contracts/transfer.train.json"
    _write(snapshot / train_route, train)
    stages = [{"name": "transfer", "updates": updates,
               "environment": {"contract": train_route, "contract_sha256": _sha(snapshot / train_route), "num_envs": num_envs},
               "scenarios": [case["name"] for case in cases]}]
    scenarios, environments = [], []
    for case in cases:
        config = deepcopy(eval_base)
        config["evaluation"]["cases"] = [deepcopy(case)]
        config["evaluation"]["stable_case_layout"] = False
        config["evaluation"]["episodes_per_case"] = evaluation_replicas
        config["target_num_envs"] = evaluation_replicas
        config["scene_groups"] = [{"name": case["name"], "fraction": 1., "terrain": [case.get("terrain", "flat")]}]
        route = f"contracts/eval.{case['name']}.json"
        _write(snapshot / route, config)
        environment = {"contract": route, "contract_sha256": _sha(snapshot / route),
                       "num_envs": evaluation_replicas, "evaluation_batch": "transfer_all"}
        environments.append(environment)
        gates = [{"path": "success_rate", "operator": "min", "value": .95},
                 {"path": "metrics.height_abs_error.mean", "operator": "max", "value": .03},
                 {"path": "metrics.vx_abs_error.mean", "operator": "max", "value": .15},
                 {"path": "metrics.wz_abs_error.mean", "operator": "max", "value": .25},
                 {"path": "metrics.tilt_angle.mean", "operator": "max", "value": .25}]
        if case.get("transfer_profile", {}).get("base_case", case["name"]).startswith("stand"):
            gates.append({"path": "metrics.drift_m.max", "operator": "max", "value": .20})
        scenarios.append({"name": case["name"], "environment": environment, "gates": gates,
                          "require_steady": case.get("task") == "survive"})
    # A suite may share a simulator only after its common configuration matches.
    merge_evaluation_contracts(snapshot, environments)
    control = control_contract(train, manifest)
    files = {str(path.relative_to(snapshot)): _sha(path) for path in sorted(snapshot.rglob("*"))
             if path.is_file()}
    identity = {"files": files, "sha256": digest(files), "control": control,
                "parent_task_sha256": _sha(task_path), "contract_validator": "packed_transfer_study"}
    _write(snapshot / "snapshot.json", identity)
    base = {"model": {"policy": {"architecture": "mlp", "history_length": 1, "mean_init_scale": 1.},
                      "critic_dim": 81, "critic_hidden": [256, 128, 64], "initial_std": 1.},
            "ppo": {"learning_rate": 3e-5, "gamma": math.sqrt(.99), "gae_lambda": math.sqrt(.95),
                    "epochs": 5, "num_minibatches": 32, "value_coef": 4., "entropy_coef": .005,
                    "target_kl": .01, "clip_ratio": .2},
            "control": control, "environment": {"snapshot": str(snapshot), "snapshot_sha256": identity["sha256"],
                                                   **stages[0]["environment"]}}
    _write(directory / "base.json", FrameTrainConfig.from_dict(base).to_dict())
    spec = {"base_config": "base.json", "environment_factory": "transformer_rl.chassis_adapter:make_env",
            "variants": network_variants(), "seeds": seeds, "stages": stages, "scenarios": scenarios,
            "training": {"rollout_steps": 48, "checkpoint_interval": updates, "max_seconds": 86400.,
                         "retention_coef": 0., "anchor_seeds": [4101, 5101]},
            "evaluation": {"validation_seeds": [701, 1701], "seeds": [2701, 3701], "steps": 4001,
                           "settle_steps": 200, "min_steady_samples": 200,
                           "min_completed_episodes": evaluation_replicas},
            "execution": {"devices": ["cuda:0"], "worker_module": "transformer_rl.frame_process",
                          "job_timeout_seconds": 86700.},
            "selection": {"min_training_seeds": 3, "std_penalty": 1., "latency_p99_ms": 8., "latency_max_ms": 10.,
                          "max_deadline_misses": 0, "retention_score_tolerance": .25, "rollback_limit": 0,
                          "objectives": [{"path": "success_rate", "direction": "maximize", "scale": 1., "weight": 4.},
                                         {"path": "metrics.height_abs_error.mean", "direction": "minimize", "scale": .03, "weight": 1.},
                                         {"path": "metrics.vx_abs_error.mean", "direction": "minimize", "scale": .15, "weight": 1.},
                                         {"path": "metrics.issued_action_rate_rms.mean", "direction": "minimize", "scale": 100., "weight": .1}]}}
    _write(directory / "study.json", spec)
    design = {"schema": "transformer_rl.transfer_study_design.v1", "parent": parent_identity,
              "policy_hz": 100, "physics_feedback_pd_hz": 200, "hardware_equivalence_verified": False,
              "updates_per_variant_seed": updates, "rollout_samples": num_envs * 48,
              "planned_transitions_per_variant_seed": updates * num_envs * 48,
              "single_stage_no_intermediate_gate": True, "all_cases_evaluated_including_failed_models": True,
              "optimization": {"learning_rate": 3e-5, "mean_init_scale": 1., "initial_std": 1.},
              "delay_schedule": {"parent": parent["signal_delay"]["schedule"],
                                 "research": train["signal_delay"]["schedule"],
                                 "clock_units": "research_completed_actor_updates_not_parent_consumption"},
              "height_schedule": {"parent": parent["new_asset_curriculum"]["height_sampling_schedule"],
                                  "research": train["new_asset_curriculum"]["height_sampling_schedule"],
                                  "clock_units": "research_completed_actor_updates_from_zero"},
              "training_population": {"noise_effective_fraction_expectation": .25, "signal_delay_fraction": .75,
                                      "actual_population_required_in_runtime_receipt": True},
              "scene_allocation": {"method": "minimum_one_then_largest_remainder_integer_quotas",
                                   "parent_fractions": {group["name"]: group["fraction"] for group in parent["scene_groups"]},
                                   "requested_counts": requested_scene_counts,
                                   "frozen_allocator_verification": scene_verification,
                                   "runtime_scene_group_counts_required": True},
              "recovery": "disabled_no_recovery_poses_in_flat_pool_50hz_FSM_not_adapted",
              "cases": [{"name": case["name"], "transfer_profile": case.get("transfer_profile")} for case in cases]}
    _write(directory / "transfer_design.json", design)
    return {"directory": str(directory), "spec": str(directory / "study.json"),
            "snapshot_sha256": identity["sha256"], "parent_task_sha256": _sha(task_path),
            "policy_hz": 100, "physics_hz": 200, "history_frames": 31,
            "variants": len(spec["variants"]), "scenarios": len(scenarios),
            "updates_per_job": updates, "training_started": False}

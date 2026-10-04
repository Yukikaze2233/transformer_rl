"""Freeze a three-arm task-schedule study without touching its source study.

All arms use the same Gated policy, reward, mechanics and cumulative delay
clock. Each arm resets its environment/history at the common phase boundary;
the runner restores learning state rather than starting a second optimizer.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil

from .chassis_adapter import network_variants
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest, json_bytes


ARM_DOMAINS = {"mixed": ("mixed", "mixed"),
               "stationary": ("stationary", "stationary"),
               "pretrain": ("stationary", "mixed")}
EVALUATION_SEEDS = (8701, 9701)
ROLLOUT_STEPS = 48
TRANSITIONS_PER_UPDATE = 49152
TASK_FIELDS = {"scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions"}
CLOCK_FIELDS = {"global_actor_update", "stage_actor_update", "task_actor_update",
                "actor_updates_consumed", "parent_updates", "consumed_updates",
                "global_actor_update_offset", "training_transitions"}


def _read(path):
    value = json.loads(Path(path).read_text())
    json_bytes(value)
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(root, route):
    if not isinstance(route, str) or Path(route).is_absolute():
        raise ValueError("curriculum artifacts require relative paths")
    path = (root / route).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError("curriculum artifact escapes its root")
    return path


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _files(root):
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "snapshot.json"
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def _verify_files(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("snapshot requires sealed files")
    for route, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid snapshot file SHA")
        path = _inside(root, route)
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"frozen curriculum dependency changed: {route}")


def _snapshot(root, expected=None):
    identity = _read(root / "snapshot.json")
    if identity.get("sha256") != digest(identity.get("files")) or (
            expected is not None and identity["sha256"] != expected):
        raise ValueError("snapshot identity differs from its sealed invocation")
    _verify_files(root, identity["files"])
    if _files(root) != identity["files"]:
        raise ValueError("snapshot contains unsealed files")
    return identity


def _parameters(warmup, total, seeds):
    if (type(warmup) is not int or type(total) is not int
            or not 0 < warmup < total or total < 600):
        raise ValueError("study requires a positive split and >=600 cumulative updates")
    reserved = {*EVALUATION_SEEDS, 701, 1701, 2701, 3701, 4101, 5101}
    if (not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 or seed in reserved for seed in seeds)):
        raise ValueError("study requires >=3 unique training seeds disjoint from evaluation")


def _parent(base, snapshot, identity):
    environment = base.environment
    if (environment.get("snapshot_sha256") != identity["sha256"]
            or environment.get("num_envs") != 1024):
        raise ValueError("curriculum requires the frozen 1024-environment transfer study")
    path = _inside(snapshot, environment["contract"])
    if _sha(path) != environment.get("contract_sha256"):
        raise ValueError("parent training contract changed")
    parent = _read(path)
    if (parent.get("contract_id"), parent.get("physics_dt"), parent.get("policy_dt"),
            parent.get("target_num_envs"), parent.get("num_steps_per_env"),
            base.model.frame_dim, base.model.critic_dim, base.model.action_dim,
            base.control["policy_dt_s"]) != ("packed-transfer-study", .005, .01, 1024, 48, 35, 81, 6, .01):
        raise ValueError("parent must retain the V6 transfer timing, observation and rollout contract")
    if (parent.get("evaluation_exact_cases") or not parent.get("dense_tracking", {}).get("enabled")
            or not parent.get("actuator_response", {}).get("enabled")
            or parent.get("global_actor_update_offset", 0) != 0
            or parent.get("new_asset_curriculum", {}).get("stage_start_actor_update", 0) != 0
            or any(parent.get(key) for key in ("task_campaign", "task_allocator", "performance_curriculum"))):
        raise ValueError("parent is not a scratch flat transfer training contract")
    if not parent.get("scene_groups") or set(parent["skill_specs"]) != {
            group["name"] for group in parent["scene_groups"]}:
        raise ValueError("parent mixed pool requires one skill per scene group")
    if any(group["terrain"] != ["flat"] for group in parent["scene_groups"]):
        raise ValueError("curriculum comparison only changes ordinary flat task scheduling")
    if base.control != identity.get("control"):
        raise ValueError("parent action/observation control differs from its snapshot")
    return parent


def _contract(parent, domain, start, total):
    result = deepcopy(parent)
    if domain == "stationary":
        result["scene_groups"] = [{"name": "stand_305mm", "fraction": 1., "terrain": ["flat"]}]
        result["skill_specs"] = {"stand_305mm": {
            "kind": "stand", "command": [0., 0., .305], "mode": 0,
            "sample_amplitude": False, "sample_yaw_sign": False, "push_m_s": 0.,
            "episode_seconds": 20., "terrain_limits": deepcopy(parent["terrain_limits"])}}
        result["behavior_pool_membership"] = {"stand_305mm": "stationary"}
        result["behavior_pool_fractions"] = {"stationary": 1.}
        # The native startup check requires a start/stop task whenever this
        # sampling timer is declared. It has no meaning for fixed standing;
        # all other curriculum settings, including the delay clock, stay frozen.
        result.get("new_asset_curriculum", {}).pop("start_stop_start_delay_seconds", None)
    elif domain != "mixed":
        raise ValueError("unknown curriculum task domain")
    # Constructor/reset sampling precedes the rollout progress hook. Initial
    # clocks therefore describe the consumed global budget, including phase 2.
    for name in CLOCK_FIELDS - {"global_actor_update_offset", "training_transitions"}:
        result[name] = start
    result["global_actor_update_offset"] = 0
    result["training_transitions"] = start * TRANSITIONS_PER_UPDATE
    result["total_updates"] = total
    return result


def _protected(parent):
    result = {key: deepcopy(value) for key, value in parent.items()
              if key not in TASK_FIELDS | CLOCK_FIELDS | {"total_updates"}}
    result.get("new_asset_curriculum", {}).pop("start_stop_start_delay_seconds", None)
    return result


def _allocation(snapshot, contract):
    # run_path compiles a pure task helper directly and does not create a .pyc
    # in the original immutable snapshot as SourceFileLoader would.
    helper = runpy.run_path(str(snapshot / "src/wheeled_tasks/chassis/task.py"))
    actual = Counter(name for name, terrain in helper["choose_scene_groups"](
        contract["scene_groups"], contract["target_num_envs"]) if terrain == "flat")
    if sum(actual.values()) != 1024 or set(actual) != set(contract["skill_specs"]):
        raise ValueError("frozen scene allocator does not preserve the task pool")
    return dict(actual)


def _gated(base):
    policy = next(variant["policy"] for variant in network_variants()
                  if variant["name"] == "transformer_gated")
    return base.with_policy(policy)


def prepare_curriculum_study(base_config_path, directory, *, warmup_updates=400,
                             total_updates=1200, seeds=(1101, 1102, 1103)):
    """Prepare mixed, stationary and pretrain arms; never launch training."""
    seeds = list(seeds)
    _parameters(warmup_updates, total_updates, seeds)
    base_path, directory = Path(base_config_path).resolve(), Path(directory).resolve()
    base = FrameTrainConfig.load(base_path)
    source_snapshot = Path(base.environment["snapshot"]).resolve()
    if directory == source_snapshot or directory.is_relative_to(source_snapshot):
        raise ValueError("output cannot be inside the immutable source snapshot")
    identity = _snapshot(source_snapshot, base.environment["snapshot_sha256"])
    parent = _parent(base, source_snapshot, identity)
    spec_path = base_path.parent / "study.json"
    spec = _read(spec_path)
    if _inside(base_path.parent, spec["base_config"]) != base_path:
        raise ValueError("source study does not identify the supplied base configuration")
    scenarios = spec.get("scenarios", [])
    if len(scenarios) != 50 or len({case["name"] for case in scenarios}) != 50:
        raise ValueError("curriculum study retains all 50 frozen transfer scenarios")
    for scenario in scenarios:
        environment = scenario["environment"]
        route = environment["contract"]
        if (_sha(_inside(source_snapshot, route)) != environment["contract_sha256"]
                or environment["num_envs"] < 8):
            raise ValueError("evaluation contract identity or trace coverage differs")
    allocations = {domain: _allocation(source_snapshot, _contract(parent, domain, 0, total_updates))
                   for domain in ("mixed", "stationary")}
    if directory.exists():
        raise FileExistsError("curriculum output already exists")
    directory.mkdir(parents=True)
    snapshot = directory / "snapshot"
    for route in identity["files"]:
        target = _inside(snapshot, route)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_inside(source_snapshot, route), target)
    for source, target in ((source_snapshot / "snapshot.json", "source_snapshot.json"),
                           (base_path, "parent_base.json"), (spec_path, "parent_study.json")):
        shutil.copy2(source, snapshot / target)
    phase_contracts, arms = {}, []
    for arm_name, domains in ARM_DOMAINS.items():
        phases = []
        for index, (domain, start, updates) in enumerate(zip(
                domains, (0, warmup_updates), (warmup_updates, total_updates - warmup_updates))):
            phase_name = f"phase{index + 1}"
            route = f"contracts/curriculum.{arm_name}.{phase_name}.json"
            contract = _contract(parent, domain, start, total_updates)
            if _protected(contract) != _protected(parent):
                raise ValueError("curriculum intervention altered reward/mechanics/timing")
            _write(_inside(snapshot, route), contract)
            phase_contracts[(arm_name, phase_name)] = route
            phases.append({"name": phase_name, "domain": domain, "updates": updates,
                           "start_update": start, "config": f"configs/{arm_name}.{phase_name}.json"})
        arms.append({"name": arm_name, "phases": phases})
    snapshot_identity = deepcopy(identity)
    snapshot_identity.update(files=_files(snapshot), source_snapshot_sha256=identity["sha256"],
                             contract_validator="packed_curriculum_study")
    snapshot_identity["sha256"] = digest(snapshot_identity["files"])
    _write(snapshot / "snapshot.json", snapshot_identity)
    model_base, configs, artifacts = _gated(base), {}, {}

    def save_config(route, environment):
        value = model_base.to_dict()
        value["environment"] = {"snapshot": str(snapshot), "snapshot_sha256": snapshot_identity["sha256"],
                                **environment}
        config = FrameTrainConfig.from_dict(value)
        path = _inside(directory, route)
        _write(path, config.to_dict())
        configs[route], artifacts[route] = digest(config.to_dict()), _sha(path)

    for arm in arms:
        for phase in arm["phases"]:
            route = phase_contracts[(arm["name"], phase["name"])]
            save_config(phase["config"], {"contract": route, "contract_sha256": _sha(snapshot / route),
                                         "num_envs": 1024})
    evaluation = []
    for scenario in scenarios:
        route = f"configs/eval.{scenario['name']}.json"
        save_config(route, scenario["environment"])
        evaluation.append({"name": scenario["name"], "config": route})
    manifest = {
        "format": "transformer_rl.curriculum_study", "schema_version": 1,
        "environment_factory": spec["environment_factory"], "arms": arms, "training_seeds": seeds,
        "training": {"rollout_steps": 48, "checkpoint_interval": 100, "max_seconds": 172800.},
        "evaluation": {"seeds": list(EVALUATION_SEEDS), "steps": 4001, "settle_steps": 200,
                       "min_steady_samples": 200, "trace_replicas": 8},
        "scenarios": evaluation, "configs": configs, "artifacts": artifacts,
        "source_identity": {"base_config_sha256": _sha(base_path), "study_spec_sha256": _sha(spec_path),
                            "snapshot_sha256": identity["sha256"],
                            "snapshot_receipt_sha256": _sha(source_snapshot / "snapshot.json"),
                            "contract": base.environment["contract"],
                            "contract_sha256": base.environment["contract_sha256"],
                            "protected_recipe_sha256": digest(_protected(parent)),
                            "learner_source": source_identity()},
        "snapshot_identity": {"path": "snapshot", "sha256": snapshot_identity["sha256"],
                              "receipt_sha256": _sha(snapshot / "snapshot.json")},
        "protocol": {"warmup_updates": warmup_updates, "total_updates": total_updates,
                     "transitions_per_update": TRANSITIONS_PER_UPDATE,
                     "planned_transitions_per_arm_seed": total_updates * TRANSITIONS_PER_UPDATE,
                     "planned_total_transitions": total_updates * TRANSITIONS_PER_UPDATE * 3 * len(seeds),
                     "same_phase_boundary_reset_all_arms": True,
                     "phase2_learning_state": "restore_model_optimizer_rng_and_update_counter",
                     "phase2_clock": "cumulative_consumed_updates_with_zero_global_offset",
                     "phase_evaluation_does_not_gate_progress": True,
                     "stationary_inactive_sampling_fields": ["new_asset_curriculum.start_stop_start_delay_seconds"],
                     "task_dose_matched": False,
                     "causal_scope": "task_schedule_and_exposure_intervention_not_order_only",
                     "allocation_counts": allocations,
                     "policy_hz": 100, "physics_feedback_pd_hz": 200,
                     "hardware_deployment_ready": False},
    }
    manifest["sha256"] = digest(manifest)
    _write(directory / "manifest.json", manifest)
    validate_curriculum_study(directory)
    return {"directory": str(directory), "manifest": str(directory / "manifest.json"),
            "sha256": manifest["sha256"], "snapshot_sha256": snapshot_identity["sha256"],
            "arms": 3, "training_seeds": seeds, "updates_per_arm_seed": total_updates,
            "training_started": False}


def validate_curriculum_study(directory):
    """Check frozen bytes and reconstruct every allowed curriculum intervention."""
    directory = Path(directory).resolve()
    if directory.is_file():
        directory = directory.parent
    manifest = _read(directory / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if (manifest.get("format") != "transformer_rl.curriculum_study"
            or manifest.get("schema_version") != 1 or manifest.get("sha256") != digest(body)):
        raise ValueError("curriculum manifest identity changed")
    protocol = manifest["protocol"]
    warmup, total = protocol["warmup_updates"], protocol["total_updates"]
    _parameters(warmup, total, manifest["training_seeds"])
    if (protocol["transitions_per_update"] != TRANSITIONS_PER_UPDATE
            or protocol["planned_transitions_per_arm_seed"] != total * TRANSITIONS_PER_UPDATE
            or protocol["planned_total_transitions"] != total * TRANSITIONS_PER_UPDATE * 3 * len(manifest["training_seeds"])
            or not protocol["same_phase_boundary_reset_all_arms"] or protocol["task_dose_matched"]):
        raise ValueError("curriculum budget or phase-reset control changed")
    if manifest["source_identity"]["learner_source"] != source_identity():
        raise ValueError("curriculum learner source changed after preparation")
    snapshot = _inside(directory, manifest["snapshot_identity"]["path"])
    identity = _snapshot(snapshot, manifest["snapshot_identity"]["sha256"])
    if _sha(snapshot / "snapshot.json") != manifest["snapshot_identity"]["receipt_sha256"]:
        raise ValueError("curriculum snapshot receipt changed")
    source = manifest["source_identity"]
    for route, expected in (("parent_base.json", source["base_config_sha256"]),
                            ("parent_study.json", source["study_spec_sha256"]),
                            ("source_snapshot.json", source["snapshot_receipt_sha256"])):
        if _sha(snapshot / route) != expected:
            raise ValueError("original curriculum source receipt changed")
    original_identity = _read(snapshot / "source_snapshot.json")
    if (original_identity["sha256"] != source["snapshot_sha256"]
            or original_identity["sha256"] != digest(original_identity["files"])
            or identity["source_snapshot_sha256"] != source["snapshot_sha256"]):
        raise ValueError("original snapshot provenance changed")
    _verify_files(snapshot, original_identity["files"])
    base = FrameTrainConfig.load(snapshot / "parent_base.json")
    parent = _parent(base, snapshot, original_identity)
    if digest(_protected(parent)) != source["protected_recipe_sha256"]:
        raise ValueError("original reward/mechanics recipe changed")
    expected_model = _gated(base)
    loaded = {}
    if set(manifest["configs"]) != set(manifest["artifacts"]):
        raise ValueError("configuration byte and canonical identities differ")
    for route, expected in manifest["configs"].items():
        path = _inside(directory, route)
        config = FrameTrainConfig.load(path)
        if digest(config.to_dict()) != expected or _sha(path) != manifest["artifacts"][route]:
            raise ValueError("frozen curriculum configuration changed")
        if (config.model != expected_model.model or config.ppo != expected_model.ppo
                or config.control != expected_model.control
                or config.environment["snapshot"] != str(snapshot)
                or config.environment["snapshot_sha256"] != identity["sha256"]):
            raise ValueError("arms must share model/PPO/control and snapshot identity")
        environment = config.environment
        contract_path = _inside(snapshot, environment["contract"])
        if _sha(contract_path) != environment["contract_sha256"]:
            raise ValueError("curriculum environment contract changed")
        loaded[route] = config
    if [arm["name"] for arm in manifest["arms"]] != list(ARM_DOMAINS):
        raise ValueError("curriculum requires exactly the mixed/stationary/pretrain arms")
    used = set()
    for arm in manifest["arms"]:
        if len(arm["phases"]) != 2:
            raise ValueError("every arm must share the two-phase boundary")
        for index, phase in enumerate(arm["phases"]):
            expected = {"name": f"phase{index + 1}", "domain": ARM_DOMAINS[arm["name"]][index],
                        "start_update": 0 if index == 0 else warmup,
                        "updates": warmup if index == 0 else total - warmup}
            if any(phase.get(key) != value for key, value in expected.items()):
                raise ValueError("curriculum phase domain, budget or cumulative clock changed")
            config = loaded[phase["config"]]
            contract = _read(_inside(snapshot, config.environment["contract"]))
            if config.environment["num_envs"] != 1024 or contract != _contract(
                    parent, expected["domain"], expected["start_update"], total):
                raise ValueError("phase altered a protected recipe or task/clock intervention")
            used.add(phase["config"])
    original_spec = _read(snapshot / "parent_study.json")
    if manifest["environment_factory"] != original_spec["environment_factory"]:
        raise ValueError("curriculum environment factory changed")
    originals = {case["name"]: case for case in original_spec["scenarios"]}
    if len(manifest["scenarios"]) != 50 or {case["name"] for case in manifest["scenarios"]} != set(originals):
        raise ValueError("curriculum evaluation scenario coverage changed")
    for case in manifest["scenarios"]:
        config = loaded[case["config"]]
        expected = {"snapshot": str(snapshot), "snapshot_sha256": identity["sha256"],
                    **originals[case["name"]]["environment"]}
        if config.environment != expected:
            raise ValueError("curriculum evaluation differs from the original fixed cases")
        used.add(case["config"])
    if used != set(loaded):
        raise ValueError("curriculum has unused or missing configurations")
    if (manifest["evaluation"] != {"seeds": list(EVALUATION_SEEDS), "steps": 4001,
                                   "settle_steps": 200, "min_steady_samples": 200, "trace_replicas": 8}
            or manifest["training"] != {"rollout_steps": 48, "checkpoint_interval": 100,
                                       "max_seconds": 172800.}):
        raise ValueError("curriculum comparison evaluation or rollout protocol changed")
    return manifest

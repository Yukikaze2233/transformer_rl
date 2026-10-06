"""Freeze the existing robot environment and connect it to independent packed PPO.

Only preparation imports pure curriculum helpers. Isaac Lab starts in make_env,
inside a dedicated frame_process worker. The running baseline checkout is never
modified, and no RSL-RL learner or legacy model checkpoint is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys

from .frame_config import FrameTrainConfig, digest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _inside(root, path):
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("snapshot dependency escapes its root")
    return result


def control_contract(base, manifest):
    names = ["command_vx", "command_vy", "command_yaw", "command_height_x5"]
    names += [f"gyro_{axis}_x0_5" for axis in "xyz"] + [f"gravity_{axis}" for axis in "xyz"]
    policy_names = base["policy_action_order"]
    names += [f"q_relative_{name}" for name in policy_names] + [f"dq_x0_1_{name}" for name in policy_names]
    names += [f"previous_issued_{name}" for name in policy_names]
    names += ["mode_normal", "mode_stair", "mode_slope_reserved", "mode_recover", "mode_jump",
              "request_height_x5", "command_clock_s"]
    settings = base["v5_control"]
    return {"policy_dt_s": .01, "observation_schema": "manual_scaled35",
            "feature_names": names, "action_names": policy_names,
            "action_bounds": [settings["action_clip"]] * 4 + [settings["wheel_action_clip"]] * 2,
            "target_scale": [settings["leg_position_scale"]] * 4 + [settings["wheel_velocity_scale"]] * 2,
            "target_offset": [manifest["nominal_joint_pos"][name] for name in policy_names[:4]] + [0., 0.],
            "target_units": ["rad"] * 4 + ["rad/s"] * 2}


def prepare_study(source_root, curriculum_path, directory, *, num_envs=4096, evaluation_replicas=16, round_name="screen"):
    """Pure file/config preparation; no simulator interaction or training occurs."""
    source_root, directory = Path(source_root).resolve(), Path(directory).resolve()
    curriculum_path = Path(curriculum_path)
    if not curriculum_path.is_absolute():
        curriculum_path = _inside(source_root, curriculum_path)
    if type(num_envs) is not int or num_envs < 32 or type(evaluation_replicas) is not int or evaluation_replicas < 4:
        raise ValueError("preparation requires >=32 training envs and >=4 evaluation replicas")
    if round_name not in ("screen", "confirm"):
        raise ValueError("round must be screen or confirm")
    plan = json.loads(curriculum_path.read_text())
    if plan.get("contract_id") != "v6-new-asset-curriculum-plan-v1":
        raise ValueError("this adapter requires the explicit new-asset curriculum")
    base = plan["runtime_base"]
    if (base["physics_dt"], base["policy_dt"], base["actor_dim"], base["critic_dim"], base["action_dim"]) != (.001, .02, 35, 81, 6):
        raise ValueError("source curriculum does not match the audited baseline")
    source_asset = _inside(source_root, base["asset_directory"])
    manifest_path = source_asset / "manifest.json"
    if _sha(manifest_path) != base["asset_manifest_sha256"]:
        raise ValueError("source asset manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for name, sha in manifest["files_sha256"].items():
        if _sha(_inside(source_asset, name)) != sha:
            raise ValueError(f"source asset changed: {name}")
    control_path = _inside(source_root, base["control_math_source"])
    if _sha(control_path) != base["control_math_sha256"]:
        raise ValueError("source control prior changed")
    # Load only pure materializers from the source root. Subsequent simulation
    # happens in fresh workers importing the copied package instead.
    sys.path.insert(0, str(source_root / "src"))
    try:
        full = importlib.import_module("wheeled_tasks.chassis.full_curriculum")
        evaluation = importlib.import_module("wheeled_tasks.chassis.evaluation")
        if not Path(full.__file__).resolve().is_relative_to(source_root / "src"):
            raise ValueError("another task package is already imported; prepare in a fresh process")
        materialized = [full.stage_contract({}, plan, recipe, 20000) for recipe in plan["stages"]]
    finally:
        sys.path.pop(0)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "snapshot"
    snapshot.mkdir()
    shutil.copytree(source_root / "src/wheeled_tasks", snapshot / "src/wheeled_tasks",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    asset_target = snapshot / base["asset_directory"]
    asset_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_asset, asset_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    prior_target = snapshot / base["control_math_source"]
    prior_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(control_path, prior_target)
    _write(snapshot / "parent_curriculum.json", plan)
    control = control_contract(base, manifest)
    snapshots = {}
    stages, scenarios, seen = [], [], set()
    for index, (recipe, config) in enumerate(zip(plan["stages"], materialized)):
        if round_name == "screen" and index:
            break
        config = deepcopy(config)
        config.update(contract_id="packed-policy-study", parent_curriculum_id=plan["contract_id"],
                      policy_dt=.01, actor_dim=35, history_length=1, target_num_envs=num_envs,
                      num_steps_per_env=48, num_mini_batches=32, ppo_minibatch_samples=num_envs * 48 // 32,
                      curriculum_reference_batch=num_envs * 48, record_diagnostics=True, diagnostic_trace=True,
                      auto_reset=False, action_difference_reference_dt=.02)
        # The old validator intentionally hard-codes the baseline's 50 Hz
        # campaign. This new study has its own frozen contract and validator;
        # it never claims to pass that baseline campaign's design table.
        config.pop("design_preflight", None)
        config["requires_design_preflight"] = True
        stage_name = recipe["name"].lower()
        train_route = f"contracts/{stage_name}.train.json"
        _write(snapshot / train_route, config)
        snapshots[train_route] = _sha(snapshot / train_route)
        current_scenarios = []
        for case in config["evaluation"]["cases"]:
            if case["name"] not in config["evaluation"]["promotion_case_names"]:
                continue
            name = case["name"].lower().replace("-", "_")
            if name in seen:
                continue
            seen.add(name)
            current_scenarios.append(name)
            eval_config = evaluation.fixed_suite_contract(config)
            eval_config["evaluation"]["cases"] = [deepcopy(case)]
            eval_config["evaluation"]["stable_case_layout"] = False
            eval_config["evaluation"]["episodes_per_case"] = evaluation_replicas
            eval_config["target_num_envs"] = evaluation_replicas
            eval_route = f"contracts/eval.{name}.json"
            _write(snapshot / eval_route, eval_config)
            snapshots[eval_route] = _sha(snapshot / eval_route)
            gates = [{"path": "success_rate", "operator": "min", "value": .95},
                     {"path": "metrics.tilt_angle.mean", "operator": "max", "value": .25}]
            task = case["task"]
            if task not in ("jump", "traverse", "sequence"):
                gates.extend(( {"path": "metrics.height_abs_error.mean", "operator": "max", "value": .03},
                               {"path": "metrics.vx_abs_error.mean", "operator": "max", "value": .15},
                               {"path": "metrics.wz_abs_error.mean", "operator": "max", "value": .25}))
            if "stand" in name and task == "survive":
                gates.append({"path": "metrics.drift_m.max", "operator": "max", "value": .20})
            scenarios.append({"name": name, "environment": {"contract": eval_route,
                "contract_sha256": snapshots[eval_route], "num_envs": evaluation_replicas,
                "evaluation_batch": stage_name}, "gates": gates, "require_steady": task == "survive"})
        stages.append({"name": stage_name, "updates": 200 if round_name == "screen" else recipe["updates"],
                       "environment": {"contract": train_route, "contract_sha256": snapshots[train_route],
                                       "num_envs": num_envs}, "scenarios": current_scenarios})
    files = {str(path.relative_to(snapshot)): _sha(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    identity = {"files": files, "sha256": digest(files), "control": control,
                "parent_curriculum_sha256": _sha(curriculum_path), "contract_validator": "packed_policy_study"}
    _write(snapshot / "snapshot.json", identity)
    train_config = {"model": {"policy": {"architecture": "mlp", "history_length": 1},
                               "critic_dim": 81, "critic_hidden": [256, 128, 64], "initial_std": 1.},
                    "ppo": {"learning_rate": 1e-4, "gamma": math.sqrt(.99), "gae_lambda": math.sqrt(.95),
                            "epochs": 5, "num_minibatches": 32, "value_coef": 4., "entropy_coef": .005,
                            "target_kl": .01, "clip_ratio": .2},
                    "control": control, "environment": {"snapshot": str(snapshot), "snapshot_sha256": identity["sha256"],
                                                           **stages[0]["environment"]}}
    _write(directory / "base.json", FrameTrainConfig.from_dict(train_config).to_dict())
    variants = network_variants()
    max_seconds = 3600. if round_name == "screen" else 14400.
    spec = {"base_config": "base.json", "environment_factory": "transformer_rl.chassis_adapter:make_env",
            "variants": variants, "seeds": [1101] if round_name == "screen" else [1101, 2202, 3303],
            "stages": stages, "scenarios": scenarios,
            "training": {"rollout_steps": 48, "checkpoint_interval": 200, "max_seconds": max_seconds,
                         "retention_coef": 0., "anchor_seeds": [4101, 5101]},
            "evaluation": {"validation_seeds": [701, 1701], "seeds": [2701, 3701], "steps": 4001, "settle_steps": 200,
                           "min_steady_samples": 200, "min_completed_episodes": evaluation_replicas},
            "execution": {"devices": ["cuda:0"], "worker_module": "transformer_rl.frame_process",
                          "job_timeout_seconds": max_seconds + 300.},
            "selection": {"min_training_seeds": 3, "std_penalty": 1., "latency_p99_ms": 8., "latency_max_ms": 10.,
                          "max_deadline_misses": 0, "retention_score_tolerance": .25, "rollback_limit": 1,
                          "objectives": [{"path": "success_rate", "direction": "maximize", "scale": 1., "weight": 4.},
                                         {"path": "metrics.height_abs_error.mean", "direction": "minimize", "scale": .03, "weight": 1.},
                                         {"path": "metrics.vx_abs_error.mean", "direction": "minimize", "scale": .15, "weight": 1.},
                                         {"path": "metrics.issued_action_rate_rms.mean", "direction": "minimize", "scale": 100., "weight": .1}]}}
    _write(directory / "study.json", spec)
    return {"directory": str(directory), "spec": str(directory / "study.json"), "snapshot_sha256": identity["sha256"],
            "policy_hz": 100, "physics_hz": 1000, "history_frames": 31, "training_started": False,
            "variants": len(variants), "scenarios": len(scenarios), "round": round_name}



def network_variants():
    """Three architecture families, with separate Transformer capacity probes."""
    variants = []
    for name, hidden in (("mlp", [256, 128, 64]), ("mlp_medium", [512, 256, 128])):
        variants.append({"name": name, "policy": {"architecture": "mlp", "history_length": 1,
                                                  "actor_hidden_dims": hidden}})
    for name, encoder, latent, hidden in (
            ("history_mlp", [128, 64], 3, [128, 64, 32]),
            ("history_mlp_wide", [256, 128, 64], 16, [256, 128, 64])):
        variants.append({"name": name, "policy": {"architecture": "history_mlp", "history_length": 31,
            "encoder_hidden_dims": encoder, "history_latent_dim": latent, "actor_hidden_dims": hidden}})
    for name, width, layers, heads, ffn, hidden, residual, readout in (
            ("transformer_small", 96, 2, 4, 192, [128, 64], "add", "last"),
            ("transformer", 128, 2, 4, 512, [256, 128], "add", "last"),
            ("transformer_query", 128, 2, 4, 512, [256, 128], "add", "query"),
            ("transformer_gated", 128, 2, 4, 512, [256, 128], "gated", "last"),
            ("transformer_large", 160, 3, 5, 640, [256, 128], "add", "last"),
            ("transformer_xlarge", 192, 4, 6, 1024, [256, 128], "add", "last")):
        variants.append({"name": name, "policy": {"architecture": "transformer", "history_length": 31,
            "d_model": width, "num_layers": layers, "num_heads": heads, "ffn_dim": ffn,
            "actor_hidden_dims": hidden, "residual_type": residual, "readout_type": readout}})
    return variants

def _validate_contract(config, control):
    physical_dt = .005 if config.get("contract_id") == "packed-transfer-study" else .001
    if (config["physics_dt"], config["policy_dt"], config["history_length"], config["actor_dim"],
            config["actor_frame_dim"], config["critic_dim"], config["action_dim"]) != (physical_dt, .01, 1, 35, 35, 81, 6):
        raise ValueError("prepared environment timing/observation contract mismatch")
    settings = config["v5_control"]
    if (settings["leg_kp"], settings["leg_kd"], control["actuators"]["wheel"]["kd"]) != (160., 2.5, .6):
        raise ValueError("prepared PC feedback gains differ from the declared baseline")
    if config.get("auto_reset") is not False or not config.get("record_diagnostics") or not config.get("diagnostic_trace"):
        raise ValueError("adapter requires PRE-reset diagnostics and explicit resets")
    if config.get("evaluation_exact_cases") and config["jump_assist"]["enabled"]:
        raise ValueError("evaluation cannot use training jump assistance")
    expected = {"physics_dt": physical_dt, "policy_dt": .01, "decimation": round(.01 / physical_dt), "pc_control_dt": physical_dt,
            "leg_kp": 160., "leg_kd": 2.5, "wheel_velocity_p": .6, "precision_tracking": False,
            "dense_tracking": True, "command_reference": True,
            "modules": {name: bool(config.get(name, {}).get("enabled")) for name in
                        ("usb_transport", "command_transport", "signal_perturbations", "dynamics_randomization",
                         "contact_domain", "step_assist", "jump_assist", "training_schedule")}}
    for name in ("actuator_response", "signal_delay"):
        if name in config:
            expected["modules"][name] = bool(config[name].get("enabled"))
            expected[name] = deepcopy(config[name])
    if config.get("command_transport", {}).get("delivery_model"):
        expected["command_transport_model"] = config["command_transport"]["delivery_model"] if expected["modules"]["command_transport"] else None
    if config.get("step_jump_task", {}).get("enabled"):
        expected["modules"].update(step_jump_task=True, step_climb_task=True)
    if config.get("recovery_training", {}).get("enabled"):
        expected["modules"]["recovery_training"] = True
    return expected


def rescale_action_differences(terms, reference_dt, policy_dt):
    """Keep the baseline physical derivative weighting as the interval changes."""
    ratio = reference_dt / policy_dt
    terms["action_rate"] *= ratio**2
    terms["leg_action_smoothness"] *= ratio**4
    terms["wheel_action_smoothness"] *= ratio**4
    return terms


def merge_evaluation_contracts(snapshot, environments):
    """Only case lists/layout/count may differ within a simulation batch."""
    configs = []
    for environment in environments:
        path = _inside(snapshot, environment["contract"])
        if _sha(path) != environment["contract_sha256"]:
            raise ValueError("evaluation contract hash mismatch")
        configs.append(json.loads(path.read_text()))
    def common(config):
        config = deepcopy(config)
        config.pop("scene_groups", None)
        config.pop("target_num_envs", None)
        config["evaluation"].pop("cases", None)
        return config
    if any(common(config) != common(configs[0]) for config in configs):
        raise ValueError("batch scenarios differ in dynamics, commands, reward or episode protocol")
    result = deepcopy(configs[0])
    cases = [case for config in configs for case in config["evaluation"]["cases"]]
    if (any(not config.get("evaluation_exact_cases") for config in configs)
            or len({case["name"] for case in cases}) != len(cases)):
        raise ValueError("evaluation batches require unique fixed cases")
    if any(not config["evaluation"]["cases"] or environment["num_envs"] < 1
           or environment["num_envs"] % len(config["evaluation"]["cases"])
           or config["target_num_envs"] != environment["num_envs"]
           for environment, config in zip(environments, configs)):
        raise ValueError("batch environment counts must equal complete case replica counts")
    replicas = [environment["num_envs"] // len(config["evaluation"]["cases"])
                for environment, config in zip(environments, configs)]
    if len(set(replicas)) != 1:
        raise ValueError("every batched case needs equal replicas")
    result["evaluation"]["cases"] = cases
    result["scene_groups"] = [{"name": case["name"], "fraction": 1 / len(cases), "terrain": [case.get("terrain", "flat")]} for case in cases]
    result["target_num_envs"] = sum(environment["num_envs"] for environment in environments)
    return result


def evaluate_suite(checkpoint, configs, outputs, *, steps, seed, device, settle_steps, min_steady_samples,
                   anchor_directory=None, max_anchors=256, control_output=None, trace_output=None, trace_replicas=2):
    """Publish separate per-case evidence from a common vectorized rollout."""
    from .frame_checkpoint import load_frame_checkpoint
    from .frame_workflow import _positive_integer, evaluate_frame_policy
    _positive_integer(max_anchors, "max_anchors")
    if len(configs) != len(outputs) or not configs:
        raise ValueError("suite configs and outputs must have equal nonzero lengths")
    for path in (control_output, trace_output):
        if path is not None and Path(path).exists():
            raise FileExistsError(path)
    if trace_output is not None and control_output is None:
        raise ValueError("trace_output requires a control_output report")
    parsed = [FrameTrainConfig.load(path) for path in configs]
    _, _, saved, _, _, _ = load_frame_checkpoint(checkpoint)
    for config, output in zip(parsed, outputs):
        if config.model != saved.model or config.control != saved.control:
            raise ValueError("suite policy/control differs from checkpoint")
        if Path(output).exists():
            raise FileExistsError(output)
    environments = [config.environment for config in parsed]
    snapshot = Path(environments[0]["snapshot"])
    if any(environment["snapshot"] != str(snapshot) or environment["snapshot_sha256"] != environments[0]["snapshot_sha256"]
           for environment in environments):
        raise ValueError("suite mixes environment snapshots")
    merge_evaluation_contracts(snapshot, environments)
    environment = {"snapshot": str(snapshot), "snapshot_sha256": environments[0]["snapshot_sha256"],
                   "contracts": environments, "num_envs": sum(e["num_envs"] for e in environments)}
    report = evaluate_frame_policy(checkpoint, make_env, environment, steps=steps, seed=seed, device=device,
        settle_steps=settle_steps, min_steady_samples=min_steady_samples, max_anchors=max_anchors, group_anchor_directory=anchor_directory,
        control_metrics=control_output is not None, trace_output=trace_output, trace_replicas=trace_replicas)
    for config, output in zip(parsed, outputs):
        case = json.loads((_inside(snapshot, config.environment["contract"])).read_text())["evaluation"]["cases"][0]
        grouped = report["groups"][case["name"]]
        grouped["environment"] = config.environment
        _write(output, grouped)
    if control_output is not None:
        _write(control_output, {"format": "transformer_rl.control_evaluation", "schema_version": 1,
            "checkpoint_sha256": report["checkpoint_sha256"], "checkpoint_update": report["checkpoint_update"],
            "seed": seed, "steps": steps, "environment_provenance": report["environment_provenance"],
            "control": report["control"], "groups": {name: value["control"] for name, value in report["groups"].items()},
            "trace": report.get("trace")})
    return {"case_reports": [str(path) for path in outputs], "simulation_envs": environment["num_envs"], "seed": seed}


class ChassisFrameAdapter:
    def __init__(self, env, config, metadata, *, enable_control_metrics=False, reset_transform=None):
        import torch
        self.env, self.config, self.metadata = env, config, metadata
        self.enable_control_metrics = enable_control_metrics
        self._reset_transform = reset_transform
        self._last_environment_metrics = {}
        self._control_motor_indices = None
        self.num_envs, self.device = env.num_envs, torch.device(env.device)
        self._previous = torch.zeros(self.num_envs, 6, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._origin = env.robot.data.root_link_pose_w.torch[:, :2].clone()
        self._survival_only = None
        if env.cfg.get("evaluation_exact_cases"):
            tasks = {case["name"]: case["task"] for case in env.cfg["evaluation"]["cases"]}
            self._survival_only = torch.tensor([tasks[name] == "survive" for name in env.scene_groups], device=self.device)

    def _observation(self, raw):
        import torch
        from .types import VectorObservation
        frame = raw["policy"]
        if frame.shape != (self.num_envs, 35):
            raise ValueError("environment must return one scaled 35D frame")
        timestamp = torch.full((self.num_envs,), self.env.tick * .01, dtype=torch.float64, device=self.device)
        return VectorObservation(frame.clone(), timestamp, frame[:, self.config.command_indices].clone(), raw["critic"].clone())

    def reset(self, seed=None):
        import torch
        if seed is not None:
            self.env.generator.manual_seed(seed)
        self._reset_env(torch.arange(self.num_envs, device=self.device))
        self._previous.zero_()
        self._fresh.fill_(True)
        self._origin.copy_(self.env.robot.data.root_link_pose_w.torch[:, :2])
        return self._observation(self.env.get_observations())

    def _reset_env(self, rows):
        self.env.reset(rows)
        if self._reset_transform is not None:
            self._reset_transform(self.env, rows)

    def training_diagnostics(self):
        """Latest observed environment diagnostics, distinct from learner statistics."""
        import torch
        result = {}
        for name, value in self._last_environment_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().item()
            if type(value) in (int, float, bool):
                if not math.isfinite(float(value)):
                    raise ValueError(f"nonfinite environment diagnostic: {name}")
                result[name.strip("/")] = value
        if getattr(self.env, "perturbations", None) is not None:
            result["transfer/noise_enabled_fraction"] = self.env.perturbations.enabled.float().mean().item()
        return result

    def set_training_progress(self, updates, transitions):
        self.env.stage_actor_update = updates
        self.env.global_actor_update = self.env.cfg.get("global_actor_update_offset", 0) + updates
        self.env.training_transitions = transitions

    def _nominal_effort_scaling(self):
        """Snapshot the reset-stable V5 multiplier, not a physical output limit."""
        import torch
        if not getattr(self.env, "is_v5", False):
            return None
        strength = getattr(self.env, "motor_strength", None)
        if (not isinstance(strength, torch.Tensor) or strength.shape != (self.num_envs, 6)
                or not strength.is_floating_point() or strength.device != self._previous.device
                or not torch.isfinite(strength).all() or (strength < 0).any()):
            raise ValueError("scaled nominal effort requires finite nonnegative motor_strength [num_envs, 6]")
        factors = [strength.detach().clone()]
        skills = getattr(self.env, "skills", None)
        if skills is not None:
            scheduled = getattr(getattr(skills, "schedule", None), "motor_scale", None)
            if (not isinstance(scheduled, torch.Tensor) or scheduled.shape != strength.shape
                    or not scheduled.is_floating_point() or scheduled.device != strength.device
                    or not torch.isfinite(scheduled).all() or (scheduled < 0).any()):
                raise ValueError("scaled nominal effort requires finite nonnegative schedule motor_scale [num_envs, 6]")
            factors.append(scheduled.detach().clone())
        if len(factors) == 2 and not torch.isfinite(factors[0] * factors[1]).all():
            raise ValueError("scaled nominal effort multiplier must be finite")
        return tuple(factors)

    def _control_packet(self, diagnostic, velocity, omega, height, tilt, done, nominal_scaling=None):
        import torch
        if self._control_motor_indices is None:
            source_order = self.metadata.get("startup", getattr(self.env, "startup_report", {})).get("active_joint_order")
            target_order = self.env.cfg.get("policy_action_order")
            if (not isinstance(source_order, list) or not isinstance(target_order, list)
                    or len(source_order) != 6 or len(target_order) != 6
                    or len(set(source_order)) != 6 or set(source_order) != set(target_order)):
                raise ValueError("control metrics require explicit matching active_joint_order and policy_action_order")
            self._control_motor_indices = [source_order.index(name) for name in target_order]
        order = self._control_motor_indices
        position = diagnostic.get("position")
        if position is None:
            position = self.env.robot.data.root_link_pose_w.torch
        # Simulator joint state is the physical endpoint; diagnostics may expose
        # delayed controller feedback. Canonical packets use four legs then wheels.
        motor_position = self.env.robot.data.joint_pos.torch[:, self.env.ids][:, order]
        motor_velocity = self.env.robot.data.joint_vel.torch[:, self.env.ids][:, order]
        bound = diagnostic["motor_effort_bounds"]
        limits = torch.stack((-bound, bound), -1) if bound.ndim == 2 else bound
        packet = {name: value.clone() for name, value in {
            "time_s": diagnostic["episode_ticks"].double() * .01,
            "command_reference": diagnostic["commands"],
            "actual": torch.stack((velocity[:, 0], omega[:, 2], height), -1),
            "position_xy": position[:, :2], "tilt": tilt,
            "leg_target": diagnostic["leg_target_position"], "wheel_target": diagnostic["wheel_target_velocity"],
            "motor_position": motor_position, "motor_velocity": motor_velocity,
            "motor_effort": diagnostic["motor_effort"][:, order],
            "requested_motor_effort": diagnostic["requested_motor_effort"][:, order], "effort_bounds": limits[:, order],
            "failure": diagnostic["terminated"].bool() & done,
            "success": diagnostic["success"].bool() & ~diagnostic["terminated"].bool() & done,
        }.items()}
        if nominal_scaling is not None:
            # Keep request and envelope in the same scaled coordinate system.
            # Filtering, transport and dynamic wheel limits remain later stages.
            packet["scaled_nominal_requested_motor_effort"] = packet["requested_motor_effort"].clone()
            packet["scaled_nominal_effort_bounds"] = packet["effort_bounds"].clone()
            # Match the parent's two in-place float operations. Combining the
            # factors first can move a true endpoint past the metric tolerance.
            for factor in nominal_scaling:
                scale = factor[:, order]
                packet["scaled_nominal_requested_motor_effort"] *= scale
                packet["scaled_nominal_effort_bounds"] *= scale[..., None]
        return packet

    def step(self, issued_action):
        import torch
        from .types import StepResult
        nominal_scaling = self._nominal_effort_scaling() if self.enable_control_metrics else None
        raw, reward, done, extras = self.env.step(issued_action)
        self._last_environment_metrics = dict(extras.get("log", {}))
        diagnostic = extras["diagnostics"]
        final_critic = raw["critic"].clone()
        truncated = extras["time_outs"].bool().clone() & done
        terminated = done & (~truncated | diagnostic["terminated"].bool() | diagnostic["success"].bool())
        truncated &= ~terminated
        velocity, omega, gravity = (diagnostic[name] for name in ("velocity", "omega", "gravity"))
        command, height = diagnostic["commands"], diagnostic["height"]
        tilt = torch.acos((-gravity[:, 2] / gravity.norm(dim=-1).clamp_min(1e-6)).clamp(-1, 1))
        rate = (issued_action - self._previous).square().mean(-1).sqrt() / .01
        rate[self._fresh] = 0.
        drift = (self.env.robot.data.root_link_pose_w.torch[:, :2] - self._origin).norm(dim=-1)
        metrics = {"height_abs_error": (height - command[:, 2]).abs(),
                   "vx_abs_error": (velocity[:, 0] - command[:, 0]).abs(),
                   "wz_abs_error": (omega[:, 2] - command[:, 1]).abs(), "tilt_angle": tilt,
                   "drift_m": drift, "issued_action_rate_rms": rate}
        signals = {"height_error": height - command[:, 2], "vx_error": velocity[:, 0] - command[:, 0],
                   "wz_error": omega[:, 2] - command[:, 1]}
        for prefix, values in (("leg_target", diagnostic["leg_target_position"]),
                               ("wheel_target", diagnostic["wheel_target_velocity"]),
                               ("effort", diagnostic["motor_effort"])):
            signals.update({f"{prefix}_{i}": values[:, i].clone() for i in range(values.shape[1])})
        # Task success and survival are distinct: only ordinary continuous tasks
        # may succeed at a healthy time limit; jump/traverse must meet geometry.
        ordinary = self._survival_only if self._survival_only is not None else self.env.mode <= 1
        episode_success = done & ~diagnostic["terminated"].bool() & (diagnostic["success"].bool() | (truncated & ordinary))
        info = {"episode_success": episode_success.clone(),
                "evaluation_metrics": {name: value.clone() for name, value in metrics.items()},
                "evaluation_signals": signals,
                "evaluation_signal_time": diagnostic["episode_ticks"].double() * .01}
        if self.enable_control_metrics:
            if nominal_scaling is not None:
                current = self._nominal_effort_scaling()
                if (current is None or len(current) != len(nominal_scaling)
                        or any(not torch.equal(before, after) for before, after in zip(nominal_scaling, current))):
                    raise ValueError("nominal effort scale changed during physical step; scaled envelope is unavailable")
            info["control_packet"] = self._control_packet(diagnostic, velocity, omega, height, tilt, done, nominal_scaling)
        self._previous.copy_(issued_action)
        self._fresh.zero_()
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self._reset_env(ids)
            self._previous[ids] = 0.
            self._fresh[ids] = True
            self._origin[ids] = self.env.robot.data.root_link_pose_w.torch[ids, :2]
            raw = self.env.get_observations()
        return StepResult(self._observation(raw), reward.clone(), terminated.clone(), truncated.clone(),
                          final_critic, done.clone(), info)

    def close(self):
        self.env.sim.stop()


def make_env(model_config, environment_config, device):
    import torch
    from .frame_process import register_app, require_worker
    require_worker()
    snapshot = Path(environment_config["snapshot"]).resolve()
    identity = json.loads((snapshot / "snapshot.json").read_text())
    if digest(identity["files"]) != identity["sha256"] or identity["sha256"] != environment_config["snapshot_sha256"]:
        raise ValueError("snapshot identity differs from the study")
    for name, sha in identity["files"].items():
        if _sha(_inside(snapshot, name)) != sha:
            raise ValueError(f"frozen environment dependency changed: {name}")
    config = merge_evaluation_contracts(snapshot, environment_config["contracts"]) if "contracts" in environment_config else None
    if config is None:
        path = _inside(snapshot, environment_config["contract"])
        if _sha(path) != environment_config["contract_sha256"]:
            raise ValueError("effective environment contract changed")
        config = json.loads(path.read_text())
    effective_sha256 = digest(config)
    if config["target_num_envs"] != environment_config["num_envs"]:
        raise ValueError("environment count differs from the frozen invocation")
    if (model_config.frame_dim, model_config.critic_dim, model_config.action_dim) != (35, 81, 6):
        raise ValueError("chassis adapter requires the explicit 35/81/6 contract")
    control = json.loads((snapshot / config["control_math_source"]).read_text())
    expected_runtime = _validate_contract(config, control)
    expected_runtime["modules"]["training_schedule"] = True
    expected_runtime["seed"] = torch.initial_seed()
    asset = snapshot / config["asset_directory"]
    manifest = json.loads((asset / "manifest.json").read_text())
    if control_contract(config, manifest) != identity["control"]:
        raise ValueError("effective action mapping differs from the deployment contract")
    sys.path.insert(0, str(snapshot / "src"))
    from isaaclab.app import AppLauncher
    app = AppLauncher({"headless": True, "enable_cameras": False, "device": str(device),
                       "kit_args": "--/exts/omni.kit.telemetry/skipDeferredStartup=true"})
    register_app(app.app)
    env_module = importlib.import_module("wheeled_tasks.chassis.env")
    if not Path(env_module.__file__).resolve().is_relative_to(snapshot / "src"):
        raise ValueError("worker imported another environment source")
    original_preflight = env_module.design_preflight
    reward_module = importlib.import_module("wheeled_tasks.chassis.rewards")
    original_terms = reward_module.reward_terms
    def timed_terms(*args, **kwargs):
        return rescale_action_differences(original_terms(*args, **kwargs),
                                         config["action_difference_reference_dt"], config["policy_dt"])
    reward_module.reward_terms = timed_terms
    def study_preflight(effective, effective_control, root, **invocation):
        if digest(effective) != digest(config) or effective_control != control:
            raise ValueError("constructor altered the frozen effective contract")
        return {"status": "packed_study_static_passed", "runtime_expected": expected_runtime,
                "contract_sha256": effective_sha256, "source": "independent_100hz_network_study"}
    env_module.design_preflight = study_preflight
    try:
        stage = config["enabled_stages"][0]
        env = env_module.ChassisEnv(config, manifest, control, snapshot, stage_name=stage,
            num_envs=environment_config["num_envs"], device=str(device), level=1., seed=torch.initial_seed())
    finally:
        env_module.design_preflight = original_preflight
    metadata = {"identity": identity["sha256"],
        "control_sha256": digest(identity["control"]), "contract_sha256": effective_sha256,
        "startup": env.startup_report,
        **({"parent_task_sha256": identity["parent_task_sha256"]} if "parent_task_sha256" in identity
           else {"parent_curriculum_sha256": identity["parent_curriculum_sha256"]}),
        "control_packet_joint_order": list(config["policy_action_order"]),
        "physics_hz": 1 / config["physics_dt"], "policy_hz": 1 / config["policy_dt"], "preflight": "packed_study_not_baseline_campaign",
        **({"evaluation_groups": list(env.scene_groups)} if config["evaluation_exact_cases"] else {})}
    reset_transform = None
    if config.get("transfer_evaluation"):
        from .transfer_profiles import configure_transfer_evaluation, apply_reset_profiles
        metadata["transfer_evaluation"] = configure_transfer_evaluation(env)
        reset_transform = apply_reset_profiles
    return ChassisFrameAdapter(env, model_config, metadata, reset_transform=reset_transform)

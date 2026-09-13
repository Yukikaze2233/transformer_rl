"""Optional external Isaac Lab research task; no legacy learner is imported."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType

import torch

from transformer_rl.adapters import _TensorEnvContract
from transformer_rl.history import pack_frame
from transformer_rl.types import StepResult, VectorObservation


def encode_observation(config, raw, previous_issued_action, timestamp, policy_dt):
    """Preserve source scaling/noise, replacing its applied-action echo with issued."""
    if raw["policy"].shape != (previous_issued_action.shape[0], 125):
        raise ValueError("external actor observation must be N x 125")
    scalar = raw["policy"][:, -25:]
    proprio = torch.cat((scalar[:, :6], scalar[:, 9:19]), dim=-1)
    command = scalar[:, 6:9].clone()
    ages = scalar.new_zeros((scalar.shape[0], 2))
    frame = pack_frame(
        config, proprio, command, previous_issued_action, ages,
        torch.ones_like(ages, dtype=torch.bool), scalar.new_full((scalar.shape[0],), policy_dt),
    )
    return VectorObservation(frame, timestamp.clone(), command, raw["critic"].clone())


def physical_metrics(linear_velocity, angular_velocity, height, command, non_wheel_force):
    """Owned PRE-reset SI diagnostics; net force does not identify contact pairs."""
    return {name: value.detach().clone() for name, value in {
        "vx_abs_error": (linear_velocity[:, 0] - command[:, 0]).abs(),
        "wz_abs_error": (angular_velocity[:, 2] - command[:, 1]).abs(),
        "height_abs_error": (height - command[:, 2]).abs(),
        "planar_speed": torch.linalg.vector_norm(linear_velocity[:, :2], dim=-1),
        "non_wheel_net_force": non_wheel_force,
    }.items()}


class IsaacLabTaskAdapter:
    """Own one SimulationApp and capture terminal state before DirectRLEnv reset."""

    def __init__(self, env, app, config, metadata, build_observation, build_critic):
        self.env, self.app, self.metadata = env, app, metadata
        self._contract = _TensorEnvContract(config, env.num_envs, env.device)
        self.num_envs, self.device = self._contract.num_envs, self._contract.device
        self.config = config
        self._tick = 0
        self._stepping = False
        self._closed = False
        self._captured = False
        self._issued = torch.zeros(self.num_envs, 6, device=self.device)
        self._final = torch.zeros(self.num_envs, 29, device=self.device)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._metrics = {}
        original_rewards, original_reset = env._get_rewards, env._reset_idx

        def rewards(source):
            reward = original_rewards()
            if self._stepping:
                q, dq = source._joint_state()
                data = source.robot.data
                linear = data.root_com_lin_vel_b.torch
                angular = data.root_com_ang_vel_b.torch
                height = source._base_height()
                # Reward decrements command timers but does not resample. Capture
                # the transition's command, before next-observation resampling.
                clean = build_observation(
                    angular, data.projected_gravity_b.torch, source.commands,
                    q, dq, source.actions, source.contract,
                )
                self._final.copy_(build_critic(clean, linear, height))
                self._metrics = physical_metrics(
                    linear, angular, height, source.commands,
                    source._contact_magnitudes()[:, source._non_wheel_body_ids].amax(-1),
                )
                self._captured = True
            return reward

        def reset_idx(source, env_ids):
            if self._stepping:
                if not self._captured:
                    raise RuntimeError("auto-reset preceded final-state capture")
                ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                       else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
                self._valid[ids] = source.reset_time_outs[ids]
            return original_reset(env_ids)

        # Installed after source construction: startup/reset cannot capture stale state.
        env._get_rewards = MethodType(rewards, env)
        env._reset_idx = MethodType(reset_idx, env)

    def _encode(self, raw):
        timestamp = torch.full((self.num_envs,), self._tick * self.env.step_dt,
                               device=self.device, dtype=torch.float64)
        return self._contract.observation(encode_observation(
            self.config, raw, self._issued, timestamp, self.env.step_dt,
        ))

    @torch.no_grad()
    def reset(self, seed=None):
        raw, _ = self.env.reset(seed=seed)
        self._issued.zero_()
        self._tick += 1
        return self._encode(raw)

    @torch.no_grad()
    def step(self, issued_action):
        self._issued.copy_(self._contract.tensor("issued_action", issued_action, (self.num_envs, 6)))
        self._valid.zero_()
        self._captured = False
        self._stepping = True
        try:
            raw, reward, terminated, truncated, info = self.env.step(self._issued.clone())
        finally:
            self._stepping = False
        if not self._captured:
            raise RuntimeError("source step did not capture PRE-reset state")
        self._tick += 1
        self._issued[terminated | truncated] = 0.0
        result = StepResult(self._encode(raw), reward, terminated, truncated,
                            self._final, self._valid,
                            {**info, "evaluation_metrics": self._metrics})
        return self._contract.step(result)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # The process owner closes App only after CLI checkpoint/receipt writes.
        self.env.close()


def make_env(model_config, environment_config, device):
    """Factory for train/evaluate: external paths are explicit and hash recorded."""
    options = dict(environment_config)
    allowed = {"task_root", "contract", "num_envs", "mode", "stage", "fixed_command", "usd_seed"}
    if set(options) - allowed:
        raise ValueError(f"unknown environment options: {sorted(set(options) - allowed)}")
    if (model_config.proprio_dim, model_config.command_dim, model_config.action_dim,
            model_config.sensor_groups, model_config.critic_dim) != (16, 3, 6, 2, 29):
        raise ValueError("research adapter requires proprio16/command3/action6/sensors2/critic29")
    root = Path(options["task_root"]).expanduser().resolve(strict=True)
    contract = Path(options["contract"])
    contract = (contract if contract.is_absolute() else root / contract).resolve(strict=True)
    mode = options.get("mode", "train")
    count = options.get("num_envs", 2)
    if mode not in {"train", "evaluation"} or type(count) is not int or count < 1:
        raise ValueError("mode must be train/evaluation and num_envs a positive integer")
    if mode == "train" and "fixed_command" in options:
        raise ValueError("fixed_command requires evaluation mode")
    sys.path.insert(0, str(root / "src"))
    # AppLauncher must precede all external task/Isaac simulation imports.
    from isaaclab.app import AppLauncher

    from examples import _isaaclab_process
    if not _isaaclab_process._active:
        raise RuntimeError("Sim6 requires python -m examples._isaaclab_process train/evaluate "
                           "so application shutdown follows artifact writes")
    launcher = AppLauncher(headless=True, device=str(device), enable_cameras=False,
                           fast_shutdown=True)
    app = launcher.app
    _isaaclab_process.register_app(app)
    env = None
    try:
        from wheeled_tasks.direct.v40_serial.env import V40Env
        from wheeled_tasks.direct.v40_serial.env_cfg import V40EnvCfg
        from wheeled_tasks.v40.core import build_observation, build_critic, load_contract, is_round2

        source_path = Path(sys.modules[V40Env.__module__].__file__).resolve()
        if not source_path.is_relative_to(root):
            raise RuntimeError("external task was already imported from another root")
        if not is_round2(load_contract(str(contract))):
            raise ValueError("research adapter requires the v2 task contract")
        cfg = V40EnvCfg()
        cfg.contract_path = str(contract)
        cfg.allow_research = True
        cfg.stage = options.get("stage", "locomotion")
        cfg.scene.num_envs = count
        cfg.seed = torch.initial_seed() % (2**31)
        cfg.sim.device = str(device)
        cfg.usd_cache_dir = tempfile.mkdtemp(prefix="transformer-rl-usd-")
        if options.get("usd_seed"):
            cfg.usd_seed = str(Path(options["usd_seed"]).resolve(strict=True))
        env = V40Env(cfg, render_mode=None)
        if mode == "evaluation":
            env.set_evaluation_command(tuple(options.get("fixed_command", (0.0, 0.0, 0.30))))
        files = sorted((root / "src").rglob("*.py"))
        source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        metadata = {
            "task_root": str(root), "contract": str(contract), "mode": mode,
            "contract_sha256": env.contract_sha256,
            "contract_file_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            "asset_manifest_sha256": env.asset_manifest_sha256,
            "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            "source_files_sha256": source_hashes,
            "usd_cache_dir": cfg.usd_cache_dir,
            "usd_seed_sha256": (hashlib.sha256(Path(cfg.usd_seed).read_bytes()).hexdigest() if cfg.usd_seed else None),
            "critic_auxiliary_semantics": {"25": "true_body_com_vx_m_s", "26": "true_body_com_vy_m_s",
                                           "27": "true_body_com_vz_m_s"},
            "sensor_age": "simulated_current_state_known_zero_not_hardware_evidence",
            "timestamp": "independent_monotonic_float64_policy_event_seconds",
            "action": "previous_issued_before_source_clipping_zero_extra_transport_delay",
            "controller": "source_joint_space_feedback_each_physics_substep",
            "final_command": "preceding_transition_command_before_resample",
            "contact": "rigid_body_net_force_history_peak_not_ground_pair",
            "hardware_deployment_ready": False,
        }
        metadata["identity"] = {key: metadata[key] for key in (
            "contract_sha256", "asset_manifest_sha256", "source_sha256", "usd_seed_sha256",
        )}
        metadata["identity"].update(
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            worker_sha256=hashlib.sha256(Path(_isaaclab_process.__file__).read_bytes()).hexdigest(),
        )
        print("ISAACLAB_TASK_METADATA=" + json.dumps(metadata, sort_keys=True), flush=True)
        return IsaacLabTaskAdapter(env, app, model_config, metadata, build_observation, build_critic)
    except BaseException:
        if env is not None:
            env.close()
        raise

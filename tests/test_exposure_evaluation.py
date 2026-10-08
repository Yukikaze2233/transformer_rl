"""Actual CPU PPO, history inference, disk traces and physical-provider checks.

The six-motor environment has synthetic physical responses and asynchronous
resets. It is not an Isaac or hardware qualification. Protocol runtime and
predecessor fixtures, where used, are explicitly scoped to CPU authorization.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from transformer_rl import exposure_evaluation as evaluator
from transformer_rl import exposure_training as training
from transformer_rl import exposure_campaign as campaign
from transformer_rl import exposure_protocol as protocol
from transformer_rl import history_study, queue_validation
from transformer_rl.config import PPOConfig
from transformer_rl.frame_checkpoint import load_frame_checkpoint
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest, json_bytes
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.frame_training import FrameHistory
from transformer_rl.frame_workflow import evaluate_frame_policy
from transformer_rl.types import StepResult, VectorObservation
from test_exposure_protocol import prepared, freeze, write_json


REAL_RUNTIME = protocol.runtime_identity


class SixMotorFixture:
    """Owned PRE-reset packets; actual issued actions drive the toy dynamics."""

    def __init__(self, model_config, environment_config, device):
        self.config = model_config
        self.options = deepcopy(environment_config)
        self.device = torch.device(device)
        self.num_envs = environment_config["num_envs"]
        self.control = environment_config.get("control")
        if self.control is None:
            self.control = json.loads((Path(environment_config["snapshot"]) / "snapshot.json").read_bytes())["control"]
        self.dt = self.control["policy_dt_s"]
        self.state = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous = torch.zeros_like(self.state)
        self.previous_issued = torch.zeros(self.num_envs, 6, device=self.device)
        self.age = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.episodes = torch.zeros_like(self.age)
        self.periods = 4 + 2 * (torch.arange(self.num_envs, device=self.device) % 2)
        self.tick = 0
        self.closed = False
        self.enable_control_metrics = False
        self.metadata = {"identity": environment_config.get("snapshot_sha256", "synthetic_six_motor_fixture"),
                         "control_sha256": digest(self.control),
                         "control_packet_joint_order": [f"motor_{i}" for i in range(6)]}
        if "contracts" in environment_config:
            from transformer_rl.chassis_adapter import merge_evaluation_contracts
            contract = merge_evaluation_contracts(Path(environment_config["snapshot"]), environment_config["contracts"])
            labels = [case["name"] for case in contract["evaluation"]["cases"]
                      for _ in range(environment_config["contracts"][0]["num_envs"])]
            self.metadata["contract_sha256"] = digest(contract)
        elif "contract" in environment_config:
            contract = json.loads((Path(environment_config["snapshot"]) / environment_config["contract"]).read_bytes())
            self.metadata["contract_sha256"] = digest(contract)
            labels = None
        else:
            labels = environment_config.get("labels", ["alpha"] * (self.num_envs // 2)
                + ["beta"] * (self.num_envs - self.num_envs // 2))
        if labels is not None:
            self.metadata["evaluation_groups"] = labels

    def _observation(self):
        frame = torch.cat((torch.ones(self.num_envs, 1, device=self.device), self.state, self.previous), -1)
        critic = torch.cat((self.state, torch.ones(self.num_envs, 1, device=self.device)), -1)
        return VectorObservation(frame,
            torch.full((self.num_envs,), self.tick * self.dt, dtype=torch.float64, device=self.device),
            frame[:, self.config.command_indices], critic)

    def reset(self, seed=None):
        self.state.zero_()
        self.previous.zero_()
        self.previous_issued.zero_()
        self.age.zero_()
        self.episodes.zero_()
        self.tick = 0
        return self._observation()

    def step(self, issued_action):
        bounds = torch.tensor(self.control["action_bounds"], device=self.device)
        assert issued_action.shape == (self.num_envs, 6)
        assert torch.isfinite(issued_action).all() and (issued_action.abs() <= bounds).all()
        issued_rate = (issued_action - self.previous_issued).square().mean(-1).sqrt() / self.dt
        issued_rate[self.age == 0] = 0.
        self.previous_issued.copy_(issued_action)
        self.state += issued_action[:, :2] * self.dt
        self.previous.copy_(issued_action[:, :2])
        self.age += 1
        self.tick += 1
        done = self.age >= self.periods
        failures = done & (torch.arange(self.num_envs, device=self.device) % 2 == 0)
        successes = done & ~failures
        final = self._observation().critic.clone()
        reference = torch.zeros(self.num_envs, 3, device=self.device)
        reference[:, 2] = .3
        actual = reference.clone()
        actual[:, 0] = .1 + .03 * self.episodes + self.state[:, 0]
        actual[:, 2] += .005 * self.episodes
        time = self.age.double() * self.dt
        positions = torch.stack((self.age.float() * .001 + self.episodes.float() * 100.,
                                 self.age.float() * .002), -1)
        legs = .2 + issued_action[:, :4] * .25
        wheels = issued_action[:, 4:] * 10.
        requested = issued_action * 20.
        effort = requested.clamp(-.5, .5)
        motor_velocity = torch.cat((issued_action[:, :4], wheels), -1)
        height_error = actual[:, 2] - reference[:, 2]
        vx_error = actual[:, 0] - reference[:, 0]
        signals = {"height_error": height_error, "vx_error": vx_error, "wz_error": torch.zeros_like(vx_error)}
        signals.update({f"leg_target_{i}": legs[:, i] for i in range(4)})
        signals.update({f"wheel_target_{i}": wheels[:, i] for i in range(2)})
        signals.update({f"effort_{i}": effort[:, i] for i in range(6)})
        info = {"episode_success": successes.clone(),
                "evaluation_metrics": {"height_abs_error": height_error.abs(), "vx_abs_error": vx_error.abs(),
                    "wz_abs_error": torch.zeros_like(vx_error), "tilt_angle": torch.zeros_like(vx_error),
                    "drift_m": torch.linalg.vector_norm(torch.stack((self.age.float() * .001,
                                                                    self.age.float() * .002), -1), dim=-1),
                    "issued_action_rate_rms": issued_rate},
                "evaluation_signals": signals, "evaluation_signal_time": time.clone()}
        if self.enable_control_metrics:
            info["control_packet"] = {"time_s": time.clone(), "command_reference": reference.clone(),
                "actual": actual.clone(), "position_xy": positions.clone(),
                "tilt": torch.zeros(self.num_envs, device=self.device),
                "leg_target": legs.clone(), "wheel_target": wheels.clone(),
                "motor_position": torch.cat((legs, torch.zeros_like(wheels)), -1),
                "motor_velocity": motor_velocity.clone(), "motor_effort": effort.clone(),
                "requested_motor_effort": requested.clone(),
                "effort_bounds": torch.tensor([[-.5, .5]], device=self.device).expand(self.num_envs, 6, 2).clone(),
                "scaled_nominal_requested_motor_effort": requested.clone(),
                "scaled_nominal_effort_bounds": torch.tensor([[-.5, .5]], device=self.device).expand(self.num_envs, 6, 2).clone(),
                "failure": failures.clone(), "success": successes.clone()}
        reward = 1. - vx_error.square() - height_error.square()
        self.state[done] = 0.
        self.previous[done] = 0.
        self.previous_issued[done] = 0.
        self.age[done] = 0
        self.episodes += done.to(torch.int64)
        return StepResult(self._observation(), reward, failures, done & ~failures, final, done, info)

    def close(self):
        self.closed = True


def make_six_motor_env(model_config, environment_config, device):
    return SixMotorFixture(model_config, environment_config, device)


def configuration(architecture="mlp", history=1, *, readout="last", residual="add", num_envs=4):
    policy = FramePolicyConfig(architecture=architecture, frame_dim=5, action_dim=6,
        history_length=history, actor_hidden_dims=(8,), encoder_hidden_dims=(8,), history_latent_dim=3,
        d_model=8, num_heads=2, num_layers=1, ffn_dim=12, readout_type=readout,
        residual_type=residual, mean_init_scale=.2)
    control = {"policy_dt_s": .01, "observation_schema": "synthetic_six_motor",
        "feature_names": [f"feature_{i}" for i in range(5)],
        "action_names": [f"motor_{i}" for i in range(6)], "action_bounds": [.001] * 6,
        "target_scale": [.25] * 4 + [10.] * 2, "target_offset": [.2] * 4 + [0.] * 2,
        "target_units": ["rad"] * 4 + ["rad/s"] * 2}
    return FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(8,),
        command_indices=(0,), initial_std=.2),
        PPOConfig(epochs=2, num_minibatches=2, learning_rate=3e-5, target_kl=.5),
        control, {"control": control, "num_envs": num_envs})


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trained_checkpoint(tmp_path, config):
    result = training.train_exposure_segment({"name": "physical_fixture", "config": config, "updates": 2},
        make_six_motor_env, "test_exposure_evaluation:make_six_motor_env", tmp_path / "training",
        job_id="six_motor_cpu", rollout_steps=2, training_seed=71, retention_seed=901,
        evaluation_seeds=(701, 1701), device="cpu", max_seconds=60.,
        expected_initial_model_sha256=training.initial_model_sha256(config, 71))
    assert result["status"] == "completed"
    endpoint = result["endpoints"][0]
    model, trainer, _, update, metadata, _ = load_frame_checkpoint(endpoint["checkpoint"]["path"])
    assert update == 2 and trainer.optimizer.state_dict()["state"]
    assert metadata["continuation_segment"]["successful_updates"] == 2
    return Path(endpoint["checkpoint"]["path"]), result


def replay(report, path, config):
    from transformer_rl.exposure_trace import verify_trace_archive
    trace = report["trace"]
    expected = {key: trace[key] for key in ("checkpoint_sha256", "checkpoint_update", "seed", "steps",
        "policy_dt_s", "sampling_hz", "control_sha256", "row_indices", "group_labels", "history_length",
        "pre_inference_age_semantics", "evaluation_metric_names", "evaluation_signal_names")}
    return verify_trace_archive(path, trace, expected, steps=report["steps"], rows=report["num_envs"],
        history_length=config.model.history_length, action_bounds=config.control["action_bounds"],
        settle_steps=1, min_steady_samples=1)


@pytest.mark.parametrize("architecture,history,readout,residual", [
    ("mlp", 1, "last", "add"), ("history_mlp", 4, "last", "add"),
    ("transformer", 4, "query", "add"), ("transformer", 4, "last", "gated")])
def test_actual_trained_policy_history_age_and_raw_action_full_trace_replay(tmp_path,
        architecture, history, readout, residual):
    config = configuration(architecture, history, readout=readout, residual=residual)
    checkpoint, _ = trained_checkpoint(tmp_path, config)
    trace_path = tmp_path / "trace.npz"
    created = []

    def factory(**arguments):
        env = make_six_motor_env(**arguments)
        created.append(env)
        return env

    report = evaluate_frame_policy(checkpoint, factory, config.environment, steps=13, seed=701,
        settle_steps=1, min_steady_samples=1, control_metrics=True, history_control=True,
        trace_output=trace_path, trace_replicas=2)
    assert created[0].closed and created[0].enable_control_metrics
    reconstructed = replay(report, trace_path, config)
    fields = ("control", "history_control", "metrics", "reward_mean", "stability",
              "completed_episodes", "failed_episodes", "success_metric_available", "success_rate")
    for key in fields:
        assert evaluator._same_metrics(report[key], reconstructed[key]), key
        for name in ("alpha", "beta"):
            assert evaluator._same_metrics(report["groups"][name][key], reconstructed["groups"][name][key]), (name, key)
    with np.load(trace_path, allow_pickle=False) as archive:
        ages = archive["pre_inference_episode_age"]
        expected_ages = np.stack([np.arange(13) % period for period in (4, 6, 4, 6)], axis=1)
        np.testing.assert_array_equal(ages, expected_ages)
        np.testing.assert_array_equal(archive["time_s"], (expected_ages + 1) * .01)
        raw, issued = archive["raw_policy_mean"], archive["issued_action"]
        np.testing.assert_array_equal(issued, np.clip(raw, -.001, .001))
        assert np.any(raw != issued)
        full_count = int((expected_ages >= history - 1).sum())
        windows = report["history_control"]["windows"]
        assert windows["full_history"]["samples"] == full_count
        assert windows["reset_filled"]["samples"] == 52 - full_count
        if history == 1:
            assert windows["reset_filled"]["tracking"]["axes"]["vx"]["mae"] is None
            assert windows["reset_filled"]["actor_clipping_fraction"] == [None] * 6
        else:
            assert windows["full_history"]["samples"] > 0 and windows["reset_filled"]["samples"] > 0
        # Large position offsets at reset are excluded from drift derivatives;
        # between-episode bias is excluded from centered motion jitter.
        assert report["control"]["planar_motion"]["full_interval"]["max_speed_m_s"] < .3
        tracking = windows["full_history"]["tracking"]["axes"]["vx"]
        assert tracking["group_mean_std"] > .01 and tracking["within_group_std"] < .001
        for category in ("reset_filled", "full_history"):
            membership = expected_ages >= history - 1
            if category == "reset_filled":
                membership = ~membership
            intervals = int((membership[1:] & membership[:-1] & (expected_ages[1:] != 0)).sum())
            assert windows[category]["rates"]["issued_action_rate"][0]["count"] == intervals
        model, _, _, _, _, _ = load_frame_checkpoint(checkpoint)
        env = make_six_motor_env(config.model, config.environment, "cpu")
        env.enable_control_metrics = True
        buffer = FrameHistory(model.config, env.num_envs, env.device)
        current = buffer.append(env.reset(seed=701))
        for tick in range(13):
            mean = model.actor(current)
            np.testing.assert_array_equal(mean.detach().numpy(), raw[tick])
            result = env.step(mean.clamp(-torch.tensor(config.control["action_bounds"]),
                                        torch.tensor(config.control["action_bounds"])))
            done = result.terminated | result.truncated
            buffer.reset(done)
            current = buffer.append(result.observation)
            if done.any():
                for row in torch.nonzero(done).flatten():
                    assert torch.equal(current.frames[row], current.frames[row, -1:].expand_as(current.frames[row]))
        env.close()


def canonical(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def synthetic_closed(*args, **kwargs):
    return {"status": "completed", "controller_live": False, "live_workers": [],
            "fixture": "synthetic predecessor closure only; no production queue proof"}


@pytest.fixture
def physical_protocol(prepared, monkeypatch):
    """Real protocol/model/runtime bytes; predecessor definitions are synthetic.

    The SDK-pinned sitecustomize module substitutes original queue closure and
    the Isaac environment factory resolver only. This is an actual CPU worker
    integration, not production queue closure, Isaac or hardware evidence.
    """
    p = prepared
    base = configuration("transformer", 4, residual="gated")
    contracts = {}
    for name in ("first", "second", "normal", "new_skill"):
        replicas = 4 if name in ("first", "second") else 2
        value = {"name": "common_synthetic_physics", "target_num_envs": replicas,
                 "physics_dt": .005, "policy_dt": .01, "evaluation_exact_cases": True,
                 "evaluation": {"cases": [{"name": name, "terrain": "flat", "task": "survive"}],
                                "episodes_per_case": 2, "stable_case_layout": False}}
        path = p["snapshot"] / "contracts" / f"{name}.json"
        write_json(path, value)
        contracts[name] = {"contract": str(path.relative_to(p["snapshot"])),
                          "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "num_envs": replicas}
    files = {str(path.relative_to(p["snapshot"])): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(p["snapshot"].rglob("*")) if path.is_file() and path.name != "snapshot.json"}
    snapshot_sha = digest(files)
    write_json(p["snapshot"] / "snapshot.json", {"files": files, "sha256": snapshot_sha, "control": base.control})
    base = replace(base, environment={"snapshot": str(p["snapshot"]), "snapshot_sha256": snapshot_sha,
                                     **contracts["first"], "control": base.control})
    spec = deepcopy(p["spec"])
    spec["environment_factory"] = "transformer_rl.chassis_adapter:make_env"
    spec["variants"] = [
        {"name": "mlp", "policy": {"architecture": "mlp", "history_length": 1, "residual_type": "add"}},
        {"name": "history_mlp", "policy": {"architecture": "history_mlp", "residual_type": "add"}},
        {"name": "gated", "policy": {"architecture": "transformer", "residual_type": "gated"}}]
    for stage in spec["stages"]:
        stage["environment"] = contracts[stage["name"]]
        stage["updates"] = 2
    for scenario in spec["scenarios"]:
        scenario["environment"] = contracts[scenario["name"]]
    spec["training"]["rollout_steps"] = 2
    spec["evaluation"]["steps"] = 13
    spec["selection"]["objectives"][0]["path"] = "metrics.vx_abs_error.mean"
    write_json(p["inputs"] / "base.json", base.to_dict())
    write_json(p["inputs"] / "study.json", spec)
    fixture_source = ("from copy import deepcopy\nimport json\nfrom pathlib import Path\nimport torch\n"
                      "from transformer_rl.frame_config import digest\n"
                      "from transformer_rl.types import StepResult, VectorObservation\n\n"
                      + inspect.getsource(SixMotorFixture) + "\n" + inspect.getsource(make_six_motor_env))
    (p["sdk"] / "physical_cpu_fixture.py").write_text(fixture_source)
    startup = '''
# CPU fixture substitutes ONLY original queue providers and Isaac resolution.
import json,os
from copy import deepcopy
from pathlib import Path
from transformer_rl import exposure_protocol as protocol,queue_validation,cli
p=json.loads(Path(os.environ['EXPOSURE_CPU_FIXTURE_PROTOCOL']).read_bytes())
protocol.predecessors._dependencies=lambda *a: (deepcopy(p['execution']['dependencies']),deepcopy(p['execution']['resource_locks']))
queue_validation.check_dependency=lambda *a,**k: {'status':'completed','controller_live':False,'live_workers':[],'fixture':'synthetic closure only'}
import physical_cpu_fixture
def cpu_factory(reference):
    if reference != 'transformer_rl.chassis_adapter:make_env':
        raise ValueError('fixture factory reference differs')
    return physical_cpu_fixture.make_six_motor_env
cli._factory=cpu_factory
'''
    # A failing fixture startup must not fall through to the real Isaac factory.
    (p["sdk"] / "sitecustomize.py").write_text("import os\ntry:\n" +
        "".join("    " + line + "\n" for line in startup.splitlines()) +
        "except BaseException:\n    os._exit(81)\n")
    p["history"] = p["tmp"] / "physical_history"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json", p["history"],
        history_lengths=[1, 4], position_reference="current")
    monkeypatch.setattr(protocol, "runtime_identity", REAL_RUNTIME)
    monkeypatch.setattr(queue_validation, "check_dependency", synthetic_closed)
    value = freeze(p)
    protocol_path = p["tmp"] / "physical_protocol.json"
    canonical(protocol_path, value)
    p["protocol"], p["protocol_path"] = value, protocol_path
    p["base"], p["spec"] = base, spec
    job = next(job for job in value["jobs"] if job["candidate"] == "gated_h4" and job["training_seed"] == 71)
    p["job"] = job
    root = Path(value["output_root"])
    root.mkdir()
    (root / job["id"]).mkdir()
    stage_root = root / job["id"] / "stage_0000"
    stage_root.mkdir()
    stage = job["stages"][0]
    config = FrameTrainConfig.from_dict(stage["config"])
    result = training.train_exposure_segment({key: stage[key] for key in ("name", "config", "updates")},
        make_six_motor_env, value["environment_factory"], stage_root / "train", job_id=job["id"],
        rollout_steps=value["execution"]["rollout_steps"], training_seed=job["training_seed"],
        retention_seed=job["retention_seed"],
        evaluation_seeds=[*value["evaluation"]["validation_seeds"], *value["evaluation"]["seeds"]],
        device="cpu", max_seconds=value["execution"]["max_seconds"],
        expected_initial_model_sha256=job["initial_model_sha256"])
    assert result["status"] == "completed"
    endpoint_path = Path(result["endpoints"][0]["checkpoint"]["path"]).parent / "endpoint.json"
    p["endpoint"] = protocol._receipt(endpoint_path)
    p["cells"] = [cell for cell in value["evaluation_cells"] if cell["job_id"] == job["id"]
                  and cell["stage_index"] == 0 and cell["role"] == "validation" and cell["seed"] == 701]
    p["directory"] = stage_root / "evaluation_validation_701"
    # Actual four-kind storage calibration. These are CPU fixture artifacts;
    # they do not estimate production Isaac startup/cache or large-model caps.
    measurements = p["tmp"] / "actual_storage_measurements"
    measurements.mkdir()
    sample_trace = measurements / "actual_cpu_trace.npz"
    calibration_report = evaluate_frame_policy(result["endpoints"][0]["checkpoint"]["path"],
        make_six_motor_env, config.environment, steps=4, seed=777, settle_steps=1,
        min_steady_samples=1, control_metrics=True, history_control=True,
        trace_output=sample_trace, trace_replicas=4)
    assert calibration_report["transitions"] == 16
    cache_root = measurements / "actual_runtime_cache"
    cache_root.mkdir()
    (cache_root / "fixture_source.bin").write_bytes((p["sdk"] / "physical_cpu_fixture.py").read_bytes())
    cache = measurements / "actual_runtime_cache_inventory.json"
    cache_files = [protocol._receipt(cache_root / "fixture_source.bin")]
    canonical(cache, {"format": "transformer_rl.exposure_runtime_cache_measurement", "schema_version": 1,
        "root": str(cache_root), "files": cache_files, "total_bytes": sum(item["bytes"] for item in cache_files)})
    storage = p["tmp"] / "actual_storage_fixture.json"
    storage_contract = {"format": "transformer_rl.exposure_storage_contract", "schema_version": 1,
        "protocol_raw_sha256": protocol._receipt(protocol_path)["sha256"], "source": value["source"],
        "caps": {"checkpoint_bytes": 16 * 1024**2, "trace_bytes_per_policy_sample": 4096,
                 "metric_bytes_per_update": 64 * 1024, "inflight_bytes": 8 * 1024**2,
                 "runtime_cache_bytes": 4 * 1024**2, "free_margin_bytes": 4 * 1024**2},
        "measurements": [{"kind": kind, "receipt": protocol._receipt(path), "units": units}
            for kind, path, units in (("checkpoint", Path(result["endpoints"][0]["checkpoint"]["path"]), 1),
                ("trace", sample_trace, 16), ("metric", stage_root / "train" / "metrics.jsonl", 2),
                ("runtime_cache", cache, 1))]}
    canonical(storage, storage_contract)
    assert campaign._validate_storage(storage_contract, value, protocol._receipt(protocol_path)) == storage_contract
    p["storage"], p["storage_path"] = storage_contract, storage
    controller = {"format": "transformer_rl.exposure_controller", "schema_version": 1,
        "protocol_sha256": value["sha256"], "source": value["source"], "runtime": value["runtime"],
        "process": campaign._identity(os.getpid()), "protocol_raw_receipt": protocol._receipt(protocol_path),
        "expected_protocol_sha256": protocol._receipt(protocol_path)["sha256"],
        "storage_contract_receipt": protocol._receipt(storage)}
    canonical(root / "controller.json", controller)
    p["controller"] = protocol._receipt(root / "controller.json")
    return p


def run_physical_worker(p):
    """Actual canonical CLI child; launcher behavior is tested separately.

    Environment startup inserts a runtime-pinned synthetic queue/factory shim.
    It does not replace request, chain, trace, lease or result validators.
    """
    value = p["protocol"]
    with campaign.resource_lease(value, lambda *args, **kwargs: None) as leases:
        planned = evaluator.plan_request(value, p["endpoint"], p["cells"], p["directory"], leases, p["controller"])
        environment = os.environ.copy()
        environment.update(PYTHONPATH=os.pathsep.join((str(Path(__file__).resolve().parent.parent / "src"), str(p["sdk"]))),
            PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(p["directory"] / "empty_python_cache"),
            EXPOSURE_CPU_FIXTURE_PROTOCOL=str(p["protocol_path"]), CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        Path(environment["PYTHONPYCACHEPREFIX"]).mkdir()
        from transformer_rl import runtime_paths
        environment = runtime_paths.prepare_worker_runtime(p["directory"], environment)
        runtime_profile = runtime_paths.profile_receipt(
            runtime_paths.validate_runtime_profile(p["directory"], environ=environment))
        with (p["directory"] / "stdout.txt").open("xb") as stdout, (p["directory"] / "stderr.txt").open("xb") as stderr:
            child = subprocess.Popen(planned["command"], stdout=stdout, stderr=stderr, env=environment,
                                     pass_fds=tuple(lease["descriptor"] for lease in leases), start_new_session=True)
            opening, sending = campaign.predecessors._pidfd_api()
            handle = opening(child.pid)
            try:
                identity = campaign._startup_identity(child.pid, planned["command"], 60.)
                process = {"format": "transformer_rl.exposure_worker_process", "schema_version": 1,
                           "process": identity, "command": planned["command"], "leases": leases,
                           "runtime_profile": runtime_profile}
                canonical(p["directory"] / "worker.process.json", process)
                code = child.wait(timeout=60.)
            except BaseException:
                if child.poll() is None:
                    import signal
                    sending(handle, signal.SIGTERM)
                child.wait(timeout=10.)
                raise
            finally:
                os.close(handle)
        canonical(p["directory"] / "worker.completion.json", {
            "format": "transformer_rl.exposure_worker_completion", "schema_version": 1,
            "process": identity, "command": planned["command"], "returncode": code, "timed_out": False,
            "runtime_profile": runtime_profile})
        assert code == 0, (p["directory"] / "stderr.txt").read_text()
        return planned


def test_actual_canonical_worker_and_complete_physical_result_reconstruction(physical_protocol):
    p = physical_protocol
    planned = run_physical_worker(p)
    result = evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
    assert result["status"] == "completed" and set(result["cells"]) == {cell["id"] for cell in p["cells"]}
    assert all(record["identity"] == cell for cell in p["cells"] for record in [result["cells"][cell["id"]]])
    assert result["trace_validation"]["steps"] == 13
    assert result["hardware_verified"] is False and result["formal_architecture_selection"] is False
    assert campaign.predecessors._process(json.loads((p["directory"] / "worker.completion.json").read_bytes())["process"]["pid"]) is None
    worker_path, process_path = p["directory"] / "worker.completion.json", p["directory"] / "worker.process.json"
    worker_bytes, process_bytes = worker_path.read_bytes(), process_path.read_bytes()
    mutations = [lambda worker, process: worker.update(returncode=1),
                 lambda worker, process: worker.update(returncode=False),
                 lambda worker, process: worker.update(timed_out=True),
                 lambda worker, process: worker.update(schema_version=True),
                 lambda worker, process: process.update(schema_version=True),
                 lambda worker, process: worker["process"].update(pid=os.getpid()),
                 lambda worker, process: worker.pop("runtime_profile"),
                 lambda worker, process: worker["runtime_profile"].update(bytes=1),
                 lambda worker, process: process["runtime_profile"].update(sha256="0" * 64),
                 lambda worker, process: worker["command"].append("--extra"),
                 lambda worker, process: (worker["process"].update(argv=["fabricated"]),
                                          process["process"].update(argv=["fabricated"]))]
    for mutate in mutations:
        worker, process = json.loads(worker_bytes), json.loads(process_bytes)
        mutate(worker, process)
        canonical(worker_path, worker)
        canonical(process_path, process)
        with pytest.raises(ValueError):
            evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
        worker_path.write_bytes(worker_bytes)
        process_path.write_bytes(process_bytes)
    assert evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])["status"] == "completed"


def test_batch_preserves_every_declared_scenario_and_forbids_held_out_selection(physical_protocol):
    p = physical_protocol
    value, cells, endpoint, directory = p["protocol"], p["cells"], p["endpoint"], p["directory"]
    actual = evaluator._batch(value, endpoint, cells, directory)
    assert [case["scenario"] for case in cells] == ["normal", "new_skill"]
    assert actual["environment"]["num_envs"] == 4
    changes = [cells[:1], list(reversed(cells)), [cells[0], cells[0]], []]
    for key, replacement in (("job_id", "unknown"), ("stage_index", 1), ("seed", True),
                             ("num_envs", 1), ("scenario", "normal"), ("role", "held_out")):
        changed = deepcopy(cells)
        changed[1][key] = replacement
        changes.append(changed)
    for changed in changes:
        with pytest.raises((ValueError, KeyError)):
            evaluator._batch(value, endpoint, changed, directory)
    held_out = [cell for cell in value["evaluation_cells"] if cell["job_id"] == p["job"]["id"]
                and cell["stage_index"] == 0 and cell["role"] == "held_out" and cell["seed"] == 2701]
    with pytest.raises(ValueError, match="immutable validation choice"):
        evaluator._batch(value, endpoint, held_out, directory.parent / "evaluation_held_out_2701")
    with pytest.raises(ValueError, match="fixed role/seed output"):
        evaluator._batch(value, endpoint, cells, directory.parent / "arbitrary")
    external_factory = deepcopy(value)
    external_factory["environment_factory"] = "test_exposure_evaluation:make_six_motor_env"
    with pytest.raises(ValueError):
        evaluator._batch(external_factory, endpoint, cells, directory)
    assert not directory.exists()


def test_request_rebuild_and_actual_checkpoint_chain_reject_identity_changes(physical_protocol):
    p = physical_protocol
    with campaign.resource_lease(p["protocol"], lambda *args, **kwargs: None) as leases:
        planned = evaluator.plan_request(p["protocol"], p["endpoint"], p["cells"], p["directory"], leases, p["controller"])
    path = Path(planned["request"]["path"])
    original = path.read_bytes()
    request = json.loads(original)
    mutations = [lambda item: item["cells"].pop(),
                 lambda item: item["source"].update(sha256="0" * 64),
                 lambda item: item["environment"].update(num_envs=2),
                 lambda item: item["key"].update(role="held_out"),
                 lambda item: item["evaluation"].update(steps=1),
                 lambda item: item.update(effective_contract_sha256="0" * 64),
                 lambda item: item["leases"].pop(),
                 lambda item: item["leases"][0].update(descriptor=True),
                 lambda item: item["leases"][0].update(controller_start=1),
                 lambda item: item.update(schema_version=True)]
    for mutate in mutations:
        changed = deepcopy(request)
        mutate(changed)
        canonical(path, changed)
        with pytest.raises(ValueError):
            evaluator._request(protocol._receipt(path))
        path.write_bytes(original)
    value, actual, batch = evaluator._request(planned["request"])
    assert actual == p["protocol"] and value == request and batch["endpoint"]["stage_index"] == 0
    endpoint = json.loads(Path(p["endpoint"]["path"]).read_bytes())
    checkpoint = Path(endpoint["checkpoint"]["path"])
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"modified actual checkpoint")
    try:
        with pytest.raises(ValueError):
            evaluator._request(planned["request"])
    finally:
        checkpoint.write_bytes(checkpoint_bytes)
    assert evaluator._request(planned["request"])[1] == p["protocol"]


def test_resealed_report_metrics_cannot_replace_full_physical_replay(physical_protocol):
    p = physical_protocol
    planned = run_physical_worker(p)
    report_path = p["directory"] / "report.json"
    completion_path = p["directory"] / "evaluation.completion.json"
    report_bytes, completion_bytes = report_path.read_bytes(), completion_path.read_bytes()
    original = json.loads(report_bytes)
    assert evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
    mutations = [lambda report: report.update(reward_mean=0.),
        lambda report: report.update(completed_episodes=0),
        lambda report: report.update(failed_episodes=0),
        lambda report: report.update(success_rate=1.),
        lambda report: report["metrics"]["vx_abs_error"].update(mean=0.),
        lambda report: report["control"]["full_interval"]["axes"]["vx"].update(mae=0.),
        lambda report: report["history_control"]["windows"]["reset_filled"].update(samples=0),
        lambda report: report["groups"]["normal"]["metrics"]["vx_abs_error"].update(mean=0.),
        lambda report: report["groups"]["new_skill"]["history_control"]["windows"]["full_history"].update(samples=0),
        lambda report: report["stability"]["signals"]["vx_error"].update(count=0),
        lambda report: report["metrics"].pop("wz_abs_error"),
        lambda report: report["stability"]["signals"].pop("wz_error"),
        lambda report: report["environment_provenance"].update(evaluation_groups=["normal"] * 4),
        lambda report: report.update(checkpoint_update=3),
        lambda report: report.update(seed=1701)]
    for mutate in mutations:
        changed = deepcopy(original)
        mutate(changed)
        canonical(report_path, changed)
        completion = json.loads(completion_bytes)
        completion["report"] = protocol._receipt(report_path)
        canonical(completion_path, completion)
        with pytest.raises(ValueError):
            evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
        report_path.write_bytes(report_bytes)
        completion_path.write_bytes(completion_bytes)
    assert evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])["status"] == "completed"


def test_metric_comparison_requires_complete_finite_typed_statistics():
    expected = {"mean": .25, "count": 1, "empty": None, "flags": [True, False]}
    assert evaluator._same_metrics(deepcopy(expected), expected)
    assert evaluator._same_metrics({**expected, "mean": .25 + 1e-11}, expected)
    for changed in ({**expected, "mean": float("nan")}, {**expected, "mean": float("inf")},
                    {**expected, "mean": True}, {**expected, "count": True},
                    {**expected, "count": 1.}, {**expected, "empty": 0.},
                    {**expected, "flags": [1, 0]}, {**expected, "extra": 0},
                    {key: value for key, value in expected.items() if key != "count"}):
        assert not evaluator._same_metrics(changed, expected)


def test_later_stage_evaluation_binds_complete_actual_learning_chain(physical_protocol):
    p = physical_protocol
    stage = p["job"]["stages"][1]
    stage_root = Path(p["protocol"]["output_root"]) / p["job"]["id"] / "stage_0001"
    stage_root.mkdir()
    result = training.train_exposure_segment({key: stage[key] for key in ("name", "config", "updates")},
        make_six_motor_env, p["protocol"]["environment_factory"], stage_root / "train", job_id=p["job"]["id"],
        rollout_steps=p["protocol"]["execution"]["rollout_steps"], training_seed=p["job"]["training_seed"],
        retention_seed=p["job"]["retention_seed"], parent_endpoint=p["endpoint"],
        evaluation_seeds=[*p["protocol"]["evaluation"]["validation_seeds"], *p["protocol"]["evaluation"]["seeds"]],
        device="cpu", max_seconds=p["protocol"]["execution"]["max_seconds"],
        expected_initial_model_sha256=p["job"]["initial_model_sha256"])
    assert result["status"] == "completed" and result["endpoints"][0]["cumulative_successful_updates"] == 4
    ancestor = json.loads(Path(p["endpoint"]["path"]).read_bytes())
    checkpoint_path = Path(result["endpoints"][0]["checkpoint"]["path"])
    p["endpoint"] = protocol._receipt(checkpoint_path.parent / "endpoint.json")
    p["cells"] = [cell for cell in p["protocol"]["evaluation_cells"] if cell["job_id"] == p["job"]["id"]
                  and cell["stage_index"] == 1 and cell["role"] == "validation" and cell["seed"] == 701]
    p["directory"] = stage_root / "evaluation_validation_701"
    planned = run_physical_worker(p)
    assert evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
    ancestor_checkpoint = Path(ancestor["checkpoint"]["path"])
    ancestor_bytes = ancestor_checkpoint.read_bytes()
    ancestor_checkpoint.write_bytes(ancestor_bytes + b"ancestor checkpoint changed")
    try:
        with pytest.raises(ValueError):
            evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])
    finally:
        ancestor_checkpoint.write_bytes(ancestor_bytes)
    assert evaluator.verify_result(p["protocol"], p["endpoint"], p["cells"], p["directory"], planned["request"])["status"] == "completed"

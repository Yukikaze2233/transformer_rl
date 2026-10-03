"""Physical-control evaluation wiring on synthetic tensors, without a simulator."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_rl.chassis_adapter import ChassisFrameAdapter
from transformer_rl.frame_checkpoint import save_frame_checkpoint
from transformer_rl.frame_config import FrameModelConfig, digest
from transformer_rl.frame_training import FrameActorCritic
from transformer_rl.frame_workflow import evaluate_frame_policy
from transformer_rl.ppo import PPOTrainer

from test_chassis_adapter import TensorChassis
from test_frame_workflow import configuration
from packed_env import PackedFixture


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class PhysicalChassis(TensorChassis):
    """Reset mutates every physical-state buffer exposed to the adapter."""

    def __init__(self, *, diagnostic_joint_state=True, missing=None):
        super().__init__()
        self.diagnostic_joint_state = diagnostic_joint_state
        self.missing = missing
        joint_order = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]
        self.startup_report = {"active_joint_order": joint_order}
        self.cfg["policy_action_order"] = joint_order.copy()
        self.ids = torch.arange(6)
        self.motor_position = torch.zeros(3, 6)
        self.motor_velocity = torch.zeros(3, 6)
        self.motor_effort = torch.zeros(3, 6)
        self.requested_motor_effort = torch.zeros(3, 6)
        self.effort_limit = torch.full((3, 6), 20.)
        self.robot.data.joint_pos = SimpleNamespace(torch=self.motor_position)
        self.robot.data.joint_vel = SimpleNamespace(torch=self.motor_velocity)

    def reset(self, rows):
        super().reset(rows)
        for values in (self.motor_position, self.motor_velocity, self.motor_effort,
                       self.requested_motor_effort, self.effort_limit):
            values[rows] = -99.

    def step(self, issued):
        raw, reward, done, extras = super().step(issued)
        self.motor_position.fill_(.4)
        self.motor_velocity.fill_(2.)
        self.motor_effort.fill_(3.)
        self.requested_motor_effort.fill_(4.)
        self.effort_limit.fill_(20.)
        diagnostic = extras["diagnostics"]
        diagnostic["motor_effort"] = self.motor_effort
        diagnostic["requested_motor_effort"] = self.requested_motor_effort
        diagnostic["motor_effort_bounds"] = self.effort_limit
        if self.diagnostic_joint_state:
            diagnostic["motor_position"] = self.motor_position
            diagnostic["motor_velocity"] = self.motor_velocity
        if self.missing is not None:
            diagnostic.pop(self.missing, None)
        return raw, reward, done, extras


@pytest.mark.parametrize("diagnostic_joint_state", (False, True))
def test_adapter_control_packet_is_owned_before_reset(diagnostic_joint_state):
    env = PhysicalChassis(diagnostic_joint_state=diagnostic_joint_state)
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {}, enable_control_metrics=True)
    adapter.reset(seed=71)
    result = adapter.step(torch.zeros(3, 6))
    packet = result.info["control_packet"]
    assert packet["time_s"].dtype == torch.float64
    assert packet["failure"].dtype == packet["success"].dtype == torch.bool
    assert packet["actual"].shape == packet["command_reference"].shape == (3, 3)
    torch.testing.assert_close(packet["actual"][:, 2], torch.full((3,), .3))
    torch.testing.assert_close(packet["position_xy"], torch.full((3, 2), .5))
    for name, expected in (("motor_position", .4), ("motor_velocity", 2.),
                           ("motor_effort", 3.), ("requested_motor_effort", 4.)):
        torch.testing.assert_close(packet[name], torch.full((3, 6), expected))
    torch.testing.assert_close(packet["effort_bounds"][..., 0], torch.full((3, 6), -20.))
    torch.testing.assert_close(packet["effort_bounds"][..., 1], torch.full((3, 6), 20.))
    assert not env.pose.any()
    assert (env.motor_position == -99.).all()
    for value in packet.values():
        assert not (value == -99.).any()


def test_adapter_default_does_not_require_optional_physical_diagnostics():
    adapter = ChassisFrameAdapter(TensorChassis(), FrameModelConfig(), {})
    adapter.reset(seed=71)
    result = adapter.step(torch.zeros(3, 6))
    assert "control_packet" not in result.info


@pytest.mark.parametrize("missing", ("requested_motor_effort", "motor_effort_bounds"))
def test_adapter_enabled_physical_diagnostics_fail_closed(missing):
    env = PhysicalChassis(missing=missing)
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {}, enable_control_metrics=True)
    adapter.reset(seed=71)
    with pytest.raises((ValueError, KeyError), match=missing):
        adapter.step(torch.zeros(3, 6))


def test_adapter_uses_physical_endpoint_and_canonical_motor_order():
    from transformer_rl.control_metrics import ControlMetrics

    source_order = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
    canonical_order = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"]

    class InterleavedChassis(PhysicalChassis):
        def step(self, issued):
            raw, reward, done, extras = super().step(issued)
            self.motor_position[:] = torch.tensor([1., 2., 9., 3., 4., 8.])
            self.motor_velocity[:] = torch.tensor([11., 12., 91., 13., 14., 81.])
            self.motor_effort[:] = torch.tensor([.1, .2, .3, .4, .5, .6])
            self.requested_motor_effort[:] = torch.tensor([.11, .22, .33, .44, .55, .66])
            self.effort_limit[:] = torch.tensor([40., 40., 4.5, 40., 40., 4.5])
            diagnostic = extras["diagnostics"]
            # Deliberately stale feedback must not replace the physical endpoint.
            diagnostic["motor_position"] = torch.full((3, 6), -7.)
            diagnostic["motor_velocity"] = torch.full((3, 6), -8.)
            diagnostic["leg_target_position"][:] = torch.tensor([1., 2., 3., 4.])
            diagnostic["wheel_target_velocity"][:] = torch.tensor([91., 81.])
            return raw, reward, done, extras

    env = InterleavedChassis()
    env.cfg["policy_action_order"] = canonical_order
    metadata = {"startup": {"active_joint_order": source_order}}
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), metadata, enable_control_metrics=True)
    adapter.reset(seed=71)
    result = adapter.step(torch.zeros(3, 6))
    packet = result.info["control_packet"]
    for name, expected in (("motor_position", [1., 2., 3., 4., 9., 8.]),
                           ("motor_velocity", [11., 12., 13., 14., 91., 81.]),
                           ("motor_effort", [.1, .2, .4, .5, .3, .6]),
                           ("requested_motor_effort", [.11, .22, .44, .55, .33, .66])):
        torch.testing.assert_close(packet[name], torch.tensor(expected).expand(3, -1))
    torch.testing.assert_close(packet["effort_bounds"][..., 1],
        torch.tensor([40., 40., 40., 40., 4.5, 4.5]).expand(3, -1))
    assert (env.motor_position == -99.).all()
    metrics = ControlMetrics(3, .01, settle_steps=0, min_steady_samples=1)
    metrics.update(packet, result.terminated | result.truncated)
    report = metrics.report()["actuation"]
    for name in ("leg_position_error", "wheel_velocity_error"):
        assert all(channel["rms"] == 0. for channel in report[name]["channels"])


@pytest.mark.parametrize("invalid", ("missing_source", "missing_target", "different_names", "duplicate_source"))
def test_adapter_rejects_missing_or_ambiguous_motor_order(invalid):
    env = PhysicalChassis()
    if invalid == "missing_source":
        env.startup_report.pop("active_joint_order")
    elif invalid == "missing_target":
        env.cfg.pop("policy_action_order")
    elif invalid == "different_names":
        env.cfg["policy_action_order"][0] = "unidentified_motor"
    else:
        env.startup_report["active_joint_order"][0] = "LL_joint1"
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {}, enable_control_metrics=True)
    adapter.reset(seed=71)
    with pytest.raises(ValueError, match="active_joint_order.*policy_action_order"):
        adapter.step(torch.zeros(3, 6))


class PhysicalFixture(PackedFixture):
    """Physical signals are six-dimensional even for a two-action toy policy."""

    def __init__(self, model_config, environment_config, device, *, fail_at=None, corrupt=None):
        super().__init__(model_config, environment_config, device)
        self.metadata["evaluation_groups"] = ["a"] * 3 + ["b"] * 3
        self.enable_control_metrics = False
        self.fail_at, self.corrupt = fail_at, corrupt
        self.enabled_at_reset = None

    def reset(self, seed=None):
        self.enabled_at_reset = self.enable_control_metrics
        return super().reset(seed)

    def step(self, issued_action):
        endpoint_time = (self.age + 1).double() * self.dt
        result = super().step(issued_action)
        if not self.enable_control_metrics:
            return result
        n, device = self.num_envs, self.device
        command = torch.zeros(n, 3, device=device)
        command[:, 2] = .3
        actual = command.clone()
        actual[3:, 2] = .1
        position = torch.stack((torch.arange(n, device=device) + self.tick * .001,
                                torch.zeros(n, device=device)), -1)
        bounds = torch.empty(n, 6, 2, device=device)
        bounds[..., 0], bounds[..., 1] = -20., 20.
        packet = {"time_s": endpoint_time, "command_reference": command, "actual": actual,
                  "position_xy": position, "tilt": torch.zeros(n, device=device),
                  "leg_target": torch.full((n, 4), .2, device=device),
                  "wheel_target": torch.full((n, 2), .3, device=device),
                  "motor_position": torch.full((n, 6), .4, device=device),
                  "motor_velocity": torch.full((n, 6), 2., device=device),
                  "motor_effort": torch.full((n, 6), 3., device=device),
                  "requested_motor_effort": torch.full((n, 6), 4., device=device),
                  "effort_bounds": bounds, "failure": result.terminated.clone(),
                  "success": (result.info["episode_success"] & ~result.terminated).clone()}
        if self.tick == self.fail_at:
            if self.corrupt == "missing":
                return result
            if self.corrupt == "nonfinite":
                packet["motor_effort"][0, 0] = float("nan")
            elif self.corrupt == "wrong_dtype":
                packet["time_s"] = packet["time_s"].float()
        result.info["control_packet"] = packet
        return result


@pytest.fixture
def checkpoint(tmp_path):
    config = configuration()
    config = replace(config, environment={**config.environment, "num_envs": 6})
    model = FrameActorCritic(config.model)
    path = tmp_path / "checkpoint.pt"
    metadata = {"environment_provenance": {"identity": "synthetic_tensor_fixture",
                                            "control_sha256": digest(config.control)}}
    save_frame_checkpoint(path, model, PPOTrainer(model, config.ppo), config, 7, metadata)
    return path, config


def factory_capture(created, **options):
    def make_env(model_config, environment_config, device):
        env = PhysicalFixture(model_config, environment_config, device, **options)
        created.append(env)
        return env
    return make_env


def test_enabled_evaluation_keeps_group_control_and_trace_provenance(tmp_path, checkpoint):
    path, config = checkpoint
    created = []
    trace_path = tmp_path / "physical_trace.npz"
    report = evaluate_frame_policy(path, factory_capture(created), config.environment,
        steps=12, seed=801, settle_steps=1, min_steady_samples=1,
        control_metrics=True, trace_output=trace_path, trace_replicas=2)
    assert created[0].enabled_at_reset is True and created[0].closed
    assert report["checkpoint_update"] == 7
    assert set(report["groups"]) == {"a", "b"}
    assert report["control"]
    assert report["control"]["full_interval"]["samples"] == 72
    for group in report["groups"].values():
        assert group["num_envs"] == 3 and group["transitions"] == 36
        assert group["control"]["full_interval"]["samples"] == 36
    good, wrong = (report["groups"][name]["control"] for name in ("a", "b"))
    assert good["full_interval"]["axes"]["height"]["rmse"] == 0.
    height = wrong["full_interval"]["axes"]["height"]
    assert height["bias"] == pytest.approx(-.2, abs=1e-7)
    assert height["rmse"] == pytest.approx(.2, abs=1e-7)
    assert height["within_group_std"] == 0.
    assert height["in_band_fraction"] == 0.
    # Timeouts are healthy in the toy fixture, but wrong-height holding is not tracking.
    assert wrong["episodes"]["success_flags"] > 0
    assert wrong["full_interval"]["all_axes_in_band_fraction"] == 0.
    receipt = report["trace"]
    assert receipt["path"] == str(trace_path)
    assert receipt["sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    with np.load(trace_path, allow_pickle=False) as trace:
        rows = trace["row_indices"]
        assert rows.tolist() == [0, 1, 3, 4]
        metadata = json.loads(str(trace["metadata_json"].item()))
        assert metadata["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert metadata["control_sha256"] == digest(config.control)
        assert metadata["seed"] == 801
        assert metadata["row_indices"] == rows.tolist()
        assert metadata["group_labels"] == ["a", "a", "b", "b"]
        assert metadata["policy_dt_s"] == .01 and metadata["sampling_hz"] == 100.
        assert trace["actual"].shape == (12, 4, 3)
        assert trace["motor_effort"].shape == (12, 4, 6)
        assert trace["time_s"].dtype == np.float64
        assert trace["done"].dtype == trace["failure"].dtype == np.bool_
        np.testing.assert_allclose(trace["actual"][:, :2, 2], .3)
        np.testing.assert_allclose(trace["actual"][:, 2:, 2], .1)
        # Terminal endpoints belong to the old episode. The next point starts a new one.
        assert trace["done"][2, 0] and trace["episode_id"][2, 0] == 0
        assert trace["episode_id"][3, 0] == 1
        assert trace["time_s"][2, 0] == .03 and trace["time_s"][3, 0] == .01
        for name in trace.files:
            assert trace[name].dtype.kind != "O"


def test_default_evaluation_preserves_existing_contract_without_control_packet(checkpoint):
    path, config = checkpoint
    created = []
    report = evaluate_frame_policy(path, factory_capture(created), config.environment,
        steps=12, seed=801, settle_steps=1, min_steady_samples=1)
    assert created[0].enabled_at_reset is False and created[0].closed
    assert "control" not in report and "trace" not in report
    assert report["completed_episodes"] > 0


@pytest.mark.parametrize("corrupt", ("missing", "nonfinite", "wrong_dtype"))
def test_late_packet_failure_closes_environment_without_publishing_trace(tmp_path, checkpoint, corrupt):
    path, config = checkpoint
    created = []
    trace_path = tmp_path / "failed_trace.npz"
    with pytest.raises((ValueError, FloatingPointError), match="control|packet|time_s|motor_effort|finite"):
        evaluate_frame_policy(path, factory_capture(created, fail_at=3, corrupt=corrupt), config.environment,
            steps=12, seed=801, settle_steps=1, min_steady_samples=1,
            control_metrics=True, trace_output=trace_path, trace_replicas=2)
    assert created[0].closed
    assert not trace_path.exists()


def test_existing_trace_is_preserved_when_evaluation_refuses_overwrite(tmp_path, checkpoint):
    path, config = checkpoint
    trace_path = tmp_path / "existing.npz"
    trace_path.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError):
        evaluate_frame_policy(path, factory_capture([]), config.environment,
            steps=12, seed=801, settle_steps=1, min_steady_samples=1,
            control_metrics=True, trace_output=trace_path, trace_replicas=2)
    assert trace_path.read_bytes() == b"existing evidence"


def test_suite_publishes_per_case_control_and_checkpoint_provenance(tmp_path, checkpoint, monkeypatch):
    from transformer_rl import chassis_adapter

    path, config = checkpoint
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    configs, outputs = [], []
    for name in ("a", "b"):
        contract = {"evaluation_exact_cases": True, "target_num_envs": 3,
                    "physics_dt": .001, "scene_groups": [{"name": name, "fraction": 1.}],
                    "evaluation": {"cases": [{"name": name, "task": "survive", "terrain": "flat"}]}}
        contract_path = snapshot / f"{name}.json"
        contract_path.write_text(json.dumps(contract))
        environment = {"snapshot": str(snapshot), "snapshot_sha256": "a" * 64,
                       "contract": contract_path.name,
                       "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                       "num_envs": 3}
        config_path = tmp_path / f"{name}_config.json"
        config_path.write_text(json.dumps(replace(config, environment=environment).to_dict()))
        configs.append(config_path)
        outputs.append(tmp_path / f"{name}_report.json")
    created = []

    def make_env(model_config, environment_config, device):
        env = PhysicalFixture(model_config, {**environment_config, "control": config.control}, device)
        created.append(env)
        return env

    monkeypatch.setattr(chassis_adapter, "make_env", make_env)
    control_path, trace_path = tmp_path / "control.json", tmp_path / "trace.npz"
    chassis_adapter.evaluate_suite(path, configs, outputs, steps=12, seed=801, device="cpu",
        settle_steps=1, min_steady_samples=1, control_output=control_path,
        trace_output=trace_path, trace_replicas=2)
    assert len(created) == 1 and created[0].closed
    control = json.loads(control_path.read_text())
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert control["checkpoint_sha256"] == expected_sha and control["seed"] == 801
    assert set(control["groups"]) == {"a", "b"}
    assert trace_path.exists()
    for name, output in zip(("a", "b"), outputs):
        report = json.loads(output.read_text())
        assert report["checkpoint_sha256"] == expected_sha and report["seed"] == 801
        assert report["num_envs"] == 3
        assert report["control"]["full_interval"]["samples"] == 36
        assert report["environment"]["contract"] == f"{name}.json"

"""PRE-reset source-adapter semantics on tensors, without importing the SDK."""
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.chassis_adapter import ChassisFrameAdapter
from transformer_rl.frame_config import FrameModelConfig


class TensorChassis:
    def __init__(self):
        self.num_envs, self.device, self.tick = 3, "cpu", 0
        self.scene_groups = ["standing", "stairs", "jump"]
        self.cfg = {"evaluation_exact_cases": True,
                    "evaluation": {"cases": [{"name": name, "task": task} for name, task in
                        zip(self.scene_groups, ("survive", "traverse", "jump"))]}}
        self.mode = torch.tensor([0, 1, 4])
        self.generator = torch.Generator()
        self.pose = torch.zeros(3, 7)
        self.robot = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=self.pose)))
        self.value = torch.zeros(3)
        self.failed = torch.zeros(3, dtype=torch.bool)
        self.success = torch.zeros(3, dtype=torch.bool)
        self.episode_limits = torch.ones(3, dtype=torch.int64)
        self.boundary = torch.zeros(3, dtype=torch.bool)
        self.blocked = torch.zeros(3, dtype=torch.bool)

    def get_observations(self):
        return {"policy": self.value[:, None].expand(-1, 35).clone(),
                "critic": self.value[:, None].expand(-1, 81).clone()}

    def reset(self, rows):
        self.value[rows] = 0.
        self.pose[rows] = 0.

    def step(self, issued):
        self.tick += 1
        self.value.copy_(torch.tensor([1., 2., 3.]))
        self.pose[:, :2] = .5
        diagnostics = {"velocity": torch.zeros(3, 3), "omega": torch.zeros(3, 3),
            "gravity": torch.tensor([[0., 0., -1.]]).expand(3, -1).clone(),
            "commands": torch.tensor([[0., 0., .3]]).expand(3, -1).clone(), "height": torch.full((3,), .3),
            "episode_ticks": torch.ones(3, dtype=torch.long), "terminated": self.failed.clone(),
            "reasons": {"boundary": self.boundary, "blocked": self.blocked},
            "success": self.success.clone(), "leg_target_position": torch.zeros(3, 4),
            "wheel_target_velocity": torch.zeros(3, 2), "motor_effort": torch.zeros(3, 6)}
        return self.get_observations(), torch.ones(3), torch.ones(3, dtype=torch.bool), {
            "diagnostics": diagnostics, "time_outs": torch.ones(3, dtype=torch.bool)}


@pytest.mark.parametrize("real_events", (False, True))
def test_terminal_critic_and_goal_success_survive_manual_reset(real_events):
    env = TensorChassis()
    if real_events:
        env.failed[1], env.success[2] = True, True
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {})
    adapter.reset(seed=71)
    result = adapter.step(torch.zeros(3, 6))
    torch.testing.assert_close(result.final_critic[:, 0], torch.tensor([1., 2., 3.]))
    assert not result.observation.critic.any() and not result.observation.frame.any()
    assert result.info["episode_success"].tolist() == [True, False, real_events]
    assert result.terminated.tolist() == [False, real_events, real_events]
    assert result.truncated.tolist() == [True, not real_events, not real_events]
    assert (result.info["evaluation_metrics"]["drift_m"] > .7).all()
    assert not result.info["evaluation_metrics"]["issued_action_rate_rms"].any()
    assert adapter._fresh.all()
    env.value.fill_(100.)
    torch.testing.assert_close(result.final_critic[:, 0], torch.tensor([1., 2., 3.]))


@pytest.mark.parametrize("device_alias", ("cpu", "cpu:0"))
def test_episode_evidence_is_owned_before_reset_and_keeps_success_distinct(device_alias):
    class ResettingChassis(TensorChassis):
        def reset(self, rows):
            super().reset(rows)
            self.episode_limits[rows] = 99
            self.boundary[rows] = False
            self.blocked[rows] = False

    env = ResettingChassis()
    env.device = device_alias
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {})
    adapter.enable_episode_outcomes = True
    adapter.reset()
    env.episode_limits.copy_(torch.tensor([1, 8, 12]))
    env.boundary[0] = True
    env.blocked[1] = True
    env.success[2] = True
    result = adapter.step(torch.zeros(3, 6))
    packet = result.info["evaluation_episode"]
    assert packet["episode_horizon_ticks"].tolist() == [1, 8, 12]
    assert packet["boundary"].tolist() == [True, False, False]
    assert packet["blocked"].tolist() == [False, True, False]
    assert packet["survival_applicable"].tolist() == [True, False, False]
    assert packet["task_success"].tolist() == [False, False, True]
    assert not packet["environment_failure"].any()
    assert result.terminated[2]  # A learning termination is a task success here.
    assert env.episode_limits.tolist() == [99, 99, 99]
    assert not env.boundary.any() and not env.blocked.any()
    assert result.info["evaluation_state"]["world_position"][0, 0] == .5
    assert not env.pose.any()


@pytest.mark.parametrize("missing", ("reasons", "boundary", "horizon", "active_blocked"))
def test_episode_evidence_rejects_missing_timeout_causes(missing):
    env = TensorChassis()
    original = env.step
    def missing_evidence(issued):
        raw, reward, done, extras = original(issued)
        if missing == "reasons":
            del extras["diagnostics"]["reasons"]
        elif missing == "boundary":
            del extras["diagnostics"]["reasons"]["boundary"]
        elif missing == "horizon":
            env.episode_limits = None
        else:
            env.step_assist = object()
            del extras["diagnostics"]["reasons"]["blocked"]
        return raw, reward, done, extras
    env.step = missing_evidence
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {})
    adapter.enable_episode_outcomes = True
    with pytest.raises(ValueError, match="episode|blocked"):
        adapter.step(torch.zeros(3, 6))


def test_episode_evidence_without_explicit_task_contract_is_unavailable():
    env = TensorChassis()
    env.cfg["evaluation_exact_cases"] = False
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {})
    adapter.enable_episode_outcomes = True
    result = adapter.step(torch.zeros(3, 6))
    assert "evaluation_episode" not in result.info


def test_default_study_contains_only_requested_architecture_families_and_consistent_presets():
    import json
    from pathlib import Path
    from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig
    from transformer_rl.chassis_adapter import network_variants

    variants = network_variants()
    assert len(variants) == 10 and len({variant["name"] for variant in variants}) == 10
    assert {variant["policy"]["architecture"] for variant in variants} == {"mlp", "history_mlp", "transformer"}
    root = Path(__file__).resolve().parents[1] / "configs" / "frame_policies"
    for variant in variants:
        config = FramePolicyConfig.from_dict(variant["policy"])
        preset = FramePolicyConfig.from_dict(json.loads((root / f'{variant["name"]}.json').read_text()))
        assert config == preset
        if config.architecture != "mlp":
            assert config.history_length == 31
        policy = FramePolicy(config)
        assert torch.isfinite(policy(torch.zeros(1, config.history_length, 35))).all()


def test_transfer_reset_profiles_apply_before_observation_and_after_episode_end():
    env = TensorChassis()
    calls = []
    def transform(actual, rows):
        calls.append(rows.tolist())
        actual.value[rows] = 7.
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {}, reset_transform=transform)
    torch.testing.assert_close(adapter.reset().frame[:, 0], torch.full((3,), 7.))
    result = adapter.step(torch.zeros(3, 6))
    torch.testing.assert_close(result.final_critic[:, 0], torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(result.observation.frame[:, 0], torch.full((3,), 7.))
    assert calls == [[0, 1, 2], [0, 1, 2]]


def test_transfer_physical_clock_requires_separate_explicit_contract():
    from transformer_rl.chassis_adapter import _validate_contract
    config = {"contract_id": "packed-transfer-study", "physics_dt": .005, "policy_dt": .01,
              "history_length": 1, "actor_dim": 35, "actor_frame_dim": 35, "critic_dim": 81,
              "action_dim": 6, "v5_control": {"leg_kp": 160., "leg_kd": 2.5},
              "auto_reset": False, "record_diagnostics": True, "diagnostic_trace": True,
              "actuator_response": {"enabled": True}, "signal_delay": {"enabled": True},
              "command_transport": {"enabled": True, "delivery_model": "phase_aware_hold_v1"}}
    control = {"actuators": {"wheel": {"kd": .6}}}
    state = _validate_contract(config, control)
    assert state["physics_dt"] == state["pc_control_dt"] == .005
    assert state["decimation"] == 2 and state["modules"]["actuator_response"]
    assert state["modules"]["signal_delay"] and state["command_transport_model"] == "phase_aware_hold_v1"
    config["contract_id"] = "packed-policy-study"
    with pytest.raises(ValueError, match="timing"):
        _validate_contract(config, control)


def test_environment_diagnostics_report_actual_mask_and_reject_nonfinite_values():
    env = TensorChassis()
    env.perturbations = SimpleNamespace(enabled=torch.tensor([True, False, False]))
    adapter = ChassisFrameAdapter(env, FrameModelConfig(), {})
    adapter._last_environment_metrics = {"/delay/strength": torch.tensor(.75)}
    diagnostics = adapter.training_diagnostics()
    assert diagnostics["delay/strength"] == .75
    assert diagnostics["transfer/noise_enabled_fraction"] == pytest.approx(1/3)
    adapter._last_environment_metrics["/bad"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.training_diagnostics()

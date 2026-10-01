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

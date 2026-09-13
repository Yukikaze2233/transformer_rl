"""CPU-only semantic checks; these do not certify SDK integration."""
import torch

from examples.isaaclab_task import encode_observation, physical_metrics
from transformer_rl.config import ModelConfig


def test_source_layout_uses_latest_noise_and_issued_echo():
    config = ModelConfig()
    scalar = torch.arange(25, dtype=torch.float32)[None].repeat(2, 1)
    raw = {"policy": torch.cat([scalar + 100] * 4 + [scalar], dim=-1),
           "critic": torch.arange(29, dtype=torch.float32)[None].repeat(2, 1)}
    issued = torch.full((2, 6), -3.0)
    timestamp = torch.tensor([1e6, 1e6 + .01], dtype=torch.float64)
    obs = encode_observation(config, raw, issued, timestamp, .01)
    torch.testing.assert_close(obs.frame[:, :16], torch.cat((scalar[:, :6], scalar[:, 9:19]), -1))
    torch.testing.assert_close(obs.command, scalar[:, 6:9])
    torch.testing.assert_close(obs.frame[:, 19:25], issued)
    torch.testing.assert_close(obs.frame[:, 25:29], torch.tensor([[0., 0., 1., 1.]]).repeat(2, 1))
    assert obs.timestamp.dtype == torch.float64
    raw["critic"].zero_()
    timestamp.zero_()
    assert obs.critic[0, 25] == 25 and obs.timestamp[0] == 1e6


def test_physical_metrics_are_si_owned_pre_reset_values():
    velocity = torch.tensor([[3., 4., 2.]])
    angular = torch.tensor([[0., 0., -2.]])
    height = torch.tensor([.32])
    command = torch.tensor([[1., 1., .30]])
    force = torch.tensor([12.])
    metrics = physical_metrics(velocity, angular, height, command, force)
    velocity.zero_()
    force.zero_()
    expected = {"vx_abs_error": 2., "wz_abs_error": 3., "height_abs_error": .02,
                "planar_speed": 5., "non_wheel_net_force": 12.}
    for key, value in expected.items():
        torch.testing.assert_close(metrics[key], torch.tensor([value]))

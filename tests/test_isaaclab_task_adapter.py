"""CPU-only semantic checks; these do not certify SDK integration."""
from types import SimpleNamespace

import pytest
import torch

from examples.isaaclab_task import (
    CONTROL_SIGNAL_UNITS, IsaacLabTaskAdapter, control_signals,
    encode_observation, physical_metrics,
)
from transformer_rl.config import ModelConfig


def test_source_layout_uses_latest_noise_and_issued_echo():
    config = ModelConfig()
    scalar = torch.arange(25, dtype=torch.float32)[None].repeat(2, 1)
    raw = {"policy": torch.cat([scalar + 100] * 4 + [scalar], dim=-1),
           "critic": torch.arange(29, dtype=torch.float32)[None].repeat(2, 1)}
    issued = torch.full((2, 6), -3.0)
    timestamp = torch.tensor([1e6, 1e6 + .01], dtype=torch.float64)
    obs = encode_observation(config, raw, issued, timestamp, .01)
    assert obs.frame.shape == (2, 30)
    torch.testing.assert_close(obs.frame[:, :16], torch.cat((scalar[:, :6], scalar[:, 9:19]), -1))
    torch.testing.assert_close(obs.command, scalar[:, 6:9])
    torch.testing.assert_close(obs.frame[:, 19:25], issued)
    torch.testing.assert_close(obs.frame[:, 25:29], torch.tensor([[0., 0., 1., 1.]]).repeat(2, 1))
    torch.testing.assert_close(obs.frame[:, 29], torch.full((2,), .01))
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


def test_control_signals_units_order_sign_and_owned_storage():
    linear = torch.tensor([[.5, 20., 30.], [-.5, 40., 50.]], requires_grad=True)
    angular = torch.tensor([[.1, -.2, -.5], [-.3, .4, .5]], requires_grad=True)
    height = torch.tensor([.25, .5], requires_grad=True)
    command = torch.tensor([[1., .25, .5], [-1., -.25, .25]], requires_grad=True)
    legs = torch.tensor([[.1, -.2, .3, -.4], [.5, -.6, .7, -.8]], requires_grad=True)
    wheels = torch.tensor([[11., -12.], [-13., 14.]], requires_grad=True)
    effort = torch.tensor([[21., -22., 23., -24., 25., -26.],
                           [-31., 32., -33., 34., -35., 36.]], requires_grad=True)
    inputs = (linear, angular, height, command, legs, wheels, effort)
    signals = control_signals(*inputs)
    expected = {
        "height_error": torch.tensor([-.25, .25]),
        "vx_error": torch.tensor([-.5, .5]),
        "wz_error": torch.tensor([-.75, .75]),
        "angular_velocity_x": torch.tensor([.1, -.3]),
        "angular_velocity_y": torch.tensor([-.2, .4]),
        **{f"leg_target_{i}": legs[:, i].detach().clone() for i in range(4)},
        **{f"wheel_target_{i}": wheels[:, i].detach().clone() for i in range(2)},
        **{f"effort_{i}": effort[:, i].detach().clone() for i in range(6)},
    }
    assert list(signals) == list(expected) == list(CONTROL_SIGNAL_UNITS)
    assert list(CONTROL_SIGNAL_UNITS.values()) == (
        ["m", "m/s", "rad/s", "rad/s", "rad/s"]
        + ["rad"] * 4 + ["rad/s"] * 2 + ["Nm"] * 6
    )
    with torch.no_grad():
        for source in inputs:
            source.fill_(999.)
    for name, value in signals.items():
        assert value.shape == (2,) and value.device.type == "cpu"
        assert not value.requires_grad and value.grad_fn is None and value._base is None
        assert all(value.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                   for source in inputs)
        torch.testing.assert_close(value, expected[name])


def test_control_signals_missing_fixture_arrays_are_not_fabricated():
    signals = control_signals(torch.zeros(2, 3), torch.zeros(2, 3),
                              torch.zeros(2), torch.zeros(2, 3))
    assert set(signals) == {"height_error", "vx_error", "wz_error",
                            "angular_velocity_x", "angular_velocity_y"}


def test_physical_metrics_true_normalized_tilt_and_owned_state():
    gravity = torch.tensor([[0., 0., -2.], [0., 3., -3.], [2., 0., 0.],
                            [0., 0., 4.], [0., 0., 0.]], requires_grad=True)
    height = torch.full((5,), .32, requires_grad=True)
    metrics = physical_metrics(torch.zeros(5, 3), torch.zeros(5, 3), height,
                               torch.zeros(5, 3), torch.zeros(5), gravity)
    assert set(metrics) == {"vx_abs_error", "wz_abs_error", "height_abs_error",
                            "planar_speed", "non_wheel_net_force", "base_height", "tilt_angle"}
    with torch.no_grad():
        gravity.zero_()
        height.zero_()
    torch.testing.assert_close(metrics["base_height"], torch.full((5,), .32))
    torch.testing.assert_close(metrics["tilt_angle"],
                               torch.tensor([0., torch.pi / 4, torch.pi / 2, torch.pi, torch.pi / 2]))
    assert all(not value.requires_grad and value._base is None for value in metrics.values())


class SyntheticSource:
    """Tensor-only reward -> reset -> observation lifecycle; no simulator imports."""

    num_envs = 2
    device = "cpu"
    step_dt = .01

    def __init__(self):
        self.linear = torch.zeros(2, 3)
        self.angular = torch.zeros(2, 3)
        self.gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
        self.height = torch.zeros(2)
        self.commands = torch.zeros(2, 3)
        self.actions = torch.zeros(2, 6)
        self.leg_targets = torch.zeros(2, 4)
        self.wheel_targets = torch.zeros(2, 2)
        self.torques = torch.zeros(2, 6)
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_com_lin_vel_b=SimpleNamespace(torch=self.linear),
            root_com_ang_vel_b=SimpleNamespace(torch=self.angular),
            projected_gravity_b=SimpleNamespace(torch=self.gravity),
        ))
        self.contract = {}
        self._non_wheel_body_ids = [0]
        self.reset_time_outs = torch.tensor([True, False])

    def _joint_state(self):
        return torch.zeros(2, 6), torch.zeros(2, 6)

    def _base_height(self):
        return self.height

    def _contact_magnitudes(self):
        return torch.full((2, 1), 12.)

    def _get_rewards(self):
        # Distinct values written here prove capture happens AFTER reward.
        self.linear[:, 0] = torch.tensor([3., -3.])
        self.angular[:] = torch.tensor([.1, -.2, 2.])
        self.height[:] = torch.tensor([.25, .5])
        self.commands[:] = torch.tensor([1., .5, .3])
        self.gravity[:] = torch.tensor([0., 2., -2.])
        self.leg_targets[:] = torch.tensor([.1, -.2, .3, -.4])
        self.wheel_targets[:] = torch.tensor([10., -11.])
        self.torques[:] = torch.arange(6) + 20.
        return torch.tensor([7., 8.])

    def _reset_idx(self, env_ids):
        for value in (self.linear, self.angular, self.height, self.commands,
                      self.leg_targets, self.wheel_targets, self.torques, self.gravity):
            value[env_ids] = 0.

    def _observations(self):
        # Deliberately unrelated noisy/reset actor input, never diagnostic truth.
        return {"policy": torch.full((2, 125), -9.), "critic": torch.zeros(2, 29)}

    def reset(self, seed=None):
        self._reset_idx(None)
        return self._observations(), {}

    def step(self, issued):
        self.actions.copy_(issued)
        reward = self._get_rewards()
        self._reset_idx(torch.tensor([0]))
        self.commands.fill_(99.)  # Next-observation resampling, including live row.
        return (self._observations(), reward, torch.zeros(2, dtype=torch.bool),
                self.reset_time_outs.clone(), {"source_tag": "preserved"})


@pytest.mark.parametrize("mode", ["train", "evaluation"])
def test_adapter_captures_before_reset_with_monotonic_owned_policy_time(mode):
    source = SyntheticSource()
    adapter = IsaacLabTaskAdapter(
        source, None, ModelConfig(), {"mode": mode},
        lambda *args: torch.zeros(2, 25),
        lambda clean, linear, height: torch.cat((clean, linear, height[:, None]), -1),
    )
    initial = adapter.reset()
    torch.testing.assert_close(initial.timestamp, torch.full((2,), .01, dtype=torch.float64))
    first = adapter.step(torch.ones(2, 6))
    signals = first.info["evaluation_signals"]
    assert set(signals) == set(CONTROL_SIGNAL_UNITS)
    torch.testing.assert_close(signals["height_error"], torch.tensor([-.05, .2]))
    torch.testing.assert_close(signals["vx_error"], torch.tensor([2., -4.]))
    torch.testing.assert_close(signals["wz_error"], torch.full((2,), 1.5))
    torch.testing.assert_close(signals["angular_velocity_x"], torch.full((2,), .1))
    torch.testing.assert_close(signals["angular_velocity_y"], torch.full((2,), -.2))
    for prefix, values in (("leg_target", [.1, -.2, .3, -.4]),
                           ("wheel_target", [10., -11.]),
                           ("effort", [20., 21., 22., 23., 24., 25.])):
        for i, value in enumerate(values):
            torch.testing.assert_close(signals[f"{prefix}_{i}"], torch.full((2,), value))
    torch.testing.assert_close(first.info["evaluation_metrics"]["base_height"], torch.tensor([.25, .5]))
    torch.testing.assert_close(first.info["evaluation_metrics"]["tilt_angle"], torch.full((2,), torch.pi / 4))
    torch.testing.assert_close(first.reward, torch.tensor([7., 8.]))
    assert first.final_critic_valid.tolist() == [True, False]
    assert first.info["source_tag"] == "preserved"
    assert first.observation.frame.shape == (2, 30)
    torch.testing.assert_close(first.observation.frame[:, :19], torch.full((2, 19), -9.))
    torch.testing.assert_close(first.observation.frame[:, 29], torch.full((2,), .01))
    saved = {key: value.clone() for key, value in signals.items()}
    second = adapter.step(torch.zeros(2, 6))
    adapter.reset()
    third = adapter.step(torch.zeros(2, 6))
    for result, time in ((first, .02), (second, .03), (third, .05)):
        tag = result.info["evaluation_signal_time"]
        assert tag.dtype == torch.float64 and tag.shape == (2,)
        torch.testing.assert_close(tag, torch.full((2,), time, dtype=torch.float64))
        torch.testing.assert_close(tag, result.observation.timestamp)
    for key, value in saved.items():
        torch.testing.assert_close(first.info["evaluation_signals"][key], value)

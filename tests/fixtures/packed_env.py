"""Small synthetic tensor contract; no robot, physics engine or hardware."""
import torch

from transformer_rl.frame_config import digest
from transformer_rl.types import StepResult, VectorObservation


class PackedFixture:
    def __init__(self, model_config, environment_config, device):
        self.config, self.options, self.device = model_config, environment_config, torch.device(device)
        self.num_envs = environment_config.get("num_envs", 3)
        self.metadata = {"identity": "synthetic_tensor_fixture", "control_sha256": digest(environment_config["control"])}
        self.dt = environment_config["control"]["policy_dt_s"]
        self.tick = 0
        self.state = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous = torch.zeros_like(self.state)
        self.age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.closed = False

    def _observation(self):
        frame = torch.cat((torch.ones(self.num_envs, 1, device=self.device), self.state, self.previous), -1)
        critic = torch.cat((self.state, torch.ones(self.num_envs, 1, device=self.device)), -1)
        return VectorObservation(frame, torch.full((self.num_envs,), self.tick * self.dt, dtype=torch.float64, device=self.device),
                                 frame[:, self.config.command_indices], critic)

    def reset(self, seed=None):
        self.state.zero_()
        self.previous.zero_()
        self.age.zero_()
        return self._observation()

    def step(self, issued_action):
        bounds = torch.tensor(self.options["control"]["action_bounds"], device=self.device)
        assert (issued_action.abs() <= bounds).all()
        self.state += issued_action * self.dt
        self.previous.copy_(issued_action)
        self.age += 1
        self.tick += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.age >= 4
        terminated[0] = self.age[0] >= 3
        truncated &= ~terminated
        done = terminated | truncated
        final = self._observation().critic.clone()
        values = self.state[:, 0].clone()
        info = {"episode_success": done & ~torch.full_like(done, self.options.get("reject", False)),
                "evaluation_metrics": {"tracking_error": values.abs()},
                "evaluation_signals": {"tracking_error": values},
                "evaluation_signal_time": self.age.double() * self.dt}
        reward = 1. - self.state.square().sum(-1)
        self.state[done] = 0.
        self.previous[done] = 0.
        self.age[done] = 0
        return StepResult(self._observation(), reward, terminated, truncated, final, done, info)

    def close(self):
        self.closed = True


def make_env(model_config, environment_config, device):
    return PackedFixture(model_config, environment_config, device)

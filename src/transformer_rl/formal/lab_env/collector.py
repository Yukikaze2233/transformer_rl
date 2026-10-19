"""M1 专用 rollout collector。

复用现有 GAE、窗口、mask、auxiliary label 和 PPO trainer；只省略旧任务
collector 的逐 step 所有权/协议校验，因为这些检查属于正式 recovery/jump
接入层，M1 平地不需要每步重复。所有策略前向在 no_grad 下完成。
"""
from __future__ import annotations

from dataclasses import fields
import time

import torch
from torch.distributions import Normal

from ..contracts import StructuredCollectionConfig
from ..learning.contracts import (AuxiliaryLabels, BoundaryMasks, ObservationWindows,
                                  masked_gae)
from ..learning.optimization import StructuredBatch
from .adapter import LabLabeledEnv


class LabCollector:
    """M1 collector；输入 LabLabeledEnv，输出 StructuredBatch。"""

    def __init__(self, env: LabLabeledEnv, model,
                 config: StructuredCollectionConfig = StructuredCollectionConfig()):
        if not isinstance(env, LabLabeledEnv):
            raise TypeError("env 必须是 LabLabeledEnv")
        if not isinstance(config, StructuredCollectionConfig):
            raise TypeError("config 必须是 StructuredCollectionConfig")
        self.env, self.model, self.config = env, model, config
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.bounds = torch.tensor(config.action_bounds, dtype=torch.float32, device=self.device)
        self.history_length = model.actor.policy.history_length
        reference = next(model.parameters())
        if reference.device != self.device or reference.dtype != torch.float32:
            raise ValueError("collector 需要与环境同 device 的 float32 模型")
        self._current = None
        self._history = None
        self.attempted_vector_steps = 0
        self.returned_vector_steps = 0
        self.last_metrics = {}
        self.last_timing = {}

    @torch.no_grad()
    def reset(self, seed=None):
        if seed is not None:
            raise ValueError("M1 collector 不在 reset 处重设全局种子")
        self._current = self.env.reset()
        self._history = self._current.frame[:, None].expand(-1, self.history_length, -1).clone()
        return self._current

    def _store(self, records, name, value, step):
        if name not in records:
            records[name] = value.new_empty((self._history_length_steps, *value.shape))
        records[name][step].copy_(value)

    @torch.no_grad()
    def collect(self, steps=48):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps 必须是正整数")
        if self._current is None:
            raise RuntimeError("collect 前必须 reset")
        self.model.eval()
        self._history_length_steps = steps
        started = time.perf_counter()
        prefix = self._history[:, :-1].transpose(0, 1).clone()
        names = ("frame", "critic", "raw", "issued", "mean", "log_std", "logp", "value",
                 "reward", "next_value", "reset", "velocity", "contact", "velocity_delta")
        records = {}
        mask_records = {}
        reset_before = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        outcomes = torch.zeros(2, dtype=torch.long, device=self.device)
        environment_timing: dict[str, float] = {}

        for step in range(steps):
            current = self._current
            mean, _, _ = self.model.actor.policy.outputs(self._history)
            log_std = self.model.actor.log_std.clamp(-4.0, 1.0).expand_as(mean)
            raw_action = Normal(mean, log_std.exp(), validate_args=False).sample()
            evaluation = self.model.actor._evaluation(mean, self.model.actor.log_std, raw_action)
            issued = raw_action.clamp(-self.bounds, self.bounds)
            old_value = self.model.critic(current.critic)
            self.attempted_vector_steps += 1
            returned = self.env.step(issued, None)
            self.returned_vector_steps += 1
            for name, elapsed in self.env.last_timing.items():
                environment_timing[name] = environment_timing.get(name, 0.0) + elapsed

            next_observation = returned.observation
            final = returned.final_labels
            same_final_epoch = ((final.episode_id == current.episode_id)
                                & (final.control_epoch == current.control_epoch))
            rollout_end = torch.full_like(returned.terminated, step == steps - 1)
            masks = BoundaryMasks.from_events(
                fully_policy_owned=returned.fully_policy_owned,
                terminated=returned.terminated,
                truncated=returned.truncated,
                final_value_valid=returned.final_critic_valid,
                handoff_after=returned.handoff_after,
                rollout_end=rollout_end,
                state_label_valid=current.labels.valid,
                next_label_valid=current.labels.valid & final.valid,
                next_same_control_epoch=same_final_epoch,
            )
            next_value = torch.zeros_like(old_value)
            ordinary = (masks.bootstrap & ~returned.truncated).nonzero().flatten()
            if ordinary.numel():
                next_value[ordinary] = self.model.critic(next_observation.critic[ordinary])
            timeout = (masks.bootstrap & returned.truncated).nonzero().flatten()
            if timeout.numel():
                next_value[timeout] = self.model.critic(returned.final_critic[timeout])

            values = {
                "frame": current.frame,
                "critic": current.critic,
                "raw": raw_action,
                "issued": issued,
                "mean": mean,
                "log_std": log_std,
                "logp": evaluation.log_prob,
                "value": old_value,
                "reward": returned.reward,
                "next_value": next_value,
                "reset": reset_before,
                "velocity": current.labels.velocity_body_m_s,
                "contact": current.labels.wheel_contact,
                "velocity_delta": final.joint_velocity_rad_s - current.labels.joint_velocity_rad_s,
            }
            for name in names:
                self._store(records, name, values[name], step)
            for field in fields(BoundaryMasks):
                value = getattr(masks, field.name)
                if field.name not in mask_records:
                    mask_records[field.name] = value.new_empty((steps, *value.shape))
                mask_records[field.name][step].copy_(value)
            outcomes += torch.stack((returned.terminated.sum(),
                                     (returned.truncated & ~returned.terminated).sum()))

            reset_before = ((next_observation.episode_id != current.episode_id)
                            | (next_observation.control_epoch != current.control_epoch))
            self._history[:, :-1] = self._history[:, 1:].clone()
            self._history[:, -1] = next_observation.frame
            self._history[reset_before] = next_observation.frame[reset_before, None]
            self._current = next_observation

        masks = BoundaryMasks(*(mask_records[field.name] for field in fields(BoundaryMasks)))
        advantage, returns = masked_gae(records["reward"], records["value"], records["next_value"],
                                        masks, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        windows = ObservationWindows(prefix, records["frame"], records["reset"])
        flat_masks = BoundaryMasks(*(getattr(masks, field.name).flatten() for field in fields(masks)))
        labels = AuxiliaryLabels(records["velocity"].flatten(0, 1), records["contact"].flatten(0, 1),
                                 records["velocity_delta"].flatten(0, 1))
        batch = StructuredBatch(windows, records["critic"].flatten(0, 1),
                                records["raw"].flatten(0, 1), records["issued"].flatten(0, 1),
                                records["mean"].flatten(0, 1), records["log_std"].flatten(0, 1),
                                records["logp"].flatten(), records["value"].flatten(),
                                advantage.flatten(), returns.flatten(), flat_masks, labels)
        batch.validate(self.model)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.last_timing = environment_timing
        self.last_metrics = {
            "vector_steps": steps,
            "sampled_transitions": steps * self.num_envs,
            "policy_owned_transitions": int(flat_masks.actor.sum()),
            "raw_history_elements": windows.raw_storage_elements,
            "collection_elapsed_s": elapsed,
            "attempted_vector_steps_lifetime": self.attempted_vector_steps,
            "returned_vector_steps_lifetime": self.returned_vector_steps,
            "terminated_rows": int(outcomes[0]),
            "truncated_rows": int(outcomes[1]),
            "policy_owned_fraction": float(flat_masks.actor.float().mean()),
            "reward_mean": float(records["reward"].mean()),
            "return_mean": float(returns.mean()),
            "advantage_std": float(advantage.std()),
        }
        if environment_timing:
            self.last_metrics["environment_host_wall_s"] = environment_timing
        return batch

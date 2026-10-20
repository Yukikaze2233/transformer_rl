"""把 M1 DirectRLEnv 适配到现有 StructuredCollector 的 LabeledStep 契约。

只做边界转换：观测/标签/回合时钟。策略、PPO、GAE 和窗口逻辑仍复用
formal.learning；不改网络或奖励公式。M1 尚未接入 recovery/jump/request。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..contracts import (LabeledObservation, LabeledStep, PhysicalLabels,
                         StructuredCollectionConfig)

if TYPE_CHECKING:
    from .env import WheeledLabEnv


class LabLabeledEnv:
    """M1 平地环境的 collector 适配器；接触标签是 2N 阈值二值量。"""

    def __init__(self, env: "WheeledLabEnv", *, contact_threshold_n: float = 2.0):
        if not hasattr(env, "num_envs") or not hasattr(env, "step_dt"):
            raise TypeError("env 必须是 WheeledLabEnv 兼容对象")
        if type(contact_threshold_n) not in (int, float) or contact_threshold_n <= 0:
            raise ValueError("contact_threshold_n 必须是正数")
        self.env = env
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.contact_threshold_n = float(contact_threshold_n)
        self._episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._control_epoch = torch.zeros_like(self._episode_id)
        self.last_timing: dict[str, float] = {}
        self.last_counts: dict[str, int] = {}
        self.terrain_stream = None

    def validate_collection_config(self, config):
        if not isinstance(config, StructuredCollectionConfig):
            raise TypeError("config 必须是 StructuredCollectionConfig")
        if abs(config.policy_dt_s - self.env.step_dt) > 1e-12:
            raise ValueError("collector 与 M1 策略周期不一致")

    def _sample_state(self):
        sample = self.env._sample()
        return self.env._observations.assembler.measure(sample), sample

    def _policy_allowed(self):
        recovery = getattr(self.env, '_recover_task', None)
        if recovery is not None:
            return ~recovery.hold_mask
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _labels(self, state, timestamp, *, episode_id=None, control_epoch=None):
        force = state.wheel_ground_force_n
        contact = (force > self.contact_threshold_n).to(torch.float32)
        valid = (torch.isfinite(state.velocity_origin_body_m_s).all(-1)
                 & torch.isfinite(state.policy_velocity_rad_s).all(-1))
        return PhysicalLabels(
            velocity_body_m_s=state.velocity_origin_body_m_s.clone(),
            wheel_contact=contact,
            joint_velocity_rad_s=state.policy_velocity_rad_s.clone(),
            valid=valid,
            timestamp_s=timestamp.clone(),
            episode_id=(self._episode_id.clone() if episode_id is None else episode_id.clone()),
            control_epoch=(self._control_epoch.clone() if control_epoch is None else control_epoch.clone()),
        )

    def _observation(self, state, timestamp):
        obs = self.env.obs_buf
        if obs is None:
            raise RuntimeError("WheeledLabEnv 尚未产生 obs_buf")
        frame = obs["policy"].reshape(self.num_envs, 5, 35)[:, -1].clone()
        critic = obs["critic"].clone()
        if critic.shape != (self.num_envs, 182):
            raise ValueError("WheeledLabEnv critic 必须是 182D 环境块")
        return LabeledObservation(
            frame=frame,
            critic=critic,
            timestamp_s=timestamp.clone(),
            episode_id=self._episode_id.clone(),
            control_epoch=self._control_epoch.clone(),
            policy_allowed=self._policy_allowed().clone(),
            labels=self._labels(state, timestamp),
        )

    def reset(self, seed=None):
        if seed is not None:
            raise ValueError("M1 适配器不在此处重设全局随机种子")
        self.env.reset()
        self._episode_id.fill_(1)
        self._control_epoch.fill_(1)
        state, _ = self._sample_state()
        return self._observation(state, self.env._clock())

    def step(self, issued_action, completed_estimate):
        policy_before = self._policy_allowed().clone()
        self.env._completed_estimate = completed_estimate
        obs, reward, terminated, truncated, extras = self.env.step(issued_action)
        self.last_timing = dict(extras.get("timing", {}))
        self.last_counts = {}
        final_state = self.env._terminal_state
        final_sample = self.env._terminal_sample
        if final_state is None or final_sample is None or self.env._terminal_critic_182 is None:
            raise RuntimeError("WheeledLabEnv 未保存 PRE-reset 终态")
        final_time = final_state.timestamp_s
        final_episode = self._episode_id.clone()
        final_control = self._control_epoch.clone()
        final_labels = self._labels(final_state, final_time,
                                    episode_id=final_episode, control_epoch=final_control)
        done = terminated | truncated
        handoff = (policy_before != self._policy_allowed()) & ~done
        self._episode_id.add_(done.long())
        self._control_epoch.add_((done | handoff).long())
        next_state, _ = self._sample_state()
        next_observation = self._observation(next_state, self.env._clock())
        final_critic = self.env._terminal_critic_182.clone()
        return LabeledStep(
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            final_critic=final_critic,
            final_critic_valid=torch.isfinite(final_critic).all(-1),
            final_labels=final_labels,
            fully_policy_owned=policy_before,
            handoff_after=handoff,
            terminal_frame=self.env._terminal_frame.clone(),
        )

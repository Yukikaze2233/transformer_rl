"""Private labelled collection boundary for the structured policy candidate.

An environment adapter must provide named PRE-reset labels and actual control
ownership. This module neither guesses privileged-state columns nor starts a
simulator. Existing public VectorEnv and checkpoint contracts are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import time

import torch
from torch.distributions import Normal

from ._policy_optimization import StructuredBatch
from ._task_protocol import CompletedEstimate
from ._training_contract import AuxiliaryLabels, BoundaryMasks, ObservationWindows, masked_gae


ACTION_ORDER = ('L_joint1', 'LL_joint1', 'R_joint1', 'RR_joint1', 'L_joint3', 'R_joint3')


@dataclass(frozen=True)
class PhysicalLabels:
    velocity_body_m_s: torch.Tensor
    wheel_contact: torch.Tensor
    joint_velocity_rad_s: torch.Tensor
    valid: torch.Tensor
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    velocity_reference: str = 'base_link_origin_body'
    joint_order: tuple[str, ...] = ACTION_ORDER

    def owned(self):
        return type(self)(**{field.name: (getattr(self, field.name).detach().clone()
                                         if isinstance(getattr(self, field.name), torch.Tensor)
                                         else getattr(self, field.name)) for field in fields(self)})


@dataclass(frozen=True)
class LabeledObservation:
    frame: torch.Tensor
    critic: torch.Tensor
    timestamp_s: torch.Tensor
    episode_id: torch.Tensor
    control_epoch: torch.Tensor
    policy_allowed: torch.Tensor
    labels: PhysicalLabels

    def owned(self):
        return type(self)(**{field.name: (self.labels.owned() if field.name == 'labels'
                                         else getattr(self, field.name).detach().clone()) for field in fields(self)})


@dataclass(frozen=True)
class LabeledStep:
    observation: LabeledObservation
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_critic: torch.Tensor
    final_critic_valid: torch.Tensor
    final_labels: PhysicalLabels
    fully_policy_owned: torch.Tensor
    handoff_after: torch.Tensor


@dataclass(frozen=True)
class StructuredCollectionConfig:
    policy_dt_s: float = 0.01
    gamma: float = 0.99498743710662
    gae_lambda: float = 0.9746794344808963
    action_bounds: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 9.0, 9.0)

    def __post_init__(self):
        if (type(self.policy_dt_s) not in (int, float) or self.policy_dt_s != 0.01
                or type(self.gamma) not in (int, float) or not 0 < self.gamma < 1
                or type(self.gae_lambda) not in (int, float) or not 0 <= self.gae_lambda <= 1
                or not isinstance(self.action_bounds, (tuple, list)) or len(self.action_bounds) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in self.action_bounds)):
            raise ValueError('collection requires the fixed 100Hz discount and six positive action bounds')
        object.__setattr__(self, 'action_bounds', tuple(self.action_bounds))


class StructuredCollector:
    """Freeze one behavior policy per complete collection call.

    ``env.step(issued_action, completed_estimate)`` may use the estimate only to
    prepare the NEXT tick's task. Its reward reference is the CURRENT task.
    Availability clocks continue across reset; episode/control IDs mark history
    boundaries. Explicit reset is required after any failed collection.
    """

    def __init__(self, env, model, config=StructuredCollectionConfig()):
        if type(env.num_envs) is not int or env.num_envs < 1:
            raise ValueError('environment requires a positive row count')
        validator = getattr(env, 'validate_collection_config', None)
        if validator is not None:
            if not callable(validator):
                raise ValueError('environment collection-contract validator must be callable')
            validator(config)
        self.env, self.model, self.config = env, model, config
        self.num_envs, self.device = env.num_envs, torch.device(env.device)
        self.history_length = model.actor.policy.history_length
        self.bounds = torch.tensor(config.action_bounds, dtype=torch.float32, device=self.device)
        reference = next(model.parameters())
        if reference.device != self.device or reference.dtype != torch.float32:
            raise ValueError('candidate collection requires a float32 model on the environment device')
        self._current = None
        self._history = None
        self.attempted_vector_steps = 0
        self.returned_vector_steps = 0
        self.last_metrics = {}

    def _tensor(self, name, value, width=None, dtype=torch.float32):
        shape = (self.num_envs, width) if width is not None else (self.num_envs,)
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype != dtype or value.device != self.device):
            raise ValueError(f'{name} requires {shape}/{dtype} on {self.device}')

    def _label_invalid(self, labels):
        if (not isinstance(labels, PhysicalLabels) or labels.velocity_reference != 'base_link_origin_body'
                or tuple(labels.joint_order) != ACTION_ORDER):
            raise ValueError('physical labels require the named link origin/body velocity and policy joint order')
        for name, width, dtype in (('velocity_body_m_s', 3, torch.float32), ('wheel_contact', 2, torch.float32),
                                    ('joint_velocity_rad_s', 6, torch.float32), ('valid', None, torch.bool),
                                    ('timestamp_s', None, torch.float64), ('episode_id', None, torch.long),
                                    ('control_epoch', None, torch.long)):
            self._tensor('labels.' + name, getattr(labels, name), width, dtype)
        invalid = (~torch.isfinite(labels.velocity_body_m_s).all(-1)
                   | ~torch.isfinite(labels.wheel_contact).all(-1)
                   | ~torch.isfinite(labels.joint_velocity_rad_s).all(-1)
                   | (labels.wheel_contact < 0).any(-1) | (labels.wheel_contact > 1).any(-1)
                   | ~torch.isfinite(labels.timestamp_s) | (labels.episode_id < 0) | (labels.control_epoch < 0))
        return invalid & labels.valid

    def _observation(self, value):
        if not isinstance(value, LabeledObservation):
            raise TypeError('candidate environment must return a LabeledObservation')
        for name, width, dtype in (('frame', 35, torch.float32), ('critic', 81, torch.float32),
                                   ('timestamp_s', None, torch.float64), ('episode_id', None, torch.long),
                                   ('control_epoch', None, torch.long), ('policy_allowed', None, torch.bool)):
            self._tensor(name, getattr(value, name), width, dtype)
        invalid = (~torch.isfinite(value.frame).all(-1) | (value.frame[:, 14:16] != 0).any(-1)
                   | ~torch.isfinite(value.timestamp_s) | (value.episode_id < 0) | (value.control_epoch < 0)
                   | (value.policy_allowed & ~torch.isfinite(value.critic).all(-1)))
        invalid |= self._label_invalid(value.labels)
        invalid |= value.policy_allowed & ~value.labels.valid
        invalid |= value.labels.valid & ((value.labels.timestamp_s - value.timestamp_s).abs() > 1e-8)
        invalid |= value.labels.valid & ((value.labels.episode_id != value.episode_id)
                                        | (value.labels.control_epoch != value.control_epoch))
        if invalid.any():
            raise ValueError('observation contains invalid features or labels from a different physical sample')
        return value.owned()

    def _step(self, value, current, issued):
        if not isinstance(value, LabeledStep):
            raise TypeError('candidate environment must return a LabeledStep')
        next_observation = self._observation(value.observation)
        for name, width, dtype in (('reward', None, torch.float32), ('terminated', None, torch.bool),
                                   ('truncated', None, torch.bool), ('final_critic', 81, torch.float32),
                                   ('final_critic_valid', None, torch.bool),
                                   ('fully_policy_owned', None, torch.bool), ('handoff_after', None, torch.bool)):
            self._tensor(name, getattr(value, name), width, dtype)
        invalid = self._label_invalid(value.final_labels)
        final = value.final_labels
        end_time = current.timestamp_s + self.config.policy_dt_s
        invalid |= (next_observation.timestamp_s - end_time).abs() > 1e-8
        invalid |= final.valid & ((final.timestamp_s - end_time).abs() > 1e-8)
        done = value.terminated | value.truncated
        episode_changed = next_observation.episode_id != current.episode_id
        control_changed = next_observation.control_epoch != current.control_epoch
        invalid |= (episode_changed != done) | (next_observation.episode_id < current.episode_id)
        invalid |= (next_observation.control_epoch < current.control_epoch)
        invalid |= value.handoff_after & ~control_changed
        invalid |= (next_observation.policy_allowed != current.policy_allowed) & ~control_changed
        invalid |= value.fully_policy_owned & ~current.policy_allowed
        invalid |= current.policy_allowed & ~value.fully_policy_owned & ~control_changed
        invalid |= value.fully_policy_owned & ~torch.isfinite(value.reward)
        timeout = value.fully_policy_owned & value.truncated & ~value.terminated & ~value.handoff_after
        invalid |= timeout & (~value.final_critic_valid | ~torch.isfinite(value.final_critic).all(-1))
        same_final_epoch = ((final.episode_id == current.episode_id)
                            & (final.control_epoch == current.control_epoch))
        expected_final_labels = value.fully_policy_owned & ~value.terminated & ~value.handoff_after
        invalid |= expected_final_labels & (~final.valid | ~same_final_epoch)
        # The endpoint metadata stamps both final critic and auxiliary labels,
        # even when a particular auxiliary channel is unavailable.
        invalid |= timeout & (~same_final_epoch | ~torch.isfinite(final.timestamp_s)
                              | ((final.timestamp_s - end_time).abs() > 1e-8))
        continuing = value.fully_policy_owned & ~done & ~value.handoff_after
        invalid |= continuing & (control_changed | ~next_observation.policy_allowed)
        ordinary = continuing & ~episode_changed & ~control_changed
        invalid |= ordinary & (next_observation.frame[:, 22:28] - issued).abs().gt(1e-6).any(-1)
        if invalid.any():
            raise ValueError('step clock/ownership/reset/issued-action or terminal-label contract differs')
        return LabeledStep(next_observation, *(getattr(value, field.name).detach().clone()
                                               if field.name != 'final_labels' else value.final_labels.owned()
                                               for field in fields(LabeledStep) if field.name != 'observation'))

    def _version(self):
        return tuple((name, id(value), value._version) for name, value in (
            *self.model.named_parameters(), *self.model.named_buffers()))

    def _check_version(self, expected):
        if self._version() != expected or any(module.training for module in self.model.modules()):
            raise RuntimeError('policy or critic changed during collection')

    @torch.no_grad()
    def reset(self, seed=None):
        self._current, self._history = None, None
        current = self._observation(self.env.reset(seed=seed))
        self._history = current.frame[:, None].expand(-1, self.history_length, -1).clone()
        self._current = current
        return current.owned()

    @torch.no_grad()
    def collect(self, steps=48):
        if type(steps) is not int or steps < 1:
            raise ValueError('collection requires a positive complete vector-step count')
        if self._current is None:
            raise RuntimeError('explicitly reset a fresh environment before collection')
        self.model.eval()
        version = self._version()
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        prefix = self._history[:, :-1].transpose(0, 1).clone()
        collected = {name: [] for name in ('frame', 'critic', 'raw', 'issued', 'mean', 'log_std', 'logp', 'value',
                                          'reward', 'next_value', 'reset', 'velocity', 'contact', 'velocity_delta')}
        all_masks = []
        reset_before = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            for step in range(steps):
                self._check_version(version)
                current = self._current
                mean, raw_state, _ = self.model.actor.policy.outputs(self._history)
                log_std = self.model.actor.log_std.clamp(-4.0, 1.0).expand_as(mean)
                raw_action = Normal(mean, log_std.exp(), validate_args=False).sample()
                evaluation = self.model.actor._evaluation(mean, self.model.actor.log_std, raw_action)
                issued = raw_action.clamp(-self.bounds, self.bounds)
                old_value = torch.zeros(self.num_envs, device=self.device)
                allowed = current.policy_allowed.nonzero().flatten()
                if allowed.numel():
                    old_value[allowed] = self.model.critic(current.critic[allowed])
                if (not torch.isfinite(raw_action).all() or not torch.isfinite(raw_state).all()
                        or not torch.isfinite(evaluation.log_prob).all() or not torch.isfinite(old_value).all()):
                    raise FloatingPointError('nonfinite behavior output before submitting an action')
                gravity = current.frame[:, 7:10]
                norm = gravity.norm(dim=-1)
                estimate = CompletedEstimate(
                    (-(raw_state[:, :3] * gravity).sum(-1) / norm.clamp_min(1e-8)).clone(),
                    raw_state[:, 6:8].sigmoid().clone(),
                    (-gravity[:, 2] / norm.clamp_min(1e-8)).clamp(-1, 1).acos(),
                    (current.frame[:, 4:7] * 2).norm(dim=-1), current.timestamp_s.clone(),
                    torch.isfinite(raw_state).all(-1) & (norm > 1e-6),
                )
                # Current inputs/labels are already owned before step can reuse
                # native buffers, apply a script override or auto-reset a row.
                self._current = None
                self.attempted_vector_steps += 1
                returned = self.env.step(issued.clone(), estimate)
                self.returned_vector_steps += 1
                result = self._step(returned, current, issued)
                self._check_version(version)
                final = result.final_labels
                same_final_epoch = ((final.episode_id == current.episode_id)
                                    & (final.control_epoch == current.control_epoch))
                rollout_end = torch.full_like(result.terminated, step == steps - 1)
                masks = BoundaryMasks.from_events(
                    fully_policy_owned=result.fully_policy_owned, terminated=result.terminated,
                    truncated=result.truncated, final_value_valid=result.final_critic_valid,
                    handoff_after=result.handoff_after, rollout_end=rollout_end,
                    state_label_valid=current.labels.valid,
                    next_label_valid=current.labels.valid & final.valid,
                    next_same_control_epoch=same_final_epoch,
                )
                next_value = torch.zeros_like(old_value)
                ordinary = (masks.bootstrap & ~result.truncated).nonzero().flatten()
                if ordinary.numel():
                    next_value[ordinary] = self.model.critic(result.observation.critic[ordinary])
                timeout = (masks.bootstrap & result.truncated).nonzero().flatten()
                if timeout.numel():
                    next_value[timeout] = self.model.critic(result.final_critic[timeout])
                values = (current.frame, current.critic, raw_action, issued, mean, log_std,
                          evaluation.log_prob, old_value, result.reward, next_value, reset_before,
                          current.labels.velocity_body_m_s, current.labels.wheel_contact,
                          final.joint_velocity_rad_s - current.labels.joint_velocity_rad_s)
                for name, tensor in zip(collected, values):
                    collected[name].append(tensor.detach().clone())
                all_masks.append(masks)
                next_observation = result.observation
                reset_before = ((next_observation.episode_id != current.episode_id)
                                | (next_observation.control_epoch != current.control_epoch))
                self._history[:, :-1] = self._history[:, 1:].clone()
                self._history[:, -1] = next_observation.frame
                self._history[reset_before] = next_observation.frame[reset_before, None]
                self._current = next_observation
            records = {name: torch.stack(values) for name, values in collected.items()}
            masks = BoundaryMasks(*(torch.stack([getattr(item, field.name) for item in all_masks])
                                    for field in fields(BoundaryMasks)))
            advantage, returns = masked_gae(records['reward'], records['value'], records['next_value'], masks,
                                            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
            windows = ObservationWindows(prefix, records['frame'], records['reset'])
            flattened_masks = BoundaryMasks(*(getattr(masks, field.name).flatten() for field in fields(masks)))
            labels = AuxiliaryLabels(records['velocity'].flatten(0, 1), records['contact'].flatten(0, 1),
                                     records['velocity_delta'].flatten(0, 1))
            batch = StructuredBatch(windows, records['critic'].flatten(0, 1), records['raw'].flatten(0, 1),
                                     records['issued'].flatten(0, 1), records['mean'].flatten(0, 1),
                                     records['log_std'].flatten(0, 1), records['logp'].flatten(),
                                     records['value'].flatten(), advantage.flatten(), returns.flatten(),
                                     flattened_masks, labels)
            batch.validate(self.model)
            self._check_version(version)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.last_metrics = {'vector_steps': steps, 'sampled_transitions': steps * self.num_envs,
                                 'policy_owned_transitions': int(flattened_masks.actor.sum()),
                                 'raw_history_elements': windows.raw_storage_elements,
                                 'collection_elapsed_s': time.perf_counter() - started,
                                 'attempted_vector_steps_lifetime': self.attempted_vector_steps,
                                 'returned_vector_steps_lifetime': self.returned_vector_steps}
            return batch
        except BaseException:
            self._current, self._history = None, None
            raise

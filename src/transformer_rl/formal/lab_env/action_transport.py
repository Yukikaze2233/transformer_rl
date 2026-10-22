"""Per-environment action transport delay at policy clock boundaries.

This models a transport/arrival path between the policy action and the
controller target latch. It is not a motor-response model and it is not a
measurement of USB one-way latency. Delays are expressed in whole policy
steps so no sub-policy step is fabricated; an experimentally measured
one-way delay smaller than one policy period therefore rounds to zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ActionTransportConfig:
    """Whole-step transport delays; nominal_fraction is the zero-delay probability."""

    enabled: bool = False
    delays_steps: tuple[int, ...] = (0,)
    nominal_fraction: float = 0.2

    def __post_init__(self):
        if (type(self.enabled) is not bool
                or not isinstance(self.delays_steps, (tuple, list)) or not self.delays_steps
                or any(type(value) is not int or value < 0 or value > 16
                       for value in self.delays_steps)
                or len(set(self.delays_steps)) != len(self.delays_steps)
                or type(self.nominal_fraction) not in (int, float)
                or not math.isfinite(self.nominal_fraction)
                or not 0 <= self.nominal_fraction <= 1):
            raise ValueError('action transport requires nonnegative whole-step delays and a nominal fraction')
        object.__setattr__(self, 'delays_steps', tuple(self.delays_steps))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'enabled', 'delays_steps', 'nominal_fraction'}:
            raise ValueError('action transport configuration fields differ')
        return cls(**value)


class DelayedActionTransport:
    """Bounded per-row action history with reset-local sampled delays.

    Rows marked ``policy_allowed=False`` are script-owned and are never
    enqueued. When ownership returns, that row's history is flushed before the
    first policy packet is stored, so no uncredited script action can arrive
    later with policy credit.
    """

    def __init__(self, num_envs, action_dim, config, *, seed, device='cpu'):
        if (type(num_envs) is not int or num_envs < 1 or type(action_dim) is not int or action_dim < 1
                or not isinstance(config, ActionTransportConfig)
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError('action transport requires explicit batch, action width, config and seed')
        self.num_envs, self.action_dim = num_envs, action_dim
        self.config, self.device = config, torch.device(device)
        self.capacity = max(config.delays_steps) + 1
        self.delays = torch.tensor(config.delays_steps, dtype=torch.long, device=self.device)
        self.history = torch.zeros(self.capacity, num_envs, action_dim,
                                   dtype=torch.float32, device=self.device)
        self.lags = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.index = 0
        self.reset_rows_consumed = 0
        # The previous interval's ownership decides whether this interval is a
        # policy handoff. New/reset rows start with no policy packet buffered.
        self._policy_allowed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._stamp_history = torch.zeros(self.capacity, num_envs, dtype=torch.float64, device=self.device)
        self.arrived_generated_at_s = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        if config.enabled:
            self.reset(torch.ones(num_envs, dtype=torch.bool, device=self.device))
        else:
            self.lags.zero_()

    @torch.no_grad()
    def reset(self, mask, *, difficulty=1.):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.device != self.device or mask.dtype != torch.bool):
            raise ValueError('action transport reset requires an explicit device mask')
        if type(difficulty) not in (int, float) or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
            raise ValueError('action delay difficulty must be a finite [0,1] scale')
        rows = mask.nonzero().flatten()
        if not rows.numel():
            return
        self.history[:, rows] = 0
        self._stamp_history[:, rows] = 0
        self.arrived_generated_at_s[rows] = 0
        self._policy_allowed[rows] = False
        if not self.config.enabled:
            self.lags[rows] = 0
            return
        # A configured zero delay is not a second base category. Sampling the
        # full set uniformly and then forcing nominal zero would make (0,1)
        # with nominal_fraction=.5 produce .5 + .5*.5 = 75% zero delay.
        # nominal_fraction is the sole zero-delay probability.
        nonzero = self.delays[self.delays > 0]
        if nonzero.numel():
            choose = torch.randint(len(nonzero), (rows.numel(),),
                                   generator=self.generator, device=self.device)
            lags = nonzero[choose]
        else:
            lags = torch.zeros(rows.numel(), dtype=torch.long, device=self.device)
        nominal = torch.rand(rows.numel(), generator=self.generator, device=self.device)
        lags = torch.where(nominal < self.config.nominal_fraction,
                           torch.zeros_like(lags), lags)
        self.lags[rows] = (lags * difficulty).round().long()
        self.reset_rows_consumed += rows.numel()

    @torch.no_grad()
    def step(self, actions, *, policy_allowed=None, issued_at_s=None):
        if (not isinstance(actions, torch.Tensor)
                or actions.shape != (self.num_envs, self.action_dim)
                or actions.device != self.device or actions.dtype != torch.float32
                or not torch.isfinite(actions).all()):
            raise ValueError('action transport requires finite float32 [N,A] actions on its device')
        if policy_allowed is not None and (
                not isinstance(policy_allowed, torch.Tensor)
                or policy_allowed.shape != (self.num_envs,)
                or policy_allowed.device != self.device or policy_allowed.dtype != torch.bool):
            raise ValueError('action transport ownership requires a device row mask')
        if issued_at_s is None:
            issued_at_s = self.arrived_generated_at_s.new_zeros(self.num_envs)
        if (not isinstance(issued_at_s, torch.Tensor) or issued_at_s.shape != (self.num_envs,)
                or issued_at_s.dtype != torch.float64 or issued_at_s.device != self.device
                or not torch.isfinite(issued_at_s).all()):
            raise ValueError('action issuance requires finite per-row timestamps')
        if not self.config.enabled:
            self.arrived_generated_at_s.copy_(issued_at_s)
            return actions
        if policy_allowed is None:
            policy_allowed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Script-owned proposals are discarded, and the first policy interval
        # after a handoff starts with a neutral queue. This prevents a delayed
        # action from the script recovery interval from being replayed with
        # policy credit after ownership returns to the actor.
        handoff = policy_allowed & ~self._policy_allowed
        blocked = ~policy_allowed | handoff
        self.history.masked_fill_(blocked[None, :, None], 0.)
        self._stamp_history.copy_(torch.where(blocked[None], issued_at_s[None], self._stamp_history))
        actions = torch.where(policy_allowed[:, None], actions, 0.)
        self.index = (self.index + 1) % self.capacity
        self.history[self.index] = actions
        self._stamp_history[self.index] = issued_at_s
        self._policy_allowed.copy_(policy_allowed)
        slots = (self.index - self.lags) % self.capacity
        self.arrived_generated_at_s.copy_(self._stamp_history[slots, torch.arange(self.num_envs, device=self.device)])
        return self.history[slots, torch.arange(self.num_envs, device=self.device)]

    def critic_steps(self):
        """Two nonnegative step counts for the privileged critic block."""
        value = self.lags.to(torch.float32)
        return torch.stack((value, value), -1) if self.config.enabled else torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device)

    def state_dict(self):
        """Save the independent random stream, not old physical action packets."""
        return {'format': 'lab_action_transport_v1', 'configuration': asdict(self.config),
                'rng': self.generator.get_state().clone(),
                'reset_rows_consumed': self.reset_rows_consumed}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict)
                or set(state) != {'format', 'configuration', 'rng', 'reset_rows_consumed'}
                or state['format'] != 'lab_action_transport_v1'
                or state['configuration'] != asdict(self.config)
                or type(state['reset_rows_consumed']) is not int
                or state['reset_rows_consumed'] < 0):
            raise ValueError('action transport checkpoint contract differs')
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].cpu())
        self.generator = generator
        self.reset_rows_consumed = state['reset_rows_consumed']
        self.history.zero_()
        self.lags.zero_()
        self._policy_allowed.zero_()
        self._stamp_history.zero_()
        self.arrived_generated_at_s.zero_()
        self.index = 0

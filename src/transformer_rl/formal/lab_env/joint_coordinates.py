"""Continuous encoder coordinates at the simulator read boundary.

Free PhysX revolute coordinates wrap, even though the physical joint and its
velocity remain continuous. Track turns before coupled-leg observation and
control; independently wrapping the two PD errors would change their relative
motion. Native state, targets, velocity and mechanical limits stay untouched.
"""
from __future__ import annotations

import math

import torch


class ContinuousJointCoordinates:
    def __init__(self, initial_position, joint_names, continuous_names, physics_dt_s):
        if (not isinstance(initial_position, torch.Tensor) or initial_position.ndim != 2
                or initial_position.dtype != torch.float32
                or initial_position.shape[1] != len(joint_names)
                or len(set(joint_names)) != len(joint_names)
                or len(set(continuous_names)) != len(continuous_names)
                or not set(continuous_names).issubset(joint_names)
                or not math.isfinite(physics_dt_s) or physics_dt_s <= 0):
            raise ValueError('continuous coordinates require exact named native rows and a physical clock')
        self.shape, self.device = initial_position.shape, initial_position.device
        self.indices = torch.tensor([joint_names.index(name) for name in continuous_names],
                                    dtype=torch.long, device=self.device)
        self.physics_dt_s = float(physics_dt_s)
        self._previous_raw = initial_position.detach().clone()
        self._position = initial_position.detach().clone()
        self._turns = torch.zeros(self.shape[0], len(continuous_names),
                                  dtype=torch.int64, device=self.device)
        self._tick = None

    def _layout(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != self.shape
                or value.device != self.device or value.dtype != torch.float32):
            raise ValueError('continuous coordinates differ from the bound native layout')

    @torch.no_grad()
    def read(self, raw_position, velocity, physical_tick):
        self._layout(raw_position)
        self._layout(velocity)
        if type(physical_tick) is not int or physical_tick < 0:
            raise ValueError('continuous coordinates require an integer physical tick')
        if self._tick == physical_tick:
            return self._position
        if self._tick is not None and physical_tick != self._tick + 1:
            raise RuntimeError('continuous coordinates lost a physical sample; reset explicitly')
        raw = raw_position.index_select(1, self.indices)
        speed = velocity.index_select(1, self.indices)
        # A half-turn per physical interval cannot be unambiguously decoded.
        # Stop numerical failures rather than interpreting them as wrap events.
        if (not torch.isfinite(raw).all() or not torch.isfinite(speed).all()
                or (speed.abs() * self.physics_dt_s >= math.pi).any()):
            raise RuntimeError('continuous joint state is nonfinite or angular sampling is ambiguous')
        previous = self._previous_raw.index_select(1, self.indices)
        # PhysX eREVOLUTE subtracts/adds 4*pi at +/-2*pi. This is a native
        # representation correction, not independent wrapping of PD errors.
        period = 2 * math.tau
        turns = torch.round((previous.double() - raw.double()) / period).long()
        delta = raw.double() - previous.double() + turns.double() * period
        if (delta.abs() >= math.pi).any():
            raise RuntimeError('continuous joint position increment is ambiguous')
        self._turns.add_(turns)
        position = raw_position.detach().clone()
        unwrapped = (raw.double() + self._turns.double() * period).float()
        position.index_copy_(1, self.indices, unwrapped)
        self._previous_raw = raw_position.detach().clone()
        self._position = position
        self._tick = physical_tick
        return position

    @torch.no_grad()
    def reset_rows(self, ids, written_position):
        """Seed the actual reset coordinates without altering continuing rows."""
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long
                or ids.device != self.device or written_position.shape != (len(ids), self.shape[1])
                or written_position.dtype != torch.float32 or written_position.device != self.device):
            raise ValueError('continuous coordinate reset differs from the native reset rows')
        # Earlier PRE-reset samples may retain these buffers. Publish new ones.
        self._previous_raw = self._previous_raw.clone()
        self._position = self._position.clone()
        self._previous_raw.index_copy_(0, ids, written_position)
        self._position.index_copy_(0, ids, written_position)
        self._turns[ids] = 0

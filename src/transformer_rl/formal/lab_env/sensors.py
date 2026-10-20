"""Reuse physical-clock sensor noise/lag with explicit same-clock reset samples."""
from __future__ import annotations

from dataclasses import fields

import torch

from ..control.observation import MeasuredState
from ..control.sensors import SensorModel


class LabSensorModel(SensorModel):
    """A reset reading is current state, never fabricated older sensor history.

    Normal physical samples use the original independent noise RNG. Reset
    projections use the reset RNG, preserving continuing rows at this endpoint
    and their future noise stream. Cold restoration retains neither cache.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_projection_rows_consumed = 0

    @torch.no_grad()
    def seed_after_reset(self, mask, state: MeasuredState):
        if (not isinstance(state, MeasuredState) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device
                or state.timestamp_s.shape != (self.num_envs,)
                or (state.timestamp_s - self.physical_tick * self.config.physics_dt_s).abs().gt(1e-8).any()
                or (mask & (~self.initialized | self.count.ne(0))).any()):
            raise ValueError('Lab reset projection requires just-reset rows at the current native clock')
        rows = mask.nonzero().flatten()
        if rows.numel() == 0:
            return
        cached = self._sample is not None and self._sample_tick == self.physical_tick
        if not cached and rows.numel() != self.num_envs:
            raise ValueError('continuing rows require their PRE-reset cached sensor endpoint')
        raw = torch.cat((state.policy_position_rad[:, :4], torch.zeros_like(state.policy_position_rad[:, 4:]),
                         state.policy_velocity_rad_s, state.angular_velocity_body_rad_s, state.gravity_body), -1)
        if raw.shape != (self.num_envs, 18) or not torch.isfinite(raw[rows]).all():
            raise ValueError('Lab reset projection requires finite native encoder/IMU readings')
        slot = self.physical_tick % self.capacity
        self.values[slot, rows] = raw[rows]
        self.times[slot, rows] = state.timestamp_s[rows]
        self.count[rows] = 1
        packet = self._read_rows(rows, self.reset_generator)
        if cached:
            for field in fields(packet):
                getattr(self._sample, field.name)[rows] = getattr(packet, field.name)
        else:
            self._sample = packet
        self._sample_tick = self.physical_tick
        self.reset_projection_rows_consumed += rows.numel()

    def checkpoint_state_dict(self):
        return {'format': 'lab_sensor_model_v1', 'physical_sensor_model': super().checkpoint_state_dict(),
                'reset_projection_rows_consumed': self.reset_projection_rows_consumed}

    def load_after_explicit_cold_setup(self, state):
        if (not isinstance(state, dict)
                or set(state) != {'format', 'physical_sensor_model', 'reset_projection_rows_consumed'}
                or state['format'] != 'lab_sensor_model_v1'
                or type(state['reset_projection_rows_consumed']) is not int
                or state['reset_projection_rows_consumed'] < 0):
            raise ValueError('Lab sensor checkpoint differs from the explicit reset-projection contract')
        super().load_after_explicit_cold_setup(state['physical_sensor_model'])
        self.reset_projection_rows_consumed = state['reset_projection_rows_consumed']

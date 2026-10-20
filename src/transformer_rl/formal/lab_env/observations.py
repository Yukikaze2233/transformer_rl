"""35D/81D 观测组装与 5 帧历史；不依赖 Isaac，便于 parity 单测。"""
from __future__ import annotations

import torch

from ..control.observation import (CommandFeatures, MechanicalBinding,
                                   NativePhysicalSample, ObservationAssembler,
                                   SensorSample)
from .contract import ACTOR_DIM, HISTORY_LENGTH, HistoryBuffer
from .contact_measurement import LabObservationAssembler


class LabObservationBuilder:
    """复用正式 ObservationAssembler，只负责历史窗口与 reset 边界。"""

    def __init__(self, binding: MechanicalBinding, num_envs: int, *, device="cpu",
                 history_length: int = HISTORY_LENGTH):
        if not isinstance(binding, MechanicalBinding):
            raise TypeError("binding 必须是 MechanicalBinding")
        self.binding = binding
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.assembler = LabObservationAssembler(binding, num_envs, self.device)
        self.history = HistoryBuffer(num_envs, length=history_length,
                                     frame_dim=ACTOR_DIM, device=self.device)

    def _command(self, value):
        if not isinstance(value, CommandFeatures):
            raise TypeError("commands 必须是 CommandFeatures")
        return value

    def reset(self, sample: NativePhysicalSample, commands: CommandFeatures,
              sensors: SensorSample, mask, *, measured=None):
        """用同一物理样本填充 reset 行；返回 actor 历史与 critic。"""
        commands = self._command(commands)
        state = self.assembler.measure(sample) if measured is None else measured
        frame = self.assembler.actor(sensors, commands, now_s=state.timestamp_s)
        self.history.reset(mask, frame)
        return self.history.owned(), self.assembler.critic(commands, state)

    def append(self, sample: NativePhysicalSample, commands: CommandFeatures,
               sensors: SensorSample, reset, *, measured=None):
        """追加一帧并返回 [N,5,35] actor 历史与 [N,81] critic。"""
        commands = self._command(commands)
        state = self.assembler.measure(sample) if measured is None else measured
        frame = self.assembler.actor(sensors, commands, now_s=state.timestamp_s)
        self.history.append(frame, reset)
        return self.history.owned(), self.assembler.critic(commands, state)


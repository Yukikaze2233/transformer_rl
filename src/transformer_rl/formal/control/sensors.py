"""仅面向Actor的传感器模型：物理时钟缓存、显式样本年龄、编码前加噪。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .observation import MeasuredState, SensorSample


@dataclass(frozen=True)
class SensorModelConfig:
    physics_dt_s: float = .005
    policy_dt_s: float = .01
    nominal_episode_fraction: float = .2
    extra_delay_probability: float = .5
    observation_delays_s: tuple[float, ...] = (.005, .01, .015)
    leg_position_halfwidth_rad: float = .025
    leg_velocity_halfwidth_rad_s: float = .5
    wheel_velocity_halfwidth_rad_s: float = 1.
    gyro_halfwidth_rad_s: float = .25
    gravity_halfwidth: float = .05

    def __post_init__(self):
        if (self.physics_dt_s not in (.005, .001) or self.policy_dt_s != .01
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                       for v in (self.nominal_episode_fraction, self.extra_delay_probability))
                or not isinstance(self.observation_delays_s, (tuple, list)) or not self.observation_delays_s
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       or abs(v / self.physics_dt_s - round(v / self.physics_dt_s)) > 1e-8
                       for v in self.observation_delays_s)
                or any(type(getattr(self, name)) not in (int, float)
                       or not math.isfinite(getattr(self, name)) or getattr(self, name) < 0
                       for name in ('leg_position_halfwidth_rad', 'leg_velocity_halfwidth_rad_s',
                                    'wheel_velocity_halfwidth_rad_s', 'gyro_halfwidth_rad_s',
                                    'gravity_halfwidth'))):
            raise ValueError('sensor model requires physical-unit noise and past-only physical-grid delays')
        object.__setattr__(self, 'observation_delays_s', tuple(self.observation_delays_s))


class SensorModel:
    """观测延迟是工程压力项，不代表USB RTT或另一个电机响应时间常数。

    PD、critic和监督标签不经过本对象。冷恢复只接续独立RNG和消费计数，
    重新采样新物理回合；不把旧缓存读数带入重建后的机器人。
    """

    def __init__(self, num_envs, seed, config=SensorModelConfig(), *, device='cpu', initial_physical_tick=0):
        if (type(num_envs) is not int or num_envs < 1 or type(seed) is not int or not 0 <= seed < 2**63
                or not isinstance(config, SensorModelConfig) or type(initial_physical_tick) is not int
                or initial_physical_tick < 0
                or initial_physical_tick % round(config.policy_dt_s / config.physics_dt_s)):
            raise ValueError('sensor model needs an explicit batch, independent seed, clock and configuration')
        self.num_envs, self.seed, self.config, self.device = num_envs, seed, config, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.reset_generator = torch.Generator(device=self.device).manual_seed(seed ^ (1 << 32))
        self.policy_substeps = round(config.policy_dt_s / config.physics_dt_s)
        self.delay_choices = torch.tensor([round(v / config.physics_dt_s) for v in config.observation_delays_s],
                                         dtype=torch.long, device=self.device)
        self.capacity = int(max(config.observation_delays_s) / config.physics_dt_s + .5) + 1
        self.values = torch.empty(self.capacity, num_envs, 18, device=self.device)
        self.times = torch.empty(self.capacity, num_envs, dtype=torch.float64, device=self.device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.nominal = torch.zeros_like(self.initialized)
        self.lags = torch.zeros(num_envs, 4, dtype=torch.long, device=self.device)
        self._rows = torch.arange(num_envs, device=self.device)
        self._columns = torch.arange(18, device=self.device)
        self._groups = torch.tensor([0] * 6 + [1] * 6 + [2] * 3 + [3] * 3, device=self.device)
        self._halfwidth = torch.tensor([config.leg_position_halfwidth_rad] * 4 + [0., 0.]
            + [config.leg_velocity_halfwidth_rad_s] * 4 + [config.wheel_velocity_halfwidth_rad_s] * 2
            + [config.gyro_halfwidth_rad_s] * 3 + [config.gravity_halfwidth] * 3, device=self.device)
        self.physical_tick = initial_physical_tick
        self.reset_rows_consumed = self.noise_rows_consumed = 0
        self._sample_tick, self._sample = None, None

    @torch.no_grad()
    def reset(self, mask, now_s):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.device != self.device or mask.dtype != torch.bool
                or not isinstance(now_s, torch.Tensor) or now_s.shape != (self.num_envs,)
                or now_s.device != self.device or now_s.dtype != torch.float64
                or not torch.isfinite(now_s).all()
                or (now_s - self.physical_tick * self.config.physics_dt_s).abs().gt(1e-8).any()):
            raise ValueError('sensor reset requires selected rows at the actual current physical boundary')
        rows = mask.nonzero().flatten()
        if not rows.numel():
            return
        nominal = torch.rand(rows.numel(), generator=self.reset_generator, device=self.device)
        extra = torch.rand(rows.numel(), generator=self.reset_generator, device=self.device)
        choices = torch.randint(len(self.delay_choices), (rows.numel(), 4),
                                generator=self.reset_generator, device=self.device)
        self.nominal[rows] = nominal < self.config.nominal_episode_fraction
        enabled = ~self.nominal[rows] & (extra < self.config.extra_delay_probability)
        self.lags[rows] = self.delay_choices[choices] * enabled[:, None]
        self.initialized[rows], self.count[rows] = True, 0
        self.reset_rows_consumed += rows.numel()
        # 回合抽样不推进噪声RNG，避免一行reset改变其他行后续噪声序列。

    @torch.no_grad()
    def push(self, state: MeasuredState, physical_tick):
        if (not isinstance(state, MeasuredState) or type(physical_tick) is not int
                or physical_tick != self.physical_tick + 1):
            raise ValueError('sensor history appends once after the next actual physical step')
        for name, width in (('policy_position_rad', 6), ('policy_velocity_rad_s', 6),
                            ('angular_velocity_body_rad_s', 3), ('gravity_body', 3)):
            value = getattr(state, name)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width)
                    or value.dtype != torch.float32 or value.device != self.device):
                raise ValueError('sensor source shape/device/dtype differs: ' + name)
        time = state.timestamp_s
        raw = torch.cat((state.policy_position_rad[:, :4], torch.zeros_like(state.policy_position_rad[:, 4:]),
                         state.policy_velocity_rad_s, state.angular_velocity_body_rad_s, state.gravity_body), -1)
        if time.shape != (self.num_envs,) or time.dtype != torch.float64 or time.device != self.device:
            raise ValueError('sensor source clock layout differs')
        invalid = (~torch.isfinite(raw).all() | ~torch.isfinite(time).all() | ~self.initialized.all()
                   | (time - physical_tick * self.config.physics_dt_s).abs().gt(1e-8).any())
        if invalid:
            raise ValueError('sensor source needs finite same-tick measurements after explicit reset')
        self.values[physical_tick % self.capacity] = raw
        self.times[physical_tick % self.capacity] = time
        self.count = (self.count + 1).clamp_max(self.capacity)
        self.physical_tick = physical_tick

    @staticmethod
    def _owned(sample):
        return SensorSample(sample.policy_position_rad.clone(), sample.policy_velocity_rad_s.clone(),
                            sample.angular_velocity_body_rad_s.clone(), sample.gravity_body.clone(),
                            sample.sampled_at_s.clone())

    @torch.no_grad()
    def __call__(self, state: MeasuredState):
        if (not isinstance(state, MeasuredState) or state.timestamp_s.shape != (self.num_envs,)
                or state.timestamp_s.dtype != torch.float64 or state.timestamp_s.device != self.device
                or not torch.isfinite(state.timestamp_s).all() or (self.count == 0).any()
                or self.physical_tick % self.policy_substeps
                or (state.timestamp_s - self.physical_tick * self.config.physics_dt_s).abs().gt(1e-8).any()):
            raise ValueError('sensor read requires the latest actual post-reset physical sample')
        if self._sample_tick == self.physical_tick:
            return self._owned(self._sample)
        # 只选本回合已写入的槽，不复制reset投影来伪造更老的传感器样本。
        lags = torch.minimum(self.lags, (self.count - 1)[:, None])
        slots = (self.physical_tick - lags) % self.capacity
        values = self.values[slots[:, self._groups], self._rows[:, None], self._columns[None]]
        stamps = self.times[slots, self._rows[:, None]]
        noise = torch.rand(values.shape, generator=self.generator, device=self.device).mul_(2).sub_(1)
        values = values + noise * self._halfwidth * ~self.nominal[:, None]
        self.noise_rows_consumed += self.num_envs
        self._sample = SensorSample(values[:, :6], values[:, 6:12], values[:, 12:15], values[:, 15:18], stamps)
        self._sample_tick = self.physical_tick
        return self._owned(self._sample)

    def checkpoint_state_dict(self):
        return {'config': asdict(self.config), 'seed': self.seed, 'num_envs': self.num_envs,
                'generator_state': self.generator.get_state().clone(),
                'reset_generator_state': self.reset_generator.get_state().clone(),
                'reset_rows_consumed': self.reset_rows_consumed, 'noise_rows_consumed': self.noise_rows_consumed,
                'cold_disposition': 'flush physical sensor cache; retain RNG; redraw new physical episodes'}

    def load_after_explicit_cold_setup(self, state):
        expected = self.checkpoint_state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[key] != expected[key] for key in ('seed', 'num_envs', 'cold_disposition'))
                or not isinstance(state['config'], dict) or state['config'].keys() != expected['config'].keys()
                or SensorModelConfig(**state['config']) != self.config
                or any(type(state[key]) is not int or state[key] < 0
                       for key in ('reset_rows_consumed', 'noise_rows_consumed'))
                or self.initialized.any() or self.count.any() or self._sample is not None):
            raise ValueError('sensor RNG restore requires its exact configuration and an empty cold cache')
        generator = torch.Generator(device=self.device)
        reset_generator = torch.Generator(device=self.device)
        try:
            generator.set_state(state['generator_state'])
            reset_generator.set_state(state['reset_generator_state'])
        except (TypeError, RuntimeError) as error:
            raise ValueError('invalid independent sensor RNG state') from error
        self.generator = generator
        self.reset_generator = reset_generator
        self.reset_rows_consumed, self.noise_rows_consumed = state['reset_rows_consumed'], state['noise_rows_consumed']

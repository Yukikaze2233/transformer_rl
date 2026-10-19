"""启动时修改原生机械参数；不改变关节、传动比或控制周期。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class StartupDynamicsConfig:
    nominal_fraction: float = .2
    base_mass_scale: tuple[float, float] = (.9, 1.3)
    link_mass_scale: tuple[float, float] = (.9, 1.1)
    base_com_offset_m: tuple[float, float, float] = (.04, .02, .02)

    def __post_init__(self):
        if (type(self.nominal_fraction) not in (int, float)
                or not math.isfinite(self.nominal_fraction) or not 0 <= self.nominal_fraction <= 1):
            raise ValueError('nominal dynamics fraction must lie in [0,1]')
        for name in ('base_mass_scale', 'link_mass_scale'):
            bounds = tuple(getattr(self, name))
            if (len(bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                    or not 0 < bounds[0] <= 1 <= bounds[1]):
                raise ValueError('mass scale bounds must be positive and contain the nominal value')
            object.__setattr__(self, name, bounds)
        offsets = tuple(self.base_com_offset_m)
        if (len(offsets) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in offsets)):
            raise ValueError('COM offset half widths require three nonnegative distances in metres')
        object.__setattr__(self, 'base_com_offset_m', offsets)


@torch.no_grad()
def apply_startup_dynamics(scene, config, seed, *, native_to_tensor, tensor_to_native):
    """一次性写入并回读，必须早于首个真实物理步和reward/reset对象创建。

    固定形状的质量缩放同时缩放惯量：I'=sI，m'=sm。不额外随机惯量、
    关节轴或主轴姿态。冷启动按相同seed重建相同参数，不恢复旧物理状态。
    """
    bridge = scene.observation
    if (not isinstance(config, StartupDynamicsConfig) or type(seed) is not int or not 0 <= seed < 2**63
            or not callable(native_to_tensor) or not callable(tensor_to_native)):
        raise ValueError('startup dynamics requires an explicit configuration,seed and native converters')
    if (bridge.physical_tick != 0 or bridge.model_epoch != 1
            or scene.readback.get('startup_dynamics') is not None
            or scene.readback.get('native_mass_sum_verified') is not True):
        raise RuntimeError('startup dynamics requires a fresh nominal scene before its first physical step')
    bridge.before_physics_step()
    view, count = scene.robot.root_view, bridge.num_envs
    names = bridge.binding.native_body_names
    if len(names) != 19 or names[0] != 'base_link':
        raise ValueError('startup dynamics requires the selected nineteen-body native order')

    def read(name, shape):
        value = native_to_tensor(getattr(view, name)())
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('invalid native mechanical readback: ' + name)
        return value.detach().cpu().clone()

    masses = read('get_masses', (count, 19))
    coms = read('get_coms', (count, 19, 7))
    inertias = read('get_inertias', (count, 19, 9))
    matrices = inertias.reshape(count, 19, 3, 3)
    if ((masses <= 0).any() or not torch.allclose(matrices, matrices.transpose(-1, -2), atol=1e-7, rtol=1e-5)
            or (torch.linalg.eigvalsh(matrices) <= 0).any()
            or not torch.allclose(coms[..., 3:].norm(dim=-1), torch.ones(count, 19), atol=1e-4, rtol=0)):
        raise ValueError('native mass,inertia or COM orientation is not physically valid')
    expected_mass = scene.readback['nominal_source_mass_kg']
    if not torch.allclose(masses.sum(-1), torch.full((count,), expected_mass), atol=1e-5, rtol=0):
        raise ValueError('startup mechanical sources differ from the selected nominal asset')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    nominal = torch.rand(count, generator=rng) < config.nominal_fraction
    low, high = config.link_mass_scale
    # 按具名body采样，SDK枚举顺序变化不应把一根连杆的扰动移到另一根。
    canonical = tuple(sorted(names))
    order = [canonical.index(name) for name in names]
    scales = (low + (high - low) * torch.rand(count, 19, generator=rng))[:, order]
    low, high = config.base_mass_scale
    scales[:, 0] = low + (high - low) * torch.rand(count, generator=rng)
    scales[nominal] = 1.
    offset = (torch.rand(count, 3, generator=rng) * 2 - 1) * torch.tensor(config.base_com_offset_m)
    offset[nominal] = 0.
    target_mass, target_inertia = masses * scales, inertias * scales[..., None]
    target_com = coms.clone()
    target_com[:, 0, :3] += offset
    indices = tensor_to_native(torch.arange(count, dtype=torch.int32))
    # setter或最终回读失败时保持bridge无效，不能带着半份参数继续物理步。
    scene.readback['startup_dynamics'] = {'status': 'writing'}
    bridge.invalidate_model()
    view.set_masses(tensor_to_native(target_mass), indices)
    view.set_coms(tensor_to_native(target_com), indices)
    view.set_inertias(tensor_to_native(target_inertia), indices)
    actual_mass = read('get_masses', target_mass.shape)
    actual_com = read('get_coms', target_com.shape)
    actual_inertia = read('get_inertias', target_inertia.shape)
    if (not torch.allclose(actual_mass, target_mass, rtol=1e-5, atol=1e-7)
            or not torch.allclose(actual_com[..., :3], target_com[..., :3], rtol=0, atol=1e-6)
            or not torch.allclose(actual_inertia, target_inertia, rtol=1e-4, atol=1e-7)
            or not torch.allclose(actual_com[..., 3:].norm(dim=-1), torch.ones(count, 19), atol=1e-4, rtol=0)):
        raise RuntimeError('final native dynamics differs from the requested mechanical parameters')
    bridge.refresh_model()
    bound = torch.cat((actual_mass.flatten(), actual_com.flatten(), actual_inertia.flatten())).numpy().tobytes()
    record = {
        'status': 'native_readback_matched', 'seed': seed, 'configuration': asdict(config),
        'body_order': list(names), 'nominal_rows': int(nominal.sum()), 'num_envs': count,
        'mass_kg_min': float(actual_mass.sum(-1).min()), 'mass_kg_max': float(actual_mass.sum(-1).max()),
        'native_parameters_sha256': hashlib.sha256(bound).hexdigest(), 'model_epoch': bridge.model_epoch,
        'inertia_rule': 'same mass scale per body; no independent inertia randomization',
        'distribution_evidence': 'engineering prior; not a measured probability distribution',
        'physical_qualification': False,
    }
    # 固定配置和seed绑定冷启动分布；完整参数仅做摘要，不扩大checkpoint。
    record['configuration_sha256'] = hashlib.sha256(json.dumps(asdict(config), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    scene.readback['startup_dynamics'] = record
    return record

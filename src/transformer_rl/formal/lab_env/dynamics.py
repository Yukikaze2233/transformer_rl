"""Startup mechanical parameters and their actual critic/reward mass source."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import torch

from ..simulation.dynamics import StartupDynamicsConfig, sample_startup_parameters


class LabStartupDynamics:
    """Apply the existing engineering prior once before the first task reset.

    SDK initialization may already have performed solver warmup. No training
    episode, observation history or policy transition exists at this boundary.
    Cold setup recreates the same parameters from the independent fixed seed.
    """

    @torch.no_grad()
    def __init__(self, robot, binding, num_envs, nominal_mass_kg, seed, config, *,
                 device, native_to_tensor=None, tensor_to_native=None):
        if (type(num_envs) is not int or num_envs < 1 or type(seed) is not int or not 0 <= seed < 2**63
                or tuple(robot.body_names) != tuple(binding.native_body_names)
                or robot.root_view.count != num_envs
                or config is not None and not isinstance(config, StartupDynamicsConfig)):
            raise ValueError('Lab startup parameters require the exact fresh native mechanical rows')
        if native_to_tensor is None or tensor_to_native is None:
            import warp as wp

            native_to_tensor = wp.to_torch if native_to_tensor is None else native_to_tensor
            tensor_to_native = (lambda value: wp.from_torch(value, dtype=(
                wp.uint32 if value.dtype == torch.int32 else wp.float32))) if tensor_to_native is None else tensor_to_native
        view = robot.root_view

        def read(name, shape):
            value = native_to_tensor(getattr(view, name)())
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError('invalid Lab native mechanical source: ' + name)
            return value.detach().cpu().clone()

        masses = read('get_masses', (num_envs, 19))
        coms = read('get_coms', (num_envs, 19, 7))
        inertias = read('get_inertias', (num_envs, 19, 9))
        nominal = masses.sum(-1)
        if (type(nominal_mass_kg) not in (int, float) or nominal_mass_kg <= 0
                or not torch.allclose(nominal, torch.full_like(nominal, nominal_mass_kg), atol=1e-5, rtol=0)):
            raise ValueError('Lab native source mass differs from the selected mechanical asset')
        target_mass, target_com, target_inertia, nominal_rows = sample_startup_parameters(
            masses, coms, inertias, binding.native_body_names,
            config if config is not None else StartupDynamicsConfig(nominal_fraction=1.), seed)
        if config is not None:
            indices = tensor_to_native(torch.arange(num_envs, dtype=torch.int32))
            view.set_masses(tensor_to_native(target_mass), indices)
            view.set_coms(tensor_to_native(target_com), indices)
            view.set_inertias(tensor_to_native(target_inertia), indices)
        actual_mass = read('get_masses', masses.shape)
        actual_com = read('get_coms', coms.shape)
        actual_inertia = read('get_inertias', inertias.shape)
        if (not torch.allclose(actual_mass, target_mass, rtol=1e-5, atol=1e-7)
                or not torch.allclose(actual_com[..., :3], target_com[..., :3], rtol=0, atol=1e-6)
                or not torch.allclose(actual_inertia, target_inertia, rtol=1e-4, atol=1e-7)
                or not torch.allclose(actual_com[..., 3:].norm(dim=-1), torch.ones(num_envs, 19), atol=1e-4, rtol=0)):
            raise RuntimeError('Lab native startup parameters failed their actual post-write readback')
        self.mass_kg_by_environment = actual_mass.sum(-1).to(device)
        self.mass_scale = (actual_mass.sum(-1) / nominal).to(device)[:, None]
        self.base_com_local_m = actual_com[:, 0, :3].to(device)
        bound = torch.cat((actual_mass.flatten(), actual_com.flatten(), actual_inertia.flatten())).numpy().tobytes()
        configuration = None if config is None else asdict(config)
        self._receipt = {
            'status': 'native_readback_matched', 'seed': seed, 'configuration': configuration,
            'body_order': list(binding.native_body_names), 'num_envs': num_envs,
            'nominal_rows': int(nominal_rows.sum()), 'randomization_active': config is not None,
            'mass_kg_min': float(self.mass_kg_by_environment.min()),
            'mass_kg_max': float(self.mass_kg_by_environment.max()),
            'native_parameters_sha256': hashlib.sha256(bound).hexdigest(),
            'inertia_rule': 'same mass scale per body; no independent inertia randomization',
            'distribution_evidence': 'engineering prior; not a measured probability distribution',
            'before_first_task_reset': True, 'SDK_solver_warmup_already_occurred': True,
            'physical_qualification': False}
        self._receipt['configuration_sha256'] = hashlib.sha256(json.dumps(configuration, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def base_com_position(self, link_position, quaternion_xyzw):
        """Use the final native local COM instead of a pre-write SDK cache."""
        q = quaternion_xyzw / quaternion_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        imaginary, scalar = q[:, :3], q[:, 3:]
        cross = torch.linalg.cross(imaginary, self.base_com_local_m)
        return link_position + self.base_com_local_m + 2 * (scalar * cross + torch.linalg.cross(imaginary, cross))

    @property
    def receipt(self):
        return deepcopy(self._receipt)

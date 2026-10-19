"""Private explicit effort writer; native construction/readback is separate.

Active outputs are Nm, spring sliders are N. This boundary neither inserts
feedback gains nor subtracts native friction. The SDK's explicit actuator
buffer is not a force sensor or evidence of electric motor power.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..control.pressure import PressureState
from ..control.observation import MechanicalBinding


@dataclass(frozen=True)
class NativeEffortSubmission:
    commanded_by_native_joint: torch.Tensor
    native_joint_units: tuple[str, ...]
    pressure_within_fitted_curve: torch.Tensor
    pressure_within_mechanical_stroke: torch.Tensor
    pressure_endpoint_clamped: torch.Tensor


class NativeEffortWriter:
    """Write the owned articulation's complete 18-joint effort target.

    The native scene must independently verify zero active stiffness/damping,
    configured armature/losses, passive damping and effort limits. Constructing
    this writer does not certify that readback or grant SDK/GPU admission.
    """

    def __init__(self, robot, binding: MechanicalBinding, model_spec, num_envs, device='cpu'):
        if not isinstance(binding, MechanicalBinding) or type(num_envs) is not int or num_envs < 1:
            raise ValueError('native writer requires a complete mechanical binding and positive batch')
        joints = {joint['name']: joint for joint in model_spec['joints']}
        if set(joints) != set(binding.native_joint_names) or not callable(robot.set_joint_effort_target):
            raise ValueError('native writer requires the complete bound joint specification and explicit target API')
        self.robot, self.binding, self.num_envs, self.device = robot, binding, num_envs, torch.device(device)
        self._view = robot.root_view
        self._identity()
        self._active = torch.tensor(binding.policy_joint_indices, device=self.device)
        self._springs = torch.tensor([binding.native_joint_names.index(name)
                                      for name in ('L_spring_slide', 'R_spring_slide')], device=self.device)
        if any(joints[name]['type'] != 'prismatic' for name in ('L_spring_slide', 'R_spring_slide')):
            raise ValueError('gas pressure must bind to native prismatic force coordinates')
        if any(joints[binding.native_joint_names[index]]['type'] not in ('continuous', 'revolute')
               for index in binding.policy_joint_indices):
            raise ValueError('all six active outputs must be rotary effort coordinates')
        self.units = tuple('N' if joints[name]['type'] == 'prismatic' else 'Nm'
                           for name in binding.native_joint_names)
        self._active_limit = torch.tensor((40., 40., 40., 40., 4.5, 4.5), device=self.device)
        self.submissions = 0

    def _identity(self):
        if (self.robot.root_view is not self._view
                or tuple(self.robot.joint_names) != self.binding.native_joint_names
                or tuple(self.robot.body_names) != self.binding.native_body_names):
            raise RuntimeError('native effort owner/view/joint/body identity changed')

    @torch.no_grad()
    def submit(self, mean_active_effort_nm, pressure: PressureState):
        self._identity()
        if (not isinstance(mean_active_effort_nm, torch.Tensor)
                or mean_active_effort_nm.shape != (self.num_envs, 6)
                or mean_active_effort_nm.device != self.device or mean_active_effort_nm.dtype != torch.float32
                or not torch.isfinite(mean_active_effort_nm).all()
                or (mean_active_effort_nm.abs() > self._active_limit + 1e-5).any()
                or not isinstance(pressure, PressureState)):
            raise ValueError('native active output requires finite bounded Nm in policy order and named pressure')
        force = pressure.slider_effort_n
        if (not isinstance(force, torch.Tensor) or force.shape != (self.num_envs, 2)
                or force.device != self.device or force.dtype != torch.float32
                or not torch.isfinite(force).all() or ((force < 0) | (force > 1000)).any()):
            raise ValueError('native spring pressure requires finite extension forces inside the explicit1000N cap')
        for value in (pressure.within_fitted_curve, pressure.within_mechanical_stroke, pressure.endpoint_clamped):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 2)
                    or value.dtype != torch.bool or value.device != self.device):
                raise ValueError('pressure domain flags must remain explicit same-batch booleans')
        command = torch.zeros(self.num_envs, 18, device=self.device)
        command[:, self._active] = mean_active_effort_nm
        command[:, self._springs] = force
        self.robot.set_joint_effort_target(command.clone())
        self.submissions += 1
        return NativeEffortSubmission(command, self.units, pressure.within_fitted_curve.clone(),
                                       pressure.within_mechanical_stroke.clone(), pressure.endpoint_clamped.clone())

    @torch.no_grad()
    def explicit_actuator_buffer(self):
        self._identity()
        values = self.robot.data.applied_torque.torch
        if (not isinstance(values, torch.Tensor) or values.shape != (self.num_envs, 18)
                or values.dtype != torch.float32 or values.device != self.device or not torch.isfinite(values).all()):
            raise ValueError('native explicit actuator buffer does not match the bound articulation')
        return {'active_output_effort_nm': values.index_select(1, self._active),
                'slider_actuator_effort_n': values.index_select(1, self._springs)}

"""Critic-only physical feature layout and fixed, grouped preprocessing.

The environment/rollout keeps SI values. Clipping and scaling happen once,
immediately before the value MLP, including PRE-reset bootstrap evaluations.
Actor observations, auxiliary labels and deployment exports are unaffected.
"""
from __future__ import annotations

import torch
from torch import nn


CRITIC182_SLICES = {
    "actor_frame": slice(0, 35),
    "root_velocity_body_m_s": slice(35, 38),
    "height_above_ground_m": slice(38, 39),
    "stiffness": slice(39, 45),
    "damping": slice(45, 51),
    "active_torque": slice(51, 57),
    "observation_delay_steps": slice(57, 61),
    "action_delay_steps": slice(61, 63),
    "wheel_velocity_body_m_s": slice(63, 69),
    "wheel_contact": slice(69, 71),
    "mass_scale": slice(71, 72),
    "wheel_material": slice(72, 78),
    "terrain_scan": slice(78, 155),
    "previous_actions": slice(155, 167),
    "joint_acceleration_rad_s2": slice(167, 173),
    "com_offset_body_m": slice(173, 176),
    "nominal_pose_delta_rad": slice(176, 182),
}

# (lower SI bound, upper SI bound, multiplier). Already encoded actor/terrain
# blocks retain their existing scales; there is no second noise application.
CRITIC_PHYSICAL_TRANSFORM = {
    "root_velocity_body_m_s": (-100., 100., 1.),
    "height_above_ground_m": (-10., 10., 5.),
    "stiffness": (-100., 100., 1.),
    "damping": (-100., 100., 1.),
    "active_torque": (-100., 100., .05),
    "observation_delay_steps": (0., 100., 1.),
    "action_delay_steps": (0., 100., 1.),
    "wheel_velocity_body_m_s": (-100., 100., 1.),
    "wheel_contact": (0., 1., 1.),
    "mass_scale": (0., 10., 1.),
    "wheel_material": (0., 100., 1.),
    "previous_actions": (-100., 100., 1.),
    "joint_acceleration_rad_s2": (-1000., 1000., .01),
    "com_offset_body_m": (-100., 100., 1.),
    "nominal_pose_delta_rad": (-100., 100., 1.),
}
CRITIC_PREPROCESSING_CONTRACT = "grouped_physical_clip_scale_185_v1"


class PrivilegedValueNetwork(nn.Sequential):
    """Same trainable MLP and state-dict keys, with fixed input conditioning.

    Nonpersistent buffers are deterministic from the source-bound contract.
    Loading an older learning state preserves weights/Adam, but intentionally
    changes its value function where the new input transform differs.
    """

    def __init__(self, input_dim, hidden_dims):
        first, second, third = hidden_dims
        super().__init__(nn.Linear(input_dim, first), nn.ELU(),
                         nn.Linear(first, second), nn.ELU(),
                         nn.Linear(second, third), nn.ELU(),
                         nn.Linear(third, 1), nn.Flatten(0))
        lower = torch.full((input_dim,), -float('inf'))
        upper = torch.full((input_dim,), float('inf'))
        scale = torch.ones(input_dim)
        if input_dim == 185:
            for name, (minimum, maximum, multiplier) in CRITIC_PHYSICAL_TRANSFORM.items():
                group = CRITIC182_SLICES[name]
                lower[group], upper[group], scale[group] = minimum, maximum, multiplier
            # Detached Actor velocity estimate uses the true-velocity convention.
            lower[182:], upper[182:] = -100., 100.
        self.register_buffer('_input_lower', lower, persistent=False)
        self.register_buffer('_input_upper', upper, persistent=False)
        self.register_buffer('_input_scale', scale, persistent=False)

    def preprocess(self, physical):
        return physical.clamp(self._input_lower, self._input_upper) * self._input_scale

    def forward(self, physical):
        return super().forward(self.preprocess(physical))

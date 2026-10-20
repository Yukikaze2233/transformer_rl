"""182D 环境特权块的显式组装。

该模块只拼接调用方提供的真实字段，不推断、不补零。185D 网络输入由
``StructuredActorCritic.critic_features`` 在 182D 之后追加 detach 的估计速度。
"""
from __future__ import annotations

import torch


CRITIC182_DIM = 182
TERRAIN_SCAN_DIM = 77
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


def _batch(name, value, width, n, device):
    if (not isinstance(value, torch.Tensor) or value.shape != (n, width)
            or value.device != device or value.dtype != torch.float32
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} 需要有限 float32 [{n},{width}]")
    return value


def environment_features_182(*, actor_frame, root_velocity_body_m_s, height_above_ground_m,
                             stiffness, damping, active_torque, observation_delay_steps,
                             action_delay_steps, wheel_velocity_body_m_s, wheel_contact,
                             mass_scale, wheel_material, terrain_scan, previous_actions,
                             joint_acceleration_rad_s2, com_offset_body_m,
                             nominal_pose_delta_rad):
    """按固定顺序拼成 182D 物理特权块。"""
    if not isinstance(actor_frame, torch.Tensor) or actor_frame.ndim != 2:
        raise ValueError("actor_frame 需要二维张量")
    n, device = actor_frame.shape[0], actor_frame.device
    if actor_frame.shape[1] != 35:
        raise ValueError("actor_frame 需要 35D")
    fields = {
        "root_velocity_body_m_s": (root_velocity_body_m_s, 3),
        "height_above_ground_m": (height_above_ground_m, 1),
        "stiffness": (stiffness, 6),
        "damping": (damping, 6),
        "active_torque": (active_torque, 6),
        "observation_delay_steps": (observation_delay_steps, 4),
        "action_delay_steps": (action_delay_steps, 2),
        "wheel_velocity_body_m_s": (wheel_velocity_body_m_s, 6),
        "wheel_contact": (wheel_contact, 2),
        "mass_scale": (mass_scale, 1),
        "wheel_material": (wheel_material, 6),
        "terrain_scan": (terrain_scan, TERRAIN_SCAN_DIM),
        "previous_actions": (previous_actions, 12),
        "joint_acceleration_rad_s2": (joint_acceleration_rad_s2, 6),
        "com_offset_body_m": (com_offset_body_m, 3),
        "nominal_pose_delta_rad": (nominal_pose_delta_rad, 6),
    }
    prepared = {"actor_frame": _batch("actor_frame", actor_frame, 35, n, device)}
    for name, (value, width) in fields.items():
        prepared[name] = _batch(name, value, width, n, device)
    if (prepared["mass_scale"] <= 0).any():
        raise ValueError("mass_scale 必须为正")
    if ((prepared["wheel_contact"] < 0) | (prepared["wheel_contact"] > 1)).any():
        raise ValueError("wheel_contact 必须位于 [0,1]")
    if (prepared["observation_delay_steps"] < 0).any() or (prepared["action_delay_steps"] < 0).any():
        raise ValueError("延迟步数不能为负")
    result = torch.cat(tuple(prepared[name] for name in CRITIC182_SLICES), dim=-1)
    if result.shape != (n, CRITIC182_DIM):
        raise RuntimeError("182D 特权块布局错误")
    return result


__all__ = ["CRITIC182_DIM", "CRITIC182_SLICES", "TERRAIN_SCAN_DIM", "environment_features_182"]

"""Private visible-clock jump reference, independent of physical success.

The request controller latches the actor-visible support-height target and
requested rise. Only a constant world-ground anchor is privileged. Measured
flight/landing phases may select reward terms, never actor inputs or this clock.
The curve is a task target, not an actuator command or feasibility proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from ._control_reward import RewardPhase, RewardReference
from ._physical_observation import CommandFeatures, CommandMode
from ._skill_evidence import SkillPhase


class ReferenceStage(IntEnum):
    COMPRESSION = 0
    EXTENSION = 1
    BALLISTIC = 2
    HOLD = 3


@dataclass(frozen=True)
class JumpReferenceConfig:
    gravity_m_s2: float = 9.81
    support_height_min_m: float = .17
    support_height_max_m: float = .40
    jump_min_m: float = .03
    jump_max_m: float = .30
    compression_depth_max_m: float = .04
    compression_duration_s: float = .20
    nominal_extension_duration_s: float = .10

    def __post_init__(self):
        values = vars(self).values()
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or self.support_height_min_m >= self.support_height_max_m or self.jump_min_m > self.jump_max_m
                or self.compression_depth_max_m >= self.support_height_max_m - self.support_height_min_m):
            raise ValueError('jump reference needs positive declared heights, durations and gravity')


@dataclass(frozen=True)
class JumpReferenceSample:
    height_world_z_m: torch.Tensor
    velocity_world_z_m_s: torch.Tensor
    target_apex_world_z_m: torch.Tensor
    clock_stage: torch.Tensor
    compression_depth_m: torch.Tensor
    extension_duration_s: torch.Tensor
    launch_velocity_m_s: torch.Tensor
    expected_launch_time_s: torch.Tensor
    expected_return_time_s: torch.Tensor
    extension_time_limited_by_support_range: torch.Tensor


class JumpReferenceProgram:
    """C1 compression/extension, followed by a gravity-aligned ballistic target.

    Extension velocity uses cubic smoothstep, whose integrated displacement is
    v_launch*T/2. Its launch velocity solves the requested TOTAL rise above the
    latched support-height baseline. Shortening extension to respect that height
    range preserves the requested apex; it does not clip the requested rise.
    No true takeoff time, current height, future sample or learned latent enters.
    """

    def __init__(self, config=JumpReferenceConfig()):
        if not isinstance(config, JumpReferenceConfig):
            raise ValueError('jump reference requires a parsed configuration')
        self.config = config

    @torch.no_grad()
    def sample(self, latched_height_target_m, requested_rise_m, elapsed_s, ground_anchor_world_z_m, *, active):
        values = (latched_height_target_m, requested_rise_m, ground_anchor_world_z_m)
        if (not isinstance(active, torch.Tensor) or active.ndim != 1 or not active.numel()
                or active.dtype != torch.bool):
            raise ValueError('jump reference requires an explicit active row mask')
        count, device = active.numel(), active.device
        for value in values:
            if (not isinstance(value, torch.Tensor) or value.shape != (count,)
                    or value.device != device or value.dtype != torch.float32):
                raise ValueError('jump reference heights require finite float32 SI rows')
        if (not isinstance(elapsed_s, torch.Tensor) or elapsed_s.shape != (count,)
                or elapsed_s.device != device or elapsed_s.dtype != torch.float64):
            raise ValueError('jump reference clock must be float64 on the same device')
        c = self.config
        invalid = (~torch.isfinite(elapsed_s) | (elapsed_s < 0)
                   | ~torch.isfinite(torch.stack(values)).all(0)
                   | (active & ((latched_height_target_m < c.support_height_min_m)
                                | (latched_height_target_m > c.support_height_max_m)
                                | (requested_rise_m < c.jump_min_m) | (requested_rise_m > c.jump_max_m))))
        if invalid.any():
            raise ValueError('jump reference received invalid visible height, request or causal clock')
        # Benign inactive arithmetic never invents an active request. All
        # inactive outputs below explicitly reduce to an ordinary held height.
        baseline = torch.where(active, latched_height_target_m, c.support_height_min_m).double()
        rise = torch.where(active, requested_rise_m, c.jump_min_m).double()
        height_min = torch.tensor(c.support_height_min_m, dtype=torch.float32, device=device).double()
        height_max = torch.tensor(c.support_height_max_m, dtype=torch.float32, device=device).double()
        depth = (baseline - height_min).clamp(min=0, max=c.compression_depth_max_m)
        low = baseline - depth
        nominal_t, g = c.nominal_extension_duration_s, c.gravity_m_s2
        root = torch.sqrt((g * nominal_t)**2 + 8 * g * (rise + depth))
        velocity = 4 * g * (rise + depth) / (root + g * nominal_t)
        travel = velocity * nominal_t / 2
        available = height_max - low
        limited = travel > available
        travel = torch.minimum(travel, available)
        limited_velocity = (2 * g * (rise + depth - travel)).clamp_min(0).sqrt()
        velocity = torch.where(limited, limited_velocity, velocity)
        extension_t = torch.where(limited, 2 * travel / velocity.clamp_min(1e-12), nominal_t)
        launch = c.compression_duration_s + extension_t
        takeoff_height = low + travel
        return_flight = (velocity + (velocity.square() + 2 * g * (takeoff_height - baseline)).clamp_min(0).sqrt()) / g
        return_time = launch + return_flight
        u = (elapsed_s / c.compression_duration_s).clamp(0, 1)
        compressed = baseline - depth * u.pow(3) * (10 - 15 * u + 6 * u.square())
        compress_v = -depth * 30 * u.square() * (1 - u).square() / c.compression_duration_s
        extension_u = ((elapsed_s - c.compression_duration_s) / extension_t).clamp(0, 1)
        extended = low + velocity * extension_t * (extension_u.pow(3) - .5 * extension_u.pow(4))
        extended_v = velocity * (3 * extension_u.square() - 2 * extension_u.pow(3))
        flight_t = (elapsed_s - launch).clamp_min(0)
        ballistic = takeoff_height + velocity * flight_t - .5 * g * flight_t.square()
        ballistic_v = velocity - g * flight_t
        is_compress = elapsed_s < c.compression_duration_s
        is_extend = ~is_compress & (elapsed_s < launch)
        is_flight = ~is_compress & ~is_extend & (elapsed_s < return_time)
        height = torch.where(is_compress, compressed, torch.where(is_extend, extended,
                                                                  torch.where(is_flight, ballistic, baseline)))
        speed = torch.where(is_compress, compress_v, torch.where(is_extend, extended_v,
                                                               torch.where(is_flight, ballistic_v, 0)))
        stage = torch.where(is_compress, int(ReferenceStage.COMPRESSION),
                            torch.where(is_extend, int(ReferenceStage.EXTENSION),
                                        torch.where(is_flight, int(ReferenceStage.BALLISTIC), int(ReferenceStage.HOLD))))
        world = height + ground_anchor_world_z_m.double()
        apex = baseline + rise + ground_anchor_world_z_m.double()
        held = latched_height_target_m + ground_anchor_world_z_m
        return JumpReferenceSample(torch.where(active, world.float(), held), torch.where(active, speed.float(), 0),
                                    torch.where(active, apex.float(), held),
                                    torch.where(active, stage, int(ReferenceStage.HOLD)),
                                    torch.where(active, depth.float(), 0), torch.where(active, extension_t, 0),
                                    torch.where(active, velocity, 0), torch.where(active, launch, 0),
                                    torch.where(active, return_time, 0), limited & active)


@torch.no_grad()
def reward_phases(commands: CommandFeatures, physical_phase):
    """Physical phases may gate reward, never the deployable mode or action."""
    if not isinstance(commands, CommandFeatures):
        raise TypeError('reward phase bridge requires explicit deployable commands')
    mode = commands.mode
    if (not isinstance(physical_phase, torch.Tensor) or physical_phase.shape != mode.shape
            or physical_phase.dtype != torch.long or physical_phase.device != mode.device
            or ((physical_phase < int(SkillPhase.IDLE)) | (physical_phase > int(SkillPhase.SETTLING))).any()):
        raise ValueError('physical skill phase differs from command rows')
    result = torch.full_like(mode, int(RewardPhase.SUPPORT))
    result = torch.where(mode == int(CommandMode.RECOVERY), int(RewardPhase.RECOVERY), result)
    jump_phase = torch.where(physical_phase <= int(SkillPhase.PREPARING), int(RewardPhase.TAKEOFF),
                              torch.where(physical_phase == int(SkillPhase.FLIGHT), int(RewardPhase.FLIGHT),
                                          torch.where(physical_phase == int(SkillPhase.LANDING),
                                                      int(RewardPhase.LANDING), int(RewardPhase.SETTLING))))
    return torch.where(mode == int(CommandMode.JUMP), jump_phase, result)


@torch.no_grad()
def reward_reference(commands: CommandFeatures, curve: JumpReferenceSample, physical_reward_phase, current_ground_world_z_m):
    """Reward-only landing support follows current ground, not an old anchor."""
    if not isinstance(commands, CommandFeatures) or not isinstance(curve, JumpReferenceSample):
        raise TypeError('reward reference requires explicit deployable commands and a visible-clock curve')
    count, device = commands.mode.numel(), commands.mode.device
    if (not isinstance(physical_reward_phase, torch.Tensor) or physical_reward_phase.shape != (count,)
            or physical_reward_phase.dtype != torch.long or physical_reward_phase.device != device
            or not isinstance(current_ground_world_z_m, torch.Tensor) or current_ground_world_z_m.shape != (count,)
            or current_ground_world_z_m.dtype != torch.float32 or current_ground_world_z_m.device != device
            or not torch.isfinite(current_ground_world_z_m).all()):
        raise ValueError('physical reward phase and ground must match the command rows')
    for name in ('height_world_z_m', 'velocity_world_z_m_s'):
        value = getattr(curve, name)
        if (not isinstance(value, torch.Tensor) or value.shape != (count,) or value.dtype != torch.float32
                or value.device != device or not torch.isfinite(value).all()):
            raise ValueError('visible-clock jump reference differs from command rows')
    jump = commands.mode == int(CommandMode.JUMP)
    landing = jump & ((physical_reward_phase == int(RewardPhase.LANDING))
                      | (physical_reward_phase == int(RewardPhase.SETTLING)))
    height = torch.where(landing, current_ground_world_z_m + commands.admitted_body_command[:, 3], curve.height_world_z_m)
    speed = torch.where(landing, 0, curve.velocity_world_z_m_s)
    return RewardReference(commands.admitted_body_command.detach().clone(), commands.mode.detach().clone(),
                            physical_reward_phase.detach().clone(), height.clone(), speed.clone(),
                            torch.zeros(count, 2, dtype=torch.float32, device=device))

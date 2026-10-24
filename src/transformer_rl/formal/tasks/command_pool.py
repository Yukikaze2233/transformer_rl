"""Within-episode command segments drawn from the SCUT V14 command pool.

A segment is one admitted-command target held for U(5, 15) s; the next starts
from the current admitted command, so standing/moving and direction changes
are trained. Modes, ranges, unlock updates and proportions follow SCUT
(Flat-v0 unlock order; Rough-v0 shares with the gimbal-translation share given
to normal motion, which this robot does not have). Course rows keep SCUT's
terrain bindings: non-flat height floor +2 cm, stair approach band.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math

import torch

from .curriculum import TaskDomain, TaskFamily
from .geometry import TerrainKind
from .trajectory import TrajectoryKind


class SegmentKind(IntEnum):
    STAND = 0
    NORMAL = 1
    SPIN_LOW = 2
    SPIN_MID = 3
    DASH = 4


SPECIAL_KINDS = (SegmentKind.SPIN_LOW, SegmentKind.SPIN_MID, SegmentKind.DASH)
# SCUT Rough-v0 effective shares: standing 13.6%, spin_low 9%, spin_mid 9%,
# dash 18%, normal 32.4% + gimbal translation 18%.
SEGMENT_WEIGHTS = (.136, .504, .09, .09, .18)
APPROACH_TERRAINS = (TerrainKind.STEP_UP, TerrainKind.STEP_DOWN, TerrainKind.STAIRS, TerrainKind.TWO_PLATFORMS)


@dataclass(frozen=True)
class CommandPoolConfig:
    """Frozen SCUT pool contract; times in s, speeds in m/s, rates in rad/s."""

    segment_s: tuple[float, float] = (5., 15.)
    special_after_s: float = 5.
    heading_fraction: float = .5
    normal_yaw_rad_s: float = 2 * math.pi
    spin_low_rad_s: tuple[float, float] = (2 * math.pi, 3.25 * math.pi)
    spin_mid_rad_s: tuple[float, float] = (3.25 * math.pi, 4.5 * math.pi)
    spin_forward_m_s: float = .1
    dash_forward_m_s: tuple[float, float] = (2., 3.)
    dash_yaw_rad_s: float = 2 * math.pi
    spin_low_update: int = 3000
    spin_mid_update: int = 4000
    dash_update: int = 2000
    # SCUT terrain bindings: non-flat heights start 2 cm higher; the stair
    # profile uses |vx| in [0.5, 2.0] m/s and non-heading yaw in [-1, 1] rad/s.
    terrain_height_floor_m: float = .02
    approach_forward_m_s: tuple[float, float] = (.5, 2.)
    approach_yaw_rad_s: float = 1.
    # SCUT ground resets get vx, yaw in U(-1, 1) for 1.5 s before normal commands.
    handoff_s: float = 1.5
    handoff_forward_m_s: float = 1.
    handoff_yaw_rad_s: float = 1.
    segment_weights: tuple[float, ...] = field(default=SEGMENT_WEIGHTS)

    def __post_init__(self):
        for name in ('segment_s', 'spin_low_rad_s', 'spin_mid_rad_s', 'dash_forward_m_s', 'approach_forward_m_s'):
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or not 0 < value[0] < value[1]):
                raise ValueError(f'{name} requires two ordered positive finite bounds')
            object.__setattr__(self, name, tuple(float(v) for v in value))
        scalars = ('special_after_s', 'normal_yaw_rad_s', 'spin_forward_m_s', 'dash_yaw_rad_s',
                   'terrain_height_floor_m', 'approach_yaw_rad_s', 'handoff_s', 'handoff_forward_m_s',
                   'handoff_yaw_rad_s')
        weights = tuple(float(v) for v in self.segment_weights)
        if (any(type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name))
                or getattr(self, name) < 0 for name in scalars)
                or type(self.heading_fraction) not in (int, float) or not 0 <= self.heading_fraction <= 1
                or any(type(getattr(self, name)) is not int or getattr(self, name) < 0
                       for name in ('spin_low_update', 'spin_mid_update', 'dash_update'))
                or len(weights) != len(SegmentKind) or any(not math.isfinite(v) or v < 0 for v in weights)
                or abs(sum(weights) - 1) > 1e-9):
            raise ValueError('command pool bounds or segment weights are invalid')
        object.__setattr__(self, 'segment_weights', weights)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'enabled'} or value['enabled'] is not True:
            raise ValueError('command pool configuration is {"enabled": true}; parameters are frozen in code')
        return cls()


@dataclass(frozen=True)
class CommandContext:
    """Curriculum inputs read when a segment is drawn."""

    domain: TaskDomain
    update: int
    request_motion_scale: float = 1.


@dataclass(frozen=True)
class SegmentDraw:
    kind: torch.Tensor                # SegmentKind
    command: torch.Tensor             # [n,4] body vx, vy=0, yaw rate, height
    duration_s: torch.Tensor          # segment length
    transition_kind: torch.Tensor     # TrajectoryKind; HOLD steps through the slew limiter
    heading_enabled: torch.Tensor     # yaw command comes from the heading loop
    heading_offset_rad: torch.Tensor  # target heading relative to current yaw


class CommandPool:
    """Sample segment targets; all randomness comes from the caller's generator."""

    def __init__(self, config: CommandPoolConfig = CommandPoolConfig()):
        if not isinstance(config, CommandPoolConfig):
            raise TypeError('command pool requires a parsed CommandPoolConfig')
        self.config = config
        self._weights = torch.tensor(config.segment_weights, dtype=torch.float32)

    @staticmethod
    def approach(terrain_kind):
        rows = torch.zeros_like(terrain_kind, dtype=torch.bool)
        for kind in APPROACH_TERRAINS:
            rows |= terrain_kind == int(kind)
        return rows

    @torch.no_grad()
    def sample(self, family, terrain_kind, elapsed_s, approach_sign, context: CommandContext, generator):
        if not isinstance(context, CommandContext) or not isinstance(context.domain, TaskDomain):
            raise TypeError('segment sampling requires the current curriculum context')
        count, device = family.shape[0], family.device
        if (terrain_kind.shape != (count,) or elapsed_s.shape != (count,) or approach_sign.shape != (count,)
                or family.dtype != torch.long or terrain_kind.dtype != torch.long):
            raise ValueError('segment rows require [n] family, terrain, clock and approach sign')
        scale = float(context.request_motion_scale)
        if not 0 <= scale <= 1:
            raise ValueError('request motion scale must be in [0,1]')
        c, d = self.config, context.domain
        uniform = lambda low, high: low + (high - low) * torch.rand(count, generator=generator, device=device)
        symmetric = lambda bound: uniform(-bound, bound)
        domain_forward = min(-d.forward_m_s[0], d.forward_m_s[1])
        domain_yaw = min(-d.yaw_rad_s[0], d.yaw_rad_s[1])
        approach = self.approach(terrain_kind)
        weights = self._weights.to(device).expand(count, -1).clone()
        # SCUT: special modes only after 5 s of episode time and from their
        # unlock update; stair courses disable them. Blocked shares go to normal.
        unlocked = {SegmentKind.SPIN_LOW: context.update >= c.spin_low_update and domain_yaw > c.spin_low_rad_s[0],
                    SegmentKind.SPIN_MID: context.update >= c.spin_mid_update and domain_yaw > c.spin_mid_rad_s[0],
                    SegmentKind.DASH: context.update >= c.dash_update and domain_forward >= c.dash_forward_m_s[0]}
        for kind in SPECIAL_KINDS:
            blocked = (elapsed_s < c.special_after_s) | approach | (not unlocked[kind])
            weights[:, int(SegmentKind.NORMAL)] += torch.where(blocked, weights[:, int(kind)], 0.)
            weights[:, int(kind)] = torch.where(blocked, 0., weights[:, int(kind)])
        kind = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
        sign = torch.where(torch.rand(count, generator=generator, device=device) < .5, -1., 1.)
        floor = torch.where(terrain_kind != int(TerrainKind.FLAT), d.height_m[0] + c.terrain_height_floor_m,
                            d.height_m[0]).clamp_max(d.height_m[1])
        height = floor + (d.height_m[1] - floor) * torch.rand(count, generator=generator, device=device)
        normal_v = symmetric(domain_forward)
        normal_w = symmetric(min(c.normal_yaw_rad_s, domain_yaw))
        spin_low = sign * uniform(c.spin_low_rad_s[0], min(c.spin_low_rad_s[1], max(domain_yaw, c.spin_low_rad_s[0])))
        spin_mid = sign * uniform(c.spin_mid_rad_s[0], min(c.spin_mid_rad_s[1], max(domain_yaw, c.spin_mid_rad_s[0])))
        creep = symmetric(c.spin_forward_m_s)
        dash = sign * uniform(c.dash_forward_m_s[0], min(c.dash_forward_m_s[1], max(domain_forward, c.dash_forward_m_s[0])))
        dash_yaw = symmetric(min(c.dash_yaw_rad_s, domain_yaw))
        # The course spawn faces the obstacle along the episode's command sign.
        approach_v = approach_sign * uniform(*c.approach_forward_m_s).clamp_max(domain_forward)
        approach_w = symmetric(c.approach_yaw_rad_s)

        K = SegmentKind
        normal = kind == K.NORMAL
        spin = (kind == K.SPIN_LOW) | (kind == K.SPIN_MID)
        vx = torch.where(normal, torch.where(approach, approach_v, normal_v), torch.zeros_like(normal_v))
        vx = torch.where(spin, creep, torch.where(kind == K.DASH, dash, vx))
        wz = torch.where(normal, torch.where(approach, approach_w, normal_w), torch.zeros_like(normal_w))
        wz = torch.where(kind == K.SPIN_LOW, spin_low, torch.where(kind == K.SPIN_MID, spin_mid, wz))
        wz = torch.where(kind == K.DASH, dash_yaw, wz)
        request = family == int(TaskFamily.REQUEST)
        vx, wz = torch.where(request, vx * scale, vx), torch.where(request, wz * scale, wz)
        command = torch.stack((vx, torch.zeros_like(vx), wz, height), -1)
        # SCUT heading envs: half of normal (non-special) rows hold a sampled
        # world heading; stair approaches keep the course heading instead.
        heading = (normal & (torch.rand(count, generator=generator, device=device) < c.heading_fraction)) | (
            approach & (kind != K.STAND))
        offset = torch.where(normal & ~approach, symmetric(math.pi), torch.zeros_like(vx))
        return SegmentDraw(kind, command, uniform(*c.segment_s),
                           torch.full_like(kind, int(TrajectoryKind.HOLD)), heading, offset)

    @torch.no_grad()
    def handoff(self, count, generator, device):
        """SCUT ground-reset command after a recovery: vx, yaw in U(-1, 1)."""
        c = self.config
        uniform = lambda bound: (torch.rand(count, generator=generator, device=device) * 2 - 1) * bound
        return uniform(c.handoff_forward_m_s), uniform(c.handoff_yaw_rad_s)

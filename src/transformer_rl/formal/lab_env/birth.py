"""SCUT V14 birth randomization translated to this asset's mechanical travel.

SCUT draws each virtual leg (length L, angle psi) uniformly and independently
per side at every reset. Here L is uniform over the knee stops and, given L,
psi is uniform over the range the policy decoder can hold (leg motor targets
inside the commandable output domain), so every birth posture is one this
robot can actually hold. psi is measured from straight down, positive toward
+x; a common hip offset rotates the rigid leg, so psi = alpha(knee) - offset.

- Landing birth (SCUT ordinary reset): released from the top of the travel
  (the deepest posture just touches the ground) with base roll/pitch and
  wheel-speed perturbations; the policy acts from the first step.
- Ground birth (SCUT ground reset, 0.3 + 0.2): the posture stands level on
  the ground with zero motor torque for 0.2 s, so physics settles or collapses
  it before the same policy stands up under the +/-1 handoff command.

Postures are solved once on CPU into a certified library and indexed at
reset. Leg joint velocities are not drawn: a closed chain admits no
independent per-joint velocity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ..control.actuation import OutputFeedbackController
from ..simulation.ground import WheelCollisionCentres
from ..tasks.geometry import TerrainKind
from .recovery import LabRecoveryReset

# Ground births need a flat spawn patch under the low body: roughness stays
# below the belly clearance of the lowest stance; slopes and step courses
# would put the level body into the terrain.
GROUND_TERRAINS = (TerrainKind.FLAT, TerrainKind.ROUGH, TerrainKind.MATERIAL)


@dataclass(frozen=True)
class BirthRandomizationConfig:
    """SCUT V14 reset distribution; angles in rad, speeds in rad/s, lengths in m."""

    enabled: bool = False
    ground_probability: float = .5        # SCUT ground modes: positive .3 + negative .2
    ground_release_s: float = .2          # SCUT zero_torque_time_s
    roll_pitch_rad: float = .15           # SCUT reset_base roll/pitch on ordinary births
    wheel_velocity_rad_s: float = 50.     # SCUT wheel_joint_vel_range
    # Tilting the deepest stance by 0.15 rad lowers one wheel by about 3.3 cm.
    tilt_clearance_m: float = .035
    library_size: int = 2048

    def __post_init__(self):
        if (type(self.enabled) is not bool or type(self.library_size) is not int or self.library_size < 16
                or type(self.ground_probability) not in (int, float) or not 0 <= self.ground_probability <= 1
                or any(type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name))
                       or getattr(self, name) < 0
                       for name in ('ground_release_s', 'roll_pitch_rad', 'wheel_velocity_rad_s',
                                    'tilt_clearance_m'))):
            raise ValueError('birth randomization bounds are invalid')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'enabled'} or value['enabled'] is not True:
            raise ValueError('birth randomization configuration is {"enabled": true}; parameters are frozen in code')
        return cls(enabled=True)


class BirthPostureLibrary:
    """Certified level-body leg postures inside the commandable output domain."""

    KNEE_MARGIN_RAD = math.radians(.5)
    CANDIDATE_FACTOR = 2
    # rad and m: rigid-leg polar map versus forward kinematics. The hip axes are
    # tilted ~0.3 deg out of the sagittal normal, which leaves ~1e-4 rad.
    MAP_TOLERANCE = 1e-3

    def __init__(self, recovery: LabRecoveryReset, wheels: WheelCollisionCentres, *, size, seed,
                 pair_domain=None):
        if (not isinstance(recovery, LabRecoveryReset) or not isinstance(wheels, WheelCollisionCentres)
                or type(size) is not int or size < 16):
            raise TypeError('birth posture library requires the selected recovery geometry and wheel colliders')
        kinematics = recovery.kinematics
        binding = kinematics.binding
        generator = torch.Generator().manual_seed(int(seed))
        low, high = (float(v) for v in kinematics.geometry.inner_limits_rad)
        knees = torch.linspace(low + self.KNEE_MARGIN_RAD, high - self.KNEE_MARGIN_RAD, 129, dtype=torch.float64)
        # Per side: the asset's hip frames are not exact mirror images.
        alpha, length, drive = self._knee_map(kinematics, knees)
        if not bool((length[1:] > length[:-1]).all()):
            raise RuntimeError('virtual leg length must increase monotonically over the knee travel')
        offset_low, offset_high = self._offset_domain(binding, knees, drive, pair_domain)

        count = self.CANDIDATE_FACTOR * size
        unit = torch.rand(count, 2, 2, generator=generator, dtype=torch.float64)
        # A common length range keeps both sides on the same SCUT distribution.
        shortest, longest = float(length[0].amax()), float(length[-1].amin())
        leg_length = shortest + (longest - shortest) * unit[..., 0]
        side = lambda grid, values, query: torch.stack(
            [self._interpolate(grid[:, s] if grid.ndim == 2 else grid, values[:, s], query[:, s])
             for s in range(2)], -1)
        knee = side(length, knees[:, None].expand(-1, 2), leg_length)
        low_o, high_o = side(knees, offset_low, knee), side(knees, offset_high, knee)
        offset = low_o + (high_o - low_o) * unit[..., 1]
        psi = side(knees, alpha, knee) - offset

        seeds = kinematics.nominal_tree_position[None].double().repeat(count, 1)
        hips = seeds[:, self._hip_columns(kinematics)] + torch.stack((offset[:, 0], -offset[:, 1]), -1)
        solved = kinematics.continue_posture(
            seeds, hips, knee, maximum_segment_angle_rad=math.pi / 24, maximum_segments=32,
            closure_tolerance_m=1e-6, maximum_iterations=80).closure
        tree = solved.tree_position
        controller = OutputFeedbackController(binding.nominal_policy_position_rad, count,
                                              pair_domain=pair_domain, device='cpu')
        policy = torch.zeros(count, 6, dtype=torch.float32)
        policy[:, :4] = tree[:, list(kinematics.policy_tree_indices[:4])].float()
        admitted = controller.reset_hold_action(policy, position_tolerance_rad=1e-6).admitted
        frame = kinematics.forward(tree)
        achieved_psi, achieved_length = self._virtual_legs(kinematics, frame)
        mapped = (((achieved_psi - psi).abs() < self.MAP_TOLERANCE)
                  & ((achieved_length - leg_length).abs() < self.MAP_TOLERANCE)).all(-1)
        level = torch.zeros(count, 4, dtype=torch.float64)
        level[:, 3] = 1
        placement = recovery.geometry.place(
            tree, level, torch.zeros(count, 2, dtype=torch.float64),
            recovery.flat_patch(torch.arange(count) % recovery.num_envs, torch.zeros(count, 3, dtype=torch.float64)),
            clearance_m=recovery.config.clearance_m, loop_tolerance_m=recovery.config.loop_tolerance_m,
            wheel_support_tolerance_m=recovery.config.wheel_support_tolerance_m)
        checks = {'converged': solved.converged, 'within_native_limits': solved.within_native_limits,
                  'command_domain': admitted, 'polar_map': mapped, 'placement': placement.geometry_ready}
        valid = torch.stack(tuple(checks.values())).all(0)
        rejected = {name: int((~value).sum()) for name, value in checks.items()}
        if int(valid.sum()) < size:
            raise RuntimeError(f'birth library certified {int(valid.sum())}/{count} candidates, needs {size}; '
                               f'rejected {rejected}')
        keep = valid.nonzero(as_tuple=False).flatten()[:size]
        native_order = [binding.tree_joint_names.index(name) for name in binding.native_joint_names]
        self.joint_position = tree[keep][:, native_order].float()
        # Ground births stand on the ground at their own posture height.
        self.ground_root_height_m = placement.root_pose_xyzw[keep, 2].float()
        # Landing births start from the top of the travel: the deepest posture touches.
        self.landing_root_height_m = float(self.ground_root_height_m.max())
        wheel_index = list(binding.wheel_body_indices)
        local = torch.tensor(wheels.local_xyz_m, dtype=torch.float64)
        self.wheel_centre_root_m = (frame.body_position_m[keep][:, wheel_index] + torch.einsum(
            'nkij,kj->nki', frame.body_rotation[keep][:, wheel_index], local)).float()
        self.leg_length_m, self.leg_angle_rad = leg_length[keep].float(), psi[keep].float()
        self.receipt = {
            'format': 'birth_posture_library_v2', 'requested': size, 'candidates': count,
            'certified': int(valid.sum()), 'rejected': rejected,
            'distribution': 'per side: L uniform over knee stops; psi uniform over the commandable range given L',
            'knee_travel_deg': [math.degrees(low), math.degrees(high)],
            'leg_length_m': [float(self.leg_length_m.min()), float(self.leg_length_m.max())],
            'leg_angle_deg': [math.degrees(float(self.leg_angle_rad.min())),
                              math.degrees(float(self.leg_angle_rad.max()))],
            'ground_root_height_m': [float(self.ground_root_height_m.min()), self.landing_root_height_m],
            'landing_root_height_m': self.landing_root_height_m,
        }

    @staticmethod
    def _hip_columns(kinematics):
        return [kinematics.policy_tree_indices[0], kinematics.policy_tree_indices[2]]

    @staticmethod
    def _virtual_legs(kinematics, frame):
        """[n,2] psi and L of the wheel centre about each hip axis."""
        names, joints = kinematics.binding.native_body_names, kinematics.binding.tree_joint_names
        wheel = (frame.body_position_m[:, [names.index('L_link3'), names.index('R_link3')]]
                 - frame.joint_origin_world_m[:, [joints.index('L_joint1'), joints.index('R_joint1')]])
        return torch.atan2(wheel[..., 0], -wheel[..., 2]), wheel[..., [0, 2]].norm(dim=-1)

    @classmethod
    def _knee_map(cls, kinematics, knees):
        """Zero-offset legs per knee and side: alpha, L and knee-drive motor position, each [k,2]."""
        seeds = kinematics.nominal_tree_position[None].double().repeat(len(knees), 1)
        solved = kinematics.continue_posture(
            seeds, seeds[:, cls._hip_columns(kinematics)].clone(), knees[:, None].expand(-1, 2).clone(),
            maximum_segment_angle_rad=math.pi / 24, maximum_segments=32,
            closure_tolerance_m=1e-6, maximum_iterations=80).closure
        if not bool(solved.converged.all()):
            raise RuntimeError('birth knee map did not converge over the knee travel')
        psi, length = cls._virtual_legs(kinematics, kinematics.forward(solved.tree_position))
        drive = solved.tree_position[:, [kinematics.policy_tree_indices[1], kinematics.policy_tree_indices[3]]]
        return psi, length, drive

    @staticmethod
    def _offset_domain(binding, knees, drive, pair_domain):
        """[k,2] admitted common-offset interval per knee and side.

        A common offset turns both coaxial leg motors of a side together; the
        decoder's own hold admission defines the interval, and the final full
        solve re-checks every posture.
        """
        grid = torch.arange(-math.pi, math.pi + 1e-9, math.radians(.1), dtype=torch.float64)
        nominal = torch.tensor(binding.nominal_policy_position_rad, dtype=torch.float64)
        rows = len(knees) * len(grid)
        controller = OutputFeedbackController(binding.nominal_policy_position_rad, rows,
                                              pair_domain=pair_domain, device='cpu')
        bounds = []
        for side, sign in ((0, 1.), (1, -1.)):
            q = nominal.repeat(rows, 1)
            offset = grid.repeat(len(knees))
            q[:, 2 * side] = nominal[2 * side] + sign * offset
            q[:, 2 * side + 1] = drive[:, side].repeat_interleave(len(grid)) + sign * offset
            admitted = controller.reset_hold_action(q.float(), position_tolerance_rad=1e-6).admitted
            admitted = admitted.view(len(knees), len(grid))
            if not bool(admitted.any(-1).all()):
                raise RuntimeError('a knee angle inside its stops has no commandable leg posture')
            first = admitted.float().argmax(-1)
            last = len(grid) - 1 - admitted.flip(-1).float().argmax(-1)
            span = torch.arange(len(grid))[None]
            if not bool(admitted.eq((span >= first[:, None]) & (span <= last[:, None])).all()):
                raise RuntimeError('commandable leg offsets are not one interval per knee')
            bounds.append((grid[first], grid[last]))
        return torch.stack((bounds[0][0], bounds[1][0]), -1), torch.stack((bounds[0][1], bounds[1][1]), -1)

    @staticmethod
    def _interpolate(grid, values, query):
        """Piecewise-linear values(query) over an increasing grid, clamped at its ends."""
        query = query.clamp(grid[0], grid[-1])
        high = torch.searchsorted(grid.contiguous(), query.contiguous()).clamp(1, len(grid) - 1)
        low = high - 1
        weight = (query - grid[low]) / (grid[high] - grid[low])
        return values[low] + weight * (values[high] - values[low])


class BirthRandomization:
    """Per-reset draws; owns its RNG and the certified birth posture library."""

    def __init__(self, recovery: LabRecoveryReset, config: BirthRandomizationConfig,
                 wheels: WheelCollisionCentres, *, seed, device, pair_domain=None):
        if not isinstance(config, BirthRandomizationConfig):
            raise TypeError('birth randomization requires a parsed BirthRandomizationConfig')
        self.config, self.device = config, torch.device(device)
        self.library = BirthPostureLibrary(recovery, wheels, size=config.library_size, seed=seed,
                                           pair_domain=pair_domain)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        self.joint_position = self.library.joint_position.to(self.device)
        self.ground_root_height_m = self.library.ground_root_height_m.to(self.device)
        self.landing_root_height_m = self.library.landing_root_height_m
        self.wheel_centre_root_m = self.library.wheel_centre_root_m.to(self.device)
        # landing births, ground births, ground births stood up, ground births failed first
        self.counts = torch.zeros(4, dtype=torch.long, device=self.device)
        # Ground births failed first, per FailureReason bit (one row may set several).
        self.failure_bits = torch.zeros(6, dtype=torch.long, device=self.device)
        self.since_birth_s = torch.full((recovery.num_envs,), float('inf'), device=self.device)
        self.standing_s = torch.zeros(recovery.num_envs, device=self.device)

    def _uniform(self, shape, bound):
        return (torch.rand(shape, generator=self.generator, device=self.device) * 2 - 1) * bound

    @staticmethod
    def ground_eligible(terrain_kind):
        rows = torch.zeros_like(terrain_kind, dtype=torch.bool)
        for kind in GROUND_TERRAINS:
            rows |= terrain_kind == int(kind)
        return rows

    @torch.no_grad()
    def draw(self, ordinary, ground_eligible):
        """[n] ground mask, [n] library index, [n,2] roll/pitch, [n,2] wheel speed."""
        count, c = ordinary.numel(), self.config
        ground = ordinary & ground_eligible & (
            torch.rand(count, generator=self.generator, device=self.device) < c.ground_probability)
        index = torch.randint(len(self.ground_root_height_m), (count,), generator=self.generator, device=self.device)
        tilt = self._uniform((count, 2), c.roll_pitch_rad)
        wheels = self._uniform((count, 2), c.wheel_velocity_rad_s)
        self.counts[0] += (ordinary & ~ground).sum()
        return ground, index, tilt, wheels

    @staticmethod
    def tilt_quaternion(roll_pitch):
        """xyzw quaternion of R_y(pitch) R_x(roll); yaw stays at the course heading."""
        half = roll_pitch / 2
        cr, sr, cp, sp = half[:, 0].cos(), half[:, 0].sin(), half[:, 1].cos(), half[:, 1].sin()
        return torch.stack((sr * cp, cr * sp, -sr * sp, cr * cp), -1)

    def release_frames(self, policy_dt_s):
        return max(1, round(self.config.ground_release_s / policy_dt_s))

    @torch.no_grad()
    def begin(self, rows):
        """Start stand-up telemetry for ground-birth ``rows``."""
        self.counts[1] += rows.numel()
        self.since_birth_s[rows] = 0
        self.standing_s[rows] = 0

    @torch.no_grad()
    def observe(self, dt_s, *, height_m, height_command_m, tilt_rad, wheel_load_n, terminated, failure_reasons=None):
        """Telemetry only: stood up = within 5 cm of the height command, tilt < 20 deg, both
        wheels loaded, held for 0.5 s."""
        active = torch.isfinite(self.since_birth_s)
        self.since_birth_s.add_(dt_s * active)
        upright = (active & ((height_m - height_command_m).abs() < .05) & (tilt_rad < math.radians(20))
                   & (wheel_load_n > 5).all(-1))
        self.standing_s.copy_(torch.where(upright, self.standing_s + dt_s, 0))
        stood = active & (self.standing_s >= .5)
        failed = active & ~stood & terminated
        self.counts[2] += stood.sum()
        self.counts[3] += failed.sum()
        if failure_reasons is not None:
            bits = 1 << torch.arange(len(self.failure_bits), device=self.device)
            self.failure_bits += (failed[:, None] & (failure_reasons[:, None] & bits).ne(0)).sum(0)
        self.since_birth_s.masked_fill_(stood | failed, float('inf'))

    def metrics(self):
        landing, ground, stood, failed = self.counts.tolist()
        return {'landing_births': landing, 'ground_births': ground, 'ground_stood_up': stood,
                'ground_failed_before_standing': failed,
                'ground_failure_bits': dict(zip(('low_support_height', 'excessive_tilt', 'forbidden_body_contact',
                                                 'passive_knee_overrun', 'knee_pivot_residual', 'pressure_stroke'),
                                                self.failure_bits.tolist())),
                'stand_up_rule': '|h-h_cmd|<0.05 m, tilt<20 deg, both wheels >5 N for 0.5 s (telemetry only)',
                'library': self.library.receipt}

    def state_dict(self):
        return {'format': 'lab_birth_randomization_v2', 'rng': self.generator.get_state().clone(),
                'configuration': asdict(self.config)}

    def restore_after_physical_reset(self, state):
        if (not isinstance(state, dict) or set(state) != {'format', 'rng', 'configuration'}
                or state['format'] != 'lab_birth_randomization_v2'
                or state['configuration'] != asdict(self.config)):
            raise ValueError('birth randomization checkpoint contract differs')
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].cpu())
        self.generator = generator
        self.since_birth_s.fill_(float('inf'))
        self.standing_s.zero_()

"""Private named physical measurements for the 35D/81D formal candidate.

No simulator is imported. Native joint/body order, quaternion convention and
velocity reference are explicit; deployment sensors never accept privileged
state. This is a new candidate binding, not a legacy checkpoint ABI claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from ._control_sources import ControlSource, canonical_coaxial_displacement, validate_issued_control
from ._structured_collection import ACTION_ORDER, PhysicalLabels


class CommandMode(IntEnum):
    NORMAL = 0
    STAIR = 1
    SLOPE = 2
    RECOVERY = 3
    JUMP = 4


@dataclass(frozen=True)
class MechanicalBinding:
    native_joint_names: tuple[str, ...]
    native_body_names: tuple[str, ...]
    tree_joint_names: tuple[str, ...]
    policy_joint_indices: tuple[int, ...]
    tree_joint_indices: tuple[int, ...]
    wheel_body_indices: tuple[int, int]
    wheel_tree_indices: tuple[int, int]
    nominal_policy_position_rad: tuple[float, ...]

    @classmethod
    def from_manifest(cls, manifest, native_joint_names, native_body_names):
        joints, bodies = tuple(native_joint_names), tuple(native_body_names)
        tree = tuple(manifest['tree_joint_names'])
        declared_bodies = tuple(manifest['rigid_body_names'])
        if (manifest.get('control_frame') != 'Xforward_Yleft_Zup'
                or len(tree) != 18 or len(set(tree)) != 18
                or len(declared_bodies) != 19 or len(set(declared_bodies)) != 19
                or len(joints) != 18 or len(set(joints)) != 18 or set(joints) != set(tree)
                or len(bodies) != 19 or len(set(bodies)) != 19 or set(bodies) != set(declared_bodies)
                or len(manifest['control_joint_names']) != 6
                or set(manifest['control_joint_names']) != set(ACTION_ORDER)
                or not set(ACTION_ORDER).issubset(tree)
                or not {'base_link', 'L_link3', 'R_link3'}.issubset(bodies)):
            raise ValueError('mechanical binding requires the complete named 18-joint/19-body topology')
        nominal = tuple(manifest['nominal_joint_pos'][name] for name in ACTION_ORDER)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in nominal):
            raise ValueError('all six nominal active positions require finite asset values')
        return cls(joints, bodies, tree, tuple(joints.index(name) for name in ACTION_ORDER),
                   tuple(joints.index(name) for name in tree),
                   (bodies.index('L_link3'), bodies.index('R_link3')),
                   (tree.index('L_joint3'), tree.index('R_joint3')), nominal)


@dataclass(frozen=True)
class NativePhysicalSample:
    joint_position_rad: torch.Tensor
    joint_velocity_rad_s: torch.Tensor
    root_link_position_world_m: torch.Tensor
    root_link_orientation_xyzw: torch.Tensor
    root_com_position_world_m: torch.Tensor
    root_com_velocity_world_m_s: torch.Tensor
    angular_velocity_world_rad_s: torch.Tensor
    terrain_contact_force_world_n: torch.Tensor
    ground_at_root_world_z_m: torch.Tensor
    ground_at_wheel_world_z_m: torch.Tensor
    environment_origin_world_m: torch.Tensor
    wheel_friction: torch.Tensor
    timestamp_s: torch.Tensor


@dataclass(frozen=True)
class RigidBodyMotion:
    """Owned world-SI COM motion in actual native body order, not actor data."""

    link_pose_world_xyzw: torch.Tensor
    com_position_world_m: torch.Tensor
    com_velocity_world_m_s: torch.Tensor
    angular_velocity_world_rad_s: torch.Tensor
    timestamp_s: torch.Tensor
    body_names: tuple[str, ...]
    physical_tick: int
    model_epoch: int


@dataclass(frozen=True)
class MeasuredState:
    """Owned SI state at the base_link origin; all six-axis fields use P order."""

    policy_position_rad: torch.Tensor
    policy_velocity_rad_s: torch.Tensor
    tree_position_rad: torch.Tensor
    tree_velocity_rad_s: torch.Tensor
    velocity_origin_body_m_s: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor
    gravity_body: torch.Tensor
    height_above_ground_m: torch.Tensor
    wheel_ground_force_n: torch.Tensor
    wheel_friction: torch.Tensor
    wheel_support_above_environment_m: torch.Tensor
    timestamp_s: torch.Tensor


@dataclass(frozen=True)
class SensorSample:
    """Deployable encoder/IMU values; sample times are q,dq,gyro,gravity."""

    policy_position_rad: torch.Tensor
    policy_velocity_rad_s: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor
    gravity_body: torch.Tensor
    sampled_at_s: torch.Tensor


@dataclass(frozen=True)
class CommandFeatures:
    admitted_body_command: torch.Tensor  # vx, vy=0, yaw rate, height; SI
    mode: torch.Tensor
    request_height_m: torch.Tensor
    command_clock_s: torch.Tensor
    previous_issued_action: torch.Tensor
    previous_control_source: torch.Tensor | None = None  # Deployable provenance,not an extra actor feature.


class ObservationAssembler:
    """Build explicit actor/critic fields without importing an environment.

    The critic retains 35+3+1+2+18+18+2+2 dimensions. Root velocity now has
    the named link-origin reference, and both critic wheel-position slots are
    zero. Those semantics are deliberately not advertised as legacy-compatible.
    Terrain-force inputs must be native terrain-filtered vectors, not unfiltered
    self-contact forces or fabricated contact probabilities.
    """

    CRITIC_SLICES = {
        'clean_actor_frame': slice(0, 35), 'velocity_origin_body_m_s': slice(35, 38),
        'height_above_ground_m': slice(38, 39), 'wheel_ground_force_div125': slice(39, 41),
        'tree_position_rad': slice(41, 59), 'tree_velocity_times0p1': slice(59, 77),
        'wheel_friction': slice(77, 79), 'wheel_support_above_environment_m': slice(79, 81),
    }

    def __init__(self, binding: MechanicalBinding, num_envs, device='cpu'):
        if not isinstance(binding, MechanicalBinding) or type(num_envs) is not int or num_envs < 1:
            raise ValueError('observation assembly requires a parsed binding and positive row count')
        self.binding, self.num_envs, self.device = binding, num_envs, torch.device(device)
        self._policy_indices = torch.tensor(binding.policy_joint_indices, device=self.device)
        self._tree_indices = torch.tensor(binding.tree_joint_indices, device=self.device)
        self._wheel_indices = torch.tensor(binding.wheel_body_indices, device=self.device)
        self._nominal = torch.tensor(binding.nominal_policy_position_rad, dtype=torch.float32, device=self.device)

    def _tensor(self, name, value, shape, dtype=torch.float32):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, *shape)
                or value.device != self.device or value.dtype != dtype):
            raise ValueError(f'{name} requires {(self.num_envs, *shape)}/{dtype} on {self.device}')

    @staticmethod
    def _inverse_rotate(quaternion, vector):
        # Quaternion convention is scalar-last; conjugation performs world->body.
        imaginary, scalar = -quaternion[:, :3], quaternion[:, 3:]
        cross = torch.linalg.cross(imaginary, vector)
        return vector + 2 * (scalar * cross + torch.linalg.cross(imaginary, cross))

    @torch.no_grad()
    def measure(self, sample: NativePhysicalSample):
        if not isinstance(sample, NativePhysicalSample):
            raise TypeError('physical measurements require the named native sample')
        shapes = {'joint_position_rad': (18,), 'joint_velocity_rad_s': (18,),
                  'root_link_position_world_m': (3,), 'root_link_orientation_xyzw': (4,),
                  'root_com_position_world_m': (3,), 'root_com_velocity_world_m_s': (3,),
                  'angular_velocity_world_rad_s': (3,), 'terrain_contact_force_world_n': (19, 3),
                  'ground_at_root_world_z_m': (), 'ground_at_wheel_world_z_m': (2,),
                  'environment_origin_world_m': (3,), 'wheel_friction': (2,)}
        for name, shape in shapes.items():
            self._tensor(name, getattr(sample, name), shape)
        self._tensor('timestamp_s', sample.timestamp_s, (), torch.float64)
        invalid = ~torch.isfinite(sample.timestamp_s) | (sample.timestamp_s < 0)
        for name in shapes:
            invalid |= ~torch.isfinite(getattr(sample, name)).reshape(self.num_envs, -1).all(-1)
        quaternion = sample.root_link_orientation_xyzw
        invalid |= (quaternion.norm(dim=-1) - 1).abs() > 2e-4
        invalid |= (sample.wheel_friction < 0).any(-1)
        if invalid.any():
            raise ValueError('physical values, unit xyzw orientation, time or friction are invalid')
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        omega = sample.angular_velocity_world_rad_s
        com_offset = sample.root_com_position_world_m - sample.root_link_position_world_m
        velocity = sample.root_com_velocity_world_m_s - torch.linalg.cross(omega, com_offset)
        gravity = sample.root_link_position_world_m.new_tensor((0., 0., -1.)).expand(self.num_envs, -1)
        tree_q = sample.joint_position_rad.index_select(1, self._tree_indices)
        # Absolute wheel angle is neither a task anchor nor a hidden clock.
        tree_q[:, list(self.binding.wheel_tree_indices)] = 0
        result = MeasuredState(
            sample.joint_position_rad.index_select(1, self._policy_indices),
            sample.joint_velocity_rad_s.index_select(1, self._policy_indices), tree_q,
            sample.joint_velocity_rad_s.index_select(1, self._tree_indices),
            self._inverse_rotate(quaternion, velocity), self._inverse_rotate(quaternion, omega),
            self._inverse_rotate(quaternion, gravity),
            sample.root_link_position_world_m[:, 2] - sample.ground_at_root_world_z_m,
            sample.terrain_contact_force_world_n.index_select(1, self._wheel_indices).norm(dim=-1),
            sample.wheel_friction.clone(),
            sample.ground_at_wheel_world_z_m - sample.environment_origin_world_m[:, 2:3],
            sample.timestamp_s.clone(),
        )
        self._measured(result)
        return result

    def _measured(self, state):
        if not isinstance(state, MeasuredState):
            raise TypeError('a named measured physical state is required')
        widths = {'policy_position_rad': (6,), 'policy_velocity_rad_s': (6,),
                  'tree_position_rad': (18,), 'tree_velocity_rad_s': (18,),
                  'velocity_origin_body_m_s': (3,), 'angular_velocity_body_rad_s': (3,),
                  'gravity_body': (3,), 'height_above_ground_m': (), 'wheel_ground_force_n': (2,),
                  'wheel_friction': (2,), 'wheel_support_above_environment_m': (2,)}
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, shape in widths.items():
            value = getattr(state, name)
            self._tensor(name, value, shape)
            invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
        self._tensor('measured timestamp_s', state.timestamp_s, (), torch.float64)
        invalid |= ~torch.isfinite(state.timestamp_s) | (state.timestamp_s < 0)
        invalid |= (state.wheel_ground_force_n < 0).any(-1) | (state.wheel_friction < 0).any(-1)
        invalid |= (state.tree_position_rad[:, list(self.binding.wheel_tree_indices)] != 0).any(-1)
        if invalid.any():
            raise ValueError('measured state contains invalid physical quantities or wheel-position leakage')

    def clean_sensors(self, state: MeasuredState):
        self._measured(state)
        return SensorSample(state.policy_position_rad.clone(), state.policy_velocity_rad_s.clone(),
                            state.angular_velocity_body_rad_s.clone(), state.gravity_body.clone(),
                            state.timestamp_s[:, None].expand(-1, 4).clone())

    def labels(self, state: MeasuredState, contact_history: WheelContactHistory, episode_id, control_epoch):
        self._measured(state)
        if (not isinstance(contact_history, WheelContactHistory)
                or contact_history.num_envs != self.num_envs or contact_history.device != self.device):
            raise ValueError('physical labels require this batch and device contact history')
        self._tensor('episode_id', episode_id, (), torch.long)
        self._tensor('control_epoch', control_epoch, (), torch.long)
        if ((episode_id < 0).any() or (control_epoch < 0).any()
                or not torch.isfinite(contact_history.last_time_s).all()
                or (contact_history.last_time_s - state.timestamp_s).abs().gt(1e-8).any()):
            raise ValueError('physical labels require same-tick contact history and valid episode/control identity')
        contacts, valid = contact_history.contacts()
        return PhysicalLabels(state.velocity_origin_body_m_s.clone(), contacts.to(torch.float32),
                              state.policy_velocity_rad_s.clone(), valid, state.timestamp_s.clone(),
                              episode_id.clone(), control_epoch.clone())

    @torch.no_grad()
    def actor(self, sensors: SensorSample, commands: CommandFeatures, *, now_s):
        if not isinstance(sensors, SensorSample) or not isinstance(commands, CommandFeatures):
            raise TypeError('actor accepts deployable sensor and command inputs only')
        for name, width in (('policy_position_rad', 6), ('policy_velocity_rad_s', 6),
                            ('angular_velocity_body_rad_s', 3), ('gravity_body', 3)):
            self._tensor(name, getattr(sensors, name), (width,))
        self._tensor('sensor sample times', sensors.sampled_at_s, (4,), torch.float64)
        self._tensor('now_s', now_s, (), torch.float64)
        for name, shape, dtype in (('admitted_body_command', (4,), torch.float32), ('mode', (), torch.long),
                                   ('request_height_m', (), torch.float32), ('command_clock_s', (), torch.float32),
                                   ('previous_issued_action', (6,), torch.float32)):
            self._tensor(name, getattr(commands, name), shape, dtype)
        age = now_s[:, None] - sensors.sampled_at_s
        invalid = (~torch.isfinite(age).all(-1) | (age < -1e-9).any(-1) | (now_s < 0)
                   | (sensors.sampled_at_s < 0).any(-1) | ~torch.isfinite(now_s)
                   | (commands.mode < 0) | (commands.mode > 4))
        for value in (sensors.policy_position_rad, sensors.policy_velocity_rad_s,
                      sensors.angular_velocity_body_rad_s, sensors.gravity_body,
                      commands.admitted_body_command, commands.previous_issued_action,
                      commands.request_height_m[:, None], commands.command_clock_s[:, None]):
            invalid |= ~torch.isfinite(value).all(-1)
        jump = commands.mode == int(CommandMode.JUMP)
        invalid |= (commands.command_clock_s < 0) | (commands.admitted_body_command[:, 3] <= 0)
        invalid |= (commands.admitted_body_command[:, 1].abs() > 1e-8)
        invalid |= jump & ((commands.request_height_m < .03) | (commands.request_height_m > .30))
        invalid |= ~jump & (commands.request_height_m != 0)
        source = commands.previous_control_source
        if source is None:
            source = torch.full((self.num_envs,), int(ControlSource.POLICY), dtype=torch.long, device=self.device)
        validate_issued_control(commands.previous_issued_action, source, num_envs=self.num_envs, device=self.device)
        if invalid.any():
            raise ValueError('actor sensor time, modes, request semantics or issued-action provenance is invalid')
        relative, _ = canonical_coaxial_displacement(sensors.policy_position_rad[:, :4], self._nominal[:4])
        relative = torch.cat((relative.float(), torch.zeros_like(sensors.policy_position_rad[:, 4:])), -1)
        modes = torch.nn.functional.one_hot(commands.mode, 5).to(torch.float32)
        command = commands.admitted_body_command
        frame = torch.cat((command[:, :3], command[:, 3:4] * 5,
                           sensors.angular_velocity_body_rad_s * .5, sensors.gravity_body,
                           relative, sensors.policy_velocity_rad_s * .1, commands.previous_issued_action,
                           modes, commands.request_height_m[:, None] * 5,
                           commands.command_clock_s[:, None].clamp_max(5)), -1)
        # Reject invalid raw inputs before clipping, so infinities cannot become
        # plausible finite observations. Ages remain metadata, never actor truth.
        if not torch.isfinite(frame).all():
            raise ValueError('actor scaling produced a nonfinite value before clipping')
        return frame.clamp(-100, 100)

    @torch.no_grad()
    def critic(self, commands: CommandFeatures, state: MeasuredState):
        # Rebuild the clean prefix here, so a delayed/noisy actor frame cannot
        # accidentally be supplied under the name "clean_actor_frame".
        clean_actor_frame = self.actor(self.clean_sensors(state), commands, now_s=state.timestamp_s)
        return torch.cat((clean_actor_frame, state.velocity_origin_body_m_s,
                          state.height_above_ground_m[:, None], state.wheel_ground_force_n / 125,
                          state.tree_position_rad, state.tree_velocity_rad_s * .1,
                          state.wheel_friction, state.wheel_support_above_environment_m), -1)


class WheelContactHistory:
    """Past-only 15ms labels at 200Hz, or at the 1kHz numerical reference.

    Reset clears the window. A row is label-valid only after a full new window
    of physical samples; settling must establish labels before PPO ownership.
    Repeated observation reads never advance this history. Forces describe own
    terrain-filtered contacts, not proof of absence from a cancelling net force.
    """

    def __init__(self, num_envs, device='cpu', *, threshold_n=2., physics_dt_s=.005):
        if (type(num_envs) is not int or num_envs < 1 or type(threshold_n) not in (int, float)
                or not math.isfinite(threshold_n) or threshold_n <= 0
                or type(physics_dt_s) not in (int, float) or physics_dt_s not in (.001, .005)):
            raise ValueError('contact history requires positive rows/threshold and the 200Hz or 1kHz clock')
        self.num_envs, self.device, self.threshold_n = num_envs, torch.device(device), threshold_n
        self.physics_dt_s, self.samples = physics_dt_s, round(.015 / physics_dt_s)
        self.history = torch.zeros(self.samples, num_envs, 2, device=self.device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.last_time_s = torch.full((num_envs,), -float('inf'), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def push(self, wheel_ground_force_n, time_s):
        if (not isinstance(wheel_ground_force_n, torch.Tensor)
                or wheel_ground_force_n.shape != (self.num_envs, 2)
                or wheel_ground_force_n.device != self.device or wheel_ground_force_n.dtype != torch.float32
                or not isinstance(time_s, torch.Tensor) or time_s.shape != (self.num_envs,)
                or time_s.dtype != torch.float64 or time_s.device != self.device):
            raise ValueError('contact sample shape/dtype/device differs')
        invalid = (~torch.isfinite(wheel_ground_force_n).all(-1) | (wheel_ground_force_n < 0).any(-1)
                   | ~torch.isfinite(time_s) | (time_s < 0)
                   | (torch.isfinite(self.last_time_s)
                      & ((time_s - self.last_time_s - self.physics_dt_s).abs() > 1e-8)))
        if invalid.any():
            raise ValueError('contact history must advance once per fresh selected physical interval')
        self.history[:-1] = self.history[1:].clone()
        self.history[-1].copy_(wheel_ground_force_n)
        self.count.add_(1).clamp_max_(self.samples)
        self.last_time_s.copy_(time_s)

    def contacts(self):
        return (self.history.amax(0) > self.threshold_n).clone(), (self.count == self.samples).clone()

    @torch.no_grad()
    def reset(self, mask, now_s):
        if (not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,)
                or mask.dtype != torch.bool or mask.device != self.device
                or not isinstance(now_s, torch.Tensor) or now_s.shape != (self.num_envs,)
                or now_s.dtype != torch.float64 or now_s.device != self.device
                or not torch.isfinite(now_s).all() or (now_s < 0).any()
                or (now_s[mask] < self.last_time_s[mask]).any()):
            raise ValueError('contact reset requires a row mask and nondecreasing finite physical clock')
        self.history[:, mask] = 0
        self.count[mask] = 0
        self.last_time_s[mask] = now_s[mask]

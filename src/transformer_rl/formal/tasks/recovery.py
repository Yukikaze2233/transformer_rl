"""Private explicit recovery requests and selected-asset geometric candidates.

This defines requested birth semantics,not physical recoverability. Infeasible
closure or output holds stay visible. No body is teleported,script is run or
extra policy is selected. The last two requests explicitly distinguish belly
contact with legs above the body from an inverted body with its legs above.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..control.actuation import OutputFeedbackController, PreparationHold
from .reset_kinematics import AssetResetKinematics, PostureClosure, PostureContinuation


RECOVERY_NAMES = (
    'upright', 'crouched', 'upright_release', 'crouched_release',
    'left_side_90', 'right_side_90', 'pitch_forward_45', 'pitch_backward_45',
    'front_down_90', 'back_down_90',
    'front_down_left_forward_right_back_90', 'front_down_left_back_right_forward_90',
    'back_down_left_forward_right_back_90', 'back_down_left_back_right_forward_90',
    'prone_belly_legs_above', 'prone_back_legs_above',
)


@dataclass(frozen=True)
class RecoveryTuning:
    """Explicit engineering recipe; none of these values are hardware-qualified."""
    crouched_inner_angle_rad: float
    asymmetric_leg_pitch_rad: float
    legs_above_pitch_rad: float
    release_clearance_m: float
    release_downward_speed_m_s: float
    output_hold_tolerance_rad: float
    loop_tolerance_m: float
    maximum_solver_iterations: int
    posture_segment_angle_rad: float
    maximum_posture_segments: int

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for name, v in vars(self).items() if name not in ('maximum_solver_iterations', 'maximum_posture_segments'))
                or type(self.maximum_solver_iterations) is not int or not 1 <= self.maximum_solver_iterations <= 256
                or type(self.maximum_posture_segments) is not int or not 1 <= self.maximum_posture_segments <= 256
                or self.posture_segment_angle_rad > math.pi/4
                or not self.asymmetric_leg_pitch_rad < self.legs_above_pitch_rad <= math.pi):
            raise ValueError('recovery tuning needs explicit finite angle/length/speed/tolerance/iteration inputs')


@dataclass(frozen=True)
class RecoveryCandidates:
    requested_posture_id: torch.Tensor
    requested_root_orientation_xyzw: torch.Tensor
    requested_inner_knee_rad: torch.Tensor
    requested_leg_forward_pitch_rad: torch.Tensor
    release_clearance_m: torch.Tensor
    origin_velocity_body_m_s: torch.Tensor
    angular_velocity_body_rad_s: torch.Tensor
    closure: PostureClosure
    required_preparation_action: torch.Tensor
    current_output_hold_admitted: torch.Tensor
    decoded_hold_error_rad: torch.Tensor
    pair_projected: torch.Tensor
    posture_path: PostureContinuation
    preparation_hold: PreparationHold
    tuning: RecoveryTuning
    physical_qualification: bool = False


class RecoveryPoseCompiler:
    def __init__(self, kinematics: AssetResetKinematics, tuning: RecoveryTuning):
        if not isinstance(kinematics, AssetResetKinematics) or not isinstance(tuning, RecoveryTuning):
            raise TypeError('recovery compiler requires selected closed-chain geometry and explicit engineering tuning')
        if not kinematics.geometry.inner_limits_rad[0] < tuning.crouched_inner_angle_rad < kinematics.geometry.inner_limits_rad[1]:
            raise ValueError('crouched request must lie within original selected geometric knee stops')
        self.kinematics, self.tuning, self.device = kinematics, tuning, kinematics.device
        nominal = kinematics.nominal_tree_position[None]
        state = kinematics.forward(nominal)
        self.nominal_inner = kinematics.inner_knee_angles(nominal)[0]
        # The requested direction is physical forward pitch. Derive each
        # mirrored output sign from its actual wheel/hinge geometry.
        indices = (kinematics.policy_tree_indices[0], kinematics.policy_tree_indices[2])
        wheels = kinematics.binding.wheel_body_indices
        sign = []
        for joint, body in zip(indices, wheels):
            lever = state.body_position_m[:, body] - state.joint_origin_world_m[:, joint]
            derivative = torch.linalg.cross(state.joint_axis_world[:, joint], lever)[:, 0]
            if derivative.abs().min() < 1e-4:
                raise ValueError('selected reset hip has no identifiable forward wheel displacement direction')
            sign.append(derivative.sign()[0])
        self._forward_sign = torch.stack(sign)

    @staticmethod
    def quaternion(roll, pitch, yaw):
        sr, cr, sp, cp, sy, cy = ((roll / 2).sin(), (roll / 2).cos(), (pitch / 2).sin(),
                                  (pitch / 2).cos(), (yaw / 2).sin(), (yaw / 2).cos())
        return torch.stack((sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
                            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy), -1)

    @torch.no_grad()
    def compile(self, posture_id, yaw_rad, controller: OutputFeedbackController, *,
                roll_pitch_offset_rad=None, inner_knee_offset_rad=None, leg_pitch_offset_rad=None):
        if (not isinstance(posture_id, torch.Tensor) or posture_id.ndim != 1 or posture_id.numel() < 1
                or posture_id.dtype != torch.long or posture_id.device != self.device
                or ((posture_id < 0) | (posture_id >= len(RECOVERY_NAMES))).any()
                or not isinstance(controller, OutputFeedbackController) or controller.device != self.device
                or controller.num_envs != len(posture_id)
                or not torch.equal(controller.nominal, torch.tensor(self.kinematics.binding.nominal_policy_position_rad,
                                                                    dtype=torch.float32, device=self.device))):
            raise ValueError('recovery IDs and actual output-controller nominal rows differ')
        count = len(posture_id)
        if (not isinstance(yaw_rad, torch.Tensor) or yaw_rad.shape != (count,) or yaw_rad.dtype != torch.float64
                or yaw_rad.device != self.device or not torch.isfinite(yaw_rad).all()):
            raise ValueError('recovery yaw requires explicit finite float64 per-row world headings')
        offset = []
        for value in (roll_pitch_offset_rad, inner_knee_offset_rad, leg_pitch_offset_rad):
            if value is None:
                value = torch.zeros(count, 2, dtype=torch.float64, device=self.device)
            if (not isinstance(value, torch.Tensor) or value.shape != (count, 2) or value.dtype != torch.float64
                    or value.device != self.device or not torch.isfinite(value).all()):
                raise ValueError('continuous recovery perturbations need explicit finite float64 left/right rows')
            offset.append(value)
        # Left/right denote the body side facing the ground; +pitch faces the
        # front toward the ground. These are body conventions,not motor signs.
        angles = torch.tensor(((0., 0.), (0., 0.), (0., 0.), (0., 0.),
                                (-math.pi/2, 0.), (math.pi/2, 0.),
                                (0., math.pi/4), (0., -math.pi/4),
                                (0., math.pi/2), (0., -math.pi/2),
                                (0., math.pi/2), (0., math.pi/2),
                                (0., -math.pi/2), (0., -math.pi/2), (0., 0.), (math.pi, 0.)),
                               dtype=torch.float64, device=self.device)[posture_id] + offset[0]
        knee = self.nominal_inner[None].expand(count, -1).clone()
        crouched = (posture_id == 1) | (posture_id == 3) | (posture_id >= 14)
        knee[crouched] = self.tuning.crouched_inner_angle_rad
        knee += offset[1]
        forward = torch.zeros(count, 2, dtype=torch.float64, device=self.device)
        asym_forward = (posture_id == 10) | (posture_id == 12)
        asym_backward = (posture_id == 11) | (posture_id == 13)
        forward[asym_forward] = forward.new_tensor((1., -1.)) * self.tuning.asymmetric_leg_pitch_rad
        forward[asym_backward] = forward.new_tensor((-1., 1.)) * self.tuning.asymmetric_leg_pitch_rad
        forward[posture_id == 14] = self.tuning.legs_above_pitch_rad
        forward += offset[2]
        indices = (self.kinematics.policy_tree_indices[0], self.kinematics.policy_tree_indices[2])
        seed = self.kinematics.nominal_tree_position[None].repeat(count, 1)
        hip = seed[:, indices] + forward * self._forward_sign
        path = self.kinematics.continue_posture(seed, hip, knee,
                                                maximum_segment_angle_rad=self.tuning.posture_segment_angle_rad,
                                                maximum_segments=self.tuning.maximum_posture_segments,
                                                closure_tolerance_m=self.tuning.loop_tolerance_m,
                                                maximum_iterations=self.tuning.maximum_solver_iterations)
        solution = path.closure
        held = controller.reset_hold_action(solution.policy_positions.float(),
                                             position_tolerance_rad=self.tuning.output_hold_tolerance_rad)
        preparation = controller.preparation_hold(solution.policy_positions.float(),
                                                   position_tolerance_rad=self.tuning.output_hold_tolerance_rad)
        release = (posture_id == 2) | (posture_id == 3)
        clearance = release.double() * self.tuning.release_clearance_m
        orientation = self.quaternion(angles[:, 0], angles[:, 1], yaw_rad)
        # Release is world-down even when a continuous birth perturbation tilts
        # the body. Store the corresponding link-origin body-frame velocity.
        world_velocity = torch.zeros(count, 3, dtype=torch.float64, device=self.device)
        world_velocity[:, 2] = -release.double() * self.tuning.release_downward_speed_m_s
        from .reset_collision import AssetCollisionGeometry

        rotation = AssetCollisionGeometry._rotation(orientation)
        velocity = (rotation.transpose(-1, -2) @ world_velocity[..., None]).squeeze(-1)
        return RecoveryCandidates(posture_id.detach().clone(), orientation,
                                    knee, forward, clearance, velocity, torch.zeros_like(velocity), solution,
                                    held.requested_action, held.admitted, held.target_error_rad, held.pair_projected,
                                    path, preparation, self.tuning)

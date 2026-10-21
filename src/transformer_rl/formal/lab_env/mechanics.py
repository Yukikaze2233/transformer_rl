"""Device-resident native loop observations; never a control or reset gate."""
from __future__ import annotations

import torch


class NativeLoopTelemetry:
    """Measure source anchors from actual link poses, including reset boundaries.

    Window statistics reset between collections. Pending first-step markers
    and lifetime peaks survive that boundary. No RNG, joint writes, host reads
    or acceptance thresholds occur in the observation path.
    """

    def __init__(self, constraints, body_names, num_envs, device):
        names = tuple(body_names)
        if (type(num_envs) is not int or num_envs < 1 or len(names) != 19
                or len(set(names)) != len(names) or len(constraints) != 6
                or len({c['name'] for c in constraints}) != 6
                or any(c['type'] != 'spherical' or c['body0'] == c['body1']
                       or c['body0'] not in names or c['body1'] not in names
                       for c in constraints)):
            raise ValueError('loop telemetry requires the six named selected-asset constraints')
        self.num_envs, self.device = num_envs, torch.device(device)
        self.names = tuple(c['name'] for c in constraints)
        self._ids = torch.tensor([[names.index(c['body0']), names.index(c['body1'])]
                                  for c in constraints], device=self.device)
        self._local = torch.tensor([[c['local_pos0_m'], c['local_pos1_m']]
                                    for c in constraints], dtype=torch.float32, device=self.device)
        if self._local.shape != (6, 2, 3) or not bool(torch.isfinite(self._local).all()):
            raise ValueError('loop telemetry source anchors must be finite metre vectors')
        self._pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._lifetime_peak = torch.zeros((), device=self.device)
        self._lifetime_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self._environment_peak = torch.zeros(num_envs, device=self.device)
        self._loop_peak = torch.zeros(6, device=self.device)
        self._reset_peak = torch.zeros(6, device=self.device)
        self._first_peak = torch.zeros(6, device=self.device)
        self._invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self._velocity_peak = torch.zeros((), device=self.device)
        self._velocity_invalid = torch.zeros((), dtype=torch.long, device=self.device)
        self._sum = torch.zeros((), device=self.device)
        self._sum_sq = torch.zeros((), device=self.device)
        self._loop_sum = torch.zeros(6, device=self.device)
        self._loop_sum_sq = torch.zeros(6, device=self.device)
        self._loop_count = torch.zeros(6, dtype=torch.long, device=self.device)
        self._count = torch.zeros((), dtype=torch.long, device=self.device)
        self.clear_window()

    def clear_window(self):
        for value in (self._environment_peak, self._loop_peak, self._reset_peak,
                      self._first_peak, self._invalid, self._velocity_peak,
                      self._velocity_invalid, self._sum, self._sum_sq,
                      self._loop_sum, self._loop_sum_sq, self._loop_count, self._count):
            value.zero_()
        self._frames = self._reset_rows = self._velocity_frames = 0
        self._first_rows = torch.zeros((), dtype=torch.long, device=self.device)

    def _gaps(self, poses):
        if (not isinstance(poses, torch.Tensor) or poses.ndim != 3
                or poses.shape[1:] != (19, 7) or poses.device != self.device
                or poses.dtype != torch.float32):
            raise ValueError('native loop poses require float32 [N,19,7] xyzw on the bound device')
        selected = poses[:, self._ids]
        quaternion = selected[..., 3:]
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
        local = self._local[None].expand(len(poses), -1, -1, -1)
        cross = torch.linalg.cross(quaternion[..., :3], local)
        anchors = selected[..., :3] + local + 2 * (
            quaternion[..., 3:] * cross + torch.linalg.cross(quaternion[..., :3], cross))
        return (anchors[:, :, 0] - anchors[:, :, 1]).norm(dim=-1)

    def _finite_reduction_values(self, gaps):
        finite = torch.isfinite(gaps)
        invalid = (~finite).sum()
        self._invalid.add_(invalid)
        self._lifetime_invalid.add_(invalid)
        # This zero is only a reduction identity. Any invalid observation
        # makes the published closure maximum unavailable, never accepted.
        return torch.where(finite, gaps, 0.)

    @torch.no_grad()
    def observe_reset(self, poses, ids):
        if (poses.shape[0] != self.num_envs or ids.ndim != 1
                or ids.device != self.device or ids.dtype != torch.long):
            raise ValueError('reset telemetry requires exact selected device row indices')
        if ids.numel() == 0:
            return
        gaps = self._finite_reduction_values(self._gaps(poses.index_select(0, ids)))
        self._reset_peak.copy_(torch.maximum(self._reset_peak, gaps.amax(0)))
        self._pending[ids] = True
        self._reset_rows += ids.numel()

    @torch.no_grad()
    def observe_physics(self, poses, joint_velocity):
        if (poses.shape[0] != self.num_envs
                or joint_velocity.shape != (self.num_envs, 18)
                or joint_velocity.device != self.device or joint_velocity.dtype != torch.float32):
            raise ValueError('physical telemetry requires complete native poses and joint velocities')
        raw = self._gaps(poses)
        finite = torch.isfinite(raw)
        gaps = self._finite_reduction_values(raw)
        # Keep a true sample mean/RMS alongside the peaks. Invalid lanes are
        # counted in the invalid counter and excluded from these moments.
        self._sum.add_(gaps.sum())
        self._sum_sq.add_(gaps.square().sum())
        self._loop_sum.add_(gaps.sum(0))
        self._loop_sum_sq.add_(gaps.square().sum(0))
        self._loop_count.add_(finite.sum(0))
        self._count.add_(finite.sum())
        self._environment_peak.copy_(torch.maximum(self._environment_peak, gaps.amax(-1)))
        self._loop_peak.copy_(torch.maximum(self._loop_peak, gaps.amax(0)))
        self._lifetime_peak.copy_(torch.maximum(self._lifetime_peak, gaps.amax()))
        first = torch.where(self._pending[:, None], gaps, 0.)
        self._first_peak.copy_(torch.maximum(self._first_peak, first.amax(0)))
        self._first_rows.add_(self._pending.sum())
        self._pending.zero_()
        finite = torch.isfinite(joint_velocity)
        self._velocity_invalid.add_((~finite).sum())
        self._velocity_peak.copy_(torch.maximum(
            self._velocity_peak, torch.where(finite, joint_velocity.abs(), 0.).amax()))
        self._frames += 1
        self._velocity_frames += 1

    def summary(self):
        p95 = torch.quantile(self._environment_peak, .95)
        scalars = torch.stack((self._environment_peak.amax(), p95,
                               self._reset_peak.amax(), self._first_peak.amax(),
                               self._lifetime_peak, self._invalid, self._lifetime_invalid,
                               self._velocity_peak, self._velocity_invalid, self._first_rows,
                               self._sum, self._sum_sq, self._count.to(torch.float64)))
        # One host transfer per summary, with no changes to observation state.
        values = torch.cat((scalars, self._loop_peak, self._loop_sum,
                            self._loop_count.to(torch.float64)))
        values = values.detach().cpu().tolist()
        bad, lifetime_bad = int(values[5]), int(values[6])
        observed = self._frames > 0
        valid = observed and bad == 0
        count = int(values[12])
        mean = values[10] / count if valid and count else None
        rms = (values[11] / count) ** 0.5 if valid and count else None
        mean_by_constraint = {
            name: (values[19 + index] / values[25 + index]
                   if valid and values[25 + index] else None)
            for index, name in enumerate(self.names)
        }
        return {
            'six_loop_scope': 'world anchor separation at every returned physical substep; reset observations use forward(dt=0)',
            'six_loop_physics_frames': self._frames,
            'six_loop_constraint_samples': self._frames * self.num_envs * 6,
            'six_loop_valid_constraint_samples': count,
            'six_loop_reset_rows': self._reset_rows,
            'six_loop_first_physics_rows': int(values[9]),
            'six_loop_nonfinite_constraint_observations': bad,
            'six_loop_mean_error_m': mean,
            'six_loop_rms_error_m': rms,
            'six_loop_mean_by_constraint_m': mean_by_constraint,
            'six_loop_closure_max_m': values[0] if valid else None,
            'six_loop_environment_peak_p95_m': values[1] if valid else None,
            'six_loop_reset_after_forward_max_m': values[2] if self._reset_rows and not bad else None,
            'six_loop_first_physics_max_m': values[3] if values[9] and not bad else None,
            'six_loop_lifetime_max_m': values[4] if observed and not lifetime_bad else None,
            'six_loop_max_by_constraint_m': dict(zip(
                self.names, values[13:19] if valid else [None] * 6)),
            'joint_velocity_max_native_units_per_s': (
                values[7] if self._velocity_frames and not values[8] else None),
            'joint_velocity_nonfinite_observations': int(values[8]),
            'physical_qualification': False,
        }

"""Pure gas pressure on the selected main-link mounting points.

The gas bodies, slider, closures and damping remain part of the physical model.
Callers must remove slider pressure when applying these wrenches,
while retaining the existing slider damping exactly once. This component does
not add damping, drive effort, control targets or a deployment actuator command.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

from ..control.pressure import gas_spring_force_n


class SpringPressureForces(NamedTuple):
    forces_world: torch.Tensor
    torques_about_com_world: torch.Tensor
    mount_positions_world: torch.Tensor
    mount_distance_m: torch.Tensor
    compression_m: torch.Tensor
    pressure_force_n: torch.Tensor
    valid: torch.Tensor
    normalized_compression: torch.Tensor
    within_mechanical_stroke: torch.Tensor
    within_fitted_domain: torch.Tensor


class SpringForceElement:
    """Pure pressure force for the authenticated two-sided mounting geometry.

    Output body order is upper-left, lower-left, upper-right, lower-right.
    Inputs use body-link poses with xyzw quaternions and actual body COM positions.
    ``valid`` describes finite force geometry, not closure or travel qualification.
    ``within_mechanical_stroke`` and ``within_fitted_domain`` are separate
    diagnostics for the *mount-point* compression. Invalid lanes yield NaN
    wrenches; they are never replaced by nominal geometry. No motor-angle
    branch is inferred or clipped here. Compression is clamped to the stroke;
    beyond the fitted domain the force continues linearly from its end slope.
    """

    def __init__(self, spec, manifest, fit, body_names, device, dtype=torch.float32):
        body_names = tuple(body_names)
        if (len(body_names) != len(set(body_names))
                or set(body_names) != set(manifest["rigid_body_names"])
                or set(body_names) != {body["name"] for body in spec["bodies"]}
                or (len(spec["bodies"]), len(spec["joints"]), len(spec["constraints"])) != (19, 18, 6)
                or manifest["spring_joint_names"] != ["L_spring_slide", "R_spring_slide"]):
            raise ValueError("Spring force element requires the bound full two-sided topology")
        coefficients = fit["monomial_coefficients_n"]
        domain = fit.get("normalized_fit_domain")
        if (fit["pressure_mpa"] != 10. or len(coefficients) != 4
                or any(not math.isfinite(value) for value in coefficients)
                or not math.isfinite(fit["stroke_m"]) or fit["stroke_m"] <= 0
                or not isinstance(domain, (list, tuple)) or len(domain) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in domain)
                or not 0 <= domain[0] < domain[1] <= 1):
            raise ValueError("Spring force element requires the existing finite 10MPa cubic and fit domain")
        by_name = {joint["name"]: joint for joint in spec["joints"]}
        constraints = {constraint["name"]: constraint for constraint in spec["constraints"]}
        bodies, points, full_lengths, strokes = [], [], [], []
        for side in ("L", "R"):
            slide_name = side+"_spring_slide"
            slide = by_name[slide_name]
            lower = by_name[side*3+"_joint2"]
            upper = constraints[side+"_spring_upper_mount"]
            binding = spec["spring_binding"][slide_name]
            if (slide["type"] != "prismatic" or lower["parent"] != side+"_link2"
                    or lower["child"] != slide["parent"] or upper["type"] != "spherical"
                    or upper["body0"] != side+"_link1" or upper["body1"] != slide["child"]
                    or binding["pressure_mpa"] != fit["pressure_mpa"]
                    or binding["stroke_m"] != fit["stroke_m"]
                    or not math.isclose(binding["source_mount_distance_m"]
                        + binding["compression_at_q_zero_m"], binding["full_extension_pin_distance_m"],
                        abs_tol=1e-12, rel_tol=0.)):
                raise ValueError("Gas mounts, coordinate datum and pressure curve are inconsistent")
            bodies.extend((upper["body0"], lower["parent"]))
            points.extend((upper["local_pos0_m"], [lower["origin"][i][3] for i in range(3)]))
            full_lengths.append(binding["full_extension_pin_distance_m"])
            strokes.append(binding["stroke_m"])
        self.device, self.dtype = torch.device(device), dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Spring force geometry requires float32 or float64")
        self.body_names, self.main_link_body_names = body_names, tuple(bodies)
        self.body_ids = tuple(body_names.index(name) for name in bodies)
        self._indices = torch.tensor(self.body_ids, device=self.device)
        self._points = torch.tensor(points, dtype=dtype, device=self.device)
        self._full_lengths = torch.tensor(full_lengths, dtype=dtype, device=self.device)
        self._strokes = torch.tensor(strokes, dtype=dtype, device=self.device)
        self._fit_domain = torch.tensor(domain, dtype=dtype, device=self.device)
        self._coefficients = tuple(float(value) for value in coefficients)
        self._fit_end = float(domain[1])
        if not bool(torch.isfinite(self._points).all() & torch.isfinite(self._full_lengths).all()
                   & torch.isfinite(self._strokes).all() & torch.isfinite(self._fit_domain).all()):
            raise ValueError("Spring mounting points and length datums must be finite")

    def evaluate(self, body_link_pose_world, body_com_position_world, spring_strength):
        """Return pressure wrenches in world coordinates, without mutating inputs.

        Strength is [N,1] for the existing common scalar or [N,2] for explicit
        left/right values. Body poses and COMs retain the supplied body order.
        No native read, CPU synchronization, damping or state update occurs here.
        """
        count = len(body_link_pose_world)
        if (body_link_pose_world.shape != (count, len(self.body_names), 7)
                or body_com_position_world.shape != (count, len(self.body_names), 3)
                or spring_strength.shape not in ((count, 1), (count, 2))
                or any(value.device != self.device or value.dtype != self.dtype
                    for value in (body_link_pose_world, body_com_position_world, spring_strength))):
            raise ValueError("Spring force inputs must match the bound body order, device and dtype")
        pose = body_link_pose_world.index_select(1, self._indices)
        com = body_com_position_world.index_select(1, self._indices)
        quaternion = pose[..., 3:7]
        quaternion_norm = quaternion.norm(dim=-1, keepdim=True)
        quaternion = quaternion/quaternion_norm
        local = self._points.expand(count, -1, -1)
        cross = torch.linalg.cross(quaternion[..., :3], local)
        mounts = pose[..., :3]+local+2*(quaternion[..., 3:] * cross
            + torch.linalg.cross(quaternion[..., :3], cross))
        pairs = mounts.reshape(count, 2, 2, 3)
        separation = pairs[:, :, 0]-pairs[:, :, 1]
        length = separation.norm(dim=-1)
        compression = self._full_lengths-length
        normalized_compression = compression/self._strokes
        u = normalized_compression.clamp(0., 1.)
        magnitude = gas_spring_force_n(u, self._coefficients, self._fit_end)*spring_strength
        upper_force = magnitude[..., None]*(separation/length[..., None])
        forces = torch.stack((upper_force, -upper_force), dim=2).flatten(1, 2)
        torques = torch.linalg.cross(mounts-com, forces)
        valid = (torch.isfinite(pose).flatten(1).all(-1) & torch.isfinite(com).flatten(1).all(-1)
            & torch.isfinite(spring_strength).all(-1) & (spring_strength >= 0).all(-1)
            & torch.isfinite(quaternion_norm).flatten(1).all(-1)
            & torch.isfinite(mounts).flatten(1).all(-1)
            & torch.isfinite(separation).flatten(1).all(-1)
            & torch.isfinite(length).all(-1) & torch.isfinite(compression).all(-1)
            & (quaternion_norm.squeeze(-1) > 0).all(-1) & (length > 0).all(-1)
            & torch.isfinite(forces).flatten(1).all(-1) & torch.isfinite(torques).flatten(1).all(-1))
        within_mechanical_stroke = (valid[:, None] & (compression >= 0.)
            & (compression <= self._strokes))
        within_fitted_domain = (valid[:, None]
            & (normalized_compression >= self._fit_domain[0])
            & (normalized_compression <= self._fit_domain[1]))
        forces = torch.where(valid[:, None, None], forces, torch.nan)
        torques = torch.where(valid[:, None, None], torques, torch.nan)
        magnitude = torch.where(valid[:, None], magnitude, torch.nan)
        return SpringPressureForces(
            forces, torques, mounts, length, compression, magnitude, valid,
            normalized_compression, within_mechanical_stroke, within_fitted_domain,
        )

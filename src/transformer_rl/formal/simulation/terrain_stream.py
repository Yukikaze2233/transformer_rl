"""活跃回合只铺设近身平地/粗糙块；远处仍由完整全局源定义。"""
from __future__ import annotations

import torch

from .terrain import NativeTerrainInstaller


class NativeTerrainStream:
    """保留近身碰撞prim，不移动机器人、重置时钟或清空接触历史。"""

    def __init__(self, installer: NativeTerrainInstaller, collision_geometry):
        if not isinstance(installer, NativeTerrainInstaller):
            raise TypeError('terrain stream requires its owned installer')
        self.installer, self.environment = installer, installer.environment
        life = installer.lifecycle
        geometry = collision_geometry
        if geometry.kinematics.binding != life.plant.scene.observation.binding:
            raise ValueError('terrain streaming needs the selected physical collision geometry')
        radius = torch.zeros(19, dtype=torch.float64, device=life.device)
        for body, points in geometry._mesh_points:
            radius[body] = torch.maximum(radius[body], points.norm(dim=-1).max())
        for body, offset, _axis, r, length in geometry._cylinders:
            radius[body] = torch.maximum(radius[body], offset.norm() + (r*r + length*length/4)**.5)
        if not torch.isfinite(radius).all() or (radius <= 0).any():
            raise ValueError('each body requires a complete finite collision bounding radius')
        coms = installer.editor._convert(life.plant.scene.robot.root_view.get_coms())
        if (not isinstance(coms, torch.Tensor) or coms.shape != (life.num_envs, 19, 7)
                or not torch.isfinite(coms).all()):
            raise ValueError('terrain streaming needs complete actual native COM readback')
        self._radius = radius
        self._com_radius = coms[..., :3].to(life.device).double().norm(dim=-1)
        self._scene = None
        self._limits = None
        self._tiled = None
        self.last_result = None

    def _bind(self):
        scene = self.installer.lifecycle.plant.scene
        if scene is self._scene:
            return
        courses = self.installer.lifecycle.ground.courses
        limits = torch.tensor(((-float('inf'), float('inf')),)*2,
                              dtype=torch.float64, device=self._radius.device).repeat(len(courses), 1, 1)
        tiled = torch.tensor([c.tile_source is not None for c in courses], device=self._radius.device)
        for row in tiled.nonzero().flatten().tolist():
            c = courses[row]
            bounds = ((min(p.x_m[0] for p in c.surface.patches), max(p.x_m[-1] for p in c.surface.patches)),
                      (min(p.y_m[0] for p in c.surface.patches), max(p.y_m[-1] for p in c.surface.patches)))
            full = (c.request.extent.x_m, c.request.extent.y_m)
            # 比驻留边界提前1m补块；完整域边缘保留原边界，不延伸虚构地面。
            for axis in range(2):
                lo, hi = bounds[axis]
                limits[row, axis, 0] = lo + (1. if lo > full[axis][0] + 1e-5 else 0.)
                limits[row, axis, 1] = hi - (1. if hi < full[axis][1] - 1e-5 else 0.)
        self._scene, self._limits, self._tiled = scene, limits, tiled

    @torch.no_grad()
    def __call__(self):
        env, life = self.environment, self.installer.lifecycle
        env._identity()
        if (life._stepped or life._pending_tick is not None or env._in_step
                or env.rewards.composer._window is not None):
            raise RuntimeError('terrain streaming requires a drained policy boundary')
        self._bind()
        if not self._tiled.any():
            return
        scene, bridge = life.plant.scene, life.plant.scene.observation
        position = scene.robot.data.body_link_pose_w.torch[..., :2].double()
        velocity = scene.robot.data.body_com_vel_w.torch.double()
        # 球包络覆盖任意身体朝向；再覆盖下一个完整策略间隔的当前运动及0.25m余量。
        dt = bridge.physics_dt_s * 2
        pad = (self._radius[None, :, None] + velocity[..., :2].abs()*dt
               + (velocity[..., 3:].norm(dim=-1)*(self._radius + self._com_radius)*dt)[..., None] + .25)
        local = position - scene.environment_origins_m[:, None, :2]
        protected = torch.stack(((local-pad).amin(1), (local+pad).amax(1)), -1)
        changed = self._tiled & ((protected[..., 0] < self._limits[..., 0]).any(-1)
                                 | (protected[..., 1] > self._limits[..., 1]).any(-1))
        if not changed.any():
            return
        courses = list(life.ground.courses)
        for row in changed.nonzero().flatten().tolist():
            c = courses[row]
            bounds = protected[row].cpu().tolist()
            keys = c.tile_source.tile_keys(bounds[0], bounds[1], cells=8, halo=1)
            courses[row] = c.with_tiles(keys)
        try:
            result = self.installer.editor.replace_courses(life.ground, tuple(courses), changed,
                                                           protected_xy_m=protected)
            self.installer.adopt(result)
            self.last_result = result
            self._bind()
        except BaseException:
            env._faulted = life._faulted = life.plant._faulted = True
            raise

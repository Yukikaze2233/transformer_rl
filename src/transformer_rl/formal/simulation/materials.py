"""按精确摩擦系数共享原生材料，最后一个碰撞体退休时释放。"""
from __future__ import annotations

import math
import struct


class NativeTerrainMaterials:
    def __init__(self, stage, sim_utils):
        self.stage, self.sim_utils = stage, sim_utils
        self.root = '/World/TerrainMaterials'
        if stage.GetPrimAtPath(self.root).IsValid():
            raise RuntimeError('owned terrain material scope already exists')
        self._owners = {}
        self._counts = {}

    @staticmethod
    def path_for_friction(friction):
        if type(friction) not in (int, float) or not math.isfinite(friction) or friction < 0:
            raise ValueError('terrain material needs finite nonnegative friction')
        # IEEE754编码不量化连续课程参数，也没有摘要碰撞。
        return '/World/TerrainMaterials/m_' + struct.pack('>d', float(friction)).hex()

    def bind(self, collider, friction):
        if collider in self._owners or not collider.startswith('/World/envs/env_'):
            raise ValueError('material binding must belong to one new owned collider')
        path = self.path_for_friction(friction)
        if path not in self._counts:
            if self.stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError('unowned terrain material appeared inside the pool')
            material = self.sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, restitution=0.,
                friction_combine_mode='multiply', restitution_combine_mode='multiply')
            material.func(path, material)
            self._counts[path] = 0
        self.sim_utils.bind_physics_material(collider, path)
        self._owners[collider] = path
        self._counts[path] += 1

    def release(self, collider):
        path = self._owners.pop(collider)
        self._counts[path] -= 1
        if self._counts[path] == 0:
            if not self.stage.RemovePrim(path):
                raise RuntimeError('unused owned terrain material could not be retired')
            del self._counts[path]

    @property
    def unique_count(self):
        return len(self._counts)

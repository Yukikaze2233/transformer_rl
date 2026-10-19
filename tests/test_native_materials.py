"""共享材料不量化摩擦；跨碰撞体引用与退休分别计数。"""
from types import SimpleNamespace

import pytest

from transformer_rl.formal.simulation.materials import NativeTerrainMaterials


def fixture():
    prims, bindings, created = set(), {}, []

    def remove(path):
        prims.remove(path)
        return True

    stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: path in prims),
                            RemovePrim=remove)

    def material(**kwargs):
        def author(path, value):
            prims.add(path)
            created.append((path, kwargs))
        return SimpleNamespace(func=author)

    utils = SimpleNamespace(RigidBodyMaterialCfg=material,
                            bind_physics_material=lambda collider, path: bindings.update({collider: path}))
    return NativeTerrainMaterials(stage, utils), prims, bindings, created


def test_one_exact_material_is_shared_and_retired_only_after_its_last_owner():
    pool, prims, bindings, created = fixture()
    first, second = '/World/envs/env_0/Terrain/a', '/World/envs/env_1/Terrain/b'
    pool.bind(first, .7)
    pool.bind(second, .7)
    assert len(created) == pool.unique_count == 1
    assert bindings[first] == bindings[second]
    pool.release(first)
    assert pool.unique_count == 1 and bindings[second] in prims
    pool.release(second)
    assert pool.unique_count == 0 and not prims
    pool.bind(first, .7)
    assert len(created) == 2


def test_continuous_coefficients_are_not_rounded_to_a_material_palette():
    pool, _, bindings, _ = fixture()
    pool.bind('/World/envs/env_0/a', .70000000001)
    pool.bind('/World/envs/env_0/b', .70000000002)
    assert pool.unique_count == 2 and len(set(bindings.values())) == 2
    with pytest.raises(ValueError, match='one new owned'):
        pool.bind('/World/envs/env_0/a', .8)

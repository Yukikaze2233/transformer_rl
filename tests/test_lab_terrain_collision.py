"""Single-material native surfaces preserve geometry and complete face lineage."""
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import pytest

from transformer_rl.formal.lab_env.terrain_collision import (
    SOURCE_COLLISION_PATH, NATIVE_COLLISION_ROOT, partition_static_collision,
)


def source_stage():
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, '/World/ground/terrain')
    parent.AddTranslateOp().Set(Gf.Vec3d(7., -3., .2))
    mesh = UsdGeom.Mesh.Define(stage, SOURCE_COLLISION_PATH)
    mesh.CreatePointsAttr([(0., 0., 0.), (1., 0., 0.), (1., 1., .2), (0., 1., .2)])
    mesh.CreateFaceVertexCountsAttr([3, 3]);mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    mesh.AddTranslateOp().Set(Gf.Vec3d(1., 2., .3))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    for index, mu in enumerate((.2, .9)):
        material = UsdShade.Material.Define(stage, '/World/ground/terrain/Material_' + str(index))
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(mu);physics.CreateDynamicFrictionAttr(mu)
        subset = api.CreateMaterialBindSubset('Surface_' + str(index), [index])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material, materialPurpose='physics')
    return stage, mesh


def triangles_world(mesh):
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    return [tuple(tuple(round(v, 7) for v in transform.Transform(Gf.Vec3d(points[i])))
                  for i in indices[start:start + 3]) for start in range(0, len(indices), 3)]


def test_native_material_parts_preserve_every_world_triangle_and_actual_material():
    stage, mesh = source_stage()
    original = triangles_world(mesh)
    record = partition_static_collision(stage)
    assert record['native_faces'] == record['source_faces'] == 2
    assert record['physical_materials_reused'] and not record['vertices_moved']
    assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is False
    reconstructed = []
    for index, part in enumerate(record['partitions']):
        native = UsdGeom.Mesh(stage.GetPrimAtPath(part['path']))
        assert len(native.GetPointsAttr().Get()) == 3
        assert not native.GetPrim().GetChildren()
        assert UsdPhysics.CollisionAPI(native).GetCollisionEnabledAttr().Get() is True
        material, _ = UsdShade.MaterialBindingAPI(native).ComputeBoundMaterial(materialPurpose='physics')
        assert str(material.GetPath()) == part['material_path']
        assert float(UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get()) == pytest.approx((.2, .9)[index])
        reconstructed += triangles_world(native)
    assert sorted(reconstructed) == sorted(original)
    with pytest.raises(ValueError, match='sole enabled'):
        partition_static_collision(stage)


@pytest.mark.parametrize('bad', ['overlap', 'missing', 'outside', 'unbound'])
def test_incomplete_or_ambiguous_materials_never_disable_the_source_collider(bad):
    stage, mesh = source_stage()
    subset = stage.GetPrimAtPath(SOURCE_COLLISION_PATH + '/Surface_1')
    if bad == 'overlap':
        subset.GetAttribute('indices').Set([0])
    elif bad == 'missing':
        subset.GetAttribute('indices').Set([])
    elif bad == 'outside':
        subset.GetAttribute('indices').Set([3])
    else:
        UsdShade.MaterialBindingAPI(subset).UnbindAllBindings()
    with pytest.raises(ValueError):
        partition_static_collision(stage)
    assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is True
    assert not stage.GetPrimAtPath(NATIVE_COLLISION_ROOT)

"""One static collision mesh per physical material, with source-face lineage.

The native backend can query our multi-material mesh while failing to generate
physical contacts for it. Splitting by material keeps the exact triangles and
coefficients, avoiding that GeomSubset simulation path without an SDK patch.
"""
from __future__ import annotations

import hashlib
import json


SOURCE_COLLISION_PATH = '/World/ground/terrain/Collision'
NATIVE_COLLISION_ROOT = '/World/ground/terrain/NativeCollision'
NATIVE_COLLISION_PATTERN = NATIVE_COLLISION_ROOT + '/Surface_.*'


def partition_static_collision(stage):
    """Run once before physics attachment; never rebuild a running terrain."""
    from pxr import UsdGeom, UsdPhysics, UsdShade

    source = UsdGeom.Mesh(stage.GetPrimAtPath(SOURCE_COLLISION_PATH))
    if (not source or stage.GetPrimAtPath(NATIVE_COLLISION_ROOT)
            or not source.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(source.GetPrim()).GetCollisionEnabledAttr().Get() is not True
            or UsdPhysics.MeshCollisionAPI(source.GetPrim()).GetApproximationAttr().Get() != 'none'):
        raise ValueError('static collision partition requires its sole enabled source triangle mesh')
    points = source.GetPointsAttr().Get()
    counts = list(source.GetFaceVertexCountsAttr().Get())
    indices = list(source.GetFaceVertexIndicesAttr().Get())
    if (not points or not counts or any(count != 3 for count in counts)
            or len(indices) != 3 * len(counts)
            or any(index < 0 or index >= len(points) for index in indices)):
        raise ValueError('static collision partition requires complete source triangles')
    geometry = json.dumps({'points': [list(point) for point in points], 'indices': indices},
                          separators=(',', ':'), allow_nan=False).encode()
    subsets = UsdShade.MaterialBindingAPI(source).GetMaterialBindSubsets()
    if not subsets:
        raise ValueError('the course source must declare its complete per-face physical materials')
    parts, covered = [], set()
    for subset in subsets:
        if subset.GetElementTypeAttr().Get() != 'face':
            raise ValueError('terrain material subsets must index faces')
        faces = tuple(int(value) for value in subset.GetIndicesAttr().Get())
        if (len(set(faces)) != len(faces) or covered.intersection(faces)
                or any(face < 0 or face >= len(counts) for face in faces)):
            raise ValueError('terrain material faces overlap or escape the source topology')
        covered.update(faces)
        if not faces:
            continue
        material, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial(materialPurpose='physics')
        if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            raise ValueError('each terrain partition needs its actual physics material')
        parts.append((subset.GetPrim().GetName(), faces, material))
    if covered != set(range(len(counts))):
        raise ValueError('terrain material partitions must cover every source face exactly once')

    root = UsdGeom.Xform.Define(stage, NATIVE_COLLISION_ROOT)
    # The source and new root are siblings. Preserve a non-identity local pose
    # without baking or rounding vertices, including any authored reset stack.
    source_transform = UsdGeom.Xformable(source).GetLocalTransformation()
    root.MakeMatrixXform().Set(source_transform)
    root.SetResetXformStack(UsdGeom.Xformable(source).GetResetXformStack())
    records = []
    for name, faces, material in parts:
        path = NATIVE_COLLISION_ROOT + '/' + name
        mesh = UsdGeom.Mesh.Define(stage, path)
        vertices = sorted({indices[3 * face + corner] for face in faces for corner in range(3)})
        remap = {source_index: local for local, source_index in enumerate(vertices)}
        mesh.CreatePointsAttr([points[index] for index in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([remap[indices[3 * face + corner]]
                                         for face in faces for corner in range(3)])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateOrientationAttr(source.GetOrientationAttr().Get())
        mesh.CreateDoubleSidedAttr(source.GetDoubleSidedAttr().Get())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
        lineage = json.dumps(faces, separators=(',', ':')).encode()
        records.append({'path': path, 'material_path': str(material.GetPath()),
                        'source_subset': str(source.GetPath()) + '/' + name,
                        'faces': len(faces), 'vertices': len(vertices),
                        'source_vertex_indices_sha256': hashlib.sha256(
                            json.dumps(vertices, separators=(',', ':')).encode()).hexdigest(),
                        'source_face_indices_sha256': hashlib.sha256(lineage).hexdigest()})
    # Switch only after all parts have been built. The referenced artifact and
    # its hash remain immutable; this override belongs to this runtime scene.
    UsdPhysics.CollisionAPI(source.GetPrim()).CreateCollisionEnabledAttr(False)
    source.CreateVisibilityAttr('invisible')
    return {'method': 'source_face_material_partitions', 'source_path': SOURCE_COLLISION_PATH,
            'collider_paths': [record['path'] for record in records], 'partitions': records,
            'source_faces': len(counts), 'native_faces': sum(record['faces'] for record in records),
            'source_geometry_sha256': hashlib.sha256(geometry).hexdigest(),
            'vertices_moved': False, 'faces_added_or_removed': False,
            'physical_materials_reused': True, 'original_collision_disabled': True,
            'rebuilt_during_rollout': False, 'physical_qualification': False}

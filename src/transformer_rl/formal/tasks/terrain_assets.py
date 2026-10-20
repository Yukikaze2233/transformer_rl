"""Source-backed static terrain preparation; no simulator or learning imports.

The original archive stays immutable. Collision cleanup and explicitly approved
training additions have distinct provenance; neither changes the visual source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def file_pin(path: Path) -> dict:
    raw = path.read_bytes()
    return {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _clip(poly, a, b, inside):
    """Clip a convex 3D polygon using a 2D oriented edge (projection in XY)."""
    result = []
    for p, q in zip(poly, (*poly[1:], poly[0])):
        edge = b[:2] - a[:2]
        dp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        dq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
        pin, qin = ((dp >= 0, dq >= 0) if inside else (dp <= 0, dq <= 0))
        if pin:
            result.append(p)
        if pin != qin:
            result.append(p + (q - p) * (dp / (dp - dq)))
    return result


def clean_collision(vertices, faces, *, plane_tolerance_m=1e-7):
    """Subtract same-facing coplanar overlap without closing holes or convexifying.

    Projection axes are chosen per source face. Only distances <= 0.1 micrometre
    are treated as coplanar; real millimetre markings and the 50mm gap remain.
    Output is recast to the exact float32 coordinates consumed by the backend.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0
            or faces.min() < 0 or faces.max() >= len(vertices) or len(faces) > 100_000):
        raise ValueError("invalid finite triangle geometry")
    output, lineage, previous = [], [], []
    degenerate, split_roundoff = [], []
    removed_area = 0.0
    source_area = 0.0
    for index, ids in enumerate(faces):
        tri = vertices[ids].astype(np.float64)
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        magnitude = np.linalg.norm(normal)
        if magnitude == 0:
            degenerate.append(index)
            continue
        source_area += magnitude / 2
        normal /= magnitude
        # A cyclic coordinate permutation preserves orientation and distance.
        axes = np.roll(np.arange(3), -((int(np.argmax(np.abs(normal))) + 1) % 3))
        tri2 = tri[:, axes]
        if np.linalg.det(np.stack((tri2[1, :2] - tri2[0, :2], tri2[2, :2] - tri2[0, :2]))) < 0:
            tri2 = tri2[[0, 2, 1]]
        pieces = [list(tri2)]
        low, high = tri.min(0), tri.max(0)
        for old, old_normal, old_low, old_high in previous:
            if (np.any(high < old_low) or np.any(old_high < low)
                    or np.dot(normal, old_normal) < 1 - 1e-10
                    or np.max(np.abs((old - tri[0]) @ normal)) > plane_tolerance_m):
                continue
            cutter = old[:, axes]
            if np.linalg.det(np.stack((cutter[1, :2] - cutter[0, :2], cutter[2, :2] - cutter[0, :2]))) < 0:
                cutter = cutter[[0, 2, 1]]
            intersection = list(tri2)
            for a, b in zip(cutter, np.roll(cutter, -1, axis=0)):
                if len(intersection) < 3:
                    break
                intersection = _clip(intersection, a, b, True)
            if len(intersection) < 3:
                continue
            intersection_area = sum(np.linalg.norm(np.cross(
                intersection[j] - intersection[0], intersection[j + 1] - intersection[0])) / 2
                for j in range(1, len(intersection) - 1))
            if intersection_area <= 1e-14:
                continue
            retained = []
            for piece in pieces:
                inner = piece
                for a, b in zip(cutter, np.roll(cutter, -1, axis=0)):
                    if len(inner) < 3:
                        break
                    outer = _clip(inner, a, b, False)
                    if len(outer) >= 3:
                        retained.append(outer)
                    inner = _clip(inner, a, b, True)
            pieces = retained
            if not pieces:
                break
        area = 0.0
        for poly in pieces:
            for j in range(1, len(poly) - 1):
                candidate = np.array((poly[0], poly[j], poly[j + 1]))[:, np.argsort(axes)]
                cross = np.cross(candidate[1] - candidate[0], candidate[2] - candidate[0])
                if np.dot(cross, normal) < 0:
                    candidate = candidate[[0, 2, 1]]
                candidate = candidate.astype(np.float32)
                area_now = np.linalg.norm(np.cross(
                    candidate[1].astype(float) - candidate[0],
                    candidate[2].astype(float) - candidate[0])) / 2
                if area_now == 0:
                    split_roundoff.append(index)
                    continue
                output.append(candidate)
                lineage.append(index)
                area += area_now
        removed_area += max(0.0, magnitude / 2 - area)
        previous.append((tri, normal, low, high))
    if not output:
        raise ValueError("collision preparation removed the entire surface")
    unique, inverse = np.unique(np.concatenate(output), axis=0, return_inverse=True)
    return unique, inverse.reshape(-1, 3), np.asarray(lineage, np.int64), {
        "input_triangles": len(faces), "output_triangles": len(output),
        "float32_zero_area_source_faces": degenerate,
        "split_zero_area_source_faces": sorted(set(split_roundoff)),
        "source_area_m2": source_area, "removed_overlap_area_m2": removed_area,
        "plane_tolerance_m": plane_tolerance_m,
        "filled_holes": False, "convexified": False, "scaled": False,
    }


def extend_front_contact(source_vertices, source_faces, specification):
    """Close only the reviewed front's lower gap, keeping its actual plane.

    The source's 50..200mm panel remains. Two added triangles connect its lower
    edge to Z=0. No top, rear, side, or under-platform surface is invented.
    """
    extension = specification['front_collision_extension']
    selected = np.asarray(extension['source_face_indices'], dtype=np.int64)
    triangles = np.asarray(source_vertices, float)[np.asarray(source_faces)[selected]]
    approach = specification['approach_axis']
    if approach not in ('+X', '+Y') or extension['scope'] != 'front_only_close_50mm_gap':
        raise ValueError('unsupported front collision extension')
    along, across = (0, 1) if approach == '+X' else (1, 0)
    points = triangles.reshape(-1, 3)
    bottom, top = points[:, 2].min(), points[:, 2].max()
    low, high = points[:, across].min(), points[:, across].max()
    interval = extension['height_interval_m']
    if (len(interval) != 2 or interval != [0., .2]
            or abs(bottom - specification['underpanel_gap_m']) > 1e-5
            or abs(top - interval[1]) > 1e-5
            or abs(high - low - specification['width_m']) > 1e-5):
        raise ValueError('reviewed panel dimensions differ from the approved extension')
    # Fit the measured panel plane; retain its tiny source rotation/roundoff.
    design = np.column_stack((points[:, across], points[:, 2], np.ones(len(points))))
    plane = np.linalg.lstsq(design, points[:, along], rcond=None)[0]
    if np.max(np.abs(design @ plane - points[:, along])) > 1e-6:
        raise ValueError('reviewed front faces are not a single plane')
    vertices = np.zeros((4, 3), np.float32)
    vertices[:, across] = (low, high, high, low)
    vertices[:, 2] = (interval[0], interval[0], bottom, bottom)
    vertices[:, along] = np.column_stack((vertices[:, across], vertices[:, 2], np.ones(4))) @ plane
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), np.int64)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    if normal[along] > 0:
        faces = faces[:, [0, 2, 1]]
    return vertices, faces, {
        'kind': 'approved_front_gap_closure', 'source_face_indices': selected.tolist(),
        'source_mesh_index': extension['source_mesh_index'], 'added_triangles': 2,
        'added_height_interval_m': [interval[0], float(bottom)],
        'result_front_height_interval_m': interval, 'width_m': float(high - low),
        'plane_along_from_lateral_z_constant': plane.tolist(),
        'visual_source_changed': False, 'other_gap_regions_filled': False,
        'authorization': extension['authorization'],
    }


def write_collision_usda(path, vertices, faces, *, static_friction, dynamic_friction,
                         restitution=0.0, visual_reference=None, face_material=None,
                         material_coefficients=None):
    """One static Mesh is also the unambiguous RayCaster target."""
    points = ",\n".join("(" + ", ".join(format(float(x), ".9g") for x in p) + ")" for p in vertices)
    indices = ", ".join(str(int(i)) for i in np.asarray(faces).ravel())
    visual = ("" if visual_reference is None else
              f'    def Xform "Visual" (prepend references = @{visual_reference}@</Terrain/Visual>) {{}}\n')
    material_text, subsets = '', ''
    if face_material is not None:
        ids = np.asarray(face_material)
        if ids.shape != (len(faces),):
            raise ValueError('one physical material is required per triangle')
        for index, coefficient in enumerate(material_coefficients):
            material_text += f'''    def Material "Surface_{index}" (prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]) {{
        float physics:staticFriction = {float(coefficient)}
        float physics:dynamicFriction = {float(coefficient)}
        float physics:restitution = 0
        token physxMaterial:frictionCombineMode = "multiply"
        token physxMaterial:restitutionCombineMode = "multiply"
    }}
'''
            selected = ', '.join(map(str, np.flatnonzero(ids == index)))
            subsets += f'''        def GeomSubset "Surface_{index}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [{selected}]
            rel material:binding:physics = </Terrain/Surface_{index}>
        }}
'''
    text = f'''#usda 1.0
(
    defaultPrim = "Terrain"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Terrain" {{
{visual}{material_text}    def Material "Material" (prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]) {{
        float physics:staticFriction = {float(static_friction)}
        float physics:dynamicFriction = {float(dynamic_friction)}
        float physics:restitution = {float(restitution)}
        token physxMaterial:frictionCombineMode = "multiply"
        token physxMaterial:restitutionCombineMode = "multiply"
    }}
    def Mesh "Collision" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]) {{
        point3f[] points = [{points}]
        int[] faceVertexCounts = [{', '.join(['3'] * len(faces))}]
        int[] faceVertexIndices = [{indices}]
        uniform token subdivisionScheme = "none"
        token visibility = "{'invisible' if visual_reference is not None else 'inherited'}"
        bool physics:collisionEnabled = true
        token physics:approximation = "none"
        rel material:binding:physics = </Terrain/Material>
        uniform token subsetFamily:materialBind:familyType = "nonOverlapping"
{subsets}
    }}
}}
'''
    Path(path).write_text(text)


def prepare_assets(archive, configuration, output):
    archive, configuration, output = Path(archive), Path(configuration), Path(output)
    config = json.loads(configuration.read_text())
    archive_pin = file_pin(archive)
    if archive_pin["sha256"] != config["package_sha256"]:
        raise ValueError("source package SHA256 differs from the reviewed archive")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"format": "prepared_terrain_assets_v1", "source_package": archive_pin,
                "processor": file_pin(Path(__file__)),
                "configuration": file_pin(configuration), "assets": {},
                "robot_qualified": False, "native_validation": "not_run"}
    try:
        with zipfile.ZipFile(archive) as source:
            for name, spec in config["assets"].items():
                folder = output / name
                folder.mkdir()
                root = config["archive_root"] + "/" + spec["directory"]
                raw_manifest = source.read(root + "/manifest.json")
                original = json.loads(raw_manifest)
                for member, pin_key in (("geometry.npz", "geometry_npz"), ("terrain.usdc", "USD")):
                    raw = source.read(root + "/" + member)
                    expected = original[pin_key]
                    if (len(raw) != expected["size_bytes"]
                            or hashlib.sha256(raw).hexdigest() != expected["sha256"]):
                        raise ValueError(f"source member identity differs: {name}/{member}")
                geometry_raw = source.read(root + "/geometry.npz")
                with np.load(io.BytesIO(geometry_raw), allow_pickle=False) as arrays:
                    vertices, faces, lineage, report = clean_collision(arrays["vertices_m"], arrays["triangles"])
                    source_mesh = arrays["source_mesh_index"][lineage]
                    source_triangle = arrays["source_triangle_index"][lineage]
                    face_kind = np.zeros(len(faces), np.uint8)
                    additions = []
                    if 'front_collision_extension' in spec:
                        expected_mesh = spec['front_collision_extension']['source_mesh_index']
                        selected = spec['front_collision_extension']['source_face_indices']
                        if not np.all(arrays['source_mesh_index'][selected] == expected_mesh):
                            raise ValueError('front extension source mesh binding differs')
                        extra_v, extra_f, addition = extend_front_contact(arrays['vertices_m'], arrays['triangles'], spec)
                        faces = np.concatenate((faces, extra_f + len(vertices)))
                        vertices = np.concatenate((vertices, extra_v))
                        lineage = np.concatenate((lineage, np.full(2, -1, np.int64)))
                        source_mesh = np.concatenate((source_mesh, np.full(2, -1, np.int64)))
                        source_triangle = np.concatenate((source_triangle, np.full(2, -1, np.int64)))
                        face_kind = np.concatenate((face_kind, np.ones(2, np.uint8)))
                        additions.append(addition)
                    report['training_additions'] = additions
                    report['final_output_triangles'] = len(faces)
                    report['cleanup_filled_holes'] = False
                    report['filled_holes'] = bool(additions)
                    report['filled_holes_scope'] = 'approved_front_lower_gap_only' if additions else None
                np.savez_compressed(folder / "collision.npz", vertices_m=vertices, triangles=faces,
                                    output_to_input_face=lineage, source_mesh_index=source_mesh,
                                    source_triangle_index=source_triangle, derived_face_kind=face_kind)
                (folder / "source.usdc").write_bytes(source.read(root + "/terrain.usdc"))
                (folder / "source_manifest.json").write_bytes(raw_manifest)
                (folder / "NOTICE.md").write_bytes(source.read(root + "/NOTICE.md"))
                material = {k: original["physical_material"][k]
                            for k in ("static_friction", "dynamic_friction", "restitution")}
                write_collision_usda(folder / "terrain.usda", vertices, faces,
                                     **material, visual_reference="source.usdc")
                manifest["assets"][name] = {
                    "specification": spec, "source_geometry_sha256": hashlib.sha256(geometry_raw).hexdigest(),
                    "source_usd_sha256": original["USD"]["sha256"], "processing": report,
                    "material": {**material, "measured": False, "combine_mode": "multiply"},
                    "files": {p.name: file_pin(p) for p in sorted(folder.iterdir())},
                }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        # Preserve a partial result for diagnosis; never make it look complete.
        (output / "INCOMPLETE").write_text("Asset preparation failed; do not load this directory.\n")
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare immutable source terrain derivatives without a simulator")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_assets(args.archive, args.configuration, args.output)
    print(json.dumps({name: record["processing"] for name, record in manifest["assets"].items()}, indent=2))


if __name__ == "__main__":
    main()

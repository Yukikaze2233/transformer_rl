#!/usr/bin/env python3
"""CPU-only export comparison; write only the explicitly selected evidence JSON.

Reuse audited pure functions through AST extraction, never import old report
modules or run their entry points. No simulator, asset conversion or GPU imports.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree


OLD = Path('/home/yukikaze/Downloads/urdf_V4.0 /urdf_V4.0')
NEW = Path('/home/yukikaze/Downloads/urdf_V4.0  (2)/urdf_V4.0')
LEGACY = Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train')
MATH_SOURCE = LEGACY / 'tools/prepare_v40_assets.py'
BORE_SOURCE = LEGACY / 'reports/knee_source_comparison/bore_witness.py'
STL_DTYPE = np.dtype([
    ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2'),
])


def load_pure_functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    folders = [OLD, NEW, LEGACY / 'assets/urdf_v40',
               LEGACY.with_name(LEGACY.name + '-60') / 'assets/urdf_v40',
               LEGACY / 'reports/knee_source_comparison']
    files = {p for folder in folders for p in folder.rglob('*') if p.is_file()}
    files.add(MATH_SOURCE)
    return {str(p): digest(p) for p in sorted(files)}


def xml_flat(root, prefix=''):
    key = prefix + '/' + root.tag
    if 'name' in root.attrib:
        key += '[' + root.get('name') + ']'
    result = {key + '/@' + k: v for k, v in root.attrib.items()}
    if root.text and root.text.strip():
        result[key + '/text'] = root.text.strip()
    counts = {}
    for child in root:
        identity = (child.tag, child.get('name'))
        counts[identity] = counts.get(identity, 0) + 1
        result.update(xml_flat(child, key + f'/{child.tag}#{counts[identity]}'))
    return result


def differences(a, b):
    return {k: {'old': a.get(k), 'new': b.get(k)}
            for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


def triangle_keys(tri):
    """Exact geometry ignoring face order, winding and stored STL normals."""
    ordered = np.empty_like(tri)
    for i, t in enumerate(tri):
        ordered[i] = t[np.lexsort((t[:, 2], t[:, 1], t[:, 0]))]
    rows = np.ascontiguousarray(ordered.astype('<f8').reshape(-1, 9))
    return np.sort(rows.view(np.dtype((np.void, rows.dtype.itemsize * 9))).ravel())


def geometry_hash(tri):
    return hashlib.sha256(triangle_keys(tri).tobytes()).hexdigest()


def mesh_stats(tri):
    vertices, faces, labels, count = connected_mesh(tri)
    components = []
    parts = []
    for i in range(count):
        ids = np.flatnonzero(labels[faces[:, 0]] == i)
        part = tri[ids]
        parts.append(part)
        v, f, _, _ = connected_mesh(part)
        components.append({
            'id': i, 'triangles': len(part), 'unique_vertices': len(v),
            'bbox_local_mm': (np.array([v.min(0), v.max(0)]) * 1000).tolist(),
            'geometry_sha256': geometry_hash(part),
            'source_triangle_indices_0based': ids.tolist(),
            **surface_stats(v, f),
        })
    return {
        'triangles': len(tri), 'unique_vertices': len(vertices),
        'component_count': count,
        'bbox_local_mm': (np.array([vertices.min(0), vertices.max(0)]) * 1000).tolist(),
        'geometry_sha256': geometry_hash(tri), 'components': components,
    }, parts


def vertex_distances(a, b):
    a = np.unique(a.reshape(-1, 3), axis=0)
    b = np.unique(b.reshape(-1, 3), axis=0)
    result = {}
    for label, src, dst in [('old_to_new', a, b), ('new_to_old', b, a)]:
        distances, _ = cKDTree(dst).query(src)
        i = int(distances.argmax())
        result[label] = {
            'max_mm': float(distances[i] * 1000),
            'p95_mm': float(np.quantile(distances, .95) * 1000),
            'vertices_over_1um': int((distances > 1e-6).sum()),
            'max_witness_local_mm': (src[i] * 1000).tolist(),
        }
    return result


def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {r['Link Name']: r for r in csv.DictReader(stream)}


def csv_inertial_check(rows, root):
    mismatches = []
    for link in root.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        fields = {'Mass': float(inertial.find('mass').get('value'))}
        for attr, value in inertial.find('inertia').attrib.items():
            fields['Moment ' + attr.capitalize()] = float(value)
        for label, attr in [('Center of Mass', 'xyz'), ('Center of Mass', 'rpy')]:
            axes = ['X', 'Y', 'Z'] if attr == 'xyz' else ['Roll', 'Pitch', 'Yaw']
            for axis, value in zip(axes, vec(inertial.find('origin').get(attr))):
                fields[label + ' ' + axis] = float(value)
        for field, value in fields.items():
            csv_value = float(rows[name][field])
            if not math.isclose(csv_value, value, rel_tol=1e-12, abs_tol=1e-15):
                mismatches.append({'link': name, 'field': field,
                                   'csv': csv_value, 'urdf': value})
    return mismatches


def bore_audit(root, meshes, c4_retained):
    results = {}
    for side, knee, qnom in [('L', 'L_joint2', .44890796255584864),
                             ('R', 'R_jonit2', -.4129956195280641)]:
        thigh, shin = meshes[f'{side}_link1'], meshes[f'{side}_link2']
        joint = root.find(f"joint[@name='{knee}']")
        lv = root.find(f"link[@name='{side}_link1']/visual")
        sv = root.find(f"link[@name='{side}_link2']/visual")
        entry = {'rod_identity_confirmed_by_exact_component_geometry': c4_retained[side]}
        rois = {
            'thigh_main': (thigh, [.180, -.030], [.240, .030]),
            'shin_main': (shin, [-.027, -.027], [.027, .027]),
            'shin_tip': (shin, [-.080, -.033], [-.050, -.003]),
        }
        if c4_retained[side]:
            rois['rod_tip'] = (thigh, [.230, .042], [.263, .074])
        for name, args in rois.items():
            try:
                entry[name] = fit_ring(*args)
            except (AssertionError, ValueError) as error:
                entry[name] = {'unresolved': str(error)}
        if 'rod_tip' not in entry or any('unresolved' in entry[k] for k in rois):
            results[side] = entry
            continue
        # Projected circular rings are interpreted as local Z axes only after
        # validating that their source vertices occupy multiple axial levels.
        for k in rois:
            assert np.ptp(entry[k]['ring_vertex_z_range_m']) > 1e-5
        entry['axis_model'] = 'Local Z from repeated XY ring across nonzero axial extent'
        a = np.r_[entry['rod_tip']['center_local_xy_m'], 0., 1.]
        b = np.r_[entry['shin_tip']['center_local_xy_m'], 0., 1.]
        entry['poses'] = {}
        sign = 1 if side == 'L' else -1
        poses = {'zero': 0., 'zero_knee_-5deg': -sign * math.radians(5),
                 'zero_knee_+5deg': sign * math.radians(5), 'nominal': qnom,
                 'nominal_knee_-5deg': qnom - sign * math.radians(5),
                 'nominal_knee_+5deg': qnom + sign * math.radians(5)}
        reference = forward(root, {knee: qnom})
        points = np.c_[thigh.reshape(-1, 3), np.ones(thigh.size // 3)]
        reference_points = points @ (reference[f'{side}_link1'] @ transform(lv)).T
        for pose, q in poses.items():
            fk = forward(root, {knee: q})
            rel = np.linalg.inv(fk[f'{side}_link1']) @ fk[f'{side}_link2']
            rod_frame, shin_frame = transform(lv), rel @ transform(sv)
            delta = (shin_frame @ b - rod_frame @ a)[:3]
            axis = rod_frame[:3, 2]
            parallel_error = np.linalg.norm(np.cross(axis, shin_frame[:3, 2]))
            assert parallel_error < 1e-10
            moved_points = points @ (fk[f'{side}_link1'] @ transform(lv)).T
            entry['poses'][pose] = {
                'raw_knee_rad': q,
                'parallel_axis_cross_norm': float(parallel_error),
                'transverse_axis_offset_mm': float(np.linalg.norm(
                    delta - np.dot(delta, axis) * axis) * 1000),
                'thigh_mesh_displacement_under_knee_motion_mm': float(np.linalg.norm(
                    moved_points[:, :3] - reference_points[:, :3], axis=1).max() * 1000),
            }
        entry['main_knee_center_offset_mm'] = {
            'thigh': float(np.linalg.norm(np.array(entry['thigh_main']['center_local_xy_m'])
                                          - vec(joint.find('origin').get('xyz'))[:2]) * 1000),
            'shin': float(np.linalg.norm(entry['shin_main']['center_local_xy_m']) * 1000),
        }
        yaw = vec(joint.find('origin').get('rpy'))[2]
        entry['inner_angle_mapping'] = {
            'zero_inner_deg': float(math.degrees(math.pi + yaw)),
            'raw_sign': sign,
            'raw_at_inner_35_80_rad': [float(sign * (math.radians(k) - math.pi - yaw))
                                       for k in (35, 80)],
            'old_nominal_inner_deg': float(math.degrees(math.pi + yaw + sign * qnom)),
        }
        results[side] = entry
    return results


def compact_report(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'source_triangle_indices_0based':
                encoded = json.dumps(item, separators=(',', ':')).encode()
                result['source_triangle_indices_summary'] = {
                    'count': len(item), 'sha256': hashlib.sha256(encoded).hexdigest(),
                    'omitted_from_summary': True,
                }
            else:
                result[key] = compact_report(item)
        return result
    if isinstance(value, list):
        return [compact_report(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compact-existing', type=Path,
                        help='Summarize an existing audit without rereading robot assets')
    args = parser.parse_args()
    assert args.output.parent.is_dir()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite an audit result')
    if args.compact_existing:
        raw = args.compact_existing.read_bytes()
        result = compact_report(json.loads(raw))
        result['summary_scope'] = 'Face index arrays omitted; measurements and source identities retained'
        result['full_report_sha256'] = hashlib.sha256(raw).hexdigest()
        result['full_report_path'] = str(args.compact_existing)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size}))
        return
    load_pure_functions(MATH_SOURCE, ['vec', 'rotation', 'origin', 'transform',
                                    'axis_rotation', 'topology', 'forward',
                                    'stl_triangles', 'connected_mesh', 'surface_stats'])
    load_pure_functions(BORE_SOURCE, ['fit_ring'])
    before = snapshot()
    result = {
        'scope': 'CPU static export audit; no simulator, GPU, training or asset writes',
        'sources': {'old': str(OLD), 'new': str(NEW)},
        'pure_function_sources_sha256': {str(p): digest(p) for p in [MATH_SOURCE, BORE_SOURCE]},
        'audit_script_sha256': digest(Path(__file__)),
        'files': {}, 'urdf': {}, 'csv': {}, 'meshes': {},
    }
    inventories = {label: {str(p.relative_to(folder)): p for p in folder.rglob('*')
                           if p.is_file() and p.suffix.lower() in {'.urdf', '.csv', '.stl'}}
                   for label, folder in [('old', OLD), ('new', NEW)]}
    roots = {}
    meshes = {'old': {}, 'new': {}}
    retained = {}
    for rel in sorted(inventories['old'].keys() | inventories['new'].keys()):
        files = {k: v.get(rel) for k, v in inventories.items()}
        result['files'][rel] = {k: None if p is None else {
            'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size}
            for k, p in files.items()}
        result['files'][rel]['byte_equal'] = all(files.values()) and digest(files['old']) == digest(files['new'])
        if not all(files.values()):
            continue
        if rel.endswith('.urdf'):
            pair = {k: ET.parse(p).getroot() for k, p in files.items()}
            roots = pair
            result['urdf'][rel] = {'attribute_diff': differences(*[xml_flat(pair[k]) for k in ('old', 'new')])}
            for k, root in pair.items():
                links, joints, base = topology(root)
                assert len(forward(root)) == len(links)
                result['urdf'][rel][k] = {
                    'link_count': len(links), 'joint_count': len(joints), 'root': base,
                    'joints': {n: xml_flat(j) for n, j in joints.items()},
                    'links': {n: xml_flat(l) for n, l in links.items()},
                    'mimic_count': len(root.findall('.//mimic')),
                    'transmission_count': len(root.findall('transmission')),
                }
        elif rel.lower().endswith('.stl'):
            print('Comparing ' + rel, flush=True)
            pair = {k: stl_triangles(p) for k, p in files.items()}
            stats, parts = {}, {}
            for k, tri in pair.items():
                meshes[k][Path(rel).stem] = tri
                stats[k], parts[k] = mesh_stats(tri)
            matches = {str(c['id']): [d['id'] for d in stats['new']['components']
                                    if c['geometry_sha256'] == d['geometry_sha256']]
                       for c in stats['old']['components']}
            stats['old_component_exact_geometry_matches_new'] = matches
            stats['exact_unoriented_triangle_geometry_equal'] = stats['old']['geometry_sha256'] == stats['new']['geometry_sha256']
            stats['vertex_distances'] = vertex_distances(pair['old'], pair['new'])
            stats['unique_triangle_intersection_count'] = len(np.intersect1d(
                triangle_keys(pair['old']), triangle_keys(pair['new'])))
            if Path(rel).stem in ('L_link1', 'R_link1'):
                retained[Path(rel).stem[0]] = bool(matches.get('4'))
            result['meshes'][rel] = stats
        elif rel.endswith('.csv'):
            pair = {k: csv_read(p) for k, p in files.items()}
            result['csv'][rel] = {
                'field_diff': differences(
                    {n + '/' + f: v for n, r in pair['old'].items() for f, v in r.items()},
                    {n + '/' + f: v for n, r in pair['new'].items() for f, v in r.items()}),
                'rows': pair,
            }
    for item in result['csv'].values():
        item['csv_vs_urdf_inertial_mismatches'] = {
            k: csv_inertial_check(rows, roots[k]) for k, rows in item['rows'].items()}
    result['bores'] = {
        'old': bore_audit(roots['old'], meshes['old'], {'L': True, 'R': True}),
        'new': bore_audit(roots['new'], meshes['new'], retained),
    }
    checks = {}
    for rel, item in result['meshes'].items():
        for version in ('old', 'new'):
            checks[f'{rel}/{version}/component_faces_cover_mesh'] = (
                sum(c['triangles'] for c in item[version]['components'])
                == item[version]['triangles'])
        if result['files'][rel]['byte_equal']:
            checks[f'{rel}/byte_equal_implies_geometry_equal'] = (
                item['exact_unoriented_triangle_geometry_equal']
                and all(d['max_mm'] == 0 for d in item['vertex_distances'].values()))
    for version, sides in result['bores'].items():
        for side, entry in sides.items():
            if 'poses' not in entry:
                continue
            # Independent 2-D closed-form rotation checks the FK axis witness.
            joint_name = 'L_joint2' if side == 'L' else 'R_jonit2'
            joint = roots[version].find(f"joint[@name='{joint_name}']")
            yaw = vec(joint.find('origin').get('rpy'))[2]
            axis_sign = vec(joint.find('axis').get('xyz'))[2]
            anchor = vec(joint.find('origin').get('xyz'))[:2]
            tip = np.array(entry['shin_tip']['center_local_xy_m'])
            rod = np.array(entry['rod_tip']['center_local_xy_m'])
            errors = []
            for pose in entry['poses'].values():
                theta = yaw + axis_sign * pose['raw_knee_rad']
                c, s = math.cos(theta), math.sin(theta)
                center = anchor + np.array([[c, -s], [s, c]]) @ tip
                errors.append(abs(np.linalg.norm(center - rod) * 1000
                                  - pose['transverse_axis_offset_mm']))
            entry['independent_2d_fk_max_error_mm'] = float(max(errors))
            checks[f'{version}/{side}/independent_2d_fk'] = bool(max(errors) < 1e-9)
    assert all(checks.values()), checks
    result['verification'] = checks
    after = snapshot()
    result['integrity'] = {'protected_file_count': len(before),
                           'before_sha256': before, 'after_sha256': after,
                           'unchanged': before == after}
    assert before == after
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'files': {k: {'byte_equal': v['byte_equal'], 'sha256': v['new']['sha256']}
                  for k, v in result['files'].items()},
        'urdf_diff': {k: v['attribute_diff'] for k, v in result['urdf'].items()},
        'mesh_summary': {k: {f: v['new'][f] for f in ['triangles', 'unique_vertices',
                                                    'component_count', 'bbox_local_mm']}
                         for k, v in result['meshes'].items()},
        'thigh_c4': {side: result['meshes'][f'meshes/{side}_link1.STL']['new']['components'][4]
                    | {'source_triangle_indices_0based': 'See evidence JSON'} for side in ('L', 'R')},
        'bores': {s: {k: {f: v[k][f] for f in ['center_local_xy_m', 'radius_m',
                                              'max_radial_fit_residual_m', 'ring_vertex_z_range_m']}
                      for k in ['rod_tip', 'shin_tip', 'thigh_main', 'shin_main'] if k in v}
                  for s, v in result['bores']['new'].items()},
        'csv_inertial': {k: v['csv_vs_urdf_inertial_mismatches'] for k, v in result['csv'].items()},
        'protected_file_count': len(before), 'verification_count': len(checks),
        'protected_files_unchanged': before == after,
    }, indent=2))


if __name__ == '__main__':
    main()

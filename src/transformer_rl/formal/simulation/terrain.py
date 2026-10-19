"""Private zero-step terrain editing on an already admitted owned context.

This implements authoring and handoff, not its SDK qualification. Visible
native state equality does not prove preservation of hidden solver caches.
Real cooking/contact/unchanged-trajectory acceptance is still required before
formal training. There is no global stop, reset, play, app update or sim step.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
import struct
import time

import torch

from .contacts import NativeTerrainContacts
from .ground import NativeGroundQuery
from .scene import NativeSceneRecipe, NativeSceneState
from ..tasks.terrain import TerrainCourse, TerrainRequest


@dataclass(frozen=True)
class TerrainEditTuning:
    cooking_timeout_s: float
    poll_interval_s: float
    vertex_encoding_tolerance_m: float
    native_query_offset_m: float

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in vars(self).values())
                or self.cooking_timeout_s > 60 or self.poll_interval_s > min(.1, self.cooking_timeout_s)):
            raise ValueError('terrain edit needs explicit bounded cooking and native encoding tolerances')


class NativeShapeQueries:
    """Require native path hits, not USD existence or echoed filter patterns.

    One interior triangle probe per collider checks presence/retirement only.
    It does not qualify every triangle, material, contact or solver behavior.
    """

    def __init__(self, query, offset_m):
        if (not callable(getattr(query, 'raycast_all', None)) or type(offset_m) not in (int, float)
                or not math.isfinite(offset_m) or offset_m <= 0):
            raise ValueError('native collider presence requires scene-query support and an explicit positive ray offset')
        self.query, self.offset = query, offset_m

    def __call__(self, paths, meshes, origins, *, required_present):
        if (type(required_present) is not bool or not isinstance(origins, torch.Tensor) or origins.ndim != 2
                or origins.shape != (len(paths), 3) or origins.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(origins).all() or not paths or len(meshes) != len(paths)
                or any(not p or len(p) != len(m) for p, m in zip(paths, meshes))):
            raise ValueError('native path probes require every nonempty collider row and finite world origins')
        count = 0
        for row, origin in enumerate(origins.detach().cpu().tolist()):
            for path, mesh in zip(paths[row], meshes[row]):
                point, normal = None, None
                for face in mesh.triangles:
                    a, b, c = [tuple(struct.unpack('<f', struct.pack('<f', x))[0] for x in mesh.vertices_m[i]) for i in face]
                    u, v = tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(a, c))
                    cross = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
                    length = math.sqrt(sum(x*x for x in cross))
                    if length > 1e-12:
                        normal = tuple(x / length for x in cross)
                        point = tuple((a[i] + b[i] + c[i]) / 3 + origin[i] for i in range(3))
                        break
                if point is None:
                    raise ValueError('native collider has no usable point3f triangle for a presence probe')
                found = [False]

                def report(hit):
                    found[0] |= str(hit.collision) == path
                    return True  # Do not let another robot/terrain occlude the own-path check.

                start = tuple(p + n*self.offset for p, n in zip(point, normal))
                result = self.query.raycast_all(start, tuple(-n for n in normal), self.offset*2, report, True)
                if (found[0] and not result) or found[0] != required_present:
                    raise RuntimeError('native collider path presence or retirement differs from the edited source')
                count += 1
        return {'colliders_queried': count, 'required_present': required_present,
                'every_triangle_or_physical_contact_equivalence': False}


class UsdTerrainAccess:
    """The SDK adapter is constructed only after external context admission."""

    def __init__(self, stage, geometry_api, physics_api, material_api, materials):
        self.stage, self.geometry_api, self.physics_api = stage, geometry_api, physics_api
        self.material_api, self.materials = material_api, materials

    def exists(self, path):
        return self.stage.GetPrimAtPath(path).IsValid()

    def author(self, root, mesh):
        return NativeSceneRecipe._author_terrain_mesh(self.stage, root, mesh, self.geometry_api,
                                                       self.physics_api, self.materials, enabled=False)

    def enabled(self, path, value):
        if not self.physics_api.CollisionAPI(self.stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Set(value):
            raise RuntimeError('owned terrain collider enabled state could not be authored')

    def verify(self, path, mesh, *, enabled, tolerance):
        prim = self.stage.GetPrimAtPath(path)
        geometry = self.geometry_api.Mesh(prim)
        points = tuple(tuple(float(x) for x in v) for v in geometry.GetPointsAttr().Get())
        encoded = tuple(tuple(struct.unpack('<f', struct.pack('<f', x))[0] for x in v) for v in mesh.vertices_m)
        error = max(abs(x - y) for a, b in zip(encoded, mesh.vertices_m) for x, y in zip(a, b))
        if (points != encoded or error > tolerance
                or tuple(geometry.GetFaceVertexCountsAttr().Get()) != (3,) * len(mesh.triangles)
                or tuple(geometry.GetFaceVertexIndicesAttr().Get()) != tuple(i for f in mesh.triangles for i in f)
                or geometry.GetSubdivisionSchemeAttr().Get() != 'none'
                or geometry.GetDoubleSidedAttr().Get() is not False
                or self.physics_api.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not enabled
                or self.physics_api.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'none'):
            raise ValueError('authored collider topology, enabled state or point3f encoding differs from source')
        root = path.removesuffix('/geometry/mesh')
        targets = tuple(str(p) for p in prim.GetRelationship('material:binding:physics').GetTargets())
        if targets != (self.materials.path_for_friction(mesh.friction),):
            raise ValueError('terrain collider is not bound to its exact shared physical material')
        material = self.stage.GetPrimAtPath(targets[0])
        value = self.physics_api.MaterialAPI(material)
        friction = struct.unpack('<f', struct.pack('<f', mesh.friction))[0]
        if (value.GetStaticFrictionAttr().Get() != friction or value.GetDynamicFrictionAttr().Get() != friction
                or value.GetRestitutionAttr().Get() != 0.
                or self.material_api.PhysxMaterialAPI(material).GetFrictionCombineModeAttr().Get() != 'multiply'
                or self.material_api.PhysxMaterialAPI(material).GetRestitutionCombineModeAttr().Get() != 'multiply'):
            raise ValueError('authored terrain material readback differs from the requested physical material')
        return error

    def static_readback(self, paths):
        return NativeSceneRecipe._static_terrain_readback(self.stage, paths, self.physics_api.RigidBodyAPI)

    def retire(self, paths):
        roots = {p.removesuffix('/geometry/mesh') for p in paths}
        parents = {p.rsplit('/', 1)[0] for p in roots}
        for root in sorted(roots):
            prim = self.stage.GetPrimAtPath(root)
            collider = self.stage.GetPrimAtPath(root + '/geometry/mesh')
            children = {str(c.GetPath()) for c in prim.GetChildren()}
            if (children != {root + '/geometry'}
                    or self.physics_api.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is not False):
                raise ValueError('retired collider root contains unowned or missing children')
            if not self.stage.RemovePrim(root) or self.exists(root):
                raise RuntimeError('retired owned collider root was not removed')
            self.materials.release(root + '/geometry/mesh')
        for parent in sorted(parents):
            if re.search(r'/Terrain/generation_[0-9]+$', parent) and self.exists(parent):
                if not tuple(self.stage.GetPrimAtPath(parent).GetChildren()):
                    if not self.stage.RemovePrim(parent):
                        raise RuntimeError('empty owned terrain generation was not retired')


@dataclass(frozen=True)
class TerrainEditResult:
    old_scene: NativeSceneState
    scene: NativeSceneState
    old_ground: NativeGroundQuery
    ground: NativeGroundQuery
    changed_rows: tuple[int, ...]
    receipt: dict
    physical_qualification: bool = False


class NativeTerrainEditor:
    def __init__(self, scene, access, physics, simulation, cooking, *, stage_id_source,
                 physics_view_source, native_to_tensor, native_geometry_probe, tuning,
                 contacts_factory=NativeTerrainContacts):
        if (not isinstance(scene, NativeSceneState) or not isinstance(tuning, TerrainEditTuning)
                or not all(callable(v) for v in (stage_id_source, physics_view_source, native_to_tensor,
                                                native_geometry_probe, contacts_factory))):
            raise ValueError('terrain editor requires an admitted owned scene and complete native adapters')
        self.scene, self.access, self.physics, self.simulation, self.cooking = scene, access, physics, simulation, cooking
        self._stage_id, self._physics_view, self._convert = stage_id_source, physics_view_source, native_to_tensor
        self.tuning, self._contacts = tuning, contacts_factory
        self._probe = native_geometry_probe
        self._view, self._busy, self._faulted = scene.robot.root_view, False, False
        self._prim_paths = tuple(str(p) for p in self._view.prim_paths)
        # SDK可将根API绑定在Robot容器或base_link，之后仍要求完整路径不变。
        if (len(self._prim_paths) != scene.observation.num_envs
                or any(path not in (env + '/Robot', env + '/Robot/' + scene.observation.binding.native_body_names[0])
                       for path, env in zip(self._prim_paths, scene.environment_paths))):
            raise ValueError('native terrain editor requires its exact environment articulation roots')
        self.last_failure = None

    @classmethod
    def on_admitted_context(cls, sim, scene, tuning):
        from pxr import PhysxSchema, UsdGeom, UsdPhysics
        import omni.physx
        from omni.physx.bindings import _physx
        import warp as wp
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils.stage import get_current_stage_id
        from isaaclab_physx.physics import PhysxManager

        scene_api = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
        if scene_api.GetEnableSceneQuerySupportAttr().Get() is not True:
            raise ValueError('native terrain replacement requires scene-query support in the admitted configuration')
        return cls(scene, UsdTerrainAccess(sim.stage, UsdGeom, UsdPhysics, PhysxSchema, scene.terrain_materials),
                    omni.physx.get_physx_interface(), omni.physx.get_physx_simulation_interface(),
                    _physx.acquire_physx_cooking_private_interface(), stage_id_source=get_current_stage_id,
                    physics_view_source=PhysxManager.get_physics_sim_view, native_to_tensor=wp.to_torch,
                    native_geometry_probe=NativeShapeQueries(omni.physx.get_physx_scene_query_interface(),
                                                              tuning.native_query_offset_m), tuning=tuning)

    def _statistics(self):
        value = self.cooking.get_cooking_statistics()
        names = ('total_scheduled_tasks', 'total_finished_tasks', 'total_warnings_failed_gpu_compatibility',
                 'total_warnings_convex_polygon_limits_reached')
        result = {name: int(getattr(value, name)) for name in names}
        if min(result.values()) < 0 or result[names[1]] > result[names[0]]:
            raise RuntimeError('native cooking statistics are invalid or changed their lifetime')
        return result

    def _flush(self, baseline):
        deadline = time.monotonic() + self.tuning.cooking_timeout_s
        while True:
            self.simulation.flush_changes()
            value = self._statistics()
            if any(value[k] < baseline[k] for k in baseline):
                raise RuntimeError('native cooking counters rewound during the owned edit')
            if any(value[k] != baseline[k] for k in baseline if k.startswith('total_warnings')):
                raise RuntimeError('terrain cooking reported GPU compatibility or polygon-limit warnings')
            if value['total_scheduled_tasks'] == value['total_finished_tasks']:
                return value
            if time.monotonic() >= deadline:
                raise TimeoutError('owned terrain cooking did not complete within its explicit wall-clock bound')
            time.sleep(self.tuning.poll_interval_s)

    def _native_state(self):
        bridge = self.scene.observation
        bridge.before_physics_step()
        n = bridge.num_envs
        expected = {'get_root_transforms': (n, 7), 'get_root_velocities': (n, 6),
                    'get_link_transforms': (n, 19, 7), 'get_link_velocities': (n, 19, 6),
                    'get_dof_positions': (n, 18), 'get_dof_velocities': (n, 18),
                    'get_coms': (n, 19, 7), 'get_masses': (n, 19), 'get_dof_limits': (n, 18, 2)}
        result = {}
        for name, shape in expected.items():
            value = self._convert(getattr(self._view, name)())
            parameter = name in ('get_coms', 'get_masses', 'get_dof_limits')
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                    or not parameter and value.device != bridge.device
                    or (torch.isnan(value).any() if name == 'get_dof_limits' else not torch.isfinite(value).all())):
                raise ValueError('native terrain edit snapshot requires complete finite root/link/DOF/model state')
            result[name] = value.detach().clone()
        result['prim_paths'] = tuple(str(p) for p in self._view.prim_paths)
        result['joint_names'], result['body_names'] = tuple(self.scene.robot.joint_names), tuple(self.scene.robot.body_names)
        if (result['prim_paths'] != self._prim_paths
                or result['joint_names'] != bridge.binding.native_joint_names
                or result['body_names'] != bridge.binding.native_body_names):
            raise ValueError('native terrain edit cannot change the actual articulation clone or named topology')
        result['timestamp_s'] = bridge.timestamp_s().clone()
        result['physical_tick'], result['model_epoch'] = bridge.physical_tick, bridge.model_epoch
        return result

    @staticmethod
    def _same_state(first, last):
        return first.keys() == last.keys() and all((first[k].device == last[k].device and torch.equal(first[k], last[k]))
                                                   if isinstance(first[k], torch.Tensor)
                                                   else first[k] == last[k] for k in first)

    @torch.no_grad()
    def replace_courses(self, ground, courses, reset_mask, *, protected_xy_m=None):
        started = time.monotonic()
        scene = self.scene
        bridge, contacts = scene.observation, scene.contacts
        if (self._busy or self._faulted or scene.readback.get('terrain_binding_invalidated') is True
                or not isinstance(ground, NativeGroundQuery) or ground.scene is not scene
                or scene.robot.root_view is not self._view or self._physics_view() is not contacts.physics_view
                or not isinstance(reset_mask, torch.Tensor) or reset_mask.shape != (bridge.num_envs,)
                or reset_mask.dtype != torch.bool or reset_mask.device != bridge.device):
            raise RuntimeError('terrain edit needs the current nonfaulted exclusive scene and exact reset rows')
        courses = tuple(courses)
        rows = set(reset_mask.nonzero().flatten().tolist())
        if (len(courses) != bridge.num_envs
                or any(not isinstance(c, TerrainCourse) or not isinstance(c.request, TerrainRequest)
                       or type(c.geometry_seed) is not int or not 0 <= c.geometry_seed < 2**63 for c in courses)
                or any(courses[i] is not ground.courses[i] for i in range(bridge.num_envs) if i not in rows)
                or contacts.environment_paths != scene.environment_paths):
            raise ValueError('terrain edit may replace only selected native-order course rows')
        changed = tuple(i for i in sorted(rows) if courses[i].surface != ground.courses[i].surface)
        if not changed:
            return TerrainEditResult(scene, scene, ground, ground, (), {'physical_step_events': 0, 'native_edit': False})
        if (bridge.physical_tick % round(.01 / bridge.physics_dt_s)
                or not self.simulation.check_results() or self.simulation.is_change_tracking_paused()
                or self.simulation.get_attached_stage() != self._stage_id()):
            raise RuntimeError('terrain edit needs a complete published step and the same attached stage')
        before, baseline = self._native_state(), self._statistics()
        streaming = protected_xy_m is not None
        retained = {}
        if streaming:
            if (not isinstance(protected_xy_m, torch.Tensor) or protected_xy_m.shape != (bridge.num_envs, 2, 2)
                    or not torch.isfinite(protected_xy_m).all()
                    or (protected_xy_m[:, :, 0] >= protected_xy_m[:, :, 1]).any()):
                raise ValueError('live tiles require complete finite protected local XY bounds')
            bounds = protected_xy_m.detach().cpu().tolist()
            for row in changed:
                old, new = ground.courses[row], courses[row]
                if (old.tile_source is None or new.tile_source is not old.tile_source
                        or old.sha256 != new.sha256):
                    raise ValueError('live streaming cannot replace a task or its complete terrain source')
                required = set(old.tile_source.tile_keys(bounds[row][0], bounds[row][1], cells=8, halo=0))
                xy = before['get_link_transforms'][row, :, :2] - scene.environment_origins_m[row, :2]
                limits = xy.new_tensor(bounds[row])
                if ((xy < limits[:, 0]).any() or (xy > limits[:, 1]).any()):
                    raise ValueError('protected streaming region omits an actual native body origin')
                shared = set(old.resident_tiles) & set(new.resident_tiles)
                if not required <= shared:
                    raise ValueError('live terrain must retain every protected body-footprint collider')
                old_mesh = dict(zip(old.resident_tiles, scene.terrain_meshes[row]))
                paths = dict(zip(old.resident_tiles, contacts.terrain_paths_by_environment[row]))
                retained[row] = {}
                for key, mesh in zip(new.resident_tiles, new.surface.meshes):
                    if key in shared:
                        if mesh is not old_mesh[key]:
                            raise ValueError('retained live tiles must keep their original exact source mesh')
                        retained[row][key] = paths[key]
        if baseline['total_scheduled_tasks'] != baseline['total_finished_tasks']:
            raise RuntimeError('previous cooking tasks are still pending before this owned edit')
        epoch = scene.readback.get('terrain_epoch', 0)
        pending = scene.readback.get('terrain_reset_required_rows', ())
        if (type(epoch) is not int or epoch < 0 or not isinstance(pending, tuple)
                or any(type(row) is not int or not 0 <= row < bridge.num_envs for row in pending)):
            raise ValueError('native terrain epoch is invalid')
        new_paths = [tuple(v) for v in contacts.terrain_paths_by_environment]
        added = {row: [] for row in changed}
        removed = {row: [] for row in changed}
        for row in changed:
            env = scene.environment_paths[row]
            if self.access.exists(env + '/Terrain/generation_' + str(epoch + 1)):
                raise RuntimeError('new terrain generation already exists;never overwrite an unverified attempt')
            for path, mesh in zip(new_paths[row], scene.terrain_meshes[row]):
                if (not path.startswith(env + '/Terrain/') or not path.endswith('/geometry/mesh')
                        or len(new_paths[row]) != len(scene.terrain_meshes[row])):
                    raise ValueError('retired terrain paths differ from the exact owned collider layout')
                self.access.verify(path, mesh, enabled=True, tolerance=self.tuning.vertex_encoding_tolerance_m)
        events = [0]
        subscription = self.physics.subscribe_physics_step_events(lambda _dt: events.__setitem__(0, events[0] + 1))
        if subscription is None:
            raise RuntimeError('native zero-step terrain guard could not subscribe')
        self._busy = True
        scene.readback['terrain_binding_invalidated'] = True
        phase = 'author_disabled_geometry'
        try:
            for row in changed:
                root = scene.environment_paths[row] + '/Terrain/generation_' + str(epoch + 1)
                paths = []
                for i, mesh in enumerate(courses[row].surface.meshes):
                    key = courses[row].resident_tiles[i] if streaming else None
                    path = retained.get(row, {}).get(key)
                    if path is None:
                        path = self.access.author(root + '/surface_' + str(i), mesh)
                        self.access.verify(path, mesh, enabled=False, tolerance=self.tuning.vertex_encoding_tolerance_m)
                        added[row].append((path, mesh))
                    paths.append(path)
                new_paths[row] = tuple(paths)
                removed[row] = [(path, mesh) for path, mesh in
                    zip(contacts.terrain_paths_by_environment[row], scene.terrain_meshes[row]) if path not in paths]
            phase = 'switch_and_flush'
            for row in changed:
                for path, _ in removed[row]:
                    self.access.enabled(path, False)
                for path, _ in added[row]:
                    self.access.enabled(path, True)
            cooked = self._flush(baseline)
            phase = 'verify_new_native_paths'
            def probe(entries, present):
                rows = [row for row in changed if entries[row]]
                return (self._probe(tuple(tuple(p for p, _ in entries[row]) for row in rows),
                    tuple(tuple(m for _, m in entries[row]) for row in rows),
                    scene.environment_origins_m[rows], required_present=present) if rows else
                    {'colliders_queried': 0, 'required_present': present})
            presence = probe(added, True)
            if events[0] or not self._same_state(before, self._native_state()):
                raise RuntimeError('native editing advanced physics or changed visible dynamic/model state')
            phase = 'create_contact_views'
            new_contacts = self._contacts(contacts.physics_view, bridge.binding, scene.environment_paths,
                                           tuple(new_paths), native_to_tensor=self._convert, device=bridge.device,
                                           patches_per_body=contacts.patches_per_body,
                                           terrain_motion_reference='authored_static_own_terrain')
            retired = tuple(path for row in changed for path, _ in removed[row])
            phase = 'retire_previous_geometry'
            self.access.retire(retired)
            self._flush(baseline)
            absence = probe(removed, False)
            for row in changed:
                for path, mesh in zip(new_paths[row], courses[row].surface.meshes):
                    self.access.verify(path, mesh, enabled=True, tolerance=self.tuning.vertex_encoding_tolerance_m)
            if (events[0] or self._physics_view() is not contacts.physics_view
                    or scene.robot.root_view is not self._view or not self._same_state(before, self._native_state())):
                raise RuntimeError('terrain retirement changed the native view, clock or visible dynamic state')
            readback = dict(scene.readback)
            readback.update(self.access.static_readback(tuple(new_paths)))
            readback.update(terrain_epoch=epoch + 1, terrain_binding_invalidated=False,
                            terrain_reset_required_rows=tuple(sorted(set(pending) | (set() if streaming else set(changed)))),
                            live_terrain_edit_physically_qualified=False, physics_qualified=False)
            fresh = replace(scene, contacts=new_contacts, terrain_meshes=tuple(c.surface.meshes for c in courses),
                            readback=readback)
            design = [None] * bridge.num_envs
            for native, original in enumerate(scene.clone_indices):
                design[original] = courses[native]
            phase = 'bind_ground'
            new_ground = NativeGroundQuery(fresh, tuple(design), ground._geometry)
            receipt = {'terrain_epoch': epoch + 1, 'changed_native_rows': changed, 'physical_step_events': events[0],
                       'visible_native_state_equal': True, 'hidden_solver_cache_equivalence_verified': False,
                       'cooking_before': baseline, 'cooking_after': cooked, 'retired_paths': retired,
                       'new_paths': tuple(new_paths), 'editor_walltime_s': time.monotonic() - started,
                       'new_native_path_probe': presence, 'retired_native_path_probe': absence,
                       'live_tile_stream': streaming, 'retained_collider_count': sum(map(len, retained.values())),
                       'physical_qualification': False}
            self.scene = fresh
            return TerrainEditResult(scene, fresh, ground, new_ground, changed, receipt)
        except BaseException as problem:
            self._faulted = True
            self.last_failure = {'phase': phase, 'exception_type': type(problem).__name__, 'message': str(problem),
                                 'changed_native_rows': changed, 'physical_step_events': events[0],
                                 'old_binding_invalidated': True, 'automatic_resume_authorized': False,
                                 'editor_walltime_s': time.monotonic() - started}
            raise
        finally:
            self._busy = False
            subscription = None


class NativeTerrainInstaller:
    """Adopt the editor's new binding across one task graph before reset.

    The mechanical bridge, root view, clock, model epoch, learning histories,
    drive response and unselected contact histories stay owned by the same
    objects. Only terrain/contact/query bindings are exchanged. A partially
    edited or partially adopted graph faults rather than rolling back physics.
    """

    def __init__(self, environment, coordinator, editor):
        from .task import NativeTaskEnvironment

        if (not isinstance(environment, NativeTaskEnvironment) or not isinstance(editor, NativeTerrainEditor)
                or coordinator.lifecycle is not environment.lifecycle
                or editor.scene is not environment.lifecycle.plant.scene):
            raise ValueError('terrain installation requires one actual task graph and its owned source/editor')
        self.environment, self.coordinator, self.editor = environment, coordinator, editor
        self.lifecycle = environment.lifecycle
        self.last_result = None

    @torch.no_grad()
    def __call__(self, plan, reset_mask):
        env, life, owner = self.environment, self.lifecycle, self.coordinator
        env._identity()
        env.rewards._identity()
        owner._identity()
        bridge, plant = life.plant.scene.observation, life.plant
        proposal = owner._issued
        if (proposal is None or proposal.plan is not plan or not torch.equal(proposal.reset_mask, reset_mask)
                or not torch.equal(proposal.next_episode_id, life.episode_id + reset_mask.long())
                or not torch.equal(proposal.requested_at_s, bridge.timestamp_s())
                or life._stepped or life._pending_tick is not None
                or (reset_mask & life._last_reset_tick.eq(bridge.physical_tick)).any()
                or bridge.physical_tick != life._last_tick or bridge.physical_tick != env.rewards._last_tick
                or env.rewards.composer._window is not None or plan.source_courses is None):
            raise RuntimeError('terrain installation requires the exact sealed source at a drained policy boundary')
        env.commands.validate_plan(plan.commands)
        if plan.preparation_hold is not None and plan.preparation_issued_action is not None:
            raise ValueError('terrain reset preparation must have one unambiguous control representation')
        if plan.preparation_issued_action is not None:
            life._actions(plan.preparation_issued_action)
        for name, value, width in (('root pose', plan.root_pose_xyzw, 7), ('COM velocity', plan.root_com_velocity, 6),
                                   ('joint position', plan.joint_position, 18), ('joint velocity', plan.joint_velocity, 18)):
            plant._matrix(name, value, width)
        if plan.preparation_hold is not None:
            plant.controller.validate_preparation_hold(
                plan.preparation_hold, plan.joint_position[:, list(bridge.binding.policy_joint_indices)], reset_mask)
        if not torch.allclose(plan.root_pose_xyzw[:, 3:].norm(dim=-1), torch.ones(life.num_envs, device=life.device),
                              atol=1e-5, rtol=0):
            raise ValueError('terrain reset requires encoded normalized native xyzw')
        old, ground = plant.scene, life.ground
        if (len(plan.source_courses) != life.num_envs or any(not isinstance(c, TerrainCourse) for c in plan.source_courses)
                or tuple(c.sha256 for c in plan.source_courses) != tuple(plan.course_sha256)):
            raise ValueError('terrain plan source descriptions differ from the sealed requested course hashes')
        if self.editor.scene is not old or owner._scene is not old or env._scene is not old:
            raise RuntimeError('terrain owner graph changed before installation')
        try:
            result = self.editor.replace_courses(ground, plan.source_courses, reset_mask)
            self.adopt(result)
        except BaseException:
            env._faulted = life._faulted = plant._faulted = True
            raise

    def adopt(self, result):
        """原生编辑后的同一对象图交接；不reset物理、历史或课程计数。"""
        self.last_result = result
        if not result.changed_rows:
            return
        env, life, owner = self.environment, self.lifecycle, self.coordinator
        plant, fresh, old = life.plant, result.scene, life.plant.scene
        if (result.old_scene is not old or result.old_ground is not life.ground or self.editor.scene is not fresh
                or fresh.robot is not old.robot or fresh.observation is not old.observation
                or fresh.efforts is not old.efforts or fresh.contacts.physics_view is not old.contacts.physics_view
                or fresh.environment_paths != old.environment_paths or fresh.clone_indices != old.clone_indices
                or not torch.equal(fresh.environment_origins_m, old.environment_origins_m)):
            raise RuntimeError('new terrain result changed a mechanical owner, clock or clone binding')
        plant.scene, plant._ground = fresh, result.ground
        life.ground, life._scene, life._meshes = result.ground, fresh, fresh.terrain_meshes
        life._courses, life._course_sha256 = result.ground.courses, tuple(c.sha256 for c in result.ground.courses)
        env.rewards._scene, env.rewards._physics_view = fresh, fresh.contacts.physics_view
        env._scene, owner._scene = fresh, fresh
        env._identity()
        env.rewards._identity()
        owner._identity()

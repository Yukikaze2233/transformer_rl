"""Terrain editing orchestration with fake SDK storage on Kaiser CPU only.

These checks do not execute pxr, cooking, GPU contacts or real physics. The
actual USD adapter and hidden solver-state equivalence remain unqualified.
"""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from test_native_task import fixture as task_fixture, advance
from transformer_rl.formal.simulation.contacts import NativeTerrainContacts
from transformer_rl.formal.simulation.ground import NativeGroundQuery
from transformer_rl.formal.simulation.terrain import NativeShapeQueries, NativeTerrainEditor, NativeTerrainInstaller, TerrainEditTuning
from transformer_rl.formal.tasks.geometry import HeightPatch, TerrainSurface, TerrainKind
from transformer_rl.formal.tasks.terrain import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning


class Storage:
    def __init__(self, scene):
        self.records, self.writes, self.retired = {}, [], []
        self.fail_verify = False
        self.missing_native, self.ghost_native, self.ghosts = False, False, set()
        for row, meshes in enumerate(scene.terrain_meshes):
            for path, mesh in zip(scene.contacts.terrain_paths_by_environment[row], meshes):
                self.records[path] = [mesh, True]

    def exists(self, path):
        return path in self.records or any(p.startswith(path + '/') for p in self.records)

    def author(self, root, mesh):
        path = root + '/geometry/mesh'
        assert not self.exists(root)
        self.records[path] = [mesh, False]
        self.writes.append(('author', path))
        return path

    def enabled(self, path, value):
        self.records[path][1] = value
        self.writes.append(('enabled', path, value))

    def verify(self, path, mesh, *, enabled, tolerance):
        if self.fail_verify and '/generation_' in path:
            raise ValueError('explicit synthetic material readback mismatch')
        assert self.records[path] == [mesh, enabled]
        return 0.

    def retire(self, paths):
        assert all(not self.records[p][1] for p in paths)
        self.retired.extend(paths)
        for p in paths:
            del self.records[p]
            if self.ghost_native:self.ghosts.add(p)

    def probe(self, paths, meshes, origins, *, required_present):
        for row in paths:
            for path in row:
                present = path in self.ghosts or path in self.records and self.records[path][1]
                if self.missing_native and '/generation_' in path:present = False
                if present != required_present:
                    raise RuntimeError('explicit synthetic native-path presence mismatch')
        return {'required_present': required_present, 'colliders_queried': sum(len(row) for row in paths)}

    def static_readback(self, paths):
        assert all(p in self.records for row in paths for p in row)
        return {'terrain_motion_reference': 'authored_static_own_terrain',
                'native_static_collision_equivalence_verified': False}


def fixture(*, reverse=False, root_at_body=False):
    env, robot, sim, writes, _ = task_fixture()
    life, plant = env.lifecycle, env.lifecycle.plant
    old_scene, courses = plant.scene, life.ground.courses
    views = []

    def create(sensors, *, filter_patterns, max_contact_data_count):
        n, width = len(sensors), len(filter_patterns[0])
        view = SimpleNamespace(sensor_paths=sensors, filter_paths=filter_patterns, sensor_count=n,
                                 filter_count=width, max_contact_data_count=max_contact_data_count,
                                 get_contact_force_matrix=lambda *, dt: torch.zeros(n, width, 3),
                                 get_contact_data=lambda *, dt: (torch.full((max_contact_data_count, 1), float('nan')),
                                    torch.zeros(max_contact_data_count, 3), torch.zeros(max_contact_data_count, 3), None,
                                    torch.zeros(n, width, dtype=torch.int32), torch.full((n, width), -1, dtype=torch.int32)))
        views.append(view)
        return view

    physics_view = SimpleNamespace(create_rigid_contact_view=create, get_gravity=lambda: (0., 0., -9.81))
    paths = tuple(tuple(env_path + '/Terrain/surface_' + str(i) + '/geometry/mesh'
                         for i in range(len(course.surface.meshes))) for env_path, course in zip(old_scene.environment_paths, courses))
    contacts = NativeTerrainContacts(physics_view, old_scene.observation.binding, old_scene.environment_paths, paths,
                                     native_to_tensor=lambda v: v, terrain_motion_reference='authored_static_own_terrain')
    scene = replace(old_scene, contacts=contacts, clone_indices=(1, 0) if reverse else (0, 1))
    design = tuple(courses[i] for i in scene.clone_indices)
    ground = NativeGroundQuery(scene, design, life.ground._geometry)
    plant.scene, plant._ground = scene, ground
    life._scene, life.ground, life._meshes, life._courses = scene, ground, scene.terrain_meshes, ground.courses
    env._scene, env.rewards._scene, env.rewards._physics_view = scene, scene, physics_view
    view = robot.root_view
    suffix = '/Robot/base_link' if root_at_body else '/Robot'
    view.prim_paths = tuple(p + suffix for p in scene.environment_paths)
    view.get_root_transforms = lambda: robot.data.root_link_pose_w.torch
    view.get_root_velocities = lambda: robot.data.root_com_vel_w.torch
    view.get_link_transforms = lambda: robot.data.body_link_pose_w.torch
    view.get_link_velocities = lambda: robot.data.body_com_vel_w.torch
    # Continuous native bounds may legitimately be infinite. Preserve them,
    # rather than rejecting them as invalid live-state measurements.
    limits = view.get_dof_limits().clone()
    limits[:, 0] = torch.tensor([-float('inf'), float('inf')])
    view.get_dof_limits = lambda: limits
    physics = SimpleNamespace(callback=None)

    def subscribe(callback):
        physics.callback = callback
        return object()

    physics.subscribe_physics_step_events = subscribe
    statistics = SimpleNamespace(total_scheduled_tasks=0, total_finished_tasks=0,
                                  total_warnings_failed_gpu_compatibility=0, total_warnings_convex_polygon_limits_reached=0)
    cooking = SimpleNamespace(get_cooking_statistics=lambda: statistics)
    simulation = SimpleNamespace(flushes=0, on_flush=None, stage_id=42, tracking_paused=False)

    def flush():
        simulation.flushes += 1
        if simulation.on_flush is not None:
            simulation.on_flush()

    simulation.flush_changes, simulation.check_results = flush, lambda: True
    simulation.get_attached_stage = lambda: simulation.stage_id
    simulation.is_change_tracking_paused = lambda: simulation.tracking_paused
    storage = Storage(scene)
    current_view = [physics_view]
    editor = NativeTerrainEditor(scene, storage, physics, simulation, cooking, stage_id_source=lambda: 42,
                                   physics_view_source=lambda: current_view[0], native_to_tensor=lambda v: v,
                                   native_geometry_probe=storage.probe, tuning=TerrainEditTuning(.01, .001, 1e-5, .01))
    return env, robot, sim, writes, editor, storage, simulation, statistics, current_view, physics


def changed(courses, *, row=0, height=.05):
    values = list(courses)
    old = values[row]
    p = old.surface.patches[0]
    patch = replace(p, z_m=tuple(tuple(height for _ in r) for r in p.z_m))
    values[row] = replace(old, geometry_seed=old.geometry_seed + 1,
                           surface=TerrainSurface((patch,)))
    return tuple(values)


def test_sdk_root_on_base_link_is_bound_once_and_later_path_changes_are_rejected():
    env, robot, sim, _, editor, _, _, _, _, _ = fixture(root_at_body=True)
    result = editor.replace_courses(env.lifecycle.ground, changed(env.lifecycle.ground.courses),
                                     torch.tensor([True, False]))
    assert sim.steps == 0 and result.changed_rows == (0,)
    robot.root_view.prim_paths = tuple(p + '/Robot' for p in result.scene.environment_paths)
    with pytest.raises(ValueError, match='actual articulation clone'):
        editor._native_state()


@pytest.mark.parametrize('reverse', [False, True])
def test_changed_terrain_uses_fresh_shapes_and_views_without_advancing_any_visible_native_state(reverse):
    env, robot, sim, _, editor, storage, native, _, _, _ = fixture(reverse=reverse)
    old, ground = env.lifecycle.plant.scene, env.lifecycle.ground
    before = editor._native_state()
    result = editor.replace_courses(ground, changed(ground.courses), torch.tensor([True, False]))
    assert result.changed_rows == (0,) and sim.steps == 0 and native.flushes == 2
    assert result.receipt['physical_step_events'] == 0 and result.receipt['visible_native_state_equal']
    assert result.scene.readback['terrain_reset_required_rows'] == (0,)
    assert not result.receipt['hidden_solver_cache_equivalence_verified'] and not result.physical_qualification
    assert editor._same_state(before, editor._native_state())
    assert result.scene.robot is robot and result.scene.observation is old.observation
    assert result.scene.contacts.physics_view is old.contacts.physics_view
    assert result.ground.courses[1] is ground.courses[1]
    assert result.scene.contacts.terrain_paths_by_environment[1] == old.contacts.terrain_paths_by_environment[1]
    assert storage.retired == list(old.contacts.terrain_paths_by_environment[0])
    assert old.readback['terrain_binding_invalidated'] and not result.scene.readback['terrain_binding_invalidated']
    with pytest.raises(RuntimeError):
        env.lifecycle.plant._identity()


def test_next_edit_supports_a_different_surface_count_without_a_fixed_atlas_or_generation_leak():
    env, _, sim, _, editor, storage, _, _, _, _ = fixture()
    first = editor.replace_courses(env.lifecycle.ground, changed(env.lifecycle.ground.courses), torch.tensor([True, False]))
    generator = ContinuousTerrainGenerator(4401, TerrainTuning(.8, (.2, 1.1), .25))
    material = generator.build(TerrainRequest(TerrainKind.MATERIAL, TerrainExtent((-5., 5.), (-5., 5.)),
                                               material_friction=(.4, .9)))
    new = (first.ground.courses[0], material)
    second = editor.replace_courses(first.ground, new, torch.tensor([False, True]))
    assert len(second.scene.contacts.terrain_paths_by_environment[1]) == 3
    third = editor.replace_courses(second.ground, changed(second.ground.courses, height=.08), torch.tensor([True, False]))
    assert third.scene.readback['terrain_epoch'] == 3 and sim.steps == 0
    assert third.scene.readback['terrain_reset_required_rows'] == (0, 1)
    assert all('/generation_1/' not in p for p in storage.records)
    assert len(storage.records) == 4 and third.ground.courses[1] is material


@pytest.mark.parametrize('fault', ['step_event', 'joint_state', 'model_state', 'view', 'cook_warning', 'cook_timeout',
                                  'readback', 'missing_native', 'ghost_native'])
def test_partial_native_edit_failure_retires_the_old_binding_and_cannot_replay_or_advance_the_task(fault):
    env, robot, sim, _, editor, storage, native, stats, current_view, physics = fixture()

    def corrupt():
        if fault == 'step_event':
            physics.callback(.005)
        elif fault == 'joint_state':
            robot.data.joint_pos.torch[0, 0] += .001
        elif fault == 'model_state':
            robot.root_view.local_com[0, 0, 0] += .001
        elif fault == 'view':
            current_view[0] = object()
        elif fault == 'cook_warning':
            stats.total_warnings_failed_gpu_compatibility += 1
        elif fault == 'cook_timeout':
            stats.total_scheduled_tasks = 1

    native.on_flush = corrupt
    storage.fail_verify = fault == 'readback'
    storage.missing_native, storage.ghost_native = fault == 'missing_native', fault == 'ghost_native'
    old = editor.scene
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        editor.replace_courses(env.lifecycle.ground, changed(env.lifecycle.ground.courses), torch.tensor([True, False]))
    assert editor._faulted and old.readback['terrain_binding_invalidated'] and sim.steps == 0
    assert editor.last_failure['automatic_resume_authorized'] is False
    with pytest.raises(RuntimeError, match='nonfaulted'):
        editor.replace_courses(env.lifecycle.ground, changed(env.lifecycle.ground.courses), torch.tensor([True, False]))
    with pytest.raises(RuntimeError):
        env.lifecycle.plant._identity()


@pytest.mark.parametrize('fault', ['stage', 'tracking', 'pending_cooking', 'unselected_course'])
def test_preflight_refusal_changes_no_shapes_or_native_binding(fault):
    env, _, sim, _, editor, storage, native, stats, _, _ = fixture()
    courses = changed(env.lifecycle.ground.courses)
    if fault == 'stage':native.stage_id += 1
    elif fault == 'tracking':native.tracking_paused = True
    elif fault == 'pending_cooking':stats.total_scheduled_tasks = 1
    else:courses = (courses[0], replace(courses[1]))
    with pytest.raises((RuntimeError, ValueError)):
        editor.replace_courses(env.lifecycle.ground, courses, torch.tensor([True, False]))
    assert not storage.writes and not editor.scene.readback.get('terrain_binding_invalidated', False)
    assert not editor._faulted and sim.steps == 0


@pytest.mark.parametrize('direct_hold', [False, True])
def test_installer_adopts_the_whole_task_graph_then_resets_only_the_ended_row(direct_hold):
    env, _, sim, _, editor, _, _, _, _, _ = fixture()
    env.reset()
    for _ in range(5):advance(env)
    life, plant = env.lifecycle, env.lifecycle.plant
    mask = torch.tensor([True, False])
    plan = env.reset_source(mask, life.episode_id + mask.long(), plant.scene.observation.timestamp_s())
    courses = changed(life.ground.courses)
    plan = replace(plan, source_courses=courses, course_sha256=tuple(c.sha256 for c in courses),
                    preparation_issued_action=torch.zeros(2, 6))
    if direct_hold:
        active = list(plant.scene.observation.binding.policy_joint_indices)
        hold = plant.controller.preparation_hold(plan.joint_position[:, active], position_tolerance_rad=1e-6)
        plan = replace(plan, preparation_issued_action=None, preparation_hold=hold)
    owner = SimpleNamespace(lifecycle=life, _scene=plant.scene, _identity=lambda: life._identity(),
                            _issued=SimpleNamespace(plan=plan, reset_mask=mask, next_episode_id=life.episode_id + mask.long(),
                                                     requested_at_s=plant.scene.observation.timestamp_s()))
    installer = NativeTerrainInstaller(env, owner, editor)
    old, steps = plant.scene, sim.steps
    drive, contacts = plant.drive, plant.contact_history
    other_drive, other_contacts = drive.state[1].clone(), contacts.history[:, 1].clone()

    def source(*args):
        installer(plan, mask)
        return plan

    env.reset_source = source
    env._reset_rows(mask)
    assert sim.steps == steps and life.episode_id.tolist() == [1, 0]
    assert plant.scene is env._scene is env.rewards._scene is owner._scene is editor.scene
    assert life._scene is plant.scene and life.ground is plant._ground is installer.last_result.ground
    assert old.readback['terrain_binding_invalidated'] and not plant.scene.readback['terrain_binding_invalidated']
    assert plant.scene.readback['terrain_reset_required_rows'] == ()
    assert plant.drive is drive and plant.contact_history is contacts
    assert torch.equal(drive.state[1], other_drive) and torch.equal(contacts.history[:, 1], other_contacts)
    env._identity()
    result = advance(env)
    assert sim.steps == steps + life.substeps and result.observation.episode_id.tolist() == [1, 0]
    assert plant.targets.control_source[0] == (2 if direct_hold else 1)


def test_invalid_hold_is_refused_before_any_USD_edit_or_native_reset():
    env, _, sim, writes, editor, storage, _, _, _, _ = fixture()
    env.reset()
    for _ in range(5):advance(env)
    life, plant = env.lifecycle, env.lifecycle.plant
    mask = torch.tensor([True, False]);steps = sim.steps;before_writes = len(writes)
    plan = env.reset_source(mask, life.episode_id + mask.long(), plant.scene.observation.timestamp_s())
    courses = changed(life.ground.courses);active = list(plant.scene.observation.binding.policy_joint_indices)
    hold = plant.controller.preparation_hold(plan.joint_position[:, active], position_tolerance_rad=1e-6)
    hold.leg_position_rad[0, 0] += .01
    plan = replace(plan, source_courses=courses, course_sha256=tuple(c.sha256 for c in courses), preparation_hold=hold)
    owner = SimpleNamespace(lifecycle=life, _scene=plant.scene, _identity=lambda: life._identity(),
                            _issued=SimpleNamespace(plan=plan, reset_mask=mask, next_episode_id=life.episode_id + mask.long(),
                                                     requested_at_s=plant.scene.observation.timestamp_s()))
    installer = NativeTerrainInstaller(env, owner, editor)
    with pytest.raises(ValueError, match='hold'):
        installer(plan, mask)
    assert sim.steps == steps and len(writes) == before_writes and not storage.writes
    assert not plant.scene.readback.get('terrain_binding_invalidated', False)


def test_cached_course_digest_preserves_the_original_canonical_identity_and_changed_sources_get_new_hashes():
    env, *_ = fixture()
    c = env.lifecycle.ground.courses[0]
    expected = hashlib.sha256(json.dumps(asdict(c), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert c.sha256 == expected and c.sha256 is c.sha256
    assert replace(c, geometry_seed=c.geometry_seed + 1).sha256 != expected


def test_new_binding_cannot_step_before_the_required_reset_and_only_reset_rows_discharge_it():
    env, robot, sim, _, editor, _, _, _, _, _ = fixture()
    life, plant = env.lifecycle, env.lifecycle.plant
    result = editor.replace_courses(life.ground, changed(life.ground.courses), torch.tensor([True, False]))
    plant.scene, plant._ground = result.scene, result.ground
    with pytest.raises(RuntimeError, match='physical reset'):
        plant.step_physics()
    values = [getattr(robot.data, name).torch.clone() for name in
              ('root_link_pose_w', 'root_com_vel_w', 'joint_pos', 'joint_vel')]
    plant.reset(torch.tensor([False, True]), *values)
    assert plant.scene.readback['terrain_reset_required_rows'] == (0,)
    plant.reset(torch.tensor([True, False]), *values)
    assert plant.scene.readback['terrain_reset_required_rows'] == () and sim.steps == 0


def test_native_path_probe_uses_all_hits_and_does_not_accept_usd_or_unrelated_collision_hits():
    env, *_ = fixture()
    mesh = env.lifecycle.ground.courses[0].surface.meshes[0]
    calls, targets = [], ['/Robot/collider', '/Owned/mesh']

    def raycast(origin, direction, distance, report, both_sides):
        calls.append((origin, direction, distance, both_sides))
        for path in targets:
            assert report(SimpleNamespace(collision=path))
        return bool(targets)

    probe = NativeShapeQueries(SimpleNamespace(raycast_all=raycast), .01)
    origins = torch.tensor([[2., -3., .5]])
    result = probe((('/Owned/mesh',),), ((mesh,),), origins, required_present=True)
    assert result['colliders_queried'] == 1 and calls[-1][2:] == (.02, True)
    assert sum(x*x for x in calls[-1][1]) == pytest.approx(1.)
    targets[:] = ['/Robot/collider']
    with pytest.raises(RuntimeError, match='presence'):
        probe((('/Owned/mesh',),), ((mesh,),), origins, required_present=True)
    probe((('/Owned/mesh',),), ((mesh,),), origins, required_present=False)

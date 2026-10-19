"""Private articulation/terrain authoring inside an already admitted SDK context.

The caller owns AppLauncher, runtime/resource admission and exclusive leases.
This module never starts a context or grants admission. Its live readbacks
must still be exercised and retained by the physical acceptance worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from ._native_actuation import NativeEffortWriter
from ._native_contacts import NativeTerrainContacts
from ._native_observation import NativeObservationBridge
from ._output_actuation import ResponseParameters
from ._physical_observation import MechanicalBinding
from ._structured_collection import ACTION_ORDER
from ._terrain_geometry import TerrainMesh


@dataclass(frozen=True)
class _ActuatorGroup:
    names: tuple[str, ...]
    damping: float
    effort_limit: float
    velocity_limit: float


@dataclass(frozen=True)
class NativeSceneState:
    robot: object
    observation: NativeObservationBridge
    efforts: NativeEffortWriter
    contacts: NativeTerrainContacts
    environment_paths: tuple[str, ...]
    environment_origins_m: torch.Tensor
    terrain_meshes: tuple[tuple[TerrainMesh, ...], ...]
    clone_indices: tuple[int, ...]
    readback: dict


class NativeSceneRecipe:
    """Own exact environment paths, copied asset bytes and explicit actuators.

    Meshes support arbitrary course geometry; authoring them is separate from
    sampling the full curriculum. A mesh is not a capability/coverage claim.
    A context reset initializes SDK views; it is not a PPO transition or proof
    that the robot has reached a healthy settled state.
    """

    def __init__(self, asset_directory, manifest_sha256, response: ResponseParameters,
                 environment_origins_m, meshes_by_environment):
        self.asset = Path(asset_directory).resolve(strict=True)
        if not isinstance(response, ResponseParameters) or not self.asset.is_dir():
            raise ValueError('scene requires a selected asset directory and complete response')
        manifest_path = self.asset / 'manifest.json'
        data = manifest_path.read_bytes()
        if manifest_path.is_symlink() or hashlib.sha256(data).hexdigest() != manifest_sha256:
            raise ValueError('selected asset manifest hash differs')
        manifest = json.loads(data)
        if (manifest['usd'] not in manifest['files_sha256'] or 'model_spec.json' not in manifest['files_sha256']
                or not isinstance(manifest.get('urdf'), str) or manifest['urdf'] not in manifest['files_sha256']
                or manifest['self_collision_enabled'] is not False
                or type(manifest['nominal_base_height_m']) not in (int, float)
                or not math.isfinite(manifest['nominal_base_height_m']) or manifest['nominal_base_height_m'] <= 0):
            raise ValueError('scene requires the complete selected asset and explicit nominal root height')
        for name, sha in manifest['files_sha256'].items():
            path = self.asset / name
            if (Path(name).is_absolute() or not path.resolve().is_relative_to(self.asset)
                    or path.is_symlink() or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != sha):
                raise ValueError('selected asset dependency escaped or changed: ' + name)
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.initial_root_position_m = (0., 0., float(manifest['nominal_base_height_m']))
        self.model_spec = json.loads((self.asset / 'model_spec.json').read_text())
        self.response = response
        self.binding = MechanicalBinding.from_manifest(manifest, manifest['tree_joint_names'], manifest['rigid_body_names'])
        self.origins = tuple(tuple(row) for row in environment_origins_m)
        self.meshes = tuple(tuple(row) for row in meshes_by_environment)
        if (not self.origins or len(set(self.origins)) != len(self.origins) or len(self.meshes) != len(self.origins)
                or any(len(row) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                       for row in self.origins)
                or any(not row or any(not isinstance(mesh, TerrainMesh) for mesh in row) for row in self.meshes)):
            raise ValueError('each distinct finite scene origin requires explicit nonempty terrain meshes')
        self.groups = self._actuator_groups()

    def _actuator_groups(self):
        springs = tuple(self.manifest['spring_joint_names'])
        passive = tuple(name for name in self.manifest['tree_joint_names'] if name not in (*ACTION_ORDER, *springs))
        damping = self.manifest['passive_damping']
        if (springs != ('L_spring_slide', 'R_spring_slide') or damping['active_Nm_s_per_rad'] != 0
                or any(type(damping[field]) not in (int, float) or not math.isfinite(damping[field])
                       or damping[field] < 0 for field in ('passive_hinge_Nm_s_per_rad', 'slider_N_s_per_m'))):
            raise ValueError('asset passive damping or active output grouping differs')
        return {'legs': _ActuatorGroup(ACTION_ORDER[:4], 0., 40., 1e9),
                'wheels': _ActuatorGroup(ACTION_ORDER[4:], 0., 4.5, 1e9),
                'passive': _ActuatorGroup(passive, damping['passive_hinge_Nm_s_per_rad'], 100., 1e9),
                'pressure': _ActuatorGroup(springs, damping['slider_N_s_per_m'], 1000., 100.)}

    def _actuator_configs(self, config_type):
        native = self.response.native_parameters()
        result = {}
        for name, group in self.groups.items():
            values = {field: {joint: native[joint][field] if joint in native else 0. for joint in group.names}
                      for field in ('armature', 'friction', 'dynamic_friction', 'viscous_friction')}
            result[name] = config_type(joint_names_expr=list(group.names), stiffness=0., damping=group.damping,
                                       effort_limit=group.effort_limit, effort_limit_sim=group.effort_limit,
                                       velocity_limit_sim=group.velocity_limit, **values)
        return result

    def _actuator_readback(self, robot, converter, device):
        """Read SDK explicit gains separately from zero solver PD gains."""
        binding = MechanicalBinding.from_manifest(self.manifest, robot.joint_names, robot.body_names)
        count = len(robot.root_view.prim_paths)
        native = self.response.native_parameters()

        def same(actual, expected, shape, label):
            if (not isinstance(actual, torch.Tensor) or actual.shape != shape or not torch.isfinite(actual).all()
                    or not torch.allclose(actual.to(device=device, dtype=torch.float64),
                                          torch.as_tensor(expected, device=device, dtype=torch.float64),
                                          rtol=1e-6, atol=1e-7)):
                raise RuntimeError('native actuator readback differs: ' + label)

        joint_names = binding.native_joint_names
        armature = [native[n]['armature'] if n in native else 0. for n in joint_names]
        friction = [[native[n][key] if n in native else 0.
                     for key in ('friction', 'dynamic_friction', 'viscous_friction')] for n in joint_names]
        same(converter(robot.root_view.get_dof_armatures()), [armature] * count, (count, 18), 'armature')
        same(converter(robot.root_view.get_dof_friction_properties()), [friction] * count, (count, 18, 3), 'friction')
        for getter in ('get_dof_stiffnesses', 'get_dof_dampings'):
            same(converter(getattr(robot.root_view, getter)()), 0., (count, 18), getter)
        covered = []
        if set(robot.actuators) != set(self.groups):
            raise RuntimeError('explicit actuator groups differ')
        for name, expected in self.groups.items():
            actuator = robot.actuators[name]
            names = tuple(actuator.joint_names)
            if actuator.is_implicit_model or set(names) != set(expected.names) or len(names) != len(expected.names):
                raise RuntimeError('an implicit, missing or duplicate joint actuator was installed')
            same(actuator.stiffness, 0., (count, len(names)), name + ' explicit stiffness')
            same(actuator.damping, expected.damping, (count, len(names)), name + ' explicit damping')
            same(actuator.effort_limit, expected.effort_limit, (count, len(names)), name + ' explicit effort limit')
            covered.extend(names)
        if len(covered) != 18 or set(covered) != set(joint_names):
            raise RuntimeError('explicit actuator coverage must be exactly the native tree')
        return {'named_armature_friction_verified': True, 'solver_pd_gains_zero': True,
                'explicit_actuator_gains_and_caps_verified': True, 'native_solver_caps_readback_verified': False,
                'hardware_calibration_verified': False}

    @staticmethod
    def _static_terrain_readback(stage, terrain_paths, rigid_body_api):
        """Static velocity is allowed only for unanimated rigid-free ancestry.

        A future terrain-replacement owner must invalidate this binding before
        edits. This is authored-USD readback, not native collision equivalence.
        """
        checked = set()
        for path in (path for row in terrain_paths for path in row):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError('authored terrain collider is missing')
            while prim.IsValid() and str(prim.GetPath()) not in checked:
                if prim.HasAPI(rigid_body_api) or any(a.GetNumTimeSamples() for a in prim.GetAttributes()):
                    raise RuntimeError('contact motion cannot assume static terrain with rigid or animated ancestry')
                checked.add(str(prim.GetPath()))
                prim = prim.GetParent()
        return {'terrain_motion_reference': 'authored_static_own_terrain',
                'static_terrain_ancestry_prims_checked': len(checked),
                'native_static_collision_equivalence_verified': False}

    @staticmethod
    def _author_terrain_mesh(stage, path, mesh, geometry_api, physics_api, sim_utils, *, enabled):
        geometry = geometry_api.Mesh.Define(stage, path + '/geometry/mesh')
        geometry.CreatePointsAttr(mesh.vertices_m)
        geometry.CreateFaceVertexCountsAttr([3] * len(mesh.triangles))
        geometry.CreateFaceVertexIndicesAttr([i for face in mesh.triangles for i in face])
        geometry.CreateSubdivisionSchemeAttr('none')
        geometry.CreateDoubleSidedAttr(True)
        physics_api.CollisionAPI.Apply(geometry.GetPrim()).CreateCollisionEnabledAttr(enabled)
        physics_api.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr('none')
        material = sim_utils.RigidBodyMaterialCfg(static_friction=mesh.friction, dynamic_friction=mesh.friction,
                                                  restitution=0., friction_combine_mode='multiply',
                                                  restitution_combine_mode='multiply')
        material.func(path + '/Material', material)
        sim_utils.bind_physics_material(str(geometry.GetPath()), path + '/Material')
        return str(geometry.GetPath())

    def build_on_admitted_context(self, sim):
        """Author this owned scene; the caller has already acquired admission."""
        if type(sim.cfg.dt) not in (int, float) or sim.cfg.dt not in (.001, .005):
            raise ValueError('scene physics must use the selected 200Hz or 1kHz interval')
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        import warp as wp
        from isaaclab import cloner
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxManager

        stage = sim.stage
        if stage.GetPrimAtPath('/World/envs').IsValid() or stage.GetPrimAtPath('/World/PlantMaterial').IsValid():
            raise RuntimeError('owned scene roots already exist; never adopt or overwrite another scene')
        env_paths = tuple('/World/envs/env_' + str(index) for index in range(len(self.origins)))
        terrain_paths = []
        for env, origin, meshes in zip(env_paths, self.origins, self.meshes):
            UsdGeom.Xform.Define(stage, env).AddTranslateOp().Set(tuple(float(v) for v in origin))
            paths = []
            for index, mesh in enumerate(meshes):
                path = env + '/Terrain/surface_' + str(index)
                paths.append(self._author_terrain_mesh(stage, path, mesh, UsdGeom, UsdPhysics, sim_utils, enabled=True))
            terrain_paths.append(tuple(paths))
        cfg = ArticulationCfg(
            prim_path='/World/envs/env_.*/Robot',
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.asset / self.manifest['usd']), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=6)),
            init_state=ArticulationCfg.InitialStateCfg(pos=self.initial_root_position_m, joint_pos=self.manifest['nominal_joint_pos'],
                                                     joint_vel={'.*': 0.}),
            actuators=self._actuator_configs(IdealPDActuatorCfg), soft_joint_pos_limit_factor=1.)
        robot = Articulation(cfg)
        material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.,
                                                  friction_combine_mode='multiply', restitution_combine_mode='multiply')
        material.func('/World/PlantMaterial', material)
        for env in env_paths:
            loops, roots = 0, 0
            for prim in Usd.PrimRange(stage.GetPrimAtPath(env + '/Robot')):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    sim_utils.bind_physics_material(str(prim.GetPath()), '/World/PlantMaterial')
                if prim.IsA(UsdPhysics.SphericalJoint):
                    if UsdPhysics.SphericalJoint(prim).GetExcludeFromArticulationAttr().Get() is not True:
                        raise RuntimeError('closed-chain point constraint entered the articulation tree')
                    loops += 1
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    actual = tuple(prim.GetAttribute('physxArticulation:' + key).Get()
                                   for key in ('solverPositionIterationCount', 'solverVelocityIterationCount'))
                    if actual != (12, 6):
                        raise RuntimeError('closed-chain solver iteration readback differs')
                    roots += 1
            if (loops, roots) != (6, 1):
                raise RuntimeError('each robot requires six excluded point loops and one articulation root')
        cloner.filter_collisions(stage, sim.cfg.physics_prim_path, '/World/collisions', list(env_paths), global_paths=[])
        if stage.GetPrimAtPath(sim.cfg.physics_prim_path).GetAttribute('physxScene:invertCollisionGroupFilter').Get() is not True:
            raise RuntimeError('cross-environment collision filtering was not authored')
        for index, env in enumerate(env_paths):
            path = '/World/collisions/group' + str(index)
            prim = stage.GetPrimAtPath(path)
            includes = tuple(str(p) for p in prim.GetRelationship('collection:colliders:includes').GetTargets())
            allowed = {str(p) for p in prim.GetRelationship('physics:filteredGroups').GetTargets()}
            if includes != (env,) or allowed != {path, '/World/collisions/global_group'}:
                raise RuntimeError('environment collision group readback differs')
        sim.reset()
        paths = tuple(str(path) for path in robot.root_view.prim_paths)
        native_envs = tuple(path.split('/Robot')[0] for path in paths)
        if len(set(native_envs)) != len(env_paths) or set(native_envs) != set(env_paths):
            raise RuntimeError('native clone paths do not match this owned scene')
        order = tuple(env_paths.index(path) for path in native_envs)
        device = torch.device(robot.device)
        bridge = NativeObservationBridge(robot, self.manifest, len(env_paths), native_to_tensor=wp.to_torch,
                                         device=device, physics_dt_s=float(sim.cfg.dt))
        readback = self._actuator_readback(robot, wp.to_torch, device)
        masses = wp.to_torch(robot.root_view.get_masses())
        if (masses.shape != (len(env_paths), 19) or not torch.isfinite(masses).all() or (masses <= 0).any()
                or not torch.allclose(masses.sum(-1), masses.new_full((len(env_paths),), self.manifest['total_mass_kg']),
                                      atol=1e-5, rtol=0)):
            raise RuntimeError('native complete body mass differs from the selected asset')
        readback.update(native_mass_sum_verified=True, six_excluded_point_loops_per_robot=True,
                        collision_groups_verified=len(env_paths), native_clone_order_resolved=True,
                        selected_asset_manifest_sha256=self.manifest_sha256,
                        selected_model_spec_sha256=self.manifest['files_sha256']['model_spec.json'],
                        selected_urdf_sha256=self.manifest['files_sha256'][self.manifest['urdf']],
                        native_wheel_collision_equivalence_verified=False,
                        nominal_source_mass_kg=self.manifest['total_mass_kg'],
                        physics_qualified=False)
        readback.update(self._static_terrain_readback(stage, terrain_paths, UsdPhysics.RigidBodyAPI))
        contacts = NativeTerrainContacts(PhysxManager.get_physics_sim_view(), bridge.binding, native_envs,
                                         tuple(terrain_paths[index] for index in order),
                                         native_to_tensor=wp.to_torch, device=device,
                                         terrain_motion_reference=readback['terrain_motion_reference'])
        return NativeSceneState(robot, bridge, NativeEffortWriter(robot, bridge.binding, self.model_spec,
                                                                 len(env_paths), device=device),
                                contacts, native_envs, torch.tensor([self.origins[i] for i in order], device=device),
                                tuple(self.meshes[i] for i in order), order, readback)

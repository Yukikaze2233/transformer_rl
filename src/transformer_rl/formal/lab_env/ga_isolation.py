"""Scene-local Env ID replication through the Isaac Lab 3 PhysX extension point.

Only the passed physics configuration receives an application-owned manager.
The SDK classes and installed clone callbacks remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .isolation import HomogeneousCloneIsolation


class GaHomogeneousCloneIsolation(HomogeneousCloneIsolation):
    """Translate one complete GA robot plan into the certified native callback."""

    def __init__(self, num_envs: int, device: str):
        super().__init__(num_envs)
        self.device = str(torch.device(device))
        if torch.device(device).type != 'cuda':
            raise ValueError('GA Env ID isolation requires a CUDA PhysX scene')
        self._plan_registered = False
        self._receipt.update(clone_plan_api='isaaclab_physx_context',
                             shared_ground_verified=False, warmup_finished=False)

    def register_plan(self, stage, plan, asset_prototype_ids, path_api):
        """Reject heterogeneous, partial or renamed worlds before native import.

        Shared sources stay outside the excluded environment namespace, so USD
        ground and lights load normally. Only the full Robot subtree is cloned.
        """
        if self._plan_registered:
            raise RuntimeError('GA collision clone plan was already registered')
        topology = plan.topology
        arrays = (topology.world_prototypes, topology.world_prototype_starts,
                  topology.world_prototype_layout)
        if any(not isinstance(value, np.ndarray) or value.ndim != 1
               or not np.issubdtype(value.dtype, np.integer)
               or not value.flags.c_contiguous for value in arrays):
            raise ValueError('GA isolation requires a contiguous host clone topology')
        members, starts, layout = arrays
        num_assets = topology.num_asset_prototypes
        if (type(num_assets) is not int or num_assets != len(plan.asset_cfgs)
                or starts.shape != (3,) or starts[0] != 0
                or not 0 <= starts[1] <= starts[2] or starts[2] != members.size
                or starts[2] - starts[1] != 1
                or layout.shape != (self.num_envs,) or np.any(layout != 0)
                or plan.env_template != '/World/envs/env_{}'
                or np.any(members < 0) or np.any(members >= num_assets)):
            raise ValueError('GA isolation requires one complete homogeneous robot world')
        shared = tuple(map(int, members[:starts[1]]))
        robot = int(members[starts[1]])
        requested_ids = tuple(asset_prototype_ids)
        if any(isinstance(index, (bool, np.bool_))
               or not isinstance(index, (int, np.integer)) for index in requested_ids):
            raise ValueError('GA isolation requires integer PhysX source IDs')
        requested = tuple(map(int, requested_ids))
        if (tuple(sorted(set(requested))) != requested
                or len(set(shared)) != len(shared) or robot in shared
                or set(requested) != {*shared, robot}):
            raise ValueError('GA isolation requires every active PhysX source exactly once')
        positions = plan.positions
        if positions is not None and (
                not isinstance(positions, np.ndarray)
                or positions.shape != (self.num_envs, 3)
                or not np.issubdtype(positions.dtype, np.floating)
                or not np.isfinite(positions).all()):
            raise ValueError('GA isolation requires finite positions for every world')

        sources = path_api.get_asset_prototype_paths(plan)
        templates, template_starts, world_ids, world_starts = (
            path_api.get_world_prototype_asset_templates(plan, include_world_indices=True))
        expected_worlds = np.r_[-1, np.arange(self.num_envs)]
        if (len(sources) != num_assets or len(templates) != members.size
                or not np.array_equal(template_starts, starts)
                or not np.array_equal(world_ids, expected_worlds)
                or not np.array_equal(world_starts, [0, 1, self.num_envs + 1])
                or sources[robot] != '/World/envs/env_0/Robot'
                or templates[starts[1]] != '/World/envs/env_{}/Robot'
                or any(not isinstance(sources[index], str)
                       or not sources[index].startswith('/World/')
                       or sources[index].startswith('/World/envs/') for index in shared)):
            raise ValueError('GA isolation requires exact source and consecutive world ID binding')

        self.replicate(stage, [sources[robot]], [templates[starts[1]]],
                       torch.arange(self.num_envs),
                       torch.ones((1, self.num_envs), dtype=torch.bool),
                       device=self.device, use_fabric=False)
        self._plan_registered = True
        self._receipt.update(
            physics_asset_prototype_ids=list(requested), robot_asset_prototype_id=robot,
            shared_source_paths=[sources[index] for index in shared],
            world_ids={'first': 0, 'last': self.num_envs - 1, 'count': self.num_envs},
            environment_namespace='/World/envs', device=self.device,
            stage_id=self._stage_id,
        )

    def confirm_shared_ground_groups(self, receipt):
        """Require the existing scene-owned USD readback, not a spacing claim."""
        if (not isinstance(receipt, dict)
                or receipt.get('environment_groups_verified') != self.num_envs
                or receipt.get('global_paths') != ['/World/ground']
                or receipt.get('inverted_filter') is not True):
            raise ValueError('GA isolation requires the shared-ground collision-group readback')
        self._receipt.update(shared_ground_verified=True, shared_ground_groups=deepcopy(receipt))

    def finish_warmup(self):
        """Retain registration through stage loading; release only after readback."""
        if not self._plan_registered or not self._receipt['shared_ground_verified']:
            raise RuntimeError('GA isolation requires the robot plan and shared-ground groups')
        super().finish_warmup()
        self._receipt['warmup_finished'] = True

    @property
    def receipt(self):
        return deepcopy(self._receipt)


def _physx_types():
    """Import the native integration only inside an active GA setup boundary."""
    from isaaclab import cloner
    from isaaclab_physx.cloner import PhysxReplicateContext
    from isaaclab_physx.physics import PhysxCfg, PhysxManager
    return PhysxCfg, PhysxManager, PhysxReplicateContext, cloner.path


def configure_ga_isolation(physics_cfg, num_envs: int, device: str):
    """Select a scene-owned manager before constructing SimulationContext.

    Keep the returned binding on the environment configuration. After cloning,
    pass it ``author_shared_ground_groups(scene)`` via
    ``confirm_shared_ground_groups``. After the initial reset/physics warmup,
    call ``finish_warmup`` and publish its receipt with the scene readback.
    """
    binding = GaHomogeneousCloneIsolation(num_envs, device)
    config_type, manager_type, context_type, path_api = _physx_types()
    if not isinstance(physics_cfg, config_type):
        raise TypeError('GA collision isolation requires a PhysxCfg')
    if physics_cfg.class_type not in (
            manager_type, '{DIR}.physx_manager:PhysxManager',
            'isaaclab_physx.physics.physx_manager:PhysxManager'):
        raise ValueError('GA collision isolation cannot replace another custom physics manager')

    class IsolatedPhysxReplicateContext(context_type):
        def replicate(self, plan, asset_prototype_ids):
            binding.register_plan(self.stage, plan, asset_prototype_ids, path_api)

    # GA AssetBase gates its PhysX prim-deletion callback by this prefix.
    class PhysxIsolatedManager(manager_type):
        clone_context_type = IsolatedPhysxReplicateContext
        collision_isolation = binding

    physics_cfg.class_type = PhysxIsolatedManager
    return binding

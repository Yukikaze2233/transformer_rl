"""Application-owned collision isolation for one homogeneous robot clone plan.

USD collision groups preserve the shared-ground contract. PhysX environment IDs
also reject cross-environment pairs in broadphase, including co-located robots.
This adapter changes this scene's clone callback, never the installed SDK.
"""
from __future__ import annotations

import torch


class HomogeneousCloneIsolation:
    """Bind Env IDs only when source and destination indices are unambiguous."""

    def __init__(self, num_envs: int):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('collision isolation requires a positive environment count')
        self.num_envs = num_envs
        self._receipt = {
            'method': 'physx_homogeneous_env_ids', 'environments': num_envs,
            'registered': False, 'applied': num_envs == 1,
            'replicated_environments': 0, 'use_env_ids': num_envs > 1,
        }

    def replicate(self, stage, sources, destinations, env_ids, mapping,
                  positions=None, quaternions=None, use_fabric=False,
                  device='cpu', exclude_self_replication=True):
        """Use the native replicator with consecutive IDs and env_0 as source.

        The installed generic cloner disables Env IDs for heterogeneous plans.
        Requiring one full source row avoids that ambiguity; a different plan
        is rejected rather than silently falling back to late pair filtering.
        """
        del positions, quaternions
        if self._receipt['registered']:
            raise RuntimeError('collision clone plan was already registered')
        expected = torch.arange(self.num_envs, device=env_ids.device)
        if (torch.device(device).type != 'cuda' or not exclude_self_replication
                or len(sources) != 1 or len(destinations) != 1
                or env_ids.shape != (self.num_envs,)
                or not torch.equal(env_ids, expected)
                or mapping.shape != (1, self.num_envs) or not bool(mapping.all())
                or destinations[0].count('{}') != 1
                or not destinations[0].startswith('/World/envs/env_{}')
                or sources[0] != destinations[0].format(0)):
            raise ValueError('Env ID isolation requires one homogeneous consecutive clone plan')
        if self.num_envs == 1:
            return
        from omni.physx import get_physx_replicator_interface
        from pxr import UsdUtils

        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        self._register(get_physx_replicator_interface(), stage_id,
                       sources[0], destinations[0], use_fabric=use_fabric)

    def _register(self, replicator, stage_id, source, destination, *, use_fabric):
        def exclude_environments(_stage_id):
            return ['/World/envs']

        def renamed_path(_path, replica_index):
            if not 0 <= replica_index < self.num_envs - 1:
                raise IndexError('native clone index is outside the bound plan')
            return destination.format(replica_index + 1)

        def replicate_after_attach(actual_stage_id):
            try:
                if actual_stage_id != stage_id:
                    raise RuntimeError('native replicator attached a different stage')
                result = replicator.replicate(
                    stage_id, source, self.num_envs - 1,
                    useEnvIds=True, useFabricForReplication=use_fabric)
                if result is False:
                    raise RuntimeError('native Env ID replication was rejected')
                self._receipt.update(applied=True,
                                     replicated_environments=self.num_envs - 1)
            finally:
                replicator.unregister_replicator(stage_id)

        self._receipt.update(registered=True, source=source, destination=destination)
        replicator.register_replicator(
            stage_id, exclude_environments, replicate_after_attach, renamed_path)

    @property
    def receipt(self):
        return dict(self._receipt)

    def require_applied(self):
        if not self._receipt['applied']:
            raise RuntimeError('native Env ID isolation has not been applied')


def isolate_scene(scene):
    """Install early GPU filtering and always author the shared-ground groups.

    Called before clone_environments, then before the first simulation reset.
    The returned binding must be checked after reset, not treated as a native
    success merely because its callback was installed.
    """
    binding = None
    if torch.device(scene.device).type == 'cuda':
        if not scene.physics_backend.startswith('physx'):
            raise ValueError('GPU collision isolation requires the verified PhysX backend')
        binding = HomogeneousCloneIsolation(scene.num_envs)
        scene.cloner_cfg.physics_clone_fn = binding.replicate
    return binding


def author_shared_ground_groups(scene):
    """Read back USD isolation once; spatial spacing is never a filter switch."""
    scene.filter_collisions(global_prim_paths=['/World/ground'])
    stage = scene.stage
    physics_scene = stage.GetPrimAtPath(scene.physics_scene_path)
    if physics_scene.GetAttribute('physxScene:invertCollisionGroupFilter').Get() is not True:
        raise RuntimeError('environment collision-group filter was not authored')
    for index, path in enumerate(scene.env_prim_paths):
        group_path = f'/World/collisions/group{index}'
        group = stage.GetPrimAtPath(group_path)
        members = [str(p) for p in group.GetRelationship('collection:colliders:includes').GetTargets()]
        allowed = {str(p) for p in group.GetRelationship('physics:filteredGroups').GetTargets()}
        if members != [path] or allowed != {group_path, '/World/collisions/global_group'}:
            raise RuntimeError(f'environment collision-group readback differs: {group_path}')
    return {'environment_groups_verified': len(scene.env_prim_paths),
            'global_paths': ['/World/ground'], 'inverted_filter': True}

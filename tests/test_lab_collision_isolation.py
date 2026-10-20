"""Clone-plan guards and real replicator arguments without an SDK context."""
from types import SimpleNamespace
import pytest
import torch

from transformer_rl.formal.lab_env.isolation import (
    HomogeneousCloneIsolation, isolate_scene,
)


class Replicator:
    def __init__(self):
        self.calls = []
        self.removed = []

    def register_replicator(self, stage, exclude, attached, rename):
        self.exclude, self.attached, self.rename = exclude, attached, rename

    def replicate(self, stage, source, count, **kwargs):
        self.calls.append((stage, source, count, kwargs))
        return True

    def unregister_replicator(self, stage):
        self.removed.append(stage)


def test_native_ids_cover_source_and_all_8191_replicas():
    binding = HomogeneousCloneIsolation(8192)
    native = Replicator()
    binding._register(native, 37, '/World/envs/env_0/Robot',
                      '/World/envs/env_{}/Robot', use_fabric=False)
    assert native.exclude(37) == ['/World/envs']
    assert native.rename('', 0) == '/World/envs/env_1/Robot'
    assert native.rename('', 8190) == '/World/envs/env_8191/Robot'
    with pytest.raises(IndexError):
        native.rename('', 8191)
    with pytest.raises(RuntimeError, match='has not been applied'):
        binding.require_applied()
    native.attached(37)
    assert native.calls == [(37, '/World/envs/env_0/Robot', 8191,
                             {'useEnvIds': True, 'useFabricForReplication': False})]
    assert native.removed == [37]
    binding.require_applied()
    assert binding.receipt['replicated_environments'] == 8191


def test_failed_native_replication_is_not_published_as_isolated():
    binding = HomogeneousCloneIsolation(4)
    native = Replicator()
    native.replicate = lambda *args, **kwargs: False
    binding._register(native, 3, '/World/envs/env_0', '/World/envs/env_{}', use_fabric=False)
    with pytest.raises(RuntimeError, match='was rejected'):
        native.attached(3)
    assert native.removed == [3]
    assert not binding.receipt['applied']


def test_wrong_attached_stage_is_rejected_and_unregistered():
    binding = HomogeneousCloneIsolation(4)
    native = Replicator()
    binding._register(native, 3, '/World/envs/env_0', '/World/envs/env_{}', use_fabric=False)
    with pytest.raises(RuntimeError, match='different stage'):
        native.attached(9)
    assert not native.calls
    assert native.removed == [3]


@pytest.mark.parametrize('change', ['heterogeneous', 'partial', 'reordered', 'source', 'cpu'])
def test_ambiguous_plan_never_reaches_the_sdk(change):
    binding = HomogeneousCloneIsolation(4)
    args = dict(stage=None, sources=['/World/envs/env_0/Robot'],
                destinations=['/World/envs/env_{}/Robot'], env_ids=torch.arange(4),
                mapping=torch.ones(1, 4, dtype=torch.bool), device='cuda:0')
    if change == 'heterogeneous':
        args.update(sources=args['sources'] * 2, destinations=args['destinations'] * 2,
                    mapping=torch.ones(2, 4, dtype=torch.bool))
    elif change == 'partial':
        args['mapping'][0, 3] = False
    elif change == 'reordered':
        args['env_ids'] = torch.tensor([1, 0, 2, 3])
    elif change == 'source':
        args['sources'] = ['/World/envs/env_1/Robot']
    else:
        args['device'] = 'cpu'
    with pytest.raises(ValueError, match='homogeneous'):
        binding.replicate(**args)
    assert not binding.receipt['registered']


def test_single_environment_does_not_register_a_zero_replica_callback():
    binding = HomogeneousCloneIsolation(1)
    binding.replicate(None, ['/World/envs/env_0/Robot'], ['/World/envs/env_{}/Robot'],
                      torch.arange(1), torch.ones(1, 1, dtype=torch.bool), device='cuda:0')
    binding.require_applied()
    assert not binding.receipt['registered']


def test_only_the_owned_scene_callback_is_replaced():
    original = object()
    scene = SimpleNamespace(device='cuda:0', physics_backend='physxmanager', num_envs=8192,
                            cloner_cfg=SimpleNamespace(physics_clone_fn=original))
    binding = isolate_scene(scene)
    assert scene.cloner_cfg.physics_clone_fn.__self__ is binding
    other = SimpleNamespace(device='cpu', cloner_cfg=SimpleNamespace(physics_clone_fn=original))
    assert isolate_scene(other) is None
    assert other.cloner_cfg.physics_clone_fn is original

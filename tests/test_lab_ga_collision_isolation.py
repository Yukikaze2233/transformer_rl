"""GA clone-context contract with synthetic host plans and native callbacks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_rl.formal.lab_env import ga_isolation


class PhysxCfg:
    class_type = '{DIR}.physx_manager:PhysxManager'


class PhysxReplicateContext:
    replicate_priority = 0

    def __init__(self, sim):
        self.stage = sim.stage

    def replicate(self, *args):
        raise AssertionError('the generic Env-ID-disabled callback must not run')


class PhysxManager:
    clone_context_type = PhysxReplicateContext
    backend = None

    @classmethod
    def initialize(cls, backend):
        cls.backend = backend

    @classmethod
    def get_physics_sim_view(cls):
        return cls.backend


class AssetCfg:
    prim_path = '/World/envs/env_.*/Robot'
    cloning_contexts = None
    spawn = object()


class PlanPaths:
    """SDK path API-shaped fixture; no stage lookup or native context."""

    def get_asset_prototype_paths(self, plan):
        return plan.sources

    def get_world_prototype_asset_templates(self, plan, *, include_world_indices):
        assert include_world_indices is True
        return (plan.templates, plan.topology.world_prototype_starts,
                plan.world_ids, plan.world_starts)


def plan(num_envs=4, *, shared=True):
    members = [0, 1] if shared else [1]
    start = 1 if shared else 0
    return SimpleNamespace(
        topology=SimpleNamespace(
            num_asset_prototypes=3,
            world_prototypes=np.array(members, dtype=np.int32),
            world_prototype_starts=np.array([0, start, len(members)], dtype=np.int64),
            world_prototype_layout=np.zeros(num_envs, dtype=np.int32),
        ),
        asset_cfgs=(object(), object(), object()),
        env_template='/World/envs/env_{}',
        positions=np.zeros((num_envs, 3), dtype=np.float32),
        sources=('/World/ground' if shared else None, '/World/envs/env_0/Robot', None),
        templates=('/World/ground', '/World/envs/env_{}/Robot') if shared
        else ('/World/envs/env_{}/Robot',),
        world_ids=np.r_[-1, np.arange(num_envs)].astype(np.int32),
        world_starts=np.array([0, 1, num_envs + 1], dtype=np.int64),
    )


class Replicator:
    def __init__(self):
        self.calls = []
        self.removed = []

    def register_replicator(self, stage_id, exclude, attach, rename):
        self.exclude, self.attach, self.rename = exclude, attach, rename

    def replicate(self, stage_id, source, count, **kwargs):
        self.calls.append((stage_id, source, count, kwargs))
        return True

    def unregister_replicator(self, stage_id):
        self.removed.append(stage_id)


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.setattr(ga_isolation, '_physx_types',
                        lambda: (PhysxCfg, PhysxManager, PhysxReplicateContext, PlanPaths()))
    def setup(cfg, num_envs, device):
        asset = getattr(cfg, '_test_robot', None)
        if asset is None:
            asset = cfg._test_robot = AssetCfg()
        return ga_isolation.configure_ga_isolation(cfg, asset, num_envs, device)

    return setup


def clone_context(cfg):
    return cfg._test_robot.cloning_contexts[0]


def bind_native(monkeypatch, binding, native):
    captured = []

    def replicate(stage, sources, destinations, env_ids, mapping, **kwargs):
        captured.append((stage, sources, destinations, env_ids.clone(), mapping.clone(), kwargs))
        if binding.num_envs > 1:
            binding._register(native, stage, sources[0], destinations[0],
                              use_fabric=kwargs['use_fabric'])

    monkeypatch.setattr(binding, 'replicate', replicate)
    return captured


def group_receipt(num_envs):
    return dict(environment_groups_verified=num_envs,
                global_paths=['/World/ground'], inverted_filter=True)


def test_only_the_owned_robot_gets_a_new_context(configure):
    first, second = PhysxCfg(), PhysxCfg()
    binding = configure(first, 8192, 'cuda:0')
    assert first.class_type == '{DIR}.physx_manager:PhysxManager'
    assert issubclass(clone_context(first), PhysxReplicateContext)
    assert clone_context(first).replicate_priority == 0
    assert second.class_type == '{DIR}.physx_manager:PhysxManager'
    assert AssetCfg.cloning_contexts is None
    assert PhysxManager.clone_context_type is PhysxReplicateContext
    assert binding.receipt['clone_plan_api'] == 'isaaclab_asset_cloning_contexts'
    with pytest.raises(ValueError, match='another clone context'):
        configure(first, 8192, 'cuda:0')


def test_two_configurations_never_share_a_plan_or_receipt(configure):
    first, second = PhysxCfg(), PhysxCfg()
    a, b = configure(first, 4, 'cuda:0'), configure(second, 8, 'cuda:0')
    assert clone_context(first) is not clone_context(second)
    assert first._test_robot is not second._test_robot
    a.confirm_shared_ground_groups(group_receipt(4))
    assert not b.receipt['shared_ground_verified']


def test_explicit_asset_route_keeps_the_original_manager_singleton(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    cfg.class_type = PhysxManager
    binding = configure(cfg, 4, 'cuda:0')
    calls = bind_native(monkeypatch, binding, native)
    # Follow GA replicate_session's explicit-override rule. USD authoring is
    # supplied by the spawner separately; the default PhysX context is excluded.
    references = vars(cfg._test_robot).get('cloning_contexts', ())
    contexts = ((cfg.class_type.clone_context_type,) if references is None
                else tuple(references))
    assert contexts == (clone_context(cfg),)
    assert PhysxReplicateContext not in contexts
    for context_type in contexts:
        context_type(SimpleNamespace(stage=37)).replicate(plan(shared=False), (1,))
    assert len(calls) == 1
    marker = object()
    monkeypatch.setattr(PhysxManager, 'backend', None)
    cfg.class_type.initialize(marker)
    assert cfg.class_type is PhysxManager
    assert cfg.class_type.get_physics_sim_view() is marker
    # The native articulation imports this hard-coded SDK class.
    assert PhysxManager.get_physics_sim_view() is marker
    assert PhysxManager.clone_context_type is PhysxReplicateContext


def test_shared_ground_routes_to_default_context_and_only_robot_to_owned_context(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    cfg.class_type = PhysxManager
    binding = configure(cfg, 4, 'cuda:0')
    calls = bind_native(monkeypatch, binding, native)
    item = plan()
    item.asset_cfgs = (SimpleNamespace(cloning_contexts=None), cfg._test_robot, object())
    physics_context = cfg.class_type.clone_context_type
    shared, routing = {0}, {}
    # GA replicate_session routes explicit declarations first, then includes
    # shared sources in the default physics context. Sets deduplicate routes.
    for index in (0, 1):
        references = vars(item.asset_cfgs[index]).get('cloning_contexts', ())
        contexts = (physics_context,) if references is None else tuple(references)
        if index in shared:
            contexts += (physics_context,)
        for context in contexts:
            routing.setdefault(context, set()).add(index)
    assert routing == {PhysxReplicateContext: {0}, clone_context(cfg): {1}}
    default_routes = []
    monkeypatch.setattr(PhysxReplicateContext, 'replicate',
                        lambda self, plan, ids: default_routes.append(ids))
    for context_type, ids in routing.items():
        context_type(SimpleNamespace(stage=37)).replicate(item, tuple(sorted(ids)))
    assert default_routes == [(0,)]
    assert len(calls) == 1 and calls[0][1] == ['/World/envs/env_0/Robot']
    native.attach(37)
    binding.confirm_shared_ground_groups(group_receipt(4))
    binding.finish_warmup()
    assert binding.receipt['physics_asset_prototype_ids'] == [1]
    assert binding.receipt['shared_source_paths'] == ['/World/ground']


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_cpu_or_non_native_device_does_not_change_configuration(configure, device):
    cfg = PhysxCfg()
    with pytest.raises(ValueError, match='CUDA PhysX'):
        configure(cfg, 4, device)
    assert cfg.class_type == '{DIR}.physx_manager:PhysxManager'


def test_wrong_physics_configuration_is_rejected(configure):
    cfg = SimpleNamespace(class_type=PhysxManager)
    with pytest.raises(TypeError, match='PhysxCfg'):
        configure(cfg, 4, 'cuda:0')
    assert cfg.class_type is PhysxManager


@pytest.mark.parametrize('change', ['subtree', 'spawn', 'explicit_context'])
def test_only_a_fresh_complete_robot_declaration_can_be_routed(configure, change):
    cfg, robot = PhysxCfg(), AssetCfg()
    if change == 'subtree':
        robot.prim_path = '/World/envs/env_.*/Robot/base_link'
    elif change == 'spawn':
        robot.spawn = None
    else:
        robot.cloning_contexts = ()
    with pytest.raises(ValueError, match='complete Robot|another clone context'):
        ga_isolation.configure_ga_isolation(cfg, robot, 4, 'cuda:0')
    assert cfg.class_type == '{DIR}.physx_manager:PhysxManager'


def test_complete_8192_plan_uses_existing_native_env_id_registration(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    binding = configure(cfg, 8192, 'cuda:0')
    captured = bind_native(monkeypatch, binding, native)
    context = clone_context(cfg)(SimpleNamespace(stage=37))
    item = plan(8192)
    before = [value.copy() for value in (
        item.topology.world_prototypes, item.topology.world_prototype_starts,
        item.topology.world_prototype_layout, item.positions)]
    rng = torch.get_rng_state().clone()
    context.replicate(item, (np.int32(1),))
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(np.array_equal(previous, current) for previous, current in zip(before, (
        item.topology.world_prototypes, item.topology.world_prototype_starts,
        item.topology.world_prototype_layout, item.positions), strict=True))
    stage, sources, destinations, ids, mapping, options = captured[0]
    assert stage == 37 and sources == ['/World/envs/env_0/Robot']
    assert destinations == ['/World/envs/env_{}/Robot']
    assert torch.equal(ids, torch.arange(8192))
    assert mapping.shape == (1, 8192) and bool(mapping.all())
    assert options == {'device': 'cuda:0', 'use_fabric': False}
    assert native.exclude(37) == ['/World/envs']
    assert native.rename('', 8190) == '/World/envs/env_8191/Robot'
    with pytest.raises(RuntimeError, match='has not been applied'):
        binding.require_applied()
    native.attach(37)
    native.attach(37)
    assert native.calls == [
        (37, sources[0], 8191, {'useEnvIds': True, 'useFabricForReplication': False}),
        (37, sources[0], 8191, {'useEnvIds': True, 'useFabricForReplication': False}),
    ]
    assert native.removed == []
    with pytest.raises(RuntimeError, match='shared-ground groups'):
        binding.finish_warmup()
    readback = group_receipt(8192)
    binding.confirm_shared_ground_groups(readback)
    readback['global_paths'].append('/World/unverified')
    binding.finish_warmup()
    binding.finish_warmup()
    assert native.removed == [37]
    receipt = binding.receipt
    assert receipt['attach_calls'] == 2 and receipt['warmup_finished'] is True
    assert receipt['shared_source_paths'] == ['/World/ground']
    assert receipt['shared_ground_groups']['global_paths'] == ['/World/ground']
    assert receipt['world_ids'] == {'first': 0, 'last': 8191, 'count': 8192}
    receipt['shared_ground_groups']['global_paths'].clear()
    assert binding.receipt['shared_ground_groups']['global_paths'] == ['/World/ground']


def test_robot_only_plan_preserves_manual_shared_ground(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    binding = configure(cfg, 4, 'cuda:0')
    bind_native(monkeypatch, binding, native)
    clone_context(cfg)(SimpleNamespace(stage=37)).replicate(
        plan(shared=False), (np.int32(1),))
    native.attach(37)
    binding.confirm_shared_ground_groups(group_receipt(4))
    binding.finish_warmup()
    assert binding.receipt['shared_source_paths'] == []
    assert binding.receipt['shared_ground_verified']


def test_single_world_requires_plan_and_group_proof_without_zero_replication(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    binding = configure(cfg, 1, 'cuda:0')
    calls = bind_native(monkeypatch, binding, native)
    with pytest.raises(RuntimeError, match='robot plan'):
        binding.finish_warmup()
    clone_context(cfg)(SimpleNamespace(stage=37)).replicate(plan(1), (1,))
    binding.confirm_shared_ground_groups(group_receipt(1))
    binding.finish_warmup()
    assert len(calls) == 1 and native.calls == [] and native.removed == []
    assert binding.receipt['applied'] and binding.receipt['warmup_finished']


@pytest.mark.parametrize('change', [
    'heterogeneous', 'two_robots', 'partial', 'noncontiguous', 'template',
    'source', 'shared_environment', 'world_ids', 'world_starts',
    'missing_robot', 'unrouted', 'duplicate', 'boolean_id', 'nonfinite',
])
def test_ambiguous_ga_plan_never_registers_native(configure, monkeypatch, change):
    cfg, native = PhysxCfg(), Replicator()
    binding = configure(cfg, 4, 'cuda:0')
    calls = bind_native(monkeypatch, binding, native)
    item, ids = plan(), (1,)
    if change == 'heterogeneous':
        item.topology.world_prototype_starts = np.array([0, 1, 2, 3])
    elif change == 'two_robots':
        item.topology.world_prototypes = np.array([0, 1, 1])
        item.topology.world_prototype_starts = np.array([0, 1, 3])
    elif change == 'partial':
        item.topology.world_prototype_layout = np.zeros(3, dtype=np.int32)
    elif change == 'noncontiguous':
        item.topology.world_prototype_layout = np.zeros(8, dtype=np.int32)[::2]
    elif change == 'template':
        item.env_template = '/World/other/env_{}'
    elif change == 'source':
        item.sources = ('/World/ground', '/World/envs/env_1/Robot', None)
    elif change == 'shared_environment':
        item.sources = ('/World/envs/env_0/Cube', item.sources[1], None)
    elif change == 'world_ids':
        item.world_ids = np.array([-1, 0, 2, 1, 3])
    elif change == 'world_starts':
        item.world_starts = np.array([0, 1, 4])
    elif change == 'missing_robot':
        ids = (0,)
    elif change == 'unrouted':
        ids = (0, 1, 2)
    elif change == 'duplicate':
        ids = (0, 1, 1)
    elif change == 'boolean_id':
        ids = (False, 1)
    else:
        item.positions[0, 2] = np.nan
    with pytest.raises(ValueError, match='GA isolation'):
        clone_context(cfg)(SimpleNamespace(stage=37)).replicate(item, ids)
    assert calls == [] and native.calls == [] and not binding.receipt['registered']


def test_a_bound_context_cannot_register_twice(configure, monkeypatch):
    cfg, native = PhysxCfg(), Replicator()
    binding = configure(cfg, 4, 'cuda:0')
    bind_native(monkeypatch, binding, native)
    context = clone_context(cfg)(SimpleNamespace(stage=37))
    context.replicate(plan(), (1,))
    with pytest.raises(RuntimeError, match='already registered'):
        context.replicate(plan(), (1,))


@pytest.mark.parametrize('change', ['count', 'global', 'inverted'])
def test_unverified_usd_groups_cannot_release_registration(change):
    binding = ga_isolation.GaHomogeneousCloneIsolation(4, 'cuda:0')
    receipt = group_receipt(4)
    if change == 'count':
        receipt['environment_groups_verified'] = 3
    elif change == 'global':
        receipt['global_paths'] = ['/World/other']
    else:
        receipt['inverted_filter'] = False
    with pytest.raises(ValueError, match='shared-ground'):
        binding.confirm_shared_ground_groups(receipt)
    assert not binding.receipt['shared_ground_verified']

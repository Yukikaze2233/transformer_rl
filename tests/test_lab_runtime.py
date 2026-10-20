"""Frozen Lab runtime inputs and failure publication, without an SDK."""
from dataclasses import asdict, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_structured_checkpoint import fixture
from transformer_rl.common.training_control import boot_id, file_pin
from transformer_rl.formal.contracts import ACTION_ORDER
from transformer_rl.formal.lab_env.contract import FlatTaskSpec
from transformer_rl.formal.lab_env import runtime as module


def inputs(tmp_path, monkeypatch):
    recipe, components, journal, _ = fixture(tmp_path, full=True, target_rollouts=35000)
    recipe = replace(recipe, device='cuda:0')
    asset = SimpleNamespace(directory=tmp_path / 'robot', manifest_sha256=recipe.asset_manifest_sha256)
    received = []

    def selected(path, *, expected_manifest_sha256):
        received.append((path, expected_manifest_sha256))
        return asset

    monkeypatch.setattr(module, 'load_asset_description', selected)
    response = tmp_path / 'response.json'
    response.write_text(json.dumps({'actuator_response': {
        'enabled': True, 'gain': 1., 'joint_names': ACTION_ORDER,
        'armature': [0.] * 6, 'frictionloss': [0.] * 6,
        'damping': [0.] * 6, 'time_constant_s': [0.] * 6}}))
    mesh = tmp_path / 'terrain.usda'
    mesh.write_text('# unit fixture; not physical geometry\n')
    atlas = tmp_path / 'atlas.json'
    atlas.write_text(json.dumps({'format': 'terrain_course_atlas_v1', 'files': {'terrain.usda': file_pin(mesh)}}))
    config = {'format': 'managed_lab_runtime_v1', 'authorized_boot': boot_id(),
              'asset_directory': str(asset.directory), 'response_source_pin': file_pin(response),
              'terrain_atlas_pin': file_pin(atlas), 'environment_spacing_m': 1.,
              'task': asdict(FlatTaskSpec()), 'pair_domain': None,
              'physics': dict(gpu_found_lost_pairs_capacity=2**22,
                              gpu_found_lost_aggregate_pairs_capacity=2**25,
                              gpu_total_aggregate_pairs_capacity=2**22,
                              gpu_max_rigid_contact_count=2**24, gpu_max_rigid_patch_count=2**22),
              'profile_collection': False}
    return recipe, components, journal, config, received


def test_frozen_runtime_pins_sources_and_preserves_exact_physical_options(tmp_path, monkeypatch):
    recipe, _, journal, config, received = inputs(tmp_path, monkeypatch)
    with journal:
        asset, response, pair, task, atlas, identity = module.parse_configuration(recipe, config, boot_id())
        assert received == [(config['asset_directory'], recipe.asset_manifest_sha256)]
        assert asset.manifest_sha256 == identity['asset_manifest_sha256']
        assert pair is None and task.action_bounds == recipe.collection.action_bounds
        assert atlas.name == 'atlas.json' and len(response.effective_time_constant_s) == 6
        assert identity['atlas_sha256'] == config['terrain_atlas_pin']['sha256']
        assert identity['response_sha256'] == config['response_source_pin']['sha256']
        assert len(identity['runtime_sha256']) == 64


@pytest.mark.parametrize('change', ('boot', 'task', 'capacity', 'profile', 'payload'))
def test_runtime_refuses_changed_contract_before_importing_sdk(tmp_path, monkeypatch, change):
    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    with journal:
        if change == 'boot':
            config['authorized_boot'] = 'not-the-authorized-boot'
        elif change == 'task':
            config['task']['action_bounds'] = [1.] * 6
        elif change == 'capacity':
            config['physics']['gpu_max_rigid_contact_count'] = 0
        elif change == 'profile':
            config['profile_collection'] = 'false'
        else:
            (tmp_path / 'terrain.usda').write_text('changed after freezing')
        with pytest.raises(ValueError):
            module.parse_configuration(recipe, config, boot_id())


def test_sdk_open_failure_is_published_without_building_a_scene(tmp_path, monkeypatch):
    recipe, components, journal, config, _ = inputs(tmp_path, monkeypatch)
    # Configuration parsing is tested separately. Use CPU components for the
    # control/lifecycle test so this test cannot create a CUDA context.
    monkeypatch.setattr(module, 'parse_configuration', lambda *_: (None, None, None, None, None, {}))
    failures = []
    control = SimpleNamespace(binding={'boot_id': boot_id()}, fail=failures.append)
    fake_package, fake_app = ModuleType('isaaclab'), ModuleType('isaaclab.app')

    def unavailable(**_kwargs):
        raise RuntimeError('owned SDK startup failure')

    fake_app.AppLauncher = unavailable
    monkeypatch.setitem(sys.modules, 'isaaclab', fake_package)
    monkeypatch.setitem(sys.modules, 'isaaclab.app', fake_app)
    with journal, pytest.raises(RuntimeError, match='owned SDK startup failure'):
        with module.runtime(replace(recipe, device='cpu'), components, config, control=control):
            pytest.fail('runtime must not yield after a startup failure')
    assert len(failures) == 1 and str(failures[0]) == 'owned SDK startup failure'

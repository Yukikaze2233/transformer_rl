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
from transformer_rl.formal.tasks.termination import NativeTaskLimits
from transformer_rl.formal.control.sensors import SensorModelConfig
from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig
from transformer_rl.formal.lab_env import runtime as module
from transformer_rl.formal.tasks.terrain_assets import file_pin as terrain_file_pin


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
    for name in ('geometry.npz', 'query_index.npz'):
        (tmp_path / name).write_bytes(b'pin-only unit fixture; not an array or physical acceptance')
    atlas = tmp_path / 'atlas.json'
    atlas.write_text(json.dumps({'format': 'terrain_course_atlas_v1', 'files': {
        name: terrain_file_pin(tmp_path / name) for name in ('terrain.usda', 'geometry.npz', 'query_index.npz')}}))
    config = {'format': 'managed_lab_runtime_v4', 'authorized_boot': boot_id(),
              'asset_directory': str(asset.directory), 'response_source_pin': file_pin(response),
              'terrain_atlas_pin': file_pin(atlas), 'environment_spacing_m': 1.,
              'task': asdict(FlatTaskSpec()), 'pair_domain': None,
              'sensors': asdict(SensorModelConfig()), 'dynamics': asdict(StartupDynamicsConfig()),
              'termination_limits': asdict(NativeTaskLimits(
                  .08, .01, .5, 2.5, ('base_link',), 2., .02, .02, .01, 5.)),
              'physics': dict(gpu_found_lost_pairs_capacity=2**22,
                              gpu_found_lost_aggregate_pairs_capacity=2**25,
                              gpu_total_aggregate_pairs_capacity=2**22,
                              gpu_max_rigid_contact_count=2**24, gpu_max_rigid_patch_count=2**22,
                              gpu_collision_stack_size=2**28),
              'profile_collection': False}
    return recipe, components, journal, config, received


def test_frozen_runtime_pins_sources_and_preserves_exact_physical_options(tmp_path, monkeypatch):
    recipe, _, journal, config, received = inputs(tmp_path, monkeypatch)
    with journal:
        asset, response, pair, task, atlas, identity, limits, sensors, dynamics = module.parse_configuration(recipe, config, boot_id())
        assert received == [(config['asset_directory'], recipe.asset_manifest_sha256)]
        assert asset.manifest_sha256 == identity['asset_manifest_sha256']
        assert pair is None and task.action_bounds == recipe.collection.action_bounds
        assert atlas.name == 'atlas.json' and len(response.effective_time_constant_s) == 6
        assert identity['atlas_sha256'] == config['terrain_atlas_pin']['sha256']
        assert identity['response_sha256'] == config['response_source_pin']['sha256']
        assert len(identity['runtime_sha256']) == 64
        assert limits.failure_tilt_rad == 2.5 and limits.forbidden_contact_confirm_s == .02
        assert sensors == SensorModelConfig() and dynamics == StartupDynamicsConfig()


@pytest.mark.parametrize('change', ('boot', 'task', 'capacity', 'stack_missing', 'sensor_missing', 'sensor_clock',
                                    'dynamics_missing', 'profile', 'payload', 'limits_missing', 'limits_value', 'old_format'))
def test_runtime_refuses_changed_contract_before_importing_sdk(tmp_path, monkeypatch, change):
    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    with journal:
        if change == 'boot':
            config['authorized_boot'] = 'not-the-authorized-boot'
        elif change == 'task':
            config['task']['action_bounds'] = [1.] * 6
        elif change == 'capacity':
            config['physics']['gpu_max_rigid_contact_count'] = 0
        elif change == 'stack_missing':
            config['physics'].pop('gpu_collision_stack_size')
        elif change == 'sensor_missing':
            config.pop('sensors')
        elif change == 'sensor_clock':
            config['sensors']['physics_dt_s'] = .001
        elif change == 'dynamics_missing':
            config.pop('dynamics')
        elif change == 'profile':
            config['profile_collection'] = 'false'
        elif change == 'limits_missing':
            config.pop('termination_limits')
        elif change == 'limits_value':
            config['termination_limits']['forbidden_contact_confirm_s'] = 0
        elif change == 'old_format':
            config['format'] = 'managed_lab_runtime_v3'
        else:
            (tmp_path / 'terrain.usda').write_text('changed after freezing')
        with pytest.raises(ValueError):
            module.parse_configuration(recipe, config, boot_id())


def test_sdk_open_failure_is_published_without_building_a_scene(tmp_path, monkeypatch):
    recipe, components, journal, config, _ = inputs(tmp_path, monkeypatch)
    # Configuration parsing is tested separately. Use CPU components for the
    # control/lifecycle test so this test cannot create a CUDA context.
    monkeypatch.setattr(module, 'parse_configuration', lambda *_: (None, None, None, None, None, {}, None, None, None))
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


def test_atlas_relative_payload_pins_survive_directory_relocation(tmp_path, monkeypatch):
    import shutil

    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    moved = tmp_path / 'moved'
    moved.mkdir()
    for name in ('atlas.json', 'terrain.usda', 'geometry.npz', 'query_index.npz'):
        shutil.copyfile(tmp_path / name, moved / name)
    config['terrain_atlas_pin'] = file_pin(moved / 'atlas.json')
    with journal:
        assert module.parse_configuration(recipe, config, boot_id())[4] == moved / 'atlas.json'


def test_atlas_payload_cannot_redirect_the_manifest_to_another_directory(tmp_path, monkeypatch):
    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    atlas = tmp_path / 'atlas.json'
    document = json.loads(atlas.read_text())
    document['files']['../response.json'] = {'file': '../response.json', 'bytes': 1, 'sha256': 'a' * 64}
    atlas.write_text(json.dumps(document))
    config['terrain_atlas_pin'] = file_pin(atlas)
    with journal, pytest.raises(ValueError, match='escapes'):
        module.parse_configuration(recipe, config, boot_id())

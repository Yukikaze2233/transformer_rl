"""The managed preflight accepts actual source-backed atlas manifests."""
from test_lab_runtime import inputs
from test_terrain_assets import real_assets
from test_terrain_courses import atlas

from transformer_rl.common.training_control import boot_id, file_pin
from transformer_rl.formal.lab_env.runtime import parse_configuration


def test_actual_prepared_atlas_passes_managed_preflight_before_sdk_setup(tmp_path, monkeypatch, atlas):
    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    config['terrain_atlas_pin'] = file_pin(atlas)
    with journal:
        selected = parse_configuration(recipe, config, boot_id())
        assert selected[4] == atlas
        assert selected[5]['atlas_sha256'] == file_pin(atlas)['sha256']

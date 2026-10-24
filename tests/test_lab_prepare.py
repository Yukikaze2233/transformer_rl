"""Fresh managed preparation freezes plans without admitting or starting SDKs."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

from test_lab_runtime import inputs
from transformer_rl.common.training_control import read_json
from transformer_rl.formal.lab_env import prepare as module
from transformer_rl.formal.learning.checkpoint import LearningRecipe
from transformer_rl.formal.learning.settings import FormalTrainingSettings


def preparation_inputs(tmp_path, monkeypatch):
    recipe, _, journal, config, _ = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'load_asset_description',
                        lambda _path: type('SelectedAsset', (), {'manifest_sha256': recipe.asset_manifest_sha256})())
    settings = FormalTrainingSettings.load(Path(__file__).parents[1] / 'configs/formal_training.json')
    settings = replace(settings, num_envs=4, rollout_steps=8)
    # The frozen runtime shares the formal collection clock (50 Hz policy).
    config['sensors'] = {**config['sensors'], 'policy_dt_s': settings.collection.policy_dt_s}
    locks = []
    for index in range(4):
        path = tmp_path / f'lease_{index}.lock'
        path.touch()
        info = path.stat()
        locks.append({'path': str(path), 'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid})
    resources = {'gpu_uuid': 'GPU-' + 'a' * 36, 'nvidia_smi': sys.executable,
                 'driver_library_directory': str(tmp_path), 'minimum_ram_bytes': 7 * 1024**3,
                 'minimum_vram_mib': 14336, 'minimum_disk_bytes': 10 * 1024**3, 'locks': locks}
    return settings, config, resources, journal


@pytest.mark.parametrize('include_jump,target,stages', ((True, 35000, 4), (False, 30000, 3)))
def test_preparation_freezes_distinct_full_budget_and_does_not_start_a_worker(
        tmp_path, monkeypatch, include_jump, target, stages):
    settings, config, resources, journal = preparation_inputs(tmp_path, monkeypatch)
    before = torch.get_rng_state().clone()
    with journal:
        session = module.prepare_session(tmp_path / 'new_formal', settings, config, resources,
                                         training_seed=2101, include_jump=include_jump)
        recipe = LearningRecipe.from_dict(read_json(session.root / 'recipe.json'))
        assert session.spec['updates'] == target and recipe.target_rollouts == target
        assert len(recipe.stage_schedule[0]) == stages
        assert recipe.checkpoint_interval == settings.checkpoint_interval == 100
        assert session.spec['runtime_factory'] == 'transformer_rl.formal.lab_env.runtime:runtime'
        assert not list((session.root / 'attempts').iterdir())
        assert not (session.root / 'journal').exists()
        assert not (session.root / 'current.json').exists()
        assert torch.equal(torch.get_rng_state(), before)
        with pytest.raises(FileExistsError, match='cannot replace'):
            module.prepare_session(session.root, settings, config, resources, training_seed=2101)


def test_preparation_refuses_an_unauthorized_boot_before_creating_the_run(tmp_path, monkeypatch):
    settings, config, resources, journal = preparation_inputs(tmp_path, monkeypatch)
    config['authorized_boot'] = 'different-boot'
    root = tmp_path / 'not_created'
    with journal, pytest.raises(ValueError, match='current-boot'):
        module.prepare_session(root, settings, config, resources, training_seed=2101)
    assert not root.exists()

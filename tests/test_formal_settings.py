"""正式采样配置到受控会话的消费路径；不启动仿真或训练。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from transformer_rl.formal.learning.checkpoint import LearningRecipe
from transformer_rl.formal.learning.settings import FormalTrainingSettings
from transformer_rl.formal.runtime.session import StructuredTrainingSession


PROFILE = Path(__file__).resolve().parents[1] / 'configs/formal_training.json'


def test_complete_plan_keeps_rollout_duration_and_accounts_all_three_training_seeds():
    settings = FormalTrainingSettings.load(PROFILE)
    assert settings.num_envs == 8192 and settings.rollout_steps == 48
    assert settings.policy.history_length == 5
    assert settings.policy.body_width == 32 and settings.policy.temporal_width == 64
    assert settings.policy.leg_hidden_dims == settings.policy.wheel_hidden_dims == (64, 32)
    assert settings.policy.critic_input_dim == 185
    assert settings.policy.critic_hidden_dims == (256, 128, 64)
    assert settings.collection.gamma**2 == pytest.approx(.99)
    assert settings.collection.gae_lambda**2 == pytest.approx(.95)
    assert settings.budget() == {
        'fresh_per_update': 393216, 'fresh_per_seed': 7864320000, 'total_fresh': 23592960000,
        'rollout_duration_s': .48, 'logical_minibatch_endpoints': 24576,
        'maximum_actor_steps_per_update': 64, 'maximum_actor_endpoint_uses_per_update': 1572864}
    for seed in settings.training_seeds:
        recipe = settings.recipe('formal_' + str(seed), seed, asset_manifest_sha256='a' * 64,
                                 learner_source_sha256='b' * 64, task_contract_sha256='c' * 64,
                                 initial_model_sha256='d' * 64)
        assert LearningRecipe.from_dict(recipe.to_dict()) == recipe
        assert recipe.journal_spec().num_envs == 8192
        assert recipe.journal_spec().target_rollouts == 20000
        assert recipe.checkpoint_interval == 100 and recipe.curriculum_promotion == 'exposure'
    assert FormalTrainingSettings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize('change', ['extra', 'seed', 'updates', 'epoch', 'batch', 'chunk', 'clock'])
def test_invalid_formal_profile_does_not_silently_change_budget_or_timing(change):
    value = deepcopy(json.loads(PROFILE.read_text()))
    if change == 'extra':
        value['unused_typo'] = 42
    elif change == 'seed':
        value['training_seeds'] = [2101, 2101]
    elif change == 'updates':
        value['updates_per_seed'] = 1000
    elif change == 'epoch':
        value['ppo']['epochs'] = 0
    elif change == 'batch':
        value['ppo']['minibatches'] = 7
    elif change == 'chunk':
        value['ppo']['training_chunk'] = True
    elif change == 'clock':
        value['collection']['policy_dt_s'] = .02
    with pytest.raises(ValueError):
        FormalTrainingSettings.from_dict(value)


def test_settings_deployment_freezes_resolved_recipe_without_creating_an_attempt(tmp_path):
    import transformer_rl

    source = Path(transformer_rl.__file__).parent
    locks = []
    for index in range(4):
        path = tmp_path / f'resource_{index}.lock'
        path.touch()
        info = path.stat()
        locks.append({'path': str(path), 'device': info.st_dev, 'inode': info.st_ino, 'uid': info.st_uid})
    driver = tmp_path / 'driver'
    driver.mkdir()
    resources = {'gpu_uuid': 'GPU-' + 'a' * 36, 'nvidia_smi': sys.executable,
                 'driver_library_directory': str(driver), 'minimum_ram_bytes': 7 * 1024**3,
                 'minimum_vram_mib': 14336, 'minimum_disk_bytes': 10 * 1024**3, 'locks': locks}
    session = StructuredTrainingSession.create_from_settings(
        tmp_path / 'formal', PROFILE, run_id='formal_2101', training_seed=2101,
        asset_manifest_sha256='a' * 64, task_contract_sha256='c' * 64, initial_model_sha256='d' * 64,
        runtime_factory='transformer_rl.formal.runtime.worker:runtime', runtime_configuration={},
        resources=resources, device='cpu', learner_source=source)
    recipe = LearningRecipe.from_dict(json.loads((session.root / 'recipe.json').read_text()))
    assert recipe.num_envs == session.spec['num_envs'] == 8192
    assert recipe.rollout_steps == 48 and recipe.checkpoint_interval == 100
    assert recipe.learner_source_sha256 == StructuredTrainingSession.source_sha256(source)
    assert not list((session.root / 'attempts').iterdir())
    assert not session.spec['automatic_evaluation']

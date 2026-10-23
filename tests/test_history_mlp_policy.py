"""MLP architecture and formal-stack compatibility on synthetic CPU data."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from test_policy_optimization import assert_identical, batch_for
from test_structured_checkpoint import fixture as checkpoint_fixture, update
from test_structured_policy import observation
from transformer_rl.common.random_state import capture_rng
from transformer_rl.formal.learning.checkpoint import LearningRecipe, StructuredCheckpointStore
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.settings import FormalTrainingSettings
from transformer_rl.formal.mlp_policy import HistoryMLPConfig, HistoryMLPPolicy
from transformer_rl.formal.policy import (
    StructuredActorCritic, StructuredPolicyConfig, policy_config_from_dict,
)


@pytest.fixture(autouse=True)
def seeded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2101)
        yield
    torch.set_num_threads(threads)


def test_shapes_parameter_budget_and_independent_critic():
    model = StructuredActorCritic(HistoryMLPConfig())
    policy = model.actor.policy
    assert isinstance(policy, HistoryMLPPolicy)
    mean, state, context = policy.outputs(observation().frames)
    assert (mean.shape, state.shape, context.shape) == ((4, 6), (4, 8), (4, 128))
    assert policy.action_decoder[0].in_features == 171
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(policy.for_deployment()) == 93742
    assert count(model.actor) == 98266
    assert count(model.critic) == 88833
    assert count(model) == 187099
    assert not any(isinstance(m, (nn.MultiheadAttention, nn.Dropout, nn.GRU, nn.LSTM))
                   for m in model.modules())
    assert not ({id(p) for p in model.actor.parameters()} & {id(p) for p in model.critic.parameters()})


def test_history_and_current_commands_are_used_without_wheel_position_leakage():
    policy = HistoryMLPPolicy()
    frames = observation().frames.requires_grad_()
    baseline = policy(frames)
    gradient, = torch.autograd.grad(baseline.square().sum(), frames)
    assert gradient[:, 0].abs().sum() > 0
    assert gradient[:, -1, :4].abs().sum() > 0
    assert gradient[..., 20:22].abs().sum() > 0
    assert torch.count_nonzero(gradient[..., 14:16]) == 0
    for value in (12345., float('nan')):
        changed = frames.detach().clone()
        changed[..., 14:16] = value
        torch.testing.assert_close(policy(changed), baseline, atol=0, rtol=0)
        with pytest.raises(ValueError):
            policy.validate_frames(changed)
    changed = frames.detach().clone()
    changed[:, 0, 4:10] += 2
    assert not torch.allclose(policy(changed), baseline)
    changed = frames.detach().clone()
    changed[:, -1, :4] += 2
    assert not torch.allclose(policy(changed), baseline)
    policy(observation(7).frames)
    torch.testing.assert_close(policy(frames), baseline, atol=0, rtol=0)
    policy.validate_frames(frames)


def test_state_auxiliary_and_prediction_keep_existing_gradient_boundaries():
    model = StructuredActorCritic(HistoryMLPConfig())
    history = observation()
    raw_action = torch.zeros(4, 6)
    evaluation, state, prediction = model.actor.evaluate_training(history, raw_action, raw_action)
    changed = model.actor.evaluate_training(history, raw_action, torch.ones_like(raw_action))
    torch.testing.assert_close(changed[0].mean, evaluation.mean, atol=0, rtol=0)
    torch.testing.assert_close(changed[1], state, atol=0, rtol=0)
    assert not torch.allclose(changed[2], prediction)
    environment = torch.randn(4, 182, requires_grad=True)
    features = model.critic_features(environment, state)
    assert not features.requires_grad
    model.critic(features).square().mean().backward()
    assert environment.grad is None
    assert all(p.grad is None for p in model.actor.parameters())
    model.zero_grad(set_to_none=True)
    evaluation.mean.square().sum().backward(retain_graph=True)
    assert model.actor.policy.state_head[0].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    (state.square().mean() + prediction.square().mean()).backward()
    assert model.actor.policy.history_encoder[0].weight.grad.abs().sum() > 0
    assert model.actor.policy.action_decoder[0].weight.grad is None


@pytest.mark.parametrize('changes', [
    {'architecture': 'unknown'}, {'history_length': 0}, {'history_length': True},
    {'history_hidden_dims': [256]}, {'history_hidden_dims': [256, False]},
    {'critic_hidden_dims': [256, 128]}, {'critic_input_dim': 0},
    {'action_hidden_dim': False}, {'state_hidden_dim': 0}, {'prediction_hidden_dim': 0},
    {'mean_init_scale': -1}, {'estimate_log_std_min': 2}, {'estimate_log_std_max': float('nan')},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(HistoryMLPConfig(), **changes)


def test_explicit_architecture_parser_preserves_legacy_and_rejects_mixed_fields():
    for config in (StructuredPolicyConfig(), HistoryMLPConfig()):
        value = json.loads(json.dumps(asdict(config)))
        assert policy_config_from_dict(value, complete=True) == config
    assert 'architecture' not in asdict(StructuredPolicyConfig())
    assert policy_config_from_dict({}) == StructuredPolicyConfig()
    assert policy_config_from_dict({'architecture': 'history_mlp'}) == HistoryMLPConfig()
    for value in ({'architecture': 'transformer'}, {'architecture': 'history_mlp', 'body_width': 32},
                  {'history_hidden_dims': [256, 128]}, []):
        with pytest.raises(ValueError):
            policy_config_from_dict(value)
    with pytest.raises(ValueError, match='fields differ'):
        policy_config_from_dict({'architecture': 'history_mlp'}, complete=True)


def test_candidate_settings_only_replace_policy_and_roundtrip_recipe():
    root = Path(__file__).resolve().parents[1]
    original = FormalTrainingSettings.load(root / 'configs/formal_training.json')
    candidate = FormalTrainingSettings.load(root / 'configs/formal_training_mlp.json')
    assert candidate == replace(original, policy=HistoryMLPConfig(),
                                ppo=replace(original.ppo, kl_early_stop=False))
    assert candidate.budget() == original.budget()
    assert FormalTrainingSettings.from_dict(candidate.to_dict()) == candidate
    recipe = candidate.recipe('mlp_fixture', 2102, asset_manifest_sha256='a' * 64,
                              learner_source_sha256='b' * 64, task_contract_sha256='c' * 64,
                              initial_model_sha256='d' * 64, device='cpu')
    assert LearningRecipe.from_dict(recipe.to_dict()) == recipe
    assert recipe.to_dict()['policy']['architecture'] == 'history_mlp'
    assert recipe.make_curriculum().stage_budgets == (5000, 20000, 5000, 5000)


def test_shared_gradient_diagnostics_cover_encoder_and_exclude_action_head():
    model = StructuredActorCritic(HistoryMLPConfig())
    trainer = StructuredPPOTrainer(model)
    shared = {id(p) for p in trainer._shared_parameters}
    assert shared == {id(p) for p in model.actor.policy.history_encoder.parameters()}
    batch = batch_for(model)
    result = trainer._gradient_diagnostics(batch, torch.arange(len(batch)), batch.advantages)
    for name in ('policy_entropy', 'velocity', 'contact', 'prediction', 'combined'):
        value = result['shared_gradient_norm/' + name]
        assert torch.isfinite(value) and value > 0


def test_ppo_updates_mlp_with_existing_raw_action_probabilities_and_accounting():
    model = StructuredActorCritic(HistoryMLPConfig())
    config = replace(StructuredPPOConfig(), epochs=1, minibatches=2, kl_early_stop=False)
    trainer = StructuredPPOTrainer(model, config)
    batch = batch_for(model)
    before = deepcopy(model.state_dict())
    result = trainer.update(batch)
    assert result['retained_actor_steps'] == result['critic_optimizer_steps'] == 2
    assert result['rolled_back_actor_steps'] == 0
    assert result['fresh_endpoints'] == len(batch)
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_script_only_rows_do_not_update_mlp_actor():
    model = StructuredActorCritic(HistoryMLPConfig())
    trainer = StructuredPPOTrainer(model, replace(StructuredPPOConfig(), kl_early_stop=False))
    before = deepcopy(model.actor.state_dict())
    result = trainer.update(batch_for(model, owned=False))
    assert result['actor_endpoint_uses'] == result['retained_actor_steps'] == 0
    assert_identical(model.actor.state_dict(), before)


def test_chunked_auxiliary_and_ppo_gradients_match_the_complete_minibatch():
    full = StructuredActorCritic(HistoryMLPConfig())
    chunked = deepcopy(full)
    batch = batch_for(full)
    config = replace(StructuredPPOConfig(), kl_early_stop=False)
    trainer = StructuredPPOTrainer(full, config)
    other = StructuredPPOTrainer(chunked, replace(config, training_chunk=3))
    endpoints = torch.arange(len(batch))
    trainer._actor_backward(batch, endpoints, batch.advantages)
    other._actor_backward(batch, endpoints, batch.advantages)
    for left, right in zip(full.actor.parameters(), chunked.actor.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=3e-7, rtol=3e-5)


def test_managed_session_records_selected_architecture_without_starting(tmp_path, monkeypatch):
    from test_structured_session import session
    managed, _ = session(tmp_path, monkeypatch, policy_config=HistoryMLPConfig())
    assert managed.spec['architecture'] == 'history_mlp'
    assert managed.spec['history_length'] == 5
    assert managed.attempt() is None
    recipe = LearningRecipe.from_dict(json.loads((managed.root / 'recipe.json').read_text()))
    assert recipe.policy == HistoryMLPConfig()


def test_mlp_complete_checkpoint_restores_next_update_adam_rng_and_course(tmp_path):
    recipe, original, journal, store = checkpoint_fixture(
        tmp_path, target_rollouts=35000, policy_config=HistoryMLPConfig())
    with journal:
        update(original, journal)
        before = capture_rng()
        store.save('mlp.pt', original)
        loaded = store.load('mlp.pt')
        assert_identical(capture_rng(), before)
        assert loaded.recipe == recipe
        assert loaded.components.curriculum.stage_budgets == (5000, 20000, 5000, 5000)
        assert_identical(loaded.components.model.state_dict(), original.model.state_dict())
        original.trainer.update(batch_for(original.model))
        expected_rng = capture_rng()
        loaded.restore_global_rng_after_cold_setup()
        loaded.components.trainer.update(batch_for(loaded.components.model))
        assert_identical(loaded.components.model.state_dict(), original.model.state_dict())
        assert_identical(loaded.components.trainer.actor_optimizer.state_dict(),
                         original.trainer.actor_optimizer.state_dict())
        assert_identical(loaded.components.trainer.critic_optimizer.state_dict(),
                         original.trainer.critic_optimizer.state_dict())
        assert_identical(capture_rng(), expected_rng)
        with pytest.raises(ValueError, match='model fields differ'):
            StructuredCheckpointStore._validate_weights(original.model.state_dict(), StructuredActorCritic())
        with pytest.raises(RuntimeError):
            StructuredActorCritic().load_state_dict(original.model.state_dict(), strict=True)


def test_script_export_and_dynamic_batches_exclude_training_prediction_head():
    policy = HistoryMLPPolicy().for_deployment()
    assert policy.prediction_head is None
    stream = io.BytesIO()
    torch.jit.save(torch.jit.script(policy), stream)
    stream.seek(0)
    loaded = torch.jit.load(stream)
    assert not any(name.startswith('prediction_head.') for name in loaded.state_dict())
    for batch in (1, 7):
        frames = observation(batch).frames
        torch.testing.assert_close(loaded(frames), policy(frames), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(loaded.encode(frames), policy.encode(frames), atol=1e-6, rtol=1e-5)


def test_onnx_dynamic_batches_match_eager_policy():
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    policy = HistoryMLPPolicy().for_deployment()
    stream = io.BytesIO()
    torch.onnx.export(policy, (observation(1).frames,), stream, input_names=['frames'],
                      output_names=['raw_mean'], dynamic_axes={'frames': {0: 'batch'},
                                                               'raw_mean': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(stream.getvalue(), providers=['CPUExecutionProvider'])
    for batch in (1, 7):
        frames = observation(batch).frames
        output = torch.from_numpy(session.run(None, {'frames': frames.numpy()})[0])
        torch.testing.assert_close(output, policy(frames), atol=2e-6, rtol=1e-4)

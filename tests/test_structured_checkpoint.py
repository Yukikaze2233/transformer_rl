"""Complete candidate snapshots using synthetic learning, without robot steps."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_policy_optimization import assert_identical, batch_for, model as small_model
from transformer_rl.formal.learning.accounting import LearningJournal, RecoveryConsumptionMismatch
from transformer_rl.formal.learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl.formal.learning.checkpoint import LearningComponents, LearningRecipe, StructuredCheckpointStore
from transformer_rl.formal.learning.collection import StructuredCollectionConfig
from transformer_rl.formal.policy import StructuredActorCritic
from transformer_rl.formal.tasks.curriculum import ContinuousTaskSampler, CurriculumBudget, TaskDomain
from transformer_rl.formal.tasks.protocol import RequestProtocol, RequestProtocolConfig
from transformer_rl.formal.tasks.terrain import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning
from transformer_rl.formal.tasks.geometry import TerrainKind
from transformer_rl.research.frame_checkpoint import capture_rng, restore_rng
from transformer_rl.common.training_control import boot_id


@pytest.fixture(autouse=True)
def rng_and_threads():
    before, threads = capture_rng(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(2101)
    yield
    restore_rng(before)
    torch.set_num_threads(threads)


def fixture(tmp_path, full=False, target_rollouts=30000):
    network = StructuredActorCritic() if full else small_model()
    ppo = replace(StructuredPPOConfig(), epochs=1, minibatches=1, probe_endpoints=16,
                  evaluation_chunk=32, learning_rate=1e-6)
    terrain = TerrainTuning(.7, (.2, 1.1), .25)
    recipe = LearningRecipe('fixture', 2101, 4, 8, 'a' * 64, 'b' * 64, 'c' * 64,
                            StructuredCheckpointStore.model_sha256(network), network.config,
                            ppo, StructuredCollectionConfig(), TaskDomain(), RequestProtocolConfig(), terrain,
                            target_rollouts=target_rollouts)
    components = LearningComponents(network, StructuredPPOTrainer(network, ppo), recipe.make_curriculum(),
                                    recipe.make_sampler(), ContinuousTerrainGenerator(2101, terrain), RequestProtocol(4))
    journal = LearningJournal.create(tmp_path / 'journal', recipe.journal_spec(), authorized_boot=boot_id())
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    return recipe, components, journal, StructuredCheckpointStore(directory, recipe, journal)


def update(components, journal, index=1):
    journal.reserve(components.curriculum.stage, active_elapsed_s=2 * index - 1)
    components.curriculum.charge_rollout()
    batch = batch_for(components.model)
    components.attempted_vector_steps += 8
    components.returned_vector_steps += 8
    components.trainer.update(batch)
    journal.finish(components.observed_consumption(4), active_elapsed_s=2 * index)


def test_legacy_recipe_keeps_identity_and_complete_schedule_is_explicit(tmp_path):
    recipe, _, journal, _ = fixture(tmp_path)
    with journal:
        legacy = recipe.to_dict()
        assert 'target_rollouts' not in legacy
        restored = LearningRecipe.from_dict(legacy)
        assert restored.sha256 == recipe.sha256
        assert restored.journal_spec().target_rollouts == 30000
        complete = replace(recipe, target_rollouts=35000)
        assert complete.to_dict()['target_rollouts'] == 35000
        assert complete.sha256 != recipe.sha256
        assert LearningRecipe.from_dict(complete.to_dict()) == complete


def test_complete_state_preserves_both_adams_quota_rng_and_nonrefundable_counts(tmp_path):
    recipe, state, journal, store = fixture(tmp_path)
    with journal:
        update(state, journal)
        state.sampler.sample(17, 0)
        state.terrain.build(TerrainRequest(TerrainKind.FLAT, TerrainExtent((-4., 4.), (-3., 3.))))
        before = capture_rng()
        receipt = store.save('pause.pt', state)
        assert_identical(capture_rng(), before)
        loaded = store.load('pause.pt')
        assert_identical(capture_rng(), before)
        assert_identical(loaded.components.model.state_dict(), state.model.state_dict())
        assert_identical(loaded.components.trainer.actor_optimizer.state_dict(), state.trainer.actor_optimizer.state_dict())
        assert_identical(loaded.components.trainer.critic_optimizer.state_dict(), state.trainer.critic_optimizer.state_dict())
        assert_identical(loaded.components.sampler.state_dict(), state.sampler.state_dict())
        assert_identical(loaded.components.terrain.state_dict(), state.terrain.state_dict())
        assert loaded.components.curriculum.state_dict() == state.curriculum.state_dict()
        assert loaded.components.trainer.consumed == state.trainer.consumed
        assert receipt['boundary']['reserved_transitions'] == 32 and not receipt['cold_physical_state_restored']


def test_explicit_35k_snapshot_restores_four_stage_budget_and_jump_sampler(tmp_path):
    recipe, state, journal, store = fixture(tmp_path, target_rollouts=35000)
    with journal:
        update(state, journal)
        store.save('complete_course.pt', state)
        restored = store.load('complete_course.pt').components
        assert restored.curriculum.stage_names == ('warmup', 'rough', 'hard', 'jump')
        assert restored.curriculum.stage_budgets == (5000, 20000, 5000, 5000)
        assert restored.curriculum.consumed == [1, 0, 0, 0]
        assert restored.trainer.consumed == state.trainer.consumed
        assert len(journal.summary()['reserved_by_stage']) == 4
        draw = restored.sampler.sample(1000, 3)
        assert torch.bincount(draw.family, minlength=6).tolist() == [200, 100, 100, 100, 500, 0]
        assert recipe.journal_spec().target_rollouts == 35000


def test_35k_recipe_does_not_accept_legacy_three_stage_components(tmp_path):
    recipe, state, journal, store = fixture(tmp_path, target_rollouts=35000)
    with journal:
        state.curriculum = CurriculumBudget()
        state.sampler = ContinuousTaskSampler(2101)
        with pytest.raises(ValueError, match='source-bound recipe'):
            store.save('wrong_course.pt', state)
        assert not (store.directory / 'wrong_course.pt').exists()


def test_extra_task_rng_survives_snapshot_and_must_restore_before_cold_sampling(tmp_path):
    class TaskSource:
        def __init__(self, seed):
            self.generator = torch.Generator().manual_seed(seed)
            self.draws = 0

        def draw(self):
            self.draws += 1
            return torch.rand(7, generator=self.generator)

        def checkpoint_state_dict(self):
            return {'pose_rng': self.generator.get_state(), 'draws': self.draws}

        def load_after_explicit_cold_setup(self, state):
            self.generator.set_state(state['pose_rng'])
            self.draws = state['draws']

    _, original, journal, store = fixture(tmp_path)
    original.task_source = TaskSource(9081)
    original.task_source.draw()
    with journal:
        update(original, journal)
        before = capture_rng()
        store.save('with_task.pt', original)
        loaded = store.load('with_task.pt')
        assert_identical(capture_rng(), before)
        with pytest.raises(RuntimeError, match='task source'):
            loaded.restore_global_rng_after_cold_setup()
        with pytest.raises(ValueError, match='deferred task source'):
            store.save('not_ready.pt', loaded.components)
        assert not (store.directory / 'not_ready.pt').exists()
        expected = original.task_source.draw()
        cold = TaskSource(0)
        loaded.restore_task_source(cold)
        assert loaded.components.task_source is cold
        assert loaded.components.deferred_task_source_state is None
        assert torch.equal(cold.draw(), expected)
        assert cold.draws == original.task_source.draws
        with pytest.raises(ValueError, match='deferred state'):
            loaded.restore_task_source(cold)
        loaded.restore_global_rng_after_cold_setup()


def test_restored_next_update_matches_weights_adam_and_rng_after_explicit_rng_restore(tmp_path):
    _, original, journal, store = fixture(tmp_path)
    with journal:
        update(original, journal)
        store.save('clean.pt', original)
        loaded = store.load('clean.pt')
        reference_batch = batch_for(original.model)
        original.trainer.update(reference_batch)
        expected_rng = capture_rng()
        loaded.restore_global_rng_after_cold_setup()
        resumed_batch = batch_for(loaded.components.model)
        loaded.components.trainer.update(resumed_batch)
        assert_identical(loaded.components.model.state_dict(), original.model.state_dict())
        assert_identical(loaded.components.trainer.actor_optimizer.state_dict(), original.trainer.actor_optimizer.state_dict())
        assert_identical(loaded.components.trainer.critic_optimizer.state_dict(), original.trainer.critic_optimizer.state_dict())
        assert_identical(capture_rng(), expected_rng)
        with pytest.raises(RuntimeError, match='already'):
            loaded.restore_global_rng_after_cold_setup()


def test_cold_restore_aborts_physical_requests_but_keeps_consumed_ids(tmp_path):
    _, state, journal, store = fixture(tmp_path)
    with journal:
        state.protocol.max_seen_id[:] = torch.tensor([7, 3, -1, 5])
        state.protocol.active_id[:] = torch.tensor([7, -1, -1, 5])
        state.protocol.height_m[:] = torch.tensor([.1, 0., 0., .2])
        state.protocol.accepted_at_s[:] = torch.tensor([.1, 0., 0., .1], dtype=torch.float64)
        state.protocol.last_time_s.fill_(.2)
        store.save('requests.pt', state)
        restored = store.load('requests.pt')
        assert restored.aborted_request_ids.tolist() == [7, -1, -1, 5]
        assert restored.components.protocol.max_seen_id.tolist() == [7, 3, -1, 5]
        assert restored.components.protocol.active_id.eq(-1).all()
        assert not restored.components.protocol.height_m.any()


def test_real_width_candidate_and_initialized_adams_roundtrip(tmp_path):
    _, state, journal, store = fixture(tmp_path, full=True)
    with journal:
        update(state, journal)
        receipt = store.save('full.pt', state)
        loaded = store.load('full.pt')
        assert sum(p.numel() for p in loaded.components.model.parameters()) == 233720
        assert loaded.components.trainer.actor_optimizer.state and loaded.components.trainer.critic_optimizer.state
        assert receipt['bytes'] > 2 * sum(p.numel() * p.element_size()
                                         for p in loaded.components.model.parameters())


def test_new_consumption_or_pending_reservation_refuses_checkpoint_rollback(tmp_path):
    _, state, journal, store = fixture(tmp_path)
    with journal:
        store.save('initial.pt', state)
        journal.reserve(0, active_elapsed_s=.1)
        with pytest.raises(RecoveryConsumptionMismatch):
            store.load('initial.pt')


def test_no_clobber_and_missing_completion_marker_are_not_restorable(tmp_path):
    _, state, journal, store = fixture(tmp_path)
    with journal:
        store.save('initial.pt', state)
        with pytest.raises(FileExistsError):
            store.save('initial.pt', state)
        (store.directory / 'unsealed.pt').write_bytes((store.directory / 'initial.pt').read_bytes())
        with pytest.raises(ValueError, match='receipt'):
            store.load('unsealed.pt')
        with pytest.raises(ValueError, match='inside'):
            store.load('../other.pt')


@pytest.mark.parametrize('change', ['fixed_mask', 'actor_moment', 'critic_moment', 'recipe', 'budget',
                                  'fresh_count', 'protocol', 'normalization'])
def test_invalid_complete_payload_is_rejected_without_touching_live_model_or_rng(tmp_path, change):
    _, state, journal, store = fixture(tmp_path)
    with journal:
        update(state, journal)
        store.save('good.pt', state)
        payload = torch.load(store.directory / 'good.pt', weights_only=True)
        if change == 'fixed_mask':
            payload['model']['actor.policy.time_allowed'].fill_(True)
        elif change in ('actor_moment', 'critic_moment'):
            key = 'actor_adam' if change == 'actor_moment' else 'critic_adam'
            entry = next(iter(payload[key]['state'].values()))
            entry['exp_avg_sq'].fill_(-1)
        elif change == 'recipe':
            payload['recipe']['task_contract_sha256'] = 'd' * 64
        elif change == 'budget':
            payload['curriculum']['consumed'][0] += 1
        elif change == 'fresh_count':
            payload['trainer']['consumed']['fresh_endpoints'] += 1
        elif change == 'normalization':
            payload['normalization'] = 'fixed_manual35_81_no_running_statistics'
        else:
            payload['protocol']['tensors']['max_seen_id'][0] = -2
        before, weights = capture_rng(), deepcopy(state.model.state_dict())
        with pytest.raises(ValueError):
            store._validate_payload(payload)
        assert_identical(state.model.state_dict(), weights)
        assert_identical(capture_rng(), before)


def test_initialization_hash_is_checked_before_first_snapshot(tmp_path):
    recipe, state, journal, store = fixture(tmp_path)
    with journal:
        with torch.no_grad():
            next(state.model.parameters()).add_(.01)
        with pytest.raises(ValueError, match='initialization'):
            store.save('wrong.pt', state)


def test_manifest_hash_is_verified_before_deserialization(tmp_path):
    _, state, journal, store = fixture(tmp_path)
    with journal:
        store.save('clean.pt', state)
        raw = bytearray((store.directory / 'clean.pt').read_bytes())
        raw[-1] ^= 1
        (store.directory / 'clean.pt').write_bytes(raw)
        with pytest.raises(ValueError, match='bytes'):
            store.load('clean.pt')

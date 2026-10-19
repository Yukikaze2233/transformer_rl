"""Complete candidate snapshots using synthetic learning, without robot steps."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_policy_optimization import assert_identical, batch_for, model as small_model
from transformer_rl._learning_accounting import LearningJournal, RecoveryConsumptionMismatch
from transformer_rl._policy_optimization import StructuredPPOConfig, StructuredPPOTrainer
from transformer_rl._structured_checkpoint import LearningComponents, LearningRecipe, StructuredCheckpointStore
from transformer_rl._structured_collection import StructuredCollectionConfig
from transformer_rl._structured_policy import StructuredActorCritic
from transformer_rl._task_curriculum import ContinuousTaskSampler, CurriculumBudget, TaskDomain
from transformer_rl._task_protocol import RequestProtocol, RequestProtocolConfig
from transformer_rl._terrain_tasks import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning
from transformer_rl._terrain_geometry import TerrainKind
from transformer_rl.frame_checkpoint import capture_rng, restore_rng
from transformer_rl.training_control import boot_id


@pytest.fixture(autouse=True)
def rng_and_threads():
    before, threads = capture_rng(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(2101)
    yield
    restore_rng(before)
    torch.set_num_threads(threads)


def fixture(tmp_path, full=False):
    network = StructuredActorCritic() if full else small_model()
    ppo = replace(StructuredPPOConfig(), epochs=1, minibatches=1, probe_endpoints=16,
                  evaluation_chunk=32, learning_rate=1e-6)
    terrain = TerrainTuning(.7, (.2, 1.1), .25)
    recipe = LearningRecipe('fixture', 2101, 4, 8, 'a' * 64, 'b' * 64, 'c' * 64,
                            StructuredCheckpointStore.model_sha256(network), network.config,
                            ppo, StructuredCollectionConfig(), TaskDomain(), RequestProtocolConfig(), terrain)
    components = LearningComponents(network, StructuredPPOTrainer(network, ppo), CurriculumBudget(),
                                    ContinuousTaskSampler(2101), ContinuousTerrainGenerator(2101, terrain), RequestProtocol(4))
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
        assert sum(p.numel() for p in loaded.components.model.parameters()) == 408824
        assert loaded.components.trainer.actor_optimizer.state and loaded.components.trainer.critic_optimizer.state
        assert receipt['bytes'] > 3_000_000


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


@pytest.mark.parametrize('change', ['fixed_mask', 'actor_moment', 'critic_moment', 'recipe', 'budget', 'fresh_count', 'protocol'])
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

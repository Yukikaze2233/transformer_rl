"""Partial proposal bookkeeping with explicit synthetic native authority."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_reset_program import fixture as program_fixture, draw
from transformer_rl.formal.simulation.episode import NativeEpisodeLifecycle
from transformer_rl.formal.runtime.reset_coordinator import ResetBatchCoordinator, ResetSourceNotReady


def controlled_draws(program, sizes):
    def sample(count, stage):
        sizes.append(count)
        value = draw([0] * count)
        value.requested_command[:, 0] = program.tasks._random(count)
        return value

    program.tasks.sample = sample


def fixture():
    program, controller, origins = program_fixture()
    program.tuning = replace(program.tuning, xy_jitter_m=0, roll_pitch_jitter_rad=0,
                              inner_knee_jitter_rad=0, leg_pitch_jitter_rad=0)
    clock = torch.zeros(2, dtype=torch.float64)

    class Observation(SimpleNamespace):
        @property
        def physical_tick(self):
            return round(float(clock[0]) / .005)

    observation = Observation(binding=program.geometry.kinematics.binding,
                                   timestamp_s=lambda: clock.clone(), _local_com_m=torch.zeros(2, 3))
    scene = SimpleNamespace(observation=observation, environment_origins_m=origins.float(),
                             readback={'selected_asset_manifest_sha256': program.geometry.kinematics.manifest_sha256,
                                       'selected_model_spec_sha256': program.geometry.kinematics.model_sha256})
    # Only the authority surface used by the planner is synthetic here. Actual
    # native writes and same-surface metadata adoption have separate tests.
    life = object.__new__(NativeEpisodeLifecycle)
    life.num_envs, life.device = 2, torch.device('cpu')
    life.plant = SimpleNamespace(scene=scene, controller=controller)
    life.episode_id, life._last = torch.full((2,), -1, dtype=torch.long), None
    life._last_reset_tick = torch.full((2,), -1, dtype=torch.long)
    life._course_sha256 = ('0' * 64, '1' * 64)
    life._identity = lambda: None
    sizes = []
    controlled_draws(program, sizes)
    return ResetBatchCoordinator(life, program, stage_source=lambda: 0), life, clock, sizes


def request(coordinator, life, mask=None):
    mask = torch.ones(2, dtype=torch.bool) if mask is None else mask
    return coordinator(mask, life.episode_id + mask.long(), life.plant.scene.observation.timestamp_s())


def apply(life, plan, mask=None):
    mask = torch.ones(2, dtype=torch.bool) if mask is None else mask
    life.episode_id += mask.long()
    life._course_sha256, life._last = plan.course_sha256, object()
    life._last_reset_tick[mask] = life.plant.scene.observation.physical_tick


def test_checkpoint_source_requires_the_reset_proposal_to_be_applied():
    coordinator, life, _, _ = fixture()
    plan = request(coordinator, life)
    with pytest.raises(RuntimeError, match='unapplied reset proposal'):
        coordinator.checkpoint_state_dict()
    apply(life, plan)
    state = coordinator.checkpoint_state_dict()
    assert state['status'] == 'ready_for_explicit_cold_setup'
    assert state['pending_proposal'] is None


def test_partial_resets_draw_only_selected_rows_and_keep_every_other_plan_field_and_course_identity():
    coordinator, life, clock, sizes = fixture()
    first = request(coordinator, life)
    apply(life, first)
    clock += .1
    second = request(coordinator, life, torch.tensor([False, True]))
    assert sizes == [2, 1] and coordinator.preparer.requested_rows == 3
    for name in ('root_pose_xyzw', 'root_com_velocity', 'joint_position', 'joint_velocity'):
        assert torch.equal(getattr(first, name)[0], getattr(second, name)[0])
    assert first.preparation_issued_action is None and second.preparation_issued_action is None
    for name in ('requested_action', 'leg_position_rad', 'common_turns', 'admitted', 'target_error_rad', 'pair_admitted'):
        assert torch.equal(getattr(first.preparation_hold, name)[0], getattr(second.preparation_hold, name)[0])
    for name in ('start', 'end', 'duration_s', 'kind', 'frequency_start_hz', 'frequency_end_hz'):
        assert torch.equal(getattr(first.commands.trajectory, name)[0], getattr(second.commands.trajectory, name)[0])
    for name in ('base_mode', 'task_horizon_s', 'jump_height_m', 'request_interval_s'):
        assert torch.equal(getattr(first.commands, name)[0], getattr(second.commands, name)[0])
    assert second.source_courses[0] is first.source_courses[0]
    assert second.course_sha256[0] == first.course_sha256[0] and second.course_sha256[1] != first.course_sha256[1]


def test_already_applied_same_tick_reset_cannot_consume_another_task_or_start_another_edit():
    coordinator, life, _, sizes = fixture()
    first = request(coordinator, life)
    apply(life, first)
    before = coordinator.preparer.state_dict()
    with pytest.raises(RuntimeError, match='already reset'):
        request(coordinator, life)
    assert sizes == [2] and coordinator.preparer.requested_rows == 2
    assert torch.equal(coordinator.preparer.generator.get_state(), before['pose_rng'])


def test_unapplied_boundary_reuses_its_plan_without_rng_advance_and_never_draws_a_replacement_request():
    coordinator, life, clock, sizes = fixture()
    first = request(coordinator, life)
    state = deepcopy(coordinator.preparer.state_dict())
    assert request(coordinator, life) is first
    assert sizes == [2] and torch.equal(coordinator.preparer.generator.get_state(), state['pose_rng'])
    clock += .01
    with pytest.raises(RuntimeError, match='has not been applied'):
        request(coordinator, life)
    assert sizes == [2]
    saved = coordinator.state_dict()
    assert saved['status'] == 'pending_not_resumable' and saved['pending_proposal']['requested_draw']['family'].tolist() == [0, 0]
    fresh, _, _, _ = fixture()
    with pytest.raises(ValueError, match='nonpending'):
        fresh.load_after_explicit_cold_setup(saved)


def test_geometric_preparation_failure_keeps_the_exact_draw_and_cannot_silently_resample():
    coordinator, life, _, sizes = fixture()

    def reject(*args):
        raise RuntimeError('explicit synthetic geometry resource rejection')

    coordinator.preparer.terrain.from_draw = reject
    with pytest.raises(ResetSourceNotReady) as first:
        request(coordinator, life)
    with pytest.raises(ResetSourceNotReady) as second:
        request(coordinator, life)
    assert first.value.proposal is second.value.proposal and sizes == [2]
    assert coordinator.preparer.requested_rows == 2 and coordinator.preparer.completed_batches == 0
    saved = coordinator.state_dict()
    assert saved['pending_proposal']['prepared_geometry'] is None
    assert 'resource rejection' in saved['pending_proposal']['preparation_error']


def test_interruption_after_the_draw_is_sealed_leaves_a_nonresumable_exact_pending_proposal():
    coordinator, life, _, sizes = fixture()

    def interrupt(*args):
        raise KeyboardInterrupt('explicit synthetic interruption')

    coordinator.preparer.terrain.from_draw = interrupt
    with pytest.raises(KeyboardInterrupt):
        request(coordinator, life)
    with pytest.raises(ResetSourceNotReady) as pending:
        request(coordinator, life)
    assert pending.value.proposal.preparation_error == 'source_preparation_in_progress'
    assert sizes == [2] and coordinator.preparer.requested_rows == 2
    assert coordinator.state_dict()['status'] == 'pending_not_resumable'


def test_complete_boundary_cold_setup_keeps_rng_and_sequence_but_does_not_replay_an_active_episode():
    first, life, clock, _ = fixture()
    plan = request(first, life)
    apply(life, plan)
    saved = deepcopy(first.state_dict())
    assert saved['status'] == 'ready_for_explicit_cold_setup' and saved['pending_proposal'] is None
    clock += .1
    expected = request(first, life)
    second, cold_life, _, sizes = fixture()
    second.load_after_explicit_cold_setup(saved)
    controlled_draws(second.preparer, sizes)
    actual = request(second, cold_life)
    assert second._sequence == first._sequence == 2
    assert actual.course_sha256 == expected.course_sha256
    assert torch.equal(actual.root_pose_xyzw, expected.root_pose_xyzw)
    assert torch.equal(actual.commands.trajectory.end, expected.commands.trajectory.end)
    assert cold_life.episode_id.tolist() == [-1, -1]
    assert second.preparer.requested_rows == first.preparer.requested_rows == 4


@pytest.mark.parametrize('fault', ['controller', 'trajectory'])
def test_changed_controller_or_issued_trajectory_is_not_adopted_as_the_previous_source_plan(fault):
    coordinator, life, _, sizes = fixture()
    value = request(coordinator, life)
    if fault == 'controller':
        life.plant.controller.action_bounds[0] += .1
    else:
        value.commands.trajectory.frequency_start_hz[0] += .1
    with pytest.raises(RuntimeError, match='changed'):
        request(coordinator, life)
    assert sizes == [2]

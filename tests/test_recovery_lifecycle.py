"""Exercise recovery ownership, causal handoff and separate physical evidence."""
from copy import deepcopy

import pytest
import torch

from transformer_rl.formal.lab_env.recovery_lifecycle import (
    RecoveryLifecycle, RecoveryLifecycleConfig,
)
from transformer_rl.formal.tasks.protocol import CompletedEstimate


def fixture():
    value = RecoveryLifecycle(2, config=RecoveryLifecycleConfig(stable_s=.03, timeout_s=.10))
    value.reset(torch.arange(2), torch.ones(2, dtype=torch.bool))
    return value


def estimate(tick):
    return CompletedEstimate(torch.zeros(2), torch.ones(2, 2), torch.zeros(2),
                             torch.zeros(2), torch.full((2,), tick * .01, dtype=torch.float64),
                             torch.ones(2, dtype=torch.bool))


def interval(value, tick, *, physical=True, owned=True, failed=False, packet='fresh'):
    ownership = torch.full((2,), owned, dtype=torch.bool)
    for _ in range(2):
        value.observe_physics(
            height_m=torch.full((2,), .32 if physical else .05),
            command_height_m=torch.full((2,), .32), tilt_rad=torch.zeros(2),
            angular_speed_rad_s=torch.zeros(2), velocity_z_m_s=torch.zeros(2),
            wheel_load_n=torch.full((2, 2), 10. if physical else 0.),
            base_load_n=torch.zeros(2), policy_owned=ownership)
    packet = estimate(tick - 1) if packet == 'fresh' else packet
    return value.finish_interval(
        torch.full((2,), tick * .01, dtype=torch.float64), packet,
        policy_owned=ownership, failed=torch.full((2,), failed, dtype=torch.bool))


def test_stable_recovery_hands_back_to_tracking_and_counts_one_physical_event():
    value = fixture()
    assert not interval(value, 1).handoff.any()
    assert not interval(value, 2).handoff.any()
    result = interval(value, 3)
    assert result.handoff.all() and not result.timeout.any()
    assert not value.active.any()
    interval(value, 4)
    assert value.metrics()['counts']['estimated_handoffs'] == 2
    assert value.metrics()['counts']['verified_physical_successes'] == 2


def test_actor_visible_handoff_cannot_depend_on_privileged_height_or_contact_truth():
    actual, altered = fixture(), fixture()
    for tick in range(1, 4):
        good, bad = interval(actual, tick), interval(altered, tick, physical=False)
        assert torch.equal(good.handoff, bad.handoff)
        assert torch.equal(good.timeout, bad.timeout)
    assert actual.metrics()['counts']['verified_physical_successes'] == 2
    assert altered.metrics()['counts']['verified_physical_successes'] == 0
    assert altered.metrics()['counts']['handoffs_without_verified_success'] == 2


def test_script_hold_neither_consumes_policy_deadline_nor_earns_success():
    value = fixture()
    for tick in range(1, 30):
        result = interval(value, tick, owned=False)
        assert not result.timeout.any() and not result.handoff.any()
    assert value.policy_elapsed_s.eq(0).all()
    assert value.metrics()['counts']['verified_physical_successes'] == 0
    assert interval(value, 30).handoff.eq(False).all()


def test_missing_estimates_timeout_without_inventing_success():
    value = fixture()
    for tick in range(1, 10):
        assert not interval(value, tick, physical=False, packet=None).timeout.any()
    result = interval(value, 10, physical=False, packet=None)
    assert result.timeout.all() and not result.handoff.any()
    assert value.metrics()['counts']['timeouts'] == 2
    assert value.metrics()['counts']['verified_physical_successes'] == 0
    assert not interval(value, 11, packet=None).timeout.any()


def test_latched_failure_wins_over_same_interval_success_and_handoff():
    value = fixture()
    interval(value, 1)
    interval(value, 2)
    result = interval(value, 3, failed=True)
    assert not result.handoff.any() and not result.timeout.any()
    assert value.metrics()['counts']['failed_before_handoff'] == 2
    assert value.metrics()['counts']['verified_physical_successes'] == 0


def test_partial_reset_preserves_continuing_row_and_never_consumes_rng():
    value = fixture()
    interval(value, 1)
    before = torch.get_rng_state().clone()
    elapsed, stable = value.policy_elapsed_s[1].clone(), value.estimated_stable_s[1].clone()
    value.reset(torch.tensor([0]), torch.tensor([False]))
    assert not value.active[0] and value.active[1]
    assert value.policy_elapsed_s[1] == elapsed and value.estimated_stable_s[1] == stable
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('timestamp', (float('nan'), 100.))
def test_bad_estimate_clock_does_not_poison_subsequent_fresh_packets(timestamp):
    value = fixture()
    bad = estimate(0)
    bad.available_at_s.fill_(timestamp)
    assert not interval(value, 1, packet=bad).handoff.any()
    assert not interval(value, 2).handoff.any()
    assert not interval(value, 3).handoff.any()
    assert interval(value, 4).handoff.all()


def test_reused_packet_cannot_manufacture_a_stable_handoff():
    value = fixture()
    packet = estimate(0)
    for tick in range(1, 10):
        assert not interval(value, tick, packet=packet).handoff.any()


def test_cold_restore_preserves_counters_and_discards_old_stability():
    value = fixture()
    interval(value, 1)
    interval(value, 2)
    state = value.state_dict()
    restored = fixture()
    restored.restore_after_physical_reset(state)
    assert restored.metrics()['counts'] == value.metrics()['counts']
    assert restored.policy_elapsed_s.eq(0).all()
    assert restored.estimated_stable_s.eq(0).all()
    assert not interval(restored, 3).handoff.any()
    bad = deepcopy(state)
    bad['counts'][0] = -1
    with pytest.raises(ValueError):
        restored.restore_after_physical_reset(bad)


def test_course_handoff_changes_next_mode_and_occupancy_without_new_episode_or_script():
    from types import SimpleNamespace
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
    from transformer_rl.formal.tasks.curriculum import TaskFamily

    value = fixture()
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.recovery = value
    course.tasks = SimpleNamespace(base_mode=torch.full((2,), int(CommandMode.RECOVERY)),
                                   family=torch.full((2,), int(TaskFamily.RECOVERY)))
    course.env = SimpleNamespace(
        _clock=lambda: torch.full((2,), .03, dtype=torch.float64),
        _completed_estimate=estimate(2), _interval_policy_owned=torch.ones(2, dtype=torch.bool),
        _terminated=torch.zeros(2, dtype=torch.bool))
    interval(value, 1)
    interval(value, 2)
    assert not course.finish_recovery_interval().any()
    assert course.tasks.base_mode.eq(int(CommandMode.NORMAL)).all()
    assert course.tasks.family.eq(int(TaskFamily.HEIGHT)).all()


def test_native_step_returns_recovery_timeout_and_preserves_pre_reset_endpoint():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import time

    path = Path(__file__).resolve().parents[1] / 'src/transformer_rl/formal/lab_env/env.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    step = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == 'step')
    namespace = {'torch': torch, 'time': time}
    exec(compile(ast.Module(body=[step], type_ignores=[]), str(path), 'exec'), namespace)
    command = torch.zeros(3)
    endpoints = []

    def observe():
        env._terminated[1] = True

    def reset(ids):
        command[ids] = 99.

    course = SimpleNamespace(
        finish_recovery_interval=lambda: torch.tensor([True, False, False]),
        router=SimpleNamespace(observe_end=lambda *_: torch.zeros(3, dtype=torch.bool)))
    env = SimpleNamespace(
        _device=torch.device('cpu'), cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1)),
        sim=SimpleNamespace(is_rendering=False, step=lambda **kwargs: None),
        scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None),
        _pre_physics_step=lambda _: None, _apply_action=lambda: None, _observe_substep=observe,
        _sim_step_counter=0, _physical_tick=0, physics_dt=.005,
        episode_length_buf=torch.zeros(3, dtype=torch.long), max_episode_length=3000, common_step_counter=0,
        _terrain_course=course, _formal_reward=None, _terminated=torch.zeros(3, dtype=torch.bool),
        _reward_policy_step=lambda: torch.full((3,), 2.), _last_reward=torch.zeros(3),
        _last_truncated=torch.zeros(3, dtype=torch.bool), _last_terminated=torch.zeros(3, dtype=torch.bool),
        reset_terminated=torch.zeros(3, dtype=torch.bool), reset_time_outs=torch.zeros(3, dtype=torch.bool),
        _previous_action=torch.zeros(3, 6), _issued_action=torch.zeros(3, 6),
        _capture_terminal=lambda: endpoints.append(command.clone()),
        _update_skill=lambda: command.add_(.03), _reset_idx=reset,
        _root_link_position=lambda: torch.zeros(3, 3),
        _get_observations=lambda: command.clone(), extras={}, _reward_terms={})
    observation, _, terminated, truncated, _ = namespace['step'](env, torch.zeros(3, 6))
    assert terminated.tolist() == [False, True, False]
    assert truncated.tolist() == [True, False, False]
    assert env.extras['recovery_timeouts'] == 1
    torch.testing.assert_close(endpoints[0], torch.full((3,), .03))
    assert observation[:2].eq(99.).all() and observation[2] == endpoints[0][2]

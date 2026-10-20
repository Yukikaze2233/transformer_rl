"""Causal command/request inputs and reward-only physical success."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.lab_env.course_tasks import LabCourseTasks, completed_policy_estimate
from transformer_rl.formal.lab_env.formal_curriculum import FormalCurriculumCapabilities, FormalTaskCurriculum, LabTaskDrawSampler
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestResponse
from transformer_rl.formal.tasks.trajectory import DriveEnvelope


def fixture(kind=1):
    curriculum = FormalTaskCurriculum(2, capabilities=FormalCurriculumCapabilities(True, True, True))
    sampler = LabTaskDrawSampler(curriculum)
    sample = sampler(torch.arange(2))
    draw = replace(sample.draw, trajectory_kind=torch.full((2,), kind), nominal=torch.zeros(2, dtype=torch.bool))
    sample = replace(sample, draw=draw, command=torch.tensor([[1., 0, 0, .35], [-1., 0, 0, .30]]))
    tasks = LabCourseTasks(2, DriveEnvelope(.06, .4))
    tasks.reset(torch.arange(2), sample, torch.zeros(2, dtype=torch.float64))
    return tasks, sample


@pytest.mark.parametrize('kind', range(7))
def test_command_families_execute_and_respect_rates(kind):
    tasks, sample = fixture(kind)
    previous = tasks.stream.current.clone()
    observed = []
    for tick in range(350):
        admission, _, _, _ = tasks.step(torch.full((2,), (tick + 1) * .01, dtype=torch.float64), None, torch.zeros(2))
        assert (admission.admitted[:, 0] - previous[:, 0]).abs().max() <= .020001
        assert (admission.admitted[:, 3] - previous[:, 3]).abs().max() <= .002001
        observed.append(admission.admitted.clone())
        previous = admission.admitted
    assert torch.isfinite(torch.stack(observed)).all()
    if kind:
        assert not torch.equal(observed[0], observed[-1]) or kind in (2, 3, 5)


def test_completed_estimate_does_not_use_privileged_values():
    frame = torch.zeros(2, 35)
    frame[:, 9] = -1
    raw = torch.tensor([[1., 2, 3, 0, 0, 0, 0, 0], [1., 2, 3, 0, 0, 0, 2, -2]])
    result = completed_policy_estimate(frame, raw, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(result.velocity_z_m_s, torch.full((2,), 3.))
    assert result.valid.all() and result.tilt_rad.eq(0).all()
    torch.testing.assert_close(result.wheel_contact_probability, raw[:, 6:8].sigmoid())


def test_no_success_for_airborne_timer_without_ascent():
    tasks = supported_jump()
    for _ in range(10):
        tasks.observe_physics(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2), torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    for _ in range(500):
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    assert tasks.success_id.eq(-1).all()


def test_physical_success_is_ready_in_current_interval_and_paid_once():
    tasks = supported_jump()
    for _ in range(3):
        tasks.observe_physics(torch.zeros(2, 2), torch.ones(2), torch.zeros(2), torch.full((2,), .42), angular_speed_rad_s=torch.zeros(2))
    for _ in range(401):
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    assert tasks.success_id.eq(1).all()
    assert tasks.paid_id.eq(1).all()
    assert tasks.physical_outcome_counts.tolist() == [2, 0, 0]
    tasks.success_id.fill_(-1)
    tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    assert tasks.success_id.eq(-1).all()
    assert tasks.physical_outcome_counts.tolist() == [2, 0, 0]


def test_request_end_and_unverified_physical_counts_survive_cold_restore():
    from transformer_rl.formal.tasks.protocol import RequestEnd

    tasks = supported_jump()
    now = torch.full((2,), 6.01, dtype=torch.float64)
    tasks.step(now, _touch_estimate(now - .01), torch.zeros(2))
    assert tasks.request_end_counts[int(RequestEnd.TIMEOUT)] == 2
    assert tasks.physical_outcome_counts.tolist() == [0, 0, 2]
    saved = tasks.state_dict()
    restored, _ = fixture()
    restored.restore_after_physical_reset(saved, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(restored.request_end_counts, tasks.request_end_counts)
    torch.testing.assert_close(restored.physical_outcome_counts, tasks.physical_outcome_counts)


def test_request_decision_rejects_same_tick_estimate_and_keeps_ids_across_reset():
    tasks, sample = fixture()
    tasks.family[:] = 4
    tasks.rise[:] = .1
    tasks.next_request_s[:] = 0
    time = torch.full((2,), .01, dtype=torch.float64)
    estimate = CompletedEstimate(torch.zeros(2), torch.ones(2, 2), torch.zeros(2), torch.zeros(2),
                                 time.clone(), torch.ones(2, dtype=torch.bool))
    tasks.step(time, estimate, torch.zeros(2))
    assert tasks.protocol.max_seen_id.eq(0).all()
    assert tasks.protocol.active_id.eq(-1).all()
    saved = tasks.state_dict()
    restored, _ = fixture()
    restored.restore_after_physical_reset(saved, torch.zeros(2, dtype=torch.float64))
    restored.reset(torch.arange(2), sample, torch.zeros(2, dtype=torch.float64))
    assert restored.protocol.max_seen_id.eq(0).all()
    assert restored.next_id.eq(1).all()


def _touch_estimate(t, valid=True):
    return CompletedEstimate(torch.zeros(2), torch.ones(2, 2), torch.zeros(2), torch.zeros(2),
                             t.clone(), torch.full((2,), valid, dtype=torch.bool))


def test_external_jump_request_is_not_gated_by_training_task_family():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    tasks, _ = fixture()
    tasks.family.zero_()
    tasks.base_mode.zero_()
    now = torch.zeros(2, dtype=torch.float64)
    command = RequestCommand(torch.tensor([17, 23]), torch.full((2,), .1))
    _, mode, rise, _ = tasks.step(now, _touch_estimate(now - .01), torch.zeros(2), command=command)
    assert tasks.protocol.active_id.tolist() == [17, 23]
    assert mode.eq(4).all()
    torch.testing.assert_close(rise, torch.full((2,), .1))


def test_manual_and_tof_requests_share_busy_cancel_and_timeout_protocol():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    tasks, _ = fixture()
    tasks.family[:] = 4
    first = torch.zeros(2, dtype=torch.float64)
    second = torch.full((2,), .01, dtype=torch.float64)
    manual = RequestCommand(torch.zeros(2, dtype=torch.long), torch.full((2,), .10),
                            source='manual')
    tasks.step(first, _touch_estimate(torch.full((2,), -.01, dtype=torch.float64)), torch.zeros(2), command=manual)
    tasks.step(second, _touch_estimate(first), torch.zeros(2), command=manual)
    assert tasks.protocol.active_id.eq(0).all()

    busy = RequestCommand(torch.ones(2, dtype=torch.long), torch.full((2,), .10), source='tof')
    tasks.step(torch.full((2,), .02, dtype=torch.float64), _touch_estimate(second), torch.zeros(2),
               command=busy)
    assert tasks.external_request_counters[RequestResponse.BUSY].item() == 2

    cancel = RequestCommand(torch.full((2,), -1, dtype=torch.long), torch.zeros(2),
                            cancel_id=torch.zeros(2, dtype=torch.long), source='manual')
    tasks.step(torch.full((2,), .03, dtype=torch.float64), _touch_estimate(torch.full((2,), .02, dtype=torch.float64)),
               torch.zeros(2), command=cancel)
    assert tasks.protocol.active_id.eq(-1).all()
    assert tasks.external_request_counters[RequestResponse.CANCELLED].item() == 0

    tof = RequestCommand(torch.full((2,), 2, dtype=torch.long), torch.full((2,), .10), source='tof')
    tasks.step(torch.full((2,), .04, dtype=torch.float64), _touch_estimate(torch.full((2,), .02, dtype=torch.float64)),
               torch.zeros(2), command=tof)
    tasks.step(torch.full((2,), 7.0, dtype=torch.float64), _touch_estimate(torch.full((2,), 6.9, dtype=torch.float64)),
               torch.zeros(2), command=RequestCommand(torch.full((2,), -1, dtype=torch.long), torch.zeros(2), source='tof'))
    assert tasks.external_request_counters[RequestResponse.ACCEPTED].item() == 4


def test_external_request_height_drives_reference_and_success_not_sampled_rise():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    tasks, _ = fixture()
    tasks.family[:] = 4
    tasks.rise[:] = .05
    external = RequestCommand(torch.zeros(2, dtype=torch.long), torch.full((2,), .20), source='tof')
    tasks.step(torch.zeros(2, dtype=torch.float64),
               _touch_estimate(torch.full((2,), -.01, dtype=torch.float64)), torch.zeros(2), command=external)
    tasks.step(torch.full((2,), .01, dtype=torch.float64),
               _touch_estimate(torch.zeros(2, dtype=torch.float64)), torch.zeros(2), command=external)
    assert tasks.protocol.active_id.eq(0).all()
    assert tasks.active_rise.eq(.20).all()
    reference = tasks.jump_reference()
    torch.testing.assert_close(reference.target_apex_world_z_m - tasks.ground_anchor - tasks.latched_height,
                               torch.full((2,), .20))
    # A 0.10m physical rise must not satisfy a 0.20m external request.
    tasks.observe_physics(torch.zeros(2, 2), torch.ones(2), torch.zeros(2), torch.full((2,), .42), angular_speed_rad_s=torch.zeros(2))
    tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .42), angular_speed_rad_s=torch.zeros(2))
    assert tasks.success_id.eq(-1).all()


def test_large_external_id_does_not_make_the_next_auto_request_replay():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    tasks, _ = fixture()
    tasks.family[:] = 4
    tasks.rise[:] = .10
    external = RequestCommand(torch.full((2,), 100, dtype=torch.long), torch.full((2,), .10), source='tof')
    tasks.step(torch.zeros(2, dtype=torch.float64),
               _touch_estimate(torch.full((2,), -.01, dtype=torch.float64)), torch.zeros(2), command=external)
    tasks.step(torch.full((2,), .01, dtype=torch.float64),
               _touch_estimate(torch.zeros(2, dtype=torch.float64)), torch.zeros(2), command=external)
    assert tasks.protocol.active_id.eq(100).all()
    cancel = RequestCommand(torch.full((2,), -1, dtype=torch.long), torch.zeros(2),
                            cancel_id=torch.full((2,), 100, dtype=torch.long), source='manual')
    tasks.step(torch.full((2,), .02, dtype=torch.float64),
               _touch_estimate(torch.full((2,), .01, dtype=torch.float64)), torch.zeros(2), command=cancel)
    assert tasks.protocol.active_id.eq(-1).all()
    assert tasks.next_id.eq(101).all()
    tasks.next_request_s[:] = 0
    tasks.step(torch.full((2,), .03, dtype=torch.float64),
               _touch_estimate(torch.full((2,), .02, dtype=torch.float64)), torch.zeros(2))
    assert tasks.protocol.max_seen_id.eq(101).all()
    assert tasks.protocol.active_id.eq(101).all()


def test_external_id_state_survives_reset_and_cold_restore_without_auto_replay():
    from transformer_rl.formal.lab_env.course_tasks import RequestCommand

    tasks, sample = fixture()
    tasks.family[:] = 4
    tasks.rise[:] = .10
    external = RequestCommand(torch.full((2,), 100, dtype=torch.long), torch.full((2,), .10), source='tof')
    tasks.step(torch.zeros(2, dtype=torch.float64),
               _touch_estimate(torch.full((2,), -.01, dtype=torch.float64)), torch.zeros(2), command=external)
    tasks.step(torch.full((2,), .01, dtype=torch.float64),
               _touch_estimate(torch.zeros(2, dtype=torch.float64)), torch.zeros(2), command=external)
    saved = tasks.state_dict()
    restored, _ = fixture()
    restored.restore_after_physical_reset(saved, torch.zeros(2, dtype=torch.float64))
    restored.reset(torch.arange(2), sample, torch.zeros(2, dtype=torch.float64))
    assert restored.protocol.max_seen_id.eq(100).all()
    assert restored.next_id.ge(101).all()
    restored.next_request_s[:] = 0
    restored.family[:] = 4
    restored.rise[:] = .10
    restored.step(torch.zeros(2, dtype=torch.float64),
                  _touch_estimate(torch.full((2,), -.01, dtype=torch.float64)), torch.zeros(2))
    assert restored.protocol.active_id.ge(101).all()


def supported_jump(*, actual_start_support=True):
    tasks, _ = fixture()
    tasks.family.fill_(4)
    tasks.next_request_s.zero_()
    tasks.next_id.fill_(1)
    tasks.rise.fill_(.1)
    tasks.trajectory.start[:, 3] = .32
    tasks.trajectory.end[:, 3] = .32
    tasks.stream.current[:, 3] = .32
    if actual_start_support:
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2),
                              torch.full((2,), .32), angular_speed_rad_s=torch.zeros(2))
    now = torch.zeros(2, dtype=torch.float64)
    tasks.step(now, _touch_estimate(now - .01), torch.zeros(2))
    assert tasks.active_id.eq(1).all()
    return tasks


@pytest.mark.parametrize('problem', ('bad_start', 'overshoot', 'spinning'))
def test_jump_evidence_rejects_unsupported_start_overshoot_or_spinning_landing(problem):
    tasks = supported_jump(actual_start_support=problem != 'bad_start')
    apex = .60 if problem == 'overshoot' else .42
    for _ in range(3):
        tasks.observe_physics(torch.zeros(2, 2), torch.ones(2), torch.zeros(2),
                              torch.full((2,), apex), angular_speed_rad_s=torch.zeros(2))
    for _ in range(401):
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2),
                              torch.full((2,), .32), angular_speed_rad_s=torch.full((2,),
                                  1.0 if problem == 'spinning' else 0.0))
    assert tasks.success_id.eq(-1).all()
    assert tasks.paid_id.eq(-1).all()


def test_early_estimated_completion_does_not_retire_unfinished_physical_evidence():
    from transformer_rl.formal.tasks.protocol import RequestEnd
    tasks = supported_jump()
    tasks.next_request_s.fill_(999)
    estimate_ended = None
    physical_successes = []
    for tick in range(1, 251):
        actual_air = tick <= 30
        for _ in range(2):
            tasks.observe_physics(
                torch.full((2, 2), 0. if actual_air else 100.),
                torch.full((2,), 1. if actual_air else 0.), torch.zeros(2),
                torch.full((2,), .42 if actual_air else .32), angular_speed_rad_s=torch.zeros(2))
        if tasks.success_id.ge(0).any():
            physical_successes.append(tick * .01)
        now = torch.full((2,), tick * .01, dtype=torch.float64)
        estimate = _touch_estimate(now - .01)
        if tick <= 2:
            estimate = replace(estimate, wheel_contact_probability=torch.zeros(2, 2))
        tasks.step(now, estimate, torch.zeros(2))
        if estimate_ended is None and tasks.protocol.active_id.eq(-1).all():
            estimate_ended = tick * .01
    assert estimate_ended is not None
    assert len(physical_successes) == 1 and physical_successes[0] > estimate_ended
    assert tasks.paid_id.eq(1).all() and tasks.physical_id.eq(-1).all()

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
    tasks, _ = fixture()
    tasks.active_id[:] = 1
    tasks.rise[:] = .1
    for _ in range(10):
        tasks.observe_physics(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2), torch.full((2,), .32))
    for _ in range(500):
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32))
    assert tasks.success_id.eq(-1).all()


def test_physical_success_is_ready_in_current_interval_and_paid_once():
    tasks, _ = fixture()
    tasks.active_id[:] = 1
    tasks.rise[:] = .1
    for _ in range(3):
        tasks.observe_physics(torch.zeros(2, 2), torch.ones(2), torch.zeros(2), torch.full((2,), .42))
    for _ in range(401):
        tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32))
    assert tasks.success_id.eq(1).all()
    assert tasks.paid_id.eq(1).all()
    tasks.success_id.fill_(-1)
    tasks.observe_physics(torch.full((2, 2), 100.), torch.zeros(2), torch.zeros(2), torch.full((2,), .32))
    assert tasks.success_id.eq(-1).all()


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

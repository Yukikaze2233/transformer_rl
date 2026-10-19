"""Synthetic request/physics separation checks. No SDK, qualification or PPO."""
from dataclasses import fields, replace
import math

import pytest
import torch

from transformer_rl.formal.tasks.skills import (
    PhysicalJumpEvidence, SkillEvidenceConfig, SkillMeasurement, SkillOutcome, SkillPhase,
)
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestEnd, RequestProtocol, RequestProtocolConfig, RequestResponse


class SkillHarness:
    def __init__(self, period=.005, stable=.03, n=2, retired=None):
        self.n, self.period = n, period
        self.protocol = RequestProtocol(n, config=RequestProtocolConfig(stable_landing_s=stable))
        if retired is not None:
            self.protocol.max_seen_id.copy_(torch.tensor(retired))
        self.tracker = PhysicalJumpEvidence(self.protocol, SkillEvidenceConfig(physics_dt_s=period, stable_landing_s=stable))
        self.tracker.reset(torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.float64),
                            torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), physical_tick=0)
        self.last_evidence = self.last_condition = self.last_receipt = None

    def tensor(self, value, width=None, dtype=torch.float32):
        shape = (self.n,) if width is None else (self.n, width)
        return torch.broadcast_to(torch.as_tensor(value, dtype=dtype), shape).clone()

    def measurement(self, **overrides):
        tick = self.tracker._tick + 1
        sample = SkillMeasurement(tick, self.tensor(tick * self.period, dtype=torch.float64),
                                   self.tracker._episode.clone(), self.tracker._control.clone(),
                                   self.tensor(.3), self.tensor(0.), self.tensor(0.), self.tensor(100., 2),
                                   self.tensor(0., 2), self.tensor(0.), self.tensor(0.), self.tensor(True, dtype=torch.bool),
                                   self.tensor(True, dtype=torch.bool), self.tensor(False, dtype=torch.bool))
        return replace(sample, **overrides)

    def interval(self, *, request_id=-1, rise=.1, cancel_id=-1, estimated_air=False, safety_abort=False,
                 edit=None, **physical):
        for _ in range(self.tracker.substeps):
            overrides = {}
            for name, value in physical.items():
                reference = getattr(self.measurement(), name)
                overrides[name] = torch.broadcast_to(torch.as_tensor(value, dtype=reference.dtype), reference.shape).clone()
            measurement = self.measurement(**overrides)
            if edit is not None:
                measurement = edit(measurement)
            self.tracker.observe(measurement)
        evidence = self.tracker.drain_policy_evidence()
        now = evidence.timestamp_s
        estimate = CompletedEstimate(self.tensor(0.), self.tensor(0. if estimated_air else 1., 2),
                                      self.tensor(0.), self.tensor(0.), now - .01, self.tensor(True, dtype=torch.bool))
        condition = self.protocol.step(now_s=now, estimate=estimate, request_id=self.tensor(request_id, dtype=torch.long),
                                       requested_height_m=self.tensor(rise), cancel_id=self.tensor(cancel_id, dtype=torch.long),
                                       allow_new=self.tensor(True, dtype=torch.bool),
                                       safety_abort=self.tensor(safety_abort, dtype=torch.bool))
        receipt = self.tracker.accept_protocol_condition(condition, self.tensor(.3))
        self.last_evidence, self.last_condition, self.last_receipt = evidence, condition, receipt
        return evidence

    def accept(self, rise=.1):
        self.interval(request_id=1, rise=rise)
        assert self.last_condition.response.eq(int(RequestResponse.ACCEPTED)).all()

    def airborne(self, peak=.4, **changes):
        return self.interval(root_world_z_m=peak, wheel_normal_load_n=0., wheel_clearance_m=.02, **changes)

    def settle(self, **changes):
        for _ in range(math.ceil((self.tracker.config.stable_landing_s + .05) / .01)):
            value = self.interval(**changes)
            if value.finished_request_id.ge(0).any():
                return value
        raise AssertionError('synthetic evidence did not finish within its declared stable interval')


def assert_protocol_equal(actual, saved):
    assert actual['configuration'] == saved['configuration']
    for key, value in saved['tensors'].items():
        assert torch.equal(actual['tensors'][key], value), key


@pytest.mark.parametrize('period', [.005, .001])
def test_real_default_two_second_landing_after_ballistic_trace_and_across_rollout_segments(period):
    h = SkillHarness(period=period, stable=2.)
    rng = torch.get_rng_state().clone()
    h.accept()
    launch = h.tracker._time.clone()
    velocity = math.sqrt(2 * 9.81 * .1)
    flight_duration = 2 * velocity / 9.81
    found = None
    for _ in range(260):
        def ballistic(value):
            t = value.timestamp_s - launch
            air = t < flight_duration
            z = torch.where(air, .3 + velocity * t - .5 * 9.81 * t.square(), .3).float()
            vz = torch.where(air, velocity - 9.81 * t, 0).float()
            # Explicit synthetic collider clearance: this is not link-Z/radius inference.
            gap = torch.where(air, (z - .3).clamp_min(.006), 0)
            loads = torch.where(air, 0., 100.)[:, None].expand(-1, 2).clone()
            return replace(value, root_world_z_m=z, vertical_velocity_world_m_s=vz,
                            wheel_clearance_m=gap[:, None].expand(-1, 2).clone(), wheel_normal_load_n=loads)

        result = h.interval(edit=ballistic)
        if result.finished_request_id.ge(0).all():
            found = result
            break
    assert found is not None
    assert found.outcome.eq(int(SkillOutcome.PHYSICAL_SUCCESS)).all()
    assert found.physical_success_request_id.eq(1).all()
    assert found.stable_landing_s.ge(2. - 1e-9).all()
    assert found.timestamp_s.gt(4 * .48).all()  # Skill state survives independent rollout slicing.
    torch.testing.assert_close(found.apex_rise_m, torch.full((2,), .1), atol=4e-5, rtol=0)
    assert found.active_id.eq(-1).all() and h.tracker.active_id.eq(-1).all()
    assert found.phase.eq(int(SkillPhase.SETTLING)).all()
    # Estimated contacts stayed grounded: physical truth never changed the FSM.
    assert h.protocol.active_id.eq(1).all() and not h.protocol.flight_seen.any()
    assert torch.equal(torch.get_rng_state(), rng)
    assert h.interval().physical_success_request_id.eq(-1).all()


def test_estimated_completion_without_real_flight_cannot_create_success_and_new_request_is_explicitly_superseded():
    h = SkillHarness()
    h.accept()
    h.interval(estimated_air=True)
    h.interval(estimated_air=True)
    for _ in range(10):
        value = h.interval()
        if h.last_condition.end_reason.eq(int(RequestEnd.ESTIMATED_COMPLETE)).all():
            break
    assert h.protocol.active_id.eq(-1).all()
    assert value.physical_success_request_id.eq(-1).all() and not value.flight_seen.any()
    assert h.tracker.active_id.eq(1).all()
    h.interval(request_id=2)
    assert h.last_receipt.accepted_id.eq(2).all()
    assert h.last_receipt.finished_id.eq(1).all()
    assert h.last_receipt.outcome.eq(int(SkillOutcome.SUPERSEDED)).all()


@pytest.mark.parametrize('peak,rise,expected', [(.30, .1, SkillOutcome.HEIGHT_MISSED),
                                              (.45, .1, SkillOutcome.HEIGHT_MISSED),
                                              (.315, .03, SkillOutcome.PHYSICAL_SUCCESS),
                                              (.314, .03, SkillOutcome.HEIGHT_MISSED)])
def test_physical_apex_error_is_two_sided_and_small_jump_tolerance_does_not_accept_no_jump(peak, rise, expected):
    h = SkillHarness()
    h.accept(rise=rise)
    h.airborne(peak=peak)
    result = h.settle()
    assert result.outcome.eq(int(expected)).all()
    torch.testing.assert_close(result.height_tolerance_m, torch.full((2,), min(.03, .5 * rise)))


@pytest.mark.parametrize('missing', ['force_only', 'one_wheel_only'])
def test_airborne_requires_both_actual_clearances_and_both_unloaded_wheels(missing):
    h = SkillHarness()
    h.accept()
    changes = {'wheel_normal_load_n': 0., 'wheel_clearance_m': 0.}
    if missing == 'one_wheel_only':
        changes = {'wheel_normal_load_n': [0., 100.], 'wheel_clearance_m': [.02, 0.]}
    for _ in range(4):
        value = h.interval(root_world_z_m=.4, **changes)
    assert not value.flight_seen.any() and value.physical_success_request_id.eq(-1).all()


def test_standing_extension_is_not_counted_as_flight_apex():
    h = SkillHarness()
    h.accept()
    h.interval(root_world_z_m=.55)
    h.airborne(peak=.4)
    result = h.settle()
    assert result.outcome.eq(int(SkillOutcome.PHYSICAL_SUCCESS)).all()
    torch.testing.assert_close(result.apex_rise_m, torch.full((2,), .1))


@pytest.mark.parametrize('field,expected', [('policy_owned', SkillOutcome.SCRIPT_ASSISTED),
                                          ('healthy', SkillOutcome.UNHEALTHY)])
def test_entire_skill_ownership_and_health_are_sticky_even_when_the_last_sample_is_good(field, expected):
    h = SkillHarness()
    h.accept()
    h.airborne(**{field: [False, True]})
    result = h.settle()
    assert result.outcome.tolist() == [int(expected), int(SkillOutcome.PHYSICAL_SUCCESS)]
    assert result.physical_success_request_id.tolist() == [-1, 1]


def test_bad_physical_support_start_is_not_hidden_by_estimated_readiness():
    h = SkillHarness()
    h.interval(request_id=1, root_world_z_m=.35)
    assert h.last_condition.response.eq(int(RequestResponse.ACCEPTED)).all()
    h.airborne()
    assert h.settle().outcome.eq(int(SkillOutcome.BAD_START)).all()


def test_unstable_landing_restarts_the_continuous_timer_instead_of_accumulating_good_fragments():
    h = SkillHarness(stable=.05)
    h.accept()
    h.airborne()
    for _ in range(4):
        h.interval()
    interrupted = h.interval(tilt_rad=.4)
    assert interrupted.stable_landing_s.eq(0).all()
    assert interrupted.physical_success_request_id.eq(-1).all()
    after = h.interval()
    assert after.stable_landing_s.lt(.02).all()
    assert h.settle().outcome.eq(int(SkillOutcome.PHYSICAL_SUCCESS)).all()


def test_cancellation_on_ground_and_in_air_never_pays_success():
    ground = SkillHarness()
    ground.accept()
    ground.interval(cancel_id=1)
    assert ground.last_receipt.outcome.eq(int(SkillOutcome.CANCELLED)).all()
    assert ground.tracker.active_id.eq(-1).all()
    air = SkillHarness()
    air.accept()
    air.airborne(estimated_air=True)
    air.airborne(estimated_air=True)
    air.airborne(estimated_air=True, cancel_id=1)
    assert air.tracker.active_id.eq(1).all()
    assert air.settle().outcome.eq(int(SkillOutcome.CANCELLED)).all()


def test_physical_failure_before_policy_endpoint_is_retained_and_protocol_abort_is_a_distinct_receipt():
    h = SkillHarness()
    h.accept()
    result = h.airborne(true_failure=[True, False])
    assert result.outcome.tolist() == [int(SkillOutcome.TRUE_FAILURE), int(SkillOutcome.NONE)]
    assert result.finished_request_id.tolist() == [1, -1]
    assert result.finished_at_s[0] < result.timestamp_s[0]
    assert result.physical_success_request_id.eq(-1).all()
    h.interval(safety_abort=True)
    assert h.last_receipt.outcome.tolist() == [int(SkillOutcome.NONE), int(SkillOutcome.SAFETY_ABORT)]


def test_no_flight_request_times_out_without_inventing_success_or_replaying_id():
    h = SkillHarness()
    h.accept()
    for _ in range(605):
        value = h.interval()
        if value.finished_request_id.ge(0).any():
            break
    assert value.outcome.eq(int(SkillOutcome.TIMEOUT)).all()
    h.interval(request_id=1)
    assert h.last_condition.response.eq(int(RequestResponse.REPLAY)).all()
    assert h.tracker.active_id.eq(-1).all()


def test_invalid_or_missing_physics_sample_fails_before_mutation_and_drain_cannot_repeat():
    h = SkillHarness()
    h.accept()
    next_sample = h.measurement()
    before_protocol = h.protocol.state_dict()
    for wrong in (replace(next_sample, physical_tick=next_sample.physical_tick + 1),
                  replace(next_sample, timestamp_s=next_sample.timestamp_s + .01),
                  replace(next_sample, episode_id=torch.ones(2, dtype=torch.long)),
                  replace(next_sample, wheel_normal_load_n=torch.full((2, 2), -1.)),
                  replace(next_sample, root_world_z_m=torch.full((2,), float('nan')))):
        before_tick = h.tracker._tick
        with pytest.raises(ValueError):
            h.tracker.observe(wrong)
        assert h.tracker._tick == before_tick
    with pytest.raises(RuntimeError):
        h.tracker.drain_policy_evidence()
    assert_protocol_equal(h.protocol.state_dict(), before_protocol)
    h.tracker.observe(next_sample)
    with pytest.raises(RuntimeError):
        h.tracker.drain_policy_evidence()
    h.tracker.observe(h.measurement())
    h.tracker.drain_policy_evidence()
    with pytest.raises(RuntimeError):
        h.tracker.drain_policy_evidence()


def test_selected_reset_preserves_other_attempt_and_cold_consumed_ids_never_start_again():
    h = SkillHarness()
    h.accept()
    h.airborne()
    mask = torch.tensor([True, False])
    h.protocol.reset(mask, h.tracker._time.clone())
    receipt = h.tracker.reset(mask, h.tracker._time.clone(), torch.tensor([1, 0]), torch.tensor([1, 0]),
                              physical_tick=h.tracker._tick)
    assert receipt.finished_id.tolist() == [1, -1]
    assert receipt.outcome.tolist() == [int(SkillOutcome.RESET), int(SkillOutcome.NONE)]
    assert h.tracker.active_id.tolist() == [-1, 1]
    result = h.settle()
    assert result.physical_success_request_id.tolist() == [-1, 1]
    cold = SkillHarness(retired=[1, 1])
    cold.interval(request_id=1)
    assert cold.last_condition.response.eq(int(RequestResponse.REPLAY)).all()
    assert cold.tracker.active_id.eq(-1).all()


def test_mismatched_protocol_receipt_and_duplicate_acceptance_are_rejected_without_changing_real_fsm():
    h = SkillHarness()
    h.accept()
    before = h.protocol.state_dict()
    for wrong in (h.last_condition, replace(h.last_condition, jump_mode=torch.zeros(2, dtype=torch.bool)),
                  replace(h.last_condition, clock_s=torch.ones(2))):
        with pytest.raises(ValueError):
            h.tracker.accept_protocol_condition(wrong, h.tensor(.3))
    assert_protocol_equal(h.protocol.state_dict(), before)

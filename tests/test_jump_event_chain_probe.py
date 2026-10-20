"""Isolated physical-event chain; no PPO, policy credit, or capability claim."""
import torch
import pytest

from test_skill_evidence import SkillHarness
from test_skill_reward import reward_window


def test_accept_flight_apex_land_settle_and_single_payment_across_rollout_cuts():
    harness = SkillHarness(stable=2.)
    harness.accept()
    assert harness.last_condition.active_id.eq(1).all()
    flight = harness.airborne(peak=.40)
    assert flight.flight_seen.all() and flight.apex_rise_m.gt(.09).all()
    evidence = harness.settle()
    assert evidence.outcome.eq(1).all()
    assert evidence.physical_success_request_id.eq(1).all()
    assert evidence.stable_landing_s.ge(2. - 1e-9).all()
    # Success occurs after several 0.48s collector slices; the event survives them.
    assert evidence.timestamp_s.gt(4 * .48).all()
    reward = reward_window(evidence)
    result = reward.finish_from_evidence(evidence, terminated=torch.zeros(2, dtype=torch.bool),
                                         truncated=torch.zeros(2, dtype=torch.bool))
    assert result.success_credited.all()
    assert result.event_rewards['physical_skill_success'].eq(1).all()
    with pytest.raises(RuntimeError, match='complete reward interval'):
        reward.finish_from_evidence(evidence, terminated=torch.zeros(2, dtype=torch.bool),
                                    truncated=torch.zeros(2, dtype=torch.bool))

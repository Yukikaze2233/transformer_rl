"""One causal SI projection for fast collection, strict collection and deployment."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.course_tasks import completed_policy_estimate
from transformer_rl.formal.learning.collection import StructuredCollector
from transformer_rl.formal.tasks.protocol import CompletedEstimate


def observations():
    frame = torch.zeros(3, 35)
    frame[:, 9] = -1
    frame[:, 4:7] = torch.tensor([[.3, .4, 0.], [0., 0., .3], [.1, .2, .3]])
    raw = torch.zeros(3, 8)
    raw[:, :3] = torch.tensor([[1., 2., 3.], [2., -1., -.5], [0., .5, 2.]])
    return frame, raw, torch.full((3,), .01, dtype=torch.float64)


def test_fast_and_strict_collectors_decode_identical_si_estimates():
    frame, raw, time = observations()
    fast = completed_policy_estimate(frame, raw, time)
    collector = StructuredCollector.__new__(StructuredCollector)
    collector.num_envs, collector.device = 3, torch.device('cpu')
    strict = collector.completed_estimate(SimpleNamespace(frame=frame, timestamp_s=time), raw)
    for field in fields(CompletedEstimate):
        torch.testing.assert_close(getattr(fast, field.name), getattr(strict, field.name), atol=0, rtol=0)
    torch.testing.assert_close(fast.angular_speed_rad_s, torch.tensor([1., .6, 2 * (.14 ** .5)]))
    torch.testing.assert_close(fast.velocity_z_m_s, raw[:, 2])
    assert fast.valid.all()
    # A 0.6rad/s rotation must not appear below the 0.5rad/s readiness limit.
    assert fast.angular_speed_rad_s[1] > .5


def test_projection_uses_only_actor_sensor_columns_and_estimated_state():
    frame, raw, time = observations()
    before = completed_policy_estimate(frame, raw, time)
    frame[:, :4] = 7
    frame[:, 10:] = 9
    after = completed_policy_estimate(frame, raw, time)
    for field in fields(CompletedEstimate):
        torch.testing.assert_close(getattr(before, field.name), getattr(after, field.name), atol=0, rtol=0)


def test_invalid_gravity_or_gyro_cannot_publish_a_valid_estimate():
    frame, raw, time = observations()
    frame[0, 7:10] = 0
    frame[1, 4] = float('nan')
    raw[2, 0] = float('nan')
    assert not completed_policy_estimate(frame, raw, time).valid.any()


def test_projection_rejects_an_unstamped_or_wrong_width_packet():
    frame, raw, time = observations()
    with pytest.raises(ValueError, match='float64 clock'):
        completed_policy_estimate(frame, raw, time.float())
    with pytest.raises(ValueError, match='actor/state rows'):
        completed_policy_estimate(frame[:, :34], raw, time)


def test_projection_inverts_the_actual_observation_assembler_gyro_scaling():
    from test_physical_observation import fixture

    _, _, assembler, sample, command = fixture()
    state = assembler.measure(sample)
    frame = assembler.actor(assembler.clean_sensors(state), command, now_s=sample.timestamp_s)
    estimate = completed_policy_estimate(frame, torch.zeros(2, 8), sample.timestamp_s)
    torch.testing.assert_close(estimate.angular_speed_rad_s, state.angular_velocity_body_rad_s.norm(dim=-1))

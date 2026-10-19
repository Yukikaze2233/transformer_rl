"""Synthetic ownership/reset/return checks, independent of any simulator."""
from dataclasses import fields, replace

import pytest
import torch

from transformer_rl.formal.learning.contracts import (
    AuxiliaryLabels, BoundaryMasks, ObservationWindows, auxiliary_losses,
    bounded_task_reward, masked_gae,
)


def events(shape):
    yes = torch.ones(shape, dtype=torch.bool)
    no = torch.zeros_like(yes)
    return dict(fully_policy_owned=yes, terminated=no.clone(), truncated=no.clone(),
                final_value_valid=yes.clone(), handoff_after=no.clone(), rollout_end=no.clone(),
                state_label_valid=yes.clone(), next_label_valid=yes.clone(),
                next_same_control_epoch=yes.clone())


def test_boundary_table_distinguishes_truncation_failure_script_and_rollout_end():
    data = events((6,))
    data['truncated'][1] = True
    data['terminated'][2] = True
    data['handoff_after'][3] = True
    data['fully_policy_owned'][4] = False
    data['rollout_end'][5] = True
    masks = BoundaryMasks.from_events(**data)
    masks.validate()
    expected = dict(actor=[1, 1, 1, 1, 0, 1], critic=[1, 1, 1, 1, 0, 1],
                    state_auxiliary=[1, 1, 1, 1, 0, 1], next_auxiliary=[1, 1, 0, 0, 0, 1],
                    trace=[1, 0, 0, 0, 0, 0], bootstrap=[1, 1, 0, 0, 0, 1])
    for name, values in expected.items():
        assert getattr(masks, name).tolist() == [bool(v) for v in values]


def test_missing_terminal_value_is_not_silently_treated_as_failure():
    data = events((1,))
    data['truncated'].fill_(True)
    data['final_value_valid'].fill_(False)
    with pytest.raises(ValueError, match='PRE-reset'):
        BoundaryMasks.from_events(**data)
    data['terminated'].fill_(True)
    masks = BoundaryMasks.from_events(**data)
    assert not masks.bootstrap.any()
    assert not masks.next_auxiliary.any()


def test_reset_observation_cannot_become_previous_episode_prediction_target():
    data = events((2,))
    data['truncated'][0] = True
    data['next_same_control_epoch'][0] = False
    masks = BoundaryMasks.from_events(**data)
    assert masks.bootstrap.tolist() == [True, True]
    assert masks.next_auxiliary.tolist() == [False, True]


def test_gae_does_not_credit_script_success_across_control_handover():
    data = events((3, 1))
    data['handoff_after'][0] = True
    data['fully_policy_owned'][1] = False
    data['rollout_end'][2] = True
    masks = BoundaryMasks.from_events(**data)
    reward = torch.tensor([[1.0], [1000.0], [3.0]])
    value = torch.tensor([[2.0], [float('nan')], [4.0]])
    next_value = torch.tensor([[float('nan')], [float('nan')], [5.0]])
    advantage, returns = masked_gae(reward, value, next_value, masks, gamma=0.9, gae_lambda=0.95)
    torch.testing.assert_close(advantage[:, 0], torch.tensor([-1.0, 0.0, 3.5]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([1.0, 0.0, 7.5]))


def test_time_limit_uses_final_value_while_true_failure_does_not():
    data = events((1, 2))
    data['truncated'][0, 0] = True
    data['terminated'][0, 1] = True
    masks = BoundaryMasks.from_events(**data)
    advantage, returns = masked_gae(torch.ones(1, 2), torch.full((1, 2), 2.0),
                                    torch.tensor([[10.0, float('nan')]]), masks,
                                    gamma=0.9, gae_lambda=0.95)
    torch.testing.assert_close(returns, torch.tensor([[10.0, 1.0]]))
    torch.testing.assert_close(advantage, torch.tensor([[8.0, -1.0]]))


def test_gae_propagates_only_within_owned_continuous_intervals():
    masks = BoundaryMasks.from_events(**events((3, 1)))
    advantage, _ = masked_gae(torch.ones(3, 1), torch.zeros(3, 1), torch.zeros(3, 1), masks,
                              gamma=0.9, gae_lambda=0.5)
    torch.testing.assert_close(advantage[:, 0], torch.tensor([1 + 0.45 + 0.45**2, 1.45, 1.0]))


@pytest.mark.parametrize('field', ['actor', 'state_auxiliary', 'next_auxiliary', 'bootstrap', 'trace'])
def test_inconsistent_masks_are_rejected(field):
    data = events((2,))
    data['fully_policy_owned'][0] = False
    masks = BoundaryMasks.from_events(**data)
    bad = getattr(masks, field).clone()
    bad[0] = True
    with pytest.raises(ValueError):
        replace(masks, **{field: bad}).validate()


def test_observation_windows_repeat_first_on_each_reset_and_own_raw_data():
    prefix = torch.arange(4.0)[:, None, None].expand(4, 2, 35).clone()
    observed = torch.stack((torch.arange(10.0, 14), torch.arange(100.0, 104)), 1)
    observed = observed[..., None].expand(4, 2, 35).clone()
    prefix[..., 14:16] = observed[..., 14:16] = 0
    reset = torch.zeros(4, 2, dtype=torch.bool)
    reset[1, 0] = reset[3, 1] = True
    windows = ObservationWindows(prefix, observed, reset)
    prefix.fill_(-777)
    observed.fill_(-888)
    frames = windows.gather(torch.arange(8))
    torch.testing.assert_close(frames[0, :, 0], torch.tensor([0., 1, 2, 3, 10]))
    torch.testing.assert_close(frames[2, :, 0], torch.full((5,), 11.0))
    torch.testing.assert_close(frames[4, :, 0], torch.tensor([11., 11, 11, 11, 12]))
    torch.testing.assert_close(frames[7, :, 0], torch.full((5,), 103.0))
    assert torch.count_nonzero(frames[..., 14:16]) == 0
    assert not frames.requires_grad
    order = torch.tensor([7, 0, 4, 1])
    torch.testing.assert_close(windows.gather(order), frames[order], atol=0, rtol=0)


def test_raw_window_storage_does_not_duplicate_the_history_for_each_endpoint():
    prefix = torch.zeros(4, 3, 35)
    current = torch.zeros(48, 3, 35)
    windows = ObservationWindows(prefix, current, torch.zeros(48, 3, dtype=torch.bool))
    assert len(windows) == 144
    assert windows.raw_storage_elements == 52 * 3 * 35
    assert windows.raw_storage_elements < 0.22 * 48 * 3 * 5 * 35
    with pytest.raises((IndexError, RuntimeError)):
        windows.gather(torch.tensor([-1]))
    with pytest.raises((IndexError, RuntimeError)):
        windows.gather(torch.tensor([144]))


def test_auxiliary_invalid_nan_labels_are_absent_without_gradient_leakage():
    raw = torch.randn(3, 8, requires_grad=True)
    prediction = torch.randn(3, 6, requires_grad=True)
    state_valid = torch.tensor([True, False, True])
    next_valid = torch.tensor([False, True, False])
    labels = AuxiliaryLabels(torch.zeros(3, 3), torch.full((3, 2), 0.5), torch.zeros(3, 6))
    labels.velocity[~state_valid] = float('nan')
    labels.wheel_contact[~state_valid] = float('nan')
    labels.velocity_delta[~next_valid] = float('nan')
    labels.validate(state_valid, next_valid)
    losses = auxiliary_losses(raw, prediction, labels, state_valid, next_valid)
    assert all(torch.isfinite(loss) for loss in losses)
    sum(losses).backward()
    assert torch.count_nonzero(raw.grad[~state_valid]) == 0
    assert torch.count_nonzero(prediction.grad[~next_valid]) == 0
    assert raw.grad[state_valid].abs().sum() > 0
    assert prediction.grad[next_valid].abs().sum() > 0


def test_auxiliary_no_valid_labels_has_zero_loss_without_fabricated_supervision():
    raw = torch.randn(2, 8, requires_grad=True)
    prediction = torch.randn(2, 6, requires_grad=True)
    valid = torch.zeros(2, dtype=torch.bool)
    labels = AuxiliaryLabels(torch.full((2, 3), float('nan')), torch.full((2, 2), float('nan')),
                             torch.full((2, 6), float('nan')))
    labels.validate(valid, valid)
    losses = auxiliary_losses(raw, prediction, labels, valid, valid)
    assert all(loss.item() == 0 for loss in losses)
    sum(losses).backward()
    assert torch.count_nonzero(raw.grad) == torch.count_nonzero(prediction.grad) == 0


def test_auxiliary_labels_are_detached_and_missing_valid_labels_rejected():
    labels = AuxiliaryLabels(torch.zeros(2, 3, requires_grad=True),
                             torch.full((2, 2), 0.5, requires_grad=True),
                             torch.zeros(2, 6, requires_grad=True))
    valid = torch.ones(2, dtype=torch.bool)
    losses = auxiliary_losses(torch.randn(2, 8, requires_grad=True),
                              torch.randn(2, 6, requires_grad=True), labels, valid, valid)
    sum(losses).backward()
    assert all(getattr(labels, field.name).grad is None for field in fields(labels))
    with pytest.raises(ValueError):
        replace(labels, velocity=torch.full((2, 3), float('nan'))).validate(valid, valid)


def test_failure_tail_does_not_make_immediate_failure_better_than_worst_safe_density():
    gamma = 0.99498743710662
    dt = 0.01
    safe_lower_return = -10 * dt / (1 - gamma)
    fail_from_best_density = bounded_task_reward(torch.tensor([8.0]), torch.tensor([0.0]),
                                                 torch.tensor([True]), policy_dt=dt, gamma=gamma)
    assert fail_from_best_density.item() < safe_lower_return
    ordinary = bounded_task_reward(torch.tensor([0.0]), torch.tensor([1e6]), torch.tensor([False]),
                                    policy_dt=dt, gamma=gamma)
    torch.testing.assert_close(ordinary, torch.tensor([-0.1]))
    # Time-limit truncation supplies failed=False; no failure tail is charged.
    positive = bounded_task_reward(torch.tensor([8.0]), torch.tensor([0.0]), torch.tensor([False]),
                                    policy_dt=dt, gamma=gamma)
    torch.testing.assert_close(positive, torch.tensor([0.08]))


@pytest.mark.parametrize('positive,cost', [(9., 0.), (-1., 0.), (1., -1.), (1., float('nan'))])
def test_reward_component_contract_cannot_be_silently_exceeded(positive, cost):
    with pytest.raises(ValueError):
        bounded_task_reward(torch.tensor([positive]), torch.tensor([cost]), torch.tensor([False]),
                            policy_dt=0.01, gamma=0.99)

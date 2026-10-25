"""Deferred value invariants: same failures, one host read per interval (CPU only)."""
import pytest
import torch

from transformer_rl.formal.control.actuation import EffectiveDriveResponse, OutputFeedbackController, ResponseParameters
from transformer_rl.formal.control.invariants import DeferredInvariants, require


def test_immediate_and_deferred_modes_raise_the_same_labels():
    with pytest.raises(ValueError, match='finite effort'):
        require(None, 'finite effort', torch.tensor([False, True]))
    invariants = DeferredInvariants('cpu')
    require(invariants, 'finite effort', torch.tensor([False, True]))
    require(invariants, 'positive bound', torch.tensor([False]))
    require(invariants, 'finite effort', torch.tensor([False]))  # a later pass does not clear it
    with pytest.raises(ValueError, match='finite effort') as caught:
        invariants.raise_if_violated()
    assert 'positive bound' not in str(caught.value)
    invariants.raise_if_violated()  # read and cleared


def test_controller_defers_nonfinite_feedback_to_the_interval_end():
    controller = OutputFeedbackController((0.,) * 6, 2, pair_domain=None)
    clock = torch.zeros(2, dtype=torch.float64)
    targets = controller.decode(torch.zeros(2, 6), torch.zeros(2, 6), clock)
    bad = torch.zeros(2, 6)
    bad[1, 2] = float('nan')
    with pytest.raises(ValueError, match='feedback_position_rad'):
        controller.effort(targets, bad, torch.zeros(2, 6), clock)
    controller.invariants = DeferredInvariants('cpu')
    controller.effort(targets, bad, torch.zeros(2, 6), clock)  # no host read here
    with pytest.raises(ValueError, match='feedback_position_rad'):
        controller.invariants.raise_if_violated()
    parameters = ResponseParameters((0.,) * 6, (0.,) * 6, (0.,) * 6, (.01,) * 6)
    drive = EffectiveDriveResponse(parameters, 2)
    drive.invariants = DeferredInvariants('cpu')
    drive.advance(torch.full((2, 6), float('inf')), torch.full((2,), .005, dtype=torch.float64), torch.ones(2, 6))
    with pytest.raises(ValueError, match='command must match'):
        drive.invariants.raise_if_violated()

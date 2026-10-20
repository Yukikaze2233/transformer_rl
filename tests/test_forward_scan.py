"""Forward step detector tests."""
import torch

from transformer_rl.formal.lab_env.forward_scan import ForwardScanConfig, ForwardStepDetector


def test_forward_height_difference_is_nonnegative_and_classifies_step_and_wall():
    detector = ForwardStepDetector(3, ForwardScanConfig(min_step_m=.01, max_step_m=.20, wall_step_m=.20))
    result = detector.update(torch.tensor([.0, .0, .0]), torch.tensor([.02, .20, .30]))
    torch.testing.assert_close(result.step_height_m, torch.tensor([.02, .20, .30]))
    assert result.is_step.tolist() == [True, True, False]
    assert result.is_wall.tolist() == [False, False, True]


def test_flat_and_invalid_inputs_are_not_claimed_as_steps():
    detector = ForwardStepDetector(2)
    result = detector.update(torch.zeros(2), torch.zeros(2))
    assert not result.is_step.any() and not result.is_wall.any()
    try:
        detector.update(torch.zeros(2), torch.tensor([float('nan'), 0.]))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite scan must fail")


def test_reset_clears_only_selected_rows():
    detector = ForwardStepDetector(2)
    detector.update(torch.zeros(2), torch.ones(2))
    detector.reset(torch.tensor([1]))
    assert torch.equal(detector.previous_ground_z_m, torch.zeros(2))

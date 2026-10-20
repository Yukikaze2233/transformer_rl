"""182D critic environment feature assembly tests."""
import pytest
import torch

from transformer_rl.formal.lab_env.critic_features import (
    CRITIC182_DIM, CRITIC182_SLICES, environment_features_182,
)


def fixture(n=3):
    values = {name: torch.zeros(n, sl.stop - sl.start) for name, sl in CRITIC182_SLICES.items()}
    values["actor_frame"].fill_(1)
    values["root_velocity_body_m_s"] = torch.tensor([[1., 2., 3.]]).repeat(n, 1)
    values["height_above_ground_m"].fill_(.32)
    values["terrain_scan"].fill_(.1)
    values["mass_scale"].fill_(1.)
    values["wheel_contact"].fill_(1.)
    return values


def test_layout_is_exact_and_ordered():
    values = fixture()
    out = environment_features_182(**values)
    assert out.shape == (3, CRITIC182_DIM)
    for name, sl in CRITIC182_SLICES.items():
        torch.testing.assert_close(out[:, sl], values[name])


def test_nonfinite_wrong_shape_and_bad_contact_are_rejected():
    values = fixture()
    values["terrain_scan"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="terrain_scan"):
        environment_features_182(**values)
    values = fixture()
    values["stiffness"] = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="stiffness"):
        environment_features_182(**values)
    values = fixture()
    values["wheel_contact"].fill_(2)
    with pytest.raises(ValueError, match="wheel_contact"):
        environment_features_182(**values)

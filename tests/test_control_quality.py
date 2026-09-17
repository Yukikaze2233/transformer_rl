"""Physical failure and drift checks independent of task reward."""
import numpy as np
import pytest
import torch

from transformer_rl.control_quality import ControlQuality, EvaluationTrace
from transformer_rl.stability import EpisodeSignalStatistics


def step(quality, t, height=0.30, x=0.0, term=False, trunc=False):
    quality.update({"height": torch.tensor([height]), "tilt": torch.zeros(1),
                    "world_position": torch.tensor([[x, 0.0, height]])},
                   torch.tensor([t], dtype=torch.float64), torch.tensor([term]),
                   torch.tensor([trunc]), 0.1)


def test_wrong_height_latches_failure_even_after_recovery_and_warmup():
    quality = ControlQuality(1, 2, 2)
    step(quality, 0.1, height=0.15)
    step(quality, 0.2, height=0.15)
    step(quality, 0.3)
    step(quality, 0.4, trunc=True)
    report = quality.report()
    assert report["healthy_timeout_fraction"] == 0
    assert report["physical_failure_episodes"] == 1
    assert report["censored_sample_fraction"] == 0


def test_drift_uses_world_positions_after_warmup_and_never_crosses_reset():
    quality = ControlQuality(1, 1, 2)
    for t, x, trunc in [(0.1, 10, False), (0.2, 11, False), (0.3, 11.2, True),
                        (0.4, -20, False), (0.5, -20, False), (0.6, -19.9, False)]:
        step(quality, t, x=x, trunc=trunc)
    report = quality.report()
    assert report["episodes"][0]["world_xy_endpoint_drift"] == pytest.approx(0.2, abs=1e-6)
    assert report["episodes"][1]["world_xy_endpoint_drift"] == pytest.approx(0.1, abs=2e-6)
    assert report["healthy_timeout_fraction"] == 1
    assert report["censored_sample_fraction"] == 0.5
    assert not report["episodes"][1]["healthy_timeout"]


def test_absolute_error_does_not_cancel_signed_bias():
    stats = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=2)
    for i, value in enumerate((-0.1, 0.1)):
        stats.update({"height_error": torch.tensor([value])},
                     torch.tensor([float(i)], dtype=torch.float64), torch.tensor([i == 1]))
    report = stats.report()["signals"]["height_error"]
    assert report["mean"] == 0 and report["mean_abs"] == pytest.approx(0.1)


def test_trace_preserves_current_endpoint_labels_empty_estimates_and_episode_ids(tmp_path):
    trace = EvaluationTrace(2)
    for step_index in range(3):
        trace.add({"observation_time": torch.tensor([step_index, step_index], dtype=torch.float64),
                   "estimated_state_t": torch.empty(2, 0), "state_target_t": torch.empty(2, 0),
                   "frame": torch.ones(2, 30) * step_index}, torch.tensor([step_index == 0, False]))
    path = tmp_path / "trace.npz"
    receipt = trace.save(path)
    with np.load(path) as data:
        assert data["estimated_state_t"].shape == (3, 2, 0)
        assert data["episode_id"].tolist() == [[0, 0], [1, 0], [1, 0]]
        assert data["frame"].dtype == np.float32
        assert data["observation_time"].dtype == np.float64
    assert receipt["steps"] == 3
    with pytest.raises(FileExistsError):
        trace.save(path)

"""Hand-computed policy-rate signal statistics using CPU tensors only."""
import json
import math

import pytest
import torch

from transformer_rl.stability import EpisodeSignalStatistics


def update(statistics, values, timestamp, done=None):
    values = torch.tensor(values, dtype=torch.float64)
    times = torch.as_tensor(timestamp, dtype=torch.float64).expand(statistics.num_envs)
    done = torch.tensor(done if done is not None else [False] * statistics.num_envs)
    statistics.update({"error": values}, times, done)


def test_constant_signed_bias_is_not_jitter():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=3)
    for t in range(3):
        update(statistics, [-100.0], t)
    report = statistics.report()
    signal = report["signals"]["error"]
    assert report["available"] is True
    assert signal["mean"] == -100
    assert signal["within_episode_std"] == signal["derivative_rms"] == 0
    assert signal["max_abs"] == 100
    assert signal["count"] == 3 and signal["derivative_count"] == 2
    assert signal["segments"] == signal["partial_segments"] == 1
    assert signal["completed_segments"] == 0
    assert statistics.report() == report
    json.dumps(report, allow_nan=False)
    with pytest.raises(RuntimeError, match="after report"):
        update(statistics, [0], 3)


def test_signed_oscillation_and_variable_dt_derivative():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=3)
    for value, timestamp in zip([-2, 2, -2], [0.0, 0.5, 2.5]):
        update(statistics, [value], timestamp)
    signal = statistics.report()["signals"]["error"]
    assert signal["mean"] == pytest.approx(-2 / 3)
    assert signal["within_episode_std"] == pytest.approx(math.sqrt(32 / 9))
    assert signal["derivative_rms"] == pytest.approx(math.sqrt((8**2 + (-2)**2) / 2))
    assert signal["max_abs"] == 2


def test_different_constant_environment_means_have_zero_pooled_jitter():
    statistics = EpisodeSignalStatistics(2, settle_steps=0, min_steady_samples=2)
    for t in range(3):
        update(statistics, [-10, 30], t)
    signal = statistics.report()["signals"]["error"]
    assert signal["mean"] == 10
    assert signal["within_episode_std"] == signal["derivative_rms"] == 0
    assert signal["episode_mean_min"] == -10
    assert signal["episode_mean_max"] == 30
    assert signal["episode_mean_std"] == 20
    assert signal["count"] == 6 and signal["segments"] == 2


def test_reset_jump_excluded_and_only_selected_rows_cleared():
    statistics = EpisodeSignalStatistics(2, settle_steps=0, min_steady_samples=2)
    update(statistics, [10, 0], [0, 0])
    update(statistics, [10, 1], [1, 1], [True, False])
    update(statistics, [100, 2], [0, 2])
    update(statistics, [100, 3], [1, 3], [True, False])
    signal = statistics.report()["signals"]["error"]
    # Three segments: [10,10], [100,100], [0,1,2,3].
    assert signal["count"] == 8 and signal["segments"] == 3
    assert signal["mean"] == pytest.approx(226 / 8)
    assert signal["within_episode_std"] == pytest.approx(math.sqrt(5 / 8))
    assert signal["derivative_count"] == 5
    assert signal["derivative_rms"] == pytest.approx(math.sqrt(3 / 5))
    assert signal["completed_segments"] == 2
    assert signal["partial_segments"] == 1
    assert signal["short_segments"] == 0


def test_warmup_transients_and_the_warmup_boundary_are_excluded_each_episode():
    statistics = EpisodeSignalStatistics(1, settle_steps=2, min_steady_samples=2)
    for offset in [0, 100]:
        for t, value in enumerate([1e9, -1e9, 7 + offset, 7 + offset]):
            update(statistics, [value], t, [t == 3])
    signal = statistics.report()["signals"]["error"]
    assert signal["mean"] == 57
    assert signal["within_episode_std"] == signal["derivative_rms"] == 0
    assert signal["max_abs"] == 107
    assert signal["count"] == signal["settled_count"] == 4
    assert signal["total_count"] == 8
    assert signal["segments"] == signal["completed_segments"] == 2
    assert signal["partial_segments"] == signal["short_segments"] == 0


def test_unequal_episode_lengths_pool_within_m2_by_sample_count():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=2)
    for t, value in enumerate([0, 2]):
        update(statistics, [value], t, [t == 1])
    for t, value in enumerate([10, 10, 10, 10]):
        update(statistics, [value], t)
    signal = statistics.report()["signals"]["error"]
    assert signal["mean"] == 7
    assert signal["within_episode_std"] == pytest.approx(math.sqrt(2 / 6))
    assert signal["episode_mean_std"] == 4.5
    assert signal["derivative_rms"] == 1


def test_short_complete_and_partial_segments_are_null_and_counted():
    statistics = EpisodeSignalStatistics(2, settle_steps=1, min_steady_samples=3)
    update(statistics, [100, 100], 0, [True, False])
    update(statistics, [100, 1], [0, 1])
    update(statistics, [2, 3], [1, 2])
    report = statistics.report()
    signal = report["signals"]["error"]
    assert report["available"] is False
    for key in ("mean", "within_episode_std", "derivative_rms", "max_abs",
                "episode_mean_min", "episode_mean_max", "episode_mean_std"):
        assert signal[key] is None
    assert signal["count"] == signal["segments"] == signal["derivative_count"] == 0
    assert signal["total_count"] == 6
    assert signal["settled_count"] == signal["short_count"] == 3
    assert signal["short_segments"] == 3
    assert signal["short_completed_segments"] == 1
    assert signal["short_partial_segments"] == 2
    json.dumps(report, allow_nan=False)


def test_short_segment_does_not_contaminate_usable_partial():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=2)
    update(statistics, [1e9], 0, [True])
    update(statistics, [4], 0)
    update(statistics, [4], 1)
    signal = statistics.report()["signals"]["error"]
    assert signal["count"] == 2 and signal["short_count"] == 1
    assert signal["mean"] == signal["max_abs"] == 4
    assert signal["within_episode_std"] == signal["derivative_rms"] == 0
    assert signal["partial_segments"] == signal["short_completed_segments"] == 1


def test_single_sample_has_no_derivative_even_if_usable():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=1)
    update(statistics, [5], 0, [True])
    signal = statistics.report()["signals"]["error"]
    assert signal["count"] == signal["segments"] == 1
    assert signal["within_episode_std"] == 0
    assert signal["derivative_rms"] is None
    assert signal["derivative_count"] == signal["partial_segments"] == 0


def test_welford_preserves_small_variance_on_large_bias():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=4)
    for t, offset in enumerate([-1, 1, -1, 1]):
        update(statistics, [1e12 + offset], t)
    signal = statistics.report()["signals"]["error"]
    assert signal["mean"] == 1e12
    assert signal["within_episode_std"] == pytest.approx(1)


def test_multiple_signals_are_independent_and_dictionary_order_can_change():
    statistics = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=2)
    statistics.update({"a": torch.tensor([2.]), "b": torch.tensor([-9.])},
                      torch.tensor([0.], dtype=torch.float64), torch.tensor([False]))
    statistics.update({"b": torch.tensor([-9.]), "a": torch.tensor([4.])},
                      torch.tensor([2.], dtype=torch.float64), torch.tensor([False]))
    signals = statistics.report()["signals"]
    assert signals["a"]["mean"] == 3
    assert signals["a"]["within_episode_std"] == signals["a"]["derivative_rms"] == 1
    assert signals["b"]["mean"] == -9
    assert signals["b"]["within_episode_std"] == signals["b"]["derivative_rms"] == 0


def test_no_signals_is_explicitly_unavailable():
    statistics = EpisodeSignalStatistics(1)
    statistics.update({}, None, torch.tensor([False]))
    assert statistics.report()["available"] is False
    assert statistics.report()["signals"] == {}


@pytest.mark.parametrize("kwargs", [
    {"settle_steps": -1}, {"settle_steps": True}, {"settle_steps": 1.0},
    {"min_steady_samples": 0}, {"min_steady_samples": -1},
    {"min_steady_samples": False}, {"min_steady_samples": 2.0}, {"num_envs": 0},
])
def test_invalid_statistics_protocol(kwargs):
    with pytest.raises(ValueError):
        EpisodeSignalStatistics(**({"num_envs": 1} | kwargs))


@pytest.mark.parametrize("signals,time,done,error", [
    ([], [0.], [False], ValueError),
    ({"": torch.tensor([1.])}, [0.], [False], ValueError),
    ({1: torch.tensor([1.])}, [0.], [False], ValueError),
    ({"x": [1.]}, [0.], [False], TypeError),
    ({"x": torch.tensor([1])}, [0.], [False], TypeError),
    ({"x": torch.tensor([True])}, [0.], [False], TypeError),
    ({"x": torch.tensor([[1.]])}, [0.], [False], ValueError),
    ({"x": torch.tensor([float("nan")])}, [0.], [False], FloatingPointError),
    ({"x": torch.tensor([float("inf")])}, [0.], [False], FloatingPointError),
    ({"x": torch.tensor([1.])}, None, [False], ValueError),
    ({"x": torch.tensor([1.])}, [float("nan")], [False], FloatingPointError),
    ({"x": torch.tensor([1.])}, [float("inf")], [False], FloatingPointError),
    ({"x": torch.tensor([1.])}, [0., 1.], [False], ValueError),
    ({"x": torch.tensor([1.])}, torch.tensor([0.]), [False], TypeError),
    ({"x": torch.tensor([1.])}, [0.], [0], TypeError),
    ({"x": torch.tensor([1.])}, [0.], [False, False], ValueError),
    ({}, [0.], [False], ValueError),
])
def test_invalid_signal_inputs(signals, time, done, error):
    statistics = EpisodeSignalStatistics(1)
    if isinstance(time, list):
        time = torch.tensor(time, dtype=torch.float64)
    with pytest.raises(error):
        statistics.update(signals, time, torch.tensor(done))


@pytest.mark.parametrize("second", [0.0, -1.0])
@pytest.mark.parametrize("settle_steps", [0, 200])
def test_time_must_strictly_increase_including_during_warmup(second, settle_steps):
    statistics = EpisodeSignalStatistics(1, settle_steps=settle_steps)
    update(statistics, [1], 0.0)
    with pytest.raises(ValueError, match="strictly increase"):
        update(statistics, [1], second)


@pytest.mark.parametrize("first,second", [("x", "y"), ("x", None), (None, "x")])
def test_signal_names_cannot_appear_disappear_or_change(first, second):
    statistics = EpisodeSignalStatistics(1)
    for index, name in enumerate([first, second]):
        signals = {} if name is None else {name: torch.tensor([1.])}
        time = None if name is None else torch.tensor([float(index)], dtype=torch.float64)
        if index == 0:
            statistics.update(signals, time, torch.tensor([True]))
        else:
            with pytest.raises(ValueError, match="names"):
                statistics.update(signals, time, torch.tensor([False]))

"""Reset-settled dynamic tracking stays distinct from constant-command jitter."""
import math

import pytest
import torch

from transformer_rl.history_control import HistoryControlStatistics


def physical_packet(ages, tick, *, changing=True):
    rows = len(ages)
    reference = torch.tensor([[tick * .03 if changing else 0., 0., .3]] * rows,
                             dtype=torch.float64)
    error = torch.tensor([[tick * .01, -.02, -.003]] * rows, dtype=torch.float64)
    return {"time_s": (ages + 1).double() * .01,
            "command_reference": reference, "actual": reference + error,
            "position_xy": torch.zeros(rows, 2), "tilt": torch.zeros(rows),
            "leg_target": torch.zeros(rows, 4), "wheel_target": torch.zeros(rows, 2),
            "motor_effort": torch.zeros(rows, 6), "motor_velocity": torch.zeros(rows, 6),
            "failure": torch.zeros(rows, dtype=torch.bool),
            "success": torch.zeros(rows, dtype=torch.bool)}


@pytest.mark.parametrize("history_length", (1, 4, 8))
def test_changing_reference_keeps_dynamic_error_after_reset_settling(history_length):
    stats = HistoryControlStatistics(1, history_length, .01, settle_steps=3, min_steady_samples=2)
    done = torch.zeros(1, dtype=torch.bool)
    action = torch.zeros(1, 6)
    for tick in range(10):
        ages = torch.tensor([tick], dtype=torch.int64)
        stats.update(physical_packet(ages, tick), done, ages, action, action)
    report = stats.report()
    full = report["windows"]["full_history"]
    errors = [tick * .01 for tick in range(max(3, history_length - 1), 10)]
    dynamic = full["post_settle_tracking"]
    assert report["schema_version"] == 2
    assert dynamic["samples"] == len(errors) and dynamic["groups"] == 1
    assert full["steady_tracking"]["samples"] == 0
    assert full["steady_tracking"]["axes"]["vx"]["rmse"] is None
    assert dynamic["axes"]["vx"]["bias"] == pytest.approx(sum(errors) / len(errors))
    assert dynamic["axes"]["vx"]["rmse"] == pytest.approx(math.sqrt(sum(x * x for x in errors) / len(errors)))
    assert dynamic["axes"]["height"]["bias"] == pytest.approx(-.003)
    assert "changing references included" in dynamic["centering"]
    assert stats.report() == report


def test_short_post_settle_windows_do_not_pool_across_async_resets_or_history_edge():
    stats = HistoryControlStatistics(2, 4, .01, settle_steps=2, min_steady_samples=3)
    ages = torch.zeros(2, dtype=torch.int64)
    action = torch.zeros(2, 6)
    for tick in range(14):
        done = ages == torch.tensor([4, 6])
        stats.update(physical_packet(ages, tick), done, ages, action, action)
        ages += 1
        ages[done] = 0
    windows = stats.report()["windows"]
    full = windows["full_history"]
    # Five samples from three short row-0 windows stay excluded. Row 1 has
    # two independent four-sample windows after filling H4.
    assert full["samples"] == 13
    assert full["post_settle_tracking"]["samples"] == 8
    assert full["post_settle_tracking"]["groups"] == 2
    assert full["discarded_short_post_settle_samples"] == 5
    assert full["steady_tracking"]["samples"] == 0
    assert windows["reset_filled"]["post_settle_tracking"]["samples"] == 0
    assert windows["reset_filled"]["discarded_short_post_settle_samples"] == 5


def test_fixed_reference_post_settle_and_constant_reference_errors_agree():
    stats = HistoryControlStatistics(1, 1, .01, settle_steps=2, min_steady_samples=2)
    action = torch.zeros(1, 6)
    for tick in range(10):
        ages = torch.tensor([tick], dtype=torch.int64)
        stats.update(physical_packet(ages, tick, changing=False), torch.zeros(1, dtype=torch.bool),
                     ages, action, action)
    full = stats.report()["windows"]["full_history"]
    dynamic, steady = full["post_settle_tracking"], full["steady_tracking"]
    assert dynamic["samples"] == steady["samples"] == 8
    assert dynamic["axes"] == steady["axes"]
    assert full["discarded_short_post_settle_samples"] == 0

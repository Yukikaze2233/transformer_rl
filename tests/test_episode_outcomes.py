"""First-episode evidence separates full survival, censoring and task success."""
import json
import math

import pytest
import torch

from transformer_rl.episode_outcomes import EpisodeOutcomeStatistics


def packet(num_envs=1, *, ticks=1, horizon=3, time_out=False, failure=False,
           success=False, boundary=False, blocked=False, applicable=True):
    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype).expand(num_envs).clone()
    return {"episode_ticks": tensor(ticks, torch.int64),
            "episode_horizon_ticks": tensor(horizon, torch.int64),
            "time_out": tensor(time_out, torch.bool),
            "environment_failure": tensor(failure, torch.bool),
            "task_success": tensor(success, torch.bool),
            "boundary": tensor(boundary, torch.bool), "blocked": tensor(blocked, torch.bool),
            "survival_applicable": tensor(applicable, torch.bool)}


def add(statistics, *, done=False, height=.3, tilt=0., **kwargs):
    n = statistics.num_envs
    statistics.update(packet(n, **kwargs), torch.as_tensor(height, dtype=torch.float64).expand(n).clone(),
                      torch.as_tensor(tilt, dtype=torch.float64).expand(n).clone(),
                      torch.as_tensor(done, dtype=torch.bool).expand(n).clone())


@pytest.mark.parametrize("reason", ("boundary", "blocked"))
def test_early_collection_cut_cannot_be_full_horizon_survival(reason):
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, done=True, time_out=True, **{reason: True})
    report = statistics.report()
    # This is a non-failure truncated episode under the old healthy-timeout
    # definition, but it did not reach its declared horizon.
    assert report["completed_episodes"] == 1
    assert report["environment_failure_episodes"] == 0
    assert report["survival"]["full_horizon_survival_rate"] == 0
    assert report["survival"]["healthy_full_horizon_rate"] == 0
    assert report["end_reasons"][reason] == 1
    assert report["all_requested_accounted"]


@pytest.mark.parametrize("events,reason", (({"boundary": True}, "boundary"),
    ({"blocked": True}, "blocked"), ({"failure": True}, "environment_failure"),
    ({"success": True}, "task_success"), ({"boundary": True, "blocked": True}, "boundary")))
def test_horizon_does_not_override_conflicting_terminal_events(events, reason):
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, horizon=1, time_out=True, done=True, **events)
    report = statistics.report()
    assert report["survival"]["full_horizon_survival_rate"] == 0
    assert report["survival"]["healthy_full_horizon_rate"] == 0
    assert report["end_reasons"][reason] == 1
    assert sum(report["end_reasons"].values()) == 1


def test_full_horizon_success_does_not_require_tracking_correctly():
    statistics = EpisodeOutcomeStatistics(1, .01)
    for tick in range(1, 4):
        add(statistics, ticks=tick, height=.22, time_out=tick == 3, done=tick == 3)
    report = statistics.report()
    assert report["survival"]["full_horizon_survival_rate"] == 1
    assert report["survival"]["healthy_full_horizon_rate"] == 1
    assert report["first_episode_outcomes"][0]["observed_duration_s"] == .03
    assert "not a tracking requirement" in report["protocol"]["health_scope"]


@pytest.mark.parametrize("physical", ({"height": .15}, {"tilt": .7}))
def test_warmup_violation_latches_health_failure_after_recovery(physical):
    statistics = EpisodeOutcomeStatistics(1, .01)
    for tick in range(1, 21):
        add(statistics, ticks=tick, horizon=22, **physical)
    add(statistics, ticks=21, horizon=22)
    add(statistics, ticks=22, horizon=22, time_out=True, done=True)
    report = statistics.report()
    assert report["survival"]["full_horizon_survival_rate"] == 1
    assert report["survival"]["healthy_full_horizon_rate"] == 0
    assert report["survival"]["health_violation_episodes"] == 1
    assert report["environment_failure_episodes"] == 0


def test_violation_must_be_continuous_and_thresholds_are_strict():
    statistics = EpisodeOutcomeStatistics(1, .01)
    for tick in range(1, 20):
        add(statistics, ticks=tick, horizon=41, height=.19)
    add(statistics, ticks=20, horizon=41, height=.20, tilt=.60)
    for tick in range(21, 40):
        add(statistics, ticks=tick, horizon=41, height=.19)
    add(statistics, ticks=40, horizon=41)
    add(statistics, ticks=41, horizon=41, time_out=True, done=True)
    assert statistics.report()["survival"]["healthy_full_horizon_rate"] == 1


def test_discrete_success_is_not_environment_failure_or_failed_survival():
    statistics = EpisodeOutcomeStatistics(1, .01)
    # A learner may mark this task success as terminated. Its PRE-reset
    # environment_failure remains False; intentional low jump height is not
    # evaluated with the continuous-standing health detector.
    add(statistics, applicable=False, success=True, done=True, height=.1)
    report = statistics.report()
    assert report["environment_failure_episodes"] == 0
    assert report["survival"]["status"] == "not_applicable"
    assert report["survival"]["full_horizon_survival_rate"] is None
    assert report["survival"]["healthy_full_horizon_rate"] is None
    assert report["task"]["task_success_rate"] == 1
    assert report["first_episode_outcomes"][0]["health_violation_latched"] is None


def test_discrete_success_rate_also_keeps_unfinished_requested_tasks():
    statistics = EpisodeOutcomeStatistics(2, .01)
    add(statistics, applicable=False, success=(True, False), done=(True, False))
    report = statistics.report()
    assert report["task"]["requested_episodes"] == 2
    assert report["task"]["completed_episodes"] == 1
    assert report["task"]["task_success_rate"] == .5
    assert report["survival"]["healthy_full_horizon_rate"] is None


def test_censored_first_episode_keeps_the_fixed_requested_denominator():
    statistics = EpisodeOutcomeStatistics(2, .01)
    add(statistics, horizon=(1, 3), done=(True, False), time_out=(True, False))
    add(statistics, ticks=(1, 2), horizon=(1, 3), done=(True, False), time_out=(True, False))
    report = statistics.report()
    assert (report["requested_episodes"], report["started_episodes"], report["completed_episodes"],
            report["censored_episodes"], report["not_started_episodes"]) == (2, 2, 1, 1, 0)
    assert report["survival"]["requested_episodes"] == 2
    assert report["survival"]["full_horizon_survival_rate"] == .5
    assert report["survival"]["healthy_full_horizon_rate"] == .5
    assert report["censored_episode_fraction"] == .5
    assert report["observed_samples"] == 3 and report["censored_samples"] == 2
    assert report["censored_sample_fraction"] == pytest.approx(2 / 3)
    assert not report["all_requested_accounted"]


def test_per_case_horizons_and_mixed_task_membership_use_own_cohorts():
    statistics = EpisodeOutcomeStatistics(3, .01)
    add(statistics, horizon=(1, 3, 2), applicable=(True, True, False),
        done=(True, False, False), time_out=(True, False, False))
    add(statistics, ticks=(1, 2, 2), horizon=(9, 3, 2), applicable=(False, True, False),
        done=(False, False, True), success=(False, False, True))
    add(statistics, ticks=(2, 3, 1), horizon=(9, 3, 9), applicable=(False, True, True),
        done=(False, True, False), time_out=(False, True, False))
    report = statistics.report()
    assert report["survival"]["requested_episodes"] == 2
    assert report["survival"]["full_horizon_survival_rate"] == 1
    assert report["task"]["requested_episodes"] == 1
    assert report["task"]["task_success_rate"] == 1
    assert report["all_requested_accounted"]
    assert [row["episode_horizon_ticks"] for row in report["first_episode_outcomes"]] == [1, 3, 2]
    assert [row["samples"] for row in report["first_episode_outcomes"]] == [1, 3, 2]


def test_frequent_failure_and_later_success_never_replaces_the_first_episode():
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, failure=True, done=True)
    for _ in range(10):
        add(statistics, ticks=1, horizon=2)
        add(statistics, ticks=2, horizon=2, time_out=True, done=True)
    report = statistics.report()
    assert report["requested_episodes"] == report["completed_episodes"] == 1
    assert report["observed_samples"] == report["environment_failure_episodes"] == 1
    assert report["survival"]["full_horizon_survival_rate"] == 0
    assert report["end_reasons"]["environment_failure"] == 1


@pytest.mark.parametrize("time_out,reason", ((True, "other_truncation"), (False, "other_termination")))
def test_unknown_early_end_is_accounted_without_fabricating_horizon(time_out, reason):
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, done=True, time_out=time_out)
    report = statistics.report()
    assert report["end_reasons"][reason] == 1
    assert report["survival"]["full_horizon_survival_rate"] == 0


def test_unknown_membership_is_unavailable_and_not_started():
    statistics = EpisodeOutcomeStatistics(2, .01)
    statistics.update(None, None, None, torch.tensor([False, True]))
    report = statistics.report()
    assert not report["available"] and not report["all_requested_accounted"]
    assert report["requested_episodes"] == report["not_started_episodes"] == 2
    assert report["started_episodes"] == report["completed_episodes"] == 0
    assert report["censored_sample_fraction"] is None
    assert report["survival"]["requested_episodes"] is None
    assert report["task"]["requested_episodes"] is None
    assert report["survival"]["status"] == report["task"]["status"] == "unavailable"
    assert report["survival"]["full_horizon_survival_rate"] is None


def test_empty_report_is_unavailable_and_finalization_is_idempotent():
    statistics = EpisodeOutcomeStatistics(1, .01)
    first = statistics.report()
    assert not first["available"] and first["not_started_episodes"] == 1
    assert first == statistics.report()
    first["requested_episodes"] = 999
    assert statistics.report()["requested_episodes"] == 1
    json.dumps(statistics.report(), allow_nan=False)
    with pytest.raises(RuntimeError, match="finalized"):
        add(statistics)


@pytest.mark.parametrize("first_missing", (True, False))
def test_metadata_availability_cannot_change(first_missing):
    statistics = EpisodeOutcomeStatistics(1, .01)
    if first_missing:
        statistics.update(None, None, None, torch.tensor([False]))
        with pytest.raises(ValueError, match="availability changed"):
            add(statistics)
    else:
        add(statistics)
        with pytest.raises(ValueError, match="availability changed"):
            statistics.update(None, None, None, torch.tensor([False]))


@pytest.mark.parametrize("kwargs", ({"ticks": 2}, {"ticks": 0}, {"horizon": 0}))
def test_first_tick_and_positive_limits_are_required(kwargs):
    statistics = EpisodeOutcomeStatistics(1, .01)
    with pytest.raises(ValueError):
        add(statistics, **kwargs)


@pytest.mark.parametrize("kwargs", ({"ticks": 1}, {"ticks": 3}, {"ticks": 2, "horizon": 4},
                                    {"ticks": 2, "applicable": False}))
def test_active_cohort_ticks_limits_and_membership_are_stable(kwargs):
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics)
    with pytest.raises(ValueError):
        add(statistics, **kwargs)
    report = statistics.report()
    assert report["observed_samples"] == 1


def test_invalid_later_row_does_not_partially_update_an_earlier_row():
    statistics = EpisodeOutcomeStatistics(2, .01)
    with pytest.raises(ValueError, match="tick 1"):
        add(statistics, ticks=(1, 2))
    report = statistics.report()
    assert report["not_started_episodes"] == 2 and report["observed_samples"] == 0


@pytest.mark.parametrize("flag", ("time_out", "failure", "success"))
def test_terminal_flags_require_done(flag):
    statistics = EpisodeOutcomeStatistics(1, .01)
    with pytest.raises(ValueError, match="requires done"):
        add(statistics, **{flag: True})


def test_environment_failure_cannot_be_task_success():
    statistics = EpisodeOutcomeStatistics(1, .01)
    with pytest.raises(ValueError, match="cannot both"):
        add(statistics, failure=True, success=True, done=True)


@pytest.mark.parametrize("field,dtype", (("episode_ticks", torch.float64),
    ("episode_horizon_ticks", torch.int32), ("time_out", torch.int64),
    ("survival_applicable", torch.float32)))
def test_metadata_dtypes_are_explicit(field, dtype):
    statistics = EpisodeOutcomeStatistics(1, .01)
    data = packet()
    data[field] = data[field].to(dtype)
    with pytest.raises(ValueError, match=field):
        statistics.update(data, torch.tensor([.3]), torch.tensor([0.]), torch.tensor([False]))


@pytest.mark.parametrize("field", ("episode_ticks", "height", "tilt"))
def test_metadata_and_physical_state_share_done_device_without_initializing_cuda(field):
    statistics = EpisodeOutcomeStatistics(1, .01)
    data = packet()
    height, tilt = torch.tensor([.3]), torch.tensor([0.])
    if field == "height":
        height = torch.empty(1, device="meta")
    elif field == "tilt":
        tilt = torch.empty(1, device="meta")
    else:
        data[field] = torch.empty(1, dtype=torch.int64, device="meta")
    with pytest.raises(ValueError):
        statistics.update(data, height, tilt, torch.tensor([False]))


@pytest.mark.parametrize("mutation", ("missing", "extra", "shape", "not_tensor", "not_dict"))
def test_metadata_structure_cannot_be_inferred_or_broadcast(mutation):
    statistics = EpisodeOutcomeStatistics(1, .01)
    data = packet()
    if mutation == "missing":
        data.pop("boundary")
    elif mutation == "extra":
        data["horizon"] = data["episode_horizon_ticks"]
    elif mutation == "shape":
        data["boundary"] = torch.tensor(False)
    elif mutation == "not_tensor":
        data["boundary"] = False
    else:
        data = []
    with pytest.raises(ValueError):
        statistics.update(data, torch.tensor([.3]), torch.tensor([0.]), torch.tensor([False]))


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
@pytest.mark.parametrize("field", ("height", "tilt"))
def test_physical_samples_must_be_finite_even_after_cohort_completion(field, value):
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, horizon=1, done=True, time_out=True)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        add(statistics, **{field: value})


@pytest.mark.parametrize("bad", ("height_int", "tilt_shape", "done_int", "done_shape"))
def test_physical_and_done_shapes_and_dtypes_are_validated(bad):
    statistics = EpisodeOutcomeStatistics(1, .01)
    height, tilt, done = torch.tensor([.3]), torch.tensor([0.]), torch.tensor([False])
    if bad == "height_int":
        height = torch.tensor([1])
    elif bad == "tilt_shape":
        tilt = torch.tensor(0.)
    elif bad == "done_int":
        done = torch.tensor([0])
    else:
        done = torch.tensor(False)
    with pytest.raises(ValueError):
        statistics.update(packet(), height, tilt, done)


def test_terminal_masks_are_checked_after_completed_rows_are_ignored():
    statistics = EpisodeOutcomeStatistics(1, .01)
    add(statistics, horizon=1, done=True, time_out=True)
    with pytest.raises(ValueError, match="requires done"):
        add(statistics, failure=True)


def test_integer_horizon_identity_is_not_lost_through_float_packing():
    statistics = EpisodeOutcomeStatistics(1, .01)
    horizon = 2**53 + 1
    add(statistics, horizon=horizon)
    assert statistics.report()["first_episode_outcomes"][0]["episode_horizon_ticks"] == horizon


@pytest.mark.parametrize("num_envs,dt", ((0, .01), (True, .01), (1, 0), (1, True),
                                       (1, math.nan), (1, math.inf), (1, -.01)))
def test_constructor_rejects_invalid_cohort_and_clock(num_envs, dt):
    with pytest.raises(ValueError):
        EpisodeOutcomeStatistics(num_envs, dt)

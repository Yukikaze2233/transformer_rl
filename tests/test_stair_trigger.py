"""Simulated forward ToF stair trigger on synthetic height profiles (CPU only)."""
import pytest
import torch

from transformer_rl.formal.lab_env.stair_trigger import SimulatedTofTrigger, TofTriggerConfig


def profile(trigger, risers, slope=0.):
    """[n,k] terrain heights along the beam: (distance, rise) risers plus a ground slope."""
    s = trigger.distances_m[None].expand(len(risers), -1)
    height = slope * s
    for row, steps in enumerate(risers):
        for distance, rise in steps:
            height[row] += rise * (s[row] >= distance)
    return height, torch.ones_like(height, dtype=torch.bool)


def trigger(count, **overrides):
    config = TofTriggerConfig(**{'early_probability': 0., 'latency_intervals': (2, 2), **overrides})
    result = SimulatedTofTrigger(count, min_height_m=.04, max_height_m=.20, config=config, seed=1)
    result.reset(torch.arange(count))
    return result


def test_detector_finds_the_first_riser_and_ignores_slopes_and_roughness():
    tof = trigger(4)
    height, valid = profile(tof, [[(.6, .2)], [(.4, .15), (1.0, .15)], [], [(.5, .02)]])
    height[2] = .22 * tof.distances_m  # a 0.22 rad slope rises 0.35 m over the beam
    found, distance, step = tof.detect(height, valid)
    assert found.tolist() == [True, True, False, False]
    assert abs(float(distance[0]) - .6) <= .05 and abs(float(distance[1]) - .4) <= .05
    torch.testing.assert_close(step[:2], torch.tensor([.2, .15]))
    valid[1, 6:] = False  # the beam ends before the riser
    assert not tof.detect(height, valid)[0][1]


def test_riser_fires_once_within_its_distance_after_the_latency():
    tof = trigger(2, trigger_distance_m=(.5, .5000001))
    idle = torch.ones(2, dtype=torch.bool)
    far, _ = profile(tof, [[(.9, .2)], [(.9, .3)]])  # row 1 is a wall above the stair band
    assert not tof.step(far, torch.ones_like(far, dtype=torch.bool), idle=idle)[0].any()
    near, valid = profile(tof, [[(.45, .2)], [(.45, .3)]])
    fired = [tof.step(near, valid, idle=idle) for _ in range(5)]
    fire = torch.stack([f for f, _ in fired])
    assert fire[:, 1].eq(False).all()
    assert fire[:, 0].tolist() == [False, False, True, False, False]  # latency 2, exactly once
    height = fired[2][1][0]
    assert .19 - 1e-6 <= float(height) <= .20 + 1e-6  # measured with noise, clamped to the band
    assert tof.counts.tolist()[:2] == [1, 1]


def test_due_request_waits_for_idle_and_rearm_detects_the_next_riser():
    tof = trigger(1, trigger_distance_m=(.5, .5000001))
    near, valid = profile(tof, [[(.3, .2)]])
    busy = torch.zeros(1, dtype=torch.bool)
    for _ in range(6):
        assert not tof.step(near, valid, idle=busy)[0].any()
    assert tof.step(near, valid, idle=~busy)[0].all()
    # Standing on the first platform the next riser of a two-level course is ahead.
    second, valid = profile(tof, [[(.35, .15)]])
    for _ in range(4):
        assert not tof.step(second, valid, idle=~busy)[0].any()  # not re-armed yet
    tof.arm(torch.ones(1, dtype=torch.bool))
    fires = [tof.step(second, valid, idle=~busy)[0] for _ in range(3)]
    assert torch.stack(fires).sum() == 1


def test_early_trigger_share_and_checkpoint():
    tof = SimulatedTofTrigger(4000, min_height_m=.04, max_height_m=.20, seed=3)
    tof.reset(torch.arange(4000))
    early = tof.trigger_distance_m > .8
    assert .17 < early.float().mean() < .23
    assert tof.trigger_distance_m[early].le(1.5).all() and tof.trigger_distance_m[~early].ge(.3).all()
    state = tof.state_dict()
    tof.arm(torch.ones(4000, dtype=torch.bool))
    drawn = tof.trigger_distance_m.clone()
    tof.restore_after_physical_reset(state)
    tof.arm(torch.ones(4000, dtype=torch.bool))
    torch.testing.assert_close(tof.trigger_distance_m, drawn)
    with pytest.raises(ValueError):
        TofTriggerConfig(trigger_distance_m=(.8, .3))

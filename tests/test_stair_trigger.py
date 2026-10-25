"""Simulated stair request source on synthetic height profiles (CPU only)."""
import pytest
import torch

from transformer_rl.formal.lab_env.stair_trigger import SimulatedTofTrigger, TofBeam, TofTriggerConfig


def beam(tof, risers, *, origin_x=0., forward=True):
    """Beams along +x from x = ``origin_x``: per row a list of (world x, rise) risers."""
    s = tof.distances_m[None].expand(len(risers), -1)
    world = origin_x + s
    height = torch.zeros_like(s)
    for row, steps in enumerate(risers):
        for at, rise in steps:
            height[row] += rise * (world[row] >= at)
    n = len(risers)
    origin = torch.zeros(n, 2)
    origin[:, 0] = origin_x
    direction = torch.zeros(n, 2)
    direction[:, 0] = 1.
    return TofBeam(height, torch.ones_like(height, dtype=torch.bool), origin, direction,
                   torch.full((n,), forward))


def trigger(count, **overrides):
    config = TofTriggerConfig(**{'early_probability': 0., 'latency_intervals': (2, 2), **overrides})
    result = SimulatedTofTrigger(count, min_height_m=.04, max_height_m=.20, config=config, seed=1)
    result.reset(torch.arange(count))
    return result


def run(tof, make, steps, idle=True):
    flags = torch.full((tof.num_envs,), idle)
    return torch.stack([tof.step(make(), idle=flags)[0] for _ in range(steps)])


def test_detector_finds_the_first_riser_and_ignores_slopes_and_roughness():
    tof = trigger(4)
    b = beam(tof, [[(.6, .2)], [(.4, .15), (1.0, .15)], [], [(.5, .02)]])
    height = b.height_m.clone()
    height[2] = .22 * tof.distances_m  # a 0.22 rad slope rises 0.35 m over the beam
    found, distance, step = tof.detect(height, b.valid)
    assert found.tolist() == [True, True, False, False]
    assert abs(float(distance[0]) - .6) <= .05 and abs(float(distance[1]) - .4) <= .05
    torch.testing.assert_close(step[:2], torch.tensor([.2, .15]))
    valid = b.valid.clone()
    valid[1, 6:] = False  # the beam ends before the riser
    assert not tof.detect(height, valid)[0][1]


def test_riser_fires_once_within_its_distance_after_the_latency():
    tof = trigger(2, trigger_distance_m=(.5, .5000001))
    assert not run(tof, lambda: beam(tof, [[(.9, .2)], [(.9, .3)]]), 1).any()
    fire = run(tof, lambda: beam(tof, [[(.45, .2)], [(.45, .3)]]), 5)  # row 1 is a wall above the band
    assert fire[:, 1].eq(False).all()
    assert fire[:, 0].tolist() == [False, False, True, False, False]
    assert tof.metrics()['requests_submitted'] == {'tof': 1, 'manual_early': 0, 'manual_reverse': 0}


def test_due_request_waits_for_idle_and_drops_when_stale():
    tof = trigger(2, trigger_distance_m=(.5, .5000001))
    near = lambda: beam(tof, [[(.3, .2)], [(.3, .2)]])
    run(tof, near, 3, idle=False)           # detected, due but not idle
    gone = beam(tof, [[(.3, .2)], []])       # row 1 turned away: the riser left the beam
    fire, _ = tof.step(gone, idle=torch.ones(2, dtype=torch.bool))
    assert fire.tolist() == [True, False]
    assert tof.metrics()['stale_requests_dropped'] == 1
    assert tof.armed[1]                      # a stale row is re-armed for its next riser


def test_same_riser_needs_to_leave_the_window_but_the_next_level_triggers_at_once():
    tof = trigger(1, trigger_distance_m=(.5, .5000001), latency_intervals=(0, 0))
    first = lambda: beam(tof, [[(.3, .2)]])
    assert run(tof, first, 1).sum() == 1
    tof.arm(torch.ones(1, dtype=torch.bool))  # the skill ended without climbing
    assert run(tof, first, 5).sum() == 0      # stuck at the same riser: no in-place retrigger
    assert run(tof, lambda: beam(tof, [[]]), 1).sum() == 0  # backing off clears it
    assert run(tof, first, 1).sum() == 1
    # On the first platform of a two-level course the second riser is new.
    tof.arm(torch.ones(1, dtype=torch.bool))
    assert run(tof, lambda: beam(tof, [[(.3, .2), (.9, .15)]], origin_x=.5), 1).sum() == 1


def test_reverse_travel_is_an_operator_request_and_checkpoint_round_trip():
    tof = trigger(1, trigger_distance_m=(.5, .5000001), latency_intervals=(0, 0))
    assert run(tof, lambda: beam(tof, [[(.3, .2)]], forward=False), 1).sum() == 1
    assert tof.metrics()['requests_submitted']['manual_reverse'] == 1
    many = SimulatedTofTrigger(4000, min_height_m=.04, max_height_m=.20, seed=3)
    many.reset(torch.arange(4000))
    early = many.trigger_distance_m > .8
    assert .17 < early.float().mean() < .23 and many.early.eq(early).all()
    state = many.state_dict()
    many.arm(torch.ones(4000, dtype=torch.bool))
    drawn = many.trigger_distance_m.clone()
    many.restore_after_physical_reset(state)
    many.arm(torch.ones(4000, dtype=torch.bool))
    torch.testing.assert_close(many.trigger_distance_m, drawn)
    with pytest.raises(ValueError):
        TofTriggerConfig(trigger_distance_m=(.8, .3))
    with pytest.raises(ValueError):  # the beam would reach beyond the ~2 m ToF range
        TofTriggerConfig(early_distance_m=(.8, 2.))

"""The heading outer loop reads the IMU heading through the same sensor lag."""
import math
from dataclasses import replace

import torch

from test_physical_observation import fixture
from test_sensor_model import no_noise
from transformer_rl.formal.lab_env.sensors import LabSensorModel


def heading_model(delay_s=.01, uplink_s=None):
    _, _, assembler, raw, _ = fixture(4)
    template = assembler.measure(raw)
    config = no_noise(nominal_episode_fraction=0., extra_delay_probability=1.,
                      observation_delays_s=(delay_s,), shared_delay=True)
    kwargs = {} if uplink_s is None else {'uplink_delay_s': torch.full((4,), uplink_s, dtype=torch.float64),
                                          'maximum_uplink_s': uplink_s}
    model = LabSensorModel(4, 5, config, **kwargs)
    now = torch.zeros(4, dtype=torch.float64)
    model.reset(torch.ones(4, dtype=torch.bool), now)
    model.seed_after_reset(torch.ones(4, dtype=torch.bool), replace(template, timestamp_s=now),
                           heading_rad=torch.zeros(4))
    return model, template


def advance(model, template, yaw):
    tick = model.physical_tick + 1
    state = replace(template, timestamp_s=torch.full((4,), tick * .005, dtype=torch.float64))
    model.push(state, tick, heading_rad=torch.full((4,), yaw))


def test_heading_lags_by_the_shared_observation_delay():
    model, template = heading_model(delay_s=.01)  # two physics ticks
    for tick, yaw in enumerate((.1, .2, .3, .4), start=1):
        advance(model, template, yaw)
    torch.testing.assert_close(model.delayed_heading(), torch.full((4,), .2))


def test_heading_interpolation_crosses_pi_without_collapsing_to_zero():
    model, template = heading_model(delay_s=.005, uplink_s=.0025)  # 1.5 ticks
    for yaw in (math.pi - .05, -math.pi + .05, -math.pi + .15):
        advance(model, template, yaw)
    heading = model.delayed_heading()
    # Halfway between pi-0.05 and -pi+0.05 is +/-pi, not 0.
    assert (heading.abs() > math.pi - 1e-3).all()

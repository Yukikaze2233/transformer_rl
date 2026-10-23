"""Private native-factory configuration checks; no SDK/application creation."""
from dataclasses import asdict
import json
import math

import pytest

from transformer_rl.formal.tasks.trajectory import CommandRates
from transformer_rl.formal.runtime.assembly import NativeRuntimeTuning
from transformer_rl.formal.simulation.task import NativeTaskLimits
from transformer_rl.formal.simulation.terrain import TerrainEditTuning
from transformer_rl.formal.runtime.worker import _initial_policy_boundary, parse_tuning
from transformer_rl.formal.tasks.recovery import RecoveryTuning
from transformer_rl.formal.tasks.reset_program import ResetProgramTuning


def test_native_factory_reconstructs_explicit_json_tuning_without_importing_an_sdk():
    import sys
    absent_before = {name for name in ('isaaclab.app', 'omni.physx') if name not in sys.modules}
    tuning = NativeRuntimeTuning(
        ResetProgramTuning(30., 60., (.5, 3.), (3., 8.), 2., (1., 1.),
                           0., 0., 0., 0., 0., .02, .001),
        RecoveryTuning(math.radians(50), .25, math.pi, .04, .15, 1e-6, 1e-6, 80, math.pi/24, 32),
        NativeTaskLimits(.08, .01, .5, 2.5, ('base_link',), 2., .02, .02, .01, 5.),
        CommandRates(2., 3., .2), TerrainEditTuning(5., .01, 1e-6, .01), None)
    document = json.loads(json.dumps(asdict(tuning)))
    assert parse_tuning(document) == tuning
    assert all(name not in sys.modules for name in absent_before)
    document['knee_margin_rad'] = float('nan')
    with pytest.raises(ValueError, match='positive knee margin'):
        parse_tuning(document)


def test_initial_scene_refresh_reaches_a_real_policy_boundary_before_the_first_reset():
    from test_native_episode import fixture, reset
    from transformer_rl.formal.simulation.episode import NativeEpisodeLifecycle

    previous, robot, sim, writes = fixture()
    plant = previous.plant
    _initial_policy_boundary(sim, plant.scene, .01)
    assert sim.steps == plant.scene.observation.physical_tick == 2
    life = NativeEpisodeLifecycle(plant, previous.ground, previous.protocol)
    receipt = reset(life, robot)
    assert receipt.physical_tick == 2 and life.episode_id.eq(0).all()
    assert writes and sim.steps == 2
    with pytest.raises(ValueError, match='fresh200Hz'):
        _initial_policy_boundary(sim, plant.scene, .01)


def test_one_refresh_is_not_a_policy_boundary_and_cannot_write_the_first_reset():
    from test_native_episode import fixture, reset
    from transformer_rl.formal.simulation.episode import NativeEpisodeLifecycle

    previous, robot, sim, writes = fixture()
    plant = previous.plant
    sim.step(render=False)
    plant.scene.observation.after_physics_step(1)
    life = NativeEpisodeLifecycle(plant, previous.ground, previous.protocol)
    with pytest.raises(RuntimeError, match='boundary'):
        reset(life, robot)
    assert not writes

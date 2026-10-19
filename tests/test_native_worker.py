"""Private native-factory configuration checks; no SDK/application creation."""
from dataclasses import asdict
import json
import math

import pytest

from transformer_rl._command_trajectory import CommandRates
from transformer_rl._native_runtime import NativeRuntimeTuning
from transformer_rl._native_task import NativeTaskLimits
from transformer_rl._native_terrain import TerrainEditTuning
from transformer_rl._native_worker import parse_tuning
from transformer_rl._recovery_poses import RecoveryTuning
from transformer_rl._reset_program import ResetProgramTuning


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

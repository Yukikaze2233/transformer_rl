"""The formal default collects no loop geometry; diagnostics remain opt-in."""
import pytest
import torch

from transformer_rl.formal.lab_env.mechanics import NativeLoopTelemetry
from transformer_rl.formal.lab_env.train import _parse_args


def observer(*, enabled):
    bodies = tuple(f'body_{i}' for i in range(19))
    constraints = [
        {'name': f'loop_{i}', 'type': 'spherical', 'body0': bodies[0],
         'body1': bodies[i + 1], 'local_pos0_m': [0., 0., 0.],
         'local_pos1_m': [0., 0., 0.]}
        for i in range(6)
    ]
    return NativeLoopTelemetry(constraints, bodies, 2, 'cpu', enabled=enabled)


def test_disabled_diagnostics_never_read_pose_or_allocate_reduction_state():
    value = observer(enabled=False)
    before = torch.get_rng_state().clone()
    # No tensor read is valid here: the disabled observer must accept no data
    # and must never manufacture a measured zero or claim qualification.
    value.observe_physics(None, None)
    value.observe_reset(None, None)
    value.clear_window()
    assert value.summary() == {
        'six_loop_enabled': False,
        'six_loop_scope': 'disabled; no native loop observations collected',
        'physical_qualification': False,
    }
    assert not hasattr(value, '_local')
    assert torch.equal(before, torch.get_rng_state())


def test_diagnostic_cli_defaults_off_and_requires_explicit_enablement():
    args = ['--runtime-json', '/unused/runtime.json', '--output', '/unused/run']
    assert _parse_args(args).loop_telemetry is False
    assert _parse_args([*args, '--loop-telemetry']).loop_telemetry is True


def test_opt_in_geometry_still_records_large_residual_without_terminating():
    value = observer(enabled=True)
    poses = torch.zeros(2, 19, 7)
    poses[..., 6] = 1.
    poses[0, 1, 0] = .02
    value.observe_physics(poses, torch.zeros(2, 18))
    result = value.summary()
    assert result['six_loop_enabled'] is True
    assert result['six_loop_closure_max_m'] == pytest.approx(.02)
    assert result['physical_qualification'] is False


@pytest.mark.parametrize('enabled', (0, 1, None, 'false'))
def test_switch_rejects_implicit_truthiness(enabled):
    with pytest.raises(TypeError, match='explicit boolean'):
        observer(enabled=enabled)

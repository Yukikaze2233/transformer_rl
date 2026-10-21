"""Native-pose loop measurements preserve peaks and physical boundaries."""
import torch

from transformer_rl.formal.lab_env.mechanics import NativeLoopTelemetry


def fixture():
    names = tuple('body_' + str(i) for i in range(19))
    constraints = [{'name': f'loop_{i}', 'type': 'spherical',
                    'body0': names[0], 'body1': names[i + 1],
                    'local_pos0_m': [.01, 0., 0.], 'local_pos1_m': [.01, 0., 0.]}
                   for i in range(6)]
    observer = NativeLoopTelemetry(constraints, names, 2, 'cpu')
    poses = torch.zeros(2, 19, 7)
    poses[..., 6] = 1.
    return observer, poses, torch.zeros(2, 18)


def test_observer_preserves_an_early_physical_peak_after_the_state_recovers():
    observer, poses, velocity = fixture()
    poses[0, 1, 0] = .008
    observer.observe_physics(poses, velocity)
    poses[0, 1, 0] = 0.
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert abs(result['six_loop_closure_max_m'] - .008) < 1e-8
    assert result['six_loop_physics_frames'] == 2
    assert result['six_loop_constraint_samples'] == 24
    assert abs(result['six_loop_max_by_constraint_m']['loop_0'] - .008) < 1e-8
    assert result['six_loop_first_physics_max_m'] is None


def test_first_physical_step_marker_survives_a_rollout_boundary_and_partial_reset():
    observer, poses, velocity = fixture()
    observer.observe_reset(poses, torch.tensor([0]))
    observer.clear_window()
    poses[0, 1, 0] = .003
    poses[1, 1, 0] = .009
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert result['six_loop_first_physics_rows'] == 1
    assert abs(result['six_loop_first_physics_max_m'] - .003) < 1e-8
    observer.observe_reset(poses, torch.tensor([1]))
    poses[0, 1, 0] = .02
    poses[1, 1, 0] = .004
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert result['six_loop_first_physics_rows'] == 2
    assert abs(result['six_loop_first_physics_max_m'] - .004) < 1e-8
    assert abs(result['six_loop_reset_after_forward_max_m'] - .009) < 1e-8


def test_window_reset_preserves_lifetime_evidence_and_summary_is_read_only():
    observer, poses, velocity = fixture()
    poses[0, 1, 0] = .007
    observer.observe_physics(poses, velocity)
    assert observer.summary() == observer.summary()
    observer.clear_window()
    poses[0, 1, 0] = .002
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert abs(result['six_loop_closure_max_m'] - .002) < 1e-8
    assert abs(result['six_loop_lifetime_max_m'] - .007) < 1e-8


def test_nonfinite_geometry_is_unavailable_instead_of_a_zero_residual():
    observer, poses, velocity = fixture()
    poses[0, 1, 0] = float('nan')
    velocity[1, 2] = float('inf')
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert result['six_loop_nonfinite_constraint_observations'] == 1
    assert result['six_loop_closure_max_m'] is None
    assert result['six_loop_lifetime_max_m'] is None
    assert result['joint_velocity_max_native_units_per_s'] is None
    assert result['joint_velocity_nonfinite_observations'] == 1


def test_source_anchor_rotation_and_translation_do_not_change_closure_or_inputs():
    observer, poses, velocity = fixture()
    poses[0, 1, 0] = .003
    original = poses.clone()
    rng = torch.get_rng_state().clone()
    observer.observe_physics(poses, velocity)
    first = observer.summary()['six_loop_closure_max_m']
    torch.testing.assert_close(poses, original, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    observer.clear_window()
    # Rotate the complete configuration 90 degrees about world Z, then
    # translate it; local anchors must rotate with their own link frames.
    poses[..., 0] = -original[..., 1] + 2.
    poses[..., 1] = original[..., 0] - 1.
    poses[..., 2] = original[..., 2] + 3.
    poses[..., 5] = 2. ** -.5
    poses[..., 6] = 2. ** -.5
    observer.observe_physics(poses, velocity)
    assert abs(observer.summary()['six_loop_closure_max_m'] - first) < 2e-7


def test_physical_mean_and_rms_are_sample_moments_not_peak_means():
    observer, poses, velocity = fixture()
    poses[0, 1, 0] = .008
    poses[1, 1, 0] = .002
    observer.observe_physics(poses, velocity)
    result = observer.summary()
    assert result['six_loop_valid_constraint_samples'] == 12
    assert abs(result['six_loop_mean_error_m'] - .010 / 12) < 1e-9
    assert abs(result['six_loop_rms_error_m'] - ((.008**2 + .002**2) / 12) ** .5) < 1e-9
    assert abs(result['six_loop_mean_by_constraint_m']['loop_0'] - .010 / 2) < 1e-9
    assert result['six_loop_mean_by_constraint_m']['loop_1'] == 0

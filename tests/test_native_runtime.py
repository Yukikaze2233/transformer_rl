from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.simulation.ground import WheelCollisionCentres
from transformer_rl.formal.runtime.assembly import NativeTaskRuntime


def test_drive_geometry_uses_selected_rotated_collider_offsets_and_body_order():
    positions = torch.zeros(1, 4, 3, dtype=torch.float64)
    rotations = torch.eye(3, dtype=torch.float64).repeat(1, 4, 1, 1)
    # Reversed body indices and mirrored local offsets must remain named.
    positions[0, 3, 1], positions[0, 1, 1] = .20, -.19
    rotations[0, 3] = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    rotations[0, 1] = rotations[0, 3]
    state = SimpleNamespace(body_position_m=positions, body_rotation=rotations)
    kin = SimpleNamespace(binding=SimpleNamespace(wheel_body_indices=(3, 1)),
                           nominal_tree_position=torch.zeros(18), forward=lambda _: state)
    wheels = WheelCollisionCentres(('left', 'right'), ((0., 0., .02), (0., 0., -.03)),
                                   (.06, .06), (.025, .025))
    envelope = NativeTaskRuntime._drive_envelope(kin, wheels)
    assert envelope.track_width_m == pytest.approx(.44)
    assert envelope.wheel_radius_m == .06
    with pytest.raises(ValueError, match='equal selected wheel radii'):
        NativeTaskRuntime._drive_envelope(kin, WheelCollisionCentres(
            wheels.body_names, wheels.local_xyz_m, (.06, .07), wheels.length_m))

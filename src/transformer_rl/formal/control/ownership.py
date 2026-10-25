"""Private command provenance and common coaxial coordinate representation."""
from enum import IntEnum
import math

import torch

from .invariants import require


class ControlSource(IntEnum):
    POLICY = 0
    BOUNDED_SCRIPT = 1
    PREPARATION_HOLD = 2


def canonical_coaxial_displacement(position_rad, nominal_rad):
    """Remove one common winding per leg,preserving both output coordinates.

    Callers validate named layouts and finite inputs. This is not a differential
    joint transform or a limit projection. Independent wrapping at +/-pi would
    create a spurious relative-phase jump in a coupled leg's observation.
    """
    pairs = (position_rad.double() - nominal_rad.double()).unflatten(-1, (2, 2))
    turns = torch.floor(pairs.mean(-1) / (2*math.pi) + .5)
    return (pairs - turns[..., None]*(2*math.pi)).flatten(-2), turns


def validate_issued_control(action, source, *, num_envs, device, policy_owned=None, invariants=None):
    """Validate the actual normalized command, without relabelling or clipping.

    A hold shares one full-turn branch per coaxial pair. Its mean displacement
    is at most pi and its relative displacement change is below pi; hence each
    position command is bounded by 1.5*pi/0.25 = 6*pi. This is a representation
    bound,not a hardware motion envelope. Holds always request zero wheel speed.
    Value checks are deferred to ``invariants`` when given (no host sync).
    """
    if (not isinstance(action, torch.Tensor) or action.shape != (num_envs, 6)
            or action.dtype != torch.float32 or action.device != torch.device(device)
            or not isinstance(source, torch.Tensor) or source.shape != (num_envs,)
            or source.dtype != torch.long or source.device != action.device):
        raise ValueError('issued control requires finite six-axis commands and exact per-row source metadata')
    require(invariants, 'issued control requires finite six-axis commands and exact per-row source metadata',
            ~torch.isfinite(action).all(-1) | (source < int(ControlSource.POLICY))
            | (source > int(ControlSource.PREPARATION_HOLD)))
    hold = source == int(ControlSource.PREPARATION_HOLD)
    bounded = (action.abs() <= action.new_tensor((3, 3, 3, 3, 9, 9)) + 1e-6).all(-1)
    held = (action[:, :4].abs() <= 6*math.pi + 1e-6).all(-1) & (action[:, 4:] == 0).all(-1)
    require(invariants, 'policy/script action bounds or canonical preparation hold coordinates are invalid',
            (~hold & ~bounded) | (hold & ~held))
    if policy_owned is not None:
        if (not isinstance(policy_owned, torch.Tensor) or policy_owned.shape != (num_envs,)
                or policy_owned.dtype != torch.bool or policy_owned.device != action.device
                or not torch.equal(source == int(ControlSource.POLICY), policy_owned)):
            raise ValueError('actual control source differs from policy ownership')

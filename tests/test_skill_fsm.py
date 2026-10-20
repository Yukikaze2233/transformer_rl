"""台阶与跳跃状态机的纯 CPU 单测。"""
import math

import torch

from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.lab_env.skill_fsm import (
    JumpFSM, JumpFSMConfig, JumpState, StairFSM, StairFSMConfig, StairState,
)


def stair_inputs(n=1, *, step=.12, force=30., body=.32, command=.32, tilt=0., base=0.,
                 auto=True, manual=False, manual_height=None):
    return dict(
        step_height_m=torch.full((n,), step),
        wheel_force_n=torch.full((n, 2), force),
        body_height_m=torch.full((n,), body),
        tilt_rad=torch.full((n,), tilt),
        base_contact_force_n=torch.full((n,), base),
        command_height_m=torch.full((n,), command),
        auto_enabled=auto,
        manual_request=torch.full((n,), manual),
        manual_height_m=None if manual_height is None else torch.full((n,), manual_height),
    )


def test_auto_stair_enters_success_and_restores_normal_mode():
    fsm = StairFSM(1)
    out = fsm.step(**stair_inputs(step=.12))
    assert out.entered.item() and out.state.item() == int(StairState.STAIR)
    assert out.mode.item() == int(CommandMode.STAIR)
    torch.testing.assert_close(out.height_command_m, torch.tensor([.34]))
    for _ in range(19):
        out = fsm.step(**stair_inputs(step=.12, body=.34))
    assert not out.success.item()
    out = fsm.step(**stair_inputs(step=.12, body=.34))
    assert out.success.item() and out.exited.item()
    assert out.state.item() == int(StairState.NORMAL)
    assert out.mode.item() == int(CommandMode.NORMAL)


def test_manual_entry_uses_same_stair_state_without_forward_detection():
    fsm = StairFSM(1)
    out = fsm.step(**stair_inputs(step=0., manual=True, manual_height=.12))
    assert out.entered.item() and out.state.item() == int(StairState.STAIR)


def test_step_up_is_bounded_and_wall_is_not_entered():
    fsm = StairFSM(1)
    out = fsm.step(**stair_inputs(step=.06, command=.34))
    assert out.state.item() == int(StairState.STEP_UP)
    assert out.height_command_m.item() <= .35
    wall_fsm = StairFSM(1)
    wall = wall_fsm.step(**stair_inputs(step=.25, auto=True))
    assert wall.wall.item() and not wall.entered.item()
    assert wall.state.item() == int(StairState.NORMAL)


def test_body_bump_allows_body_contact_and_tucks_after_start():
    fsm = StairFSM(1)
    out = fsm.step(**stair_inputs(step=.18, force=30.))
    assert out.state.item() == int(StairState.BODY_BUMP)
    assert out.body_contact_allowed.item()
    for _ in range(20):
        out = fsm.step(**stair_inputs(step=.18, force=30., body=.32))
    assert out.leg_tuck.item() and out.stage.item() >= 0


def test_stair_failure_on_drop_tilt_and_base_contact():
    for kwargs in ({'body': .20}, {'tilt': math.radians(50)}, {'base': 2.}):
        fsm = StairFSM(1)
        fsm.step(**stair_inputs(step=.12))
        out = fsm.step(**stair_inputs(step=.12, **kwargs))
        assert out.failure.item() and out.exited.item()


def jump_inputs(n=1, *, request=True, rise=.10, command=.32, body=.32, force=30.,
                tilt=0., base=0., vertical=0.):
    return dict(
        request=torch.full((n,), request),
        request_height_m=torch.full((n,), rise),
        command_height_m=torch.full((n,), command),
        body_height_m=torch.full((n,), body),
        wheel_force_n=torch.full((n, 2), force),
        tilt_rad=torch.full((n,), tilt),
        base_contact_force_n=torch.full((n,), base),
        vertical_velocity_m_s=torch.full((n,), vertical),
    )


def test_jump_phases_and_success_require_settle_time():
    fsm = JumpFSM(1, JumpFSMConfig(preload_s=.02, takeoff_s=.02, flight_timeout_s=1.,
                                   landing_timeout_s=1., settle_s=.05))
    out = fsm.step(**jump_inputs())
    assert out.state.item() == int(JumpState.PRELOAD) and out.mode.item() == int(CommandMode.JUMP)
    for _ in range(2):
        out = fsm.step(**jump_inputs())
    assert out.state.item() == int(JumpState.TAKEOFF)
    for _ in range(2):
        out = fsm.step(**jump_inputs())
    assert out.state.item() == int(JumpState.FLIGHT)
    out = fsm.step(**jump_inputs(body=.43, vertical=0.))
    assert out.state.item() == int(JumpState.LANDING)
    for _ in range(5):
        out = fsm.step(**jump_inputs(body=.43, vertical=0., force=30.))
    assert out.state.item() == int(JumpState.SETTLE)
    for _ in range(5):
        out = fsm.step(**jump_inputs(body=.43, vertical=0., force=30.))
    assert out.success.item() and out.exited.item()
    assert out.mode.item() == int(CommandMode.NORMAL)


def test_jump_request_zero_is_ignored_when_idle_and_failure_aborts():
    fsm = JumpFSM(1)
    out = fsm.step(**jump_inputs(request=False, rise=0.))
    assert out.state.item() == int(JumpState.NORMAL)
    fsm.step(**jump_inputs())
    out = fsm.step(**jump_inputs(tilt=math.radians(50)))
    assert out.failure.item() and out.exited.item()

"""Partial resets must not leak neutral targets into other rows' latch delay."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from transformer_rl.formal.control.actuation import OutputFeedbackController
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.control.ownership import ControlSource
from transformer_rl.formal.lab_env.action_transport import ActionLatchConfig, SubstepActionLatch
from transformer_rl.formal.lab_env import env as env_module


def env_methods():
    path = Path(env_module.__file__)
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    names = ('_pre_physics_step', '_apply_action', '_reset_targets')
    methods = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch, 'ControlSource': ControlSource, 'ACTION_DIM': 6, 'CommandMode': CommandMode,
                 'replace': replace, '_select_targets': env_module._select_targets}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def fake_env(count=2):
    controller = OutputFeedbackController((0.,) * 6, count, pair_domain=None)
    used = []
    effort = controller.effort

    def recording_effort(targets, *args, **kwargs):
        used.append(targets.admitted_leg_position_rad[:, 0].clone())
        return effort(targets, *args, **kwargs)

    controller.effort = recording_effort
    clock = {'now': torch.zeros(count, dtype=torch.float64)}
    latch = SubstepActionLatch(count, ActionLatchConfig(enabled=True, delay_substeps=(2,), nominal_fraction=0.),
                               seed=3)
    latch.reset(torch.ones(count, dtype=torch.bool))
    env = SimpleNamespace(
        num_envs=count, _device=torch.device('cpu'), _before_action=torch.zeros(count, 6),
        _previous_action=torch.zeros(count, 6), _issued_action=torch.zeros(count, 6),
        _action_bounds=torch.tensor([3., 3., 3., 3., 9., 9.]), _apply_push=lambda: None,
        _targets=None, _clock=lambda: clock['now'], _encoder_position=lambda q: q,
        _joint_positions=lambda: torch.zeros(count, 6), _active_indices=torch.arange(6),
        _recover_task=None, _recovery_preparation=None, _action_transport=None, _formal_reward=None,
        _control=SimpleNamespace(decode=controller.decode, controller=controller, drive=None),
        _action_latch=latch, _latch_active=torch.zeros(count, dtype=torch.bool), _held_targets=None,
        _held_generated_at_s=torch.zeros(count, dtype=torch.float64), _substep=0,
        _as_torch=lambda value: value, _episode_randomization=None, _motor_envelope=None,
        _spring_force=None, _last_effort=torch.zeros(count, 6),
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(count, 6)),
                              set_joint_effort_target=lambda value: None))
    return env, used, clock


def interval(methods, env, clock, action):
    methods['_pre_physics_step'](env, torch.full((env.num_envs, 6), float(action)))
    for _ in range(4):
        methods['_apply_action'](env)
        clock['now'] = clock['now'] + .005


def test_resetting_one_of_two_rows_keeps_the_other_rows_delayed_target():
    methods = env_methods()
    env, used, clock = fake_env()
    interval(methods, env, clock, 1.)          # both rows latch 0.25 rad
    used.clear()
    reset = torch.tensor([True, False])
    methods['_reset_targets'](env, reset, torch.zeros(2, 6))
    env._action_latch.reset(reset)
    env._latch_active[reset] = False
    interval(methods, env, clock, 2.)          # new target 0.5 rad
    used = torch.stack(used)                   # [substep, row]
    # Row 1 was not reset: its two lagged substeps still apply its own 0.25 rad
    # target, never the neutral 0 rad target rebuilt for row 0.
    torch.testing.assert_close(used[:2, 1], torch.full((2,), .25))
    torch.testing.assert_close(used[2:, 1], torch.full((2,), .5))
    # The reset row has no held policy target and latches its first target at once.
    torch.testing.assert_close(used[:, 0], torch.full((4,), .5))

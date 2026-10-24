"""Critic source/time invariants, using native stand-ins without starting an SDK."""
from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from transformer_rl.formal.control.observation import ObservationAssembler
from transformer_rl.formal.lab_env.critic_features import (
    CRITIC182_SLICES, CriticActionHistory, environment_features_182,
)


def env_methods(*names):
    path = Path(__file__).resolve().parents[1] / 'src/transformer_rl/formal/lab_env/env.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    methods = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    namespace = {'torch': torch, 'environment_features_182': environment_features_182}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return type('NativeContractMethods', (), {name: namespace[name] for name in names})


def critic_env(count=3):
    from test_physical_observation import fixture

    _, binding, assembler, sample, commands = fixture(count)
    env = env_methods('_critic_182', '_wheel_velocity_body', '_capture_terminal', '_get_observations')()
    env.num_envs, env._device = count, torch.device('cpu')
    env.step_dt, env.physics_dt = .01, .005
    env._scan_debug = False
    env._as_torch = lambda value: value
    env._binding = binding
    env._wheel_body_indices = torch.tensor(binding.wheel_body_indices)
    env._root_link_quat = lambda: sample.root_link_orientation_xyzw
    env._root_link_position = lambda: sample.root_link_position_world_m
    env._commands = lambda: commands
    env._observations = SimpleNamespace(assembler=assembler, append=lambda *args, **kwargs: (
        torch.zeros(count, 5, 35), None))
    env._sample = lambda: sample
    env._last_sample, env._last_state = sample, assembler.measure(sample)
    env._sensor_packet = lambda state: replace(
        assembler.clean_sensors(state), sampled_at_s=state.timestamp_s[:, None] -
        torch.tensor([0., .005, .010, .015], dtype=torch.float64)[None])
    env.robot = SimpleNamespace(data=SimpleNamespace(
        body_lin_vel_w=sample.root_com_velocity_world_m_s[:, None].expand(-1, 19, -1).clone(),
        root_com_vel_w=torch.cat((sample.root_com_velocity_world_m_s, sample.angular_velocity_world_rad_s), -1)))
    env._stiffness = torch.tensor([160., 160., 160., 160., 0., 0.])
    env._damping = torch.tensor([2.5, 2.5, 2.5, 2.5, .6, .6])
    env._last_effort = torch.full((count, 6), 2.)
    env._last_joint_acceleration = torch.full((count, 6), 3.)
    env._last_policy_velocity = env._last_state.policy_velocity_rad_s - .02
    env._action_transport = SimpleNamespace(critic_steps=lambda: torch.ones(count, 2))
    env._episode_randomization = None
    env._startup_dynamics = SimpleNamespace(mass_scale=torch.ones(count, 1),
                                            base_com_delta_local_m=torch.full((count, 3), .01))
    env._control = SimpleNamespace(controller=SimpleNamespace(nominal=assembler._nominal + .02))
    env._wheel_materials = SimpleNamespace(features=lambda: torch.tensor([[.7, .6, 0., .7, .6, 0.]]).repeat(count, 1))
    env._terrain_scan_77 = lambda: (torch.full((count, 77), .1), sample.ground_at_root_world_z_m)
    env._ground_at_root = torch.zeros(count)
    env._critic_action_history = CriticActionHistory(count, device='cpu')
    env._reset_mask = torch.zeros(count, dtype=torch.bool)
    env._preparation_handoff = torch.zeros_like(env._reset_mask)
    return env, sample


def test_wheel_relative_velocity_is_invariant_to_shared_world_translation():
    env, sample = critic_env()
    # Both wheel COMs move with the base COM: translation is already in 35:38.
    torch.testing.assert_close(env._wheel_velocity_body(), torch.zeros(env.num_envs, 6))
    wheel_indices = env._wheel_body_indices
    env.robot.data.body_lin_vel_w[:, wheel_indices[0]] += torch.tensor([1., 2., 3.])
    env.robot.data.body_lin_vel_w[:, wheel_indices[1]] += torch.tensor([-2., 1., 0.])
    env._root_link_quat = lambda: torch.tensor(
        [[0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]]).repeat(env.num_envs, 1)
    expected = torch.tensor([[2., -1., 3., 1., 2., 0.]]).repeat(env.num_envs, 1)
    torch.testing.assert_close(env._wheel_velocity_body(), expected)
    env.robot.data.body_lin_vel_w += torch.tensor([10., -7., 2.])
    env.robot.data.root_com_vel_w[:, :3] += torch.tensor([10., -7., 2.])
    torch.testing.assert_close(env._wheel_velocity_body(), expected)


def test_critic_uses_physical_ticks_and_parameter_offsets_instead_of_pose_errors():
    env, sample = critic_env()
    before_rng = torch.get_rng_state().clone()
    actual = env._critic_182(env._last_state, sample)
    assert torch.equal(before_rng, torch.get_rng_state())
    torch.testing.assert_close(actual[:, CRITIC182_SLICES['observation_delay_steps']],
                               torch.tensor([[0., 1., 2., 3.]]).repeat(env.num_envs, 1))
    torch.testing.assert_close(actual[:, CRITIC182_SLICES['action_delay_steps']], torch.full((env.num_envs, 2), 2.))
    torch.testing.assert_close(actual[:, CRITIC182_SLICES['com_offset_body_m']], torch.full((env.num_envs, 3), .01))
    torch.testing.assert_close(actual[:, CRITIC182_SLICES['nominal_pose_delta_rad']], torch.full((env.num_envs, 6), .02))
    # Articulation motion changes the clean actor frame, not constant DR parameters.
    moved = replace(env._last_state, policy_position_rad=env._last_state.policy_position_rad + .1)
    changed = env._critic_182(moved, sample)
    assert not torch.equal(actual[:, :35], changed[:, :35])
    torch.testing.assert_close(actual[:, 173:182], changed[:, 173:182])
    # Existing physical scales and already-scaled terrain are preserved exactly.
    torch.testing.assert_close(actual[:, 39:45], env._stiffness[None].expand(env.num_envs, -1))
    torch.testing.assert_close(actual[:, 51:57], env._last_effort)
    torch.testing.assert_close(actual[:, 78:155], torch.full((env.num_envs, 77), .1))


def test_history_contains_second_and_third_previous_issued_actions_and_cuts_epochs():
    history = CriticActionHistory(3, device='cpu')
    owned = torch.ones(3, dtype=torch.bool)
    history.record(torch.full((3, 6), .1), policy_owned=owned)
    history.record(torch.full((3, 6), .2), policy_owned=owned)
    history.record(torch.full((3, 6), .3), policy_owned=owned)
    torch.testing.assert_close(history.older_actions(), torch.tensor([[.2] * 6 + [.1] * 6]).repeat(3, 1))
    # Reset one row and hand off another; a continuing row retains its own history.
    history.clear(torch.tensor([True, True, False]))
    assert history.older_actions()[:2].eq(0).all()
    history.record(torch.full((3, 6), .4), policy_owned=torch.tensor([True, False, True]))
    assert history.older_actions()[:2].eq(0).all()
    torch.testing.assert_close(history.older_actions()[2], torch.tensor([.3] * 6 + [.2] * 6))
    # The script-owned .4 proposal cannot appear after row 1 returns to policy.
    history.record(torch.full((3, 6), .5), policy_owned=owned)
    assert history.older_actions()[1].eq(0).all()


def test_terminal_snapshot_precedes_reset_and_handoff_history_clear():
    env, sample = critic_env()
    for value in (.1, .2, .3):
        env._critic_action_history.record(torch.full((3, 6), value), policy_owned=torch.ones(3, dtype=torch.bool))
    env._capture_terminal()
    terminal = env._terminal_critic_182.clone()
    torch.testing.assert_close(terminal[:, 155:167], torch.tensor([[.2] * 6 + [.1] * 6]).repeat(3, 1))
    # Subtracting float32 joint speeds near 18 rad/s loses a few ulps before
    # division by 10ms; the physical acceleration tolerance is 1e-4 rad/s^2.
    torch.testing.assert_close(terminal[:, 167:173], torch.full((3, 6), 2.), atol=1e-4, rtol=0)
    env._reset_mask[0], env._preparation_handoff[1] = True, True
    returned = env._get_observations()['critic']
    assert returned[:2, 155:167].eq(0).all()
    torch.testing.assert_close(returned[2, 155:167], terminal[2, 155:167])
    assert returned[0, 167:173].eq(0).all()
    torch.testing.assert_close(env._terminal_critic_182, terminal)
    assert not env._reset_mask.any() and not env._preparation_handoff.any()

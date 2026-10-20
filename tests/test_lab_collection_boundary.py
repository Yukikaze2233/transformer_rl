"""Fast LabCollector boundary checks for the 182D -> 185D path."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from transformer_rl.formal.contracts import (LabeledObservation, LabeledStep,
                                             PhysicalLabels, StructuredCollectionConfig)
from transformer_rl.formal.lab_env.adapter import LabLabeledEnv
from transformer_rl.formal.lab_env.collector import LabCollector
from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig


def test_native_step_bootstrap_captures_next_command_before_physical_reset():
    """Exercise the actual step body without creating a simulator application."""
    import ast
    from pathlib import Path
    import time

    path = Path(__file__).resolve().parents[1] / 'src/transformer_rl/formal/lab_env/env.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    step = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == 'step')
    namespace = {'torch': torch, 'time': time}
    exec(compile(ast.Module(body=[step], type_ignores=[]), str(path), 'exec'), namespace)

    command = torch.zeros(3)
    terminal = []
    def observe():
        env._terminated[0] = True
    def capture():
        terminal.append(command.clone())
    def reset(ids):
        command[ids] = 99.
    env = SimpleNamespace(
        _device=torch.device('cpu'), cfg=SimpleNamespace(decimation=1, sim=SimpleNamespace(render_interval=1)),
        sim=SimpleNamespace(is_rendering=False, step=lambda **kwargs: None),
        scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None),
        _pre_physics_step=lambda action: None, _apply_action=lambda: None, _observe_substep=observe,
        _sim_step_counter=0, _physical_tick=0, physics_dt=.005,
        episode_length_buf=torch.tensor([0, 8, 0]), max_episode_length=10, common_step_counter=0,
        _terrain_course=None, _formal_reward=None, _terminated=torch.zeros(3, dtype=torch.bool),
        _reward_policy_step=lambda: command.clone() + 2., _last_reward=torch.zeros(3),
        _last_truncated=torch.zeros(3, dtype=torch.bool), _last_terminated=torch.zeros(3, dtype=torch.bool),
        reset_terminated=torch.zeros(3, dtype=torch.bool), reset_time_outs=torch.zeros(3, dtype=torch.bool),
        _previous_action=torch.zeros(3, 6), _issued_action=torch.full((3, 6), .5),
        _capture_terminal=capture, _update_skill=lambda: command.add_(.03), _reset_idx=reset,
        _get_observations=lambda: command.clone(), extras={}, _reward_terms={})
    observation, reward, terminated, truncated, _ = namespace['step'](env, torch.zeros(3, 6))
    torch.testing.assert_close(reward, torch.full((3,), 2.))
    torch.testing.assert_close(terminal[0], torch.full((3,), .03))
    assert observation[:2].eq(99).all() and observation[2] == terminal[0][2]
    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [False, True, False]
    assert env.extras['termination_counts'].tolist() == [1, 1, 0, 0]


def _labels(timestamp, episode, control):
    return PhysicalLabels(
        velocity_body_m_s=torch.zeros(1, 3),
        wheel_contact=torch.zeros(1, 2),
        joint_velocity_rad_s=torch.zeros(1, 6),
        valid=torch.ones(1, dtype=torch.bool),
        timestamp_s=torch.full((1,), timestamp, dtype=torch.float64),
        episode_id=torch.full((1,), episode, dtype=torch.long),
        control_epoch=torch.full((1,), control, dtype=torch.long),
    )


def _observation(frame, critic, timestamp, episode, control):
    return LabeledObservation(
        frame=frame,
        critic=critic,
        timestamp_s=torch.full((1,), timestamp, dtype=torch.float64),
        episode_id=torch.full((1,), episode, dtype=torch.long),
        control_epoch=torch.full((1,), control, dtype=torch.long),
        policy_allowed=torch.ones(1, dtype=torch.bool),
        labels=_labels(timestamp, episode, control),
    )


class _TerminalEnv(LabLabeledEnv):
    def __init__(self):
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.step_dt = .01
        self.last_timing = {}
        self.last_counts = {}
        self.terrain_stream = None
        self.frame = torch.zeros(1, 35)
        self.critic = torch.zeros(1, 182)
        self.terminal_critic = torch.ones(1, 182)
        self.terminal_frame = torch.full((1, 35), 2.0)
        self.calls = 0

    def validate_collection_config(self, config):
        assert isinstance(config, StructuredCollectionConfig)

    def reset(self, seed=None):
        return _observation(self.frame, self.critic, 0.0, 1, 1)

    def step(self, issued_action, completed_estimate):
        assert issued_action.shape == (1, 6)
        self.calls += 1
        self.last_counts = dict(zip(
            ('true_failure', 'time_limit', 'spatial_exit', 'simultaneous_truncations'),
            torch.tensor([0, 1, 1, 1])))
        next_obs = _observation(self.frame + 1, self.critic + 3, .01, 2, 2)
        return LabeledStep(
            observation=next_obs,
            reward=torch.tensor([2.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.ones(1, dtype=torch.bool),
            final_critic=self.terminal_critic.clone(),
            final_critic_valid=torch.ones(1, dtype=torch.bool),
            final_labels=_labels(.01, 1, 1),
            fully_policy_owned=torch.ones(1, dtype=torch.bool),
            handoff_after=torch.zeros(1, dtype=torch.bool),
            terminal_frame=self.terminal_frame.clone(),
        )


def test_lab_collector_preserves_terminal_reward_and_uses_five_frame_critic():
    model = StructuredActorCritic(StructuredPolicyConfig())
    env = _TerminalEnv()
    collector = LabCollector(env, model)
    collector.reset()
    batch = collector.collect(steps=1)
    assert collector.last_metrics["reward_mean"] == 2.0
    assert batch.critic.shape == (1, 185)
    assert batch.masks.bootstrap.tolist() == [True]
    assert batch.masks.trace.tolist() == [False]
    assert torch.isfinite(batch.advantages).all()
    assert collector.last_metrics["terminated_rows"] == 0
    assert collector.last_metrics["truncated_rows"] == 1
    assert collector.last_metrics['termination_causes'] == {
        'true_failure': 0, 'time_limit': 1, 'spatial_exit': 1, 'simultaneous_truncations': 1}


def test_adapter_excludes_script_hold_and_separates_handoff_from_partial_reset():
    from test_physical_observation import fixture
    _, _, assembler, sample, _ = fixture()

    class FakeLab:
        num_envs, step_dt, device = 2, .01, 'cpu'

        def __init__(self):
            self._observations = SimpleNamespace(assembler=assembler)
            self._recover_task = SimpleNamespace(hold_mask=torch.tensor([True, False]))
            self.obs_buf = {'policy': torch.zeros(2, 175), 'critic': torch.zeros(2, 182)}
            self.sample = sample

        def _sample(self):
            return self.sample

        def _clock(self):
            return self.sample.timestamp_s

        def reset(self):
            pass

        def step(self, action):
            self.sample = replace(sample, timestamp_s=torch.full((2,), .01, dtype=torch.float64))
            self._terminal_sample = self.sample
            self._terminal_state = assembler.measure(self.sample)
            self._terminal_critic_182 = torch.ones(2, 182)
            self._terminal_frame = torch.ones(2, 35)
            self._recover_task.hold_mask[:] = False
            return self.obs_buf, torch.ones(2), torch.tensor([False, False]), torch.tensor([False, True]), {}

    adapter = LabLabeledEnv(FakeLab())
    initial = adapter.reset()
    assert initial.policy_allowed.tolist() == [False, True]
    returned = adapter.step(torch.zeros(2, 6), None)
    assert returned.fully_policy_owned.tolist() == [False, True]
    assert returned.handoff_after.tolist() == [True, False]
    assert returned.final_labels.control_epoch.tolist() == [1, 1]
    assert returned.observation.control_epoch.tolist() == [2, 2]
    assert returned.observation.episode_id.tolist() == [1, 2]


class _ResetAfterSecondActionEnv(_TerminalEnv):
    """One environment that resets after the second action; frames are step ids."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def reset(self, seed=None):
        self.calls = 0
        return _observation(torch.zeros(1, 35), self.critic, 0.0, 1, 1)

    def step(self, issued_action, completed_estimate):
        self.calls += 1
        episode = 1 if self.calls < 2 else 2
        frame = torch.zeros(1, 35)
        frame[:, 0] = float(self.calls)
        next_obs = _observation(frame, self.critic + 3, .01 * self.calls, episode, 1)
        return LabeledStep(
            observation=next_obs,
            reward=torch.ones(1),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            final_critic=self.terminal_critic.clone(),
            final_critic_valid=torch.ones(1, dtype=torch.bool),
            final_labels=_labels(.01 * self.calls, episode, 1),
            fully_policy_owned=torch.ones(1, dtype=torch.bool),
            handoff_after=torch.zeros(1, dtype=torch.bool),
            terminal_frame=frame.clone(),
        )


def test_lab_collector_records_reset_before_the_next_endpoint():
    model = StructuredActorCritic(StructuredPolicyConfig())
    collector = LabCollector(_ResetAfterSecondActionEnv(), model)
    collector.reset()
    batch = collector.collect(steps=3)
    # Endpoint 0: initial repeated frame 0.
    torch.testing.assert_close(batch.windows.gather(torch.tensor([0]))[0, :, :1].flatten(),
                               torch.zeros(5))
    # Endpoint 1: history before reset, ending at frame 1.
    torch.testing.assert_close(batch.windows.gather(torch.tensor([1]))[0, :, :1].flatten(),
                               torch.tensor([0., 0., 0., 0., 1.]))
    # The reset happened after endpoint 1; endpoint 2 starts a fresh window.
    torch.testing.assert_close(batch.windows.gather(torch.tensor([2]))[0, :, :1].flatten(),
                               torch.full((5,), 2.))


def _two_labels(timestamp, episode, control):
    return PhysicalLabels(
        velocity_body_m_s=torch.zeros(2, 3),
        wheel_contact=torch.zeros(2, 2),
        joint_velocity_rad_s=torch.zeros(2, 6),
        valid=torch.ones(2, dtype=torch.bool),
        timestamp_s=torch.full((2,), timestamp, dtype=torch.float64),
        episode_id=torch.tensor(episode, dtype=torch.long),
        control_epoch=torch.tensor(control, dtype=torch.long),
    )


def _two_observation(frame, critic, timestamp, episode, control):
    return LabeledObservation(
        frame=frame,
        critic=critic,
        timestamp_s=torch.full((2,), timestamp, dtype=torch.float64),
        episode_id=torch.tensor(episode, dtype=torch.long),
        control_epoch=torch.tensor(control, dtype=torch.long),
        policy_allowed=torch.ones(2, dtype=torch.bool),
        labels=_two_labels(timestamp, episode, control),
    )


class _PartialResetEnv(_TerminalEnv):
    """Two rows: env 1 truncates at step 1, env 0 terminates at step 2."""

    def __init__(self):
        super().__init__()
        self.num_envs = 2
        self.frame = torch.zeros(2, 35)
        self.critic = torch.zeros(2, 182)
        self.terminal_critic = torch.ones(2, 182)
        self.calls = 0

    def reset(self, seed=None):
        self.calls = 0
        return _two_observation(self.frame, self.critic, 0.0, [1, 1], [1, 1])

    def step(self, issued_action, completed_estimate):
        self.calls += 1
        step = self.calls
        episode = [1, 1]
        control = [1, 1]
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        if step >= 2:
            episode[1] = 2
            control[1] = 2
            truncated[1] = True
        if step >= 3:
            episode[0] = 2
            control[0] = 2
            terminated[0] = True
        frame = torch.zeros(2, 35)
        frame[:, 0] = float(step)
        next_obs = _two_observation(frame, self.critic + step, .01 * step, episode, control)
        return LabeledStep(
            observation=next_obs,
            reward=torch.ones(2),
            terminated=terminated,
            truncated=truncated,
            final_critic=self.terminal_critic.clone(),
            final_critic_valid=torch.ones(2, dtype=torch.bool),
            final_labels=_two_labels(.01 * step, [1, 1], [1, 1]),
            fully_policy_owned=torch.ones(2, dtype=torch.bool),
            handoff_after=torch.zeros(2, dtype=torch.bool),
            terminal_frame=frame.clone(),
        )


def test_partial_reset_and_timeout_record_each_rows_boundary_independently():
    model = StructuredActorCritic(StructuredPolicyConfig())
    collector = LabCollector(_PartialResetEnv(), model)
    collector.reset()
    batch = collector.collect(steps=4)
    # Endpoint 2: env 1 reset after endpoint 1, env 0 did not.
    w2 = batch.windows.gather(torch.tensor([4, 5]))
    assert w2[1, :, 0].tolist() == [2., 2., 2., 2., 2.]
    assert w2[0, :, 0].tolist() == [0., 0., 0., 1., 2.]
    # Endpoint 3: env 0 reset after endpoint 2, env 1 stays in its new episode.
    w3 = batch.windows.gather(torch.tensor([6, 7]))
    assert w3[0, :, 0].tolist() == [3., 3., 3., 3., 3.]
    assert w3[1, :, 0].tolist() == [2., 2., 2., 2., 3.]
    # Behavior statistics must reconstruct exactly from the stored windows.
    endpoints = batch.masks.actor.nonzero().flatten()
    reconstructed = model.actor.policy(batch.windows.gather(endpoints))
    torch.testing.assert_close(reconstructed, batch.old_mean[endpoints])


def test_lab_collector_cross_rollout_reset_starts_a_fresh_window():
    model = StructuredActorCritic(StructuredPolicyConfig())
    collector = LabCollector(_PartialResetEnv(), model)
    first = collector.reset()
    batch1 = collector.collect(steps=4)
    second = collector.reset()
    batch2 = collector.collect(steps=4)
    assert first.episode_id.tolist() == [1, 1]
    assert second.episode_id.tolist() == [1, 1]
    assert batch1.windows.gather(torch.tensor([0]))[0, :, 0].tolist() == [0.] * 5
    assert batch2.windows.gather(torch.tensor([0]))[0, :, 0].tolist() == [0.] * 5

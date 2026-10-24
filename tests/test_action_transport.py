"""Action transport is a policy-step delay, never a motor-response claim."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.lab_env.action_transport import (
    ActionTransportConfig, DelayedActionTransport,
)


def config(**changes):
    value = ActionTransportConfig(enabled=True, delays_steps=(0, 1, 2), nominal_fraction=0.0)
    return replace(value, **changes)


def test_disabled_transport_returns_current_action_and_zero_critic_steps():
    transport = DelayedActionTransport(3, 6, ActionTransportConfig(), seed=7)
    action = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    assert torch.equal(transport.step(action), action)
    assert torch.equal(transport.critic_steps(), torch.zeros(3, 2))


def test_transport_holds_previous_action_and_uses_zero_history_after_reset():
    transport = DelayedActionTransport(1, 2, config(delays_steps=(1,), nominal_fraction=0.), seed=7)
    first = torch.tensor([[1., 2.]])
    second = torch.tensor([[3., 4.]])
    assert torch.equal(transport.step(first), torch.zeros(1, 2))
    assert torch.equal(transport.step(second), first)
    transport.reset(torch.tensor([True]))
    assert torch.equal(transport.step(second), torch.zeros(1, 2))


def test_transport_keeps_per_row_lags_and_only_zero_delay_is_nominal():
    transport = DelayedActionTransport(2, 1, config(delays_steps=(0, 2), nominal_fraction=0.), seed=11)
    # Force row lags without relying on random sampling.
    transport.lags[:] = torch.tensor([0, 2])
    values = [torch.full((2, 1), float(index)) for index in range(4)]
    expected = [torch.zeros(2, 1), torch.tensor([[1.], [0.]]),
                torch.tensor([[2.], [0.]]), torch.tensor([[3.], [1.]])]
    for value, want in zip(values, expected):
        assert torch.equal(transport.step(value), want)
    assert torch.equal(transport.critic_steps(), torch.tensor([[0., 0.], [2., 2.]]))


def test_reset_samples_only_configured_nonnegative_whole_step_delays():
    transport = DelayedActionTransport(100, 1, config(delays_steps=(0, 1, 3), nominal_fraction=.25), seed=5)
    transport.reset(torch.ones(100, dtype=torch.bool))
    assert set(transport.lags.tolist()) <= {0, 1, 3}
    assert transport.reset_rows_consumed == 200


@pytest.mark.parametrize('delays', [(), (-1,), (1, 1), (17,)])
def test_invalid_delay_sets_are_rejected(delays):
    with pytest.raises(ValueError):
        ActionTransportConfig(enabled=True, delays_steps=delays)


def test_transport_rejects_bad_actions_and_masks_atomically():
    transport = DelayedActionTransport(2, 2, config(), seed=1)
    with pytest.raises(ValueError):
        transport.step(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        transport.reset(torch.zeros(2, dtype=torch.float32))
    # A rejected call must leave the delay state unchanged.
    assert transport.history.numel() == transport.capacity * 2 * 2


def test_nominal_fraction_is_actual_zero_delay_probability_not_an_extra_zero_mixture():
    transport = DelayedActionTransport(
        100000, 1, config(delays_steps=(0, 1), nominal_fraction=.5), seed=2102)
    fraction = float((transport.lags == 0).float().mean())
    # Uniformly sampling the full set (0,1) and then forcing nominal zero
    # would produce 75% zero delay, not the declared 50%.
    assert .49 < fraction < .51


def test_cold_restore_preserves_delay_rng_without_replaying_old_action_packets():
    first = DelayedActionTransport(32, 2, config(), seed=2102)
    first.step(torch.full((32, 2), 7.))
    state = first.state_dict()
    cold = DelayedActionTransport(32, 2, config(), seed=9)
    cold.restore_after_physical_reset(state)
    assert cold.history.eq(0).all() and cold.lags.eq(0).all()
    assert torch.equal(cold.generator.get_state(), first.generator.get_state())
    rows = torch.ones(32, dtype=torch.bool)
    first.reset(rows)
    cold.reset(rows)
    assert torch.equal(first.lags, cold.lags)
    assert first.reset_rows_consumed == cold.reset_rows_consumed


def test_policy_handoff_flushes_script_period_actions_before_policy_credit():
    transport = DelayedActionTransport(
        1, 1, config(delays_steps=(2,), nominal_fraction=0.), seed=2102)
    allowed = torch.tensor([True])
    blocked = torch.tensor([False])
    transport.step(torch.tensor([[1.]]), policy_allowed=allowed)
    transport.step(torch.tensor([[2.]]), policy_allowed=allowed)
    # The script interval must not enter the transport history, even though
    # the caller still has a policy proposal for the same interval.
    assert transport.step(torch.tensor([[99.]]), policy_allowed=blocked).item() == 0.
    assert transport.history.eq(0).all()
    # Ownership returns; the queue is neutral, so only the new policy packet
    # can later arrive. Neither old policy packets nor the script action can.
    assert transport.step(torch.tensor([[4.]]), policy_allowed=allowed).item() == 0.
    assert transport.step(torch.tensor([[5.]]), policy_allowed=allowed).item() == 0.
    assert transport.step(torch.tensor([[6.]]), policy_allowed=allowed).item() == 4.
    assert not transport.history.eq(99).any()


def test_script_proposals_cannot_reappear_after_handoff_or_clear_other_rows():
    transport = DelayedActionTransport(
        2, 1, config(delays_steps=(1,), nominal_fraction=0.), seed=2102)
    both = torch.ones(2, dtype=torch.bool)
    transport.step(torch.tensor([[1.], [2.]]), policy_allowed=both)
    rng = transport.generator.get_state().clone()
    applied = transport.step(torch.tensor([[99.], [3.]]),
                             policy_allowed=torch.tensor([False, True]))
    assert applied.tolist() == [[0.], [2.]]
    assert transport.step(torch.tensor([[4.], [5.]]), policy_allowed=both).tolist() == [[0.], [3.]]
    assert transport.step(torch.tensor([[6.], [7.]]), policy_allowed=both).tolist() == [[4.], [5.]]
    assert torch.equal(rng, transport.generator.get_state())


def test_native_action_history_keeps_issuance_separate_from_transport_arrival():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from transformer_rl.formal.control.ownership import ControlSource

    path = Path(__file__).resolve().parents[1] / 'src/transformer_rl/formal/lab_env/env.py'
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    method = next(node for node in definition.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_pre_physics_step')
    namespace = {'torch': torch, 'ControlSource': ControlSource, 'ACTION_DIM': 6}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    arrivals, rewards = [], []

    def decode(action, *_):
        arrivals.append(action.clone())
        return SimpleNamespace(issued_action=action.clone(),
                               control_source=torch.full((1,), int(ControlSource.POLICY)))

    env = SimpleNamespace(
        num_envs=1, _device=torch.device('cpu'), _before_action=torch.zeros(1, 6),
        _previous_action=torch.zeros(1, 6), _issued_action=torch.zeros(1, 6),
        _action_bounds=torch.full((6,), 9.), _joint_positions=lambda: torch.zeros(1, 6),
        _active_indices=torch.arange(6), _control=SimpleNamespace(decode=decode),
        _clock=lambda: torch.zeros(1, dtype=torch.float64), _recover_task=None,
        _recovery_preparation=SimpleNamespace(active=torch.zeros(1, dtype=torch.bool),
                                             targets=lambda target, _: target),
        _action_transport=DelayedActionTransport(
            1, 6, config(delays_steps=(1,), nominal_fraction=0.), seed=2102),
        _formal_reward=SimpleNamespace(composer=SimpleNamespace(
            begin=lambda action, *args, **kwargs: rewards.append(action.clone()))),
        _as_torch=lambda value: value,
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(1, 6))),
        _episode_id=torch.ones(1, dtype=torch.long), _control_epoch=torch.ones(1, dtype=torch.long),
        _command=torch.zeros(1, 4), _command_mode=torch.zeros(1, dtype=torch.long),
        _apply_push=lambda: None, _encoder_position=lambda q: q, _targets=None, _action_latch=None)
    namespace['_pre_physics_step'](env, torch.ones(1, 6))
    namespace['_pre_physics_step'](env, torch.full((1, 6), 2.))
    assert arrivals[0].eq(0).all() and arrivals[1].eq(1).all()
    assert rewards[0].eq(1).all() and rewards[1].eq(2).all()
    assert env._issued_action.eq(2).all()

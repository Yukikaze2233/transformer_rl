"""Preparation references and ownership are independent of policy sampling."""
from dataclasses import fields
import math

import torch

from transformer_rl.formal.control.actuation import OutputFeedbackController, PairTargetDomain
from transformer_rl.formal.control.ownership import ControlSource
from transformer_rl.formal.lab_env.recovery import _RecoveryPreparation


def fixture():
    controller = OutputFeedbackController((0.,) * 6, 3, pair_domain=None)
    preparation = _RecoveryPreparation(controller, policy_dt_s=.01, timeout_s=5.)
    position = torch.zeros(3, 6)
    position[0, :4] = 1.2
    position[1, :4] = .4
    position[2, :4] = 1.3
    clock = torch.zeros(3, dtype=torch.float64)
    preparation.reset_rows(torch.arange(3), position,
                           torch.tensor([True, True, False]), clock)
    return controller, preparation, position, clock


def test_only_recovery_rows_outside_the_original_policy_box_enter_preparation():
    _, preparation, _, _ = fixture()
    assert preparation.active.tolist() == [True, False, False]


def test_preparation_reference_is_rate_bounded_and_actual_PD_keeps_its_caps():
    controller, preparation, position, clock = fixture()
    sampled = torch.full((3, 6), .2)
    policy = controller.decode(sampled, position, clock)
    before = {field.name: getattr(policy, field.name).clone() for field in fields(policy)}
    rng = torch.get_rng_state().clone()
    result = preparation.targets(policy, clock)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(result.admitted_leg_position_rad[0], position[0, :4] - .01,
                               rtol=0, atol=1e-7)
    assert result.control_source.tolist() == [int(ControlSource.PREPARATION_HOLD), 0, 0]
    assert result.issued_action[0, :4].abs().max() > 3
    assert not result.wheel_velocity_rad_s[0].any()
    effort = controller.effort(result, position, torch.zeros_like(position), clock)
    assert effort.capped_effort.abs().le(effort.effort_bounds).all()
    for field in fields(policy):
        assert torch.equal(getattr(policy, field.name), before[field.name])
        assert torch.equal(getattr(result, field.name)[1:], before[field.name][1:])


def test_handoff_requires_measured_target_representability_and_five_intervals():
    controller, preparation, position, clock = fixture()
    measured = position.clone()
    for step in range(60):
        policy = controller.decode(torch.zeros_like(position), measured, clock)
        command = preparation.targets(policy, clock)
        measured[:, :4] = command.admitted_leg_position_rad
        handoff, timeout = preparation.finish_interval(measured)
        assert not timeout.any()
        if handoff[0]:
            assert step + 1 >= 5
            assert controller.reset_hold_action(measured, position_tolerance_rad=1e-6).admitted[0]
            break
        clock += .01
    else:
        raise AssertionError('representable preparation reference never handed back to policy')
    assert not preparation.active.any()
    following = controller.decode(torch.zeros_like(position), measured, clock)
    assert preparation.targets(following, clock) is following


def test_timed_out_preparation_stays_unowned_until_actual_reset():
    controller, _, position, clock = fixture()
    preparation = _RecoveryPreparation(controller, policy_dt_s=.01, timeout_s=.05)
    preparation.reset_rows(torch.arange(3), position, torch.tensor([True, False, False]), clock)
    for step in range(5):
        policy = controller.decode(torch.zeros_like(position), position, clock)
        preparation.targets(policy, clock)
        handoff, timeout = preparation.finish_interval(position)
        assert not handoff.any()
        assert bool(timeout[0]) == (step == 4)
        clock += .01
    assert preparation.active.tolist() == [True, False, False]
    preparation.reset_rows(torch.tensor([0]), position, torch.zeros(3, dtype=torch.bool), clock)
    assert not preparation.active.any()


def test_partial_reset_preserves_other_rows_and_common_full_turns():
    controller, preparation, position, clock = fixture()
    position[0, :4] += 2 * math.pi
    preparation.reset_rows(torch.tensor([0]), position, torch.tensor([True, False, False]), clock)
    policy = controller.decode(torch.zeros_like(position), position, clock)
    command = preparation.targets(policy, clock)
    torch.testing.assert_close(command.admitted_leg_position_rad[0], position[0, :4] - .01,
                               rtol=0, atol=1e-6)
    assert command.common_turns[0].tolist() == [1., 1.]
    elapsed, start = preparation.elapsed[0].clone(), preparation._start[0].clone()
    preparation.reset_rows(torch.tensor([1]), position, torch.zeros(3, dtype=torch.bool), clock)
    assert preparation.active[0]
    assert preparation.elapsed[0] == elapsed
    assert torch.equal(preparation._start[0], start)


def test_preparation_does_not_claim_the_unused_policy_targets_were_projected():
    controller = OutputFeedbackController((0.,) * 6, 1,
                                         pair_domain=PairTargetDomain(((-.1, .1),) * 2, (1, 1)))
    preparation = _RecoveryPreparation(controller, policy_dt_s=.01, timeout_s=5.)
    position = torch.tensor([[1.2, 1.2, 1.2, 1.2, 0., 0.]])
    clock = torch.zeros(1, dtype=torch.float64)
    preparation.reset_rows(torch.tensor([0]), position, torch.ones(1, dtype=torch.bool), clock)
    policy = controller.decode(torch.tensor([[3., -3., 3., -3., 0., 0.]]), position, clock)
    assert policy.pair_projected.all()
    assert not preparation.targets(policy, clock).pair_projected.any()


def test_actual_lab_collector_masks_preparation_and_restarts_handoff_history():
    from dataclasses import replace

    from test_lab_collection_boundary import _TerminalEnv, _labels, _observation
    from transformer_rl.formal.contracts import LabeledStep
    from transformer_rl.formal.lab_env.collector import LabCollector
    from transformer_rl.formal.learning.optimization import StructuredPPOTrainer
    from transformer_rl.formal.policy import StructuredActorCritic, StructuredPolicyConfig

    class HandoffEnv(_TerminalEnv):
        def reset(self, seed=None):
            return replace(super().reset(seed), policy_allowed=torch.zeros(1, dtype=torch.bool))

        def step(self, issued_action, completed_estimate):
            self.calls += 1
            clock = self.calls * .01
            before = 1 if self.calls == 1 else 2
            observation = _observation(self.frame + self.calls, self.critic, clock, 1, 2)
            observation.frame[:, 14:16] = 0
            return LabeledStep(
                observation=observation, reward=torch.tensor([1000. if self.calls == 1 else 1.]),
                terminated=torch.zeros(1, dtype=torch.bool), truncated=torch.zeros(1, dtype=torch.bool),
                final_critic=self.critic.clone(), final_critic_valid=torch.ones(1, dtype=torch.bool),
                final_labels=_labels(clock, 1, before),
                fully_policy_owned=torch.tensor([self.calls > 1]),
                handoff_after=torch.tensor([self.calls == 1]), terminal_frame=observation.frame.clone(),
            )

    model = StructuredActorCritic(StructuredPolicyConfig())
    collector = LabCollector(HandoffEnv(), model)
    collector.reset()
    batch = collector.collect(steps=2)
    assert batch.masks.actor.tolist() == [False, True]
    assert batch.masks.state_auxiliary.tolist() == [False, True]
    assert not batch.masks.trace.any()
    assert batch.advantages[0] == 0 and batch.returns[0] == 0
    history = batch.windows.gather(torch.tensor([1]))
    expected = torch.ones(1, 5, 35)
    expected[:, :, 14:16] = 0
    torch.testing.assert_close(history, expected, rtol=0, atol=0)
    StructuredPPOTrainer(model)._validate_behavior(batch, torch.arange(2))

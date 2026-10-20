"""Command/protocol causality and continuous task-plan checks on Kaiser."""
from dataclasses import replace

import pytest
import torch

from transformer_rl.formal.tasks.trajectory import CommandRates, CommandStream, DriveEnvelope, TrajectoryKind
from transformer_rl.formal.tasks.jump import JumpReferenceProgram
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.tasks.commands import RequestInputs, TaskCommandPlan, TaskCommandProgram
from transformer_rl.formal.tasks.curriculum import ContinuousTaskSampler, TaskFamily
from transformer_rl.formal.tasks.protocol import CompletedEstimate, RequestProtocol, RequestResponse


def fixture(count=120, stage=2):
    protocol = RequestProtocol(count)
    start = torch.tensor([[0., 0., 0., .32]]).repeat(count, 1)
    program = TaskCommandProgram(protocol, CommandStream(start, DriveEnvelope(.06, .4373), CommandRates(2., 3., .2)),
                                  JumpReferenceProgram())
    draw = ContinuousTaskSampler(2101).sample(count, stage)
    plan = TaskCommandPlan.from_draw(draw, start, torch.ones(count), torch.full((count,), 20., dtype=torch.float64),
                                      torch.full((count,), .1, dtype=torch.float64))
    return program, draw, plan


def boundary(program, now, inputs=None):
    n = program.num_envs
    now = torch.full((n,), now, dtype=torch.float64)
    request = program.requests(now, torch.ones(n, dtype=torch.bool), inputs or RequestInputs.empty(n, program.device))
    estimate = CompletedEstimate(torch.zeros(n), torch.ones(n, 2), torch.zeros(n), torch.zeros(n), now - .01,
                                  torch.ones(n, dtype=torch.bool))
    condition = program.protocol.step(now_s=now, estimate=estimate, request_id=request.request_id,
                                        requested_height_m=request.requested_height_m, cancel_id=request.cancel_id,
                                        allow_new=request.allow_new, safety_abort=request.safety_abort)
    program.advance(now, condition)
    return condition, program.commands(now, torch.zeros(n, 6))


@pytest.mark.parametrize('stage', range(3))
def test_all_continuous_stage_draws_map_to_explicit_plans_without_becoming_a_second_policy(stage):
    p, draw, plan = fixture(stage=stage)
    p.reset(torch.ones(p.num_envs, dtype=torch.bool), plan, torch.zeros(p.num_envs, dtype=torch.float64))
    _, commands = boundary(p, .01)
    assert commands.admitted_body_command.shape == (p.num_envs, 4)
    assert commands.previous_issued_action.eq(0).all()
    assert commands.mode[draw.family == int(TaskFamily.RECOVERY)].eq(int(CommandMode.RECOVERY)).all()
    hold = plan.trajectory.kind == int(TrajectoryKind.HOLD)
    torch.testing.assert_close(plan.trajectory.start[hold], plan.trajectory.end[hold])
    nominal = plan.trajectory.end.new_tensor((0., 0., 0., .32))
    torch.testing.assert_close(plan.trajectory.end[draw.nominal], nominal.expand(int(draw.nominal.sum()), -1))
    assert set(draw.family.tolist()).issubset(set(range(6)))


def test_request_height_latch_and_reward_anchor_do_not_follow_later_height_targets():
    p, _, plan = fixture(count=2)
    plan = replace(plan, jump_height_m=torch.full((2,), .1))
    p.reset(torch.ones(2, dtype=torch.bool), plan, torch.zeros(2, dtype=torch.float64))
    for i in range(1, 10):
        boundary(p, i * .01)
    before = p.stream.current[:, 3].clone()
    receipt, cmd = boundary(p, .1)
    assert receipt.response.eq(int(RequestResponse.ACCEPTED)).all()
    assert cmd.mode.eq(int(CommandMode.JUMP)).all()
    torch.testing.assert_close(cmd.admitted_body_command[:, 3], before)
    for i in range(11, 15):
        _, cmd = boundary(p, i * .01)
        torch.testing.assert_close(cmd.admitted_body_command[:, 3], before)
    curve = p.curve(cmd, torch.full((2,), .145, dtype=torch.float64),
                      reward_ground_anchor_world_z_m=torch.full((2,), 2.))
    torch.testing.assert_close(curve.target_apex_world_z_m, before + 2. + .1, atol=3e-7, rtol=0)


def test_deployable_commands_do_not_store_or_require_a_privileged_ground_reference():
    p, _, plan = fixture(count=2)
    p.reset(torch.ones(2, dtype=torch.bool), plan, torch.zeros(2, dtype=torch.float64))
    _, commands = boundary(p, .01)
    before = commands.admitted_body_command.clone()
    p.curve(commands, torch.full((2,), .015, dtype=torch.float64),
               reward_ground_anchor_world_z_m=torch.full((2,), 100.))
    assert not hasattr(p, '_ground_anchor')
    assert torch.equal(before, p.commands(torch.full((2,), .01, dtype=torch.float64), torch.zeros(2, 6)).admitted_body_command)


def test_manual_preparation_request_is_consumed_without_delayed_automatic_replay():
    p, _, plan = fixture(count=2)
    plan = replace(plan, jump_height_m=torch.zeros(2))
    p.reset(torch.ones(2, dtype=torch.bool), plan, torch.zeros(2, dtype=torch.float64))
    incoming = replace(RequestInputs.empty(2, p.device), request_id=torch.tensor([8, 9]),
                        requested_height_m=torch.full((2,), .1), allow_new=torch.zeros(2, dtype=torch.bool))
    result, _ = boundary(p, .01, incoming)
    assert result.response.eq(int(RequestResponse.NOT_READY)).all()
    again = replace(incoming, allow_new=torch.ones(2, dtype=torch.bool))
    result, cmd = boundary(p, .02, again)
    assert result.response.eq(int(RequestResponse.REPLAY)).all() and not cmd.mode.eq(int(CommandMode.JUMP)).any()


@pytest.mark.parametrize('bad', ['jump_mode', 'partial_horizon', 'rise', 'wrong_dtype'])
def test_bad_plan_is_rejected_before_mutating_command_or_protocol_state(bad):
    p, _, plan = fixture(count=2)
    if bad == 'jump_mode':plan = replace(plan, base_mode=torch.full((2,), int(CommandMode.JUMP)))
    elif bad == 'partial_horizon':plan = replace(plan, task_horizon_s=torch.full((2,), .015, dtype=torch.float64))
    elif bad == 'rise':plan = replace(plan, jump_height_m=torch.full((2,), .31))
    else:plan = replace(plan, task_horizon_s=torch.ones(2))
    before = p.stream.current.clone()
    with pytest.raises(ValueError):
        p.reset(torch.ones(2, dtype=torch.bool), plan, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(before, p.stream.current) and p.protocol.max_seen_id.eq(-1).all()

"""Continuous reset composition with source geometry, not native acceptance."""
from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from test_joint_boundaries import encode
from test_recovery_poses import tuning as recovery_tuning
from test_reset_collision import fixture as collision_fixture
from test_reset_kinematics import rotation
from transformer_rl._native_episode import TerrainRebuildRequired
from transformer_rl._output_actuation import OutputFeedbackController
from transformer_rl._physical_observation import MechanicalBinding
from transformer_rl._recovery_poses import RecoveryPoseCompiler
from transformer_rl._reset_collision import AssetCollisionGeometry
from transformer_rl._reset_kinematics import AssetResetKinematics
from transformer_rl._reset_program import ResetProgramTuning, TaskResetPreparer
from transformer_rl._task_curriculum import ContinuousTaskSampler, TaskDraw, TaskFamily
from transformer_rl._terrain_geometry import TerrainKind
from transformer_rl._terrain_tasks import ContinuousTerrainGenerator, TerrainTuning


def fixture(count=2):
    # The synthetic constraints are calibrated at zero. Keep their source
    # nominal consistent instead of relaxing closure to admit a test pose.
    original, raw, spec_raw, mesh = collision_fixture()
    import json

    manifest, spec = json.loads(raw), json.loads(spec_raw)
    manifest['nominal_joint_pos'] = {k: 0. for k in manifest['nominal_joint_pos']}
    spec['nominal_joint_pos'] = dict(manifest['nominal_joint_pos'])
    binding = MechanicalBinding.from_manifest(manifest, original.kinematics.binding.native_joint_names,
                                                original.kinematics.binding.native_body_names)
    raw, spec_raw = encode(manifest, spec)
    kin = AssetResetKinematics(raw, spec_raw, binding)
    geometry = AssetCollisionGeometry(kin, raw, spec_raw, {'collisions/body.obj': mesh})
    tasks = ContinuousTaskSampler(2101)
    terrain = ContinuousTerrainGenerator(2102, TerrainTuning(.8, (.2, 1.1), 1.))
    # Explicit short CPU-interface horizons. Formal30s/60s remains unchanged.
    tuning = ResetProgramTuning(1., 2., (.2, .8), (.2, .5), 2., (.8, .8),
                                .01, .001, .001, .001, 0., .02, .001)
    program = TaskResetPreparer(tasks, terrain, RecoveryPoseCompiler(kin, recovery_tuning()), geometry,
                                tuning, pose_seed=2103)
    controller = OutputFeedbackController(binding.nominal_policy_position_rad, count, pair_domain=None)
    origins = torch.zeros(count, 3, dtype=torch.float64)
    return program, controller, origins


def draw(family, *, posture=None, nominal=None, terrain=None):
    n = len(family)
    long = lambda value: torch.tensor(value, dtype=torch.long)
    scalar = lambda value: torch.full((n,), value)
    command = torch.tensor([[1., 0., .2, .32]]).repeat(n, 1)
    return TaskDraw(long(family), torch.zeros(n, dtype=torch.long), command, scalar(.1), scalar(.1), scalar(.012),
                    scalar(.12), scalar(.1), long([0] * n if terrain is None else terrain),
                    long([0] * n if posture is None else posture), torch.zeros(n, dtype=torch.long),
                    torch.tensor([False] * n if nominal is None else nominal))


def test_six_families_keep_full_requested_draw_and_complete_courses_without_failed_pose_replacement():
    program, controller, origins = fixture(7)
    requested = draw([0, 1, 2, 3, 4, 5, 5], posture=[0, 0, 0, 0, 0, 4, 14],
                       nominal=[False, False, False, True, False, False, False],
                       terrain=[0, 0, 0, int(TerrainKind.ROUGH), 0, 0, 0])
    value = program.prepare_draw(requested, origins, controller)
    for f in fields(requested):
        assert torch.equal(getattr(value.draw, f.name), getattr(requested, f.name))
    assert value.commands.task_horizon_s.tolist() == [1., 1., 1., 1., 2., 2., 2.]
    assert [c.request.extent.x_m for c in value.courses] == [(-7., 7.)] * 4 + [(-12., 12.)] * 3
    assert value.courses[3].request.kind == TerrainKind.ROUGH
    assert any(abs(z) > 0 for row in value.courses[3].surface.patches[0].z_m for z in row)
    assert value.commands.trajectory.end[3].tolist() == pytest.approx([0., 0., 0., .305])
    assert value.recovery.requested_posture_id.tolist() == [0, 0, 0, 0, 0, 4, 14]
    assert not value.source_ready[-1] and value.recovery.required_preparation_action[-1].abs().max() > 3
    assert value.recovery.requested_leg_forward_pitch_rad[-1].min() > 3
    assert program.requested_rows == 7 and program.attempted_batches == program.completed_batches == 1
    assert not value.physical_qualification


def test_sampler_geometry_pose_rng_and_fair_quotas_resume_without_touching_global_rng():
    first, controller, origins = fixture(8)
    first.sample(8, 5, origins, controller)
    state = deepcopy(first.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    expected = first.sample(8, 5, origins, controller)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    second, other_controller, other_origins = fixture(8)
    second.load_state_dict(state)
    actual = second.sample(8, 5, other_origins, other_controller)
    assert actual.patch.course_sha256 == expected.patch.course_sha256
    for f in fields(actual.draw):
        assert torch.equal(getattr(actual.draw, f.name), getattr(expected.draw, f.name))
    assert torch.equal(actual.placement.root_pose_xyzw, expected.placement.root_pose_xyzw)
    assert torch.equal(actual.recovery.closure.tree_position, expected.recovery.closure.tree_position)
    assert torch.equal(actual.commands.request_interval_s, expected.commands.request_interval_s)
    assert second.requested_rows == first.requested_rows == 16
    bad = deepcopy(state)
    bad['terrain']['tuning']['entry_m'] += .1
    before = deepcopy(second.state_dict())
    with pytest.raises(ValueError, match='tuning'):
        second.load_state_dict(bad)
    assert torch.equal(second.generator.get_state(), before['pose_rng'])
    assert second.requested_rows == before['requested_rows']


def test_native_binding_rechecks_encoded_holds_and_converts_link_origin_to_actual_com_velocity():
    program, controller, origins = fixture()
    program.tuning = replace(program.tuning, xy_jitter_m=0, roll_pitch_jitter_rad=0,
                              inner_knee_jitter_rad=0, leg_pitch_jitter_rad=0)
    value = program.prepare_draw(draw([0, 0]), origins, controller)
    assert value.source_ready.all()
    velocity = torch.tensor([[.2, -.3, -.4], [.1, .2, .3]], dtype=torch.float64)
    omega = torch.tensor([[0., 0., 2.], [1., 2., 3.]], dtype=torch.float64)
    value = replace(value, recovery=replace(value.recovery, origin_velocity_body_m_s=velocity,
                                            angular_velocity_body_rad_s=omega))
    com = torch.tensor([[.1, .2, .3], [-.3, .2, .1]], dtype=torch.float64)
    plan, placement = value.native_plan(program.geometry, controller,
                                        actual_course_sha256=value.patch.course_sha256, actual_base_com_body_m=com)
    orient = plan.root_pose_xyzw[:, 3:].double()
    orient /= orient.norm(dim=-1, keepdim=True)
    r = rotation(orient)
    linear = velocity + torch.linalg.cross(omega, com)
    expected = torch.cat(((r @ linear[..., None]).squeeze(-1), (r @ omega[..., None]).squeeze(-1)), -1).float()
    torch.testing.assert_close(plan.root_com_velocity, expected, atol=1e-7, rtol=0)
    q = plan.joint_position[:, list(program.geometry.kinematics.binding.policy_joint_indices)]
    decoded = controller.decode(plan.preparation_issued_action, q, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(decoded.admitted_leg_position_rad, q[:, :4], atol=1e-6, rtol=0)
    assert not decoded.wheel_velocity_rad_s.any() and not placement.physical_qualification
    with pytest.raises(TerrainRebuildRequired) as changed:
        value.native_plan(program.geometry, controller, actual_course_sha256=('a' * 64, 'b' * 64),
                           actual_base_com_body_m=com)
    assert changed.value.rows == (0, 1)
    value.recovery.closure.tree_position[0, 0] += .01
    with pytest.raises(ValueError, match='changed'):
        value.native_plan(program.geometry, controller, actual_course_sha256=value.patch.course_sha256,
                           actual_base_com_body_m=com)


def test_failed_preparation_retains_sampler_progress_and_attempt_accounting_without_retry():
    program, controller, origins = fixture()
    before = deepcopy(program.tasks.state_dict())

    def fail(*args):
        raise RuntimeError('explicit synthetic scene resource rejection')

    program.terrain.from_draw = fail
    with pytest.raises(RuntimeError, match='resource rejection'):
        program.sample(2, 5, origins, controller)
    assert not torch.equal(program.tasks.generator.get_state(), before['rng'])
    assert program.attempted_batches == 1 and program.requested_rows == 2 and program.completed_batches == 0


def test_release_world_down_is_not_a_tilted_body_z_velocity():
    program, controller, origins = fixture()
    value = program.recovery.compile(torch.tensor([2, 3]), torch.tensor([.5, -.3], dtype=torch.float64), controller,
                                      roll_pitch_offset_rad=torch.tensor([[.2, -.1], [-.1, .2]], dtype=torch.float64))
    world = (rotation(value.requested_root_orientation_xyzw) @ value.origin_velocity_body_m_s[..., None]).squeeze(-1)
    torch.testing.assert_close(world, torch.tensor([[0., 0., -.15]] * 2, dtype=torch.float64), atol=1e-15, rtol=0)


def test_source_command_trajectory_mutation_is_part_of_the_preparation_numeric_guard():
    program, controller, origins = fixture()
    value = program.prepare_draw(draw([0, 0]), origins, controller)
    value.commands.trajectory.end[0, 0] += .1
    with pytest.raises(ValueError, match='numeric inputs changed'):
        value.encode_plan(program.geometry, controller, base_com_body_m=torch.zeros(2, 3, dtype=torch.float64))


@pytest.mark.parametrize('fault', ['unknown_family', 'invalid_height', 'nonfinite', 'dtype', 'wrong_nominal'])
def test_invalid_source_configuration_is_rejected_before_sampling_geometry_or_advancing_pose_rng(fault):
    program, controller, origins = fixture()
    value = draw([0, 0])
    if fault == 'unknown_family':
        value.family[0] = 6
    elif fault == 'invalid_height':
        value.requested_command[0, 3] = .1
    elif fault == 'nonfinite':
        value.roughness_m[0] = float('nan')
    elif fault == 'dtype':
        value = replace(value, nominal=value.nominal.long())
    else:
        controller.nominal[0] += .1
    before = program.state_dict()
    with pytest.raises(ValueError):
        program.prepare_draw(value, origins, controller)
    assert program.attempted_batches == program.requested_rows == 0
    assert torch.equal(program.generator.get_state(), before['pose_rng'])
    assert torch.equal(program.terrain.generator.get_state(), before['terrain']['rng'])

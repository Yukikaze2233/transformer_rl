"""SCUT birth randomization inside the selected asset's travel (CPU only)."""
import math
from pathlib import Path

import pytest
import torch

from transformer_rl.formal.lab_env.birth import BirthRandomization, BirthRandomizationConfig
from transformer_rl.formal.tasks.geometry import TerrainKind

ASSET = Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/reports/v7_training_review_20261011/'
             'mechanical_stop_candidates_20261011/stop107_reference_v2/asset')


def test_configuration_is_frozen_and_tilt_quaternion_recovers_roll_pitch():
    assert BirthRandomizationConfig.from_dict({'enabled': True}).ground_probability == .5
    with pytest.raises(ValueError):
        BirthRandomizationConfig.from_dict({'enabled': True, 'roll_pitch_rad': .3})
    roll_pitch = torch.tensor([[.15, -.1], [-.05, .12]])
    x, y, z, w = BirthRandomization.tilt_quaternion(roll_pitch).unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    torch.testing.assert_close(torch.stack((roll, pitch), -1), roll_pitch)


def test_ground_births_need_a_flat_spawn_patch():
    kinds = torch.tensor([int(kind) for kind in TerrainKind])
    eligible = BirthRandomization.ground_eligible(kinds)
    expected = {TerrainKind.FLAT, TerrainKind.ROUGH, TerrainKind.MATERIAL}
    assert eligible.tolist() == [kind in expected for kind in TerrainKind]


@pytest.fixture(scope='module')
def selected():
    if not ASSET.is_dir():
        pytest.skip('selected 107-degree asset is not present')
    from transformer_rl.formal.lab_env.recovery import LabRecoveryReset
    from transformer_rl.formal.lab_env.settings import load_asset_description
    from transformer_rl.formal.simulation.ground import WheelCollisionCentres

    asset = load_asset_description(ASSET)
    recovery = LabRecoveryReset(asset, 16)
    wheels = WheelCollisionCentres.from_urdf((asset.directory / asset.manifest['urdf']).read_bytes(), asset.binding)
    birth = BirthRandomization(recovery, BirthRandomizationConfig(enabled=True, library_size=512), wheels,
                               seed=2102, device='cpu')
    return recovery, birth


def test_library_postures_stay_inside_the_commandable_travel(selected):
    from transformer_rl.formal.control.actuation import OutputFeedbackController

    recovery, birth = selected
    kinematics = recovery.kinematics
    binding = kinematics.binding
    assert birth.library.receipt['certified'] >= 512
    tree_order = [binding.native_joint_names.index(name) for name in binding.tree_joint_names]
    tree = birth.joint_position[:, tree_order].double()
    assert kinematics.closure(tree).norm(dim=-1).amax() < 2e-6
    # Every posture is one the policy decoder can hold (no scripted preparation needed).
    policy = torch.zeros(len(tree), 6)
    policy[:, :4] = tree[:, list(kinematics.policy_tree_indices[:4])].float()
    controller = OutputFeedbackController(binding.nominal_policy_position_rad, len(tree), pair_domain=None)
    assert controller.reset_hold_action(policy, position_tolerance_rad=1e-6).admitted.all()
    # Lengths span the knee stops; heights span the measured stance travel 0.22..0.43 m.
    length = birth.library.leg_length_m
    assert length.min() < .17 and length.max() > .36
    assert birth.ground_root_height_m.min() > .21 and birth.landing_root_height_m < .44
    torch.testing.assert_close(torch.tensor(birth.landing_root_height_m), birth.ground_root_height_m.max())
    # Both swing directions are covered, and both sides are drawn independently.
    angle = birth.library.leg_angle_rad
    assert (angle > math.radians(25)).any() and (angle < math.radians(-25)).any()
    assert abs(float(torch.corrcoef(angle.T)[0, 1])) < .2


def test_wheel_centres_match_the_placed_stance(selected):
    _, birth = selected
    # Standing on the ground: the lowest wheel bottom sits at the placement clearance.
    bottom = birth.ground_root_height_m + birth.wheel_centre_root_m[..., 2].amin(-1) - .06
    assert bottom.abs().max() < .01
    assert (birth.wheel_centre_root_m[:, 0, 1] > 0).all() and (birth.wheel_centre_root_m[:, 1, 1] < 0).all()


def test_draws_split_ground_and_landing_births(selected):
    _, birth = selected
    ordinary = torch.tensor([True] * 3000 + [False] * 1000)
    eligible = torch.tensor([True] * 2000 + [False] * 2000)
    before = birth.counts.clone()
    ground, index, tilt, wheels = birth.draw(ordinary, eligible)
    assert not ground[2000:].any() and .45 < ground[:2000].float().mean() < .55
    assert birth.counts[0] - before[0] == int((ordinary & ~ground).sum())
    assert tilt.abs().le(.15).all() and wheels.abs().le(50).all() and index.lt(512).all()
    state = birth.state_dict()
    again = birth.draw(ordinary, eligible)
    birth.restore_after_physical_reset(state)
    torch.testing.assert_close(birth.draw(ordinary, eligible)[1], again[1])


def test_stand_up_telemetry_counts_ground_births(selected):
    _, birth = selected
    rows = torch.tensor([0, 1])
    birth.begin(rows)
    base = birth.counts.clone()
    height, command = torch.full((16,), .30), torch.full((16,), .30)
    loaded = torch.full((16, 2), 40.)
    tilt = torch.zeros(16)
    terminated = torch.zeros(16, dtype=torch.bool)
    terminated[1] = True
    height[0] = .1  # still down
    reasons = torch.zeros(16, dtype=torch.long)
    reasons[1] = 2 | 4  # excessive tilt and forbidden body contact
    birth.observe(.02, height_m=height, height_command_m=command, tilt_rad=tilt, wheel_load_n=loaded,
                  terminated=terminated, failure_reasons=reasons)
    assert birth.counts[3] - base[3] == 1
    bits = birth.metrics()['ground_failure_bits']
    assert bits['excessive_tilt'] >= 1 and bits['forbidden_body_contact'] >= 1 and bits['low_support_height'] == 0
    height[0] = .29
    for _ in range(25):
        birth.observe(.02, height_m=height, height_command_m=command, tilt_rad=tilt, wheel_load_n=loaded,
                      terminated=torch.zeros(16, dtype=torch.bool))
    assert birth.counts[2] - base[2] == 1


def test_reset_rows_get_ground_or_landing_postures_and_recovery_rows_keep_theirs(selected):
    import ast
    from types import SimpleNamespace
    from transformer_rl.formal.control.observation import CommandMode
    from transformer_rl.formal.lab_env import env as env_module

    recovery, birth = selected
    path = Path(env_module.__file__)
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'WheeledLabEnv')
    method = next(node for node in definition.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_randomize_births')
    namespace = {'torch': torch, 'CommandMode': CommandMode, '_quat_mul': env_module._quat_mul}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    binding = recovery.kinematics.binding
    env = SimpleNamespace(_birth_randomization=birth, _terrain_course=None, _device=torch.device('cpu'),
                          _task=SimpleNamespace(nominal_height_m=.32), _recovery_reset=recovery, _binding=binding)
    count = 400
    mode = torch.full((count,), int(CommandMode.NORMAL))
    mode[:40] = int(CommandMode.RECOVERY)
    terrain = torch.full((count,), int(TerrainKind.FLAT))
    terrain[200:] = int(TerrainKind.STAIRS)
    task = SimpleNamespace(mode=mode, terrain_kind=terrain)
    ids = torch.arange(count) + 7
    root = torch.zeros(count, 7)
    root[:, 2], root[:, 6] = 1. + .32, 1.  # local ground at z = 1
    joints, velocity = torch.full((count, 18), 9.), torch.zeros(count, 18)
    ground = namespace['_randomize_births'](env, ids, task, root, joints, velocity)
    landing = ~ground
    landing[:40] = False
    assert not ground[:40].any() and not ground[200:].any() and .3 < ground[40:200].float().mean() < .7
    assert joints[:40].eq(9.).all() and root[:40, 2].eq(1.32).all() and root[:40, 3:6].eq(0).all()
    assert env._pending_ground_rows.tolist() == ids[ground].tolist()
    # Ground: level, standing on the ground at the posture height. Landing: top of travel, tilted.
    assert root[ground, 3:6].eq(0).all() and root[ground, 2].ge(1 + .21).all()
    torch.testing.assert_close(root[landing, 2], torch.full((int(landing.sum()),), 1 + birth.landing_root_height_m + .035))
    assert root[landing, 3:5].abs().gt(0).any(-1).all()
    assert velocity[40:].abs().max() <= 50 and velocity[40:].ne(0).any()
    wheels = [binding.native_joint_names.index(name) for name in ('L_joint3', 'R_joint3')]
    others = [i for i in range(18) if i not in wheels]
    assert velocity[:, others].eq(0).all() and velocity[:40].eq(0).all()

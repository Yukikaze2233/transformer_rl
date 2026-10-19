"""Native reward-source binding with synthetic buffers, never a real SDK."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_control_reward import reference
from test_joint_boundaries import asset_sources, encode, forward_poses
from test_native_plant import fixture as plant_fixture
from test_structured_collection import model as small_model
from transformer_rl.formal.tasks.reward import RewardPhase
from transformer_rl.formal.control.joints import LegBoundaryGeometry
from transformer_rl.formal.simulation.reward import NativeRewardBridge
from transformer_rl.formal.control.observation import CommandMode
from transformer_rl.formal.learning.collection import StructuredCollectionConfig, StructuredCollector


def fixture(period=.005, *, native_plant=None, initialize=True):
    plant, robot, sim, trace = plant_fixture(period) if native_plant is None else native_plant
    binding = plant.scene.observation.binding
    manifest, spec = asset_sources(binding)
    raw, model = encode(manifest, spec)
    geometry = LegBoundaryGeometry.from_asset_sources(raw, model, binding)
    plant.scene.readback.update(selected_asset_manifest_sha256=geometry.manifest_sha256,
                                selected_model_spec_sha256=geometry.model_spec_sha256)
    masses = torch.tensor([20., 10.])[:, None].repeat(1, 19) / 19
    limits = torch.tensor([-float('inf'), float('inf')])[None, None].repeat(2, 18, 1)
    for side in geometry.sides:
        limits[:, binding.native_joint_names.index(side.knee_joint)] = torch.tensor(side.native_limits_rad)
    reads = {'mass': 0, 'limits': 0, 'gravity': 0}

    def mass_read():
        reads['mass'] += 1
        return masses

    def limit_read():
        reads['limits'] += 1
        return limits

    def gravity_read():
        reads['gravity'] += 1
        return (0., 0., -9.81)

    robot.root_view.get_masses, robot.root_view.get_dof_limits = mass_read, limit_read
    plant.scene.contacts.physics_view.get_gravity = gravity_read
    sim.cfg.gravity = (0., 0., -9.81)
    old_step = sim.step

    def step(*, render):
        old_step(render=render)
        tree = robot.data.joint_pos.torch[:, list(binding.tree_joint_indices)]
        robot.data.body_link_pose_w.torch.copy_(forward_poses(spec, binding, tree, robot.data.root_link_pose_w.torch))

    sim.step = step
    config = StructuredCollectionConfig()
    reward = NativeRewardBridge(plant, geometry, config, knee_margin_rad=.05,
                                 retired_request_ids=torch.full((2,), -1, dtype=torch.long))
    if initialize:
        reward.composer.reset(torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.float64),
                                torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    return reward, plant, robot, sim, masses, limits, reads, trace


def begin(reward, ref=None, owned=None):
    ref = reference() if ref is None else ref
    owned = torch.ones(2, dtype=torch.bool) if owned is None else owned
    issued = torch.zeros(2, 6)
    plant, composer = reward.plant, reward.composer
    plant.accept_issued_action(issued, control_source=torch.where(owned, 0, 1))
    velocity = plant.scene.robot.data.joint_vel.torch[:, list(plant.scene.observation.binding.policy_joint_indices)]
    composer.begin(issued, velocity, now_s=plant.scene.observation.timestamp_s(), episode_id=composer._episode.clone(),
                     control_epoch=composer._control.clone(), policy_owned=owned,
                     admitted_body_command=ref.admitted_body_command, command_mode=ref.mode)
    return ref, owned


def interval(reward, ref=None, owned=None):
    ref, owned = begin(reward, ref, owned)
    samples = []
    for _ in range(reward.composer.substeps):
        tick = reward.plant.step_physics()
        samples.append(reward.observe(tick, ref, episode_id=reward.composer._episode,
                                       control_epoch=reward.composer._control, policy_owned=owned))
    result = reward.composer.finish(terminated=torch.zeros(2, dtype=torch.bool), truncated=torch.zeros(2, dtype=torch.bool),
                                     physical_success_request_id=torch.full((2,), -1, dtype=torch.long),
                                     entire_skill_policy_owned=owned)
    return result, samples


@pytest.mark.parametrize('period', [.005, .001])
def test_actual_tick_sources_feed_reward_without_zero_slip_mass_defaults_or_extra_physics(period):
    reward, plant, robot, sim, _, _, reads, _ = fixture(period)
    before = torch.get_rng_state().clone()
    result, samples = interval(reward)
    assert len(samples) == sim.steps == reward.composer.substeps
    assert reads == {'mass': 1, 'limits': 1, 'gravity': 1}
    assert reward.composer.tuning.gamma == reward.collection_config.gamma
    torch.testing.assert_close(reward.composer._mass_kg, torch.tensor([20., 10.]))
    assert reward.composer.tuning.robot_mass_kg == 13.886  # Declared nominal,not used as both environments' live mass.
    assert torch.equal(before, torch.get_rng_state())
    for sample in samples:
        assert sample.physics.boundary_reference == 'geometric_passive_knees_then_four_output_targets'
        assert sample.physics.wheel_contact_tangent_speed_m_s.gt(0).all()
        assert not sample.boundary.shortfall_rad[:, :2].any()
        torch.testing.assert_close(sample.physics.velocity_origin_world_m_s, torch.tensor([[.4, -.3, .2]]).repeat(2, 1))
        assert not sample.output_pair_admission_enabled  # Fixture is an explicitly incomplete control probe.
    assert result.timestamp_s.eq(.01).all() and torch.isfinite(result.learning_reward).all()
    robot.data.body_com_vel_w.torch.zero_()
    assert samples[-1].physics.wheel_contact_tangent_speed_m_s.gt(0).all()


def test_landing_impact_uses_each_native_environment_mass_not_one_nominal_or_test_mass():
    reward, plant, *_ = fixture()
    for group in plant.scene.contacts.groups:
        force, _, _, _, counts, _ = group.view.get_contact_data(dt=.005)
        force[:int(counts.sum())] = 300.
    result, _ = interval(reward, reference(CommandMode.JUMP, RewardPhase.LANDING))
    impact = result.raw_cost_densities['landing_impact']
    assert impact[0] > 0 and impact[1] > 10 * impact[0]


def test_mass_snapshot_is_owned_and_refresh_requires_new_final_epoch_at_closed_policy_boundary():
    reward, plant, _, _, masses, _, _, _ = fixture()
    interval(reward)
    previous = reward.mass_kg_by_environment
    exposed = reward.mass_kg_by_environment
    exposed.zero_()
    torch.testing.assert_close(reward.mass_kg_by_environment, previous)
    masses[0] *= 2
    torch.testing.assert_close(reward.composer._mass_kg, previous)
    with pytest.raises(ValueError, match='new final'):
        reward.refresh_model()
    plant.scene.observation.invalidate_model()
    plant.scene.observation.refresh_model()
    reward.refresh_model()
    torch.testing.assert_close(reward.mass_kg_by_environment, torch.tensor([40., 10.]))
    torch.testing.assert_close(reward.composer._mass_kg, torch.tensor([40., 10.]))
    assert reward.composer._action_count.eq(1).all()  # Model refresh did not reset learning/issued history.


def test_open_interval_rejects_mass_changes_before_readback_and_stale_packet_never_replays():
    reward, plant, _, _, _, _, reads, _ = fixture()
    ref, owned = begin(reward)
    snapshot = dict(reads)
    with pytest.raises(RuntimeError, match='policy boundary'):
        reward.refresh_model()
    assert reads == snapshot
    tick = plant.step_physics()
    reward.observe(tick, ref, episode_id=reward.composer._episode, control_epoch=reward.composer._control, policy_owned=owned)
    with pytest.raises(ValueError, match='next actual'):
        reward.observe(tick, ref, episode_id=reward.composer._episode, control_epoch=reward.composer._control, policy_owned=owned)


@pytest.mark.parametrize('fault', ['manifest', 'model_spec', 'mass_nan', 'mass_zero', 'mass_shape',
                                  'native_limit', 'gravity_missing', 'gravity_tilt', 'gravity_recipe'])
def test_invalid_or_unbound_model_snapshot_blocks_native_reward(fault):
    reward, plant, robot, sim, masses, limits, reads, _ = fixture()
    if fault == 'manifest':plant.scene.readback['selected_asset_manifest_sha256'] = '0'*64
    elif fault == 'model_spec':plant.scene.readback['selected_model_spec_sha256'] = '0'*64
    elif fault == 'mass_nan':masses[0, 0] = float('nan')
    elif fault == 'mass_zero':masses[0, 0] = 0.
    elif fault == 'mass_shape':robot.root_view.get_masses = lambda: torch.ones(2, 18)
    elif fault == 'native_limit':limits[:, plant.scene.observation.binding.native_joint_names.index(reward.geometry.sides[0].knee_joint), 1] += .01
    elif fault == 'gravity_missing':plant.scene.contacts.physics_view.get_gravity = lambda: None
    elif fault == 'gravity_tilt':plant.scene.contacts.physics_view.get_gravity = lambda: (.1, 0., -9.81)
    else:sim.cfg.gravity = (0., 0., -10.)
    before = reads['mass']
    with pytest.raises(ValueError):
        NativeRewardBridge(plant, reward.geometry, reward.collection_config, knee_margin_rad=.05,
                             retired_request_ids=torch.full((2,), -1, dtype=torch.long))
    if fault in ('manifest', 'model_spec'):assert reads['mass'] == before
    assert sim.steps == 0


def test_failed_model_refresh_does_not_leave_old_mass_as_a_valid_current_model():
    reward, plant, _, _, _, limits, _, _ = fixture()
    plant.scene.observation.invalidate_model()
    plant.scene.observation.refresh_model()
    index = plant.scene.observation.binding.native_joint_names.index(reward.geometry.sides[0].knee_joint)
    limits[:, index, 1] += .01
    with pytest.raises(ValueError, match='passive knee limits'):
        reward.refresh_model()
    assert not reward._ready


def test_collector_rejects_a_different_discount_before_accessing_the_model():
    reward, *_ = fixture()
    env = SimpleNamespace(num_envs=2, device='cpu', validate_collection_config=reward.validate_collection_config)
    with pytest.raises(ValueError, match='same100Hz discount'):
        StructuredCollector(env, None, replace(reward.collection_config, gamma=.9))
    collector = StructuredCollector(env, small_model(), reward.collection_config)
    assert collector.config is reward.collection_config


def test_replaced_tick_or_different_episode_is_rejected_without_consuming_the_real_sample():
    reward, plant, *_ = fixture()
    ref, owned = begin(reward)
    tick = plant.step_physics()
    kwargs = dict(episode_id=reward.composer._episode, control_epoch=reward.composer._control, policy_owned=owned)
    with pytest.raises(ValueError, match='next actual'):
        reward.sample(replace(tick), **kwargs)
    with pytest.raises(ValueError, match='episode, control'):
        reward.sample(tick, **{**kwargs, 'episode_id': kwargs['episode_id'] + 1})
    assert reward._last_tick == 0 and reward.composer._window['steps'] == 0
    reward.observe(tick, ref, **kwargs)
    assert reward._last_tick == reward.composer._window['steps'] == 1


def test_model_readback_clones_mass_before_another_native_getter_reuses_its_allocation():
    reward, plant, robot, _, masses, limits, _, _ = fixture()
    plant.scene.observation.invalidate_model()
    plant.scene.observation.refresh_model()

    def reused_limits():
        masses.fill_(5.)
        return limits

    robot.root_view.get_dof_limits = reused_limits
    reward.refresh_model()
    torch.testing.assert_close(reward.mass_kg_by_environment, torch.tensor([20., 10.]))


def test_script_interval_keeps_physical_sources_but_receives_no_learning_credit():
    reward, *_ = fixture()
    result, samples = interval(reward, owned=torch.tensor([False, True]))
    assert result.learning_reward[0] == 0
    assert samples[-1].physics.wheel_contact_tangent_speed_m_s[0].gt(0).all()
    assert result.raw_cost_densities['contact_slip'][0] > 0


def test_live_selected_source_metadata_cannot_change_under_an_existing_reward_binding():
    reward, plant, _, sim, *_ = fixture()
    plant.scene.readback['selected_model_spec_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='identity changed'):
        reward.refresh_model()
    assert sim.steps == 0


def test_native_action_bounds_and_restored_protocol_ids_are_required_before_model_readback():
    reward, plant, _, sim, _, _, reads, _ = fixture()
    old = dict(reads)
    with pytest.raises(ValueError, match='consumed IDs'):
        NativeRewardBridge(plant, reward.geometry, reward.collection_config, knee_margin_rad=.05, retired_request_ids=None)
    with pytest.raises(ValueError, match='output controller'):
        NativeRewardBridge(plant, reward.geometry, replace(reward.collection_config, action_bounds=(1.,)*6),
                             knee_margin_rad=.05, retired_request_ids=torch.full((2,), -1, dtype=torch.long))
    assert old == reads and sim.steps == 0
    cold = NativeRewardBridge(plant, reward.geometry, reward.collection_config, knee_margin_rad=.05,
                               retired_request_ids=torch.tensor([17, 23]))
    assert cold.composer._success_seen.tolist() == [17, 23]

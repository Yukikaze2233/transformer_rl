"""End-to-end packet binding checks with explicit synthetic native evidence."""
from dataclasses import replace

import pytest
import torch

from test_native_reward import begin, fixture
from transformer_rl._native_plant import GroundReference
from transformer_rl._wheel_clearance import CylinderGap, WheelClearanceSample


def sample(fault=None):
    reward, plant, robot, sim, *_ = fixture()
    scene, bridge = plant.scene, plant.scene.observation
    scene.readback['selected_urdf_sha256'] = 'a' * 64
    ground = plant._ground

    def query(*args):
        old = ground(*args)
        gap = torch.full((2, 2), .007)
        valid = torch.ones(2, 2, dtype=torch.bool)
        points = torch.zeros(2, 2, 3, dtype=torch.float64)
        points[..., 2] = .007
        geometry = CylinderGap(gap, points, torch.zeros_like(points), torch.zeros(2, 2, dtype=torch.long),
                                valid.clone(), valid.clone())
        names = tuple(bridge.binding.native_body_names[i] for i in bridge.binding.wheel_body_indices)
        value = WheelClearanceSample(geometry, scene.environment_origins_m.clone(), bridge.timestamp_s(),
                                      bridge.physical_tick, bridge.model_epoch, names, 'a' * 64, True, True)
        if fault == 'no_native_equivalence':
            value = replace(value, native_collision_equivalence_verified=False)
        elif fault == 'no_source':
            value = replace(value, source_geometry_bound=False)
        elif fault == 'wrong_urdf':
            value = replace(value, urdf_sha256='b' * 64)
        elif fault == 'no_sha':
            scene.readback['selected_urdf_sha256'] = None
            value = replace(value, urdf_sha256=None)
        elif fault == 'invalid':
            geometry.valid[0, 0] = False
        elif fault == 'hole':
            geometry.domain_covered[0, 0] = False
        elif fault == 'nan':
            geometry.vertical_gap_m[0, 0] = float('nan')
        elif fault == 'stale_tick':
            value = replace(value, physical_tick=bridge.physical_tick - 1)
        elif fault == 'stale_clock':
            value = replace(value, timestamp_s=bridge.timestamp_s() - .005)
        return GroundReference(old.root_world_z_m, old.wheel_world_z_m, old.wheel_friction, value)

    plant._ground = query
    ref, owned = begin(reward)
    tick = plant.step_physics()
    value = reward.observe(tick, ref, episode_id=reward.composer._episode,
                            control_epoch=reward.composer._control, policy_owned=owned)
    return reward, tick, value, sim


def test_skill_measurement_uses_actual_tick_owned_reward_world_velocity_and_positive_normal_load():
    reward, tick, value, sim = sample()
    healthy, failed = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    measurement = reward.skill_measurement(tick, value, healthy=healthy, true_failure=failed)
    assert sim.steps == 1 and measurement.physical_tick == tick.physical_tick
    assert measurement.clearance_reference == 'whole_cylinder_minimum_world_z_gap_over_own_terrain_footprint'
    torch.testing.assert_close(measurement.vertical_velocity_world_m_s, torch.full((2,), .2))
    assert measurement.wheel_normal_load_n.eq(20).all()
    assert not tick.measured.wheel_ground_force_n.any()  # Vector cancellation is not contact absence.
    assert measurement.wheel_clearance_m.eq(.007).all()
    tick.wheel_clearance.geometry.vertical_gap_m.zero_()
    assert measurement.wheel_clearance_m.eq(.007).all()


@pytest.mark.parametrize('fault', ['no_native_equivalence', 'no_source', 'wrong_urdf', 'no_sha', 'invalid', 'hole', 'nan'])
def test_unqualified_or_missing_clearance_never_becomes_physical_skill_truth(fault):
    reward, tick, value, _ = sample(fault)
    with pytest.raises(ValueError):
        reward.skill_measurement(tick, value, healthy=torch.ones(2, dtype=torch.bool),
                                  true_failure=torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize('fault', ['stale_tick', 'stale_clock'])
def test_mixed_tick_clearance_faults_before_the_plant_publishes_a_physical_packet(fault):
    with pytest.raises(ValueError, match='clearance differs'):
        sample(fault)


def test_copied_or_old_reward_packet_is_not_a_new_skill_measurement():
    reward, tick, value, _ = sample()
    with pytest.raises(ValueError, match='actual last'):
        reward.skill_measurement(tick, replace(value), healthy=torch.ones(2, dtype=torch.bool),
                                  true_failure=torch.zeros(2, dtype=torch.bool))


def test_skill_measurement_rechecks_the_native_model_after_assembling_the_packet():
    reward, tick, value, _ = sample()
    bridge = reward.plant.scene.observation
    original, calls = bridge.before_state_read, []

    def changed():
        calls.append(True)
        if len(calls) == 2:
            bridge.invalidate_model()
        original()

    bridge.before_state_read = changed
    with pytest.raises(RuntimeError):
        reward.skill_measurement(tick, value, healthy=torch.ones(2, dtype=torch.bool),
                                  true_failure=torch.zeros(2, dtype=torch.bool))

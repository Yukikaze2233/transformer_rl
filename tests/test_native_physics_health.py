"""PhysX的异步错误必须阻断当前采样，不以正常Python返回值冒充成功。"""
from dataclasses import dataclass, replace

import pytest

from transformer_rl.formal.simulation.health import NativePhysicsHealth
from transformer_rl.formal.runtime.worker import _physics_configuration


class Logging:
    def add_logger(self, callback):
        self.callback = callback
        return 41

    def remove_logger(self, handle):
        assert handle == 41
        self.callback = None


def test_physics_overflow_is_latched_until_the_interval_is_rejected():
    logging = Logging()
    health = NativePhysicsHealth(logging)
    logging.callback('omni.physx.plugin', 2, 'file', 7, 'ordinary TGS warning')
    health.check()
    message = 'PhysX error: capacity too small; simulation will miss interactions'
    logging.callback('omni.physx.plugin', 3, 'file', 7, message)
    logging.callback('omni.physx.plugin', 3, 'file', 8, 'PhysX error: second')
    assert health.messages_seen == 3
    assert health.first_error == message
    with pytest.raises(RuntimeError, match='invalid interval'):
        health.check()
    health.close()
    health.close()
    assert logging.callback is None


def test_unrelated_messages_cannot_fail_physics():
    health = NativePhysicsHealth(Logging())
    health._on_message('another.plugin', 3, '', 0, 'PhysX error: unrelated text')
    health.check()
    health.close()


def test_full_scale_pair_buffers_are_passed_to_the_native_config():
    @dataclass
    class Configuration:
        enable_external_forces_every_iteration: bool
        enable_scene_query_support: bool
        gpu_found_lost_pairs_capacity: int
        gpu_found_lost_aggregate_pairs_capacity: int
        gpu_total_aggregate_pairs_capacity: int

    value = _physics_configuration(Configuration)
    assert value.gpu_found_lost_pairs_capacity > 61882941
    assert value.gpu_found_lost_aggregate_pairs_capacity > 61882941
    assert value.gpu_total_aggregate_pairs_capacity > 2891559
    assert value.enable_external_forces_every_iteration and value.enable_scene_query_support


def test_native_overflow_does_not_publish_a_transition_or_replay_the_step():
    import torch
    from test_native_plant import fixture

    plant, robot, sim, writes = fixture()
    health = NativePhysicsHealth(Logging())
    plant.scene = replace(plant.scene, physics_health=health)
    plant.accept_issued_action(torch.zeros(plant.num_envs, 6))
    original = sim.step

    def overflow(**kwargs):
        original(**kwargs)
        health._on_message('omni.physx.plugin', 3, '', 0, 'PhysX error: lost contacts')

    sim.step = overflow
    with pytest.raises(RuntimeError, match='invalid interval'):
        plant.step_physics()
    assert sim.steps == 1 and plant.scene.observation.physical_tick == 0
    with pytest.raises(RuntimeError, match='faulted'):
        plant.step_physics()
    assert sim.steps == 1
    health.close()

"""Kaiser CPU contracts for live native wheel-shape material provenance."""
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.materials import WheelMaterialReader


class MaterialView:
    def __init__(self):
        self.count, self.max_shapes, self.reads = 3, 5, 0
        self.link_paths = [[f'/World/envs/env_{row}/Robot/{name}'
                            for name in ('base_link', 'L_link3', 'R_link3')] for row in range(3)]
        # Two shapes on each wheel; the base material is intentionally unrelated.
        self.values = torch.tensor([[[.1, .1, 0], [.9, .7, .02], [.9, .7, .02],
                                     [.8, .6, .03], [.8, .6, .03]]]).expand(3, -1, -1).clone()
        self.values[1, 1:3, 0] = 1.1

    def get_material_properties(self):
        self.reads += 1
        return self.values


@pytest.fixture
def robot():
    view = MaterialView()
    source = SimpleNamespace(create_rigid_body_view=lambda path: SimpleNamespace(
        max_shapes={'base_link': 1, 'L_link3': 2, 'R_link3': 2}[path.rsplit('/', 1)[-1]]))
    return SimpleNamespace(root_view=view, body_names=['base_link', 'L_link3', 'R_link3'],
                           _physics_sim_view=source)


def reader(robot, wheels=(1, 2), **kwargs):
    return WheelMaterialReader(robot, wheels, num_envs=3, device='cpu',
                               native_to_tensor=lambda value: value, **kwargs)


def test_feature_order_is_native_wheel_material_without_averaging_or_ground_substitution(robot):
    value = reader(robot)
    expected = torch.tensor([[.9, .7, .02, .8, .6, .03],
                             [1.1, .7, .02, .8, .6, .03], [.9, .7, .02, .8, .6, .03]])
    torch.testing.assert_close(value.features(), expected, rtol=0, atol=0)
    assert value.receipt['wheel_shape_ranges'] == [[1, 3], [3, 5]]
    assert not value.receipt['terrain_material_substitution']
    changed = value.receipt
    changed['wheel_body_names'][0] = 'not_the_wheel'
    assert value.receipt['wheel_body_names'] == ['L_link3', 'R_link3']
    for _ in range(100):
        value.features()
    assert robot.root_view.reads == 1


def test_explicit_row_refresh_is_transactional_and_does_not_alias_native_storage(robot):
    value = reader(robot)
    previous = value.features().clone()
    robot.root_view.values[1, 1:3] = torch.tensor([1.2, 1., .1])
    torch.testing.assert_close(value.features(), previous)
    value.refresh(torch.tensor([1]))
    torch.testing.assert_close(value.features()[[0, 2]], previous[[0, 2]])
    torch.testing.assert_close(value.features()[1, :3], torch.tensor([1.2, 1., .1]))
    before_failure = value.features().clone()
    robot.root_view.values[1, 4, 1] = .123
    with pytest.raises(ValueError, match='different shape materials'):
        value.refresh(torch.tensor([1]))
    torch.testing.assert_close(value.features(), before_failure, rtol=0, atol=0)


def test_reversed_wheel_binding_reverses_the_complete_material_triples(robot):
    forward = reader(robot)
    reversed_value = reader(robot, (2, 1))
    torch.testing.assert_close(reversed_value.features(),
                               torch.cat((forward.features()[:, 3:], forward.features()[:, :3]), -1))


@pytest.mark.parametrize('rows', ([3], [-1], [0, 0], [[0]], [0.5], [True]))
def test_invalid_refresh_rows_are_rejected_without_cache_change(robot, rows):
    value = reader(robot)
    before = value.features().clone()
    with pytest.raises(ValueError, match='environment rows'):
        value.refresh(rows)
    torch.testing.assert_close(value.features(), before, rtol=0, atol=0)


@pytest.mark.parametrize('bad', (float('nan'), float('inf'), -.1))
def test_nonfinite_or_negative_native_friction_is_rejected(robot, bad):
    robot.root_view.values[:, 1:3, 0] = bad
    with pytest.raises(ValueError, match='physical parameters'):
        reader(robot)


def test_invalid_restitution_is_rejected(robot):
    robot.root_view.values[:, 3:5, 2] = 1.1
    with pytest.raises(ValueError, match='physical parameters'):
        reader(robot)


def test_mixed_wheel_shape_materials_are_not_silently_averaged(robot):
    robot.root_view.values[0, 2, 1] = .4
    with pytest.raises(ValueError, match='different shape materials'):
        reader(robot)


@pytest.mark.parametrize('wheels', ((1, 1), (0, 3), (True, 2), (1,), (1, 2, 0)))
def test_invalid_body_binding_is_rejected(robot, wheels):
    with pytest.raises(ValueError, match='binding'):
        reader(robot, wheels)


def test_native_link_order_must_match_body_binding(robot):
    robot.root_view.link_paths[0].reverse()
    with pytest.raises(ValueError, match='link order'):
        reader(robot)


def test_shape_count_sum_must_match_the_native_material_buffer(robot):
    robot.root_view.max_shapes = 4
    with pytest.raises(ValueError, match='shape counts'):
        reader(robot)


def test_missing_wheel_collision_shapes_are_rejected(robot):
    robot._physics_sim_view.create_rigid_body_view = lambda path: SimpleNamespace(
        max_shapes=5 if path.endswith('R_link3') else 0)
    with pytest.raises(ValueError, match='no native collision'):
        reader(robot)


def test_unexpected_native_buffer_shape_and_view_replacement_are_rejected(robot):
    robot.root_view.values = robot.root_view.values[:, :, :2]
    with pytest.raises(ValueError, match='native material buffer'):
        reader(robot)
    robot.root_view = MaterialView()
    value = reader(robot)
    robot.root_view = MaterialView()
    with pytest.raises(RuntimeError, match='view changed'):
        value.refresh()

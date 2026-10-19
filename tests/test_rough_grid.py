"""Kaiser上的完整高度场/分块同几何检查，不等同于原生碰撞验收。"""
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from transformer_rl.formal.tasks.geometry import TerrainKind, TerrainQueryBatch, TerrainSurface
from transformer_rl.formal.tasks.terrain import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning


def fixture(*, direction=1, seed=731, extent=None, amplitude=.037):
    generator = ContinuousTerrainGenerator(2101, TerrainTuning(.7, (.2, 1.1), .25))
    request = TerrainRequest(TerrainKind.ROUGH, extent or TerrainExtent((-8., 8.), (-3., 3.)),
                             direction=direction, roughness_m=amplitude)
    return generator, request, generator.rough_grid(request, geometry_seed=seed)


@pytest.mark.parametrize('direction', [-1, 1])
def test_partitioned_mesh_and_global_queries_match_dense_mesh_at_edges_and_interiors(direction):
    generator, request, grid = fixture(direction=direction)
    dense = generator.build(request, geometry_seed=grid.geometry_seed)
    rows, columns = ((size - 2) // 7 + 1 for size in grid.shape)
    pieces = [grid.tile(i, j, cells=7) for i in range(rows) for j in range(columns)]
    quilt = TerrainQueryBatch((TerrainSurface(tuple(pieces)),))
    reference = TerrainQueryBatch((dense.surface,))
    rng = torch.Generator().manual_seed(17)
    random = torch.rand(1, 500, 2, generator=rng) * torch.tensor([16., 6.]) - torch.tensor([8., 3.])
    edges = torch.tensor([[[-8., -3.], [-8., 3.], [8., -3.], [8., 3.], [0., 0.],
                            [direction * 1.5, 0.], [direction * 2., 0.]]])
    points = torch.cat((random, edges), dim=1)
    actual, expected, sparse = quilt.at(points), reference.at(points), grid.at(points)
    assert actual.valid.all() and sparse.valid.all()
    torch.testing.assert_close(actual.height_m, expected.height_m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(sparse.height_m, expected.height_m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(sparse.friction, expected.friction)
    full = grid.patch(0, grid.shape[0] - 1, 0, grid.shape[1] - 1)
    assert dense.surface.patches == (full,)


def test_adjacent_tiles_have_bitwise_identical_vertices_and_order_never_advances_rng():
    _, _, grid = fixture()
    before = torch.random.get_rng_state().clone()
    a, b = grid.tile(0, 0, cells=7), grid.tile(1, 0, cells=7)
    c = grid.tile(0, 1, cells=7)
    assert a.x_m[-1] == b.x_m[0] and a.z_m[-1] == b.z_m[0]
    assert a.y_m[-1] == c.y_m[0]
    assert tuple(row[-1] for row in a.z_m) == tuple(row[0] for row in c.z_m)
    assert grid.tile(1, 0, cells=7) == b and grid.tile(0, 0, cells=7) == a
    assert torch.equal(before, torch.random.get_rng_state())


def test_full_sixty_second_domain_materializes_only_requested_tiles():
    _, request, grid = fixture(extent=TerrainExtent.for_horizon(60., 5., 2.))
    assert request.extent.x_m == request.extent.y_m == (-302., 302.)
    assert grid.shape[0] * grid.shape[1] > 2_000_000
    keys = grid.tile_keys((20., 20.), (0., 0.))
    assert len(keys) == 9
    tiles = [grid.tile(*key) for key in keys]
    assert all(len(tile.mesh.vertices_m) <= 33 * 33 for tile in tiles)
    assert sum(len(tile.mesh.vertices_m) for tile in tiles) <= 9 * 33 * 33
    assert not hasattr(grid, 'z_m')  # Only axes and the global seed are retained.
    points = torch.tensor([[[-302., -302.], [302., 302.], [250., 250.], [-250., -250.]]])
    assert grid.at(points).valid.all()


@pytest.mark.parametrize('direction', [-1, 1])
def test_window_selects_physical_coordinates_and_keeps_all_boundary_cells(direction):
    _, _, grid = fixture(direction=direction)
    bounds = (direction * 2.3, direction * 4.1)
    keys = grid.tile_keys(tuple(sorted(bounds)), (-.7, .9), cells=4)
    surface = TerrainSurface(tuple(grid.tile(*key, cells=4) for key in keys))
    points = torch.tensor([[[bounds[0], -.7], [bounds[1], .9], [direction * 3., .5]]])
    sampled = TerrainQueryBatch((surface,)).at(points)
    assert sampled.valid.all()
    torch.testing.assert_close(sampled.height_m, grid.at(points).height_m, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match='outside'):
        grid.tile_keys((-9., 0.), (-1., 1.))


def test_8192_seed_queries_need_four_vertex_values_per_point_without_full_fields():
    _, _, grid = fixture(extent=TerrainExtent.for_horizon(60., 5., 2.))
    points = torch.tensor([[[3., .1], [10., -1.], [301., 301.]]]).repeat(8192, 1, 1)
    seeds = torch.arange(8192, dtype=torch.long) + 41
    before = torch.random.get_rng_state().clone()
    actual = grid.at(points, seeds=seeds)
    assert actual.height_m.shape == (8192, 3) and actual.valid.all()
    for row in (0, 17, 8191):
        _, _, source = fixture(extent=TerrainExtent.for_horizon(60., 5., 2.), seed=int(seeds[row]))
        torch.testing.assert_close(source.at(points[row:row + 1]).height_m, actual.height_m[row:row + 1])
    assert torch.equal(before, torch.random.get_rng_state())


def test_source_identity_binds_seed_amplitude_direction_and_global_layout():
    _, _, a = fixture()
    _, _, b = fixture()
    _, _, c = fixture(seed=732)
    _, _, d = fixture(amplitude=.038)
    _, _, e = fixture(direction=-1)
    assert a.sha256 == b.sha256 and len({a.sha256, c.sha256, d.sha256, e.sha256}) == 4
    with pytest.raises(FrozenInstanceError):
        a.geometry_seed = 2
    points = torch.tensor([[[3.1, .2], [4.3, 1.]]])
    assert not torch.equal(a.at(points).height_m, c.at(points).height_m)


def test_invalid_queries_remain_missing_and_zero_amplitude_is_exactly_flat():
    _, _, grid = fixture(amplitude=0)
    points = torch.tensor([[[0., 0.], [4., 1.], [9., 0.]]])
    result = grid.at(points)
    assert result.valid.tolist() == [[True, True, False]]
    assert result.height_m[0, :2].eq(0).all() and result.height_m[0, 2].isnan()
    assert result.friction[0, 2].isnan() and result.patch_index[0, 2] == -1


def test_source_generator_state_replays_tile_and_dense_choices_without_new_geometry_draws():
    generator, request, _ = fixture()
    before = generator.state_dict()['rng'].clone()
    first = generator.rough_grid(request, geometry_seed=123)
    assert torch.equal(before, generator.state_dict()['rng'])
    second = generator.rough_grid(replace(request, direction=-1), geometry_seed=123)
    assert first.sha256 != second.sha256
    a = first.tile(0, 0)
    assert first.tile(0, 0) == a and torch.equal(before, generator.state_dict()['rng'])

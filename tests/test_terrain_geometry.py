"""Authored triangle geometry versus privileged queries; no SDK contexts."""
from dataclasses import replace
import math

import pytest
import torch

from transformer_rl._terrain_geometry import HeightPatch, TerrainQueryBatch, TerrainSurface


def at(surface, points):
    return TerrainQueryBatch((surface,)).at(torch.tensor([points], dtype=torch.float32))


def test_nonplanar_cell_uses_its_actual_triangle_instead_of_bilinear_height():
    patch = HeightPatch((0., 1.), (0., 1.), ((0., 0.), (0., 2.)), .8)
    result = at(TerrainSurface((patch,)), ((.75, .25), (.25, .75), (.5, .5)))
    torch.testing.assert_close(result.height_m, torch.tensor([[.5, .5, 1.]]))
    assert result.height_m[0, 0] != pytest.approx(.75 * .25 * 2)
    assert patch.mesh.triangles == ((0, 2, 3), (0, 3, 1))


def test_query_matches_independent_barycentric_intersections_with_authored_triangles():
    patch = HeightPatch((-2., -.5, .75, 2.), (-1., .25, 1.),
                        ((.1, -.2, .3), (.4, .5, -.1), (-.3, .2, .1), (.2, -.4, .6)), .7)
    points = [(-1.8, -.5), (-.7, .7), (.4, .1), (.9, -.3), (1.8, .8)]
    result = at(TerrainSurface((patch,)), points)
    expected = []
    for x, y in points:
        hits = []
        for face in patch.mesh.triangles:
            a, b, c = (patch.mesh.vertices_m[i] for i in face)
            det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            u = ((x - a[0]) * (c[1] - a[1]) - (y - a[1]) * (c[0] - a[0])) / det
            v = ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / det
            if u >= 0 and v >= 0 and u + v <= 1:
                hits.append(a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]))
        assert hits
        expected.append(max(hits))
    torch.testing.assert_close(result.height_m, torch.tensor([expected]), atol=2e-7, rtol=1e-6)


def test_variable_grid_groups_and_multiple_patches_keep_native_environment_order():
    a = HeightPatch((-2., 0.), (-1., 1.), ((0., 0.), (0., 0.)), .2)
    b = HeightPatch((0., 1., 2.), (-1., 0., 1.), ((.3, .3, .3), (.3, .3, .3), (.3, .3, .3)), .9)
    c = HeightPatch((-2., 2.), (-1., 1.), ((-.5, -.5), (-.5, -.5)), .6)
    result = TerrainQueryBatch((TerrainSurface((a, b)), TerrainSurface((c,)))).at(
        torch.tensor([[[-1., 0.], [1., 0.]], [[-1., 0.], [1., 0.]]]))
    torch.testing.assert_close(result.height_m, torch.tensor([[0., .3], [-.5, -.5]]))
    torch.testing.assert_close(result.friction, torch.tensor([[.2, .9], [.6, .6]]))
    assert result.valid.all()


def test_step_edge_selects_highest_surface_and_material_seam_uses_one_real_material():
    low = HeightPatch((-2., 0.), (-1., 1.), ((0., 0.), (0., 0.)), .2)
    high = HeightPatch((0., 2.), (-1., 1.), ((.25, .25), (.25, .25)), .9)
    result = at(TerrainSurface((low, high)), ((0., 0.),))
    assert result.height_m.item() == .25 and result.friction.item() == pytest.approx(.9)
    equal = replace(high, z_m=((0., 0.), (0., 0.)))
    seam = at(TerrainSurface((low, equal)), ((0., 0.),))
    assert seam.friction.item() == pytest.approx(.2)


def test_holes_and_outside_edges_are_invalid_without_an_infinite_plane_fallback():
    left = HeightPatch((-2., -1.), (-1., 1.), ((0., 0.), (0., 0.)), .8)
    right = HeightPatch((1., 2.), (-1., 1.), ((0., 0.), (0., 0.)), .8)
    result = at(TerrainSurface((left, right)), ((0., 0.), (-2.01, 0.), (1.5, 2.), (-1.5, 0.)))
    assert result.valid.tolist() == [[False, False, False, True]]
    assert torch.isnan(result.height_m[~result.valid]).all()
    assert result.patch_index.tolist() == [[-1, -1, -1, 0]]


def test_packed_geometry_and_outputs_do_not_alias_mutable_inputs():
    heights = [[0., 0.], [0., .3]]
    patch = HeightPatch((-1., 1.), (-1., 1.), heights, .8)
    batch = TerrainQueryBatch((TerrainSurface((patch,)),))
    xy = torch.tensor([[[.5, .5]]])
    first = batch.at(xy)
    expected = first.height_m.clone()
    heights[1][1] = 999
    xy.zero_()
    batch.at(xy)
    torch.testing.assert_close(first.height_m, expected)


def test_native_float32_rounding_is_shared_by_authoring_and_query_even_under_double_default():
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        patch = HeightPatch((0., .3), (0., .7), ((0., .1), (.2, .3)), .8)
        query = TerrainQueryBatch((TerrainSurface((patch,)),))
        result = query.at(torch.tensor([[[patch.x_m[1], patch.y_m[1]]]], dtype=torch.float32))
        assert result.height_m.dtype == torch.float32 and result.valid.all()
        assert result.height_m.item() == patch.mesh.vertices_m[-1][2]
    finally:
        torch.set_default_dtype(before)


@pytest.mark.parametrize('change', ['collapsed_float32', 'unordered', 'missing_z', 'nonfinite'])
def test_invalid_native_grid_is_rejected_before_authoring(change):
    x, y, z = (0., 1.), (0., 1.), ((0., 0.), (0., 0.))
    if change == 'collapsed_float32':
        x = (1e8, 1e8 + 1)
    elif change == 'unordered':
        x = (1., 0.)
    elif change == 'missing_z':
        z = ((0.,), (0.,))
    else:
        z = ((0., 0.), (float('nan'), 0.))
    with pytest.raises(ValueError):
        HeightPatch(x, y, z, .8)


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nan'])
def test_inconsistent_query_contract_is_rejected(change):
    patch = HeightPatch((-1., 1.), (-1., 1.), ((0., 0.), (0., 0.)), .8)
    batch = TerrainQueryBatch((TerrainSurface((patch,)),))
    xy = torch.zeros(1, 2, 2)
    if change == 'shape':
        xy = xy[0]
    elif change == 'dtype':
        xy = xy.double()
    else:
        xy[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        batch.at(xy)

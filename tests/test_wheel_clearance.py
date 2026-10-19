"""Finite-wheel geometry checks; all execution belongs on Kaiser."""
import math

import pytest
import torch

from transformer_rl._terrain_geometry import HeightPatch, TerrainKind, TerrainMesh, TerrainQueryBatch, TerrainSurface
from transformer_rl._wheel_clearance import CylinderTerrainGap


def plane(sx=0., sy=0., height=0., extent=(-2., 2.)):
    x0, x1 = extent
    z = tuple(tuple(height + sx * x + sy * y for y in extent) for x in extent)
    return TerrainSurface((HeightPatch((x0, x1), (x0, x1), z, .8),))


def run(surface, centre=(0., 0., .1), axis=(0., 1., 0.), radius=.06, length=.025):
    terrain = TerrainQueryBatch((surface,))
    query = CylinderTerrainGap(terrain, (radius,), (length,))
    result = query.at(torch.tensor([[centre]], dtype=torch.float64), torch.tensor([[axis]], dtype=torch.float64))
    return result


@pytest.mark.parametrize('axis,expected', [((0., 1., 0.), .04), ((0., 0., 1.), .0875),
                                          ((0., math.sqrt(.5), math.sqrt(.5)), .1 - .0725 * math.sqrt(.5))])
def test_complete_cylinder_width_and_axis_define_the_flat_ground_support(axis, expected):
    sample = run(plane(), axis=axis)
    assert sample.valid.all() and sample.domain_covered.all()
    assert sample.vertical_gap_m.item() == pytest.approx(expected, abs=2e-8)
    torch.testing.assert_close(sample.cylinder_point_local_m[..., 2] - sample.terrain_point_local_m[..., 2],
                               sample.vertical_gap_m.double(), atol=2e-8, rtol=0)


@pytest.mark.parametrize('sx,sy,axis', [(0.5, 0., (0., 1., 0.)), (0., .3, (0., 1., 0.)),
                                     (.4, -.2, (.6, 0., .8)), (-.2, .6, (0., 0., 1.))])
def test_slope_uses_the_whole_shape_support_not_world_lowest_point_or_centre_height(sx, sy, axis):
    normal = torch.tensor([-sx, -sy, 1.], dtype=torch.float64)
    a = torch.tensor(axis, dtype=torch.float64)
    along = normal.dot(a).item()
    expected = .1 - .0125 * abs(along) - .06 * math.sqrt(normal.dot(normal).item() - along * along)
    sample = run(plane(sx, sy), axis=axis)
    assert sample.vertical_gap_m.item() == pytest.approx(expected, abs=4e-8)


def test_high_step_under_rim_is_seen_even_when_centre_is_above_the_low_plane():
    surface = TerrainSurface((HeightPatch((-2., .05), (-2., 2.), ((0., 0.), (0., 0.)), .7),
                              HeightPatch((.05, 2.), (-2., 2.), ((.05, .05), (.05, .05)), .7)))
    sample = run(surface)
    expected = .1 - math.sqrt(.06**2 - .05**2) - .05
    assert sample.vertical_gap_m.item() == pytest.approx(expected, abs=3e-8)
    assert sample.cylinder_point_local_m[0, 0, 0].item() == pytest.approx(.05, abs=2e-8)
    assert sample.vertical_gap_m.item() < .04


def test_small_raised_corner_requires_a_vertex_active_set_instead_of_plane_extrapolation():
    # Vertical axis: its bottom disk covers the high-patch corner, while the
    # high patch does not cover the cylinder centre.
    surface = TerrainSurface((HeightPatch((-2., 2.), (-2., 2.), ((0., 0.), (0., 0.)), .7),
                              HeightPatch((.03, .05), (.03, .05), ((.04, .04), (.04, .04)), .7)))
    sample = run(surface, axis=(0., 0., 1.))
    assert sample.vertical_gap_m.item() == pytest.approx(.1 - .0125 - .04, abs=3e-8)
    point = sample.cylinder_point_local_m[0, 0]
    assert point[0] >= .03 - 2e-8 and point[1] >= .03 - 2e-8


def test_far_high_patch_does_not_extend_its_supporting_plane_under_the_wheel():
    surface = TerrainSurface((HeightPatch((-2., 2.), (-2., 2.), ((0., 0.), (0., 0.)), .7),
                              HeightPatch((.08, 1.), (-1., 1.), ((1., 1.), (1., 1.)), .7)))
    assert run(surface).vertical_gap_m.item() == pytest.approx(.04, abs=2e-8)


def test_penetration_is_signed_and_not_zeroed_like_an_unsigned_distance():
    sample = run(plane(), centre=(0., 0., .055))
    assert sample.vertical_gap_m.item() == pytest.approx(-.005, abs=2e-8)


def test_hidden_extra_wall_cannot_be_ignored_by_a_height_based_query():
    top = plane().patches
    wall = TerrainMesh(((0., -1., 0.), (0., 1., 0.), (0., 1., .2), (0., -1., .2)),
                        ((0, 1, 2), (0, 2, 3)), .8)
    with pytest.raises(ValueError, match='unrepresented'):
        run(TerrainSurface(top, (wall,)))


@pytest.mark.parametrize('kind', list(TerrainKind))
@pytest.mark.parametrize('direction', [-1, 1])
def test_every_existing_continuous_course_family_and_its_risers_bind_the_same_geometry(kind, direction):
    from test_terrain_tasks import generator, request
    course = generator().build(request(kind, direction=direction))
    sample = run(course.surface, centre=(0., 0., .15))
    assert sample.valid.all() and sample.domain_covered.all()
    assert sample.vertical_gap_m.item() == pytest.approx(.09, abs=2e-8)


@pytest.mark.parametrize('centre', [(1.98, 0., .1), (0., -1.995, .1), (3., 0., .1)])
def test_any_unknown_wheel_footprint_at_the_authored_outer_edge_is_invalid(centre):
    sample = run(plane(), centre=centre)
    assert not sample.valid.any() and not sample.domain_covered.any()
    assert torch.isnan(sample.vertical_gap_m).all() and sample.triangle_index.eq(-1).all()


def test_hole_under_rim_is_unknown_even_if_wheel_centre_has_ground():
    surface = TerrainSurface((HeightPatch((-2., .03), (-2., 2.), ((0., 0.), (0., 0.)), .8),
                              HeightPatch((.04, 2.), (-2., 2.), ((0., 0.), (0., 0.)), .8)))
    assert not run(surface).valid.any()
    assert run(surface, centre=(-.1, 0., .1)).valid.all()


def test_missing_rectangle_inside_aabb_but_outside_the_actual_projected_disk_is_not_a_hole_under_wheel():
    surface = TerrainSurface((HeightPatch((-2., .05), (-2., 2.), ((0., 0.), (0., 0.)), .8),
                              HeightPatch((.05, 2.), (-2., .05), ((0., 0.), (0., 0.)), .8)))
    # AABB overlaps the missing upper-right quadrant, but radius .06 cannot
    # reach its nearest corner sqrt(.05**2+.05**2).
    sample = run(surface, axis=(0., 0., 1.))
    assert sample.valid.all() and sample.domain_covered.all()


def test_two_environments_and_multiple_grid_shapes_keep_geometry_and_witnesses_separate():
    grid = HeightPatch((-1., -.03, .01, .08, 1.), (-1., 0., 1.),
                        tuple((0., 0., 0.) for _ in range(5)), .8)
    surfaces = (TerrainSurface((grid,)), plane(height=.2))
    query = CylinderTerrainGap(TerrainQueryBatch(surfaces), (.06, .04), (.025, .03))
    centres = torch.tensor([[[0., 0., .1], [.1, 0., .2]], [[0., 0., .3], [.1, 0., .4]]])
    axes = torch.tensor([[[0., 1., 0.], [0., 0., 1.]]]).repeat(2, 1, 1)
    sample = query.at(centres, axes)
    torch.testing.assert_close(sample.vertical_gap_m, torch.tensor([[.04, .185], [.04, .185]]), atol=4e-8, rtol=0)
    before = sample.cylinder_point_local_m.clone()
    centres.zero_()
    assert torch.equal(before, sample.cylinder_point_local_m)


def test_witness_matches_independent_dense_solid_boundary_samples_on_piecewise_terrain():
    torch.manual_seed(18)
    surface = TerrainSurface((HeightPatch((-1., -.04, .02, 1.), (-1., -.02, .03, 1.),
                                          ((.01, .02, .03, .01), (.04, .01, -.01, .02),
                                           (.0, .03, .02, .01), (.01, .01, .01, .01)), .7),))
    terrain = TerrainQueryBatch((surface,))
    theta = torch.linspace(0, 2 * math.pi, 2001, dtype=torch.float64)
    for _ in range(12):
        axis = torch.randn(3, dtype=torch.float64)
        axis /= axis.norm()
        e = torch.linalg.cross(axis, torch.tensor([0., 0., 1.], dtype=torch.float64))
        e /= e.norm()
        f = torch.linalg.cross(axis, e)
        centre = torch.tensor([.005, -.003, .13], dtype=torch.float64)
        query = CylinderTerrainGap(terrain, (.06,), (.025,))
        result = query.at(centre[None, None], axis[None, None])
        # Independent cylinder side and cap samples provide an upper bound.
        s = torch.linspace(-.0125, .0125, 151, dtype=torch.float64)
        circle = theta.cos()[:, None] * e + theta.sin()[:, None] * f
        points = centre + s[:, None, None] * axis + .06 * circle[None]
        points = points.flatten(0, 1)
        radii = torch.linspace(0, .06, 101, dtype=torch.float64)
        cap = radii[:, None, None] * circle[None]
        caps = torch.cat((centre - .0125 * axis + cap, centre + .0125 * axis + cap), 0).flatten(0, 1)
        points = torch.cat((points, caps))
        z = terrain.at(points[None, :, :2].float()).height_m[0].double()
        sampled = (points[:, 2] - z).min().item()
        exact = result.vertical_gap_m.item()
        assert exact <= sampled + 2e-7
        assert sampled - exact < .0005
        p = result.cylinder_point_local_m
        witness_ground = terrain.at(p[..., :2].float()).height_m.double()
        torch.testing.assert_close(p[..., 2] - witness_ground, result.vertical_gap_m.double(), atol=2e-7, rtol=0)


@pytest.mark.parametrize('change', ['zero_axis', 'axis_nan', 'wrong_shape', 'negative_radius'])
def test_invalid_geometry_does_not_become_zero_clearance(change):
    query = CylinderTerrainGap(TerrainQueryBatch((plane(),)), (.06,), (.025,))
    centre = torch.tensor([[[0., 0., .1]]])
    axis = torch.tensor([[[0., 1., 0.]]])
    if change == 'zero_axis':
        axis.zero_()
    elif change == 'axis_nan':
        axis[0, 0, 0] = float('nan')
    elif change == 'wrong_shape':
        centre = centre[0]
    else:
        with pytest.raises(ValueError):
            CylinderTerrainGap(query.terrain, (-.06,), (.025,))
        return
    with pytest.raises(ValueError):
        query.at(centre, axis)

"""Continuous terrain geometry and sampling-state checks without native physics."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from transformer_rl._task_curriculum import ContinuousTaskSampler, TaskFamily
from transformer_rl._terrain_geometry import TerrainKind, TerrainQueryBatch
from transformer_rl._terrain_tasks import ContinuousTerrainGenerator, TerrainExtent, TerrainRequest, TerrainTuning


def tuning():
    return TerrainTuning(.7, (.2, 1.1), .25)


def generator(seed=2101):
    return ContinuousTerrainGenerator(seed, tuning())


def request(kind, **kwargs):
    return TerrainRequest(kind, TerrainExtent((-8., 8.), (-3., 3.)), **kwargs)


def query(course, points):
    return TerrainQueryBatch((course.surface,)).at(torch.tensor([points], dtype=torch.float32))


@pytest.mark.parametrize('kind', list(TerrainKind))
def test_every_named_course_is_authored_as_meshes_and_valid_at_spawn(kind):
    course = generator().build(request(kind, slope_rad=.1, roughness_m=.017, step_height_m=.08))
    assert course.surface.meshes
    initial = query(course, ((0., 0.),))
    assert initial.valid.all() and initial.height_m.item() == 0
    assert all(len(mesh.triangles) > 0 for mesh in course.surface.meshes)


@pytest.mark.parametrize('kind,height', [(TerrainKind.STEP_UP, .237), (TerrainKind.STEP_DOWN, -.149)])
def test_step_height_is_continuous_and_has_an_actual_vertical_riser(kind, height):
    course = generator().build(request(kind, step_height_m=height, material_friction=(.1, 1.2)))
    result = query(course, ((1.49, 0.), (1.51, 0.)))
    torch.testing.assert_close(result.height_m, torch.tensor([[0., height]]))
    assert len(course.surface.vertical_faces) == 1
    wall = course.surface.vertical_faces[0]
    assert set(v[0] for v in wall.vertices_m) == {1.5}
    assert max(v[2] for v in wall.vertices_m) - min(v[2] for v in wall.vertices_m) == pytest.approx(abs(height))
    assert result.friction.eq(torch.tensor(.7)).all()


@pytest.mark.parametrize('rise', [.04, -.03])
def test_continuous_staircase_preserves_every_increment_for_ascent_and_descent(rise):
    course = generator().build(request(TerrainKind.STAIRS, step_height_m=rise))
    result = query(course, ((1.51, 0.), (1.81, 0.), (2.11, 0.), (2.41, 0.)))
    torch.testing.assert_close(result.height_m, torch.tensor([[rise * i for i in (1, 2, 3, 4)]]))
    assert len(course.surface.vertical_faces) == 4


def test_two_platform_absolute_tops_are_point20_and_point25_not_point20_plus_point25():
    course = generator().build(request(TerrainKind.TWO_PLATFORMS, platform_tops_m=(.20, .25)))
    result = query(course, ((1.6, 0.), (2.2, 0.)))
    torch.testing.assert_close(result.height_m, torch.tensor([[.20, .25]]))
    assert len(course.surface.vertical_faces) == 2
    assert max(v[2] for wall in course.surface.vertical_faces for v in wall.vertices_m) == .25


def test_reverse_approach_mirrors_platforms_and_their_real_risers():
    source = request(TerrainKind.TWO_PLATFORMS)
    forward = generator().build(source)
    reverse = generator().build(replace(source, direction=-1))
    torch.testing.assert_close(query(forward, ((1.6, 0.), (2.2, 0.))).height_m,
                               query(reverse, ((-1.6, 0.), (-2.2, 0.))).height_m)
    assert sorted({v[0] for w in reverse.surface.vertical_faces for v in w.vertices_m}) == pytest.approx([-2.1, -1.5])


@pytest.mark.parametrize('angle,bearing', [(.2, 0.), (-.2, 0.), (.2, math.pi / 2)])
def test_longitudinal_and_cross_slopes_have_the_requested_physical_gradient(angle, bearing):
    course = generator().build(request(TerrainKind.SLOPE, slope_rad=angle, slope_bearing_rad=bearing))
    result = query(course, ((3., 0.), (4., 0.), (3., 1.)))
    z = result.height_m[0]
    assert (z[1] - z[0]).item() == pytest.approx(math.tan(angle) * math.cos(bearing), abs=2e-6)
    assert (z[2] - z[0]).item() == pytest.approx(math.tan(angle) * math.sin(bearing), abs=2e-6)


def test_rough_geometry_is_reproducible_has_both_requested_extremes_and_a_flat_entry():
    req = request(TerrainKind.ROUGH, roughness_m=.0173)
    a = generator().build(req, geometry_seed=99)
    b = generator().build(req, geometry_seed=99)
    c = generator().build(req, geometry_seed=100)
    assert a.surface.meshes == b.surface.meshes and a.surface.meshes != c.surface.meshes
    heights = [v[2] for v in a.surface.meshes[0].vertices_m]
    assert max(heights) == pytest.approx(.0173, abs=1e-8)
    assert min(heights) == pytest.approx(-.0173, abs=1e-8)
    assert query(a, ((0., 0.),)).height_m.item() == 0


def test_material_course_changes_real_materials_without_a_geometry_step():
    course = generator().build(request(TerrainKind.MATERIAL, material_friction=(.23, 1.07)))
    result = query(course, ((0., 0.), (2., 0.), (7., 0.)))
    assert not result.height_m.any() and not course.surface.vertical_faces
    torch.testing.assert_close(result.friction, torch.tensor([[.23, 1.07, .23]]))


def test_generator_resume_preserves_geometry_choices_and_uses_no_global_rng():
    global_before = torch.random.get_rng_state().clone()
    sampler = ContinuousTaskSampler(2101)
    draws = sampler.sample(256, 5)
    first = generator()
    first.from_draw(draws, (TerrainExtent((-8., 8.), (-3., 3.)),) * 256)
    saved = first.state_dict()
    restored = generator(999)
    restored.load_state_dict(saved)
    more = sampler.sample(128, 5)
    extents = (TerrainExtent((-8., 8.), (-3., 3.)),) * 128
    assert first.from_draw(more, extents) == restored.from_draw(more, extents)
    assert torch.equal(global_before, torch.random.get_rng_state())
    saved['rng'].zero_()
    assert restored.state_dict()['rng'].any()


def test_full_sampler_emits_both_stairs_and_two_platforms_without_relabeling_them():
    sampler = ContinuousTaskSampler(2101)
    draw = sampler.sample(1024, 5)
    terrain_rows = draw.family == TaskFamily.TERRAIN
    assert set(draw.terrain_kind[terrain_rows].tolist()) == set(range(1, 8))
    courses = generator().from_draw(draw, (TerrainExtent((-8., 8.), (-3., 3.)),) * 1024)
    assert [int(c.request.kind) for c in courses] == draw.terrain_kind.tolist()
    assert any(c.request.kind == TerrainKind.STAIRS and c.request.step_height_m < 0 for c in courses)
    assert any(c.request.kind == TerrainKind.STAIRS and c.request.step_height_m > 0 for c in courses)


def test_too_short_geometry_is_rejected_without_reducing_requested_step_or_course():
    req = TerrainRequest(TerrainKind.TWO_PLATFORMS, TerrainExtent((-1., 2.5), (-1., 1.)))
    with pytest.raises(ValueError, match='complete treads'):
        generator().build(req)
    assert req.platform_tops_m == (.2, .25)


def test_horizon_extent_keeps_the_complete_requested_duration_and_speed():
    extent = TerrainExtent.for_horizon(120., 5., 4.)
    assert extent.x_m == extent.y_m == (-604., 604.)


def test_checkpoint_config_rejection_keeps_existing_rng_unchanged():
    gen = generator()
    saved = gen.state_dict()
    bad = deepcopy(saved)
    bad['tuning']['entry_m'] = 0
    with pytest.raises(ValueError):
        gen.load_state_dict(bad)
    assert torch.equal(gen.state_dict()['rng'], saved['rng'])

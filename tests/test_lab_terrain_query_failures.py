"""Course errors retain the actual source query and bounded diagnostics."""
import json
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.formal.lab_env.terrain import LabTerrainCourse
from test_terrain_query_diagnostics import _query


def course_fixture(tmp_path, position):
    course = LabTerrainCourse.__new__(LabTerrainCourse)
    course.query = _query(tmp_path)
    course.env = SimpleNamespace(
        _physical_tick=17, _root_link_position=lambda: position,
        _root_link_quat=lambda: torch.tensor([[0., 0., 0., 1.]]).expand(len(position), -1),
        _ground_at_root=torch.full((len(position),), 9.),
        _task=SimpleNamespace(nominal_height_m=.32))
    course.router = SimpleNamespace(tile=torch.zeros(len(position), dtype=torch.long),
                                    origins=torch.zeros_like(position))
    course._wheel_centres = lambda: position[:, None].expand(-1, 2, -1)
    course.wheel_ground = torch.full((len(position), 2), 8.)
    course.wheel_material = torch.full((len(position), 2), .7)
    course._ground_token = None
    course._grid_offsets = torch.zeros(77, 2)
    return course


@pytest.mark.parametrize('kind,position', [
    ('geometry_gap', [.9, .9, .3]),
    ('spatial_outside', [1.1, .1, .3]),
    ('nonfinite_state', [.1, .1, float('nan')]),
    ('index_miss', [.1, .1, .3]),
])
def test_ground_error_classifies_the_source_and_preserves_cached_state(tmp_path, kind, position):
    course = course_fixture(tmp_path, torch.tensor([position]))
    if kind == 'index_miss':
        course.query._buffers['offsets'].zero_()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='no physical terrain support') as caught:
        course.ground()
    report = json.loads(str(caught.value).split(': ', 1)[1])
    assert report['rows'] == [0] and report['physical_tick'] == 17
    diagnostic = report['query_diagnostics']
    assert diagnostic['failed_columns'] == 3
    assert {item['reason'] for item in diagnostic['samples']} == {kind}
    assert course.env._ground_at_root.eq(9).all()
    assert course.wheel_ground.eq(8).all() and course.wheel_material.eq(.7).all()
    assert course._ground_token is None
    assert course.query.queries == 3
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize('position,reason', [
    ([.9, .9, .3], 'geometry_gap'),
    ([1.1, .1, .3], 'spatial_outside'),
    ([.1, .1, float('nan')], 'nonfinite_state'),
])
def test_scan_failure_has_bounded_evidence_and_keeps_nonfinite_physics(tmp_path, position, reason):
    course = course_fixture(tmp_path, torch.tensor([position]))
    with pytest.raises(RuntimeError, match='privileged terrain scan') as caught:
        course.scan_77()
    report = json.loads(str(caught.value).split(': ', 1)[1])
    diagnostic = report['query_diagnostics']
    assert diagnostic['failed_columns'] == 77 and len(diagnostic['samples']) == 8
    assert {item['reason'] for item in diagnostic['samples']} == {reason}
    assert course.query.queries == 77


@pytest.mark.parametrize('secondary', [RuntimeError('diagnostic failure'), SystemExit(0)])
def test_secondary_diagnostic_errors_do_not_mask_the_ground_failure(tmp_path, secondary):
    course = course_fixture(tmp_path, torch.tensor([[.9, .9, .3]]))

    def fail(*_args, **_kwargs):
        raise secondary

    course.query._diagnose_misses = fail
    with pytest.raises(RuntimeError, match='no physical terrain support') as caught:
        course.ground()
    assert any(type(secondary).__name__ in note for note in caught.value.__notes__)
    assert course.env._ground_at_root.eq(9).all()


def test_a_fallen_body_queries_real_ground_without_running_diagnostics(tmp_path):
    course = course_fixture(tmp_path, torch.tensor([[.1, .1, -6.]]))

    def forbidden(*_args, **_kwargs):
        raise AssertionError('valid support must not enter the diagnostic path')

    course.query._diagnose_misses = forbidden
    ground, wheels = course.ground()
    torch.testing.assert_close(ground, torch.tensor([.2]))
    torch.testing.assert_close(wheels, torch.full((1, 2), .2))
    scan, center = course.scan_77()
    assert torch.isfinite(scan).all()
    torch.testing.assert_close(center, torch.tensor([.2]))
    assert course._ground_token == 17


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


def cached_course(tmp_path):
    course = course_fixture(tmp_path, torch.tensor([[.1, .1, .3], [.2, .2, .3]]))
    course._ground_ready = torch.zeros(2, dtype=torch.bool)
    course._scan_ready = torch.zeros(2, dtype=torch.bool)
    course.query_invalid = torch.zeros(2, dtype=torch.bool)
    course.query_rejected_intervals = torch.zeros((), dtype=torch.long)
    course._scan_heights = torch.zeros(2, 77)
    course._scan_token = None
    course.ground()
    course.scan_77()
    course.env._physical_tick += 1
    return course


def test_one_established_outside_row_is_excluded_without_fabricating_a_hit(tmp_path):
    course = cached_course(tmp_path)
    course.env._root_link_position()[0, 0] = 1.1
    before_rng = torch.get_rng_state().clone()
    ground, _ = course.ground()
    invalid = course.finish_queries()
    assert invalid.tolist() == [True, False]
    assert course.query_rejected_intervals == 1
    assert course.query.misses > 0
    evidence = course._query_rejection_evidence
    assert set(evidence['environment'].tolist()) == {0}
    assert len(evidence['column']) <= 8
    # The placeholder is the last measured support, never a zero-height hit.
    torch.testing.assert_close(ground, torch.full((2,), .2))
    queries = course.query.queries
    course.scan_77()
    assert course.query.queries == queries
    assert torch.equal(before_rng, torch.get_rng_state())
    course.env._root_link_position()[0, 0] = .1
    assert evidence['root_position_world_m'][:, 0].eq(1.1).all()


@pytest.mark.parametrize('failure', ['hole', 'nonfinite', 'initial', 'whole_batch', 'index'])
def test_quarantine_does_not_hide_asset_or_systemic_errors(tmp_path, failure):
    course = cached_course(tmp_path)
    position = course.env._root_link_position()
    position[0] = torch.tensor([1.1, .1, .3])
    if failure == 'hole':
        position[0, :2] = .9
    elif failure == 'nonfinite':
        position[0, 2] = float('nan')
    elif failure == 'initial':
        course._ground_ready[0] = False
    elif failure == 'whole_batch':
        position[1, 0] = 1.2
    else:
        course.query._buffers['offsets'].zero_()
    with pytest.raises(RuntimeError, match='no physical terrain support'):
        course.ground()
    assert not course.query_invalid.any()



@pytest.mark.parametrize('finish,isolated', [(.5, True), (1., False)])
def test_miss_beyond_the_declared_course_is_a_row_exit_and_inside_it_a_hole(tmp_path, finish, isolated):
    course = cached_course(tmp_path)
    # The physical tile is larger than the authored course rectangle.
    course.router.starts = torch.tensor([-1.])
    course.router.finishes = torch.tensor([finish])
    course.router.half_widths = torch.tensor([1.])
    course.env._root_link_position()[0, :2] = .9  # inside physical bounds, beyond x = 0.5
    if isolated:
        course.ground()
        assert course.finish_queries().tolist() == [True, False]
    else:
        with pytest.raises(RuntimeError, match='terrain'):
            course.ground()
            course.finish_queries()

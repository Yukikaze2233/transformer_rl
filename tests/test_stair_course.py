"""Stair course classification and double-step gate tests."""
import pytest

from transformer_rl.formal.tasks.stair_course import StairClass, StairCourseSpec


def test_stair_classes_cover_curb_step_and_200mm_ledge():
    spec = StairCourseSpec()
    assert spec.classify(0.) == StairClass.CURB
    assert spec.classify(.06) == StairClass.LOW_STEP
    assert spec.classify(.12) == StairClass.MID_STEP
    assert spec.classify(.17) == StairClass.HIGH_STEP
    assert spec.classify(.20) == StairClass.SINGLE_LEDGE
    assert spec.task(StairClass.SINGLE_LEDGE).body_bump
    assert spec.task(StairClass.SINGLE_LEDGE).rise_m == .20
    assert spec.task(StairClass.DOUBLE_LEDGE).step_count == 2
    assert spec.task(StairClass.DOUBLE_LEDGE).platform_depth_m == .80


def test_double_step_requires_attempts_and_success_fraction():
    spec = StairCourseSpec(double_unlock_min_attempts=10, double_unlock_success_fraction=.8)
    assert not spec.double_unlocked(7, 10)
    assert not spec.double_unlocked(8, 9)
    assert spec.double_unlocked(8, 10)
    assert spec.double_unlocked(80, 100)
    with pytest.raises(ValueError):
        spec.double_unlocked(11, 10)


def test_height_and_depth_are_not_interchangeable():
    spec = StairCourseSpec()
    with pytest.raises(ValueError):
        spec.classify(.21)
    assert spec.task(StairClass.SINGLE_LEDGE).rise_m == .20
    assert spec.task(StairClass.SINGLE_LEDGE).platform_depth_m == .80

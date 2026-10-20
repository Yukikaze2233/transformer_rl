"""自适应地形等级和指令包络的纯逻辑单测。"""
import torch

from transformer_rl.formal.tasks.adaptive import (
    CommandRangeConfig, CommandRangeCurriculum, TerrainLevelConfig, TerrainLevelTracker,
)


def test_terrain_levels_only_change_on_reset_and_respect_distance_and_tracking():
    tracker = TerrainLevelTracker(8, seed=7, config=TerrainLevelConfig(initial_max_level=0))
    before = tracker.levels.clone()
    update = tracker.update(torch.zeros(8, dtype=torch.bool), torch.full((8,), 4.0),
                            torch.zeros(8))
    assert torch.equal(tracker.levels, before)
    assert not update.moved_up.any() or not update.moved_down.any()

    tracker.levels.fill_(3)
    update = tracker.update(torch.ones(8, dtype=torch.bool), torch.full((8,), 4.0),
                            torch.ones(8))
    assert update.moved_up.all() and not update.moved_down.any()
    assert torch.equal(tracker.levels, torch.full((8,), 4))

    tracker.levels.fill_(3)
    update = tracker.update(torch.ones(8, dtype=torch.bool), torch.zeros(8),
                            torch.full((8,), 0.2))
    assert update.moved_down.all() and not update.moved_up.any()
    assert torch.equal(tracker.levels, torch.full((8,), 2))


def test_terrain_top_level_randomizes_only_reset_rows_and_checkpoint_roundtrips():
    tracker = TerrainLevelTracker(16, seed=3, config=TerrainLevelConfig(max_level=2, initial_max_level=0))
    tracker.levels.fill_(2)
    reset = torch.zeros(16, dtype=torch.bool)
    reset[:4] = True
    update = tracker.update(reset, torch.full((16,), 8.0), torch.ones(16))
    assert update.reached_top[:4].all() and not update.reached_top[4:].any()
    assert (tracker.levels[4:] == 2).all()
    saved = tracker.state_dict()
    restored = TerrainLevelTracker(16, seed=99, config=tracker.config)
    restored.load_state_dict(saved)
    assert torch.equal(restored.levels, tracker.levels)
    torch.testing.assert_close(restored.update(reset, torch.zeros(16), torch.full((16,), .2)).levels,
                               tracker.update(reset, torch.zeros(16), torch.full((16,), .2)).levels)


def test_command_envelope_expands_shrinks_and_is_bounded():
    curriculum = CommandRangeCurriculum(CommandRangeConfig(
        max_forward_m_s=2.7, max_yaw_rad_s=6.0, initial_forward_m_s=1.5,
        initial_yaw_rad_s=1.5, min_forward_m_s=.5, min_yaw_rad_s=.3,
        forward_step_m_s=.1, yaw_step_rad_s=.2,
    ))
    assert curriculum.state.forward_m_s == (-1.5, 1.5)
    expanded = curriculum.update(.8, 0.0)
    assert expanded.expanded and expanded.forward_m_s == (-1.6, 1.6)
    assert expanded.yaw_rad_s == (-1.7, 1.7)
    for _ in range(100):
        curriculum.update(1.0, 0.0)
    assert curriculum.state.forward_m_s == (-2.7, 2.7)
    assert curriculum.state.yaw_rad_s == (-6.0, 6.0)
    shrunk = curriculum.update(0.0, .5)
    assert shrunk.shrunk and shrunk.forward_m_s == (-2.5, 2.5)
    assert shrunk.yaw_rad_s == (-5.6, 5.6)
    for _ in range(100):
        curriculum.update(0.0, 1.0)
    assert curriculum.state.forward_m_s == (-.5, .5)
    assert curriculum.state.yaw_rad_s == (-.3, .3)

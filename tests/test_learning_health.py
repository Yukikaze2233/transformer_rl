"""Synthetic checks for the learning-health guard; no simulator or robot is used."""
import json

import pytest

from transformer_rl.formal.learning.loop import (
    ActorUpdateHealth, ActorUpdateStalled, last_actor_zero_streak,
)
from transformer_rl.formal.learning.optimization import _is_cuda_fatal


def test_actor_update_health_counts_only_attempted_zero_retention_updates():
    health = ActorUpdateHealth(limit=3)
    assert health.observe({'actor_endpoint_uses': 10, 'retained_actor_steps': 0}) == 1
    assert health.observe({'actor_endpoint_uses': 10, 'retained_actor_steps': 0}) == 2
    # A script-only update has no attempted actor transaction and must not count.
    assert health.observe({'actor_endpoint_uses': 0, 'retained_actor_steps': 0}) == 0
    assert health.observe({'actor_endpoint_uses': 10, 'retained_actor_steps': 1}) == 0


def test_actor_update_health_fails_after_the_declared_streak():
    health = ActorUpdateHealth(limit=2)
    health.observe({'actor_endpoint_uses': 10, 'retained_actor_steps': 0})
    health.observe({'actor_endpoint_uses': 10, 'retained_actor_steps': 0})
    with pytest.raises(ActorUpdateStalled, match='2 consecutive updates'):
        health.require_healthy()


def test_last_actor_zero_streak_reads_the_last_committed_metrics_row(tmp_path):
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text(
        json.dumps({'update': 1, 'optimization': {'actor_consecutive_zero_updates': 3}}) + '\n' +
        json.dumps({'update': 2, 'optimization': {'actor_consecutive_zero_updates': 4}}) + '\n',
        encoding='utf-8')
    assert last_actor_zero_streak(metrics) == 4
    metrics.write_text(json.dumps({'update': 3, 'optimization': {}}) + '\n', encoding='utf-8')
    assert last_actor_zero_streak(metrics) == 0
    assert last_actor_zero_streak(tmp_path / 'missing.jsonl') == 0
    metrics.write_text('not-json\n', encoding='utf-8')
    assert last_actor_zero_streak(metrics) == 0


def test_cuda_error_classification_does_not_hide_other_failures():
    assert _is_cuda_fatal(RuntimeError('CUDA error: unknown error'))
    assert _is_cuda_fatal(RuntimeError('cudaErrorUnknown'))
    assert not _is_cuda_fatal(ValueError('ordinary configuration error'))

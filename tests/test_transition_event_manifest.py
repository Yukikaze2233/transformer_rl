"""Large inherited histories must fit the existing small control protocol."""
import json

from transformer_rl.formal.runtime.transition import _inherited_events_summary


def test_large_event_index_is_bounded_and_still_binds_every_hash():
    events = {f'{index:08d}.json': f'{index:064x}' for index in range(10000)}
    summary = _inherited_events_summary(events)
    assert len(json.dumps(events)) > 256 * 1024
    assert len(json.dumps(summary)) < 128
    assert summary['count'] == 10000
    assert summary == _inherited_events_summary(dict(reversed(list(events.items()))))
    events['00000001.json'] = 'f' * 64
    assert summary['sha256'] != _inherited_events_summary(events)['sha256']

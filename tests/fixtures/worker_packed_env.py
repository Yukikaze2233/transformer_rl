"""Synthetic SDK ownership contract exercised by a real frozen worker process."""
import sys

from packed_env import make_env as make_packed_env


class OwnedApplication:
    def close(self, *, wait_for_replicator, exit_code):
        assert wait_for_replicator is False
        assert type(exit_code) is int


def make_env(model_config, environment_config, device):
    from transformer_rl import frame_process

    assert frame_process is sys.modules["__main__"]
    frame_process.require_worker()
    frame_process.register_app(OwnedApplication())
    return make_packed_env(model_config, environment_config, device)

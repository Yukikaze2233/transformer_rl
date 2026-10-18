"""A synchronous CPU fixture with observable episode state and bounded delay."""
import json
from pathlib import Path
import time

from packed_env import PackedFixture


class ControlledFixture(PackedFixture):
    def __init__(self, model_config, environment_config, device):
        super().__init__(model_config, environment_config, device)
        self.metadata.update(environment_config.get("fixture_provenance", {}))

    def step(self, issued_action):
        time.sleep(self.options.get("step_delay", .01))
        return super().step(issued_action)

    def close(self):
        path = self.options.get("capture_file")
        if path:
            Path(path).write_text(json.dumps({"tick": self.tick, "state": self.state.tolist(),
                "previous": self.previous.tolist(), "age": self.age.tolist()}))
        super().close()


def make_env(model_config, environment_config, device):
    return ControlledFixture(model_config, environment_config, device)

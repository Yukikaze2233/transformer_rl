"""Inspect a deterministic packed-frame policy without starting an environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = FramePolicyConfig.from_dict(json.loads(args.config.read_text()))
    policy = FramePolicy(config)
    print(json.dumps({
        "config": config.to_dict(),
        "mean_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "input": ["batch", config.history_length, config.frame_dim],
        "flat_input": ["batch", config.history_length * config.frame_dim],
        "output": ["batch", config.action_dim],
        "input_scaling": "already scaled by the observation contract; no rescaling",
        "history_order": "oldest_to_newest_including_current",
        "history_reset": "caller_owned; repeat the first frame for the V6 comparison",
        "scope": "deterministic mean network; excludes Gaussian, critic and optimizer",
        "runner_adapter_required": True,
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()

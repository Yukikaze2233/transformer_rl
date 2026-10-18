"""Kaiser CPU comparisons against the authenticated original learner package."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from transformer_rl.training_compat import instrument_package
from transformer_rl.training_control import TrainingSession
from test_frame_continuation import assert_same, configuration
from test_training_control import wait_for


@pytest.mark.parametrize("architecture", ["mlp", "history_mlp", "transformer"])
def test_original_learner_and_controlled_copy_have_identical_learning_and_episode(tmp_path, monkeypatch, architecture):
    source = os.environ.get("TRAINCTL_FROZEN_SOURCE")
    if not source:
        pytest.skip("requires an explicitly authenticated original learner on Kaiser")
    source = Path(source)
    original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*.py") if "__pycache__" not in p.parts}
    instrumented = tmp_path / "controlled/transformer_rl"
    proof = instrument_package(source, instrumented, original)
    assert proof["changed_existing_files"] == ["frame_cli.py", "frame_workflow.py"]
    for name in original.keys() - {"frame_cli.py", "frame_workflow.py"}:
        assert hashlib.sha256((instrumented / name).read_bytes()).hexdigest() == original[name]
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    config = configuration(architecture)
    capture = tmp_path / "episode.json"
    config = replace(config, environment={**config.environment, "capture_file": str(capture), "step_delay": .01})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    fixtures = Path(__file__).parent / "fixtures"
    reference = tmp_path / "reference"
    code = ("import torch; torch.set_num_threads(1); "
            "from transformer_rl.frame_config import FrameTrainConfig; "
            "from transformer_rl.frame_workflow import train_frame_policy; "
            "from controlled_env import make_env; "
            f"train_frame_policy(FrameTrainConfig.load({str(path)!r}),make_env,'controlled_env:make_env',"
            f"{str(reference)!r},updates=6,seed=71,rollout_steps=4,tensorboard=False)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(source.parent), str(fixtures)])})
    assert result.returncode == 0, result.stderr
    original_episode = json.loads(capture.read_text())
    session = TrainingSession.create(tmp_path / "session", path, env_factory="controlled_env:make_env",
        updates=6, rollout_steps=4, seed=71, checkpoint_interval=6,
        python_paths=[fixtures], learner_source=instrumented)
    session.start(paused=True)
    try:
        wait_for(session, lambda state: state["phase"] == "paused")
        time.sleep(.15)
        session.command("resume", wait_seconds=10)
        state = wait_for(session, lambda value: value["phase"] == "completed" and not value["active"])
    finally:
        if session.status()["active"]:
            session.command("stop", wait_seconds=20)
    expected = torch.load(reference / "checkpoints/final.pt", map_location="cpu", weights_only=True)
    observed = torch.load(state["checkpoint"], map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "rng", "update", "config"):
        assert_same(expected[key], observed[key])
    assert json.loads(capture.read_text()) == original_episode

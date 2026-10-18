"""Exercise frozen acceptance on new CPU fixtures, never archived experiments."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from transformer_rl.training_compat import instrument_package
from transformer_rl.training_control import (TrainingSession, boot_id, file_pin,
                                            read_json, write_json)
from test_frame_continuation import configuration
from test_training_control import wait_for


def test_frozen_endpoint_accepts_real_stopped_and_resumed_learning(tmp_path, monkeypatch):
    source = os.environ.get("TRAINCTL_FROZEN_SOURCE")
    verifier = os.environ.get("TRAINCTL_FROZEN_VERIFIER")
    if not source or not verifier:
        pytest.skip("requires authenticated frozen learner and acceptance source on Kaiser")
    source = Path(source)
    original = {str(p.relative_to(source)): file_pin(p)["sha256"]
                for p in source.rglob("*.py") if "__pycache__" not in p.parts}
    root = tmp_path / "queue"
    root.mkdir()
    learner = tmp_path / "learner/src/transformer_rl"
    learner.parent.mkdir(parents=True)
    proof = instrument_package(source, learner, original)
    identity = {"files": proof["controlled_files"], "sha256": hashlib.sha256(json.dumps(
        proof["controlled_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    contract = tmp_path / "contract.json"
    contract.write_text('{"fixture":"CPU synthetic provenance"}\n')
    from transformer_rl.frame_config import digest
    config = configuration("mlp")
    provenance = {"identity": "synthetic_tensor_fixture", "policy_hz": 100., "physics_hz": 200.,
                  "contract_sha256": digest(json.loads(contract.read_text())),
                  "startup": {"scene_group_counts": {"synthetic": 1024}}}
    config = replace(config, ppo=replace(config.ppo, epochs=5, num_minibatches=32, target_kl=100.),
        environment={**config.environment, "num_envs": 1024, "step_delay": .02,
                     "snapshot": str(tmp_path), "contract": "contract.json",
                     "contract_sha256": file_pin(contract)["sha256"],
                     "snapshot_sha256": "synthetic_tensor_fixture", "fixture_provenance": provenance})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "transfer_design.json").write_text(json.dumps({
        "scene_allocation": {"requested_counts": {"synthetic": 1024}}}))
    runtime = subprocess.run([sys.executable, "-c", "import json,platform,torch,numpy; print(json.dumps({"
        "'python':platform.python_version(),'torch':str(torch.__version__),'numpy':numpy.__version__,"
        "'cuda':torch.version.cuda,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}))"],
        capture_output=True, text=True, check=True)
    initial = subprocess.run([sys.executable, "-c",
        "import torch; from transformer_rl.frame_config import FrameTrainConfig; "
        "from transformer_rl.frame_training import FrameActorCritic; "
        "from transformer_rl.frame_workflow import _model_state_sha256; torch.manual_seed(71); "
        f"print(_model_state_sha256(FrameActorCritic(FrameTrainConfig.load({str(path)!r}).model)))"],
        env={**os.environ, "PYTHONPATH": str(source.parent)}, capture_output=True, text=True, check=True)
    session = TrainingSession.create(root / "sessions/fixture", path,
        env_factory="controlled_env:make_env", updates=4, rollout_steps=48, seed=71,
        checkpoint_interval=1200, learner_source=learner,
        python_paths=[Path(__file__).parent / "fixtures"],
        expected_initial_model_sha256=initial.stdout.strip(), max_seconds=120)
    queue = SimpleNamespace(root=root, spec={"authorized_boot": boot_id(), "learner_root": str(learner.parents[1]),
        "learner_identity": identity, "verifier": file_pin(verifier),
        "runtime": {"checkpoint_runtime": json.loads(runtime.stdout)}},
        parent_manifest={"output_root": str(root), "device": "cpu", "inputs": {"root": str(tmp_path)}})
    import transformer_rl.training_queue_validation as validation
    original_loader = validation.load_helpers

    def load_fixture_helpers(pin, dependencies=()):
        module, predicate, predicate_source = original_loader(pin, dependencies)
        # The synthetic factory is a test fixture, never a production predicate.
        predicate.__globals__["FACTORY"] = "controlled_env:make_env"
        return module, predicate, predicate_source

    monkeypatch.setattr(validation, "load_helpers", load_fixture_helpers)
    session.start()
    try:
        wait_for(session, lambda value: value.get("update", 0) >= 1)
        stopped = session.command("stop", wait_seconds=20)
        assert stopped["phase"] == "stopped" and 1 <= stopped["update"] < 4
        process = session.submitted_process
        assert process.wait(timeout=20) == 2
        first = session.attempt()
        write_json(first / "exit.json", {"identity": read_json(first / "launch.json"), "returncode": 2})
        evidence = validation.validate_training(queue, session, first)
        assert evidence["full_learning_state_validated"] is True
        assert evidence["cumulative_transitions"] == stopped["update"] * 49152
        session.start(paused=True)
        resumed = wait_for(session, lambda value: value["phase"] == "paused")
        assert resumed["update"] == stopped["update"]
        session.command("resume", wait_seconds=20)
        wait_for(session, lambda value: value["phase"] == "completed" and not value["active"])
        assert session.submitted_process.wait(timeout=20) == 0
        last = session.attempt()
        write_json(last / "exit.json", {"identity": read_json(last / "launch.json"), "returncode": 0})
        final = validation.validate_training(queue, session, last, evidence)
        assert final["final_update"] == 4 and final["cumulative_transitions"] == 4 * 49152
        assert final["cumulative_optimizer_steps"] == 4 * 160
    finally:
        if session.status()["active"]:
            session.command("stop", wait_seconds=20)
            session.submitted_process.wait(timeout=20)


def test_evaluation_pause_preserves_process_and_stop_finishes_window(tmp_path):
    from threading import Thread
    from transformer_rl.training_control import EvaluationControl, _lease
    root = tmp_path / "evaluation"
    root.mkdir()
    control = root / "control"
    control.mkdir()
    descriptor = _lease(root)
    write_json(root / "request.json", {"protocol": "transformer_rl.training_control.v1", "token": "fixture",
        "boot_id": boot_id(), "root": str(root), "attempt": str(root), "lease_fd": descriptor,
        "start_update": 0, "target_update": 9, "prior_active_elapsed_s": 0, "prior_transitions": 0})
    write_json(control / "request.json", {"token": "fixture", "sequence": 0, "operation": "pause"})
    try:
        observer = EvaluationControl(control)
        thread = Thread(target=lambda: observer.boundary(3, 9))
        thread.start()
        import time
        deadline = time.monotonic() + 5
        while read_json(control / "status.json")["phase"] != "paused" and time.monotonic() < deadline:
            time.sleep(.01)
        assert thread.is_alive()
        write_json(control / "request.json", {"token": "fixture", "sequence": 1, "operation": "stop"}, replace=True)
        thread.join(5)
        assert not thread.is_alive() and observer.state["phase"] == "stopping"
        assert not (control / "result.json").exists()
        observer.boundary(8, 9)
        observer.finish_evaluation(9)
        assert read_json(control / "result.json")["summary"]["update"] == 9
    finally:
        os.close(descriptor)

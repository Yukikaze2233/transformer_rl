"""CPU process/interface checks; no robot assets, SDK, GPU or other sessions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time

import pytest
import torch

from transformer_rl.frame_checkpoint import load_frame_checkpoint
from transformer_rl.frame_workflow import train_frame_policy
from transformer_rl.training_cli import main, render
from transformer_rl.training_control import TrainingSession, read_json, write_json

from test_frame_continuation import assert_same, configuration

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from controlled_env import make_env


def wait_for(session, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = session.status()
        if predicate(state):
            return state
        if state["phase"] in ("failed", "interrupted"):
            log = Path(state["log"]).read_text()[-4000:]
            pytest.fail(f"{state}\n{log}")
        time.sleep(.03)
    pytest.fail(f"worker did not reach expected state: {session.status()}")


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    sessions = []

    def create(architecture="history_mlp", updates=10, **options):
        config = configuration(architecture)
        config = replace(config, environment={**config.environment, "step_delay": .015,
                         "capture_file": str(tmp_path / f"capture-{len(sessions)}.json")})
        config_path = tmp_path / f"config-{len(sessions)}.json"
        config_path.write_text(json.dumps(config.to_dict()))
        session = TrainingSession.create(tmp_path / f"session {len(sessions)}", config_path,
            env_factory="controlled_env:make_env", python=sys.executable, updates=updates,
            seed=71, rollout_steps=4, checkpoint_interval=3, max_seconds=options.pop("max_seconds", 30),
            python_paths=[Path(__file__).parent / "fixtures"], **options)
        sessions.append(session)
        return session, config

    yield create
    for session in sessions:
        if session.status()["active"]:
            session.command("stop", wait_seconds=30)
            wait_for(session, lambda state: not state["active"])
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("architecture", ["mlp", "history_mlp", "transformer"])
def test_live_pause_matches_uninterrupted_model_adam_rng_and_episode(session_factory, tmp_path, architecture):
    session, config = session_factory(architecture)
    session.start(paused=True)
    first = wait_for(session, lambda state: state["phase"] == "paused")
    assert first["update"] == 0 and first["checkpoint_update"] == 0
    session.command("resume", wait_seconds=20)
    wait_for(session, lambda state: state.get("update", 0) >= 2)
    paused = session.command("pause", wait_seconds=20)
    assert paused["phase"] == "paused" and 2 <= paused["update"] < 10
    time.sleep(.25)
    again = session.status()
    assert (again["update"], again["cumulative_transitions"], again["identity"]) == (
        paused["update"], paused["cumulative_transitions"], paused["identity"])
    assert abs(again["active_elapsed_s"] - paused["active_elapsed_s"]) < .05
    assert Path(paused["checkpoint"]).is_file()
    session.command("resume", wait_seconds=20)
    done = wait_for(session, lambda state: state["phase"] == "completed" and not state["active"])
    observed = load_frame_checkpoint(done["checkpoint"])
    episode = json.loads(Path(config.environment["capture_file"]).read_text())
    reference = train_frame_policy(config, make_env, "controlled_env:make_env", tmp_path / "reference",
        updates=10, seed=71, rollout_steps=4, checkpoint_interval=3, tensorboard=False)
    expected = load_frame_checkpoint(reference["checkpoint"])
    assert_same(observed[0].state_dict(), expected[0].state_dict())
    assert_same(observed[1].optimizer.state_dict(), expected[1].optimizer.state_dict())
    assert_same(observed[5], expected[5])
    assert episode == json.loads(Path(config.environment["capture_file"]).read_text())
    assert done["cumulative_transitions"] == 120
    assert done["paused_elapsed_s"] >= .25


def test_stop_commits_before_ack_and_cold_resume_preserves_learning_budget(session_factory):
    session, _ = session_factory(updates=14)
    session.start()
    wait_for(session, lambda state: state.get("update", 0) >= 2)
    stopped = session.command("stop", wait_seconds=20)
    assert stopped["phase"] == "stopped"
    assert read_json(session.attempt() / "control" / "result.json")
    assert 2 <= stopped["update"] < 14
    wait_for(session, lambda state: not state["active"])
    parent = load_frame_checkpoint(stopped["checkpoint"])
    first_attempt = session.attempt()
    session.start(paused=True)
    resumed = wait_for(session, lambda state: state["phase"] == "paused")
    assert session.attempt() != first_attempt
    assert resumed["update"] == stopped["update"]
    assert resumed["session_active_elapsed_s"] > stopped["session_active_elapsed_s"]
    actual = load_frame_checkpoint(resumed["checkpoint"])
    assert_same(parent[0].state_dict(), actual[0].state_dict())
    assert_same(parent[1].optimizer.state_dict(), actual[1].optimizer.state_dict())
    assert_same(parent[5], actual[5])
    assert actual[4]["episode_state_restored"] is False
    session.command("resume", wait_seconds=20)
    done = wait_for(session, lambda state: state["phase"] == "completed" and not state["active"])
    assert done["update"] == 14 and done["cumulative_transitions"] == 168
    assert session.start()["phase"] == "completed"


def test_concurrent_start_and_repeated_commands_keep_one_worker(session_factory):
    session, _ = session_factory()
    with ThreadPoolExecutor(2) as executor:
        list(executor.map(lambda _: session.start(paused=True), range(2)))
    state = wait_for(session, lambda state: state["phase"] == "paused")
    assert len(list((session.root / "attempts").iterdir())) == 1
    for _ in range(3):
        assert session.command("pause", wait_seconds=1)["identity"] == state["identity"]
    assert session.command("stop", wait_seconds=20)["phase"] == "stopped"


def test_changed_boot_never_polls_retired_pid_and_requires_explicit_start(session_factory, monkeypatch):
    session, _ = session_factory()
    session.start(paused=True)
    wait_for(session, lambda state: state["phase"] == "paused")
    session.command("stop", wait_seconds=20)
    wait_for(session, lambda state: not state["active"])
    import transformer_rl.training_control as control
    with monkeypatch.context() as changed:
        changed.setattr(control, "boot_id", lambda: "another-boot")
        changed.setattr(control, "process_identity", lambda _: pytest.fail("retired PID was polled"))
        assert session.status()["boot_changed"]
        with pytest.raises(ValueError, match="boot changed"):
            session.start()


def test_changed_source_or_checkpoint_refuses_restart(session_factory):
    session, _ = session_factory()
    source = session.root / "source" / "transformer_rl" / "training_control.py"
    original = source.read_bytes()
    source.write_bytes(original + b"\n# changed\n")
    with pytest.raises(ValueError, match="artifact changed"):
        session.start()
    source.write_bytes(original)
    session.start(paused=True)
    wait_for(session, lambda state: state["phase"] == "paused")
    state = session.command("stop", wait_seconds=20)
    wait_for(session, lambda state: not state["active"])
    Path(state["checkpoint"] + ".json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        session.start()


def test_unclean_attempt_cannot_silently_replay_a_periodic_checkpoint(session_factory):
    session, _ = session_factory()
    attempt = session.root / "attempts" / "0001-123456789abc"
    attempt.mkdir()
    (attempt / "control").mkdir()
    from transformer_rl.training_control import boot_id
    write_json(attempt / "request.json", {"token": "fixture", "boot_id": boot_id(), "submitted_epoch": 0})
    write_json(attempt / "control" / "request.json", {"token": "fixture", "operation": "resume", "sequence": 0})
    write_json(session.root / "current.json", {"attempt": attempt.name})
    assert session.status()["phase"] == "interrupted"
    with pytest.raises(ValueError, match="unclean exit"):
        session.start()


def test_recipe_pin_cannot_redirect_reads_outside_the_session(session_factory, monkeypatch):
    session, _ = session_factory()
    write_json(session.root / "session.pin.json", {"path": "/outside/forbidden.pt"}, replace=True)
    import transformer_rl.training_control as control
    monkeypatch.setattr(control, "verify_pin", lambda _: pytest.fail("outside artifact was read"))
    with pytest.raises(ValueError, match="escapes the session"):
        TrainingSession(session.root)


def test_pause_time_does_not_exhaust_the_training_deadline(session_factory):
    session, _ = session_factory(updates=2, max_seconds=3)
    session.start(paused=True)
    wait_for(session, lambda state: state["phase"] == "paused")
    time.sleep(3.2)
    assert session.status()["phase"] == "paused"
    session.command("resume", wait_seconds=20)
    done = wait_for(session, lambda state: state["phase"] == "completed" and not state["active"])
    assert done["paused_elapsed_s"] >= 3.2 and done["active_elapsed_s"] < 3


def test_dashboard_detach_and_ctrl_c_do_not_stop_the_worker(session_factory):
    session, _ = session_factory()
    session.start(paused=True)
    state = wait_for(session, lambda state: state["phase"] == "paused")
    entry = str(Path(__file__).resolve().parents[1] / "tools/training.py")
    for interrupt in (False, True):
        master, slave = pty.openpty()
        ui = subprocess.Popen([sys.executable, entry, "watch", str(session.root)],
            stdin=slave, stdout=slave, stderr=slave, env={**os.environ, "TERM": "xterm-256color"})
        os.close(slave)
        try:
            output = b""
            deadline = time.monotonic() + 10
            while b"TRAINING" not in output and time.monotonic() < deadline:
                if select.select([master], [], [], .2)[0]:
                    output += os.read(master, 65536)
            assert b"TRAINING" in output and b"\x1b[?1049h" in output
            if interrupt:
                ui.send_signal(signal.SIGINT)
            else:
                os.write(master, b"q")
            assert ui.wait(timeout=10) == (130 if interrupt else 0)
            assert session.status()["identity"] == state["identity"]
            assert session.status()["phase"] == "paused"
        finally:
            if ui.poll() is None:
                ui.terminate()
                ui.wait(timeout=10)
            os.close(master)


def test_frontend_is_torch_free_json_and_terminal_safe(session_factory, capsys):
    session, _ = session_factory(name="模型\x1b[31m")
    assert main(["status", str(session.root), "--json"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["phase"] == "ready"
    rendered = render(state, width=64, color=False)
    assert "\x1b" not in rendered and "训练控制" in rendered
    from transformer_rl.training_cli import _width
    assert all(_width(line) == 64 for line in rendered.splitlines())
    assert main(["watch", str(session.root)]) == 0
    assert "\x1b" not in capsys.readouterr().out
    result = subprocess.run([sys.executable, "-c",
        "import sys; from transformer_rl.training_cli import main; "
        "assert main([]) == 0; assert 'torch' not in sys.modules"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr

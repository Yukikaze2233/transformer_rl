"""CLI orchestration tests use inert factories, not a simulator or training task."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from transformer_rl import cli
from transformer_rl.config import ModelConfig, PPOConfig, load_config


def test_inspect_reports_actual_architecture_without_factory(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_factory", lambda *_: pytest.fail("inspection started an environment"))
    assert cli.main(["inspect"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frame_dim"] == 30
    assert result["actor_parameters"] == 69772
    assert result["critic_parameters"] == 48897
    assert result["environment_started"] is False


@pytest.mark.parametrize("section,field,value", [
    ("model", "d_model", 63), ("model", "initial_std", True),
    ("ppo", "normalize_advantage", "false"), ("ppo", "gamma", True),
    ("ppo", "learning_rate", float("nan")), ("model", "unknown", 1),
])
def test_invalid_configuration_rejected(tmp_path, section, field, value):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({section: {field: value}}))
    with pytest.raises(ValueError):
        load_config(config)


def test_example_configuration_loads():
    path = Path(__file__).resolve().parents[1] / "configs/control.json"
    model, ppo, environment = load_config(path)
    assert model == ModelConfig()
    assert ppo == PPOConfig()
    assert environment == {}


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    """Replace collection, optimization and publication with inert call records."""
    import transformer_rl.checkpoint as checkpoint
    import transformer_rl.ppo as ppo
    import transformer_rl.runner as runner

    events = []
    state = SimpleNamespace(mode="normal", saved=[], env=None)
    factory_module = ModuleType("orchestration_fixture")

    class Environment:
        def close(self):
            events.append("close")

    def create_env(**kwargs):
        assert set(kwargs) == {"model_config", "environment_config", "device"}
        events.append("factory")
        state.env = Environment()
        return state.env

    factory_module.create_env = create_env
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)

    class Collector:
        def __init__(self, env, model, ppo_config, action_clip):
            self.total_transitions = 0
            self.last_metrics = {}

        def reset(self, seed=None):
            events.append("reset")

        def collect(self, steps, should_stop):
            events.append("collect")
            if state.mode == "stop":
                should_stop.__self__.reason = "SIGTERM"
                return None
            self.total_transitions += steps * 2
            self.last_metrics = {"transitions": steps * 2}
            return object()

    class Trainer:
        def __init__(self, model, config):
            self.config = config

        def update(self, batch):
            events.append("update")
            if state.mode == "failure":
                raise RuntimeError("scripted optimizer failure")
            return {"optimizer_steps": 1, "loss": 0.0}

    def save(path, model, trainer, update, metadata):
        events.append("save")
        state.saved.append((update, metadata))
        return {"path": str(path), "update": update, "sha256": "fixture"}

    monkeypatch.setattr(runner, "RolloutCollector", Collector)
    monkeypatch.setattr(ppo, "PPOTrainer", Trainer)
    monkeypatch.setattr(checkpoint, "save_checkpoint", save)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": asdict(ModelConfig()), "ppo": asdict(PPOConfig())}))
    run_dir = tmp_path / "run"
    arguments = ["train", "--config", str(config), "--env-factory", "orchestration_fixture:create_env",
                 "--run-dir", str(run_dir), "--updates", "2", "--rollout-steps", "3",
                 "--checkpoint-interval", "1"]
    return arguments, run_dir, state, events


def test_explicit_orchestration_publishes_after_each_update(orchestration):
    args, directory, state, events = orchestration
    assert cli.main(args) == 0
    report = json.loads((directory / "completion.json").read_text())
    assert report["updates_completed"] == 2
    assert report["collected_transitions"] == 12
    assert [update for update, _ in state.saved] == [1, 2]
    assert events == ["factory", "reset", "collect", "update", "save", "collect", "update", "save", "close"]
    assert len((directory / "metrics.jsonl").read_text().splitlines()) == 2
    assert json.loads((directory / "run.json").read_text())["metadata"]["diagnostics"] is False
    assert all(metadata["diagnostics"] is False for _, metadata in state.saved)


def test_diagnostics_keyword_and_metadata_are_opt_in(orchestration, monkeypatch):
    from transformer_rl import ppo

    args, directory, state, events = orchestration
    calls = []

    def update(self, batch, *, diagnostics):
        calls.append(diagnostics)
        events.append("update")
        return {"optimizer_steps": 0, "first_step_kl": None, "final_kl": 0.0}

    monkeypatch.setattr(ppo.PPOTrainer, "update", update)
    assert cli.main([*args, "--diagnostics"]) == 0
    assert calls == [True, True]
    assert json.loads((directory / "run.json").read_text())["metadata"]["diagnostics"] is True
    assert all(metadata["diagnostics"] is True for _, metadata in state.saved)
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert all(row["optimization"]["first_step_kl"] is None for row in rows)


def test_signal_boundary_saves_without_optimization(orchestration):
    args, directory, state, events = orchestration
    state.mode = "stop"
    assert cli.main(args) == 0
    report = json.loads((directory / "completion.json").read_text())
    assert report["status"] == "stopped"
    assert report["stop_reason"] == "SIGTERM"
    assert report["updates_completed"] == 0
    assert "update" not in events
    assert events[-2:] == ["close", "save"]


def test_failed_optimization_marks_partial_state_and_closes(orchestration):
    args, directory, state, events = orchestration
    state.mode = "failure"
    assert cli.main(args) == 1
    report = json.loads((directory / "failure.json").read_text())
    assert report["optimizer_update_may_be_partial"] is True
    assert report["cumulative_completed_update"] == 0
    assert not (directory / "completion.json").exists()
    assert not state.saved
    assert events[-1] == "close"


def test_existing_run_is_preserved(orchestration):
    args, directory, _, events = orchestration
    directory.mkdir()
    sentinel = directory / "user.txt"
    sentinel.write_text("user work")
    assert cli.main(args) == 1
    assert sentinel.read_text() == "user work"
    assert events == []

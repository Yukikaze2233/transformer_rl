"""训练控制台只读取冻结配置与小状态卡，不启动学习或读取模型。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from transformer_rl.common import training_cli


@pytest.fixture
def session(tmp_path):
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({
        "collection": {"policy_dt_s": .02}, "rollout_steps": 24,
        "ppo": {"epochs": 5, "minibatches": 4}}))
    raw = recipe_path.read_bytes()
    state = dict(name="formal", root=str(tmp_path), phase="collecting", active=True,
                 target_update=35000, update=1954, architecture="structured_transformer",
                 seed=2102, device="cuda:0", structured=True, num_envs=4096,
                 history_length=5, checkpoint_update=None,
                 latest_checkpoint=str(tmp_path / "update_01900.pt"),
                 solver_position_iterations=12, solver_velocity_iterations=6)
    return SimpleNamespace(root=tmp_path, status=lambda: dict(state),
                           spec={"pins": [{"path": str(recipe_path), "bytes": len(raw),
                                           "sha256": hashlib.sha256(raw).hexdigest()}]})


def test_dashboard_uses_frozen_recipe_and_published_checkpoint(session):
    state = training_cli._display_state(session)
    output = training_cli.render(state)
    assert "4096 环境 × 24 步" in output
    assert "策略 50Hz" in output and "历史跨度 80 ms" in output
    assert "5 epochs × 4 minibatches = 20" in output
    assert "最近保存  update 1900" in output
    assert not (session.root / "update_01900.pt").exists()
    assert session.status()["checkpoint_update"] is None


def test_dashboard_does_not_assume_timing_without_frozen_metadata(session):
    session.spec["pins"] = []
    output = training_cli.render(training_cli._display_state(session))
    assert "Hz" not in output and "跨度" not in output


def test_legacy_research_dashboard_does_not_read_structured_recipe(session):
    state = session.status()
    state.pop("structured")
    session.spec = {}
    assert training_cli._display_state(session, state) == state


def test_dashboard_rejects_changed_recipe_instead_of_showing_wrong_timing(session):
    (session.root / "recipe.json").write_text('{}')
    with pytest.raises(ValueError, match="artifact changed"):
        training_cli._display_state(session)


def test_status_and_noninteractive_watch_use_same_configuration(session, monkeypatch, capsys):
    monkeypatch.setattr(training_cli, "open_session", lambda _: session)
    assert training_cli.main(["status", "formal", "--json"]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["policy_dt_s"] == .02 and state["checkpoint_update"] == 1900
    assert training_cli.main(["watch", "formal"]) == 0
    assert "策略 50Hz" in capsys.readouterr().out


def test_frontend_import_does_not_import_learning_or_simulator():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import transformer_rl.common.training_cli; "
        "assert not any(k == 'torch' or k.startswith(('isaaclab', 'isaacsim')) "
        "for k in sys.modules)"],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr

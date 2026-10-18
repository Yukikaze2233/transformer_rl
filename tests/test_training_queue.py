"""Queue orchestration checks using private synthetic files and no model runs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformer_rl.training_control import _lease, boot_id, file_pin
from transformer_rl.training_queue import TrainingQueue, cell_key


def prepare(tmp_path):
    source = tmp_path / "learner/src/transformer_rl"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    config = tmp_path / "prepared/config.json"
    config.parent.mkdir()
    config.write_text("{}")
    cells = [{"rate_id": "rate", "variant": f"network_{i}", "training_seed": 1101,
              "training_config": {"path": "config.json", "sha256": file_pin(config)["sha256"]},
              "updates": 1200, "rollout_steps": 48,
              "initial_model_sha256": "a" * 64} for i in range(90)]
    manifest = {"sha256": "b" * 64, "output_root": str(tmp_path / "original"),
                "inputs": {"cells": cells, "root": str(config.parent), "protocol": {"evaluation": {}}},
                "development_seeds": [701, 1701, 2701, 3701], "device": "cpu"}
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(manifest))
    rows = {cell_key(cell): {"status": "incomplete" if i < 37 else "completed",
        "training": {"charged_updates": 1200, "completed_updates": 0 if i < 37 else 1200},
        "development": {str(s): None if i < 37 else {"status": "completed"} for s in manifest["development_seeds"]}}
        for i, cell in enumerate(cells[:39])}
    summary = tmp_path / "inherited.json"
    summary.write_text(json.dumps({"manifest_sha256": manifest["sha256"], "results": rows}))
    locks = []
    for index in range(4):
        path = tmp_path / f"resource-{index}.lock"
        path.touch()
        st = path.stat()
        locks.append({"path": str(path), "device": st.st_dev, "inode": st.st_ino, "uid": st.st_uid})
    verifier = tmp_path / "verifier.py"
    verifier.write_text("# Synthetic verifier identity\n")
    plan = {"parent_manifest": file_pin(parent), "inherited_summary": file_pin(summary),
            "expected_total_cells": 90, "retained_keys": [cell_key(cell) for cell in cells[:40]],
            "unstarted_keys": [cell_key(cell) for cell in cells[40:]],
            "authorized_boot": boot_id(), "blocked": {cell_key(cells[39]): {"verified_updates_lower_bound": 387,
                "checkpoint_available": False, "charged_updates": 1200}}, "python": os.sys.executable,
            "runtime": {}, "environment": {}, "learner_root": str(source.parents[1]),
            "locks": locks, "verifier": file_pin(verifier)}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, cells, rows


def test_adoption_skips_all_old_work_and_runs_original_remaining_grid_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan, cells, inherited = prepare(tmp_path)
    queue = TrainingQueue.adopt(tmp_path / "queue", plan, name="research")
    assert queue.status()["queue_counts"] == {"completed": 2, "retained_incomplete": 38, "not_started": 50, "total": 90}
    trained, evaluated = [], []
    import transformer_rl.training_queue as module
    def create(root, *_, **kwargs):
        root.mkdir(parents=True)
        assert kwargs["updates"] == 1200 and kwargs["rollout_steps"] == 48
        return SimpleNamespace(root=root, spec=kwargs)
    monkeypatch.setattr(module.TrainingSession, "create", create)
    def train(session, descriptors):
        trained.append(session.root.name)
        assert len(descriptors) == 5 and all(os.fstat(fd) for fd in descriptors)
        return {"final_update": 1200, "checkpoint": {"path": "/unused/synthetic.pt", "sha256": "d" * 64}}
    monkeypatch.setattr(queue, "_wait_training", train)
    monkeypatch.setattr(queue, "_evaluation", lambda job, checkpoint, seed, fds: evaluated.append((cell_key(job), seed)))
    descriptor = _lease(queue.root)
    directory = queue.root / "controllers/test"
    directory.mkdir()
    from transformer_rl.training_control import write_json
    write_json(queue.root / "request.json", {"operation": "resume", "sequence": 0})
    try:
        assert queue.run(descriptor, directory) == 0
    finally:
        os.close(descriptor)
    assert len(trained) == 50
    assert evaluated == [(cell_key(cell), seed) for cell in cells[40:] for seed in [701, 1701, 2701, 3701]]
    rows = queue.jobs()
    assert [row["status"] for row in rows[:37]] == ["incomplete"] * 37
    assert [row["status"] for row in rows[37:39]] == ["completed"] * 2
    assert rows[39]["status"] == "interrupted_no_checkpoint"
    assert all(row["status"] == "completed" for row in rows[40:])
    assert queue.status()["phase"] == "incomplete"


def test_adoption_refuses_relabeling_or_restarting_a_previously_started_cell(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan, cells, _ = prepare(tmp_path)
    original = tmp_path / "original/cells" / cell_key(cells[40]) / "training/attempt_0000"
    original.mkdir(parents=True)
    with pytest.raises(ValueError, match="already has an attempt"):
        TrainingQueue.adopt(tmp_path / "queue", plan, name="research")


def test_registered_queue_works_in_cli_and_changed_boot_never_polls_old_owner(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan, _, _ = prepare(tmp_path)
    queue = TrainingQueue.adopt(tmp_path / "queues/research", plan, name="research")
    from transformer_rl.training_cli import main
    assert main(["jobs", "research", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 90
    assert main(["list", str(queue.root.parent), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["queue_counts"]["completed"] == 2
    import transformer_rl.training_queue as module
    monkeypatch.setattr(module, "boot_id", lambda: "changed-boot")
    monkeypatch.setattr(module, "process_identity", lambda _: pytest.fail("retired owner was polled"))
    assert queue.status()["boot_changed"]
    with pytest.raises(ValueError, match="new audited adoption"):
        queue.start(accept_new_boot=True)


def test_idle_queue_stop_and_restart_hold_all_inherited_leases(tmp_path, monkeypatch):
    import time
    from transformer_rl.training_control import read_json
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan_path, _, _ = prepare(tmp_path)
    script = tmp_path / "gpu-query"
    script.write_text(f"#!{os.sys.executable}\nprint('synthetic-gpu, 0')\n")
    script.chmod(0o700)
    plan = json.loads(plan_path.read_text())
    plan["resource_gate"] = {"nvidia_smi": str(script), "gpu_uuid": "synthetic-gpu",
        "minimum_gpu_free_mib": 1, "minimum_ram_available_bytes": 1, "minimum_disk_available_bytes": 1}
    plan_path.write_text(json.dumps(plan))
    queue = TrainingQueue.adopt(tmp_path / "queue", plan_path, name="research")
    try:
        owners = []
        for _ in range(2):
            queue.start()
            deadline = time.monotonic() + 10
            while queue.status()["phase"] != "waiting_resources" and time.monotonic() < deadline:
                time.sleep(.02)
            state = queue.status()
            assert state["phase"] == "waiting_resources" and state["resources"]["gpu_free_mib"] == 0
            assert state["queue_counts"]["not_started"] == 50 and state["active_cell"] is None
            from transformer_rl.training_cli import render
            assert "可用 RAM" in render(state)
            owners.append(read_json(queue.root / "controller.json")["identity"])
            # An inherited FLOCK can retain its creator's PID after fork. Probe
            # exclusivity without sending signals or consulting other processes.
            with pytest.raises(BlockingIOError):
                _lease(queue.root)
            state = queue.command("stop", wait_seconds=15)
            assert state["phase"] == "stopped" and not state["active"]
        assert owners[0] != owners[1]
    finally:
        if queue.status()["active"]:
            queue.command("stop", wait_seconds=15)

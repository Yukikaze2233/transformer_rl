"""Controller guards with tiny synthetic artifacts and actual OS flock handles.

Orchestration stubs do not establish teacher qualification or physical results.
The original qualification/preparation tests cover their full immutable inputs.
"""
from copy import deepcopy
from pathlib import Path
import fcntl
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import time
import types

import pytest

from transformer_rl import retention_campaign as campaign


def small_protocol(tmp_path):
    parent = tmp_path / "teacher.pt"
    parent.write_bytes(b"synthetic placeholder; never deserialized")
    primary, study = tmp_path / "resource.lock", tmp_path / "study.lock"
    primary.touch(); study.touch()
    branch = {"id": "branch_test", "schedule": "stationary", "training_seed": 71,
        "requested_k": 4, "coefficient": 0., "anchors": [], "consumed_update_offset": 2,
        "parent_checkpoint": {**campaign._receipt(parent), "update": 2, "cumulative_transitions": 24}}
    execution = {"updates": 2, "rollout_steps": 4, "transitions_per_update": 12,
        "fresh_transition_budget": 24, "consumed_update_budget": 2, "max_seconds": 60.,
        "checkpoint_interval": 1, "device": "cpu", "tensorboard": False,
        "worker_timeout_seconds": 61., "max_wait_seconds": .1, "poll_seconds": .01,
        "resource_lock": str(primary), "resource_locks": [campaign._lock_signature(p)
            for p in sorted([primary, study])], "dependencies": [{"role": role}
                for role in ("curriculum", "diagnostics", "learning")]}
    return campaign._sealed({"format": campaign.FORMAT, "schema_version": 1,
        "source": campaign.learner_source(), "output_root": str(tmp_path / "new_campaign"),
        "branches": [branch], "schedules": {"stationary": {"model": {}, "ppo": {},
            "control": {}, "environment": {"num_envs": 3}}}, "execution": execution,
        "environment_factory": "packed_env:make_env", "retention_seed": 901,
        "retention_batch_size": 2, "evaluation": {"seeds": [801, 802], "steps": 4,
            "trace_replicas": 8, "settle_steps": 0, "min_steady_samples": 1},
        "evaluation_cases": [{"name": "case", "config": "case.json"}],
        "study_root": str(tmp_path), "preparation_protocol_sha256": "a" * 64,
        "preparation_manifest_sha256": "b" * 64, "expected_branches": 1,
        "expected_evaluation_suites": 2, "expected_evaluation_cells": 2,
        "skill_scope": "synthetic controller interface only"})


def test_provider_executes_actual_bytes_even_with_valid_stale_pyc(tmp_path, monkeypatch):
    path = tmp_path / "prepare_nested_anchors.py"
    path.write_bytes(b"marker='old'\n")
    info = path.stat()
    py_compile.compile(str(path), doraise=True)
    path.write_bytes(b"marker='new'\n")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    monkeypatch.setattr(campaign, "TOOLS", tmp_path)
    assert campaign._preparer().marker == "new"


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, raw):
    path = tmp_path / "input.json"; path.write_bytes(raw)
    with pytest.raises(ValueError):
        campaign._read(path)


def test_not_ready_provider_cannot_touch_original_or_new_output(tmp_path, monkeypatch):
    provider = types.SimpleNamespace(validate_preparation=lambda *args: {
        "status": "not_ready", "ready_cells": 0, "expected_cells": 30})
    monkeypatch.setattr(campaign, "_preparer", lambda: provider)
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="must be ready"):
        campaign._context({}, tmp_path)
    assert list(tmp_path.iterdir()) == before


def test_branch_identity_cannot_conflate_bool_and_float(tmp_path):
    protocol = small_protocol(tmp_path)
    branch = deepcopy(protocol["branches"][0]); branch["coefficient"] = False
    with pytest.raises(ValueError, match="frozen grid"):
        campaign.build_worker_request(protocol, tmp_path / "protocol.json", branch, tmp_path / "train", [])


def test_sampler_and_budget_request_is_bound_to_preparation_and_schedule(tmp_path):
    protocol = small_protocol(tmp_path)
    branch = protocol["branches"][0]
    request = campaign.build_worker_request(protocol, tmp_path / "protocol.json", branch, tmp_path / "train", [])
    assert request["checkpoint"]["consumed_updates"] == 2
    assert request["execution"]["fresh_transition_budget"] == 24
    assert request["execution"]["resume"] is False
    assert request["retention"]["seed"] == 901
    assert request["provenance"]["branch_id"] == branch["id"]
    assert request["provenance"]["preparation_manifest_sha256"] == "b" * 64
    assert request["sha256"] == campaign._digest({k:v for k,v in request.items() if k != "sha256"})


def controller_files(protocol):
    root = Path(protocol["output_root"]); root.mkdir()
    process = campaign._process(os.getpid())
    campaign._write_new(root / "controller.json", {"protocol_sha256": protocol["sha256"],
        "source": protocol["source"], "process": {k: process[k] for k in ("pid", "start", "argv")}})
    return process


@pytest.mark.parametrize("mutation", [None, "unlocked", "wrong_inode", "wrong_parent", "descriptor_bool"])
def test_actual_inherited_lock_lease_positive_and_negative(tmp_path, mutation):
    protocol = small_protocol(tmp_path); process = controller_files(protocol)
    leases, descriptors = [], []
    try:
        for pin in protocol["execution"]["resource_locks"]:
            fd = os.open(pin["path"], os.O_RDWR); descriptors.append(fd)
            if mutation != "unlocked":fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            leases.append({**pin, "descriptor": fd, "controller_pid": process["pid"],
                           "controller_start": process["start"]})
        if mutation == "wrong_inode":leases[0]["inode"] += 1
        if mutation == "wrong_parent":leases[0]["controller_start"] += 1
        if mutation == "descriptor_bool":leases[0]["descriptor"] = True
        payload = tmp_path / "lease.json"; payload.write_text(json.dumps([protocol, leases]))
        code = "import json,sys;from transformer_rl.retention_campaign import validate_lease;p,l=json.load(open(sys.argv[1]));validate_lease(p,l)"
        result = subprocess.run([sys.executable, "-B", "-c", code, str(payload)],
            env=campaign._environment(tmp_path), pass_fds=tuple(descriptors), capture_output=True, timeout=15.)
        assert (result.returncode == 0) == (mutation is None), result.stderr.decode()
    finally:
        for fd in descriptors:os.close(fd)


def test_resource_wait_does_not_create_or_replace_original_lock(tmp_path, monkeypatch):
    from transformer_rl import queue_validation
    protocol = small_protocol(tmp_path)
    before = deepcopy(protocol["execution"]["resource_locks"])
    monkeypatch.setattr(queue_validation, "check_dependency", lambda *args, **kwargs: {"status": "pending"})
    with pytest.raises(ValueError, match="wait expired"):
        with campaign._resource_lease(protocol, lambda *args, **kwargs: None):
            pytest.fail("pending queues must not authorize a worker")
    assert [campaign._lock_signature(p["path"]) for p in before] == before


def test_held_resource_is_not_stolen_after_predecessor_closes(tmp_path, monkeypatch):
    from transformer_rl import queue_validation
    protocol = small_protocol(tmp_path)
    monkeypatch.setattr(queue_validation, "check_dependency", lambda *args, **kwargs: {"status": "completed"})
    with open(protocol["execution"]["resource_locks"][0]["path"], "r+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="wait expired"):
            with campaign._resource_lease(protocol, lambda *args, **kwargs: None):
                pytest.fail("held lock cannot be stolen")


def test_child_timeout_is_terminal_and_never_restarted(tmp_path):
    directory = tmp_path / "worker"; directory.mkdir()
    command = [sys.executable, "-B", "-c", "import time;time.sleep(30)"]
    with pytest.raises(ValueError, match="no retry"):
        campaign._launch(command, directory, [], .05, lambda *args, **kwargs: None)
    receipt = campaign._read(directory / "worker.completion.json")
    assert receipt["timed_out"] is True and receipt["returncode"] < 0
    assert campaign._process(receipt["process"]["pid"]) is None
    assert len(list(directory.glob("worker.process.json"))) == 1


def test_unrecorded_child_is_stopped_through_its_kernel_handle(tmp_path, monkeypatch):
    directory = tmp_path / "worker"; directory.mkdir()
    spawned = []
    actual_popen = campaign.subprocess.Popen
    def popen(*args, **kwargs):
        process = actual_popen(*args, **kwargs); spawned.append(process)
        return process
    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    monkeypatch.setattr(campaign, "_process", lambda pid: None)
    with pytest.raises(ValueError, match="handle vanished"):
        campaign._launch([sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                         directory, [], 1., lambda *args, **kwargs: None)
    assert len(spawned) == 1 and spawned[0].returncode < 0
    assert campaign._read(directory / "worker.completion.json")["process"] is None


def test_unavailable_kernel_process_handles_fail_before_spawning(tmp_path, monkeypatch):
    directory = tmp_path / "worker"; directory.mkdir()
    def missing(*args):raise OSError("synthetic unavailable pidfd")
    monkeypatch.setattr(campaign, "_pidfd_api", lambda: (missing, missing))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(OSError, match="unavailable pidfd"):
        campaign._launch([sys.executable, "-c", "pass"], directory, [], 1., lambda *args, **kwargs: None)


def test_new_outputs_cannot_overlap_original_inputs_or_source_trees(tmp_path):
    protected = tmp_path / "original" / "source"
    protected.mkdir(parents=True)
    for destination in (protected, protected / "new_run", protected.parent):
        with pytest.raises(ValueError, match="overlaps"):
            campaign._guard_output(destination, [str(protected)])
    alias = tmp_path / "alias"; alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        campaign._guard_output(alias / "new_protocol.json", [str(protected)])
    campaign._guard_output(tmp_path / "separate" / "new_run", [str(protected)])


@pytest.mark.parametrize("device", ["cuda:00", "cuda", "cuda:-1", True])
def test_noncanonical_device_is_rejected_before_reading_or_creating_inputs(tmp_path, device):
    with pytest.raises(ValueError, match="device must be"):
        campaign.freeze(tmp_path / "missing_protocol.json", tmp_path / "missing_anchors",
            output_root=tmp_path / "new_campaign", retention_seed=901,
            diagnostic_dependency=tmp_path / "missing_diag", learning_dependency=tmp_path / "missing_lr",
            device=device)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [["--help"], ["freeze", "--help"], ["run", "--help"]])
def test_cli_help_exits_successfully_without_a_failure_traceback(arguments, capsys):
    with pytest.raises(SystemExit) as caught:
        campaign.main(arguments)
    assert caught.value.code == 0
    output = capsys.readouterr()
    assert "usage:" in output.out and output.err == ""


def test_evaluation_command_preserves_full_declared_case_and_replica_coverage(tmp_path):
    protocol = small_protocol(tmp_path)
    protocol["evaluation"].update(steps=4001, seeds=[8701, 9701], trace_replicas=8,
                                  settle_steps=200, min_steady_samples=200)
    protocol["evaluation_cases"] = [{"name": f"case_{i}", "config": f"case_{i}.json"} for i in range(50)]
    command = campaign._evaluation_command(protocol, {"path": "/fake/checkpoint.pt"}, 8701, tmp_path)
    assert command[command.index("--trace-replicas") + 1] == "8"
    assert command[command.index("--steps") + 1] == "4001"
    assert command[command.index("--settle-steps") + 1] == "200"
    assert command.index("--outputs") - command.index("--configs") - 1 == 50
    assert command.index("--steps") - command.index("--outputs") - 1 == 50


def test_existing_campaign_directory_cannot_be_retried(tmp_path, monkeypatch):
    protocol = small_protocol(tmp_path)
    Path(protocol["output_root"]).mkdir()
    path = tmp_path / "protocol.json"; campaign._write_new(path, protocol)
    monkeypatch.setattr(campaign, "validate_protocol", lambda value: value)
    with pytest.raises(ValueError, match="no automatic retry"):
        campaign.run_protocol(path)


def test_source_budget_and_extra_field_changes_cannot_be_equivalent(tmp_path):
    protocol = small_protocol(tmp_path)
    for mutate in (lambda p:p["execution"].update(updates=True),
                   lambda p:p.update(extra="unsupported"),
                   lambda p:p["source"].update(sha256="0"*64)):
        changed = deepcopy(protocol); mutate(changed)
        with pytest.raises(ValueError, match="changed"):
            campaign._equivalent(changed, protocol)


@pytest.mark.parametrize("fail_training", [False, True])
def test_synthetic_grid_accounts_reservation_without_refunding_failure(tmp_path, monkeypatch, fail_training):
    # Only orchestration is exercised. These stubs cannot authorize real teachers.
    from contextlib import contextmanager
    from transformer_rl import retention_evaluation
    protocol = small_protocol(tmp_path)
    first = protocol["branches"][0]
    second = deepcopy(first); second["id"] = "branch_second"
    protocol["branches"].append(second)
    protocol.update(expected_branches=2, expected_evaluation_suites=4, expected_evaluation_cells=4)
    protocol = campaign._sealed({k:v for k,v in protocol.items() if k != "sha256"})
    path = tmp_path / "protocol.json"; campaign._write_new(path, protocol)
    monkeypatch.setattr(campaign, "validate_protocol", lambda value: value)
    @contextmanager
    def lease(*args):yield []
    monkeypatch.setattr(campaign, "_resource_lease", lease)
    launched = []
    def launch(command, directory, *args):
        launched.append(command)
        if fail_training:raise RuntimeError("synthetic failed worker")
        return {"returncode": 0}
    monkeypatch.setattr(campaign, "_launch", launch)
    monkeypatch.setattr(campaign, "verify_training", lambda *args: {
        "actual_samples": 24, "checkpoint": {"path": str(tmp_path / "placeholder.pt")}})
    monkeypatch.setattr(retention_evaluation, "verify_suite", lambda *args: {"status": "synthetic"})
    if fail_training:
        with pytest.raises(RuntimeError, match="failed worker"):campaign.run_protocol(path)
    else:campaign.run_protocol(path)
    summary = campaign._read(Path(protocol["output_root"]) / "summary.json")
    assert summary["charged_updates"] == (2 if fail_training else 4)
    assert summary["reserved_samples"] == (24 if fail_training else 48)
    assert summary["verified_actual_samples"] == (0 if fail_training else 48)
    assert summary["status"] == ("incomplete" if fail_training else "completed")
    assert len(launched) == (1 if fail_training else 6)
    assert len(summary["results"]) == (1 if fail_training else 2)
    assert summary["formal_architecture_selection"] is False
    assert summary["hardware_verified"] is False
    if fail_training:
        assert summary["results"][first["id"]]["status"] == "incomplete"
        assert not (Path(protocol["output_root"]) / second["id"]).exists()
        with pytest.raises(ValueError, match="no automatic retry"):campaign.run_protocol(path)

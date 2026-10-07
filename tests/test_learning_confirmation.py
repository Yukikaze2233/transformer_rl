"""CPU synthetic protocol/fault tests; never run a simulator or an optimizer."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("confirmation_under_test", ROOT / "tools/run_learning_confirmation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
REAL_VERIFY_OUTPUTS = runner.diagnostic.verify_outputs
FIXTURE_SPEC = importlib.util.spec_from_file_location("confirmation_selection_fixture", ROOT / "tests/test_learning_selection.py")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    """Synthetic 90-cell choice bridge; development physics is a declared stub."""
    ready = fixtures.ready.__wrapped__(tmp_path, monkeypatch)
    selection, campaign, control = fixtures.selection, fixtures.campaign, fixtures.control
    for name, value in (("selection", selection), ("campaign", campaign), ("control", control),
                        ("diagnostic", selection.diagnostic), ("require", selection.require), ("read", selection.read)):
        monkeypatch.setattr(runner, name, value)
    manifest = ready["manifest"]
    learning_root = Path(manifest["output_root"])
    resource, study = tmp_path / "shared-resource.lock", tmp_path / "old-study/.run.lock"
    for path in (resource, study, learning_root / ".learning.lock"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    manifest.update(resource_lock=str(resource), study_lock=str(study), dependencies={},
        locks={str(path): runner.diagnostic.lock_identity(path) for path in (resource, study, learning_root / ".learning.lock")})
    fixtures.put(learning_root / "manifest.json", manifest)
    definition = ready["definition"]
    definition["campaign_manifest"] = campaign.artifact(learning_root / "manifest.json")
    fixtures.resign(ready["selection_path"], definition)
    for key, cell in ready["audit"]["cells"].items():
        for seed, entry in cell["development"].items():
            directory = Path(entry["receipt"]["path"]).parent
            report = fixtures.physical_report()
            fixtures.put(directory / "control.json", report)
            entry["receipt"] = fixtures.put(directory / "receipt.json", {"status": "completed",
                "artifacts": {"control": control.artifact(directory / "control.json", learning_root)}})
    fixtures.put(learning_root / "controller.json", {"pid": 999999993, "start": "1", "status": "running",
                 "manifest_sha256": manifest["sha256"]})
    audit_item = fixtures.put(learning_root / "audit.json", ready["audit"])
    fixtures.put(learning_root / "summary.json", {"status": "completed", "manifest_sha256": manifest["sha256"],
        "grid_status": "development_complete", "audit": audit_item, "expected_cells": 90,
        "results": {key: {"status": "completed"} for key in ready["audit"]["cells"]}})
    monkeypatch.setattr(campaign, "runtime_identity", lambda source: deepcopy(manifest["runtime"]))
    if getattr(request, "param", None) == "mlp_no_eligible":
        for rate in ("rate_000", "rate_001", "rate_002"):
            for train in (1101, 1102, 1103):
                for seed in campaign.DEVELOPMENT_SEEDS:
                    ready["overrides"][(f"{rate}/mlp/seed_{train}", seed, ready["cases"][0])] = {"success_rate": .5}
    choice = selection.seal_choice(ready["selection_path"])
    confirmation_root = tmp_path / "confirmation"
    confirmation = selection.prepare_confirmation(ready["selection_path"],
        Path(ready["definition"]["output_root"]) / "choice.json", confirmation_root)
    path = confirmation_root / "manifest.json"
    execution_root = tmp_path / "confirmation-execution"
    execution = runner.prepare(path, execution_root, max_wait_seconds=0., poll_seconds=.01, worker_timeout_seconds=2.)
    return {**ready, "choice": choice, "confirmation": confirmation, "confirmation_path": path,
            "execution": execution, "execution_path": execution_root / "manifest.json"}


def fake_worker(monkeypatch, fixture, *, failure=None):
    """Generate synthetic endpoint reports while exercising real runner audits."""
    campaign, selection = runner.campaign, runner.selection
    calls = []
    def work(request, directory, environment, timeout, publish):
        original = request["original"]
        calls.append(original)
        pid = 999999991
        process = {"format": "transformer_rl.learning_confirmation_worker", "schema_version": 1,
            "execution_sha256": request["execution_sha256"], "request_sha256": runner.control.file_sha(directory / "request.json"),
            "command": original["command"], "environment": request["environment"], "cache": request["cache"],
            "controller": request["controller"], "timeout_seconds": timeout, "status": "finished", "returncode": 0,
            "timed_out": False, "pid": pid, "start": "1", "recovery": {"remaining": []},
            "group_handles": [{"pid": pid, "start": "1", "pgid": pid}],
            "observed": {"pid": pid, "start": "1", "pgid": pid, "argv": original["command"], "environment": request["environment"]}}
        if failure == "timeout":
            process.update(returncode=-15, timed_out=True)
        elif failure == "nonzero":
            process["returncode"] = 3
        elif failure == "argv":
            process["observed"]["argv"] = ["wrong"]
        elif failure == "env":
            process["observed"]["environment"] = {**request["environment"], "PYTHONPYCACHEPREFIX": "/wrong"}
        elif failure == "live":
            process.update(pid=os.getpid(), start=runner.control.process_start(os.getpid()))
        campaign.write(directory / "worker.process.json", process)
        if failure == "exception":
            raise RuntimeError("synthetic worker failure")
        for case in original["configs"]:
            fixtures.put(directory / f"{case}.json", fixtures.physical_report())
        (directory / "trace.npz").write_bytes(b"synthetic trace, no simulator executed")
        fixtures.put(directory / "control.json", {"synthetic": True, "trace": {"path": str(directory / "trace.npz"),
            "sha256": runner.control.file_sha(directory / "trace.npz"), "steps": 4001}})
        return process
    def outputs(directory, shim, variant, seed):
        if failure == "missing_report":
            raise ValueError("one required report missing")
        return {name: runner.control.artifact(directory / filename, fixture["confirmation"]["output_root"])
                for name, filename in [(case, f"{case}.json") for case in fixture["cases"]]
                + [("control", "control.json"), ("trace", "trace.npz")]}
    monkeypatch.setattr(runner, "worker", work)
    monkeypatch.setattr(runner.diagnostic, "verify_outputs", outputs)
    return calls


def test_prepare_and_read_only_audit_do_not_queue_or_change_original(prepared):
    original = prepared["confirmation_path"].read_bytes()
    report = runner.audit(prepared["execution_path"])
    assert report["status"] == "not_ready" and report["expected_suites"] == 60 and report["completed_suites"] == 0
    assert len(report["missing"]) == 60 and prepared["confirmation_path"].read_bytes() == original
    assert prepared["confirmation"]["execution_implemented"] is False
    assert prepared["execution"]["execution_implemented"] is True
    assert not (Path(prepared["execution"]["output_root"]) / "controllers").exists()
    assert {request["evaluation_seed"] for request in prepared["confirmation"]["requests"]} == {11701, 12701}
    assert {request["training_seed"] for request in prepared["confirmation"]["requests"]} == {1101, 1102, 1103}


def test_serial_complete_confirmation_is_once_and_no_reselection(prepared, monkeypatch):
    calls = fake_worker(monkeypatch, prepared)
    old = prepared["confirmation_path"].read_bytes()
    result = runner.run(prepared["execution_path"])
    assert result["status"] == "completed" and result["confirmation_status"] == "confirmed"
    assert len(calls) == 60 and len({(r["cell"], r["evaluation_seed"]) for r in calls}) == 60
    report = runner.audit(prepared["execution_path"])
    assert report["completed_suites"] == 60 and report["original_choices"] == prepared["choice"]["choices"]
    assert report["no_reselection"] and not report["formal_architecture_selection"] and not report["hardware_deployment_ready"]
    assert prepared["confirmation_path"].read_bytes() == old
    runner.run(prepared["execution_path"])
    assert len(calls) == 60


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "argv", "env", "exception", "missing_report"])
def test_failed_attempt_is_sealed_missing_and_never_retried(prepared, monkeypatch, failure):
    calls = fake_worker(monkeypatch, prepared, failure=failure)
    request = prepared["confirmation"]["requests"][0]
    root = Path(prepared["execution"]["output_root"])
    controller_path = root / "controllers/controller_0000.process.json"
    handle = {"path": str(controller_path), "pid": 999999991, "start": "1"}
    fixtures.put(controller_path, {"execution_sha256": prepared["execution"]["sha256"], "pid": handle["pid"], "start": handle["start"]})
    receipt = runner.evaluate_request(prepared["execution"], prepared["confirmation"], prepared["manifest"], request, handle, lambda **k: None)
    assert receipt["status"] == "incomplete" and receipt["error"]
    directory = Path(prepared["confirmation"]["output_root"]) / request["directory"]
    assert not (directory / "receipt.json").exists()
    assert runner.audit(prepared["execution_path"])["status"] == "not_ready"
    assert runner.evaluate_request(prepared["execution"], prepared["confirmation"], prepared["manifest"], request, handle,
                                   lambda **k: None)["status"] == "already_attempted"
    assert len(calls) == 1


def test_unsealed_directory_and_unknown_attempt_cannot_be_retried(prepared, monkeypatch):
    calls = fake_worker(monkeypatch, prepared)
    original = prepared["confirmation"]["requests"][0]
    directory = Path(prepared["confirmation"]["output_root"]) / original["directory"]
    directory.mkdir(parents=True)
    audit = runner.audit(prepared["execution_path"])
    assert audit["attempts"][f"{original['cell']}/{original['evaluation_seed']}"]["status"] == "unsealed"
    assert runner.evaluate_request(prepared["execution"], prepared["confirmation"], prepared["manifest"], original, {},
                                   lambda **k: None)["status"] == "already_attempted"
    assert not calls
    (directory.parent / "attempt_0001").mkdir()
    with pytest.raises(ValueError, match="extra confirmation attempts"):
        runner.audit(prepared["execution_path"])


@pytest.mark.parametrize("reason", ["controller", "worker", "grid", "resource_lock"])
def test_live_predecessor_or_locked_resource_blocks_every_worker(prepared, monkeypatch, reason):
    calls = fake_worker(monkeypatch, prepared)
    hold = None
    if reason == "controller":
        old = runner.control.process_start
        monkeypatch.setattr(runner.control, "process_start", lambda pid: "1" if pid == 999999993 else old(pid))
    elif reason == "worker":
        monkeypatch.setattr(runner.campaign, "workers_in", lambda roots: [{"pid": 999, "start": "1"}])
    elif reason == "grid":
        original = runner.campaign.audit
        monkeypatch.setattr(runner.campaign, "audit", lambda path: {**original(path), "status": "not_ready"})
    else:
        import fcntl
        hold = Path(prepared["manifest"]["resource_lock"]).open("r+")
        fcntl.flock(hold, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if reason == "grid":
            with pytest.raises(ValueError, match="evidence changed after LR choice"):
                runner.run(prepared["execution_path"])
            assert not calls
            return
        result = runner.run(prepared["execution_path"])
        assert result["status"] == "waiting" and not calls
        assert not (Path(prepared["confirmation"]["output_root"]) / "evaluations").exists()
    finally:
        if hold:
            hold.close()


@pytest.mark.parametrize("target", ["helper", "closure", "runtime", "lock", "cache", "request_seed"])
def test_definition_input_runtime_lock_cache_tamper_fails_closed(prepared, monkeypatch, target):
    definition = runner.read(prepared["execution_path"])
    if target == "helper":
        definition["helpers"][str(ROOT / "tools/run_learning_confirmation.py")] = "0" * 64
        fixtures.resign(prepared["execution_path"], definition)
    elif target == "closure":
        path = next(iter(definition["input_closure"]))
        Path(path).write_bytes(Path(path).read_bytes() + b" ")
    elif target == "runtime":
        monkeypatch.setattr(runner.campaign, "runtime_identity", lambda source: {"different": True})
    elif target == "lock":
        route = prepared["manifest"]["resource_lock"]
        replacement = Path(route).with_suffix(".replacement")
        replacement.touch()
        replacement.replace(route)
    elif target == "cache":
        (Path(definition["cache"]["path"]) / "old.pyc").write_bytes(b"stale")
    else:
        confirmation = runner.read(prepared["confirmation_path"])
        confirmation["requests"][0]["evaluation_seed"] = 999
        fixtures.resign(prepared["confirmation_path"], confirmation)
        definition["confirmation_manifest"] = runner.campaign.artifact(prepared["confirmation_path"])
        fixtures.resign(prepared["execution_path"], definition)
    with pytest.raises((ValueError, KeyError)):
        runner.validate(prepared["execution_path"])


def test_complete_numeric_failure_keeps_selected_rate_and_is_not_confirmed(prepared, monkeypatch):
    fake_worker(monkeypatch, prepared)
    case = prepared["cases"][0]
    prepared["overrides"][("rate_000/mlp/seed_1101", 11701, case)] = {"success_rate": .5}
    result = runner.run(prepared["execution_path"])
    assert result["status"] == "completed" and result["confirmation_status"] == "not_confirmed"
    report = runner.audit(prepared["execution_path"])
    assert report["status"] == "not_confirmed" and report["original_choices"] == prepared["choice"]["choices"]
    assert report["original_choices"]["mlp"]["rate_id"] == "rate_000"


def test_attempt_report_or_worker_bytes_mutation_is_detected(prepared, monkeypatch):
    fake_worker(monkeypatch, prepared)
    runner.run(prepared["execution_path"])
    original = prepared["confirmation"]["requests"][0]
    directory = Path(prepared["confirmation"]["output_root"]) / original["directory"]
    (directory / "trace.npz").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="sealed output closure"):
        runner.audit(prepared["execution_path"])


def test_late_selector_check_never_observes_public_ready_receipt(prepared, monkeypatch):
    calls = fake_worker(monkeypatch, prepared)
    original = prepared["confirmation"]["requests"][0]
    directory = Path(prepared["confirmation"]["output_root"]) / original["directory"]
    root = Path(prepared["execution"]["output_root"])
    controller_path = root / "controllers/controller_0000.process.json"
    handle = {"path": str(controller_path), "pid": 999999991, "start": "1"}
    fixtures.put(controller_path, {"execution_sha256": prepared["execution"]["sha256"], "pid": handle["pid"], "start": handle["start"]})
    saw = []
    def rejected(*args, **kwargs):
        candidate = Path(args[3]["path"])
        assert candidate == directory / ".validation/receipt.json"
        assert not (directory / "receipt.json").exists()
        saw.append(candidate)
        raise ValueError("synthetic final selector model/provenance check failure")
    monkeypatch.setattr(runner.selection, "load_suite", rejected)
    result = runner.evaluate_request(prepared["execution"], prepared["confirmation"], prepared["manifest"], original, handle,
                                    lambda **k: None)
    assert result["status"] == "incomplete" and len(saw) == 1 and len(calls) == 1
    assert not (directory / "receipt.json").exists()
    assert (directory / ".validation/receipt.json").is_file()
    assert runner.evaluate_request(prepared["execution"], prepared["confirmation"], prepared["manifest"], original, handle,
                                   lambda **k: None)["status"] == "already_attempted"


def test_real_frozen_diagnostic_trace_path_bridge_preserves_original_bytes(tmp_path, tmp_path_factory):
    """Real NPY/NPZ schema and 50×8×4001 checks, with synthetic CPU arrays."""
    spec = importlib.util.spec_from_file_location("confirmation_real_diagnostic_fixture", ROOT / "tests/test_frame_diagnostic_campaign.py")
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    arrays = diagnostics.trace_arrays.__wrapped__(tmp_path_factory)
    cases = [f"case_{index:02d}" for index in range(50)]
    checkpoint = {"checkpoint_sha256": "a" * 64, "control_sha256": "b" * 64}
    shim = {"output_root": str(tmp_path), "inputs": {"cases": cases, "checkpoints": {"mlp": checkpoint},
        "environments": {"mlp": {case: {"synthetic": case} for case in cases}}, "snapshots": {"synthetic": {"sha256": "c" * 64}}}}
    directory = tmp_path / "attempt_0000"
    diagnostics.write_outputs(directory, shim, "mlp", 11701, arrays)
    original_bytes = (directory / "control.json").read_bytes()
    artifacts = REAL_VERIFY_OUTPUTS(directory, shim, "mlp", 11701)
    suite = {"identity": {"synthetic": True}, "status": "completed", "directory": "attempt_0000", "artifacts": artifacts}
    staging = runner.stage_candidate(directory, {"output_root": str(tmp_path)}, shim, "mlp", 11701, suite)
    assert not (directory / "receipt.json").exists()
    assert (directory / "control.json").read_bytes() == original_bytes
    shadow = runner.read(staging / "control.json")
    original = runner.read(directory / "control.json")
    assert shadow == {**original, "trace": {**original["trace"], "path": str(staging / "trace.npz")}}
    assert REAL_VERIFY_OUTPUTS(staging, shim, "mlp", 11701) == runner.read(staging / "receipt.json")["artifacts"]
    assert REAL_VERIFY_OUTPUTS(directory, shim, "mlp", 11701) == artifacts


def worker_input(tmp_path, command):
    directory, cache = tmp_path / "attempt", tmp_path / "cache"
    directory.mkdir()
    cache.mkdir()
    manifest = {"source_root": str(tmp_path / "source")}
    request = {"original": {"command": command, "directory": "synthetic"}, "execution_sha256": "1" * 64,
        "cache": runner.cache_identity(cache), "environment": runner.environment_recipe(manifest, cache), "controller": {}}
    runner.campaign.write(directory / "request.json", request)
    return directory, request, runner.run_environment(manifest, cache)


def test_real_cpu_worker_records_actual_bare_argv_env_and_terminal_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    command = [sys.executable, "-c", "import time; time.sleep(.3)"]
    directory, request, env = worker_input(tmp_path, command)
    process = runner.worker(request, directory, env, 3., lambda **k: None)
    assert process["command"] == process["observed"]["argv"] == command
    assert process["observed"]["environment"]["CUDA_VISIBLE_DEVICES"] == {"present": True, "value": "0"}
    assert process["status"] == "finished" and process["returncode"] == 0
    assert runner.control.process_start(process["pid"]) != process["start"]
    assert not list(Path(request["cache"]["path"]).iterdir())


def test_real_cpu_timeout_reclaims_parent_and_descendant_without_retry(tmp_path):
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); time.sleep(20)"
    directory, request, env = worker_input(tmp_path, [sys.executable, "-c", code])
    process = runner.worker(request, directory, env, .3, lambda **k: None)
    assert process["timed_out"] and process["returncode"] != 0
    assert not runner.group_members(process["pid"])
    assert not runner.selection.terminal_process(process, request["original"]["command"])


def test_reused_numeric_pgid_without_any_original_start_anchor_is_never_signalled(monkeypatch):
    stranger = {"pid": 555, "start": "new", "pgid": 555}
    monkeypatch.setattr(runner, "group_members", lambda pgid: [stranger])
    monkeypatch.setattr(runner.control, "process_start", lambda pid: "new")
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 555)
    kills = []
    monkeypatch.setattr(runner.os, "killpg", lambda *args: kills.append(args))
    with pytest.raises(ValueError, match="original observed group handle"):
        runner.reclaim_group(555, [{"pid": 555, "start": "old", "pgid": 555}])
    assert not kills


def test_original_observed_descendant_handle_can_anchor_cleanup(monkeypatch):
    original = {"pid": 556, "start": "original", "pgid": 555}
    members, kills = [original], []
    monkeypatch.setattr(runner, "group_members", lambda pgid: list(members))
    monkeypatch.setattr(runner.control, "process_start", lambda pid: "original")
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 555)
    def kill(pgid, sig):
        kills.append((pgid, sig))
        members.clear()
    monkeypatch.setattr(runner.os, "killpg", kill)
    assert runner.reclaim_group(555, [original])["remaining"] == []
    assert len(kills) == 1


def test_unresolved_original_group_blocks_remaining_requests_even_after_leader_exits(prepared, monkeypatch):
    calls = fake_worker(monkeypatch, prepared)
    original = prepared["confirmation"]["requests"][0]
    directory = Path(prepared["confirmation"]["output_root"]) / original["directory"]
    directory.mkdir(parents=True)
    fixtures.put(directory / "worker.process.json", {"status": "failed", "pid": 999999991, "start": "1",
        "group_handles": [{"pid": 999999994, "start": "original", "pgid": 999999991}],
        "recovery": {"ownership_unresolved": True, "remaining": [{"pid": 999999994, "start": "new", "pgid": 999999991}]}})
    assert runner.control.process_start(999999991) is None
    with pytest.raises(ValueError, match="original group"):
        runner.run(prepared["execution_path"])
    assert not calls


@pytest.mark.parametrize("prepared", ["mlp_no_eligible"], indirect=True)
def test_no_eligible_architecture_keeps_identity_and_receives_no_extra_request(prepared):
    assert prepared["choice"]["choices"]["mlp"]["status"] == "no_eligible_rate"
    assert len(prepared["confirmation"]["requests"]) == 54
    assert not any(request["cell"].split("/")[1] == "mlp" for request in prepared["confirmation"]["requests"])
    result = runner.audit(prepared["execution_path"])
    assert result["original_choices"]["mlp"]["status"] == "no_eligible_rate"


@pytest.mark.parametrize("values", [(0, 0, 1), (1, -1, 1), (1, 0, 31), (float("nan"), 0, 1), (True, 0, 1)])
def test_invalid_budget_or_timeout_cannot_be_prepared(values):
    with pytest.raises(ValueError, match="finite timeout"):
        runner.bounded_options(*values)


def test_ready_receipt_publication_is_atomic_exclusive_and_never_overwrites(tmp_path):
    payload = {"status": "completed", "synthetic": True}
    runner.publish_suite(tmp_path, payload)
    assert runner.read(tmp_path / "receipt.json") == payload
    old = (tmp_path / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.publish_suite(tmp_path, {"status": "different"})
    assert (tmp_path / "receipt.json").read_bytes() == old

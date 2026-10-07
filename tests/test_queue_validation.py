"""Small interface fixtures; fake helpers never qualify actual queue evidence.

Production resolves fixed historical helpers and validates all real artifacts.
The clearly patched providers below exercise association, counters, leases and
cache invalidation with tiny JSON files, without learners or simulator imports.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import py_compile
from types import SimpleNamespace

import pytest

from transformer_rl import queue_validation as queue


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(queue._bytes(value) + b"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resign(value):
    value["sha256"] = queue._digest({key: item for key, item in value.items() if key != "sha256"})
    return value


@pytest.fixture
def tiny_learning(tmp_path, monkeypatch):
    # These providers test only the new interface. They are not production
    # validators and cannot be selected by a definition or public argument.
    root, source, prepared = [tmp_path / name for name in ("queue", "source", "prepared")]
    root.mkdir()
    package = source / "src/transformer_rl"
    package.mkdir(parents=True)
    learner = package / "frame_process.py"
    learner.write_text("# Interface fixture; never executed as a learner.\n")
    prepared.mkdir()
    input_file = prepared / "frozen.json"
    write(input_file, {"synthetic_input": True})
    paths = [tmp_path / "resource.lock", tmp_path / "study.lock", root / ".learning.lock"]
    for path in paths:
        path.touch()
    cells = [{"rate_id": f"rate_{rate:03d}", "variant": f"variant_{variant:02d}", "training_seed": seed}
             for rate in range(3) for variant in range(10) for seed in (1101, 1102, 1103)]
    source_files = {"frame_process.py": sha(learner)}
    source_sha = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    definition = {"format": "transformer_rl.learning_campaign", "schema_version": 1,
        "output_root": str(root), "source_root": str(source),
        "inputs": {"root": str(prepared), "source": {"files": source_files, "sha256": source_sha},
                   "files": {"frozen.json": sha(input_file)}, "cells": cells},
        "resource_lock": str(paths[0]), "study_lock": str(paths[1]),
        "locks": {str(path): {key: queue.lock_identity(path)[key] for key in ("device", "inode")} for path in paths},
        "controllers": {}, "dependency_file": {"path": str(prepared / "dependencies.json"), "sha256": ""}}
    write(prepared / "dependencies.json", {"synthetic_declarations": True})
    definition["dependency_file"]["sha256"] = sha(prepared / "dependencies.json")
    resign(definition)
    write(root / "manifest.json", definition)
    write(root / "controller.json", {"pid": 99_999_999, "start": "123", "status": "running",
                                     "manifest_sha256": definition["sha256"]})
    checkpoint = root / "synthetic_endpoint.bin"
    checkpoint.write_bytes(b"tiny interface endpoint, not a tensor checkpoint")
    control_path = tmp_path / "run_frame_control_campaign.py"
    control_path.write_text("# Private fake helper for this interface fixture.\n")
    control = SimpleNamespace(__file__=str(control_path))
    owner = SimpleNamespace(receipts={}, modules=[control])
    calls = []

    def cell_key(cell):
        return f"{cell['rate_id']}/{cell['variant']}/seed_{cell['training_seed']}"

    training = {"status": "completed", "completed_updates": 1200, "charged_updates": 1200,
        "verified_fresh_samples": 58_982_400,
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint), "update": 1200}}
    results = {cell_key(cell): {"status": "completed", "training": deepcopy(training),
        "development": {str(seed): {"status": "completed"} for seed in (701, 1701, 2701, 3701)}} for cell in cells}
    report = {"manifest_sha256": definition["sha256"], "status": "development_complete",
        "expected_cells": 90, "actual_cells": 90, "completed_training_cells": 90,
        "completed_development_cells": 90, "cells": deepcopy(results)}
    development_file = root / "synthetic_development_receipt.json"
    write(development_file, {"status": "completed"})
    # Reusing this tiny interface receipt does not model a real suite. The real
    # original audit verifies each unique cell/seed/command/artifact itself.
    for result in report["cells"].values():
        for item in result["development"].values():
            item["receipt"] = {"path": str(development_file), "sha256": sha(development_file)}

    def validate(path):
        calls.append("validate")
        return deepcopy(definition)

    def audit(path):
        calls.append("audit")
        control.file_sha(checkpoint)
        control.read(input_file)
        return deepcopy(report)

    primary = SimpleNamespace(validate=validate, audit=audit, cell_key=cell_key, control=control)
    monkeypatch.setattr(queue, "_providers", lambda role, value: (owner, primary, primary))
    monkeypatch.setattr(queue, "_process_start", lambda pid: None)
    monkeypatch.setattr(queue, "_PROOFS", {})
    summary = {"status": "waiting", "manifest_sha256": definition["sha256"], "results": results,
               "grid_status": "development_complete", "audit": {"path": str(root / "audit.json"), "sha256": ""}}
    write(root / "audit.json", report)
    summary["audit"]["sha256"] = sha(root / "audit.json")
    write(root / "summary.json", summary)
    return SimpleNamespace(root=root, definition=definition, summary=summary, report=report,
        source=source, input_file=input_file, checkpoint=checkpoint, calls=calls,
        complete=lambda: write(root / "summary.json", {**summary, "status": "completed"}))


def test_freeze_learning_uses_inputs_source_and_leaves_status_mutable(tiny_learning):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    assert "source" not in fixture.definition
    assert seal["role"] == "learning" and seal["controller"] == {"pid": 99_999_999, "start": 123}
    assert seal["summary_path"] == str(fixture.root / "summary.json")
    assert len(seal["locks"]) == 2 and len(seal["original_lock_pins"]) == 3
    assert not any(name in seal for name in ("status", "summary_status", "eligible"))
    assert fixture.calls == []
    assert queue.check_dependency(seal)["status"] == "pending"
    assert fixture.calls == []
    fixture.complete()
    proof = queue.check_dependency(seal)
    assert proof["status"] == "completed"
    assert proof["counts"]["training_cells"] == 90
    assert proof["counts"]["development_suites"] == 360
    assert proof["counts"]["development_cells"] == 18_000
    assert fixture.calls == ["validate", "audit"]


def test_pending_only_checks_handles_without_replaying_evidence(tiny_learning, monkeypatch):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    fixture.complete()
    monkeypatch.setattr(queue, "_process_start", lambda pid: 123)
    state = queue.check_dependency(seal)
    assert state["status"] == "pending" and state["controller_live"] is True
    assert fixture.calls == []
    monkeypatch.setattr(queue, "_process_start", lambda pid: None)
    fixture.input_file.write_bytes(b"changed while still waiting")
    write(fixture.root / "summary.json", {**fixture.summary, "status": "training"})
    assert queue.check_dependency(seal)["status"] == "pending"
    assert fixture.calls == []
    fixture.complete()
    with pytest.raises(ValueError, match="bytes changed"):
        queue.check_dependency(seal)


def test_unresolved_or_matching_original_worker_keeps_queue_pending(tiny_learning, monkeypatch):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    fixture.complete()
    path = fixture.root / "cells" / "worker.process.json"
    write(path, {"status": "launching", "pid": None, "start": None})
    assert queue.check_dependency(seal)["live_workers"]
    assert fixture.calls == []
    write(path, {"status": "finished", "pid": 1234, "start": "456", "returncode": 0})
    monkeypatch.setattr(queue, "_process_start", lambda pid: 456 if pid == 1234 else None)
    assert queue.check_dependency(seal)["status"] == "pending"
    monkeypatch.setattr(queue, "_process_start", lambda pid: 457 if pid == 1234 else None)
    assert queue.check_dependency(seal)["status"] == "completed"


def test_closed_proof_cache_rehashes_actual_endpoint_bytes(tiny_learning):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    fixture.complete()
    first = queue.check_dependency(seal)
    second = queue.check_dependency(seal)
    assert first == second and fixture.calls.count("audit") == 1
    second["counts"]["training_cells"] = 1
    assert queue.check_dependency(seal)["counts"]["training_cells"] == 90
    fixture.checkpoint.write_bytes(b"edited endpoint")
    with pytest.raises(ValueError, match="bytes changed"):
        queue.check_dependency(seal)
    assert fixture.calls.count("audit") == 1


def test_cached_closure_rejects_extra_attempt_directory(tiny_learning):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    fixture.complete()
    queue.check_dependency(seal)
    (fixture.root / "unexpected_attempt").mkdir()
    with pytest.raises(ValueError, match="membership"):
        queue.check_dependency(seal)


@pytest.mark.parametrize("mutation", ["association", "audit_sha", "audit_path", "grid_status", "extra_cell",
    "charged", "actual", "completed_bool", "missing_seed", "bad_status", "counter_bool"])
def test_claimed_completion_with_invalid_grid_rejects_instead_of_pending(tiny_learning, mutation):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    summary = {**deepcopy(fixture.summary), "status": "completed"}
    key = next(iter(summary["results"]))
    if mutation == "association":
        summary["manifest_sha256"] = "0" * 64
    elif mutation == "audit_sha":
        summary["audit"]["sha256"] = "0" * 64
    elif mutation == "audit_path":
        copy = fixture.root / "copied_audit.json"
        copy.write_bytes((fixture.root / "audit.json").read_bytes())
        summary["audit"]["path"] = str(copy)
    elif mutation == "grid_status":
        summary["grid_status"] = "not_ready"
    elif mutation == "extra_cell":
        summary["results"]["extra"] = deepcopy(summary["results"][key])
    elif mutation == "charged":
        summary["results"][key]["training"]["charged_updates"] = 1199
    elif mutation == "actual":
        summary["results"][key]["training"]["verified_fresh_samples"] = 58_982_399
    elif mutation == "completed_bool":
        summary["results"][key]["training"]["completed_updates"] = True
    elif mutation == "missing_seed":
        del summary["results"][key]["development"]["3701"]
    elif mutation == "bad_status":
        summary["results"][key]["development"]["3701"]["status"] = "failed"
    else:
        fixture.report["actual_cells"] = True
    write(fixture.root / "summary.json", summary)
    with pytest.raises(ValueError):
        queue.check_dependency(seal)


def test_actual_audit_json_must_equal_fresh_original_audit(tiny_learning):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    saved = deepcopy(fixture.report)
    saved["unexpected_claim"] = True
    write(fixture.root / "audit.json", saved)
    summary = {**fixture.summary, "status": "completed", "audit": {
        "path": str(fixture.root / "audit.json"), "sha256": sha(fixture.root / "audit.json")}}
    write(fixture.root / "summary.json", summary)
    with pytest.raises(ValueError, match="fresh contents"):
        queue.check_dependency(seal)


@pytest.mark.parametrize("mutation", ["definition", "source", "controller", "lock"])
def test_immutable_identity_edits_reject(tiny_learning, mutation):
    fixture = tiny_learning
    seal = queue.freeze_dependency("learning", fixture.root / "summary.json")
    fixture.complete()
    if mutation == "definition":
        path = fixture.root / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "source":
        (fixture.source / "src/transformer_rl/frame_process.py").write_text("# changed source\n")
    elif mutation == "controller":
        write(fixture.root / "controller.json", {"pid": 99_999_999, "start": "124",
            "manifest_sha256": fixture.definition["sha256"]})
    else:
        path = Path(seal["locks"][0]["path"])
        path.rename(path.with_suffix(".old"))
        path.touch()
    with pytest.raises(ValueError):
        queue.check_dependency(seal)


@pytest.mark.parametrize("path_kind", ["relative", "wrong_filename", "symlink"])
def test_only_canonical_same_directory_summary_is_accepted(tiny_learning, path_kind, monkeypatch):
    fixture = tiny_learning
    if path_kind == "relative":
        monkeypatch.chdir(fixture.root)
        path = Path("summary.json")
    elif path_kind == "wrong_filename":
        path = fixture.root / "copied.json"
        path.write_bytes((fixture.root / "summary.json").read_bytes())
    else:
        path = fixture.root / "linked.json"
        path.symlink_to(fixture.root / "summary.json")
    with pytest.raises(ValueError):
        queue.freeze_dependency("learning", path)


def test_lock_fd_and_path_must_identify_same_original_inode(tmp_path):
    path = tmp_path / "existing.lock"
    path.touch()
    expected = queue.lock_identity(path)
    with path.open("r+") as stream:
        queue.check_open_lock(stream, expected)
        path.rename(tmp_path / "original.lock")
        path.touch()
        with pytest.raises(ValueError, match="replaced"):
            queue.check_open_lock(stream, expected)
    assert not (tmp_path / "missing.lock").exists()
    with pytest.raises(ValueError):
        queue.lock_identity(tmp_path / "missing.lock")


def test_nested_provider_compiles_actual_bytes_despite_valid_stale_pyc(tmp_path):
    # Fake helper source tests the loader only, never campaign qualification.
    leaf = tmp_path / "run_frame_control_campaign.py"
    leaf.write_text("VALUE = 'old'\n")
    before = leaf.stat()
    py_compile.compile(str(leaf), doraise=True)
    leaf.write_text("VALUE = 'new'\n")
    os.utime(leaf, ns=(before.st_atime_ns, before.st_mtime_ns))
    parent = tmp_path / "run_transfer_campaign.py"
    parent.write_text("import importlib.util\nfrom pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('leaf', Path(__file__).with_name('run_frame_control_campaign.py'))\n"
        "leaf = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(leaf)\nVALUE = leaf.VALUE\n")
    cached = sorted(tmp_path.rglob("*.pyc"))
    loader = queue._Providers()
    value = loader.load(parent)
    assert value.VALUE == "new"
    assert value.__file__ == str(parent) and value.leaf.__file__ == str(leaf)
    assert set(loader.receipts) == {str(parent), str(leaf)}
    assert sorted(tmp_path.rglob("*.pyc")) == cached


def test_real_original_helper_loading_has_no_learner_or_simulator_imports():
    loader = queue._Providers()
    helper = loader.load(queue.TOOLS / "run_learning_campaign.py")
    assert callable(helper.audit) and callable(helper.validate)
    assert {Path(path).name for path in loader.receipts} == {
        "run_learning_campaign.py", "run_frame_diagnostic_campaign.py",
        "run_transfer_campaign.py", "run_frame_control_campaign.py"}


@pytest.mark.parametrize("role", ["curriculum", "diagnostics", "learning"])
def test_original_definition_association_rules_are_distinct(tmp_path, role):
    format_name, filename, field, _ = queue._ROLES[role]
    value = {"format": format_name, "schema_version": 1, "output_root": str(tmp_path)}
    association = queue._digest(value)
    if role != "curriculum":
        value["sha256"] = association
    write(tmp_path / filename, value)
    path, actual, identity, summary_field = queue._definition(role, tmp_path / "summary.json")
    assert path == tmp_path / filename and actual == value
    assert identity == association and summary_field == field
    if role == "curriculum":
        value["sha256"] = association
    else:
        value["sha256"] = "0" * 64
    write(path, value)
    with pytest.raises(ValueError):
        queue._definition(role, tmp_path / "summary.json")


@pytest.mark.parametrize("role", ["curriculum", "diagnostics"])
def test_legacy_source_inventory_uses_repository_relative_python_paths(tmp_path, role):
    source = tmp_path / "original_source"
    package = source / "src/transformer_rl"
    package.mkdir(parents=True)
    path = package / "frame_process.py"
    path.write_text("# Frozen interface source, never invoked.\n")
    files = {"src/transformer_rl/frame_process.py": sha(path)}
    definition = {"source_root": str(source), "source": {"root": str(source), "files": files,
                                                           "sha256": queue._digest(files)}}
    assert queue._source_pins(role, definition) == [queue._receipt(path)]
    (package / "unlisted.py").write_text("# Extra source is not frozen.\n")
    with pytest.raises(ValueError, match="inventory changed"):
        queue._source_pins(role, definition)


@pytest.mark.parametrize("role", ["curriculum", "diagnostics", "learning"])
def test_provider_dispatch_uses_original_role_and_never_learning_else(tmp_path, monkeypatch, role):
    # A loader spy tests routing only. Actual production helpers are fixed and
    # loaded from the trusted source bytes, as separately checked above.
    calls = []

    class LoaderSpy:
        def __init__(self):
            self.receipts, self.modules = {}, []

        def load(self, path):
            calls.append(Path(path).name)
            return SimpleNamespace(filename=Path(path).name)

    definition = {"source": {field: sha(queue.TOOLS / filename) for field, filename in (
        ("curriculum_controller_sha256", "run_curriculum_campaign.py"),
        ("transfer_helper_sha256", "run_transfer_campaign.py"),
        ("controller_sha256", "run_frame_control_campaign.py"))}}
    if role != "curriculum":
        definition["controllers"] = {str(queue.TOOLS / name): sha(queue.TOOLS / name)
            for name in (queue._ROLES[role][3], "run_transfer_campaign.py", "run_frame_control_campaign.py")}
    monkeypatch.setattr(queue, "_Providers", LoaderSpy)
    _, primary, adapter = queue._providers(role, definition)
    assert primary.filename == queue._ROLES[role][3]
    assert adapter.filename == "run_learning_campaign.py"
    assert calls == (["run_learning_campaign.py"] if role == "learning"
                     else [queue._ROLES[role][3], "run_learning_campaign.py"])

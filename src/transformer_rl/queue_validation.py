"""Read-only closure of the three historical predecessor queues.

Pending queues are inspected through their original PID/start handles and
process receipts. Completed evidence is authenticated by the original helpers,
compiled from actual source bytes, and by a fresh learning-grid audit. This
module never acquires resource locks, invokes a learner, or publishes a report.
"""
from __future__ import annotations

import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import types


TOOLS = Path(__file__).resolve().parents[2] / "tools"
_ROLES = {
    "curriculum": ("transformer_rl.curriculum_campaign", "campaign.json", "campaign_sha256",
                   "run_curriculum_campaign.py"),
    "diagnostics": ("transformer_rl.frame_diagnostic_campaign", "manifest.json", "manifest_sha256",
                    "run_frame_diagnostic_campaign.py"),
    "learning": ("transformer_rl.learning_campaign", "manifest.json", "manifest_sha256",
                 "run_learning_campaign.py"),
}
_HELPERS = {"run_curriculum_campaign.py", "run_frame_diagnostic_campaign.py",
            "run_learning_campaign.py", "run_transfer_campaign.py", "run_frame_control_campaign.py"}
_PROOFS = {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _plain(value):
    _require(isinstance(value, (str, Path)), "an absolute canonical path is required")
    path = Path(value)
    _require(path.is_absolute() and all(not item.is_symlink() for item in (path, *path.parents))
             and str(path) == str(path.resolve()), "an absolute canonical nonsymlink path is required")
    return path


def _inside(root, route):
    route = Path(route)
    path = _plain(route if route.is_absolute() else root / route)
    _require(path != root and path.is_relative_to(root), "artifact path escapes its historical root")
    return path


def _sha(path):
    path = _plain(path)
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            value = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as error:
        raise ValueError(f"historical input is unavailable: {path}") from error
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
             == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns),
             f"historical input changed while reading: {path}")
    return value


def _receipt(path):
    path = _plain(path)
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _checked(item):
    _require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}
             and type(item["bytes"]) is int and item["bytes"] >= 0
             and type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
             "invalid historical file receipt")
    path = _plain(item["path"])
    _require(_receipt(path) == item, f"historical file bytes changed: {path}")
    return path


def _read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate historical JSON field")
            value[key] = item
        return value
    try:
        value = json.loads(_plain(path).read_bytes(), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite historical JSON")))
        _bytes(value)
        return value
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid historical JSON: {path}") from error


def lock_identity(path):
    """Pin an existing lock without creating it or changing its flock state."""
    path = _plain(path)
    _require(path.is_file(), "the original lock file must already exist")
    value = path.stat()
    return {"path": str(path), "device": value.st_dev, "inode": value.st_ino}


def check_open_lock(stream, expected):
    """Verify both an inherited descriptor and the current lock pathname."""
    _require(type(expected) is dict and set(expected) == {"path", "device", "inode"}
             and all(type(expected[key]) is int for key in ("device", "inode")), "invalid lock identity")
    actual = os.fstat(stream.fileno())
    _require((actual.st_dev, actual.st_ino) == (expected["device"], expected["inode"])
             and lock_identity(expected["path"]) == expected, "original lock inode was replaced")


def _handle(value):
    _require(type(value) is dict and set(value) == {"pid", "start"}
             and type(value["pid"]) is int and value["pid"] > 0, "an original controller PID/start handle is required")
    start = value["start"]
    _require(type(start) is int and start > 0 or type(start) is str and start.isascii()
             and start.isdigit() and int(start) > 0, "invalid original controller start time")
    return {"pid": value["pid"], "start": int(start)}


def _process_start(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None
    except (OSError, IndexError, ValueError) as error:
        raise ValueError(f"cannot verify original process handle: {pid}") from error


class _SourceLoader:
    def __init__(self, owner, path):
        self.owner, self.path = owner, path

    def create_module(self, specification):
        return None

    def exec_module(self, module):
        self.owner.execute(module, self.path)


class _Providers:
    """A private importlib view that never reads nested helper bytecode caches."""
    def __init__(self):
        self.receipts, self.modules = {}, []
        self.util = types.ModuleType("importlib.util")
        self.util.__dict__.update(vars(importlib.util))
        self.util.spec_from_file_location = self.specification
        self.importlib = types.ModuleType("importlib")
        self.importlib.util = self.util

    def specification(self, name, location, *args, **kwargs):
        path = _plain(location)
        _require(path.name in _HELPERS, "unexpected historical helper import")
        return importlib.util.spec_from_file_location(name, str(path), loader=_SourceLoader(self, path))

    def importing(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name == "importlib.util" and level == 0:
            return self.util if fromlist else self.importlib
        return builtins.__import__(name, globals, locals, fromlist, level)

    def execute(self, module, path):
        receipt = _receipt(path)
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == receipt["sha256"], "helper changed while loading its source bytes")
        prior = self.receipts.setdefault(str(path), receipt)
        _require(prior == receipt, "nested helper source changed during import")
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self.importing}
        self.modules.append(module)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        _checked(receipt)

    def load(self, path):
        specification = self.specification("historical_queue_" + str(len(self.modules)), path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module


def _provider_path(role, definition):
    name = _ROLES[role][3]
    declared = definition.get("controllers", {})
    if declared:
        matches = [Path(route) for route in declared if Path(route).name == name]
        _require(len(matches) == 1, "original queue controller path is ambiguous")
        for route, expected in declared.items():
            path = _plain(route)
            _require(path.name in _HELPERS and expected == _sha(TOOLS / path.name)
                     and _sha(path) == expected, "original controller differs from the trusted helper bytes")
        return _plain(matches[0])
    _require(role == "curriculum", "original queue controller source is absent")
    source = definition["source"]
    for field, filename in (("curriculum_controller_sha256", name),
                            ("transfer_helper_sha256", "run_transfer_campaign.py"),
                            ("controller_sha256", "run_frame_control_campaign.py")):
        _require(source.get(field) == _sha(TOOLS / filename), "original curriculum helper source changed")
    return TOOLS / name


def _providers(role, definition):
    owner = _Providers()
    primary = owner.load(_provider_path(role, definition))
    # This adapter is used only for its curriculum/diagnostic branches. Learning
    # is always validated by its own validate() and audit(), never by the else.
    adapter = primary if role == "learning" else owner.load(TOOLS / "run_learning_campaign.py")
    return owner, primary, adapter


def _source_pins(role, definition):
    source = definition["inputs"]["source"] if role == "learning" else definition["source"]
    root = _plain(definition.get("source_root", source.get("root")))
    package = root / "src/transformer_rl"
    files = source.get("files")
    _require(type(files) is dict and files and package.is_dir(), "complete original source inventory is required")
    base = root if all(name.startswith("src/transformer_rl/") for name in files) else package
    actual = {}
    for path in sorted(package.rglob("*")):
        _plain(path)
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and (role == "learning" or path.suffix == ".py"):
            actual[str(path.relative_to(base))] = _sha(path)
    _require(actual == files, "original learner source inventory changed")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=role == "learning", allow_nan=False).encode()
    _require(hashlib.sha256(encoded).hexdigest() == source.get("sha256"), "original learner source digest differs")
    return [_receipt(base / route) for route in sorted(files)]


def _input_pins(role, definition):
    pins = {item["path"]: item for item in _source_pins(role, definition)}
    if role == "curriculum":
        path = _plain(definition["manifest"])
        _require(_sha(path) == definition["manifest_file_sha256"], "curriculum input manifest bytes changed")
        manifest = _read(path)
        _require(manifest.get("sha256") == definition["manifest_sha256"], "curriculum manifest association differs")
        pins[str(path)] = _receipt(path)
        for route, expected in manifest["configs"].items():
            configured = _inside(path.parent, route)
            _require(_digest(_read(configured)) == expected, "curriculum configuration changed")
            pins[str(configured)] = _receipt(configured)
    else:
        inputs = definition["inputs"]
        if role == "learning":
            root = _plain(inputs["root"])
            for route, expected in inputs["files"].items():
                path = _inside(root, route)
                _require(_sha(path) == expected, "prepared learning input bytes changed")
                pins[str(path)] = _receipt(path)
        def walk(value):
            if type(value) is dict:
                if "path" in value and "sha256" in value:
                    path = Path(value["path"])
                    path = _plain(path) if path.is_absolute() else _inside(_plain(inputs["root"]), path)
                    _require(_sha(path) == value["sha256"], "original immutable input receipt changed")
                    pins[str(path)] = _receipt(path)
                else:
                    for item in value.values():
                        walk(item)
            elif type(value) is list:
                for item in value:
                    walk(item)
        walk(inputs)
        if role == "learning":
            path = _plain(definition["dependency_file"]["path"])
            _require(_sha(path) == definition["dependency_file"]["sha256"], "original dependency file changed")
            pins[str(path)] = _receipt(path)
    return [pins[path] for path in sorted(pins)]


def _definition(role, summary):
    _require(role in _ROLES, "unknown predecessor role")
    summary = _plain(summary)
    _require(summary.name == "summary.json", "the canonical same-directory summary.json is required")
    format_name, filename, field, _ = _ROLES[role]
    path = summary.parent / filename
    value = _read(path)
    _require(type(value) is dict and value.get("format") == format_name
             and type(value.get("schema_version")) is int and value["schema_version"] == 1,
             "original predecessor definition format differs")
    if role == "curriculum":
        _require("sha256" not in value, "curriculum campaign has no self SHA field")
        association = _digest(value)
    else:
        association = _digest({key: item for key, item in value.items() if key != "sha256"})
        _require(association == value.get("sha256") and _plain(value["output_root"]) == summary.parent,
                 "original manifest self identity or output association differs")
    return path, value, association, field


def _freeze_dependency(role, summary_path, *, controller=None):
    """Freeze immutable inputs and original handles, leaving live status mutable.

    Legacy curriculum/diagnostic handles must come from the original frozen LR
    dependency declarations. The LR queue owns its canonical controller.json.
    """
    path, definition, association, field = _definition(role, summary_path)
    controller_path = None
    if controller is None:
        controller_path = path.parent / "controller.json"
        value = _read(controller_path)
        _require(value.get(field) == association, "original controller belongs to another manifest")
        controller = {key: value[key] for key in ("pid", "start")}
    handle = _handle(controller)
    owner, _, _ = _providers(role, definition)
    routes = [_plain(definition["resource_lock"])]
    if role == "learning":
        routes.append(_plain(definition["study_lock"]))
    elif role == "diagnostics":
        routes.append(_plain(definition["inputs"]["study_root"]) / ".run.lock")
    locks = [lock_identity(route) for route in dict.fromkeys(routes)]
    original_locks = [lock_identity(_plain(route)) for route in definition.get("locks", {})]
    for item in original_locks:
        _require({key: item[key] for key in ("device", "inode")} == definition["locks"][item["path"]],
                 "original manifest lock inode changed")
    roots = [str(path.parent)]
    if role == "diagnostics":
        roots.append(str(_plain(definition["inputs"]["study_root"])))
    body = {"format": "transformer_rl.queue_dependency", "schema_version": 1,
        "role": role, "summary_path": str(_plain(summary_path)), "definition": _receipt(path),
        "association_sha256": association, "summary_identity_field": field,
        "controller": handle, "controller_path": str(controller_path) if controller_path else None,
        "locks": locks, "original_lock_pins": original_locks,
        "helper_receipts": sorted(owner.receipts.values(), key=lambda item: item["path"]),
        "input_receipts": _input_pins(role, definition), "worker_roots": roots}
    return {**body, "sha256": _digest(body)}


def _workers(roots):
    live = []
    for path in sorted({path for root in roots for path in _plain(root).rglob("*.process.json")}):
        value = _read(path)
        pid, start = value.get("pid"), value.get("start")
        if value.get("status") in {"launching", "running"} and (pid is None or start is None):
            live.append({"path": str(path), "reason": "original worker handle is unresolved"})
        elif pid is not None and start is not None:
            handle = _handle({"pid": pid, "start": start})
            if _process_start(handle["pid"]) == handle["start"]:
                live.append({"path": str(path), **handle})
    return live


class _Tracker:
    def __init__(self):
        self.receipts = {}

    def sha(self, path):
        item = _receipt(_plain(path))
        prior = self.receipts.setdefault(item["path"], item)
        _require(item == prior, "closed evidence changed during its audit")
        return item["sha256"]

    def read(self, path):
        self.sha(path)
        value = _read(path)
        self.sha(path)
        return value

    def attach(self, owner):
        for module in owner.modules:
            if Path(module.__file__).name == "run_frame_control_campaign.py":
                module.read, module.file_sha = self.read, self.sha


def _strict_int(value, expected, name):
    _require(type(value) is int and value == expected, f"closed {name} count differs")


def _close_legacy(seal, definition, summary, primary, adapter):
    role, root = seal["role"], Path(seal["summary_path"]).parent
    dependency = {"definition": {key: seal["definition"][key] for key in ("path", "sha256")},
        "sha256": seal["association_sha256"], "summary": seal["summary_path"],
        "summary_identity_field": seal["summary_identity_field"],
        "controller": {"pid": seal["controller"]["pid"], "start": str(seal["controller"]["start"])},
        "root": str(root), "immutable_inputs": adapter.dependency_inputs(definition)}
    _require(adapter.dependency_state(dependency).get("ready") is True, "completed predecessor closure is unproven")
    if role == "curriculum":
        manifest = primary.load_manifest(definition["manifest"])
        source = Path(definition["source"]["root"])
        _require(primary.source_identity(source) == definition["source"], "curriculum original source identity changed")
        primary.verify_learner_source(source, manifest)
        primary.verify_configs(Path(definition["manifest"]).parent, manifest)
        _require(len(manifest["training_seeds"]) == 3 and len(manifest["scenarios"]) == 50
                 and len(manifest["evaluation"]["seeds"]) == 2, "curriculum original 36 by 50 grid differs")
        for result in summary["results"].values():
            for phase, expected in zip(result["phases"], next(arm for arm in manifest["arms"] if arm["name"] == result["arm"])["phases"]):
                _strict_int(expected["updates"], 400 if expected["start_update"] == 0 else 800, "phase budget")
                cursor = expected["start_update"]
                for item in phase["training"]["attempts"]:
                    receipt = adapter.control.read(adapter.control.checked(root, item))
                    directory = adapter.control.checked(root, item).parent
                    request = adapter.control.read(directory / "request.json")
                    completion = adapter.control.read(directory / "train/completion.json")
                    _require(receipt["request_sha256"] == adapter.control.file_sha(directory / "request.json")
                             and receipt["status"] in {"completed", "resumable"}, "curriculum sealed request or status changed")
                    _strict_int(request["start_update"], cursor, "attempt start update")
                    _strict_int(request["prior_transitions"], cursor * 49152, "attempt prior transitions")
                    _require(type(completion["completed_updates"]) is int and completion["completed_updates"] > 0,
                             "curriculum complete attempt has no actual update")
                    evidence = primary.transfer.metrics_evidence(directory / "train/metrics.jsonl", sealed=completion)
                    _strict_int(completion["attempted_updates"], completion["completed_updates"], "attempted updates")
                    _strict_int(completion["consumed_transitions"], completion["completed_updates"] * 49152, "actual phase samples")
                    _require(receipt.get("evidence", {}).get("batch_samples") == evidence["batch_samples"]
                             and evidence["batch_samples"] == completion["consumed_transitions"]
                             and completion["start_update"] == request["start_update"]
                             and completion["final_update"] == request["start_update"] + completion["completed_updates"],
                             "curriculum original optimization evidence differs")
                    cursor += completion["completed_updates"]
                    _strict_int(completion["cumulative_transitions"], cursor * 49152, "attempt cumulative transitions")
                _strict_int(cursor, expected["start_update"] + expected["updates"], "complete phase update")
                _strict_int(phase["training"]["checkpoint"]["cumulative_transitions"], cursor * 49152,
                            "phase checkpoint transitions")
        return {"evaluation_suites": 36, "evaluation_cells": 1800, "training_endpoints": 18}
    _require(primary.validate(Path(seal["definition"]["path"])) == definition, "diagnostic original validator rejected its definition")
    _require(len(definition["inputs"]["variants"]) == 10 and len(definition["inputs"]["cases"]) == 50
             and len(definition["protocol"]["seeds"]) == 2, "diagnostic original 20 by 50 grid differs")
    for variant in definition["inputs"]["variants"]:
        for seed in definition["protocol"]["seeds"]:
            receipt = summary["results"][f"{variant}/seed_{seed}"]
            directory = _inside(root, receipt["directory"])
            _require(primary.verify_outputs(directory, definition, variant, seed) == receipt["artifacts"],
                     "diagnostic complete control/trace artifacts changed")
            _require(adapter.control.read(directory / "receipt.json") == receipt, "diagnostic summary and actual receipt differ")
    return {"evaluation_suites": 20, "evaluation_cells": 1000}


def _close_learning(seal, definition, summary, primary):
    path = Path(seal["definition"]["path"])
    _require(primary.validate(path) == definition, "original learning validator rejected the definition")
    report = primary.audit(path)
    _require(summary.get("grid_status") == "development_complete" and report.get("status") == "development_complete",
             "learning development grid is not closed")
    for name in ("expected_cells", "actual_cells", "completed_training_cells", "completed_development_cells"):
        _strict_int(report.get(name), 90, name)
    cells = definition["inputs"]["cells"]
    expected = {primary.cell_key(cell) for cell in cells}
    _require(len(cells) == len(expected) == 90 and set(report["cells"]) == set(summary["results"]) == expected,
             "learning exact training/development cell keys differ")
    suites = 0
    for key in expected:
        result, recorded = report["cells"][key], summary["results"][key]
        _require(result.get("status") == recorded.get("status") == "completed", "learning grid contains an incomplete cell")
        for training in (result["training"], recorded["training"]):
            _require(training.get("status") == "completed", "learning training endpoint is not complete")
            _strict_int(training.get("completed_updates"), 1200, "completed updates")
            _strict_int(training.get("charged_updates"), 1200, "charged updates")
            _strict_int(training.get("verified_fresh_samples"), 58_982_400, "actual samples")
        _require(result["training"]["checkpoint"] == recorded["training"]["checkpoint"], "learning summary checkpoint differs from fresh audit")
        for development in (result["development"], recorded["development"]):
            _require(set(development) == {"701", "1701", "2701", "3701"}
                     and all(type(item) is dict and item.get("status") == "completed" for item in development.values()),
                     "learning four development suites are incomplete")
        for seed, development in result["development"].items():
            receipt = development.get("receipt")
            _require(type(receipt) is dict and set(receipt) == {"path", "sha256"}
                     and primary.control.file_sha(_plain(receipt["path"])) == receipt["sha256"]
                     and _bytes(primary.control.read(receipt["path"])) == _bytes(recorded["development"][seed]),
                     "learning summary suite differs from its actual audited receipt")
        suites += len(result["development"])
    _strict_int(suites, 360, "development suites")
    audit = summary.get("audit")
    _require(type(audit) is dict and set(audit) == {"path", "sha256"}
             and _plain(audit["path"]) == path.parent / "audit.json"
             and _sha(Path(audit["path"])) == audit["sha256"]
             and _bytes(_read(audit["path"])) == _bytes(report), "learning actual audit path/SHA or fresh contents differ")
    return {"training_cells": 90, "development_suites": 360, "development_cells": 18000,
            "audit": _receipt(audit["path"])}


def _membership(roots):
    return {str(root): sorted(str(path.relative_to(root)) for path in root.rglob("*")
                             if "__pycache__" not in path.parts and path.suffix != ".pyc") for root in roots}


def _closure_roots(seal, definition):
    roots = [Path(route) for route in seal["worker_roots"]]
    if seal["role"] == "curriculum":
        roots.append(Path(definition["manifest"]).parent)
    if seal["role"] == "learning":
        roots.append(Path(definition["inputs"]["root"]))
    source = definition["inputs"]["source"] if seal["role"] == "learning" else definition["source"]
    roots.append(Path(definition.get("source_root", source.get("root"))) / "src/transformer_rl")
    return list(dict.fromkeys(roots))


def _check_dependency(seal, *, require_complete=True):
    """Return pending for live work; reject invalid identity or claimed closure.

    A cached closed proof is reused only after rehashing every byte receipt the
    original validators read and checking complete directory membership again.
    """
    _require(type(require_complete) is bool and type(seal) is dict and seal.get("format") == "transformer_rl.queue_dependency"
             and type(seal.get("schema_version")) is int and seal["schema_version"] == 1
             and _digest({key: value for key, value in seal.items() if key != "sha256"}) == seal.get("sha256"),
             "queue dependency seal identity differs")
    _checked(seal["definition"])
    path, definition, association, field = _definition(seal["role"], seal["summary_path"])
    _require(str(path) == seal["definition"]["path"] and association == seal["association_sha256"]
             and field == seal["summary_identity_field"], "queue canonical association changed")
    for item in [*seal["locks"], *seal["original_lock_pins"]]:
        _require(lock_identity(item["path"]) == item, "original queue lock inode changed")
    for item in seal["helper_receipts"]:
        _checked(item)
    controller = _handle(seal["controller"])
    if seal["controller_path"] is not None:
        value = _read(seal["controller_path"])
        _require(_handle({key: value[key] for key in ("pid", "start")}) == controller
                 and value.get(field) == association, "original controller handle or association changed")
    controller_live = _process_start(controller["pid"]) == controller["start"]
    live = _workers(seal["worker_roots"])
    summary_path = Path(seal["summary_path"])
    summary = _read(summary_path) if summary_path.exists() else None
    if summary is not None:
        _require(type(summary) is dict and summary.get(field) == association, "summary belongs to another original queue")
    if summary is None or summary.get("status") != "completed" or controller_live or live:
        return {"status": "pending", "role": seal["role"], "controller_live": controller_live,
                "live_workers": live, "summary_status": summary.get("status") if summary else None,
                "reason": "original queue has not closed its terminal evidence"}
    if not require_complete:
        return {"status": "pending", "role": seal["role"], "controller_live": False,
                "live_workers": [], "summary_status": "completed", "reason": "complete evidence audit was not requested"}
    for item in seal["input_receipts"]:
        _checked(item)
    summary_receipt = _receipt(summary_path)
    key = (seal["sha256"], summary_receipt["sha256"])
    cached = _PROOFS.get(key)
    if cached:
        for item in cached["receipts"]:
            _checked(item)
        _require(_membership(cached["roots"]) == cached["membership"], "closed queue directory membership changed")
        return deepcopy(cached["proof"])
    roots = _closure_roots(seal, definition)
    membership = _membership(roots)
    owner, primary, adapter = _providers(seal["role"], definition)
    tracker = _Tracker()
    tracker.attach(owner)
    counts = (_close_learning(seal, definition, summary, primary) if seal["role"] == "learning"
              else _close_legacy(seal, definition, summary, primary, adapter))
    receipts = {item["path"]: item for item in [*seal["input_receipts"], *seal["helper_receipts"], summary_receipt,
                                               *owner.receipts.values(), *tracker.receipts.values()]}
    if "audit" in counts:
        receipts[counts["audit"]["path"]] = counts["audit"]
    for item in receipts.values():
        _checked(item)
    _require(not _workers(seal["worker_roots"]) and _process_start(controller["pid"]) != controller["start"],
             "original worker or controller reappeared during closure")
    _require(_membership(roots) == membership, "closed queue directory membership changed during audit")
    proof = {"status": "completed", "role": seal["role"], "association_sha256": association,
             "summary": summary_receipt, "counts": counts,
             "closure_receipts": sorted(receipts.values(), key=lambda item: item["path"]),
             "controller_live": False, "live_workers": []}
    proof["sha256"] = _digest(proof)
    _PROOFS[key] = {"proof": deepcopy(proof), "receipts": list(receipts.values()),
                    "roots": roots, "membership": membership}
    return proof


def freeze_dependency(role, summary_path, *, controller=None):
    """Pin a canonical predecessor definition and its authenticated old handle."""
    try:
        return _freeze_dependency(role, summary_path, controller=controller)
    except (KeyError, TypeError, OSError, IndexError, StopIteration) as error:
        raise ValueError("original dependency schema or inputs are invalid") from error


def check_dependency(seal, *, require_complete=True):
    """Recheck live handles or return a completely closed byte-bound proof."""
    try:
        return _check_dependency(seal, require_complete=require_complete)
    except (KeyError, TypeError, OSError, IndexError, StopIteration) as error:
        raise ValueError("original dependency closure schema is invalid") from error

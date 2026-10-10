"""Qualified, frozen retention branches; no historical learner code executes.

The provider is the original Gated fixed-window curriculum pilot. A new plan
adds stationary and mixed continuations from each same full teacher checkpoint.
This module never changes existing preparation, qualification or queue helpers.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import types


FORMAT = "transformer_rl.retention_campaign_protocol"
REQUEST_FORMAT = "transformer_rl.continuation_request"
PACKAGE = Path(__file__).resolve().parent
TOOLS = PACKAGE.parents[1] / "tools"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _sealed(value):
    return {**value, "sha256": _digest(value)}


def _read(path):
    def pairs(items):
        result = {}
        for name, value in items:
            _require(name not in result, "duplicate JSON field")
            result[name] = value
        return result
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _bytes(result)
    return result


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _checked(receipt):
    _require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256", "bytes"}
             and isinstance(receipt["path"], str) and Path(receipt["path"]).is_absolute()
             and isinstance(receipt["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
             and type(receipt["bytes"]) is int and receipt["bytes"] >= 0,
             "invalid absolute artifact receipt")
    path = Path(receipt["path"])
    _require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
             "artifact receipt path must be canonical and must not traverse symlinks")
    _require(_sha(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"],
             f"artifact changed: {path}")
    return path


def learner_source():
    files = {str(path.relative_to(PACKAGE)): _sha(path) for path in sorted(PACKAGE.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    # Match experiments.source_identity's default ASCII JSON convention.
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


def _preparer():
    path = TOOLS / "prepare_nested_anchors.py"
    raw = path.read_bytes()
    module = types.ModuleType("retention_anchor_provider")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    _require(path.read_bytes() == raw, "anchor provider changed while importing")
    return module


def _context(preparation_protocol, preparation_directory):
    provider = _preparer()
    result = provider.validate_preparation(preparation_protocol, preparation_directory)
    _require(result["status"] == "ready" and result["ready_cells"] == result["expected_cells"]
             and result["qualified_teacher_case_pairs"] == result["expected_teacher_case_pairs"],
             "all original teacher/case pairs and prepared capacities must be ready")
    auditor = provider._auditor()
    reader = auditor.Reader()
    q = preparation_protocol["qualification_protocol"]
    bundle = auditor._bundle(reader, q["campaign_root"])
    summary = reader.read(bundle["root"] / "summary.json")
    _require(summary.get("status") == "completed", "original curriculum must be completed and stable")
    schedules = {}
    for name in ("stationary", "mixed"):
        selected = next(entry for entry in bundle["manifest"]["arms"]
                        if entry["name"] == ("stationary" if name == "stationary" else "pretrain"))
        phase = next(item for item in selected["phases"] if item["name"] == "phase2")
        _require(phase["domain"] == name and phase["start_update"] == 400 and phase["updates"] == 800,
                 "continuation schedule or original phase budget differs")
        schedules[name] = deepcopy(bundle["configs"][phase["config"]])
    reference = schedules["mixed"]
    _require(reference["model"] == q["model"] and reference["environment"]["num_envs"] == 1024
             and bundle["manifest"]["training"]["rollout_steps"] == 48,
             "qualified model or full rollout budget differs")
    for cfg in schedules.values():
        _require(all(cfg[k] == reference[k] for k in ("model", "ppo", "control")),
                 "counterfactual schedules differ in model, PPO or control")
    reader.unchanged()
    return result, bundle, schedules


def _dependencies(curriculum, diagnostics, learning, resource_lock):
    from .queue_validation import freeze_dependency
    learning = Path(learning).resolve()
    _require(learning.name == "summary.json", "learning summary basename differs")
    definition = _read(learning.parent / "manifest.json")
    result = []
    for role, summary in (("curriculum", curriculum), ("diagnostics", diagnostics)):
        original = definition["dependencies"][role]
        summary = Path(summary).resolve()
        _require(str(summary) == original["summary"], "predecessor differs from the frozen learning queue")
        seal = freeze_dependency(role, summary, controller=original["controller"])
        _require(seal["definition"]["path"] == original["definition"]["path"]
                 and seal["definition"]["sha256"] == original["definition"]["sha256"],
                 "predecessor definition differs from the frozen learning queue")
        result.append(seal)
    result.append(freeze_dependency("learning", learning))
    locks = {}
    for dependency in result:
        _require(any(item["path"] == resource_lock for item in dependency["locks"]),
                 "predecessor queues do not share the original resource lock")
        for item in dependency["locks"]:
            previous = locks.setdefault(item["path"], item)
            _require(previous == item, "predecessors disagree about a lock inode")
    _require(len(locks) == 2, "the original resource and transfer-study locks are required")
    return result, [locks[path] for path in sorted(locks)]


def _protected_roots(bundle, preparation_directory, dependencies):
    roots = {Path(preparation_directory).resolve(), PACKAGE.resolve()}
    roots.update(Path(bundle[key]).resolve() for key in
                 ("experiment", "study", "snapshot", "source_root"))
    def declared(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"root", "study_root", "source_root", "snapshot", "snapshot_root"}:
                    if isinstance(item, str) and Path(item).is_absolute():
                        roots.add(Path(item).resolve())
                declared(item)
        elif isinstance(value, list):
            for item in value:declared(item)
    for dependency in dependencies:
        roots.update(Path(root).resolve() for root in dependency["worker_roots"])
        definition = _read(dependency["definition"]["path"])
        declared(definition.get("inputs", {}))
        declared(definition.get("source", {}))
        if isinstance(definition.get("source_root"), str):
            roots.add(Path(definition["source_root"]).resolve())
    return [str(root) for root in sorted(roots)]


def _guard_output(destination, protected):
    destination = Path(destination).resolve()
    _require(all(not destination.is_relative_to(Path(root))
                 and not Path(root).is_relative_to(destination) for root in protected),
             "new output overlaps an original evidence, input or source tree")


def freeze(preparation_protocol_path, preparation_directory, *, output_root,
           retention_seed, diagnostic_dependency, learning_dependency,
           device="cpu", retention_batch_size=256, tensorboard=True,
           worker_timeout_seconds=172900., max_wait_seconds=1814400., poll_seconds=20.,
           _allow_existing_output=False):
    """Freeze both schedules and all prepared seeds/K/λ; never launch a job."""
    _require(type(retention_seed) is int and 0 <= retention_seed < 2**32,
             "retention seed must be uint32")
    _require(type(retention_batch_size) is int and retention_batch_size > 0
             and type(tensorboard) is bool, "invalid sampler or TensorBoard settings")
    _require(type(device) is str and re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", device),
             "device must be cpu or an explicit CUDA index")
    for name, value in (("worker_timeout_seconds", worker_timeout_seconds),
                        ("max_wait_seconds", max_wait_seconds), ("poll_seconds", poll_seconds)):
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                 f"{name} must be finite and positive")
    _require(poll_seconds <= 30, "controller polling must be bounded")
    prep_path = Path(preparation_protocol_path).resolve()
    prep = _read(prep_path)
    result, bundle, schedules = _context(prep, preparation_directory)
    q = prep["qualification_protocol"]
    _require(retention_seed not in [*prep["training_seeds"], *q["evaluation"]["seeds"], prep["anchor_seed"]],
             "retention seed must differ from training, evaluation and capture seeds")
    destination = Path(output_root)
    _require(not destination.is_symlink() and (_allow_existing_output or not destination.exists()),
             "campaign output already exists")
    destination = destination.resolve()
    for protected in (bundle["experiment"], Path(preparation_directory).resolve()):
        _require(not destination.is_relative_to(protected), "output must be outside original evidence and anchors")
    resource_lock = bundle["campaign"].get("resource_lock")
    _require(isinstance(resource_lock, str) and Path(resource_lock).is_absolute(),
             "original shared resource lock is required")
    dependencies, resource_locks = _dependencies(bundle["root"] / "summary.json",
        diagnostic_dependency, learning_dependency, resource_lock)
    protected = _protected_roots(bundle, preparation_directory, dependencies)
    _guard_output(destination, protected)
    _require(len({item["summary_path"] for item in dependencies}) == 3,
             "three separate predecessor queues are required")
    max_seconds = bundle["manifest"]["training"]["max_seconds"]
    _require(worker_timeout_seconds > max_seconds, "process timeout must exceed the worker soft deadline")
    branches = []
    for original in result["branches"]:
        _require(original["status"] == "ready" and original["updates_per_branch"] == 800
                 and original["transitions_per_update"] == 49152
                 and original["transitions_per_branch"] == 800 * 49152,
                 "prepared branch budget differs")
        for schedule in schedules:
            identity = {"schedule": schedule, "training_seed": original["training_seed"],
                        "requested_k": original["requested_k"], "coefficient": original["coefficient"]}
            branches.append({**identity, "id": "branch_" + _digest(identity)[:24],
                             "parent_checkpoint": deepcopy(original["parent_checkpoint"]),
                             "consumed_update_offset": original["consumed_update_offset"],
                             "anchors": [_receipt(path) for path in original["anchor_paths"]]})
    body = {"format": FORMAT, "schema_version": 1, "prepared_at": datetime.now(timezone.utc).isoformat(),
            "source": learner_source(), "provider_source": deepcopy(prep["source_identity"]),
            "physical_validator_source": _receipt(TOOLS / "run_frame_diagnostic_campaign.py"),
            "preparation_protocol": _receipt(prep_path), "preparation_protocol_sha256": prep["sha256"],
            "preparation_directory": str(Path(preparation_directory).resolve()),
            "preparation_manifest": _receipt(Path(preparation_directory) / "manifest.json"),
            "preparation_manifest_sha256": result["sha256"], "output_root": str(destination),
            "protected_roots": protected,
            "environment_factory": bundle["manifest"]["environment_factory"], "schedules": schedules,
            "retention_seed": retention_seed, "retention_batch_size": retention_batch_size,
            "evaluation": deepcopy(q["evaluation"]), "evaluation_cases": deepcopy(bundle["manifest"]["scenarios"]),
            "study_root": str(bundle["study"]), "branches": branches,
            "expected_prepared_branches": result["expected_branches"],
            "expected_branches": 2 * result["expected_branches"],
            "expected_evaluation_suites": 4 * result["expected_branches"],
            "expected_evaluation_cells": 4 * result["expected_branches"] * len(bundle["manifest"]["scenarios"]),
            "execution": {"updates": 800, "rollout_steps": 48, "transitions_per_update": 49152,
                "fresh_transition_budget": 800 * 49152, "consumed_update_budget": 800,
                "max_seconds": max_seconds, "checkpoint_interval": bundle["manifest"]["training"]["checkpoint_interval"],
                "device": device, "tensorboard": tensorboard,
                "worker_timeout_seconds": worker_timeout_seconds, "max_wait_seconds": max_wait_seconds,
                "poll_seconds": poll_seconds, "resource_lock": resource_lock,
                "resource_locks": resource_locks,
                "dependencies": dependencies},
            "failure_policy": "reserve_unverified_remaining_budget_no_automatic_retry",
            "skill_scope": q["skill_scope"], "hardware_verified": False,
            "formal_architecture_selection": False, "execution_status": "unexecuted"}
    return _sealed(body)


def validate_protocol(protocol):
    _require(isinstance(protocol, dict) and protocol.get("format") == FORMAT
             and type(protocol.get("schema_version")) is int and protocol["schema_version"] == 1
             and _digest({k: v for k, v in protocol.items() if k != "sha256"}) == protocol.get("sha256"),
             "retention campaign protocol identity differs")
    _require(protocol.get("source") == learner_source(), "retention producer source changed")
    _checked(protocol["physical_validator_source"])
    prep_path = _checked(protocol["preparation_protocol"])
    _checked(protocol["preparation_manifest"])
    execution = protocol["execution"]
    dependencies = execution["dependencies"]
    _require(isinstance(dependencies, list) and [item.get("role") for item in dependencies]
             == ["curriculum", "diagnostics", "learning"], "predecessor queue coverage differs")
    for item in dependencies:
        _checked(item["definition"])
    # Recompute every branch/config/denominator with the unmodified provider.
    # Existing output is permitted solely for this revalidation call.
    expected = freeze(prep_path, protocol["preparation_directory"], output_root=protocol["output_root"],
        retention_seed=protocol["retention_seed"], diagnostic_dependency=dependencies[1]["summary_path"],
        learning_dependency=dependencies[2]["summary_path"], device=execution["device"],
        retention_batch_size=protocol["retention_batch_size"], tensorboard=execution["tensorboard"],
        worker_timeout_seconds=execution["worker_timeout_seconds"], max_wait_seconds=execution["max_wait_seconds"],
        poll_seconds=execution["poll_seconds"], _allow_existing_output=True)
    _equivalent(protocol, expected)
    return protocol


def _equivalent(protocol, expected):
    ignored = {"prepared_at", "sha256"}
    _require(_bytes({k: v for k, v in protocol.items() if k not in ignored})
             == _bytes({k: v for k, v in expected.items() if k not in ignored}),
             "retention campaign source, branches, configuration or budget changed")


def build_worker_request(protocol, protocol_path, branch, run_directory, controller_lease):
    """Only a fresh branch request; failed or partial attempts are not refunded."""
    _require(any(_bytes(branch) == _bytes(item) for item in protocol["branches"]),
             "worker branch is not in the frozen grid")
    execution = protocol["execution"]
    parent = branch["parent_checkpoint"]
    body = {"format": REQUEST_FORMAT, "schema_version": 1,
        "config": deepcopy(protocol["schedules"][branch["schedule"]]),
        "environment_factory": protocol["environment_factory"],
        "checkpoint": {"path": parent["path"], "sha256": parent["sha256"], "update": parent["update"],
                       "cumulative_transitions": parent["cumulative_transitions"],
                       "consumed_updates": branch["consumed_update_offset"]},
        "training_seed": branch["training_seed"],
        "retention": {"seed": protocol["retention_seed"], "coefficient": branch["coefficient"],
                      "batch_size": protocol["retention_batch_size"], "anchors": deepcopy(branch["anchors"]),
                      "evaluation_seeds": list(protocol["evaluation"]["seeds"])},
        "execution": {**{k: execution[k] for k in ("updates", "rollout_steps", "transitions_per_update",
            "fresh_transition_budget", "consumed_update_budget", "max_seconds", "checkpoint_interval", "device", "tensorboard")},
            "run_directory": str(Path(run_directory).resolve()), "resume": False},
        "provenance": {"controller_protocol_path": str(Path(protocol_path).resolve()),
            "controller_protocol_sha256": protocol["sha256"],
            "preparation_protocol_sha256": protocol["preparation_protocol_sha256"],
            "preparation_manifest_sha256": protocol["preparation_manifest_sha256"], "branch_id": branch["id"],
            "controller_lease": deepcopy(controller_lease)},
        "source": deepcopy(protocol["source"])}
    return _sealed(body)


def validate_worker_request(request):
    """Reaudit qualification, manifest and the exact authorized branch before startup."""
    _require(isinstance(request, dict) and _digest({k: v for k, v in request.items() if k != "sha256"})
             == request.get("sha256"), "worker request identity differs")
    provenance = request.get("provenance", {})
    protocol_path = provenance.get("controller_protocol_path")
    _require(isinstance(protocol_path, str) and Path(protocol_path).is_absolute(),
             "worker requires an absolute frozen controller protocol")
    protocol = validate_protocol(_read(protocol_path))
    _require(protocol["sha256"] == provenance.get("controller_protocol_sha256"),
             "worker controller protocol differs")
    candidates = [item for item in protocol["branches"] if item["id"] == provenance.get("branch_id")]
    _require(len(candidates) == 1, "worker branch is missing or duplicated")
    branch = candidates[0]
    expected_directory = Path(protocol["output_root"]) / branch["id"] / "attempt_001" / "train"
    leases = provenance.get("controller_lease")
    validate_lease(protocol, leases)
    expected = build_worker_request(protocol, protocol_path, branch, expected_directory, leases)
    _require(_bytes(request) == _bytes(expected), "worker request configuration, teacher, branch, sampler or budget differs")
    for receipt in branch["anchors"]:
        _checked(receipt)
    parent = branch["parent_checkpoint"]
    _checked({k: parent[k] for k in ("path", "sha256", "bytes")})
    from .queue_validation import check_dependency
    _require(all(check_dependency(item, require_complete=True)["status"] == "completed"
                 for item in protocol["execution"]["dependencies"]), "predecessor queues are not closed")
    attempt = expected_directory.parent
    _require((attempt / "request.json").read_bytes() == _bytes(request) + b"\n",
             "worker request is not the controller's published request")
    reservation = _read(attempt / "reservation.json")
    _require(reservation == reservation_for(protocol, branch), "worker reservation or branch budget differs")
    return request


def _process(pid):
    _require(type(pid) is int and pid > 0, "invalid process PID")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return {"pid": pid, "start": int(fields[19]), "state": fields[0],
                "argv": [value.decode(errors="replace") for value in argv if value]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _lock_signature(path):
    path = Path(path)
    _require(path.is_absolute() and not path.is_symlink() and path.resolve() == path,
             "lock path must be absolute and canonical")
    info = path.stat()
    _require(path.is_file(), "lock must be an existing regular file")
    return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def validate_lease(protocol, leases):
    """Verify inherited, actually locked FDs and the live controller parent."""
    _require(type(leases) is list and len(leases) == 2, "two inherited controller leases are required")
    expected = protocol["execution"]["resource_locks"]
    controller = _read(Path(protocol["output_root"]) / "controller.json")
    parent = _process(os.getppid())
    _require(parent is not None and parent["state"] != "Z"
             and controller["process"] == {k: parent[k] for k in ("pid", "start", "argv")}
             and controller["protocol_sha256"] == protocol["sha256"]
             and controller["source"] == protocol["source"], "worker controller parent or source differs")
    _require(len({item.get("descriptor") for item in leases}) == 2,
             "inherited lease descriptors must be distinct")
    for item, pin in zip(leases, expected):
        _require(type(item) is dict and set(item) == {"path", "device", "inode", "descriptor",
                 "controller_pid", "controller_start"}
                 and all(type(item[k]) is int for k in ("device", "inode", "descriptor", "controller_pid", "controller_start"))
                 and item["descriptor"] >= 0 and item["controller_pid"] == parent["pid"]
                 and item["controller_start"] == parent["start"]
                 and {k: item[k] for k in ("path", "device", "inode")} == pin,
                 "lease identity differs from the frozen resource locks")
        descriptor = item["descriptor"]
        own = os.fstat(descriptor)
        parent_fd = Path(f"/proc/{parent['pid']}/fd/{descriptor}").stat()
        _require((own.st_dev, own.st_ino) == (pin["device"], pin["inode"])
                 == (parent_fd.st_dev, parent_fd.st_ino)
                 and _lock_signature(pin["path"]) == pin, "inherited resource lock was replaced")
        info = Path(f"/proc/self/fdinfo/{descriptor}").read_text()
        _require(any("FLOCK" in line and "ADVISORY" in line and "WRITE" in line
                     for line in info.splitlines() if line.startswith("lock:")),
                 "inherited descriptor does not hold an exclusive flock")
        # A separate open must conflict. Merely inheriting a file descriptor
        # without holding its lock cannot authorize a simulator worker.
        check = os.open(pin["path"], os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(check, fcntl.LOCK_UN)
                raise ValueError("resource lease is not exclusively held")
        finally:
            os.close(check)


def reservation_for(protocol, branch):
    execution = protocol["execution"]
    return {"format": "transformer_rl.continuation_reservation", "schema_version": 1,
            "protocol_sha256": protocol["sha256"], "branch_id": branch["id"],
            "reserved_updates": execution["consumed_update_budget"],
            "reserved_samples": execution["fresh_transition_budget"],
            "parent_checkpoint_sha256": branch["parent_checkpoint"]["sha256"],
            "no_automatic_retry": True}


def _write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish(path, value):
    temporary = Path(str(path) + ".pending")
    _write_new(temporary, value)
    os.replace(temporary, path)


@contextmanager
def _resource_lease(protocol, publish):
    from .queue_validation import check_dependency
    start = time.monotonic()
    execution = protocol["execution"]
    streams = []
    try:
        while True:
            states = [check_dependency(item, require_complete=True) for item in execution["dependencies"]]
            ready = all(item["status"] == "completed" for item in states)
            if ready:
                try:
                    for pin in execution["resource_locks"]:
                        _require(_lock_signature(pin["path"]) == pin, "frozen queue lock changed")
                        descriptor = os.open(pin["path"], os.O_RDWR | os.O_NOFOLLOW)
                        streams.append(descriptor)
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    for descriptor in streams:
                        os.close(descriptor)
                    streams.clear()
            publish("waiting", dependency_states=states)
            _require(time.monotonic() - start < execution["max_wait_seconds"],
                     "bounded predecessor wait expired; no worker launched")
            time.sleep(execution["poll_seconds"])
        # The same inode leases stay alive in a child even if its controller
        # dies; no next queue can run while an orphan simulator still owns them.
        _require(all(check_dependency(item, require_complete=True)["status"] == "completed"
                     for item in execution["dependencies"]), "predecessor changed while acquiring resources")
        process = _process(os.getpid())
        leases = [{**pin, "descriptor": descriptor, "controller_pid": process["pid"],
                   "controller_start": process["start"]}
                  for pin, descriptor in zip(execution["resource_locks"], streams)]
        yield leases
    finally:
        # Closing, rather than LOCK_UN, preserves an inherited child's lease
        # after an interrupted controller exits.
        for descriptor in streams:
            os.close(descriptor)


def _environment(directory):
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(PACKAGE.parent), PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(Path(directory) / "empty_python_cache"), OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return environment


def _pidfd_api():
    """Use the Linux kernel handle even in Python builds without pidfd APIs."""
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        return os.pidfd_open, lambda fd, sig: signal.pidfd_send_signal(fd, sig)
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    def checked(result):
        if result < 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        return result
    if not (hasattr(library, "pidfd_open") and hasattr(library, "pidfd_send_signal")):
        import platform
        # Older Python and libc builds can omit wrappers for an available
        # kernel ABI. Never fall back to signaling a reusable numeric PID.
        if (sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8
                or platform.machine() not in ("x86_64", "aarch64")):
            raise OSError("pidfd syscall fallback requires Linux x86_64/aarch64 LP64")
        syscall = library.syscall
        syscall.argtypes, syscall.restype = [ctypes.c_long], ctypes.c_long
        return (lambda pid: checked(syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))),
                lambda fd, sig: checked(syscall(424, ctypes.c_int(fd), ctypes.c_int(sig),
                                               ctypes.c_void_p(), ctypes.c_uint(0))))
    opening = library.pidfd_open
    opening.argtypes, opening.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    sending = library.pidfd_send_signal
    sending.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    sending.restype = ctypes.c_int
    return (lambda pid: checked(opening(pid, 0)),
            lambda fd, sig: checked(sending(fd, sig, None, 0)))


def _launch(command, directory, leases, timeout, publish):
    """Wait for this exact child; observation timeouts never restart a job."""
    directory = Path(directory)
    process = None
    child_handle = None
    identity = None
    timed_out = False
    interrupted = None
    started = time.monotonic()
    open_handle, signal_handle = _pidfd_api()
    # Fail before creating a child if its kernel cannot provide process handles.
    probe = open_handle(os.getpid())
    os.close(probe)
    with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                env=_environment(directory), pass_fds=tuple(item["descriptor"] for item in leases))
            # A kernel handle stays bound to this child even if /proc cannot be
            # read before its process receipt is published.
            child_handle = open_handle(process.pid)
            live = _process(process.pid)
            _require(live is not None, "child process handle vanished before its identity was recorded")
            identity = {k: live[k] for k in ("pid", "start", "argv")}
            _write_new(directory / "worker.process.json", {"format": "transformer_rl.retention_worker_process",
                "schema_version": 1, "command": command, "process": identity,
                "leases": leases, "started_at": datetime.now(timezone.utc).isoformat()})
            while True:
                try:
                    code = process.wait(timeout=min(20., max(.01, timeout - (time.monotonic() - started))))
                    break
                except subprocess.TimeoutExpired:
                    live = _process(process.pid)
                    _require(live is not None and live["start"] == identity["start"],
                             "child process identity changed during observation")
                    publish("running", worker=identity, elapsed_s=time.monotonic() - started)
                    if time.monotonic() - started >= timeout:
                        timed_out = True
                        break
        except BaseException as error:
            interrupted = error
        finally:
            if process is not None and process.poll() is None:
                live = _process(process.pid) if identity is not None else None
                if identity is None:
                    _require(child_handle is not None, "child has no verified kernel process handle")
                    signal_handle(child_handle, signal.SIGTERM)
                else:
                    _require(live is not None and live["start"] == identity["start"],
                             "refusing to signal a worker with an unverified process identity")
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    if identity is None:
                        signal_handle(child_handle, signal.SIGKILL)
                    else:
                        live = _process(process.pid)
                        _require(live is not None and live["start"] == identity["start"], "worker changed before hard shutdown")
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10.)
            if child_handle is not None:
                os.close(child_handle)
    result = {"format": "transformer_rl.retention_worker_completion", "schema_version": 1,
              "process": identity, "command": command, "returncode": process.returncode if process else None,
              "timed_out": timed_out, "elapsed_s": time.monotonic() - started}
    _write_new(directory / "worker.completion.json", result)
    if interrupted is not None:
        raise interrupted
    _require(result["returncode"] == 0 and not timed_out, "retention child failed or timed out; no retry")
    return result


def verify_training(protocol, branch, request, directory):
    """Reconstruct complete rollout/Adam/private-sampler evidence on CPU."""
    import torch
    from .frame_checkpoint import load_frame_checkpoint
    from .private_retention import PrivateAnchorRegularizer
    from .frame_workflow import _model_state_sha256
    directory = Path(directory)
    execution = protocol["execution"]
    report = _read(directory / "completion.json")
    parent = request["checkpoint"]
    expected_update = parent["update"] + execution["updates"]
    expected_samples = parent["cumulative_transitions"] + execution["fresh_transition_budget"]
    expected = {"format": "transformer_rl.continuation_execution", "schema_version": 1,
        "status": "completed", "request_sha256": request["sha256"], "provenance": request["provenance"],
        "source": protocol["source"], "start_update": parent["update"], "final_update": expected_update,
        "requested_updates": execution["updates"], "completed_updates": execution["updates"],
        "attempted_updates": execution["updates"], "actual_samples": execution["fresh_transition_budget"],
        "consumed_transitions": execution["fresh_transition_budget"], "cumulative_transitions": expected_samples,
        "consumed_updates": parent["consumed_updates"] + execution["updates"], "partial_rollouts": 0,
        "partial_samples": 0, "discarded_samples": 0, "failed_optimizer_samples": 0, "failed_collection_samples": 0,
        "unsealed_successful_updates": 0, "unsealed_samples": 0, "optimizer_update_may_be_partial": False,
        "reserved_update_budget": 0, "reserved_transition_budget": 0, "shutdown_errors": [],
        "episode_state_restored": False, "history_reset": "repeat_first", "stop_reason": None,
        "failure_stage": None, "retry_policy": "reserve_remaining_budget_no_automatic_retry"}
    _require(_bytes({k: report.get(k) for k in expected}) == _bytes(expected), "worker completion/budget differs")
    checkpoint = directory / "checkpoints" / "final.pt"
    _require(report["checkpoint"] == str(checkpoint) and _sha(checkpoint) == report["checkpoint_sha256"],
             "final checkpoint path or bytes differ")
    sealed_checkpoint = {**_receipt(checkpoint), "update": expected_update,
        "cumulative_transitions": expected_samples,
        "consumed_updates": parent["consumed_updates"] + execution["updates"]}
    _require(_bytes(report.get("last_sealed_checkpoint")) == _bytes(sealed_checkpoint),
             "completed worker sealed checkpoint ledger differs")
    _require(_bytes(_read(directory / "run.json")) == _bytes(request), "published worker request differs")
    _require(_sha(parent["path"]) == parent["sha256"], "teacher checkpoint bytes changed")
    rows = [_read_line(line) for line in (directory / "metrics.jsonl").read_bytes().splitlines()]
    _require(len(rows) == execution["updates"], "worker PPO row coverage differs")
    optimizer_steps = 0
    for offset, row in enumerate(rows, 1):
        opt = row["optimization"]
        samples = execution["transitions_per_update"]
        chunks = min(samples, request["config"]["ppo"]["num_minibatches"])
        planned_steps = chunks * request["config"]["ppo"]["epochs"]
        steps = opt["optimizer_steps"]
        _require(type(steps) is int and 0 < steps <= planned_steps,
                 "applied optimizer count exceeds the PPO recipe")
        full_epochs, last_chunks = divmod(steps, chunks)
        small, extra = divmod(samples, chunks)
        applied_samples = full_epochs * samples + last_chunks * small + min(last_chunks, extra)
        _require(type(row["update"]) is int and row["update"] == parent["update"] + offset
                 and type(row["batch_samples"]) is int and row["batch_samples"] == execution["transitions_per_update"]
                 and type(opt["optimizer_steps"]) is int and opt["optimizer_steps"] > 0
                 and type(opt["sample_count"]) is int and opt["sample_count"] == applied_samples
                 and type(opt["planned_optimizer_steps"]) is int
                 and opt["planned_optimizer_steps"] == planned_steps
                 and type(row["consumed_updates"]) is int
                 and row["consumed_updates"] == parent["consumed_updates"] + offset
                 and type(row["cumulative_transitions"]) is int
                 and row["cumulative_transitions"] == parent["cumulative_transitions"] + offset * samples,
                 "applied PPO samples or optimizer count differ")
        optimizer_steps += opt["optimizer_steps"]
    with torch.device("cpu"):
        model, trainer, config, update, metadata, rng = load_frame_checkpoint(checkpoint)
        parent_model, parent_trainer, parent_config, parent_update, parent_metadata, _ = load_frame_checkpoint(parent["path"])
        sampler = PrivateAnchorRegularizer(model.actor, config, [a["path"] for a in branch["anchors"]],
            branch["coefficient"], seed=protocol["retention_seed"], batch_size=protocol["retention_batch_size"])
        sampler.load_state_dict(metadata["continuation"]["retention"], max_replay_calls=optimizer_steps)
    binding = {key: request["provenance"][key] for key in ("controller_protocol_sha256",
        "preparation_protocol_sha256", "preparation_manifest_sha256", "branch_id")}
    clock = {"consumed_updates": parent["consumed_updates"] + execution["updates"],
             "collected_transitions": expected_samples, "rollout_steps": execution["rollout_steps"]}
    continuation = metadata.get("continuation")
    _require(isinstance(continuation, dict)
             and set(continuation) == {"format", "schema_version", "clock", "retention"}
             and continuation["format"] == "transformer_rl.frame_continuation"
             and type(continuation["schema_version"]) is int and continuation["schema_version"] == 1
             and _bytes(continuation["clock"]) == _bytes(clock), "unsupported continuation state or clock")
    _require(not execution["device"].startswith("cuda:") or bool(rng["cuda"]),
             "CUDA continuation checkpoint has no saved CUDA learning RNG")
    parent_binding = {"path": parent["path"], "sha256": parent["sha256"],
                      "update": parent["update"], "resume": False}
    segment = {"start_update": parent["update"], "successful_updates": execution["updates"],
               "attempted_updates": execution["updates"],
               "fresh_transitions": execution["fresh_transition_budget"], "discarded_transitions": 0}
    _require(_bytes(metadata.get("continuation_parent")) == _bytes(parent_binding)
             and _bytes(metadata.get("continuation_segment")) == _bytes(segment)
             and metadata.get("continuation_device") == execution["device"]
             and metadata.get("initial_model_sha256") == _model_state_sha256(parent_model)
             and metadata.get("initial_model_hash_format") == "sorted_named_tensor_contents_v1"
             and parent_update == parent["update"]
             and parent_metadata.get("collected_transitions") == parent["cumulative_transitions"]
             and parent_metadata.get("continuation") is None
             and type(parent_metadata.get("seed")) is int
             and parent_metadata["seed"] == request["training_seed"]
             and parent_metadata.get("environment_factory") == request["environment_factory"]
             and _bytes(metadata.get("parent_source")) == _bytes(parent_metadata.get("source"))
             and all(parent_config.to_dict()[key] == request["config"][key]
                     for key in ("model", "ppo", "control")), "continuation parent, segment or device differs")
    _require(_bytes(_read(directory / "environment.json")) == _bytes(metadata["environment_provenance"])
             and metadata["environment_provenance"]["identity"]
                 == parent_metadata["environment_provenance"]["identity"], "environment provenance differs")
    _require(update == expected_update and config.to_dict() == request["config"]
             and metadata["source"] == protocol["source"] and metadata["continuation_branch"] == binding
             and metadata["seed"] == request["training_seed"]
             and metadata["environment_factory"] == request["environment_factory"]
             and metadata["continuation_request_sha256"] == request["sha256"]
             and _bytes(metadata["continuation"]["clock"]) == _bytes(clock)
             and metadata["collected_transitions"] == expected_samples
             and metadata["continuation"]["retention"]["draw_count"]
                 == (optimizer_steps if branch["coefficient"] > 0 else 0), "checkpoint branch/private RNG/clock differs")
    old_states, states = parent_trainer.optimizer.state_dict(), trainer.optimizer.state_dict()
    _require(states["param_groups"] == old_states["param_groups"] and set(states["state"]) == set(old_states["state"]),
             "optimizer parameter groups or state coverage changed")
    for key, state in states["state"].items():
        _require(state["step"].item() == old_states["state"][key]["step"].item() + optimizer_steps,
                 "actual Adam step continuity differs from PPO log")
    sidecar = _read(str(checkpoint) + ".json")
    _require(sidecar["format"] == "transformer_rl.packed_checkpoint"
             and type(sidecar["schema_version"]) is int and sidecar["schema_version"] == 1
             and sidecar["sha256"] == _sha(checkpoint)
             and _bytes(sidecar["metadata"]) == _bytes(metadata)
             and _bytes(sidecar["config"]) == _bytes(config.to_dict())
             and type(sidecar["update"]) is int and sidecar["update"] == update,
             "checkpoint sidecar differs from the actual CPU-deserialized learning state")
    _require(_sha(checkpoint) == report["checkpoint_sha256"] and _sha(parent["path"]) == parent["sha256"],
             "learning state bytes changed during validation")
    return {"status": "completed", "checkpoint": {**_receipt(checkpoint), "update": update,
            "cumulative_transitions": expected_samples, "consumed_updates": clock["consumed_updates"]},
            "actual_updates": execution["updates"], "actual_samples": execution["fresh_transition_budget"],
            "optimizer_steps": optimizer_steps, "artifacts": {name: _receipt(directory / name)
                for name in ("run.json", "environment.json", "completion.json", "metrics.jsonl")},
            "sidecar": _receipt(str(checkpoint) + ".json"), "full_learning_state_validated": True}


def _read_line(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate PPO row field")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite PPO row")))
    _bytes(value)
    return value


def _evaluation_command(protocol, checkpoint, seed, directory):
    evaluation = protocol["evaluation"]
    cases = protocol["evaluation_cases"]
    return [sys.executable, "-B", "-m", "transformer_rl.frame_process", "evaluate-suite",
        "--checkpoint", checkpoint["path"], "--configs",
        *[str(Path(protocol["study_root"]) / item["config"]) for item in cases], "--outputs",
        *[str(Path(directory) / (item["name"] + ".json")) for item in cases],
        "--steps", str(evaluation["steps"]), "--seed", str(seed), "--device", protocol["execution"]["device"],
        "--control-output", str(Path(directory) / "control.json"), "--trace-output", str(Path(directory) / "trace.npz"),
        "--trace-replicas", str(evaluation["trace_replicas"]), "--settle-steps", str(evaluation["settle_steps"]),
        "--min-steady-samples", str(evaluation["min_steady_samples"])]


def _space_required(protocol):
    # Reserve uncompressed output plus one in-progress trace and checkpoint
    # publication. Compression is not assumed to save any space.
    execution = protocol["execution"]
    trace = protocol["evaluation"]["steps"] * len(protocol["evaluation_cases"]) * 8 * 1024
    checkpoint = max(branch["parent_checkpoint"]["bytes"] for branch in protocol["branches"])
    copies = execution["updates"] // execution["checkpoint_interval"] + 2
    return (protocol["expected_evaluation_suites"] + 2) * trace + protocol["expected_branches"] * copies * checkpoint


def run_protocol(protocol_path):
    """Run the entire frozen grid serially once; preserve failures and missing cells."""
    from .retention_evaluation import verify_suite
    protocol_path = Path(protocol_path)
    _require(protocol_path.is_absolute() and protocol_path.resolve() == protocol_path,
             "controller protocol path must be absolute and canonical")
    protocol = validate_protocol(_read(protocol_path))
    root = Path(protocol["output_root"])
    _require(not os.path.lexists(root), "campaign output exists; no automatic retry or refund")
    root.mkdir(mode=0o700)
    process = _process(os.getpid())
    _write_new(root / "controller.json", {"format": "transformer_rl.retention_controller", "schema_version": 1,
        "protocol_sha256": protocol["sha256"], "source": protocol["source"],
        "process": {k: process[k] for k in ("pid", "start", "argv")}})
    summary = {"format": "transformer_rl.retention_campaign", "schema_version": 1,
        "protocol_sha256": protocol["sha256"], "status": "waiting", "results": {},
        "expected_branches": protocol["expected_branches"], "expected_evaluation_suites": protocol["expected_evaluation_suites"],
        "expected_evaluation_cells": protocol["expected_evaluation_cells"], "charged_updates": 0,
        "reserved_samples": 0, "verified_actual_samples": 0, "formal_architecture_selection": False,
        "hardware_verified": False, "skill_scope": protocol["skill_scope"]}
    def publish(status, **fields):
        summary.update(status=status, **fields, updated_at=datetime.now(timezone.utc).isoformat())
        _publish(root / "summary.json", summary)
    publish("waiting")
    try:
        with _resource_lease(protocol, publish) as leases:
            remaining_bytes = _space_required(protocol)
            for branch in protocol["branches"]:
                validate_protocol(protocol)
                fs = os.statvfs(root)
                _require(fs.f_bavail * fs.f_frsize >= remaining_bytes, "insufficient disk space for the remaining frozen grid")
                attempt = root / branch["id"] / "attempt_001"
                attempt.mkdir(parents=True, mode=0o700, exist_ok=False)
                request = build_worker_request(protocol, protocol_path, branch, attempt / "train", leases)
                _write_new(attempt / "reservation.json", reservation_for(protocol, branch))
                _write_new(attempt / "request.json", request)
                summary["charged_updates"] += protocol["execution"]["consumed_update_budget"]
                summary["reserved_samples"] += protocol["execution"]["fresh_transition_budget"]
                result = {"status": "reserved", "reservation": _receipt(attempt / "reservation.json"),
                          "request": _receipt(attempt / "request.json"), "evaluations": {}}
                summary["results"][branch["id"]] = result
                publish("training", active_branch=branch["id"])
                command = [sys.executable, "-B", "-m", "transformer_rl.continuation_process", "--request", str(attempt / "request.json")]
                result["worker"] = _launch(command, attempt, leases, protocol["execution"]["worker_timeout_seconds"], publish)
                result["training"] = verify_training(protocol, branch, request, attempt / "train")
                summary["verified_actual_samples"] += result["training"]["actual_samples"]
                for seed in protocol["evaluation"]["seeds"]:
                    validate_protocol(protocol)
                    directory = root / branch["id"] / "evaluation" / f"seed_{seed}"
                    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
                    publish("evaluating", active_branch=branch["id"], active_evaluation_seed=seed)
                    worker = _launch(_evaluation_command(protocol, result["training"]["checkpoint"], seed, directory),
                        directory, leases, protocol["execution"]["worker_timeout_seconds"], publish)
                    receipt = verify_suite(protocol, branch, result["training"]["checkpoint"], seed, directory)
                    result["evaluations"][str(seed)] = {"worker": worker, "suite": receipt}
                    _write_new(directory / "receipt.json", receipt)
                result["status"] = "completed"
                _write_new(attempt / "receipt.json", result)
                # The reservation stays charged. Only the unspent disk reserve
                # decreases after this complete branch is closed.
                trace = protocol["evaluation"]["steps"] * len(protocol["evaluation_cases"]) * 8 * 1024
                remaining_bytes -= 2 * trace + (protocol["execution"]["updates"] //
                    protocol["execution"]["checkpoint_interval"] + 2) * branch["parent_checkpoint"]["bytes"]
                publish("ready_for_next_branch", active_branch=None, active_evaluation_seed=None)
        _require(len(summary["results"]) == protocol["expected_branches"]
                 and all(item["status"] == "completed" for item in summary["results"].values()),
                 "full retention grid coverage differs")
        publish("completed", active_branch=None, active_evaluation_seed=None)
    except BaseException as error:
        active = summary.get("active_branch")
        if active in summary["results"]:
            summary["results"][active]["status"] = "incomplete"
        publish("incomplete", error=f"{type(error).__name__}: {error}", no_automatic_retry=True)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--preparation-protocol", type=Path, required=True)
    freezing.add_argument("--preparation-directory", type=Path, required=True)
    freezing.add_argument("--output-root", type=Path, required=True)
    freezing.add_argument("--protocol-output", type=Path, required=True)
    freezing.add_argument("--diagnostic-summary", type=Path, required=True)
    freezing.add_argument("--learning-summary", type=Path, required=True)
    freezing.add_argument("--retention-seed", type=int, required=True)
    freezing.add_argument("--device", default="cpu")
    running = commands.add_parser("run")
    running.add_argument("--protocol", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.operation == "freeze":
            result = freeze(arguments.preparation_protocol, arguments.preparation_directory,
                output_root=arguments.output_root, retention_seed=arguments.retention_seed,
                diagnostic_dependency=arguments.diagnostic_summary, learning_dependency=arguments.learning_summary,
                device=arguments.device)
            target = arguments.protocol_output.resolve()
            _require(not target.is_relative_to(Path(result["output_root"])), "protocol must be outside the new campaign output")
            _guard_output(target, result["protected_roots"])
            _write_new(target, result)
        else:
            result = run_protocol(arguments.protocol)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

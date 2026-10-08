"""Externally pinned, serial OS execution of a complete exposure definition.

Training closure and physical evaluation closure are deliberately separate.
Every stage owns a fresh process, while the controller owns one whole-job
reservation and two inherited kernel leases. A new protocol hash is never an
implicit authorization to execute changed inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile

import numpy as np

from . import exposure_protocol as definition
from . import exposure_training as training
from . import retention_campaign as predecessors
from . import runtime_paths
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest, json_bytes


class CampaignIntegrityError(ValueError):
    """An input, learning proof, process identity or authorization changed."""


class CampaignDiskError(OSError):
    """The fixed remaining outputs no longer fit their declared storage caps."""


class CampaignWorkerError(RuntimeError):
    """An unknown or incomplete worker result must stop the campaign."""


def _require(condition, message):
    if not condition:
        raise CampaignIntegrityError(message)


def _read(path):
    path = definition._path(path)
    value = definition._read(path)
    _require(path.read_bytes() == json_bytes(value) + b"\n", "canonical published JSON required")
    return value


def _checked(receipt):
    _require(type(receipt) is dict and set(receipt) == {"path", "sha256", "bytes"},
             "actual file receipt required")
    _require(definition._receipt(receipt["path"]) == receipt, "input receipt bytes changed")
    return Path(receipt["path"])


def _new(path, value):
    predecessors._write_new(path, value)


def _publish(path, value):
    predecessors._publish(path, value)


def read_authorized_protocol(path, expected_protocol_sha256):
    """Compare actual raw bytes to a caller's fixed authorization before parse."""
    _require(type(expected_protocol_sha256) is str and re.fullmatch(
        r"[0-9a-f]{64}", expected_protocol_sha256), "explicit external raw protocol SHA required")
    receipt = definition._receipt(path)
    _require(receipt["sha256"] == expected_protocol_sha256,
             "raw protocol differs from external authorization")
    protocol = definition.validate_protocol(definition._read(path))
    _require(definition._receipt(path) == receipt, "protocol changed during validation")
    return protocol, receipt


def _validate_storage(contract, protocol, protocol_receipt):
    fields = {"format", "schema_version", "protocol_raw_sha256", "source", "caps", "measurements"}
    _require(type(contract) is dict and set(contract) == fields
             and contract["format"] == "transformer_rl.exposure_storage_contract"
             and type(contract["schema_version"]) is int and contract["schema_version"] == 1
             and contract["protocol_raw_sha256"] == protocol_receipt["sha256"]
             and contract["source"] == protocol["source"], "storage contract identity differs")
    caps = contract["caps"]
    cap_names = {"checkpoint_bytes", "trace_bytes_per_policy_sample", "metric_bytes_per_update",
                 "inflight_bytes", "runtime_cache_bytes", "free_margin_bytes"}
    _require(type(caps) is dict and set(caps) == cap_names
             and all(type(v) is int and v > 0 for v in caps.values()), "positive integer storage caps required")
    measurements = contract["measurements"]
    _require(type(measurements) is list and bool(measurements), "actual storage measurements required")
    observed = set()
    for item in measurements:
        _require(type(item) is dict and set(item) == {"kind", "receipt", "units"}
                 and item["kind"] in {"checkpoint", "trace", "metric", "runtime_cache"}
                 and type(item["units"]) is int and item["units"] > 0,
                 "invalid storage measurement")
        measurement_path = _checked(item["receipt"])
        measured_bytes = item["receipt"]["bytes"]
        if item["kind"] == "trace":
            measured_bytes = measured_trace_bytes(measurement_path, item["units"])
        if item["kind"] == "runtime_cache":
            inventory = _read(measurement_path)
            _require(set(inventory) == {"format", "schema_version", "root", "files", "total_bytes"}
                     and inventory["format"] == "transformer_rl.exposure_runtime_cache_measurement"
                     and inventory["schema_version"] == 1
                     and type(inventory["files"]) is list and bool(inventory["files"]),
                     "actual complete runtime-cache inventory required")
            cache_root = definition._path(inventory["root"])
            actual = []
            for base, directories, names in os.walk(cache_root, followlinks=False):
                for name in directories:
                    directory = definition._path(Path(base) / name)
                    _require(stat.S_ISDIR(directory.lstat().st_mode), "cache tree special directory")
                for name in names:
                    actual.append(definition._receipt(Path(base) / name))
            actual.sort(key=lambda receipt: receipt["path"])
            _require(actual == inventory["files"] and sum(f["bytes"] for f in actual) == inventory["total_bytes"],
                     "actual runtime-cache inventory or bytes changed")
            measured_bytes = inventory["total_bytes"]
        key = {"checkpoint": "checkpoint_bytes", "trace": "trace_bytes_per_policy_sample",
               "metric": "metric_bytes_per_update", "runtime_cache": "runtime_cache_bytes"}[item["kind"]]
        if item["kind"] in {"checkpoint", "runtime_cache"}:
            _require(item["units"] == 1, "checkpoint/cache measurement units must be one")
        _require(caps[key] * item["units"] >= measured_bytes,
                 "storage cap is below an actual measured artifact")
        observed.add(item["kind"])
    _require(observed == {"checkpoint", "trace", "metric", "runtime_cache"},
             "checkpoint, raw trace, optimizer log and runtime cache measurements required")
    return contract


def measured_trace_bytes(path, units):
    """Use complete NPZ payload bytes and actual steps x rows, not compression."""
    _require(type(units) is int and units > 0, "positive trace measurement units required")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _require(bool(members) and len({m.filename for m in members}) == len(members)
                 and all(m.filename.endswith(".npy") and "/" not in m.filename
                         and not m.is_dir() for m in members), "complete NPZ trace measurement required")
        _require(archive.testzip() is None, "trace measurement CRC differs")
        raw_bytes = sum(m.file_size for m in members)
    with np.load(path, allow_pickle=False) as trace:
        _require({"episode_id", "done", "row_indices", "metadata_json"} <= set(trace.files),
                 "trace measurement lacks declared samples")
        episodes, done, rows = trace["episode_id"], trace["done"], trace["row_indices"]
        metadata = json.loads(str(trace["metadata_json"].item()))
        _require(episodes.dtype == np.int64 and episodes.ndim == 2 and all(episodes.shape)
                 and done.dtype == np.bool_ and done.shape == episodes.shape
                 and rows.dtype == np.int64 and rows.shape == (episodes.shape[1],)
                 and metadata["steps"] == episodes.shape[0]
                 and metadata["row_indices"] == rows.tolist()
                 and episodes.size == units, "trace measurement policy-sample units differ")
        for name in trace.files:
            values = trace[name]
            _require(values.dtype.kind in "biufUS" and not values.dtype.hasobject,
                     "trace measurement contains object or unsupported data")
            if values.dtype.kind == "f":
                _require(bool(np.isfinite(values).all()), "trace measurement contains nonfinite data")
    return raw_bytes


def worker_storage_guard(protocol, contract, directory, required_remaining_bytes, *,
                         checkpoint_paths=(), checkpoint_limit=0, metric_paths=(), metric_limit=0,
                         trace_prefix=None, trace_sample_limit=0, active_worker=False):
    """Credit only this worker's declared outputs; cap all other owned bytes."""
    directory = definition._path(directory)
    root_stat = directory.lstat()
    _require(stat.S_ISDIR(root_stat.st_mode) and root_stat.st_uid == os.getuid(),
             "worker output must be an owned directory")
    caps = contract["caps"]
    _require(type(required_remaining_bytes) is int and required_remaining_bytes > 0,
             "fixed positive remaining storage reservation required")
    _require(type(active_worker) is bool, "active worker reservation flag must be boolean")
    active_headroom = caps["inflight_bytes"] + caps["runtime_cache_bytes"] if active_worker else 0
    _require(required_remaining_bytes > active_headroom,
             "active namespace headroom must fit the fixed storage reservation")
    effective_remaining = required_remaining_bytes - active_headroom
    checkpoint_paths, metric_paths = set(map(Path, checkpoint_paths)), set(map(Path, metric_paths))
    _require(all(p.is_absolute() and p.is_relative_to(directory)
                 for p in checkpoint_paths | metric_paths)
             and len(checkpoint_paths) <= 1 and len(metric_paths) <= 1
             and all(type(x) is int and x >= 0 for x in
                     (checkpoint_limit, metric_limit, trace_sample_limit)),
             "exact worker output paths and nonnegative byte limits required")
    trace_prefix = Path(trace_prefix) if trace_prefix is not None else None
    _require(trace_prefix is None or trace_prefix == directory / "trace.npz",
             "trace must belong to the declared evaluation worker")
    sizes = {name: 0 for name in ("checkpoint", "metric", "trace_maps", "trace_archive", "other", "cache")}
    credits = {name: 0 for name in sizes}
    trace_directories, seen = set(), {}
    for base, directories, names in os.walk(directory, followlinks=False):
        base_path = definition._path(base)
        base_stat = base_path.lstat()
        _require(stat.S_ISDIR(base_stat.st_mode) and base_stat.st_uid == os.getuid()
                 and base_stat.st_dev == root_stat.st_dev,
                 "worker directory is not owned or crosses its filesystem")
        retained = []
        for name in directories:
            p = Path(base) / name
            try:
                observed = p.lstat()
            except FileNotFoundError:
                continue
            _require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid()
                     and observed.st_dev == root_stat.st_dev,
                     "worker directory contains a symlink, special entry, foreign owner or filesystem")
            retained.append(name)
        directories[:] = retained
        for name in names:
            p = Path(base) / name
            try:
                s = p.lstat()
            except FileNotFoundError:
                # A live publisher removes its private temporary links and
                # memmaps concurrently. Not crediting a vanished file is
                # conservative; stable required artifacts are checked later.
                continue
            _require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
                     and s.st_dev == root_stat.st_dev,
                     "worker output is not an owned regular file on its filesystem")
            relative = p.relative_to(directory)
            category = "other"
            if relative.parts[0] in ("empty_python_cache", "runtime"):
                category = "cache"
            elif p in checkpoint_paths or any(p.parent == cp.parent and re.fullmatch(
                    re.escape("." + cp.name + ".") + r"[a-zA-Z0-9_-]+\.tmp", p.name) for cp in checkpoint_paths):
                category = "checkpoint"
            elif p in metric_paths:
                category = "metric"
            elif trace_prefix is not None and p == trace_prefix:
                category = "trace_archive"
            elif trace_prefix is not None and len(relative.parts) == 2 \
                    and relative.parts[0].startswith(".control-trace-"):
                trace_directories.add(relative.parts[0])
                if re.fullmatch(r"field_[0-9]+\.npy", name):
                    category = "trace_maps"
                elif name == "trace.npz":
                    category = "trace_archive"
            _require(category in ("cache", "checkpoint") or p.suffix != ".pt",
                     "undeclared checkpoint cannot consume a later stage's reservation")
            inode = (s.st_dev, s.st_ino)
            _require(inode not in seen or seen[inode] == category,
                     "hardlink crosses worker storage budget categories")
            if inode not in seen:
                sizes[category] += s.st_size
                # A new memmap can have its complete logical length while
                # still sparse. Holes are future writes, not spent reserve.
                credits[category] += min(s.st_size, s.st_blocks * 512)
                seen[inode] = category
    _require(len(trace_directories) <= 1, "worker contains multiple trace publications")
    for category, limit in (("checkpoint", checkpoint_limit), ("metric", metric_limit),
                            ("other", caps["inflight_bytes"]), ("cache", caps["runtime_cache_bytes"])):
        _require(sizes[category] <= limit, f"worker {category} storage cap exceeded")
    trace_limit = trace_sample_limit * caps["trace_bytes_per_policy_sample"]
    _require(sizes["trace_maps"] <= trace_limit and sizes["trace_archive"] <= trace_limit,
             "worker raw trace or publication storage cap exceeded")
    if trace_prefix is not None and trace_prefix.is_file():
        _require(measured_trace_bytes(trace_prefix, trace_sample_limit) <= trace_limit,
                 "published uncompressed trace exceeds measured cap")
    credited = sum(credits[name] for name in ("checkpoint", "metric", "trace_maps", "trace_archive"))
    floor = sum(caps[name] for name in ("inflight_bytes", "runtime_cache_bytes", "free_margin_bytes"))
    disk = check_disk(directory, max(floor, effective_remaining - credited))
    return {**disk, "credited_declared_output_bytes": credited, "owned_bytes": sizes,
            "undeclared_output_budget_credit": 0, "active_runtime_headroom_bytes": active_headroom}


def remaining_storage_bytes(protocol, contract, completed_stages=(), closed_cells=()):
    """Reserve all remaining uncompressed output, publication and runtime caps.

The caller supplies measured caps rather than assuming compression or silently
discarding traces. Completed bytes already occupy the filesystem and are not
counted again. Missing cells may be closed explicitly; unexecuted cells remain.
Boundary checks retain one extra active namespace headroom. The running worker
guard consumes that fixed headroom without crediting any cache/log bytes.
"""
    caps = contract["caps"]
    completed, closed = set(completed_stages), set(closed_cells)
    checkpoint_count = updates = 0
    for job in protocol["jobs"]:
        for stage in job["stages"]:
            if (job["id"], stage["index"]) not in completed:
                checkpoint_count += 1
                updates += stage["updates"]
    samples = sum(cell["expected_policy_samples"] for cell in protocol["evaluation_cells"]
                  if cell["id"] not in closed)
    batches = {}
    for cell in protocol["evaluation_cells"]:
        if cell["id"] not in closed:
            key = (cell["job_id"], cell["stage_index"], cell["role"], cell["seed"])
            batches[key] = batches.get(key, 0) + cell["expected_policy_samples"]
    largest_trace = max(batches.values(), default=0) * caps["trace_bytes_per_policy_sample"]
    # Every OS stage/batch keeps its own namespace and runtime proof. Closed
    # files already reduce statvfs available bytes; future caches and ordinary
    # publications must each be reserved, rather than assuming one shared
    # reusable cache or silently deleting historical worker evidence.
    future_worker_count = checkpoint_count + len(batches)
    namespace_cap = caps["inflight_bytes"] + caps["runtime_cache_bytes"]
    active_headroom = namespace_cap if future_worker_count else 0
    return (checkpoint_count * caps["checkpoint_bytes"]
            + updates * caps["metric_bytes_per_update"]
            + samples * caps["trace_bytes_per_policy_sample"]
            + largest_trace
            + caps["checkpoint_bytes"]
            + future_worker_count * namespace_cap
            + active_headroom
            + caps["free_margin_bytes"])


def check_disk(root, required_bytes):
    fs = os.statvfs(root)
    available = fs.f_bavail * fs.f_frsize
    if available < required_bytes:
        raise CampaignDiskError(f"remaining output reserve {required_bytes} exceeds available {available}")
    return {"available_bytes": available, "required_bytes": required_bytes}


def _identity(pid):
    live = predecessors._process(pid)
    _require(live is not None and live["state"] != "Z", "live process identity required")
    return {key: live[key] for key in ("pid", "start", "argv")}


def _startup_identity(pid, command, timeout):
    """Wait for exec's actual argv, without replacing or relaunching the child.

Even after Popen returns, /proc/cmdline can briefly be empty during exec.
The pidfd already binds this process. Such an observation is not a terminal
worker or a usable complete identity receipt.
"""
    deadline = time.monotonic() + min(2., timeout)
    first_start = None
    while True:
        identity = _identity(pid)
        if first_start is None:
            first_start = identity["start"]
        _require(identity["start"] == first_start, "child start changed during exec observation")
        if identity["argv"] == command:
            return identity
        _require(time.monotonic() < deadline, "child argv did not reach the authorized command")
        time.sleep(.005)


def validate_controller_lease(protocol, leases, controller_receipt):
    """Validate actual controller bytes, direct live parent and inherited locks."""
    controller_path = _checked(controller_receipt)
    _require(controller_path == Path(protocol["output_root"]) / "controller.json",
             "controller receipt is outside the fixed campaign")
    controller = _read(controller_path)
    _require(controller.get("format") == "transformer_rl.exposure_controller"
             and controller.get("schema_version") == 1
             and controller["source"] == protocol["source"]
             and controller["runtime"] == protocol["runtime"]
             and controller["process"] == _identity(os.getppid()),
             "worker direct parent, source or runtime differs")
    pinned_protocol = _checked(controller["protocol_raw_receipt"])
    _require(controller["expected_protocol_sha256"] == controller["protocol_raw_receipt"]["sha256"],
             "controller raw authorization differs")
    # The complete protocol reconstruction precedes worker authorization.
    # During rollout callbacks recheck its actual bytes and parsed identity;
    # rebuilding every candidate model on each physical step would distort the
    # control workload. Controller boundaries perform the full reconstruction.
    _require(definition._read(pinned_protocol) == protocol
             and definition._receipt(pinned_protocol) == controller["protocol_raw_receipt"],
             "controller protocol bytes differ")
    _checked(controller["storage_contract_receipt"])
    predecessors.validate_lease(protocol, leases)
    for item in leases:
        info = Path(f"/proc/{os.getppid()}/fdinfo/{item['descriptor']}").read_text()
        _require(any("FLOCK" in line and "ADVISORY" in line and "WRITE" in line
                     for line in info.splitlines() if line.startswith("lock:")),
                 "parent descriptor no longer holds the inherited flock")
    _checked(controller_receipt)


@contextmanager
def resource_lease(protocol, publish):
    """Reuse the original three-queue closure proof and exact two flock pins."""
    with predecessors._resource_lease(protocol, publish) as leases:
        yield leases


def _worker_environment(directory):
    directory = definition._path(directory)
    cache = directory / "empty_python_cache"
    _require(not os.path.lexists(cache), "each worker requires a new empty bytecode cache")
    cache.mkdir(mode=0o700)
    environment = predecessors._environment(directory)
    environment["PYTHONPYCACHEPREFIX"] = str(cache)
    return runtime_paths.prepare_worker_runtime(directory, environment)


def launch_owned_worker(command, directory, leases, timeout, publish, *, worker_env=None, monitor=None):
    """Launch once and signal only this child through its kernel pidfd."""
    _require(type(command) is list and bool(command) and all(type(x) is str for x in command),
             "explicit OS command required")
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
             "positive finite worker timeout required")
    _require(monitor is None or callable(monitor), "worker monitor must be callable")
    directory = definition._path(directory)
    environment = _worker_environment(directory)
    if worker_env is not None:
        _require(type(worker_env) is dict
                 and not (runtime_paths.RESERVED_ENVIRONMENT & set(worker_env)),
                 "worker environment cannot override input or runtime path isolation")
        environment.update(worker_env)
    runtime_profile = runtime_paths.profile_receipt(
        runtime_paths.validate_runtime_profile(directory, environ=environment))

    def guard_runtime():
        _require(runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(
            directory, environ=environment)) == runtime_profile,
            "worker runtime profile changed during execution")
        if monitor is not None:
            monitor()

    opening, sending = predecessors._pidfd_api()
    # Verify both pidfd opening AND signal support before Popen. Signal zero is
    # an existence/permission probe and does not modify the controller.
    probe = opening(os.getpid())
    try:
        sending(probe, 0)
    finally:
        os.close(probe)
    process = handle = identity = None
    failure, timed_out = None, False
    started = time.monotonic()
    with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                start_new_session=True, env=environment,
                pass_fds=tuple(item["descriptor"] for item in leases))
            handle = opening(process.pid)
            identity = _startup_identity(process.pid, command, timeout)
            _new(directory / "worker.process.json", {"format": "transformer_rl.exposure_worker_process",
                "schema_version": 1, "process": identity, "command": command, "leases": leases,
                "runtime_profile": runtime_profile,
                "started_at": datetime.now(timezone.utc).isoformat()})
            while True:
                guard_runtime()
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(1., remaining))
                    guard_runtime()
                    break
                except subprocess.TimeoutExpired:
                    if process.poll() is not None:
                        guard_runtime()
                        break
                    live = predecessors._process(process.pid)
                    if live is not None:
                        _require(live["pid"] == identity["pid"] and live["start"] == identity["start"],
                                 "child identity changed during observation")
                        _require(live["argv"] == identity["argv"] or not live["argv"],
                                 "child command changed during observation")
                        if live["state"] != "Z" and live["argv"] == identity["argv"]:
                            publish("running", worker=identity, elapsed_s=time.monotonic() - started)
                            continue
                    # Kernel exit clears cmdline/mm before the same child is
                    # necessarily waitable. A second poll can still return
                    # None. Accept that observation only after this exact
                    # Popen handle reaches a terminal status; a changed,
                    # nonempty command or start time never gets this grace.
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        process.wait(timeout=min(1., remaining))
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started >= timeout:
                            timed_out = True
                            break
                        raise CampaignIntegrityError("child exit observation did not become terminal")
                    guard_runtime()
                    break
        except BaseException as error:
            failure = error
        finally:
            if process is not None and process.poll() is None:
                # Never use a reused PID or process group. If obtaining a
                # kernel handle failed, leave the child holding its inherited
                # leases and stop; guessing a signal target is unsafe.
                if handle is None:
                    failure = CampaignIntegrityError("worker has no kernel handle; inherited orphan leases retained")
                else:
                    sending(handle, signal.SIGTERM)
                    try:
                        process.wait(timeout=10.)
                    except subprocess.TimeoutExpired:
                        sending(handle, signal.SIGKILL)
                        process.wait(timeout=10.)
            if handle is not None:
                os.close(handle)
    result = {"format": "transformer_rl.exposure_worker_completion", "schema_version": 1,
              "process": identity, "command": command,
              "runtime_profile": runtime_profile,
              "returncode": process.returncode if process else None,
              "timed_out": timed_out, "elapsed_s": time.monotonic() - started}
    _new(directory / "worker.completion.json", result)
    if failure is not None:
        raise failure
    return result


def reservation_for(protocol, protocol_receipt, job):
    return {"format": "transformer_rl.exposure_whole_job_reservation", "schema_version": 1,
            "protocol_sha256": protocol["sha256"], "protocol_raw_receipt": protocol_receipt,
            "job_id": job["id"], "training_seed": job["training_seed"],
            "reserved_updates": job["reserved_updates"],
            "reserved_fresh_transitions": job["reserved_fresh_transitions"],
            "stages": [{key: stage[key] for key in ("name", "index", "updates", "fresh_transitions")}
                       for stage in job["stages"]],
            "charge_scope": "whole_job_once_before_first_child",
            "refund": False, "automatic_retries": 0}


def build_stage_request(protocol, protocol_receipt, job, stage, directory, leases,
                        controller_receipt, reservation_receipt, parent_endpoint, storage_contract_receipt,
                        remaining_bytes):
    value = {"format": "transformer_rl.exposure_worker_request", "schema_version": 1,
             "protocol": protocol_receipt, "job_id": job["id"], "stage_index": stage["index"],
             "output_root": str(Path(directory) / "train"), "leases": deepcopy(leases),
             "controller": controller_receipt, "whole_job_reservation": reservation_receipt,
             "parent_endpoint": deepcopy(parent_endpoint), "storage_contract": storage_contract_receipt,
             "required_remaining_bytes": remaining_bytes, "source": protocol["source"]}
    return {**value, "sha256": digest(value)}


def validate_stage_request(request, request_receipt):
    """CPU authorization and complete parent proof before factory import."""
    path = _checked(request_receipt)
    _require(_read(path) == request, "worker request bytes differ from controller publication")
    fields = {"format", "schema_version", "protocol", "job_id", "stage_index", "output_root", "leases",
              "controller", "whole_job_reservation", "parent_endpoint", "storage_contract",
              "required_remaining_bytes", "source", "sha256"}
    _require(type(request) is dict and set(request) == fields
             and request["format"] == "transformer_rl.exposure_worker_request"
             and request["schema_version"] == 1
             and digest({k: v for k, v in request.items() if k != "sha256"}) == request["sha256"],
             "worker request schema or digest differs")
    protocol_path = _checked(request["protocol"])
    protocol, raw = read_authorized_protocol(protocol_path, request["protocol"]["sha256"])
    _require(raw == request["protocol"] and request["source"] == protocol["source"] == source_identity(),
             "worker protocol or source differs")
    validate_controller_lease(protocol, request["leases"], request["controller"])
    from .queue_validation import check_dependency
    states = [check_dependency(item, require_complete=True)
              for item in protocol["execution"]["dependencies"]]
    _require(all(item["status"] == "completed" and item.get("controller_live") is False
                 and item.get("live_workers") == [] for item in states),
             "original controllers, workers or queue closure are still pending")
    job = next((job for job in protocol["jobs"] if job["id"] == request["job_id"]), None)
    _require(job is not None and type(request["stage_index"]) is int
             and 0 <= request["stage_index"] < len(job["stages"]), "worker job/stage membership differs")
    stage = job["stages"][request["stage_index"]]
    directory = Path(protocol["output_root"]) / job["id"] / f"stage_{stage['index']:04d}"
    _require(path == directory / "request.json" and request["output_root"] == str(directory / "train")
             and not os.path.lexists(request["output_root"]), "worker output identity differs or is reused")
    reservation_path = _checked(request["whole_job_reservation"])
    _require(reservation_path == directory.parent / "reservation.json"
             and _read(reservation_path) == reservation_for(protocol, raw, job),
             "whole-job reservation differs")
    contract_path = _checked(request["storage_contract"])
    contract = _validate_storage(_read(contract_path), protocol, raw)
    controller = _read(_checked(request["controller"]))
    _require(controller["storage_contract_receipt"] == request["storage_contract"],
             "worker storage contract differs from controller authorization")
    _require(type(request["required_remaining_bytes"]) is int and request["required_remaining_bytes"] > 0,
             "remaining disk reserve required")
    check_disk(directory, request["required_remaining_bytes"])
    parent = request["parent_endpoint"]
    if stage["index"] == 0:
        _require(parent is None, "first stage must start fresh")
    else:
        endpoint_path = _checked(parent)
        previous = job["stages"][stage["index"] - 1]
        _require(endpoint_path == directory.parent / f"stage_{previous['index']:04d}" / "train"
                 / f"stage_{previous['index']:04d}_{previous['name']}" / "endpoint.json",
                 "parent is not the immediately preceding stage")
        prove_stage_endpoint(protocol, job, previous, endpoint_path)
    _checked(request_receipt)
    return protocol, job, stage, contract


def prove_stage_endpoint(protocol, job, stage, endpoint_path):
    """Use actual model/Adam/RNG/metric proof, including the complete chain."""
    receipt = definition._receipt(endpoint_path)
    config = FrameTrainConfig.from_dict(stage["config"])
    plan, _ = training._definition([{key: stage[key] for key in ("name", "config", "updates")}],
        job_id=job["id"], rollout_steps=protocol["execution"]["rollout_steps"],
        training_seed=job["training_seed"], retention_seed=job["retention_seed"],
        evaluation_seeds=[*protocol["evaluation"]["validation_seeds"], *protocol["evaluation"]["seeds"]],
        device=protocol["execution"]["device"], expected_initial_model_sha256=job["initial_model_sha256"],
        max_seconds=protocol["execution"]["max_seconds"], environment_reference=protocol["environment_factory"])
    proof = training._segment_parent(receipt, plan, config)
    endpoint = proof["endpoint"]
    _require(proof["config"].to_dict() == stage["config"],
             "actual checkpoint environment differs from the authorized stage")
    _require(endpoint["stage_index"] == stage["index"] and endpoint["stage"] == stage["name"]
             and endpoint["stage_updates"] == stage["updates"]
             and endpoint["stage_fresh_transitions"] == stage["fresh_transitions"]
             and endpoint["cumulative_successful_updates"] == stage["expected_cumulative_updates"]
             and endpoint["cumulative_attempted_updates"] == stage["expected_cumulative_updates"]
             and endpoint["cumulative_collected_transitions"] == stage["expected_cumulative_transitions"],
             "actual endpoint differs from the full fixed stage budget")
    _checked(receipt)
    return {"endpoint": receipt, "checkpoint": endpoint["checkpoint"],
            "completion": proof["completion"], "metrics": proof["metrics"],
            "actual_updates": stage["updates"],
            "actual_fresh_transitions": stage["fresh_transitions"]}


def verify_segment_endpoint(protocol, job, stage_index, endpoint_receipt):
    """Return the actual endpoint only after its complete learning proof."""
    _require(job in protocol["jobs"], "endpoint job is outside the fixed definition")
    _require(type(stage_index) is int and 0 <= stage_index < len(job["stages"]),
             "invalid endpoint stage index")
    path = _checked(endpoint_receipt)
    stage = job["stages"][stage_index]
    expected = Path(protocol["output_root"]) / job["id"] / f"stage_{stage_index:04d}" / "train" \
        / f"stage_{stage_index:04d}_{stage['name']}" / "endpoint.json"
    _require(path == expected, "endpoint is outside the authorized job stage")
    prove_stage_endpoint(protocol, job, stage, path)
    return _read(path)


def _provider(value):
    if value is None:
        return None
    _require(value == "transformer_rl.exposure_evaluation", "only the source-pinned physical evaluator is supported")
    from . import exposure_evaluation
    _require(callable(exposure_evaluation.plan_request) and callable(exposure_evaluation.verify_result),
             "physical evaluator interface is unavailable")
    return exposure_evaluation


def run_protocol(protocol_path, *, expected_protocol_sha256, storage_contract_path,
                 expected_storage_sha256, evaluation_provider=None, confirm_heldout=True):
    """Execute the full authorized training grid once, without score decisions."""
    protocol, raw = read_authorized_protocol(protocol_path, expected_protocol_sha256)
    storage_receipt = definition._receipt(storage_contract_path)
    _require(storage_receipt["sha256"] == expected_storage_sha256, "storage bytes differ from external authorization")
    contract = _validate_storage(_read(storage_contract_path), protocol, raw)
    root = definition._path(protocol["output_root"])
    protected = [*map(Path, protocol["protected_roots"]), Path(raw["path"]), Path(storage_contract_path),
                 *[Path(m["receipt"]["path"]) for m in contract["measurements"]]]
    protected.extend(Path(_read(m["receipt"]["path"])["root"]) for m in contract["measurements"]
                     if m["kind"] == "runtime_cache")
    _require(not os.path.lexists(root) and root.parent.is_dir(), "campaign output exists; no retry or reuse")
    _require(not any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected),
             "campaign output overlaps protected definitions or measurements")
    provider = _provider(evaluation_provider)
    _require(type(confirm_heldout) is bool, "held-out confirmation switch must be boolean")
    check_disk(root.parent, remaining_storage_bytes(protocol, contract))
    root.mkdir(mode=0o700)
    controller = {"format": "transformer_rl.exposure_controller", "schema_version": 1,
        "protocol_sha256": protocol["sha256"], "protocol_raw_receipt": raw,
        "expected_protocol_sha256": expected_protocol_sha256,
        "storage_contract_receipt": storage_receipt, "source": protocol["source"],
        "runtime": protocol["runtime"], "process": _identity(os.getpid()),
        "evaluation_provider": evaluation_provider}
    _new(root / "controller.json", controller)
    controller_receipt = definition._receipt(root / "controller.json")
    summary = {"format": "transformer_rl.exposure_campaign", "schema_version": 1,
        "protocol": raw, "status": "waiting", "budget": protocol["budget"], "jobs": {},
        "evaluation_cells": {c["id"]: {"identity": deepcopy(c), "status": "missing",
            "reason": "not_executed"} for c in protocol["evaluation_cells"]},
        "charged_updates": 0, "charged_fresh_transitions": 0,
        "verified_successful_updates": 0, "verified_fresh_transitions": 0,
        "automatic_retries": 0, "refund": False, "independent_evaluation_performed": False,
        "full_evaluation_matrix_closed": False, "formal_architecture_selection": False,
        "hardware_verified": False}
    completed_stages, closed_cells = set(), set()

    def publish(status, **fields):
        summary.update(status=status, **fields, updated_at=datetime.now(timezone.utc).isoformat())
        _publish(root / "summary.json", summary)

    def audit_inputs():
        actual, pin = read_authorized_protocol(protocol_path, expected_protocol_sha256)
        _require(actual == protocol and pin == raw, "campaign inputs changed")
        _checked(storage_receipt)
        _validate_storage(_read(storage_contract_path), protocol, raw)
        _checked(controller_receipt)
        return check_disk(root, remaining_storage_bytes(protocol, contract, completed_stages, closed_cells))

    def evaluate_batch(job, stage, endpoint, cells, leases, *, selection_receipt=None):
        audit_inputs()
        first = cells[0]
        evaluation_root = root / job["id"] / f"stage_{stage['index']:04d}" / f"evaluation_{first['role']}_{first['seed']}"
        progress = {"completed_stages": [list(item) for item in sorted(completed_stages)],
                    "closed_cells": sorted(closed_cells)}
        planned = provider.plan_request(protocol, endpoint, cells, evaluation_root, leases, controller_receipt,
            selection_receipt=selection_receipt, storage_progress=progress)
        _checked(planned["request"])
        request = _read(planned["request"]["path"])

        def monitor():
            _checked(planned["request"])
            _checked(raw)
            _checked(storage_receipt)
            _checked(controller_receipt)
            _require(source_identity() == protocol["source"], "physical worker inputs changed")
            return worker_storage_guard(protocol, contract, evaluation_root, request["required_remaining_bytes"],
                trace_prefix=evaluation_root / "trace.npz", trace_sample_limit=sum(c["expected_policy_samples"] for c in cells),
                active_worker=True)

        publish("evaluating", active_job=job["id"], active_stage=stage["index"],
                active_evaluation_role=first["role"], active_evaluation_seed=first["seed"])
        worker = launch_owned_worker(planned["command"], evaluation_root, leases,
            protocol["execution"]["worker_timeout_seconds"], publish, monitor=monitor)
        if worker["returncode"] != 0 or worker["timed_out"]:
            raise CampaignWorkerError("physical evaluation worker did not close normally")
        evaluated = provider.verify_result(protocol, endpoint, cells, evaluation_root, planned["request"])
        _require(evaluated["status"] == "completed" and set(evaluated["cells"]) == {c["id"] for c in cells},
                 "physical evaluator omitted the declared denominator")
        monitor()
        for cell in cells:
            summary["evaluation_cells"][cell["id"]].update(evaluated["cells"][cell["id"]])
            closed_cells.add(cell["id"])
        _new(evaluation_root / "receipt.json", evaluated)
        summary["independent_evaluation_performed"] = True
        audit_inputs()

    publish("waiting")
    try:
        with resource_lease(protocol, publish) as leases:
            for job in protocol["jobs"]:
                disk = audit_inputs()
                job_root = root / job["id"]
                job_root.mkdir(mode=0o700)
                reservation = reservation_for(protocol, raw, job)
                _new(job_root / "reservation.json", reservation)
                reservation_receipt = definition._receipt(job_root / "reservation.json")
                summary["charged_updates"] += job["reserved_updates"]
                summary["charged_fresh_transitions"] += job["reserved_fresh_transitions"]
                result = {"status": "reserved", "reservation": reservation_receipt, "stages": []}
                summary["jobs"][job["id"]] = result
                publish("training", active_job=job["id"], disk=disk)
                parent = None
                for stage in job["stages"]:
                    audit_inputs()
                    directory = job_root / f"stage_{stage['index']:04d}"
                    directory.mkdir(mode=0o700)
                    remaining = remaining_storage_bytes(protocol, contract, completed_stages, closed_cells)
                    request = build_stage_request(protocol, raw, job, stage, directory, leases,
                        controller_receipt, reservation_receipt, parent, storage_receipt, remaining)
                    _new(directory / "request.json", request)
                    request_receipt = definition._receipt(directory / "request.json")
                    publish("training", active_stage=stage["index"])
                    command = [sys.executable, "-B", "-m", "transformer_rl.exposure_process",
                               "--request", request_receipt["path"],
                               "--expected-request-sha256", request_receipt["sha256"]]
                    checkpoint_path = directory / "train" / f"stage_{stage['index']:04d}_{stage['name']}" / "endpoint.pt"

                    def monitor_training():
                        _checked(request_receipt)
                        _checked(raw)
                        _checked(storage_receipt)
                        _require(source_identity() == protocol["source"], "training worker source changed")
                        return worker_storage_guard(protocol, contract, directory, remaining,
                            checkpoint_paths=[checkpoint_path], checkpoint_limit=contract["caps"]["checkpoint_bytes"],
                            metric_paths=[directory / "train" / "metrics.jsonl"],
                            metric_limit=stage["updates"] * contract["caps"]["metric_bytes_per_update"],
                            active_worker=True)

                    worker = launch_owned_worker(command, directory, leases,
                        protocol["execution"]["worker_timeout_seconds"], publish, monitor=monitor_training)
                    outcome_path = directory / "outcome.json"
                    _require(outcome_path.is_file(), "worker outcome is absent")
                    outcome = _read(outcome_path)
                    _require(outcome.get("request") == request_receipt, "worker outcome request differs")
                    _require(outcome.get("source") == protocol["source"], "worker outcome source differs")
                    _require(worker["process"] == outcome.get("process"), "worker outcome process identity differs")
                    _require(worker["runtime_profile"] == outcome.get("runtime_profile"),
                             "worker outcome runtime profile differs")
                    runtime_paths.validate_runtime_artifact(directory, worker["runtime_profile"])
                    item = {"stage_index": stage["index"], "worker": worker,
                            "outcome": definition._receipt(outcome_path)}
                    result["stages"].append(item)
                    if outcome["status"] == "numerical_failure" and worker["returncode"] == 20:
                        _require(not worker["timed_out"] and outcome.get("typed_numerical_origin")
                                 in ("environment_step", "optimizer_update")
                                 and outcome.get("shutdown_errors") == [], "unverified numerical failure")
                        failure_completion = _read(_checked(outcome["training_completion"]))
                        _require(failure_completion.get("status") == "failed"
                                 and failure_completion.get("error", {}).get("type") == "FloatingPointError"
                                 and failure_completion.get("error", {}).get("phase") == "collect_optimize"
                                 and failure_completion.get("source") == protocol["source"]
                                 and failure_completion.get("job_id") == job["id"]
                                 and failure_completion.get("shutdown_errors") == [],
                                 "numerical outcome lacks its actual failed training record")
                        for key, upper in (("successful_updates", stage["updates"]),
                                           ("attempted_updates", stage["updates"]),
                                           ("actual_collected_transitions", stage["fresh_transitions"])):
                            _require(type(failure_completion.get(key)) is int
                                     and 0 <= failure_completion[key] <= upper,
                                     "numerical failure actual counters exceed fixed reservation")
                        item["failed_training"] = {"completion": outcome["training_completion"],
                            "successful_updates": failure_completion["successful_updates"],
                            "attempted_updates": failure_completion["attempted_updates"],
                            "actual_collected_transitions": failure_completion["actual_collected_transitions"],
                            "optimizer_steps_of_failed_update": failure_completion["failed_update_optimizer_steps"],
                            "accounting_scope": "actual counters; no sealed endpoint for this stage"}
                        result["status"] = "numerical_failure"
                        for cell in protocol["evaluation_cells"]:
                            if cell["job_id"] == job["id"] and cell["stage_index"] >= stage["index"]:
                                summary["evaluation_cells"][cell["id"]].update(reason="training_numerical_failure")
                                closed_cells.add(cell["id"])
                        # The whole-job reservation remains charged, including
                        # its unexecuted stages. No replacement seed or retry.
                        completed_stages.update((job["id"], s["index"]) for s in job["stages"]
                                                if s["index"] >= stage["index"])
                        break
                    if worker["returncode"] != 0 or worker["timed_out"] or outcome["status"] != "completed":
                        raise CampaignWorkerError("worker failed, interrupted or unknown; campaign stopped without retry")
                    endpoint_path = directory / "train" / f"stage_{stage['index']:04d}_{stage['name']}" / "endpoint.json"
                    proof = prove_stage_endpoint(protocol, job, stage, endpoint_path)
                    _require(outcome.get("training_completion") == proof["completion"],
                             "worker completed a different training publication")
                    item["training"] = proof
                    parent = proof["endpoint"]
                    completed_stages.add((job["id"], stage["index"]))
                    summary["verified_successful_updates"] += proof["actual_updates"]
                    summary["verified_fresh_transitions"] += proof["actual_fresh_transitions"]
                    _require(proof["checkpoint"]["bytes"] <= contract["caps"]["checkpoint_bytes"],
                             "actual checkpoint exceeds measured storage cap")
                    _require(proof["metrics"]["bytes"] <= stage["updates"] * contract["caps"]["metric_bytes_per_update"],
                             "actual optimizer log exceeds measured storage cap")
                    if provider is not None:
                        # Complete validation before fixing every endpoint.
                        for seed in protocol["evaluation"]["validation_seeds"]:
                            cells = [c for c in protocol["evaluation_cells"] if c["job_id"] == job["id"]
                                and c["stage_index"] == stage["index"] and c["role"] == "validation" and c["seed"] == seed]
                            evaluate_batch(job, stage, parent, cells, leases)
                    publish("training", active_evaluation_role=None, active_evaluation_seed=None)
                if result["status"] != "numerical_failure":
                    result["status"] = "training_completed"
                _new(job_root / "receipt.json", result)
                publish("ready_for_next_job", active_job=None, active_stage=None)
            audit_inputs()
            if provider is not None and confirm_heldout:
                from .exposure_selection import freeze_selection, verify_selection
                publish("sealing_validation_choice", active_job=None, active_stage=None)
                frozen = freeze_selection(protocol_path, expected_protocol_sha256=expected_protocol_sha256)
                summary["validation_choice"] = frozen["receipt"]
                choice = verify_selection(frozen["receipt"])
                for job in protocol["jobs"]:
                    for stage in job["stages"]:
                        endpoint = next((entry["training"]["endpoint"] for entry in summary["jobs"][job["id"]]["stages"]
                                         if entry["stage_index"] == stage["index"] and "training" in entry), None)
                        if endpoint is None:
                            continue
                        for seed in protocol["evaluation"]["seeds"]:
                            cells = [c for c in protocol["evaluation_cells"] if c["job_id"] == job["id"]
                                and c["stage_index"] == stage["index"] and c["role"] == "held_out" and c["seed"] == seed]
                            evaluate_batch(job, stage, endpoint, cells, leases, selection_receipt=frozen["receipt"])
                _require(verify_selection(frozen["receipt"]) == choice, "held-out changed the immutable validation choice")
        _require(len(summary["jobs"]) == len(protocol["jobs"])
                 and summary["charged_updates"] == protocol["budget"]["training_updates"]
                 and summary["charged_fresh_transitions"] == protocol["budget"]["fresh_transitions"],
                 "full training grid reservations differ")
        closed = len(closed_cells) == len(protocol["evaluation_cells"])
        summary["full_evaluation_matrix_closed"] = provider is not None and closed
        validation_ids = {c["id"] for c in protocol["evaluation_cells"] if c["role"] == "validation"}
        summary["validation_matrix_closed"] = provider is not None and validation_ids <= closed_cells
        status = ("training_completed_evaluation_pending" if provider is None else
                  "comparison_closed_deployment_qualification_pending" if confirm_heldout else
                  "validation_closed_heldout_selection_pending")
        publish(status, active_job=None, active_stage=None)
    except BaseException as error:
        publish("stopped", error={"type": type(error).__name__, "message": str(error)},
                no_automatic_retry=True)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--storage-contract", required=True)
    parser.add_argument("--expected-storage-sha256", required=True)
    parser.add_argument("--evaluation-provider", choices=["transformer_rl.exposure_evaluation"])
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    run_protocol(args.protocol, expected_protocol_sha256=args.expected_protocol_sha256,
        storage_contract_path=args.storage_contract, expected_storage_sha256=args.expected_storage_sha256,
        evaluation_provider=args.evaluation_provider, confirm_heldout=not args.validation_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

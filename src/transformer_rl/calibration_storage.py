"""Sample one owned calibration namespace and observe its actual processes.

Observed maxima are sampled filesystem facts, not continuous-time peaks,
SDK measurements by themselves, a system-wide quota, or hardware validation.
No checkpoint/trace contents are read and no process is signalled.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time


_CATEGORIES = ("runtime", "checkpoint", "trace", "metric", "other")
_JOURNAL = "observer.samples.jsonl"


class CalibrationStorageError(ValueError):
    """Owned storage, its identity, or an observation ceiling was violated."""


def _require(condition, message):
    if not condition:
        raise CalibrationStorageError(message)


def _identity(value):
    return {"device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
            "gid": value.st_gid, "mode": value.st_mode}


def _root(value):
    _require(isinstance(value, (str, Path)) and bool(str(value)), "explicit owned root required")
    path = Path(value)
    _require(path.is_absolute() and path == path.resolve(strict=True)
             and not any(p.is_symlink() for p in (path, *path.parents)),
             "root must be absolute, canonical and contain no symlink")
    observed = path.lstat()
    _require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid(),
             "root must be an owned directory")
    return path, _identity(observed)


def _category(relative):
    parts, name = relative.parts, relative.name
    # Calibration workers live below job/train or job/evaluate namespaces.
    # Their runtime cache contents remain cache bytes regardless of filenames.
    if any(part in ("runtime", "empty_python_cache") for part in parts):
        return "runtime"
    if name.endswith(".pt") or re.fullmatch(r"\..+\.pt\.[A-Za-z0-9_-]+\.tmp", name):
        return "checkpoint"
    if (name.endswith(".npz") or re.fullmatch(r"\..+\.npz\.[A-Za-z0-9_-]+\.tmp", name)
            or any(p.startswith(".control-trace-") or p == ".maps" or p.endswith(".maps")
                   for p in parts)):
        return "trace"
    return "metric" if name == "metrics.jsonl" else "other"


def _scan(root, expected):
    """Use no-follow directory descriptors, permitting vanished temporary files."""
    actual, pin = _root(root)
    _require(pin == expected, "owned root identity changed")
    categories = {name: {"logical_bytes": 0, "allocated_bytes": 0, "file_count": 0,
                         "directory_count": 0} for name in _CATEGORIES}
    seen = {}
    vanished = 0

    def add(relative, value, directory=False):
        category = _category(relative)
        key = value.st_dev, value.st_ino
        _require(key not in seen or seen[key] == category,
                 "hardlink crosses storage categories")
        if key not in seen:
            seen[key] = category
            item = categories[category]
            item["logical_bytes"] += value.st_size
            item["allocated_bytes"] += value.st_blocks * 512
            item["directory_count" if directory else "file_count"] += 1

    def walk(descriptor, path, relative):
        nonlocal vanished
        before = os.fstat(descriptor)
        _require(stat.S_ISDIR(before.st_mode) and before.st_uid == expected["uid"]
                 and before.st_dev == expected["device"],
                 "directory must be owned on the root filesystem")
        add(relative, before, True)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                child = relative / entry.name
                try:
                    value = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    vanished += 1
                    continue
                _require(value.st_uid == expected["uid"] and value.st_dev == expected["device"],
                         "entry has a foreign owner or crosses the root filesystem")
                if stat.S_ISDIR(value.st_mode):
                    try:
                        child_fd = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                           dir_fd=descriptor)
                    except FileNotFoundError:
                        vanished += 1
                        continue
                    try:
                        _require(_identity(os.fstat(child_fd)) == _identity(value),
                                 "directory changed during observation")
                        walk(child_fd, path / entry.name, child)
                    finally:
                        os.close(child_fd)
                else:
                    _require(stat.S_ISREG(value.st_mode), "entry is a symlink or special file")
                    add(child, value)
        try:
            current = path.lstat()
        except FileNotFoundError:
            # A publication can remove its private maps while they are sampled.
            vanished += 1
        else:
            _require(_identity(current) == _identity(before),
                     "directory was replaced during observation")

    descriptor = os.open(actual, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require(_identity(os.fstat(descriptor)) == expected, "root changed during observation")
        walk(descriptor, actual, Path("."))
    finally:
        os.close(descriptor)
    _require(_root(root)[1] == expected, "owned root identity changed during observation")
    fs = os.statvfs(root)
    return {"categories": categories,
            "logical_bytes": sum(c["logical_bytes"] for c in categories.values()),
            "allocated_bytes": sum(c["allocated_bytes"] for c in categories.values()),
            "available_bytes": fs.f_bavail * fs.f_frsize, "filesystem_block_bytes": fs.f_frsize,
            "vanished_entries": vanished}


def guard_owned_storage(root, max_owned_bytes, free_margin_bytes, expected_root_identity=None):
    """Read-only worker guard; share the parent's ceiling, never its journal.

    Passing the parent's fixed root identity detects replacement across calls.
    A caller that omits it receives only the identity of this individual scan.
    """
    path, current = _root(root)
    _require(type(max_owned_bytes) is int and max_owned_bytes > 0,
             "positive integer owned-byte ceiling required")
    _require(type(free_margin_bytes) is int and free_margin_bytes >= 0,
             "nonnegative integer free margin required")
    if expected_root_identity is not None:
        _require(type(expected_root_identity) is dict and set(expected_root_identity) == set(current)
                 and all(type(value) is int for value in expected_root_identity.values()),
                 "fixed owned root identity required")
        _require(current == expected_root_identity, "owned root identity changed")
    observed = _scan(path, current if expected_root_identity is None else expected_root_identity)
    _require(observed["logical_bytes"] <= max_owned_bytes and observed["allocated_bytes"] <= max_owned_bytes,
             "owned logical or allocated bytes exceed the caller ceiling")
    _require(observed["available_bytes"] >= free_margin_bytes,
             "filesystem free bytes are below the caller margin")
    return {"root": str(path), "root_identity": current, **observed,
            "scope": "read-only sampled owned regular files and directories",
            "continuous_peak_verified": False, "all_system_cap_verified": False,
            "hardware_verified": False}


class CalibrationObserver:
    """An explicit caller ceiling and sampled peaks for one exclusive worker.

    The journal records scans before its current line is appended. Guard checks
    and reported maxima additionally include that line's actual logical/block
    growth. Directories and the observer's journal belong to the owned budget.
    """

    def __init__(self, root, max_owned_bytes, free_margin_bytes, interval_s=0.1):
        self.root, self.root_identity = _root(root)
        _require(type(max_owned_bytes) is int and max_owned_bytes > 0,
                 "positive integer owned-byte ceiling required")
        _require(type(free_margin_bytes) is int and free_margin_bytes >= 0,
                 "nonnegative integer free margin required")
        _require(type(interval_s) in (int, float) and math.isfinite(interval_s) and interval_s > 0,
                 "positive finite sampling interval required")
        self.max_owned_bytes, self.free_margin_bytes = max_owned_bytes, free_margin_bytes
        self.interval_s = float(interval_s)
        self.path = self.root / _JOURNAL
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        journal_fd = None
        try:
            _require(_identity(os.fstat(root_fd)) == self.root_identity,
                     "root changed before journal creation")
            journal_fd = os.open(_JOURNAL, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=root_fd)
            _require(_root(self.root)[1] == self.root_identity,
                     "root changed during journal creation")
            self._fd = journal_fd
            self._journal_identity = _identity(os.fstat(self._fd))
        except BaseException:
            if journal_fd is not None:
                os.close(journal_fd)
            raise
        finally:
            os.close(root_fd)
        self._lock = threading.RLock()
        self._stop_event, self._thread = threading.Event(), None
        self._started, self._closed, self._error = False, False, None
        self._created = time.monotonic()
        self._last = self._first = None
        self._count, self._max_gap = 0, 0.0
        self._peaks = {"logical_bytes": 0, "allocated_bytes": 0,
                       "categories": {name: {"logical_bytes": 0, "allocated_bytes": 0}
                                      for name in _CATEGORIES}}
        self._minimum_available = None
        self._latest = None

    def _raise_error(self):
        if self._error is not None:
            raise CalibrationStorageError(
                f"calibration observer failed: {type(self._error).__name__}: {self._error}") from self._error

    def _journal_stat(self):
        value = os.fstat(self._fd)
        _require(_identity(value) == self._journal_identity and value.st_nlink == 1
                 and _identity(self.path.lstat()) == self._journal_identity,
                 "observer journal identity changed or was hardlinked")
        return value

    def sample(self):
        with self._lock:
            self._raise_error()
            _require(not self._closed, "observer is closed")
            try:
                started = time.monotonic()
                observed = _scan(self.root, self.root_identity)
                before = self._journal_stat()
                journal = {"sample": self._count + 1, "monotonic_s": started,
                           "utc": datetime.now(timezone.utc).isoformat(),
                           "phase": "scan_before_current_journal_append", **observed}
                raw = (json.dumps(journal, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode()
                offset = 0
                while offset < len(raw):
                    written = os.write(self._fd, raw[offset:])
                    _require(written > 0, "observer journal write made no progress")
                    offset += written
                after = self._journal_stat()
                logical_delta = after.st_size - before.st_size
                allocated_delta = (after.st_blocks - before.st_blocks) * 512
                _require(logical_delta == len(raw) and allocated_delta >= 0,
                         "observer journal changed outside its writer")
                observed["categories"]["other"]["logical_bytes"] += logical_delta
                observed["categories"]["other"]["allocated_bytes"] += allocated_delta
                observed["logical_bytes"] += logical_delta
                observed["allocated_bytes"] += allocated_delta
                fs = os.statvfs(self.root)
                observed["available_bytes"] = fs.f_bavail * fs.f_frsize
                observed.update(sample=self._count + 1, monotonic_s=started,
                                scan_elapsed_s=time.monotonic() - started,
                                journal_append_bytes=logical_delta)
                self._count += 1
                self._first = started if self._first is None else self._first
                if self._last is not None:
                    self._max_gap = max(self._max_gap, started - self._last)
                self._last, self._latest = started, observed
                self._minimum_available = (observed["available_bytes"] if self._minimum_available is None
                                           else min(self._minimum_available, observed["available_bytes"]))
                for name in ("logical_bytes", "allocated_bytes"):
                    self._peaks[name] = max(self._peaks[name], observed[name])
                    for category in _CATEGORIES:
                        self._peaks["categories"][category][name] = max(
                            self._peaks["categories"][category][name], observed["categories"][category][name])
                _require(observed["logical_bytes"] <= self.max_owned_bytes
                         and observed["allocated_bytes"] <= self.max_owned_bytes,
                         "owned logical or allocated bytes exceed the caller ceiling")
                _require(observed["available_bytes"] >= self.free_margin_bytes,
                         "filesystem free bytes are below the caller margin")
                return deepcopy(observed)
            except BaseException as error:
                self._error = error
                raise

    def guard(self):
        """Propagate a sticky background failure and sample before continuing."""
        return self.sample()

    def _run(self):
        while not self._stop_event.wait(self.interval_s):
            try:
                self.sample()
            except BaseException:
                self._stop_event.set()
                return

    def start(self):
        with self._lock:
            _require(not self._started and not self._closed, "observer cannot be started twice")
            self.sample()
            self._started = True
            self._thread = threading.Thread(target=self._run, name="calibration-storage-observer", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            _require(not self._thread.is_alive(), "observer sampling thread did not close")
        with self._lock:
            if not self._closed:
                try:
                    self.sample()
                finally:
                    os.close(self._fd)
                    self._closed = True
            self._raise_error()
            return self.report()

    def report(self):
        with self._lock:
            return {"format": "transformer_rl.calibration_storage_observation", "schema_version": 1,
                    "root": str(self.root), "root_identity": deepcopy(self.root_identity),
                    "journal_path": str(self.path), "sample_count": self._count,
                    "interval_s": self.interval_s, "max_sample_gap_s": self._max_gap,
                    "elapsed_s": time.monotonic() - self._created,
                    "first_sample_monotonic_s": self._first, "last_sample_monotonic_s": self._last,
                    "max_owned_bytes": self.max_owned_bytes, "free_margin_bytes": self.free_margin_bytes,
                    "minimum_available_bytes": self._minimum_available, "peaks": deepcopy(self._peaks),
                    "last_sample": deepcopy(self._latest), "closed": self._closed,
                    "thread_alive": self._thread is not None and self._thread.is_alive(),
                    "error": None if self._error is None else {
                        "type": type(self._error).__name__, "message": str(self._error)},
                    "scope": "sampled owned regular files and directories, including the observer journal",
                    "journal_scope": "scans before current journal append; reported maxima include append growth",
                    "continuous_peak_verified": False, "all_system_cap_verified": False,
                    "hardware_verified": False}


def _process(pid, unreadable):
    directory = Path("/proc") / str(pid)
    try:
        owner = directory.stat().st_uid
        if owner != os.getuid():
            return None, "foreign_uid"
        raw = (directory / "stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()
        status = (directory / "status").read_text().splitlines()
        uids = next(line.split()[1:] for line in status if line.startswith("Uid:"))
        if any(int(uid) != os.getuid() for uid in uids):
            return None, "foreign_uid"
        argv = [arg.decode(errors="replace") for arg in (directory / "cmdline").read_bytes().split(b"\0") if arg]
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if int(fields[19]) != int(after[19]) or directory.stat().st_uid != owner:
            unreadable["identity"] += 1
            return None, "unreadable"
        return {"pid": pid, "start": int(fields[19]), "uid": owner,
                "parent": int(fields[1]), "state": after[0], "argv": argv}, None
    except (FileNotFoundError, ProcessLookupError):
        return None, "gone"
    except (OSError, ValueError, IndexError, StopIteration):
        unreadable["identity"] += 1
        return None, "unreadable"


def _reference(value, root, *, deleted=False, require_exists=True):
    if deleted and value.endswith(" (deleted)"):
        value = value[:-10]
    try:
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(root):
            return None
        if require_exists:
            if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
                return None
        return str(path)
    except (OSError, ValueError):
        return None


def _references(process, root, unreadable):
    directory = Path("/proc") / str(process["pid"])
    result = {"argv": [], "environment": [], "cwd": [], "fds": []}
    for index, arg in enumerate(process["argv"]):
        value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
        path = _reference(value, root)
        if path is not None:
            result["argv"].append({"index": index, "path": path})
    try:
        for value in (directory / "environ").read_bytes().split(b"\0"):
            key, separator, content = value.partition(b"=")
            if separator:
                for item in content.decode(errors="replace").split(":"):
                    path = _reference(item, root)
                    if path is not None:
                        result["environment"].append({"key": key.decode(errors="replace"), "path": path})
    except (FileNotFoundError, ProcessLookupError):
        pass
    except OSError:
        unreadable["environment"] += 1
    try:
        target = os.readlink(directory / "cwd")
        path = _reference(target, root, deleted=True, require_exists=False)
        if path is not None:
            result["cwd"].append({"path": path, "deleted": target.endswith(" (deleted)")})
    except (FileNotFoundError, ProcessLookupError):
        pass
    except OSError:
        unreadable["cwd"] += 1
    try:
        with os.scandir(directory / "fd") as entries:
            for entry in entries:
                try:
                    target = os.readlink(entry.path)
                    path = _reference(target, root, deleted=True, require_exists=False)
                    if path is not None:
                        observed = os.stat(entry.path)
                        result["fds"].append({"descriptor": int(entry.name), "path": path,
                            "deleted": target.endswith(" (deleted)"), "device": observed.st_dev,
                            "inode": observed.st_ino, "uid": observed.st_uid})
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError:
                    unreadable["fd"] += 1
    except (FileNotFoundError, ProcessLookupError):
        pass
    except OSError:
        unreadable["fd_directory"] += 1
    return result


def _observer_ancestry(unreadable):
    """Read only PID/start/UID ancestry, including foreign-user launchers.

    A shell or controller can legitimately carry this namespace in its argv,
    cwd, or environment. It is an observer ancestor, not an SDK helper. Pin the
    start time as well as PID so an exited ancestor's reused PID stays visible.
    Failure to read the chain remains an explicit incomplete observation.
    """
    excluded = {}
    pid = os.getpid()
    while pid > 0:
        if pid in excluded:
            unreadable["identity"] += 1
            return excluded, False, "cycle"
        directory = Path("/proc") / str(pid)
        try:
            owner = directory.stat().st_uid
            before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if (int(before[19]), int(before[1]), owner) != (
                    int(after[19]), int(after[1]), directory.stat().st_uid):
                unreadable["identity"] += 1
                return excluded, False, "identity_changed"
            excluded[pid] = {"pid": pid, "start": int(after[19]), "uid": owner}
            pid = int(after[1])
        except (FileNotFoundError, ProcessLookupError):
            unreadable["identity"] += 1
            return excluded, False, "gone"
        except (OSError, ValueError, IndexError):
            unreadable["identity"] += 1
            return excluded, False, "unreadable"
    return excluded, True, None


def helper_inventory(root, tracked_handles=()):
    """Retain seen PID/start/UID identities after their root references disappear.

    Descendants of directly referencing processes and still-live retained helper
    identities are observed, including forks after a helper releases its paths.
    The caller and its PID/start/UID-pinned ancestor chain are excluded from
    discovery and descendant expansion. Observer journal FDs and launcher shell
    arguments are not helpers. Explicitly tracked identities are still checked.
    Missing visibility remains unknown; absence of a path reference never proves
    a previously seen process exited. Environment output contains only matching
    path keys, never unrelated environment values.
    """
    root, pin = _root(root)
    _require(isinstance(tracked_handles, (list, tuple)), "tracked handles must be a sequence")
    tracked = {}
    for item in tracked_handles:
        _require(type(item) is dict and set(item) == {"pid", "start", "uid"}
                 and all(type(item[k]) is int for k in item) and item["pid"] > 0 and item["start"] >= 0
                 and item["uid"] == os.getuid(), "tracked PID/start/UID handle differs")
        key = item["pid"], item["start"], item["uid"]
        _require(key not in tracked, "duplicate tracked process handle")
        tracked[key] = deepcopy(item)
    unreadable = {name: 0 for name in ("identity", "environment", "cwd", "fd", "fd_directory")}
    ancestry, ancestry_complete, ancestry_stop_reason = _observer_ancestry(unreadable)
    excluded = {(p["pid"], p["start"], p["uid"]) for p in ancestry.values()}
    caller_pid = os.getpid()

    def is_observer(process):
        return (process["pid"] == caller_pid or
                (process["pid"], process["start"], process["uid"]) in excluded)

    processes, unavailable, references = {}, {}, {}
    for entry in os.scandir("/proc"):
        if entry.name.isdecimal():
            pid = int(entry.name)
            process, reason = _process(pid, unreadable)
            if process is None:
                unavailable[pid] = reason
            else:
                processes[pid] = process
                if not is_observer(process):
                    refs = _references(process, root, unreadable)
                    after, reason = _process(pid, unreadable)
                    if after is not None and (after["start"], after["uid"]) == (process["start"], process["uid"]):
                        references[pid] = refs
                    else:
                        unreadable["identity"] += 1
    direct = {pid for pid, refs in references.items() if any(refs.values())}
    retained = set()
    for pid, expected_start, expected_uid in tracked:
        process = processes.get(pid)
        if (process is not None and (process["start"], process["uid"]) == (expected_start, expected_uid)
                and process["state"] not in ("Z", "X", "x") and not is_observer(process)):
            retained.add(pid)
    selected = direct | retained
    while True:
        descendants = {pid for pid, p in processes.items()
                       if p["parent"] in selected and not is_observer(p)}
        if descendants <= selected:
            break
        selected.update(descendants)
    for pid in selected:
        p = processes[pid]
        key = p["pid"], p["start"], p["uid"]
        tracked.setdefault(key, {name: p[name] for name in ("pid", "start", "uid")})
    statuses, visible = [], []
    for key, handle in sorted(tracked.items()):
        pid, expected_start, expected_uid = key
        # Re-read after reference discovery: the earlier snapshot can race exit.
        current, reason = _process(pid, unreadable)
        status = reason
        terminal = reason == "gone"
        if current is not None:
            if (current["start"], current["uid"]) != (expected_start, expected_uid):
                status, terminal = "reused", True
            else:
                terminal = current["state"] in ("Z", "X", "x")
                status = "exited" if terminal else "alive"
                original = processes.get(pid)
                stable = original is not None and original["start"] == expected_start
                visible.append({**current, "references": references.get(pid, {}) if stable else {},
                    "reference_kind": "direct" if stable and pid in direct else
                                      "descendant" if stable and pid in selected and pid not in retained else "tracked",
                    "terminal": terminal})
        statuses.append({"handle": handle, "status": status, "terminal": terminal})
    _require(_root(root)[1] == pin, "owned helper root identity changed during inventory")
    return {"format": "transformer_rl.calibration_helper_inventory", "schema_version": 1,
            "root": str(root), "root_identity": pin, "processes": sorted(visible, key=lambda p: (p["pid"], p["start"])),
            "handles": [tracked[k] for k in sorted(tracked)], "tracked_status": statuses,
            "all_tracked_terminal": all(s["terminal"] for s in statuses),
            "caller_excluded_from_reference_discovery": caller_pid,
            "reference_discovery_exclusions": [ancestry[k] for k in sorted(ancestry)],
            "observer_ancestry_complete": ancestry_complete,
            "observer_ancestry_stop_reason": ancestry_stop_reason,
            "unreadable": unreadable, "unreadable_count": sum(unreadable.values()),
            "scope": "same-UID root references and descendants excluding observer ancestry; retained PID/start/UID handles",
            "complete_system_coverage": False, "hardware_verified": False}

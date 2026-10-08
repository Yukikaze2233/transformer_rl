"""CPU filesystem/process evidence; these tests are not an SDK calibration."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from transformer_rl import calibration_storage as storage


@pytest.fixture
def owned(tmp_path):
    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    return root


@pytest.fixture
def observer(owned):
    value = storage.CalibrationObserver(owned, 64 * 1024**2, 0, interval_s=0.01)
    yield value
    if not value.report()["closed"]:
        try:
            value.stop()
        except storage.CalibrationStorageError:
            pass


def create(root, route, content=b"contents"):
    path = root / route
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("owned test condition did not become true")


def test_categories_deduplicate_same_inode_and_include_observer_journal(observer, owned):
    cache = create(owned, "runtime/cache/blob", b"x" * 2048)
    os.link(cache, cache.parent / "alias")
    create(owned, "empty_python_cache/item.pyc", b"bytecode")
    create(owned, "train/endpoint.pt", b"checkpoint")
    create(owned, "train/.endpoint.pt.random.tmp", b"inflight checkpoint")
    create(owned, "trace.npz", b"trace")
    create(owned, ".control-trace-test/field_0.npy", b"map")
    create(owned, "work.maps/values.npy", b"other map")
    create(owned, "train/metrics.jsonl", b"metric\n")
    create(owned, "train/endpoint.pt.json", b"sidecar")
    sample = observer.sample()
    categories = sample["categories"]
    assert categories["runtime"]["file_count"] == 2
    assert categories["checkpoint"]["file_count"] == 2
    assert categories["trace"]["file_count"] == 3
    assert categories["metric"]["file_count"] == 1
    assert categories["other"]["file_count"] == 2
    assert categories["checkpoint"]["logical_bytes"] == len(b"checkpointinflight checkpoint")
    assert categories["runtime"]["logical_bytes"] >= 2048 + len(b"bytecode")
    assert categories["other"]["logical_bytes"] >= observer.path.stat().st_size + len(b"sidecar")
    assert sample["logical_bytes"] == sum(c["logical_bytes"] for c in categories.values())
    assert sample["allocated_bytes"] == sum(c["allocated_bytes"] for c in categories.values())
    journal = json.loads(observer.path.read_text().splitlines()[0])
    assert sample["logical_bytes"] == journal["logical_bytes"] + sample["journal_append_bytes"]
    assert observer.report()["peaks"]["allocated_bytes"] == sample["allocated_bytes"]
    report = observer.stop()
    assert report["closed"] and not report["thread_alive"]
    assert not report["hardware_verified"] and not report["continuous_peak_verified"]
    assert not report["all_system_cap_verified"]


def test_nested_train_and_evaluation_namespaces_keep_runtime_and_artifact_categories(observer, owned):
    cache = create(owned, "job_a/train/runtime/cache/endpoint.pt", b"cache")
    os.link(cache, cache.parent / "same_inode")
    create(owned, "job_a/train/empty_python_cache/module.pyc", b"bytecode")
    create(owned, "job_a/evaluate/runtime/cache/trace.npz", b"cache trace name")
    create(owned, "job_a/evaluate/empty_python_cache/module.pyc", b"more bytecode")
    create(owned, "job_a/train/train/stage_0000/endpoint.pt", b"checkpoint")
    create(owned, "job_a/train/train/stage_0000/.endpoint.pt.random.tmp", b"staging")
    create(owned, "job_a/train/train/metrics.jsonl", b"metric\n")
    create(owned, "job_a/evaluate/trace.npz", b"trace")
    create(owned, "job_a/evaluate/.control-trace-test/field_0.npy", b"memmap")
    create(owned, "job_a/evaluate/session.maps/field.npy", b"map")
    create(owned, "job_a/evaluate/.trace.npz.random.tmp", b"staged trace")
    create(owned, "job_a/evaluate/report.json", b"report")
    actual = observer.sample()["categories"]
    assert actual["runtime"]["file_count"] == 4
    assert actual["runtime"]["logical_bytes"] >= len(b"cachebytecodecache trace namemore bytecode")
    assert actual["checkpoint"]["file_count"] == 2
    assert actual["checkpoint"]["logical_bytes"] >= len(b"checkpointstaging")
    assert actual["metric"]["file_count"] == 1
    assert actual["trace"]["file_count"] == 4
    assert actual["other"]["file_count"] == 2  # report plus the root observer journal


def test_nested_runtime_cross_category_hardlink_is_rejected(observer, owned):
    cache = create(owned, "job_a/train/runtime/cache/blob", b"same bytes")
    checkpoint = owned / "job_a/train/train/endpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    os.link(cache, checkpoint)
    with pytest.raises(storage.CalibrationStorageError, match="crosses storage categories"):
        observer.guard()


def test_renamed_files_remain_visible_and_peaks_do_not_shrink(observer, owned):
    path = create(owned, "runtime/cache/blob", b"x" * 20000)
    first = observer.sample()
    path.rename(path.with_name("renamed"))
    second = observer.sample()
    assert second["categories"]["runtime"] == first["categories"]["runtime"]
    path.with_name("renamed").unlink()
    last = observer.sample()
    assert last["categories"]["runtime"]["file_count"] == 0
    assert observer.report()["peaks"]["categories"]["runtime"]["logical_bytes"] == first["categories"]["runtime"]["logical_bytes"]


def test_scan_reads_no_checkpoint_or_trace_contents(observer, owned, monkeypatch):
    create(owned, "train/endpoint.pt")
    create(owned, "trace.npz")
    monkeypatch.setattr(Path, "read_bytes", lambda path: (_ for _ in ()).throw(AssertionError("content read")))
    observer.sample()


def test_read_only_worker_guard_uses_fixed_identity_and_creates_no_journal(owned):
    path = create(owned, "runtime/cache/entry")
    before = path.stat()
    initial = storage.guard_owned_storage(owned, 1024**2, 0)
    actual = storage.guard_owned_storage(owned, 1024**2, 0, initial["root_identity"])
    assert actual["logical_bytes"] == initial["logical_bytes"]
    assert actual["allocated_bytes"] == initial["allocated_bytes"]
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not (owned / "observer.samples.jsonl").exists()
    assert not actual["hardware_verified"]
    owned.rename(owned.with_name("old_owned"))
    owned.mkdir()
    with pytest.raises(storage.CalibrationStorageError, match="root identity"):
        storage.guard_owned_storage(owned, 1024**2, 0, initial["root_identity"])


def test_read_only_guard_checks_byte_ceiling_and_identity_schema(owned):
    create(owned, "overshoot", b"x" * 20000)
    with pytest.raises(storage.CalibrationStorageError, match="caller ceiling"):
        storage.guard_owned_storage(owned, 10000, 0)
    with pytest.raises(storage.CalibrationStorageError, match="fixed owned root identity"):
        storage.guard_owned_storage(owned, 1024**2, 0, {"uid": os.getuid()})
    assert not (owned / "observer.samples.jsonl").exists()


def test_journal_creation_does_not_follow_a_replaced_root(owned, monkeypatch):
    original = os.open
    old = owned.with_name("old_owned")

    def replace_root(path, flags, *args, **kwargs):
        if path == "observer.samples.jsonl" and kwargs.get("dir_fd") is not None:
            owned.rename(old)
            owned.mkdir()
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", replace_root)
    with pytest.raises(storage.CalibrationStorageError, match="root changed during"):
        storage.CalibrationObserver(owned, 1024**2, 0)
    assert not (owned / "observer.samples.jsonl").exists()
    assert (old / "observer.samples.jsonl").exists()
    targets = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            targets.append(os.readlink(entry))
        except FileNotFoundError:
            pass
    assert str(old / "observer.samples.jsonl") not in targets


def test_cross_category_hardlink_is_rejected(observer, owned):
    path = create(owned, "runtime/cache/item")
    os.link(path, owned / "trace.npz")
    with pytest.raises(storage.CalibrationStorageError, match="hardlink crosses"):
        observer.sample()


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
def test_symlinks_and_special_entries_are_rejected(observer, owned, kind):
    if kind == "fifo":
        os.mkfifo(owned / "pipe")
    else:
        outside = owned.parent / "outside"
        if kind == "directory_symlink":
            outside.mkdir()
        else:
            outside.write_bytes(b"outside")
        (owned / "escape").symlink_to(outside, target_is_directory=kind == "directory_symlink")
    with pytest.raises(storage.CalibrationStorageError, match="symlink|special"):
        observer.sample()


def test_root_replacement_cannot_keep_its_path_authorization(observer, owned):
    owned.rename(owned.with_name("old_owned"))
    owned.mkdir(mode=0o700)
    with pytest.raises(storage.CalibrationStorageError, match="root identity"):
        observer.guard()


def test_journal_identity_and_single_link_are_fixed(observer, owned):
    os.link(observer.path, owned / "journal-alias")
    with pytest.raises(storage.CalibrationStorageError, match="journal identity"):
        observer.sample()


@pytest.mark.parametrize("field,value", [("st_uid", -1), ("st_dev", -1)])
def test_file_owner_and_filesystem_are_checked(observer, owned, monkeypatch, field, value):
    create(owned, "payload")
    original = os.stat

    def changed(path, *args, **kwargs):
        actual = original(path, *args, **kwargs)
        if path == "payload" and kwargs.get("dir_fd") is not None:
            fields = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
            fields[field] = value
            return SimpleNamespace(**fields)
        return actual

    monkeypatch.setattr(storage.os, "stat", changed)
    with pytest.raises(storage.CalibrationStorageError, match="foreign owner|filesystem"):
        observer.sample()


def test_logical_sparse_overshoot_is_sticky_and_reported(owned):
    observer = storage.CalibrationObserver(owned, 1024**2, 0)
    with (owned / "sparse").open("wb") as stream:
        stream.truncate(2 * 1024**2)
    with pytest.raises(storage.CalibrationStorageError, match="caller ceiling"):
        observer.sample()
    assert observer.report()["peaks"]["logical_bytes"] > 1024**2
    with pytest.raises(storage.CalibrationStorageError, match="observer failed"):
        observer.guard()
    with pytest.raises(storage.CalibrationStorageError):
        observer.stop()
    assert observer.report()["closed"]


def test_allocated_small_file_overshoot_is_independent_of_logical_bytes(owned):
    observer = storage.CalibrationObserver(owned, 1024**2, 0)
    for index in range(300):
        (owned / f"small_{index}").write_bytes(b"x")
    with pytest.raises(storage.CalibrationStorageError, match="caller ceiling"):
        observer.sample()
    report = observer.report()
    assert report["last_sample"]["logical_bytes"] < 1024**2
    assert report["last_sample"]["allocated_bytes"] > 1024**2
    with pytest.raises(storage.CalibrationStorageError):
        observer.stop()


def test_free_margin_is_rechecked_after_own_journal_append(owned, monkeypatch):
    observer = storage.CalibrationObserver(owned, 1024**2, 1)
    original = storage.os.statvfs
    monkeypatch.setattr(storage.os, "statvfs", lambda path: SimpleNamespace(
        f_bavail=0, f_frsize=original(path).f_frsize))
    with pytest.raises(storage.CalibrationStorageError, match="free bytes"):
        observer.sample()
    assert observer.report()["minimum_available_bytes"] == 0
    with pytest.raises(storage.CalibrationStorageError):
        observer.stop()


def test_background_sampling_gap_and_exception_reach_controller(observer, owned):
    observer.start()
    wait_until(lambda: observer.report()["sample_count"] >= 3)
    assert observer.report()["max_sample_gap_s"] > 0
    (owned / "escape").symlink_to(owned.parent)
    wait_until(lambda: observer.report()["error"] is not None)
    with pytest.raises(storage.CalibrationStorageError, match="observer failed"):
        observer.guard()
    with pytest.raises(storage.CalibrationStorageError):
        observer.stop()
    assert observer.report()["closed"] and not observer.report()["thread_alive"]


def test_exclusive_journal_and_closed_lifecycle(observer, owned):
    with pytest.raises(FileExistsError):
        storage.CalibrationObserver(owned, 1024**2, 0)
    observer.start()
    with pytest.raises(storage.CalibrationStorageError, match="twice"):
        observer.start()
    stopped = observer.stop()
    assert stopped["sample_count"] >= 2
    assert observer.stop()["closed"]
    with pytest.raises(storage.CalibrationStorageError, match="closed"):
        observer.sample()


@pytest.mark.parametrize("kwargs", [
    {"max_owned_bytes": 0}, {"max_owned_bytes": True}, {"max_owned_bytes": 1.5},
    {"free_margin_bytes": -1}, {"free_margin_bytes": False},
    {"interval_s": 0}, {"interval_s": float("nan")}, {"interval_s": True},
])
def test_invalid_ceilings_and_intervals_do_not_create_journal(owned, kwargs):
    arguments = {"max_owned_bytes": 1024**2, "free_margin_bytes": 0, "interval_s": 0.1, **kwargs}
    with pytest.raises(storage.CalibrationStorageError):
        storage.CalibrationObserver(owned, **arguments)
    assert not (owned / "observer.samples.jsonl").exists()


def test_relative_symlink_and_non_directory_roots_are_rejected(owned, monkeypatch):
    monkeypatch.chdir(owned.parent)
    link = owned.parent / "link"
    link.symlink_to(owned, target_is_directory=True)
    file = owned.parent / "file"
    file.write_bytes(b"file")
    for path in ("owned", link, file):
        with pytest.raises(storage.CalibrationStorageError):
            storage.CalibrationObserver(path, 1024**2, 0)


@pytest.fixture
def owned_process(owned):
    fd = os.open(create(owned, "payload"), os.O_RDONLY)
    script = """import os,sys
descriptor=int(sys.argv[1])
print('ready',flush=True)
for command in sys.stdin:
    if command.strip()=='release':
        os.chdir('/')
        os.close(descriptor)
        print('released',flush=True)
    elif command.strip()=='exit':
        break
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", script, str(fd)], cwd=owned,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(fd,), text=True)
    os.close(fd)
    assert select.select([process.stdout], [], [], 5)[0]
    assert process.stdout.readline().strip() == "ready"
    yield process
    if process.poll() is None:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # This exact CPU subprocess belongs to this fixture, not the SDK.
            process.kill()
            process.wait(timeout=5)
    for stream in (process.stdout, process.stderr):
        stream.close()


def test_actual_fd_and_cwd_references_survive_release_until_real_exit(owned, owned_process):
    initial = storage.helper_inventory(owned)
    process = next(p for p in initial["processes"] if p["pid"] == owned_process.pid)
    assert process["uid"] == os.getuid() and process["parent"] == os.getpid()
    assert process["references"]["cwd"] and process["references"]["fds"]
    handle = {key: process[key] for key in ("pid", "start", "uid")}
    owned_process.stdin.write("release\n")
    owned_process.stdin.flush()
    assert select.select([owned_process.stdout], [], [], 5)[0]
    assert owned_process.stdout.readline().strip() == "released"
    released = storage.helper_inventory(owned, [handle])
    matched = next(p for p in released["processes"] if p["pid"] == owned_process.pid)
    assert matched["reference_kind"] == "tracked"
    assert not any(matched["references"].values())
    assert released["tracked_status"] == [{"handle": handle, "status": "alive", "terminal": False}]
    assert not released["all_tracked_terminal"]
    owned_process.stdin.write("exit\n")
    owned_process.stdin.flush()
    assert owned_process.wait(timeout=5) == 0
    ended = storage.helper_inventory(owned, [handle])
    assert ended["tracked_status"] == [{"handle": handle, "status": "gone", "terminal": True}]
    assert ended["all_tracked_terminal"] and not ended["hardware_verified"]
    assert not ended["complete_system_coverage"]


def test_observer_own_journal_fd_is_not_an_sdk_helper(observer, owned):
    observer.sample()
    inventory = storage.helper_inventory(owned)
    assert not inventory["handles"] and inventory["all_tracked_terminal"]
    assert inventory["caller_excluded_from_reference_discovery"] == os.getpid()
    assert inventory["observer_ancestry_complete"]
    assert inventory["observer_ancestry_stop_reason"] is None
    assert os.getpid() in {row["pid"] for row in inventory["reference_discovery_exclusions"]}


def test_real_launcher_ancestry_does_not_select_unrelated_sibling(owned):
    payload = create(owned, "request.json")
    probe = """import json,os,sys
from transformer_rl.calibration_storage import helper_inventory
inventory=helper_inventory(sys.argv[1])
ancestor=next(row for row in inventory['reference_discovery_exclusions'] if row['pid']==os.getppid())
tracked=helper_inventory(sys.argv[1],[ancestor])
print(json.dumps({'inventory':inventory,'tracked':tracked,'probe_pid':os.getpid()}),flush=True)
"""
    supervisor = """import json,os,subprocess,sys
children=[]
try:
    waiter='import sys;sys.stdin.read()'
    unrelated=subprocess.Popen([sys.executable,'-B','-c',waiter],cwd='/',env={},stdin=subprocess.PIPE)
    children.append(unrelated)
    helper=subprocess.Popen([sys.executable,'-B','-c',waiter],cwd=sys.argv[1],env={},stdin=subprocess.PIPE)
    children.append(helper)
    probe=subprocess.run([sys.executable,'-B','-c',sys.argv[2],sys.argv[1]],cwd='/',
        env=os.environ.copy(),capture_output=True,text=True,timeout=10,check=True)
    result=json.loads(probe.stdout)
    result.update(supervisor_pid=os.getpid(),unrelated_pid=unrelated.pid,helper_pid=helper.pid)
    print(json.dumps(result),flush=True)
    sys.stdin.read()
finally:
    for child in children:
        child.stdin.close()
    for child in children:
        child.wait(timeout=5)
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(storage.__file__).parents[1]),
                   "PYTHONDONTWRITEBYTECODE": "1", "OWNED_LAUNCHER_TEST_PATH": str(owned)}
    process = subprocess.Popen([sys.executable, "-B", "-c", supervisor, str(owned), probe,
        f"--request={payload}"], cwd=owned, env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 15)[0]
        result = json.loads(process.stdout.readline())
        actual = result["inventory"]
        assert actual["observer_ancestry_complete"]
        excluded = {row["pid"] for row in actual["reference_discovery_exclusions"]}
        assert {result["probe_pid"], result["supervisor_pid"], os.getpid()} <= excluded
        observed = {row["pid"] for row in actual["processes"]}
        assert result["helper_pid"] in observed
        assert not ({result["probe_pid"], result["supervisor_pid"], result["unrelated_pid"], os.getpid()} & observed)
        helper = next(row for row in actual["processes"] if row["pid"] == result["helper_pid"])
        assert helper["reference_kind"] == "direct" and helper["references"]["cwd"]
        tracked = result["tracked"]
        ancestor = next(row for row in tracked["processes"] if row["pid"] == process.pid)
        assert ancestor["reference_kind"] == "tracked" and ancestor["references"] == {}
        assert not ancestor["terminal"] and not tracked["all_tracked_terminal"]
        assert result["unrelated_pid"] not in {row["pid"] for row in tracked["processes"]}
        process.stdin.close()
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.stdin.close()
            process.wait(timeout=8)
        process.stdout.close()
        process.stderr.close()


def test_reused_ancestor_pid_does_not_hide_current_helper(owned, owned_process, monkeypatch):
    initial = storage.helper_inventory(owned)
    current = next(row for row in initial["processes"] if row["pid"] == owned_process.pid)
    original = storage._observer_ancestry

    def stale_ancestor(counts):
        excluded, complete, reason = original(counts)
        excluded[owned_process.pid] = {"pid": owned_process.pid, "start": current["start"] - 1,
                                      "uid": current["uid"]}
        return excluded, complete, reason

    monkeypatch.setattr(storage, "_observer_ancestry", stale_ancestor)
    actual = storage.helper_inventory(owned)
    assert any(row["pid"] == owned_process.pid and row["reference_kind"] == "direct"
               for row in actual["processes"])


def test_unreadable_observer_ancestry_is_reported_as_incomplete(owned, monkeypatch):
    monkeypatch.setattr(storage, "_observer_ancestry", lambda counts: ({}, False, "unreadable"))
    actual = storage.helper_inventory(owned)
    assert not actual["observer_ancestry_complete"]
    assert actual["observer_ancestry_stop_reason"] == "unreadable"
    assert actual["reference_discovery_exclusions"] == []
    assert actual["caller_excluded_from_reference_discovery"] == os.getpid()
    assert not actual["complete_system_coverage"]


def test_actual_argv_and_environment_references_filter_unrelated_environment(owned):
    payload = create(owned, "request.json")
    script = "import sys;print('ready',flush=True);sys.stdin.read()"
    process = subprocess.Popen([sys.executable, "-B", "-c", script, f"--request={payload}"],
        cwd="/", env={"OWNED_PATH_FOR_TEST": str(owned), "PRIVATE_UNRELATED_TEST_VALUE": "never-export-this-value"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "ready"
        actual = storage.helper_inventory(owned)
        row = next(p for p in actual["processes"] if p["pid"] == process.pid)
        assert row["references"]["argv"] == [{"index": 4, "path": str(payload)}]
        assert row["references"]["environment"] == [{"key": "OWNED_PATH_FOR_TEST", "path": str(owned)}]
        assert "never-export-this-value" not in json.dumps(actual)
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        assert storage.helper_inventory(owned, actual["handles"])["all_tracked_terminal"]
    finally:
        if process.poll() is None:
            process.stdin.close()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_actual_start_mismatch_is_a_reused_handle_not_a_live_original(owned, owned_process):
    result = storage.helper_inventory(owned)
    process = next(p for p in result["processes"] if p["pid"] == owned_process.pid)
    handle = {"pid": process["pid"], "start": process["start"] - 1, "uid": process["uid"]}
    actual = storage.helper_inventory(owned, [handle])
    row = next(p for p in actual["tracked_status"] if p["handle"] == handle)
    assert row["status"] == "reused" and row["terminal"]
    assert any(h["start"] == process["start"] for h in actual["handles"])


def test_unreadable_tracked_identity_remains_unknown(owned, owned_process, monkeypatch):
    original = storage._process
    result = storage.helper_inventory(owned)
    process = next(p for p in result["processes"] if p["pid"] == owned_process.pid)
    handle = {key: process[key] for key in ("pid", "start", "uid")}

    def unreadable(pid, counts):
        if pid == owned_process.pid:
            counts["identity"] += 1
            return None, "unreadable"
        return original(pid, counts)

    monkeypatch.setattr(storage, "_process", unreadable)
    inventory = storage.helper_inventory(owned, [handle])
    assert inventory["tracked_status"] == [{"handle": handle, "status": "unreadable", "terminal": False}]
    assert not inventory["all_tracked_terminal"] and inventory["unreadable"]["identity"] > 0


def test_real_descendant_is_tracked_without_direct_root_references(owned):
    script = """import subprocess,sys
child=subprocess.Popen([sys.executable,'-B','-c','import sys;sys.stdin.read()'],
    stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    cwd='/',env={})
print(child.pid,flush=True)
sys.stdin.read()
child.stdin.close()
child.wait(timeout=5)
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", script], cwd=owned,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        child_pid = int(process.stdout.readline())
        inventory = storage.helper_inventory(owned)
        child = next(p for p in inventory["processes"] if p["pid"] == child_pid)
        assert child["parent"] == process.pid and child["reference_kind"] == "descendant"
        assert not any(child["references"].values())
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        ended = storage.helper_inventory(owned, inventory["handles"])
        assert ended["all_tracked_terminal"]
    finally:
        if process.poll() is None:
            process.stdin.close()
            process.wait(timeout=8)
        process.stdout.close()
        process.stderr.close()


def test_retained_helper_discovers_new_fork_after_releasing_root_references(owned):
    descriptor = os.open(create(owned, "payload"), os.O_RDONLY)
    script = """import os,subprocess,sys
descriptor=int(sys.argv[1])
children=[]
try:
    print('ready',flush=True)
    for command in sys.stdin:
        if command.strip()=='release':
            os.chdir('/')
            os.close(descriptor)
            print('released',flush=True)
        elif command.strip()=='fork':
            child=subprocess.Popen([sys.executable,'-B','-c','import sys;sys.stdin.read()'],
                cwd='/',env={},stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children.append(child)
            print(child.pid,flush=True)
        elif command.strip()=='exit':
            break
finally:
    for child in children:
        child.stdin.close()
    for child in children:
        child.wait(timeout=5)
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", script, str(descriptor)], cwd=owned,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=(descriptor,), text=True)
    os.close(descriptor)
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "ready"
        initial = storage.helper_inventory(owned)
        parent = next(row for row in initial["processes"] if row["pid"] == process.pid)
        handle = {key: parent[key] for key in ("pid", "start", "uid")}
        process.stdin.write("release\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline().strip() == "released"
        released = storage.helper_inventory(owned, [handle])
        parent = next(row for row in released["processes"] if row["pid"] == process.pid)
        assert parent["reference_kind"] == "tracked" and not any(parent["references"].values())
        process.stdin.write("fork\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 5)[0]
        child_pid = int(process.stdout.readline())
        unregistered = storage.helper_inventory(owned)
        assert not ({process.pid, child_pid} & {row["pid"] for row in unregistered["processes"]})
        stale_handle = {**handle, "start": handle["start"] - 1}
        stale = storage.helper_inventory(owned, [stale_handle])
        assert stale["tracked_status"] == [{"handle": stale_handle, "status": "reused", "terminal": True}]
        assert child_pid not in {row["pid"] for row in stale["processes"]}
        actual = storage.helper_inventory(owned, [handle])
        child = next(row for row in actual["processes"] if row["pid"] == child_pid)
        assert child["parent"] == process.pid and child["reference_kind"] == "descendant"
        assert not child["terminal"] and not any(child["references"].values())
        assert not actual["all_tracked_terminal"]
        process.stdin.write("exit\n")
        process.stdin.flush()
        assert process.wait(timeout=8) == 0
        closed = storage.helper_inventory(owned, actual["handles"])
        assert closed["all_tracked_terminal"]
    finally:
        if process.poll() is None:
            process.stdin.close()
            process.wait(timeout=8)
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize("handle", [
    {"pid": 0, "start": 1, "uid": os.getuid()},
    {"pid": 1, "start": True, "uid": os.getuid()},
    {"pid": 1, "start": 1, "uid": -1},
    {"pid": 1, "start": 1},
])
def test_invalid_process_handles_fail_before_proc_scan(owned, handle):
    with pytest.raises(storage.CalibrationStorageError, match="handle"):
        storage.helper_inventory(owned, [handle])

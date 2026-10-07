"""Execute the original sealed LR confirmation requests once, without training.

Preparation and auditing are read-only with respect to the original selection,
learning campaign, frozen learner, and confirmation definition. Only ``run``
may create evaluation outputs, and it never changes a rate, seed, or request.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


_SELECTOR = Path(__file__).with_name("select_learning_rate.py")
_SPEC = importlib.util.spec_from_file_location("confirmation_selection", _SELECTOR)
selection = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selection)
campaign, control, diagnostic = selection.campaign, selection.control, selection.diagnostic
FORMAT = "transformer_rl.learning_confirmation_execution"
SCOPE = {"execution_implemented": True, "no_reselection": True,
         "scope": "heldout_noise_stream_only", "new_initial_conditions": False,
         "new_perturbation_domains": False, "training_implemented": False,
         "formal_architecture_selection": False, "hardware_deployment_ready": False}
require, read = selection.require, selection.read


def helper_identity():
    return {**selection.helper_identity(), str(campaign.plain(__file__)): control.file_sha(__file__)}


def cache_identity(path):
    path = campaign.plain(path)
    require(path.is_dir() and not any(path.iterdir()), "exclusive bytecode cache must remain empty")
    return {"path": str(path), **diagnostic.lock_identity(path)}


def original_inputs(path):
    """Reuse the frozen selector's full choice/90-cell/latency validation."""
    path = campaign.plain(path)
    confirm = selection.sealed(read(path))
    require(path == Path(confirm["output_root"]) / "manifest.json"
            and confirm.get("format") == selection.CONFIRMATION_FORMAT and confirm.get("schema_version") == 1
            and confirm["helpers"] == selection.helper_identity() and confirm["status"] == "prepared_not_queued"
            and confirm["execution_implemented"] is False and confirm["no_reselection"] is True
            and confirm["formal_architecture_selection"] is False and confirm["hardware_deployment_ready"] is False,
            "original confirmation preparation identity differs")
    selection_path = campaign.checked(confirm["selection_manifest"])
    definition, manifest = selection.validate(selection_path)
    choice, assessment = selection.load_choice(campaign.checked(confirm["choice"]), definition)
    require(confirm["choice_sha256"] == choice["sha256"] and confirm["campaign_sha256"] == manifest["sha256"]
            and confirm["protocol"] == definition["protocol"]["confirmation"]
            and confirm["requests"] == selection.confirmation_requests(manifest, choice, assessment, Path(confirm["output_root"])),
            "original chosen rates, training seeds, or noise-only requests changed")
    return confirm, definition, manifest, choice, assessment


def learning_controller(manifest):
    path = Path(manifest["output_root"]) / "controller.json"
    record = read(path)
    require(record.get("manifest_sha256") == manifest["sha256"] and type(record.get("pid")) is int
            and record["pid"] > 0 and isinstance(record.get("start"), str) and record["start"].isdigit()
            and record.get("status") == "running", "learning controller requires its original PID/start identity")
    # The frozen learning controller leaves status='running' after its exit.
    return {"receipt": campaign.artifact(path), "pid": record["pid"], "start": record["start"]}


def evidence_closure(confirm_path, confirm, definition, manifest, choice, assessment, controller):
    """Freeze all original absolute receipts plus each relative report/trace."""
    result = {}
    def add(item):
        target = campaign.checked(item)
        result[str(target)] = item["sha256"]
    def walk(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys() and isinstance(value["path"], str) and Path(value["path"]).is_absolute():
                add({"path": value["path"], "sha256": value["sha256"]})
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for item in (campaign.artifact(confirm_path), confirm["selection_manifest"], confirm["choice"],
                 definition["campaign_manifest"], choice["assessment"], controller["receipt"]):
        add(item)
    for value in (confirm, definition, assessment):
        walk(value)
    for item in assessment["cells"].values():
        for receipt in item["development_receipts"].values():
            data = read(campaign.checked(receipt))
            walk(data)
            for artifact in data["artifacts"].values():
                add(campaign.artifact(control.checked(manifest["output_root"], artifact)))
        latency = item.get("latency")
        if latency:
            walk(latency)
            data = read(campaign.checked(latency["receipt"]))
            walk(data)
            bundle = read(campaign.checked(data["bundle"]))
            for name, sha in bundle["files"].items():
                add({"path": str(campaign.inside(Path(data["bundle"]["path"]).parent, name)), "sha256": sha})
    for name in ("summary.json", "audit.json"):
        target = Path(manifest["output_root"]) / name
        require(target.is_file(), "learning completion summary and audit must be sealed before preparing confirmation execution")
        add(campaign.artifact(target))
    return result


def prepare(confirmation_manifest, output_root, *, worker_timeout_seconds=21600., max_wait_seconds=604800., poll_seconds=30.):
    bounded_options(worker_timeout_seconds, max_wait_seconds, poll_seconds)
    confirm, definition, manifest, choice, assessment = original_inputs(confirmation_manifest)
    root = selection.independent_root(output_root, [confirm["output_root"], definition["output_root"],
        manifest["output_root"], manifest["source_root"], manifest["inputs"]["root"]])
    controller = learning_controller(manifest)
    closure = evidence_closure(confirmation_manifest, confirm, definition, manifest, choice, assessment, controller)
    # Preparation freezes only evidence already present; it does not schedule a worker.
    original = campaign.audit(definition["campaign_manifest"]["path"])
    require(original["status"] == "development_complete", "whole ninety-cell development grid is required")
    summary = read(Path(manifest["output_root"]) / "summary.json")
    require(summary.get("status") == "completed" and summary.get("manifest_sha256") == manifest["sha256"]
            and summary.get("grid_status") == "development_complete"
            and summary.get("expected_cells") == 90 and set(summary.get("results", {})) == set(original["cells"])
            and all(value.get("status") == "completed" for value in summary["results"].values())
            and read(campaign.checked(summary["audit"])) == original,
            "learning completed summary must match the real grid audit")
    require(not campaign.workers_in([confirm["output_root"]]), "original confirmation has an unresolved worker")
    require(not any((Path(confirm["output_root"]) / "evaluations").rglob("attempt_*")),
            "confirmation execution must own the original unstarted attempts")
    root.mkdir(parents=True, exist_ok=False)
    own = root / ".confirmation.lock"
    own.touch(exist_ok=False)
    cache = root / ".bytecode-cache"
    cache.mkdir()
    shared = {route: manifest["locks"][route] for route in
              (manifest["resource_lock"], manifest["study_lock"], str(Path(manifest["output_root"]) / ".learning.lock"))}
    result = selection.seal({"format": FORMAT, "schema_version": 1, "status": "prepared_not_queued",
        "output_root": str(root), "confirmation_manifest": campaign.artifact(confirmation_manifest),
        "confirmation_sha256": confirm["sha256"], "choice_sha256": choice["sha256"], "campaign_sha256": manifest["sha256"],
        "helpers": helper_identity(), "runtime": campaign.runtime_identity(manifest["source_root"]),
        "cache": cache_identity(cache), "locks": {**shared, str(own): diagnostic.lock_identity(own)},
        "learning_controller": controller, "input_closure": closure,
        "protocol": {"request_count": len(confirm["requests"]), "one_original_attempt": True, "retry": False,
            "steps": 4001, "environments": 8, "cases": 50, "settle_steps": 200, "min_steady_samples": 200,
            "noise_seeds": list(selection.NOISE_SEEDS), "worker_timeout_seconds": worker_timeout_seconds,
            "max_wait_seconds": max_wait_seconds, "poll_seconds": poll_seconds},
        "original_preparer_execution_implemented": False, **SCOPE})
    campaign.write(root / "manifest.json", result)
    return result


def bounded_options(timeout, wait, poll):
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (timeout, wait, poll))
            and timeout > 0 and wait >= 0 and 0 < poll <= 30, "finite timeout, wait, and bounded polling are required")


def validate(path):
    path = campaign.plain(path)
    execution = selection.sealed(read(path))
    require(execution.get("format") == FORMAT and execution.get("schema_version") == 1
            and path == Path(execution["output_root"]) / "manifest.json" and execution["status"] == "prepared_not_queued"
            and execution["helpers"] == helper_identity() and all(execution.get(k) == v for k, v in SCOPE.items())
            and execution["original_preparer_execution_implemented"] is False,
            "confirmation runner definition or frozen helpers changed")
    confirm_path = campaign.checked(execution["confirmation_manifest"])
    confirm, definition, manifest, choice, assessment = original_inputs(confirm_path)
    require(execution["confirmation_sha256"] == confirm["sha256"] and execution["choice_sha256"] == choice["sha256"]
            and execution["campaign_sha256"] == manifest["sha256"]
            and execution["learning_controller"] == learning_controller(manifest), "runner input or original controller handle changed")
    require(execution["input_closure"] == evidence_closure(confirm_path, confirm, definition, manifest, choice, assessment,
                execution["learning_controller"]), "original input artifact closure changed")
    require(execution["runtime"] == campaign.runtime_identity(manifest["source_root"]), "Python executable or CPU runtime changed")
    require(execution["cache"] == cache_identity(execution["cache"]["path"]), "CPU audit bytecode cache identity changed")
    expected_locks = {route: manifest["locks"][route] for route in
        (manifest["resource_lock"], manifest["study_lock"], str(Path(manifest["output_root"]) / ".learning.lock"))}
    expected_locks[str(Path(execution["output_root"]) / ".confirmation.lock")] = diagnostic.lock_identity(
        Path(execution["output_root"]) / ".confirmation.lock")
    require(execution["locks"] == expected_locks == {route: diagnostic.lock_identity(route) for route in execution["locks"]},
            "shared resource, study, learning, or confirmation lock inode changed")
    protocol = execution["protocol"]
    bounded_options(protocol["worker_timeout_seconds"], protocol["max_wait_seconds"], protocol["poll_seconds"])
    require(set(protocol) == {"request_count", "one_original_attempt", "retry", "steps", "environments", "cases",
                "settle_steps", "min_steady_samples", "noise_seeds", "worker_timeout_seconds", "max_wait_seconds", "poll_seconds"}
            and protocol["request_count"] == len(confirm["requests"]) and protocol["one_original_attempt"] is True
            and protocol["retry"] is False and protocol["steps"] == 4001 and protocol["environments"] == 8
            and protocol["cases"] == 50 and protocol["settle_steps"] == protocol["min_steady_samples"] == 200
            and protocol["noise_seeds"] == list(selection.NOISE_SEEDS), "confirmation execution protocol changed")
    return execution, confirm, definition, manifest, choice


def waiting_state(execution, definition, manifest):
    handle = execution["learning_controller"]
    live = control.process_start(handle["pid"]) == handle["start"]
    roots = [manifest["output_root"], str(Path(manifest["study_lock"]).parent),
             *(item["root"] for item in manifest.get("dependencies", {}).values())]
    workers = campaign.workers_in(roots)
    dependencies = {name: campaign.dependency_state(item) for name, item in manifest.get("dependencies", {}).items()}
    grid = campaign.audit(definition["campaign_manifest"]["path"])
    summary = read(Path(manifest["output_root"]) / "summary.json")
    complete = (grid["status"] == "development_complete" and summary.get("status") == "completed"
                and summary.get("manifest_sha256") == manifest["sha256"] and summary.get("grid_status") == "development_complete"
                and read(campaign.checked(summary["audit"])) == grid)
    return {"ready": complete and not live and not workers and all(v["ready"] for v in dependencies.values()),
            "learning_controller_live": live, "live_workers": workers, "dependencies": dependencies,
            "grid_status": grid["status"], "completed_summary_matches": complete}


@contextmanager
def acquire_resources(execution, definition, manifest, publish):
    routes = [manifest["resource_lock"], manifest["study_lock"], str(Path(manifest["output_root"]) / ".learning.lock")]
    streams = [Path(route).open("r+") for route in routes]
    started = time.monotonic()
    try:
        while True:
            for stream in streams:
                diagnostic.check_open_lock(stream, execution["locks"][stream.name])
            before = waiting_state(execution, definition, manifest)
            acquired = []
            if before["ready"]:
                try:
                    for stream in streams:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired.append(stream)
                except BlockingIOError:
                    pass
            if len(acquired) == len(streams):
                try:
                    after = waiting_state(execution, definition, manifest)
                    if after == before and after["ready"]:
                        for stream in streams:
                            diagnostic.check_open_lock(stream, execution["locks"][stream.name])
                        yield after
                        return
                finally:
                    for stream in reversed(acquired):
                        fcntl.flock(stream, fcntl.LOCK_UN)
            else:
                for stream in reversed(acquired):
                    fcntl.flock(stream, fcntl.LOCK_UN)
            publish("waiting", blockers=before, shared_locks_available=False, active=None)
            remaining = execution["protocol"]["max_wait_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                yield None
                return
            time.sleep(min(remaining, execution["protocol"]["poll_seconds"]))
    finally:
        for stream in streams:
            stream.close()


def environment_recipe(manifest, cache):
    # Record controlled values only; never serialize secrets in inherited env.
    cache_identity(cache)
    return {"PYTHONPATH": str(Path(manifest["source_root"]) / "src"), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(cache), "CUDA_VISIBLE_DEVICES": {"present": "CUDA_VISIBLE_DEVICES" in os.environ,
                                                                        "value": os.environ.get("CUDA_VISIBLE_DEVICES")}}


def run_environment(manifest, cache):
    result = campaign.run_environment(manifest)
    result["PYTHONPYCACHEPREFIX"] = str(cache)
    return result


def process_observation(pid, start, recipe):
    require(control.process_start(pid) == start, "worker PID/start ownership cannot be observed")
    root = Path(f"/proc/{pid}")
    argv = [v.decode() for v in (root / "cmdline").read_bytes().split(b"\0") if v]
    values = dict(v.decode().split("=", 1) for v in (root / "environ").read_bytes().split(b"\0") if v and b"=" in v)
    filtered = {key: values.get(key) for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    filtered["CUDA_VISIBLE_DEVICES"] = {"present": "CUDA_VISIBLE_DEVICES" in values, "value": values.get("CUDA_VISIBLE_DEVICES")}
    require(control.process_start(pid) == start and os.getpgid(pid) == pid and filtered == recipe,
            "worker process group or actual controlled environment differs")
    return {"pid": pid, "start": start, "pgid": pid, "argv": argv, "environment": filtered}


def group_members(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            start = control.process_start(pid)
            state = (path / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if start is not None and state != "Z" and os.getpgid(pid) == pgid:
                members.append({"pid": pid, "start": start, "pgid": pgid})
        except (OSError, IndexError):
            pass
    return sorted(members, key=lambda item: item["pid"])


def reclaim_group(pgid, anchors):
    """Reclaim surviving descendants only through an observed original group."""
    def anchored(members, known):
        return any(member == original and control.process_start(member["pid"]) == member["start"]
                   and os.getpgid(member["pid"]) == pgid for member in members for original in known)
    first = group_members(pgid)
    if first:
        require(anchored(first, anchors), "original observed group handle cannot anchor descendant ownership")
        # These members were observed while an original handle still anchored
        # the original group; preserve them across TERM before checking KILL.
        anchors = [*anchors, *first]
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 1.
        while group_members(pgid) and time.monotonic() < deadline:
            time.sleep(.05)
        second = group_members(pgid)
        if second:
            require(anchored(second, anchors), "original descendant handles changed during recovery")
            os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 1.
            while group_members(pgid) and time.monotonic() < deadline:
                time.sleep(.05)
    remaining = group_members(pgid)
    require(not remaining, "worker descendants remain live after endpoint recovery")
    return {"observed_descendants": first, "remaining": remaining}


def confirmation_workers(root):
    """Keep unresolved group ownership blocking even after its leader exits."""
    result = campaign.workers_in([root])
    for path in Path(root).rglob("worker.process.json"):
        value = read(path)
        if value.get("recovery", {}).get("ownership_unresolved"):
            result.append({"receipt": str(path), "reason": "original process group ownership remains unresolved",
                           "pgid": value.get("pid"), "original_handles": value.get("group_handles", [])})
    return result


def worker(request, directory, environment, timeout, publish):
    """Actual bare argv, /proc identity/env, group ownership, and terminal SHA."""
    command, recipe = request["original"]["command"], request["environment"]
    require(environment.get("PYTHONPYCACHEPREFIX") == request["cache"]["path"], "worker cache environment differs")
    record = {"format": "transformer_rl.learning_confirmation_worker", "schema_version": 1,
        "execution_sha256": request["execution_sha256"], "request_sha256": control.file_sha(directory / "request.json"),
        "command": command, "environment": recipe, "cache": request["cache"], "controller": request["controller"],
        "status": "launching", "pid": None, "start": None, "started_at": control.now(), "timeout_seconds": timeout}
    process = None
    campaign.write(directory / "worker.process.json", record)
    with (directory / "worker.log").open("x") as stream:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            record.update(pid=process.pid, start=control.process_start(process.pid), status="running")
            record["observed"] = process_observation(process.pid, record["start"], recipe)
            require(record["observed"]["argv"] == command, "actual worker argv differs from the frozen bare command")
            campaign.write(directory / "worker.process.json", record, replace=True)
            started, beat = time.monotonic(), time.monotonic()
            record["timed_out"] = False
            record["group_handles"] = [{"pid": process.pid, "start": record["start"], "pgid": process.pid}]
            while True:
                # Observe descendants while the original leader is still
                # reserved (including before poll reaps a just-exited leader).
                if control.process_start(process.pid) == record["start"]:
                    observed = group_members(process.pid)
                    require(control.process_start(process.pid) == record["start"], "original leader changed during group observation")
                    fresh = [v for v in observed if v not in record["group_handles"]]
                    if fresh:
                        record["group_handles"].extend(fresh)
                        campaign.write(directory / "worker.process.json", record, replace=True)
                if process.poll() is not None:
                    break
                if time.monotonic() - started >= timeout:
                    record["timed_out"] = True
                    control.terminate_owned(process, record["start"])
                    break
                if time.monotonic() - beat >= 30:
                    publish(active=request["original"]["directory"])
                    beat = time.monotonic()
                time.sleep(.1)
            record["recovery"] = reclaim_group(process.pid, record["group_handles"])
            record.update(status="finished", returncode=process.poll(), finished_at=control.now())
        except BaseException as error:
            if process is not None:
                try:
                    control.terminate_owned(process, record["start"])
                    record["recovery"] = reclaim_group(process.pid, record.get("group_handles", []))
                except BaseException as recovery_error:
                    record["recovery"] = {"ownership_unresolved": True, "remaining": group_members(process.pid),
                                          "error": f"{type(recovery_error).__name__}: {recovery_error}"}
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          returncode=process.poll() if process else None, finished_at=control.now(),
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            campaign.write(directory / "worker.process.json", record, replace=True)
    return record


def expected_identity(execution, confirm, original):
    return {"manifest_sha256": execution["campaign_sha256"], "cell": original["cell"],
        "evaluation_seed": original["evaluation_seed"], "checkpoint_sha256": original["checkpoint"]["sha256"],
        "checkpoint_update": 1200, "use": "heldout_noise_stream_only", "choice_sha256": confirm["choice_sha256"],
        "confirmation_sha256": confirm["sha256"]}


def request_cache(execution, original):
    return campaign.inside(execution["output_root"], Path("caches") / original["cell"] / f"seed_{original['evaluation_seed']}")


def verify_execution(execution, confirm, manifest, original, directory, receipt=None):
    request = selection.sealed(read(directory / "request.json"))
    require(request.get("format") == "transformer_rl.learning_confirmation_request" and request.get("schema_version") == 1
            and request["execution_sha256"] == execution["sha256"] and request["confirmation_sha256"] == confirm["sha256"]
            and request["identity"] == expected_identity(execution, confirm, original) and request["original"] == original
            and request["timeout_seconds"] == execution["protocol"]["worker_timeout_seconds"], "actual execution request differs")
    cache = request_cache(execution, original)
    require(request["cache"] == cache_identity(cache) and request["environment"]["PYTHONPATH"] == str(Path(manifest["source_root"]) / "src")
            and request["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
            and request["environment"]["PYTHONPYCACHEPREFIX"] == str(cache), "worker cache or controlled environment changed")
    process = read(directory / "worker.process.json")
    require(process.get("format") == "transformer_rl.learning_confirmation_worker" and process.get("schema_version") == 1
            and process["execution_sha256"] == execution["sha256"]
            and process["request_sha256"] == control.file_sha(directory / "request.json")
            and process["command"] == original["command"] and process["environment"] == request["environment"]
            and process["cache"] == request["cache"] and process["controller"] == request["controller"]
            and process["timeout_seconds"] == request["timeout_seconds"], "actual worker request/command/environment differs")
    observed = process.get("observed", {})
    require(observed == {"pid": process["pid"], "start": process["start"], "pgid": process["pid"],
                "argv": original["command"], "environment": request["environment"]}
            and {"pid": process["pid"], "start": process["start"], "pgid": process["pid"]} in process.get("group_handles", [])
            and selection.terminal_process(process, original["command"]) and process.get("recovery", {}).get("remaining") == []
            and not process.get("recovery", {}).get("ownership_unresolved")
            and not group_members(process["pid"]), "actual worker ownership or terminal endpoint differs")
    controller = request["controller"]
    require(set(controller) == {"path", "pid", "start"} and type(controller["pid"]) is int and controller["pid"] > 0
            and isinstance(controller["start"], str) and controller["start"].isdigit(), "runner controller handle is invalid")
    route = campaign.inside(execution["output_root"], controller["path"])
    actual_controller = read(route)
    require(actual_controller["execution_sha256"] == execution["sha256"]
            and actual_controller["pid"] == controller["pid"] and actual_controller["start"] == controller["start"],
            "worker belongs to a different runner controller")
    artifacts = {"request": campaign.artifact(directory / "request.json"), "worker": campaign.artifact(directory / "worker.process.json")}
    if receipt is not None:
        require(receipt["execution"] == artifacts, "published suite execution receipt changed")
    return artifacts


def attempt_closure(directory):
    return {str(path.relative_to(directory)): control.file_sha(campaign.plain(path)) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in {"execution.receipt.json"}}


def stage_candidate(directory, confirm, shim, variant, seed, suite):
    """Shadow only the embedded trace route; leave original report bytes intact."""
    staging = directory / ".validation"
    staging.mkdir(exist_ok=False)
    for name, item in suite["artifacts"].items():
        source = control.checked(confirm["output_root"], item)
        if name != "control":
            os.link(source, staging / source.name)
    original_control = read(directory / "control.json")
    require(Path(original_control["trace"]["path"]).resolve() == (directory / "trace.npz").resolve(),
            "original control trace route differs")
    shadow_control = {**original_control, "trace": {**original_control["trace"], "path": str(staging / "trace.npz")}}
    campaign.write(staging / "control.json", shadow_control)
    staged_artifacts = diagnostic.verify_outputs(staging, shim, variant, seed)
    candidate = {**suite, "directory": str(staging.relative_to(Path(confirm["output_root"]))), "artifacts": staged_artifacts}
    campaign.write(staging / "receipt.json", candidate)
    return staging


def publish_suite(directory, suite):
    """Publish complete bytes atomically and exclusively after all checks."""
    pending = directory / ".ready.receipt.json"
    campaign.write(pending, suite)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(pending, directory / "receipt.json")  # fails if already published
    pending.unlink()


def evaluate_request(execution, confirm, manifest, original, controller, publish):
    directory = campaign.inside(confirm["output_root"], original["directory"])
    require(directory == Path(confirm["output_root"]) / "evaluations" / original["cell"] / f"seed_{original['evaluation_seed']}" / "attempt_0000",
            "only the original attempt_0000 is allowed")
    require(set(directory.parent.glob("attempt_*")) <= {directory}, "confirmation retries are forbidden")
    if directory.exists():
        # A directory created before a crash already consumed the only attempt.
        return {"status": "already_attempted", "directory": original["directory"]}
    for item in [original["checkpoint"], *original["configs"].values()]:
        campaign.checked(item)
    directory.mkdir(parents=True, exist_ok=False)
    cache = request_cache(execution, original)
    cache.mkdir(parents=True, exist_ok=False)
    request = selection.seal({"format": "transformer_rl.learning_confirmation_request", "schema_version": 1,
        "execution_sha256": execution["sha256"], "confirmation_sha256": confirm["sha256"],
        "identity": expected_identity(execution, confirm, original), "original": original,
        "controller": controller, "cache": cache_identity(cache), "environment": environment_recipe(manifest, cache),
        "timeout_seconds": execution["protocol"]["worker_timeout_seconds"]})
    campaign.write(directory / "request.json", request)
    status, error = "incomplete", None
    try:
        worker(request, directory, run_environment(manifest, cache), request["timeout_seconds"], publish)
        evidence = verify_execution(execution, confirm, manifest, original, directory)
        cell = next(cell for cell in manifest["inputs"]["cells"] if campaign.cell_key(cell) == original["cell"])
        _, shim = campaign.evaluation_inputs(manifest, cell, original["checkpoint"])
        shim["output_root"] = confirm["output_root"]
        artifacts = diagnostic.verify_outputs(directory, shim, cell["variant"], original["evaluation_seed"])
        suite = {"identity": request["identity"], "status": "completed", "directory": original["directory"],
                 "artifacts": artifacts, "execution": evidence}
        # The selector insists on directory/receipt.json. Validate a candidate
        # in a private child directory using hardlinks, before the canonical
        # public receipt exists. A crash here can never look selector-ready.
        staging = stage_candidate(directory, confirm, shim, cell["variant"], original["evaluation_seed"], suite)
        require(selection.load_suite(manifest, cell, original["checkpoint"], campaign.artifact(staging / "receipt.json"),
                             original["evaluation_seed"], confirmation=confirm) is not None, "candidate suite is incomplete")
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        publish_suite(directory, suite)
        status = "completed"
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        if isinstance(failure, KeyboardInterrupt):
            status = "interrupted"
        # Invalid output never becomes a selector-ready receipt. Preserve any
        # staging files as partial evidence of the consumed original attempt.
        if (directory / "receipt.json").exists():
            (directory / "receipt.json").rename(directory / "unpublished.receipt.json")
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        result = selection.seal({"format": "transformer_rl.learning_confirmation_attempt", "schema_version": 1,
            "execution_sha256": execution["sha256"], "confirmation_sha256": confirm["sha256"],
            "identity": request["identity"], "directory": original["directory"], "status": status,
            "error": error, "finished_at": control.now(), "artifacts": attempt_closure(directory)})
        campaign.write(directory / "execution.receipt.json", result)
    return result


def audit(path):
    execution, confirm, definition, manifest, choice = validate(path)
    states, missing = {}, []
    expected_dirs = {campaign.inside(confirm["output_root"], item["directory"]) for item in confirm["requests"]}
    existing_dirs = set((Path(confirm["output_root"]) / "evaluations").rglob("attempt_*"))
    require(existing_dirs <= expected_dirs, "extra confirmation attempts or undeclared architecture/seed found")
    for original in confirm["requests"]:
        key = f"{original['cell']}/{original['evaluation_seed']}"
        directory = campaign.inside(confirm["output_root"], original["directory"])
        receipt_path = directory / "execution.receipt.json"
        if not receipt_path.exists():
            states[key] = {"status": "unsealed" if directory.exists() else "not_started"}
            missing.append(key)
            continue
        receipt = selection.sealed(read(receipt_path))
        require(receipt.get("format") == "transformer_rl.learning_confirmation_attempt" and receipt.get("schema_version") == 1
                and receipt["execution_sha256"] == execution["sha256"] and receipt["confirmation_sha256"] == confirm["sha256"]
                and receipt["identity"] == expected_identity(execution, confirm, original)
                and receipt["directory"] == original["directory"] and receipt["artifacts"] == attempt_closure(directory),
                "attempt identity or sealed output closure changed")
        require(receipt["status"] in {"completed", "incomplete", "interrupted"}, "unknown attempt terminal status")
        if receipt["status"] == "completed":
            suite = read(directory / "receipt.json")
            require(suite["status"] == "completed" and suite["identity"] == receipt["identity"], "suite differs from runner attempt")
            verify_execution(execution, confirm, manifest, original, directory, suite)
        else:
            require(not (directory / "receipt.json").exists(), "failed attempt cannot publish a ready selector receipt")
            # A failed/unsealed original attempt stays missing even when its
            # process handle is unresolved. It is never authorized to relaunch.
            missing.append(key)
        states[key] = {"status": receipt["status"], "receipt": campaign.artifact(receipt_path), "error": receipt["error"]}
    # The independent selector retains every selected/zero-selected identity and
    # all physical gates. This runner additionally verifies execution provenance.
    selected = selection.audit_confirmation(execution["confirmation_manifest"]["path"])
    require(selected["choice_sha256"] == execution["choice_sha256"] and selected["original_choices"] == choice["choices"],
            "confirmation audit changed the original LR choices")
    return {"format": "transformer_rl.learning_confirmation_execution_audit", "schema_version": 1,
        "execution_sha256": execution["sha256"], "confirmation_sha256": confirm["sha256"],
        "choice_sha256": choice["sha256"], "status": "not_ready" if missing else selected["status"],
        "expected_suites": len(confirm["requests"]), "completed_suites": sum(v["status"] == "completed" for v in states.values()),
        "missing": missing, "attempts": states, "selector_audit": selected, "original_choices": choice["choices"], **SCOPE}


def run(path):
    execution, confirm, definition, manifest, _ = validate(path)
    root = Path(execution["output_root"])
    state = {"status": "waiting", "execution_sha256": execution["sha256"], "started_at": control.now(), "results": {}}
    def publish(status=None, **values):
        if status:
            state["status"] = status
        state.update(values, updated_at=control.now())
        campaign.write(root / "summary.json", state, replace=True)
    with (root / ".confirmation.lock").open("r+") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic.check_open_lock(own, execution["locks"][own.name])
        controllers = root / "controllers"
        controllers.mkdir(exist_ok=True)
        index = len(list(controllers.glob("controller_*.process.json")))
        controller_path = controllers / f"controller_{index:04d}.process.json"
        require(not controller_path.exists(), "controller invocation cannot overwrite an earlier receipt")
        handle = {"path": str(controller_path), "pid": os.getpid(), "start": control.process_start(os.getpid())}
        process = {"execution_sha256": execution["sha256"], "pid": handle["pid"], "start": handle["start"],
            "command": [v.decode() for v in Path("/proc/self/cmdline").read_bytes().split(b"\0") if v],
            "status": "running", "started_at": control.now()}
        campaign.write(controller_path, process)
        try:
            # Reject undeclared attempts or changed completed output before any
            # new worker is allowed to consume another original request.
            audit(path)
            with acquire_resources(execution, definition, manifest, publish) as ready:
                if ready is None:
                    publish("waiting", active=None)
                    return state
                validate(path)
                publish("evaluating", dependencies=ready, active=None)
                for original in confirm["requests"]:
                    require(not confirmation_workers(confirm["output_root"]), "confirmation already owns a live/unresolved worker or original group")
                    publish(active=original["directory"])
                    result = evaluate_request(execution, confirm, manifest, original, handle, publish)
                    state["results"][f"{original['cell']}/{original['evaluation_seed']}"] = result
                    publish(active=None)
                report = audit(path)
                target = root / "audits" / f"controller_{index:04d}.json"
                campaign.write(target, selection.seal(report))
                publish("completed" if report["status"] in {"confirmed", "not_confirmed"} else "incomplete",
                        confirmation_status=report["status"], audit=campaign.artifact(target), active=None)
        except BaseException as error:
            publish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", active=None,
                    error=f"{type(error).__name__}: {error}")
            raise
        finally:
            process.update(status="finished", outcome=state["status"], finished_at=control.now())
            campaign.write(controller_path, process, replace=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    preparer = operations.add_parser("prepare", help="freeze a separate runner definition; never queue a worker")
    preparer.add_argument("--confirmation-manifest", required=True, type=Path)
    preparer.add_argument("--output-root", required=True, type=Path)
    preparer.add_argument("--worker-timeout-seconds", type=float, default=21600.)
    preparer.add_argument("--max-wait-seconds", type=float, default=604800.)
    preparer.add_argument("--poll-seconds", type=float, default=30.)
    for name in ("validate", "audit", "run"):
        sub = operations.add_parser(name)
        sub.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    old = {}
    def interrupted(*_):
        raise KeyboardInterrupt
    try:
        if args.operation == "run":
            for number in (signal.SIGINT, signal.SIGTERM):
                old[number] = signal.signal(number, interrupted)
        if args.operation == "prepare":
            result = prepare(args.confirmation_manifest, args.output_root, worker_timeout_seconds=args.worker_timeout_seconds,
                             max_wait_seconds=args.max_wait_seconds, poll_seconds=args.poll_seconds)
        elif args.operation == "validate":
            result = {"status": "validated", "sha256": validate(args.manifest)[0]["sha256"]}
        elif args.operation == "audit":
            result = audit(args.manifest)
        else:
            result = run(args.manifest)
    except KeyboardInterrupt:
        return 130
    finally:
        for number, handler in old.items():
            signal.signal(number, handler)
    print(campaign.json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if args.operation != "run" or result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare, validate and queue paired diagnostics of ten sealed transfer models.

Preparation and validation use only the standard library and never start a
simulator. The run interface can invoke only evaluate-suite. Training artifacts,
predecessor campaigns and their source checkouts are read-only inputs.
"""
from __future__ import annotations

import argparse
from array import array
import ast
from contextlib import contextmanager
from collections import Counter
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import sys
import time
import zipfile


_HELPER = Path(__file__).with_name("run_transfer_campaign.py")
_SPEC = importlib.util.spec_from_file_location("diagnostic_transfer_helpers", _HELPER)
transfer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(transfer)
control, write = transfer.control, transfer.write

SOURCE_COMMIT = "9b576e7d8f72e90ad361df83139bc2bb78e05165"
# Derived from the 37 src/transformer_rl/*.py Git blobs at SOURCE_COMMIT.
# A matching HEAD or a self-described archive receipt cannot bless dirty code.
SOURCE_PACKAGE_SHA256 = "45fce95d3abae0291d04d325baa3b759eb8c4ab1d41be2d6c145f5d6c5677c1a"
PROTOCOL = {"checkpoint_update": 1200, "steps": 4001, "seeds": [8701, 9701],
            "settle_steps": 200, "min_steady_samples": 200, "cases": 50,
            "case_replicas": 8, "trace_replicas": 2, "policy_dt_s": .01,
            "seed_role": "paired_diagnostic_retest_of_existing_control_seeds",
            "independent_holdout": False}
TRACE_SHAPES = {"time_s": (), "command_reference": (3,), "actual": (3,),
    "position_xy": (2,), "tilt": (), "leg_target": (4,), "wheel_target": (2,),
    "motor_position": (6,), "motor_velocity": (6,), "motor_effort": (6,),
    "requested_motor_effort": (6,), "effort_bounds": (6, 2),
    "scaled_nominal_requested_motor_effort": (6,), "scaled_nominal_effort_bounds": (6, 2),
    "failure": (), "success": (), "done": (), "episode_id": ()}


def read(path):
    value = control.read(path)
    json.dumps(value, allow_nan=False)
    return value


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a SHA256 identity")
    return value


def absolute_artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": control.file_sha(path)}


def checked(item):
    path = Path(item["path"])
    if not path.is_absolute() or control.file_sha(path) != sha(item["sha256"]):
        raise ValueError(f"sealed artifact hash mismatch: {path}")
    return path


def controller_identity():
    return {str(path.resolve()): control.file_sha(path) for path in
            (Path(__file__), _HELPER, _HELPER.with_name("run_frame_control_campaign.py"))}


def source_identity(root):
    value = control.source_identity(root)
    return {key: value[key] for key in ("root", "files", "sha256", "git_head")}


def verify_source(root, expected, origin=None):
    actual = source_identity(root)
    if (actual["files"] != expected["files"] or actual["sha256"] != expected["sha256"]
            or expected["git_head"] != SOURCE_COMMIT
            or control.digest(expected["files"]) != sha(expected["sha256"])
            or expected["sha256"] != SOURCE_PACKAGE_SHA256):
        raise ValueError("evaluation source differs from the sealed 9b576e7 package")
    if origin is not None:
        evidence = read(checked(origin))
        for key in ("files", "sha256", "git_head"):
            if evidence.get(key) != expected[key]:
                raise ValueError("source origin receipt differs")


def lock_identity(path):
    value = Path(path).stat()
    return {"device": value.st_dev, "inode": value.st_ino}


def check_open_lock(stream, expected):
    value = os.fstat(stream.fileno())
    actual = {"device": value.st_dev, "inode": value.st_ino}
    if actual != expected or lock_identity(stream.name) != expected:
        raise ValueError("existing resource or study lock was replaced")


def snapshot_identity(path, expected):
    root = Path(path).resolve()
    receipt = absolute_artifact(root / "snapshot.json")
    identity = read(receipt["path"])
    if control.digest(identity["files"]) != identity["sha256"] or identity["sha256"] != expected:
        raise ValueError("environment snapshot identity differs")
    for route, value in identity["files"].items():
        control.checked(root, {"path": route, "sha256": sha(value)})
    return {"root": str(root), "sha256": expected, "receipt": receipt}


def dependency_definition(receipt, expected, resource):
    receipt = Path(receipt).resolve()
    campaign = absolute_artifact(receipt.parent / "campaign.json")
    definition = read(campaign["path"])
    if control.digest(definition) != sha(expected):
        raise ValueError("dependency campaign identity differs")
    if Path(definition["resource_lock"]).resolve() != resource:
        raise ValueError("dependency campaigns do not share the declared resource lock")
    return {"receipt": str(receipt), "campaign_sha256": expected, "campaign": campaign}


def input_identity(study, training_seed):
    plan = control.load_plan(study)
    names = [variant["name"] for variant in plan["spec"]["variants"]]
    stages = plan["spec"]["stages"]
    cases = stages[0]["scenarios"]
    if (len(names) != 10 or len(stages) != 1 or stages[0]["updates"] != 1200
            or len(cases) != 50 or training_seed not in plan["spec"]["seeds"]):
        raise ValueError("diagnostics require ten variants, fifty cases and the exact 1200-update study")
    if set(plan["spec"]["seeds"]).intersection(PROTOCOL["seeds"]):
        raise ValueError("paired diagnostic seeds cannot overlap training seeds")
    configs = {}
    snapshots = {}
    environments = {}
    controls = {}
    for route, expected in plan["configs"].items():
        path = control.inside(study, route)
        config = read(path)
        if control.digest(config) != expected:
            raise ValueError("training plan configuration changed")
        configs[route] = absolute_artifact(path)
    for name in names:
        environments[name] = {}
        for case in cases:
            config = read(configs[f"configs/{name}.eval.{case}.json"]["path"])
            environment = config["environment"]
            if environment["num_envs"] != 8 or config["control"]["policy_dt_s"] != .01:
                raise ValueError("case rows or policy interval differ from the diagnostic protocol")
            controls.setdefault(name, config["control"])
            if config["control"] != controls[name]:
                raise ValueError("case control contracts differ")
            snapshot = Path(environment["snapshot"]).resolve()
            if str(snapshot) not in snapshots:
                snapshots[str(snapshot)] = snapshot_identity(snapshot, environment["snapshot_sha256"])
            if snapshots[str(snapshot)]["sha256"] != environment["snapshot_sha256"]:
                raise ValueError("case snapshot identities differ")
            contract = control.checked(snapshot, {"path": environment["contract"],
                "sha256": environment["contract_sha256"]})
            value = read(contract)
            if (value.get("evaluation_exact_cases") is not True or value["target_num_envs"] != 8
                    or [entry["name"] for entry in value["evaluation"]["cases"]] != [case]):
                raise ValueError("evaluation contract is not the declared eight-row fixed case")
            environments[name][case] = environment
    if len(snapshots) != 1:
        raise ValueError("diagnostics must use one frozen training environment snapshot")
    checkpoints = {}
    for name in names:
        checkpoint = control.ready_checkpoint(study, name, training_seed, 1200, plan)
        if checkpoint["status"] != "ready" or checkpoint["update"] != 1200:
            raise ValueError("an exact sealed 1200-update checkpoint is required for every variant")
        completion = absolute_artifact(control.checked(study, checkpoint["training_completion"]))
        model_path = Path(checkpoint["checkpoint"])
        sidecar = absolute_artifact(str(model_path) + ".json")
        metadata = read(sidecar["path"])
        if (metadata["update"] != 1200 or metadata["sha256"] != checkpoint["checkpoint_sha256"]
                or metadata["config"]["control"] != controls[name]
                or metadata["metadata"]["environment_provenance"]["identity"] != next(iter(snapshots.values()))["sha256"]):
            raise ValueError("checkpoint sidecar update, control or training snapshot differs")
        checkpoints[name] = {"checkpoint": str(model_path),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"], "update": 1200,
            "completion": completion, "sidecar": sidecar,
            "training_state": absolute_artifact(control.checked(study, checkpoint["training_state"])),
            "control_sha256": control.digest(controls[name])}
    return {"study_root": str(study), "plan_sha256": plan["sha256"],
        "plan": absolute_artifact(study / "plan.json"), "variants": names, "cases": cases,
        "training_seed": training_seed, "configs": configs, "snapshots": snapshots,
        "environments": environments, "checkpoints": checkpoints}


def prepare(args):
    study, root, source, resource = (Path(getattr(args, key)).resolve() for key in
                                   ("study_root", "output_root", "source_root", "resource_lock"))
    if not resource.is_file() or not (study / ".run.lock").is_file():
        raise ValueError("both existing resource and training study lock files are required")
    inputs = input_identity(study, args.training_seed)
    actual = source_identity(source)
    origin = absolute_artifact(args.source_identity) if args.source_identity else None
    expected = read(origin["path"]) if origin else {**actual, "git_head": SOURCE_COMMIT}
    expected = {key: expected[key] for key in ("files", "sha256", "git_head")}
    verify_source(source, expected, origin)
    dependencies = {name: dependency_definition(getattr(args, name + "_receipt"),
        getattr(args, name + "_campaign_sha256"), resource) for name in ("transfer", "curriculum")}
    if Path(read(dependencies["transfer"]["campaign"]["path"])["study_root"]).resolve() != study:
        raise ValueError("transfer dependency belongs to a different checkpoint study")
    forbidden = [study, source, resource.parent, *(Path(d["receipt"]).parent for d in dependencies.values()),
                 *(Path(value) for value in inputs["snapshots"])]
    if any(root == path or root.is_relative_to(path) or path.is_relative_to(root) for path in forbidden):
        raise ValueError("diagnostic output must be separate from all frozen input roots")
    definition = {"format": "transformer_rl.frame_diagnostic_campaign", "schema_version": 1,
        "inputs": inputs, "source_root": str(source), "source_commit": SOURCE_COMMIT,
        "source": expected, "source_origin": origin, "controllers": controller_identity(),
        "resource_lock": str(resource), "dependencies": dependencies,
        "locks": {str(path): lock_identity(path) for path in (resource, study / ".run.lock")},
        "output_root": str(root), "device": args.device, "protocol": dict(PROTOCOL),
        "formal_architecture_selection": False, "hardware_deployment_ready": False}
    definition["sha256"] = control.digest(definition)
    if root.exists() and any(root.iterdir()):
        raise ValueError("prepare requires a new or empty diagnostic output root")
    root.mkdir(parents=True, exist_ok=True)
    write(root / "manifest.json", definition)
    return definition


def validate(path):
    path = Path(path).resolve()
    manifest = read(path)
    if (manifest.get("format") != "transformer_rl.frame_diagnostic_campaign"
            or manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL
            or manifest.get("source_commit") != SOURCE_COMMIT
            or control.digest({key: value for key, value in manifest.items() if key != "sha256"}) != manifest.get("sha256")):
        raise ValueError("diagnostic manifest identity or protocol differs")
    if path != Path(manifest["output_root"]) / "manifest.json":
        raise ValueError("manifest is not in its sealed diagnostic output root")
    if manifest["controllers"] != controller_identity():
        raise ValueError("sealed controller or helper source changed")
    verify_source(manifest["source_root"], manifest["source"], manifest["source_origin"])
    inputs = manifest["inputs"]
    if input_identity(Path(inputs["study_root"]), inputs["training_seed"]) != inputs:
        raise ValueError("sealed training inputs or exact checkpoints changed")
    resource = Path(manifest["resource_lock"])
    if not resource.is_file() or not (Path(inputs["study_root"]) / ".run.lock").is_file():
        raise ValueError("existing resource lock disappeared")
    if manifest["locks"] != {path: lock_identity(path) for path in manifest["locks"]}:
        raise ValueError("existing resource or study lock was replaced")
    for dependency in manifest["dependencies"].values():
        checked(dependency["campaign"])
        if dependency_definition(dependency["receipt"], dependency["campaign_sha256"], resource) != dependency:
            raise ValueError("sealed dependency definition changed")
    return manifest


def dependency_state(dependency):
    path = Path(dependency["receipt"])
    if not path.exists():
        return {"ready": False, "reason": "completion receipt has not been written"}
    value = read(path)
    if value.get("campaign_sha256") != dependency["campaign_sha256"]:
        raise ValueError("dependency completion belongs to another campaign")
    return {"ready": value.get("status") == "completed", "status": value.get("status"),
            **absolute_artifact(path)}


def active_workers(manifest):
    result = []
    study_roots = {Path(manifest["inputs"]["study_root"]), Path(manifest["resource_lock"]).parent}
    roots = {Path(manifest["output_root"]), *(Path(d["receipt"]).parent for d in manifest["dependencies"].values())}
    receipts = {path for root in study_roots for path in (root / "jobs").rglob("*.process.json")}
    receipts.update(path for root in roots for path in root.rglob("worker.process.json"))
    for path in sorted(receipts):
        record = read(path)
        if record.get("status") in {"launching", "running"} and (record.get("start") is None or record.get("pid") is None):
            result.append({"receipt": str(path), "reason": "worker ownership is not verifiable"})
        elif (record.get("start") is not None and record.get("pid") is not None
                and control.process_start(record["pid"]) == str(record["start"])):
            result.append({"receipt": str(path), "pid": record["pid"], "start": record["start"]})
    return result


@contextmanager
def acquire_resources(manifest, args, publish):
    started = time.monotonic()
    locks = [Path(manifest["resource_lock"]), Path(manifest["inputs"]["study_root"]) / ".run.lock"]
    # Deduplicate while retaining the shared-resource-before-study acquisition order.
    with locks[0].open("r+") as resource:
        with locks[-1].open("r+") as study:
            streams = [resource] if locks[0] == locks[-1] else [resource, study]
            while True:
                for stream in streams:
                    check_open_lock(stream, manifest["locks"][str(Path(stream.name))])
                dependencies = {name: dependency_state(value) for name, value in manifest["dependencies"].items()}
                live = active_workers(manifest)
                acquired = []
                if all(value["ready"] for value in dependencies.values()) and not live:
                    try:
                        for stream in streams:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            acquired.append(stream)
                    except BlockingIOError:
                        pass
                if len(acquired) == len(streams):
                    try:
                        current = {name: dependency_state(value) for name, value in manifest["dependencies"].items()}
                        if current == dependencies and not active_workers(manifest):
                            for stream in streams:
                                check_open_lock(stream, manifest["locks"][str(Path(stream.name))])
                            yield dependencies
                            return
                    finally:
                        for stream in reversed(acquired):
                            fcntl.flock(stream, fcntl.LOCK_UN)
                else:
                    for stream in reversed(acquired):
                        fcntl.flock(stream, fcntl.LOCK_UN)
                blockers = {"dependencies": dependencies, "live_workers": live, "resource_or_study_lock_held": True}
                remaining = args.max_wait_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    publish("blocked", blockers=blockers, active=None)
                    yield None
                    return
                publish("waiting", blockers=blockers, active=None)
                time.sleep(min(args.poll_seconds, remaining))


def finite(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("diagnostic statistic must be finite with its declared sign")


def count(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("diagnostic statistic has an invalid sample count")


def validate_planar_pool(pool, maximum, *, require_available=False):
    if type(pool["available"]) is not bool:
        raise ValueError("planar availability must be explicit")
    count(pool["intervals"], maximum)
    if pool["available"] != bool(pool["intervals"]) or (require_available and not pool["available"]):
        raise ValueError("planar motion has no valid velocity intervals")
    for key in ("observed_duration_s", "path_length_m", "overflow_duration_s"):
        finite(pool[key], nonnegative=True)
    if pool["overflow_duration_s"] > pool["observed_duration_s"] + 1e-7:
        raise ValueError("planar histogram duration differs")
    if pool["p95_bin_width_m_s"] != .001 or pool["p95_overflow_from_m_s"] != 10. or not isinstance(pool["p95_method"], str):
        raise ValueError("planar speed quantile protocol differs")
    if set(pool["velocity_world"]) != {"vx", "vy"}:
        raise ValueError("planar velocity must contain both world axes")
    if len(pool["p95_bin_m_s"]) != 2:
        raise ValueError("planar quantile bin differs")
    for key in ("mean_speed_m_s", "rms_speed_m_s", "max_speed_m_s"):
        if pool["available"]:
            finite(pool[key], nonnegative=True)
        elif pool[key] is not None:
            raise ValueError("unavailable planar measurements must remain null")
    for axis in ("vx", "vy"):
        for key in ("mean_m_s", "rms_m_s"):
            value = pool["velocity_world"][axis][key]
            if pool["available"]:
                finite(value, nonnegative=key == "rms_m_s")
            elif value is not None:
                raise ValueError("unavailable planar axis measurements must remain null")
    lower, upper = pool["p95_bin_m_s"]
    if pool["available"]:
        if pool["observed_duration_s"] <= 0:
            raise ValueError("planar motion observed duration must be positive")
        finite(lower, nonnegative=True)
        if upper is None:
            if lower != 10. or pool["p95_speed_m_s"] is not None or pool["overflow_duration_s"] <= 0:
                raise ValueError("planar overflow quantile differs")
        else:
            finite(upper, nonnegative=True)
            finite(pool["p95_speed_m_s"], nonnegative=True)
            if not lower <= pool["p95_speed_m_s"] <= upper:
                raise ValueError("planar quantile lies outside its histogram bin")
    elif (pool["observed_duration_s"] != 0 or pool["path_length_m"] != 0
            or pool["overflow_duration_s"] != 0 or lower is not None or upper is not None
            or pool["p95_speed_m_s"] is not None):
        raise ValueError("unavailable planar pool contains fabricated measurements")


def validate_metrics(value, rows):
    samples = 4001 * rows
    if value.get("available") is not True or value["full_interval"]["samples"] != samples:
        raise ValueError("complete control samples unavailable")
    for key, expected in (("policy_dt_s", .01), ("settle_steps", 200), ("min_steady_samples", 200)):
        if value["protocol"][key] != expected:
            raise ValueError("control sampling protocol differs")
    planar = value["planar_motion"]
    if (planar["coordinate_frame"] != "world_xy" or planar["num_envs"] != rows
            or planar["physical_samples"] != samples or planar["full_interval"]["available"] is not True
            or planar["full_interval"]["intervals"] <= 0):
        raise ValueError("complete two-dimensional planar motion unavailable")
    for key in ("velocity_source", "scope", "weighting", "steady_eligibility"):
        if not isinstance(planar[key], str) or not planar[key]:
            raise ValueError("planar measurement semantics missing")
    for name in ("full_interval", "steady", "stationary", "stationary_steady"):
        validate_planar_pool(planar[name], 4000 * rows, require_available=name == "full_interval")
    stationary = planar["stationary"]
    count(stationary["samples"], samples)
    count(stationary["runs"], stationary["samples"])
    for key in ("reference", "origin"):
        if not isinstance(stationary[key], str) or not stationary[key]:
            raise ValueError("stationary planar reference missing")
    for key in ("endpoint_displacement_m", "max_excursion_m"):
        signal = stationary[key]
        if signal["count"] != stationary["runs"]:
            raise ValueError("stationary planar run counts differ")
        for statistic in ("mean", "rms", "mean_abs", "max_abs"):
            if signal["count"]:
                finite(signal[statistic], nonnegative=True)
            elif signal[statistic] is not None:
                raise ValueError("unavailable stationary run measurement must remain null")
    scaled = value["actuation"]["scaled_nominal_envelope"]
    if (scaled["available"] is not True or scaled["sample_count"] != samples
            or value["actuation"]["sample_count"] != samples):
        raise ValueError("scaled nominal envelope unavailable")
    bounds = scaled["active_bound_samples"]
    if len(bounds) != 6 or any(type(count) is not int or count != samples for count in bounds):
        raise ValueError("scaled envelope channel counts differ")
    for name in ("applied_at_bound_fraction", "requested_outside_bounds_fraction", "applied_outside_bounds_fraction"):
        fractions = scaled[name]
        if len(fractions) != 6 or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
            raise ValueError("scaled envelope fractions differ")


def npy_header(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY trace member")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise ValueError("unsupported NPY trace version")
    size = struct.unpack("<H" if version[0] == 1 else "<I", stream.read(2 if version[0] == 1 else 4))[0]
    if size > 65536:
        raise ValueError("NPY header exceeds schema budget")
    value = ast.literal_eval(stream.read(size).decode("utf-8" if version[0] == 3 else "latin1").strip())
    if set(value) != {"descr", "fortran_order", "shape"} or value["fortran_order"] is not False:
        raise ValueError("trace NPY must be a plain C-order array")
    match = re.fullmatch(r"([<>=|])([fibuU])(\d+)", value["descr"])
    if not match or not isinstance(value["shape"], tuple) or any(type(v) is not int or v < 0 for v in value["shape"]):
        raise ValueError("unsupported trace NPY dtype or shape")
    width = int(match[3])
    if (width <= 0 or (match[2] == "f" and width not in (4, 8))
            or (match[2] == "b" and width != 1)
            or (match[2] in "iu" and width not in (1, 2, 4, 8))):
        raise ValueError("unsupported trace NPY dtype width")
    size = int(match[3]) * (4 if match[2] == "U" else 1)
    return value, match[2], math.prod(value["shape"]) * size


def validate_trace(path, manifest, checkpoint, seed, control_report):
    trace = control_report["trace"]
    if (Path(trace["path"]).resolve() != path.resolve() or trace["sha256"] != control.file_sha(path)
            or trace["steps"] != 4001):
        raise ValueError("trace receipt identity differs")
    labels = control_report["environment_provenance"]["evaluation_groups"]
    if len(labels) != 400 or Counter(labels) != Counter({name: 8 for name in manifest["inputs"]["cases"]}):
        raise ValueError("evaluation provenance does not cover all fifty eight-row groups")
    rows = sorted(index for name in set(labels) for index in [i for i, value in enumerate(labels) if value == name][:2])
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or any(not re.fullmatch(r"[a-z][a-z0-9_]*\.npy", name) for name in members):
            raise ValueError("trace archive members differ")
        fields = {name[:-4] for name in members}
        if (not (set(TRACE_SHAPES) | {"metadata_json", "row_indices"}).issubset(fields)
                or fields - {"metadata_json", "row_indices"} != set(trace["fields"])
                or fields - (set(TRACE_SHAPES) | {"metadata_json", "row_indices", "command_request"})):
            raise ValueError("trace is missing declared physical or scaled nominal fields")
        metadata = None
        time_values, episodes, dones = None, None, None
        for name in members:
            field = name[:-4]
            with archive.open(name) as stream:
                header, kind, size = npy_header(stream)
                if field == "metadata_json":
                    if header["shape"] != () or kind != "U" or size > 262144:
                        raise ValueError("trace metadata schema differs")
                    metadata = json.loads(stream.read(size).decode("utf-32-be" if header["descr"].startswith(">") else "utf-32-le").rstrip("\x00"))
                    consumed = size
                elif field == "row_indices":
                    if header["shape"] != (100,) or header["descr"] not in ("<i8", "=i8"):
                        raise ValueError("trace row indices schema differs")
                    encoded = stream.read(size)
                    if list(struct.unpack("<100q", encoded)) != rows:
                        raise ValueError("trace does not select the first two declared replicas")
                    consumed = len(encoded)
                else:
                    suffix = TRACE_SHAPES.get(field, (3,))
                    if header["shape"] != (4001, 100, *suffix):
                        raise ValueError(f"trace shape differs: {field}")
                    expected_kind = "b" if field in ("done", "failure", "success") else "i" if field == "episode_id" else "f"
                    if kind != expected_kind:
                        raise ValueError(f"trace dtype differs: {field}")
                    if field == "episode_id" and header["descr"] not in ("<i8", "=i8"):
                        raise ValueError("trace episode IDs must use signed int64")
                    consumed = 0
                    captured = bytearray() if field in {"time_s", "episode_id", "done"} else None
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        consumed += len(block)
                        if kind == "f":
                            values = array("f" if header["descr"][-1] == "4" else "d")
                            values.frombytes(block)
                            if ((header["descr"].startswith(">") and sys.byteorder == "little")
                                    or (header["descr"].startswith("<") and sys.byteorder == "big")):
                                values.byteswap()
                            if not all(math.isfinite(value) for value in values):
                                raise ValueError(f"nonfinite physical trace sample: {field}")
                        elif kind == "b" and any(value > 1 for value in block):
                            raise ValueError("trace boolean payload differs")
                        if captured is not None:
                            captured.extend(block)
                    if field == "time_s":
                        time_values = array("f" if header["descr"][-1] == "4" else "d")
                        time_values.frombytes(captured)
                        if ((header["descr"].startswith(">") and sys.byteorder == "little")
                                or (header["descr"].startswith("<") and sys.byteorder == "big")):
                            time_values.byteswap()
                    elif field == "episode_id":
                        episodes = array("q")
                        episodes.frombytes(captured)
                    elif field == "done":
                        dones = captured
                if consumed != size or stream.read(1):
                    raise ValueError("trace NPY payload length differs")
        expected = {"checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": 1200,
            "seed": seed, "steps": 4001, "policy_dt_s": .01, "sampling_hz": 100.,
            "control_sha256": checkpoint["control_sha256"], "row_indices": rows,
            "group_labels": [labels[index] for index in rows]}
        if metadata is None or any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("trace metadata identity or declared replicas differ")
        if any(trace.get(key) != value for key, value in expected.items()):
            raise ValueError("trace report and archive metadata differ")
        if any(value != 0 for value in episodes[:100]) or any(value <= 0 for value in time_values):
            raise ValueError("trace episode origin or physical time differs")
        for index in range(100, len(time_values)):
            previous = index - 100
            if (episodes[index] != episodes[previous] + int(dones[previous])
                    or (episodes[index] == episodes[previous] and time_values[index] <= time_values[previous])):
                raise ValueError("trace episode continuity or within-episode time differs")


def verify_outputs(directory, manifest, variant, seed):
    checkpoint = manifest["inputs"]["checkpoints"][variant]
    expected = {"checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": 1200,
                "seed": seed, "steps": 4001}
    cases = manifest["inputs"]["cases"]
    result = {}
    controls = {}
    for case in cases:
        path = directory / f"{case}.json"
        report = read(path)
        if (any(report.get(key) != value for key, value in expected.items())
                or report.get("num_envs") != 8 or report.get("transitions") != 32008
                or report.get("environment") != manifest["inputs"]["environments"][variant][case]):
            raise ValueError(f"case report identity or row coverage differs: {case}")
        validate_metrics(report["control"], 8)
        controls[case] = report["control"]
        result[case] = control.artifact(path, manifest["output_root"])
    report = read(directory / "control.json")
    if (any(report.get(key) != value for key, value in expected.items())
            or set(report["groups"]) != set(cases) or report["groups"] != controls):
        raise ValueError("suite identity or complete group coverage differs")
    if report["environment_provenance"]["identity"] != next(iter(manifest["inputs"]["snapshots"].values()))["sha256"]:
        raise ValueError("evaluation training snapshot provenance differs")
    validate_metrics(report["control"], 400)
    validate_trace(directory / "trace.npz", manifest, checkpoint, seed, report)
    result["control"] = control.artifact(directory / "control.json", manifest["output_root"])
    result["trace"] = control.artifact(directory / "trace.npz", manifest["output_root"])
    return result


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = validate(manifest_path)
    root = Path(manifest["output_root"])
    summary = {"status": "waiting", "started_at": control.now(), "manifest_sha256": manifest["sha256"],
               "results": {}, "paired_diagnostic_retest": True, "formal_architecture_selection": False}

    def publish(status=None, **values):
        if status is not None:
            summary["status"] = status
        summary.update(values, updated_at=control.now())
        write(root / "summary.json", summary, replace=True)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = manifest["source_root"] + "/src" + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with (root / ".diagnostic.lock").open("a") as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with acquire_resources(manifest, args, publish) as dependencies:
            if dependencies is None:
                return summary
            if validate(manifest_path) != manifest:
                raise ValueError("sealed diagnostic manifest changed while queued")
            publish("evaluating", dependencies=dependencies, blockers=[], active=None)
            for variant in manifest["inputs"]["variants"]:
                for seed in PROTOCOL["seeds"]:
                    key = f"{variant}/seed_{seed}"
                    checkpoint = manifest["inputs"]["checkpoints"][variant]
                    identity = {"manifest_sha256": manifest["sha256"], "variant": variant,
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "seed": seed}
                    job = root / "evaluations" / variant / f"seed_{seed}"
                    previous = control.previous_success(job, identity, root)
                    if previous:
                        verify_outputs(control.inside(root, previous["directory"]), manifest, variant, seed)
                        summary["results"][key] = previous
                        continue
                    # Each invocation rechecks sealed inputs before launching another suite.
                    validate(manifest_path)
                    if active_workers(manifest):
                        publish("blocked", reason="a verified worker remains alive", active=None)
                        return summary
                    directory = transfer.next_attempt(job)
                    configs = [manifest["inputs"]["configs"][f"configs/{variant}.eval.{case}.json"]["path"] for case in manifest["inputs"]["cases"]]
                    outputs = [directory / f"{case}.json" for case in manifest["inputs"]["cases"]]
                    command = transfer.command(Path(manifest["source_root"]), "evaluate-suite",
                        "--checkpoint", checkpoint["checkpoint"], "--configs", *configs, "--outputs", *outputs,
                        "--steps", 4001, "--seed", seed, "--device", manifest["device"],
                        "--control-output", directory / "control.json", "--trace-output", directory / "trace.npz",
                        "--settle-steps", 200, "--min-steady-samples", 200, "--trace-replicas", 2)
                    receipt = {"identity": identity, "status": "failed", "started_at": control.now(),
                               "directory": str(directory.relative_to(root)), "dependencies": dependencies}
                    publish("evaluating", active={"variant": variant, "seed": seed, "directory": str(directory)})
                    try:
                        worker = control.run_worker(command, directory, environment, args.worker_timeout_seconds, publish)
                        receipt["worker"] = worker
                        if worker["returncode"] != 0 or worker["timed_out"]:
                            raise RuntimeError("diagnostic evaluation failed or timed out")
                        if active_workers(manifest):
                            raise RuntimeError("worker completion is not terminal")
                        artifacts = verify_outputs(directory, manifest, variant, seed)
                        validate(manifest_path)
                        receipt.update(status="completed", artifacts=artifacts)
                    except Exception as error:
                        receipt["error"] = f"{type(error).__name__}: {error}"
                    except BaseException:
                        receipt.update(error="diagnostic campaign interrupted", finished_at=control.now())
                        write(directory / "receipt.json", receipt)
                        publish("interrupted", active=None)
                        raise
                    receipt["finished_at"] = control.now()
                    write(directory / "receipt.json", receipt)
                    summary["results"][key] = receipt
                    publish(active=None)
                    if active_workers(manifest):
                        publish("blocked", reason="worker completion is not terminal", active=None)
                        return summary
            publish("completed" if all(value["status"] == "completed" for value in summary["results"].values()) else "incomplete", active=None)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    preparer = commands.add_parser("prepare")
    for name in ("study-root", "output-root", "source-root", "resource-lock", "transfer-receipt", "curriculum-receipt"):
        preparer.add_argument("--" + name, type=Path, required=True)
    for name in ("transfer-campaign-sha256", "curriculum-campaign-sha256"):
        preparer.add_argument("--" + name, required=True)
    preparer.add_argument("--source-identity", type=Path)
    preparer.add_argument("--training-seed", type=int, default=1101)
    preparer.add_argument("--device", default="cuda:0")
    for name in ("validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            sub.add_argument("--max-wait-seconds", type=float, default=604800.)
            sub.add_argument("--worker-timeout-seconds", type=float, default=3600.)
            sub.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        result = prepare(args)
        print(json.dumps({"manifest": str(args.output_root.resolve() / "manifest.json"), "sha256": result["sha256"]}))
        return 0
    if args.operation == "validate":
        result = validate(args.manifest)
        print(json.dumps({"status": "validated", "sha256": result["sha256"]}))
        return 0
    if (not math.isfinite(args.max_wait_seconds) or args.max_wait_seconds < 0
            or not math.isfinite(args.worker_timeout_seconds) or args.worker_timeout_seconds <= 0
            or not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 30):
        parser.error("finite bounded waiting and positive worker budgets are required")
    old_handlers = {}
    def interrupted(*_):
        raise KeyboardInterrupt
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, interrupted)
        result = run(args)
        print(json.dumps({"status": result["status"], "summary": str(args.manifest.parent / "summary.json")}))
        return 0 if result["status"] == "completed" else 2
    except KeyboardInterrupt:
        return 130
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())

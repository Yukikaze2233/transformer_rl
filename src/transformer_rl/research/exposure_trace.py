"""Strict, bounded replay of complete independent physical evaluation traces.

Dimensions are supplied by the authorized evaluation definition. Parser limits
are resource ceilings, never an authorization to expand a declared evaluation.
No simulator, checkpoint deserializer, or pickle loader is used here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import struct
import zipfile

import numpy as np
import torch

from .control_metrics import ControlMetrics
from .evaluation import _MetricAccumulator
from .episode_outcomes import EpisodeOutcomeStatistics, TRACE_FIELDS, TRACE_METADATA_KEY, trace_metadata
from .stability import EpisodeSignalStatistics


MAX_STEPS = 1_000_000
MAX_ROWS = 8192
MAX_UNCOMPRESSED_BYTES = 8 * 1024**3
_SHAPES = {
    "time_s": (), "command_reference": (3,), "actual": (3,),
    "position_xy": (2,), "tilt": (), "leg_target": (4,), "wheel_target": (2,),
    "motor_position": (6,), "motor_velocity": (6,), "motor_effort": (6,),
    "requested_motor_effort": (6,), "effort_bounds": (6, 2),
    "scaled_nominal_requested_motor_effort": (6,), "scaled_nominal_effort_bounds": (6, 2),
    "failure": (), "success": (), "done": (), "episode_id": (),
    "pre_inference_episode_age": (), "raw_policy_mean": (6,), "issued_action": (6,),
}
_BOOL = {"failure", "success", "done"}
_INTEGER = {"episode_id", "pre_inference_episode_age"}
_SPECIAL = {"metadata_json", "row_indices"}
_ARTIFACT_KEYS = {"path", "sha256", "fields", "rows"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate trace metadata field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))
    _canonical(value)
    return value


def _path(value):
    path = Path(value).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink trace path")
    _require(path == path.resolve(strict=True), "trace path must be canonical")
    _require(stat.S_ISREG(path.stat().st_mode), "trace must be a regular file")
    return path


def _identity(path):
    item = path.stat()
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns


def _hash(stream):
    stream.seek(0)
    result = hashlib.file_digest(stream, "sha256").hexdigest()
    stream.seek(0)
    return result


def _archive_bounds(source, archive, size):
    """Reject ZIP polyglot prefixes, comments and ignored trailing payloads."""
    _require(size >= 22, "truncated ZIP end record")
    source.seek(size - 22)
    end = source.read(22)
    signature, disk, central_disk, disk_entries, entries, central_size, central_offset, comment = struct.unpack(
        "<4s4H2IH", end)
    _require(signature == b"PK\x05\x06" and disk == central_disk == comment == 0,
             "ZIP end record or trailing payload differs")
    central_end = size - 22
    if disk_entries == 65535 or entries == 65535 or central_size == 2**32 - 1 or central_offset == 2**32 - 1:
        _require(size >= 98, "truncated ZIP64 end records")
        source.seek(size - 42)
        locator, locator_disk, offset, disks = struct.unpack("<4sIQI", source.read(20))
        _require(locator == b"PK\x06\x07" and locator_disk == 0 and disks == 1
                 and offset + 56 == size - 42, "ZIP64 locator or extended end record differs")
        source.seek(offset)
        record = struct.unpack("<4sQ2H2I4Q", source.read(56))
        _require(record[0] == b"PK\x06\x06" and record[1] == 44 and record[4] == record[5] == 0,
                 "ZIP64 end record differs")
        disk_entries, entries, central_size, central_offset = record[6:]
        central_end = offset
    _require(disk_entries == entries == len(archive.infolist())
             and central_offset + central_size == central_end
             and archive.start_dir == central_offset
             and min(info.header_offset for info in archive.infolist()) == 0,
             "ZIP member, prefix or central-directory boundaries differ")
    ordered = sorted(archive.infolist(), key=lambda info: info.header_offset)
    for index, info in enumerate(ordered):
        source.seek(info.header_offset)
        header = struct.unpack("<4s5H3I2H", source.read(30))
        _require(header[0] == b"PK\x03\x04" and header[2] == info.flag_bits
                 and header[3] == info.compress_type and source.read(header[9]) == info.filename.encode("ascii"),
                 "ZIP local and central member headers differ")
        payload_end = info.header_offset + 30 + header[9] + header[10] + info.compress_size
        next_offset = ordered[index + 1].header_offset if index + 1 < len(ordered) else central_offset
        trailing = next_offset - payload_end
        if info.flag_bits & 8:
            _require(trailing in (12, 16, 20, 24), "ZIP data descriptor or member boundaries differ")
            source.seek(payload_end)
            raw = source.read(trailing)
            if trailing in (16, 24):
                _require(raw[:4] == b"PK\x07\x08", "ZIP data descriptor signature differs")
                raw = raw[4:]
            descriptor = struct.unpack("<III" if len(raw) == 12 else "<IQQ", raw)
            _require(descriptor == (info.CRC, info.compress_size, info.file_size), "ZIP data descriptor differs")
        else:
            _require(trailing == 0 and header[6] == info.CRC,
                     "ZIP member has unclaimed payload or inconsistent CRC")


def _header(stream):
    """Parse only plain C-order NPY headers, before any array allocation."""
    _require(stream.read(6) == b"\x93NUMPY", "invalid NPY signature")
    version = stream.read(2)
    _require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "unsupported NPY version")
    length_size = 2 if version[0] == 1 else 4
    encoded = stream.read(length_size)
    _require(len(encoded) == length_size, "truncated NPY header length")
    length = int.from_bytes(encoded, "little")
    _require(0 < length <= 65536, "NPY header size exceeds parser limit")
    raw = stream.read(length)
    _require(len(raw) == length, "truncated NPY header")
    try:
        value = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1").strip())
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise ValueError("invalid NPY header") from error
    _require(type(value) is dict and set(value) == {"descr", "fortran_order", "shape"}
             and value["fortran_order"] is False and type(value["shape"]) is tuple
             and all(type(n) is int and n >= 0 for n in value["shape"])
             and type(value["descr"]) is str, "NPY must be a plain C-order array")
    match = re.fullmatch(r"([<>=|])([fibU])(\d+)", value["descr"])
    _require(match is not None, "unsupported NPY dtype")
    order, kind, width = match[1], match[2], int(match[3])
    _require(width > 0 and (kind != "f" or width in (4, 8))
             and (kind != "i" or width == 8) and (kind != "b" or width == 1)
             and (order != "|" or width == 1), "unsupported NPY dtype width or byte order")
    dtype = np.dtype(value["descr"])
    size = math.prod(value["shape"]) * dtype.itemsize
    _require(size <= MAX_UNCOMPRESSED_BYTES, "NPY payload exceeds parser limit")
    return value, dtype, size, 8 + length_size + length


def _scan(stream, dtype, size, kind):
    consumed = 0
    for raw in iter(lambda: stream.read(1024 * 1024), b""):
        consumed += len(raw)
        _require(consumed <= size and len(raw) % dtype.itemsize == 0, "extra or unaligned NPY payload")
        if kind == "b":
            _require(all(byte <= 1 for byte in raw), "invalid boolean trace payload")
        else:
            values = np.frombuffer(raw, dtype=dtype)
            _require(bool(np.isfinite(values).all()), "nonfinite trace sample")
            if kind == "i":
                _require(bool((values >= 0).all()), "negative episode identifier or inference age")
    _require(consumed == size, "truncated NPY payload")


def _tensor(raw, dtype, shape):
    """Own native-endian CPU arrays without changing the recorded precision."""
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    array = array.astype(dtype.newbyteorder("="), copy=True)
    return torch.from_numpy(array)


def _arguments(expected, steps, rows, history_length, action_bounds, settle_steps, min_steady_samples):
    for name, value, low, high in (("steps", steps, 1, MAX_STEPS), ("rows", rows, 1, MAX_ROWS),
                                  ("history_length", history_length, 1, MAX_STEPS),
                                  ("settle_steps", settle_steps, 0, MAX_STEPS),
                                  ("min_steady_samples", min_steady_samples, 1, MAX_STEPS)):
        _require(type(value) is int and low <= value <= high, f"invalid declared {name}")
    _require(type(action_bounds) in (list, tuple) and len(action_bounds) == 6
             and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in action_bounds),
             "six finite positive action bounds required")
    _require(type(expected) is dict, "expected trace metadata requires a dictionary")
    _canonical(expected)
    required = {"checkpoint_sha256", "checkpoint_update", "seed", "steps", "policy_dt_s",
                "sampling_hz", "control_sha256", "row_indices", "group_labels"}
    _require(required <= set(expected) and not set(expected) & _ARTIFACT_KEYS,
             "authoritative trace metadata coverage differs")
    for name in ("checkpoint_sha256", "control_sha256"):
        _require(type(expected[name]) is str and re.fullmatch(r"[0-9a-f]{64}", expected[name]),
                 f"invalid expected {name}")
    _require(type(expected["checkpoint_update"]) is int and expected["checkpoint_update"] >= 0
             and type(expected["seed"]) is int and 0 <= expected["seed"] < 2**32,
             "invalid expected checkpoint clock or evaluation seed")
    dt, hz = expected["policy_dt_s"], expected["sampling_hz"]
    _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0
             and type(hz) in (int, float) and math.isfinite(hz)
             and math.isclose(hz, 1 / dt, rel_tol=1e-12, abs_tol=0.), "invalid physical sampling interval")
    _require(type(expected["steps"]) is int and expected["steps"] == steps
             and _canonical(expected["row_indices"]) == _canonical(list(range(rows))),
             "declared complete row or step coverage differs")
    labels = expected["group_labels"]
    _require(type(labels) is list and len(labels) == rows
             and all(type(label) is str and label for label in labels), "invalid declared group labels")
    if "history_length" in expected:
        _require(type(expected["history_length"]) is int and expected["history_length"] == history_length,
                 "expected history length differs")
    return float(dt), labels


def _evaluation_fields(expected):
    keys = ("evaluation_metric_names", "evaluation_signal_names")
    present = [key in expected for key in keys]
    _require(not any(present) or all(present), "evaluation field declarations must be paired")
    if not all(present):
        return {}, None
    names = {}
    for key in keys:
        value = expected[key]
        _require(type(value) is list and all(type(name) is str and re.fullmatch(
            r"[a-z][a-z0-9_]*", name) for name in value) and value == sorted(set(value)),
            "evaluation field names must be unique sorted safe identifiers")
        names[key] = value
    fields = {"eval_reward": (), "eval_episode_success": (), "eval_signal_time": ()}
    fields.update({"eval_metric_" + name: () for name in names[keys[0]]})
    fields.update({"eval_signal_" + name: () for name in names[keys[1]]})
    return fields, names


def _episode_fields(expected, trace, rows):
    _require(type(trace) is dict, "trace report requires a dictionary")
    declaration = trace.get(TRACE_METADATA_KEY)
    if declaration is None:
        _require(TRACE_METADATA_KEY not in trace and TRACE_METADATA_KEY not in expected,
                 "missing declared episode outcome trace protocol")
        available = False
    else:
        _require(type(declaration) is dict and type(declaration.get("available")) is bool
                 and _canonical(declaration) == _canonical(trace_metadata(declaration["available"])),
                 "episode outcome trace protocol differs")
        available = declaration["available"]
    contract = expected.get("episode_outcome_contract")
    if contract is not None:
        _require(available and type(contract) is dict
                 and set(contract) == {"episode_horizon_ticks", "survival_applicable"},
                 "explicit available episode outcome contract required")
        for name, dtype in (("episode_horizon_ticks", int), ("survival_applicable", bool)):
            values = contract[name]
            _require(type(values) is list and len(values) == rows and all(type(v) is dtype for v in values)
                     and (dtype is bool or all(0 < v < 2**63 for v in values)),
                     "invalid declared episode outcome row contract")
    return {name: () for name in TRACE_FIELDS} if available else {}, available, contract


def verify_trace_archive(path, trace, expected, *, steps, rows, history_length,
                         action_bounds, settle_steps, min_steady_samples):
    """Validate every payload and replay all authorized rows, including failures.

    The inference age is checked against the preceding terminal flag, rather
    than inferred from an already advanced physical sample timestamp. JSON
    report values are not taken as evidence of measured control performance.
    """
    from .history_control import HistoryControlStatistics

    dt, labels = _arguments(expected, steps, rows, history_length, action_bounds,
                            settle_steps, min_steady_samples)
    evaluation_fields, evaluation_names = _evaluation_fields(expected)
    episode_fields, episode_available, episode_contract = _episode_fields(expected, trace, rows)
    episode_contract_tensors = ({name: torch.tensor(values, dtype=torch.bool if name == "survival_applicable" else torch.int64)
                                for name, values in episode_contract.items()} if episode_contract is not None else {})
    # Row contracts are authorization inputs, not evidence self-declared by the
    # trace producer. All ordinary provenance keys remain archive-bound.
    expected_metadata = {k: v for k, v in expected.items() if k != "episode_outcome_contract"}
    shapes = {**_SHAPES, **evaluation_fields, **episode_fields}
    path = _path(path)
    identity = _identity(path)
    _require(type(trace) is dict and _path(trace["path"]) == path, "trace report path differs")
    _require(_canonical(trace.get("rows")) == _canonical(list(range(rows)))
             and all(_canonical(trace.get(k)) == _canonical(v) for k, v in expected_metadata.items()),
             "trace report provenance differs")
    required = set(shapes) | _SPECIAL
    with path.open("rb") as source:
        initial_sha = _hash(source)
        _require(type(trace.get("sha256")) is str and trace["sha256"] == initial_sha, "trace actual SHA differs")
        try:
            with zipfile.ZipFile(source) as archive:
                members = archive.namelist()
                _require(len(members) == len(set(members)) and all(
                    re.fullmatch(r"[a-z][a-z0-9_]*\.npy", name) for name in members), "trace ZIP members differ")
                _require(bool(members), "empty trace ZIP")
                _archive_bounds(source, archive, identity[2])
                fields = {name[:-4] for name in members}
                declared = trace.get("fields")
                _require(type(declared) is list and all(type(n) is str for n in declared)
                         and len(declared) == len(set(declared)) and required <= fields
                         and fields <= required | {"command_request"}
                         and set(declared) == fields - _SPECIAL, "complete trace field coverage differs")
                total_size = sum(info.file_size for info in archive.infolist())
                _require(0 < total_size <= MAX_UNCOMPRESSED_BYTES, "trace uncompressed size exceeds parser limit")
                layouts, metadata = {}, None
                for name in members:
                    field, info = name[:-4], archive.getinfo(name)
                    mode = info.external_attr >> 16
                    _require(not info.flag_bits & 1 and info.compress_type in
                             (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                             and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), "unsupported trace ZIP encoding or file type")
                    with archive.open(name) as stream:
                        header, dtype, size, prefix = _header(stream)
                        _require(info.file_size == prefix + size, "NPY member declared payload length differs")
                        if field == "metadata_json":
                            _require(header["shape"] == () and dtype.kind == "U" and size <= 262144,
                                     "trace metadata NPY schema differs")
                            raw = stream.read(size)
                            _require(len(raw) == size, "truncated trace metadata")
                            metadata = _json(np.frombuffer(raw, dtype=dtype).item())
                        elif field == "row_indices":
                            _require(header["shape"] == (rows,) and dtype.kind == "i" and dtype.itemsize == 8,
                                     "trace row indices NPY schema differs")
                            raw = stream.read(size)
                            _require(len(raw) == size and np.frombuffer(raw, dtype=dtype).tolist() == list(range(rows)),
                                     "trace row indices differ")
                        else:
                            suffix = shapes.get(field, (3,))
                            _require(header["shape"] == (steps, rows, *suffix), f"trace shape differs: {field}")
                            episode_kind = TRACE_FIELDS.get(field)
                            kind = ("b" if field in _BOOL or field == "eval_episode_success" or episode_kind == "bool"
                                    else "i" if field in _INTEGER or episode_kind == "int64" else "f")
                            _require(dtype.kind == kind, f"trace dtype differs: {field}")
                            _require(field not in ("time_s", "eval_signal_time") or dtype.itemsize == 8,
                                     "physical signal time requires float64")
                            _scan(stream, dtype, size, kind)
                            layouts[field] = {"dtype": dtype, "shape": (rows, *suffix),
                                              "tick_bytes": rows * math.prod(suffix) * dtype.itemsize}
                        _require(not stream.read(1), "extra NPY payload bytes")
                _require(type(metadata) is dict and all(
                    _canonical(metadata.get(k)) == _canonical(v) for k, v in expected_metadata.items()), "trace archive metadata differs")
                _require(_canonical(metadata) == _canonical({k: v for k, v in trace.items() if k not in _ARTIFACT_KEYS}),
                         "trace report and archive metadata differ")

                def metrics(count):
                    result = {"control": ControlMetrics(count, dt, settle_steps=settle_steps,
                                                       min_steady_samples=min_steady_samples),
                            "history_control": HistoryControlStatistics(count, history_length, dt,
                                settle_steps=settle_steps,
                                min_steady_samples=min_steady_samples),
                            "episode_outcomes": EpisodeOutcomeStatistics(count, dt)}
                    if evaluation_names is not None:
                        result["evaluation"] = {"reward": _MetricAccumulator(),
                            "metrics": {name: _MetricAccumulator() for name in
                                        evaluation_names["evaluation_metric_names"]},
                            "statistics": EpisodeSignalStatistics(count, settle_steps=settle_steps,
                                                                  min_steady_samples=min_steady_samples),
                            "completed": 0, "successes": 0, "failures": 0}
                    return result

                def evaluation_update(state, values, done):
                    state["reward"].add(values["eval_reward"])
                    for name, accumulator in state["metrics"].items():
                        accumulator.add(values["eval_metric_" + name])
                    signals = {name: values["eval_signal_" + name] for name in
                               evaluation_names["evaluation_signal_names"]}
                    state["statistics"].update(signals, values["eval_signal_time"] if signals else None, done)
                    success = values["eval_episode_success"]
                    state["completed"] += int(done.sum())
                    state["successes"] += int(success.sum())
                    state["failures"] += int((done & ~success).sum())

                overall = metrics(rows)
                groups = {label: {"rows": torch.tensor([i for i, value in enumerate(labels) if value == label]),
                                  **metrics(labels.count(label))} for label in sorted(set(labels))}
                streams = {}
                try:
                    for field in layouts:
                        streams[field] = archive.open(field + ".npy")
                        _header(streams[field])
                    previous = None
                    for _ in range(steps):
                        values = {}
                        for field, layout in layouts.items():
                            raw = streams[field].read(layout["tick_bytes"])
                            _require(len(raw) == layout["tick_bytes"], "truncated trace replay tick")
                            values[field] = _tensor(raw, layout["dtype"], layout["shape"])
                        done, age, episode, time = (values[k] for k in
                            ("done", "pre_inference_episode_age", "episode_id", "time_s"))
                        if previous is None:
                            _require(bool((age == 0).all() and (episode == 0).all()), "trace age or episode origin differs")
                            actual_dt = time
                        else:
                            expected_age = torch.where(previous["done"], 0, previous["pre_inference_episode_age"] + 1)
                            _require(torch.equal(age, expected_age), "trace pre-inference age/reset continuity differs")
                            _require(torch.equal(episode, previous["episode_id"] + previous["done"].to(torch.int64)),
                                     "trace episode continuity differs")
                            actual_dt = torch.where(previous["done"], time, time - previous["time_s"])
                        _require(bool((time > 0).all()) and torch.allclose(actual_dt, torch.full_like(time, dt), rtol=0., atol=5e-6),
                                 "trace physical policy tick clock differs")
                        _require(not bool((values["failure"] & values["success"]).any())
                                 and not bool(((values["failure"] | values["success"]) & ~done).any()),
                                 "trace terminal flags disagree with done")
                        if evaluation_names is not None:
                            _require(not bool((values["eval_episode_success"] & ~done).any())
                                     and not bool((values["eval_episode_success"] & values["failure"]).any()),
                                     "evaluation success disagrees with physical terminal flags")
                            _require(torch.equal(values["eval_signal_time"], time),
                                     "evaluation signal time differs from PRE-reset physical time")
                        outcome = None
                        if episode_available:
                            outcome = {name.removeprefix("outcome_"): values[name] for name in episode_fields
                                       if name not in ("outcome_height", "outcome_tilt")}
                            _require(torch.equal(outcome["episode_ticks"], age + 1),
                                     "episode outcome ticks differ from pre-inference age")
                            if previous is not None:
                                active = ~previous["done"]
                                for name in ("episode_horizon_ticks", "survival_applicable"):
                                    _require(torch.equal(outcome[name][active], previous["outcome_" + name][active]),
                                             "episode outcome protocol changed within an episode")
                            _require(torch.equal(outcome["environment_failure"], values["failure"])
                                     and torch.equal(outcome["task_success"], values["success"]),
                                     "episode outcome terminal flags differ from physical evidence")
                            _require(not bool(((outcome["boundary"] | outcome["blocked"]) & ~done).any()),
                                     "episode outcome collection cut requires done")
                            for name, physical in (("height", values["actual"][:, 2]), ("tilt", values["tilt"])):
                                recorded = values["outcome_" + name]
                                _require(recorded.dtype == physical.dtype and torch.equal(recorded, physical),
                                         f"episode outcome {name} differs from PRE-reset physical evidence")
                            if episode_contract is not None:
                                for name, declared in episode_contract_tensors.items():
                                    _require(torch.equal(outcome[name], declared),
                                             f"episode outcome {name} differs from authorized case contract")
                        overall["episode_outcomes"].update(outcome, values.get("outcome_height"),
                                                          values.get("outcome_tilt"), done)
                        packet = {key: value for key, value in values.items() if key not in
                                  {"episode_id", "done", "pre_inference_episode_age", "raw_policy_mean", "issued_action"}
                                  and key not in evaluation_fields and key not in episode_fields}
                        if "command_request" in packet:
                            packet["request"] = packet.pop("command_request")
                        raw_mean, issued = values["raw_policy_mean"], values["issued_action"]
                        bounds = torch.tensor(action_bounds, dtype=raw_mean.dtype)
                        _require(raw_mean.dtype == issued.dtype and torch.equal(issued, raw_mean.clamp(-bounds, bounds)),
                                 "issued action differs from declared raw mean clamp")
                        overall["control"].update(packet, done)
                        overall["history_control"].update(packet, done, age, raw_mean, issued)
                        if evaluation_names is not None:
                            evaluation_update(overall["evaluation"], values, done)
                        for group in groups.values():
                            indices = group["rows"]
                            grouped = {key: value[indices] for key, value in packet.items()}
                            group["control"].update(grouped, done[indices])
                            group["history_control"].update(grouped, done[indices], age[indices], raw_mean[indices], issued[indices])
                            if evaluation_names is not None:
                                evaluation_update(group["evaluation"], {key: value[indices] for key, value in values.items()}, done[indices])
                            group["episode_outcomes"].update(
                                {key: value[indices] for key, value in outcome.items()} if outcome is not None else None,
                                values["outcome_height"][indices] if episode_available else None,
                                values["outcome_tilt"][indices] if episode_available else None, done[indices])
                        previous = values
                    _require(all(not stream.read(1) for stream in streams.values()), "extra trace replay payload")
                finally:
                    for stream in streams.values():
                        stream.close()
        except (zipfile.BadZipFile, EOFError, UnicodeError, struct.error) as error:
            raise ValueError("invalid or corrupt trace archive") from error
        final_sha = _hash(source)
    _require(_path(path) == path and _identity(path) == identity and final_sha == initial_sha,
             "trace changed during verification")
    def report(state):
        result = {"control": state["control"].report(), "history_control": state["history_control"].report(),
                  "episode_outcomes": state["episode_outcomes"].report()}
        if evaluation_names is not None:
            item = state["evaluation"]
            result.update(metrics={name: value.report() for name, value in item["metrics"].items()},
                reward_mean=item["reward"].report()["mean"], stability=item["statistics"].report(),
                completed_episodes=item["completed"], failed_episodes=item["failures"],
                success_metric_available=True,
                success_rate=item["successes"] / item["completed"] if item["completed"] else None)
        return result

    return {"trace_validation": {"path": str(path), "sha256": final_sha, "bytes": identity[2],
        "steps": steps, "rows": rows, "fields": sorted(fields - _SPECIAL), "metadata": metadata,
        "all_physical_arrays_finite": True, "complete_payload_and_crc_checked": True,
        "episode_done_time_consistent": True, "pre_inference_history_age_checked": True,
        "episode_outcome_evidence": "explicit PRE-reset replay" if episode_available else "unavailable; no inference from legacy done",
        "episode_outcome_contract_checked": episode_contract is not None,
        "replay": "all authorized rows; original recorded precision; no outcome subsampling",
        "hardware_verified": False},
        **report(overall), "groups": {label: report(group) for label, group in groups.items()}}

"""Read-only physical evidence for the original fifty-case retention pilot.

The caller owns process termination and leases. This verifier never starts an
environment, deserializes a learner, writes a report, or changes qualification.
"""
from __future__ import annotations

from array import array
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import types
import zipfile


STEPS, CASES, REPLICAS = 4001, 50, 8
TRACE_SHAPES = {
    "time_s": (), "command_reference": (3,), "actual": (3,),
    "position_xy": (2,), "tilt": (), "leg_target": (4,), "wheel_target": (2,),
    "motor_position": (6,), "motor_velocity": (6,), "motor_effort": (6,),
    "requested_motor_effort": (6,), "effort_bounds": (6, 2),
    "scaled_nominal_requested_motor_effort": (6,), "scaled_nominal_effort_bounds": (6, 2),
    "failure": (), "success": (), "done": (), "episode_id": (),
}
TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _equal(left, right):
    return _bytes(left) == _bytes(right)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _bytes(value)
    return value


def _path(value):
    path = Path(value).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink evidence path")
    return path.resolve()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _exact_module(path, expected_sha):
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_sha, "provider source SHA differs")
    module = types.ModuleType("retention_evaluation_original_provider")
    module.__file__ = str(path.resolve())
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    _require(path.read_bytes() == raw, "provider source changed while loading")
    return module


def _physical_validators(pin):
    """Compile four pure original metric validators, never its helper imports."""
    path = TOOLS / "run_frame_diagnostic_campaign.py"
    raw = path.read_bytes()
    _require(type(pin) is dict and set(pin) == {"path", "sha256", "bytes"}
             and pin["path"] == str(_path(path)) and type(pin["bytes"]) is int
             and pin["bytes"] == len(raw) and pin["sha256"] == hashlib.sha256(raw).hexdigest(),
             "frozen physical validator source differs")
    names = {"finite", "count", "validate_planar_pool", "validate_metrics"}
    tree = ast.parse(raw, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    _require({n.name for n in nodes} == names, "original physical validator interface differs")
    namespace = {"math": math, "__file__": str(path.resolve())}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    _require(path.read_bytes() == raw, "physical validator source changed while loading")
    return namespace["validate_metrics"], {"path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "execution": "AST-selected unchanged pure functions", "functions": sorted(names)}


def _npy_header(stream):
    _require(stream.read(6) == b"\x93NUMPY", "invalid NPY signature")
    version = stream.read(2)
    _require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "unsupported NPY version")
    length_size = 2 if version[0] == 1 else 4
    encoded = stream.read(length_size)
    _require(len(encoded) == length_size, "truncated NPY header length")
    length = int.from_bytes(encoded, "little")
    _require(0 < length <= 65536, "NPY header size differs")
    raw = stream.read(length)
    _require(len(raw) == length, "truncated NPY header")
    try:
        header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin1").strip())
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise ValueError("invalid NPY header") from error
    _require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
             and header["fortran_order"] is False and type(header["shape"]) is tuple
             and all(type(n) is int and n >= 0 for n in header["shape"])
             and type(header["descr"]) is str, "NPY must be a plain C-order array")
    match = re.fullmatch(r"([<>=|])([fibuU])(\d+)", header["descr"])
    _require(match is not None, "unsupported NPY dtype")
    order, kind, width = match[1], match[2], int(match[3])
    _require(width > 0 and (kind != "f" or width in (4, 8))
             and (kind != "b" or width == 1)
             and (kind not in "iu" or width in (1, 2, 4, 8)), "unsupported NPY dtype width")
    _require(order != "|" or width == 1, "multi-byte NPY requires declared byte order")
    return header, kind, width, math.prod(header["shape"]) * width * (4 if kind == "U" else 1)


def _numeric(raw, header, width):
    _require(len(raw) % width == 0, "unaligned NPY payload")
    values = array("f" if width == 4 else "d")
    values.frombytes(raw)
    order = header["descr"][0]
    if (order == ">" and sys.byteorder == "little") or (order == "<" and sys.byteorder == "big"):
        values.byteswap()
    return values


def _episodes(raw, header):
    _require(len(raw) % 8 == 0, "unaligned episode payload")
    values = array("q")
    values.frombytes(raw)
    if header["descr"].startswith("<") and sys.byteorder == "big":
        values.byteswap()
    return values


def validate_trace_archive(path, trace, expected, *, steps, rows):
    """Generic streaming ZIP/NPY validator; verify_suite fixes 4001 by 400.

    Small dimensions are solely useful for CPU tests of corruption boundaries.
    All physical arrays are read to EOF, including each ZIP member's CRC check.
    """
    _require(type(steps) is int and 0 < steps <= STEPS
             and type(rows) is int and 0 < rows <= CASES * REPLICAS,
             "trace dimensions must be bounded positive integers")
    path = _path(path)
    _require(type(trace) is dict and _path(trace["path"]) == path
             and trace["sha256"] == _sha(path), "trace actual SHA or path differs")
    indices = list(range(rows))
    _require(_equal(expected["row_indices"], indices)
             and type(expected["group_labels"]) is list and len(expected["group_labels"]) == rows
             and all(type(label) is str and label for label in expected["group_labels"])
             and _equal(expected["steps"], steps), "trace expected row or step coverage differs")
    _require(all(_equal(trace.get(k), v) for k, v in expected.items())
             and _equal(trace.get("rows"), indices), "trace report provenance differs")
    required = set(TRACE_SHAPES) | {"metadata_json", "row_indices"}
    captured = {}
    metadata = None
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            _require(len(members) == len(set(members)) and all(
                re.fullmatch(r"[a-z][a-z0-9_]*\.npy", n) for n in members), "trace ZIP members differ")
            fields = {n[:-4] for n in members}
            declared = trace.get("fields")
            _require(type(declared) is list and all(type(n) is str for n in declared)
                     and len(declared) == len(set(declared))
                     and required.issubset(fields) and fields <= required | {"command_request"}
                     and set(declared) == fields - {"metadata_json", "row_indices"}, "physical trace field coverage differs")
            for name in members:
                field = name[:-4]
                info = archive.getinfo(name)
                _require(not info.flag_bits & 1 and info.compress_type in
                         (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "unsupported trace ZIP encoding")
                with archive.open(name) as stream:
                    header, kind, width, size = _npy_header(stream)
                    if field == "metadata_json":
                        _require(header["shape"] == () and kind == "U" and size <= 262144,
                                 "trace metadata NPY schema differs")
                        raw = stream.read(size)
                        _require(len(raw) == size, "truncated trace metadata")
                        order = "big" if header["descr"].startswith(">") else sys.byteorder if header["descr"].startswith("=") else "little"
                        metadata = _json(raw.decode("utf-32-be" if order == "big" else "utf-32-le").rstrip("\x00"))
                    elif field == "row_indices":
                        _require(header["shape"] == (rows,) and header["descr"] in ("<i8", "=i8"),
                                 "trace row indices NPY schema differs")
                        raw = stream.read(size)
                        _require(len(raw) == size and list(struct.unpack(
                            ("=" if header["descr"].startswith("=") else "<") + str(rows) + "q", raw)) == indices,
                            "trace row indices differ")
                    else:
                        suffix = TRACE_SHAPES.get(field, (3,))
                        _require(header["shape"] == (steps, rows, *suffix), f"trace shape differs: {field}")
                        expected_kind = "b" if field in ("done", "failure", "success") else "i" if field == "episode_id" else "f"
                        _require(kind == expected_kind, f"trace dtype differs: {field}")
                        if kind == "i":
                            _require(header["descr"] in ("<i8", "=i8"), "episode IDs require signed int64")
                        saved = bytearray() if field in {"time_s", "episode_id", "done", "failure", "success"} else None
                        consumed = 0
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            consumed += len(block)
                            _require(consumed <= size, "extra NPY payload bytes")
                            if kind == "f":
                                _require(all(math.isfinite(v) for v in _numeric(block, header, width)),
                                         f"nonfinite physical trace sample: {field}")
                            elif kind == "b":
                                _require(all(v <= 1 for v in block), "invalid boolean trace payload")
                            if saved is not None:
                                saved.extend(block)
                        _require(consumed == size, "truncated NPY payload")
                        if saved is not None:
                            captured[field] = (_numeric(saved, header, width) if field == "time_s" else
                                _episodes(saved, header) if field == "episode_id" else saved)
                    _require(not stream.read(1), "extra NPY payload bytes")
            _require(type(metadata) is dict and all(_equal(metadata.get(k), v) for k, v in expected.items()),
                     "trace archive metadata differs")
    except (zipfile.BadZipFile, EOFError, struct.error, UnicodeError) as error:
        raise ValueError("invalid or corrupt trace archive") from error
    times, episodes, dones = [captured[k] for k in ("time_s", "episode_id", "done")]
    _require(all(v == 0 for v in episodes[:rows]) and all(v > 0 for v in times),
             "trace episode origin or positive physical time differs")
    failures, successes = captured["failure"], captured["success"]
    dt = expected["policy_dt_s"]
    _require(type(dt) in (int, float) and math.isfinite(dt) and dt > 0,
             "trace policy time interval differs")
    for index in range(steps * rows):
        _require(not (failures[index] and successes[index])
                 and (not (failures[index] or successes[index]) or dones[index]),
                 "terminal trace events disagree with done")
        if index >= rows:
            prior = index - rows
            _require(episodes[index] == episodes[prior] + dones[prior]
                     and (episodes[index] != episodes[prior] or times[index] > times[prior]),
                     "trace episode continuity or within-episode time differs")
            actual_dt = times[index] if dones[prior] else times[index] - times[prior]
        else:
            actual_dt = times[index]
        # The packed adapter publishes episode_ticks * policy_dt after each
        # step. Float32 traces need tolerance near the 40-second full horizon.
        _require(math.isclose(actual_dt, dt, rel_tol=0., abs_tol=5e-6),
                 "trace physical time does not match the policy tick clock")
    final_sha = _sha(path)
    _require(final_sha == trace["sha256"], "trace changed during validation")
    return {"path": str(path), "sha256": final_sha, "bytes": path.stat().st_size,
            "steps": steps, "rows": rows, "fields": sorted(fields), "metadata": metadata,
            "all_physical_arrays_finite": True, "episode_done_time_consistent": True,
            "validation": "complete standard-library ZIP/NPY payload and CRC scan"}


def _fixed_invocation(protocol, branch, checkpoint, seed):
    _require(type(protocol) is dict and type(branch) is dict and type(checkpoint) is dict,
             "retention evaluation requires dictionaries")
    evaluation = protocol.get("evaluation", {})
    _require(_equal(evaluation, {"seeds": [8701, 9701], "steps": STEPS, "settle_steps": 200,
        "min_steady_samples": 200, "trace_replicas": REPLICAS}), "fixed original evaluation protocol differs")
    cases = protocol.get("evaluation_cases", [])
    _require(type(cases) is list and len(cases) == CASES and all(type(c) is dict
             and type(c.get("name")) is str for c in cases)
             and len({c["name"] for c in cases}) == CASES, "all fifty original cases are required")
    _require(type(seed) is int and seed in evaluation["seeds"], "evaluation seed differs")
    _require(any(_equal(branch, b) for b in protocol.get("branches", [])), "branch differs from frozen grid")
    _require(set(checkpoint) == {"path", "sha256", "bytes", "update", "cumulative_transitions", "consumed_updates"}
             and type(checkpoint["bytes"]) is int and checkpoint["bytes"] > 0
             and type(checkpoint["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
             and _equal({k: checkpoint[k] for k in ("update", "cumulative_transitions", "consumed_updates")},
                {"update": 1200, "cumulative_transitions": 58982400, "consumed_updates": 1200}),
             "complete CP1200 checkpoint or consumed clock differs")


def _merged_contract(reader, bundle):
    contracts = []
    for entry in bundle["manifest"]["scenarios"]:
        environment = bundle["configs"][entry["config"]]["environment"]
        contract = reader.read(bundle["snapshot"] / environment["contract"])
        _require(contract.get("target_num_envs") == REPLICAS and contract.get("evaluation_exact_cases") is True
                 and len(contract["evaluation"]["cases"]) == 1
                 and contract["evaluation"]["cases"][0]["name"] == entry["name"], "original case contract layout differs")
        contracts.append(contract)
    def common(value):
        value = deepcopy(value)
        value.pop("scene_groups", None)
        value.pop("target_num_envs", None)
        value["evaluation"].pop("cases", None)
        return value
    _require(all(_equal(common(c), common(contracts[0])) for c in contracts), "suite contracts differ beyond cases")
    merged = deepcopy(contracts[0])
    merged["evaluation"]["cases"] = [c["evaluation"]["cases"][0] for c in contracts]
    merged["scene_groups"] = [{"name": c["name"], "fraction": 1 / CASES,
                              "terrain": [c.get("terrain", "flat")]} for c in merged["evaluation"]["cases"]]
    merged["target_num_envs"] = CASES * REPLICAS
    return _digest(merged)


def verify_suite(protocol, branch, checkpoint, seed, directory):
    """Recheck actual CP1200, all original gates and all 400 raw trace rows."""
    _fixed_invocation(protocol, branch, checkpoint, seed)
    # The campaign authorizes the entire branch, including K when lambda=0.
    from .retention_campaign import validate_protocol, learner_source
    validate_protocol(protocol)
    pins = protocol["provider_source"]
    provider_path = TOOLS / "prepare_nested_anchors.py"
    provider = _exact_module(provider_path, pins[str(provider_path.resolve())])
    auditor = provider._auditor()
    analyzer_path = TOOLS / "analyze_curriculum_retention.py"
    _require(hashlib.sha256(auditor._source_bytes).hexdigest() == pins[str(analyzer_path.resolve())],
             "original analyzer source differs")
    reader = auditor.Reader()
    prep_path = reader.checked(Path("/"), protocol["preparation_protocol"])
    prepared = reader.read(prep_path)
    qualification = prepared["qualification_protocol"]
    bundle = auditor._bundle(reader, qualification["campaign_root"])
    actual = auditor._protocol_body(reader, bundle)
    _require(_equal({k: v for k, v in qualification.items() if k not in {"prepared_at", "sha256"}}, actual),
             "original physical qualification or source changed")
    cases = [c["name"] for c in bundle["manifest"]["scenarios"]]
    _require(_equal(protocol["evaluation_cases"], bundle["manifest"]["scenarios"]), "original case configs differ")
    directory = _path(directory)
    _require(directory.is_dir() and directory.is_relative_to(_path(protocol["output_root"]) / branch["id"]),
             "suite directory escapes its authorized branch")
    cp_path = _path(checkpoint["path"])
    _require(cp_path.is_relative_to(_path(protocol["output_root"]) / branch["id"]),
             "checkpoint belongs to another branch")
    reader.checked(Path("/"), checkpoint)
    sidecar_path = _path(str(cp_path) + ".json")
    sidecar = reader.read(sidecar_path)
    metadata = sidecar["metadata"]
    expected_branch = {"controller_protocol_sha256": protocol["sha256"],
        "preparation_protocol_sha256": protocol["preparation_protocol_sha256"],
        "preparation_manifest_sha256": protocol["preparation_manifest_sha256"], "branch_id": branch["id"]}
    _require(sidecar.get("format") == "transformer_rl.packed_checkpoint"
             and type(sidecar.get("schema_version")) is int and sidecar["schema_version"] == 1
             and _equal(sidecar.get("update"), 1200) and sidecar.get("sha256") == checkpoint["sha256"]
             and _equal(sidecar.get("config"), protocol["schedules"][branch["schedule"]])
             and _equal(metadata.get("source"), protocol["source"])
             and _equal(metadata.get("seed"), branch["training_seed"])
             and metadata.get("environment_factory") == protocol["environment_factory"]
             and _equal(metadata.get("collected_transitions"), 58982400)
             and _equal(metadata.get("continuation_branch"), expected_branch)
             and _equal(metadata.get("continuation", {}).get("clock"), {"consumed_updates": 1200,
                 "collected_transitions": 58982400, "rollout_steps": 48}), "CP1200 producer or branch identity differs")
    control_path, trace_path = _path(directory / "control.json"), _path(directory / "trace.npz")
    control = reader.read(control_path)
    expected = {"checkpoint_sha256": checkpoint["sha256"], "checkpoint_update": 1200,
                "seed": seed, "steps": STEPS}
    _require(control.get("format") == "transformer_rl.control_evaluation"
             and type(control.get("schema_version")) is int and control["schema_version"] == 1
             and all(_equal(control.get(k), v) for k, v in expected.items())
             and set(control.get("groups", {})) == set(cases), "control suite identity or cases differ")
    provenance = control["environment_provenance"]
    cfg = bundle["configs"][bundle["manifest"]["scenarios"][0]["config"]]
    labels = provenance.get("evaluation_groups")
    _require(type(labels) is list and len(labels) == CASES * REPLICAS
             and all(type(label) is str and label for label in labels)
             and Counter(labels) == Counter({c: REPLICAS for c in cases})
             and provenance.get("identity") == qualification["snapshot_sha256"]
             and provenance.get("control_sha256") == _digest(cfg["control"])
             and provenance.get("contract_sha256") == _merged_contract(reader, bundle)
             and metadata.get("environment_provenance", {}).get("identity") == provenance["identity"],
             "suite snapshot, control or effective merged contract differs")
    metrics_validator, metric_source = _physical_validators(protocol["physical_validator_source"])
    metrics_validator(control["control"], CASES * REPLICAS)
    cp = {"checkpoint_sha256": checkpoint["sha256"], "update": 1200}
    cells, artifacts = {}, {}
    for case in cases:
        path = _path(directory / f"{case}.json")
        report = reader.read(path)
        entry = next(c for c in bundle["manifest"]["scenarios"] if c["name"] == case)
        case_cfg = bundle["configs"][entry["config"]]
        case_identity = {**expected, "format": "transformer_rl.packed_evaluation", "schema_version": 1,
            "model": case_cfg["model"], "environment": case_cfg["environment"],
            "control_sha256": _digest(case_cfg["control"]), "num_envs": REPLICAS,
            "transitions": STEPS * REPLICAS}
        _require(all(_equal(report.get(k), v) for k, v in case_identity.items())
                 and _equal(report.get("control"), control["groups"][case])
                 and _equal(report.get("environment_provenance"), provenance)
                 and type(report.get("stability", {}).get("available")) is bool,
                 "case identity, physical provenance or stability availability differs")
        metrics_validator(report["control"], REPLICAS)
        cell = auditor._cell(reader, bundle, qualification, case, seed, cp, path, control)
        cells[case] = {**cell, "metrics_complete": report["metrics"],
            "control": report["control"], "stability": report["stability"],
            "success_metric_available": report.get("success_metric_available"),
            "success_rate": report.get("success_rate"), "reward_mean": report.get("reward_mean")}
        artifacts[case] = reader.receipt(path)
    trace_expected = {**expected, "policy_dt_s": .01, "sampling_hz": 100.,
        "control_sha256": _digest(cfg["control"]), "row_indices": list(range(CASES * REPLICAS)),
        "group_labels": labels}
    trace = validate_trace_archive(trace_path, control["trace"], trace_expected,
                                   steps=STEPS, rows=CASES * REPLICAS)
    artifacts.update(control=reader.receipt(control_path), trace=reader.receipt(trace_path),
                     checkpoint=reader.receipt(cp_path), checkpoint_sidecar=reader.receipt(sidecar_path))
    reader.unchanged()
    _require(_equal(learner_source(), protocol["source"]) and _sha(metric_source["path"]) == metric_source["sha256"],
             "evaluation verifier or physical helper source changed")
    body = {"format": "transformer_rl.retention_physical_evaluation", "schema_version": 1,
        "status": "completed", "protocol_sha256": protocol["sha256"], "branch_id": branch["id"],
        "checkpoint": deepcopy(checkpoint), "evaluation_seed": seed, "steps": STEPS,
        "case_count": CASES, "case_replicas": REPLICAS, "transitions": STEPS * CASES * REPLICAS,
        "cells": cells, "control": control["control"], "environment_provenance": provenance,
        "trace_validation": trace, "artifacts": artifacts, "source": deepcopy(protocol["source"]),
        "original_physical_validator": metric_source, "input_receipts": reader.files,
        "worker_terminal_evidence": "required and verified separately by controller",
        "formal_architecture_selection": False, "hardware_verified": False}
    return {**body, "sha256": _digest(body)}

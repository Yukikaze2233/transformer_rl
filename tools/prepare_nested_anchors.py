#!/usr/bin/env python3
"""Prepare nested CPU anchor subsets; never collect data or start a learner.

Only the existing Gated H31 curriculum qualification provider is supported.
Behavior-anchor v1 does not identify a case or capture seed: a frozen pool and
capture-report receipt must supply those missing links. Reset windows remain
part of the original pool; this tool does not claim steady-only sampling.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types


TOOL_ROOT = Path(__file__).resolve().parent
QUALIFICATION_TOOL = TOOL_ROOT / "analyze_curriculum_retention.py"
FORMAT = "transformer_rl.nested_anchor_preparation_protocol"
PERMUTATION = "sha256_pool_seed_case_row_order_v1"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _auditor():
    source = QUALIFICATION_TOOL.read_bytes()
    module = types.ModuleType("nested_anchor_qualification")
    module.__file__ = str(QUALIFICATION_TOOL.resolve())
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    _require(QUALIFICATION_TOOL.read_bytes() == source, "qualification analyzer changed while importing")
    module._source_bytes = source
    return module


def _source_identity():
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), QUALIFICATION_TOOL)}


def _receipt(path, data=None):
    path = Path(path).resolve()
    data = path.read_bytes() if data is None else data
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _read_checked(reader, receipt):
    _require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256", "bytes"}
             and Path(receipt["path"]).is_absolute()
             and type(receipt["bytes"]) is int and receipt["bytes"] >= 0,
             "a pool/report requires an absolute file receipt with size and SHA256")
    path = reader.checked(Path("/"), receipt)
    return path, reader.raw(path)


def freeze(qualification_protocol, *, capacities, coefficients, anchor_seed, pool_receipts=()):
    """Freeze an unexecuted plan. Qualification is reaudited during preparation."""
    audit = _auditor()
    q = qualification_protocol
    _require(isinstance(q, dict) and q.get("format") == "transformer_rl.curriculum_retention_protocol"
             and type(q.get("schema_version")) is int and q["schema_version"] == 1
             and audit.digest({k: v for k, v in q.items() if k != "sha256"}) == q.get("sha256"),
             "qualification protocol identity differs")
    _require(q["architecture"] == "transformer_gated" and q["old_cases"] == audit.OLD_CASES
             and q["new_cases"] == audit.NEW_CASES and len(q["training_seeds"]) >= 3
             and q["evaluation"]["seeds"] == [8701, 9701], "unsupported qualification provider scope")
    policy = q["model"]["policy"]
    _require(policy.get("architecture") == "transformer" and policy.get("residual_type") == "gated"
             and policy.get("history_length") == 31
             and all(type(policy.get(k)) is int and policy[k] > 0 for k in ("frame_dim", "action_dim")),
             "the supported teacher is Gated H31 with explicit frame/action dimensions")
    capacities, coefficients = list(capacities), list(coefficients)
    _require(capacities and len(set(capacities)) == len(capacities)
             and all(type(k) is int and k > 0 for k in capacities), "capacities require unique positive integers")
    _require(coefficients and len(set(coefficients)) == len(coefficients)
             and all(type(c) in (int, float) and math.isfinite(c) and c >= 0 for c in coefficients),
             "coefficients require unique finite nonnegative numbers")
    _require(type(anchor_seed) is int and 0 <= anchor_seed < 2**32, "anchor_seed must be uint32")
    _require(anchor_seed not in q["training_seeds"] + q["evaluation"]["seeds"],
             "capture anchor_seed must be independent of training and qualification seeds")
    pools = json.loads(_bytes(list(pool_receipts)))
    known, pool_paths, pool_hashes, report_paths = set(), set(), set(), set()
    for entry in pools:
        _require(set(entry) == {"training_seed", "case", "pool", "report"}, "unexpected pool receipt fields")
        key = entry["training_seed"], entry["case"]
        _require(type(key[0]) is int and key[0] in q["training_seeds"] and key[1] in q["old_cases"]
                 and key not in known, "duplicate or out-of-grid pool; cases cannot be selected")
        for kind in ("pool", "report"):
            ref = entry[kind]
            _require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "bytes"}
                     and isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
                     and audit.SHA.fullmatch(str(ref["sha256"])) and type(ref["bytes"]) is int and ref["bytes"] >= 0,
                     "pool/report receipt fields differ")
        pool_path, report_path = str(Path(entry["pool"]["path"]).resolve()), str(Path(entry["report"]["path"]).resolve())
        _require(pool_path not in pool_paths and entry["pool"]["sha256"] not in pool_hashes
                 and report_path not in report_paths, "a common pool cannot be reused across teacher/case pairs")
        known.add(key)
        pool_paths.add(pool_path)
        pool_hashes.add(entry["pool"]["sha256"])
        report_paths.add(report_path)
    body = {"format": FORMAT, "schema_version": 1, "prepared_at": datetime.now(timezone.utc).isoformat(),
        "architecture_scope": {"supported_provider": "strict_curriculum_retention_v1",
            "architecture": "transformer_gated", "teacher_history_length": 31,
            "other_architectures": "unexecuted; require their own trusted acquisition providers"},
        "qualification_protocol": json.loads(_bytes(q)), "source_identity": _source_identity(),
        "teacher": {"arm": "pretrain", "phase": "phase1", "checkpoint_update": 400},
        "training_seeds": q["training_seeds"], "cases": q["old_cases"], "anchor_seed": anchor_seed,
        "capacities": capacities, "coefficients": coefficients, "pool_receipts": pools,
        "capture_provider": "single_case_evaluate_frame_policy_v1",
        "permutation": PERMUTATION, "sampling_scope": "unaltered pool, including reset and transient windows",
        "budgets": {"execution_status": "unexecuted", "updates_per_branch": 800,
                    "transitions_per_update": 49152, "transitions_per_branch": 800 * 49152},
        "expected_qualification_cells": 3 * len(q["training_seeds"]) * 2 * 2 * 50,
        "expected_teacher_case_pairs": len(q["training_seeds"]) * len(q["old_cases"]),
        "expected_anchor_files": len(q["training_seeds"]) * len(q["old_cases"]) * len(capacities),
        "expected_cells": len(q["training_seeds"]) * len(q["old_cases"]) * len(capacities) * len(coefficients),
        "expected_branches": len(q["training_seeds"]) * len(capacities) * len(coefficients)}
    return {**body, "sha256": _digest(body)}


def _validate_protocol(protocol):
    _require(isinstance(protocol, dict) and _digest({k: v for k, v in protocol.items() if k != "sha256"})
             == protocol.get("sha256"), "preparation protocol SHA differs")
    expected = freeze(protocol["qualification_protocol"], capacities=protocol["capacities"],
        coefficients=protocol["coefficients"], anchor_seed=protocol["anchor_seed"], pool_receipts=protocol["pool_receipts"])
    ignored = {"prepared_at", "sha256"}
    _require({k: v for k, v in protocol.items() if k not in ignored}
             == {k: v for k, v in expected.items() if k not in ignored},
             "preparation provider, source, grid or frozen branch settings differ")


def _permutation(pool_sha, seed, case, count):
    return sorted(range(count), key=lambda i: (hashlib.sha256(
        _bytes([PERMUTATION, pool_sha, seed, case, i])).digest(), i))


def _tensor_receipt(tensor):
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()}


def _load_pool(reader, entry, protocol, teacher_pair):
    import torch
    q = protocol["qualification_protocol"]
    path, raw = _read_checked(reader, entry["pool"])
    report_path, report_raw = _read_checked(reader, entry["report"])
    report = _auditor()._json(report_raw)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and set(payload) == {"format", "schema_version", "control_sha256",
        "policy_config", "teacher_checkpoint_sha256", "frames", "mean", "std"}, "invalid behavior anchor v1 fields")
    reference = reader.read(teacher_pair[0]["report"]["path"])
    environment = reference["environment"]
    contract_path = reader.checked(Path(environment["snapshot"]),
        {"path": environment["contract"], "sha256": environment["contract_sha256"]})
    effective_contract_sha256 = _digest(reader.read(contract_path))
    provenance = report.get("environment_provenance", {})
    anchors = report.get("anchors", {})
    _require(report.get("format") == "transformer_rl.packed_evaluation"
        and type(report.get("schema_version")) is int and report["schema_version"] == 1
        and all(type(report.get(k)) is int for k in ("checkpoint_update", "seed", "steps", "num_envs", "transitions"))
        and report.get("checkpoint_update") == 400 and report.get("checkpoint_sha256") == teacher_pair[0]["checkpoint_sha256"]
        and report.get("model") == q["model"] and report.get("control_sha256") == reference["control_sha256"]
        and report.get("environment") == reference["environment"] and report.get("seed") == protocol["anchor_seed"]
        and report.get("steps") == q["evaluation"]["steps"] and report.get("num_envs") == 8
        and report.get("transitions") == q["evaluation"]["steps"] * 8
        and report.get("policy") == "deterministic_raw_mean_then_declared_action_limits"
        and provenance.get("identity") == q["snapshot_sha256"]
        and provenance.get("control_sha256") == reference["control_sha256"]
        and provenance.get("contract_sha256") == effective_contract_sha256,
        "pool capture teacher/model/case/seed/effective contract differs; merged-suite capture is unsupported")
    _require(payload.get("format") == "transformer_rl.behavior_anchors"
        and type(payload.get("schema_version")) is int and payload["schema_version"] == 1
        and payload.get("control_sha256") == reference["control_sha256"]
        and payload.get("policy_config") == q["model"]["policy"]
        and payload.get("teacher_checkpoint_sha256") == teacher_pair[0]["checkpoint_sha256"],
        "pool teacher, architecture or control identity differs")
    frames, mean, std = (payload[k] for k in ("frames", "mean", "std"))
    policy = q["model"]["policy"]
    _require(all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.layout == torch.strided
                 and t.device.type == "cpu" and not t.requires_grad and torch.isfinite(t).all()
                 for t in (frames, mean, std)),
             "anchor tensors require finite CPU strided float32 values")
    _require(frames.ndim == 3 and frames.shape[1:] == (31, policy["frame_dim"]) and len(frames) > 0
             and mean.shape == (len(frames), policy["action_dim"]) and std.shape == mean.shape
             and (std > 0).all(), "anchor tensor shape, history or positive std contract differs")
    _require(Path(anchors.get("path", "")).resolve() == path and anchors.get("sha256") == entry["pool"]["sha256"]
             and type(anchors.get("samples")) is int and anchors["samples"] == len(frames)
             and type(anchors.get("max_samples")) is int and anchors["max_samples"] >= len(frames),
             "pool and capture report anchor receipt differ")
    return payload, {"pool": _receipt(path, raw), "report": _receipt(report_path, report_raw),
        "teacher_checkpoint_sha256": teacher_pair[0]["checkpoint_sha256"], "teacher_model_sha256": _digest(q["model"]),
        "control_sha256": reference["control_sha256"], "source_sha256": q["source_sha256"],
        "capture_provider": protocol["capture_provider"], "effective_contract_sha256": effective_contract_sha256,
        "snapshot_sha256": q["snapshot_sha256"], "pool_tensor_receipts": {k: _tensor_receipt(payload[k])
                                                                             for k in ("frames", "mean", "std")}}


def assess(qualification_protocol):
    """Pure readiness audit, with no output paths, locks or tensor loading.

    The analyzer itself revalidates the original frozen source and physical
    gates. Its source bytes identify the actual imported provider; no cached
    analysis or caller-supplied eligible boolean can qualify a teacher.
    """
    audit = _auditor()
    analysis = audit.analyze(qualification_protocol, analyzer_source=audit._source_bytes)
    q = qualification_protocol
    pairs = []
    for seed in q["training_seeds"]:
        for case in q["old_cases"]:
            cells = [next(c for c in analysis["cells"] if c["arm"] == "pretrain" and c["phase"] == "phase1"
                          and c["training_seed"] == seed and c["case"] == case and c["evaluation_seed"] == ev)
                     for ev in q["evaluation"]["seeds"]]
            acquired = all(c["status"] == "ready" and c["passed"] is True
                           and c["checkpoint_update"] == 400 for c in cells)
            if acquired:
                _require(cells[0]["checkpoint_sha256"] == cells[1]["checkpoint_sha256"], "teacher checkpoint pair differs")
            pairs.append({"training_seed": seed, "case": case, "qualified": acquired, "evaluations": cells})
    return {"format": "transformer_rl.nested_anchor_teacher_readiness", "schema_version": 1,
        "status": "ready" if all(p["qualified"] for p in pairs) else "not_ready",
        "architecture": "transformer_gated", "teacher": {"arm": "pretrain", "phase": "phase1", "checkpoint_update": 400},
        "expected_teacher_case_pairs": len(pairs), "qualified_teacher_case_pairs": sum(p["qualified"] for p in pairs),
        "pairs": pairs, "analysis": analysis, "anchor_publication": False, "execution_status": "unexecuted"}


def build(protocol, output_directory):
    """Read and audit all evidence; return bytes without publishing any files."""
    _validate_protocol(protocol)
    audit = _auditor()
    q = protocol["qualification_protocol"]
    output = Path(output_directory).resolve()
    _require(not output.is_relative_to(Path(q["campaign_root"]).resolve().parent),
             "preparation output must be outside the frozen experiment")
    analysis = assess(q)["analysis"]
    _require(analysis["analyzer_sha256"] == protocol["source_identity"][str(QUALIFICATION_TOOL.resolve())]
             and analysis["expected_cells"] == protocol["expected_qualification_cells"], "qualification audit scope differs")
    reader = audit.Reader()
    for path, sha in protocol["source_identity"].items():
        reader.checked(Path("/"), {"path": path, "sha256": sha})
    for path, receipt in analysis["input_receipts"].items():
        reader.checked(Path("/"), {"path": path, **receipt})
    pools = {(e["training_seed"], e["case"]): e for e in protocol["pool_receipts"]}
    qualified, missing = {}, []
    available_pools = set()
    for key, entry in pools.items():
        available = True
        for kind in ("pool", "report"):
            try:
                _read_checked(reader, entry[kind])
            except FileNotFoundError:
                available = False
                missing.append({"training_seed": key[0], "case": key[1], "reason": f"{kind}_file_not_available"})
        if available:
            available_pools.add(key)
    for seed in protocol["training_seeds"]:
        for case in protocol["cases"]:
            pair = [next(c for c in analysis["cells"] if c["arm"] == "pretrain" and c["phase"] == "phase1"
                         and c["training_seed"] == seed and c["case"] == case and c["evaluation_seed"] == ev)
                    for ev in q["evaluation"]["seeds"]]
            acquired = all(c["status"] == "ready" and c["passed"] is True and c["checkpoint_update"] == 400 for c in pair)
            if acquired:
                _require(pair[0]["checkpoint_sha256"] == pair[1]["checkpoint_sha256"], "teacher checkpoint pair differs")
                qualified[seed, case] = pair
            else:
                missing.append({"training_seed": seed, "case": case, "reason": "teacher_acquisition_not_qualified",
                                "evaluations": pair})
            if (seed, case) not in pools:
                missing.append({"training_seed": seed, "case": case, "reason": "common_pool_not_available"})
    ready = not missing
    outputs, prepared, loaded, content_identities = {}, {}, {}, set()
    if ready:
        for (seed, case), pair in qualified.items():
            payload, identity = _load_pool(reader, pools[seed, case], protocol, pair)
            content_identity = _digest(identity["pool_tensor_receipts"])
            _require(content_identity not in content_identities,
                     "identical tensor content cannot establish independent teacher/case capture")
            content_identities.add(content_identity)
            identity["pool_tensor_content_sha256"] = content_identity
            loaded[seed, case] = payload, identity
            for k in protocol["capacities"]:
                prepared[seed, case, k] = {**identity, "actual_n": min(k, len(payload["frames"])),
                                           "pool_sha256": identity["pool"]["sha256"]}
                if len(payload["frames"]) < k:
                    missing.append({"training_seed": seed, "case": case, "reason": "pool_capacity_not_available",
                                    "requested_k": k, "actual_n": len(payload["frames"])})
        ready = not missing
    if ready:
        import torch
        for (seed, case), (payload, identity) in loaded.items():
            order = _permutation(identity["pool"]["sha256"], protocol["anchor_seed"], case, len(payload["frames"]))
            for k in protocol["capacities"]:
                indices = order[:min(k, len(order))]
                subset = {**payload, **{name: payload[name][indices].contiguous() for name in ("frames", "mean", "std")}}
                stream = io.BytesIO()
                torch.save(subset, stream)
                anchor_path = output / f"seed_{seed}__{case}__k_{k}.pt"
                index_path = output / f"seed_{seed}__{case}__k_{k}.indices.json"
                index_data = _bytes({"format": "transformer_rl.nested_anchor_indices", "schema_version": 1,
                    "permutation": PERMUTATION, "permutation_sha256": _digest(order),
                    "pool_sha256": identity["pool"]["sha256"], "anchor_seed": protocol["anchor_seed"],
                    "case": case, "requested_k": k, "actual_n": len(indices), "indices": indices}) + b"\n"
                outputs[anchor_path], outputs[index_path] = stream.getvalue(), index_data
                prepared[seed, case, k] = {**identity, "actual_n": len(indices),
                    "pool_sha256": identity["pool"]["sha256"], "permutation_sha256": _digest(order),
                    "anchor": _receipt(anchor_path, outputs[anchor_path]), "indices": _receipt(index_path, index_data),
                    "tensor_receipts": {name: _tensor_receipt(subset[name]) for name in ("frames", "mean", "std")}}
    cells, branches = [], []
    summary = reader.read(Path(q["campaign_root"]) / "summary.json") if qualified else {"results": {}}
    for seed in protocol["training_seeds"]:
        parent_checkpoint = None
        if any(s == seed for s, _ in qualified):
            phase = next(p for p in summary["results"][f"pretrain/seed_{seed}"]["phases"] if p["name"] == "phase1")
            endpoint = phase["training"]["checkpoint"]
            reference = next(pair[0] for (s, _), pair in qualified.items() if s == seed)
            _require(endpoint["update"] == 400 and endpoint["checkpoint_sha256"] == reference["checkpoint_sha256"]
                     and endpoint["cumulative_transitions"] == 400 * 49152, "branch CP400 parent or consumed clock differs")
            parent_path = reader.checked(Path("/"), {"path": endpoint["checkpoint"], "sha256": endpoint["checkpoint_sha256"]})
            parent_checkpoint = {**reader.receipt(parent_path), "update": 400, "cumulative_transitions": 400 * 49152}
        for k in protocol["capacities"]:
            for coefficient in protocol["coefficients"]:
                branch_cells = []
                for case in protocol["cases"]:
                    cell = {"training_seed": seed, "case": case, "requested_k": k, "coefficient": coefficient,
                            "status": "ready" if ready else "not_ready", **prepared.get((seed, case, k), {})}
                    cells.append(cell)
                    branch_cells.append(cell)
                branches.append({"training_seed": seed, "requested_k": k, "coefficient": coefficient,
                    "status": "ready" if ready else "not_ready", "execution_status": "unexecuted",
                    "teacher": protocol["teacher"], "model_sha256": _digest(q["model"]),
                    "parent_checkpoint": parent_checkpoint, "consumed_update_offset": 400,
                    "required_resume_state": ["model", "optimizer", "rng", "clock"],
                    "resume_executor": "not_implemented", "retention_rng": "independent_generator_not_implemented",
                    "anchor_seed": protocol["anchor_seed"], "case_order": protocol["cases"],
                    "anchor_paths": [c["anchor"]["path"] for c in branch_cells] if ready and coefficient > 0 else [],
                    "retention_enabled": coefficient > 0, **protocol["budgets"]})
    reader.unchanged()
    result = {"format": "transformer_rl.nested_anchor_preparation", "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(), "status": "ready" if ready else "not_ready",
        "protocol_sha256": protocol["sha256"], "architecture_scope": protocol["architecture_scope"],
        "qualification_protocol_sha256": q["sha256"], "qualification_analyzer_sha256": analysis["analyzer_sha256"],
        "qualification_ready_cells": analysis["ready_cells"], "expected_qualification_cells": analysis["expected_cells"],
        "expected_teacher_case_pairs": protocol["expected_teacher_case_pairs"], "qualified_teacher_case_pairs": len(qualified),
        "expected_cells": protocol["expected_cells"], "ready_cells": len(cells) if ready else 0,
        "expected_anchor_files": protocol["expected_anchor_files"], "anchor_file_count": len(prepared) if ready else 0,
        "pool_file_count": len(available_pools), "declared_pool_file_count": len(pools),
        "expected_branches": protocol["expected_branches"],
        "cells": cells, "branches": branches, "missing": missing, "input_receipts": reader.files,
        "execution_status": "unexecuted", "qualification_scope": "paired original CP400 gates; no eligible-flag provider"}
    result["sha256"] = _digest(result)
    outputs[output / "manifest.json"] = _bytes(result) + b"\n"
    return result, outputs


def _publish_exclusive(output, outputs):
    """Reserve a new directory; fsync and link files, with manifest last.

    Whole-directory crash atomicity is not promised. Only the complete final
    manifest plus successful validation permits consumption of any anchors.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output / "manifest.json"
    _require(marker in outputs and all(path.parent == output for path in outputs),
             "publication requires a completion manifest and contained file paths")
    ordered_outputs = [(path, data) for path, data in outputs.items() if path != marker]
    ordered_outputs.append((marker, outputs[marker]))
    with tempfile.TemporaryDirectory(prefix=".nested-anchor-stage-", dir=output.parent) as staging:
        staged, published = {}, []
        for number, (path, data) in enumerate(ordered_outputs):
            temporary = Path(staging) / str(number)
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            staged[path] = temporary
        output.mkdir(exist_ok=False)
        try:
            for path, temporary in staged.items():
                published.append((temporary, path))
                os.link(temporary, path)
            descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            for temporary, path in reversed(published):
                try:
                    before, after = temporary.stat(), path.stat(follow_symlinks=False)
                    if (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
                        path.unlink()
                except FileNotFoundError:
                    pass
            try:
                output.rmdir()
            except OSError:
                pass
            raise


def prepare(protocol, output_directory):
    requested_output = Path(output_directory)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(requested_output)
    output = requested_output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result, outputs = build(protocol, output)
    if result["status"] != "ready":
        return result
    _publish_exclusive(output, outputs)
    return result


def validate_preparation(protocol, output_directory):
    output = Path(output_directory).resolve()
    raw_manifest = (output / "manifest.json").read_bytes()
    result = _auditor()._json(raw_manifest)
    _require(raw_manifest == _bytes(result) + b"\n", "preparation manifest bytes differ from canonical publication")
    _require(_digest({k: v for k, v in result.items() if k != "sha256"}) == result.get("sha256")
             and result.get("protocol_sha256") == protocol["sha256"], "preparation manifest identity differs")
    reader = _auditor().Reader()
    for path, receipt in result["input_receipts"].items():
        reader.checked(Path("/"), {"path": path, **receipt})
    fresh, _ = build(protocol, output)
    ignored = {"captured_at", "sha256"}
    _require({k: v for k, v in result.items() if k not in ignored}
             == {k: v for k, v in fresh.items() if k not in ignored}, "preparation inputs or branch descriptors changed")
    names = {"manifest.json"}
    for cell in result["cells"]:
        if cell["status"] != "ready":
            continue
        for key in ("anchor", "indices"):
            path, _ = _read_checked(reader, cell[key])
            _require(path.parent == output, "prepared anchor route escapes output directory")
            names.add(path.name)
    _require({p.name for p in output.iterdir()} == names, "unexpected files in prepared output")
    reader.unchanged()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    plan = sub.add_parser("freeze")
    plan.add_argument("--qualification-protocol", type=Path, required=True)
    plan.add_argument("--capacities", type=int, nargs="+", required=True)
    plan.add_argument("--coefficients", type=float, nargs="+", required=True)
    plan.add_argument("--anchor-seed", type=int, required=True)
    plan.add_argument("--pool-receipts", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    readiness = sub.add_parser("assess")
    readiness.add_argument("--qualification-protocol", type=Path, required=True)
    for operation in ("prepare", "validate"):
        command = sub.add_parser(operation)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "assess":
        q = _auditor()._json(args.qualification_protocol.read_bytes())
        result = assess(q)
    elif args.operation == "freeze":
        q = _auditor()._json(args.qualification_protocol.read_bytes())
        _require(not args.output.resolve().is_relative_to(Path(q["campaign_root"]).resolve().parent),
                 "protocol output must be outside the frozen experiment")
        pools = _auditor()._json(args.pool_receipts.read_bytes()) if args.pool_receipts else []
        result = freeze(q, capacities=args.capacities, coefficients=args.coefficients,
                        anchor_seed=args.anchor_seed, pool_receipts=pools)
        with tempfile.NamedTemporaryFile(dir=args.output.parent) as temporary:
            temporary.write(_bytes(result) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            os.link(temporary.name, args.output)
    else:
        protocol = _auditor()._json(args.protocol.read_bytes())
        result = (prepare if args.operation == "prepare" else validate_preparation)(protocol, args.output_directory)
    print(json.dumps({k: result[k] for k in ("status", "ready_cells", "expected_cells", "anchor_file_count",
                      "qualified_teacher_case_pairs", "expected_teacher_case_pairs")
                      if k in result}))


if __name__ == "__main__":
    main()

"""Small standard-library ZIP fixtures, never a real 4001 by 400 trace.

Generic array tests exercise complete payload validation on four steps/two rows.
Fixed-suite negatives prove the production entrypoint does not relax its grid.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

import pytest

from transformer_rl import retention_evaluation as evaluation


def npy(shape, dtype, payload, *, version=b"\x01\x00", fortran=False):
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("latin1") + b"\n"
    return b"\x93NUMPY" + version + len(header).to_bytes(2 if version[0] == 1 else 4, "little") + header + payload


def encode(shape, dtype, values):
    count = math.prod(shape)
    assert len(values) == count
    kind, width = dtype[1], int(dtype[2:])
    if kind == "b":
        payload = bytes(values)
    else:
        payload = struct.pack((">" if dtype[0] == ">" else "<") + str(count)
                              + ("q" if kind == "i" else "f" if width == 4 else "d"), *values)
    return npy(shape, dtype, payload)


def metadata_member(value):
    raw = json.dumps(value).encode("utf-32-le")
    return npy((), "<U" + str(len(raw) // 4), raw)


@pytest.fixture
def trace(tmp_path):
    steps, rows = 4, 2
    expected = {"checkpoint_sha256": "a" * 64, "checkpoint_update": 1200, "seed": 8701,
        "steps": steps, "policy_dt_s": .01, "sampling_hz": 100., "control_sha256": "b" * 64,
        "row_indices": [0, 1], "group_labels": ["first", "second"]}
    members = {}
    for name, suffix in evaluation.TRACE_SHAPES.items():
        shape = (steps, rows, *suffix)
        values = [0.] * math.prod(shape)
        dtype = "<f4"
        if name == "time_s":
            values, dtype = [.01, .01, .02, .02, .01, .03, .02, .04], "<f8"
        elif name == "episode_id":
            values, dtype = [0, 0, 0, 0, 1, 0, 1, 0], "<i8"
        elif name in {"done", "failure", "success"}:
            values, dtype = ([0, 0, 1, 0, 0, 0, 0, 0] if name in {"done", "success"}
                             else [0] * (steps * rows)), "|b1"
        members[name + ".npy"] = encode(shape, dtype, values)
    members["row_indices.npy"] = encode((rows,), "<i8", [0, 1])
    members["metadata_json.npy"] = metadata_member(expected)
    path = tmp_path / "small-trace.npz"

    def publish(changes=None, *, duplicate=None):
        data = {**members, **(changes or {})}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in data.items():
                if raw is not None:
                    archive.writestr(name, raw)
            if duplicate:
                archive.writestr(duplicate, members[duplicate])
        report = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": [0, 1], "fields": [name[:-4] for name, raw in data.items()
                if raw is not None and name not in {"metadata_json.npy", "row_indices.npy"}], **expected}
        return report
    return path, expected, members, publish


def check(trace, report=None):
    path, expected, _, publish = trace
    return evaluation.validate_trace_archive(path, report or publish(), expected, steps=4, rows=2)


def test_complete_generic_trace_and_source_free_import(trace):
    result = check(trace)
    assert result["episode_done_time_consistent"] is True
    assert result["all_physical_arrays_finite"] is True
    assert result["rows"] == 2 and result["steps"] == 4
    assert result["bytes"] < 20_000


@pytest.mark.parametrize("field", ["actual", "position_xy", "effort_bounds", "scaled_nominal_effort_bounds"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_physical_payload_rejects_nonfinite(trace, field, value):
    _, _, _, publish = trace
    shape = (4, 2, *evaluation.TRACE_SHAPES[field])
    values = [0.] * math.prod(shape)
    values[-1] = value
    report = publish({field + ".npy": encode(shape, "<f4", values)})
    with pytest.raises(ValueError, match="nonfinite physical"):
        check(trace, report)


def test_big_endian_floats_are_scanned(trace):
    _, _, _, publish = trace
    report = publish({"actual.npy": encode((4, 2, 3), ">f8", [0.] * 24)})
    check(trace, report)
    report = publish({"actual.npy": encode((4, 2, 3), ">f8", [0.] * 23 + [float("nan")])})
    with pytest.raises(ValueError, match="nonfinite physical"):
        check(trace, report)


@pytest.mark.parametrize("field", ["done", "failure", "success"])
def test_boolean_payload_has_only_zero_and_one(trace, field):
    _, _, _, publish = trace
    report = publish({field + ".npy": encode((4, 2), "|b1", [0] * 7 + [2])})
    with pytest.raises(ValueError, match="boolean"):
        check(trace, report)


@pytest.mark.parametrize("change,reason", [
    ({"episode_id.npy": encode((4, 2), "<i8", [-1, 0, 0, 0, 1, 0, 1, 0])}, "origin"),
    ({"episode_id.npy": encode((4, 2), "<i8", [0] * 8)}, "continuity"),
    ({"done.npy": encode((4, 2), "|b1", [0] * 8)}, "terminal"),
    ({"failure.npy": encode((4, 2), "|b1", [0, 0, 1, 0, 0, 0, 0, 0])}, "terminal"),
    ({"time_s.npy": encode((4, 2), "<f8", [.01, .01, .01, .02, .01, .03, .02, .04])}, "time"),
    ({"time_s.npy": encode((4, 2), "<f8", [.01, .01, .02, .02, 0., .03, .02, .04])}, "positive"),
])
def test_episode_done_and_time_semantics(trace, change, reason):
    _, _, _, publish = trace
    with pytest.raises(ValueError, match=reason):
        check(trace, publish(change))


@pytest.mark.parametrize("field", ["seed", "checkpoint_update", "checkpoint_sha256", "control_sha256", "group_labels"])
def test_archive_metadata_must_equal_report_and_frozen_identity(trace, field):
    _, expected, _, publish = trace
    broken = deepcopy(expected)
    broken[field] = ["wrong", "second"] if field == "group_labels" else "wrong"
    report = publish({"metadata_json.npy": metadata_member(broken)})
    with pytest.raises(ValueError, match="metadata differs"):
        check(trace, report)


def test_bool_does_not_alias_metadata_integer(trace):
    _, expected, _, publish = trace
    broken = deepcopy(expected)
    broken["row_indices"] = [False, True]
    with pytest.raises(ValueError, match="metadata differs"):
        check(trace, publish({"metadata_json.npy": metadata_member(broken)}))


@pytest.mark.parametrize("change,reason", [
    ({"row_indices.npy": encode((2,), "<i8", [1, 0])}, "row indices"),
    ({"actual.npy": encode((4, 1, 3), "<f4", [0.] * 12)}, "shape"),
    ({"episode_id.npy": encode((4, 2), "<f4", [0.] * 8)}, "dtype"),
    ({"time_s.npy": encode((4, 2), "<i8", [1] * 8)}, "dtype"),
    ({"actual.npy": npy((4, 2, 3), "|O8", b"x" * 192)}, "dtype"),
    ({"actual.npy": npy((4, 2, 3), "<f4", b"\0" * 96, fortran=True)}, "C-order"),
])
def test_shapes_indices_and_dtype_schema(trace, change, reason):
    _, _, _, publish = trace
    with pytest.raises(ValueError, match=reason):
        check(trace, publish(change))


@pytest.mark.parametrize("field", ["position_xy.npy", "done.npy", "scaled_nominal_effort_bounds.npy"])
def test_missing_physical_field_fails(trace, field):
    _, _, _, publish = trace
    with pytest.raises(ValueError, match="coverage"):
        check(trace, publish({field: None}))


def test_unknown_duplicate_or_traversal_members_rejected(trace):
    _, _, _, publish = trace
    with pytest.raises(ValueError, match="coverage"):
        check(trace, publish({"invented.npy": encode((4, 2), "<f4", [0.] * 8)}))
    with pytest.warns(UserWarning, match="Duplicate"):
        report = publish(duplicate="done.npy")
    with pytest.raises(ValueError, match="ZIP members"):
        check(trace, report)
    with pytest.raises(ValueError, match="ZIP members"):
        check(trace, publish({"../done.npy": encode((4, 2), "|b1", [0] * 8)}))


@pytest.mark.parametrize("tail", [b"extra", None])
def test_exact_payload_length(trace, tail):
    _, _, members, publish = trace
    original = members["actual.npy"]
    changed = original + tail if tail is not None else original[:-1]
    with pytest.raises(ValueError, match="payload"):
        check(trace, publish({"actual.npy": changed}))


def test_actual_file_sha_is_not_self_described(trace):
    _, _, _, publish = trace
    report = publish()
    report["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="actual SHA"):
        check(trace, report)


def test_optional_command_request_still_scanned(trace):
    _, _, _, publish = trace
    report = publish({"command_request.npy": encode((4, 2, 3), "<f4", [0.] * 24)})
    check(trace, report)
    with pytest.raises(ValueError, match="nonfinite physical"):
        check(trace, publish({"command_request.npy": encode((4, 2, 3), "<f4", [0.] * 23 + [float("nan")])}))


def test_positive_but_incorrect_physical_tick_clock_is_rejected(trace):
    _, _, _, publish = trace
    with pytest.raises(ValueError, match="policy tick clock"):
        check(trace, publish({"time_s.npy": encode((4, 2), "<f8",
            [.01, .01, .03, .02, .01, .03, .02, .04])}))


def test_short_actual_arrays_cannot_claim_the_fixed_full_trace(trace):
    path, expected, _, publish = trace
    report = publish()
    expected = {**expected, "steps": 4001, "row_indices": list(range(400)),
                "group_labels": [f"case_{i // 8}" for i in range(400)]}
    report.update(expected, rows=list(range(400)))
    with pytest.raises(ValueError, match="trace shape differs"):
        evaluation.validate_trace_archive(path, report, expected, steps=4001, rows=400)


def test_zip_crc_is_checked_even_with_resealed_outer_sha(trace):
    path, _, members, publish = trace
    report = publish()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    raw = bytearray(path.read_bytes())
    # Corrupt one physical payload while retaining the original member CRC.
    offset = raw.index(members["actual.npy"])
    raw[offset + len(members["actual.npy"]) - 1] ^= 1
    path.write_bytes(raw)
    report["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="corrupt trace archive"):
        check(trace, report)


def test_symlink_trace_rejected(trace, tmp_path):
    path, expected, _, publish = trace
    report = publish()
    alias = tmp_path / "alias.npz"
    alias.symlink_to(path)
    report["path"] = str(alias)
    with pytest.raises(ValueError, match="symlink"):
        evaluation.validate_trace_archive(alias, report, expected, steps=4, rows=2)


def fixed():
    branch = {"id": "branch_original", "training_seed": 1101, "requested_k": 256,
              "coefficient": .1, "schedule": "stationary"}
    protocol = {"evaluation": {"seeds": [8701, 9701], "steps": 4001,
        "settle_steps": 200, "min_steady_samples": 200, "trace_replicas": 8},
        "evaluation_cases": [{"name": f"case_{i}"} for i in range(50)], "branches": [branch]}
    checkpoint = {"path": "/unopened/real.pt", "sha256": "a" * 64, "bytes": 10,
                  "update": 1200, "cumulative_transitions": 58982400, "consumed_updates": 1200}
    return protocol, branch, checkpoint


@pytest.mark.parametrize("field,value", [("trace_replicas", 2), ("steps", 4), ("settle_steps", 0),
    ("min_steady_samples", True)])
def test_real_suite_refuses_reduced_generic_dimensions_before_authorization(field, value):
    protocol, branch, checkpoint = fixed()
    protocol["evaluation"][field] = value
    with pytest.raises(ValueError, match="fixed original"):
        evaluation.verify_suite(protocol, branch, checkpoint, 8701, "/unopened/eval")


def test_real_suite_refuses_incomplete_case_grid_before_any_file_io():
    protocol, branch, checkpoint = fixed()
    protocol["evaluation_cases"].pop()
    with pytest.raises(ValueError, match="fifty"):
        evaluation.verify_suite(protocol, branch, checkpoint, 8701, "/unopened/eval")


@pytest.mark.parametrize("field,value", [("update", 400), ("consumed_updates", 1201),
    ("cumulative_transitions", 58982401), ("bytes", True)])
def test_terminal_checkpoint_requires_exact_nonrefunded_clock(field, value):
    protocol, branch, checkpoint = fixed()
    checkpoint[field] = value
    with pytest.raises(ValueError, match="checkpoint or consumed clock"):
        evaluation.verify_suite(protocol, branch, checkpoint, 8701, "/unopened/eval")


def test_unfrozen_lambda_zero_branch_does_not_alias_another_k():
    protocol, branch, checkpoint = fixed()
    protocol["branches"][0] = {**branch, "coefficient": 0.}
    branch = {**protocol["branches"][0], "requested_k": 512}
    with pytest.raises(ValueError, match="frozen grid"):
        evaluation.verify_suite(protocol, branch, checkpoint, 8701, "/unopened/eval")


def test_pure_original_metric_validators_are_exact_byte_sourced():
    path = evaluation.TOOLS / "run_frame_diagnostic_campaign.py"
    raw = path.read_bytes()
    pin = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    validator, receipt = evaluation._physical_validators(pin)
    assert receipt["sha256"] == hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest()
    assert "validate_metrics" in receipt["functions"]
    with pytest.raises(ValueError, match="complete control samples"):
        validator({"available": False}, 8)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", True),
    ("bytes", 1), ("path", "/a/different/tool.py")])
def test_metric_validator_requires_frozen_actual_source(field, value):
    path = evaluation.TOOLS / "run_frame_diagnostic_campaign.py"
    raw = path.read_bytes()
    pin = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    pin[field] = value
    with pytest.raises(ValueError, match="frozen physical validator"):
        evaluation._physical_validators(pin)


def test_effective_suite_contract_is_merged_from_every_actual_case(tmp_path):
    scenarios, configs, contracts = [], {}, []
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for i in range(50):
        name, route = f"case_{i}", f"case_{i}.json"
        contract = {"target_num_envs": 8, "evaluation_exact_cases": True,
            "scene_groups": [{"name": name}], "policy_dt": .01,
            "evaluation": {"cases": [{"name": name, "terrain": "flat"}], "stable_case_layout": False}}
        (snapshot / route).write_text(json.dumps(contract))
        contracts.append(contract)
        scenarios.append({"name": name, "config": name})
        configs[name] = {"environment": {"contract": route}}
    bundle = {"snapshot": snapshot, "configs": configs, "manifest": {"scenarios": scenarios}}

    class Reader:
        def read(self, path):
            return json.loads(path.read_bytes())

    merged = deepcopy(contracts[0])
    merged["target_num_envs"] = 400
    merged["evaluation"]["cases"] = [c["evaluation"]["cases"][0] for c in contracts]
    merged["scene_groups"] = [{"name": f"case_{i}", "fraction": .02, "terrain": ["flat"]} for i in range(50)]
    expected_sha = hashlib.sha256(json.dumps(merged, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    actual_sha = evaluation._merged_contract(Reader(), bundle)
    assert actual_sha == expected_sha
    assert actual_sha != evaluation._digest(contracts[0])
    changed = contracts[49] | {"policy_dt": .02}
    (snapshot / "case_49.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="beyond cases"):
        evaluation._merged_contract(Reader(), bundle)

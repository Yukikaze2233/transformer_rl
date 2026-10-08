"""Real disk-backed trace writing and CPU replay, never a simulator proof."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import zipfile

import numpy as np
import pytest
import torch

from transformer_rl.control_metrics import ControlMetrics
from transformer_rl.control_trace import ControlTrace
from transformer_rl.exposure_trace import verify_trace_archive
from transformer_rl.evaluation import _MetricAccumulator
from transformer_rl.history_control import HistoryControlStatistics
from transformer_rl.stability import EpisodeSignalStatistics


BOUNDS = [.5] * 6


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value)
    return stream.getvalue()


def make_trace(tmp_path, *, history_length=3, steps=7, command_change=None,
               terminal_pattern=None, bias_after_reset=False, include_evaluation=False):
    rows, labels, dt = 2, ["alpha", "beta"], .01
    metadata = {"checkpoint_sha256": "a" * 64, "checkpoint_update": 7, "seed": 71,
                "policy_dt_s": dt, "sampling_hz": 100., "control_sha256": "b" * 64,
                "history_length": history_length,
                "pre_inference_age_semantics": "policy steps since reset, captured before actor inference"}
    if include_evaluation:
        metadata.update(evaluation_metric_names=["height_abs_error", "vx_abs_error"],
                        evaluation_signal_names=["height_error", "vx_error"])
    path = tmp_path / "trace.npz"
    writer = ControlTrace(path, steps=steps, num_envs=rows, replicas=rows,
                          groups=labels, metadata=metadata)
    control = ControlMetrics(rows, dt, settle_steps=0, min_steady_samples=1)
    history = HistoryControlStatistics(rows, history_length, dt, settle_steps=0, min_steady_samples=1)
    group_metrics = {label: {"control": ControlMetrics(1, dt, settle_steps=0, min_steady_samples=1),
                            "history_control": HistoryControlStatistics(1, history_length, dt,
                                settle_steps=0, min_steady_samples=1)} for label in labels}
    def evaluation_state(count):
        return {"reward": _MetricAccumulator(),
                "metrics": {name: _MetricAccumulator() for name in metadata["evaluation_metric_names"]},
                "statistics": EpisodeSignalStatistics(count, settle_steps=0, min_steady_samples=1),
                "completed": 0, "successes": 0, "failures": 0}

    def evaluation_update(state, data, done):
        state["reward"].add(data["eval_reward"])
        for name, value in state["metrics"].items():
            value.add(data["eval_metric_" + name])
        signals = {name: data["eval_signal_" + name] for name in metadata["evaluation_signal_names"]}
        state["statistics"].update(signals, data["eval_signal_time"], done)
        state["completed"] += int(done.sum())
        state["successes"] += int(data["eval_episode_success"].sum())
        state["failures"] += int((done & ~data["eval_episode_success"]).sum())

    def evaluation_report(state):
        return {"metrics": {name: value.report() for name, value in state["metrics"].items()},
                "reward_mean": state["reward"].report()["mean"], "stability": state["statistics"].report(),
                "completed_episodes": state["completed"], "failed_episodes": state["failures"],
                "success_metric_available": True,
                "success_rate": state["successes"] / state["completed"] if state["completed"] else None}

    evaluations = {"all": evaluation_state(rows), **{label: evaluation_state(1) for label in labels}} if include_evaluation else {}
    ages = torch.zeros(rows, dtype=torch.int64)
    episodes = torch.zeros(rows, dtype=torch.int64)
    packets = []
    try:
        for tick in range(steps):
            done = torch.tensor(terminal_pattern[tick] if terminal_pattern is not None else
                                [tick in (2, 5), tick == 4], dtype=torch.bool)
            reference = torch.tensor([[0., 0., .3]] * rows)
            if command_change is not None and tick >= command_change:
                reference[:, 0] = .2
            actual = reference.clone()
            if bias_after_reset:
                actual[:, 0] += episodes.float()
            positions = torch.tensor([[tick * .01 + int(episodes[0]) * 100., tick * .02],
                                      [tick * .03, tick * .04 + int(episodes[1]) * 200.]])
            raw = torch.full((rows, 6), tick / 10.)
            issued = raw.clamp(-torch.tensor(BOUNDS), torch.tensor(BOUNDS))
            packet = {"time_s": (ages + 1).double() * dt,
                "command_reference": reference, "actual": actual, "position_xy": positions,
                "tilt": torch.zeros(rows), "leg_target": torch.full((rows, 4), tick / 100.),
                "wheel_target": torch.full((rows, 2), tick / 10.),
                "motor_position": torch.full((rows, 6), tick / 100.),
                "motor_velocity": torch.full((rows, 6), .1),
                "motor_effort": torch.full((rows, 6), tick / 20.),
                "requested_motor_effort": torch.full((rows, 6), tick / 10.),
                "effort_bounds": torch.tensor([[[-1., 1.]] * 6] * rows),
                "scaled_nominal_requested_motor_effort": torch.full((rows, 6), tick / 10.),
                "scaled_nominal_effort_bounds": torch.tensor([[[-1., 1.]] * 6] * rows),
                "failure": done & torch.tensor([True, False]),
                "success": done & torch.tensor([False, True])}
            control.update(packet, done)
            history.update(packet, done, ages, raw, issued)
            extra = {}
            if include_evaluation:
                vx_error = actual[:, 0] - reference[:, 0]
                height_error = actual[:, 2] - reference[:, 2]
                extra = {"eval_reward": torch.tensor([tick + .1, -tick - .2]),
                         "eval_episode_success": done & ~packet["failure"],
                         "eval_signal_time": packet["time_s"].clone(),
                         "eval_metric_height_abs_error": height_error.abs(),
                         "eval_metric_vx_abs_error": vx_error.abs(),
                         "eval_signal_height_error": height_error,
                         "eval_signal_vx_error": vx_error}
                evaluation_update(evaluations["all"], extra, done)
            for index, label in enumerate(labels):
                grouped = {name: value[index:index + 1] for name, value in packet.items()}
                group_metrics[label]["control"].update(grouped, done[index:index + 1])
                group_metrics[label]["history_control"].update(grouped, done[index:index + 1],
                    ages[index:index + 1], raw[index:index + 1], issued[index:index + 1])
                if include_evaluation:
                    evaluation_update(evaluations[label], {name: value[index:index + 1] for name, value in extra.items()},
                                      done[index:index + 1])
            writer.add({**packet, "pre_inference_episode_age": ages.clone(),
                        "raw_policy_mean": raw, "issued_action": issued, **extra}, done)
            packets.append((deepcopy(packet), done.clone(), ages.clone(), raw.clone(), issued.clone()))
            ages += 1
            ages[done] = 0
            episodes += done.to(torch.int64)
        trace = writer.publish()
    finally:
        writer.close()
    expected = {**metadata, "steps": steps, "row_indices": list(range(rows)), "group_labels": labels}
    online = {"control": control.report(), "history_control": history.report(),
              "groups": {label: {name: value.report() for name, value in metrics.items()}
                         for label, metrics in group_metrics.items()}}
    if include_evaluation:
        online.update(evaluation_report(evaluations["all"]))
        for label in labels:
            online["groups"][label].update(evaluation_report(evaluations[label]))
    return {"path": path, "trace": trace, "expected": expected, "steps": steps, "rows": rows,
            "history_length": history_length, "online": online, "packets": packets}


def verify(fixture, *, trace=None, expected=None, **kwargs):
    arguments = {"steps": fixture["steps"], "rows": fixture["rows"],
                 "history_length": fixture["history_length"], "action_bounds": BOUNDS,
                 "settle_steps": 0, "min_steady_samples": 1, **kwargs}
    return verify_trace_archive(fixture["path"], trace or fixture["trace"],
                                expected or fixture["expected"], **arguments)


def rewrite(fixture, changes, *, extra_members=(), compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(fixture["path"]) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    for name, value in changes.items():
        if value is None:
            members.pop(name)
        else:
            members[name] = value if isinstance(value, bytes) else _npy(value)
    with zipfile.ZipFile(fixture["path"], "w", compression=compression) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
        for name, value in extra_members:
            archive.writestr(name, value)
    report = deepcopy(fixture["trace"])
    report["sha256"] = hashlib.sha256(fixture["path"].read_bytes()).hexdigest()
    report["fields"] = [name[:-4] for name in members if name not in ("metadata_json.npy", "row_indices.npy")]
    return report


def array(fixture, field):
    with zipfile.ZipFile(fixture["path"]) as archive:
        return np.load(io.BytesIO(archive.read(field + ".npy")), allow_pickle=False)


def test_real_complete_trace_replay_matches_online_metrics_exactly(tmp_path):
    fixture = make_trace(tmp_path)
    replay = verify(fixture)
    assert {key: replay[key] for key in fixture["online"]} == fixture["online"]
    receipt = replay["trace_validation"]
    assert receipt["complete_payload_and_crc_checked"]
    assert receipt["pre_inference_history_age_checked"]
    assert receipt["hardware_verified"] is False
    assert fixture["online"]["control"]["full_interval"]["samples"] == 14


def test_declared_task_metrics_reward_success_and_stability_replay_exactly(tmp_path):
    fixture = make_trace(tmp_path, include_evaluation=True, bias_after_reset=True)
    replay = verify(fixture)
    assert {key: replay[key] for key in fixture["online"]} == fixture["online"]
    assert replay["metrics"]["vx_abs_error"]["count"] == 14
    assert replay["completed_episodes"] == 3 and replay["failed_episodes"] == 2
    assert replay["groups"]["beta"]["success_rate"] == 1.


@pytest.mark.parametrize("field", ["eval_reward", "eval_metric_height_abs_error", "eval_signal_vx_error",
                                   "eval_signal_time"])
def test_task_trace_payload_is_fully_finite_scanned(tmp_path, field):
    fixture = make_trace(tmp_path, include_evaluation=True)
    broken = array(fixture, field)
    broken[-1, -1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": broken}))


def test_declared_metric_cannot_be_missing_or_replaced_by_unlisted_dynamic_field(tmp_path):
    fixture = make_trace(tmp_path, include_evaluation=True)
    with pytest.raises(ValueError, match="field coverage"):
        verify(fixture, trace=rewrite(fixture, {"eval_metric_height_abs_error.npy": None,
                                               "eval_metric_fake.npy": np.zeros((7, 2))}))


def test_dynamic_evaluation_fields_require_explicit_authoritative_declarations(tmp_path):
    fixture = make_trace(tmp_path)
    with pytest.raises(ValueError, match="field coverage"):
        verify(fixture, trace=rewrite(fixture, {"eval_reward.npy": np.zeros((7, 2))}))


def test_evaluation_success_and_signal_clock_cannot_be_rebound(tmp_path):
    fixture = make_trace(tmp_path, include_evaluation=True)
    broken = array(fixture, "eval_episode_success")
    broken[0, 0] = True
    with pytest.raises(ValueError, match="evaluation success"):
        verify(fixture, trace=rewrite(fixture, {"eval_episode_success.npy": broken}))
    fixture = make_trace(tmp_path / "clock", include_evaluation=True)
    broken = array(fixture, "eval_signal_time")
    broken[0, 0] += 1e-8
    with pytest.raises(ValueError, match="signal time differs"):
        verify(fixture, trace=rewrite(fixture, {"eval_signal_time.npy": broken}))


@pytest.mark.parametrize("names", [["../escape"], ["duplicated", "duplicated"], ["z", "a"], [True]])
def test_dynamic_names_reject_paths_duplicates_unsorted_or_bool(tmp_path, names):
    fixture = make_trace(tmp_path, include_evaluation=True)
    expected = deepcopy(fixture["expected"])
    expected["evaluation_metric_names"] = names
    with pytest.raises(ValueError, match="unique sorted safe"):
        verify(fixture, expected=expected)


def test_h1_empty_reset_window_has_null_metrics_and_full_denominator(tmp_path):
    result = verify(make_trace(tmp_path, history_length=1))
    reset = result["history_control"]["windows"]["reset_filled"]
    full = result["history_control"]["windows"]["full_history"]
    assert reset["samples"] == 0 and full["samples"] == 14
    assert reset["tracking"]["axes"]["vx"]["mae"] is None
    assert reset["rates"]["issued_action_rate"][0]["count"] == 0
    assert reset["rates"]["issued_action_rate"][0]["rms"] is None
    assert reset["actor_clipping_fraction"] == [None] * 6


def test_pre_inference_age_boundary_and_asynchronous_reset_counts(tmp_path):
    fixture = make_trace(tmp_path, history_length=3)
    result = verify(fixture)
    # Row alpha ages: 0,1,2(done),0,1,2(done),0. Beta: 0,1,2,3,4(done),0,1.
    alpha = result["groups"]["alpha"]["history_control"]["windows"]
    beta = result["groups"]["beta"]["history_control"]["windows"]
    assert alpha["full_history"]["samples"] == 2
    assert alpha["reset_filled"]["samples"] == 5
    assert alpha["full_history"]["terminal_events"]["failure"] == 2
    assert beta["full_history"]["samples"] == 3
    assert beta["reset_filled"]["samples"] == 4
    assert beta["full_history"]["rates"]["issued_action_rate"][0]["count"] == 2
    assert alpha["full_history"]["rates"]["issued_action_rate"][0]["count"] == 0
    assert result["control"]["planar_motion"]["full_interval"]["max_speed_m_s"] < 6.


def test_short_episodes_never_gain_full_history_but_keep_all_samples(tmp_path):
    fixture = make_trace(tmp_path, history_length=4, steps=6,
                         terminal_pattern=[[False, False], [True, True]] * 3)
    result = verify(fixture)
    windows = result["history_control"]["windows"]
    assert windows["full_history"]["samples"] == 0
    assert windows["reset_filled"]["samples"] == 12
    assert windows["full_history"]["tracking"]["axes"]["vx"]["within_group_std"] is None
    assert result["control"]["full_interval"]["samples"] == 12


def test_episode_offsets_change_bias_without_becoming_within_episode_jitter(tmp_path):
    result = verify(make_trace(tmp_path, history_length=1, bias_after_reset=True))
    vx = result["history_control"]["windows"]["full_history"]["tracking"]["axes"]["vx"]
    assert vx["within_group_std"] == 0.
    assert vx["bias"] > 0. and vx["group_mean_std"] > 0.


def test_command_boundary_excludes_actor_and_target_rate_interval(tmp_path):
    fixture = make_trace(tmp_path, history_length=1, steps=4, command_change=2,
                         terminal_pattern=[[False, False]] * 4)
    result = verify(fixture)
    full = result["history_control"]["windows"]["full_history"]
    assert full["samples"] == 8
    assert full["rates"]["issued_action_rate"][0]["count"] == 4
    assert full["physical_planar_motion"]["intervals"] == 4


@pytest.mark.parametrize("field", ["actual", "motor_effort", "raw_policy_mean", "issued_action",
                                   "scaled_nominal_effort_bounds", "time_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_float_array_is_scanned_even_with_resealed_outer_sha(tmp_path, field, value):
    fixture = make_trace(tmp_path)
    broken = array(fixture, field)
    broken.reshape(-1)[-1] = value
    with pytest.raises(ValueError, match="nonfinite"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": broken}))


@pytest.mark.parametrize("field", ["pre_inference_episode_age", "episode_id"])
def test_integer_age_and_episode_must_match_terminal_continuity(tmp_path, field):
    fixture = make_trace(tmp_path)
    broken = array(fixture, field)
    broken[3, 0] += 1
    with pytest.raises(ValueError, match="continuity"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": broken}))


@pytest.mark.parametrize("field", ["pre_inference_episode_age", "episode_id"])
def test_age_episode_origin_and_nonnegative_values(tmp_path, field):
    fixture = make_trace(tmp_path)
    broken = array(fixture, field)
    broken[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": broken}))


def test_same_episode_positive_but_wrong_policy_clock_rejected(tmp_path):
    fixture = make_trace(tmp_path)
    broken = array(fixture, "time_s")
    broken[1, 0] += .005
    with pytest.raises(ValueError, match="tick clock"):
        verify(fixture, trace=rewrite(fixture, {"time_s.npy": broken}))


def test_terminal_events_cannot_appear_before_done(tmp_path):
    fixture = make_trace(tmp_path)
    broken = array(fixture, "failure")
    broken[0, 0] = True
    with pytest.raises(ValueError, match="terminal flags"):
        verify(fixture, trace=rewrite(fixture, {"failure.npy": broken}))


def test_issued_actions_must_equal_authorized_mean_clamp(tmp_path):
    fixture = make_trace(tmp_path)
    broken = array(fixture, "issued_action")
    broken[-1, 0, 0] = .49
    with pytest.raises(ValueError, match="raw mean clamp"):
        verify(fixture, trace=rewrite(fixture, {"issued_action.npy": broken}))


@pytest.mark.parametrize("field", ["raw_policy_mean", "pre_inference_episode_age", "motor_effort"])
def test_missing_required_fields_cannot_be_redeclared_as_optional(tmp_path, field):
    fixture = make_trace(tmp_path)
    with pytest.raises(ValueError, match="field coverage"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": None}))


@pytest.mark.parametrize("field,dtype", [("actual", object), ("actual", np.complex64),
                                        ("pre_inference_episode_age", np.int32),
                                        ("time_s", np.float32), ("done", np.uint8)])
def test_no_object_pickle_complex_unsigned_or_wrong_precision_schema(tmp_path, field, dtype):
    fixture = make_trace(tmp_path)
    broken = array(fixture, field).astype(dtype)
    with pytest.raises(ValueError, match="dtype|float64"):
        verify(fixture, trace=rewrite(fixture, {field + ".npy": broken}))


def test_big_endian_arrays_replay_without_precision_changes(tmp_path):
    fixture = make_trace(tmp_path)
    changes = {field + ".npy": array(fixture, field).astype(
        array(fixture, field).dtype.newbyteorder(">")) for field in
        ("actual", "time_s", "pre_inference_episode_age", "episode_id")}
    result = verify(fixture, trace=rewrite(fixture, changes))
    assert result["control"] == fixture["online"]["control"]
    assert result["history_control"] == fixture["online"]["history_control"]


def test_shape_cannot_claim_more_complete_rows_or_steps_than_written(tmp_path):
    fixture = make_trace(tmp_path)
    broken = array(fixture, "actual")[:-1]
    with pytest.raises(ValueError, match="shape differs"):
        verify(fixture, trace=rewrite(fixture, {"actual.npy": broken}))


@pytest.mark.parametrize("tail", [b"tail", b"\x00"])
def test_extra_member_payload_bytes_rejected(tmp_path, tail):
    fixture = make_trace(tmp_path)
    broken = _npy(array(fixture, "actual")) + tail
    with pytest.raises(ValueError, match="payload length"):
        verify(fixture, trace=rewrite(fixture, {"actual.npy": broken}))


@pytest.mark.parametrize("side", ["prefix", "suffix", "between_members_and_directory"])
def test_unclaimed_whole_zip_bytes_rejected_after_resealed_sha(tmp_path, side):
    fixture = make_trace(tmp_path)
    raw = bytearray(fixture["path"].read_bytes())
    if side == "prefix":
        raw = bytearray(b"prefix") + raw
    elif side == "suffix":
        raw.extend(b"suffix")
    else:
        central_offset = struct.unpack_from("<I", raw, len(raw) - 6)[0]
        raw[central_offset:central_offset] = b"x"
        struct.pack_into("<I", raw, len(raw) - 6, central_offset + 1)
    fixture["path"].write_bytes(raw)
    report = {**fixture["trace"], "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="prefix|trailing payload|unclaimed payload"):
        verify(fixture, trace=report)


@pytest.mark.parametrize("corrupt", [False, True])
def test_streaming_zip_data_descriptors_are_checked_and_replayed(tmp_path, corrupt):
    fixture = make_trace(tmp_path)
    with zipfile.ZipFile(fixture["path"]) as archive:
        members = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]

    class Output(io.BytesIO):
        def seek(self, *_):
            raise io.UnsupportedOperation("write-only streaming fixture")

    output = Output()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members:
            archive.writestr(name, raw)
    payload = bytearray(output.getvalue())
    if corrupt:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = archive.infolist()[0]
        filename, extra = struct.unpack_from("<HH", payload, info.header_offset + 26)
        descriptor = info.header_offset + 30 + filename + extra + info.compress_size
        payload[descriptor + 4] ^= 1
    fixture["path"].write_bytes(payload)
    report = {**fixture["trace"], "sha256": hashlib.sha256(payload).hexdigest()}
    with zipfile.ZipFile(fixture["path"]) as archive:
        assert all(info.flag_bits & 8 for info in archive.infolist())
    if corrupt:
        with pytest.raises(ValueError, match="descriptor differs"):
            verify(fixture, trace=report)
    else:
        replay = verify(fixture, trace=report)
        assert replay["control"] == fixture["online"]["control"]


def test_zip64_end_records_are_checked_without_allocating_large_arrays(tmp_path):
    fixture = make_trace(tmp_path)
    payload = fixture["path"].read_bytes()
    end = struct.unpack("<4s4H2IH", payload[-22:])
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, end[3], end[4], end[5], end[6])
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(payload) - 22, 1)
    legacy = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 2**32 - 1, 2**32 - 1, 0)
    payload = payload[:-22] + record + locator + legacy
    fixture["path"].write_bytes(payload)
    report = {**fixture["trace"], "sha256": hashlib.sha256(payload).hexdigest()}
    result = verify(fixture, trace=report)
    assert result["control"] == fixture["online"]["control"]


def test_unknown_duplicate_and_traversal_members_rejected(tmp_path):
    fixture = make_trace(tmp_path)
    with pytest.raises(ValueError, match="field coverage"):
        verify(fixture, trace=rewrite(fixture, {"invented.npy": np.zeros((7, 2))}))
    fixture = make_trace(tmp_path / "duplicate")
    with pytest.warns(UserWarning, match="Duplicate"):
        report = rewrite(fixture, {}, extra_members=[("actual.npy", _npy(array(fixture, "actual")))])
    with pytest.raises(ValueError, match="ZIP members"):
        verify(fixture, trace=report)
    fixture = make_trace(tmp_path / "traversal")
    report = rewrite(fixture, {}, extra_members=[("../actual.npy", _npy(np.zeros((7, 2, 3))) )])
    with pytest.raises(ValueError, match="ZIP members"):
        verify(fixture, trace=report)


def test_boolean_byte_values_must_be_zero_or_one(tmp_path):
    fixture = make_trace(tmp_path)
    broken = bytearray(_npy(array(fixture, "done")))
    broken[-1] = 2
    with pytest.raises(ValueError, match="boolean"):
        verify(fixture, trace=rewrite(fixture, {"done.npy": bytes(broken)}))


def test_archive_metadata_and_report_identity_must_both_match_expected(tmp_path):
    fixture = make_trace(tmp_path)
    metadata = dict(fixture["trace"])
    metadata = {key: value for key, value in metadata.items() if key not in {"path", "sha256", "fields", "rows"}}
    metadata["seed"] = 72
    report = rewrite(fixture, {"metadata_json.npy": np.array(json.dumps(metadata))})
    with pytest.raises(ValueError, match="archive metadata"):
        verify(fixture, trace=report)
    report["seed"] = 72
    with pytest.raises(ValueError, match="provenance"):
        verify(fixture, trace=report)


def test_crc_corruption_rejected_even_when_outer_sha_is_resealed(tmp_path):
    fixture = make_trace(tmp_path)
    report = rewrite(fixture, {}, compression=zipfile.ZIP_STORED)
    with zipfile.ZipFile(fixture["path"]) as archive:
        info = archive.getinfo("actual.npy")
    raw = bytearray(fixture["path"].read_bytes())
    offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    raw[offset + info.file_size - 1] ^= 1
    fixture["path"].write_bytes(raw)
    report["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="corrupt trace"):
        verify(fixture, trace=report)


def test_symlink_paths_and_special_zip_members_rejected(tmp_path):
    fixture = make_trace(tmp_path)
    alias = tmp_path / "alias.npz"
    alias.symlink_to(fixture["path"])
    with pytest.raises(ValueError, match="symlink"):
        verify_trace_archive(alias, {**fixture["trace"], "path": str(alias)}, fixture["expected"],
            steps=7, rows=2, history_length=3, action_bounds=BOUNDS, settle_steps=0, min_steady_samples=1)
    with zipfile.ZipFile(fixture["path"]) as archive:
        members = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(fixture["path"], "w") as archive:
        for name, raw in members:
            info = zipfile.ZipInfo(name)
            if name == "actual.npy":
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, raw)
    report = {**fixture["trace"], "sha256": hashlib.sha256(fixture["path"].read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="file type"):
        verify(fixture, trace=report)


def test_optional_command_request_is_scanned_and_replayed_as_request(tmp_path):
    fixture = make_trace(tmp_path)
    request = array(fixture, "command_reference").copy()
    request[:, :, 0] += .2
    result = verify(fixture, trace=rewrite(fixture, {"command_request.npy": request}))
    assert result["control"]["request_reference_difference"]["available"]
    assert result["control"]["request_reference_difference"]["axes"]["vx"]["mean"] == pytest.approx(-.2)


@pytest.mark.parametrize("argument,value", [("steps", True), ("rows", 0), ("history_length", False),
                                           ("settle_steps", -1), ("min_steady_samples", 0),
                                           ("action_bounds", [float("nan")] * 6)])
def test_invalid_authorized_arguments_rejected_before_replay(tmp_path, argument, value):
    fixture = make_trace(tmp_path)
    with pytest.raises(ValueError):
        verify(fixture, **{argument: value})


def test_new_dynamic_steps_are_not_limited_to_old_retention_campaign(tmp_path):
    fixture = make_trace(tmp_path, history_length=1, steps=4002,
                         terminal_pattern=[[True, True]] * 4002)
    result = verify(fixture)
    assert result["control"]["full_interval"]["samples"] == 8004
    assert result["trace_validation"]["steps"] == 4002

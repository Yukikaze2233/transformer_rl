"""Frozen paired diagnostics and resource ownership, without a simulator."""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest


MODULE = Path(__file__).parents[1] / "tools/run_frame_diagnostic_campaign.py"
SPEC = importlib.util.spec_from_file_location("frame_diagnostic_campaign", MODULE)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
control = campaign.control


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    study, output, source, snapshot = [tmp_path / name for name in ("study", "output", "source", "snapshot")]
    (source / "src/transformer_rl").mkdir(parents=True)
    (source / "src/transformer_rl/frame_process.py").write_text("# synthetic package; never executed\n")
    identity = campaign.source_identity(source)
    # Production pins the real commit's Git-blob digest. These tests declare a
    # different, small source package explicitly instead of weakening the check.
    monkeypatch.setattr(campaign, "SOURCE_PACKAGE_SHA256", identity["sha256"])
    identity["git_head"] = campaign.SOURCE_COMMIT
    origin = tmp_path / "source_origin.json"
    campaign.write(origin, identity)
    names = [f"variant_{index:02d}" for index in range(10)]
    cases = [f"case_{index:02d}" for index in range(50)]
    snapshot.mkdir()
    files = {}
    for case in cases:
        path = snapshot / f"{case}.json"
        campaign.write(path, {"evaluation_exact_cases": True, "target_num_envs": 8,
            "evaluation": {"cases": [{"name": case}]}})
        files[path.name] = control.file_sha(path)
    snapshot_sha = control.digest(files)
    campaign.write(snapshot / "snapshot.json", {"files": files, "sha256": snapshot_sha})
    plan = {"spec": {"variants": [{"name": name} for name in names], "seeds": [1101],
        "stages": [{"name": "transfer", "updates": 1200, "scenarios": cases}]}, "configs": {}}
    ctl = {"policy_dt_s": .01, "action_names": ["leg0", "leg1", "leg2", "leg3", "wheel0", "wheel1"]}
    for name in names:
        for case in cases:
            route = f"configs/{name}.eval.{case}.json"
            config = {"model": {"name": name}, "control": ctl, "environment": {
                "snapshot": str(snapshot), "snapshot_sha256": snapshot_sha,
                "contract": f"{case}.json", "contract_sha256": files[f"{case}.json"], "num_envs": 8}}
            campaign.write(study / route, config)
            plan["configs"][route] = control.digest(config)
    plan["sha256"] = control.digest(plan)
    campaign.write(study / "plan.json", plan)
    (study / ".run.lock").touch()
    for name in names:
        directory = study / "jobs" / name / "seed_1101/transfer/attempt_0000/train"
        checkpoint = directory / "checkpoints/final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(name.encode())
        campaign.write(str(checkpoint) + ".json", {"update": 1200, "sha256": control.file_sha(checkpoint),
            "config": {"control": ctl}, "metadata": {"environment_provenance": {"identity": snapshot_sha}}})
        campaign.write(directory / "completion.json", {"final_update": 1200, "checkpoint": str(checkpoint),
            "checkpoint_sha256": control.file_sha(checkpoint)})
        campaign.write(study / "jobs" / name / "seed_1101/state.json", {"plan_sha256": plan["sha256"],
            "variant": name, "seed": 1101, "status": "rejected", "stages": [{"name": "transfer", "attempts": [{
                "budget_charged": True, "training": control.artifact(directory / "completion.json", study),
                "checkpoint": control.artifact(checkpoint, study)}]}]})
    resource = tmp_path / "predecessor/.run.lock"
    resource.parent.mkdir()
    resource.touch()
    dependencies = {}
    for name in ("transfer", "curriculum"):
        root = tmp_path / name
        definition = {"resource_lock": str(resource), "study_root": str(study), "name": name}
        campaign.write(root / "campaign.json", definition)
        digest = control.digest(definition)
        campaign.write(root / "summary.json", {"status": "completed", "campaign_sha256": digest})
        dependencies[name] = (root / "summary.json", digest)
    args = argparse.Namespace(study_root=study, output_root=output, source_root=source,
        source_identity=origin, resource_lock=resource, training_seed=1101, device="cpu",
        transfer_receipt=dependencies["transfer"][0], transfer_campaign_sha256=dependencies["transfer"][1],
        curriculum_receipt=dependencies["curriculum"][0], curriculum_campaign_sha256=dependencies["curriculum"][1])
    manifest = campaign.prepare(args)
    runner = argparse.Namespace(manifest=output / "manifest.json", max_wait_seconds=0.,
        worker_timeout_seconds=.1, poll_seconds=.01)
    return args, runner, manifest


def metrics(rows):
    samples = 4001 * rows
    def pool(available):
        return {"available": available, "intervals": 4000 * rows if available else 0,
            "observed_duration_s": 40. * rows if available else 0., "path_length_m": 0.,
            "mean_speed_m_s": 0. if available else None, "rms_speed_m_s": 0. if available else None,
            "max_speed_m_s": 0. if available else None, "p95_speed_m_s": .0005 if available else None,
            "p95_bin_m_s": [0., .001] if available else [None, None], "p95_method": "duration-weighted histogram",
            "p95_bin_width_m_s": .001, "p95_overflow_from_m_s": 10., "overflow_duration_s": 0.,
            "velocity_world": {axis: {"mean_m_s": 0. if available else None, "rms_m_s": 0. if available else None}
                               for axis in ("vx", "vy")}}
    stationary = {**pool(False), "samples": 0, "runs": 0, "reference": "zero effective planar command",
        "origin": "first stationary sample", "endpoint_displacement_m": {"count": 0, "mean": None, "rms": None, "mean_abs": None, "max_abs": None},
        "max_excursion_m": {"count": 0, "mean": None, "rms": None, "mean_abs": None, "max_abs": None}}
    return {"available": True, "full_interval": {"samples": samples},
        "protocol": {"policy_dt_s": .01, "settle_steps": 200, "min_steady_samples": 200},
        "planar_motion": {"coordinate_frame": "world_xy", "num_envs": rows, "physical_samples": samples,
            "velocity_source": "consecutive PRE-reset XY samples", "scope": "all rows", "weighting": "observed duration",
            "steady_eligibility": "unchanged reference segments", "full_interval": pool(True),
            "steady": pool(False), "stationary": stationary, "stationary_steady": pool(False)},
        "actuation": {"sample_count": samples, "scaled_nominal_envelope": {"available": True, "sample_count": samples,
            "active_bound_samples": [samples] * 6, "applied_at_bound_fraction": [0.] * 6,
            "requested_outside_bounds_fraction": [0.] * 6, "applied_outside_bounds_fraction": [0.] * 6}}}


def npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


@pytest.fixture(scope="module")
def trace_arrays(tmp_path_factory):
    """Compress physical arrays once; suite-specific metadata stays separate."""
    path = tmp_path_factory.mktemp("diagnostic_trace") / "arrays.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for field, suffix in campaign.TRACE_SHAPES.items():
            dtype = np.bool_ if field in ("done", "failure", "success") else np.int64 if field == "episode_id" else np.float32
            value = np.zeros((4001, 100, *suffix), dtype=dtype)
            if field == "time_s":
                value[:] = (np.arange(1, 4002, dtype=np.float32) * .01)[:, None]
            archive.writestr(field + ".npy", npy_bytes(value))
    return path


def write_outputs(directory, manifest, variant, seed, trace_arrays, *, corrupt=None):
    cases = manifest["inputs"]["cases"]
    checkpoint = manifest["inputs"]["checkpoints"][variant]
    identity = {"checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": 1200,
                "seed": seed, "steps": 4001}
    groups = {case: metrics(8) for case in cases}
    if corrupt == "scaled":
        groups[cases[0]]["actuation"]["scaled_nominal_envelope"]["available"] = False
    if corrupt == "planar":
        del groups[cases[0]]["planar_motion"]
    for case in cases:
        report = {**identity, "num_envs": 8, "transitions": 32008, "control": groups[case],
            "environment": manifest["inputs"]["environments"][variant][case]}
        if corrupt == "update":
            report["checkpoint_update"] = 1201
        campaign.write(directory / f"{case}.json", report)
    labels = [case for case in cases for _ in range(8)]
    rows = [index for index in range(400) if index % 8 < 2]
    metadata = {**identity, "policy_dt_s": .01, "sampling_hz": 100.,
        "control_sha256": checkpoint["control_sha256"], "row_indices": rows,
        "group_labels": [labels[index] for index in rows]}
    if corrupt == "trace_seed":
        metadata["seed"] += 1
    trace = directory / "trace.npz"
    # Copy the immutable compressed member payloads directly; no repeated 120 MB
    # compression is needed for the twenty-suite resume test.
    if corrupt in {"trace_field", "trace_shape", "trace_nan"}:
        with zipfile.ZipFile(trace, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for field, suffix in campaign.TRACE_SHAPES.items():
                if corrupt == "trace_field" and field == "scaled_nominal_effort_bounds":
                    continue
                dtype = np.bool_ if field in ("done", "failure", "success") else np.int64 if field == "episode_id" else np.float32
                value = np.zeros((1 if corrupt == "trace_shape" else 4001, 100, *suffix), dtype=dtype)
                if field == "time_s":
                    value[:] = (np.arange(1, len(value) + 1, dtype=np.float32) * .01)[:, None]
                if corrupt == "trace_nan" and field == "position_xy":
                    value[0, 0, 0] = np.nan
                archive.writestr(field + ".npy", npy_bytes(value))
    else:
        trace.write_bytes(trace_arrays.read_bytes())
    with zipfile.ZipFile(trace, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata_json.npy", npy_bytes(np.array(json.dumps(metadata))))
        if corrupt != "trace_rows":
            archive.writestr("row_indices.npy", npy_bytes(np.asarray(rows, dtype=np.int64)))
    trace_receipt = {**metadata, "path": str(trace), "sha256": control.file_sha(trace),
                     "fields": list(campaign.TRACE_SHAPES)}
    report = {**identity, "groups": groups, "control": metrics(400), "trace": trace_receipt,
        "environment_provenance": {"identity": next(iter(manifest["inputs"]["snapshots"].values()))["sha256"],
                                   "evaluation_groups": labels}}
    if corrupt == "group":
        report["groups"].pop(cases[-1])
    campaign.write(directory / "control.json", report)


def synthetic_worker(manifest, calls, trace_arrays, *, fail=False):
    def worker(command, directory, environment, timeout, heartbeat):
        calls.append(command)
        assert command[3] == "evaluate-suite"
        assert "--resume" not in command and "--updates" not in command
        option = lambda name: command[command.index(name) + 1]
        variant = next(name for name, value in manifest["inputs"]["checkpoints"].items()
                       if value["checkpoint"] == option("--checkpoint"))
        if fail:
            (directory / "worker.log").write_text("retained partial evidence")
            return {"returncode": 1, "timed_out": True}
        write_outputs(directory, manifest, variant, int(option("--seed")), trace_arrays)
        for path in (Path(manifest["resource_lock"]), Path(manifest["inputs"]["study_root"]) / ".run.lock"):
            with path.open("r+") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"returncode": 0, "timed_out": False}
    return worker


def test_prepare_validate_seal_exact_inputs_without_starting_worker(prepared, monkeypatch):
    args, runner, manifest = prepared
    monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("preparation/validation launched a worker"))
    assert campaign.validate(runner.manifest) == manifest
    assert len(manifest["inputs"]["checkpoints"]) == 10
    assert all(value["update"] == 1200 for value in manifest["inputs"]["checkpoints"].values())
    assert manifest["protocol"]["seed_role"] == "paired_diagnostic_retest_of_existing_control_seeds"
    assert not manifest["protocol"]["independent_holdout"]
    assert campaign.main(["validate", "--manifest", str(runner.manifest)]) == 0


@pytest.mark.parametrize("target", ["source", "origin", "controller", "checkpoint", "completion", "config", "snapshot"])
def test_changed_sealed_input_rejected(prepared, target, monkeypatch):
    args, runner, manifest = prepared
    first = manifest["inputs"]["checkpoints"][manifest["inputs"]["variants"][0]]
    paths = {"source": args.source_root / "src/transformer_rl/frame_process.py", "origin": args.source_identity,
        "checkpoint": Path(first["checkpoint"]), "completion": Path(first["completion"]["path"]),
        "config": Path(next(iter(manifest["inputs"]["configs"].values()))["path"]),
        "snapshot": Path(next(iter(manifest["inputs"]["snapshots"].values()))["root"]) / "case_00.json"}
    if target == "controller":
        monkeypatch.setattr(campaign, "controller_identity", lambda: {})
    else:
        paths[target].write_text("changed sealed input")
    monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("changed inputs launched evaluation"))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        campaign.run(runner)


@pytest.mark.parametrize("dependency", ["transfer", "curriculum"])
def test_dependency_identity_and_completion_required(prepared, dependency, monkeypatch):
    _, runner, manifest = prepared
    path = Path(manifest["dependencies"][dependency]["receipt"])
    calls = []
    monkeypatch.setattr(control, "run_worker", lambda *_: calls.append("unexpected"))
    value = campaign.read(path)
    value["status"] = "evaluating"
    campaign.write(path, value, replace=True)
    assert campaign.run(runner)["status"] == "blocked" and not calls
    value.update(status="completed", campaign_sha256="a" * 64)
    campaign.write(path, value, replace=True)
    with pytest.raises(ValueError, match="another campaign"):
        campaign.run(runner)
    assert not calls


@pytest.mark.parametrize("lock_name", ["resource", "study"])
def test_existing_shared_and_study_locks_block_launch(prepared, lock_name, monkeypatch):
    _, runner, manifest = prepared
    path = Path(manifest["resource_lock"]) if lock_name == "resource" else Path(manifest["inputs"]["study_root"]) / ".run.lock"
    monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("held lock launched evaluation"))
    with path.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert campaign.run(runner)["status"] == "blocked"


@pytest.mark.parametrize("status,location", [("running", "campaign"), ("finished", "campaign"), ("finished", "study")])
def test_real_live_worker_is_nonterminal_and_never_killed(prepared, status, location, monkeypatch):
    _, runner, manifest = prepared
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path = (Path(manifest["dependencies"]["curriculum"]["receipt"]).parent / "attempt_0000/worker.process.json"
                if location == "campaign" else Path(manifest["inputs"]["study_root"]) / "jobs/other/train.log.process.json")
        campaign.write(path, {"pid": process.pid, "start": control.process_start(process.pid), "status": status})
        monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("live worker launched evaluation"))
        assert campaign.run(runner)["status"] == "blocked"
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_inputs_rechecked_after_resource_acquisition(prepared, monkeypatch):
    args, runner, _ = prepared
    original = campaign.acquire_resources
    @contextmanager
    def changed(manifest, budgets, publish):
        with original(manifest, budgets, publish) as value:
            (args.source_root / "src/transformer_rl/frame_process.py").write_text("# changed while waiting\n")
            yield value
    monkeypatch.setattr(campaign, "acquire_resources", changed)
    monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("queued source change launched evaluation"))
    with pytest.raises(ValueError, match="source differs"):
        campaign.run(runner)


@pytest.mark.parametrize("corrupt", ["scaled", "planar", "update", "group", "trace_seed", "trace_rows", "trace_field", "trace_shape", "trace_nan"])
def test_new_fields_group_coverage_and_trace_identity_are_required(prepared, trace_arrays, corrupt):
    _, _, manifest = prepared
    directory = Path(manifest["output_root"]) / "synthetic"
    directory.mkdir()
    variant = manifest["inputs"]["variants"][0]
    write_outputs(directory, manifest, variant, 8701, trace_arrays, corrupt=corrupt)
    with pytest.raises((ValueError, KeyError)):
        campaign.verify_outputs(directory, manifest, variant, 8701)


def test_successful_twenty_suites_resume_and_recheck_sealed_outputs(prepared, trace_arrays, monkeypatch):
    args, runner, manifest = prepared
    frozen_roots = [args.study_root, args.source_root, args.transfer_receipt.parent, args.curriculum_receipt.parent]
    before = {str(path): control.file_sha(path) for root in frozen_roots for path in root.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(manifest, calls, trace_arrays))
    result = campaign.run(runner)
    assert result["status"] == "completed" and len(calls) == 20
    assert campaign.run(runner)["status"] == "completed" and len(calls) == 20
    after = {str(path): control.file_sha(path) for root in frozen_roots for path in root.rglob("*") if path.is_file()}
    assert before == after
    first = next(iter(result["results"].values()))
    path = args.output_root / first["directory"] / "case_00.json"
    value = campaign.read(path)
    value["control"]["actuation"]["scaled_nominal_envelope"]["available"] = False
    campaign.write(path, value, replace=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.run(runner)
    assert len(calls) == 20


def test_failed_timeout_attempts_preserved_and_retry_uses_new_directory(prepared, trace_arrays, monkeypatch):
    args, runner, manifest = prepared
    calls = []
    monkeypatch.setattr(control, "run_worker", synthetic_worker(manifest, calls, trace_arrays, fail=True))
    assert campaign.run(runner)["status"] == "incomplete"
    receipts = {str(path): path.read_bytes() for path in args.output_root.rglob("receipt.json")}
    assert len(receipts) == 20
    assert all(campaign.read(path)["status"] == "failed" for path in receipts)
    assert campaign.run(runner)["status"] == "incomplete"
    assert len(list(args.output_root.rglob("receipt.json"))) == 40
    assert all(Path(path).read_bytes() == data for path, data in receipts.items())


def test_real_worker_timeout_cannot_produce_completed_receipt(prepared, monkeypatch):
    args, runner, manifest = prepared
    original = control.run_worker
    def timed_out(command, directory, environment, timeout, heartbeat):
        return original([sys.executable, "-c", "import time; time.sleep(30)"], directory,
                        environment, .02, heartbeat)
    monkeypatch.setattr(control, "run_worker", timed_out)
    result = campaign.run(runner)
    assert result["status"] == "incomplete"
    for receipt in result["results"].values():
        assert receipt["status"] == "failed" and receipt["worker"]["timed_out"]
        assert control.process_start(receipt["worker"]["pid"]) is None


def test_wrong_source_origin_commit_and_nonexact_checkpoint_rejected(prepared):
    args, _, manifest = prepared
    origin = campaign.read(args.source_identity)
    origin["git_head"] = "a" * 40
    campaign.write(args.source_identity, origin, replace=True)
    args.output_root = args.output_root.parent / "second_output"
    with pytest.raises(ValueError, match="9b576e7"):
        campaign.prepare(args)
    origin["git_head"] = campaign.SOURCE_COMMIT
    campaign.write(args.source_identity, origin, replace=True)
    first = next(iter(manifest["inputs"]["checkpoints"].values()))
    path = Path(first["completion"]["path"])
    completion = campaign.read(path)
    completion["final_update"] = 1201
    campaign.write(path, completion, replace=True)
    state_path = Path(first["training_state"]["path"])
    state = campaign.read(state_path)
    state["stages"][0]["attempts"][0]["training"] = control.artifact(path, args.study_root)
    campaign.write(state_path, state, replace=True)
    with pytest.raises(ValueError, match="exact sealed 1200"):
        campaign.prepare(args)


def test_refreshed_dirty_source_receipt_cannot_claim_frozen_commit(prepared):
    args, _, _ = prepared
    (args.source_root / "src/transformer_rl/frame_process.py").write_text("# dirty package\n")
    origin = campaign.source_identity(args.source_root)
    origin["git_head"] = campaign.SOURCE_COMMIT
    campaign.write(args.source_identity, origin, replace=True)
    args.output_root = args.output_root.parent / "new_output"
    with pytest.raises(ValueError, match="9b576e7"):
        campaign.prepare(args)


def test_replaced_lock_file_is_rejected_even_with_same_path(prepared, monkeypatch):
    args, runner, manifest = prepared
    original = campaign.acquire_resources
    @contextmanager
    def replaced(manifest, budgets, publish):
        with original(manifest, budgets, publish) as value:
            args.resource_lock.unlink()
            args.resource_lock.touch()
            yield value
    monkeypatch.setattr(campaign, "acquire_resources", replaced)
    monkeypatch.setattr(control, "run_worker", lambda *_: pytest.fail("replaced lock launched evaluation"))
    with pytest.raises(ValueError, match="lock was replaced"):
        campaign.run(runner)


@pytest.mark.parametrize("descriptor", ["<f0", "<f3", "|b2"])
def test_invalid_npy_dtype_width_rejected(descriptor):
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {"descr": descriptor, "fortran_order": False, "shape": (1,)})
    payload.seek(0)
    with pytest.raises(ValueError, match="dtype width"):
        campaign.npy_header(payload)


def test_validator_accepts_real_planar_and_scaled_metric_schemas():
    import torch
    from transformer_rl.control_metrics import ControlMetrics
    from test_control_metrics import packet, with_scaled_nominal_envelope

    collector = ControlMetrics(1, .01, settle_steps=200, min_steady_samples=200)
    for index in range(4001):
        current = packet(time=(index + 1) * .01, reference=(.1, 0., .3),
                         position=(index * .001, index * .002))
        collector.update(with_scaled_nominal_envelope(current, .85), torch.tensor([False]))
    result = collector.report()
    campaign.validate_metrics(result, 1)
    assert result["planar_motion"]["full_interval"]["velocity_world"]["vy"]["mean_m_s"] == pytest.approx(.2)
    assert result["planar_motion"]["stationary"]["available"] is False
    assert result["actuation"]["scaled_nominal_envelope"]["sample_count"] == 4001

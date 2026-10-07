"""CPU-only evidence producer and fault tests; no simulator, rollout, or PPO."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_learning_selection import ready, selection as tested_selection, put, terminal


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("latency_under_test", ROOT / "tools/run_learning_latency.py")
latency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latency)
REAL_WORKER = latency.campaign.worker
REAL_WORKERS_IN = latency.campaign.workers_in
REAL_CPU_IDENTITY = latency.cpu_identity


@pytest.fixture
def producer(ready, monkeypatch, tmp_path):
    # Keep the selector's strict bundle/process/parity verifier real.
    monkeypatch.setattr(latency, "selection", tested_selection)
    monkeypatch.setattr(latency, "campaign", tested_selection.campaign)
    monkeypatch.setattr(latency, "control", tested_selection.control)
    monkeypatch.setattr(latency, "diagnostic", tested_selection.diagnostic)
    manifest = ready["manifest"]
    control, campaign = latency.control, latency.campaign
    locks = tmp_path / "resources"
    locks.mkdir()
    for name in ("resource.lock", "study.lock"):
        (locks / name).touch()
    manifest.update(dependencies={}, resource_lock=str(locks / "resource.lock"), study_lock=str(locks / "study.lock"))
    executable = {"requested": sys.executable, "resolved": sys.executable, "sha256": "e" * 64, "device": 1, "inode": 2}
    manifest["runtime"] = {**manifest.get("runtime", {}), "executable": executable}
    manifest["runtime"].setdefault("checkpoint_runtime", {"torch": "fixture", "numpy": "fixture"})
    campaign_manifest = Path(manifest["output_root"]) / "manifest.json"
    put(campaign_manifest, manifest)
    first = next(iter(ready["index"]["cells"].values()))
    old_receipt = tested_selection.read(first["receipt"]["path"])
    machine = tested_selection.read(old_receipt["benchmark"]["path"])["machine"]
    identity = {"versions": {"python": "fixture", "torch": "fixture", "numpy": "fixture", "onnx": "fixture", "onnxruntime": "test"},
        "machine": machine, "cuda_initialized": False, "executable": executable}
    monkeypatch.setattr(latency, "cpu_identity", lambda source: deepcopy(identity))
    output = tmp_path / "cpu_producer"
    definition = latency.prepare(campaign_manifest, output)
    calls = []
    def worker(command, directory, environment, timeout, publish):
        assert environment["CUDA_VISIBLE_DEVICES"] == "" and environment["PYTHONDONTWRITEBYTECODE"] == "1"
        assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "1"
        assert not any(Path(environment["PYTHONPYCACHEPREFIX"]).iterdir())
        calls.append((directory.name, command))
        request = tested_selection.read(directory.parent / "request.json")
        key = request["identity"]["cell"]
        reference = tested_selection.read(ready["index"]["cells"][key]["receipt"]["path"])
        if directory.name == "export":
            bundle_dir = directory.parent / "bundle"
            bundle_dir.mkdir()
            old_bundle = tested_selection.read(reference["bundle"]["path"])
            for name in old_bundle["files"]:
                original = Path(reference["bundle"]["path"]).parent / name
                (bundle_dir / name).write_bytes(original.read_bytes())
            old_bundle["runtime_versions"] = {key: value for key, value in identity["versions"].items() if key != "python"}
            put(bundle_dir / "manifest.json", old_bundle)
        elif directory.name == "benchmark":
            benchmark = tested_selection.read(reference["benchmark"]["path"])
            benchmark["manifest_sha256"] = control.file_sha(directory.parent / "bundle/manifest.json")
            put(directory.parent / "benchmark.json", benchmark)
        else:
            parity = request["parity_request"]
            put(directory.parent / "parity.json", {"format": "transformer_rl.learning_export_parity_reference", "schema_version": 1,
                "identity": parity["identity"], "protocol": tested_selection.PARITY_INPUTS, "reference_max_abs": 10., "cuda_initialized": False})
        process = terminal(command)
        put(directory / "worker.process.json", process)
        return process
    monkeypatch.setattr(campaign, "worker", worker)
    return {**ready, "campaign_manifest": campaign_manifest, "producer": definition, "producer_path": output / "manifest.json",
            "producer_root": output, "cpu_identity": identity, "calls": calls, "synthetic_worker": worker}


def produce_first(producer):
    manifest, audited = producer["manifest"], producer["audit"]
    cell = manifest["inputs"]["cells"][0]
    checkpoint = latency.endpoint(audited, cell)
    result = latency.produce_cell(producer["producer"], manifest, cell, checkpoint, lambda *args, **kwargs: None)
    directory = producer["producer_root"] / "cells" / latency.campaign.cell_key(cell) / "attempt_0000"
    return cell, checkpoint, result, directory


def test_prepare_is_input_only_and_empty_cpu_index_retains_all_ninety(producer):
    definition, manifest = latency.validate(producer["producer_path"])
    assert definition == producer["producer"] and manifest == producer["manifest"]
    assert producer["calls"] == []
    index = latency.audit(producer["producer_path"])
    assert index["status"] == "not_ready" and index["expected_cells"] == index["actual_cells"] == 90
    assert index["completed_cells"] == 0
    assert all(item["receipt"] is None for item in index["cells"].values())
    assert definition["protocol"]["warmup"] == 50 and definition["protocol"]["iterations"] == 1000
    assert not definition["hardware_deployment_ready"]


def test_real_selector_accepts_new_cell_closure_and_repeat_is_read_only(producer):
    cell, checkpoint, result, directory = produce_first(producer)
    assert result["status"] == "completed"
    measured = tested_selection.latency_cell(producer["manifest"], cell, checkpoint, result["receipt"])
    assert measured["passed"] and measured["p99_ms"] == .8
    receipt = tested_selection.read(result["receipt"]["path"])
    assert set(receipt["artifact_closure"]) == {"bundle", "benchmark", "policy.pt", "policy.onnx"}
    assert receipt["parity_reference_max_abs"] == 10.
    assert len(producer["calls"]) == 3
    assert latency.produce_cell(producer["producer"], producer["manifest"], cell, checkpoint,
        lambda *args, **kwargs: None) == result
    assert len(producer["calls"]) == 3
    index = latency.audit(producer["producer_path"])
    assert index["status"] == "not_ready" and index["completed_cells"] == 1 and len(index["cells"]) == 90


@pytest.mark.parametrize("fault", ["checkpoint", "graph", "benchmark", "parity", "request", "process", "environment", "cache", "extra_attempt", "receipt"])
def test_successful_evidence_cannot_hide_tampering(producer, fault):
    cell, checkpoint, result, directory = produce_first(producer)
    if fault == "checkpoint":
        Path(checkpoint["path"]).write_bytes(b"changed checkpoint")
    elif fault == "graph":
        (directory / "bundle/policy.onnx").write_bytes(b"changed graph")
    elif fault == "benchmark":
        value = tested_selection.read(directory / "benchmark.json")
        value["iterations"] = 999
        put(directory / "benchmark.json", value)
    elif fault == "parity":
        value = tested_selection.read(directory / "parity.json")
        value["reference_max_abs"] = float("nan")
        (directory / "parity.json").write_text(json.dumps(value))
    elif fault == "request":
        value = tested_selection.read(directory / "request.json")
        value["commands"]["benchmark"][-1] = "10"
        put(directory / "request.json", value)
    elif fault in {"process", "environment"}:
        value = tested_selection.read(directory / "benchmark/worker.process.json")
        if fault == "process":
            value["pid"] = None
        else:
            value["cpu_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        put(directory / "benchmark/worker.process.json", value)
    elif fault == "cache":
        (directory / "benchmark/.bytecode-cache/hidden.pyc").write_bytes(b"stale")
    elif fault == "extra_attempt":
        (directory.parent / "attempt_0001").mkdir()
    else:
        value = tested_selection.read(directory / "receipt.json")
        value.pop("sha256")
        value["producer_sha256"] = "f" * 64
        put(directory / "receipt.json", tested_selection.seal(value))
    with pytest.raises((ValueError, OSError)):
        latency.audit(producer["producer_path"])


@pytest.mark.parametrize("fault", ["failed", "timeout", "wrong_command", "missing_output", "machine", "parity_range", "runtime", "unlisted_graph"])
def test_failed_or_incompatible_cpu_phases_are_sealed_without_retry(producer, monkeypatch, fault):
    original = producer["synthetic_worker"]
    def failed(command, directory, *args):
        process = original(command, directory, *args)
        if directory.name == "benchmark":
            if fault == "failed":
                process["returncode"] = 1
            elif fault == "timeout":
                process["timed_out"] = True
            elif fault == "wrong_command":
                process["command"] = [*command, "unexpected"]
            elif fault == "missing_output":
                (directory.parent / "benchmark.json").unlink()
            elif fault == "machine":
                value = tested_selection.read(directory.parent / "benchmark.json")
                value["machine"]["node"] = "different-cpu"
                put(directory.parent / "benchmark.json", value)
            elif fault == "parity_range":
                value = tested_selection.read(directory.parent / "bundle/manifest.json")
                value["validation"]["onnx_max_abs_error"] = 100.
                put(directory.parent / "bundle/manifest.json", value)
                benchmark = tested_selection.read(directory.parent / "benchmark.json")
                benchmark["manifest_sha256"] = latency.control.file_sha(directory.parent / "bundle/manifest.json")
                put(directory.parent / "benchmark.json", benchmark)
            elif fault == "runtime":
                value = tested_selection.read(directory.parent / "bundle/manifest.json")
                value["runtime_versions"]["torch"] = "different"
                put(directory.parent / "bundle/manifest.json", value)
                benchmark = tested_selection.read(directory.parent / "benchmark.json")
                benchmark["manifest_sha256"] = latency.control.file_sha(directory.parent / "bundle/manifest.json")
                put(directory.parent / "benchmark.json", benchmark)
            else:
                (directory.parent / "bundle/foreign.bin").write_bytes(b"unexpected")
        return process
    monkeypatch.setattr(latency.campaign, "worker", failed)
    cell, checkpoint, result, directory = produce_first(producer)
    assert result["status"] == "incomplete"
    before = len(producer["calls"])
    assert latency.produce_cell(producer["producer"], producer["manifest"], cell, checkpoint,
        lambda *args, **kwargs: None)["status"] == "incomplete"
    assert len(producer["calls"]) == before
    index = latency.audit(producer["producer_path"])
    assert index["status"] == "not_ready" and index["completed_cells"] == 0 and len(index["cells"]) == 90


def test_unsealed_job_and_live_handle_are_never_restarted(producer, monkeypatch):
    manifest, definition = producer["manifest"], producer["producer"]
    cell = manifest["inputs"]["cells"][0]
    cp = latency.endpoint(producer["audit"], cell)
    directory = producer["producer_root"] / "cells" / latency.campaign.cell_key(cell) / "attempt_0000"
    directory.mkdir(parents=True)
    put(directory / "request.json", latency.request_for(definition, manifest, cell, cp, directory))
    monkeypatch.setattr(latency.campaign, "workers_in", lambda roots: [{"pid": 42, "start": "100"}])
    assert latency.produce_cell(definition, manifest, cell, cp, lambda *args: None)["status"] == "waiting"
    assert not (directory / "receipt.json").exists() and not producer["calls"]
    monkeypatch.setattr(latency.campaign, "workers_in", lambda roots: [])
    assert latency.audit(producer["producer_path"])["cells"][latency.campaign.cell_key(cell)]["status"] == "unsealed"
    result = latency.produce_cell(definition, manifest, cell, cp, lambda *args: None)
    assert result["status"] == "incomplete" and not producer["calls"]


def test_sole_attempt_0001_cannot_replace_the_declared_first_attempt(producer):
    manifest, definition = producer["manifest"], producer["producer"]
    cell = manifest["inputs"]["cells"][0]
    checkpoint = latency.endpoint(producer["audit"], cell)
    directory = producer["producer_root"] / "cells" / latency.campaign.cell_key(cell) / "attempt_0001"
    directory.mkdir(parents=True)
    # A complete, internally consistent request must still fail the no-retry protocol.
    put(directory / "request.json", latency.request_for(definition, manifest, cell, checkpoint, directory))
    with pytest.raises(ValueError, match="only permits attempt_0000"):
        latency.produce_cell(definition, manifest, cell, checkpoint, lambda *args: None)
    with pytest.raises(ValueError, match="only permits attempt_0000"):
        latency.audit(producer["producer_path"])
    assert not producer["calls"] and not (directory / "receipt.json").exists()


def test_no_complete_runtime_does_not_launch_export_or_fabricate_zero(producer, monkeypatch):
    audited = deepcopy(producer["audit"])
    key = next(iter(audited["cells"]))
    audited["cells"][key]["development"]["3701"] = None
    audited.update(status="not_ready", completed_development_cells=89)
    monkeypatch.setattr(latency.campaign, "audit", lambda path: audited)
    result = latency.run(producer["producer_path"])
    assert result["status"] == "waiting" and not producer["calls"]
    index = tested_selection.read(producer["producer_root"] / "index.json")
    assert index["status"] == "not_ready" and len(index["cells"]) == 90
    assert all(value["receipt"] is None for value in index["cells"].values())


def test_cpu_identity_and_helpers_are_frozen(producer, monkeypatch):
    value = deepcopy(producer["cpu_identity"])
    value["machine"]["cpu_affinity"] = [0, 1]
    monkeypatch.setattr(latency, "cpu_identity", lambda source: value)
    with pytest.raises(ValueError, match="affinity"):
        latency.validate(producer["producer_path"])
    monkeypatch.setattr(latency, "cpu_identity", lambda source: deepcopy(producer["cpu_identity"]))
    monkeypatch.setattr(latency, "helper_identity", lambda: {"changed": "f" * 64})
    with pytest.raises(ValueError, match="helpers"):
        latency.validate(producer["producer_path"])


@pytest.mark.parametrize("fault", ["wrong_campaign", "missing_cell", "foreign_cell"])
def test_invalid_runtime_audit_cannot_launch_cpu_worker(producer, monkeypatch, fault):
    audited = deepcopy(producer["audit"])
    if fault == "wrong_campaign":
        audited["manifest_sha256"] = "f" * 64
    elif fault == "missing_cell":
        audited["cells"].pop(next(iter(audited["cells"])))
    else:
        audited["cells"]["foreign/variant/seed_0"] = audited["cells"].pop(next(iter(audited["cells"])))
    monkeypatch.setattr(latency.campaign, "audit", lambda path: audited)
    with pytest.raises(ValueError, match="grid"):
        latency.run(producer["producer_path"])
    assert not producer["calls"]


def test_actual_cpu_identity_does_not_initialize_cuda_or_kit():
    result = REAL_CPU_IDENTITY(ROOT)
    assert result["cuda_initialized"] is False
    assert result["machine"]["cpu_affinity"] == sorted(os.sched_getaffinity(0))
    assert result["versions"]["onnxruntime"] == result["machine"]["onnxruntime"]


def test_actual_cpu_worker_deadline_reaps_only_owned_handle(tmp_path):
    directory = tmp_path / "phase"
    directory.mkdir()
    cache = directory / ".bytecode-cache"
    cache.mkdir()
    record = REAL_WORKER([sys.executable, "-c", "import time;time.sleep(10)"], directory,
        latency.cpu_environment(ROOT, cache), .1, lambda *args: None)
    assert record["timed_out"] and record["status"] == "finished"
    assert latency.control.process_start(record["pid"]) != record["start"]
    assert not REAL_WORKERS_IN([directory])


def test_actual_cpu_cache_recipe_avoids_old_source_pyc(tmp_path):
    import py_compile
    source = tmp_path / "source"
    package = source / "src"
    package.mkdir(parents=True)
    module = package / "latency_cache_fixture.py"
    module.write_text("result='old'\n")
    stamp = module.stat().st_mtime_ns
    py_compile.compile(str(module), doraise=True)
    module.write_text("result='new'\n")
    os.utime(module, ns=(stamp, stamp))
    cache = tmp_path / "cache"
    cache.mkdir()
    process = subprocess.run([sys.executable, "-c", "import latency_cache_fixture;print(latency_cache_fixture.result)"],
        env=latency.cpu_environment(source, cache), capture_output=True, text=True, check=True)
    assert process.stdout.strip() == "new" and not any(cache.iterdir())
    (cache / "hidden.pyc").write_bytes(b"stale")
    with pytest.raises(ValueError, match="cache"):
        latency.cpu_environment(source, cache)


def test_full_ninety_cell_index_is_accepted_by_independent_selector(producer):
    result = latency.run(producer["producer_path"])
    assert result["status"] == "completed" and len(producer["calls"]) == 270
    index = latency.audit(producer["producer_path"])
    assert index["status"] == "completed" and index["completed_cells"] == 90
    definition = {"latency_index": str(producer["producer_root"] / "index.json")}
    values, missing, artifact = tested_selection.load_latency(definition, producer["manifest"], producer["audit"])
    assert len(values) == 90 and missing == [] and artifact["sha256"]
    assert len({value["machine_sha256"] for value in values.values()}) == 1
    latency.run(producer["producer_path"])
    assert len(producer["calls"]) == 270


def test_public_cpu_export_benchmark_and_raw_mean_parity_with_tensor_checkpoint(tmp_path):
    import torch
    from transformer_rl.config import PPOConfig
    from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig
    from transformer_rl.frame_policy import FramePolicyConfig
    from transformer_rl.frame_training import FrameActorCritic
    from transformer_rl.frame_checkpoint import save_frame_checkpoint
    from transformer_rl.ppo import PPOTrainer
    # Synthetic tensor fixture: the update label is not training consumption evidence.
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{index}" for index in range(5)], "action_names": ["position", "velocity"],
        "action_bounds": [.2, .5], "target_scale": [.25, 10.], "target_offset": [.1, -.2], "target_units": ["rad", "rad/s"]}
    config = FrameTrainConfig(FrameModelConfig(policy=FramePolicyConfig(architecture="mlp", frame_dim=5,
        action_dim=2, history_length=1, actor_hidden_dims=(4,)), critic_dim=3, critic_hidden=(4,)), PPOConfig(), control, {})
    model = FrameActorCritic(config.model)
    checkpoint = tmp_path / "tensor-fixture.pt"
    save_frame_checkpoint(checkpoint, model, PPOTrainer(model, config.ppo), config, 1200, {"synthetic_tensor_fixture": True})
    config_path = tmp_path / "prepared/config.json"
    put(config_path, config.to_dict())
    manifest = {"sha256": "1" * 64, "source_root": str(ROOT), "inputs": {"root": str(config_path.parent),
        "source": {"sha256": "2" * 64}}, "controllers": {},
        "runtime": {"checkpoint_runtime": {"torch": str(torch.__version__), "numpy": __import__("numpy").__version__}}}
    cell = {"rate_id": "rate_000", "variant": "tensor_fixture", "training_seed": 1101,
        "training_config": {"path": "config.json", "canonical_sha256": latency.control.digest(config.to_dict())}}
    cp = latency.campaign.artifact(checkpoint)
    bundle, benchmark = tmp_path / "bundle", tmp_path / "benchmark.json"
    parity_path, parity_output = tmp_path / "parity.request.json", tmp_path / "parity.json"
    put(parity_path, tested_selection.parity_request(manifest, cell, cp, parity_output))
    commands = {
        "export": latency.campaign.transfer.command(ROOT, "export", "--checkpoint", checkpoint, "--directory", bundle),
        "benchmark": latency.campaign.transfer.command(ROOT, "benchmark", "--directory", bundle, "--output", benchmark,
            "--backend", "onnx", "--threads", 1, "--iterations", 1000),
        "parity": tested_selection.parity_command(manifest, parity_path)}
    records = {}
    for name, command in commands.items():
        directory = tmp_path / name
        directory.mkdir()
        cache = directory / ".bytecode-cache"
        cache.mkdir()
        record = REAL_WORKER(command, directory, latency.cpu_environment(ROOT, cache), 60., lambda *args: None)
        assert tested_selection.terminal_process(record, command)
        assert not any(cache.iterdir())
        records[name] = latency.campaign.artifact(directory / "worker.process.json")
    parity = tested_selection.read(parity_output)
    assert parity["cuda_initialized"] is False
    benchmark_value = tested_selection.read(benchmark)
    assert benchmark_value["iterations"] == 1000 and benchmark_value["threads"] == 1
    receipt = {"format": latency.CELL_FORMAT, "schema_version": 1, "status": "completed",
        "identity": tested_selection.read(parity_path)["identity"], "controllers": {},
        "bundle": latency.campaign.artifact(bundle / "manifest.json"), "benchmark": latency.campaign.artifact(benchmark),
        "export_process": records["export"], "benchmark_process": records["benchmark"],
        "parity": {"request": latency.campaign.artifact(parity_path), "output": latency.campaign.artifact(parity_output), "process": records["parity"]},
        "parity_reference_max_abs": parity["reference_max_abs"]}
    item = put(tmp_path / "receipt.json", tested_selection.seal(receipt))
    measured = tested_selection.latency_cell(manifest, cell, cp, item)
    assert measured["p99_ms"] >= 0 and measured["max_ms"] >= measured["p99_ms"]
    assert not torch.cuda.is_initialized()

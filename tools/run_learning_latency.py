#!/usr/bin/env python3
"""Independent CPU export and latency evidence for the exact ninety-cell campaign.

This controller never imports a simulator, launches PPO, or selects a winner.
Missing and failed cells remain in the fixed ninety-cell index.
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
import tempfile


_SPEC = importlib.util.spec_from_file_location("learning_latency_selection", Path(__file__).with_name("select_learning_rate.py"))
selection = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selection)
campaign, control, diagnostic = selection.campaign, selection.control, selection.diagnostic
FORMAT = "transformer_rl.learning_latency_producer"
CELL_FORMAT = "transformer_rl.learning_latency_cell"
INDEX_FORMAT = "transformer_rl.learning_latency_index"
PROTOCOL = {"backend": "onnx", "threads": 1, "iterations": 1000, "warmup": 50,
    "warmup_evidence": "frozen source FrameRuntime.benchmark default; public CLI has no warmup argument",
    "scope": selection.LATENCY_SCOPE, "parity_inputs": selection.PARITY_INPUTS,
    "expected_cells": 90, "no_unsealed_retry": True, "no_selection": True}
REQUIRED_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def helper_identity():
    return {**selection.helper_identity(), str(campaign.plain(__file__)): control.file_sha(__file__)}


def cpu_environment(source, cache):
    cache = campaign.plain(cache)
    selection.require(cache.is_dir() and not any(cache.iterdir()), "CPU cache prefix must be exclusive and empty")
    return {**campaign.cpu_environment(source), "PYTHONPYCACHEPREFIX": str(cache)}


def environment_recipe(source, cache):
    return {**REQUIRED_ENVIRONMENT, "PYTHONPATH": str(campaign.plain(source) / "src"),
            "PYTHONPYCACHEPREFIX": str(campaign.plain(cache))}


def cpu_identity(source):
    """Measure the actual CPU process identity, not a claim of hardware qualification."""
    script = """import json,os,platform,torch,numpy,onnx,onnxruntime
assert not torch.cuda.is_initialized()
print(json.dumps({'versions':{'python':platform.python_version(),'torch':str(torch.__version__),
    'numpy':numpy.__version__,'onnx':onnx.__version__,'onnxruntime':onnxruntime.__version__},
    'machine':{'node':platform.node(),'system':platform.system(),'architecture':platform.machine(),
    'processor':platform.processor(),'cpu_count':os.cpu_count(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
    'onnxruntime':onnxruntime.__version__},'cuda_initialized':torch.cuda.is_initialized()},allow_nan=False))
"""
    with tempfile.TemporaryDirectory(prefix="learning-latency-cpu-") as cache:
        result = subprocess.run([sys.executable, "-c", script], env=cpu_environment(source, cache),
                                capture_output=True, text=True, timeout=120)
    selection.require(result.returncode == 0, "CPU export dependencies unavailable: " + result.stderr[-4000:])
    value = campaign.json.loads(result.stdout)
    selection.require(value["cuda_initialized"] is False, "CPU dependency probe initialized CUDA")
    return {**value, "executable": campaign.runtime_identity(source)["executable"]}


def prepare(campaign_manifest, output_root, *, worker_timeout_seconds=14400.):
    manifest = campaign.validate(campaign_manifest)
    selection.protocol(manifest)
    forbidden = [manifest["output_root"], manifest["source_root"], manifest["inputs"]["root"],
                 *(item["root"] for item in manifest["dependencies"].values()),
                 str(Path(manifest["resource_lock"]).parent), str(Path(manifest["study_lock"]).parent)]
    root = selection.independent_root(output_root, forbidden)
    selection.require(type(worker_timeout_seconds) in (int, float) and math.isfinite(worker_timeout_seconds)
                      and worker_timeout_seconds > 0, "finite positive CPU worker deadline required")
    identity = cpu_identity(manifest["source_root"])
    selection.require(identity["executable"] == manifest["runtime"]["executable"],
                      "producer must use the sealed campaign Python executable")
    selection.require(all(identity["versions"][key] == manifest["runtime"]["checkpoint_runtime"][key]
                          for key in ("torch", "numpy")), "CPU export Torch/NumPy differ from the learning runtime")
    value = {"format": FORMAT, "schema_version": 1, "output_root": str(root),
        "campaign_manifest": campaign.artifact(campaign_manifest), "campaign_sha256": manifest["sha256"],
        "helpers": helper_identity(), "cpu_identity": identity, "protocol": PROTOCOL,
        "worker_timeout_seconds": worker_timeout_seconds, "formal_architecture_selection": False,
        "hardware_deployment_ready": False}
    root.mkdir(parents=True, exist_ok=False)
    lock = root / ".latency.lock"
    lock.touch(exist_ok=False)
    value["lock"] = {"path": str(lock), **diagnostic.lock_identity(lock)}
    value = selection.seal(value)
    campaign.write(root / "manifest.json", value)
    # This complete empty index is a declaration, never a fabricated measurement.
    index = make_index(value, manifest, {})
    campaign.write(root / "index.json", index)
    return value


def validate(path):
    path = campaign.plain(path)
    definition = selection.sealed(selection.read(path))
    selection.require(definition.get("format") == FORMAT and definition.get("schema_version") == 1
        and path == Path(definition["output_root"]) / "manifest.json" and definition["helpers"] == helper_identity()
        and definition["protocol"] == PROTOCOL, "producer definition or frozen helpers changed")
    manifest = campaign.validate(campaign.checked(definition["campaign_manifest"]))
    selection.require(manifest["sha256"] == definition["campaign_sha256"], "execution campaign identity changed")
    selection.protocol(manifest)
    selection.require(cpu_identity(manifest["source_root"]) == definition["cpu_identity"],
                      "CPU versions, executable, machine, or affinity changed")
    lock = definition["lock"]
    selection.require(diagnostic.lock_identity(lock["path"]) == {key: value for key, value in lock.items() if key != "path"},
                      "producer lock inode changed")
    selection.require(type(definition["worker_timeout_seconds"]) in (int, float)
        and math.isfinite(definition["worker_timeout_seconds"]) and definition["worker_timeout_seconds"] > 0,
        "CPU deadline changed to an invalid budget")
    return definition, manifest


@contextmanager
def producer_lock(definition):
    with Path(definition["lock"]["path"]).open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic.check_open_lock(stream, {key: value for key, value in definition["lock"].items() if key != "path"})
        yield


def make_index(definition, manifest, cells):
    keys = {campaign.cell_key(cell) for cell in manifest["inputs"]["cells"]}
    selection.require(len(keys) == 90 and not set(cells) - keys, "latency index must retain the exact ninety cells")
    values = {campaign.cell_key(cell): cells.get(campaign.cell_key(cell), {"status": "not_started", "receipt": None})
              for cell in manifest["inputs"]["cells"]}
    return selection.seal({"format": INDEX_FORMAT, "schema_version": 1, "campaign_sha256": manifest["sha256"],
        "producer_sha256": definition["sha256"], "expected_cells": 90, "actual_cells": len(values),
        "status": "completed" if all(value["status"] == "completed" for value in values.values()) else "not_ready",
        "completed_cells": sum(value["status"] == "completed" for value in values.values()), "cells": values,
        "formal_architecture_selection": False, "hardware_deployment_ready": False})


def endpoint(audited, cell):
    item = audited["cells"][campaign.cell_key(cell)]
    training = item.get("training")
    if (audited.get("manifest_sha256") is None or item.get("status") != "completed" or not training
            or training.get("status") != "completed" or training.get("completed_updates") != 1200
            or training.get("charged_updates") != 1200 or training.get("verified_fresh_samples") != campaign.TOTAL
            or set(item.get("development", {})) != {str(seed) for seed in campaign.DEVELOPMENT_SEEDS}
            or any(value is None or value.get("status") != "completed" for value in item["development"].values())):
        return None
    campaign.checked(training["checkpoint"])
    return training["checkpoint"]


def validate_runtime_audit(manifest, audited):
    keys = {campaign.cell_key(cell) for cell in manifest["inputs"]["cells"]}
    selection.require(audited.get("manifest_sha256") == manifest["sha256"]
        and audited.get("expected_cells") == audited.get("actual_cells") == 90
        and set(audited.get("cells", {})) == keys, "runtime audit belongs to another or incomplete grid")


def request_for(definition, manifest, cell, checkpoint, directory):
    parity_path, parity_output = directory / "parity.request.json", directory / "parity.json"
    bundle, benchmark = directory / "bundle", directory / "benchmark.json"
    commands = {
        "export": campaign.transfer.command(Path(manifest["source_root"]), "export", "--checkpoint", checkpoint["path"], "--directory", bundle),
        "benchmark": campaign.transfer.command(Path(manifest["source_root"]), "benchmark", "--directory", bundle,
            "--output", benchmark, "--backend", "onnx", "--threads", 1, "--iterations", 1000),
        "parity": selection.parity_command(manifest, parity_path)}
    parity = selection.parity_request(manifest, cell, checkpoint, parity_output)
    return {"producer_sha256": definition["sha256"], "identity": parity["identity"], "checkpoint": checkpoint,
        "protocol": definition["protocol"], "cpu_identity": definition["cpu_identity"], "commands": commands,
        "parity_request": parity, "environments": {name: environment_recipe(manifest["source_root"], directory / name / ".bytecode-cache")
                                                   for name in commands}}


def verify_request(definition, manifest, cell, checkpoint, directory):
    request = selection.read(directory / "request.json")
    selection.require(request == request_for(definition, manifest, cell, checkpoint, directory),
                      "CPU request changed its source/checkpoint/commands/environment")
    return request


def verify_receipt(definition, manifest, cell, checkpoint, directory, receipt):
    selection.sealed(receipt)
    request = verify_request(definition, manifest, cell, checkpoint, directory)
    selection.require(receipt.get("format") == CELL_FORMAT and receipt.get("schema_version") == 1
        and receipt.get("identity") == request["identity"] and receipt.get("controllers") == manifest["controllers"]
        and receipt.get("producer_sha256") == definition["sha256"] and receipt.get("no_retry") is True
        and receipt.get("request") == campaign.artifact(directory / "request.json"), "CPU receipt/request seal differs")
    for item in receipt["artifacts"].values():
        campaign.checked(item)
    return request


def verify_completed(definition, manifest, cell, checkpoint, directory, receipt, *, receipt_path=None):
    request = verify_receipt(definition, manifest, cell, checkpoint, directory, receipt)
    for phase, command in request["commands"].items():
        path = directory / phase / "worker.process.json"
        process = selection.read(path)
        selection.require(selection.terminal_process(process, command), "CPU worker is not the original successful terminal process")
        selection.require(process.get("cpu_environment") == request["environments"][phase], "CPU worker environment seal differs")
        cache = Path(request["environments"][phase]["PYTHONPYCACHEPREFIX"])
        selection.require(cache.is_dir() and not any(cache.iterdir()), "CPU worker cache prefix no longer empty")
    for item in receipt["artifact_closure"].values():
        campaign.checked(item)
    bundle = selection.read(campaign.checked(receipt["bundle"]))
    expected_closure = {"bundle": receipt["bundle"], "benchmark": receipt["benchmark"],
        **{name: campaign.artifact(directory / "bundle" / name) for name in bundle["files"]}}
    selection.require(receipt["artifact_closure"] == expected_closure, "export graph closure has missing or foreign files")
    selection.require(campaign.inventory(directory / "bundle") == {"manifest.json": receipt["bundle"]["sha256"], **bundle["files"]},
                      "export bundle contains unlisted or missing files")
    candidate = receipt_path or directory / "receipt.json"
    measured = selection.latency_cell(manifest, cell, checkpoint, campaign.artifact(candidate))
    benchmark = selection.read(campaign.checked(receipt["benchmark"]))
    selection.require(benchmark["machine"] == definition["cpu_identity"]["machine"], "benchmark CPU differs from frozen measured CPU")
    selection.require(bundle["runtime_versions"] == {key: value for key, value in definition["cpu_identity"]["versions"].items() if key != "python"},
                      "export runtime versions differ")
    return measured


def seal_result(definition, manifest, cell, checkpoint, directory, *, error=None):
    request = verify_request(definition, manifest, cell, checkpoint, directory)
    receipt = {"format": CELL_FORMAT, "schema_version": 1, "status": "incomplete", "identity": request["identity"],
        "controllers": manifest["controllers"], "producer_sha256": definition["sha256"],
        "request": campaign.artifact(directory / "request.json"), "finished_at": control.now(),
        "artifacts": {}, "no_retry": True}
    for path in directory.rglob("*"):
        if path.is_file() and path.name != "receipt.json":
            receipt["artifacts"][str(path.relative_to(directory))] = campaign.artifact(path)
    if error is not None:
        receipt["error"] = error
    else:
        try:
            bundle = directory / "bundle/manifest.json"
            benchmark = directory / "benchmark.json"
            parity = directory / "parity.json"
            receipt.update(status="completed", bundle=campaign.artifact(bundle), benchmark=campaign.artifact(benchmark),
                export_process=campaign.artifact(directory / "export/worker.process.json"),
                benchmark_process=campaign.artifact(directory / "benchmark/worker.process.json"),
                parity_reference_max_abs=selection.read(parity)["reference_max_abs"],
                parity={"request": campaign.artifact(directory / "parity.request.json"),
                        "output": campaign.artifact(parity), "process": campaign.artifact(directory / "parity/worker.process.json")})
            bundle_value = selection.read(bundle)
            receipt["artifact_closure"] = {"bundle": receipt["bundle"], "benchmark": receipt["benchmark"],
                **{name: campaign.artifact(directory / "bundle" / name) for name in bundle_value["files"]}}
            candidate = directory / "candidate.json"
            candidate_receipt = selection.seal(receipt)
            campaign.write(candidate, candidate_receipt)
            verify_completed(definition, manifest, cell, checkpoint, directory, candidate_receipt, receipt_path=candidate)
        except (ValueError, KeyError, TypeError, OSError) as exception:
            receipt["status"] = "incomplete"
            receipt["error"] = f"{type(exception).__name__}: {exception}"
    receipt = selection.seal(receipt)
    campaign.write(directory / "receipt.json", receipt)
    return receipt


def produce_cell(definition, manifest, cell, checkpoint, publish):
    root = Path(definition["output_root"])
    job = root / "cells" / campaign.cell_key(cell)
    attempts = sorted(job.glob("attempt_*"))
    selection.require(len(attempts) <= 1 and (not attempts or attempts[0].name == "attempt_0000"),
                      "CPU cell only permits attempt_0000; undeclared retries are rejected")
    if attempts:
        directory = attempts[0]
        verify_request(definition, manifest, cell, checkpoint, directory)
        if campaign.workers_in([directory]):
            return {"status": "waiting", "receipt": None, "reason": "original CPU handle is live or unresolved"}
        if (directory / "receipt.json").exists():
            receipt = selection.sealed(selection.read(directory / "receipt.json"))
            if receipt["status"] == "completed":
                verify_completed(definition, manifest, cell, checkpoint, directory, receipt)
            else:
                selection.require(receipt["status"] == "incomplete", "unknown CPU endpoint status")
                verify_receipt(definition, manifest, cell, checkpoint, directory, receipt)
            return {"status": receipt["status"], "receipt": campaign.artifact(directory / "receipt.json")}
        # No phase is restarted or salvaged as a favorable measurement.
        receipt = seal_result(definition, manifest, cell, checkpoint, directory, error="unsealed CPU job retained; no execution retry")
        return {"status": receipt["status"], "receipt": campaign.artifact(directory / "receipt.json")}
    selection.require(not campaign.workers_in([root]), "another original CPU worker is live or unresolved")
    directory = job / "attempt_0000"
    directory.mkdir(parents=True, exist_ok=False)
    request = request_for(definition, manifest, cell, checkpoint, directory)
    campaign.write(directory / "request.json", request)
    campaign.write(directory / "parity.request.json", request["parity_request"])
    error = None
    try:
        for phase, command in request["commands"].items():
            validate(root / "manifest.json")
            campaign.checked(checkpoint)
            phase_directory = directory / phase
            phase_directory.mkdir()
            cache = phase_directory / ".bytecode-cache"
            cache.mkdir()
            publish("running", active={"cell": campaign.cell_key(cell), "phase": phase})
            process = campaign.worker(command, phase_directory, cpu_environment(manifest["source_root"], cache),
                                      definition["worker_timeout_seconds"], publish)
            process["cpu_environment"] = request["environments"][phase]
            campaign.write(phase_directory / "worker.process.json", process, replace=True)
            selection.require(selection.terminal_process(process, command), "CPU phase failed or exceeded its wall deadline")
        campaign.checked(checkpoint)
    except (ValueError, KeyError, TypeError, OSError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    receipt = seal_result(definition, manifest, cell, checkpoint, directory, error=error)
    return {"status": receipt["status"], "receipt": campaign.artifact(directory / "receipt.json")}


def audit(path):
    definition, manifest = validate(path)
    audited = campaign.audit(campaign.checked(definition["campaign_manifest"]))
    validate_runtime_audit(manifest, audited)
    cells, machine_ids = {}, set()
    for cell in manifest["inputs"]["cells"]:
        key = campaign.cell_key(cell)
        attempts = sorted((Path(definition["output_root"]) / "cells" / key).glob("attempt_*"))
        selection.require(len(attempts) <= 1 and (not attempts or attempts[0].name == "attempt_0000"),
                          "CPU evidence only permits attempt_0000; undeclared extra attempts are rejected")
        checkpoint = endpoint(audited, cell)
        if not attempts:
            cells[key] = {"status": "not_started", "receipt": None, "checkpoint_ready": checkpoint is not None}
            continue
        directory = attempts[0]
        selection.require(checkpoint is not None, "CPU job is not based on the complete development CP1200")
        verify_request(definition, manifest, cell, checkpoint, directory)
        if campaign.workers_in([directory]):
            cells[key] = {"status": "waiting", "receipt": None, "reason": "original handle is live or unresolved"}
        elif not (directory / "receipt.json").exists():
            cells[key] = {"status": "unsealed", "receipt": None, "no_retry": True}
        else:
            receipt = selection.sealed(selection.read(directory / "receipt.json"))
            if receipt["status"] == "completed":
                measured = verify_completed(definition, manifest, cell, checkpoint, directory, receipt)
                machine_ids.add(measured["machine_sha256"])
            else:
                selection.require(receipt["status"] == "incomplete", "unknown CPU endpoint status")
                verify_receipt(definition, manifest, cell, checkpoint, directory, receipt)
            cells[key] = {"status": receipt["status"], "receipt": campaign.artifact(directory / "receipt.json")}
    selection.require(len(machine_ids) <= 1, "ninety CPU cells were measured on different machines")
    return make_index(definition, manifest, cells)


def run(path):
    definition, manifest = validate(path)
    root = Path(definition["output_root"])
    summary = {"status": "waiting", "campaign_sha256": manifest["sha256"], "producer_sha256": definition["sha256"],
               "expected_cells": 90, "started_at": control.now(), "results": {}}
    def publish(status=None, **values):
        if status:
            summary["status"] = status
        summary.update(values, updated_at=control.now())
        campaign.write(root / "summary.json", summary, replace=True)
    with producer_lock(definition):
        audited = campaign.audit(campaign.checked(definition["campaign_manifest"]))
        validate_runtime_audit(manifest, audited)
        if (audited["status"] != "development_complete" or audited["completed_training_cells"] != 90
                or audited["completed_development_cells"] != 90 or any(endpoint(audited, cell) is None for cell in manifest["inputs"]["cells"])):
            publish("waiting", reason="all ninety complete CP1200 and four development seeds are required", active=None)
            campaign.write(root / "index.json", audit(path), replace=True)
            return summary
        if campaign.workers_in([manifest["output_root"], root]):
            publish("waiting", reason="original campaign or CPU worker is live or unresolved", active=None)
            return summary
        for cell in manifest["inputs"]["cells"]:
            key = campaign.cell_key(cell)
            result = produce_cell(definition, manifest, cell, endpoint(audited, cell), publish)
            summary["results"][key] = result
            if result["status"] == "waiting":
                publish("waiting", active=None)
                return summary
            campaign.write(root / "index.json", make_index(definition, manifest, summary["results"]), replace=True)
            publish(active=None)
        index = audit(path)
        campaign.write(root / "index.json", index, replace=True)
        publish("completed" if index["status"] == "completed" else "incomplete", active=None, index=campaign.artifact(root / "index.json"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    command = operations.add_parser("prepare")
    command.add_argument("--campaign-manifest", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--worker-timeout-seconds", type=float, default=14400.)
    for name in ("validate", "audit", "run"):
        item = operations.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.operation == "prepare":
            result = prepare(args.campaign_manifest, args.output_root, worker_timeout_seconds=args.worker_timeout_seconds)
        elif args.operation == "validate":
            result, _ = validate(args.manifest)
        elif args.operation == "audit":
            result = audit(args.manifest)
        else:
            result = run(args.manifest)
        print(campaign.json.dumps(result, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("CPU producer interrupted; original job evidence retained", file=sys.stderr)
        return 130
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())

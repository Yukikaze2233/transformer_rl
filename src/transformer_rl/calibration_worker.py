"""Fresh-process runtime calibration, separate from architecture selection.

Calibration artifacts describe an explicitly authorized small training run and
its private evaluation. They never supply validation/held-out grades, a winning
architecture, production storage authorization or hardware qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import stat
import sys
import time
import traceback

from . import calibration
from . import calibration_storage
from . import exposure_campaign as campaign
from . import exposure_training as training
from . import runtime_paths
from .cli import _factory
from .continuation_process import _ApplicationRegistry
from .experiments import source_identity
from .frame_workflow import _provenance, evaluate_frame_policy


def _startup_profile(directory):
    """Require a bound runtime profile and a genuinely new bytecode cache."""
    directory = Path(directory)
    profile = runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))
    cache = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
    campaign._require(cache == directory / "empty_python_cache",
                      "calibration bytecode cache is outside its worker directory")
    observed = cache.lstat()
    campaign._require(stat.S_ISDIR(observed.st_mode) and observed.st_uid == os.getuid()
                      and observed.st_dev == directory.stat().st_dev and not list(cache.iterdir()),
                      "calibration worker did not start with a new owned empty bytecode cache")
    campaign._require(sys.dont_write_bytecode is True and sys.pycache_prefix == str(cache),
                      "calibration interpreter bytecode settings differ from its worker cache")
    return profile


def _lazy_factory(protocol, config, guard, provenances, shutdown_errors):
    """Resolve the real factory after authorization and retain actual metadata.

    The existing application registry owns SDK applications registered by the
    chassis adapter. An environment rejected before returning to the workflow
    must also be closed here; the workflow does not yet own that object.
    """
    def factory(**kwargs):
        if guard():
            raise InterruptedError("calibration stopped before environment construction")
        resolved = _factory(protocol["environment_factory"])
        if guard():
            raise InterruptedError("calibration stopped after factory resolution")
        environment = resolved(**kwargs)
        try:
            actual = _provenance(environment, config.control)
            if guard():
                raise InterruptedError("calibration stopped after environment construction")
        except BaseException:
            try:
                environment.close()
            except BaseException as error:
                shutdown_errors.append({"owner": "rejected_environment",
                                        "error": f"{type(error).__name__}: {error}"})
            raise
        provenances.append(deepcopy(actual))
        return environment
    return factory


def _training_counters(completion):
    """Retain observed learner accounting, including incomplete attempts."""
    fields = (
        "successful_updates", "attempted_updates", "actual_collected_transitions",
        "successful_full_rollout_samples", "optimizer_steps", "optimization_sample_uses",
        "reserved_updates", "charged_updates", "charged_fresh_transition_budget",
        "unsealed_successful_updates", "recorded_metric_updates", "recorded_full_rollout_samples",
        "unverified_or_unoptimized_samples", "optimizer_update_may_be_partial",
        "failed_update_optimizer_steps", "refund", "automatic_retries",
    )
    counters = {name: deepcopy(completion[name]) for name in fields if name in completion}
    for alias, original in (
        ("completed_updates", "successful_updates"),
        ("collected_transitions", "actual_collected_transitions"),
        ("full_rollout_transitions", "successful_full_rollout_samples"),
    ):
        if original in counters:
            counters[alias] = counters[original]
    return counters


def run_worker(request_receipt):
    """Consume one dedicated calibration authorization in a fresh process."""
    request, protocol, job, batch = calibration.validate_request(request_receipt, child=True)
    directory = Path(request["directory"])
    runtime_profile = _startup_profile(directory)
    campaign._require(not any(os.path.lexists(directory / name) for name in
        ("outcome.json", "report.json", "trace.npz")), "calibration output cannot be overwritten")
    own = campaign._identity(os.getpid())
    started = time.monotonic()
    registry = _ApplicationRegistry()
    previous_handlers, provenances, shutdown_errors = {}, [], []
    completion_receipt = report_receipt = trace_receipt = checkpoint = None
    actual_counters, completion, error = {}, None, None
    work_completed, stop_requested = False, False
    limits = request["limits"]

    def request_stop(number, frame):
        nonlocal stop_requested
        stop_requested = True

    def guard():
        if stop_requested or time.monotonic() - started >= limits["timeout_s"]:
            return True
        for receipt in (request_receipt, request["plan"], request["protocol"]):
            campaign._checked(receipt)
        campaign._require(runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))
                          == runtime_profile, "calibration worker runtime profile changed")
        campaign._require(sys.dont_write_bytecode is True
                          and sys.pycache_prefix == str(directory / "empty_python_cache"),
                          "calibration interpreter bytecode settings changed")
        campaign._require(source_identity() == protocol["source"], "calibration worker source changed")
        calibration.validate_controller_lease(protocol, request["leases"], request["controller"])
        calibration_storage.guard_owned_storage(batch["storage_root"], limits["max_owned_bytes"],
            limits["free_margin_bytes"], expected_root_identity=batch["storage_root_identity"])
        return False

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, request_stop)
        campaign._require(not guard(), "calibration stopped before application activation")
        registry.activate()
        factory = _lazy_factory(protocol, batch["config"], guard, provenances, shutdown_errors)
        if request["kind"] == "train":
            max_seconds = min(protocol["execution"]["max_seconds"], limits["timeout_s"] - 2.)
            campaign._require(max_seconds > 0, "calibration training deadline must be positive")
            completion = training.train_exposure_job([deepcopy(batch["stage"])], factory,
                protocol["environment_factory"], directory / "train", job_id="calibration_" + job["id"],
                rollout_steps=protocol["execution"]["rollout_steps"], training_seed=job["training_seed"],
                retention_seed=job["retention_seed"], evaluation_seeds=[batch["evaluation_seed"]],
                device=protocol["execution"]["device"], expected_initial_model_sha256=job["initial_model_sha256"],
                max_seconds=max_seconds, should_stop=guard, protected_paths=protocol["protected_roots"])
            actual_counters = _training_counters(completion)
            completion_receipt = campaign.definition._receipt(directory / "train" / "completion.json")
            shutdown_errors.extend(deepcopy(completion.get("shutdown_errors", [])))
            if completion["status"] == "completed":
                campaign._require(len(completion["endpoints"]) == 1,
                                  "calibration training did not seal exactly one endpoint")
                endpoint = completion["endpoints"][0]["checkpoint"]
                checkpoint = campaign.definition._receipt(endpoint["path"])
                campaign._require(checkpoint == endpoint, "calibration checkpoint publication differs")
                work_completed = True
            else:
                error = deepcopy(completion.get("error")) or {
                    "type": "IncompleteCalibrationTraining", "message": completion["status"]}
        else:
            evaluation = batch["evaluation"]
            report = evaluate_frame_policy(batch["checkpoint_receipt"]["path"], factory,
                batch["evaluation_environment"], steps=evaluation["steps"], seed=batch["evaluation_seed"],
                device=protocol["execution"]["device"], settle_steps=evaluation["settle_steps"],
                min_steady_samples=evaluation["min_steady_samples"], control_metrics=True, history_control=True,
                trace_output=directory / "trace.npz", trace_replicas=batch["trace_replicas"], should_stop=guard)
            actual_counters = {name: deepcopy(report[name]) for name in
                ("steps", "num_envs", "transitions", "checkpoint_update") if name in report}
            campaign._require(not guard(), "calibration stopped before evaluation report publication")
            campaign._new(directory / "report.json", report)
            report_receipt = campaign.definition._receipt(directory / "report.json")
            trace_receipt = campaign.definition._receipt(directory / "trace.npz")
            checkpoint = deepcopy(batch["checkpoint_receipt"])
            work_completed = True
        campaign._require(not guard(), "calibration stopped after workflow publication")
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure),
                 "traceback": traceback.format_exc()}
    finally:
        shutdown_errors.extend(registry.close(0 if work_completed and error is None else 1))
        try:
            calibration.validate_request(request_receipt, child=True)
            campaign._require(not guard(), "calibration stopped during application shutdown")
        except BaseException as failure:
            shutdown_errors.append({"owner": "worker_guard",
                                    "error": f"{type(failure).__name__}: {failure}"})
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    status = "completed" if work_completed and error is None and not shutdown_errors else "failed"
    outcome = {"format": "transformer_rl.calibration_worker_outcome", "schema_version": 1,
        "request": deepcopy(request_receipt), "source": source_identity(), "process": own,
        "status": status, "runtime_profile": runtime_profile, "environment_provenance": provenances,
        "training_completion": completion_receipt, "report": report_receipt, "trace": trace_receipt,
        "checkpoint": checkpoint, "actual_counters": actual_counters,
        "shutdown_errors": shutdown_errors, "error": error,
        "formal_architecture_selection": False, "production_storage_authorized": False,
        "hardware_verified": False}
    campaign._new(directory / "outcome.json", outcome)
    if status == "completed":
        campaign._require(not guard(), "calibration storage or authorization changed after outcome publication")
    return 0 if status == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("worker")
    command.add_argument("--request", required=True)
    command.add_argument("--expected-request-sha256", required=True)
    args = parser.parse_args(argv)
    receipt = campaign.definition._receipt(args.request)
    campaign._require(receipt["sha256"] == args.expected_request_sha256,
                      "calibration worker raw request differs from external authorization")
    return run_worker(receipt)


if __name__ == "__main__":
    raise SystemExit(main())

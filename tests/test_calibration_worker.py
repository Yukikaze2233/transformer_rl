"""CPU lifecycle checks; synthetic metadata is never SDK qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_rl import calibration_worker as worker
from transformer_rl import frame_process
from transformer_rl import runtime_paths
from transformer_rl.frame_config import digest


def factory_fixture(monkeypatch, *, correct_control=True, close_failure=False):
    control = {"fixture": "CPU factory control contract"}
    metadata = {"identity": "synthetic CPU fixture; no Isaac readback",
        "control_sha256": digest(control) if correct_control else "changed",
        "runtime_paths": {"fixture": "copied metadata only; not an SDK observation"}}
    environment = SimpleNamespace(metadata=metadata, close_calls=0)

    def close():
        environment.close_calls += 1
        if close_failure:
            raise RuntimeError("actual fixture close failure")
    environment.close = close
    calls = []

    def resolve(reference):
        assert reference == "synthetic_cpu_fixture:make_env"

        def factory(**kwargs):
            calls.append(deepcopy(kwargs))
            return environment
        return factory
    monkeypatch.setattr(worker, "_factory", resolve)
    return control, environment, calls


def test_lazy_factory_retains_actual_control_and_runtime_metadata_without_inventing_fields(monkeypatch):
    control, environment, calls = factory_fixture(monkeypatch)
    provenances, errors = [], []
    factory = worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
        SimpleNamespace(control=control), lambda: False, provenances, errors)
    assert factory(device="cpu") is environment
    assert calls == [{"device": "cpu"}] and provenances == [environment.metadata]
    environment.metadata["runtime_paths"]["fixture"] = "later changed caller object"
    assert provenances[0]["runtime_paths"]["fixture"] == "copied metadata only; not an SDK observation"
    assert environment.close_calls == 0 and errors == []


@pytest.mark.parametrize("close_failure", [False, True])
def test_rejected_provenance_closes_the_unreturned_environment_and_preserves_close_failure(monkeypatch, close_failure):
    control, environment, _ = factory_fixture(monkeypatch, correct_control=False, close_failure=close_failure)
    provenances, errors = [], []
    factory = worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
        SimpleNamespace(control=control), lambda: False, provenances, errors)
    with pytest.raises(ValueError, match="contract differs"):
        factory()
    assert environment.close_calls == 1 and provenances == []
    assert errors == ([{"owner": "rejected_environment", "error": "RuntimeError: actual fixture close failure"}]
                      if close_failure else [])


def test_stop_after_construction_closes_the_unreturned_environment(monkeypatch):
    control, environment, _ = factory_fixture(monkeypatch)
    answers = iter((False, False, True))
    provenances, errors = [], []
    factory = worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
        SimpleNamespace(control=control), lambda: next(answers), provenances, errors)
    with pytest.raises(InterruptedError, match="after environment"):
        factory()
    assert environment.close_calls == 1 and provenances == [] and errors == []


def test_stop_during_factory_resolution_prevents_construction(monkeypatch):
    stopped = False

    def resolve(reference):
        nonlocal stopped
        stopped = True
        return lambda **kwargs: pytest.fail("constructed after factory resolution stop")
    monkeypatch.setattr(worker, "_factory", resolve)
    with pytest.raises(InterruptedError, match="after factory resolution"):
        worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
            SimpleNamespace(control={}), lambda: stopped, [], [])()


def test_stop_before_factory_resolution_does_not_import_the_environment(monkeypatch):
    monkeypatch.setattr(worker, "_factory", lambda _: pytest.fail("resolved after caller stop"))
    with pytest.raises(InterruptedError, match="before environment"):
        worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
            SimpleNamespace(control={}), lambda: True, [], [])()


def test_unserializable_actual_provenance_closes_the_unreturned_environment(monkeypatch):
    control, environment, _ = factory_fixture(monkeypatch)
    environment.metadata["runtime_paths"] = object()
    provenances, errors = [], []
    factory = worker._lazy_factory({"environment_factory": "synthetic_cpu_fixture:make_env"},
        SimpleNamespace(control=control), lambda: False, provenances, errors)
    with pytest.raises(TypeError, match="JSON serializable"):
        factory()
    assert environment.close_calls == 1 and provenances == [] and errors == []


def prepare_profile(tmp_path, monkeypatch):
    directory = tmp_path / "worker"
    directory.mkdir(mode=0o700)
    cache = directory / "empty_python_cache"
    cache.mkdir(mode=0o700)
    environment = runtime_paths.prepare_worker_runtime(directory, {
        **os.environ, "PYTHONPYCACHEPREFIX": str(cache), "PYTHONDONTWRITEBYTECODE": "1"})
    for key in runtime_paths.RESERVED_ENVIRONMENT:
        if key in environment:
            monkeypatch.setenv(key, environment[key])
    # These are explicit in-process interpreter fixtures. Real workers must
    # acquire these settings at interpreter startup.
    monkeypatch.setattr(worker.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(worker.sys, "pycache_prefix", str(cache))
    return directory, cache


def test_startup_requires_actual_profile_and_a_new_empty_owned_cache(tmp_path, monkeypatch):
    directory, _ = prepare_profile(tmp_path, monkeypatch)
    receipt = worker._startup_profile(directory)
    assert receipt == runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))


@pytest.mark.parametrize("mutation", ["nonempty", "symlink", "outside", "profile_sha",
                                     "interpreter_prefix", "interpreter_writes"])
def test_startup_rejects_reused_or_unbound_runtime_before_factory_resolution(tmp_path, monkeypatch, mutation):
    directory, cache = prepare_profile(tmp_path, monkeypatch)
    if mutation == "nonempty":
        (cache / "old.pyc").write_bytes(b"old bytecode")
    elif mutation == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        cache.rmdir()
        cache.symlink_to(outside, target_is_directory=True)
    elif mutation == "outside":
        monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path))
    elif mutation == "profile_sha":
        monkeypatch.setenv(runtime_paths.PROFILE_SHA_ENV, "0" * 64)
    elif mutation == "interpreter_prefix":
        monkeypatch.setattr(worker.sys, "pycache_prefix", str(tmp_path))
    else:
        monkeypatch.setattr(worker.sys, "dont_write_bytecode", False)
    monkeypatch.setattr(worker, "_factory", lambda _: pytest.fail("factory resolved before startup authorization"))
    with pytest.raises(ValueError, match="cache|SHA-256"):
        worker._startup_profile(directory)


def test_fresh_os_process_uses_real_launch_time_bytecode_settings(tmp_path, monkeypatch):
    directory, cache = prepare_profile(tmp_path, monkeypatch)
    environment = dict(os.environ)
    source = str(Path(worker.__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, "-B", "-c",
        "import json,os,sys; from transformer_rl.calibration_worker import _startup_profile; "
        "print(json.dumps({'pid':os.getpid(),'prefix':sys.pycache_prefix,"
        "'dont_write':sys.dont_write_bytecode,'profile':_startup_profile(sys.argv[1])}))",
        str(directory)], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["pid"] != os.getpid() and actual["pid"] > 0
    assert actual["prefix"] == str(cache) and actual["dont_write"] is True
    assert actual["profile"] == worker._startup_profile(directory)
    assert not list(cache.iterdir())


@pytest.fixture
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def authorized_cpu_fixture(tmp_path, monkeypatch, *, kind="train", config=None, checkpoint=None):
    """Synthetic authorization only; worker learning and storage are real CPU I/O."""
    from test_exposure_training import config as tensor_config

    config = config or tensor_config(num_envs=2)
    directory, cache = prepare_profile(tmp_path, monkeypatch)
    protocol = {"source": worker.source_identity(), "environment_factory": "synthetic_cpu_fixture:make_env",
        "execution": {"rollout_steps": 2, "device": "cpu", "max_seconds": 30.}, "protected_roots": []}
    job = {"id": "synthetic_job", "training_seed": 71, "retention_seed": 901,
        "initial_model_sha256": worker.training.initial_model_sha256(config, 71)}
    worker.campaign._new(tmp_path / "plan.json", {"fixture": "synthetic CPU authorization"})
    worker.campaign._new(tmp_path / "protocol.json", protocol)
    worker.campaign._new(tmp_path / "controller.json", {"fixture": "synthetic CPU controller lease"})
    request = {"format": "transformer_rl.calibration_worker_request", "schema_version": 1,
        "plan": worker.campaign.definition._receipt(tmp_path / "plan.json"),
        "protocol": worker.campaign.definition._receipt(tmp_path / "protocol.json"),
        "source": protocol["source"], "job_id": job["id"], "kind": kind,
        "directory": str(directory), "updates": 1,
        "controller": worker.campaign.definition._receipt(tmp_path / "controller.json"),
        "leases": [], "checkpoint": checkpoint,
        "limits": {"max_owned_bytes": 256 * 1024**2, "free_margin_bytes": 0,
                   "interval_s": .1, "timeout_s": 30.}}
    request["sha256"] = digest(request)
    worker.campaign._new(directory / "request.json", request)
    receipt = worker.campaign.definition._receipt(directory / "request.json")
    root_identity = worker.calibration_storage._identity(tmp_path.stat())
    batch = {"config": config, "stage": {"name": "first_stage", "config": config.to_dict(), "updates": 1},
        "evaluation_environment": config.environment, "cells": [],
        "evaluation": {"steps": 12, "settle_steps": 2, "min_steady_samples": 2},
        "evaluation_seed": 811, "trace_replicas": 2,
        "expected_policy_samples": 12 * config.environment["num_envs"],
        "checkpoint_receipt": checkpoint, "storage_root": str(tmp_path),
        "storage_root_identity": root_identity}
    authorizations, lease_checks = [], []

    def authorize(actual_receipt, *, child):
        assert actual_receipt == receipt and child is True
        authorizations.append(deepcopy(actual_receipt))
        return request, protocol, job, batch

    def lease(actual_protocol, leases, controller):
        assert actual_protocol == protocol and leases == [] and controller == request["controller"]
        lease_checks.append(deepcopy(controller))

    monkeypatch.setattr(worker.calibration, "validate_request", authorize)
    monkeypatch.setattr(worker.calibration, "validate_controller_lease", lease)
    # A child must never instantiate the journal-writing parent observer.
    monkeypatch.setattr(worker.calibration_storage, "CalibrationObserver",
                        lambda *a, **k: pytest.fail("worker created a second observer"))
    return SimpleNamespace(directory=directory, cache=cache, receipt=receipt, request=request,
        protocol=protocol, job=job, batch=batch, authorizations=authorizations, lease_checks=lease_checks)


def registered_cpu_factory(monkeypatch, *, make_environment=None, fail_constructor=False, fail_close=False,
                           mutate_plan=None, interpreter_prefix=None):
    """Exercise the existing application registry without constructing an SDK app."""
    from packed_env import make_env

    application = SimpleNamespace(close_calls=[], fixture="synthetic CPU application lifecycle")
    environments = []

    def close(*, wait_for_replicator, exit_code):
        application.close_calls.append((wait_for_replicator, exit_code))
        if fail_close:
            raise RuntimeError("synthetic application shutdown failure")
    application.close = close

    def factory(**kwargs):
        frame_process.register_app(application)
        if fail_constructor:
            raise RuntimeError("synthetic constructor failed after registering its application")
        environment = (make_environment or make_env)(**kwargs)
        if interpreter_prefix is not None:
            worker.sys.pycache_prefix = str(interpreter_prefix)
        if mutate_plan is not None:
            original_step = environment.step

            def mutate(*args, **options):
                result = original_step(*args, **options)
                mutate_plan.write_bytes(b"changed authorized plan bytes\n")
                return result
            environment.step = mutate
        environments.append(environment)
        return environment

    def resolve(reference):
        assert reference == "synthetic_cpu_fixture:make_env"
        return factory
    monkeypatch.setattr(worker, "_factory", resolve)
    return application, environments


def test_worker_real_cpu_learning_retains_small_reservation_and_closes_registry(tmp_path, monkeypatch, one_cpu_thread):
    fixture = authorized_cpu_fixture(tmp_path, monkeypatch)
    application, environments = registered_cpu_factory(monkeypatch)
    previous_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    assert worker.run_worker(fixture.receipt) == 0
    outcome = worker.campaign._read(fixture.directory / "outcome.json")
    completion = worker.campaign._read(worker.campaign._checked(outcome["training_completion"]))
    reservation = worker.campaign._read(fixture.directory / "train" / "reservation.json")
    assert outcome["status"] == completion["status"] == "completed"
    assert reservation["charged_updates"] == completion["charged_updates"] == 1
    assert reservation["charged_fresh_transition_budget"] == completion["actual_collected_transitions"] == 4
    assert reservation["refund"] is False and completion["automatic_retries"] == 0
    assert completion["job_id"] == "calibration_synthetic_job"
    assert outcome["actual_counters"]["completed_updates"] == 1
    assert outcome["actual_counters"]["collected_transitions"] == 4
    assert outcome["actual_counters"]["full_rollout_transitions"] == 4
    assert outcome["checkpoint"] == worker.campaign.definition._receipt(completion["endpoints"][0]["checkpoint"]["path"])
    payload = torch.load(outcome["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert payload["update"] == 1 and payload["optimizer"]["state"]
    assert outcome["environment_provenance"] == [environments[0].metadata]
    assert environments[0].closed and application.close_calls == [(False, 0)]
    assert outcome["error"] is None and outcome["shutdown_errors"] == []
    assert len(fixture.authorizations) == 2 and len(fixture.lease_checks) > 2
    assert frame_process._active is False and application not in frame_process._apps
    assert all(signal.getsignal(number) == handler for number, handler in previous_handlers.items())
    assert not list(fixture.cache.iterdir())
    assert not (tmp_path / "observer.samples.jsonl").exists()
    assert all(outcome[name] is False for name in
               ("formal_architecture_selection", "production_storage_authorized", "hardware_verified"))


@pytest.mark.parametrize("failure", ["constructor", "application_close", "plan_mutation", "interpreter_prefix"])
def test_worker_failure_keeps_real_accounting_and_closes_only_registered_apps(tmp_path, monkeypatch, one_cpu_thread, failure):
    fixture = authorized_cpu_fixture(tmp_path, monkeypatch)
    application, environments = registered_cpu_factory(monkeypatch,
        fail_constructor=failure == "constructor", fail_close=failure == "application_close",
        mutate_plan=tmp_path / "plan.json" if failure == "plan_mutation" else None,
        interpreter_prefix=tmp_path if failure == "interpreter_prefix" else None)
    assert worker.run_worker(fixture.receipt) == 1
    outcome = worker.campaign._read(fixture.directory / "outcome.json")
    completion = worker.campaign._read(worker.campaign._checked(outcome["training_completion"]))
    assert outcome["status"] == "failed" and completion["refund"] is False
    assert completion["charged_updates"] == 1 and completion["automatic_retries"] == 0
    assert len(application.close_calls) == 1 and application.close_calls[0][0] is False
    assert all(environment.closed for environment in environments)
    assert frame_process._active is False and application not in frame_process._apps
    if failure == "constructor":
        assert outcome["error"]["type"] == "RuntimeError" and application.close_calls == [(False, 1)]
        assert outcome["actual_counters"]["collected_transitions"] == 0
    elif failure == "application_close":
        assert completion["status"] == "completed" and outcome["checkpoint"] is not None
        assert any(item["owner"] == "application" for item in outcome["shutdown_errors"])
    elif failure == "plan_mutation":
        assert completion["status"] == "failed" and outcome["checkpoint"] is None
        assert any(item["owner"] == "worker_guard" for item in outcome["shutdown_errors"])
        assert 0 < outcome["actual_counters"]["collected_transitions"] < 4
    else:
        assert completion["status"] == "failed" and outcome["checkpoint"] is None
        assert outcome["actual_counters"]["collected_transitions"] == 0
        assert "interpreter bytecode settings changed" in outcome["error"]["message"]


def test_worker_real_cpu_evaluation_traces_every_scenario_row_without_grades(tmp_path, monkeypatch, one_cpu_thread):
    from test_exposure_evaluation import configuration, make_six_motor_env

    config = configuration(num_envs=4)
    teacher = worker.training.train_exposure_job(
        [{"name": "cpu_checkpoint", "config": config, "updates": 1}], make_six_motor_env,
        "synthetic_cpu_fixture:make_env", tmp_path / "checkpoint", job_id="synthetic_checkpoint",
        rollout_steps=2, training_seed=71, retention_seed=901, evaluation_seeds=[811], device="cpu",
        expected_initial_model_sha256=worker.training.initial_model_sha256(config, 71), max_seconds=30.)
    assert teacher["status"] == "completed"
    checkpoint = teacher["endpoints"][0]["checkpoint"]
    fixture = authorized_cpu_fixture(tmp_path, monkeypatch, kind="evaluate", config=config, checkpoint=checkpoint)
    application, environments = registered_cpu_factory(monkeypatch, make_environment=make_six_motor_env)
    assert worker.run_worker(fixture.receipt) == 0
    outcome = worker.campaign._read(fixture.directory / "outcome.json")
    report = worker.campaign._read(worker.campaign._checked(outcome["report"]))
    assert outcome["training_completion"] is None and outcome["checkpoint"] == checkpoint
    assert outcome["actual_counters"] == {"steps": 12, "num_envs": 4, "transitions": 48, "checkpoint_update": 1}
    assert report["seed"] == 811 and set(report["groups"]) == {"alpha", "beta"}
    assert all("grade" not in group for group in report["groups"].values())
    with np.load(worker.campaign._checked(outcome["trace"]), allow_pickle=False) as trace:
        assert trace["row_indices"].tolist() == [0, 1, 2, 3]
        assert trace["raw_policy_mean"].shape == (12, 4, 6)
        assert trace["issued_action"].shape == (12, 4, 6)
    assert environments[0].closed and application.close_calls == [(False, 0)]
    assert outcome["environment_provenance"] == [report["environment_provenance"]]
    assert all(outcome[name] is False for name in
               ("formal_architecture_selection", "production_storage_authorized", "hardware_verified"))


def test_cli_rejects_raw_request_hash_before_worker_authorization(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    worker.campaign._new(request, {"fixture": "CPU byte pin only"})
    monkeypatch.setattr(worker, "run_worker", lambda _: pytest.fail("worker called for an unauthorized raw hash"))
    with pytest.raises(ValueError, match="raw request"):
        worker.main(["worker", "--request", str(request), "--expected-request-sha256", "0" * 64])


def test_real_os_cli_accepts_controller_worker_arguments_before_rejecting_raw_bytes(tmp_path):
    request = tmp_path / "request.json"
    worker.campaign._new(request, {"fixture": "CPU CLI byte authorization only; no SDK"})
    cache = tmp_path / "empty_python_cache"
    cache.mkdir(mode=0o700)
    environment = {**os.environ, "PYTHONPATH": str(Path(worker.__file__).resolve().parents[1]),
        "PYTHONPYCACHEPREFIX": str(cache), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-B", "-m", "transformer_rl.calibration_worker", "worker",
        "--request", str(request), "--expected-request-sha256", "0" * 64],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "calibration worker raw request differs from external authorization" in result.stderr
    assert "unrecognized arguments" not in result.stderr and not list(cache.iterdir())

"""Controller-authorized continuation worker with exclusive local artifacts.

This worker consumes authorization from retention_campaign; it is not a teacher
qualification provider. Deadlines and signals are soft collection boundaries.
The legacy process registry is adapted only in this worker process, without
changing its source or intercepting its CLI.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import torch

from .checkpoint import _publish_new_files, _validate_json_value
from .cli import _StopBudget, _factory
from .experiments import source_identity
from .frame_config import FrameTrainConfig, digest, json_bytes
from .frame_continuation import FrameContinuation
from .frame_training import reward_component_scalars


_FIELDS = {"format", "schema_version", "config", "environment_factory", "checkpoint",
           "training_seed", "retention", "execution", "provenance", "source", "sha256"}
_CHECKPOINT = {"path", "sha256", "update", "cumulative_transitions", "consumed_updates"}
_RETENTION = {"seed", "coefficient", "batch_size", "anchors", "evaluation_seeds"}
_EXECUTION = {"updates", "rollout_steps", "transitions_per_update", "fresh_transition_budget",
              "consumed_update_budget", "max_seconds", "checkpoint_interval", "device",
              "tensorboard", "run_directory", "resume"}
_BRANCH = {"controller_protocol_sha256", "preparation_protocol_sha256",
           "preparation_manifest_sha256", "branch_id"}
_PROVENANCE = _BRANCH | {"controller_protocol_path", "controller_lease"}
_LEASE = {"path", "descriptor", "device", "inode", "controller_pid", "controller_start"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, fields, name):
    _require(type(value) is dict and set(value) == fields, f"invalid {name} fields")


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value, name, minimum, inclusive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid and (value >= minimum if inclusive else value > minimum), f"invalid {name}")


def _path(value, name):
    _require(type(value) is str and Path(value).is_absolute(), f"{name} must be an absolute path")
    path = Path(value)
    _require(not path.is_symlink() and str(path.resolve()) == value, f"{name} must be canonical and not a symlink")
    return path


def _file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_inputs(request):
    _require(request["source"] == source_identity(), "worker producer source bytes changed")
    checkpoint = request["checkpoint"]
    _require(_file_sha(Path(checkpoint["path"])) == checkpoint["sha256"], "worker checkpoint SHA256 mismatch")
    for item in request["retention"]["anchors"]:
        path = Path(item["path"])
        _require(path.stat().st_size == item["bytes"] and _file_sha(path) == item["sha256"],
                 "worker anchor receipt changed")


def _authorize(request):
    # Resolve the fixed producer after cheap request checks, never from a
    # caller-selected validator or an eligible flag in a request.
    from .retention_campaign import validate_worker_request
    authorized = validate_worker_request(deepcopy(request))
    _require(type(authorized) is dict and authorized == request, "controller did not authorize the exact request")


def _parent_metadata(request):
    """Read actual CPU checkpoint metadata before any environment import.

    FrameContinuation.open subsequently validates all model/Adam/RNG contents
    before it invokes the lazy factory. Only small JSON metadata survives this
    preliminary read; model and optimizer tensors are not copied.
    """
    checkpoint = request["checkpoint"]
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    _require(type(payload) is dict and payload.get("format") == "transformer_rl.packed_checkpoint"
             and type(payload.get("schema_version")) is int and payload["schema_version"] == 1
             and type(payload.get("update")) is int and payload["update"] == checkpoint["update"],
             "worker checkpoint update or format mismatch")
    metadata = payload.get("metadata")
    _require(type(metadata) is dict and type(metadata.get("seed")) is int
             and metadata["seed"] == request["training_seed"]
             and metadata.get("environment_factory") == request["environment_factory"]
             and type(metadata.get("collected_transitions")) is int
             and metadata["collected_transitions"] == checkpoint["cumulative_transitions"],
             "worker checkpoint seed, factory or transition clock mismatch")
    if request["execution"]["resume"]:
        branch = metadata.get("continuation_branch")
        _keys(branch, _BRANCH, "checkpoint continuation_branch")
        _require(branch == {key: request["provenance"][key] for key in _BRANCH},
                 "worker resume branch identity mismatch")
        _require(metadata.get("source") == request["source"]
                 and metadata.get("continuation_device") == request["execution"]["device"],
                 "worker resume source or device mismatch")
    _check_inputs(request)


def validate_request(request):
    """Validate schema, bytes and fixed-controller authorization without output."""
    _validate_json_value(request)
    _keys(request, _FIELDS, "worker request")
    _require(request["format"] == "transformer_rl.continuation_request"
             and type(request["schema_version"]) is int and request["schema_version"] == 1,
             "unsupported worker request format")
    _require(_sha(request["sha256"]) and digest({key: value for key, value in request.items() if key != "sha256"})
             == request["sha256"], "worker request canonical SHA256 mismatch")
    config = FrameTrainConfig.from_dict(request["config"])
    _require(type(request["environment_factory"]) is str and re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", request["environment_factory"]),
        "invalid worker environment factory")
    checkpoint, retention, execution, provenance = (request[key] for key in
        ("checkpoint", "retention", "execution", "provenance"))
    _keys(checkpoint, _CHECKPOINT, "checkpoint")
    _path(checkpoint["path"], "checkpoint path")
    _require(_sha(checkpoint["sha256"]), "invalid checkpoint SHA256")
    for key in ("update", "cumulative_transitions", "consumed_updates"):
        _integer(checkpoint[key], "checkpoint " + key)
    _require(checkpoint["consumed_updates"] >= checkpoint["update"], "consumed update offset is before checkpoint")
    _keys(retention, _RETENTION, "retention")
    for key, value in (("training_seed", request["training_seed"]), ("retention seed", retention["seed"])):
        _integer(value, key)
        _require(value < 2**32, key + " must be uint32")
    seeds = retention["evaluation_seeds"]
    _require(type(seeds) is list and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
             and len(seeds) == len(set(seeds)), "invalid worker evaluation seeds")
    _require(retention["seed"] not in [request["training_seed"], *seeds], "retention seed must be independent")
    _number(retention["coefficient"], "retention coefficient", 0, inclusive=True)
    _integer(retention["batch_size"], "retention batch_size", 1)
    anchors = retention["anchors"]
    _require(type(anchors) is list and bool(anchors) == (retention["coefficient"] > 0),
             "worker coefficient and anchor paths must be supplied together")
    for item in anchors:
        _keys(item, {"path", "sha256", "bytes"}, "anchor receipt")
        _path(item["path"], "anchor path")
        _require(_sha(item["sha256"]), "invalid anchor SHA256")
        _integer(item["bytes"], "anchor bytes", 1)
    _require(len({item["path"] for item in anchors}) == len(anchors), "duplicate worker anchor paths")
    _keys(execution, _EXECUTION, "execution")
    for key in ("updates", "rollout_steps", "transitions_per_update", "fresh_transition_budget",
                "consumed_update_budget", "checkpoint_interval"):
        _integer(execution[key], key, 1)
    num_envs = config.environment.get("num_envs")
    _integer(num_envs, "config num_envs", 1)
    _require(execution["transitions_per_update"] == execution["rollout_steps"] * num_envs,
             "worker transitions_per_update differs from steps times num_envs")
    _require(execution["updates"] == execution["consumed_update_budget"]
             and execution["fresh_transition_budget"] == execution["updates"] * execution["transitions_per_update"],
             "worker full update and sample budgets differ")
    _number(execution["max_seconds"], "max_seconds", 0)
    _require(type(execution["device"]) is str and (execution["device"] == "cpu"
             or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", execution["device"])), "worker device must be cpu or explicit cuda index")
    _require(type(execution["tensorboard"]) is bool and type(execution["resume"]) is bool,
             "worker tensorboard and resume must be boolean")
    run = _path(execution["run_directory"], "run_directory")
    if os.path.lexists(run):
        raise FileExistsError(f"refusing existing worker output: {run}")
    _require(run.parent.is_dir(), "worker output parent does not exist")
    _keys(provenance, _PROVENANCE, "provenance")
    _path(provenance["controller_protocol_path"], "controller protocol path")
    for key in _BRANCH - {"branch_id"}:
        _require(_sha(provenance[key]), "invalid " + key)
    _require(type(provenance["branch_id"]) is str and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}",
             provenance["branch_id"]), "invalid worker branch_id")
    leases = provenance["controller_lease"]
    _require(type(leases) is list and len(leases) == 2, "worker requires resource and study controller leases")
    for lease in leases:
        _keys(lease, _LEASE, "controller lease")
        _path(lease["path"], "controller lease path")
        for key in ("descriptor", "device", "inode", "controller_pid", "controller_start"):
            _integer(lease[key], "controller lease " + key, 0 if key in {"descriptor", "device"} else 1)
    _require(len({item["path"] for item in leases}) == 2
             and len({item["descriptor"] for item in leases}) == 2, "controller leases must be distinct")
    _check_inputs(request)
    _authorize(request)
    _parent_metadata(request)
    return config


def _write(path, value):
    _publish_new_files({path: json_bytes(value) + b"\n"})


class _BeforeEnvironmentStop(Exception):
    pass


class _ApplicationRegistry:
    def __init__(self):
        from . import frame_process
        self.process = frame_process
        self.start = None

    def activate(self):
        _require(not self.process._active, "continuation worker cannot nest in an active frame worker")
        self.start = len(self.process._apps)
        self.process._active = True

    def close(self, exit_code):
        failures = []
        if self.start is None:
            return failures
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException as error:
            failures.append({"owner": "stdio", "error": f"{type(error).__name__}: {error}"})
        try:
            for app in reversed(self.process._apps[self.start:]):
                try:
                    app.close(wait_for_replicator=False, exit_code=exit_code)
                except BaseException as error:
                    failures.append({"owner": "application", "error": f"{type(error).__name__}: {error}"})
        finally:
            del self.process._apps[self.start:]
            self.process._active = False
            self.start = None
        return failures


def run_request(request):
    """Execute one authorized attempt; failures reserve all unverified budget."""
    config = validate_request(request)
    request = deepcopy(request)
    execution, parent, retention = (request[key] for key in ("execution", "checkpoint", "retention"))
    run = Path(execution["run_directory"])
    run.mkdir(mode=0o700, exist_ok=False)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    _write(run / "run.json", request)
    started = time.monotonic()
    session = writer = log = None
    registry = _ApplicationRegistry()
    error = error_traceback = None
    shutdown = []
    stage = "startup"
    partial_rollouts = failed_optimizer_samples = failed_collection_samples = 0
    optimizer_may_be_partial = False
    last_sealed = {**parent, "bytes": Path(parent["path"]).stat().st_size}
    final_checkpoint = None
    last_step_start = last_attempt_start = 0
    stop = _StopBudget(execution["max_seconds"])
    try:
        with stop:
            registry.activate()
            try:
                def lazy_factory(**kwargs):
                    _check_inputs(request)
                    _authorize(request)
                    if stop.stopped():
                        raise _BeforeEnvironmentStop()
                    factory = _factory(request["environment_factory"])
                    env = factory(**kwargs)
                    try:
                        _require(type(env.num_envs) is int and env.num_envs == config.environment["num_envs"],
                                 "runtime environment num_envs differs from the worker budget")
                        stop.install()
                        return env
                    except BaseException:
                        try:
                            env.close()
                        except BaseException as caught:
                            shutdown.append({"owner": "environment", "error": f"{type(caught).__name__}: {caught}"})
                        raise

                session = FrameContinuation.open(config, lazy_factory, request["environment_factory"], parent["path"],
                    checkpoint_sha256=parent["sha256"], parent_update=parent["update"],
                    cumulative_transitions=parent["cumulative_transitions"], consumed_updates=parent["consumed_updates"],
                    rollout_steps=execution["rollout_steps"], training_seed=request["training_seed"],
                    retention_seed=retention["seed"], anchors=[item["path"] for item in retention["anchors"]],
                    retention_coef=retention["coefficient"], retention_batch_size=retention["batch_size"],
                    evaluation_seeds=retention["evaluation_seeds"], device=execution["device"], resume=execution["resume"])
                stop.install()
                session.metadata["continuation_branch"] = {key: request["provenance"][key] for key in _BRANCH}
                session.metadata["continuation_request_sha256"] = request["sha256"]
                _write(run / "environment.json", session.metadata["environment_provenance"])
                if execution["tensorboard"]:
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(str(run / "tensorboard"))
                log = (run / "metrics.jsonl").open("x", encoding="utf-8", buffering=1)
                while session.update - parent["update"] < execution["updates"] and not stop.stopped():
                    fresh = session.collector.total_transitions
                    _require(type(session.env.num_envs) is int and session.env.num_envs == config.environment["num_envs"],
                             "runtime environment num_envs changed before step")
                    _require(session.attempted_updates < execution["consumed_update_budget"]
                             and fresh + execution["transitions_per_update"] <= execution["fresh_transition_budget"],
                             "remaining worker budget cannot fit a complete rollout")
                    last_step_start, last_attempt_start = fresh, session.attempted_updates
                    stage = "step"
                    record = session.step(should_stop=stop.stopped)
                    delta = session.collector.total_transitions - last_step_start
                    _require(0 <= delta <= execution["transitions_per_update"]
                             and session.collector.total_transitions <= execution["fresh_transition_budget"]
                             and session.attempted_updates <= execution["consumed_update_budget"],
                             "runtime worker collection exceeded its frozen budget")
                    if record is None:
                        partial_rollouts += int(delta > 0)
                        break
                    _require(delta == execution["transitions_per_update"], "completed worker update has wrong sample count")
                    stage = "metrics"
                    record["elapsed_s"] = time.monotonic() - started
                    diagnostics = getattr(session.env, "training_diagnostics", None)
                    if diagnostics is not None:
                        record["environment_diagnostics"] = diagnostics()
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    if writer is not None:
                        for group, items in (("ppo", record["optimization"]), ("rollout", record["collection"]),
                                             ("environment", record.get("environment_diagnostics", {}))):
                            for name, value in items.items():
                                if type(value) in (int, float, bool):
                                    writer.add_scalar(f"{group}/{name}", value, session.update)
                        for name, value in reward_component_scalars(record["collection"]).items():
                            writer.add_scalar(name, value, session.update)
                    print(json.dumps({"update": session.update, "transitions": session.collector.total_transitions}), flush=True)
                    if session.update % execution["checkpoint_interval"] == 0:
                        stage = "checkpoint"
                        _check_inputs(request)
                        path = checkpoints / f"update_{session.update:06d}.pt"
                        receipt = session.save(path)
                        last_sealed = {"path": str(path), "sha256": receipt["sha256"], "bytes": path.stat().st_size,
                            "update": session.update, "cumulative_transitions": session.collected_transitions,
                            "consumed_updates": session.consumed_updates}
                stage = "final_checkpoint"
                _check_inputs(request)
                final_path = checkpoints / "final.pt"
                receipt = session.save(final_path)
                last_sealed = {"path": str(final_path), "sha256": receipt["sha256"], "bytes": final_path.stat().st_size,
                    "update": session.update, "cumulative_transitions": session.collected_transitions,
                    "consumed_updates": session.consumed_updates}
                # A file left behind by a failed publication is not a sealed
                # final checkpoint and must not inherit an earlier receipt.
                final_checkpoint = final_path
            except _BeforeEnvironmentStop:
                stage = "before_environment_stop"
            except BaseException as caught:
                error, error_traceback = caught, caught.__traceback__
                if stage == "step" and session is not None:
                    delta = session.collector.total_transitions - last_step_start
                    if session.attempted_updates > last_attempt_start:
                        optimizer_may_be_partial = True
                        failed_optimizer_samples = delta
                    else:
                        failed_collection_samples = delta
            finally:
                for name, owner in (("metrics", log), ("tensorboard", writer), ("environment", session)):
                    if owner is not None:
                        try:
                            owner.close()
                        except BaseException as caught:
                            shutdown.append({"owner": name, "error": f"{type(caught).__name__}: {caught}"})
                shutdown.extend(registry.close(1 if error is not None or shutdown else 0))
    except BaseException as caught:
        if error is None:
            error, error_traceback = caught, caught.__traceback__
        shutdown.extend(registry.close(1))
    if error is None and shutdown:
        error = RuntimeError("continuation worker shutdown failed: " + "; ".join(item["error"] for item in shutdown))
        stage = "shutdown"
    actual = session.collector.total_transitions if session is not None and session.collector is not None else 0
    update = session.update if session is not None else parent["update"]
    attempts = session.attempted_updates if session is not None else 0
    completed = update - parent["update"]
    partial = session.discarded_transitions if session is not None else 0
    status = "failed" if error is not None else "completed" if completed == execution["updates"] else "stopped"
    report = {"format": "transformer_rl.continuation_execution", "schema_version": 1,
        "status": status, "request_sha256": request["sha256"], "provenance": deepcopy(request["provenance"]),
        "start_update": parent["update"], "final_update": update, "requested_updates": execution["updates"],
        "completed_updates": completed, "attempted_updates": attempts, "actual_samples": actual,
        "consumed_transitions": actual, "cumulative_transitions": parent["cumulative_transitions"] + actual,
        "consumed_updates": parent["consumed_updates"] + attempts,
        "partial_rollouts": partial_rollouts, "partial_samples": partial,
        "discarded_samples": max(0, actual - completed * execution["transitions_per_update"]),
        "failed_optimizer_samples": failed_optimizer_samples, "failed_collection_samples": failed_collection_samples,
        "unsealed_successful_updates": max(0, update - last_sealed["update"]),
        "unsealed_samples": max(0, parent["cumulative_transitions"] + actual - last_sealed["cumulative_transitions"]),
        "optimizer_update_may_be_partial": optimizer_may_be_partial,
        "reserved_update_budget": max(0, execution["consumed_update_budget"] - attempts) if status != "completed" else 0,
        "reserved_transition_budget": max(0, execution["fresh_transition_budget"] - actual) if status != "completed" else 0,
        "stop_reason": stop.reason, "failure_stage": stage if error is not None else None,
        "last_sealed_checkpoint": last_sealed,
        "checkpoint": str(final_checkpoint) if final_checkpoint is not None and final_checkpoint.exists() else None,
        "checkpoint_sha256": last_sealed["sha256"] if final_checkpoint is not None and final_checkpoint.exists() else None,
        "elapsed_s": time.monotonic() - started, "source": deepcopy(request["source"]),
        "shutdown_errors": shutdown, "episode_state_restored": False, "history_reset": "repeat_first",
        "sample_count_scope": "collector-confirmed returned vector environment steps; failed step side effects are not estimated",
        "retry_policy": "reserve_remaining_budget_no_automatic_retry"}
    if session is not None and session.collector is not None:
        components = session.collector.last_metrics.get("reward_components")
        if components is not None:
            report["last_collection_reward_components"] = components
    if error is not None:
        report["error"] = f"{type(error).__name__}: {error}"
    _write(run / ("failure.json" if error is not None else "completion.json"), report)
    if error is not None:
        raise error.with_traceback(error_traceback)
    return report


def load_request(path):
    """Read one canonical JSON request; reject duplicate keys and nonfinite data."""
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate worker request JSON key")
            value[key] = item
        return value

    raw = _path(str(path), "request path").read_bytes()
    request = json.loads(raw, object_pairs_hook=unique,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite worker request JSON")))
    _require(raw == json_bytes(request) + b"\n", "worker request file is not canonical JSON")
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute one controller-authorized continuation request")
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = run_request(load_request(arguments.request))
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

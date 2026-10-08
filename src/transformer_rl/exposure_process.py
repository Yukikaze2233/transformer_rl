"""One fixed-authorized OS stage with full learning-state continuation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import traceback

from . import exposure_campaign as campaign
from . import exposure_training as training
from .cli import _factory
from .continuation_process import _ApplicationRegistry
from .experiments import source_identity
from .frame_continuation import FrameContinuation
from .ppo import PPOTrainer
from . import runtime_paths


def run_request(request, request_receipt):
    protocol, job, stage, contract = campaign.validate_stage_request(request, request_receipt)
    directory = Path(request_receipt["path"]).parent
    cache = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
    campaign._require(cache == directory / "empty_python_cache" and cache.is_dir()
                      and not list(cache.iterdir()), "worker did not start with a new empty bytecode cache")
    runtime_profile = runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))
    own = campaign._identity(os.getpid())
    registry = _ApplicationRegistry()
    numerical = {"origin": None}
    errors, caught, completion = [], None, None
    stop_requested = False
    original_handlers = {}
    optimizer_update = PPOTrainer.update
    continuation_save = FrameContinuation.save
    output_root = Path(request["output_root"])
    checkpoint_records = campaign.learning_checkpoint_records(protocol, job, stage)
    checkpoints = {Path(item["checkpoint_path"]): item["checkpoint_update"] for item in checkpoint_records}
    checkpoint_caps = ({path: contract["caps"]["checkpoint_bytes"] for path in checkpoints}
                       if "checkpoint_updates" in stage else None)

    def on_signal(number, frame):
        nonlocal stop_requested
        stop_requested = True

    def guard():
        if stop_requested:
            return True
        campaign._checked(request_receipt)
        campaign._require(runtime_paths.profile_receipt(runtime_paths.validate_runtime_profile(directory))
                          == runtime_profile, "worker runtime profile changed")
        campaign.validate_controller_lease(protocol, request["leases"], request["controller"])
        campaign._require(source_identity() == protocol["source"], "worker learner source changed")
        # Only this stage's declared data may spend its output reserve. SDK,
        # stdout and publication temporaries cannot consume future jobs' caps.
        campaign.worker_storage_guard(protocol, contract, directory,
            request["required_remaining_bytes"], active_worker=True, checkpoint_paths=checkpoints,
            checkpoint_limit=len(checkpoints) * contract["caps"]["checkpoint_bytes"],
            checkpoint_byte_limits=checkpoint_caps,
            metric_paths=(output_root / "metrics.jsonl",),
            metric_limit=stage["updates"] * contract["caps"]["metric_bytes_per_update"])
        return False

    def guarded_save(self, path):
        campaign._require(Path(path) in checkpoints and self.update == checkpoints[Path(path)],
                          "worker checkpoint publication path or update differs")
        # The final PPO update and metric publication occur after the last
        # collector guard. Recheck before creating an immutable endpoint.
        campaign._require(not guard(), "worker stopped before checkpoint publication")
        result = continuation_save(self, path)
        campaign._require(not guard(), "worker stopped after checkpoint publication")
        return result

    def typed_optimizer(self, *args, **kwargs):
        try:
            return optimizer_update(self, *args, **kwargs)
        except BaseException as error:
            if type(error) is FloatingPointError:
                numerical["origin"] = "optimizer_update"
            raise

    def lazy_factory(**kwargs):
        campaign.validate_controller_lease(protocol, request["leases"], request["controller"])
        campaign._require(not guard(), "worker stopped before environment construction")
        env = _factory(protocol["environment_factory"])(**kwargs)
        original_step = env.step

        def typed_step(*args, **options):
            try:
                return original_step(*args, **options)
            except BaseException as error:
                if type(error) is FloatingPointError:
                    numerical["origin"] = "environment_step"
                raise
        env.step = typed_step
        return env

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            original_handlers[number] = signal.signal(number, on_signal)
        registry.activate()
        # This interception records the actual built-in exception object;
        # an exception's message or a completion string cannot grant permission
        # to continue other jobs. It is scoped to this disposable OS worker.
        PPOTrainer.update = typed_optimizer
        FrameContinuation.save = guarded_save
        completion = training.train_exposure_segment(
            campaign.stage_definition(stage),
            lazy_factory, protocol["environment_factory"], request["output_root"],
            job_id=job["id"], rollout_steps=protocol["execution"]["rollout_steps"],
            training_seed=job["training_seed"], retention_seed=job["retention_seed"],
            evaluation_seeds=[*protocol["evaluation"]["validation_seeds"], *protocol["evaluation"]["seeds"]],
            device=protocol["execution"]["device"], expected_initial_model_sha256=job["initial_model_sha256"],
            max_seconds=protocol["execution"]["max_seconds"], parent_endpoint=request["parent_endpoint"],
            should_stop=guard, protected_paths=protocol["protected_roots"])
        if completion["status"] == "completed":
            campaign._require(not guard(), "worker was interrupted after learning publication")
    except BaseException as error:
        caught = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        PPOTrainer.update = optimizer_update
        FrameContinuation.save = continuation_save
        errors.extend(registry.close(0 if completion is not None and completion["status"] == "completed" else 1))
        try:
            campaign._require(not guard(), "worker stopped during shutdown")
        except BaseException as error:
            errors.append({"owner": "worker_guard", "error": f"{type(error).__name__}: {error}"})
        for number, previous in original_handlers.items():
            signal.signal(number, previous)
    status = "unknown_failure"
    if completion is not None and completion["status"] == "completed" and caught is None and not errors:
        status = "completed"
    elif (completion is not None and completion["status"] == "failed" and caught is None and not errors
          and numerical["origin"] is not None and completion.get("shutdown_errors") == []
          and completion.get("error", {}).get("type") == "FloatingPointError"
          and completion.get("error", {}).get("phase") == "collect_optimize"):
        status = "numerical_failure"
    outcome = {"format": "transformer_rl.exposure_stage_outcome", "schema_version": 1,
        "request": deepcopy(request_receipt), "source": source_identity(), "process": own,
        "status": status, "typed_numerical_origin": numerical["origin"],
        "runtime_profile": runtime_profile,
        "training_completion": campaign.definition._receipt(Path(request["output_root"]) / "completion.json")
            if completion is not None else None,
        "shutdown_errors": errors, "error": caught}
    campaign._new(directory / "outcome.json", outcome)
    return 0 if status == "completed" else 20 if status == "numerical_failure" else 30


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--expected-request-sha256", required=True)
    args = parser.parse_args(argv)
    receipt = campaign.definition._receipt(args.request)
    campaign._require(receipt["sha256"] == args.expected_request_sha256,
                      "worker raw request differs from external authorization")
    return run_request(campaign._read(args.request), receipt)


if __name__ == "__main__":
    raise SystemExit(main())

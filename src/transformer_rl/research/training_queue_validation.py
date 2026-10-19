"""Reuse frozen research predicates exclusively for new managed artifacts."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path

from ..common.training_control import file_pin, read_json, verify_pin, write_json


def load_helpers(pin, dependencies=()):
    for dependency in dependencies:
        verify_pin(dependency)
    path = verify_pin(pin)
    spec = importlib.util.spec_from_file_location("managed_frozen_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "training_evidence")
    source = ast.get_source_segment(path.read_text(), node)
    before = 'or control.process_start(process["pid"]) == process["start"]'
    after = ('or process.get("boot_id") != manifest["authorized_boot"]\n'
             '            or process.get("actual_waitpid_returncode") not in (0, 2)')
    if source.count(before) != 1 or source.count('{"time_budget", "SIGINT", "SIGTERM"}') != 1:
        raise ValueError("unsupported frozen training acceptance predicate")
    source = source.replace(before, after).replace(
        '{"time_budget", "SIGINT", "SIGTERM"}', '{"time_budget", "SIGINT", "SIGTERM", "control_stop"}')
    namespace = dict(module.__dict__)
    exec(compile(source, str(path) + ":managed-endpoint", "exec"), namespace)
    # All update/rollout/Adam/RNG/config/provenance predicates remain the frozen
    # predicates. Only real waitpid evidence and explicit cooperative stop differ.
    return module, namespace["training_evidence"], source


def validate_training(queue, session, attempt, parent_evidence=None):
    attempt = Path(attempt).resolve(strict=True)
    if not attempt.is_relative_to(queue.root / "sessions"):
        raise ValueError("only new managed training attempts may be consumed")
    cached = read_json(attempt / "research_validation.json")
    if cached:
        return cached["evidence"]
    exit_receipt = read_json(attempt / "exit.json")
    launch = read_json(attempt / "launch.json")
    if (not exit_receipt or exit_receipt["returncode"] not in (0, 2) or exit_receipt["identity"] != launch
            or launch["boot_id"] != queue.spec["authorized_boot"]):
        raise ValueError("managed child has no matching successful waitpid receipt")
    birth = read_json(attempt / "request.json")
    result = read_json(attempt / "control/result.json")
    if not result or birth["token"] != result["token"]:
        raise ValueError("uncommitted managed learning endpoint")
    if result["summary"]["phase"] != ("completed" if exit_receipt["returncode"] == 0 else "stopped"):
        raise ValueError("actual exit status differs from the learning endpoint")
    module, validator, source = load_helpers(queue.spec["verifier"], queue.spec.get("verifier_dependencies", []))
    manifest = deepcopy(queue.parent_manifest)
    manifest.update(output_root=str(queue.root), source_root=queue.spec["learner_root"],
                    authorized_boot=queue.spec["authorized_boot"],
                    learner_max_seconds=session.spec["max_seconds"] - birth["prior_active_elapsed_s"])
    manifest["runtime"] = queue.spec["runtime"]
    parent = None
    if parent_evidence:
        parent = parent_evidence["checkpoint"]
    config = read_json(session.root / "config.json")
    request = {"parent": parent, "runtime": manifest["runtime"], "config": config,
               "seed": session.spec["seed"], "updates": session.spec["updates"] - birth["start_update"],
               "start_update": birth["start_update"], "prior_transitions": birth["prior_transitions"],
               "initial_model_sha256": session.spec["expected_initial_model_sha256"],
               "source": queue.spec["learner_identity"], "command": launch["argv"],
               "config_sha256": module.control.digest(config)}
    write_json(attempt / "worker.process.json", {
        "command": launch["argv"], "pid": launch["pid"], "start": str(launch["start_ticks"]),
        "status": "finished", "returncode": exit_receipt["returncode"], "boot_id": launch["boot_id"],
        "actual_waitpid_returncode": exit_receipt["returncode"], "receipt_origin": "managed_child_waitpid"})
    evidence = validator(manifest, request, attempt, parent_evidence)
    write_json(attempt / "research_validation.json", {
        "evidence": evidence, "request": request, "control_result_pin": file_pin(attempt / "control/result.json"),
        "exit_pin": file_pin(attempt / "exit.json"), "frozen_verifier": queue.spec["verifier"],
        "managed_predicate_source": source, "old_research_artifacts_consumed": False})
    return evidence


def validate_evaluation(queue, job, checkpoint, seed, directory):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_relative_to(queue.root / "evaluations"):
        raise ValueError("only new managed evaluation outputs may be consumed")
    cached = read_json(directory / "research_validation.json")
    if cached:
        return cached
    module, _, _ = load_helpers(queue.spec["verifier"], queue.spec.get("verifier_dependencies", []))
    manifest = deepcopy(queue.parent_manifest)
    manifest["output_root"] = str(queue.root)
    configs, shim = module.evaluation_inputs(manifest, job, checkpoint)
    artifacts = module.diagnostic.verify_outputs(directory, shim, job["variant"], seed)
    result = {"status": "completed", "cell": queue.cell_key(job), "evaluation_seed": seed,
              "checkpoint": checkpoint, "directory": str(directory), "artifacts": artifacts,
              "exit_pin": file_pin(directory / "exit.json"), "frozen_verifier": queue.spec["verifier"]}
    write_json(directory / "research_validation.json", result)
    return result

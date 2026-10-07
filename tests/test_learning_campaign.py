"""Real ledger invariants and fault paths, using CPU fixtures only."""
import argparse
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from contextlib import contextmanager
import py_compile

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("learning_campaign", ROOT / "tools/run_learning_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)
control = campaign.control
ACTUAL_PREPARATION_VALIDATOR = campaign.validate_prepared
ACTUAL_RUNTIME_IDENTITY = campaign.runtime_identity
from test_learning_study import template


def replace_json(path, value):
    campaign.write(path, value, replace=True)


def resign(path, value):
    value = deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = control.digest(value)
    replace_json(path, value)
    return value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs, source, output = [tmp_path / name for name in ("prepared", "source", "output")]
    package = source / "src/transformer_rl"
    package.mkdir(parents=True)
    (package / "frame_process.py").write_text("# CPU fixture; never executed\n")
    (package / "frame_workflow.py").write_text("# initialization guard fixture\n")
    names = [f"variant_{index:02d}" for index in range(10)]
    cases = [f"case_{index:02d}" for index in range(50)]
    rates, seeds = [1e-5, 3e-5, 1e-4], [1101, 1102, 1103]
    protocol = {"expected_training_jobs": 90, "updates_per_job": 1200, "planned_transitions_per_job": campaign.TOTAL,
        "planned_total_transitions": campaign.TOTAL * 90, "rollout_steps": 48, "num_envs": 1024,
        "retention_coef": 0., "variants": names, "training_seeds": seeds, "learning_rates": rates,
        "development": {"validation_seeds": [701, 1701], "packed_final_seeds": [2701, 3701]},
        "evaluation": {"case_names": cases}}
    initializations = [{"variant": name, "training_seed": seed, "initial_model_sha256": control.digest([name, seed]),
        "model_sha256": control.digest({"variant": name})} for name in names for seed in seeds]
    manifest = {"format": "transformer_rl.learning_rate_study", "schema_version": 1,
        "protocol": protocol, "source": campaign.learner_identity(source), "initializations": initializations,
        "children": [], "cells": []}
    population = {"stand": 1024}
    campaign.write(inputs / "parent/transfer_design.json", {"scene_allocation": {"requested_counts": population}})
    for index, rate in enumerate(rates):
        name = f"rate_{index:03d}"
        packed = inputs / name / "study"
        (packed / "jobs").mkdir(parents=True)
        plan = {"source": manifest["source"], "spec": {"environment_factory": campaign.FACTORY,
            "variants": [{"name": n} for n in names], "seeds": seeds,
            "stages": [{"scenarios": cases}]}, "configs": {}}
        child = {"id": name, "learning_rate": rate, "root": f"{name}/study", "configurations": []}
        for variant in names:
            config = {"model": {"variant": variant}, "ppo": {"learning_rate": rate, "epochs": 5, "num_minibatches": 32},
                "control": {"policy_dt_s": .01}, "environment": {"num_envs": 1024,
                    "snapshot_sha256": "1" * 64, "snapshot": str(inputs / "snapshot"),
                    "contract_sha256": "2" * 64}}
            routes = [f"{variant}.train.transfer.json", *(f"{variant}.eval.{case}.json" for case in cases)]
            for suffix in routes:
                configured = deepcopy(config)
                if ".eval." in suffix:
                    configured["environment"]["num_envs"] = 8
                path = packed / "configs" / suffix
                campaign.write(path, configured)
                route = str(path.relative_to(inputs))
                item = {"path": route, "sha256": control.file_sha(path), "canonical_sha256": control.digest(configured)}
                child["configurations"].append(item)
                plan["configs"][f"configs/{suffix}"] = control.digest(configured)
            training_route = f"{name}/study/configs/{variant}.train.transfer.json"
            receipt = next(v for v in child["configurations"] if v["path"] == training_route)
            for seed in seeds:
                init = next(v for v in initializations if (v["variant"], v["training_seed"]) == (variant, seed))
                manifest["cells"].append({"rate_id": name, "learning_rate": rate, "variant": variant,
                    "training_seed": seed, "job": f"{name}/study/jobs/{variant}/seed_{seed}",
                    "training_config": receipt, "initial_model_sha256": init["initial_model_sha256"],
                    "updates": 1200, "rollout_steps": 48, "num_envs": 1024, "planned_transitions": campaign.TOTAL})
        plan["sha256"] = control.digest(plan)
        campaign.write(packed / "plan.json", plan)
        manifest["children"].append(child)
    manifest["sha256"] = control.digest(manifest)
    campaign.write(inputs / "manifest.json", manifest)
    resource, study_lock = tmp_path / "resource/.run.lock", tmp_path / "old_study/.run.lock"
    resource.parent.mkdir()
    study_lock.parent.mkdir()
    resource.touch()
    study_lock.touch()
    dependencies = {}
    for name, (kind, field) in campaign.DEPENDENCIES.items():
        root = tmp_path / name
        definition = {"format": kind, "resource_lock": str(resource), "source": {
            "root": str(source), "files": {"src/transformer_rl/" + route: sha
                                           for route, sha in manifest["source"]["files"].items()}}}
        if name == "diagnostics":
            definition.update(inputs={"variants": names}, protocol={"seeds": [8701, 9701]})
            definition["sha256"] = control.digest(definition)
        route = root / ("manifest.json" if name == "diagnostics" else "campaign.json")
        campaign.write(route, definition)
        expected = definition["sha256"] if name == "diagnostics" else control.digest(definition)
        campaign.write(root / "summary.json", {field: expected, "status": "waiting", "results": {}})
        dependencies[name] = {"definition": str(route), "sha256": expected, "summary": str(root / "summary.json"),
                              "controller": {"pid": 987654321, "start": "12345"}}
    dependency_path = tmp_path / "dependencies.json"
    campaign.write(dependency_path, dependencies)
    observed = []
    def validate_prepared(root, source_root):
        observed.append((root, source_root))
        return {"sha256": manifest["sha256"], "status": "prepared"}
    monkeypatch.setattr(campaign, "validate_prepared", validate_prepared)
    runtime = {"checkpoint_runtime": {"python": "fixture", "torch": "fixture", "numpy": "fixture", "cuda": None,
        "deterministic_algorithms": False}, "executable": {"requested": sys.executable, "resolved": sys.executable,
        "sha256": "e" * 64, "device": 1, "inode": 2}, "default_dtype": "torch.float32", "cuda_initialized": False}
    monkeypatch.setattr(campaign, "runtime_identity", lambda *args: deepcopy(runtime))
    args = argparse.Namespace(prepared_root=inputs, source_root=source, output_root=output,
        resource_lock=resource, study_lock=study_lock, dependencies=dependency_path,
        prepared_manifest_sha256=manifest["sha256"], device="cpu", max_run_attempts=8,
        learner_max_seconds=604800., worker_timeout_seconds=605100.)
    frozen = campaign.prepare(args)
    return args, frozen, manifest, observed


def metric(update, count=49152, total=None, applied=32):
    total = count if total is None else total
    return {"update": update, "batch_samples": count,
        "optimization": {"optimizer_steps": applied, "planned_optimizer_steps": 160,
            "sample_count": applied * count // 32, "early_stopped": applied < 160,
            "first_step_kl": 1e-6, "final_kl": .001},
        "collection": {"vector_steps": count // 1024, "transitions": count, "total_steps": total // 1024,
            "total_transitions": total, "early_stopped": count != 49152}}


def endpoint(prepared, monkeypatch, *, updates=2, start=0, parent=None, prior=None, directory=None):
    args, frozen, manifest, _ = prepared
    cell = manifest["cells"][0]
    if directory is None:
        directory = args.output_root / "cells" / campaign.cell_key(cell) / "training/attempt_0000"
    request = campaign.training_request(frozen, cell, directory, start, parent)
    if not (directory / "request.json").exists():
        campaign.write(directory / "request.json", request)
    else:
        assert campaign.read(directory / "request.json") == request
    run = {"config": deepcopy(request["config"]), "seed": request["seed"], "environment_factory": campaign.FACTORY,
        "updates": request["updates"], "rollout_steps": 48, "device": "cpu", "source": request["source"],
        "resume": parent["path"] if parent else None, "initialize_from": None, "restore_learning_from": None,
        "episode_state_restored": False, "history_reset": "repeat_first", "retention_coef": 0.,
        "initial_model_hash_format": "sorted_named_tensor_contents_v1", "max_seconds": frozen["learner_max_seconds"], "checkpoint_interval": 1200,
        "initial_model_sha256": prior["model_sha256"] if prior else cell["initial_model_sha256"]}
    if not parent:
        run["initialization_guard"] = {"expected_sha256": cell["initial_model_sha256"],
                                      "actual_sha256": cell["initial_model_sha256"], "verified": True}
    campaign.write(directory / "train/run.json", run)
    campaign.write(directory / "worker.process.json", {"status": "finished", "command": request["command"],
        "pid": 987654321, "start": "123456", "returncode": 0 if updates == request["updates"] else 2, "timed_out": False})
    environment = {"identity": "1" * 64, "control_sha256": control.digest(request["config"]["control"]),
        "contract_sha256": "2" * 64, "policy_hz": 100., "physics_hz": 200., "startup": {"scene_group_counts": {"stand": 1024}}}
    campaign.write(directory / "train/environment.json", environment)
    rows = [metric(start + index + 1, total=(index + 1) * campaign.SAMPLES) for index in range(updates)]
    (directory / "train/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = directory / "train/checkpoints/final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(f"synthetic {start} {updates}".encode())
    completion = {"status": "completed" if updates == request["updates"] else "stopped", "stop_reason": "time_budget",
        "start_update": start, "final_update": start + updates, "completed_updates": updates,
        "attempted_updates": updates, "consumed_transitions": updates * campaign.SAMPLES,
        "cumulative_transitions": (start + updates) * campaign.SAMPLES,
        "config_sha256": request["config_sha256"], "checkpoint": str(checkpoint), "checkpoint_sha256": control.file_sha(checkpoint)}
    campaign.write(directory / "train/completion.json", completion)
    metadata = {"environment_factory": campaign.FACTORY, "environment_provenance": environment,
        "seed": request["seed"], "source": request["source"], "anchors": [], "retention_coef": 0.,
        "initial_model_sha256": run["initial_model_sha256"], "initial_model_hash_format": run["initial_model_hash_format"],
        "episode_state_restored": False, "collected_transitions": completion["cumulative_transitions"],
        "runtime": frozen["runtime"]["checkpoint_runtime"]}
    if not parent:
        metadata["initialization_guard"] = run["initialization_guard"]
    total_steps = (prior["cumulative_optimizer_steps"] if prior else 0) + updates * 32
    probe = {"checkpoint_sha256": control.file_sha(checkpoint), "config": deepcopy(request["config"]), "update": start + updates,
        "metadata": metadata, "model_sha256": control.digest([start, updates]), "optimizer_sha256": "a" * 64,
        "rng_sha256": "b" * 64, "optimizer_groups": [{"lr": cell["learning_rate"], "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0, "amsgrad": False}], "optimizer_state_count": 10, "optimizer_parameter_count": 10,
        "optimizer_step_min": total_steps, "optimizer_step_max": total_steps,
        "rng_validated": True, "model_validated": True, "optimizer_validated": True, "cuda_initialized": False,
        "optimizer_recipe_matches_fresh": True}
    campaign.write(str(checkpoint) + ".json", {"format": "transformer_rl.packed_checkpoint", "schema_version": 1,
        "sha256": control.file_sha(checkpoint), "config": request["config"],
        "update": start + updates, "metadata": metadata})
    monkeypatch.setattr(campaign, "checkpoint_probe", lambda *args: deepcopy(probe))
    return directory, request, run, completion, probe


def test_preparation_freezes_inputs_only_and_empty_grid_is_not_ready(prepared):
    args, frozen, _, observed = prepared
    before = campaign.inventory(args.prepared_root)
    assert len(observed) == 1
    assert len(frozen["inputs"]["cells"]) == 90
    assert not (args.output_root / "cells").exists()
    assert campaign.validate(args.output_root / "manifest.json") == frozen
    report = campaign.audit(args.output_root / "manifest.json")
    assert report["status"] == "not_ready"
    assert report["actual_cells"] == report["expected_cells"] == 90
    assert report["completed_training_cells"] == report["completed_development_cells"] == 0
    assert all(item["training"] is None and all(v is None for v in item["development"].values()) for item in report["cells"].values())
    assert not report["selection_implemented"]
    assert not report["confirmation_implemented"]
    assert campaign.inventory(args.prepared_root) == before


@pytest.mark.parametrize("mutation", ["missing_cell", "duplicate_cell", "initial_sha", "source", "extra_jobs", "paired_config", "budget"])
def test_prepared_semantic_and_complete_grid_reject_mutations(prepared, mutation):
    args, _, manifest, _ = prepared
    if mutation == "missing_cell":
        manifest["cells"].pop()
    elif mutation == "duplicate_cell":
        manifest["cells"][-1] = manifest["cells"][0]
    elif mutation == "initial_sha":
        manifest["cells"][0]["initial_model_sha256"] = "f" * 64
    elif mutation == "source":
        (args.source_root / "src/transformer_rl/frame_workflow.py").write_text("# changed\n")
    elif mutation == "extra_jobs":
        campaign.write(args.prepared_root / "rate_000/study/jobs/illegal.json", {})
    elif mutation == "paired_config":
        path = args.prepared_root / manifest["cells"][30]["training_config"]["path"]
        config = campaign.read(path)
        config["ppo"]["epochs"] = 6
        replace_json(path, config)
        item = manifest["children"][1]["configurations"][0]
        # Re-signing a manifest cannot bless a changed training recipe.
        for entry in manifest["children"][1]["configurations"]:
            if entry["path"] == str(path.relative_to(args.prepared_root)):
                entry.update(sha256=control.file_sha(path), canonical_sha256=control.digest(config))
    else:
        manifest["protocol"]["planned_total_transitions"] -= 1
    changed = resign(args.prepared_root / "manifest.json", manifest)
    with pytest.raises((ValueError, KeyError)):
        campaign.prepared_identity(args.prepared_root, args.source_root, changed["sha256"])


@pytest.mark.parametrize("route", ["resource_lock", "study_lock", "own"])
def test_replaced_lock_inode_rejected(prepared, route):
    args, _, _, _ = prepared
    path = args.output_root / ".learning.lock" if route == "own" else getattr(args, route)
    old = path.with_suffix(".old")
    path.rename(old)
    path.touch()
    with pytest.raises(ValueError, match="inode"):
        campaign.validate(args.output_root / "manifest.json")


def test_dependency_definition_and_helpers_are_frozen(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    monkeypatch.setattr(campaign, "controllers", lambda: {})
    with pytest.raises(ValueError, match="helpers"):
        campaign.validate(args.output_root / "manifest.json")
    monkeypatch.undo()
    path = Path(frozen["dependencies"]["transfer"]["definition"]["path"])
    definition = campaign.read(path)
    definition["resource_lock"] = "different"
    replace_json(path, definition)
    with pytest.raises(ValueError):
        campaign.validate(args.output_root / "manifest.json")


def test_dynamic_wait_summary_does_not_invalidate_immutable_definition(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    dependency = frozen["dependencies"]["transfer"]
    path = Path(dependency["summary"])
    before = campaign.dependency_state(dependency)
    assert not before["ready"]
    summary = campaign.read(path)
    summary.update(status="training", updated_at="later")
    replace_json(path, summary)
    after = campaign.dependency_state(dependency)
    assert not after["ready"] and after["summary"] != before["summary"]
    assert campaign.validate(args.output_root / "manifest.json") == frozen
    monkeypatch.setattr(control, "process_start", lambda pid: dependency["controller"]["start"])
    assert campaign.dependency_state(dependency)["controller_live"]


def test_dependency_completed_label_with_missing_coverage_is_rejected(prepared):
    _, frozen, _, _ = prepared
    dependency = frozen["dependencies"]["diagnostics"]
    summary = campaign.read(dependency["summary"])
    summary["status"] = "completed"
    replace_json(dependency["summary"], summary)
    with pytest.raises(ValueError, match="coverage"):
        campaign.dependency_state(dependency)


def test_observation_timeout_keeps_waiting_and_never_spawns_worker(prepared, monkeypatch):
    args, _, _, _ = prepared
    monkeypatch.setattr(campaign, "worker", lambda *args: pytest.fail("no worker may start while dependencies are incomplete"))
    runner = argparse.Namespace(manifest=args.output_root / "manifest.json", max_wait_seconds=0., poll_seconds=.01)
    result = campaign.run(runner)
    assert result["status"] == "waiting"
    assert not (args.output_root / "cells").exists()
    assert not campaign.read(args.output_root / "controller.json")["start"] is None


def test_lock_held_wait_does_not_steal_resource(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    monkeypatch.setattr(campaign, "dependency_state", lambda d: {"ready": True})
    with args.resource_lock.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = argparse.Namespace(max_wait_seconds=0., poll_seconds=.01)
        calls = []
        with campaign.acquire_resources(frozen, runner, lambda *args, **kwargs: calls.append(kwargs)) as result:
            assert result is None
        assert calls[-1]["blockers"]["resource_or_study_lock_unavailable"]


def test_matching_full_batch_endpoint_and_exact_resume_preserve_budget(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch, updates=2)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "resumable"
    assert receipt["charged_updates"] == 2
    assert receipt["verified_actual"]["fresh_samples"] == 2 * campaign.SAMPLES
    prior = receipt["evidence"]
    second = directory.parent / "attempt_0001"
    second_dir, second_request, _, _, _ = endpoint(prepared, monkeypatch, updates=1198, start=2,
        parent=prior["checkpoint"], prior=prior, directory=second)
    assert "--expected-initial-model-sha256" not in second_request["command"]
    assert "--resume" in second_request["command"]
    resumed = campaign.seal_training(frozen, second_request, second_dir, prior)
    assert resumed["status"] == "completed"
    assert resumed["evidence"]["cumulative_transitions"] == campaign.TOTAL
    assert resumed["evidence"]["final_update"] == 1200
    assert resumed["evidence"]["cumulative_optimizer_steps"] == 1200 * 32
    assert campaign.inventory(args.prepared_root) == frozen["inputs"]["files"]


@pytest.mark.parametrize("field", ["initial_sha", "guard", "source", "seed", "factory", "parent", "anchors",
    "config", "sample_count", "optimizer_steps", "first_kl", "final_kl", "short", "counter", "cp_sha",
    "cp_update", "cp_metadata", "cp_config", "cp_lr", "cp_steps", "cp_rng", "population", "sidecar"])
def test_runtime_faults_are_terminal_and_charge_entire_reservation(prepared, monkeypatch, field):
    _, frozen, _, _ = prepared
    directory, request, run, completion, probe = endpoint(prepared, monkeypatch)
    if field == "initial_sha":
        run["initial_model_sha256"] = "f" * 64
    elif field == "guard":
        run.pop("initialization_guard")
    elif field == "source":
        run["source"] = {"files": {}, "sha256": "f" * 64}
    elif field == "seed":
        run["seed"] += 1
    elif field == "factory":
        run["environment_factory"] = "other:factory"
    elif field == "parent":
        run["resume"] = "parent.pt"
    elif field == "anchors":
        probe["metadata"]["anchors"] = [{"path": "unexpected"}]
    elif field == "config":
        run["config"]["ppo"]["learning_rate"] *= 2
    elif field in {"sample_count", "optimizer_steps", "first_kl", "final_kl", "short", "counter"}:
        rows = [json.loads(line) for line in (directory / "train/metrics.jsonl").read_text().splitlines()]
        if field == "short":
            rows[0] = metric(1, count=1024)
        elif field == "counter":
            rows[0]["collection"]["total_transitions"] -= 1
        else:
            key = "first_step_kl" if field == "first_kl" else field
            rows[0]["optimization"][key] = None if "kl" in field else 0
        (directory / "train/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif field == "cp_sha":
        Path(completion["checkpoint"]).write_bytes(b"changed")
    elif field == "cp_update":
        probe["update"] += 1
    elif field == "cp_metadata":
        probe["metadata"]["collected_transitions"] -= 1
    elif field == "cp_config":
        probe["config"]["ppo"]["learning_rate"] *= 2
    elif field == "cp_lr":
        probe["optimizer_groups"][0]["lr"] *= 2
    elif field == "cp_steps":
        probe["optimizer_step_min"] -= 1
    elif field == "cp_rng":
        probe["rng_validated"] = False
    elif field == "population":
        environment = campaign.read(directory / "train/environment.json")
        environment["startup"]["scene_group_counts"] = {"stand": 1023}
        replace_json(directory / "train/environment.json", environment)
    else:
        sidecar = campaign.read(completion["checkpoint"] + ".json")
        sidecar["update"] = 999
        replace_json(completion["checkpoint"] + ".json", sidecar)
    replace_json(directory / "train/run.json", run)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete"
    assert receipt["charged_updates"] == 1200
    assert receipt["charged_samples"] == campaign.TOTAL
    assert receipt["accounting"] == "entire_reservation_charged_not_actual_consumption_no_refund"
    assert receipt["verified_actual"]["fresh_samples"] <= 2 * campaign.SAMPLES


def test_unsealed_crash_and_zero_update_endpoint_never_refund_or_reseed(prepared, monkeypatch):
    _, frozen, _, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch, updates=0)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete" and receipt["charged_updates"] == 1200
    assert receipt["verified_actual"]["fresh_samples"] == 0


def test_live_handle_and_unknown_launch_are_not_terminal(prepared, monkeypatch):
    _, frozen, _, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch)
    process = {"status": "running", "pid": 111, "start": "777"}
    replace_json(directory / "worker.process.json", process)
    monkeypatch.setattr(control, "process_start", lambda pid: "777")
    assert campaign.seal_training(frozen, request, directory)["status"] == "waiting"
    assert not (directory / "receipt.json").exists()
    process.update(status="launching", pid=None, start=None)
    replace_json(directory / "worker.process.json", process)
    monkeypatch.setattr(control, "process_start", lambda pid: None)
    assert campaign.seal_training(frozen, request, directory)["status"] == "waiting"
    assert not (directory / "receipt.json").exists()


def test_reused_pid_allows_audit_without_killing_unrelated_process(prepared, monkeypatch):
    _, frozen, _, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch)
    process = campaign.read(directory / "worker.process.json")
    process.update(status="running", pid=111, start="777")
    replace_json(directory / "worker.process.json", process)
    monkeypatch.setattr(control, "process_start", lambda pid: "888")
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "resumable"


def test_unsealed_metric_suffix_is_not_accepted_as_complete_endpoint(prepared, monkeypatch):
    _, frozen, _, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch)
    metrics = directory / "train/metrics.jsonl"
    metrics.write_text(metrics.read_text().rstrip("\n"))
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete"
    assert receipt["charged_updates"] == 1200
    assert receipt["verified_actual"]["fresh_samples"] == campaign.SAMPLES
    assert not receipt["verified_actual"]["complete"]


def test_actual_cpu_checkpoint_probe_validates_model_adam_and_rng(tmp_path):
    import torch
    from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig
    from transformer_rl.frame_policy import FramePolicyConfig
    from transformer_rl.frame_training import FrameActorCritic
    from transformer_rl.frame_checkpoint import save_frame_checkpoint
    from transformer_rl.ppo import PPOTrainer
    from transformer_rl.config import PPOConfig
    control_contract = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)], "action_names": ["position", "velocity"],
        "action_bounds": [.2, .5], "target_scale": [.25, 10.], "target_offset": [.1, -.2],
        "target_units": ["rad", "rad/s"]}
    config = FrameTrainConfig(FrameModelConfig(policy=FramePolicyConfig(architecture="mlp",
        frame_dim=5, action_dim=2, history_length=1, actor_hidden_dims=(4,)), critic_dim=3, critic_hidden=(4,)),
        PPOConfig(), control_contract, {})
    model = FrameActorCritic(config.model)
    trainer = PPOTrainer(model, config.ppo)
    # A synthetic differentiable scalar populates real Adam moments without an environment.
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    trainer.optimizer.step()
    checkpoint = tmp_path / "checkpoint.pt"
    save_frame_checkpoint(checkpoint, model, trainer, config, 1, {"fixture": True})
    result = campaign.checkpoint_probe({"source_root": str(ROOT)}, checkpoint)
    assert result["model_validated"] and result["optimizer_validated"] and result["rng_validated"]
    assert result["optimizer_step_min"] == result["optimizer_step_max"] == 1
    assert result["checkpoint_sha256"] == control.file_sha(checkpoint)
    assert not result["cuda_initialized"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="learning-state"):
        campaign.checkpoint_probe({"source_root": str(ROOT)}, checkpoint)


def test_real_cpu_preparation_validator_and_controller_preserve_inputs(prepared, template, tmp_path, monkeypatch):
    from transformer_rl.learning_study import prepare_learning_study
    args, _, _, _ = prepared
    parent, packed = tmp_path / "real_parent", tmp_path / "real_prepared"
    shutil.copytree(template, parent)
    base = campaign.read(parent / "base.json")
    base["environment"]["snapshot"] = str(parent / "snapshot")
    replace_json(parent / "base.json", base)
    result = prepare_learning_study(parent, packed)
    before = campaign.inventory(packed)
    monkeypatch.setattr(campaign, "validate_prepared", ACTUAL_PREPARATION_VALIDATOR)
    monkeypatch.setattr(campaign, "runtime_identity", ACTUAL_RUNTIME_IDENTITY)
    args.prepared_root = packed
    args.source_root = ROOT
    args.output_root = tmp_path / "real_output"
    args.prepared_manifest_sha256 = result["sha256"]
    frozen = campaign.prepare(args)
    assert frozen["inputs"]["source"] == campaign.learner_identity(ROOT)
    assert len(frozen["inputs"]["cells"]) == 90
    assert campaign.validate(args.output_root / "manifest.json") == frozen
    assert campaign.audit(args.output_root / "manifest.json")["status"] == "not_ready"
    assert campaign.inventory(packed) == before
    assert all(not campaign.inventory(packed / child["root"] / "jobs") for child in campaign.read(packed / "manifest.json")["children"])


@pytest.mark.parametrize("fault", ["missing", "attempted", "consumed", "cumulative", "final", "too_many"])
def test_completion_and_missing_seal_faults_do_not_refund(prepared, monkeypatch, fault):
    _, frozen, _, _ = prepared
    directory, request, _, completion, _ = endpoint(prepared, monkeypatch)
    path = directory / "train/completion.json"
    if fault == "missing":
        path.unlink()
    else:
        field = {"attempted": "attempted_updates", "consumed": "consumed_transitions", "cumulative": "cumulative_transitions",
                 "final": "final_update", "too_many": "completed_updates"}[fault]
        completion[field] += 1 if fault != "too_many" else 1200
        replace_json(path, completion)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete"
    assert receipt["charged_updates"] == 1200 and receipt["charged_samples"] == campaign.TOTAL
    assert receipt["verified_actual"]["fresh_samples"] == 2 * campaign.SAMPLES


def test_failed_cell_cannot_relaunch_or_reseed_and_retains_all_grid_denominators(prepared, monkeypatch):
    args, frozen, manifest, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch)
    (directory / "train/completion.json").unlink()
    receipt = campaign.seal_training(frozen, request, directory)
    monkeypatch.setattr(campaign, "worker", lambda *args: pytest.fail("an incomplete cell cannot start a replacement"))
    result = campaign.train_cell(frozen, manifest["cells"][0], lambda *args, **kwargs: None)
    assert result["status"] == "incomplete" and result["charged_updates"] == 1200
    assert len(list(directory.parent.glob("attempt_*"))) == 1
    report = campaign.audit(args.output_root / "manifest.json")
    assert report["status"] == "not_ready" and report["actual_cells"] == 90
    assert report["cells"][campaign.cell_key(manifest["cells"][0])]["training"]["status"] == "incomplete"
    assert report["completed_training_cells"] == 0
    # Re-signing the receipt cannot hide missing budget or inflate actual samples.
    receipt["verified_actual"]["fresh_samples"] += 49152
    replace_json(directory / "receipt.json", receipt)
    with pytest.raises(ValueError, match="actual"):
        campaign.audit(args.output_root / "manifest.json")


def test_fresh_request_has_guard_zero_clock_and_no_parent_or_anchors(prepared):
    args, frozen, manifest, _ = prepared
    cell = manifest["cells"][0]
    directory = args.output_root / "cells" / campaign.cell_key(cell) / "training/attempt_0000"
    request = campaign.training_request(frozen, cell, directory, 0, None)
    command = request["command"]
    assert request["parent"] is None and request["start_update"] == request["prior_transitions"] == 0
    assert command[command.index("--expected-initial-model-sha256") + 1] == cell["initial_model_sha256"]
    assert command[command.index("--retention-coef") + 1] == "0.0"
    assert not {"--resume", "--initialize-from", "--restore-learning-from", "--anchors"}.intersection(command)
    assert request["updates"] == 1200 and request["reserved_samples"] == campaign.TOTAL


def test_worker_launch_receipt_precedes_spawn_and_spawn_failure_is_terminal(tmp_path, monkeypatch):
    directory = tmp_path / "attempt"
    directory.mkdir()
    def fail_spawn(*args, **kwargs):
        value = campaign.read(directory / "worker.process.json")
        assert value["status"] == "launching" and value["pid"] is None
        raise OSError("synthetic spawn failure")
    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="spawn"):
        campaign.worker([sys.executable, "-c", "pass"], directory, {"PYTHONPATH": ""}, 1., lambda: None)
    assert campaign.read(directory / "worker.process.json")["status"] == "launch_failed"


def test_owned_cpu_worker_deadline_proves_terminal_original_handle(tmp_path):
    directory = tmp_path / "attempt"
    directory.mkdir()
    observed = []
    def heartbeat():
        value = campaign.read(directory / "worker.process.json")
        assert control.process_start(value["pid"]) == value["start"]
        observed.append((value["pid"], value["start"]))
    result = campaign.worker([sys.executable, "-c", "import time; time.sleep(10)"], directory,
        campaign.cpu_environment(ROOT), .3, heartbeat)
    assert observed and result["status"] == "finished" and result["timed_out"]
    assert result["returncode"] != 0
    assert control.process_start(result["pid"]) != result["start"]
    assert not campaign.workers_in([directory])


def test_owned_cpu_worker_interrupt_does_not_leave_live_handle(tmp_path):
    directory = tmp_path / "attempt"
    directory.mkdir()
    def heartbeat():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        campaign.worker([sys.executable, "-c", "import time; time.sleep(10)"], directory,
            campaign.cpu_environment(ROOT), 30., heartbeat)
    record = campaign.read(directory / "worker.process.json")
    assert record["status"] == "finished" and record["interrupted"]
    assert control.process_start(record["pid"]) != record["start"]


def test_live_dependency_controller_never_passes_completed_label(prepared):
    _, frozen, _, _ = prepared
    dependency = deepcopy(frozen["dependencies"]["diagnostics"])
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        dependency["controller"] = {"pid": process.pid, "start": control.process_start(process.pid)}
        summary = campaign.read(dependency["summary"])
        summary["status"] = "completed"
        replace_json(dependency["summary"], summary)
        state = campaign.dependency_state(dependency)
        assert not state["ready"] and state["controller_live"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_failing_cpu_initialization_preflight_cannot_create_execution_or_start_ppo(prepared, monkeypatch, tmp_path):
    args, _, _, _ = prepared
    args.output_root = tmp_path / "rejected_campaign"
    def reject(*args):
        raise ValueError("CPU initialization SHA differs")
    monkeypatch.setattr(campaign, "validate_prepared", reject)
    monkeypatch.setattr(campaign, "worker", lambda *args: pytest.fail("no PPO worker may be constructed"))
    with pytest.raises(ValueError, match="initialization SHA"):
        campaign.prepare(args)
    assert not args.output_root.exists()


def test_dependencies_require_all_three_definitions_and_reject_rewritten_source(prepared):
    args, frozen, _, _ = prepared
    declared = campaign.read(args.dependencies)
    declared.pop("curriculum")
    replace_json(args.dependencies, declared)
    with pytest.raises(ValueError):
        campaign.validate(args.output_root / "manifest.json")
    path = args.source_root / "src/transformer_rl/extra.py"
    path.write_text("# undeclared predecessor source\n")
    with pytest.raises(ValueError, match="inventory"):
        campaign.dependency_state(frozen["dependencies"]["transfer"])


def test_request_tampering_prevents_any_resume_or_new_worker(prepared, monkeypatch):
    _, frozen, manifest, _ = prepared
    directory, request, _, _, _ = endpoint(prepared, monkeypatch)
    request["initial_model_sha256"] = "c" * 64
    replace_json(directory / "request.json", request)
    monkeypatch.setattr(campaign, "worker", lambda *args: pytest.fail("a rewritten request must not start a worker"))
    with pytest.raises(ValueError, match="request"):
        campaign.train_cell(frozen, manifest["cells"][0], lambda *args, **kwargs: None)


def test_short_rollout_audit_does_not_confuse_charged_samples_with_actual_endpoints(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    directory, request, _, completion, _ = endpoint(prepared, monkeypatch, updates=1)
    row = metric(1, count=1024)
    (directory / "train/metrics.jsonl").write_text(json.dumps(row) + "\n")
    completion.update(consumed_transitions=1024, cumulative_transitions=1024)
    replace_json(directory / "train/completion.json", completion)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete"
    assert receipt["charged_samples"] == 58982400
    assert receipt["verified_actual"]["fresh_samples"] == 1024
    assert receipt["verified_actual"]["optimization_samples"] == 1024
    assert campaign.audit(args.output_root / "manifest.json")["status"] == "not_ready"


@pytest.mark.parametrize("field", ["planned_steps", "early_stop", "vector_steps", "step_type", "count_type", "batch_type"])
def test_rollout_and_ppo_record_types_and_exact_recipe_are_verified(tmp_path, field):
    row = metric(1)
    if field == "planned_steps":
        row["optimization"]["planned_optimizer_steps"] = 159
    elif field == "early_stop":
        row["optimization"]["early_stopped"] = False
    elif field == "vector_steps":
        row["collection"]["vector_steps"] = 47
    elif field == "step_type":
        row["optimization"]["optimizer_steps"] = 32.
    elif field == "count_type":
        row["collection"]["transitions"] = 49152.
    else:
        row["batch_samples"] = 49152.
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        campaign.optimization_records(path, 0)


@pytest.mark.parametrize("mutation", ["clock", "pid", "start", "returncode", "runtime"])
def test_actual_worker_command_and_runtime_are_required(prepared, monkeypatch, mutation):
    _, frozen, _, _ = prepared
    directory, request, _, _, probe = endpoint(prepared, monkeypatch)
    path = directory / "worker.process.json"
    process = campaign.read(path)
    if mutation == "clock":
        offset = process["command"].index("--consumed-update-offset") + 1
        process["command"][offset] = "99"
    elif mutation == "pid":
        process["pid"] = None
    elif mutation == "start":
        process["start"] = "unverified"
    elif mutation == "returncode":
        process["returncode"] = None
    else:
        probe["metadata"]["runtime"]["torch"] = "changed-runtime"
    replace_json(path, process)
    receipt = campaign.seal_training(frozen, request, directory)
    assert receipt["status"] == "incomplete"
    assert receipt["charged_samples"] == campaign.TOTAL


def test_runtime_and_empty_cache_are_revalidated(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    changed = deepcopy(frozen["runtime"])
    changed["executable"]["inode"] += 1
    monkeypatch.setattr(campaign, "runtime_identity", lambda *args: changed)
    with pytest.raises(ValueError, match="runtime"):
        campaign.validate(args.output_root / "manifest.json")
    monkeypatch.setattr(campaign, "runtime_identity", lambda *args: deepcopy(frozen["runtime"]))
    (args.output_root / ".bytecode-cache/stale.pyc").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="cache"):
        campaign.validate(args.output_root / "manifest.json")


def test_exclusive_cache_prefix_avoids_reading_existing_stale_bytecode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    module = source / "cached_fixture.py"
    module.write_text("value = 'old'\n")
    stamp = module.stat().st_mtime_ns
    py_compile.compile(str(module), doraise=True)
    module.write_text("value = 'new'\n")
    os.utime(module, ns=(stamp, stamp))
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    script = "import cached_fixture;print(cached_fixture.value)"
    old = subprocess.run([sys.executable, "-B", "-c", script], env=env, capture_output=True, text=True, check=True)
    assert old.stdout.strip() == "old"
    cache = tmp_path / "exclusive-cache"
    cache.mkdir()
    fresh = subprocess.run(campaign.python_command(sys.executable, cache, "-c", script), env=env,
        capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == "new"
    assert not campaign.inventory(cache)


@pytest.mark.parametrize("sealed", [True, False])
def test_failed_resume_retains_prior_actual_and_full_reservation(prepared, monkeypatch, sealed):
    args, frozen, manifest, _ = prepared
    first, first_request, _, _, first_probe = endpoint(prepared, monkeypatch)
    first_receipt = campaign.seal_training(frozen, first_request, first)
    evidence = first_receipt["evidence"]
    second = first.parent / "attempt_0001"
    directory, request, _, completion, _ = endpoint(prepared, monkeypatch, updates=1, start=2,
        parent=evidence["checkpoint"], prior=evidence, directory=second)
    monkeypatch.setattr(campaign, "checkpoint_probe", lambda *args: deepcopy(first_probe))
    (directory / "train/metrics.jsonl").write_text(json.dumps(metric(3, count=1024)) + "\n")
    completion.update(consumed_transitions=1024, cumulative_transitions=99328)
    replace_json(directory / "train/completion.json", completion)
    if sealed:
        failed = campaign.seal_training(frozen, request, directory, evidence)
        assert failed["status"] == "incomplete"
    else:
        (directory / "train/completion.json").unlink()
    report = campaign.audit(args.output_root / "manifest.json")
    assert report["status"] == "not_ready" and report["expected_cells"] == 90
    training = report["cells"][campaign.cell_key(manifest["cells"][0])]["training"]
    assert training["verified_fresh_samples_before_failed_attempt"] == 98304
    assert training["verified_actual"]["fresh_samples"] == 1024
    assert training["verified_known_prefix_fresh_samples"] == 99328
    assert training["charged_updates"] == 1200 and training["charged_samples"] == 58982400
    if sealed:
        extra = directory.parent / "attempt_0002"
        extra.mkdir()
        with pytest.raises(ValueError, match="extra attempt"):
            campaign.audit(args.output_root / "manifest.json")


def test_run_evaluates_all_complete_cells_even_when_control_is_poor(prepared, monkeypatch):
    args, frozen, _, _ = prepared
    @contextmanager
    def ready(*args):
        yield {"status": "completed"}
    monkeypatch.setattr(campaign, "acquire_resources", ready)
    monkeypatch.setattr(campaign, "train_cell", lambda *args: {"status": "completed", "success_rate": 0., "tracking_error": 99.})
    calls = []
    def evaluation(manifest, cell, training, seed, publish):
        calls.append((campaign.cell_key(cell), seed))
        return {"status": "completed", "success_rate": 0.}
    monkeypatch.setattr(campaign, "evaluate_cell", evaluation)
    monkeypatch.setattr(campaign, "audit", lambda *args: {"status": "development_complete"})
    summary = campaign.run(argparse.Namespace(manifest=args.output_root / "manifest.json", max_wait_seconds=0., poll_seconds=.01))
    assert summary["status"] == "completed"
    assert len(calls) == 360
    assert set(calls) == {(campaign.cell_key(cell), seed) for cell in frozen["inputs"]["cells"] for seed in campaign.DEVELOPMENT_SEEDS}


@pytest.mark.parametrize("failed", [False, True])
def test_development_fixed_suite_seal_and_no_favorable_retry(prepared, monkeypatch, failed):
    args, frozen, manifest, _ = prepared
    cp = args.output_root / "fixture-final.pt"
    cp.write_bytes(b"CPU only checkpoint fixture")
    training = {"status": "completed", "checkpoint": campaign.artifact(cp)}
    cell, commands, verifications = manifest["cells"][0], [], []
    def synthetic_worker(command, directory, *args):
        commands.append(command)
        campaign.write(directory / "worker.process.json", {"command": command, "status": "finished", "returncode": 0,
            "timed_out": False, "pid": 987654321, "start": "123456"})
    def verify(directory, shim, variant, seed):
        verifications.append(seed)
        assert len(shim["inputs"]["cases"]) == 50
        if failed:
            raise ValueError("missing fixed control output")
        return {}
    monkeypatch.setattr(campaign, "worker", synthetic_worker)
    monkeypatch.setattr(campaign.diagnostic, "verify_outputs", verify)
    for seed in campaign.DEVELOPMENT_SEEDS:
        result = campaign.evaluate_cell(frozen, cell, training, seed, lambda *args, **kwargs: None)
        assert result["status"] == ("incomplete" if failed else "completed")
        command = commands[-1]
        assert command[command.index("--steps") + 1] == "4001"
        assert command[command.index("--settle-steps") + 1] == "200"
        assert command.index("--outputs") - command.index("--configs") - 1 == 50
        assert campaign.evaluate_cell(frozen, cell, training, seed, lambda *args, **kwargs: None)["status"] == result["status"]
    assert len(commands) == 4
    assert set(verifications) == set(campaign.DEVELOPMENT_SEEDS)

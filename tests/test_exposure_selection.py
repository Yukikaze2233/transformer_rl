"""Selection semantics and actual CPU campaign-to-choice reconstruction.

Logical grade fixtures exercise ranking arithmetic only. The OS integration
executes actual canonical CLI workers, PPO/Adam, complete checkpoints, physical
NPZ traces and replay. Only predecessor closure, Isaac factory resolution and
the test startup search path are substituted by a runtime-pinned CPU fixture.
No production queue, PhysX row identity, latency or hardware proof is claimed.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys

import pytest
import torch

from transformer_rl import exposure_campaign as campaign
from transformer_rl import exposure_evaluation as evaluator
from transformer_rl import exposure_protocol as protocol
from transformer_rl import exposure_selection as selection
from transformer_rl import exposure_training as training
from transformer_rl import history_study, queue_validation
from transformer_rl.frame_config import digest, json_bytes
from transformer_rl.frame_study import grade_report
from transformer_rl.frame_workflow import evaluate_frame_policy
from test_exposure_evaluation import (REAL_RUNTIME, SixMotorFixture, configuration,
                                     make_six_motor_env, synthetic_closed)
from test_exposure_protocol import prepared, freeze, write_json


def canonical(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def logical_grid(stages=2):
    """Explicit arithmetic fixture; these records are not physical evidence."""
    value = {"jobs": [], "evaluation_cells": [], "selection": {
        "min_training_seeds": 3, "std_penalty": 1., "retention_score_tolerance": .5}}
    training, cells = {}, {}
    for name, architecture in (("mlp_h1", "mlp"), ("gated_h4", "transformer"), ("query_h4", "transformer")):
        for seed in (71, 97, 101):
            job_id = f"{name}_seed_{seed}"
            stage_list = [{"index": index, "config": {"model": {"policy": {"architecture": architecture}}}}
                          for index in range(stages)]
            value["jobs"].append({"id": job_id, "candidate": name, "training_seed": seed, "stages": stage_list})
            training[job_id] = {"status": "training_completed", "stages": [
                {"stage_index": index, "endpoint": {"path": f"{job_id}/{index}/endpoint.json"},
                 "checkpoint": {"path": f"{job_id}/{index}/endpoint.pt"}} for index in range(stages)]}
            for index in range(stages):
                for evaluation_seed in (701, 1701):
                    for scenario in ("normal", "new_skill"):
                        identity = {"id": f"{job_id}_{index}_{evaluation_seed}_{scenario}", "job_id": job_id,
                                    "stage_index": index, "role": "validation", "seed": evaluation_seed,
                                    "scenario": scenario}
                        value["evaluation_cells"].append(identity)
                        # grade_report is real; input report is a logical fixture.
                        grade = grade_report({"completed_episodes": 2, "success_rate": 1., "error": 1.},
                            {"gates": [{"path": "success_rate", "operator": "min", "value": .8}]},
                            {"min_completed_episodes": 1},
                            [{"path": "error", "weight": 1., "scale": 1., "direction": "minimize"}])
                        cells[identity["id"]] = {"identity": deepcopy(identity), "status": "completed", "grade": grade}
    return value, training, cells


def update_grades(cells, candidate, *, stage=None, seed=None, scenario=None, score=1., passed=True):
    for record in cells.values():
        identity = record["identity"]
        if not identity["job_id"].startswith(candidate) or stage is not None and identity["stage_index"] != stage \
                or seed is not None and not identity["job_id"].endswith(f"_{seed}") \
                or scenario is not None and identity["scenario"] != scenario:
            continue
        record["grade"] = {"passed": passed, "reasons": [] if passed else ["success_rate"], "score": score}


def test_final_stage_only_ranking_and_acquisition_conditioned_retention():
    value, training, cells = logical_grid()
    update_grades(cells, "gated", stage=0, scenario="new_skill", score=30., passed=False)
    update_grades(cells, "gated", stage=1, score=.75)
    update_grades(cells, "query", stage=0, score=.8)
    update_grades(cells, "query", stage=1, score=.9)
    rank = selection._rank(value, training, cells)
    assert rank["best_transformer"]["candidate"] == "gated_h4"
    assert rank["best_transformer"]["score_mean"] == .75
    assert all(seed["eligible"] for seed in rank["best_transformer"]["training_seeds"])
    # The same later failure is disqualifying once this skill was acquired.
    update_grades(cells, "gated", stage=0, scenario="new_skill", score=.1)
    rank = selection._rank(value, training, cells)
    gated = next(candidate for candidate in rank["candidates"] if candidate["candidate"] == "gated_h4")
    assert not gated["eligible"] and all(seed["retention_regressions"] for seed in gated["training_seeds"])
    assert rank["best_transformer"]["candidate"] == "query_h4"


def test_all_original_training_seeds_and_missing_cells_remain_in_denominator():
    value, training, cells = logical_grid()
    training["gated_h4_seed_71"]["status"] = "numerical_failure"
    record = next(record for record in cells.values() if record["identity"]["job_id"] == "query_h4_seed_101"
                  and record["identity"]["stage_index"] == 1)
    record.update(status="missing", grade=None)
    rank = selection._rank(value, training, cells)
    assert rank["best_transformer"] is None and rank["best_overall"]["architecture"] == "mlp"
    query = next(candidate for candidate in rank["candidates"] if candidate["candidate"] == "query_h4")
    assert query["score_mean"] is query["score_sample_std"] is query["rank_score"] is None
    assert len(query["training_seeds"]) == 3
    failed = next(seed for seed in query["training_seeds"] if seed["training_seed"] == 101)
    assert failed["expected_validation_cells"] == 8 and failed["completed_validation_cells"] == 7
    assert len(rank["mlp_controls"]) == 1


def test_sample_standard_deviation_penalty_and_exact_final_median_seed_checkpoint():
    value, training, cells = logical_grid()
    value["selection"]["retention_score_tolerance"] = 5.
    for seed, score in ((71, 1.), (97, 2.), (101, 4.)):
        update_grades(cells, "gated", stage=1, seed=seed, score=score)
    rank = selection._rank(value, training, cells)
    gated = next(candidate for candidate in rank["candidates"] if candidate["candidate"] == "gated_h4")
    assert gated["score_mean"] == statistics.mean((1., 2., 4.))
    assert gated["score_sample_std"] == statistics.stdev((1., 2., 4.))
    assert gated["rank_score"] == gated["score_mean"] + gated["score_sample_std"]
    assert gated["representative"] == {"job_id": "gated_h4_seed_97", "training_seed": 97, "stage_index": 1,
        "endpoint": {"path": "gated_h4_seed_97/1/endpoint.json"},
        "checkpoint": {"path": "gated_h4_seed_97/1/endpoint.pt"}}
    assert rank["best_transformer"]["candidate"] == "query_h4"


def test_all_legal_but_bad_grades_yield_no_eligible_candidate():
    value, training, cells = logical_grid()
    for name in ("mlp", "gated", "query"):
        update_grades(cells, name, stage=1, score=100., passed=False)
    rank = selection._rank(value, training, cells)
    assert rank["best_overall"] is rank["best_transformer"] is None
    assert all(not candidate["eligible"] and candidate["score_mean"] == 100. for candidate in rank["candidates"])


def test_full_history_minimum_is_a_quality_gate_not_a_replacement_or_score_edit(tmp_path):
    config = configuration("transformer", 4)
    path = tmp_path / "config.json"
    canonical(path, config.to_dict())
    cell = {"config_receipt": protocol._receipt(path), "scenario": "stand", "num_envs": 2}
    value = {"evaluation": {"min_completed_episodes": 1, "min_steady_samples": 2, "settle_steps": 1},
             "scenarios": [{"name": "stand", "gates": [], "require_steady": True}],
             "selection": {"objectives": [{"path": "metrics.error.mean", "direction": "minimize", "scale": 1., "weight": 1.}]}}
    metrics = {"completed_episodes": 2, "stability": {"available": True}, "metrics": {"error": {"mean": .5}},
        "history_control": {"history_length": 4, "minimum_full_age": 3, "policy_dt_s": .01,
            "settle_steps": 1, "min_steady_samples": 2,
            "windows": {"full_history": {"samples": 3, "steady_tracking": {"samples": 2}}}}}
    measured = {"metrics": metrics, "grade": grade_report(metrics, value["scenarios"][0], value["evaluation"],
                                                          value["selection"]["objectives"])}
    grade, gate = selection._physical_grade(value, cell, measured)
    assert not grade["passed"] and grade["score"] == .5 and gate["minimum_samples"] == 4
    assert set(grade["reasons"]) == {"insufficient_full_history_samples", "insufficient_full_history_steady_samples"}
    metrics["history_control"]["windows"]["full_history"].update(samples=4, steady_tracking={"samples": 4})
    assert selection._physical_grade(value, cell, measured)[0]["passed"]
    for name, replacement in (("history_length", True), ("minimum_full_age", 2), ("policy_dt_s", .02)):
        original = metrics["history_control"][name]
        metrics["history_control"][name] = replacement
        with pytest.raises(ValueError):
            selection._physical_grade(value, cell, measured)
        metrics["history_control"][name] = original


@pytest.fixture
def physical_grid(prepared, monkeypatch):
    """Minimum complete legal protocol, with every ordinary dependency actual.

    Runtime-pinned sitecustomize substitutes ONLY the original queue closure
    providers and Isaac factory resolver. A test _worker_environment wrapper
    inserts this SDK startup search path; it does not weaken the production
    whitelist. Actual production launcher commands remain canonical -m, and
    parent/child pidfds, two inherited flocks, protocol/source/runtime hashes,
    full learning chains, NPZ verification and selection are never patched.
    """
    p = prepared
    base = configuration("transformer", 4, residual="gated")
    contracts = {}
    for name in ("first", "second", "normal", "new_skill"):
        replicas = 4 if name in ("first", "second") else 2
        path = p["snapshot"] / "contracts" / f"{name}.json"
        write_json(path, {"name": "common_synthetic_physics", "target_num_envs": replicas,
            "physics_dt": .005, "policy_dt": .01, "evaluation_exact_cases": True, "episode_seconds": .06,
            "evaluation": {"cases": [{"name": name, "terrain": "flat", "task": "survive"}],
                           "episodes_per_case": 2, "stable_case_layout": False}})
        contracts[name] = {"contract": str(path.relative_to(p["snapshot"])),
                          "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "num_envs": replicas}
    files = {str(path.relative_to(p["snapshot"])): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(p["snapshot"].rglob("*")) if path.is_file() and path.name != "snapshot.json"}
    snapshot_sha = digest(files)
    write_json(p["snapshot"] / "snapshot.json", {"files": files, "sha256": snapshot_sha, "control": base.control})
    base = replace(base, environment={"snapshot": str(p["snapshot"]), "snapshot_sha256": snapshot_sha,
                                     **contracts["first"], "control": base.control})
    spec = deepcopy(p["spec"])
    spec["environment_factory"] = "transformer_rl.chassis_adapter:make_env"
    spec["variants"] = [
        {"name": "mlp", "policy": {"architecture": "mlp", "history_length": 1, "residual_type": "add"}},
        {"name": "history_mlp", "policy": {"architecture": "history_mlp", "residual_type": "add"}},
        {"name": "gated", "policy": {"architecture": "transformer", "residual_type": "gated"}}]
    spec["stages"] = [{**spec["stages"][0], "environment": contracts["first"], "updates": 1,
                       "scenarios": [scenario["name"] for scenario in spec["scenarios"]]}]
    for scenario in spec["scenarios"]:
        scenario["environment"] = contracts[scenario["name"]]
        # Permissive logical qualification allows this toy pipeline to select;
        # this threshold must never be mistaken for a robot acceptance limit.
        scenario["gates"][0]["value"] = 0.
    spec["training"]["rollout_steps"] = 2
    spec["evaluation"]["steps"] = 13
    spec["selection"]["objectives"][0]["path"] = "metrics.vx_abs_error.mean"
    write_json(p["inputs"] / "base.json", base.to_dict())
    write_json(p["inputs"] / "study.json", spec)
    fixture_source = ("from copy import deepcopy\nimport json\nfrom pathlib import Path\nimport torch\n"
        "from transformer_rl.frame_config import digest\nfrom transformer_rl.types import StepResult, VectorObservation\n\n"
        + inspect.getsource(SixMotorFixture) + "\n" + inspect.getsource(make_six_motor_env) + '''
original_make_env = make_six_motor_env
def make_six_motor_env(model_config, environment_config, device):
    env = original_make_env(model_config, environment_config, device)
    if model_config.policy.architecture == 'mlp' and torch.initial_seed() == 71:
        def failure(action):
            raise FloatingPointError('actual CPU fixture nonfinite training step')
        env.step = failure
    return env
''')
    (p["sdk"] / "physical_cpu_fixture.py").write_text(fixture_source)
    startup = '''
# ONLY synthetic original closure providers and Isaac factory resolution.
import json, os
from copy import deepcopy
from pathlib import Path
from transformer_rl import exposure_protocol as protocol, queue_validation, cli
p = json.loads(Path(os.environ['EXPOSURE_CPU_FIXTURE_PROTOCOL']).read_bytes())
protocol.predecessors._dependencies = lambda *a: (deepcopy(p['execution']['dependencies']), deepcopy(p['execution']['resource_locks']))
queue_validation.check_dependency = lambda *a, **k: {'status': 'completed', 'controller_live': False, 'live_workers': [], 'fixture': 'synthetic closure only'}
import physical_cpu_fixture
def cpu_factory(reference):
    if reference != 'transformer_rl.chassis_adapter:make_env':
        raise ValueError('fixture factory reference differs')
    return physical_cpu_fixture.make_six_motor_env
cli._factory = cpu_factory
'''
    (p["sdk"] / "sitecustomize.py").write_text("import os\ntry:\n" +
        "".join("    " + line + "\n" for line in startup.splitlines()) + "except BaseException:\n    os._exit(81)\n")
    p["history"] = p["tmp"] / "selection_history"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json", p["history"],
                                       history_lengths=[1, 4], position_reference="current")
    monkeypatch.setattr(protocol, "runtime_identity", REAL_RUNTIME)
    monkeypatch.setattr(queue_validation, "check_dependency", synthetic_closed)
    value = freeze(p)
    p["protocol_path"] = p["tmp"] / "authorized_selection_protocol.json"
    canonical(p["protocol_path"], value)
    p["protocol"], p["base"], p["spec"] = value, base, spec
    measurements = p["tmp"] / "actual_storage_measurements"
    measurements.mkdir()
    calibration = training.train_exposure_segment({"name": "storage_calibration", "config": base, "updates": 1},
        make_six_motor_env, value["environment_factory"], measurements / "actual_cpu_training",
        job_id="selection_storage_calibration", rollout_steps=2, training_seed=19, retention_seed=91002,
        evaluation_seeds=(777, 1777), device="cpu", max_seconds=60.,
        expected_initial_model_sha256=training.initial_model_sha256(base, 19))
    assert calibration["status"] == "completed"
    checkpoint = Path(calibration["endpoints"][0]["checkpoint"]["path"])
    trace_path = measurements / "actual_cpu_trace.npz"
    calibration_report = evaluate_frame_policy(checkpoint, make_six_motor_env, base.environment,
        steps=4, seed=777, settle_steps=1, min_steady_samples=1, control_metrics=True,
        history_control=True, trace_output=trace_path, trace_replicas=4)
    assert calibration_report["transitions"] == 16
    metrics = measurements / "actual_cpu_training" / "metrics.jsonl"
    cache_root = measurements / "actual_runtime_cache"
    cache_root.mkdir()
    (cache_root / "fixture_source.bin").write_bytes((p["sdk"] / "physical_cpu_fixture.py").read_bytes())
    cache = measurements / "actual_runtime_cache_inventory.json"
    cache_files = [protocol._receipt(cache_root / "fixture_source.bin")]
    canonical(cache, {"format": "transformer_rl.exposure_runtime_cache_measurement", "schema_version": 1,
        "root": str(cache_root), "files": cache_files, "total_bytes": sum(item["bytes"] for item in cache_files)})
    p["storage_path"] = p["tmp"] / "actual_storage_contract.json"
    canonical(p["storage_path"], {"format": "transformer_rl.exposure_storage_contract", "schema_version": 1,
        "protocol_raw_sha256": protocol._receipt(p["protocol_path"])["sha256"], "source": value["source"],
        "caps": {"checkpoint_bytes": 16 * 1024**2, "trace_bytes_per_policy_sample": 4096,
                 "metric_bytes_per_update": 64 * 1024, "inflight_bytes": 8 * 1024**2,
                 "runtime_cache_bytes": 4 * 1024**2, "free_margin_bytes": 4 * 1024**2},
        "measurements": [{"kind": kind, "receipt": protocol._receipt(path), "units": units}
            for kind, path, units in (("checkpoint", checkpoint, 1), ("trace", trace_path, 16),
                                      ("metric", metrics, 1), ("runtime_cache", cache, 1))]})
    original_environment = campaign._worker_environment

    def fixture_environment(directory):
        environment = original_environment(directory)
        environment["PYTHONPATH"] = os.pathsep.join((environment["PYTHONPATH"], str(p["sdk"])))
        environment["EXPOSURE_CPU_FIXTURE_PROTOCOL"] = str(p["protocol_path"])
        return environment

    monkeypatch.setattr(campaign, "_worker_environment", fixture_environment)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield p
    torch.set_num_threads(previous)


def test_actual_full_grid_selection_and_heldout_authorization_without_summary_trust(physical_grid, monkeypatch):
    p = physical_grid
    value = p["protocol"]
    raw = protocol._receipt(p["protocol_path"])
    observed = []
    original_freeze = selection.freeze_selection

    def observe_real_seal(*args, **kwargs):
        # This observer calls the actual selector unmodified. It only observes
        # phase order and the actual still-live parent/two-FD boundary.
        assert not any(selection._directory(value, cell).exists() for cell in value["evaluation_cells"]
                       if cell["role"] == "held_out")
        first_request = next(Path(value["output_root"]).glob("*/stage_0000/evaluation_validation_701/request.json"))
        leases = json.loads(first_request.read_bytes())["leases"]
        assert len(leases) == 2
        for lease in leases:
            assert lease["controller_pid"] == os.getpid()
            identity = os.fstat(lease["descriptor"])
            assert (identity.st_dev, identity.st_ino) == (lease["device"], lease["inode"])
            assert "FLOCK" in Path(f"/proc/self/fdinfo/{lease['descriptor']}").read_text()
        forbidden = selection._directory(value, next(cell for cell in value["evaluation_cells"]
                                                     if cell["role"] == "held_out"))
        forbidden.mkdir()
        try:
            with pytest.raises(ValueError, match="held-out output existed"):
                original_freeze(*args, **kwargs)
        finally:
            forbidden.rmdir()
        assert not (Path(value["output_root"]) / "selection").exists()
        actual = original_freeze(*args, **kwargs)
        observed.append(deepcopy(actual))
        return actual

    with monkeypatch.context() as phase_observer:
        phase_observer.setattr(selection, "freeze_selection", observe_real_seal)
        summary = campaign.run_protocol(p["protocol_path"], expected_protocol_sha256=raw["sha256"],
            storage_contract_path=p["storage_path"], expected_storage_sha256=protocol._receipt(p["storage_path"])["sha256"],
            evaluation_provider="transformer_rl.exposure_evaluation")
    assert summary["charged_updates"] == 15 and summary["charged_fresh_transitions"] == 120
    assert summary["verified_successful_updates"] == 14 and len(summary["jobs"]) == 15
    assert summary["status"] == "comparison_closed_deployment_qualification_pending"
    assert summary["full_evaluation_matrix_closed"] and summary["validation_matrix_closed"]
    assert len(observed) == 1 and summary["validation_choice"] == observed[0]["receipt"]
    for role in ("validation", "held_out"):
        records = [record for record in summary["evaluation_cells"].values() if record["identity"]["role"] == role]
        assert len(records) == 60 and sum(record["status"] == "completed" for record in records) == 56
        assert sum(record["status"] == "missing" for record in records) == 4
    root = Path(value["output_root"])
    with pytest.raises(ValueError, match="external authorization"):
        selection.freeze_selection(p["protocol_path"], expected_protocol_sha256="0" * 64)
    heldout = [cell for cell in value["evaluation_cells"] if cell["role"] == "held_out" and
               cell["job_id"] == value["jobs"][-1]["id"] and cell["seed"] == 2701]
    heldout_directory = selection._directory(value, heldout[0])
    sealed = observed[0]
    choice, receipt = sealed["choice"], sealed["receipt"]
    assert selection.verify_selection(receipt) == choice
    for cell in value["evaluation_cells"]:
        if cell["role"] != "held_out" or choice["held_out_cells"][cell["id"]]["endpoint"] is None:
            continue
        request = json.loads((selection._directory(value, cell) / "request.json").read_bytes())
        assert request["selection"] == receipt
        assert request["endpoint"] == choice["held_out_cells"][cell["id"]]["endpoint"]
    assert choice["status"] == "selected_provisional"
    assert choice["denominator"] == {"jobs": 15, "validation_cells": 60, "completed_validation_cells": 56, "held_out_cells": 60}
    assert choice["deployment"]["status"] == "unverified" and not choice["deployment"]["latency_gate_applied"]
    assert not choice["hardware_verified"] and not choice["formal_architecture_selection"]
    assert len(choice["mlp_controls"]) == 3 and len(choice["candidates"]) == 5
    mlp = next(item for item in choice["candidates"] if item["architecture"] == "mlp")
    assert not mlp["eligible"] and len(mlp["training_seeds"]) == 3 and mlp["score_mean"] is None
    assert sum(record["endpoint"] is None for record in choice["held_out_cells"].values()) == 4
    for candidate in choice["candidates"]:
        scores = [item["score"] for item in candidate["training_seeds"]]
        if all(score is not None for score in scores):
            assert candidate["score_mean"] == statistics.mean(scores)
            assert candidate["score_sample_std"] == statistics.stdev(scores)
    winner = choice["best_transformer"]
    assert winner["architecture"] == "transformer" and winner["eligible"]
    assert winner["representative"]["stage_index"] == 0
    assert selection.verify_selection(receipt) == choice
    summary_path = root / "summary.json"
    summary_bytes = summary_path.read_bytes()
    fabricated_summary = json.loads(summary_bytes)
    fabricated_summary.update(best_transformer="fabricated_csv_winner", reward_mean=1e12)
    canonical(summary_path, fabricated_summary)
    assert selection.verify_selection(receipt) == choice
    summary_path.write_bytes(summary_bytes)
    with pytest.raises(ValueError, match="overwritten or retried"):
        selection.freeze_selection(p["protocol_path"], expected_protocol_sha256=raw["sha256"])
    fixed = choice["held_out_cells"][heldout[0]["id"]]["endpoint"]
    authorization = selection.authorize_heldout_batch(receipt, fixed, heldout, heldout_directory)
    assert authorization["selection_may_change"] is False and authorization["endpoint"] == fixed
    for changed_cells, directory in ((heldout[:1], heldout_directory), (list(reversed(heldout)), heldout_directory),
                                     (heldout, heldout_directory.parent / "unfixed")):
        with pytest.raises(ValueError):
            selection.authorize_heldout_batch(receipt, fixed, changed_cells, directory)
    other = next(item["endpoint"] for item in choice["held_out_cells"].values()
                 if item["endpoint"] is not None and item["endpoint"] != fixed)
    with pytest.raises(ValueError, match="sealed checkpoint"):
        selection.authorize_heldout_batch(receipt, other, heldout, heldout_directory)
    missing_cell = next(item["identity"] for item in choice["held_out_cells"].values() if item["endpoint"] is None)
    missing_batch = [cell for cell in value["evaluation_cells"] if cell["role"] == "held_out" and
                     cell["job_id"] == missing_cell["job_id"] and cell["seed"] == missing_cell["seed"]]
    with pytest.raises(ValueError, match="lacks a trained endpoint"):
        selection.authorize_heldout_batch(receipt, fixed, missing_batch, selection._directory(value, missing_cell))
    # Selection never consumes heldout metrics. Publication after a seal cannot
    # alter the exact candidate or any checkpoint fixed in its original matrix.
    if not heldout_directory.exists():
        heldout_directory.mkdir()
        canonical(heldout_directory / "arbitrary_heldout_score.json", {"score": -1e12})
    assert selection.verify_selection(receipt) == choice
    path = Path(receipt["path"])
    original = path.read_bytes()
    for mutate in (lambda item: item.update(status="no_eligible"),
                   lambda item: item["best_transformer"]["representative"].update(endpoint=other),
                   lambda item: item["deployment"].update(latency_gate_applied=True),
                   lambda item: item["denominator"].update(validation_cells=56),
                   lambda item: item["held_out_cells"].pop(heldout[0]["id"])):
        changed = deepcopy(choice)
        mutate(changed)
        changed.pop("sha256")
        changed["sha256"] = digest(changed)
        canonical(path, changed)
        with pytest.raises(ValueError, match="actual full evidence"):
            selection.verify_selection(protocol._receipt(path))
        path.write_bytes(original)
    # Fast heldout authorization checks frozen actual inode/ctime signatures,
    # without rehashing/replaying every large NPZ. Changing bytes (even later
    # restoring the same bytes) permanently invalidates this original seal.
    trace_path = Path(next(batch for batch in choice["validation_batches"].values()
                           if batch["status"] == "completed")["artifacts"]["trace.npz"]["path"])
    trace_bytes = trace_path.read_bytes()
    trace_path.write_bytes(trace_bytes + b"changed frozen trace bytes")
    try:
        with pytest.raises(ValueError, match="signature changed"):
            selection.authorize_heldout_batch(receipt, fixed, heldout, heldout_directory)
    finally:
        trace_path.write_bytes(trace_bytes)
    with pytest.raises(ValueError, match="signature changed"):
        selection.authorize_heldout_batch(receipt, fixed, heldout, heldout_directory)
    # A fabricated CSV/ledger score cannot override raw report/NPZ evidence.
    batch = next(batch for batch in choice["validation_batches"].values() if batch["status"] == "completed")
    report_path = Path(batch["artifacts"]["report.json"]["path"])
    completion_path = report_path.parent / "evaluation.completion.json"
    report_bytes, completion_bytes = report_path.read_bytes(), completion_path.read_bytes()
    report = json.loads(report_bytes)
    report["groups"]["normal"]["metrics"]["vx_abs_error"]["mean"] = -1e12
    canonical(report_path, report)
    completion = json.loads(completion_bytes)
    completion["report"] = protocol._receipt(report_path)
    canonical(completion_path, completion)
    try:
        with pytest.raises(ValueError, match="full trace replay"):
            selection.verify_selection(receipt)
    finally:
        report_path.write_bytes(report_bytes)
        completion_path.write_bytes(completion_bytes)
    checkpoint = Path(winner["representative"]["checkpoint"]["path"])
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"modified actual learning state")
    try:
        with pytest.raises(ValueError):
            selection.verify_selection(receipt)
    finally:
        checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(ValueError, match="actual full evidence"):
        selection.verify_selection(receipt)
    with pytest.raises(ValueError, match="overwritten or retried"):
        selection.freeze_selection(p["protocol_path"], expected_protocol_sha256=raw["sha256"])

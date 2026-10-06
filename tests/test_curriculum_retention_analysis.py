"""Report-based retention, qualification, missingness and sealed path identities."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


TOOL = Path(__file__).parents[1] / "tools/analyze_curriculum_retention.py"
SPEC = importlib.util.spec_from_file_location("curriculum_retention_analysis", TOOL)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path, root):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def frozen(tmp_path):
    """The actual 50-case/three-arm layout, containing no runnable learner."""
    experiment = tmp_path / "experiment"
    root, study, source = [experiment / n for n in ("curriculum_campaign", "prepared", "source")]
    source_file = source / "src/transformer_rl/frame_process.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("raise AssertionError('historical learner must never execute')\n")
    source_files = {"src/transformer_rl/frame_process.py": hashlib.sha256(source_file.read_bytes()).hexdigest()}
    source_data = {"root": str(source), "files": source_files, "sha256": analysis.digest(source_files)}
    for key, name in (("controller_sha256", "run_frame_control_campaign.py"),
                      ("transfer_controller_sha256", "run_transfer_campaign.py"),
                      ("curriculum_controller_sha256", "run_curriculum_campaign.py"),
                      ("transfer_helper_sha256", "run_transfer_campaign.py")):
        path = source / "tools" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("raise AssertionError('controller must never execute')\n")
        source_data[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    model = {"policy": {"architecture": "transformer", "residual_type": "gated", "history_length": 31}}
    control, ppo = {"policy_dt_s": .01}, {"learning_rate": 3e-5}
    snapshot = study / "snapshot"
    scenarios, contracts, arms = [], {}, []
    for name in analysis.OLD_CASES + analysis.NEW_CASES:
        route = f"contracts/eval.{name}.json"
        contracts[route] = write(snapshot / route, {"evaluation": {"cases": [{"name": name}]}})
        gates = [{"path": "success_rate", "operator": "min", "value": .95}]
        for key, limit in (("height_abs_error", .03), ("vx_abs_error", .15),
                           ("wz_abs_error", .25), ("tilt_angle", .25)):
            gates.append({"path": f"metrics.{key}.mean", "operator": "max", "value": limit})
        if name in analysis.OLD_CASES:
            gates.append({"path": "metrics.drift_m.max", "operator": "max", "value": .2})
        scenarios.append({"name": name, "environment": {"contract": route, "contract_sha256": contracts[route],
            "num_envs": 8, "evaluation_batch": "transfer_all"}, "gates": gates, "require_steady": True})
    for name, domains in analysis.ARMS.items():
        phases = []
        for i, domain in enumerate(domains):
            route = f"contracts/{name}.phase{i + 1}.json"
            contracts[route] = write(snapshot / route, {"domain": domain, "start_update": (0, 400)[i]})
            phases.append({"name": f"phase{i + 1}", "domain": domain, "updates": (400, 800)[i],
                "start_update": (0, 400)[i], "config": f"configs/{name}.phase{i + 1}.json"})
        arms.append({"name": name, "phases": phases})
    parent = {"scenarios": scenarios, "evaluation": {"steps": 4001, "settle_steps": 200,
              "min_steady_samples": 200, "min_completed_episodes": 8}}
    write(snapshot / "parent_study.json", parent)
    files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in snapshot.rglob("*") if p.is_file()}
    snapshot_data = {"files": files, "sha256": analysis.digest(files)}
    snapshot_file_sha = write(snapshot / "snapshot.json", snapshot_data)
    configs, config_files = {}, {}

    def config(route, environment):
        value = {"model": model, "ppo": ppo, "control": control,
            "environment": {"snapshot": str(snapshot), "snapshot_sha256": snapshot_data["sha256"], **environment}}
        configs[route] = analysis.digest(value)
        config_files[route] = write(study / route, value)

    for arm in arms:
        for phase in arm["phases"]:
            route = f"contracts/{arm['name']}.{phase['name']}.json"
            config(phase["config"], {"contract": route, "contract_sha256": contracts[route], "num_envs": 1024})
    evaluation_cases = []
    for scenario in scenarios:
        route = f"configs/eval.{scenario['name']}.json"
        config(route, scenario["environment"])
        evaluation_cases.append({"name": scenario["name"], "config": route})
    learner = {k.removeprefix("src/transformer_rl/"): v for k, v in source_files.items()}
    manifest = {"format": "transformer_rl.curriculum_study", "schema_version": 1,
        "environment_factory": "transformer_rl.chassis_adapter:make_env",
        "arms": arms, "training_seeds": [1101, 1102, 1103], "scenarios": evaluation_cases,
        "configs": configs, "artifacts": config_files,
        "training": {"rollout_steps": 48, "checkpoint_interval": 100, "max_seconds": 172800.},
        "evaluation": {"seeds": [8701, 9701], "steps": 4001, "settle_steps": 200,
                       "min_steady_samples": 200, "trace_replicas": 8},
        "source_identity": {"learner_source": {"files": learner, "sha256": analysis.digest(learner, ascii=True)}},
        "snapshot_identity": {"path": "snapshot", "sha256": snapshot_data["sha256"], "receipt_sha256": snapshot_file_sha},
        "protocol": {"warmup_updates": 400, "total_updates": 1200, "transitions_per_update": 49152,
            "planned_transitions_per_arm_seed": 58982400, "planned_total_transitions": 58982400 * 9,
            "same_phase_boundary_reset_all_arms": True,
            "phase2_learning_state": "restore_model_optimizer_rng_and_update_counter",
            "phase2_clock": "cumulative_consumed_updates_with_zero_global_offset",
            "phase_evaluation_does_not_gate_progress": True, "task_dose_matched": False,
            "causal_scope": "task_schedule_and_exposure_intervention_not_order_only",
            "policy_hz": 100, "physics_feedback_pd_hz": 200}}
    manifest["sha256"] = analysis.digest(manifest)
    manifest_sha = write(study / "manifest.json", manifest)
    write(root / "manifest.json", manifest)
    campaign = {"format": "transformer_rl.curriculum_campaign", "schema_version": 1,
        "device": "cpu",
        "manifest": str(study / "manifest.json"), "manifest_file_sha256": manifest_sha,
        "manifest_sha256": manifest["sha256"], "source": source_data}
    write(root / "campaign.json", campaign)
    summary = {"status": "waiting", "campaign_sha256": analysis.digest(campaign), "results": {},
               "formal_architecture_selection": False}
    write(root / "summary.json", summary)
    return root, study, manifest, summary


def populate(root, study, manifest, summary, *, learned=True, improved=False):
    for seed in manifest["training_seeds"]:
        for arm in manifest["arms"]:
            name, parent = arm["name"], None
            result = {"arm": name, "training_seed": seed, "phases": [], "status": "completed"}
            summary["results"][f"{name}/seed_{seed}"] = result
            for phase in arm["phases"]:
                offset, updates = phase["start_update"], phase["updates"]
                cfg = read(study / phase["config"])
                directory = root / "jobs" / name / f"seed_{seed}" / phase["name"] / "training/attempt_001"
                train = directory / "train"
                train.mkdir(parents=True)
                checkpoint_path = train / "final.pt"
                checkpoint_path.write_bytes(f"synthetic-{seed}-{name}-{phase['name']}".encode())
                checkpoint = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": artifact(checkpoint_path, root)["sha256"],
                    "update": offset + updates, "cumulative_transitions": (offset + updates) * 49152}
                identity = {"manifest_sha256": manifest["sha256"], "arm": name, "phase": phase["name"],
                    "training_seed": seed, "config_sha256": manifest["configs"][phase["config"]],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"] if parent else None,
                    "consumed_update_offset": offset}
                initialization = {"restore_learning_from": parent["checkpoint"]} if parent else {}
                request = {"identity": identity, "config": cfg, "initialization": initialization,
                    "updates": updates, "start_update": offset, "prior_transitions": offset * 49152,
                    "batch_samples_per_update": 49152}
                request_sha = write(directory / "request.json", request)
                initial = analysis.digest({"model_seed": seed, "restored": parent["checkpoint_sha256"] if parent else None})
                write(Path(str(checkpoint_path) + ".json"), {"format": "transformer_rl.packed_checkpoint", "schema_version": 1,
                    "sha256": checkpoint["checkpoint_sha256"], "update": checkpoint["update"], "config": cfg,
                    "metadata": {"source": manifest["source_identity"]["learner_source"], "seed": seed,
                        "environment_factory": manifest["environment_factory"], "retention_coef": 0., "anchors": [],
                        "episode_state_restored": False, "initial_model_sha256": initial,
                        "initial_model_hash_format": "sorted_named_tensor_contents_v1",
                        "collected_transitions": checkpoint["cumulative_transitions"]}})
                write(train / "run.json", {"config": cfg, "seed": seed, "initial_model_sha256": initial, **initialization,
                    "environment_factory": manifest["environment_factory"], "rollout_steps": 48,
                    "source": manifest["source_identity"]["learner_source"], "retention_coef": 0.,
                    "episode_state_restored": False, "history_reset": "repeat_first", "updates": updates,
                    "initial_model_hash_format": "sorted_named_tensor_contents_v1", "device": "cpu",
                    "max_seconds": manifest["training"]["max_seconds"], "checkpoint_interval": 100})
                write(train / "completion.json", {"status": "completed", "completed_updates": updates,
                    "attempted_updates": updates, "start_update": offset, "final_update": offset + updates,
                    "config_sha256": identity["config_sha256"], "consumed_transitions": updates * 49152,
                    "cumulative_transitions": checkpoint["cumulative_transitions"],
                    "checkpoint": checkpoint["checkpoint"], "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
                (train / "metrics.jsonl").write_text("".join(json.dumps({"update": offset + i,
                    "batch_samples": 49152, "optimization": {"optimizer_steps": 2, "sample_count": 98304}}) + "\n"
                    for i in range(1, updates + 1)))
                receipt = {"identity": identity, "request_sha256": request_sha, "status": "completed",
                    "initial_model_sha256": initial, "consumed_updates": updates, "consumed_transitions": updates * 49152,
                    "cumulative_transitions": checkpoint["cumulative_transitions"], "checkpoint": checkpoint,
                    "evidence": {"updates": updates, "last_update": offset + updates, "optimizer_steps": updates * 2,
                        "batch_samples": updates * 49152, "ppo_verified": True,
                        "metrics_sha256": artifact(train / "metrics.jsonl", root)["sha256"],
                        "optimization_samples": updates * 98304},
                    "artifacts": {k: artifact(train / f, root) for k, f in
                        (("run", "run.json"), ("completion", "completion.json"), ("metrics", "metrics.jsonl"), ("checkpoint", "final.pt"))}}
                write(directory / "receipt.json", receipt)
                training = {"status": "completed", "consumed_updates": updates, "checkpoint": checkpoint,
                    "initial_model_sha256": initial, "attempts": [artifact(directory / "receipt.json", root)]}
                item = {"name": phase["name"], "training": training, "evaluations": {}}
                result["phases"].append(item)
                for eval_seed in manifest["evaluation"]["seeds"]:
                    evdir = root / "control" / name / f"train_{seed}" / phase["name"] / f"seed_{eval_seed}" / "attempt_001"
                    evdir.mkdir(parents=True)
                    groups, reports, artifacts = {}, {}, {}
                    for case in manifest["scenarios"]:
                        case_name, old = case["name"], case["name"] in analysis.OLD_CASES
                        height, vx, yaw, drift = (.01, .03, .05, .05) if old else (.07, .30, .40, .05)
                        if old and not learned:
                            height = .08
                        if offset:
                            if old and name == "pretrain":
                                height, vx, drift = (.005, .02, .04) if improved else (.05 if learned else .1, .2, .3)
                            elif old and name == "stationary":
                                height, vx, drift = (.012 if learned else .08), .04, .06
                            elif not old and name in ("pretrain", "mixed"):
                                height, vx, yaw = .015, .05, .1
                        quality = {"full_interval": {"samples": 32008}, "actuation": {"sample_count": 32008},
                            "protocol": {"policy_dt_s": .01, "settle_steps": 200, "min_steady_samples": 200,
                                         "tracking_tolerance": [.15, .25, .03]}}
                        groups[case_name] = quality
                        report = {"format": "transformer_rl.packed_evaluation", "schema_version": 1,
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": checkpoint["update"],
                            "seed": eval_seed, "steps": 4001, "num_envs": 8, "transitions": 32008,
                            "model": cfg["model"], "control_sha256": analysis.digest(cfg["control"]),
                            "environment": read(study / case["config"])["environment"],
                            "completed_episodes": 8, "failed_episodes": 0, "success_rate": 1.,
                            "metrics": {k: {"mean": v, "max": v, "count": 32008} for k, v in
                                (("height_abs_error", height), ("vx_abs_error", vx), ("wz_abs_error", yaw),
                                 ("tilt_angle", .05), ("drift_m", drift))},
                            "stability": {"available": True, "protocol": {"settle_steps": 200, "min_steady_samples": 200}},
                            "control": quality}
                        path = evdir / f"{case_name}.json"
                        write(path, report)
                        artifacts[case_name] = artifact(path, root)
                        reports[case_name] = {"report": artifacts[case_name]}
                    trace_path = evdir / "trace.npz"
                    trace_path.write_bytes(b"synthetic trace is only hashed, never loaded")
                    trace = {"path": str(trace_path), "sha256": artifact(trace_path, root)["sha256"],
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": checkpoint["update"],
                        "seed": eval_seed, "steps": 4001, "rows": list(range(400)), "row_indices": list(range(400)),
                        "group_labels": [c["name"] for c in manifest["scenarios"] for _ in range(8)]}
                    write(evdir / "control.json", {"format": "transformer_rl.control_evaluation", "schema_version": 1,
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "checkpoint_update": checkpoint["update"],
                        "seed": eval_seed, "steps": 4001, "groups": groups, "trace": trace})
                    artifacts.update(control=artifact(evdir / "control.json", root), trace=artifact(trace_path, root))
                    evreceipt = {"status": "completed", "identity": {"manifest_sha256": manifest["sha256"],
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "evaluation_seed": eval_seed,
                        "protocol": manifest["evaluation"]}, "scenarios": reports, "artifacts": artifacts}
                    write(evdir / "receipt.json", evreceipt)
                    item["evaluations"][str(eval_seed)] = evreceipt
                parent = checkpoint
    summary["status"] = "completed"
    write(root / "summary.json", summary)


def test_waiting_is_1800_unknown_cells_and_input_is_unchanged(tmp_path):
    root, _, _, _ = frozen(tmp_path)
    before = {str(p): p.read_bytes() for p in root.parent.rglob("*") if p.is_file()}
    protocol = analysis.prepare(root)
    result = analysis.analyze(protocol)
    assert result["status"] == "not_ready" and result["expected_cells"] == len(result["cells"]) == 1800
    assert result["ready_cells"] == 0 and len(result["unready"]) == 1800
    assert all(c["passed"] is None for c in result["cells"])
    assert all(r["retained_case_fraction"] is None for r in result["training_seed_results"])
    assert before == {str(p): p.read_bytes() for p in root.parent.rglob("*") if p.is_file()}
    assert not result["formal_architecture_selection"]
    assert "torch" not in analysis.__dict__ and "numpy" not in analysis.__dict__


def test_real_gate_regression_and_new_gain_have_correct_units_and_control_difference(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    result = analysis.analyze(protocol)
    assert result["status"] == "complete" and result["ready_cells"] == 1800
    old = next(c for c in result["paired_changes"] if c["case"] == analysis.OLD_CASES[0])
    assert old["qualified_forgetting"]["height_mae_m"] == pytest.approx(.04)
    assert old["qualified_forgetting_difference"]["height_mae_m"] == pytest.approx(.038)
    assert old["positive_forgetting"]["vx_mae_m_s"] == pytest.approx(.17)
    assert old["retained"] is False
    new = next(c for c in result["paired_changes"] if c["case"] == analysis.NEW_CASES[0])
    assert new["new_error_reduction"]["vx_mae_m_s"] == pytest.approx(.25)
    assert new["qualified_forgetting"] is None
    assert all(r["acquired_old_case_count"] == 10 and r["retained_case_fraction"] == 0 for r in result["training_seed_results"])
    assert result["aggregate"]["retained_case_fraction"]["n_training_seeds"] == 3
    assert len(result["paired_changes"]) == 150
    assert all(c["gate_values"]["success_rate"] == 1 for c in result["cells"])


def test_never_acquired_is_not_forgetting_despite_later_degradation(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary, learned=False)
    result = analysis.analyze(protocol)
    assert result["status"] == "complete"
    old = [c for c in result["paired_changes"] if c["role"] == "old"]
    assert all(c["qualified_forgetting"] is None and c["retained"] is None for c in old)
    assert old[0]["old_error_increase"]["height_mae_m"] == pytest.approx(.02)
    assert all(r["retained_case_fraction"] is None and r["retention_reason"] == "old_task_not_acquired"
               for r in result["training_seed_results"])
    assert result["aggregate"]["retained_case_fraction"]["mean"] is None


def test_signed_positive_transfer_is_retained_and_nonnegative_forgetting_is_zero(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary, improved=True)
    result = analysis.analyze(protocol)
    old = next(c for c in result["paired_changes"] if c["role"] == "old")
    assert old["qualified_forgetting"]["height_mae_m"] == pytest.approx(-.005)
    assert old["positive_forgetting"]["height_mae_m"] == 0
    assert all(r["retained_case_fraction"] == 1 for r in result["training_seed_results"])


def test_missing_post_case_is_unknown_without_shrinking_acquired_denominator(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    path = root / "control/pretrain/train_1101/phase2/seed_8701/attempt_001" / f"{analysis.OLD_CASES[0]}.json"
    path.unlink()
    result = analysis.analyze(protocol)
    assert result["status"] == "not_ready" and result["ready_cells"] == 1799
    row = result["training_seed_results"][0]
    assert row["acquired_old_case_count"] == 10 and row["known_lost_case_count"] == 9
    assert row["retained_case_fraction"] is None and row["unknown_post_cases"] == [analysis.OLD_CASES[0]]
    assert result["aggregate"]["retained_case_fraction"]["mean"] is None


def test_report_tampering_is_rejected_even_when_summary_says_completed(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    path = root / "control/pretrain/train_1101/phase2/seed_8701/attempt_001" / f"{analysis.OLD_CASES[0]}.json"
    report = read(path)
    report["metrics"]["height_abs_error"]["mean"] = 0
    write(path, report)
    with pytest.raises(ValueError, match="SHA mismatch"):
        analysis.analyze(protocol)


@pytest.mark.parametrize("mutation", ["weights", "gates", "cases"])
def test_resealing_protocol_cannot_change_predeclared_scientific_rules(tmp_path, mutation):
    root, _, _, _ = frozen(tmp_path)
    protocol = analysis.prepare(root)
    if mutation == "weights":
        protocol["weighting"] = "choose only successful cases"
    elif mutation == "gates":
        protocol["gates"][analysis.OLD_CASES[0]]["gates"][1]["value"] = 1.
    else:
        protocol["old_cases"] = protocol["old_cases"][:-1]
    protocol["sha256"] = analysis.digest({k: v for k, v in protocol.items() if k != "sha256"})
    with pytest.raises(ValueError, match="protocol changed"):
        analysis.analyze(protocol)


@pytest.mark.parametrize("escape", ["relative", "symlink"])
def test_source_paths_cannot_escape_experiment(tmp_path, escape):
    root, _, _, _ = frozen(tmp_path)
    path = root / "campaign.json"
    campaign = read(path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "relative":
        campaign["source"]["root"] = "../outside"
    else:
        link = root.parent / "external_source"
        link.symlink_to(outside, target_is_directory=True)
        campaign["source"]["root"] = str(link)
    write(path, campaign)
    with pytest.raises(ValueError, match="escapes"):
        analysis.prepare(root)


def test_checkpoint_parent_and_training_seed_continuum_are_verified(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    result = summary["results"]["pretrain/seed_1101"]
    phase = result["phases"][1]
    receipt_path = root / phase["training"]["attempts"][0]["path"]
    receipt = read(receipt_path)
    request_path = receipt_path.parent / "request.json"
    request = read(request_path)
    request["identity"]["parent_checkpoint_sha256"] = "0" * 64
    receipt["identity"] = request["identity"]
    receipt["request_sha256"] = write(request_path, request)
    write(receipt_path, receipt)
    phase["training"]["attempts"][0] = artifact(receipt_path, root)
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match="training request identity"):
        analysis.analyze(protocol)


@pytest.mark.parametrize("field,value", [("source", {"files": {}, "sha256": "0" * 64}),
    ("environment_factory", "wrong.module:environment"), ("rollout_steps", 47),
    ("retention_coef", .2), ("history_reset", "persist"), ("source", None)])
def test_resealed_run_cannot_change_source_or_training_recipe(tmp_path, field, value):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    phase = summary["results"]["mixed/seed_1101"]["phases"][0]
    receipt_path = root / phase["training"]["attempts"][0]["path"]
    receipt = read(receipt_path)
    run_path = root / receipt["artifacts"]["run"]["path"]
    run = read(run_path)
    if value is None:
        del run[field]
    else:
        run[field] = value
    write(run_path, run)
    receipt["artifacts"]["run"] = artifact(run_path, root)
    write(receipt_path, receipt)
    phase["training"]["attempts"][0] = artifact(receipt_path, root)
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match="run source, recipe"):
        analysis.analyze(protocol)


def test_existing_summary_requires_campaign_sha_but_missing_summary_is_not_started(tmp_path):
    root, _, _, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    del summary["campaign_sha256"]
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match="another campaign"):
        analysis.analyze(protocol)
    (root / "summary.json").unlink()
    result = analysis.analyze(protocol)
    assert result["status"] == "not_ready" and result["campaign_status"] == "not_started"


def reseal_case(root, summary, *, arm="pretrain", phase="phase1", case=None, mutate):
    case = case or analysis.OLD_CASES[0]
    item = summary["results"][f"{arm}/seed_1101"]["phases"][int(phase[-1]) - 1]
    for eval_seed in (8701, 9701):
        receipt = item["evaluations"][str(eval_seed)]
        path = root / receipt["artifacts"][case]["path"]
        report = read(path)
        mutate(report)
        write(path, report)
        receipt["artifacts"][case] = artifact(path, root)
        receipt["scenarios"][case]["report"] = artifact(path, root)
        write(path.parent / "receipt.json", receipt)
    write(root / "summary.json", summary)


def test_missing_old_reference_keeps_unknown_denominator_and_no_qualified_forgetting(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    path = root / "control/pretrain/train_1101/phase1/seed_8701/attempt_001" / f"{analysis.OLD_CASES[0]}.json"
    path.unlink()
    result = analysis.analyze(protocol)
    row = result["training_seed_results"][0]
    assert row["acquired_old_case_count"] == 9 and row["retained_case_fraction"] is None
    assert row["unknown_reference_cases"] == [analysis.OLD_CASES[0]]
    assert row["qualified_old_forgetting"] is None


def test_stationary_unlearned_reference_blocks_only_qualified_controlled_forgetting(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    reseal_case(root, summary, arm="stationary", mutate=lambda r: r["metrics"]["height_abs_error"].update(mean=.05))
    result = analysis.analyze(protocol)
    row = next(c for c in result["paired_changes"] if c["training_seed"] == 1101 and c["case"] == analysis.OLD_CASES[0])
    assert row["qualified_forgetting"] is not None and row["qualified_forgetting_difference"] is None
    assert row["pretrain_minus_stationary_error_change"]["height_mae_m"] == pytest.approx(.078)


@pytest.mark.parametrize("mutation", ["initial_model", "weights_only_initializer"])
def test_same_seed_initial_weights_and_preserved_optimizer_initialization_are_required(tmp_path, mutation):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    arm, index = ("stationary", 0) if mutation == "initial_model" else ("pretrain", 1)
    phase = summary["results"][f"{arm}/seed_1101"]["phases"][index]
    receipt_path = root / phase["training"]["attempts"][0]["path"]
    receipt = read(receipt_path)
    run_path = root / receipt["artifacts"]["run"]["path"]
    run = read(run_path)
    if mutation == "initial_model":
        run["initial_model_sha256"] = receipt["initial_model_sha256"] = phase["training"]["initial_model_sha256"] = "f" * 64
        sidecar_path = Path(str(root / receipt["artifacts"]["checkpoint"]["path"]) + ".json")
        sidecar = read(sidecar_path)
        sidecar["metadata"]["initial_model_sha256"] = "f" * 64
        write(sidecar_path, sidecar)
        match = "initialized different models"
    else:
        request_path = receipt_path.parent / "request.json"
        request = read(request_path)
        parent = request["initialization"]["restore_learning_from"]
        request["initialization"] = {"initialize_from": parent}
        receipt["request_sha256"] = write(request_path, request)
        run.pop("restore_learning_from")
        run["initialize_from"] = parent
        match = "parent or sample clock"
    write(run_path, run)
    receipt["artifacts"]["run"] = artifact(run_path, root)
    write(receipt_path, receipt)
    phase["training"]["attempts"][0] = artifact(receipt_path, root)
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match=match):
        analysis.analyze(protocol)


def test_missing_finite_metric_is_unknown_but_wrong_sample_count_is_rejected(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    reseal_case(root, summary, phase="phase2", mutate=lambda r: r["metrics"]["height_abs_error"].update(mean=None))
    result = analysis.analyze(protocol)
    assert result["status"] == "not_ready" and result["ready_cells"] == 1798
    assert result["training_seed_results"][0]["retained_case_fraction"] is None
    reseal_case(root, summary, phase="phase2", mutate=lambda r: r["metrics"]["height_abs_error"].update(mean=.05, count=32007))
    with pytest.raises(ValueError, match="metric sample coverage"):
        analysis.analyze(protocol)


def test_evaluation_receipt_path_and_seed_cannot_be_reassigned(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    receipt = summary["results"]["pretrain/seed_1101"]["phases"][0]["evaluations"]["8701"]
    receipt_path = root / "control/pretrain/train_1101/phase1/seed_8701/attempt_001/receipt.json"
    receipt["identity"]["evaluation_seed"] = 9701
    write(receipt_path, receipt)
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match="evaluation receipt identity"):
        analysis.analyze(protocol)
    receipt["identity"]["evaluation_seed"] = 8701
    receipt["artifacts"][analysis.OLD_CASES[0]]["path"] = str(tmp_path / "outside.json")
    write(receipt_path, receipt)
    write(root / "summary.json", summary)
    with pytest.raises(ValueError, match="escapes"):
        analysis.analyze(protocol)


def test_full_rollout_time_stop_then_resume_preserves_continuous_sample_ledger(tmp_path):
    root, study, manifest, summary = frozen(tmp_path)
    protocol = analysis.prepare(root)
    populate(root, study, manifest, summary)
    phases = summary["results"]["mixed/seed_1101"]["phases"]
    first_path = root / phases[0]["training"]["attempts"][0]["path"]
    original = read(first_path)
    original_files = {k: root / a["path"] for k, a in original["artifacts"].items()}
    model_bytes = original_files["checkpoint"].read_bytes()
    rows = original_files["metrics"].read_text().splitlines()
    cfg = read(original_files["run"])["config"]
    initial = original["initial_model_sha256"]
    prior = None
    attempts = []
    for index, (start, count) in enumerate(((0, 100), (100, 300)), 1):
        directory = first_path.parent.parent / f"attempt_{index:03d}"
        train = directory / "train"
        train.mkdir(parents=True, exist_ok=True)
        checkpoint_path = train / "final.pt"
        checkpoint_path.write_bytes(b"synthetic full-rollout C100" if index == 1 else model_bytes)
        checkpoint = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": artifact(checkpoint_path, root)["sha256"],
            "update": start + count, "cumulative_transitions": (start + count) * 49152}
        request = read(first_path.parent / "request.json")
        request.update(updates=400 - start, start_update=start, prior_transitions=start * 49152,
                       initialization={"resume": prior["checkpoint"]} if prior else {})
        request["identity"].update(consumed_update_offset=start,
                                  parent_checkpoint_sha256=prior["checkpoint_sha256"] if prior else None)
        request_sha = write(directory / "request.json", request)
        run = read(original_files["run"])
        run.update(updates=400 - start, resume=prior["checkpoint"] if prior else None,
                   initial_model_sha256=initial if index == 1 else analysis.digest({"resumed": "C100"}))
        write(train / "run.json", run)
        completion = {"status": "stopped" if index == 1 else "completed", "stop_reason": "time_budget" if index == 1 else None,
            "completed_updates": count, "attempted_updates": count, "start_update": start, "final_update": start + count,
            "config_sha256": request["identity"]["config_sha256"], "consumed_transitions": count * 49152,
            "cumulative_transitions": checkpoint["cumulative_transitions"], "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"]}
        write(train / "completion.json", completion)
        (train / "metrics.jsonl").write_text("\n".join(rows[start:start + count]) + "\n")
        sidecar = read(Path(str(original_files["checkpoint"]) + ".json"))
        sidecar.update(sha256=checkpoint["checkpoint_sha256"], update=start + count)
        sidecar["metadata"].update(initial_model_sha256=run["initial_model_sha256"], collected_transitions=(start + count) * 49152)
        write(Path(str(checkpoint_path) + ".json"), sidecar)
        receipt = deepcopy(original)
        receipt.update(identity=request["identity"], request_sha256=request_sha, checkpoint=checkpoint,
            status="resumable" if index == 1 else "completed", consumed_updates=count, consumed_transitions=count * 49152,
            cumulative_transitions=(start + count) * 49152, initial_model_sha256=run["initial_model_sha256"],
            evidence={"updates": count, "last_update": start + count, "optimizer_steps": count * 2,
                "batch_samples": count * 49152, "ppo_verified": True,
                "metrics_sha256": artifact(train / "metrics.jsonl", root)["sha256"], "optimization_samples": count * 98304},
            artifacts={k: artifact(train / filename, root) for k, filename in
                (("completion", "completion.json"), ("run", "run.json"), ("metrics", "metrics.jsonl"), ("checkpoint", "final.pt"))})
        write(directory / "receipt.json", receipt)
        attempts.append(artifact(directory / "receipt.json", root))
        prior = checkpoint
    phases[0]["training"].update(checkpoint=prior, attempts=attempts)
    second_path = root / phases[1]["training"]["attempts"][0]["path"]
    second = read(second_path)
    request_path = second_path.parent / "request.json"
    request = read(request_path)
    request["initialization"]["restore_learning_from"] = prior["checkpoint"]
    second["request_sha256"] = write(request_path, request)
    run_path = root / second["artifacts"]["run"]["path"]
    run = read(run_path)
    run["restore_learning_from"] = prior["checkpoint"]
    write(run_path, run)
    second["artifacts"]["run"] = artifact(run_path, root)
    write(second_path, second)
    phases[1]["training"]["attempts"][0] = artifact(second_path, root)
    write(root / "summary.json", summary)
    result = analysis.analyze(protocol)
    assert result["status"] == "complete" and result["ready_cells"] == 1800
    assert any("attempt_002/request.json" in p for p in result["input_receipts"])


def test_cli_only_creates_external_new_files_and_waiting_is_successful_capture(tmp_path):
    root, _, _, _ = frozen(tmp_path)
    protocol_path, report_path = tmp_path / "analysis_protocol.json", tmp_path / "analysis.json"
    prepared = subprocess.run([sys.executable, "-B", str(TOOL), "prepare", "--campaign-root", str(root),
        "--output", str(protocol_path)], capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stderr
    captured = subprocess.run([sys.executable, "-B", str(TOOL), "analyze", "--protocol", str(protocol_path),
        "--output", str(report_path)], capture_output=True, text=True)
    assert captured.returncode == 0, captured.stderr
    assert read(report_path)["status"] == "not_ready"
    with pytest.raises(FileExistsError):
        analysis.write_external(report_path, {}, root)
    with pytest.raises(ValueError, match="outside"):
        analysis.write_external(root.parent / "prepared/analysis.json", {}, root)
    assert not (root.parent / "prepared/analysis.json").exists()


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"value":NaN}', '{"value":1e1000}'])
def test_invalid_json_is_not_silently_accepted(text):
    with pytest.raises(ValueError):
        analysis._json(text)

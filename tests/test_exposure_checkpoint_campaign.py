"""Continuous checkpoint integration with actual CPU learners and OS workers.

The predecessor and six-motor physics providers are synthetic fixtures. These
checks do not establish production queue closure, Isaac behavior or deployment.
"""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from transformer_rl import exposure_campaign as campaign
from transformer_rl import exposure_evaluation as evaluator
from transformer_rl import exposure_protocol as protocol
from transformer_rl import exposure_training as training
from transformer_rl import exposure_selection as selection
from transformer_rl import history_study
from transformer_rl.frame_config import FrameTrainConfig
from test_exposure_campaign import campaign_inputs, canonical, run
from test_exposure_evaluation import (make_six_motor_env, physical_protocol,
                                      run_physical_worker)
from test_exposure_protocol import prepared, freeze, write_json
from test_exposure_selection import physical_grid


def scheduled_protocol(p, *, interval=1):
    value = freeze(p, checkpoint_interval=interval)
    canonical(p["protocol_path"], value)
    p["protocol"] = value
    storage = deepcopy(p["storage"])
    storage["protocol_raw_sha256"] = protocol._receipt(p["protocol_path"])["sha256"]
    storage["source"] = value["source"]
    canonical(p["storage_path"], storage)
    p["storage"] = storage
    return value


def test_actual_full_actor_grid_keeps_one_process_per_stage_and_every_saved_point(campaign_inputs):
    p = campaign_inputs
    spec = deepcopy(p["spec"])
    for stage in spec["stages"]:
        stage["updates"] = 2
    write_json(p["inputs"] / "study.json", spec)
    p["history"] = p["tmp"] / "scheduled_history"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json",
        p["history"], history_lengths=[1, 3], position_reference="current")
    value = scheduled_protocol(p)
    result = run(p)
    assert result["status"] == "training_completed_evaluation_pending"
    assert result["charged_updates"] == value["budget"]["training_updates"]
    assert result["verified_successful_updates"] == result["charged_updates"]
    assert result["verified_fresh_transitions"] == value["budget"]["fresh_transitions"]
    assert len(result["evaluation_cells"]) == len(value["evaluation_cells"])
    assert all(cell["status"] == "missing" for cell in result["evaluation_cells"].values())
    handles = set()
    for job in value["jobs"]:
        record = result["jobs"][job["id"]]
        assert record["status"] == "training_completed"
        for stage, entry in zip(job["stages"], record["stages"]):
            handle = entry["worker"]["process"]
            handles.add((handle["pid"], handle["start"]))
            assert len(entry["checkpoints"]) == 2
            assert [point["kind"] for point in entry["checkpoints"]] == ["intermediate", "endpoint"]
            expected = stage["expected_cumulative_updates"]
            assert [point["checkpoint_update"] for point in entry["checkpoints"]] == [expected-1, expected]
            assert entry["training"]["endpoint"] == entry["checkpoints"][-1]["record"]
            for point in entry["checkpoints"]:
                actual = campaign.verify_learning_checkpoint(value, job, stage["index"],
                    point["checkpoint_update"], point["record"])
                assert actual["checkpoint"] == point["checkpoint"]
                payload = torch.load(point["checkpoint"]["path"], map_location="cpu", weights_only=True)
                assert payload["update"] == point["checkpoint_update"]
                assert payload["metadata"]["fixed_exposure_job"]["stage_index"] == stage["index"]
    assert len(handles) == sum(len(job["stages"]) for job in value["jobs"])


def test_independent_physical_worker_evaluates_the_exact_intermediate_model(physical_protocol):
    p = physical_protocol
    p["output"] = p["tmp"] / "scheduled_physical_campaign"
    p["protocol_path"] = p["tmp"] / "scheduled_physical_protocol.json"
    value = scheduled_protocol(p)
    job = next(item for item in value["jobs"] if item["candidate"] == "gated_h4"
               and item["training_seed"] == 71)
    stage = job["stages"][0]
    root = Path(value["output_root"])
    stage_root = root / job["id"] / "stage_0000"
    stage_root.mkdir(parents=True)
    completion = training.train_exposure_segment(campaign.stage_definition(stage), make_six_motor_env,
        value["environment_factory"], stage_root / "train", job_id=job["id"],
        rollout_steps=value["execution"]["rollout_steps"], training_seed=job["training_seed"],
        retention_seed=job["retention_seed"],
        evaluation_seeds=[*value["evaluation"]["validation_seeds"], *value["evaluation"]["seeds"]],
        device="cpu", max_seconds=value["execution"]["max_seconds"],
        expected_initial_model_sha256=job["initial_model_sha256"])
    assert completion["status"] == "completed"
    point = completion["sealed_checkpoints"][0]
    assert point["kind"] == "intermediate" and point["checkpoint_update"] == 1
    p["job"], p["endpoint"] = job, point["record"]
    p["cells"] = [cell for cell in value["evaluation_cells"]
                  if cell["job_id"] == job["id"] and cell["stage_index"] == 0
                  and cell["checkpoint_update"] == 1 and cell["role"] == "validation" and cell["seed"] == 701]
    p["directory"] = campaign.evaluation_directory(value, p["cells"][0])
    controller = {"format": "transformer_rl.exposure_controller", "schema_version": 1,
        "protocol_sha256": value["sha256"], "source": value["source"], "runtime": value["runtime"],
        "process": campaign._identity(__import__("os").getpid()),
        "protocol_raw_receipt": protocol._receipt(p["protocol_path"]),
        "expected_protocol_sha256": protocol._receipt(p["protocol_path"])["sha256"],
        "storage_contract_receipt": protocol._receipt(p["storage_path"])}
    canonical(root / "controller.json", controller)
    p["controller"] = protocol._receipt(root / "controller.json")
    planned = run_physical_worker(p)
    result = evaluator.verify_result(value, p["endpoint"], p["cells"], p["directory"], planned["request"])
    assert set(result["cells"]) == {cell["id"] for cell in p["cells"]}
    for cell in p["cells"]:
        assert result["cells"][cell["id"]]["identity"]["checkpoint_update"] == 1
    final = completion["sealed_checkpoints"][-1]["record"]
    with pytest.raises(ValueError, match="outside the authorized stage"):
        campaign.verify_learning_checkpoint(value, job, 0, 1, final)
    config = FrameTrainConfig.from_dict(stage["config"])
    plan, _ = training._definition([campaign.stage_definition(stage)], job_id=job["id"],
        rollout_steps=value["execution"]["rollout_steps"], training_seed=job["training_seed"],
        retention_seed=job["retention_seed"], evaluation_seeds=[701,1701,2701,3701], device="cpu",
        max_seconds=value["execution"]["max_seconds"],
        expected_initial_model_sha256=job["initial_model_sha256"], environment_reference=value["environment_factory"])
    with pytest.raises(ValueError, match="endpoint|schema|complete"):
        training._segment_parent(point["record"], plan, config)


def prepare_failed_prefix(p, updates):
    fixture = p["sdk"] / "physical_cpu_fixture.py"
    source = fixture.read_text()
    old = """        def failure(action):
            raise FloatingPointError('actual CPU fixture nonfinite training step')
        env.step = failure"""
    new = """        calls = 0
        original_step = env.step
        def failure(action):
            nonlocal calls
            calls += 1
            if calls > 2:
                raise FloatingPointError('actual CPU fixture fails after a sealed update')
            return original_step(action)
        env.step = failure"""
    assert source.count(old) == 1
    fixture.write_text(source.replace(old, new))
    spec = deepcopy(p["spec"])
    spec["stages"][0]["updates"] = updates
    write_json(p["inputs"] / "study.json", spec)
    p["history"] = p["tmp"] / "failed_prefix_history"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json",
        p["history"], history_lengths=[1, 4], position_reference="current")
    p["storage"] = campaign._read(p["storage_path"])
    return scheduled_protocol(p)


def test_actual_failed_worker_prefix_is_validated_and_evaluated_without_a_final_model(physical_grid, monkeypatch):
    p = physical_grid
    value = prepare_failed_prefix(p, 3)
    first = value["jobs"][0]
    assert first["training_seed"] == 71 and first["candidate"] == "mlp_h1"
    original_launch = campaign.launch_owned_worker
    launched = []

    def stop_after_first_real_job(command, directory, *args, **kwargs):
        if "transformer_rl.exposure_process" in command and Path(directory).parent.name != first["id"]:
            raise campaign.CampaignWorkerError("CPU fixture ends after the first actual failed-prefix job")
        result = original_launch(command, directory, *args, **kwargs)
        launched.append(deepcopy(result))
        return result

    monkeypatch.setattr(campaign, "launch_owned_worker", stop_after_first_real_job)
    with pytest.raises(campaign.CampaignWorkerError, match="fixture ends"):
        run(p, evaluation_provider="transformer_rl.exposure_evaluation", confirm_heldout=False)
    summary = campaign._read(Path(value["output_root"]) / "summary.json")
    assert summary["status"] == "stopped" and summary["full_evaluation_matrix_closed"] is False
    assert len(summary["evaluation_cells"]) == len(value["evaluation_cells"])
    entry = summary["jobs"][first["id"]]["stages"][0]
    assert entry["worker"]["returncode"] == 20 and entry["worker"]["timed_out"] is False
    assert entry["failed_training"]["successful_updates"] == 1
    assert [point["checkpoint_update"] for point in entry["checkpoints"]] == [1]
    cells = [cell for cell in summary["evaluation_cells"].values() if cell["identity"]["job_id"] == first["id"]]
    assert all(cell["status"] == "completed" for cell in cells
               if cell["identity"]["role"] == "validation" and cell["identity"]["checkpoint_update"] == 1)
    assert all(cell["status"] == "missing" for cell in cells if cell["identity"]["checkpoint_update"] in (2,3))
    assert len(launched) == 3  # One failed training worker and both validation seeds.
    assert len({(worker["process"]["pid"], worker["process"]["start"]) for worker in launched}) == 3
    stage = first["stages"][0]
    final = campaign.learning_checkpoint_records(value, first, stage)[-1]
    assert not Path(final["path"]).exists() and not Path(final["checkpoint_path"]).exists()
    assert summary["jobs"][first["id"]]["reservation"]["path"]
    assert summary["charged_updates"] >= first["reserved_updates"]


def test_actual_scheduled_validation_and_heldout_close_with_verified_failed_prefix(physical_grid, monkeypatch):
    p = physical_grid
    value = prepare_failed_prefix(p, 2)
    original_seal = selection.freeze_selection
    observed_choices = []

    def observe_choice(*args, **kwargs):
        assert not any(campaign.evaluation_directory(value, cell).exists()
                       for cell in value["evaluation_cells"] if cell["role"] == "held_out")
        sealed = original_seal(*args, **kwargs)
        observed_choices.append(deepcopy(sealed))
        return sealed

    monkeypatch.setattr(selection, "freeze_selection", observe_choice)
    summary = run(p, evaluation_provider="transformer_rl.exposure_evaluation")
    assert summary["status"] == "comparison_closed_deployment_qualification_pending"
    assert summary["full_evaluation_matrix_closed"] is True
    assert summary["validation_matrix_closed"] is True
    assert summary["charged_updates"] == value["budget"]["training_updates"]
    assert summary["charged_fresh_transitions"] == value["budget"]["fresh_transitions"]
    assert len(summary["evaluation_cells"]) == len(value["evaluation_cells"]) == 240
    first = value["jobs"][0]
    assert summary["jobs"][first["id"]]["status"] == "numerical_failure"
    for cell in summary["evaluation_cells"].values():
        failed_final = cell["identity"]["job_id"] == first["id"] and cell["identity"]["checkpoint_update"] == 2
        assert cell["status"] == ("missing" if failed_final else "completed")
    assert len(observed_choices) == 1
    choice = selection.verify_selection(observed_choices[0]["receipt"])
    assert choice["best_transformer"] is not None
    chosen = choice["best_transformer"]["representative"]
    assert chosen["stage_index"] == 0
    endpoint = campaign._read(chosen["endpoint"]["path"])
    assert endpoint["cumulative_successful_updates"] == 2
    assert Path(endpoint["checkpoint"]["path"]).name == "endpoint.pt"
    assert choice["hardware_verified"] is False

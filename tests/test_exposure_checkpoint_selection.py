"""Scheduled-save selection rules, with explicit logical rather than physical data.

These fixtures exercise selection authorization and denominator semantics. They
do not claim actual PPO, simulation measurements, deployment or worker closure;
the campaign/training integration suites prove the independent save evidence.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from transformer_rl import exposure_campaign as campaign
from transformer_rl import exposure_selection as selection


def grid():
    protocol = {"schema_version": 2, "jobs": [], "evaluation_cells": [], "selection": {
        "min_training_seeds": 3, "std_penalty": 1., "retention_score_tolerance": 2.}}
    training, cells = {}, {}
    for name in ("gated", "query"):
        for seed in (71, 97, 101):
            job_id = f"{name}_{seed}"
            stages = [{"index": index, "updates": 100, "checkpoint_updates": [50, 100],
                "expected_cumulative_updates": (index + 1) * 100,
                "config": {"model": {"policy": {"architecture": "transformer"}}}} for index in range(2)]
            protocol["jobs"].append({"id": job_id, "candidate": name, "training_seed": seed, "stages": stages})
            records = [{"stage_index": index, "endpoint": {"path": f"{job_id}/{index}/endpoint.json"},
                "checkpoint": {"path": f"{job_id}/{index}/endpoint.pt"}, "checkpoints": []} for index in range(2)]
            training[job_id] = {"status": "training_completed", "stages": records}
            for index in range(2):
                for update in (index * 100 + 50, index * 100 + 100):
                    for eval_seed in (701, 1701):
                        identity = {"id": f"{job_id}_{index}_{update}_{eval_seed}", "job_id": job_id,
                            "stage_index": index, "checkpoint_update": update, "role": "validation",
                            "seed": eval_seed, "scenario": "stand"}
                        protocol["evaluation_cells"].append(identity)
                        cells[identity["id"]] = {"identity": deepcopy(identity), "status": "completed",
                            "grade": {"passed": True, "score": 1., "reasons": []}}
    return protocol, training, cells


def candidate(rank, name):
    return next(item for item in rank["candidates"] if item["candidate"] == name)


def test_better_middle_checkpoint_never_replaces_final_score_or_representative():
    protocol, training, cells = grid()
    for record in cells.values():
        if record["identity"]["job_id"].startswith("gated"):
            record["grade"]["score"] = .01 if record["identity"]["checkpoint_update"] == 150 else 1.
        else:
            record["grade"]["score"] = .75
    rank = selection._rank(protocol, training, cells)
    gated = candidate(rank, "gated")
    assert gated["eligible"] and gated["score_mean"] == 1.
    assert gated["representative"]["checkpoint_update"] == 200
    assert gated["representative"]["endpoint"]["path"].endswith("/1/endpoint.json")
    assert rank["best_transformer"]["candidate"] == "query"
    assert all(seed["expected_validation_cells"] == 8 for seed in gated["training_seeds"])


def test_unacquired_before_and_postlearn_failures_are_not_called_forgetting():
    protocol, training, cells = grid()
    for record in cells.values():
        if record["identity"]["job_id"].startswith("gated") and record["identity"]["checkpoint_update"] < 200:
            record["grade"].update(passed=False, score=100.)
    # Put later checkpoints first; acquisition must still follow actual time.
    protocol["evaluation_cells"].reverse()
    gated = candidate(selection._rank(protocol, training, cells), "gated")
    assert gated["eligible"]
    for seed in gated["training_seeds"]:
        assert not seed["retention_regressions"]
        curve = seed["learning_observations"]
        assert [row["checkpoint_update"] for row in curve] == [50, 50, 100, 100, 150, 150, 200, 200]
        assert all(row["acquisition_status"] == "not_yet_acquired" for row in curve[:6])
        assert all(row["acquisition_status"] == "acquired" for row in curve[6:])


def test_regression_requires_an_actual_earlier_passed_gate():
    protocol, training, cells = grid()
    for record in cells.values():
        if record["identity"]["job_id"].startswith("gated") and record["identity"]["checkpoint_update"] == 150:
            record["grade"].update(passed=False, score=10.)
    protocol["evaluation_cells"].reverse()
    gated = candidate(selection._rank(protocol, training, cells), "gated")
    assert not gated["eligible"]
    for seed in gated["training_seeds"]:
        assert len(seed["retention_regressions"]) == 2
        assert all(row["acquired_cell"].split("_")[-2] == "50" for row in seed["retention_regressions"])
        assert sum(row["acquisition_status"] == "regressed_after_acquisition"
                   for row in seed["learning_observations"]) == 2


def test_failure_prefix_is_observable_but_missing_final_cannot_win():
    protocol, training, cells = grid()
    for job_id, data in training.items():
        if not job_id.startswith("gated"):
            continue
        data["status"] = "numerical_failure"
        data["stages"][-1].update(endpoint=None, checkpoint=None,
            checkpoints=[{"checkpoint_update": 150, "kind": "intermediate",
                "record": {"path": f"{job_id}/1/checkpoint_00000050.json"},
                "checkpoint": {"path": f"{job_id}/1/checkpoint_00000050.pt"}}])
    for record in cells.values():
        if record["identity"]["job_id"].startswith("gated"):
            record["grade"]["score"] = .001
            if record["identity"]["checkpoint_update"] == 200:
                record.update(status="missing", grade=None)
            routed = selection._observation_record(training[record["identity"]["job_id"]]["stages"][-1],
                record["identity"])
            if record["identity"]["checkpoint_update"] == 150:
                assert routed["endpoint"]["path"].endswith("checkpoint_00000050.json")
            elif record["identity"]["checkpoint_update"] == 200:
                assert routed["endpoint"] is routed["checkpoint"] is None
    rank = selection._rank(protocol, training, cells)
    gated = candidate(rank, "gated")
    assert not gated["eligible"] and gated["score_mean"] is None
    assert "representative" not in gated
    assert rank["best_transformer"]["candidate"] == "query"
    assert all(seed["expected_validation_cells"] == 8 and seed["completed_validation_cells"] == 6
               for seed in gated["training_seeds"])


def test_missing_middle_observation_keeps_the_original_denominator():
    protocol, training, cells = grid()
    missing = next(record for record in cells.values() if record["identity"]["job_id"] == "gated_71"
                   and record["identity"]["checkpoint_update"] == 150)
    missing.update(status="missing", grade=None)
    gated = candidate(selection._rank(protocol, training, cells), "gated")
    assert not gated["eligible"] and gated["score_mean"] == 1.
    seed = next(item for item in gated["training_seeds"] if item["training_seed"] == 71)
    assert seed["expected_validation_cells"] == 8 and seed["completed_validation_cells"] == 7


class PinFixture:
    def __init__(self, record):
        self.record, self.checked_paths, self.received_paths = record, [], []

    def checked(self, receipt):
        self.checked_paths.append(receipt["path"])
        return Path(receipt["path"])

    def receipt(self, path):
        self.received_paths.append(str(path))
        return {"path": str(path)}

    def read(self, path):
        return self.record


@pytest.mark.parametrize("completed", (True, False))
def test_training_checkpoint_ledger_must_equal_independently_verified_records(tmp_path, monkeypatch, completed):
    protocol = {"output_root": str(tmp_path)}
    job, stage = {"id": "job"}, {"index": 0, "checkpoint_updates": [4, 8]}
    verified = [{"checkpoint_update": 4, "kind": "intermediate",
        "record": {"path": str(tmp_path / "cp.json")}, "checkpoint": {"path": str(tmp_path / "cp.pt")}}]
    calls = []

    def verify(actual_protocol, actual_job, actual_stage, receipt, *, completed):
        calls.append((actual_protocol, actual_job, actual_stage, receipt, completed))
        return deepcopy(verified)

    monkeypatch.setattr(campaign, "verified_stage_checkpoints", verify)
    completion = {"path": str(tmp_path / "completion.json")}
    pins = PinFixture({"sidecar": {"path": str(tmp_path / "cp.pt.json")}})
    result = selection._training_checkpoints(protocol, job, stage, {"checkpoints": verified},
        completion, pins, completed=completed)
    assert result == verified and calls[-1][-1] is completed
    assert str(tmp_path / "cp.json") in pins.checked_paths
    assert str(tmp_path / "cp.pt") in pins.checked_paths
    assert str(tmp_path / "cp.pt.json") in pins.received_paths
    assert str(tmp_path / "job/stage_0000/train/metrics.jsonl") in pins.received_paths
    with pytest.raises(ValueError, match="job ledger changes"):
        selection._training_checkpoints(protocol, job, stage, {"checkpoints": []},
            completion, pins, completed=completed)
    substituted = deepcopy(verified)
    substituted[0]["checkpoint"]["path"] = str(tmp_path / "replacement.pt")
    with pytest.raises(ValueError, match="job ledger changes"):
        selection._training_checkpoints(protocol, job, stage, {"checkpoints": substituted},
            completion, pins, completed=completed)


def heldout_fixture(tmp_path, monkeypatch):
    job = {"id": "job", "stages": [{"expected_cumulative_updates": 8}]}
    protocol = {"schema_version": 2, "output_root": str(tmp_path), "jobs": [job], "evaluation_cells": []}
    choice = {"status": "no_eligible", "held_out_cells": {}}
    receipts = {}
    for update in (4, 8):
        receipt = {"path": str(tmp_path / f"checkpoint_{update}.json")}
        receipts[update] = receipt
        for scenario in ("stand", "move"):
            cell = {"id": f"{update}_{scenario}", "job_id": "job", "stage_index": 0,
                "checkpoint_update": update, "role": "held_out", "seed": 2701, "scenario": scenario}
            protocol["evaluation_cells"].append(cell)
            choice["held_out_cells"][cell["id"]] = {"identity": deepcopy(cell), "endpoint": receipt,
                "checkpoint": {"path": str(tmp_path / f"checkpoint_{update}.pt")}}
    monkeypatch.setattr(selection, "_sealed_inputs", lambda receipt: (choice, protocol, {"path": "protocol"}))
    monkeypatch.setattr(campaign, "_checked", lambda receipt: Path(receipt["path"]))
    calls = []

    def verify(actual, actual_job, index, update, receipt):
        calls.append((update, receipt))
        assert receipt == receipts[update]
        return {"checkpoint": {"path": str(tmp_path / f"checkpoint_{update}.pt")}}

    monkeypatch.setattr(campaign, "verify_learning_checkpoint", verify)
    return protocol, receipts, calls


def test_heldout_authorizes_all_checkpoints_only_at_their_exact_frozen_routes(tmp_path, monkeypatch):
    protocol, receipts, calls = heldout_fixture(tmp_path, monkeypatch)
    selection_receipt = {"path": str(tmp_path / "choice.json")}
    for update in (4, 8):
        cells = [cell for cell in protocol["evaluation_cells"] if cell["checkpoint_update"] == update]
        directory = campaign.evaluation_directory(protocol, cells[0])
        assert f"checkpoint_{update:08d}" in str(directory)
        authorization = selection.authorize_heldout_batch(selection_receipt, receipts[update], cells, directory)
        assert authorization["cells"] == cells and authorization["selection_may_change"] is False
        with pytest.raises(ValueError, match="shrinks or changes"):
            selection.authorize_heldout_batch(selection_receipt, receipts[update], cells[:1], directory)
        with pytest.raises(ValueError, match="shrinks or changes"):
            selection.authorize_heldout_batch(selection_receipt, receipts[update], list(reversed(cells)), directory)
        with pytest.raises(ValueError, match="changes its sealed checkpoint"):
            selection.authorize_heldout_batch(selection_receipt, receipts[12 - update], cells, directory)
        with pytest.raises(ValueError, match="output route differs"):
            selection.authorize_heldout_batch(selection_receipt, receipts[update], cells, directory.parent)
    assert [update for update, _ in calls] == [4, 8]


def test_default_endpoint_routes_and_rules_keep_v1_shape():
    protocol = {"output_root": "/tmp/campaign", "schema_version": 1}
    cell = {"job_id": "job", "stage_index": 0, "role": "validation", "seed": 701}
    assert str(selection._directory(protocol, cell)) == "/tmp/campaign/job/stage_0000/evaluation_validation_701"
    assert campaign.evaluation_batch_key(cell) == cell
    record = {"endpoint": {"path": "endpoint.json"}, "checkpoint": {"path": "endpoint.pt"}}
    assert selection._observation_record(record, cell) is record
    assert selection._rule(protocol) == selection._RULE
    assert selection._training_checkpoints(protocol, {}, {}, {}, {}, None, completed=True) is None

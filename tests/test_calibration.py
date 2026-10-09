"""Actual CPU processes and artifacts; synthetic original closure and Isaac factory.

These checks exercise independent calibration, never production queue closure,
real SDK peak storage, hardware timing or an architecture winner.
"""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from transformer_rl import calibration, exposure_campaign as campaign
from transformer_rl import exposure_protocol as protocol, queue_validation
from transformer_rl.episode_outcomes import TRACE_FIELDS, TRACE_METADATA_KEY, trace_metadata
from transformer_rl.frame_config import digest, json_bytes
from test_exposure_protocol import freeze, prepared
from test_exposure_evaluation import physical_protocol
from test_exposure_trace import array, rewrite


REAL_LAUNCH = campaign.launch_owned_worker
BOOTSTRAP = r'''
import json,sys
from pathlib import Path
from copy import deepcopy
from transformer_rl import exposure_protocol as protocol,queue_validation,cli
request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
p=json.loads(Path(request['protocol']['path']).read_bytes())
protocol.predecessors._dependencies=lambda *a: (deepcopy(p['execution']['dependencies']),deepcopy(p['execution']['resource_locks']))
queue_validation.check_dependency=lambda *a,**k: {'status':'completed','controller_live':False,'live_workers':[],'fixture':'synthetic closure only'}
sys.path.insert(0,p['runtime_roots'][0])
import physical_cpu_fixture
cli._factory=lambda reference: physical_cpu_fixture.make_six_motor_env
from transformer_rl.calibration_worker import main
raise SystemExit(main(sys.argv[1:]))
'''


def publish(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def plan_inputs(p, **changes):
    options = {"expected_protocol_sha256": protocol._receipt(p["protocol_path"])["sha256"],
        "output_root": p["tmp"] / "calibration", "training_seed": 71, "updates": 2,
        "evaluation_seed": 92001, "max_owned_bytes": 64 * 1024**2,
        "free_margin_bytes": 1024**2, "interval_s": .02, "timeout_s": 60.}
    options.update(changes)
    return calibration.build_plan(p["protocol_path"], **options)


def plan_file(p, **changes):
    value = plan_inputs(p, **changes)
    path = p["tmp"] / "calibration.plan.json"
    publish(path, value)
    return path, value


def fixture_launch(monkeypatch):
    def launch(command, directory, leases, timeout, publish, **kwargs):
        # Only original queue providers and the factory resolver are synthetic.
        # Actual CLI, runtime profile, Linux parent/flocks, Adam and trace remain.
        command = [sys.executable, "-B", "-c", BOOTSTRAP, *command[4:]]
        return REAL_LAUNCH(command, directory, leases, timeout, publish, **kwargs)
    monkeypatch.setattr(campaign, "launch_owned_worker", launch)


def scheduled_protocol(p):
    value = freeze(p, checkpoint_interval=1, _allow_existing_output=True)
    publish(p["protocol_path"], value)
    p["protocol"] = value
    return value


def completed_private_evaluation(p, monkeypatch):
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert result["status"] == "completed", result["error"]
    evaluation = next(w for w in result["workers"] if w["kind"] == "evaluate")
    outcome = json.loads(Path(evaluation["outcome"]["path"]).read_bytes())
    job = next(j for j in p["protocol"]["jobs"] if j["id"] == evaluation["job_id"])
    return plan, job, outcome


def test_plan_preserves_all_candidates_recipe_and_private_complete_scenarios(physical_protocol):
    p = physical_protocol
    path, plan = plan_file(p)
    parsed, frozen = calibration.validate_plan(protocol._receipt(path))
    assert parsed == plan and frozen == p["protocol"]
    jobs = [j for j in frozen["jobs"] if j["training_seed"] == 71]
    assert plan["job_ids"] == [j["id"] for j in jobs]
    assert plan["coverage"]["all_original_candidates"] is True
    assert plan["coverage"]["worker_namespaces"] == 2 * len(jobs)
    assert plan["coverage"]["fresh_transition_budget"] == sum(2 * 2 * 4 for _ in jobs)
    assert plan["coverage"]["evaluation_policy_samples"] == sum(13 * 2 * 2 for _ in jobs)
    assert not Path(plan["output_root"]).exists()
    assert not plan["formal_architecture_selection"] and not plan["production_storage_authorized"]


@pytest.mark.parametrize("updates,evaluation_seed", [(1, 92002), (2, 92001)])
def test_continuous_checkpoints_supply_one_complete_private_template(physical_protocol, updates, evaluation_seed):
    p = physical_protocol
    legacy = p["protocol"]
    job = next(j for j in legacy["jobs"] if j["training_seed"] == 71)
    assert legacy["schema_version"] == 1
    assert all("checkpoint_update" not in cell for cell in calibration._cells(legacy, job))
    frozen = scheduled_protocol(p)
    original = deepcopy(frozen)
    path, plan = plan_file(p, updates=updates, evaluation_seed=evaluation_seed)
    assert calibration.validate_plan(protocol._receipt(path)) == (plan, frozen)
    assert frozen["schema_version"] == 2
    assert len(frozen["evaluation_cells"]) == 2 * len(legacy["evaluation_cells"])
    jobs = [j for j in frozen["jobs"] if j["training_seed"] == 71]
    for job in jobs:
        assert job["stages"][0]["checkpoint_updates"] == [1, 2]
        cells = calibration._cells(frozen, job)
        assert [cell["scenario"] for cell in cells] == [s["name"] for s in frozen["scenarios"]]
        assert {cell["checkpoint_update"] for cell in cells} == {1}
        assert {cell["seed"] for cell in cells} == {frozen["evaluation"]["validation_seeds"][0]}
        cells[0]["checkpoint_update"] = 999  # The template is a copy, not a protocol rewrite.
    assert plan["updates"] == updates and plan["evaluation_seed"] == evaluation_seed
    assert plan["coverage"]["evaluation_policy_samples"] == len(jobs) * 13 * 2 * 2
    assert frozen == original and not Path(plan["output_root"]).exists()


@pytest.mark.parametrize("change", [
    {"evaluation_seed": 701}, {"evaluation_seed": 71}, {"evaluation_seed": 4101},
    {"updates": 3}, {"updates": True}, {"training_seed": 1},
    {"candidates": ["unknown"]}, {"candidates": ["mlp_h1", "mlp_h1"]},
    {"max_owned_bytes": 0}, {"interval_s": 2.}, {"timeout_s": 2.},
])
def test_invalid_budget_or_seed_refused_before_any_calibration_output(physical_protocol, change):
    p = physical_protocol
    with pytest.raises((ValueError, campaign.CampaignIntegrityError)):
        plan_inputs(p, **change)
    assert not (p["tmp"] / "calibration").exists()


def test_raw_authorization_and_signed_scope_cannot_be_silently_replaced(physical_protocol):
    p = physical_protocol
    path, plan = plan_file(p)
    with pytest.raises(campaign.CampaignIntegrityError, match="external authorization"):
        calibration.run(path, expected_plan_sha256="a" * 64)
    changed = deepcopy(plan)
    changed["scope"] = "replacement claims formal evaluation"
    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
    publish(path, changed)
    with pytest.raises(campaign.CampaignIntegrityError, match="coverage or recipe changed"):
        calibration.validate_plan(protocol._receipt(path))
    assert not Path(plan["output_root"]).exists()


def test_output_protection_and_immutable_external_plan(physical_protocol):
    p = physical_protocol
    for protected in (p["history"] / "inside", p["sdk"] / "inside", p["output"] / "inside"):
        with pytest.raises(campaign.CampaignIntegrityError, match="protected"):
            plan_inputs(p, output_root=protected)
    path, plan = plan_file(p, candidates=["gated_h4"])
    Path(plan["output_root"]).mkdir()
    assert calibration.validate_plan(protocol._receipt(path))[0] == plan
    with pytest.raises(campaign.CampaignIntegrityError, match="cannot be reused"):
        calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])


def test_actual_original_flock_conflict_starts_no_calibration_worker(physical_protocol, monkeypatch):
    p = physical_protocol
    value = deepcopy(p["protocol"])
    value["execution"].update(max_wait_seconds=.05, poll_seconds=.01)
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    publish(p["protocol_path"], value)
    path, plan = plan_file(p, candidates=["gated_h4"])
    monkeypatch.setattr(campaign, "launch_owned_worker", lambda *a, **k: pytest.fail("launched under an occupied original lock"))
    with p["shared"].open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert result["status"] == "failed" and result["workers"] == []
    assert "wait expired" in result["error"]["message"]
    assert not list(Path(plan["output_root"]).glob("job_*"))


@pytest.mark.parametrize("continuous_checkpoints", [False, True])
def test_actual_complete_cpu_calibration_all_architectures_and_private_rows(physical_protocol, monkeypatch,
                                                                          continuous_checkpoints):
    p = physical_protocol
    if continuous_checkpoints:
        scheduled_protocol(p)
    original = deepcopy(p["protocol"])
    path, plan = plan_file(p)
    fixture_launch(monkeypatch)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert result["status"] == "completed", json.dumps(result.get("error"))
    assert len(result["workers"]) == plan["coverage"]["worker_namespaces"]
    assert set(w["candidate"] for w in result["workers"]) == set(plan["candidates"])
    assert {m["kind"] for m in result["measurements"]} == {"runtime_cache", "checkpoint", "metric", "trace"}
    for worker in result["workers"]:
        request = json.loads(Path(worker["request"]["path"]).read_bytes())
        assert request["kind"] == worker["kind"]
        if worker["kind"] == "train":
            assert worker["actual_counters"]["successful_updates"] == 2
            assert worker["actual_counters"]["actual_collected_transitions"] == 16
        else:
            outcome = json.loads(Path(worker["outcome"]["path"]).read_bytes())
            report = json.loads(Path(outcome["report"]["path"]).read_bytes())
            assert report["seed"] == 92001 and report["transitions"] == 52 and report["checkpoint_update"] == 2
            assert report["trace"]["checkpoint_update"] == 2
            assert report["environment_provenance"]["evaluation_groups"] == ["normal", "new_skill", "normal", "new_skill"]
            assert set(report["groups"]) == {"normal", "new_skill"}
            assert all(group["num_envs"] == 2 for group in report["groups"].values())
            outcomes = report["episode_outcomes"]
            assert outcomes["available"] and outcomes["requested_episodes"] == outcomes["completed_episodes"] == 4
            assert outcomes["survival"]["healthy_full_horizon_episodes"] == 2
            assert [row["episode_horizon_ticks"] for row in outcomes["first_episode_outcomes"]] == [4, 6, 4, 6]
            for name, horizon in (("normal", 4), ("new_skill", 6)):
                group = report["groups"][name]["episode_outcomes"]
                assert group["requested_episodes"] == 2 and group["available"]
                assert [row["episode_horizon_ticks"] for row in group["first_episode_outcomes"]] == [horizon] * 2
            measurement = next(m for m in result["measurements"] if m["kind"] == "trace"
                               and m["receipt"] == outcome["trace"])
            assert measurement["trace_validation"]["episode_outcome_contract_checked"] is True
            if continuous_checkpoints:
                job = next(j for j in p["protocol"]["jobs"] if j["id"] == worker["job_id"])
                assert {cell["checkpoint_update"] for cell in calibration._cells(p["protocol"], job)} == {1}
            assert "grade" not in report and "selection" not in report
    root = Path(plan["output_root"])
    completion = json.loads((root / "completion.json").read_bytes())
    assert completion["terminal_storage_guard_after_publication"] is True
    assert result["observed_helpers"]["all_tracked_terminal"] is True
    assert result["sampled_storage"]["closed"] is True
    assert result["sampled_storage"]["sample_count"] > 1
    assert result["sampled_storage"]["peaks"]["categories"]["checkpoint"]["logical_bytes"] > 0
    assert result["sampled_storage"]["peaks"]["categories"]["trace"]["logical_bytes"] > 0
    assert not completion["production_storage_authorized"] and not completion["hardware_verified"]
    assert p["protocol"] == original


@pytest.mark.parametrize("field", ["seed", "checkpoint_sha256", "num_envs"])
def test_private_report_mismatch_is_not_accepted_as_a_storage_measurement(physical_protocol, monkeypatch, field):
    p = physical_protocol
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert result["status"] == "completed", result["error"]
    evaluation = next(w for w in result["workers"] if w["kind"] == "evaluate")
    outcome = json.loads(Path(evaluation["outcome"]["path"]).read_bytes())
    report_path = Path(outcome["report"]["path"])
    report = json.loads(report_path.read_bytes())
    report[field] = "a" * 64 if field == "checkpoint_sha256" else report[field] + 1
    publish(report_path, report)
    outcome["report"] = protocol._receipt(report_path)
    job = next(j for j in p["protocol"]["jobs"] if j["id"] == evaluation["job_id"])
    with pytest.raises(campaign.CampaignIntegrityError, match="private calibration report changes"):
        calibration._verify_private_evaluation(plan, p["protocol"], job, outcome["checkpoint"], outcome)


def test_resealed_private_episode_outcomes_and_row_identity_must_match_trace(physical_protocol, monkeypatch):
    p = physical_protocol
    scheduled_protocol(p)
    plan, job, outcome = completed_private_evaluation(p, monkeypatch)
    report_path = Path(outcome["report"]["path"])
    original = json.loads(report_path.read_bytes())
    # Each mutation is resealed, so rejection must come from semantic replay or
    # immutable physical identity, not from the external artifact checksum.
    for group, mutation, reason in (
        (None, "healthy_count", "aggregate report differs"),
        (None, "missing", "aggregate report differs"),
        (None, "first_horizon", "aggregate report differs"),
        ("new_skill", "healthy_count", "scenario metrics or replicas differ"),
        ("normal", "missing", "scenario metrics or replicas differ"),
        ("new_skill", "first_horizon", "scenario metrics or replicas differ"),
        ("normal", "checkpoint_update", "scenario checkpoint, seed, clock or row identity differs"),
        ("normal", "seed", "scenario checkpoint, seed, clock or row identity differs"),
        (None, "row_count", "report changes declared physical rows"),
        (None, "row_type", "report changes declared physical rows"),
        (None, "contract_sha256", "effective contract or timing provenance differs"),
    ):
        report = deepcopy(original)
        target = report if group is None else report["groups"][group]
        if mutation == "healthy_count":
            target["episode_outcomes"]["survival"]["healthy_full_horizon_episodes"] += 1
        elif mutation == "missing":
            del target["episode_outcomes"]
        elif mutation == "first_horizon":
            target["episode_outcomes"]["first_episode_outcomes"][0]["episode_horizon_ticks"] -= 1
        elif mutation in ("checkpoint_update", "seed"):
            target[mutation] += 1
        elif mutation == "row_count":
            target["environment_provenance"]["evaluation_groups"][0] = "new_skill"
        elif mutation == "row_type":
            target["environment_provenance"]["evaluation_groups"][0] = ["normal"]
        else:
            target["environment_provenance"][mutation] = "a" * 64
        publish(report_path, report)
        changed = {**outcome, "report": protocol._receipt(report_path)}
        with pytest.raises(campaign.CampaignIntegrityError, match=reason):
            calibration._verify_private_evaluation(plan, p["protocol"], job, outcome["checkpoint"], changed)


def test_resealed_private_trace_requires_explicit_immutable_episode_evidence(physical_protocol, monkeypatch):
    p = physical_protocol
    scheduled_protocol(p)
    plan, job, outcome = completed_private_evaluation(p, monkeypatch)
    report_path, trace_path = Path(outcome["report"]["path"]), Path(outcome["trace"]["path"])
    original = json.loads(report_path.read_bytes())
    original_trace = trace_path.read_bytes()
    fixture = {"path": trace_path, "trace": original["trace"]}
    for mutation, reason in (
        ("short_horizon", "episode_horizon_ticks differs from authorized case contract"),
        ("false_task", "survival_applicable differs from authorized case contract"),
        ("missing_horizon", "complete trace field coverage differs"),
        ("missing_height", "complete trace field coverage differs"),
        ("missing_protocol", "missing declared episode outcome trace protocol"),
        ("unavailable", "explicit available episode outcome contract required"),
    ):
        trace_path.write_bytes(original_trace)
        report = deepcopy(original)
        if mutation in ("short_horizon", "false_task"):
            name = "outcome_episode_horizon_ticks" if mutation == "short_horizon" else "outcome_survival_applicable"
            values = array(fixture, name).copy()
            if mutation == "short_horizon":
                values[:, 1] -= 1
            else:
                values[:, 1] = ~values[:, 1]
            changes = {name + ".npy": values}
        elif mutation.startswith("missing_") and mutation != "missing_protocol":
            name = "outcome_episode_horizon_ticks" if mutation == "missing_horizon" else "outcome_height"
            changes = {name + ".npy": None}
        else:
            metadata = json.loads(array(fixture, "metadata_json").item())
            if mutation == "missing_protocol":
                del metadata[TRACE_METADATA_KEY]
                del report["trace"][TRACE_METADATA_KEY]
            else:
                metadata[TRACE_METADATA_KEY] = trace_metadata(False)
                report["trace"][TRACE_METADATA_KEY] = trace_metadata(False)
            changes = {name + ".npy": None for name in TRACE_FIELDS}
            changes["metadata_json.npy"] = np.array(json.dumps(metadata))
        fixture["trace"] = report["trace"]
        report["trace"] = rewrite(fixture, changes)
        publish(report_path, report)
        changed = {**outcome, "report": protocol._receipt(report_path), "trace": protocol._receipt(trace_path)}
        with pytest.raises(ValueError, match=reason):
            calibration._verify_private_evaluation(plan, p["protocol"], job, outcome["checkpoint"], changed)


def test_lease_exit_error_after_complete_workers_cannot_claim_completion(physical_protocol, monkeypatch):
    p = physical_protocol
    path, _ = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    real_lease = campaign.resource_lease

    @contextmanager
    def lease(*args):
        with real_lease(*args) as descriptors:
            yield descriptors
            raise RuntimeError("injected lease exit failure")

    monkeypatch.setattr(campaign, "resource_lease", lease)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert len(result["workers"]) == 2
    assert result["status"] == "failed" and result["error"]["message"] == "injected lease exit failure"
    assert json.loads(Path(result["completion"]["path"]).read_bytes())["status"] == "failed"


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_helper_observation_failure_keeps_original_flocks_until_actual_exit(physical_protocol, monkeypatch, failure_type):
    p = physical_protocol
    path, _ = plan_file(p, candidates=["gated_h4"])
    real_inventory = calibration.helper_inventory
    process = None
    calls = 0

    def assert_original_locks_busy():
        for lock in (p["shared"], p["study_lock"]):
            with lock.open("rb") as descriptor:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def launch(command, directory, *args, **kwargs):
        nonlocal process
        process = subprocess.Popen([sys.executable, "-B", "-c", "import sys;sys.stdin.buffer.read(1)"],
                                   cwd=directory, stdin=subprocess.PIPE)
        return {"returncode": 1, "timed_out": False}

    def inventory(root, handles=()):
        nonlocal calls
        if process is not None:
            calls += 1
            assert_original_locks_busy()
            if calls == 2:
                raise failure_type("injected helper observation interruption")
            if calls >= 3 and process.poll() is None:
                process.stdin.write(b"x")
                process.stdin.flush()
                assert process.wait(timeout=5) == 0
        return real_inventory(root, handles)

    monkeypatch.setattr(campaign, "launch_owned_worker", launch)
    monkeypatch.setattr(calibration, "helper_inventory", inventory)
    try:
        result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
        assert calls >= 3 and process.returncode == 0
        assert result["status"] == "failed"
        assert result["helper_closure_failures"]["count"] == 1
        assert result["helper_closure_failures"]["first"]["type"] == failure_type.__name__
        assert result["observed_helpers"]["all_tracked_terminal"] is True
    finally:
        if process is not None:
            if process.poll() is None:
                process.stdin.write(b"x")
                process.stdin.flush()
                process.wait(timeout=5)
            process.stdin.close()


def test_progress_publication_failure_preserves_failed_terminal_receipt(physical_protocol, monkeypatch):
    p = physical_protocol
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    original = campaign._publish

    def fail_result_progress(path, value):
        if path.name == "progress.json" and "result" in value:
            raise OSError("injected result progress I/O failure")
        return original(path, value)

    monkeypatch.setattr(campaign, "_publish", fail_result_progress)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert len(result["workers"]) == 2 and result["sampled_storage"]["closed"]
    assert result["status"] == "failed" and result["error"]["type"] == "OSError"
    completed = json.loads((Path(plan["output_root"]) / "completion.json").read_bytes())
    assert completed["status"] == "failed" and completed["error"]["message"] == "injected result progress I/O failure"


def test_protocol_change_during_helper_closure_is_terminal_failure(physical_protocol, monkeypatch):
    p = physical_protocol
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    original = calibration.helper_inventory
    changed = False

    def change_after_last_worker(root, handles=()):
        nonlocal changed
        progress = root / "progress.json"
        if not changed and progress.exists() and json.loads(progress.read_bytes()).get("completed_workers") == 2:
            with p["protocol_path"].open("ab") as stream:
                stream.write(b"\n")
            changed = True
        return original(root, handles)

    monkeypatch.setattr(calibration, "helper_inventory", change_after_last_worker)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert changed and len(result["workers"]) == 2
    assert result["status"] == "failed" and result["completion"] is not None
    completed = json.loads((Path(plan["output_root"]) / "completion.json").read_bytes())
    assert completed["status"] == "failed" and not completed["terminal_storage_guard_after_publication"]


def test_final_guard_and_persistent_failure_io_cannot_leave_success_receipt(physical_protocol, monkeypatch):
    p = physical_protocol
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    original_guard = calibration.guard_owned_storage
    original_publish = campaign._publish

    def fail_after_completion(root, *args):
        if (Path(root) / "completion.json").exists():
            raise OSError("injected final guard failure")
        return original_guard(root, *args)

    def fail_failure_completion(path, value):
        if path.name == "completion.json":
            raise OSError("injected persistent failure publication error")
        return original_publish(path, value)

    monkeypatch.setattr(calibration, "guard_owned_storage", fail_after_completion)
    monkeypatch.setattr(campaign, "_publish", fail_failure_completion)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert len(result["workers"]) == 2 and result["status"] == "failed"
    assert result["completion"] is None
    assert result["terminal_publication_error"]["message"] == "injected persistent failure publication error"
    assert not (Path(plan["output_root"]) / "completion.json").exists()


@pytest.mark.parametrize("field,value", [("refund", True), ("charged_updates", 1)])
def test_actual_training_reservation_cannot_be_refunded_or_undercharged(physical_protocol, monkeypatch, field, value):
    p = physical_protocol
    path, plan = plan_file(p, candidates=["gated_h4"])
    fixture_launch(monkeypatch)
    result = calibration.run(path, expected_plan_sha256=protocol._receipt(path)["sha256"])
    assert result["status"] == "completed", result["error"]
    train = next(w for w in result["workers"] if w["kind"] == "train")
    directory = Path(train["request"]["path"]).parent
    reservation_path = directory / "train" / "reservation.json"
    reservation = json.loads(reservation_path.read_bytes())
    reservation[field] = value
    publish(reservation_path, reservation)
    job = next(j for j in p["protocol"]["jobs"] if j["id"] == train["job_id"])
    with pytest.raises(ValueError, match="reservation or no-retry"):
        calibration._training_checkpoint(directory, protocol._receipt(path), p["protocol"], job, plan)

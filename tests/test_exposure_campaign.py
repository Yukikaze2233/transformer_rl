"""Real OS/FD/CPU learning checks; original queue closure is a fixture provider.

These tests never authorize the production predecessor queues or a simulator.
History/config/source/runtime/snapshot bytes, Linux pidfds, inherited flocks,
independent processes, real Adam updates and learning-state chains are actual.
"""
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from transformer_rl import exposure_campaign as campaign
from transformer_rl import exposure_protocol as protocol
from transformer_rl import exposure_training as training
from transformer_rl import history_study
from transformer_rl import queue_validation
from transformer_rl.control_trace import ControlTrace
from transformer_rl.frame_config import digest, json_bytes
from transformer_rl.frame_training import FrameActorCritic
from test_exposure_protocol import prepared, freeze, write_json
from test_exposure_training import make_env


REAL_RUNTIME = protocol.runtime_identity
REAL_LAUNCH = campaign.launch_owned_worker
FIXTURES = Path(__file__).parent / "fixtures"


BOOTSTRAP = r'''
import json,sys
from copy import deepcopy
from pathlib import Path
from transformer_rl import exposure_protocol as protocol, queue_validation
request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
p=json.loads(Path(request['protocol']['path']).read_bytes())
protocol.predecessors._dependencies=lambda *a: (deepcopy(p['execution']['dependencies']),deepcopy(p['execution']['resource_locks']))
queue_validation.check_dependency=lambda *a,**k: {'status':'completed','controller_live':False,'live_workers':[],'fixture':'synthetic closure only'}
sys.path.insert(0,p['runtime_roots'][0])
from transformer_rl.exposure_process import main
raise SystemExit(main(sys.argv[1:]))
'''


def canonical(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def closed(*args, **kwargs):
    return {"status": "completed", "controller_live": False, "live_workers": [],
            "fixture": "synthetic original closure provider; no production proof"}


@pytest.fixture
def campaign_inputs(prepared, monkeypatch):
    """Prepare all four permitted actor modes and all mandatory seed/cell IDs."""
    p = prepared
    source = (FIXTURES / "packed_env.py").read_text()
    source += '''
original_make_env = make_env
def make_env(model_config, environment_config, device):
    env = original_make_env(model_config, environment_config, device)
    mode = environment_config.get('fixture_failure')
    if mode and model_config.policy.architecture == 'mlp' and torch.initial_seed() == 71:
        def failure(action):
            if mode == 'numerical': raise FloatingPointError('actual CPU fixture nonfinite step')
            raise RuntimeError('nan in error text does not make this numerical')
        env.step = failure
    return env
'''
    (p["sdk"] / "campaign_cpu_fixture.py").write_text(source)
    base = replace(p["base"], environment={**p["base"].environment, "control": p["base"].control})
    write_json(p["inputs"] / "base.json", base.to_dict())
    spec = deepcopy(p["spec"])
    spec["environment_factory"] = "campaign_cpu_fixture:make_env"
    spec["variants"][-1]["policy"].update(residual_type="gated", readout_type="last")
    spec["variants"].append({"name": "transformer_query", "policy": {
        "architecture": "transformer", "residual_type": "gated", "readout_type": "query"}})
    for stage in spec["stages"]:
        stage["updates"] = 1
    spec["training"]["rollout_steps"] = 2
    write_json(p["inputs"] / "study.json", spec)
    p["history"] = p["tmp"] / "campaign_history"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json",
        p["history"], history_lengths=[1, 3], position_reference="current")
    monkeypatch.setattr(protocol, "runtime_identity", REAL_RUNTIME)
    monkeypatch.setattr(queue_validation, "check_dependency", closed)
    p["spec"], p["base"] = spec, base
    value = freeze(p)
    protocol_path = p["tmp"] / "authorized_protocol.json"
    canonical(protocol_path, value)
    p["protocol"], p["protocol_path"] = value, protocol_path
    measurement_root = p["tmp"] / "storage_measurements"
    measurement_root.mkdir()
    cp = measurement_root / "initial_state.pt"
    torch.save(FrameActorCritic(base.model).state_dict(), cp)
    traces = measurement_root / "actual_synthetic_trace.npz"
    trace = ControlTrace(traces, steps=2, num_envs=3, replicas=3, groups=None,
                         metadata={"fixture": "CPU tensor trace storage sample"})
    try:
        for step in range(2):
            trace.add({"time_s": torch.full((3,), (step + 1) * .01, dtype=torch.float64),
                       "issued_action": torch.full((3, 2), .1),
                       "fixture_payload": torch.zeros((3, 512))},
                      torch.zeros(3, dtype=torch.bool))
        trace.publish()
    finally:
        trace.close()
    metrics = measurement_root / "actual_synthetic_metric.jsonl"
    metrics.write_text(json.dumps({"fixture": "CPU tensor storage sample", "samples": 6}) + "\n")
    cache_root = measurement_root / "actual_runtime_cache"
    cache_root.mkdir()
    (cache_root / "fixture_cache.bin").write_bytes((p["sdk"] / "campaign_cpu_fixture.py").read_bytes())
    cache = measurement_root / "actual_runtime_cache_inventory.json"
    cache_files = [protocol._receipt(cache_root / "fixture_cache.bin")]
    canonical(cache, {"format": "transformer_rl.exposure_runtime_cache_measurement", "schema_version": 1,
        "root": str(cache_root), "files": cache_files, "total_bytes": sum(f["bytes"] for f in cache_files)})
    p["storage"] = {"format": "transformer_rl.exposure_storage_contract", "schema_version": 1,
        "protocol_raw_sha256": protocol._receipt(protocol_path)["sha256"], "source": value["source"],
        "caps": {"checkpoint_bytes": 16 * 1024**2, "trace_bytes_per_policy_sample": 4096,
                 "metric_bytes_per_update": 64 * 1024, "inflight_bytes": 8 * 1024**2,
                 "runtime_cache_bytes": 4 * 1024**2, "free_margin_bytes": 4 * 1024**2},
        "measurements": [{"kind": kind, "receipt": protocol._receipt(path),
                          "units": 6 if kind == "trace" else 1}
                         for kind, path in (("checkpoint", cp), ("trace", traces),
                                            ("metric", metrics), ("runtime_cache", cache))]}
    p["storage_path"] = p["tmp"] / "storage_contract.json"
    canonical(p["storage_path"], p["storage"])

    def actual_cpu_launch(command, directory, leases, timeout, publish, **kwargs):
        # Only the predecessor closure provider is replaced inside this test
        # child. The actual worker CLI and every other validator are executed.
        command = [sys.executable, "-B", "-c", BOOTSTRAP, *command[4:]]
        return REAL_LAUNCH(command, directory, leases, timeout, publish, **kwargs)
    monkeypatch.setattr(campaign, "launch_owned_worker", actual_cpu_launch)
    return p


def run(p, **kwargs):
    return campaign.run_protocol(p["protocol_path"],
        expected_protocol_sha256=protocol._receipt(p["protocol_path"])["sha256"],
        storage_contract_path=p["storage_path"],
        expected_storage_sha256=protocol._receipt(p["storage_path"])["sha256"], **kwargs)


def test_raw_external_authorization_cannot_be_replaced_by_self_signature(campaign_inputs):
    p = campaign_inputs
    with pytest.raises(campaign.CampaignIntegrityError, match="external authorization"):
        campaign.read_authorized_protocol(p["protocol_path"], "a" * 64)
    assert not p["output"].exists()


def test_storage_actual_measurements_and_remaining_complete_matrix(campaign_inputs):
    p = campaign_inputs
    raw = protocol._receipt(p["protocol_path"])
    assert campaign._validate_storage(p["storage"], p["protocol"], raw) == p["storage"]
    full = campaign.remaining_storage_bytes(p["protocol"], p["storage"])
    job = p["protocol"]["jobs"][0]
    cell = p["protocol"]["evaluation_cells"][0]
    remaining = campaign.remaining_storage_bytes(p["protocol"], p["storage"],
        [(job["id"], 0)], [cell["id"]])
    caps = p["storage"]["caps"]
    assert full - remaining == (caps["checkpoint_bytes"]
        + job["stages"][0]["updates"] * caps["metric_bytes_per_update"]
        + cell["expected_policy_samples"] * caps["trace_bytes_per_policy_sample"]
        + caps["inflight_bytes"] + caps["runtime_cache_bytes"])
    bad = deepcopy(p["storage"])
    bad["caps"]["checkpoint_bytes"] = 1
    with pytest.raises(campaign.CampaignIntegrityError, match="actual measured"):
        campaign._validate_storage(bad, p["protocol"], raw)
    measurement = Path(p["storage"]["measurements"][1]["receipt"]["path"])
    measurement.write_bytes(b"changed raw trace")
    with pytest.raises(campaign.CampaignIntegrityError, match="receipt bytes changed"):
        run(p)
    assert not p["output"].exists()


def test_trace_storage_cap_uses_actual_uncompressed_payload_not_archive_size(campaign_inputs):
    p = campaign_inputs
    measured = next(item for item in p["storage"]["measurements"] if item["kind"] == "trace")
    units, compressed = measured["units"], measured["receipt"]["bytes"]
    raw = campaign.measured_trace_bytes(measured["receipt"]["path"], units)
    bad = deepcopy(p["storage"])
    bad["caps"]["trace_bytes_per_policy_sample"] = (compressed + units - 1) // units
    assert compressed <= bad["caps"]["trace_bytes_per_policy_sample"] * units < raw
    with pytest.raises(campaign.CampaignIntegrityError, match="below an actual measured artifact"):
        campaign._validate_storage(bad, p["protocol"], protocol._receipt(p["protocol_path"]))


def test_trace_measurement_units_must_match_actual_steps_times_rows(campaign_inputs):
    p = campaign_inputs
    bad = deepcopy(p["storage"])
    measured = next(item for item in bad["measurements"] if item["kind"] == "trace")
    measured["units"] += 1
    with pytest.raises(campaign.CampaignIntegrityError, match="policy-sample units differ"):
        campaign._validate_storage(bad, p["protocol"], protocol._receipt(p["protocol_path"]))


def test_full_remaining_budget_counts_each_retained_stage_and_evaluation_namespace():
    # Two stages and four independent role/seed batches. Each batch has two
    # cases; closing one case cannot release its still-required namespace.
    value = {"jobs": [{"id": "job_a", "stages": [{"index": 0, "updates": 3},
                                                      {"index": 1, "updates": 5}]}],
        "evaluation_cells": [{"id": f"cell_{role}_{seed}_{case}", "job_id": "job_a",
            "stage_index": 0, "role": role, "seed": seed, "expected_policy_samples": samples}
            for role in ("validation", "held_out") for seed in (701, 1701)
            for case, samples in (("case_a", 10), ("case_b", 20))]}
    caps = {"checkpoint_bytes": 100, "trace_bytes_per_policy_sample": 2,
        "metric_bytes_per_update": 7, "inflight_bytes": 11, "runtime_cache_bytes": 13,
        "free_margin_bytes": 17}
    contract = {"caps": caps}
    expected = 2 * 100 + 8 * 7 + 120 * 2 + 30 * 2 + 100 + 7 * (11 + 13) + 17
    initial = campaign.remaining_storage_bytes(value, contract)
    assert initial == expected
    after_stage = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)])
    assert initial - after_stage == 100 + 3 * 7 + 11 + 13
    first = value["evaluation_cells"][0]["id"]
    second = value["evaluation_cells"][1]["id"]
    after_case = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)], [first])
    assert after_stage - after_case == 10 * 2
    after_batch = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)], [first, second])
    assert after_case - after_batch == 20 * 2 + 11 + 13
    closed = [cell["id"] for cell in value["evaluation_cells"]]
    final = campaign.remaining_storage_bytes(value, contract, [("job_a", 0), ("job_a", 1)], closed)
    # All historic namespaces remain on disk and reduce statvfs available;
    # zero future workers means no repeated cache/inflight reservation.
    assert final == 100 + 17


def test_publication_peak_tracks_largest_actual_remaining_batch_not_worker_count():
    value = {"jobs": [], "evaluation_cells": [
        {"id": "large", "job_id": "job_a", "stage_index": 0, "role": "validation",
         "seed": 701, "expected_policy_samples": 100},
        {"id": "small", "job_id": "job_a", "stage_index": 0, "role": "held_out",
         "seed": 1701, "expected_policy_samples": 25}]}
    caps = {"checkpoint_bytes": 100, "trace_bytes_per_policy_sample": 2,
        "metric_bytes_per_update": 7, "inflight_bytes": 11, "runtime_cache_bytes": 13,
        "free_margin_bytes": 17}
    contract = {"caps": caps}
    initial = campaign.remaining_storage_bytes(value, contract)
    remaining = campaign.remaining_storage_bytes(value, contract, closed_cells=["large"])
    # Release only this batch's retained archive, namespace and change in the
    # one concurrent staging/archive peak. There is no per-worker peak copy.
    assert initial - remaining == 100 * 2 + 11 + 13 + (100 - 25) * 2


@pytest.fixture
def worker_storage(tmp_path, monkeypatch):
    directory = tmp_path / "worker"
    directory.mkdir()
    contract = {"caps": {"checkpoint_bytes": 128, "metric_bytes_per_update": 64,
        "trace_bytes_per_policy_sample": 256, "inflight_bytes": 64,
        "runtime_cache_bytes": 32, "free_margin_bytes": 16}}
    available = {"bytes": 10_000}
    monkeypatch.setattr(campaign.os, "statvfs", lambda path: SimpleNamespace(
        f_bavail=available["bytes"], f_frsize=1))
    return directory, contract, available


def test_exact_boundary_reserve_survives_current_runtime_cap_without_future_credit(worker_storage):
    directory, contract, available = worker_storage
    contract["caps"].update(runtime_cache_bytes=4096, inflight_bytes=4096)
    value = {"jobs": [{"id": "job_a", "stages": [{"index": 0, "updates": 1},
                                                      {"index": 1, "updates": 1}]}],
             "evaluation_cells": []}
    boundary = campaign.remaining_storage_bytes(value, contract)
    namespace_cap = contract["caps"]["runtime_cache_bytes"] + contract["caps"]["inflight_bytes"]
    active = boundary - namespace_cap
    available["bytes"] = boundary
    campaign.check_disk(directory, boundary)
    cache = directory / "runtime"
    cache.mkdir()
    cache_file, log_file = cache / "current_cache.bin", directory / "stdout.txt"
    cache_file.write_bytes(b"r" * 4096)
    log_file.write_bytes(b"s" * 4096)
    # The disk availability observation is controlled, while both category
    # bytes and allocated blocks come from the real written fixture files.
    written = sum(path.stat().st_blocks * 512 for path in (cache_file, log_file))
    assert written == namespace_cap
    available["bytes"] -= written
    checked = campaign.worker_storage_guard(value, contract, directory, boundary, active_worker=True)
    assert checked["required_bytes"] == active == available["bytes"]
    assert checked["active_runtime_headroom_bytes"] == namespace_cap
    assert checked["owned_bytes"]["cache"] == 4096 and checked["owned_bytes"]["other"] == 4096
    assert checked["credited_declared_output_bytes"] == checked["undeclared_output_budget_credit"] == 0
    after_stage = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)])
    assert boundary - after_stage == (contract["caps"]["checkpoint_bytes"]
        + contract["caps"]["metric_bytes_per_update"] + namespace_cap)
    campaign.check_disk(directory, after_stage)
    assert cache_file.is_file() and log_file.is_file()


@pytest.mark.parametrize("flag", [1, "true", None])
def test_active_namespace_headroom_requires_explicit_boolean(worker_storage, flag):
    directory, contract, _ = worker_storage
    with pytest.raises(campaign.CampaignIntegrityError, match="flag must be boolean"):
        campaign.worker_storage_guard({}, contract, directory, 1000, active_worker=flag)


def test_active_namespace_headroom_cannot_spend_the_entire_reservation(worker_storage):
    directory, contract, _ = worker_storage
    namespace_cap = contract["caps"]["inflight_bytes"] + contract["caps"]["runtime_cache_bytes"]
    with pytest.raises(campaign.CampaignIntegrityError, match="headroom must fit"):
        campaign.worker_storage_guard({}, contract, directory, namespace_cap, active_worker=True)


def test_stdout_cannot_spend_later_jobs_reserved_storage(worker_storage):
    directory, contract, available = worker_storage
    (directory / "stdout.txt").write_bytes(b"actual stdout bytes" * 3)
    # The old all-files credit would reduce 1000 below the available 990.
    available["bytes"] = 990
    with pytest.raises(campaign.CampaignDiskError, match="reserve 1000 exceeds available 990"):
        campaign.worker_storage_guard({}, contract, directory, 1000)
    available["bytes"] = 1000
    result = campaign.worker_storage_guard({}, contract, directory, 1000)
    assert result["required_bytes"] == 1000
    assert result["credited_declared_output_bytes"] == 0
    assert result["undeclared_output_budget_credit"] == 0


def test_only_declared_own_checkpoint_and_metrics_spend_reserved_storage(worker_storage):
    directory, contract, available = worker_storage
    checkpoint = directory / "endpoint.pt"
    metrics = directory / "metrics.jsonl"
    checkpoint.write_bytes(b"c" * 40)
    metrics.write_bytes(b"m" * 10)
    (directory / "stdout.txt").write_bytes(b"s" * 30)
    cache = directory / "empty_python_cache"
    cache.mkdir()
    (cache / "sdk_cache.bin").write_bytes(b"r" * 16)
    sibling = directory.parent / "other_job"
    sibling.mkdir()
    (sibling / "endpoint.pt").write_bytes(b"other job" * 1000)
    available["bytes"] = 950
    result = campaign.worker_storage_guard({}, contract, directory, 1000,
        checkpoint_paths=(checkpoint,), checkpoint_limit=128,
        metric_paths=(metrics,), metric_limit=64)
    assert result["required_bytes"] == 950 and result["credited_declared_output_bytes"] == 50
    assert result["owned_bytes"] == {"checkpoint": 40, "metric": 10,
        "trace_maps": 0, "trace_archive": 0, "other": 30, "cache": 16}


def test_multiple_checkpoints_cannot_use_future_stages_reserved_storage(worker_storage):
    directory, contract, _ = worker_storage
    checkpoint, extra = directory / "endpoint.pt", directory / "extra.pt"
    checkpoint.write_bytes(b"c" * 20)
    extra.write_bytes(b"e" * 20)
    with pytest.raises(campaign.CampaignIntegrityError, match="undeclared checkpoint"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(checkpoint,), checkpoint_limit=128)
    with pytest.raises(campaign.CampaignIntegrityError, match="exact worker output paths"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(checkpoint, extra), checkpoint_limit=128)


def test_continuous_checkpoint_reserve_counts_one_training_namespace_and_distinct_evaluators():
    value = {"jobs": [{"id": "job_a", "stages": [{"index": 0, "updates": 12,
        "expected_cumulative_updates": 12, "checkpoint_updates": [4, 8, 12]}]}],
        "evaluation_cells": [{"id": f"cell_{update}_{case}", "job_id": "job_a", "stage_index": 0,
            "checkpoint_update": update, "role": "validation", "seed": 701,
            "expected_policy_samples": samples}
            for update in (4, 8, 12) for case, samples in (("a", 10), ("b", 20))]}
    caps = {"checkpoint_bytes": 100, "trace_bytes_per_policy_sample": 2,
        "metric_bytes_per_update": 7, "inflight_bytes": 11, "runtime_cache_bytes": 13,
        "free_margin_bytes": 17}
    contract = {"caps": caps}
    # Three retained checkpoints, one continuous training process, three
    # independent checkpoint evaluation workers and one active headroom.
    initial = campaign.remaining_storage_bytes(value, contract)
    assert initial == 3 * 100 + 12 * 7 + 90 * 2 + 30 * 2 + 100 + 5 * 24 + 17
    after_stage = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)])
    assert initial - after_stage == 3 * 100 + 12 * 7 + 24
    after_one_case = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)], ["cell_4_a"])
    assert after_stage - after_one_case == 10 * 2
    after_first_batch = campaign.remaining_storage_bytes(value, contract, [("job_a", 0)],
        ["cell_4_a", "cell_4_b"])
    assert after_one_case - after_first_batch == 20 * 2 + 24
    closed = [cell["id"] for cell in value["evaluation_cells"]]
    assert campaign.remaining_storage_bytes(value, contract, [("job_a", 0)], closed) == 100 + 17


def test_explicit_checkpoint_caps_credit_each_immutable_publication_once(worker_storage):
    directory, contract, available = worker_storage
    first, final = directory / "update_000004.pt", directory / "endpoint.pt"
    temporary = directory / ".update_000004.pt.actual-publication.tmp"
    temporary.write_bytes(b"a" * 70)
    os.link(temporary, first)
    final.write_bytes(b"b" * 40)
    (directory / ".endpoint.pt.json.sidecar.tmp").write_bytes(b"j" * 10)
    available["bytes"] = 890
    result = campaign.worker_storage_guard({}, contract, directory, 1000,
        checkpoint_paths=(first, final), checkpoint_limit=256,
        checkpoint_byte_limits={str(first): 128, final: 128})
    assert result["checkpoint_owned_bytes"] == {str(first): 70, str(final): 40}
    assert result["owned_bytes"]["checkpoint"] == 110
    assert result["owned_bytes"]["other"] == 10
    assert result["credited_declared_output_bytes"] == 110 and result["required_bytes"] == 890


@pytest.mark.parametrize("invalid", ["missing", "extra", "aggregate", "oversize", "boolean", "duplicate"])
def test_explicit_checkpoint_caps_cannot_change_the_authorized_budget(worker_storage, invalid):
    directory, contract, _ = worker_storage
    first, final = directory / "update_000004.pt", directory / "endpoint.pt"
    limits, total = {first: 128, final: 128}, 256
    if invalid == "missing":
        limits.pop(first)
    elif invalid == "extra":
        limits[directory / "extra.pt"] = 128
    elif invalid == "aggregate":
        total = 257
    elif invalid == "oversize":
        limits[first] = 129
        total = 257
    elif invalid == "boolean":
        limits[first] = True
    else:
        limits[str(first)] = 128
        total = 384
    with pytest.raises(campaign.CampaignIntegrityError, match="per-checkpoint caps"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(first, final), checkpoint_limit=total, checkpoint_byte_limits=limits)


def test_explicit_checkpoint_cap_cannot_spend_another_snapshots_unused_bytes(worker_storage):
    directory, contract, _ = worker_storage
    first, final = directory / "update_000004.pt", directory / "endpoint.pt"
    first.write_bytes(b"a" * 129)
    final.write_bytes(b"b" * 10)
    with pytest.raises(campaign.CampaignIntegrityError, match="per-checkpoint storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(first, final), checkpoint_limit=256,
            checkpoint_byte_limits={first: 128, final: 128})
    assert first.read_bytes() == b"a" * 129 and final.read_bytes() == b"b" * 10


def test_explicit_checkpoint_publication_copies_share_only_their_own_cap(worker_storage):
    directory, contract, _ = worker_storage
    first, final = directory / "update_000004.pt", directory / "endpoint.pt"
    first.write_bytes(b"a" * 70)
    (directory / ".update_000004.pt.independent-copy.tmp").write_bytes(b"t" * 70)
    with pytest.raises(campaign.CampaignIntegrityError, match="per-checkpoint storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(first, final), checkpoint_limit=256,
            checkpoint_byte_limits={first: 128, final: 128})


@pytest.mark.parametrize("alias", ["checkpoint", "sidecar", "cache"])
def test_explicit_checkpoints_cannot_alias_another_budget_identity(worker_storage, alias):
    directory, contract, _ = worker_storage
    first, final = directory / "update_000004.pt", directory / "endpoint.pt"
    first.write_bytes(b"a" * 20)
    if alias == "checkpoint":
        destination = final
    elif alias == "sidecar":
        destination = directory / ".endpoint.pt.json.sidecar.tmp"
    else:
        (directory / "runtime").mkdir()
        destination = directory / "runtime" / "cache.bin"
    os.link(first, destination)
    with pytest.raises(campaign.CampaignIntegrityError, match="hardlink crosses"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(first, final), checkpoint_limit=256,
            checkpoint_byte_limits={first: 128, final: 128})


def test_inflight_files_have_an_aggregate_cap_and_cache_has_its_own_cap(worker_storage):
    directory, contract, _ = worker_storage
    stdout, sdk = directory / "stdout.txt", directory / "sdk.log"
    stdout.write_bytes(b"s" * 40)
    sdk.write_bytes(b"d" * 30)
    with pytest.raises(campaign.CampaignIntegrityError, match="other storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000)
    sdk.unlink()
    cache = directory / "empty_python_cache"
    cache.mkdir()
    (cache / "runtime.bin").write_bytes(b"r" * 33)
    with pytest.raises(campaign.CampaignIntegrityError, match="cache storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000)


def test_hardlinks_cannot_cross_declared_data_and_inflight_budget_categories(worker_storage):
    directory, contract, _ = worker_storage
    checkpoint = directory / "endpoint.pt"
    checkpoint.write_bytes(b"c" * 20)
    os.link(checkpoint, directory / "stdout.txt")
    with pytest.raises(campaign.CampaignIntegrityError, match="hardlink crosses"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(checkpoint,), checkpoint_limit=128)


def test_checkpoint_publication_same_inode_is_counted_once_and_independent_copies_are_capped(worker_storage):
    directory, contract, _ = worker_storage
    checkpoint = directory / "endpoint.pt"
    temporary = directory / ".endpoint.pt.actual-publication.tmp"
    temporary.write_bytes(b"c" * 70)
    os.link(temporary, checkpoint)
    result = campaign.worker_storage_guard({}, contract, directory, 1000,
        checkpoint_paths=(checkpoint,), checkpoint_limit=128)
    assert result["owned_bytes"]["checkpoint"] == 70
    assert result["credited_declared_output_bytes"] == 70 and result["required_bytes"] == 930
    independent = directory / ".endpoint.pt.other-publication.tmp"
    independent.write_bytes(b"d" * 70)
    with pytest.raises(campaign.CampaignIntegrityError, match="checkpoint storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(checkpoint,), checkpoint_limit=128)


def test_sparse_trace_memmap_credits_allocated_blocks_not_its_future_logical_writes(worker_storage):
    directory, contract, available = worker_storage
    temporary = directory / ".control-trace-sparse-fixture"
    temporary.mkdir()
    path = temporary / "field_0.npy"
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(16, 8192))
    values[0, 0] = 1.
    values.flush()
    del values
    measured = path.stat()
    allocated = measured.st_blocks * 512
    assert 0 < allocated < measured.st_size
    contract["caps"]["trace_bytes_per_policy_sample"] = 2 * 1024**2
    required = 8 * 1024**2
    available["bytes"] = required
    result = campaign.worker_storage_guard({}, contract, directory, required,
        trace_prefix=directory / "trace.npz", trace_sample_limit=1)
    assert result["owned_bytes"]["trace_maps"] == measured.st_size
    assert result["credited_declared_output_bytes"] == allocated
    assert result["required_bytes"] == required - allocated
    # Crediting the memmap's unallocated tail would incorrectly permit this
    # disk state even though its future writes remain reserved.
    available["bytes"] = required - measured.st_size + 1
    with pytest.raises(campaign.CampaignDiskError, match="exceeds available"):
        campaign.worker_storage_guard({}, contract, directory, required,
            trace_prefix=directory / "trace.npz", trace_sample_limit=1)


def test_checkpoint_sidecar_temporary_is_inflight_and_never_checkpoint_credit(worker_storage):
    directory, contract, _ = worker_storage
    checkpoint = directory / "endpoint.pt"
    checkpoint.write_bytes(b"c" * 20)
    sidecar = directory / ".endpoint.pt.json.actual-publication.tmp"
    sidecar.write_bytes(b"j" * 50)
    result = campaign.worker_storage_guard({}, contract, directory, 1000,
        checkpoint_paths=(checkpoint,), checkpoint_limit=128)
    assert result["owned_bytes"]["checkpoint"] == 20 and result["owned_bytes"]["other"] == 50
    assert result["credited_declared_output_bytes"] == 20 and result["required_bytes"] == 980
    (directory / ".endpoint.pt.json.other-publication.tmp").write_bytes(b"j" * 20)
    with pytest.raises(campaign.CampaignIntegrityError, match="other storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000,
            checkpoint_paths=(checkpoint,), checkpoint_limit=128)


def test_whole_worker_runtime_tree_is_cache_with_no_future_output_credit(worker_storage):
    directory, contract, _ = worker_storage
    runtime = directory / "runtime/home/nested"
    runtime.mkdir(parents=True)
    (runtime / "sdk_cache.bin").write_bytes(b"r" * 24)
    (directory / "runtime.profile.json").write_bytes(b"profile" * 4)
    result = campaign.worker_storage_guard({}, contract, directory, 1000)
    assert result["owned_bytes"]["cache"] == 24 and result["owned_bytes"]["other"] == 28
    assert result["credited_declared_output_bytes"] == 0 and result["required_bytes"] == 1000
    (runtime / "later_cache.bin").write_bytes(b"x" * 9)
    with pytest.raises(campaign.CampaignIntegrityError, match="cache storage cap exceeded"):
        campaign.worker_storage_guard({}, contract, directory, 1000)


@pytest.mark.parametrize("kind", ["directory_symlink", "file_symlink", "fifo"])
def test_worker_runtime_cache_cannot_hide_links_or_special_entries(worker_storage, kind):
    directory, contract, _ = worker_storage
    cache = directory / "runtime"
    cache.mkdir()
    sentinel = directory.parent / "outside_original.bin"
    sentinel.write_bytes(b"outside file must remain intact")
    if kind == "directory_symlink":
        (cache / "mounted_cache").symlink_to(directory.parent, target_is_directory=True)
    elif kind == "file_symlink":
        (cache / "foreign.bin").symlink_to(sentinel)
    else:
        os.mkfifo(cache / "special.bin")
    with pytest.raises(campaign.CampaignIntegrityError, match="symlink|regular file"):
        campaign.worker_storage_guard({}, contract, directory, 1000)
    assert sentinel.read_bytes() == b"outside file must remain intact"


@pytest.mark.parametrize("field", ["st_uid", "st_dev"])
@pytest.mark.parametrize("kind", ["root", "directory", "file"])
def test_foreign_owner_or_filesystem_cannot_spend_the_worker_output_reserve(worker_storage, monkeypatch, field, kind):
    directory, contract, _ = worker_storage
    cache = directory / "runtime"
    cache.mkdir()
    file = cache / "sdk_cache.bin"
    file.write_bytes(b"actual owned regular fixture")
    target = {"root": directory, "directory": cache, "file": file}[kind]
    actual_lstat = Path.lstat

    def changed_stat(path, *args, **kwargs):
        observed = actual_lstat(path, *args, **kwargs)
        if path != target:
            return observed
        values = {name: getattr(observed, name) for name in dir(observed) if name.startswith("st_")}
        values[field] += 1
        return SimpleNamespace(**values)

    # Only the ownership/device observation is counterfactual. The actual
    # path traversal, file contents, budget and statvfs guard are unchanged.
    monkeypatch.setattr(Path, "lstat", changed_stat)
    if kind == "root" and field == "st_dev":
        message = "filesystem"
    else:
        message = "owned|owner|filesystem"
    with pytest.raises(campaign.CampaignIntegrityError, match=message):
        campaign.worker_storage_guard({}, contract, directory, 1000)


@pytest.mark.parametrize("name", sorted(campaign.runtime_paths.RESERVED_ENVIRONMENT))
def test_worker_environment_cannot_override_any_protected_runtime_path(tmp_path, monkeypatch, name):
    directory = tmp_path / "new_worker"
    directory.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid runtime environment must fail before Popen")

    monkeypatch.setattr(campaign.subprocess, "Popen", forbidden)
    with pytest.raises(campaign.CampaignIntegrityError, match="runtime path isolation"):
        REAL_LAUNCH([sys.executable, "-B", "-c", "pass"], directory, [], 1., lambda *a, **k: None,
                    worker_env={name: "/unowned/override"})


def test_pending_original_processes_do_not_launch_or_reserve(campaign_inputs, monkeypatch):
    p = campaign_inputs
    calls = []
    monkeypatch.setattr(queue_validation, "check_dependency", lambda *a, **k: {
        "status": "pending", "controller_live": True, "live_workers": [{"pid": os.getpid()}]})
    monkeypatch.setattr(campaign, "launch_owned_worker", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(campaign.predecessors.time, "sleep", lambda value: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run(p)
    summary = json.loads((p["output"] / "summary.json").read_bytes())
    assert calls == [] and summary["charged_updates"] == 0 and summary["jobs"] == {}
    assert summary["status"] == "stopped"


def test_real_entire_CPU_training_grid_independent_stages_all_four_actor_modes(campaign_inputs):
    p = campaign_inputs
    result = run(p)
    assert result["status"] == "training_completed_evaluation_pending"
    assert result["charged_updates"] == result["verified_successful_updates"] == p["protocol"]["budget"]["training_updates"]
    assert result["charged_fresh_transitions"] == result["verified_fresh_transitions"] == p["protocol"]["budget"]["fresh_transitions"]
    assert result["formal_architecture_selection"] is False
    assert result["independent_evaluation_performed"] is False
    assert result["full_evaluation_matrix_closed"] is False
    assert len(result["evaluation_cells"]) == len(p["protocol"]["evaluation_cells"])
    assert all(cell["status"] == "missing" for cell in result["evaluation_cells"].values())
    identities = set()
    for job in p["protocol"]["jobs"]:
        record = result["jobs"][job["id"]]
        assert record["status"] == "training_completed" and len(record["stages"]) == 2
        reservation = json.loads(Path(record["reservation"]["path"]).read_bytes())
        assert reservation == campaign.reservation_for(p["protocol"], protocol._receipt(p["protocol_path"]), job)
        assert reservation["charge_scope"] == "whole_job_once_before_first_child"
        parent_checkpoint = None
        for stage, record in zip(job["stages"], record["stages"]):
            identities.add((record["worker"]["process"]["pid"], record["worker"]["process"]["start"]))
            endpoint = campaign.verify_segment_endpoint(p["protocol"], job, stage["index"],
                record["training"]["endpoint"])
            payload = torch.load(endpoint["checkpoint"]["path"], map_location="cpu", weights_only=True)
            assert payload["optimizer"]["state"]
            assert payload["metadata"]["continuation_parent"] is None if stage["index"] == 0 else (
                payload["metadata"]["continuation_parent"]["path"] == parent_checkpoint["path"])
            assert endpoint["cumulative_collected_transitions"] == stage["expected_cumulative_transitions"]
            parent_checkpoint = endpoint["checkpoint"]
    assert len(identities) == sum(len(j["stages"]) for j in p["protocol"]["jobs"])
    assert all(campaign.predecessors._process(pid) is None for pid, start in identities)
    with pytest.raises(campaign.CampaignIntegrityError, match="exists"):
        run(p)
    for pin in p["protocol"]["execution"]["resource_locks"]:
        descriptor = os.open(pin["path"], os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


def test_actual_final_optimizer_output_overflow_stops_before_checkpoint_publication(campaign_inputs, monkeypatch):
    p = campaign_inputs
    late_output = r'''
from transformer_rl.ppo import PPOTrainer
original_update = PPOTrainer.update
def update_with_late_sdk_output(self, *args, **kwargs):
    result = original_update(self, *args, **kwargs)
    contract = json.loads(Path(request['storage_contract']['path']).read_bytes())
    destination = Path(request['output_root']).parent / 'late_sdk_output.bin'
    with destination.open('xb') as stream:
        stream.write(b'x' * (contract['caps']['inflight_bytes'] + 1))
    return result
PPOTrainer.update = update_with_late_sdk_output
'''
    bootstrap = BOOTSTRAP.replace("from transformer_rl.exposure_process import main",
                                  late_output + "\nfrom transformer_rl.exposure_process import main")
    launches = []

    def launch(command, directory, leases, timeout, publish, **kwargs):
        launches.append(Path(directory))
        # Isolate the worker's pre-publication guard. Parent monitoring has a
        # separate actual pidfd/byte-growth test below, without this omission.
        kwargs.pop("monitor", None)
        return REAL_LAUNCH([sys.executable, "-B", "-c", bootstrap, *command[4:]],
                          directory, leases, timeout, publish, **kwargs)

    monkeypatch.setattr(campaign, "launch_owned_worker", launch)
    with pytest.raises(campaign.CampaignWorkerError, match="campaign stopped without retry"):
        run(p)
    assert len(launches) == 1
    directory = launches[0]
    completion = json.loads((directory / "train" / "completion.json").read_bytes())
    assert completion["successful_updates"] == completion["recorded_metric_updates"] == 1
    assert completion["error"]["phase"] == "publish_checkpoint"
    assert completion["error"]["type"] == "CampaignIntegrityError"
    assert "other storage cap exceeded" in completion["error"]["message"]
    assert not list((directory / "train").rglob("endpoint.pt"))
    assert not list((directory / "train").rglob("endpoint.json"))
    outcome = json.loads((directory / "outcome.json").read_bytes())
    assert outcome["status"] == "unknown_failure" and outcome["typed_numerical_origin"] is None
    assert outcome["shutdown_errors"][0]["owner"] == "worker_guard"
    summary = json.loads((p["output"] / "summary.json").read_bytes())
    assert summary["status"] == "stopped" and len(summary["jobs"]) == 1
    assert summary["charged_updates"] == p["protocol"]["jobs"][0]["reserved_updates"]


def test_actual_checkpoint_cap_overflow_preserves_unsealed_file_without_endpoint(campaign_inputs, monkeypatch):
    p = campaign_inputs
    actual_launch = campaign.launch_owned_worker

    def worker_guard_only(*args, **kwargs):
        # Prevent the parent from winning the detection race in this dedicated
        # post-save guard test. All kernel/process/learner checks remain actual.
        kwargs.pop("monitor", None)
        return actual_launch(*args, **kwargs)

    monkeypatch.setattr(campaign, "launch_owned_worker", worker_guard_only)
    initial = next(item for item in p["storage"]["measurements"] if item["kind"] == "checkpoint")
    # This intentionally insufficient CPU contract covers a measured model
    # state, but not the full Adam/RNG checkpoint produced by real learning.
    p["storage"]["caps"]["checkpoint_bytes"] = initial["receipt"]["bytes"]
    canonical(p["storage_path"], p["storage"])
    with pytest.raises(campaign.CampaignWorkerError, match="campaign stopped without retry"):
        run(p)
    first = p["protocol"]["jobs"][0]
    stage = first["stages"][0]
    directory = p["output"] / first["id"] / "stage_0000"
    checkpoint = directory / "train" / f"stage_0000_{stage['name']}" / "endpoint.pt"
    assert checkpoint.is_file() and checkpoint.stat().st_size > p["storage"]["caps"]["checkpoint_bytes"]
    assert not checkpoint.with_name("endpoint.json").exists()
    completion = json.loads((directory / "train" / "completion.json").read_bytes())
    assert completion["status"] == "failed" and completion["error"]["phase"] == "publish_checkpoint"
    assert "checkpoint storage cap exceeded" in completion["error"]["message"]
    assert completion["unsealed_checkpoint_paths"] == [str(checkpoint)]
    assert completion["last_sealed_checkpoint"] is None
    summary = json.loads((p["output"] / "summary.json").read_bytes())
    assert summary["status"] == "stopped" and len(summary["jobs"]) == 1
    assert summary["verified_successful_updates"] == 0


def test_initial_parent_monitor_storage_rejection_starts_no_child_and_preserves_terminal_record(tmp_path, monkeypatch):
    directory = tmp_path / "rejected_worker"
    directory.mkdir()
    (directory / "existing_burst.bin").write_bytes(b"x" * 262144)
    contract = {"caps": {"checkpoint_bytes": 128, "metric_bytes_per_update": 64,
        "trace_bytes_per_policy_sample": 256, "inflight_bytes": 65536,
        "runtime_cache_bytes": 32, "free_margin_bytes": 16}}
    original_api = campaign.predecessors._pidfd_api
    launches, signals = [], []

    def observed_api():
        opening, sending = original_api()

        def send_pid(descriptor, number):
            signals.append(number)
            return sending(descriptor, number)

        return opening, send_pid

    def forbidden_popen(*args, **kwargs):
        launches.append(True)
        raise AssertionError("initial storage rejection must precede child creation")

    monkeypatch.setattr(campaign.predecessors, "_pidfd_api", observed_api)
    monkeypatch.setattr(campaign.subprocess, "Popen", forbidden_popen)
    command = [sys.executable, "-B", "-c", "pass"]
    with pytest.raises(campaign.CampaignIntegrityError, match="other storage cap exceeded"):
        REAL_LAUNCH(command, directory, [], 10., lambda *a, **k: None,
            monitor=lambda: campaign.worker_storage_guard({}, contract, directory, 10_000))
    assert launches == [] and not (directory / "worker.process.json").exists()
    terminal = json.loads((directory / "worker.completion.json").read_bytes())
    assert terminal["command"] == command and terminal["process"] is None
    assert terminal["returncode"] is None and terminal["timed_out"] is False
    assert signals == [0]
    assert (directory / "existing_burst.bin").stat().st_size == 262144


def test_actual_parent_monitor_output_burst_signals_only_owned_pidfd_and_preserves_terminal_record(tmp_path, monkeypatch):
    directory = tmp_path / "monitored_worker"
    directory.mkdir()
    contract = {"caps": {"checkpoint_bytes": 128, "metric_bytes_per_update": 64,
        "trace_bytes_per_policy_sample": 256, "inflight_bytes": 65536,
        "runtime_cache_bytes": 32, "free_margin_bytes": 16}}
    original_api, original_popen = campaign.predecessors._pidfd_api, campaign.subprocess.Popen
    launches, signals, monitor_launch_counts = [], [], []

    def observed_api():
        opening, sending = original_api()
        identities = {}

        def open_pid(pid):
            descriptor = opening(pid)
            identities[descriptor] = pid
            return descriptor

        def send_pid(descriptor, number):
            signals.append((identities[descriptor], number))
            return sending(descriptor, number)

        return open_pid, send_pid

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        launches.append(child.pid)
        return child

    def monitor():
        monitor_launch_counts.append(len(launches))
        return campaign.worker_storage_guard({}, contract, directory, 10_000)

    monkeypatch.setattr(campaign.predecessors, "_pidfd_api", observed_api)
    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    program = "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_bytes(b'x'*262144);time.sleep(20)"
    command = [sys.executable, "-B", "-c", program, str(directory / "sdk_burst.bin")]
    with pytest.raises(campaign.CampaignIntegrityError, match="other storage cap exceeded"):
        REAL_LAUNCH(command, directory, [], 10., lambda *a, **k: None,
            monitor=monitor)
    assert len(launches) == 1
    assert monitor_launch_counts[0] == 0 and 1 in monitor_launch_counts
    actual = json.loads((directory / "worker.process.json").read_bytes())
    terminal = json.loads((directory / "worker.completion.json").read_bytes())
    assert actual["command"] == terminal["command"] == actual["process"]["argv"] == command
    assert terminal["process"]["pid"] == launches[0]
    assert terminal["returncode"] == -signal.SIGTERM and terminal["timed_out"] is False
    assert signals == [(os.getpid(), 0), (launches[0], signal.SIGTERM)]
    assert campaign.predecessors._process(launches[0]) is None
    assert (directory / "sdk_burst.bin").stat().st_size == 262144
    assert not list(directory.rglob("endpoint.pt")) and not list(directory.rglob("endpoint.json"))


def test_actual_runtime_output_burst_keeps_profiles_pinned_and_only_terminates_the_owned_child(tmp_path):
    directory = tmp_path / "runtime_burst_worker"
    directory.mkdir()
    contract = {"caps": {"checkpoint_bytes": 128, "metric_bytes_per_update": 64,
        "trace_bytes_per_policy_sample": 256, "inflight_bytes": 65536,
        "runtime_cache_bytes": 64, "free_margin_bytes": 16}}
    program = ("import os,pathlib,time;"
               "pathlib.Path(os.environ['XDG_CACHE_HOME'],'sdk_burst.bin').write_bytes(b'x'*512);"
               "time.sleep(20)")
    command = [sys.executable, "-B", "-c", program]
    with pytest.raises(campaign.CampaignIntegrityError, match="cache storage cap exceeded"):
        REAL_LAUNCH(command, directory, [], 10., lambda *a, **k: None,
            monitor=lambda: campaign.worker_storage_guard({}, contract, directory, 100_000))
    actual = json.loads((directory / "worker.process.json").read_bytes())
    terminal = json.loads((directory / "worker.completion.json").read_bytes())
    assert actual["process"] == terminal["process"] and terminal["returncode"] == -signal.SIGTERM
    assert actual["runtime_profile"] == terminal["runtime_profile"]
    profile = campaign.runtime_paths.validate_runtime_artifact(directory, actual["runtime_profile"])
    assert (Path(profile["paths"]["xdg_cache"]) / "sdk_burst.bin").read_bytes() == b"x" * 512
    assert campaign.predecessors._process(actual["process"]["pid"]) is None


def test_actual_child_runtime_profile_tamper_is_terminal_without_relaunch(tmp_path):
    directory = tmp_path / "tampered_runtime_worker"
    directory.mkdir()
    program = ("import os,pathlib,time;time.sleep(.2);"
               "p=pathlib.Path(os.environ['TRANSFORMER_RL_RUNTIME_PROFILE']);"
               "p.write_bytes(p.read_bytes()+b' ');time.sleep(20)")
    command = [sys.executable, "-B", "-c", program]
    with pytest.raises(ValueError, match="external SHA-256"):
        REAL_LAUNCH(command, directory, [], 10., lambda *a, **k: None)
    actual = json.loads((directory / "worker.process.json").read_bytes())
    terminal = json.loads((directory / "worker.completion.json").read_bytes())
    assert terminal["command"] == command and terminal["process"] == actual["process"]
    assert terminal["returncode"] == -signal.SIGTERM and not terminal["timed_out"]
    assert actual["runtime_profile"] == terminal["runtime_profile"]
    with pytest.raises(ValueError, match="external SHA-256"):
        campaign.runtime_paths.validate_runtime_artifact(directory, terminal["runtime_profile"])


def test_pidfd_both_APIs_are_probed_before_Popen(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign.predecessors, "_pidfd_api", lambda: (
        lambda pid: os.open("/dev/null", os.O_RDONLY),
        lambda fd, number: (_ for _ in ()).throw(OSError("actual probe failed"))))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: calls.append(True))
    with pytest.raises(OSError, match="probe failed"):
        REAL_LAUNCH([sys.executable, "-B", "-c", "pass"], tmp_path, [], 1., lambda *a, **k: None)
    assert calls == [] and not (tmp_path / "worker.process.json").exists()


def test_actual_pidfd_timeout_only_exact_owned_child(tmp_path, monkeypatch):
    original = campaign.predecessors._pidfd_api
    signals = []
    def observed():
        opening, sending = original()
        def send(fd, number):
            signals.append(number)
            return sending(fd, number)
        return opening, send
    monkeypatch.setattr(campaign.predecessors, "_pidfd_api", observed)
    result = REAL_LAUNCH([sys.executable, "-B", "-c", "import time;time.sleep(20)"],
        tmp_path, [], .1, lambda *a, **k: None)
    assert result["timed_out"] is True and result["returncode"] == -signal.SIGTERM
    assert signals == [0, signal.SIGTERM]
    assert campaign.predecessors._process(result["process"]["pid"]) is None


def test_actual_child_empty_exec_argv_observation_waits_without_relaunch(tmp_path, monkeypatch):
    identity = campaign._identity
    popen = campaign.subprocess.Popen
    observations, launches = [], []
    def empty_first(pid):
        actual = identity(pid)
        observations.append(pid)
        if len(observations) == 1:
            return {**actual, "argv": []}
        return actual
    def launched(*args, **kwargs):
        child = popen(*args, **kwargs)
        launches.append(child.pid)
        return child
    monkeypatch.setattr(campaign, "_identity", empty_first)
    monkeypatch.setattr(campaign.subprocess, "Popen", launched)
    command = [sys.executable, "-B", "-c", "import time;time.sleep(.1)"]
    result = REAL_LAUNCH(command, tmp_path, [], 2., lambda *a, **k: None)
    assert result["returncode"] == 0 and result["process"]["argv"] == command
    assert len(launches) == 1 and len(observations) >= 2
    assert set(observations) == set(launches)


def test_actual_child_empty_exit_argv_observation_waits_for_real_terminal_status(tmp_path, monkeypatch):
    """Mock only one /proc observation after an actual one-second wait.

    The child, Popen.wait/poll, pidfd and exit status remain real. The injected
    same-start empty cmdline represents kernel exit observation before the
    original Popen handle is waitable; it must not authorize a new process.
    """
    read_process = campaign.predecessors._process
    injected = []
    command = [sys.executable, "-B", "-c", "import time;time.sleep(1.3)"]

    def empty_during_exit(pid):
        actual = read_process(pid)
        if not injected and actual is not None and (tmp_path / "worker.process.json").is_file():
            assert actual["argv"] == command and actual["state"] != "Z"
            injected.append(actual.copy())
            return {**actual, "argv": []}
        return actual

    monkeypatch.setattr(campaign.predecessors, "_process", empty_during_exit)
    result = REAL_LAUNCH(command, tmp_path, [], 3., lambda *a, **k: None,
                         monitor=lambda: None)
    assert len(injected) == 1
    assert result["returncode"] == 0 and result["timed_out"] is False
    assert result["process"] == {key: injected[0][key] for key in ("pid", "start", "argv")}
    process = json.loads((tmp_path / "worker.process.json").read_bytes())
    terminal = json.loads((tmp_path / "worker.completion.json").read_bytes())
    assert process["process"] == terminal["process"] == result["process"]
    assert process["command"] == terminal["command"] == command
    assert read_process(result["process"]["pid"]) is None
    canonical(tmp_path / "exit_observation_fixture.json", {
        "scope": "one proc-reader empty argv observation; actual Popen/pidfd/wait/exit status",
        "actual_observed_process": injected[0], "injected_argv": [],
        "actual_terminal": result})


def test_actual_child_nonempty_changed_exit_argv_rejected_and_owned_child_terminated(tmp_path, monkeypatch):
    """A changed nonempty proc observation never receives terminal grace.

    Only that proc-reader value is a fixture. The original sleeping child is
    actually terminated and reaped through the launcher's real owned pidfd.
    """
    read_process = campaign.predecessors._process
    injected = []
    command = [sys.executable, "-B", "-c", "import time;time.sleep(20)"]
    changed = [*command, "fixture_changed_command"]

    def changed_during_observation(pid):
        actual = read_process(pid)
        if not injected and actual is not None and (tmp_path / "worker.process.json").is_file():
            assert actual["argv"] == command and actual["state"] != "Z"
            injected.append(actual.copy())
            return {**actual, "argv": changed}
        return actual

    monkeypatch.setattr(campaign.predecessors, "_process", changed_during_observation)
    with pytest.raises(campaign.CampaignIntegrityError, match="child command changed"):
        REAL_LAUNCH(command, tmp_path, [], 5., lambda *a, **k: None,
                    monitor=lambda: None)
    assert len(injected) == 1
    process = json.loads((tmp_path / "worker.process.json").read_bytes())
    terminal = json.loads((tmp_path / "worker.completion.json").read_bytes())
    identity = {key: injected[0][key] for key in ("pid", "start", "argv")}
    assert process["process"] == terminal["process"] == identity
    assert process["command"] == terminal["command"] == command
    assert terminal["returncode"] == -signal.SIGTERM and terminal["timed_out"] is False
    assert read_process(identity["pid"]) is None
    canonical(tmp_path / "exit_observation_fixture.json", {
        "scope": "one proc-reader nonempty changed argv; actual owned pidfd termination",
        "actual_observed_process": injected[0], "injected_argv": changed,
        "actual_terminal": terminal})


def change_failure(p, mode):
    base = replace(p["base"], environment={**p["base"].environment, "fixture_failure": mode})
    write_json(p["inputs"] / "base.json", base.to_dict())
    p["base"] = base
    p["history"] = p["tmp"] / f"history_{mode}"
    history_study.prepare_history_study(p["inputs"] / "study.json", p["inputs"] / "base.json",
        p["history"], history_lengths=[1, 3], position_reference="current")
    p["protocol"] = freeze(p)
    canonical(p["protocol_path"], p["protocol"])
    p["storage"]["protocol_raw_sha256"] = protocol._receipt(p["protocol_path"])["sha256"]
    canonical(p["storage_path"], p["storage"])


def test_actual_typed_numerical_failure_charges_whole_job_and_advances_next_seed(campaign_inputs, monkeypatch):
    p = campaign_inputs
    change_failure(p, "numerical")
    original = campaign.launch_owned_worker
    launches = []

    def stop_after_verified_advance(command, directory, *args, **kwargs):
        launches.append(Path(directory).parent.name)
        if len(launches) == 2:
            raise KeyboardInterrupt()
        return original(command, directory, *args, **kwargs)
    monkeypatch.setattr(campaign, "launch_owned_worker", stop_after_verified_advance)
    with pytest.raises(KeyboardInterrupt):
        run(p)
    summary = json.loads((p["output"] / "summary.json").read_bytes())
    first, second = p["protocol"]["jobs"][:2]
    assert launches == [first["id"], second["id"]]
    failed = summary["jobs"][first["id"]]
    assert failed["status"] == "numerical_failure"
    assert failed["stages"][0]["failed_training"]["actual_collected_transitions"] == 0
    assert len(failed["stages"]) == 1
    assert summary["charged_updates"] == first["reserved_updates"] + second["reserved_updates"]
    assert summary["charged_fresh_transitions"] == first["reserved_fresh_transitions"] + second["reserved_fresh_transitions"]
    assert summary["refund"] is False and summary["automatic_retries"] == 0
    assert all(cell["status"] == "missing" and cell["reason"] == "training_numerical_failure"
               for cell in summary["evaluation_cells"].values() if cell["identity"]["job_id"] == first["id"])


def test_error_text_cannot_grant_numerical_failure_or_another_seed(campaign_inputs):
    p = campaign_inputs
    change_failure(p, "unknown")
    with pytest.raises(campaign.CampaignWorkerError, match="unknown"):
        run(p)
    summary = json.loads((p["output"] / "summary.json").read_bytes())
    assert len(summary["jobs"]) == 1 and summary["status"] == "stopped"
    first = p["protocol"]["jobs"][0]
    assert summary["charged_updates"] == first["reserved_updates"]
    outcome = json.loads(Path(summary["jobs"][first["id"]]["stages"][0]["outcome"]["path"]).read_bytes())
    assert outcome["typed_numerical_origin"] is None and outcome["status"] == "unknown_failure"


def test_controller_lease_checks_actual_inherited_FLOCK_parent_and_controller_bytes(campaign_inputs):
    p = campaign_inputs
    root = p["output"]
    root.mkdir()
    controller = {"format": "transformer_rl.exposure_controller", "schema_version": 1,
        "protocol_sha256": p["protocol"]["sha256"],
        "protocol_raw_receipt": protocol._receipt(p["protocol_path"]),
        "expected_protocol_sha256": protocol._receipt(p["protocol_path"])["sha256"],
        "storage_contract_receipt": protocol._receipt(p["storage_path"]),
        "source": p["protocol"]["source"], "runtime": p["protocol"]["runtime"],
        "process": campaign._identity(os.getpid()), "evaluation_provider": None}
    canonical(root / "controller.json", controller)
    receipt = protocol._receipt(root / "controller.json")
    with campaign.resource_lease(p["protocol"], lambda *a, **k: None) as leases:
        program = r'''
import json,sys
from pathlib import Path
from transformer_rl.exposure_campaign import validate_controller_lease
data=json.loads(Path(sys.argv[1]).read_bytes())
validate_controller_lease(data['protocol'],data['leases'],data['controller'])
'''
        descriptor = leases[0]["descriptor"]
        # A separate open of the same inode is not an inherited locked handle.
        unlocked = os.open(leases[0]["path"], os.O_RDWR)
        try:
            changed = deepcopy(leases)
            changed[0]["descriptor"] = unlocked
            directory = root / "bad_lease"
            directory.mkdir()
            data = {"protocol": p["protocol"], "leases": changed, "controller": receipt}
            canonical(directory / "data.json", data)
            result = REAL_LAUNCH([sys.executable, "-B", "-c", program, str(directory / "data.json")],
                directory, changed, 10., lambda *a, **k: None)
            assert result["returncode"] != 0
            assert "exclusive flock" in (directory / "stderr.txt").read_text()
        finally:
            os.close(unlocked)
        assert os.fstat(descriptor).st_ino == leases[0]["inode"]
        directory = root / "good_lease"
        directory.mkdir()
        data = {"protocol": p["protocol"], "leases": leases, "controller": receipt}
        canonical(directory / "data.json", data)
        result = REAL_LAUNCH([sys.executable, "-B", "-c", program, str(directory / "data.json")],
            directory, leases, 10., lambda *a, **k: None)
        assert result["returncode"] == 0


def test_actual_checkpoint_wrong_environment_same_model_count_and_budget_rejected(campaign_inputs):
    p = campaign_inputs
    job = p["protocol"]["jobs"][0]
    declared = job["stages"][0]
    actual = deepcopy(declared["config"])
    actual["environment"]["recipe"] = "actually-trained-different-environment"
    output = p["output"] / job["id"] / "stage_0000" / "train"
    output.parent.mkdir(parents=True)
    result = training.train_exposure_segment({"name": declared["name"], "config": actual,
        "updates": declared["updates"]}, make_env, p["protocol"]["environment_factory"], output,
        job_id=job["id"], rollout_steps=p["protocol"]["execution"]["rollout_steps"],
        training_seed=job["training_seed"], retention_seed=job["retention_seed"],
        evaluation_seeds=[*p["protocol"]["evaluation"]["validation_seeds"], *p["protocol"]["evaluation"]["seeds"]],
        device="cpu", expected_initial_model_sha256=job["initial_model_sha256"],
        max_seconds=p["protocol"]["execution"]["max_seconds"])
    assert result["status"] == "completed"
    endpoint = output / f"stage_0000_{declared['name']}" / "endpoint.json"
    with pytest.raises(campaign.CampaignIntegrityError, match="checkpoint environment"):
        campaign.verify_segment_endpoint(p["protocol"], job, 0, protocol._receipt(endpoint))

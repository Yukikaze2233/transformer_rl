"""Tiny CPU worker-to-proof checks, never a teacher qualification provider.

Only the fixed campaign auditor callback and worker tensor-fixture factory
resolver are replaced by the imported synthetic rig. Actual FrameContinuation,
PPO, checkpoint serialization and verify_training execute. The parent has
explicitly initialized Adam moments; it does not represent real pretraining.
One module-scoped baseline is restored after each corruption test, avoiding
large per-case model copies or simulated experiment grids.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import pytest
import torch

from transformer_rl import retention_campaign as campaign
from transformer_rl.experiments import source_identity
from transformer_rl.frame_checkpoint import save_frame_checkpoint
from transformer_rl.frame_config import digest, json_bytes
from transformer_rl.ppo import PPOTrainer

from test_continuation_process import file_sha, reseal, rig as synthetic_rig


@dataclass
class Proof:
    request: dict
    protocol: dict
    branch: dict
    directory: Path
    parent: Path
    checkpoint: Path

    def verify(self):
        return campaign.verify_training(self.protocol, self.branch, self.request, self.directory)


def make_proof(directory, patch, coefficient=.3):
    """Construct small real learning artifacts with synthetic authorization."""
    rig = synthetic_rig.__wrapped__(directory, patch)
    trainer = PPOTrainer(rig.model, rig.config.ppo)
    for parameter in rig.model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    parent = directory / "initialized_synthetic_parent.pt"
    save_frame_checkpoint(parent, rig.model, trainer, rig.config, 1,
        {"seed": 71, "environment_factory": "packed_env:make_env",
         "environment_provenance": {"identity": "synthetic_tensor_fixture",
                                    "control_sha256": digest(rig.config.control)},
         "collected_transitions": 4, "source": source_identity()})
    rig.request["checkpoint"].update(path=str(parent), sha256=file_sha(parent),
        update=1, cumulative_transitions=4, consumed_updates=1)
    if coefficient > 0:
        rig.anchor(directory)
        rig.request["retention"]["coefficient"] = coefficient
    report = rig.worker.run_request(reseal(rig.request))
    protocol = {"execution": deepcopy(rig.request["execution"]),
                "source": deepcopy(rig.request["source"]),
                "retention_seed": 901, "retention_batch_size": 3}
    branch = {"coefficient": coefficient, "anchors": deepcopy(rig.request["retention"]["anchors"])}
    proof = Proof(deepcopy(rig.request), protocol, branch, rig.directory, parent, Path(report["checkpoint"]))
    assert sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) < 1_000_000
    assert proof.verify()["full_learning_state_validated"]
    return proof


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("small_retention_training_proof")
    patch = pytest.MonkeyPatch()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        proof = make_proof(directory, patch)
        original = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        yield proof, original
    finally:
        patch.undo()
        torch.set_num_threads(threads)


@pytest.fixture
def proof(baseline):
    original, snapshots = baseline
    value = Proof(deepcopy(original.request), deepcopy(original.protocol), deepcopy(original.branch),
                  original.directory, original.parent, original.checkpoint)
    try:
        yield value
    finally:
        for path, raw in snapshots.items():
            path.write_bytes(raw)


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_bytes(json_bytes(value) + b"\n")


def put(value, route, replacement):
    target = value
    for field in route[:-1]:
        target = target[field]
    target[route[-1]] = replacement


def rewrite_checkpoint(proof, change):
    """Reseal all redundant byte receipts, preserving the corruption itself."""
    payload = torch.load(proof.checkpoint, map_location="cpu", weights_only=True)
    change(payload)
    torch.save(payload, proof.checkpoint)
    identity = file_sha(proof.checkpoint)
    sidecar_path = Path(str(proof.checkpoint) + ".json")
    sidecar = read_json(sidecar_path)
    sidecar.update(sha256=identity, metadata=payload["metadata"],
                   config=payload["config"], update=payload["update"])
    write_json(sidecar_path, sidecar)
    report_path = proof.directory / "completion.json"
    report = read_json(report_path)
    report["checkpoint_sha256"] = identity
    report["last_sealed_checkpoint"].update(sha256=identity, bytes=proof.checkpoint.stat().st_size)
    write_json(report_path, report)


def rewrite_parent(proof, change):
    """Close the changed parent's SHA in every binding to isolate validation."""
    payload = torch.load(proof.parent, map_location="cpu", weights_only=True)
    change(payload)
    torch.save(payload, proof.parent)
    proof.request["checkpoint"]["sha256"] = file_sha(proof.parent)
    reseal(proof.request)
    write_json(proof.directory / "run.json", proof.request)
    report_path = proof.directory / "completion.json"
    report = read_json(report_path)
    report["request_sha256"] = proof.request["sha256"]
    write_json(report_path, report)

    def close_bindings(value):
        value["metadata"]["continuation_parent"]["sha256"] = proof.request["checkpoint"]["sha256"]
        value["metadata"]["continuation_request_sha256"] = proof.request["sha256"]

    rewrite_checkpoint(proof, close_bindings)


def test_complete_real_cpu_learning_state_closes_all_clocks_and_private_draws(proof):
    result = proof.verify()
    assert result["actual_updates"] == result["optimizer_steps"] == 2
    assert result["actual_samples"] == 8
    assert result["checkpoint"]["update"] == 3
    assert result["checkpoint"]["cumulative_transitions"] == 12
    assert result["checkpoint"]["consumed_updates"] == 3
    state = torch.load(proof.checkpoint, map_location="cpu", weights_only=True)
    assert state["metadata"]["continuation"]["retention"]["draw_count"] == 2
    assert state["metadata"]["continuation"]["retention"]["endpoint_draw_count"] == 6


def test_zero_coefficient_proof_has_no_anchor_or_private_rng_draw(tmp_path):
    patch = pytest.MonkeyPatch()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        proof = make_proof(tmp_path, patch, coefficient=0.)
        state = torch.load(proof.checkpoint, map_location="cpu", weights_only=True)
        assert proof.verify()["optimizer_steps"] == 2
        assert proof.request["retention"]["anchors"] == []
        assert state["metadata"]["continuation"]["retention"]["draw_count"] == 0
    finally:
        patch.undo()
        torch.set_num_threads(threads)


@pytest.mark.parametrize("route,replacement", [
    (("continuation_device",), "cuda:99"),
    (("continuation", "format"), "synthetic.invalid.continuation"),
    (("continuation", "schema_version"), True),
    (("continuation", "unexpected"), "reject"),
    (("continuation", "clock", "rollout_steps"), True),
    (("continuation_parent", "path"), "/synthetic/unbound_parent.pt"),
    (("continuation_parent", "sha256"), "0" * 64),
    (("continuation_parent", "update"), 2),
    (("continuation_parent", "resume"), True),
    (("continuation_parent", "unexpected"), "reject"),
    (("continuation_segment", "start_update"), 0),
    (("continuation_segment", "successful_updates"), 1),
    (("continuation_segment", "attempted_updates"), 1),
    (("continuation_segment", "fresh_transitions"), 7),
    (("continuation_segment", "discarded_transitions"), 1),
    (("initial_model_sha256",), "0" * 64),
    (("initial_model_hash_format",), "unsupported"),
    (("environment_provenance", "identity"), "wrong_environment"),
    (("continuation_request_sha256",), "0" * 64),
    (("continuation_branch", "branch_id"), "wrong_branch"),
])
def test_resealed_checkpoint_metadata_cannot_bypass_learning_binding(proof, route, replacement):
    rewrite_checkpoint(proof, lambda value: put(value["metadata"], route, replacement))
    with pytest.raises(ValueError):
        proof.verify()


@pytest.mark.parametrize("mutation", ["adam_step", "adam_nonfinite", "cpu_rng", "private_draw_count"])
def test_resealed_actual_learning_tensors_and_sampler_are_checked(proof, mutation):
    def change(value):
        if mutation == "adam_step":
            next(iter(value["optimizer"]["state"].values()))["step"].add_(1)
        elif mutation == "adam_nonfinite":
            next(iter(value["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
        elif mutation == "cpu_rng":
            value["rng"]["torch"] = value["rng"]["torch"][:1]
        else:
            state = value["metadata"]["continuation"]["retention"]
            state["draw_count"] -= 1
            state["sha256"] = digest({key: item for key, item in state.items() if key != "sha256"})

    rewrite_checkpoint(proof, change)
    with pytest.raises((ValueError, FloatingPointError)):
        proof.verify()


@pytest.mark.parametrize("route,replacement", [
    (("config", "ppo", "learning_rate"), .0003),
    (("config", "control", "policy_dt_s"), .02),
    (("metadata", "continuation"), {"synthetic": True}),
    (("metadata", "collected_transitions"), 3),
])
def test_resealed_parent_must_be_the_original_full_learning_contract(proof, route, replacement):
    rewrite_parent(proof, lambda value: put(value, route, replacement))
    with pytest.raises(ValueError):
        proof.verify()


@pytest.mark.parametrize("route,replacement", [
    (("format",), "unsupported"), (("schema_version",), True),
    (("stop_reason",), "SIGTERM"), (("failure_stage",), "step"),
    (("attempted_updates",), 1), (("actual_samples",), 7),
    (("reserved_transition_budget",), 1), (("partial_samples",), 1),
])
def test_completion_cannot_claim_full_grid_after_failure_or_changed_budget(proof, route, replacement):
    path = proof.directory / "completion.json"
    value = read_json(path)
    put(value, route, replacement)
    write_json(path, value)
    with pytest.raises(ValueError):
        proof.verify()


@pytest.mark.parametrize("route,replacement", [
    (("optimization", "optimizer_steps"), 2),
    (("optimization", "sample_count"), 3),
    (("optimization", "planned_optimizer_steps"), 2),
    (("consumed_updates",), 1), (("cumulative_transitions",), 7),
])
def test_actual_ppo_row_counts_must_close_recipe_and_collection_clock(proof, route, replacement):
    path = proof.directory / "metrics.jsonl"
    rows = [json.loads(row) for row in path.read_bytes().splitlines()]
    put(rows[0], route, replacement)
    path.write_bytes(b"".join(json_bytes(row) + b"\n" for row in rows))
    with pytest.raises(ValueError):
        proof.verify()


@pytest.mark.parametrize("artifact", ["run", "environment", "sidecar", "duplicate_row", "duplicate_completion"])
def test_redundant_artifacts_are_independently_parsed_and_bound(proof, artifact):
    if artifact == "run":
        path = proof.directory / "run.json"
        value = read_json(path)
        value["retention"]["seed"] += 1
        write_json(path, reseal(value))
    elif artifact == "environment":
        path = proof.directory / "environment.json"
        value = read_json(path)
        value["identity"] = "synthetic_changed_assets"
        write_json(path, value)
    elif artifact == "sidecar":
        path = Path(str(proof.checkpoint) + ".json")
        value = read_json(path)
        value["schema_version"] = True
        write_json(path, value)
    elif artifact == "duplicate_row":
        path = proof.directory / "metrics.jsonl"
        lines = path.read_bytes().splitlines()
        lines[0] = b'{"update":2,' + lines[0][1:]
        path.write_bytes(b"\n".join(lines) + b"\n")
    else:
        path = proof.directory / "completion.json"
        raw = path.read_bytes()
        path.write_bytes(b'{"schema_version":1,' + raw[1:])
    with pytest.raises(ValueError):
        proof.verify()

"""Real history/configuration/initial-state and byte-bound protocol checks.

Only the expensive host runtime and predecessor providers are synthetic. These
checks do not grant queue closure, simulator execution or hardware qualification.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch

from transformer_rl import experiments, exposure_protocol as protocol
from transformer_rl import exposure_training, frame_study, history_study
from transformer_rl.config import PPOConfig
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest
from transformer_rl.frame_policy import FramePolicyConfig
from transformer_rl.frame_training import FrameActorCritic
from transformer_rl.frame_workflow import _model_state_sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def resign(value, change):
    result = deepcopy(value)
    change(result)
    result.pop("sha256")
    result["sha256"] = digest(result)
    return result


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs, snapshot, sdk = [tmp_path / name for name in ("inputs", "snapshot", "sdk")]
    for root in (inputs, snapshot, sdk):
        root.mkdir()
    (snapshot / "environment.py").write_text("fixture_environment = True\n")
    (snapshot / "robot.bin").write_bytes(b"synthetic robot bytes; no simulator asset")
    (sdk / "SDK_VERSION").write_text("synthetic-cpu-sdk-v1\n")
    (sdk / "loader.py").write_text("sdk_fixture = True\n")
    contracts = {}
    for name, count in (("first", 3), ("second", 5), ("normal", 2), ("new_skill", 7)):
        path = snapshot / "contracts" / f"{name}.json"
        write_json(path, {"name": name, "target_num_envs": count, "physics_dt": .005,
                          "policy_dt": .01, "fixture": "never constructed"})
        contracts[name] = {"contract": f"contracts/{name}.json",
                           "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "num_envs": count}
    files = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    snapshot_sha = digest(files)
    write_json(snapshot / "snapshot.json", {"files": files, "sha256": snapshot_sha})
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)], "action_names": ["leg", "wheel"],
        "action_bounds": [.2, .5], "target_scale": [.25, 10.], "target_offset": [.1, -.2],
        "target_units": ["rad", "rad/s"]}
    policy = FramePolicyConfig(architecture="transformer", frame_dim=5, action_dim=2,
        history_length=3, actor_hidden_dims=(12, 8), encoder_hidden_dims=(12,),
        d_model=8, num_heads=2, num_layers=1, ffn_dim=16, mean_init_scale=.2)
    base = FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(12, 8),
        command_indices=(0,), initial_std=.8),
        PPOConfig(epochs=2, num_minibatches=2, learning_rate=3e-5, target_kl=.01),
        control, {"snapshot": str(snapshot), "snapshot_sha256": snapshot_sha,
                  **contracts["first"], "recipe": "fixed"})
    spec = {"base_config": "base.json", "environment_factory": "never_imported_for_protocol:make_env",
        "variants": [{"name": "mlp", "policy": {"architecture": "mlp", "history_length": 1}},
                     {"name": "history_mlp", "policy": {"architecture": "history_mlp", "history_latent_dim": 3}},
                     {"name": "transformer", "policy": {}}],
        "seeds": [71, 97, 101],
        "stages": [{"name": "first", "updates": 2, "environment": contracts["first"], "scenarios": ["normal"]},
                   {"name": "second", "updates": 3, "environment": contracts["second"], "scenarios": ["new_skill"]}],
        "scenarios": [{"name": name, "environment": contracts[name],
            "gates": [{"path": "success_rate", "operator": "min", "value": 1.}],
            "require_steady": True} for name in ("normal", "new_skill")],
        "training": {"rollout_steps": 4, "checkpoint_interval": 2, "max_seconds": 60.,
                     "retention_coef": 0., "anchor_seeds": [4101, 5101]},
        "evaluation": {"validation_seeds": [701, 1701], "seeds": [2701, 3701], "steps": 12,
                       "settle_steps": 1, "min_steady_samples": 1, "min_completed_episodes": 1},
        "execution": {"devices": ["cpu"], "worker_module": "transformer_rl.frame_process",
                      "job_timeout_seconds": 61.},
        "selection": {"min_training_seeds": 3, "objectives": [{"path": "metrics.error.mean",
            "direction": "minimize", "scale": 1., "weight": 1.}], "std_penalty": 1.,
            "latency_p99_ms": 9., "latency_max_ms": 10., "max_deadline_misses": 0,
            "retention_score_tolerance": .1, "rollback_limit": 1}}
    write_json(inputs / "base.json", base.to_dict())
    write_json(inputs / "study.json", spec)
    history = tmp_path / "history"
    history_study.prepare_history_study(inputs / "study.json", inputs / "base.json", history,
        history_lengths=[1, 3], position_reference="current")
    predecessor_root = tmp_path / "predecessors"
    predecessor_root.mkdir()
    shared, study = predecessor_root / "resource.lock", predecessor_root / "study.lock"
    shared.touch()
    study.touch()
    summaries = {}
    for role in ("curriculum", "diagnostics", "learning"):
        directory = predecessor_root / role
        directory.mkdir()
        write_json(directory / "definition.json", {"role": role, "fixture": "immutable definition"})
        write_json(directory / "summary.json", {"role": role, "status": "pending"})
        summaries[role] = directory / "summary.json"
    state = {"runtime_marker": "original", "dependency_marker": "original"}

    def runtime_provider(roots):
        if type(roots) is not list or not roots:
            raise ValueError("explicit external runtime roots required")
        roots = [protocol._path(root) for root in roots]
        if len(roots) != len(set(roots)) or any(a.is_relative_to(b) for i, a in enumerate(roots)
                for j, b in enumerate(roots) if i != j):
            raise ValueError("runtime roots must be distinct and must not overlap")
        return {"python_invocation": sys.executable, "python_version": sys.version,
                "files": {"python": protocol._receipt(Path(sys.executable).resolve())},
                "declared_sdk_trees": [protocol._tree(root) for root in sorted(roots)],
                "fixture_marker": state["runtime_marker"], "hardware_verified": False}

    def dependency_provider(curriculum, diagnostic, learning, resource_lock):
        expected = [summaries[role] for role in ("curriculum", "diagnostics", "learning")]
        assert [Path(curriculum), Path(diagnostic), Path(learning)] == expected
        assert Path(resource_lock) == shared
        pins = [protocol.predecessors._lock_signature(path) for path in sorted((shared, study))]
        shared_pin = next(pin for pin in pins if pin["path"] == str(shared))
        dependencies = []
        for role in ("curriculum", "diagnostics", "learning"):
            dependencies.append({"role": role, "summary_path": str(summaries[role]),
                "definition": protocol._receipt(summaries[role].parent / "definition.json"),
                "worker_roots": [str(summaries[role].parent)], "input_receipts": [],
                "locks": [shared_pin] if role == "curriculum" else deepcopy(pins),
                "controller": {"pid": 10, "start": 123}, "fixture_marker": state["dependency_marker"]})
        return dependencies, pins

    monkeypatch.setattr(protocol, "runtime_identity", runtime_provider)
    monkeypatch.setattr(protocol.predecessors, "_dependencies", dependency_provider)
    return {"tmp": tmp_path, "inputs": inputs, "history": history, "snapshot": snapshot,
            "sdk": sdk, "spec": spec, "base": base, "state": state, "summaries": summaries,
            "shared": shared, "study_lock": study, "dependency_provider": dependency_provider,
            "runtime_provider": runtime_provider, "output": tmp_path / "new_campaign"}


def freeze(prepared, **changes):
    arguments = {"output_root": prepared["output"], "retention_seed": 91001, "device": "cpu",
        "curriculum_summary": prepared["summaries"]["curriculum"],
        "diagnostic_summary": prepared["summaries"]["diagnostics"],
        "learning_summary": prepared["summaries"]["learning"],
        "resource_lock": prepared["shared"], "runtime_roots": [str(prepared["sdk"])],
        "worker_timeout_seconds": 61., "max_wait_seconds": 5., "poll_seconds": .1}
    arguments.update(changes)
    return protocol.freeze(prepared["history"], **arguments)


def test_real_complete_grid_initial_states_stage_and_evaluation_budgets_without_execution(prepared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("protocol preparation must not start training, an environment or a process")
    monkeypatch.setattr(frame_study, "_process", forbidden)
    monkeypatch.setattr(frame_study, "_run_job", forbidden)
    monkeypatch.setattr(frame_study, "run_study", forbidden)
    monkeypatch.setattr(exposure_training, "train_exposure_job", forbidden)
    monkeypatch.setattr(protocol.predecessors.subprocess, "Popen", forbidden)
    before_rng = torch.get_rng_state().clone()
    before_cuda = torch.cuda.is_initialized()
    before_source = experiments.source_identity()
    source_input_bytes = {path: path.read_bytes() for path in prepared["inputs"].iterdir()}
    result = freeze(prepared)
    assert protocol.validate_protocol(result) == result
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.cuda.is_initialized() == before_cuda
    assert experiments.source_identity() == before_source
    assert not prepared["output"].exists()
    assert not list((prepared["history"] / "study/jobs").iterdir())
    assert "never_imported_for_protocol" not in sys.modules
    assert all(path.read_bytes() == raw for path, raw in source_input_bytes.items())
    assert len(result["jobs"]) == 15
    assert len({job["id"] for job in result["jobs"]}) == 15
    assert {job["training_seed"] for job in result["jobs"]} == {71, 97, 101}
    assert {job["candidate"] for job in result["jobs"]} == {
        "mlp_h1", "history_mlp_h1", "history_mlp_h3", "transformer_h1", "transformer_h3"}
    assert result["budget"] == {"jobs": 15, "training_updates": 75, "fresh_transitions": 1260,
        "validation_cells": 120, "held_out_cells": 120, "evaluation_transition_upper": 12960}
    expected_cells = {(job["id"], stage, role, seed, case)
        for job in result["jobs"] for stage in (0, 1)
        for role, seeds in (("validation", (701, 1701)), ("held_out", (2701, 3701)))
        for seed in seeds for case in ("normal", "new_skill")}
    actual_cells = {(cell["job_id"], cell["stage_index"], cell["role"], cell["seed"], cell["scenario"])
                    for cell in result["evaluation_cells"]}
    assert len(result["evaluation_cells"]) == len({c["id"] for c in result["evaluation_cells"]}) == 240
    assert actual_cells == expected_cells
    for cell in result["evaluation_cells"]:
        count = 2 if cell["scenario"] == "normal" else 7
        assert cell["num_envs"] == count and cell["expected_policy_samples"] == 12 * count
        assert hashlib.sha256(Path(cell["config_receipt"]["path"]).read_bytes()).hexdigest() == cell["config_receipt"]["sha256"]
    for job in result["jobs"]:
        assert job["retention_seed"] == 91001
        assert job["reserved_updates"] == 5 and job["reserved_fresh_transitions"] == 84
        assert [s["index"] for s in job["stages"]] == [0, 1]
        assert [s["name"] for s in job["stages"]] == ["first", "second"]
        assert [s["updates"] for s in job["stages"]] == [2, 3]
        assert [s["fresh_transitions"] for s in job["stages"]] == [24, 60]
        assert [s["expected_cumulative_updates"] for s in job["stages"]] == [2, 5]
        assert [s["expected_cumulative_transitions"] for s in job["stages"]] == [24, 84]
        configs = [FrameTrainConfig.from_dict(s["config"]) for s in job["stages"]]
        assert configs[0].model == configs[1].model
        assert configs[0].ppo == configs[1].ppo and configs[0].control == configs[1].control
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(job["training_seed"])
            actual_initial_state = _model_state_sha256(FrameActorCritic(configs[0].model))
        assert job["initial_model_sha256"] == actual_initial_state
    assert result["policy"]["stage_execution"] == "one_OS_process_per_stage_full_learning_state_resume"
    assert result["policy"]["selection"] == "validation_only_then_immutable_exact_checkpoint_choice_then_held_out_confirmation"
    assert result["execution_status"] == "unexecuted"
    assert result["needs_fixed_authorization_OS_controller"]
    assert not result["independent_evaluation_performed"]
    assert not result["formal_architecture_selection"] and not result["hardware_verified"]


@pytest.mark.parametrize("change", [
    lambda p: p["jobs"].pop(),
    lambda p: p["jobs"].append(deepcopy(p["jobs"][0])),
    lambda p: p["jobs"][1].update(id=p["jobs"][0]["id"]),
    lambda p: p["jobs"][0].update(training_seed=97),
    lambda p: p["jobs"][0].update(initial_model_sha256="0" * 64),
    lambda p: p["jobs"][0].update(reserved_updates=1, reserved_fresh_transitions=12),
    lambda p: p["jobs"][0]["stages"][1].update(updates=1, fresh_transitions=20),
    lambda p: p["jobs"][0]["stages"][1].update(expected_cumulative_transitions=24),
    lambda p: p["jobs"][0]["stages"].pop(),
    lambda p: p["evaluation_cells"].pop(),
    lambda p: p["evaluation_cells"].append(deepcopy(p["evaluation_cells"][0])),
    lambda p: p["evaluation_cells"][0].update(role="held_out"),
    lambda p: p["evaluation_cells"][0].update(stage_index=1),
    lambda p: p["evaluation_cells"][0].update(expected_policy_samples=1),
    lambda p: p["budget"].update(jobs=1, training_updates=1, fresh_transitions=1),
    lambda p: p["budget"].update(validation_cells=1, held_out_cells=1, evaluation_transition_upper=1),
    lambda p: p.update(environment_factory="unauthorized:factory"),
    lambda p: p["policy"].update(selection="held_out_can_select_winner"),
    lambda p: p.update(caller_validator="anything:eligible", extra=True),
    lambda p: p.update(schema_version=True),
])
def test_resigning_jobs_full_matrix_or_budget_does_not_authorize_changes(prepared, change):
    original = freeze(prepared)
    changed = resign(original, change)
    assert changed["sha256"] == digest({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError):
        protocol.validate_protocol(changed)
    assert not prepared["output"].exists()


def test_coherent_resigned_reduced_job_and_evaluation_denominator_is_rejected(prepared):
    original = freeze(prepared)
    def shrink(p):
        p["jobs"] = p["jobs"][:3]
        ids = {job["id"] for job in p["jobs"]}
        p["evaluation_cells"] = [cell for cell in p["evaluation_cells"] if cell["job_id"] in ids]
        p["budget"] = {"jobs": len(p["jobs"]),
            "training_updates": sum(j["reserved_updates"] for j in p["jobs"]),
            "fresh_transitions": sum(j["reserved_fresh_transitions"] for j in p["jobs"]),
            "validation_cells": sum(c["role"] == "validation" for c in p["evaluation_cells"]),
            "held_out_cells": sum(c["role"] == "held_out" for c in p["evaluation_cells"]),
            "evaluation_transition_upper": sum(c["expected_policy_samples"] for c in p["evaluation_cells"])}
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(resign(original, shrink))


@pytest.mark.parametrize("seed", [True, -1, 2**32, 71, 97, 101, 701, 1701, 2701, 3701, 4101, 5101])
def test_private_seed_must_be_uint32_and_independent_of_all_declared_streams(prepared, seed):
    with pytest.raises(ValueError, match="private seed"):
        freeze(prepared, retention_seed=seed)
    assert not prepared["output"].exists()


@pytest.mark.parametrize("device", [True, None, "cuda", "cuda:00", "cuda:-1", "cuda:0", "CPU"])
def test_device_must_be_canonical_and_declared(prepared, device):
    with pytest.raises(ValueError, match="device"):
        freeze(prepared, device=device)
    assert not prepared["output"].exists()


@pytest.mark.parametrize("arguments", [
    {"worker_timeout_seconds": 60.}, {"worker_timeout_seconds": 59.},
    {"worker_timeout_seconds": True}, {"worker_timeout_seconds": float("inf")},
    {"max_wait_seconds": 0}, {"poll_seconds": -1},
    {"poll_seconds": 6.}, {"poll_seconds": 61., "max_wait_seconds": 90.},
])
def test_deadlines_and_observation_interval_are_explicit_and_bounded(prepared, arguments):
    with pytest.raises(ValueError):
        freeze(prepared, **arguments)
    assert not prepared["output"].exists()


@pytest.mark.parametrize("role", ["curriculum", "diagnostics", "learning"])
def test_actual_predecessor_definition_change_is_detected(prepared, role):
    original = freeze(prepared)
    definition = prepared["summaries"][role].parent / "definition.json"
    definition.write_bytes(definition.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(original)


@pytest.mark.parametrize("lock_name", ["shared", "study_lock"])
def test_original_lock_path_replacement_invalidates_protocol(prepared, lock_name):
    original = freeze(prepared)
    path = prepared[lock_name]
    before = path.stat().st_ino
    preserved = path.with_suffix(".old")
    path.rename(preserved)
    path.touch()
    assert path.stat().st_ino != before
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)


def test_resigned_disagreement_of_dependency_lock_identity_is_rejected(prepared):
    original = freeze(prepared)
    def alter(p):
        p["execution"]["dependencies"][1]["locks"][0]["inode"] += 1
    with pytest.raises(ValueError):
        protocol.validate_protocol(resign(original, alter))


def test_stale_lock_provider_pin_is_rejected_before_protocol_authorization(prepared, monkeypatch):
    actual = prepared["dependency_provider"]
    def stale(*args):
        dependencies, locks = actual(*args)
        locks[0]["inode"] += 1
        return dependencies, locks
    monkeypatch.setattr(protocol.predecessors, "_dependencies", stale)
    with pytest.raises(ValueError, match="inode"):
        freeze(prepared)


@pytest.mark.parametrize("kind", ["runtime", "snapshot", "history", "original_input", "dependency", "lock"])
def test_actual_inputs_changed_during_cpu_guard_construction_are_not_sealed(prepared, monkeypatch, kind):
    actual_guard = protocol.initial_model_sha256
    calls = []
    def guard(config, seed):
        actual = actual_guard(config, seed)
        calls.append((config.model.policy.architecture, seed))
        if len(calls) == 1:
            if kind == "runtime":
                path = prepared["sdk"] / "SDK_VERSION"
                path.write_text("changed while freezing\n")
            elif kind == "snapshot":
                path = prepared["snapshot"] / "environment.py"
                path.write_bytes(path.read_bytes() + b"changed while freezing\n")
            elif kind == "history":
                path = prepared["history"] / "study/configs/transformer_h3.train.second.json"
                path.write_bytes(path.read_bytes() + b" \n")
            elif kind == "original_input":
                path = prepared["inputs"] / "study.json"
                path.write_bytes(path.read_bytes() + b" \n")
            elif kind == "dependency":
                path = prepared["summaries"]["diagnostics"].parent / "definition.json"
                path.write_bytes(path.read_bytes() + b" \n")
            else:
                path = prepared["shared"]
                path.rename(path.with_suffix(".original"))
                path.touch()
        return actual
    monkeypatch.setattr(protocol, "initial_model_sha256", guard)
    with pytest.raises(ValueError):
        freeze(prepared)
    assert calls and not prepared["output"].exists()


@pytest.mark.parametrize("change", ["bytes", "added_file", "identity"])
def test_actual_runtime_provider_change_invalidates_frozen_protocol(prepared, change):
    original = freeze(prepared)
    if change == "bytes":
        (prepared["sdk"] / "SDK_VERSION").write_text("synthetic-cpu-sdk-v2\n")
    elif change == "added_file":
        (prepared["sdk"] / "extension.so").write_bytes(b"different declared SDK membership")
    else:
        prepared["state"]["runtime_marker"] = "changed-runtime-provider"
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(original)


def test_real_source_bytes_change_is_detected_without_modifying_shared_checkout(prepared, monkeypatch):
    original = freeze(prepared)
    copy = prepared["tmp"] / "changed_package"
    shutil.copytree(Path(experiments.__file__).resolve().parent, copy,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = copy / "frame_policy.py"
    path.write_bytes(path.read_bytes() + b"\n# synthetic changed source bytes\n")
    monkeypatch.setattr(experiments, "__file__", str(copy / "experiments.py"))
    assert experiments.source_identity() != original["source"]
    with pytest.raises(ValueError, match="source"):
        protocol.validate_protocol(original)


@pytest.mark.parametrize("route", ["history_plan.json", "expanded_spec.json", "inputs/spec.json",
    "study/plan.json", "study/configs/transformer_h3.train.second.json",
    "study/policy_source/transformer_rl/frame_policy.py"])
def test_actual_history_plan_or_frozen_member_bytes_change_is_detected(prepared, route):
    original = freeze(prepared)
    path = prepared["history"] / route
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)


@pytest.mark.parametrize("member", ["environment.py", "robot.bin", "contracts/first.json", "contracts/normal.json"])
def test_actual_environment_and_contract_bytes_change_is_detected(prepared, member):
    original = freeze(prepared)
    path = prepared["snapshot"] / member
    path.write_bytes(path.read_bytes() + b"changed input")
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)


@pytest.mark.parametrize("kind", ["undeclared_file", "undeclared_symlink", "declared_member_symlink",
                                   "pyc_symlink", "cache_directory_symlink", "git_directory_symlink"])
def test_extra_environment_members_and_symlinks_are_rejected(prepared, kind):
    original = freeze(prepared)
    if kind == "undeclared_file":
        (prepared["snapshot"] / "unexpected.txt").write_text("not declared by snapshot")
    elif kind == "undeclared_symlink":
        (prepared["snapshot"] / "unexpected.py").symlink_to(prepared["sdk"] / "loader.py")
    elif kind == "pyc_symlink":
        (prepared["snapshot"] / "ignored.pyc").symlink_to(prepared["sdk"] / "loader.py")
    elif kind == "cache_directory_symlink":
        (prepared["snapshot"] / "__pycache__").symlink_to(prepared["sdk"], target_is_directory=True)
    elif kind == "git_directory_symlink":
        (prepared["snapshot"] / ".git").symlink_to(prepared["sdk"], target_is_directory=True)
    else:
        path = prepared["snapshot"] / "environment.py"
        path.unlink()
        path.symlink_to(prepared["sdk"] / "loader.py")
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)


@pytest.mark.parametrize("root_name", ["inputs", "history", "snapshot", "sdk"])
def test_campaign_outputs_cannot_be_inside_original_inputs_or_sdk(prepared, root_name):
    destination = prepared[root_name] / "new_campaign"
    with pytest.raises(ValueError, match="overlap"):
        freeze(prepared, output_root=destination)
    assert not destination.exists()


def test_campaign_output_parent_cannot_contain_protected_trees(prepared):
    with pytest.raises(ValueError):
        freeze(prepared, output_root=prepared["tmp"], _allow_existing_output=True)


def test_existing_output_is_rejected_and_not_modified(prepared):
    prepared["output"].mkdir()
    sentinel = prepared["output"] / "sentinel.txt"
    sentinel.write_text("preserve caller output\n")
    with pytest.raises(ValueError, match="new output"):
        freeze(prepared)
    assert list(prepared["output"].iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve caller output\n"


def test_revalidation_can_inspect_existing_output_without_writing_or_reauthorizing_retry(prepared):
    original = freeze(prepared)
    prepared["output"].mkdir()
    sentinel = prepared["output"] / "status.json"
    sentinel.write_text('{"status":"partial"}\n')
    before = sentinel.read_bytes()
    assert protocol.validate_protocol(original) == original
    assert sentinel.read_bytes() == before and list(prepared["output"].iterdir()) == [sentinel]
    assert original["policy"]["reservation"] == "charge_full_job_before_first_child_no_refund"


def test_revalidation_refuses_a_regular_file_at_the_campaign_output_path(prepared):
    original = freeze(prepared)
    prepared["output"].write_bytes(b"existing caller file, not a campaign directory\n")
    before = prepared["output"].read_bytes()
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)
    assert prepared["output"].is_file() and prepared["output"].read_bytes() == before


@pytest.mark.parametrize("route", ["inputs", "history", "snapshot"])
def test_declared_runtime_tree_cannot_overlap_training_or_environment_inputs(prepared, route):
    with pytest.raises(ValueError, match="overlap"):
        freeze(prepared, runtime_roots=[str(prepared[route])])


def test_runtime_roots_must_be_nonempty_distinct_and_nonoverlapping(prepared):
    nested = prepared["sdk"] / "nested"
    nested.mkdir()
    (nested / "version").write_text("nested sdk\n")
    for roots in ([], [str(prepared["sdk"])] * 2, [str(prepared["sdk"]), str(nested)]):
        with pytest.raises(ValueError):
            freeze(prepared, runtime_roots=roots)


@pytest.mark.parametrize("route", ["sdk_alias", "snapshot_alias", "output_alias"])
def test_symlinked_roots_are_not_treated_as_canonical_inputs(prepared, route):
    alias = prepared["tmp"] / route
    if route == "sdk_alias":
        alias.symlink_to(prepared["sdk"], target_is_directory=True)
        changes = {"runtime_roots": [str(alias)]}
    elif route == "snapshot_alias":
        alias.symlink_to(prepared["snapshot"], target_is_directory=True)
        changes = {"output_root": alias / "new_campaign"}
    else:
        alias.symlink_to(prepared["tmp"], target_is_directory=True)
        changes = {"output_root": alias / "new_campaign"}
    with pytest.raises(ValueError):
        freeze(prepared, **changes)


def test_existing_history_job_artifacts_require_a_new_preparation(prepared):
    path = prepared["history"] / "study/jobs" / "existing_attempt.json"
    write_json(path, {"status": "partial; do not reuse this H preparation"})
    with pytest.raises(ValueError, match="new unexecuted"):
        freeze(prepared)


def test_nonzero_anchor_retention_recipe_is_not_reinterpreted_as_architecture_exposure(prepared):
    path = prepared["inputs"] / "study.json"
    spec = json.loads(path.read_text())
    spec["training"]["retention_coef"] = .2
    write_json(path, spec)
    history = prepared["tmp"] / "retention_history"
    history_study.prepare_history_study(path, prepared["inputs"] / "base.json", history,
        history_lengths=[1, 3], position_reference="current")
    changed = {**prepared, "history": history}
    with pytest.raises(ValueError, match="lambda zero"):
        freeze(changed)
    assert not prepared["output"].exists()


@pytest.mark.parametrize("suffix", [".txt", ".pyc"])
def test_special_input_file_is_rejected_before_cache_exclusion_or_blocking_read(prepared, suffix):
    original = freeze(prepared)
    path = prepared["snapshot"] / ("unexpected_fifo" + suffix)
    os.mkfifo(path)
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)
    assert path.exists() and not path.is_file()


@pytest.mark.parametrize("directory,kind", [
    ("__pycache__", "symlink"), ("__pycache__", "fifo"),
    (".git", "symlink"), (".git", "fifo"),
])
def test_excluded_subtree_descendants_still_require_regular_nonsymlink_input_types(prepared, directory, kind):
    original = freeze(prepared)
    cache = prepared["snapshot"] / directory / "nested"
    cache.mkdir(parents=True)
    path = cache / "ignored.pyc"
    if kind == "symlink":
        path.symlink_to(prepared["sdk"] / "loader.py")
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError):
        protocol.validate_protocol(original)


def test_regular_rebuildable_cache_bytes_are_excluded_without_waiving_input_type_guards(prepared):
    original = freeze(prepared)
    for root in (prepared["snapshot"], prepared["sdk"]):
        for directory in ("__pycache__", ".git"):
            cache = root / directory / "nested"
            cache.mkdir(parents=True)
            (cache / "ignored.pyc").write_bytes(b"rebuildable regular cache payload")
        (root / "ignored.pyc").write_bytes(b"regular bytecode placeholder")
    assert protocol.validate_protocol(original) == original


def cli_freeze_arguments(prepared, destination):
    return ["freeze", "--history-root", str(prepared["history"]),
        "--output-root", str(prepared["output"]), "--protocol-output", str(destination),
        "--curriculum-summary", str(prepared["summaries"]["curriculum"]),
        "--diagnostic-summary", str(prepared["summaries"]["diagnostics"]),
        "--learning-summary", str(prepared["summaries"]["learning"]),
        "--resource-lock", str(prepared["shared"]), "--runtime-roots", str(prepared["sdk"]),
        "--retention-seed", "91001", "--device", "cpu", "--worker-timeout-seconds", "61",
        "--max-wait-seconds", "5", "--poll-seconds", ".1"]


@pytest.mark.parametrize("root_name", ["inputs", "history", "snapshot", "sdk", "output"])
def test_cli_protocol_publication_cannot_enter_protected_inputs_or_future_job_output(prepared, capsys, root_name):
    destination = prepared[root_name] / "new_protocol.json"
    with pytest.raises(SystemExit) as caught:
        protocol.main(cli_freeze_arguments(prepared, destination))
    assert caught.value.code == 1
    assert "overlap" in capsys.readouterr().err
    assert not destination.exists() and not prepared["output"].exists()


def test_cli_protocol_publication_never_overwrites_an_existing_unprotected_file(prepared, capsys):
    destination = prepared["tmp"] / "existing_report.json"
    destination.write_bytes(b"caller report must survive\n")
    before = destination.read_bytes()
    with pytest.raises(SystemExit) as caught:
        protocol.main(cli_freeze_arguments(prepared, destination))
    assert caught.value.code == 1 and capsys.readouterr().err
    assert destination.read_bytes() == before and not prepared["output"].exists()


def test_cli_protocol_publication_does_not_create_missing_parent_directories(prepared, capsys):
    destination = prepared["tmp"] / "missing_parent" / "new_protocol.json"
    with pytest.raises(SystemExit) as caught:
        protocol.main(cli_freeze_arguments(prepared, destination))
    assert caught.value.code == 1
    assert "parent" in capsys.readouterr().err
    assert not destination.parent.exists() and not prepared["output"].exists()


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_ambiguous_nonfinite_or_nonobject_input_json_is_rejected(tmp_path, raw):
    path = tmp_path / "ambiguous.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        protocol._read(path)

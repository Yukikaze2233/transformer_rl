"""History study integrity and real packed configuration/model interface checks."""
import hashlib
import json
import fcntl
import os
from pathlib import Path
import py_compile

import pytest
import torch

from transformer_rl import frame_study, history_study
from transformer_rl.config import PPOConfig
from transformer_rl.frame_config import FrameModelConfig, FrameTrainConfig, digest
from transformer_rl.frame_policy import FramePolicy, FramePolicyConfig


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    control = {"policy_dt_s": .01, "observation_schema": "tensor_fixture",
        "feature_names": [f"feature_{i}" for i in range(5)], "action_names": ["leg", "wheel"],
        "action_bounds": [.2, .5], "target_scale": [.25, 10.], "target_offset": [.1, -.2],
        "target_units": ["rad", "rad/s"]}
    policy = FramePolicyConfig(architecture="transformer", frame_dim=5, action_dim=2,
        history_length=4, actor_hidden_dims=(12, 8), encoder_hidden_dims=(12,),
        d_model=8, num_heads=2, num_layers=1, ffn_dim=16, mean_init_scale=.2)
    base = FrameTrainConfig(FrameModelConfig(policy, critic_dim=3, critic_hidden=(12, 8),
        command_indices=(0,), initial_std=.8),
        PPOConfig(epochs=2, num_minibatches=2, learning_rate=3e-5, target_kl=.01),
        control, {"num_envs": 3, "physics_dt_s": .005, "recipe": "fixed"})
    spec = {"base_config": "base.json", "environment_factory": "never_imported:make_env",
        "variants": [{"name": "mlp", "policy": {"architecture": "mlp", "history_length": 1}},
            {"name": "mlp_wide", "policy": {"architecture": "mlp", "history_length": 1,
                                            "actor_hidden_dims": [16, 12, 8]}},
            {"name": "history_mlp", "policy": {"architecture": "history_mlp", "history_latent_dim": 3}},
            {"name": "transformer", "policy": {}},
            {"name": "gated", "policy": {"residual_type": "gated"}},
            {"name": "query", "policy": {"readout_type": "query"}}],
        "seeds": [71, 97, 101],
        "stages": [{"name": "first", "updates": 2, "environment": {}, "scenarios": ["normal"]},
                   {"name": "second", "updates": 3, "environment": {"num_envs": 5, "stage": "second"},
                    "scenarios": ["new_skill"]}],
        "scenarios": [{"name": name, "environment": {"case": name, "num_envs": count},
            "gates": [{"path": "success_rate", "operator": "min", "value": 1.}],
            "require_steady": True} for name, count in (("normal", 2), ("new_skill", 7))],
        "training": {"rollout_steps": 4, "checkpoint_interval": 2, "max_seconds": 60.,
                     "retention_coef": .2, "anchor_seeds": [4101], "max_anchors": 13},
        "evaluation": {"validation_seeds": [701], "seeds": [2701, 2801], "steps": 12,
                       "settle_steps": 1, "min_steady_samples": 1, "min_completed_episodes": 6},
        "execution": {"devices": ["cpu"], "worker_module": "transformer_rl.frame_process",
                      "job_timeout_seconds": 60.},
        "selection": {"min_training_seeds": 3, "objectives": [{"path": "metrics.error.mean",
            "direction": "minimize", "scale": 1., "weight": 1.}], "std_penalty": 1.,
            "latency_p99_ms": 9., "latency_max_ms": 10., "max_deadline_misses": 0,
            "retention_score_tolerance": .1, "rollback_limit": 1}}
    (directory / "base.json").write_text(json.dumps(base.to_dict(), indent=3) + "\n\n")
    (directory / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    return directory / "spec.json", directory / "base.json", tmp_path / "study"


def prepare(inputs, **options):
    return history_study.prepare_history_study(*inputs,
        history_lengths=options.get("history_lengths", [1, 3, 7]),
        position_reference=options.get("position_reference", "current"))


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value) + "\n")


def reseal(path, change):
    value = json.loads(path.read_text())
    change(value)
    value.pop("sha256")
    value["sha256"] = digest(value)
    path.write_text(json.dumps(value) + "\n")


def test_complete_grid_real_study_interface_budget_and_no_execution(inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not execute a training job")
    monkeypatch.setattr(frame_study, "_process", forbidden)
    monkeypatch.setattr(frame_study, "_run_job", forbidden)
    spec_path, base_path, root = inputs
    original = {p: p.read_bytes() for p in (spec_path, base_path)}
    original_source = frame_study.source_identity()
    planner_bytes = Path(history_study.__file__).read_bytes()
    result = prepare(inputs)
    assert history_study.validate_history_study(root) == result
    frozen = frame_study.validate_study(root / "study", source=True)
    spec, base = json.loads(original[spec_path]), FrameTrainConfig.load(base_path)
    assert len(result["candidates"]) == 14
    assert {c["history_length"] for c in result["candidates"] if c["architecture"] == "mlp"} == {1}
    for candidate in result["candidates"]:
        variant = next(v for v in spec["variants"] if v["name"] == candidate["source_variant"])
        expected = base.with_policy(variant["policy"]).model.policy.to_dict()
        expected["history_length"] = candidate["history_length"]
        if candidate["architecture"] == "transformer":
            expected["position_reference"] = "current"
        assert candidate["policy"] == expected
        assert candidate["span_s"] == pytest.approx((candidate["history_length"] - 1) * .01)
        assert candidate["minimum_full_history_episode_age"] == candidate["history_length"] - 1
        for kind, entries in (("train", spec["stages"]), ("eval", spec["scenarios"])):
            for entry in entries:
                config = FrameTrainConfig.load(root / "study/configs" / f"{candidate['name']}.{kind}.{entry['name']}.json")
                value, reference = config.to_dict(), base.to_dict()
                reference["model"]["policy"] = expected
                reference["environment"].update(entry["environment"])
                assert value == reference
    for name in set(spec) - {"base_config", "variants"}:
        assert frozen["spec"][name] == spec[name]
    budget = result["budget"]
    assert budget["job_count"] == 42 and budget["updates_per_job_upper"] == 5
    assert budget["updates_all_jobs_upper"] == 210
    assert budget["fresh_transitions_all_jobs_upper"] == 3528
    assert budget["final_held_out_evaluation_cells"] == 168
    assert budget["final_evaluation_transition_upper"] == 9072
    assert [s["num_envs"] for s in budget["stages"]] == [3, 5]
    assert [s["held_out_evaluation_cells_at_stage_endpoint"] for s in budget["stages"]] == [84, 168]
    assert [s["validation_cells_per_checkpoint"] for s in budget["stages"]] == [42, 84]
    assert [s["validation_checkpoints_per_job_if_full_chunks"] for s in budget["stages"]] == [1, 2]
    assert [s["anchor_cells_at_promotion"] for s in budget["stages"]] == [42, 84]
    assert result["execution_semantics"]["needs_independent_equal_exposure_executor"]
    assert not result["execution_semantics"]["preparation_started_training"]
    assert result["execution_state"] == "unobserved"
    assert not result["execution_semantics"]["equal_exposure_comparison_ready"]
    assert not list((root / "study/jobs").iterdir())
    for role, path in (("spec", spec_path), ("base", base_path)):
        assert result["inputs"][role]["sha256"] == hashlib.sha256(original[path]).hexdigest()
        assert (root / f"inputs/{role}.json").read_bytes() == original[path]
        assert path.read_bytes() == original[path]
    assert Path(history_study.__file__).read_bytes() == planner_bytes
    assert frame_study.source_identity() == original_source
    assert result["planner_source"]["sha256"] == hashlib.sha256(planner_bytes).hexdigest()
    assert (root / "study/policy_source/transformer_rl/history_study.py").read_bytes() == planner_bytes


def test_models_accept_complete_grid_and_align_current_frame_ages(inputs):
    result = prepare(inputs)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        encodings, parameter_counts = {}, {}
        for candidate in result["candidates"]:
            config = FramePolicyConfig.from_dict(candidate["policy"])
            model = FramePolicy(config)
            frames = torch.randn(2, config.history_length, config.frame_dim, requires_grad=True)
            action = model(frames)
            assert action.shape == (2, 2) and torch.isfinite(action).all()
            action.square().mean().backward()
            assert frames.grad is not None and torch.isfinite(frames.grad).all()
            parameter_counts[candidate["name"]] = sum(p.numel() for p in model.parameters())
            if config.architecture == "transformer":
                if candidate["source_variant"] not in encodings:
                    encodings[candidate["source_variant"]] = model.position_encoding.detach().clone()
                else:
                    short = encodings[candidate["source_variant"]]
                    assert torch.equal(model.position_encoding[:, -short.shape[1]:], short)
        assert parameter_counts["transformer_h1"] == parameter_counts["transformer_h3"] == parameter_counts["transformer_h7"]
        assert parameter_counts["history_mlp_h1"] < parameter_counts["history_mlp_h3"] < parameter_counts["history_mlp_h7"]
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("lengths", ([1], [3, 7], [1, 1, 3], [1, 0], [1, -2], [True, 3], [1, 3.], "1,3"))
def test_invalid_length_grids_are_rejected_before_writes(inputs, lengths):
    with pytest.raises(ValueError, match="history_lengths"):
        prepare(inputs, history_lengths=lengths)
    assert not inputs[2].exists()


@pytest.mark.parametrize("position", ("oldest", None, ""))
def test_current_time_reference_is_mandatory(inputs, position):
    with pytest.raises(ValueError, match="explicit current"):
        prepare(inputs, position_reference=position)
    assert not inputs[2].exists()


@pytest.mark.parametrize("change", (
    lambda s: s.update(seeds=[71, 97]),
    lambda s: s["evaluation"].update(seeds=[2701]),
    lambda s: s["selection"].update(min_training_seeds=2),
    lambda s: s["selection"].update(min_training_seeds=4),
    lambda s: s["training"].update(anchor_seeds=[71]),
    lambda s: s["evaluation"].update(validation_seeds=[2701]),
    lambda s: s["training"].update(anchor_seeds=[]),
    lambda s: s["evaluation"].pop("validation_seeds"),
    lambda s: s["variants"][0]["policy"].update(architecture="rnn"),
    lambda s: s["variants"][0]["policy"].update(architecture="frame_stack_mlp"),
    lambda s: s.update(variants=[v for v in s["variants"] if v["name"] != "history_mlp"]),
    lambda s: s["variants"][0]["policy"].update(mean_init_scale=.5),
    lambda s: s["variants"][0]["policy"].update(frame_dim=6),
))
def test_seed_independence_complete_families_and_recipe_invariants(inputs, change):
    edit(inputs[0], change)
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("count", (None, 0, -1, True, 3., "3"))
def test_num_envs_is_not_assumed(inputs, count):
    edit(inputs[1], lambda b: b["environment"].update(num_envs=count))
    with pytest.raises(ValueError, match="num_envs"):
        prepare(inputs)
    assert not inputs[2].exists()


def test_missing_eval_environment_count_is_rejected(inputs):
    edit(inputs[0], lambda s: s["scenarios"][0]["environment"].update(num_envs=None))
    with pytest.raises(ValueError, match="num_envs"):
        prepare(inputs)


def test_missing_num_envs_is_rejected_without_implicit_defaults(inputs):
    edit(inputs[1], lambda b: b["environment"].pop("num_envs"))
    with pytest.raises(ValueError, match="num_envs"):
        prepare(inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("route", ("expanded_spec.json", "inputs/spec.json", "inputs/base.json",
    "study/configs/gated_h3.train.first.json", "study/plan.json",
    "study/policy_source/transformer_rl/history_study.py"))
def test_frozen_real_artifact_bytes_detect_tamper(inputs, route):
    prepare(inputs)
    path = inputs[2] / route
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError):
        history_study.validate_history_study(inputs[2])


def test_resigning_budget_or_inventory_cannot_replace_actual_sources(inputs):
    prepare(inputs)
    manifest = inputs[2] / "history_plan.json"
    reseal(manifest, lambda m: m["budget"].update(fresh_transitions_all_jobs_upper=1))
    with pytest.raises(ValueError, match="definition"):
        history_study.validate_history_study(inputs[2])


def test_resigned_inventory_still_requires_every_expected_file(inputs):
    prepare(inputs)
    route = "inputs/spec.json"
    (inputs[2] / route).unlink()
    reseal(inputs[2] / "history_plan.json", lambda m: m["files"].pop(route))
    with pytest.raises((ValueError, OSError)):
        history_study.validate_history_study(inputs[2])


def test_validator_rereads_original_input_bytes(inputs):
    prepare(inputs)
    inputs[1].write_bytes(inputs[1].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="original inputs"):
        history_study.validate_history_study(inputs[2])


def test_new_unlisted_artifact_is_rejected(inputs):
    prepare(inputs)
    (inputs[2] / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        history_study.validate_history_study(inputs[2])


def test_required_empty_job_directory_is_not_optional(inputs):
    prepare(inputs)
    (inputs[2] / "study/jobs").rmdir()
    with pytest.raises(ValueError, match="required study directory"):
        history_study.validate_history_study(inputs[2])


def test_job_artifacts_do_not_mutate_the_plan_or_prove_unexecuted_training(inputs, capsys):
    before = prepare(inputs)
    original_bytes = (inputs[2] / "history_plan.json").read_bytes()
    job = inputs[2] / "study/jobs/gated_h3/seed_71"
    job.mkdir(parents=True)
    (job / "state.json").write_text('{"status":"running"}\n')
    result = history_study.validate_history_study(inputs[2])
    assert result["sha256"] == before["sha256"]
    assert result["execution_state"] == "job_artifacts_present"
    assert "training_executed" not in result and "training_executed" not in result["execution_semantics"]
    assert result["execution_semantics"]["preparation_started_training"] is False
    assert (inputs[2] / "history_plan.json").read_bytes() == original_bytes
    assert history_study.main(["validate", "--output-root", str(inputs[2])]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["execution_state"] == "job_artifacts_present"
    assert "training_executed" not in observed and observed["preparation_started_training"] is False


def test_original_runner_lock_and_real_source_bytecode_preserve_static_validation(inputs):
    before = prepare(inputs)
    original_bytes = (inputs[2] / "history_plan.json").read_bytes()
    study_root = inputs[2] / "study"
    source = study_root / "policy_source/transformer_rl/frame_policy.py"
    bytecode = Path(py_compile.compile(str(source), doraise=True))
    assert bytecode.is_file() and "__pycache__" in bytecode.parts
    # The original runner uses this exact reusable lock path and flock mode.
    with (study_root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frame_study.validate_study(study_root, source=True)
        result = history_study.validate_history_study(inputs[2])
        assert result["sha256"] == before["sha256"]
        assert result["execution_state"] == "unobserved"
        assert "training_executed" not in result
        assert result["execution_semantics"]["preparation_started_training"] is False
    assert (inputs[2] / "history_plan.json").read_bytes() == original_bytes
    assert history_study.validate_history_study(inputs[2])["sha256"] == before["sha256"]


@pytest.mark.parametrize("route", ("outside.pyc", "__pycache__/fake.pyc",
    "study/configs/__pycache__/fake.pyc", "study/.run.lock.extra"))
def test_runtime_exclusion_does_not_hide_foreign_or_configuration_files(inputs, route):
    prepare(inputs)
    path = inputs[2] / route
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="inventory"):
        history_study.validate_history_study(inputs[2])


@pytest.mark.parametrize("route", ("study/.run.lock",
    "study/policy_source/transformer_rl/__pycache__/frame_policy.pyc"))
def test_ignored_runtime_paths_still_reject_symlinks(inputs, route):
    prepare(inputs)
    path = inputs[2] / route
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(inputs[0])
    with pytest.raises(ValueError, match="symlinks"):
        history_study.validate_history_study(inputs[2])


def test_ignored_lock_path_still_rejects_unsupported_artifacts(inputs):
    prepare(inputs)
    os.mkfifo(inputs[2] / "study/.run.lock")
    with pytest.raises(ValueError, match="unsupported artifact"):
        history_study.validate_history_study(inputs[2])


def test_partial_plan_is_never_ready_or_overwritten(inputs, monkeypatch):
    real = frame_study.plan_study
    def interrupted(*args):
        real(*args)
        raise RuntimeError("simulated preparation interruption")
    monkeypatch.setattr(frame_study, "plan_study", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        prepare(inputs)
    assert inputs[2].is_dir() and not (inputs[2] / "history_plan.json").exists()
    with pytest.raises(ValueError, match="partial"):
        history_study.validate_history_study(inputs[2])
    with pytest.raises(ValueError, match="output must be new"):
        prepare(inputs)


def test_output_cannot_overlap_inputs_or_replace_existing_directory(inputs):
    sentinel = inputs[0].parent / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="overlap"):
        prepare((inputs[0], inputs[1], inputs[0].parent / "new"))
    inputs[2].mkdir()
    (inputs[2] / "keep").write_text("untouched")
    with pytest.raises(ValueError, match="output must be new"):
        prepare(inputs)
    assert (inputs[2] / "keep").read_text() == "untouched" and sentinel.read_text() == "keep"


def test_source_tree_cannot_be_output(inputs):
    with pytest.raises(ValueError, match="overlap"):
        prepare((inputs[0], inputs[1], Path(history_study.__file__).parent / "new_history_output"))


def test_explicit_base_must_match_original_spec(inputs):
    other = inputs[1].parent / "other.json"
    other.write_bytes(inputs[1].read_bytes())
    with pytest.raises(ValueError, match="explicit base path"):
        prepare((inputs[0], other, inputs[2]))


@pytest.mark.parametrize("kind", ("input", "output_parent", "frozen_file"))
def test_symlinks_are_rejected(inputs, kind):
    if kind == "input":
        linked = inputs[0].parent / "linked.json"
        linked.symlink_to(inputs[0])
        with pytest.raises(ValueError, match="symlinks"):
            prepare((linked, inputs[1], inputs[2]))
    elif kind == "output_parent":
        linked = inputs[2].parent / "linked"
        linked.symlink_to(inputs[2].parent, target_is_directory=True)
        with pytest.raises(ValueError, match="symlinks"):
            prepare((inputs[0], inputs[1], linked / "new"))
    else:
        prepare(inputs)
        path = inputs[2] / "inputs/spec.json"
        path.unlink()
        path.symlink_to(inputs[0])
        with pytest.raises(ValueError, match="symlinks"):
            history_study.validate_history_study(inputs[2])


def test_no_retention_budget_is_invented(inputs):
    edit(inputs[0], lambda s: s["training"].update(retention_coef=0))
    result = prepare(inputs)
    assert all(s["anchor_cells_at_promotion"] == 0 for s in result["budget"]["stages"])
    assert result["budget"]["anchor_seeds"] == [4101]


def test_cli_only_prepares_or_validates_and_requires_explicit_options(inputs, capsys):
    with pytest.raises(SystemExit) as error:
        history_study.main(["run"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        history_study.main(["prepare", "--spec", str(inputs[0]), "--base-config", str(inputs[1]),
                            "--output-root", str(inputs[2])])
    assert error.value.code == 2
    assert history_study.main(["prepare", "--spec", str(inputs[0]), "--base-config", str(inputs[1]),
        "--output-root", str(inputs[2]), "--history-lengths", "1", "3", "7", "--position-reference", "current"]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["preparation_started_training"] is False and prepared["budget"]["candidate_count"] == 14
    assert prepared["execution_state"] == "unobserved" and "training_executed" not in prepared
    assert history_study.main(["validate", "--output-root", str(inputs[2])]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated == prepared


def test_duplicate_json_input_fields_are_rejected_before_writes(inputs):
    inputs[0].write_bytes(b'{"base_config":"base.json","base_config":"other.json"}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        prepare(inputs)
    assert not inputs[2].exists()

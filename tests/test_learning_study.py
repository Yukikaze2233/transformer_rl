"""File/CPU checks for preparation, never simulator or robot PPO execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from transformer_rl import learning_study as learning
from transformer_rl import transfer_profiles as real_profiles
from transformer_rl.frame_config import digest
from transformer_rl.transfer_study import prepare_transfer_study
from test_transfer_study import source as transfer_source


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def resign(root, manifest):
    manifest["sha256"] = digest({key: value for key, value in manifest.items() if key != "sha256"})
    write(root / "manifest.json", manifest)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    directory = tmp_path_factory.mktemp("learning-template")
    with pytest.MonkeyPatch.context() as patch:
        source, task, parent = transfer_source.__wrapped__(directory, patch)
        patch.setitem(sys.modules, "transformer_rl.transfer_profiles", real_profiles)
        parent["evaluation"]["cases"] = [
            {"name": name, "task": "survive", "terrain": "flat"}
            for name in real_profiles.REPRESENTATIVE_CASES]
        write(task, parent)
        prepared = directory / "prepared"
        prepare_transfer_study(source, task, prepared, num_envs=1024, evaluation_replicas=8,
                               updates=1200, seeds=[1101])
    return prepared


@pytest.fixture
def parent(template, tmp_path):
    root = tmp_path / "parent"
    shutil.copytree(template, root)
    base = learning._read(root / "base.json")
    base["environment"]["snapshot"] = str(root / "snapshot")
    write(root / "base.json", base)
    return root


@pytest.fixture
def sealed(parent, tmp_path):
    root = tmp_path / "prepared"
    learning.prepare_learning_study(parent, root)
    return root


def test_complete_grid_preserves_parent_and_declares_only_preparation(parent, tmp_path):
    before = {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    state = torch.get_rng_state().clone()
    root = tmp_path / "prepared"
    result = learning.prepare_learning_study(parent, root)
    assert torch.equal(state, torch.get_rng_state())
    assert result["status"] == "prepared" and result["expected_training_jobs"] == 90
    assert result["initial_models"] == 30 and result["planned_total_transitions"] == 5_308_416_000
    assert result["training_started"] is result["queued"] is result["execution_implemented"] is False
    assert result["selection_implemented"] is result["formal_architecture_selection"] is False
    manifest = learning._read(root / "manifest.json")
    assert "winner" not in manifest and "eligible" not in manifest
    assert len(manifest["cells"]) == 90 and len(manifest["initializations"]) == 30
    assert len({(cell["variant"], cell["training_seed"], cell["learning_rate"]) for cell in manifest["cells"]}) == 90
    assert [child["id"] for child in manifest["children"]] == ["rate_000", "rate_001", "rate_002"]
    assert len(list(root.rglob("snapshot"))) == 1
    for initialization in manifest["initializations"]:
        cells = [cell for cell in manifest["cells"] if cell["variant"] == initialization["variant"]
                 and cell["training_seed"] == initialization["training_seed"]]
        assert len(cells) == 3
        assert {cell["initial_model_sha256"] for cell in cells} == {initialization["initial_model_sha256"]}
    assert before == {str(path.relative_to(parent)): path.read_bytes() for path in parent.rglob("*") if path.is_file()}
    assert "learning_study.py" in manifest["source"]["files"]
    assert learning.validate_learning_study(root)["sha256"] == result["sha256"]


def test_all_train_and_eval_pairs_differ_only_in_ppo_rate(sealed):
    manifest = learning._read(sealed / "manifest.json")
    normalized = {}
    for child in manifest["children"]:
        for receipt in child["configurations"]:
            config = learning._read(sealed / receipt["path"])
            route = receipt["path"].split("/study/", 1)[1]
            assert config["ppo"]["learning_rate"] == child["learning_rate"]
            assert config["environment"]["snapshot"] == str(sealed / "snapshot")
            model = config["model"]
            assert model["initial_std"] == model["policy"]["mean_init_scale"] == 1.
            assert model["policy"].get("position_reference", "oldest") == "oldest"
            assert model["policy"]["history_length"] in (1, 31)
            projected = learning._without_rate(config)
            if route in normalized:
                assert projected == normalized[route]
            normalized[route] = projected
    assert len(normalized) == 510
    native = learning._read(sealed / "snapshot/contracts/transfer.train.json")
    assert native["learning_rate"] == 3e-5


def test_development_and_confirmation_are_disjoint_and_scope_is_noise_only(sealed):
    manifest = learning._read(sealed / "manifest.json")
    protocol = manifest["protocol"]
    confirmation = protocol["confirmation"]
    assert confirmation["noise_seeds"] == [11701, 12701]
    assert confirmation["scope"] == "heldout_noise_stream_only"
    assert confirmation["new_initial_conditions"] is confirmation["new_perturbation_domains"] is False
    assert confirmation["deterministic_cases"] == 38 and confirmation["stochastic_noise_or_combined_cases"] == 12
    for child in manifest["children"]:
        spec = learning._read(sealed / child["spec"]["path"])
        used = set(spec["seeds"] + spec["evaluation"]["seeds"] + spec["evaluation"]["validation_seeds"])
        assert not used.intersection(confirmation["noise_seeds"])
    assert protocol["future_selection"]["no_eligible_candidate"] == "no_winner"
    assert protocol["future_selection"]["implemented"] is False
    assert protocol["future_runtime_checks"]["implemented"] is False


@pytest.mark.parametrize("kwargs", [
    {"learning_rates": [True, 3e-5, 1e-4]}, {"learning_rates": [1e-5, 1e-5, 1e-4]},
    {"learning_rates": [1e-5, float("nan"), 1e-4]}, {"learning_rates": [1e-5, 0., 1e-4]},
    {"training_seeds": [True, 1102, 1103]}, {"training_seeds": [1101, 1101, 1103]},
    {"training_seeds": [1101, 1102]}, {"training_seeds": [1101, 1102, 11701]},
    {"confirmatory_noise_seeds": [701, 12701]}, {"confirmatory_noise_seeds": [11701, 11701]},
    {"confirmatory_noise_seeds": [True, 12701]},
])
def test_invalid_rates_or_seed_pools_fail_before_creating_output(parent, tmp_path, kwargs):
    root = tmp_path / "bad"
    with pytest.raises(ValueError):
        learning.prepare_learning_study(parent, root, **kwargs)
    assert not root.exists()


@pytest.mark.parametrize("section,key,value", [
    ("ppo", "gamma", .99), ("ppo", "gae_lambda", .5), ("ppo", "entropy_coef", .5),
    ("ppo", "normalize_advantage", False), ("model", "command_indices", [0, 1]),
])
def test_changed_parent_recipe_is_not_accepted_as_lr_only(parent, tmp_path, section, key, value):
    base = learning._read(parent / "base.json")
    base[section][key] = value
    write(parent / "base.json", base)
    root = tmp_path / "bad"
    with pytest.raises(ValueError, match="recipe"):
        learning.prepare_learning_study(parent, root)
    assert not root.exists()


@pytest.mark.parametrize("section,key,value", [
    ("selection", "latency_p99_ms", 9.), ("execution", "worker_module", "another.worker"),
    ("training", "checkpoint_interval", 100),
])
def test_parent_execution_and_validation_cannot_be_silently_rewritten(parent, tmp_path, section, key, value):
    spec = learning._read(parent / "study.json")
    spec[section][key] = value
    write(parent / "study.json", spec)
    with pytest.raises(ValueError, match="recipe|protocol"):
        learning.prepare_learning_study(parent, tmp_path / "bad")


def test_missing_pure_dependency_and_extra_snapshot_file_are_rejected(parent, tmp_path):
    dependency = parent / "snapshot/contracts/lookup.json"
    data = dependency.read_bytes()
    dependency.unlink()
    with pytest.raises(ValueError, match="inventory"):
        learning.prepare_learning_study(parent, tmp_path / "missing")
    dependency.write_bytes(data)
    (parent / "snapshot/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        learning.prepare_learning_study(parent, tmp_path / "extra")


def test_snapshot_identity_cannot_override_its_own_sha(parent, tmp_path):
    path = parent / "snapshot/snapshot.json"
    identity = learning._read(path)
    identity["files"]["snapshot.json"] = "0" * 64
    identity["sha256"] = digest(identity["files"])
    write(path, identity)
    base = learning._read(parent / "base.json")
    base["environment"]["snapshot_sha256"] = identity["sha256"]
    write(parent / "base.json", base)
    with pytest.raises(ValueError, match="exclude itself"):
        learning.prepare_learning_study(parent, tmp_path / "bad")


@pytest.mark.parametrize("mutation", ["task_sha", "scene_counts", "cases", "delay_clock", "height_schedule"])
def test_parent_design_must_match_authenticated_contracts(parent, tmp_path, mutation):
    design = learning._read(parent / "transfer_design.json")
    if mutation == "task_sha":
        design["parent"]["task_contract_sha256"] = "0" * 64
    elif mutation == "scene_counts":
        design["scene_allocation"]["requested_counts"] = {"other": 1024}
    elif mutation == "cases":
        design["cases"] = []
    elif mutation == "delay_clock":
        design["delay_schedule"]["research"]["start"] = 0
    else:
        design["height_schedule"]["research"]["actor_update_boundary"] = 0
    write(parent / "transfer_design.json", design)
    with pytest.raises(ValueError, match="design"):
        learning.prepare_learning_study(parent, tmp_path / "bad")


def test_input_overlap_overwrite_and_symlinks_are_rejected(parent, tmp_path, sealed):
    with pytest.raises(ValueError, match="independent"):
        learning.prepare_learning_study(parent, parent / "child")
    with pytest.raises(FileExistsError):
        learning.prepare_learning_study(parent, sealed)
    link = tmp_path / "parent-link"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        learning.prepare_learning_study(link, tmp_path / "linked")
    dependency = parent / "snapshot/contracts/lookup.json"
    dependency.unlink()
    dependency.symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        learning.prepare_learning_study(parent, tmp_path / "escape")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "budget", "initial", "runtime", "split", "escape"])
def test_rehashed_manifest_does_not_bypass_semantic_checks(sealed, mutation):
    manifest = learning._read(sealed / "manifest.json")
    if mutation == "missing":
        manifest["cells"].pop()
    elif mutation == "duplicate":
        manifest["cells"][-1] = deepcopy(manifest["cells"][0])
    elif mutation == "budget":
        manifest["protocol"]["updates_per_job"] = 1199
    elif mutation == "initial":
        manifest["initializations"][0]["initial_model_sha256"] = "0" * 64
        for cell in manifest["cells"]:
            if cell["variant"] == "mlp" and cell["training_seed"] == 1101:
                cell["initial_model_sha256"] = "0" * 64
    elif mutation == "runtime":
        manifest["initialization_runtime"]["torch_version"] = "another-runtime"
    elif mutation == "split":
        manifest["protocol"]["confirmation"]["noise_seeds"] = [701, 12701]
    else:
        manifest["cells"][0]["job"] = "../outside"
    resign(sealed, manifest)
    with pytest.raises(ValueError):
        learning.validate_learning_study(sealed)


def test_resigned_child_config_change_is_rejected(sealed):
    manifest = learning._read(sealed / "manifest.json")
    child = manifest["children"][0]
    base_path = sealed / child["base"]["path"]
    base = learning._read(base_path)
    base["ppo"]["entropy_coef"] = .5
    write(base_path, base)
    child["base"] = learning._receipt(sealed, child["base"]["path"], canonical=True)
    resign(sealed, manifest)
    with pytest.raises(ValueError, match="LR-only"):
        learning.validate_learning_study(sealed)


def test_config_raw_bytes_are_authenticated(sealed):
    manifest = learning._read(sealed / "manifest.json")
    config = sealed / manifest["cells"][0]["training_config"]["path"]
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="SHA"):
        learning.validate_learning_study(sealed)


def test_sealed_snapshot_bytes_and_inventory_are_authenticated(sealed):
    dependency = sealed / "snapshot/contracts/lookup.json"
    original = dependency.read_bytes()
    dependency.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="inventory"):
        learning.validate_learning_study(sealed)
    dependency.write_bytes(original)
    (sealed / "snapshot/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        learning.validate_learning_study(sealed)


def test_changed_child_source_is_rejected(sealed):
    path = sealed / "rate_000/study/policy_source/transformer_rl/frame_workflow.py"
    path.write_text(path.read_text() + "\n# changed source\n")
    with pytest.raises(ValueError, match="source"):
        learning.validate_learning_study(sealed)


def test_runtime_artifacts_do_not_return_prepared_and_not_started(sealed):
    directory = sealed / "rate_000/study/jobs/mlp/seed_1101"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="runtime job outputs"):
        learning.validate_learning_study(sealed)


def test_cpu_initializer_restores_rng_on_success_and_failure(parent, monkeypatch):
    import transformer_rl.frame_training as training
    base, spec = learning._read(parent / "base.json"), learning._read(parent / "study.json")
    state = torch.get_rng_state().clone()
    for name in ("manual_seed_all", "_lazy_init", "get_rng_state_all", "set_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: pytest.fail("CUDA API called"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args, **kwargs: pytest.fail("global CUDA-capable seeding called"))
    runtime, values = learning._initializations(base, spec["variants"][:1], [1101, 1102])
    assert runtime["device"] == "cpu" and len(values) == 2
    assert values[0]["initial_model_sha256"] != values[1]["initial_model_sha256"]
    assert torch.equal(state, torch.get_rng_state())
    def broken(config):
        torch.rand(7)
        raise RuntimeError("synthetic construction fault")
    monkeypatch.setattr(training, "FrameActorCritic", broken)
    with pytest.raises(RuntimeError, match="construction fault"):
        learning._initializations(base, spec["variants"][:1], [1101])
    assert torch.equal(state, torch.get_rng_state())


def test_cli_has_only_prepare_and_validate(parent, tmp_path, capsys):
    root = tmp_path / "cli"
    assert learning.main(["prepare", "--parent", str(parent), "--directory", str(root)]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["training_started"] is prepared["queued"] is False
    assert learning.main(["validate", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["sha256"] == prepared["sha256"]
    with pytest.raises(SystemExit):
        learning.main(["run", "--root", str(root)])

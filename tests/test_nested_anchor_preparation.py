"""Nested anchor publication from audited CP400 reports and synthetic CPU pools."""
from contextlib import contextmanager
from copy import deepcopy
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_curriculum_retention_analysis import analysis, frozen, populate, read, reseal_case, write


TOOL = Path(__file__).parents[1] / "tools/prepare_nested_anchors.py"
SPEC = importlib.util.spec_from_file_location("nested_anchor_preparation", TOOL)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def receipt(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@contextmanager
def preserved(*paths):
    before = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, data in before.items():
            path.write_bytes(data)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audited_nested_anchors")
    root, study, manifest, summary = frozen(directory)
    # The strict analyzer keeps H31 Gated while accepting smaller synthetic
    # frame/action dimensions. Every one of the 56 sealed configs is updated.
    for route in manifest["configs"]:
        config = read(study / route)
        config["model"]["policy"].update(frame_dim=5, action_dim=2)
        manifest["configs"][route] = analysis.digest(config)
        manifest["artifacts"][route] = write(study / route, config)
    manifest["sha256"] = analysis.digest({k: v for k, v in manifest.items() if k != "sha256"})
    manifest_file_sha = write(study / "manifest.json", manifest)
    write(root / "manifest.json", manifest)
    campaign = read(root / "campaign.json")
    campaign.update(manifest_file_sha256=manifest_file_sha, manifest_sha256=manifest["sha256"])
    write(root / "campaign.json", campaign)
    summary["campaign_sha256"] = analysis.digest(campaign)
    populate(root, study, manifest, summary)
    qualification = analysis.prepare(root)
    pools = []
    for seed in qualification["training_seeds"]:
        endpoint = summary["results"][f"pretrain/seed_{seed}"]["phases"][0]
        for case_number, case in enumerate(qualification["old_cases"]):
            captured = directory / "anchor_capture" / f"seed_{seed}" / case
            captured.mkdir(parents=True)
            pool_path, report_path = captured / "pool.pt", captured / "report.json"
            sample_count = 513
            frames = torch.arange(sample_count * 31 * 5, dtype=torch.float32).reshape(sample_count, 31, 5)
            frames = frames / 1000 + case_number + seed / 10000
            mean = torch.arange(sample_count * 2, dtype=torch.float32).reshape(sample_count, 2) / 100
            std = torch.full_like(mean, .6)
            report = deepcopy(read(root / endpoint["evaluations"]["8701"]["artifacts"][case]["path"]))
            payload = {"format": "transformer_rl.behavior_anchors", "schema_version": 1,
                "control_sha256": report["control_sha256"], "policy_config": report["model"]["policy"],
                "teacher_checkpoint_sha256": endpoint["training"]["checkpoint"]["checkpoint_sha256"],
                "frames": frames, "mean": mean, "std": std}
            torch.save(payload, pool_path)
            report.update(seed=4101, policy="deterministic_raw_mean_then_declared_action_limits",
                environment_provenance={"identity": qualification["snapshot_sha256"],
                    "control_sha256": report["control_sha256"],
                    "contract_sha256": analysis.digest(read(study / "snapshot" / report["environment"]["contract"]))},
                anchors={"path": str(pool_path.resolve()), "sha256": receipt(pool_path)["sha256"],
                    "samples": sample_count, "max_samples": sample_count})
            write(report_path, report)
            pools.append({"training_seed": seed, "case": case, "pool": receipt(pool_path), "report": receipt(report_path)})
    return {"directory": directory, "root": root, "study": study, "manifest": manifest,
        "summary": summary, "qualification": qualification, "pools": pools}


def freeze(evidence, *, capacities=(256, 512), coefficients=(.1, .2), pools=None):
    return preparation.freeze(evidence["qualification"], capacities=list(capacities),
        coefficients=list(coefficients), anchor_seed=4101,
        pool_receipts=deepcopy(evidence["pools"] if pools is None else pools))


def output_path(directory, item):
    path = Path(item["path"])
    return path if path.is_absolute() else directory / path


def cells(directory):
    return read(directory / "manifest.json")["cells"]


def index_values(directory, cell):
    value = read(output_path(directory, cell["indices"]))
    return value if isinstance(value, list) else value["indices"]


def payload_for(directory, cell):
    return torch.load(output_path(directory, cell["anchor"]), map_location="cpu", weights_only=True)


def rewrite_pool_entry(entry, mutate):
    updated = deepcopy(entry)
    pool_path, report_path = Path(entry["pool"]["path"]), Path(entry["report"]["path"])
    payload = torch.load(pool_path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, pool_path)
    updated["pool"] = receipt(pool_path)
    report = read(report_path)
    report["anchors"].update(sha256=updated["pool"]["sha256"], samples=len(payload["frames"]))
    write(report_path, report)
    updated["report"] = receipt(report_path)
    return updated


def test_ready_preparation_is_nested_across_capacity_and_independent_of_coefficients(evidence, tmp_path):
    plan = freeze(evidence)
    directory = tmp_path / "nested"
    preparation.prepare(plan, directory)
    manifest = read(directory / "manifest.json")
    assert manifest["status"] == "ready"
    assert len(manifest["cells"]) == 3 * 10 * 2 * 2
    assert manifest["qualified_teacher_case_pairs"] == 30
    assert manifest["anchor_file_count"] == manifest["expected_anchor_files"] == 60
    assert len(list(directory.glob("*.pt"))) == 60
    assert len(manifest["branches"]) == 12
    assert all(branch["execution_status"] == "unexecuted" for branch in manifest["branches"])
    assert all(cell["status"] == "ready" for cell in manifest["cells"])
    source = {(entry["training_seed"], entry["case"]): entry for entry in evidence["pools"]}
    grouped = {}
    for cell in manifest["cells"]:
        key = cell["training_seed"], cell["case"]
        grouped.setdefault(key, {})[cell["requested_k"], cell["coefficient"]] = cell
        assert cell["actual_n"] == cell["requested_k"]
        assert cell["pool_sha256"] == source[key]["pool"]["sha256"]
        indices = index_values(directory, cell)
        assert len(indices) == len(set(indices)) == cell["actual_n"]
        original = torch.load(source[key]["pool"]["path"], map_location="cpu", weights_only=True)
        selected = payload_for(directory, cell)
        for name in ("frames", "mean", "std"):
            torch.testing.assert_close(selected[name], original[name][indices], rtol=0, atol=0)
            tensor_receipt = cell["tensor_receipts"][name]
            assert tensor_receipt["dtype"] == "torch.float32"
            assert tensor_receipt["shape"] == list(original[name][indices].shape)
            assert tensor_receipt["sha256"] == hashlib.sha256(original[name][indices].contiguous().numpy().tobytes()).hexdigest()
        assert selected["teacher_checkpoint_sha256"] == original["teacher_checkpoint_sha256"]
        assert selected["policy_config"] == original["policy_config"]
        assert set(cell["tensor_receipts"]) == {"frames", "mean", "std"}
        for name in ("anchor", "indices"):
            actual = receipt(output_path(directory, cell[name]))
            assert actual["sha256"] == cell[name]["sha256"] and actual["bytes"] == cell[name]["bytes"]
    for choices in grouped.values():
        for coefficient in (.1, .2):
            assert index_values(directory, choices[256, coefficient]) == index_values(directory, choices[512, coefficient])[:256]
        for capacity in (256, 512):
            assert index_values(directory, choices[capacity, .1]) == index_values(directory, choices[capacity, .2])
    checked = preparation.validate_preparation(plan, directory)
    assert checked["status"] == "ready"
    independently_prepared = tmp_path / "only_256"
    preparation.prepare(freeze(evidence, capacities=(256,), coefficients=(.2,)), independently_prepared)
    for cell in cells(independently_prepared):
        assert index_values(independently_prepared, cell) == index_values(directory, grouped[cell["training_seed"], cell["case"]][256, .2])


def test_zero_coefficient_branch_has_no_training_anchor_paths(evidence, tmp_path):
    directory = tmp_path / "baseline"
    preparation.prepare(freeze(evidence, capacities=(256,), coefficients=(0., .1)), directory)
    manifest = read(directory / "manifest.json")
    assert manifest["anchor_file_count"] == 30 and len(manifest["cells"]) == 60
    for branch in manifest["branches"]:
        assert branch["execution_status"] == "unexecuted"
        assert branch["teacher"] == {"arm": "pretrain", "phase": "phase1", "checkpoint_update": 400}
        assert branch["updates_per_branch"] == 800
        assert branch["transitions_per_branch"] == 800 * 49152
        endpoint = evidence["summary"]["results"][f"pretrain/seed_{branch['training_seed']}"]["phases"][0]["training"]["checkpoint"]
        assert branch["parent_checkpoint"] == {
            **receipt(Path(endpoint["checkpoint"])), "update": 400, "cumulative_transitions": 400 * 49152}
        assert branch["parent_checkpoint"]["sha256"] == endpoint["checkpoint_sha256"]
        assert branch["consumed_update_offset"] == 400
        assert branch["required_resume_state"] == ["model", "optimizer", "rng", "clock"]
        assert branch["resume_executor"] == "not_implemented"
        assert branch["retention_rng"] == "independent_generator_not_implemented"
        if branch["coefficient"] == 0:
            assert branch["anchor_paths"] == [] and branch["retention_enabled"] is False
        else:
            assert len(branch["anchor_paths"]) == 10 and branch["retention_enabled"] is True


@pytest.mark.parametrize("sample_count", [17, 255])
def test_short_pool_records_actual_count_and_blocks_entire_publication(evidence, tmp_path, sample_count):
    entry = evidence["pools"][0]
    with preserved(Path(entry["pool"]["path"]), Path(entry["report"]["path"])):
        pools = deepcopy(evidence["pools"])
        def shorten(payload):
            for name in ("frames", "mean", "std"):
                payload[name] = payload[name][:sample_count].clone()
        pools[0] = rewrite_pool_entry(entry, shorten)
        directory = tmp_path / "small_pool"
        result = preparation.prepare(freeze(evidence, pools=pools), directory)
        matching = [cell for cell in result["cells"] if (cell["training_seed"], cell["case"]) == (entry["training_seed"], entry["case"])]
        assert len(matching) == 4
        assert all(cell["actual_n"] == sample_count for cell in matching)
        assert result["status"] == "not_ready" and result["ready_cells"] == result["anchor_file_count"] == 0
        assert len(result["cells"]) == result["expected_cells"] == 120
        assert all(cell["status"] == "not_ready" and "anchor" not in cell and "indices" not in cell for cell in result["cells"])
        assert not directory.exists()


@pytest.mark.parametrize("fault", ["dtype", "history", "shape", "policy", "teacher", "control", "nan", "zero_std", "extra_key"])
def test_resealed_invalid_pool_is_rejected_instead_of_trusting_file_sha(evidence, tmp_path, fault):
    entry = evidence["pools"][0]
    with preserved(Path(entry["pool"]["path"]), Path(entry["report"]["path"])):
        def corrupt(payload):
            if fault == "dtype":
                payload["frames"] = payload["frames"].double()
            elif fault == "history":
                payload["frames"] = payload["frames"][:, -30:]
            elif fault == "shape":
                payload["mean"] = payload["mean"][:, :1]
            elif fault == "policy":
                payload["policy_config"] = {**payload["policy_config"], "residual_type": "add"}
            elif fault == "teacher":
                payload["teacher_checkpoint_sha256"] = "0" * 64
            elif fault == "control":
                payload["control_sha256"] = "0" * 64
            elif fault == "nan":
                payload["mean"][0, 0] = float("nan")
            elif fault == "zero_std":
                payload["std"][0, 0] = 0
            else:
                payload["unsealed_metadata"] = True
        pools = deepcopy(evidence["pools"])
        pools[0] = rewrite_pool_entry(entry, corrupt)
        with pytest.raises((ValueError, TypeError)):
            preparation.prepare(freeze(evidence, pools=pools), tmp_path / "invalid")


def test_pool_hash_cannot_be_replaced_by_an_unverified_claim(evidence, tmp_path):
    pools = deepcopy(evidence["pools"])
    pools[0]["pool"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        preparation.prepare(freeze(evidence, pools=pools), tmp_path / "invalid_hash")


@pytest.mark.parametrize("anchor_seed", [1101, 8701])
def test_capture_seed_must_be_independent_of_training_and_qualification(evidence, anchor_seed):
    with pytest.raises(ValueError, match="independent"):
        preparation.freeze(evidence["qualification"], capacities=[256], coefficients=[.1],
            anchor_seed=anchor_seed, pool_receipts=deepcopy(evidence["pools"]))


def test_copied_pool_bytes_cannot_qualify_a_second_case(evidence):
    source, target = evidence["pools"][:2]
    pool_path, report_path = Path(target["pool"]["path"]), Path(target["report"]["path"])
    with preserved(pool_path, report_path):
        pool_path.write_bytes(Path(source["pool"]["path"]).read_bytes())
        pools = deepcopy(evidence["pools"])
        pools[1]["pool"] = receipt(pool_path)
        report = read(report_path)
        report["anchors"]["sha256"] = pools[1]["pool"]["sha256"]
        write(report_path, report)
        pools[1]["report"] = receipt(report_path)
        assert pools[1]["pool"]["sha256"] == source["pool"]["sha256"]
        with pytest.raises(ValueError, match="reused"):
            freeze(evidence, pools=pools)


def test_reserialized_identical_tensors_cannot_establish_independent_case_capture(evidence, tmp_path):
    source, target = evidence["pools"][:2]
    original = torch.load(source["pool"]["path"], map_location="cpu", weights_only=True)
    with preserved(Path(target["pool"]["path"]), Path(target["report"]["path"])):
        def copy_tensors(payload):
            for name in ("frames", "mean", "std"):
                payload[name] = original[name].clone()
            items = list(payload.items())
            payload.clear()
            payload.update(reversed(items))
        pools = deepcopy(evidence["pools"])
        pools[1] = rewrite_pool_entry(target, copy_tensors)
        assert pools[1]["pool"]["sha256"] != source["pool"]["sha256"]
        directory = tmp_path / "reused_content"
        with pytest.raises(ValueError, match="identical tensor content"):
            preparation.prepare(freeze(evidence, pools=pools), directory)
        assert not directory.exists()


@pytest.mark.parametrize("fault", ["model", "checkpoint_update", "evaluation_seed", "snapshot", "raw_contract_sha", "merged_contract_sha"])
def test_resealed_capture_report_must_match_qualified_teacher_and_environment(evidence, tmp_path, fault):
    pools = deepcopy(evidence["pools"])
    report_path = Path(pools[0]["report"]["path"])
    with preserved(report_path):
        report = read(report_path)
        if fault == "model":
            report["model"]["policy"]["frame_dim"] = 6
        elif fault == "checkpoint_update":
            report["checkpoint_update"] = 1200
        elif fault == "evaluation_seed":
            report["seed"] = 8701
        elif fault == "snapshot":
            report["environment_provenance"]["identity"] = "0" * 64
        elif fault == "raw_contract_sha":
            assert report["environment_provenance"]["contract_sha256"] != report["environment"]["contract_sha256"]
            report["environment_provenance"]["contract_sha256"] = report["environment"]["contract_sha256"]
        else:
            merged = {"evaluation": {"cases": [{"name": case} for case in evidence["qualification"]["old_cases"]]}}
            report["environment_provenance"]["contract_sha256"] = analysis.digest(merged)
        write(report_path, report)
        pools[0]["report"] = receipt(report_path)
        with pytest.raises(ValueError):
            preparation.prepare(freeze(evidence, pools=pools), tmp_path / "invalid_capture")


def test_missing_pool_blocks_all_tensor_publication(evidence, tmp_path):
    directory = tmp_path / "missing_pool"
    result = preparation.prepare(freeze(evidence, pools=evidence["pools"][:-1]), directory)
    assert result["status"] == "not_ready" and result["ready_cells"] == 0
    assert len(result["cells"]) == result["expected_cells"] == 120
    assert not directory.exists()


@pytest.mark.parametrize("input_kind", ["pool", "report"])
def test_unready_preparation_still_seals_all_supplied_input_files(evidence, tmp_path, input_kind):
    plan = freeze(evidence, pools=evidence["pools"][:-1])
    directory = tmp_path / "unready"
    result = preparation.prepare(plan, directory)
    assert result["status"] == "not_ready" and not directory.exists()
    assert evidence["pools"][0][input_kind]["path"] in result["input_receipts"]
    path = Path(evidence["pools"][0][input_kind]["path"])
    with preserved(path):
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            preparation.prepare(plan, directory)
    assert not directory.exists()


def test_one_missing_paired_qualification_blocks_all_tensor_publication(evidence, tmp_path):
    root, case = evidence["root"], evidence["qualification"]["old_cases"][0]
    path = root / "control/pretrain/train_1101/phase1/seed_9701/attempt_001" / f"{case}.json"
    with preserved(path):
        path.unlink()
        directory = tmp_path / "missing_qualification"
        result = preparation.prepare(freeze(evidence), directory)
        assert result["status"] == "not_ready" and result["ready_cells"] == 0
        assert result["qualified_teacher_case_pairs"] == 29
        assert len(result["cells"]) == result["expected_cells"] == 120
        assert not directory.exists()


def test_assess_missing_teacher_is_pure_strict_audit_without_torch_import(evidence, monkeypatch):
    root, case = evidence["root"], evidence["qualification"]["old_cases"][0]
    path = root / "control/pretrain/train_1101/phase1/seed_9701/attempt_001" / f"{case}.json"
    original_import = builtins.__import__
    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("pure assessment must not import torch")
        return original_import(name, *args, **kwargs)
    with preserved(path):
        path.unlink()
        before = {str(p) for p in evidence["directory"].rglob("*")}
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", reject_torch)
            result = preparation.assess(evidence["qualification"])
        assert result["status"] == "not_ready"
        assert result["expected_teacher_case_pairs"] == len(result["pairs"]) == 30
        assert result["qualified_teacher_case_pairs"] == 29
        assert result["anchor_publication"] is False and result["execution_status"] == "unexecuted"
        assert "torch" not in preparation.__dict__
        assert before == {str(p) for p in evidence["directory"].rglob("*")}


def test_missing_summary_is_not_ready_without_publication(evidence, tmp_path):
    summary_path = evidence["root"] / "summary.json"
    with preserved(summary_path):
        summary_path.unlink()
        directory = tmp_path / "no_summary"
        result = preparation.prepare(freeze(evidence), directory)
        assert result["status"] == "not_ready" and result["qualified_teacher_case_pairs"] == 0
        assert result["expected_teacher_case_pairs"] == 30
        assert len(result["cells"]) == result["expected_cells"] == 120
        assert all(branch["parent_checkpoint"] is None for branch in result["branches"])
        assert not directory.exists()


def test_original_gate_failure_cannot_be_promoted_from_other_qualified_cases(evidence, tmp_path):
    root, case = evidence["root"], evidence["qualification"]["old_cases"][0]
    paths = [root / "summary.json"]
    for seed in (8701, 9701):
        directory = root / f"control/pretrain/train_1101/phase1/seed_{seed}/attempt_001"
        paths.extend((directory / f"{case}.json", directory / "receipt.json"))
    with preserved(*paths):
        summary = read(root / "summary.json")
        reseal_case(root, summary, mutate=lambda report: report["metrics"]["height_abs_error"].update(mean=.08))
        directory = tmp_path / "failed_original_gate"
        result = preparation.prepare(freeze(evidence), directory)
        assert result["status"] == "not_ready" and result["ready_cells"] == 0
        assert result["qualified_teacher_case_pairs"] == 29
        assert len(result["cells"]) == result["expected_cells"] == 120
        assert not directory.exists()


def test_sealed_source_edit_is_rejected_before_output_publication(evidence, tmp_path):
    source = evidence["root"].parent / "source/src/transformer_rl/frame_process.py"
    plan = freeze(evidence)
    with preserved(source):
        source.write_text(source.read_text() + "# edited after freezing\n")
        directory = tmp_path / "source_changed"
        with pytest.raises(ValueError):
            preparation.prepare(plan, directory)
        assert not directory.exists() or not list(directory.rglob("*.pt"))


@pytest.mark.parametrize("artifact_name", ["anchor", "indices", "manifest"])
def test_validation_rejects_output_edits(evidence, tmp_path, artifact_name):
    plan, directory = freeze(evidence, capacities=(256,), coefficients=(.1,)), tmp_path / "prepared"
    preparation.prepare(plan, directory)
    path = directory / "manifest.json" if artifact_name == "manifest" else output_path(directory, cells(directory)[0][artifact_name])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        preparation.validate_preparation(plan, directory)


def test_validation_reaudits_input_and_does_not_trust_ready_manifest(evidence, tmp_path):
    plan, directory = freeze(evidence, capacities=(256,), coefficients=(.1,)), tmp_path / "prepared"
    preparation.prepare(plan, directory)
    source = evidence["root"].parent / "source/src/transformer_rl/frame_process.py"
    with preserved(source):
        source.write_text(source.read_text() + "# edited after preparing\n")
        with pytest.raises(ValueError):
            preparation.validate_preparation(plan, directory)


def test_validation_reaudits_capture_report_receipts(evidence, tmp_path):
    plan, directory = freeze(evidence, capacities=(256,), coefficients=(.1,)), tmp_path / "prepared"
    preparation.prepare(plan, directory)
    report_path = Path(evidence["pools"][0]["report"]["path"])
    with preserved(report_path):
        report_path.write_bytes(report_path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            preparation.validate_preparation(plan, directory)


def test_output_directory_is_exclusive_and_cannot_overwrite_existing_files(evidence, tmp_path):
    plan, directory = freeze(evidence, capacities=(256,), coefficients=(.1,)), tmp_path / "prepared"
    preparation.prepare(plan, directory)
    before = {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        preparation.prepare(plan, directory)
    assert before == {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    empty = tmp_path / "already_exists"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        preparation.prepare(plan, empty)
    assert not list(empty.iterdir())


def test_raw_output_dangling_symlink_is_rejected_before_resolving(evidence, tmp_path):
    plan = freeze(evidence, capacities=(256,), coefficients=(.1,))
    directory, target = tmp_path / "dangling", tmp_path / "missing_target"
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        preparation.prepare(plan, directory)
    assert directory.is_symlink() and not target.exists()

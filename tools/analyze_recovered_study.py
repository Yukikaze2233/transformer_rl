"""Read-only CPU acceptance of the frozen architecture-study recovery.

Only derived JSON/CSV are published, exclusively outside the recovery root.
No environment factory, inference, optimizer step, or remote operation is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile


REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("last_token_attention", "time_attention", "index_attention", "gated_attention")
SEEDS = (1011, 1022, 1033)
ARCHIVE_SHA = "b798cdfe296da6ee2524549cad0e3d0cdefcd85a991d955e93197f6949d1eb92"
MANIFEST_SHA = "4d8ba084fb983eb4c1e4c89df1c2851d24339fc4cddd393fc02919628f393b4a"
SOURCE_COMMIT = "80a45b40c9afc7e6e3356ccbb6fa15f8933a3970"
PACKAGE_SHA = "e6ef86cae8177052999b773d2615af5edaa2c46774a9942548fa0f47242af5c7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def confined(root, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe relative path: {relative}")
    root = root.resolve()
    path = (root / str(relative)).resolve(strict=True)
    require(path != root and path.is_relative_to(root), f"path escapes root: {relative}")
    return path


def map_remote_checkpoint(remote_path, remote_study, local_study):
    """Map an explicit absolute prefix; never rewrite source metadata."""
    path, prefix = PurePosixPath(remote_path), PurePosixPath(remote_study)
    require(path.is_absolute() and prefix.is_absolute(), "absolute study paths required")
    require(".." not in path.parts, "checkpoint traversal is forbidden")
    try:
        relative = path.relative_to(prefix)
    except ValueError as error:
        raise ValueError("checkpoint outside explicit remote study prefix") from error
    return confined(local_study, str(relative))


def training_seed_statistics(values):
    """Retain missing values explicitly, using sample SD across training seeds."""
    present = [value for value in values if value is not None]
    return {"n": len(present), "missing": len(values) - len(present),
            "mean": statistics.mean(present) if present else None,
            "std": statistics.stdev(present) if len(present) > 1 else None}


def grouped_derivative(signals, names):
    """Equal channel RMS, only when all channels share the same retained support."""
    selected = [signals[name] for name in names]
    require(len({signal["derivative_count"] for signal in selected}) == 1,
            "cannot aggregate derivatives with different sample support")
    values = [signal["derivative_rms"] for signal in selected]
    if any(value is None for value in values):
        return None
    return math.sqrt(statistics.mean(value * value for value in values))


class RecoveryAnalysis:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.extracted = self.root / "extracted"
        self.study = self.extracted / "study"
        self.run = self.study / "run"
        self.failures = []

    def verify_inventory(self):
        manifest_path = self.root / "manifest.json"
        require(sha256(manifest_path) == MANIFEST_SHA, "manifest anchor mismatch")
        manifest = read_json(manifest_path)
        receipt = read_json(self.root / "recovery_receipt.json")
        require(receipt["manifest_sha256"] == MANIFEST_SHA, "receipt manifest mismatch")
        require(manifest["archive"] == receipt["archive"], "archive metadata mismatch")
        archive = confined(self.root, manifest["archive"]["path"])
        require(archive.stat().st_size == manifest["archive"]["size"] == 226433945,
                "archive size mismatch")
        require(sha256(archive) == manifest["archive"]["sha256"] == ARCHIVE_SHA,
                "archive hash mismatch")
        entries = {entry["path"]: entry for entry in manifest["files"]}
        require(len(entries) == len(manifest["files"]) == manifest["file_count"] == 729,
                "inventory count mismatch")
        require(sum(entry["size"] for entry in entries.values())
                == manifest["total_bytes"] == 313976774, "inventory bytes mismatch")
        seen = set()
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                require(member.isfile() and member.name in entries and member.name not in seen,
                        f"unexpected archive member: {member.name}")
                entry = entries[member.name]
                require(member.size == entry["size"], f"member size: {member.name}")
                with stream.extractfile(member) as content:
                    digest = hashlib.file_digest(content, "sha256").hexdigest()
                require(digest == entry["sha256"], f"member hash: {member.name}")
                seen.add(member.name)
        require(seen == set(entries), "archive inventory mismatch")
        for name, entry in entries.items():
            path = confined(self.extracted, name)
            require(path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"],
                    f"local file integrity: {name}")
        actual = {str(path.relative_to(self.extracted)) for path in self.extracted.rglob("*")
                  if path.is_file()}
        require(actual == set(entries), "local inventory differs from manifest")
        self.manifest = manifest
        self.entries = entries
        return {"verified_files": len(entries), "verified_bytes": manifest["total_bytes"],
                "archive": manifest["archive"], "manifest_sha256": MANIFEST_SHA,
                "recovery_receipt_sha256": sha256(self.root / "recovery_receipt.json"),
                "skipped": manifest["skipped"],
                "partial_jsonl_tails": receipt["partial_jsonl_tails"],
                "snapshot_start": manifest["snapshot_start"],
                "snapshot_end": manifest["snapshot_end"],
                "snapshot_semantics": manifest["semantics"],
                "snapshot_queue_counts": manifest["status_end"]["value"]["queue_counts"],
                "path_mapping": {"remote_prefix": manifest["study"],
                                 "local_prefix": str(self.study)}}

    def verify_environment(self, provenance):
        identity = provenance["identity"]
        task = self.extracted / "task"
        contract = read_json(task / "contracts/own_v40_v2.json")
        canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
        require(hashlib.sha256(canonical.encode()).hexdigest() == identity["contract_sha256"],
                "contract canonical hash mismatch")
        require(sha256(task / "contracts/own_v40_v2.json") == provenance["contract_file_sha256"],
                "contract file hash mismatch")
        asset_root = confined(task, contract["asset"]["directory"])
        asset_path = confined(asset_root, contract["asset"]["manifest"])
        require(sha256(asset_path) == identity["asset_manifest_sha256"], "asset manifest hash")
        assets = read_json(asset_path)
        for name, digest in assets["files_sha256"].items():
            require(sha256(confined(asset_root, name)) == digest, f"asset hash: {name}")
        files = provenance["source_files_sha256"]
        for name, digest in files.items():
            require(sha256(confined(task, name)) == digest, f"task source hash: {name}")
        require(hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                == identity["source_sha256"], "task source aggregate hash")
        for key, filename in (("adapter_sha256", "isaaclab_task.py"),
                              ("worker_sha256", "_isaaclab_process.py")):
            for source_root in (REPO, self.extracted / "source"):
                require(sha256(source_root / "examples" / filename) == identity[key],
                        f"adapter/worker source mismatch: {filename}")
        require(identity["usd_seed_sha256"] is None, "unexpected USD seed")
        self.identity = identity
        self.signal_units = provenance["evaluation_signals"]["units"]
        self.contract = contract
        return {"identity": identity, "task_source_files_verified": len(files),
                "asset_files_verified": len(assets["files_sha256"]),
                "contact": provenance["contact"],
                "signal_semantics": provenance["evaluation_signals"]}

    def verify_checkpoint(self, path, job, expected_update, config):
        import torch
        from transformer_rl.checkpoint import (
            _DTYPES, _PAYLOAD_KEYS, _json_metadata, _new_model, _parse_config,
            _validate_optimizer_state, _validate_tensor, _validated_components,
        )
        from transformer_rl.config import ModelConfig, PPOConfig

        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        entry = self.entries[str(path.relative_to(self.extracted))]
        require(digest == entry["sha256"], "checkpoint changed after inventory verification")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        require(payload["schema_version"] == payload["source_schema_version"] == 4,
                "expected native schema 4")
        require(payload["model_config"] == config["model"]
                and payload["ppo_config"] == config["ppo"], "checkpoint config mismatch")
        strict_error = None
        try:
            _validated_components(payload, device="cpu")
        except ValueError as error:
            strict_error = str(error)
            self.failures.append({"job": job["id"], "scope": "strict_checkpoint_loader",
                                  "error": strict_error})
        # Inspect independently after a strict-load failure. This does not load a
        # modified payload or relax the production loader's acceptance contract.
        require(set(payload) == _PAYLOAD_KEYS | {"source_schema_version"}
                and payload["format"] == "transformer_rl.checkpoint", "checkpoint payload schema")
        parsed_config = _parse_config(payload["model_config"], ModelConfig)
        _parse_config(payload["ppo_config"], PPOConfig)
        model = _new_model(parsed_config, _DTYPES[payload["model_dtype"]])
        expected_state = model.state_dict()
        require(set(payload["model_state"]) == set(expected_state), "model state keys")
        for name, reference in expected_state.items():
            _validate_tensor(name, payload["model_state"][name], reference.shape, reference.dtype)
        require(payload["optimizer_type"] == "Adam", "optimizer type")
        _validate_optimizer_state(payload["optimizer_state"], model)
        differences = []
        for name, reference in model.named_buffers():
            actual = payload["model_state"][name]
            if not torch.equal(actual, reference):
                delta = (actual.double() - reference.double()).abs()
                differences.append({"name": name, "different_elements": int((actual != reference).sum()),
                                    "max_abs_difference": delta.max().item(),
                                    "max_relative_difference": (delta / reference.double().abs().clamp_min(1e-300)).max().item(),
                                    "indices": (actual != reference).nonzero().tolist(),
                                    "stored_differing_values": actual[actual != reference].tolist(),
                                    "local_differing_values": reference[actual != reference].tolist()})
        update, metadata = payload["update"], _json_metadata(payload["metadata"])
        require(update == expected_update, "checkpoint update mismatch")
        require(metadata["source"] == {"commit": SOURCE_COMMIT, "dirty": False},
                "checkpoint source mismatch")
        require(metadata["seed"] == job["seed"] and metadata["action_clip"] == 100.0,
                "checkpoint seed/action clip mismatch")
        require(metadata["environment"] == config["environment"], "checkpoint environment config")
        require(metadata["environment_factory"] == "examples.isaaclab_task:make_env",
                "checkpoint environment factory")
        require(metadata["environment_provenance"]["identity"] == self.identity,
                "checkpoint environment identity")
        require(metadata["collected_transitions"] == update * 512 * 32,
                "checkpoint transition count")
        require(all(parameter.device.type == "cpu" for parameter in model.parameters()),
                "CPU-only checkpoint validation required")
        result = {"path": str(path.relative_to(self.extracted)), "sha256": digest,
                  "update": update, "transitions": metadata["collected_transitions"],
                  "schema_version": 4, "model_parameters": sum(p.numel() for p in model.parameters()),
                  "adam_state_entries": len(payload["optimizer_state"]["state"]),
                  "weights_only_deserialized": True, "tensor_and_adam_schema_validated": True,
                  "strict_loader_passed": strict_error is None, "strict_loader_error": strict_error,
                  "buffer_differences": differences}
        steps = [entry["step"].item() for entry in payload["optimizer_state"]["state"].values()]
        require(steps and min(steps) > 0, "trained Adam state must be populated")
        result["adam_step_range"] = [min(steps), max(steps)]
        del model, payload
        return result

    def metrics_row(self, report, job, scenario):
        signals, metrics = report["stability"]["signals"], report["metrics"]
        height = signals["height_error"]
        row = {"variant": job["variant"], "training_seed": job["seed"],
               "scenario": scenario, "evaluation_seed": 301,
               "height_command_m": report["environment"]["fixed_command"][2],
               "height_signed_bias_mm": None if height["mean"] is None else height["mean"] * 1000,
               "height_within_episode_std_mm": (None if height["within_episode_std"] is None
                                                 else height["within_episode_std"] * 1000),
               "vx_signed_error_m_s": signals["vx_error"]["mean"],
               "wz_signed_error_rad_s": signals["wz_error"]["mean"],
               "height_full_interval_mae_mm": metrics["height_abs_error"]["mean"] * 1000,
               "planar_speed_full_interval_m_s": metrics["planar_speed"]["mean"],
               "tilt_full_interval_rad": metrics["tilt_angle"]["mean"],
               "nonwheel_netforce_full_interval_N": metrics["non_wheel_net_force"]["mean"],
               "vx_full_interval_mae_m_s": metrics["vx_abs_error"]["mean"],
               "wz_full_interval_mae_rad_s": metrics["wz_abs_error"]["mean"],
               "terminated_count": report["terminated_count"],
               "truncated_count": report["truncated_count"], "done_count": report["done_count"],
               "steady_coverage": height["count"] / report["transitions"],
               "steady_samples": height["count"], "steady_segments": height["segments"],
               "short_segments": height["short_segments"],
               "short_retained_samples": height["short_count"],
               "reward_mean_diagnostic": report["reward_mean"]}
        for prefix, indices, unit, output in (
            ("leg_target", range(4), "rad", "leg_target_derivative_rms_rad_s"),
            ("wheel_target", range(2), "rad/s", "wheel_target_derivative_rms_rad_s2"),
            ("effort", self.contract["joints"]["leg_indices"], "Nm", "leg_effort_derivative_rms_Nm_s"),
            ("effort", self.contract["joints"]["wheel_indices"], "Nm", "wheel_effort_derivative_rms_Nm_s"),
        ):
            names = [f"{prefix}_{index}" for index in indices]
            require(all(self.signal_units[name] == unit for name in names), "signal unit mismatch")
            row[output] = grouped_derivative(signals, names)
        return row

    def analyze(self):
        import torch
        from transformer_rl.experiments import _report, validate_plan

        torch.set_num_threads(1)
        integrity = self.verify_inventory()
        plan = validate_plan(self.run, check_source=True)
        require(plan["source"]["sha256"] == PACKAGE_SHA, "package source anchor mismatch")
        for name, digest in plan["source"]["files"].items():
            path = confined(self.extracted / "source/src/transformer_rl", name)
            require(sha256(path) == digest, f"recovered package source mismatch: {name}")
        first = self.run / "jobs/last_token_attention/seed_1011/train/completion.json"
        environment = self.verify_environment(read_json(first)["environment_provenance"])
        jobs, rows = [], []
        for job in plan["jobs"]:
            directory = self.run / job["directory"]
            config = read_json(self.run / job["config"])
            item = {"job": job["id"], "snapshot_status": "not_started"}
            jobs.append(item)
            if job["variant"] not in VARIANTS:
                checkpoints = sorted((directory / "train/checkpoints").glob("checkpoint_*.pt"))
                if checkpoints:
                    item["snapshot_status"] = "partial"
                    expected = {1011: 700, 1022: 600}[job["seed"]]
                    try:
                        item["checkpoint"] = self.verify_checkpoint(checkpoints[-1], job, expected, config)
                    except (ValueError, KeyError, OSError, RuntimeError) as error:
                        item["validation_error"] = str(error)
                        self.failures.append({"job": job["id"], "error": str(error)})
                item["excluded_from_equal_budget_comparison"] = True
                continue
            try:
                result = read_json(directory / "result.json")
                require(result["status"] == "completed" and result["job"] == job["id"]
                        and result["plan_sha256"] == plan["plan_sha256"], "result status/provenance")
                completion = read_json(directory / "train/completion.json")
                require(completion["status"] == "completed"
                        and completion["stop_reason"] == "updates_completed"
                        and completion["updates_completed"] == completion["cumulative_update"] == 977
                        and completion["collected_transitions"] == 16007168, "completion budget")
                require(completion["environment_provenance"]["identity"] == self.identity,
                        "completion environment identity")
                for saved in completion["checkpoints"]:
                    local = map_remote_checkpoint(saved["path"], self.manifest["study"], self.study)
                    require(local.parent == (directory / "train/checkpoints").resolve(),
                            "checkpoint maps to another job")
                    require(sha256(local) == saved["sha256"], "completion checkpoint hash")
                last = completion["checkpoints"][-1]
                require(last["update"] == 977, "last checkpoint update")
                item["checkpoint"] = self.verify_checkpoint(local, job, 977, config)
                item["snapshot_status"] = "completed"
                item["evaluation_files_verified"] = 0
                for scenario, route in job["eval_configs"].items():
                    try:
                        path = directory / f"evaluation_{scenario}_301.json"
                        expected_env = read_json(self.run / route)["environment"]
                        report = _report(path, last["sha256"], 301, plan["spec"], expected_env)
                        require(report["checkpoint_update"] == 977 and report["num_envs"] == 8
                                and report["transitions"] == 16000, "evaluation budget")
                        require(report["environment_provenance"]["identity"] == self.identity,
                                "evaluation environment identity")
                        require(report["environment_provenance"]["evaluation_signals"]["units"]
                                == self.signal_units, "evaluation units")
                        require(report["physical_metrics_available"], "physical metrics unavailable")
                        require(all(metric["count"] == 16000 for metric in report["metrics"].values()),
                                "physical metric support mismatch")
                        rows.append(self.metrics_row(report, job, scenario))
                        item["evaluation_files_verified"] += 1
                    except (ValueError, KeyError, OSError, TypeError) as error:
                        self.failures.append({"job": job["id"], "scenario": scenario, "error": str(error)})
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                item["validation_error"] = str(error)
                self.failures.append({"job": job["id"], "error": str(error)})
        summaries = []
        keys = [key for key in (rows[0] if rows else {}) if key not in (
            "variant", "training_seed", "scenario", "evaluation_seed", "height_command_m")]
        for variant in VARIANTS:
            for scenario in (s["name"] for s in plan["spec"]["evaluation"]["scenarios"]):
                selected = {row["training_seed"]: row for row in rows
                            if row["variant"] == variant and row["scenario"] == scenario}
                summaries.append({"variant": variant, "scenario": scenario,
                                  "training_seeds": sorted(selected),
                                  "metrics": {key: training_seed_statistics(
                                      [selected.get(seed, {}).get(key) for seed in SEEDS]) for key in keys}})
        report = {"schema_version": 1, "integrity": integrity,
                  "plan_sha256": plan["plan_sha256"], "source_commit": SOURCE_COMMIT,
                  "package_sha256": PACKAGE_SHA, "environment": environment,
                  "verification": {"checkpoint_weights_only_deserialized_cpu": sum("checkpoint" in j for j in jobs),
                                   "tensor_and_adam_schema_validated": sum("checkpoint" in j for j in jobs),
                                   "strict_checkpoint_loader_passed": sum(j.get("checkpoint", {}).get(
                                       "strict_loader_passed", False) for j in jobs),
                                   "local_torch_version": torch.__version__,
                                   "local_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                                   "evaluation_files_verified": len(rows), "failures": self.failures},
                  "jobs": jobs, "aggregation": "per scenario; sample SD (ddof=1) over independent training seeds",
                  "derivative_aggregation": "sqrt(mean(channel derivative_rms squared)); equal channel support; legs/wheels/effort separated",
                  "post_snapshot_status": {
                      "evidence_source": "parent-agent handoff, not the pre-stop snapshot",
                      "stopped_at": "2026-09-15T16:02:56+08:00", "abort_reason": "signal_SIGTERM",
                      "completed": 12, "manually_stopped": 2, "never_started": 7,
                      "all_batch_pids_exited": True,
                      "untransferred_final_updates": {"supervised_attention/seed_1011": 789,
                                                     "supervised_attention/seed_1022": 732},
                      "transfer_blocker": "SSH connection reset reported by parent; no remote access attempted"},
                  "limitations": ["snapshot is per-file, pre-stop, not globally atomic",
                                  "partial supervised checkpoints are 700/600, not final stopped 789/732",
                                  "no same-budget MLP/GRU results; supervised incomplete",
                                  "one evaluation seed, three training seeds per complete variant",
                                  "net force has no ground-pair identity",
                                  "planar speed, tilt, force and MAE include transients and resets",
                                  "derivatives sample policy boundaries, not full physics-rate traces",
                                  "CPU load validates model and Adam state, not inference or physical quality"],
                  "summaries": summaries}
        return report, rows


def print_tables(report):
    for standing in (True, False):
        keys = (["height_signed_bias_mm", "height_within_episode_std_mm", "vx_signed_error_m_s",
                 "planar_speed_full_interval_m_s", "tilt_full_interval_rad",
                 "nonwheel_netforce_full_interval_N"] if standing else [
                     "vx_full_interval_mae_m_s", "wz_full_interval_mae_rad_s",
                     "vx_signed_error_m_s", "wz_signed_error_rad_s"])
        print("| Variant | Scenario | " + " | ".join(keys) + " |")
        for summary in report["summaries"]:
            if summary["scenario"].startswith("stand_") != standing:
                continue
            cells = []
            for key in keys:
                stat = summary["metrics"].get(key, {})
                cells.append(f"{stat['mean']:.4f} ± {stat['std']:.4f}"
                             if stat.get("std") is not None else "missing")
            print(f"| {summary['variant']} | {summary['scenario']} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--print-tables", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "set CUDA_VISIBLE_DEVICES='' explicitly")
    outputs = [path.resolve() for path in (args.json, args.csv)]
    require(outputs[0] != outputs[1], "output paths must differ")
    for path in outputs:
        require(not path.is_relative_to(args.root.resolve()), "outputs must be outside recovery root")
        require(path.parent.is_dir() and not path.exists(), f"output must be new: {path}")
    sys.path.insert(0, str(REPO / "src"))
    report, rows = RecoveryAnalysis(args.root).analyze()
    with args.json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with args.csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "variant", "training_seed", "scenario"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["verification"], indent=2))
    if args.print_tables:
        print_tables(report)
    return int(bool(report["verification"]["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())

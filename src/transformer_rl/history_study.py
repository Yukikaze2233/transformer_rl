"""Prepare and validate complete history-length studies without executing jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

from . import frame_study
from .frame_config import FrameTrainConfig, digest, json_bytes


FORMAT = "transformer_rl.history_study"
PACKAGE = Path(__file__).absolute().parent
MANIFEST = "history_plan.json"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _path(value):
    path = Path(value)
    _require(path.is_absolute(), "paths must be explicit absolute paths")
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             "paths must not traverse symlinks")
    path = Path(os.path.abspath(path))
    _require(not any(p.is_symlink() for p in (path, *path.parents)),
             "paths must not traverse symlinks")
    return path


def _raw(path):
    path = _path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode), "inputs and frozen artifacts must be regular files")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _require(identity(before) == identity(after) == identity(path.stat()),
             "file changed while reading")
    return raw


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    json_bytes(value)
    return value


def _receipt(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _write(path, raw):
    _path(path)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _lengths(values):
    _require(isinstance(values, (list, tuple)) and len(values) >= 2
             and all(type(n) is int and n > 0 for n in values)
             and len(set(values)) == len(values) and 1 in values,
             "history_lengths require at least two unique positive integers including H1")
    return list(values)


def _num_envs(environment):
    count = environment.get("num_envs")
    _require(type(count) is int and count > 0,
             "every resolved environment.num_envs must be an explicit positive integer")
    return count


def _expand(spec, base, lengths, position_reference):
    _require(position_reference == "current", "Transformer position_reference must be explicit current")
    lengths = _lengths(lengths)
    try:
        frame_study._validate_spec(spec)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("invalid complete study specification") from error
    _require(len(spec["seeds"]) >= 3 and len(spec["evaluation"]["seeds"]) >= 2,
             "history studies require at least three training and two held-out evaluation seeds")
    _require(3 <= spec["selection"]["min_training_seeds"] <= len(spec["seeds"]),
             "selection.min_training_seeds must preserve at least three training seeds")
    # Validate every original override with the same recipe restrictions as FrameStudy.
    frame_study._configs(base, spec)
    expanded = deepcopy(spec)
    expanded["base_config"] = "inputs/base.json"
    expanded["variants"] = []
    candidates = []
    architectures = set()
    for variant in spec["variants"]:
        original = base.with_policy(variant["policy"]).model.policy
        architecture = original.architecture
        _require(architecture in {"mlp", "history_mlp", "transformer"},
                 "history studies exclude frame-stack MLP and recurrent architectures")
        architectures.add(architecture)
        for length in ([1] if architecture == "mlp" else lengths):
            name = f"{variant['name']}_h{length}"
            policy = deepcopy(variant["policy"])
            policy["history_length"] = length
            if architecture == "transformer":
                policy["position_reference"] = position_reference
            candidate = base.with_policy(policy).model.policy.to_dict()
            invariant = original.to_dict()
            invariant["history_length"] = length
            if architecture == "transformer":
                invariant["position_reference"] = position_reference
            _require(candidate == invariant, "expansion changed another architecture setting")
            expanded["variants"].append({"name": name, "policy": policy})
            candidates.append({"name": name, "source_variant": variant["name"],
                "architecture": architecture, "history_length": length,
                "policy": candidate, "span_s": (length - 1) * base.control["policy_dt_s"],
                "minimum_full_history_episode_age": length - 1})
    _require(architectures == {"mlp", "history_mlp", "transformer"},
             "the study must retain single-frame MLP, history MLP and Transformer comparisons")
    frame_study._validate_spec(expanded)
    configs = frame_study._configs(base, expanded)
    for config in configs.values():
        _num_envs(config["environment"])
    return expanded, candidates, configs


def _budget(spec, base, candidates, configs):
    jobs = len(candidates) * len(spec["seeds"])
    rollout = spec["training"]["rollout_steps"]
    stages, required = [], []
    total_updates = total_transitions = 0
    for stage in spec["stages"]:
        required.extend(name for name in stage["scenarios"] if name not in required)
        count = _num_envs(configs[f"configs/{candidates[0]['name']}.train.{stage['name']}.json"]["environment"])
        updates = stage["updates"] * jobs
        transitions = updates * rollout * count
        total_updates += updates
        total_transitions += transitions
        stages.append({"name": stage["name"], "updates_per_job_upper": stage["updates"],
            "num_envs": count, "fresh_transitions_per_job_upper": stage["updates"] * rollout * count,
            "updates_all_jobs_upper": updates, "fresh_transitions_all_jobs_upper": transitions,
            "required_scenarios": list(required),
            "held_out_evaluation_cells_at_stage_endpoint": jobs * len(required) * len(spec["evaluation"]["seeds"]),
            "validation_cells_per_checkpoint": jobs * len(required) * len(spec["evaluation"]["validation_seeds"]),
            "validation_checkpoints_per_job_if_full_chunks": ((stage["updates"] + spec["training"]["checkpoint_interval"] - 1)
                                                             // spec["training"]["checkpoint_interval"]),
            "anchor_cells_at_promotion": (jobs * len(required) * len(spec["training"]["anchor_seeds"])
                                           if spec["training"]["retention_coef"] > 0 else 0)})
    eval_environments = {scenario["name"]: _num_envs(configs[
        f"configs/{candidates[0]['name']}.eval.{scenario['name']}.json"]["environment"])
        for scenario in spec["scenarios"]}
    return {"scope": "requested upper bound for complete fresh rollout batches, not optimizer sample uses; partial/failed/unsealed actual transitions require a separate runtime ledger",
        "candidate_count": len(candidates), "job_count": jobs,
        "training_seeds": list(spec["seeds"]), "held_out_evaluation_seeds": list(spec["evaluation"]["seeds"]),
        "validation_seeds": list(spec["evaluation"]["validation_seeds"]),
        "anchor_seeds": list(spec["training"]["anchor_seeds"]),
        "rollout_steps": rollout, "updates_per_job_upper": sum(s["updates"] for s in spec["stages"]),
        "updates_all_jobs_upper": total_updates, "fresh_transitions_all_jobs_upper": total_transitions,
        "stages": stages, "evaluation_num_envs_by_scenario": eval_environments,
        "final_held_out_evaluation_cells": jobs * len(spec["scenarios"]) * len(spec["evaluation"]["seeds"]),
        "final_evaluation_transition_upper": jobs * len(spec["evaluation"]["seeds"])
             * spec["evaluation"]["steps"] * sum(eval_environments.values()),
        "cell_definition": "candidate × training seed × scenario × evaluation seed; replicas are separate",
        "failure_denominator": "all declared candidate/training-seed jobs; incomplete and rejected jobs remain missing"}


def _execution():
    return {"preparation_started_training": False, "equal_exposure_comparison_ready": False,
        "needs_independent_equal_exposure_executor": True,
        "runner": "transformer_rl.frame_study",
        "stage_initialization": "initialize-from resets Adam and learning RNG; no full-state continuation guarantee",
        "within_stage": "resume inherits state; retention regression may restore a protected checkpoint",
        "promotion": "gates, rollback limit and early rejection can change exposure or omit later stages",
        "evaluation": "runner validates cumulative scenarios at checkpoints and held-out scenarios at final completion; stage endpoint held-out cells are planned obligations, not runner output",
        "required_actual_records": ["requested/attempted/completed updates", "fresh rollout transitions",
            "checkpoint parents and optimizer/RNG restore mode", "promotions/rollbacks/rejections",
            "all missing candidate/seed/stage/evaluation cells", "wall time and compute cost"],
        "conclusions": "preparation does not establish qualified teachers, memory retention, forgetting or a winning architecture"}


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        _path(path)
        mode = path.stat().st_mode
        _require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "study tree contains an unsupported artifact")
        if path.is_file():
            route = str(path.relative_to(root))
            parts = path.relative_to(root).parts
            package_file = parts[:3] == ("study", "policy_source", "transformer_rl")
            # Match FrameStudy's frozen source cache exclusion, only inside its package.
            source_cache = package_file and ("__pycache__" in parts[3:] or path.suffix == ".pyc")
            mutable = route == "study/.run.lock" or route.startswith("study/jobs/") or source_cache
            if route != MANIFEST and not mutable:
                result[route] = _receipt(_raw(path))
    return result


def _definition(root, spec_path, spec_raw, base_path, base_raw, lengths, position_reference):
    spec, base = _parse(spec_raw), FrameTrainConfig.from_dict(_parse(base_raw))
    _require(isinstance(spec, dict), "study specification requires an object")
    _require(isinstance(spec.get("base_config"), str) and spec["base_config"], "base_config requires a path")
    _require(_path(spec_path.parent / spec["base_config"]) == base_path,
             "explicit base path differs from the original specification")
    expanded, candidates, configs = _expand(spec, base, lengths, position_reference)
    source = frame_study.source_identity()
    definition = {"format": FORMAT, "schema_version": 1, "status": "prepared", "root": str(root),
        "inputs": {"spec": {"path": str(spec_path), **_receipt(spec_raw)},
                   "base": {"path": str(base_path), **_receipt(base_raw)}},
        "history_lengths": list(lengths), "position_reference": position_reference,
        "history": {"stride": 1, "order": "oldest_to_current", "reset": "repeat_first",
                    "episode_age_unit": "policy steps since reset", "policy_dt_s": base.control["policy_dt_s"],
                    "fully_observed_rule": "episode_age >= history_length - 1; reset-filled windows are separately reported"},
        "candidates": candidates, "budget": _budget(expanded, base, candidates, configs),
        "execution_semantics": _execution(), "source": source,
        "planner_source": {"path": str(PACKAGE / "history_study.py"), **_receipt(_raw(PACKAGE / "history_study.py"))},
        "expanded_spec_sha256": digest(expanded), "study_root": "study"}
    return definition, expanded, base


def _guard_output(root, inputs):
    _require(root.parent.is_dir() and not root.exists(), "output parent must exist and output must be new")
    for protected in [PACKAGE, *(path.parent for path in inputs)]:
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                 "output must not overlap an input or learner source tree")


def prepare_history_study(spec_path, base_config_path, output_root, *, history_lengths, position_reference):
    """Freeze a new complete grid and actual source bytes; never import a factory or run a job.

    Failure after directory creation leaves an unsealed partial directory. Reuse is
    refused; only the last, successfully validated manifest constitutes preparation.
    """
    spec_path, base_path, root = map(_path, (spec_path, base_config_path, output_root))
    _guard_output(root, (spec_path, base_path))
    spec_raw, base_raw = _raw(spec_path), _raw(base_path)
    definition, expanded, base = _definition(root, spec_path, spec_raw, base_path, base_raw,
                                             history_lengths, position_reference)
    root.mkdir(exist_ok=False)
    (root / "inputs").mkdir()
    _write(root / "inputs/spec.json", spec_raw)
    _write(root / "inputs/base.json", base_raw)
    _write(root / "expanded_spec.json", _json(expanded))
    frame_study.plan_study(root / "expanded_spec.json", root / "study")
    frozen = frame_study.validate_study(root / "study", source=True)
    _require(frozen["base"] == base.to_dict() and frozen["spec"] == expanded
             and frozen["source"] == definition["source"], "frozen study differs from its inputs")
    _require(_raw(spec_path) == spec_raw and _raw(base_path) == base_raw,
             "original inputs changed while preparing")
    _require(frame_study.source_identity() == definition["source"], "learner source changed while preparing")
    definition["plan_sha256"] = frozen["sha256"]
    definition["files"] = _inventory(root)
    definition["sha256"] = digest(definition)
    _write(root / MANIFEST, _json(definition))
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return validate_history_study(root)


def validate_history_study(output_root):
    """Rebuild the grid from live original inputs and verify all frozen artifacts."""
    root = _path(output_root)
    manifest_path = root / MANIFEST
    _require(manifest_path.is_file(), "partial history study has no sealed manifest")
    manifest = _parse(_raw(manifest_path))
    _require(isinstance(manifest, dict) and manifest.get("format") == FORMAT
             and manifest.get("schema_version") == 1
             and digest({k: v for k, v in manifest.items() if k != "sha256"}) == manifest.get("sha256"),
             "history manifest hash/format mismatch")
    for route in ("inputs", "study/configs", "study/jobs", "study/policy_source/transformer_rl"):
        _require(_path(root / route).is_dir(), f"required study directory is missing: {route}")
    spec_path = _path(manifest["inputs"]["spec"]["path"])
    base_path = _path(manifest["inputs"]["base"]["path"])
    spec_raw, base_raw = _raw(spec_path), _raw(base_path)
    expected, expanded, base = _definition(root, spec_path, spec_raw, base_path, base_raw,
        manifest["history_lengths"], manifest["position_reference"])
    _require({k: v for k, v in manifest.items() if k not in {"sha256", "files", "plan_sha256"}} == expected,
             "history definition, original inputs or learner source changed")
    _require(_raw(root / "inputs/spec.json") == spec_raw and _raw(root / "inputs/base.json") == base_raw,
             "frozen input copies differ from actual original bytes")
    _require(_parse(_raw(root / "expanded_spec.json")) == expanded, "expanded specification changed")
    frozen = frame_study.validate_study(root / "study", source=True)
    _require(frozen["spec"] == expanded and frozen["base"] == base.to_dict()
             and frozen["source"] == expected["source"] and frozen["sha256"] == manifest["plan_sha256"],
             "packed study differs from the complete history definition")
    expected_routes = {"inputs/spec.json", "inputs/base.json", "expanded_spec.json", "study/plan.json"}
    expected_routes.update(f"study/{route}" for route in frozen["configs"])
    expected_routes.update(f"study/policy_source/transformer_rl/{route}" for route in expected["source"]["files"])
    inventory = _inventory(root)
    _require(set(inventory) == expected_routes and inventory == manifest["files"],
             "frozen artifact inventory changed or is incomplete")
    # Artifact presence is a filesystem observation, not a process or completion proof.
    execution_state = "job_artifacts_present" if next((root / "study/jobs").iterdir(), None) else "unobserved"
    return {**manifest, "execution_state": execution_state}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--history-lengths", type=int, nargs="+", required=True)
    prepare.add_argument("--position-reference", choices=("current",), required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (prepare_history_study(args.spec, args.base_config, args.output_root,
                    history_lengths=args.history_lengths, position_reference=args.position_reference)
                  if args.operation == "prepare" else validate_history_study(args.output_root))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"history study: {error}\n")
    print(json.dumps({"root": result["root"], "status": result["status"], "sha256": result["sha256"],
                      "budget": result["budget"], "preparation_started_training": False,
                      "execution_state": result["execution_state"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

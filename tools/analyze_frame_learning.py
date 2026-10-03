#!/usr/bin/env python3
"""Read-only, standard-library analysis of sealed frame-study learning logs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def parse(text):
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    # JSON's valid numeric syntax can still overflow a binary float.
    digest(value)
    return value


def integer(value, label, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{label} requires an integer >= {minimum}")
    return value


def number(value, label, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} requires a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("invalid study identifier")
    return value


class Reader:
    """Keep byte identities for every file actually consumed by the analysis."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}
        self.cache = {}

    def path(self, route):
        if not isinstance(route, str) or Path(route).is_absolute():
            raise ValueError("study routes must be relative")
        path = (self.root / route).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("study route escapes root")
        return path

    def raw(self, route):
        if route not in self.cache:
            data = self.path(route).read_bytes()
            self.cache[route] = data
            self.files[route] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return self.cache[route]

    def json(self, route):
        return parse(self.raw(route).decode("utf-8"))


def _sealed_plan(reader):
    plan = reader.json("plan.json")
    if (plan.get("format") != "transformer_rl.packed_study" or plan.get("schema_version") != 1
            or digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256")):
        raise ValueError("plan SHA/format mismatch")
    if digest(plan["source"]["files"]) != plan["source"].get("sha256"):
        raise ValueError("plan source identity mismatch")
    spec, expected = plan["spec"], {}
    names = [name(v["name"]) for v in spec["variants"]]
    seeds = [integer(s, "training seed") for s in spec["seeds"]]
    if not names or len(set(names)) != len(names) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("empty or duplicate variants/seeds")
    stage_names = [name(s["name"]) for s in spec["stages"]]
    if not stage_names or len(set(stage_names)) != len(stage_names):
        raise ValueError("empty or duplicate stages")
    for variant in spec["variants"]:
        for kind, entries in (("train", spec["stages"]), ("eval", spec["scenarios"])):
            for entry in entries:
                config = deepcopy(plan["base"])
                config["model"]["policy"].update(variant["policy"])
                config["environment"].update(entry["environment"])
                route = f"configs/{variant['name']}.{kind}.{name(entry['name'])}.json"
                expected[route] = digest(config)
    if expected != plan["configs"]:
        raise ValueError("configs differ from sealed base/spec")
    for route, sha in plan["configs"].items():
        if digest(reader.json(route)) != sha:
            raise ValueError(f"config SHA mismatch: {route}")
    return plan


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "sample_std": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def _horizon(factor, dt):
    if not 0 <= factor <= 1:
        raise ValueError("discount factors must be in [0,1]")
    return {"seconds": -dt / math.log(factor) if 0 < factor < 1 else (0. if factor == 0 else None),
            "infinite": factor == 1}


def _row(record, run, full_samples, line):
    update = integer(record["update"], f"{line}: update", 1)
    batch = integer(record["batch_samples"], f"{line}: batch_samples", 1)
    collection, opt = record["collection"], record["optimization"]
    envs = run["config"]["environment"]["num_envs"]
    steps = integer(collection["vector_steps"], f"{line}: vector_steps", 1)
    if (batch != integer(collection["transitions"], f"{line}: transitions", 1)
            or steps * envs != batch or batch > full_samples):
        raise ValueError(f"{line}: rollout sample identity mismatch")
    if type(collection["early_stopped"]) is not bool or collection["early_stopped"] != (batch < full_samples):
        raise ValueError(f"{line}: partial rollout identity mismatch")
    ppo = run["config"]["ppo"]
    epochs, chunks = ppo["epochs"], min(ppo["num_minibatches"], batch)
    planned = integer(opt["planned_optimizer_steps"], f"{line}: planned_optimizer_steps", 1)
    applied = integer(opt["optimizer_steps"], f"{line}: optimizer_steps")
    used = integer(opt["sample_count"], f"{line}: sample_count")
    if planned != epochs * chunks or applied > planned:
        raise ValueError(f"{line}: optimizer step identity mismatch")
    # tensor_split may create uneven minibatches; count all endpoint uses exactly.
    quotient, remainder = divmod(batch, chunks)
    complete_epochs, leftover = divmod(applied, chunks)
    expected_used = complete_epochs * batch + leftover * quotient + min(leftover, remainder)
    if used != expected_used:
        raise ValueError(f"{line}: optimization endpoint-use identity mismatch")
    if type(opt["early_stopped"]) is not bool or (not opt["early_stopped"] and applied != planned):
        raise ValueError(f"{line}: KL stop identity mismatch")
    for key in ("grad_norm", "kl", "final_kl", "stop_kl", "clip_fraction"):
        number(opt[key], f"{line}: {key}", 0.)
    if opt.get("first_step_kl") is None:
        if applied:
            raise ValueError(f"{line}: missing first-step KL despite optimizer step")
    else:
        number(opt["first_step_kl"], f"{line}: first_step_kl", 0.)
    number(collection["reward_mean"], f"{line}: reward_mean")
    return update


def _stage(reader, plan, variant, seed, stage, state, first, last):
    spec = plan["spec"]
    config_route = f"configs/{variant}.train.{stage['name']}.json"
    config = reader.json(config_route)
    ppo, policy, dt = config["ppo"], config["model"]["policy"], config["control"]["policy_dt_s"]
    dt = number(dt, "policy_dt_s", 0.)
    if not dt:
        raise ValueError("policy_dt_s must be positive")
    envs = integer(config["environment"]["num_envs"], "num_envs", 1)
    length = integer(policy["history_length"], "history_length", 1)
    rollout = integer(spec["training"]["rollout_steps"], "rollout_steps", 1)
    full_samples = envs * rollout
    for key in ("epochs", "num_minibatches"):
        integer(ppo[key], key, 1)
    for key in ("learning_rate", "target_kl"):
        if not number(ppo[key], key, 0.):
            raise ValueError(f"{key} must be positive")
    gamma = number(ppo["gamma"], "gamma", 0.)
    gae = number(ppo["gae_lambda"], "gae_lambda", 0.)
    if gamma > 1 or gae > 1:
        raise ValueError("discount factors must be <= 1")
    entry = next((e for e in state.get("stages", []) if e["name"] == stage["name"]), None)
    configured_retention = number(spec["training"]["retention_coef"], "configured retention_coef", 0.)
    anchors = entry.get("anchors", []) if entry else []
    if not isinstance(anchors, list):
        raise ValueError("stage anchors must be a list")
    effective_retention = configured_retention if anchors else 0.
    records, attempts = {}, []
    for attempt in entry.get("attempts", []) if entry else []:
        directory = attempt["directory"]
        expected_parent = reader.path(f"jobs/{variant}/seed_{seed}/{stage['name']}")
        resolved = reader.path(directory)
        if resolved.parent != expected_parent or not re.fullmatch(r"attempt_[0-9]{4,}", resolved.name):
            raise ValueError("attempt directory identity mismatch")
        route = f"{directory}/train/run.json"
        if not reader.path(route).exists():
            if reader.path(f"{directory}/train/metrics.jsonl").exists():
                raise ValueError("metrics exist without run identity")
            attempts.append({"directory": directory, "status": "no_run_identity", "rows": 0})
            continue
        run = reader.json(route)
        integer(run.get("seed"), "run seed")
        integer(run.get("rollout_steps"), "run rollout_steps", 1)
        number(run.get("retention_coef"), "run retention_coef", 0.)
        if (run.get("seed") != seed or run.get("environment_factory") != spec["environment_factory"]
                or run.get("config") != config or run.get("source") != plan["source"]
                or run.get("rollout_steps") != rollout
                or run.get("retention_coef") != effective_retention):
            raise ValueError(f"run identity/config differs from sealed plan: {route}")
        integer(run["updates"], "requested run updates", 1)
        metrics_route = f"{directory}/train/metrics.jsonl"
        rows = []
        if reader.path(metrics_route).exists():
            text = reader.raw(metrics_route).decode("utf-8")
            if text and not text.endswith("\n"):
                raise ValueError(f"incomplete JSONL record: {metrics_route}")
            for index, line in enumerate(text.splitlines(), 1):
                record = parse(line)
                identity = f"{metrics_route}:{index}"
                update = _row(record, run, full_samples, identity)
                if rows and update != rows[-1]["update"] + 1:
                    raise ValueError(f"nonconsecutive updates: {identity}")
                if update in records:
                    raise ValueError(f"duplicate/conflicting resumed update {update}: {identity}")
                records[update] = record
                rows.append(record)
        if len(rows) > run["updates"]:
            raise ValueError("log exceeds requested updates")
        if rows and not (run.get("resume") or run.get("restore_learning_from")) and rows[0]["update"] != 1:
            raise ValueError("new learning run must start at update 1")
        completion_route = f"{directory}/train/completion.json"
        if reader.path(completion_route).exists():
            completion = reader.json(completion_route)
            if completion.get("config_sha256") != digest(config):
                raise ValueError("completion config SHA mismatch")
            start = integer(completion["start_update"], "start_update")
            final = integer(completion["final_update"], "final_update")
            integer(completion["completed_updates"], "completed_updates")
            integer(completion["consumed_transitions"], "consumed_transitions")
            if (final - start != len(rows) or completion["completed_updates"] != len(rows)
                    or (rows and (rows[0]["update"] != start + 1 or rows[-1]["update"] != final))):
                raise ValueError("completion/log update identity mismatch")
            if completion["consumed_transitions"] < sum(r["batch_samples"] for r in rows):
                raise ValueError("completion lost collected samples")
        attempts.append({"directory": directory, "rows": len(rows), "status": attempt.get("status"),
                         "first_update": rows[0]["update"] if rows else None,
                         "last_update": rows[-1]["update"] if rows else None,
                         "history_reset": run.get("history_reset"),
                         "episode_state_restored": run.get("episode_state_restored"),
                         "resume_mode": next((k for k in ("resume", "initialize_from", "restore_learning_from")
                                              if run.get(k) is not None), "new")})
    selected = [records[u] for u in range(first, last + 1) if u in records and records[u]["batch_samples"] == full_samples]
    missing = [u for u in range(first, last + 1) if u not in records]
    partial = [{"update": u, "batch_samples": r["batch_samples"], "in_requested_interval": first <= u <= last}
               for u, r in sorted(records.items()) if r["batch_samples"] != full_samples]
    partial_window = [item["update"] for item in partial if item["in_requested_interval"]]
    complete = not missing and not partial_window
    metrics = {}
    for key in ("first_step_kl", "final_kl", "kl", "grad_norm", "clip_fraction", "stop_kl"):
        metrics[key] = _summary([r["optimization"][key] for r in selected if r["optimization"].get(key) is not None])
    used = sum(r["optimization"]["sample_count"] for r in selected)
    budget = sum(r["batch_samples"] * ppo["epochs"] for r in selected)
    applied_steps = sum(r["optimization"]["optimizer_steps"] for r in selected)
    planned_steps = sum(r["optimization"]["planned_optimizer_steps"] for r in selected)
    return {"variant": variant, "training_seed": seed, "stage": stage["name"],
            "job_status": state.get("status"), "sealed_config_sha256": digest(config),
            "learning_rate": ppo["learning_rate"], "target_kl": ppo["target_kl"],
            "gamma": gamma, "gae_lambda": gae,
            "reward_credit_e_fold_horizon": _horizon(gamma, dt),
            "gae_credit_e_fold_horizon": _horizon(gamma * gae, dt),
            "history_frames": length, "history_span_s": (length - 1) * dt,
            "policy_dt_s": dt, "policy_capacity_config": policy,
            "configured_retention_coef": configured_retention,
            "effective_retention_coef": effective_retention, "retention_coef": effective_retention,
            "anchors_capacity": None, "anchors_capacity_reason": "not declared by sealed study",
            "actual_logged_samples": sum(r["batch_samples"] for r in records.values()),
            "actual_partial_rollouts": partial, "attempts": attempts,
            "window": {"first_update": first, "last_update": last, "complete_full_rollouts": complete,
                       "missing_updates": missing, "excluded_partial_updates": partial_window,
                       "selected_updates": [r["update"] for r in selected],
                       "full_rollout_samples": full_samples, "selected_samples": sum(r["batch_samples"] for r in selected),
                       "actual_optimizer_steps": applied_steps,
                       "planned_optimizer_steps": planned_steps,
                       "optimizer_step_fraction": applied_steps / planned_steps if planned_steps else None,
                       "endpoint_uses": used, "planned_endpoint_uses": budget,
                       "endpoint_use_fraction": used / budget if budget else None,
                       "early_stop_fraction": sum(r["optimization"]["early_stopped"] for r in selected) / len(selected) if selected else None,
                       "optimization": metrics,
                       "reward_diagnostic_only": _summary([r["collection"]["reward_mean"] for r in selected])}}


def analyze(study_root, first_update=101, last_update=120):
    integer(first_update, "first_update", 1)
    integer(last_update, "last_update", first_update)
    reader = Reader(study_root)
    plan = _sealed_plan(reader)
    snapshot = None
    if reader.path("snapshot.json").exists():
        snapshot = reader.json("snapshot.json")
        if snapshot.get("plan_sha256") != plan["sha256"]:
            raise ValueError("snapshot plan SHA mismatch")
        for route, receipt in snapshot["files"].items():
            reader.raw(route)
            if reader.files[route] != {"sha256": receipt["sha256"], "bytes": receipt["bytes"]}:
                raise ValueError(f"snapshot file SHA/size mismatch: {route}")
        stamp = snapshot["captured_at"]
        parsed = datetime.fromisoformat(stamp)
        if parsed.tzinfo is None:
            raise ValueError("snapshot captured_at requires timezone")
    else:
        stamp = datetime.now(timezone.utc).isoformat()
    jobs = []
    for variant in plan["spec"]["variants"]:
        for seed in plan["spec"]["seeds"]:
            route = f"jobs/{variant['name']}/seed_{seed}/state.json"
            exists = reader.path(route).exists()
            state = reader.json(route) if exists else {"status": "not_started"}
            if exists and (state.get("variant") != variant["name"]
                    or state.get("seed") != seed or state.get("plan_sha256") != plan["sha256"]):
                raise ValueError("job state identity/plan SHA mismatch")
            if exists:
                integer(state.get("seed"), "state seed")
                stage_names = [name(e["name"]) for e in state["stages"]]
                if len(set(stage_names)) != len(stage_names) or set(stage_names) - {s["name"] for s in plan["spec"]["stages"]}:
                    raise ValueError("job stage identity mismatch")
            for stage in plan["spec"]["stages"]:
                jobs.append(_stage(reader, plan, variant["name"], seed, stage, state, first_update, last_update))
    comparisons = {}
    for stage in plan["spec"]["stages"]:
        items = [j for j in jobs if j["stage"] == stage["name"]]
        complete = all(j["window"]["complete_full_rollouts"] for j in items)
        comparisons[stage["name"]] = {
            "complete_common_update_interval": complete,
            "equal_selected_rollout_samples": complete and len({j["window"]["selected_samples"] for j in items}) == 1,
            "equal_actual_optimizer_steps": complete and len({j["window"]["actual_optimizer_steps"] for j in items}) == 1,
            "equal_endpoint_uses": complete and len({j["window"]["endpoint_uses"] for j in items}) == 1,
            "equal_all_logged_samples": len({j["actual_logged_samples"] for j in items}) == 1,
            "fairness_scope": "common complete rollout window; equal updates do not imply equal total samples or optimizer work",
            "unready": [{"variant": j["variant"], "training_seed": j["training_seed"]}
                        for j in items if not j["window"]["complete_full_rollouts"]]}
    return {"format": "transformer_rl.frame_learning_analysis", "schema_version": 1,
            "snapshot_captured_at": stamp, "snapshot_is_file_sealed": snapshot is not None,
            "study_root": str(reader.root), "plan_sha256": plan["sha256"],
            "policy_source_sha256": plan["source"]["sha256"],
            "policy_source_verification": "manifest and run identity metadata; source code not executed",
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "first_update": first_update, "last_update": last_update, "jobs": jobs,
            "comparisons": comparisons,
            "forgetting": {"available": False, "skill_retention_rate": None,
                           "stage_count": len(plan["spec"]["stages"]),
                           "reason": "missing verified acquired-skill reference and matched later control evaluation",
                           "matrix_consumed": False},
            "interpretation": {"kl_gradient_reward": "optimization diagnostics; not evidence of forgetting or physical jitter",
                               "horizons": "reward/GAE credit decay; not temporal observation memory",
                               "statistics": "across updates, not across independent training seeds; sample_std is not a confidence interval",
                               "branches": "overlapping rollback/resume update identities require separate branch analysis; this tool rejects them"},
            "source_files": reader.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-update", type=int, default=101)
    parser.add_argument("--last-update", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve().is_relative_to(args.study_root.resolve()):
        raise ValueError("analysis output must be outside the read-only study")
    report = analyze(args.study_root, args.first_update, args.last_update)
    # Never mutate the study or overwrite an existing analysis artifact.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "plan_sha256": report["plan_sha256"],
                      "jobs": len(report["jobs"]), "forgetting_available": False}))


if __name__ == "__main__":
    main()

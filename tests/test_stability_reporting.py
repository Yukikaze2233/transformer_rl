"""Stability reporting uses constructed reports and fake subprocesses, never an env."""
import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformer_rl import checkpoint, cli, evaluation, experiments as ex
from transformer_rl.config import ModelConfig
from transformer_rl.stability import EpisodeSignalStatistics
from test_experiments import fake_runner, specification  # noqa: F401
from test_scenario_experiments import scenario_runner, scenario_spec  # noqa: F401


STATS = ("mean", "within_episode_std", "derivative_rms", "max_abs")


def stability_report(transitions=20, *, count=None, bias=100, jitter=0, minimum=2):
    """Unqualified samples are singleton failure segments, without settling."""
    count = transitions if count is None else count
    segments = int(count > 0)
    short = transitions - count
    signal = {
        "mean": bias if count else None, "within_episode_std": jitter if count else None,
        "derivative_rms": jitter * 10 if count > 1 else None,
        "max_abs": abs(bias) + jitter if count else None,
        "count": count, "segments": segments, "derivative_count": count - segments,
        "short_count": short, "short_segments": short, "total_count": transitions,
        "settled_count": 0, "completed_segments": 0, "partial_segments": segments,
        "short_completed_segments": short, "short_partial_segments": 0,
        "episode_mean_min": bias if count else None, "episode_mean_max": bias if count else None,
        "episode_mean_std": 0 if count else None,
    }
    protocol = EpisodeSignalStatistics(1, settle_steps=0, min_steady_samples=minimum).report()["protocol"]
    return {"available": count > 0, "protocol": protocol, "signals": {"pitch_error": signal}}


def evaluation_report(stability):
    return {"checkpoint_sha256": "digest", "seed": 101, "vector_steps": 10,
            "transitions": 20, "reward_mean": 1, "terminated_count": 0,
            "truncated_count": 0, "done_count": 0, "environment": {}, "action_clip": None,
            "policy": "deterministic_mean",
            "metrics": {"tracking_error": {"mean": 1, "rms": 1, "min": 1, "max": 1, "count": 20}},
            "stability": stability}


@pytest.mark.parametrize("windows", [{}, {"settle_steps": 0}, {"min_steady_samples": 1},
                                      {"settle_steps": 3, "min_steady_samples": 4}])
def test_cli_forwards_keyword_only_windows(tmp_path, monkeypatch, windows):
    config, output = tmp_path / "config.json", tmp_path / "report.json"
    config.write_text("{}")
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *_: (
        SimpleNamespace(config=ModelConfig()), None, None, None))
    factory = object()
    monkeypatch.setattr(cli, "_factory", lambda *_: factory)
    calls = []

    def evaluate(*args, **kwargs):
        assert args[1] is factory and args[5] == "cpu"
        calls.append(kwargs)
        return {"fake": True}

    signature = inspect.signature(evaluation.evaluate_policy)
    monkeypatch.setattr(evaluation, "evaluate_policy", evaluate)
    flags = [item for key, value in windows.items() for item in ("--" + key.replace("_", "-"), str(value))]
    assert cli.main(["evaluate", "--config", str(config), "--checkpoint", "fake.pt",
                     "--env-factory", "unused:factory", "--steps", "10", "--seed", "101",
                     "--output", str(output), *flags]) == 0
    expected = {key: windows.get(key, signature.parameters[key].default)
                for key in ("settle_steps", "min_steady_samples")}
    assert calls == [expected]
    assert all(signature.parameters[k].kind == inspect.Parameter.KEYWORD_ONLY for k in expected)


@pytest.mark.parametrize("flag,value", [("--settle-steps", "-1"), ("--settle-steps", "1.5"),
                                        ("--min-steady-samples", "0"), ("--min-steady-samples", "true")])
def test_cli_rejects_invalid_windows(flag, value, monkeypatch):
    monkeypatch.setattr(cli, "_factory", lambda *_: pytest.fail("must not resolve factory"))
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "--config", "unused", "--checkpoint", "unused",
                  "--env-factory", "unused:factory", "--steps", "10", "--seed", "1",
                  "--output", "unused", flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize("windows", [{}, {"settle_steps": 0}, {"min_steady_samples": 1},
                                      {"settle_steps": 200, "min_steady_samples": 200}])
def test_worker_only_appends_explicit_flags_and_hashes_defaults(specification, windows):
    path, root, spec = specification
    spec["seeds"] = [11]
    spec["evaluation"]["seeds"] = [101]
    spec["evaluation"].update(windows)
    spec["execution"]["devices"] = ["cpu"]
    path.write_text(json.dumps(spec))
    manifest = ex.plan(path, root)
    assert manifest["spec"] == spec
    assert {"stability.py", "evaluation.py", "cli.py", "experiments.py"} <= set(manifest["source"]["files"])
    calls = []

    def runner(argv, log, deadline, stop):
        calls.append(argv[3])
        for key in ("settle_steps", "min_steady_samples"):
            flag = "--" + key.replace("_", "-")
            assert (flag in argv) == (argv[3] == "evaluate" and key in windows)
            if flag in argv:
                assert argv[argv.index(flag) + 1] == str(windows[key])
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            report = ex._read(output)
            report["stability"] = EpisodeSignalStatistics(1, **windows).report()
            output.write_text(json.dumps(report))

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == "completed"
    assert calls == ["train", "evaluate"]
    summary = ex.summarize(root)
    assert summary["stability_protocol"]["settle_steps"] == windows.get("settle_steps", 200)
    assert summary["stability_protocol"]["min_steady_samples"] == windows.get("min_steady_samples", 200)
    assert not summary["variants"][0]["stability_complete"]


@pytest.mark.parametrize("key,value", [("settle_steps", -1), ("settle_steps", True),
                                       ("settle_steps", 0.0), ("min_steady_samples", 0),
                                       ("min_steady_samples", False), ("min_steady_samples", 2.0)])
def test_invalid_spec_windows_do_not_create_plan(specification, key, value):
    path, root, spec = specification
    spec["evaluation"][key] = value
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=key):
        ex.plan(path, root)
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["settle", "minimum", "reset", "weighting", "bool_window",
                                     "negative_count", "bool_count", "derivative_count", "total_count",
                                     "nan", "infinity", "missing_value", "negative_std", "false_available",
                                     "fake_zero", "fake_derivative", "termination_budget", "metric_budget", "done_budget"])
def test_report_rejects_protocol_and_semantic_corruption(specification, tmp_path, mutation):
    _, _, spec = specification
    spec["evaluation"].update(settle_steps=0, min_steady_samples=2)
    block = stability_report(count=0 if mutation == "fake_zero" else 20)
    signal, protocol = block["signals"]["pitch_error"], block["protocol"]
    changes = {"settle": (protocol, "settle_steps", 1), "minimum": (protocol, "min_steady_samples", 3),
               "reset": (protocol, "centering", "whole_rollout"),
               "weighting": (protocol, "derivative_weighting", "duration"),
               "bool_window": (protocol, "settle_steps", False),
               "negative_count": (signal, "short_segments", -1), "bool_count": (signal, "count", True),
               "derivative_count": (signal, "derivative_count", 20),
               "total_count": (signal, "total_count", 21), "nan": (signal, "mean", float("nan")),
               "infinity": (signal, "max_abs", float("inf")), "missing_value": (signal, "mean", None),
               "negative_std": (signal, "within_episode_std", -1),
               "false_available": (block, "available", False), "fake_zero": (signal, "within_episode_std", 0),
               "fake_derivative": (signal, "derivative_rms", None)}
    report = evaluation_report(block)
    changes.update(termination_budget=(report, "terminated_count", 21), done_budget=(report, "done_count", 1),
                   metric_budget=(report["metrics"]["tracking_error"], "count", 21))
    target, key, value = changes[mutation]
    target[key] = value
    output = tmp_path / "report.json"
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        ex._report(output, "digest", 101, spec, {})


def test_singleton_derivative_null_is_valid(specification, tmp_path):
    _, _, spec = specification
    spec["evaluation"].update(settle_steps=0, min_steady_samples=1)
    block = stability_report(minimum=1)
    block["signals"]["pitch_error"].update(segments=20, completed_segments=20, partial_segments=0,
                                            derivative_count=0, derivative_rms=None)
    output = tmp_path / "report.json"
    report = evaluation_report(block)
    report.update(terminated_count=20, done_count=20)
    ex._write(output, report)
    report = ex._report(output, "digest", 101, spec, {})
    means, coverage = ex._stability_seed_means([report], "")
    assert means["stability.pitch_error.within_episode_std"] == 0
    assert means["stability.pitch_error.derivative_rms"] is None
    assert coverage["stability.pitch_error.derivative_rms"]["missing_eval_seeds"] == [101]


@pytest.mark.parametrize("partial", [False, True])
def test_scenario_seed_hierarchy_and_coverage_do_not_rank_bias_as_success(scenario_spec, partial):
    path, root, spec = scenario_spec
    spec["evaluation"].update(settle_steps=0, min_steady_samples=2)
    spec["execution"]["devices"] = ["cpu"]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        scenario_runner(argv, log, deadline, stop)
        if argv[3] != "evaluate":
            return
        output = Path(argv[argv.index("--output") + 1])
        report = ex._read(output)
        train_seed = int(output.parent.name.split("_")[1])
        mlp = output.parent.parent.name == "mlp"
        low = "stand_low" in output.name
        # Different qualified sample counts must not become a fairness requirement.
        count = 10 if mlp else report["transitions"]
        if partial and low and (train_seed == 33 or (train_seed == 11 and report["seed"] == 103)):
            count = 0
        report["stability"] = stability_report(
            report["transitions"], count=count, bias=report["reward_mean"])
        report["terminated_count"] = report["transitions"] - count
        # A fully missing block in one training seed exercises the union across seeds.
        if partial and train_seed == 33 and low:
            del report["stability"]
        output.write_text(json.dumps(report))

    assert all(j["status"] == "completed" for j in ex.run(root, runner=runner)["jobs"])
    summary = ex.summarize(root)
    for row in summary["variants"]:
        assert row["completed"] == 3 and row["failed"] == 0
        assert row["evaluation_complete"] and row["physical_metrics_complete"] and row["comparison_available"]
        assert row["stability_complete"] is (not partial)
        assert row["stability_comparison_available"] is (not partial)
        high = "scenarios.stand_high.stability.pitch_error."
        low = "scenarios.stand_low.stability.pitch_error."
        assert row["evaluation"][high + "mean"] == {"n": 3, "mean": 464, "std": 11}
        assert row["evaluation"][high + "within_episode_std"] == {"n": 3, "mean": 0, "std": 0}
        assert row["evaluation"][low + "mean"]["mean"] == (418 if partial else 424)
        if partial:
            assert row["evaluation"][low + "mean"]["n"] == 2
            assert row["evaluation"][low + "mean"]["std"] == pytest.approx(12 / 2**0.5)
            coverage = row["stability_coverage"][low + "mean"]
            assert coverage["available_eval_runs"] == 3 and coverage["missing_eval_runs"] == 3
            assert coverage["missing_training_seeds"] == [33]
            assert coverage["incomplete_training_seeds"] == [11, 33]
            assert coverage["transitions"] == 120
            assert coverage["sample_coverage"] == (0.5 if row["variant"] == "attention" else 0.25)
            assert row["seeds"][2]["stability_coverage"]["scenarios.stand_low.stability"][
                "missing_report_eval_seeds"] == [101, 103]
    checks = summary["fairness_checks"]["architecture"]
    assert checks["scenarios"]["scenarios.stand_high"]["stability_comparison_available"]
    assert checks["stability_comparison_available"] is (not partial)
    assert "winner" not in summary and all("winner" not in row for row in summary["variants"])
    assert "large bias" in summary["interpretation"] and "not standing success" in summary["interpretation"]
    with (root / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(r["metric"] == "scenarios.stand_low.stability.pitch_error.max_abs" for r in rows)
    low_mean = next(r for r in rows if r["metric"] == "scenarios.stand_low.stability.pitch_error.mean")
    assert json.loads(low_mean["stability_coverage"])["available_eval_runs"] == (3 if partial else 6)


def test_all_null_and_legacy_reports_preserve_physical_results(specification):
    path, root, spec = specification
    spec["seeds"] = [11, 22]
    spec["evaluation"].update(settle_steps=0, min_steady_samples=2)
    spec["execution"]["devices"] = ["cpu"]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            if output.parent.name == "seed_11":
                report = ex._read(output)
                report["stability"] = stability_report(count=0)
                output.write_text(json.dumps(report))

    assert all(j["status"] == "completed" for j in ex.run(root, runner=runner)["jobs"])
    row = ex.summarize(root)["variants"][0]
    assert row["evaluation_complete"] and row["physical_metrics_complete"]
    assert not row["stability_complete"] and not row["stability_comparison_available"]
    assert row["evaluation"]["reward_mean"] == {"n": 2, "mean": 118.5, "std": pytest.approx(11 / 2**0.5)}
    for stat in STATS:
        key = "stability.pitch_error." + stat
        assert row["evaluation"][key] == {"n": 0, "mean": None, "std": None}
        assert row["seeds"][0]["evaluation_seed_means"][key] is None
        assert row["stability_coverage"][key]["missing_training_seeds"] == [11, 22]
    with (root / "summary.csv").open() as stream:
        metric = next(r for r in csv.DictReader(stream) if r["metric"] == "stability.pitch_error.mean")
    assert metric["mean"] == "" and metric["n"] == "0"


def test_old_report_is_explicitly_missing_stability(specification, tmp_path):
    _, _, spec = specification
    report = evaluation_report(None)
    del report["stability"]
    output = tmp_path / "report.json"
    ex._write(output, report)
    assert ex._report(output, "digest", 101, spec, {})["stability_missing"]


def test_singleton_job_completes_but_derivative_comparison_is_unavailable(specification):
    path, root, spec = specification
    spec["seeds"] = [11]
    spec["evaluation"].update(seeds=[101], settle_steps=0, min_steady_samples=1)
    spec["execution"]["devices"] = ["cpu"]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            report = ex._read(output)
            block = stability_report(minimum=1)
            block["signals"]["pitch_error"].update(segments=20, completed_segments=20, partial_segments=0,
                                                    derivative_count=0, derivative_rms=None)
            report["stability"] = block
            report.update(terminated_count=20, done_count=20)
            output.write_text(json.dumps(report))

    assert ex.run(root, runner=runner)["jobs"][0]["status"] == "completed"
    row = ex.summarize(root)["variants"][0]
    assert row["completed"] == 1 and row["physical_metrics_complete"]
    assert not row["stability_complete"] and not row["stability_comparison_available"]
    assert row["evaluation"]["stability.pitch_error.mean"]["mean"] == 100
    assert row["evaluation"]["stability.pitch_error.within_episode_std"]["mean"] == 0
    assert row["evaluation"]["stability.pitch_error.derivative_rms"] == {"n": 0, "mean": None, "std": None}


def test_summary_rejects_cross_training_seed_protocol_mismatch(specification):
    path, root, spec = specification
    spec["seeds"] = [11, 22]
    spec["evaluation"].update(seeds=[101], settle_steps=0, min_steady_samples=2)
    spec["execution"]["devices"] = ["cpu"]
    path.write_text(json.dumps(spec))
    ex.plan(path, root)

    def runner(argv, log, deadline, stop):
        fake_runner(argv, log, deadline, stop)
        if argv[3] == "evaluate":
            output = Path(argv[argv.index("--output") + 1])
            report = ex._read(output)
            report["stability"] = stability_report()
            output.write_text(json.dumps(report))

    ex.run(root, runner=runner)
    target = root / "jobs/attention/seed_22/evaluation_101.json"
    report = ex._read(target)
    report["stability"]["protocol"]["centering"] = "whole_rollout"
    target.write_text(json.dumps(report))
    row = ex.summarize(root)["variants"][0]
    assert row["completed"] == 1 and row["failed"] == 1
    assert "stability protocol mismatch" in row["seeds"][1]["error"]
    assert row["evaluation"]["stability.pitch_error.mean"]["n"] == 1
    assert not row["stability_comparison_available"]


@pytest.mark.parametrize("qualified", [False, True])
def test_default_protocol_accepts_short_and_qualified_windows(specification, tmp_path, qualified):
    _, _, spec = specification
    spec["evaluation"]["steps"] = 400
    block = stability_report(800, count=400 if qualified else 0)
    block["protocol"] = EpisodeSignalStatistics(1).report()["protocol"]
    block["signals"]["pitch_error"].update(
        settled_count=400 if qualified else 800, short_count=0,
        short_segments=0 if qualified else 8, short_completed_segments=0 if qualified else 8,
        segments=2 if qualified else 0, partial_segments=2 if qualified else 0,
        derivative_count=398 if qualified else 0)
    report = evaluation_report(block)
    report.update(transitions=800, vector_steps=400, terminated_count=0 if qualified else 8,
                  done_count=0 if qualified else 8)
    output = tmp_path / "report.json"
    ex._write(output, report)
    assert ex._report(output, "digest", 101, spec, {})["stability"]["available"] is qualified

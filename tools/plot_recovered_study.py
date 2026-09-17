"""Plot the completed recovered study using only CPU JSON analysis and Agg.

Usage: python tools/plot_recovered_study.py --output artifacts/plots/new-directory
The output directory must not exist. No checkpoint or simulator is loaded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = REPO / (
    "artifacts/recovered/architecture-16m-20260915T0540/"
    "recovery-post-stop-20260917T0900Z/extracted/study"
)
MODELS = ("last_token_attention", "time_attention", "index_attention", "gated_attention")
LABELS = ("Last-token", "Time", "Index", "Gated")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
STYLES = ("-", "--", "-.", ":")
SEEDS = (1011, 1022, 1033)
MARKERS = ("o", "s", "^")
HEIGHTS = ("stand_low", "stand_mid", "stand_high")
SCENARIOS = HEIGHTS + ("reverse", "forward", "turn_right", "turn_left")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


class StudyPlots:
    def __init__(self, study, window):
        self.study = study.resolve(strict=True)
        self.window = window
        self.inputs = {}
        self.training = {}
        self.evaluation = {}
        self.runs = []
        self.excluded = []
        self.recovery = self.study.parent.parent
        manifest_path = self.recovery / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.inputs[str(manifest_path)] = file_record(manifest_path)
        self.inventory = {entry["path"]: entry for entry in self.manifest["files"]}

    def read(self, path, jsonl=False):
        record = file_record(path)
        relative = str(path.relative_to(self.study.parent))
        expected = self.inventory[relative]
        require(record["sha256"] == expected["sha256"]
                and record["bytes"] == expected["size"], f"SHA/size mismatch: {path}")
        record["manifest_verified"] = True
        self.inputs[str(path)] = record
        text = path.read_text(encoding="utf-8")
        if jsonl:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return json.loads(text)

    def load(self):
        run = self.study / "run"
        self.plan = self.read(run / "plan.json")
        commands = {s["name"]: s["environment"]["fixed_command"]
                    for s in self.plan["spec"]["evaluation"]["scenarios"]}
        for model in MODELS:
            for seed in SEEDS:
                job = run / "jobs" / model / f"seed_{seed}"
                completion = self.read(job / "train/completion.json")
                metadata = self.read(job / "train/run.json")
                result = self.read(job / "result.json")
                require(result["status"] == completion["status"] == "completed"
                        and completion["stop_reason"] == "updates_completed",
                        f"Incomplete job: {job}")
                rows = self.read(job / "train/metrics.jsonl", jsonl=True)
                require(len(rows) == completion["updates_completed"] == 977,
                        f"Unexpected update budget: {job}")
                require([r["update"] for r in rows] == list(range(1, 978)),
                        f"Missing/duplicate updates: {job}")
                cumulative = np.cumsum([r["collection"]["transitions"] for r in rows])
                x = np.array([r["collection"].get("total_transitions", int(total))
                              for r, total in zip(rows, cumulative)])
                require(np.array_equal(x, cumulative) and np.all(np.diff(x) > 0),
                        f"Collection counters disagree: {job}")
                require(x[-1] == completion["collected_transitions"] == 16007168,
                        f"Unexpected transition budget: {job}")
                require(metadata["requested_updates"] == 977
                        and metadata["environment"]["num_envs"] == 512
                        and metadata["rollout_steps"] == 32
                        and metadata["metadata"]["seed"] == seed, "Run identity/budget")
                optim = [r["optimization"] for r in rows]
                actual = np.array([o["optimizer_steps"] for o in optim])
                planned = np.array([o["planned_optimizer_steps"] for o in optim])
                require(np.all(planned > 0) and np.all(actual >= 0)
                        and np.all(actual <= planned), "Invalid optimizer utilization")
                values = np.array([
                    [r["collection"]["reward_mean"] for r in rows],
                    [o["value_loss"] for o in optim],
                    [o["actor_loss"] for o in optim],
                    [o["kl"] for o in optim],
                    [o["entropy"] for o in optim],
                    actual / planned * 100,
                ], dtype=float)
                require(np.isfinite(values).all(), "Nonfinite training data")
                self.training[model, seed] = (x / 1e6, values)
                self.runs.append({
                    "model": model, "training_seed": seed, "status": "completed",
                    "updates": len(rows), "collected_transitions": int(x[-1]),
                    "num_envs": 512, "rollout_steps": 32,
                    "started_at": metadata["started_at"],
                    "elapsed_s": completion["elapsed_s"],
                    "source": metadata["metadata"]["source"],
                    "optimizer_steps": int(actual.sum()),
                    "planned_optimizer_steps": int(planned.sum()),
                    "optimizer_utilization": float(actual.sum() / planned.sum()),
                })
                checkpoint_sha = next(c["sha256"] for c in completion["checkpoints"]
                                      if c["update"] == 977)
                for scenario in SCENARIOS:
                    report = self.read(job / f"evaluation_{scenario}_301.json")
                    require(report["checkpoint_sha256"] == checkpoint_sha
                            and report["checkpoint_update"] == 977, "Eval checkpoint identity")
                    require(report["seed"] == 301 and report["num_envs"] == 8
                            and report["vector_steps"] == 2000
                            and report["transitions"] == 16000
                            and report["policy"] == "deterministic_mean", "Eval budget")
                    require([report[k] for k in ("terminated_count", "truncated_count", "done_count")]
                            == [0, 8, 8], "Unexpected termination counts")
                    protocol = report["stability"]["protocol"]
                    require(protocol["settle_steps"] == protocol["min_steady_samples"] == 200
                            and protocol["centering"] == "per_environment_episode",
                            "Steady-state protocol mismatch")
                    command = report["environment"]["fixed_command"]
                    require(command == commands[scenario], "Scenario command mismatch")
                    signals = report["stability"]["signals"]
                    for name in ("height_error", "vx_error", "wz_error"):
                        require(signals[name]["count"] == 14392
                                and signals[name]["segments"] == 8, "Steady sample support")
                    row = {
                        "height_command_m": command[2],
                        "height_signed_bias_mm": signals["height_error"]["mean"] * 1000,
                        "height_within_episode_std_mm": signals["height_error"]["within_episode_std"] * 1000,
                        "vx_signed_error_m_s": signals["vx_error"]["mean"],
                        "wz_signed_error_rad_s": signals["wz_error"]["mean"],
                        "nonwheel_netforce_full_interval_N": report["metrics"]["non_wheel_net_force"]["mean"],
                        "vx_full_interval_mae_m_s": report["metrics"]["vx_abs_error"]["mean"],
                        "wz_full_interval_mae_rad_s": report["metrics"]["wz_abs_error"]["mean"],
                        "actual_height_m": command[2] + signals["height_error"]["mean"],
                        "actual_vx_m_s": command[0] + signals["vx_error"]["mean"],
                        "actual_wz_rad_s": command[1] + signals["wz_error"]["mean"],
                    }
                    require(np.isfinite(list(row.values())).all(), "Nonfinite eval data")
                    self.evaluation[model, seed, scenario] = row
        for job in self.plan["jobs"]:
            if job["variant"] in MODELS:
                continue
            path = run / job["directory"] / "train/completion.json"
            item = {"model": job["variant"], "training_seed": job["seed"]}
            if path.exists():
                completion = self.read(path)
                item.update(status=completion["status"], reason=completion["stop_reason"],
                            updates=completion["updates_completed"],
                            collected_transitions=completion["collected_transitions"])
            else:
                item.update(status="not_started", reason="No training completion or metrics")
                require(not (path.parent / "metrics.jsonl").exists(), "Unaccounted partial job")
            self.excluded.append(item)

    def crosscheck_csv(self, path):
        self.inputs[str(path)] = file_record(path)
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 84, "Expected 84 reference CSV rows")
        checked, seen = 0, set()
        for row in rows:
            key = (row["variant"], int(row["training_seed"]), row["scenario"])
            require(key not in seen, "Duplicate CSV key")
            seen.add(key)
            for name, value in self.evaluation[key].items():
                if name in row:
                    require(np.isclose(value, float(row[name]), rtol=1e-12, atol=1e-12),
                            f"CSV discrepancy: {key}/{name}")
                    checked += 1
        require(seen == set(self.evaluation), "CSV coverage mismatch")
        return {"rows": len(rows), "numeric_values": checked, "matched": True,
                "usage": "Cross-check only; plots derived from post-stop raw JSON"}

    def frame(self, title, subtitle, shape, footer):
        fig, axes = plt.subplots(*shape, figsize=(16, 11 if shape == (3, 2) else 10))
        fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.80,
                            hspace=.48, wspace=.28)
        fig.suptitle(title, fontsize=23, fontweight="bold", y=.975)
        fig.text(.5, .932, subtitle, ha="center", fontsize=12)
        handles = [Line2D([], [], color=c, ls=s, lw=2.5, label=l)
                   for c, s, l in zip(COLORS, STYLES, LABELS)]
        handles += [Line2D([], [], color="#555555", marker=m, ls="", markersize=5,
                           label=f"Seed {s}") for m, s in zip(MARKERS, SEEDS)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .908),
                   ncol=7, frameon=False, fontsize=11)
        fig.text(.075, .025, footer, fontsize=11, linespacing=1.6)
        for ax in axes.flat:
            ax.grid(True, color="#e4e7eb", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
        return fig, axes

    def training_figure(self):
        fig, axes = self.frame(
            "Training dynamics | raw metrics.jsonl",
            f"12 complete runs | 977 updates / 16.007168M transitions each | "
            f"{self.window}-update trailing mean per seed, then mean +/- sample SD (not CI)",
            (3, 2),
            "Thin lines: individual smoothed training seeds; shading: between-seed sample SD (n=3). "
            "Early windows use available updates.\n"
            "KL: pre-step minibatch estimate, NOT final-policy full-rollout KL. "
            "Policy loss = raw actor_loss; reward is per collected transition.\n"
            "Entropy is differential entropy: negative values alone do not imply collapse. "
            "X counts collected transitions, never optimizer sample reuse.")
        titles = ("Reward per transition", "Value loss", "Policy loss (actor_loss)",
                  "Pre-step minibatch KL (not final rollout KL)", "Differential entropy",
                  "Optimizer utilization: actual / planned")
        for index, ax in enumerate(axes.flat):
            for model, color, style in zip(MODELS, COLORS, STYLES):
                curves = []
                for seed in SEEDS:
                    x, values = self.training[model, seed]
                    y = values[index]
                    sums = np.concatenate(([0.], np.cumsum(y)))
                    ends = np.arange(1, len(y) + 1)
                    starts = np.maximum(0, ends - self.window)
                    smooth = (sums[ends] - sums[starts]) / (ends - starts)
                    curves.append(smooth)
                    ax.plot(x, smooth, color=color, alpha=.22, linewidth=.55)
                require(all(np.array_equal(x, self.training[model, s][0]) for s in SEEDS),
                        "Training seeds have different x grids")
                mean, sd = np.mean(curves, axis=0), np.std(curves, axis=0, ddof=1)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.13)
                ax.plot(x, mean, color=color, ls=style, lw=1.9)
            ax.set_title(titles[index], loc="left", fontsize=13, fontweight="bold")
            ax.set_xlabel("Collected transitions (million)")
            ax.set_xlim(0, 16.007168)
            if index == 5:
                ax.set_ylabel("% of planned optimizer steps")
                ax.set_ylim(-2, 105)
        return fig

    def evaluation_panel(self, ax, scenarios, x, field, title, xlabel, ylabel, target=False):
        x = np.asarray(x)
        if target:
            ax.plot(x, x, color="#222222", ls="--", lw=1.5)
            ax.text(.03, .94, "Dashed black: ideal y = x", transform=ax.transAxes,
                    fontsize=10, va="top")
        else:
            ax.axhline(0, color="#888888", lw=.8)
        for model, color, style in zip(MODELS, COLORS, STYLES):
            values = np.array([[self.evaluation[model, seed, s][field] for s in scenarios]
                               for seed in SEEDS])
            for marker, ys in zip(MARKERS, values):
                ax.plot(x, ys, color=color, marker=marker, markersize=4,
                        lw=.7, alpha=.40)
            ax.errorbar(x, values.mean(axis=0), yerr=values.std(axis=0, ddof=1),
                        color=color, ls=style, lw=2, capsize=4, elinewidth=1.3)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.margins(x=.12, y=.12)

    def command_figure(self):
        fig, axes = self.frame(
            "Independent evaluation | command response",
            "4 architectures x 3 training seeds | 7 scenarios each | "
            "lines: mean +/- sample SD; points: individual seeds (not 8 envs)",
            (2, 2),
            "Actual steady response = command + signed steady error; discard first 200 steps per env/episode; "
            "minimum 200 retained samples.\n"
            "Eval seed 301; 2,000 vector steps x 8 envs; deterministic mean. "
            "14,392/16,000 steady samples per report; timeout/reset included.\n"
            "Net force is FULL-INTERVAL non-wheel rigid-body net-force history peak, NOT ground-pair force. "
            "Low jitter is not successful tracking.")
        panels = [
            (HEIGHTS, [.28, .30, .32], "actual_height_m", "Height response: mostly flat", "Height command (m)", "Actual steady height (m)", True),
            (("reverse", "stand_mid", "forward"), [-.5, 0, .5], "actual_vx_m_s", "Forward / reverse response: near zero", "vx command (m/s)", "Actual steady vx (m/s)", True),
            (("turn_right", "stand_mid", "turn_left"), [-1, 0, 1], "actual_wz_rad_s", "Yaw response: inconsistent across seeds", "wz command (rad/s)", "Actual steady wz (rad/s)", True),
            (HEIGHTS, [.28, .30, .32], "nonwheel_netforce_full_interval_N", "Non-wheel net force | full interval", "Height command (m)", "Mean net-force metric (N)", False),
        ]
        for ax, panel in zip(axes.flat, panels):
            self.evaluation_panel(ax, *panel)
        return fig

    def steady_figure(self):
        fig, axes = self.frame(
            "Standing quality | bias and jitter must be read together",
            "Low / mid / high commands | mean +/- sample SD across 3 independent training seeds (not CI)",
            (2, 2),
            "Height within_episode_std: centered separately per environment/episode, then pooled by retained samples; "
            "NOT between-seed SD.\n"
            "Steady metrics discard 200 steps per episode (minimum 200 retained); error bars are across training seeds, "
            "not the 8 evaluation envs.\n"
            "Net force includes transients/resets and is NOT ground-pair force. "
            "Symmetric sample-SD bars may cross zero; no negative jitter/force observation implied.")
        panels = [
            ("height_signed_bias_mm", "Height signed bias | actual - command", "Steady height bias (mm)"),
            ("height_within_episode_std_mm", "Within-episode height jitter", "Within-episode std (mm)"),
            ("vx_signed_error_m_s", "Stationary-command vx drift", "Steady signed vx (m/s)"),
            ("nonwheel_netforce_full_interval_N", "Non-wheel net force | full interval", "Mean net-force metric (N)"),
        ]
        for ax, (field, title, ylabel) in zip(axes.flat, panels):
            self.evaluation_panel(ax, HEIGHTS, [.28, .30, .32], field, title,
                                  "Height command (m)", ylabel)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path,
                        default=REPO / "artifacts/plots/architecture-recovery-20260917")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--reference-csv", type=Path,
                        default=REPO / "docs/evidence/training-recovery.csv")
    args = parser.parse_args()
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "Output directory already exists")
    require(output.resolve().is_relative_to(REPO / "artifacts/plots"),
            "Output must be under ignored artifacts/plots")
    require(not output.resolve().is_relative_to(args.study.resolve()), "Output overlaps input")
    require(1 <= args.window <= 977, "Window must be between 1 and 977 updates")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.labelsize": 11})
    plots = StudyPlots(args.study, args.window)
    plots.load()
    csv_check = plots.crosscheck_csv(args.reference_csv.resolve())
    output.mkdir(parents=True, exist_ok=False)
    images = []
    for filename, make in (("training_curves.png", plots.training_figure),
                           ("command_tracking.png", plots.command_figure),
                           ("steady_state.png", plots.steady_figure)):
        fig = make()
        path = output / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        with Image.open(path) as image:
            image.load()
            require(image.format == "PNG" and image.width >= 1400, "Invalid PNG")
            record = file_record(path)
            record["dimensions_px"] = list(image.size)
            require(np.asarray(image).std() > 1, "Empty image")
        images.append(record)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": str(plots.study), "execution": "CPU-only numpy/matplotlib Agg; no model loading",
        "script": file_record(Path(__file__).resolve()),
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "snapshot": {k: plots.manifest[k] for k in ("snapshot_start", "snapshot_end", "semantics")},
        "source": plots.plan["source"], "runs": plots.runs,
        "excluded_incomplete_or_unstarted": plots.excluded,
        "checks": {"complete_runs": 12, "metrics_rows": 12 * 977, "eval_reports": 84,
                   "input_SHA_size_vs_manifest": "all consumed study files matched",
                   "csv_crosscheck": csv_check, "png_decode_and_nonempty": "passed"},
        "statistics": {
            "training": {"source": "raw train/metrics.jsonl", "window_updates": args.window,
                         "smoothing": "trailing arithmetic mean per seed; min_periods=1",
                         "aggregation": "mean +/- sample SD (ddof=1), n=3 training seeds, not CI",
                         "x": "collection.total_transitions checked against cumulative collection.transitions / 1e6",
                         "policy_loss": "optimization.actor_loss",
                         "kl": "optimization.kl: pre-step minibatch estimate, not final rollout KL",
                         "entropy": "raw differential entropy; negative does not imply collapse",
                         "utilization": "100 * optimizer_steps / planned_optimizer_steps per update before smoothing"},
            "evaluation": {"source": "raw evaluation_*_301.json", "seed": 301,
                           "num_envs": 8, "vector_steps": 2000, "transitions": 16000,
                           "policy": "deterministic_mean", "settle_steps_per_episode": 200,
                           "min_steady_samples": 200, "retained_samples": 14392,
                           "response": "command + signed error mean",
                           "within_episode_std": "per-environment/episode centered; retained-sample denominator; height converted to mm",
                           "error_bars": "sample SD across 3 independent training seeds (ddof=1), not envs or CI; untruncated symmetric bars",
                           "netforce": "full-interval non-wheel rigid-body net-force history peak, not ground-pair force"},
        },
        "derived_evaluation": [{"model": m, "training_seed": s, "scenario": c, **v}
                               for (m, s, c), v in plots.evaluation.items()],
        "inputs": list(plots.inputs.values()), "images": images,
    }
    with (output / "plot_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"images": images, "checks": receipt["checks"],
                      "receipt": str(output / "plot_receipt.json")}, indent=2))


if __name__ == "__main__":
    main()

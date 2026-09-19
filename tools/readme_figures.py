"""Prepare hash-verified aggregate data and render the README's research figures.

Rendering uses only the committed CSVs. Preparing requires the recovered archives;
neither operation imports the policy, starts a simulator, or performs training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
SEEDS = (1011, 1022, 1033)
LABELS = {
    "last_token_attention": "Last-token", "time_attention": "Time",
    "index_attention": "Index", "gated_attention": "Gated",
    "supervised_attention": "Supervised*", "history_mlp": "History MLP",
    "history_gru": "History GRU",
}
COLORS = ("#2468a0", "#ce7d28", "#218878", "#bd4e50", "#8a6bb2", "#4b5966", "#b87298")
COMMANDS = {
    "stand_low": (0, 0, .28), "stand_mid": (0, 0, .30), "stand_high": (0, 0, .32),
    "forward": (.5, 0, .30), "reverse": (-.5, 0, .30),
    "turn_left": (0, 1, .30), "turn_right": (0, -1, .30),
}


class Archive:
    """Read only manifest members, checking bytes before parsing them."""

    def __init__(self, root, expected_sha):
        self.root = root
        raw = (root / "manifest.json").read_bytes()
        self.manifest_sha = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha != expected_sha:
            raise ValueError("unexpected recovery manifest: " + str(root))
        self.files = {item["path"]: item for item in json.loads(raw)["files"]}
        self.checked = {}

    def read_bytes(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("archive member escapes root")
        path = self.root / "extracted" / relative
        if not path.resolve().is_relative_to((self.root / "extracted").resolve()):
            raise ValueError("archive member resolves outside root")
        raw = path.read_bytes()
        expected = self.files[name]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != expected["size"] or digest != expected["sha256"]:
            raise ValueError("archive member mismatch: " + name)
        self.checked[name] = digest
        return raw

    def json(self, name):
        return json.loads(self.read_bytes(name))

    def metrics(self, name):
        return [json.loads(line) for line in self.read_bytes(name).splitlines() if line]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    old = Archive(args.round1, "6e001fdef4c1152aa15878ad48bbdc5e7583c3f31c8c696f05a2e34ad54281d8")
    new = Archive(args.wiring, "6cb289fce8e652a38dd650924c45b0d563185040662c77d0469c54bb01f0f900")
    plan = old.json("original/run/plan.json")
    lineage = {job["id"]: job for job in old.json("study/resume-plan.json")["jobs"]}
    training, evaluations = {}, []
    for job in plan["jobs"]:
        origin = "original/run/" + job["directory"]
        location = "study/" + job["directory"] if job["id"] in lineage else origin
        completion = old.json(location + "/train/completion.json")
        if completion["status"] != "completed" or completion["cumulative_update"] != 977:
            raise ValueError("incomplete round-one job")
        metrics = old.metrics(location + "/train/metrics.jsonl")
        if job["id"] in lineage and lineage[job["id"]]["start_update"]:
            metrics = old.metrics(origin + "/train/metrics.jsonl") + metrics
        if [row["update"] for row in metrics] != list(range(1, 978)):
            raise ValueError("missing or duplicate training updates")
        training[job["variant"], job["seed"]] = np.array([row["collection"]["reward_mean"] for row in metrics])
        for scenario in job["eval_configs"]:
            report = old.json(location + f"/evaluation_{scenario}_301.json")
            if (report["checkpoint_sha256"] != completion["checkpoints"][-1]["sha256"]
                    or report["checkpoint_update"] != 977 or report["transitions"] != 16000):
                raise ValueError("evaluation/checkpoint identity mismatch")
            signals = report["stability"]["signals"]
            vx, wz, height = COMMANDS[scenario]
            evaluations.append({
                "variant": job["variant"], "seed": job["seed"], "scenario": scenario,
                "command_vx_m_s": vx, "command_wz_rad_s": wz, "command_height_m": height,
                "actual_vx_m_s": vx + signals["vx_error"]["mean"],
                "actual_height_m": height + signals["height_error"]["mean"],
                "height_bias_mm": 1000 * signals["height_error"]["mean"],
                "vx_full_interval_mae_m_s": report["metrics"]["vx_abs_error"]["mean"],
            })
    if len(training) != 21 or len(evaluations) != 147:
        raise ValueError("expected 21 complete runs and 147 evaluations")
    curves = []
    for variant in LABELS:
        values = np.stack([training[variant, seed] for seed in SEEDS])
        smoothed = np.stack([values[:, max(0, i - 24):i + 1].mean(1) for i in range(977)], axis=1)
        for i in range(0, 977, 8):
            curves.append({"variant": variant, "update": i + 1, "seed_count": 3,
                           "reward_mean": smoothed[:, i].mean(), "reward_seed_sd": smoothed[:, i].std(ddof=1)})
    traces = []
    for variant in ("direct_mlp", "direct_transformer", "velocity_mlp"):
        base = f"study/wiring/plan/jobs/{variant}/seed_2003"
        report_name = base + "/evaluation_000080/stand_mid_3003.json"
        report = new.json(report_name)
        completion = new.json(base + "/train/completion.json")
        if (report["checkpoint_sha256"] != completion["checkpoints"][-1]["sha256"]
                or report["checkpoint_update"] != 80 or completion["collected_transitions"] != 1310720):
            raise ValueError("wiring evaluation/checkpoint identity mismatch")
        raw = new.read_bytes(report_name.replace(".json", ".npz"))
        if hashlib.sha256(raw).hexdigest() != report["trajectory"]["sha256"]:
            raise ValueError("trajectory identity mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as trace:
            if (trace["height"].shape != (1000, 8) or trace["terminated"].any()
                    or trace["truncated"].any()):
                raise ValueError("unexpected wiring episode boundaries")
            times = trace["signal_time"][:, 0] - trace["observation_time"][0, 0]
            for i in sorted(set(range(0, 1000, 5)) | {999}):
                traces.append({"variant": variant, "time_s": times[i],
                               "height_env_mean_m": trace["height"][i].mean(dtype=np.float64)})
    data = FIGURES / "data"
    data.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("training.csv", curves), ("evaluation.csv", evaluations), ("wiring.csv", traces)):
        write_csv(data / filename, rows)
    provenance = {
        "schema_version": 1,
        "scope": "Aggregate README data; original checkpoints and logs remain in ignored recovered archives.",
        "training": "Per seed: trailing mean of up to 25 updates; then 3-seed mean and sample SD (ddof=1); every 8 updates, including 1 and 977.",
        "evaluation": "Final checkpoint 977, evaluation seed 301; steady means after 200 settling steps per reset; vx MAE column uses full interval.",
        "wiring": "Seed 2003, checkpoint 80, evaluation seed 3003; 8-env height mean sampled every 5 steps and final step; 10s censored segments.",
        "resumed": {"supervised_attention": {"1011": 789, "1022": 732}},
        "manifest_sha256": {"round1": old.manifest_sha, "wiring": new.manifest_sha},
        "verified_inputs": {"round1": old.checked, "wiring": new.checked},
        "csv_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(data.glob("*.csv"))},
    }
    (data / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(curves)} curve points, {len(evaluations)} evaluations, {len(traces)} trace points.")


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = font_manager.findfont("Noto Sans CJK SC", fallback_to_default=False)
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({
        "font.family": "Noto Sans CJK SC", "font.size": 12, "axes.titlesize": 15,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9c6cd",
        "text.color": "#263746", "axes.labelcolor": "#263746", "xtick.color": "#536775",
        "ytick.color": "#536775", "axes.unicode_minus": False, "savefig.facecolor": "white",
    })
    data = FIGURES / "data"
    provenance = json.loads((data / "provenance.json").read_text())
    tables = {}
    for name, digest in provenance["csv_sha256"].items():
        raw = (data / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("published CSV changed: " + name)
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode())))

    def values(table, variant, field, scenario=None):
        return np.array([float(row[field]) for row in tables[table]
                         if row["variant"] == variant and (scenario is None or row.get("scenario") == scenario)])

    def finish(fig, axes, title, subtitle, footer, filename):
        fig.suptitle(title, x=.07, y=.97, ha="left", fontsize=23, fontweight="bold")
        fig.text(.07, .865, subtitle, fontsize=12, color="#536775")
        fig.text(.07, .03, footer, fontsize=10, color="#536775", linespacing=1.6)
        for ax in np.asarray(axes).reshape(-1):
            ax.grid(color="#e7edf0", linewidth=.8)
            ax.set_axisbelow(True)
        fig.savefig(FIGURES / filename, dpi=160, metadata={"Software": "tools/readme_figures.py"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.7), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.08, right=.97, top=.66, bottom=.22, wspace=.43)
    handles = []
    for i, (variant, label) in enumerate(LABELS.items()):
        x = values("training.csv", variant, "update")
        mean = values("training.csv", variant, "reward_mean")
        sd = values("training.csv", variant, "reward_seed_sd")
        line, = axes[0].plot(x, mean, color=COLORS[i], label=label, linewidth=1.9)
        handles.append(line)
        axes[0].fill_between(x, mean - sd, mean + sd, color=COLORS[i], alpha=.09)
        bias = values("evaluation.csv", variant, "height_bias_mm", "stand_mid")
        axes[1].errorbar(bias.mean(), i, xerr=bias.std(ddof=1), fmt="o", color=COLORS[i], capsize=3)
        axes[1].scatter(bias, np.full(3, i), color=COLORS[i], alpha=.45, s=22)
    axes[0].set(title="训练 reward", xlabel="累计 PPO 更新数", ylabel="Reward / transition", xlim=(0, 1000))
    axes[1].set(title="独立站立评估：目标 0.30 m", xlabel="实际高度 − 目标高度（mm）",
                yticks=range(7), yticklabels=list(LABELS.values()))
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#263746", linestyle="--", linewidth=1.3)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .81), ncol=4, frameon=False, fontsize=11)
    finish(fig, axes, "训练曲线上升，不等于控制任务完成", "首轮 7 种网络 × 3 个训练 seed · 每项 977 updates / 16,007,168 条完整 rollout 样本",
           "左：先按 seed 做 25-update 后向均值，再取跨 seed 均值 ± 样本标准差；每 8 updates 展示一个点。\n右：小点为单 seed，大点与误差线为均值 ± 样本标准差。* 两项 Supervised 分段续训，环境与历史重置。",
           "training-and-height.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6))
    fig.subplots_adjust(left=.08, right=.97, top=.66, bottom=.23, wspace=.26)
    for i, (variant, label) in enumerate(LABELS.items()):
        for ax, scenarios, command, actual in (
            (axes[0], ("reverse", "stand_mid", "forward"), [-.5, 0, .5], "actual_vx_m_s"),
            (axes[1], ("stand_low", "stand_mid", "stand_high"), [.28, .30, .32], "actual_height_m"),
        ):
            ys = [values("evaluation.csv", variant, actual, scenario).mean() for scenario in scenarios]
            ax.plot(command, ys, "o-", color=COLORS[i], linewidth=1.8, markersize=5, label=label)
    for ax, bounds in ((axes[0], [-.5, .5]), (axes[1], [.28, .32])):
        ax.plot(bounds, bounds, "--", color="#263746", label="理想跟踪", linewidth=1.6)
    axes[0].set(title="前进 / 后退响应", xlabel="指令 vx（m/s）", ylabel="实际稳态 vx（m/s）", ylim=(-.56, .56))
    axes[1].set(title="高度响应", xlabel="指令高度（m）", ylabel="实际稳态高度（m）", ylim=(.18, .335), xticks=[.28, .30, .32])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.06, .81), ncol=4, frameon=False, fontsize=11)
    finish(fig, axes, "不同指令下，实际速度与高度响应有限", "最终模型独立评估 · 每个指令对应独立固定场景 · 三个训练 seed 的等权均值",
           "每次 reset 后去掉前 200 步；100 Hz 策略。此图展示均值，不展示跨 seed 离散程度。\n同任务、同预算下各网络均未解决这些跟踪场景，不能据此选出合格赢家；不包含估计器长训或其他项目的 MLP 结果。",
           "command-response.png")

    fig, ax = plt.subplots(figsize=(13.6, 5.5))
    fig.subplots_adjust(left=.08, right=.97, top=.68, bottom=.24)
    for variant, label, color in (("direct_mlp", "Direct MLP", COLORS[5]),
                                  ("direct_transformer", "Direct Transformer", COLORS[0]),
                                  ("velocity_mlp", "Velocity MLP", COLORS[2])):
        ax.plot(values("wiring.csv", variant, "time_s"), values("wiring.csv", variant, "height_env_mean_m"),
                color=color, label=label, linewidth=2)
    ax.axhline(.30, color="#263746", linestyle="--", linewidth=1.5, label="目标 0.30 m")
    ax.axvspan(0, 2, color="#e7edf0", zorder=0)
    height_max = max(float(row["height_env_mean_m"]) for row in tables["wiring.csv"])
    ax.set(xlabel="重置后的时间（s）", ylabel="实测高度（m）", xlim=(0, 10),
           ylim=(.20, max(.31, height_max) + .005))
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.06, .79), ncol=4, frameon=False, fontsize=11)
    finish(fig, [ax], "估计器接线短训：有响应，但还没跟住高度", "Seed 2003 · 每项 80 updates / 1,310,720 samples · 仅展示三个已完成评估的网络",
           "曲线为 8 个评估环境的高度均值，每 5 步展示一个点；灰区为前 2 秒。单 seed，10 秒片段，未覆盖完整 20 秒 episode。\nVelocity Transformer 评估中断，两个 Context 网络未启动；这组图用于接线诊断，不能推断长训效果或架构优劣。",
           "wiring-height.png")
    print("Rendered three README figures from hash-verified CSVs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("prepare")
    source.add_argument("--round1", type=Path, required=True)
    source.add_argument("--wiring", type=Path, required=True)
    commands.add_parser("render")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else render()


if __name__ == "__main__":
    main()

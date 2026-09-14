"""CPU-only audit of single-env, fixed-command pre-reset reference CSVs.

Read CSV bytes once for both SHA256 and parsing. Emit deterministic JSON to
stdout; never import a model, simulator, array framework, or write source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys


DT = 0.01
SIM_TIME_TOL = 1e-9
EPISODE_TIME_TOL = 2e-6  # The source stores episode time in FP32.
HEIGHT_TOL = 1e-7
SIGNED_COLUMNS = [
    "height_signed_bias_mm", "height_demeaned_std_mm",
    "vx_signed_mean_m_s", "vx_demeaned_std_m_s",
    "wz_signed_mean_rad_s", "wz_demeaned_std_rad_s",
]
DISPLACEMENT_COLUMNS = ["dx_m", "dy_m", "net_m", "path_m", "max_from_start_m"]
BOOL_COLUMNS = {
    "terminated", "timeout", "exploratory", "action_command_ood",
    "next_command_ood",
}
COMMAND_COLUMNS = {
    f"{prefix}_{axis}"
    for prefix in ("action_cmd", "reward_cmd", "next_cmd", "requested")
    for axis in ("vx", "wz", "height")
}
REQUIRED = COMMAND_COLUMNS | {
    "step", "sim_s", "policy_tick", "episode_step", "episode_time_s",
    "sample_kind", "x", "y", "z", "vx", "vy", "wz", "terminated", "timeout",
    "non_wheel_net_force_max_n", "diagnostic_non_wheel_contact",
    "non_wheel_contact_n",
}


def parse_trace(data: bytes, height: float) -> list[list[dict]]:
    """Reject malformed rows; never drop, interpolate or join across a reset."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    header = reader.fieldnames or []
    if len(header) != len(set(header)) or not REQUIRED.issubset(header):
        raise ValueError("duplicate or missing required CSV columns")
    episodes = []
    previous = None
    for index, raw in enumerate(reader, 1):
        def require(condition, message):
            if not condition:
                raise ValueError(f"CSV line {reader.line_num}: {message}")

        # A telemetry row occupies exactly one physical line. Detect blank rows
        # that DictReader would otherwise silently skip.
        require(reader.line_num == index + 1, "blank or multiline row")
        require(None not in raw and None not in raw.values(), "row width mismatch")
        row = {}
        for name, value in raw.items():
            if name == "sample_kind":
                require(value == "pre_reset", "sample must be pre_reset")
                row[name] = value
            elif name in BOOL_COLUMNS or name.startswith(("diagnostic_", "termination_")):
                require(value in ("True", "False"), f"invalid boolean {name}")
                row[name] = value == "True"
            else:
                try:
                    row[name] = float(value)
                except ValueError as exc:
                    raise ValueError(f"CSV line {reader.line_num}: invalid {name}") from exc
                require(math.isfinite(row[name]), f"nonfinite {name}")
        require(row["step"] == row["policy_tick"] == index, "missing/duplicate policy tick")
        require(abs(row["sim_s"] - index * DT) <= SIM_TIME_TOL, "invalid sim time")
        if previous is not None:
            require(abs(row["sim_s"] - previous["sim_s"] - DT) <= SIM_TIME_TOL,
                    "nonuniform sim time")
        reset = previous is None or previous["terminated"] or previous["timeout"]
        expected_step = 1 if reset else previous["episode_step"] + 1
        require(row["episode_step"] == expected_step, "episode step/reset mismatch")
        require(abs(row["episode_time_s"] - expected_step * DT) <= EPISODE_TIME_TOL,
                "episode time inconsistent with tick")
        if not reset:
            require(abs(row["episode_time_s"] - previous["episode_time_s"] - DT)
                    <= EPISODE_TIME_TOL, "nonuniform episode time")
        for name in COMMAND_COLUMNS:
            expected = height if name.endswith("height") else 0.0
            tolerance = HEIGHT_TOL if name.endswith("height") else 0.0
            require(abs(row[name] - expected) <= tolerance, f"nonconstant command {name}")
        force = row["non_wheel_net_force_max_n"]
        require(force == row["non_wheel_contact_n"], "contact aliases disagree")
        require(row["diagnostic_non_wheel_contact"] == (force > 1.0),
                "contact diagnostic inconsistent with threshold")
        if reset:
            episodes.append([])
        episodes[-1].append(row)
        previous = row
    if reader.line_num != sum(map(len, episodes)) + 1:
        raise ValueError("trailing blank CSV row")
    if not episodes:
        raise ValueError("empty trace")
    return episodes


def moments(values):
    """Two-pass population moments using compensated summation of FP64 values."""
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


def pool_moments(groups):
    """Return weighted mean, within variance and between variance separately."""
    count = sum(n for n, _, _ in groups)
    mean = math.fsum(n * mu for n, mu, _ in groups) / count
    within = math.fsum(n * variance for n, _, variance in groups) / count
    between = math.fsum(n * (mu - mean) ** 2 for n, mu, _ in groups) / count
    return mean, within, between


def displacement(rows):
    first = rows[0]
    dx = rows[-1]["x"] - first["x"]
    dy = rows[-1]["y"] - first["y"]
    path = math.fsum(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                     for a, b in zip(rows, rows[1:]))
    maximum = max(math.hypot(r["x"] - first["x"], r["y"] - first["y"])
                  for r in rows)
    return [dx, dy, math.hypot(dx, dy), path, maximum]


def analyze_trace(data, height, expected_sha256, expected_samples=6000):
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError("CSV SHA256 differs from historical evidence")
    episodes = parse_trace(data, height)
    rows = [row for episode in episodes for row in episode]
    if len(rows) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, got {len(rows)}")
    records = []
    groups = [[], [], []]
    retained = []
    for episode_id, episode in enumerate(episodes, 1):
        steady = [r for r in episode if r["episode_time_s"] > 2.0]
        retained.extend(steady)
        record = {
            "episode_id": episode_id,
            "all_steps": [int(episode[0]["step"]), int(episode[-1]["step"])],
            "all_samples": len(episode), "steady_samples": len(steady),
            "terminated": episode[-1]["terminated"], "timeout": episode[-1]["timeout"],
            "all_xy": displacement(episode),
            "steady_steps": None, "steady_sim_s": None, "steady_episode_s": None,
            "signed": None, "steady_xy": None,
        }
        if steady:
            signals = [[(r["z"] - height) * 1000 for r in steady],
                       [r["vx"] for r in steady], [r["wz"] for r in steady]]
            signed = []
            for group, signal in zip(groups, signals):
                mean, variance = moments(signal)
                group.append((len(steady), mean, variance))
                signed.extend([mean, math.sqrt(variance)])
            record.update(
                steady_steps=[int(steady[0]["step"]), int(steady[-1]["step"])],
                steady_sim_s=[steady[0]["sim_s"], steady[-1]["sim_s"]],
                steady_episode_s=[steady[0]["episode_time_s"], steady[-1]["episode_time_s"]],
                signed=signed, steady_xy=displacement(steady),
            )
        records.append(record)
    pooled = None
    if retained:
        pooled = {"signed": [], "within_variances": [], "between_variances": []}
        for group in groups:
            mean, within, between = pool_moments(group)
            pooled["signed"].extend([mean, math.sqrt(within)])
            pooled["within_variances"].append(within)
            pooled["between_variances"].append(between)
    sim_deltas = [b["sim_s"] - a["sim_s"] for a, b in zip(rows, rows[1:])]
    episode_deltas = [b["episode_time_s"] - a["episode_time_s"]
                      for ep in episodes for a, b in zip(ep, ep[1:])]
    return {
        "height_m": height, "csv_sha256": digest,
        "csv_bytes": len(data), "sha256_matches_historical": True,
        "samples": len(rows), "steady_samples": len(retained),
        "excluded_initial_samples": len(rows) - len(retained),
        "anomalous_rows": 0, "dropped_rows": 0,
        "sim_dt_min_max_s": [min(sim_deltas), max(sim_deltas)],
        "episode_dt_min_max_s": [min(episode_deltas), max(episode_deltas)],
        "sim_dt_max_abs_error_s": max(abs(delta - DT) for delta in sim_deltas),
        "episode_dt_max_abs_error_s": max(abs(delta - DT) for delta in episode_deltas),
        "episode_time_max_abs_error_s": max(abs(r["episode_time_s"] - r["episode_step"] * DT)
                                            for r in rows),
        "non_wheel_contact_frames_all_steady": [
            sum(r["diagnostic_non_wheel_contact"] for r in rows),
            sum(r["diagnostic_non_wheel_contact"] for r in retained)],
        "episodes": records, "pooled_within_episode": pooled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-evidence", type=Path, required=True,
                        help="Historical JSON containing heights and csv_sha256")
    args = parser.parse_args()
    reference = json.loads(args.reference_evidence.read_text())
    results = []
    for source in reference["heights"]:
        height = source["height_m"]
        path = args.root / f"h{round(height * 100):03d}" / "telemetry.csv"
        result = analyze_trace(path.read_bytes(), height, source["csv_sha256"])
        results.append(result)
    output = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "signed_columns": SIGNED_COLUMNS,
        "displacement_columns": DISPLACEMENT_COLUMNS,
        "variance_units_order": ["mm^2", "(m/s)^2", "(rad/s)^2"],
        "height_results": results,
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

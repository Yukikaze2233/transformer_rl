"""Control statistics conditioned on the history used for each policy action."""
from __future__ import annotations

import math

import torch

from .control_metrics import _ErrorPool, _Errors, _PlanarMotion, _Scalar


class HistoryControlStatistics:
    """Keep reset-filled and fully observed windows separate, with bounded memory.

    Samples are PRE-reset physical responses to the action whose input age is
    provided explicitly. Finite differences require both endpoints in the same
    window, episode and unchanged-reference segment. Empty windows stay null.
    """

    def __init__(self, num_envs, history_length, policy_dt_s, *, settle_steps=200,
                 min_steady_samples=200):
        for name, value, minimum in (("num_envs", num_envs, 1),
                                     ("history_length", history_length, 1),
                                     ("settle_steps", settle_steps, 0),
                                     ("min_steady_samples", min_steady_samples, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(policy_dt_s) not in (int, float) or not math.isfinite(policy_dt_s) or policy_dt_s <= 0:
            raise ValueError("positive finite policy_dt_s required")
        self.num_envs, self.history_length = num_envs, history_length
        self.dt = float(policy_dt_s)
        self.settle_steps, self.min_steady_samples = settle_steps, min_steady_samples
        self._expected_age = torch.zeros(num_envs, dtype=torch.int64)
        self._rows = [self._row() for _ in range(num_envs)]
        self._windows = {name: self._window() for name in ("reset_filled", "full_history")}
        self._report = None

    @staticmethod
    def _row():
        return {"window": None, "errors": _Errors(), "post_settle": _Errors(), "steady": _Errors(),
                "previous": None, "reference_age": 0}

    @staticmethod
    def _window():
        widths = {"leg_target_rate": 4, "wheel_target_acceleration": 2,
                  "physical_effort_rate": 6, "raw_policy_mean_rate": 6,
                  "issued_action_rate": 6, "physical_effort": 6}
        return {"full": _ErrorPool(), "post_settle": _ErrorPool(), "steady": _ErrorPool(),
                "rates": {key: [_Scalar() for _ in range(width)] for key, width in widths.items()},
                "planar": _PlanarMotion(), "stationary": _PlanarMotion(),
                "tilt": _Scalar(), "mechanical_power_proxy": _Scalar(),
                "actor_clipped": [0] * 6, "samples": 0,
                "terminal": {"done": 0, "failure": 0, "success": 0},
                "discarded_short_steady_samples": 0, "discarded_short_post_settle_samples": 0}

    def _finish_steady(self, row):
        steady = row["steady"]
        if row["window"] is not None and steady.count:
            window = self._windows[row["window"]]
            if steady.count >= self.min_steady_samples:
                window["steady"].merge(steady)
            else:
                window["discarded_short_steady_samples"] += steady.count
        row["steady"] = _Errors()

    def _finish_window(self, row):
        self._finish_steady(row)
        if row["window"] is not None:
            window = self._windows[row["window"]]
            window["full"].merge(row["errors"])
            if row["post_settle"].count >= self.min_steady_samples:
                window["post_settle"].merge(row["post_settle"])
            else:
                window["discarded_short_post_settle_samples"] += row["post_settle"].count
        row["errors"] = _Errors()
        row["post_settle"] = _Errors()

    def update(self, packet, done, ages, raw_mean, issued_action):
        if self._report is not None:
            raise RuntimeError("cannot update after report")
        if (not isinstance(ages, torch.Tensor) or ages.dtype != torch.int64
                or ages.shape != (self.num_envs,)
                or not isinstance(done, torch.Tensor) or done.dtype != torch.bool
                or done.shape != (self.num_envs,)):
            raise ValueError("int64 pre-inference ages and bool done vectors required")
        age = ages.detach().cpu()
        if not torch.equal(age, self._expected_age):
            raise ValueError("pre-inference age must begin at zero and reset after done")
        shapes = {"time_s": (), "command_reference": (3,), "actual": (3,),
                  "position_xy": (2,), "tilt": (), "leg_target": (4,),
                  "wheel_target": (2,), "motor_effort": (6,), "motor_velocity": (6,),
                  "failure": (), "success": ()}
        values = {}
        for name, shape in shapes.items():
            value = packet.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, *shape):
                raise ValueError(f"invalid physical field: {name}")
            if name in ("failure", "success"):
                if value.dtype != torch.bool:
                    raise ValueError("terminal fields must be bool")
            elif not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("physical history samples must be finite floats")
            values[name] = value.detach().double().cpu().tolist()
        for name, value in (("raw_mean", raw_mean), ("issued_action", issued_action)):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, 6)
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError("six finite raw mean and issued action values required")
            values[name] = value.detach().double().cpu().tolist()
        endings = done.detach().cpu().tolist()
        for index, row in enumerate(self._rows):
            data = {name: value[index] for name, value in values.items()}
            window_name = "full_history" if int(age[index]) >= self.history_length - 1 else "reset_filled"
            if not math.isclose(data["time_s"], (int(age[index]) + 1) * self.dt,
                                rel_tol=0., abs_tol=5e-6):
                raise ValueError("physical clock disagrees with input age")
            if ((data["failure"] or data["success"]) and not endings[index]
                    or data["failure"] and data["success"]):
                raise ValueError("terminal history flags disagree")
            previous = row["previous"]
            changed = previous is not None and any(abs(a - b) > 1e-5 for a, b in
                zip(data["command_reference"], previous["command_reference"]))
            if changed:
                self._finish_steady(row)
                row["reference_age"] = 0
            elif previous is not None:
                row["reference_age"] += 1
            if row["window"] != window_name:
                self._finish_window(row)
                row["window"] = window_name
            window = self._windows[window_name]
            error = [a - b for a, b in zip(data["actual"], data["command_reference"])]
            row["errors"].add(error, (.15, .25, .03))
            # Dynamic references remain part of post-reset tracking. Only the
            # constant-reference statistic restarts its settling clock.
            if int(age[index]) >= self.settle_steps:
                row["post_settle"].add(error, (.15, .25, .03))
            steady = int(age[index]) >= self.settle_steps and row["reference_age"] >= self.settle_steps
            if steady:
                row["steady"].add(error, (.15, .25, .03))
            window["samples"] += 1
            window["tilt"].add(data["tilt"])
            window["mechanical_power_proxy"].add(sum(abs(a * b) for a, b in
                zip(data["motor_effort"], data["motor_velocity"])))
            for joint, (raw, issued, effort) in enumerate(zip(data["raw_mean"], data["issued_action"], data["motor_effort"])):
                window["actor_clipped"][joint] += raw != issued
                window["rates"]["physical_effort"][joint].add(effort)
            # The incoming interval is excluded at a reset, command or history
            # category edge. Physical targets and issued actions stay distinct.
            if previous is not None and previous["window"] == window_name and not changed:
                dt = data["time_s"] - previous["time_s"]
                if not math.isclose(dt, self.dt, rel_tol=0., abs_tol=5e-6):
                    raise ValueError("noncontiguous history control interval")
                window["planar"].add(previous["position_xy"], data["position_xy"], dt)
                if all(abs(v) <= 1e-5 for v in data["command_reference"][:2]):
                    window["stationary"].add(previous["position_xy"], data["position_xy"], dt)
                for metric, field in (("leg_target_rate", "leg_target"),
                                      ("wheel_target_acceleration", "wheel_target"),
                                      ("physical_effort_rate", "motor_effort"),
                                      ("raw_policy_mean_rate", "raw_mean"),
                                      ("issued_action_rate", "issued_action")):
                    for signal, now, before in zip(window["rates"][metric], data[field], previous[field]):
                        signal.add((now - before) / dt)
            data["window"] = window_name
            row["previous"] = data
            if endings[index]:
                for name in ("done", "failure", "success"):
                    window["terminal"][name] += bool(endings[index] if name == "done" else data[name])
                self._finish_window(row)
                self._rows[index] = self._row()
        self._expected_age = age + 1
        self._expected_age[done.detach().cpu()] = 0

    def report(self):
        if self._report is None:
            for row in self._rows:
                self._finish_window(row)
            windows = {}
            for name, window in self._windows.items():
                count = window["samples"]
                windows[name] = {"samples": count,
                    "tracking": window["full"].report("within each environment/episode/history-window"),
                    "post_settle_tracking": window["post_settle"].report("within each environment/episode/history-window after reset settling; changing references included"),
                    "steady_tracking": window["steady"].report("within each environment/episode/constant-reference segment"),
                    "discarded_short_steady_samples": window["discarded_short_steady_samples"],
                    "discarded_short_post_settle_samples": window["discarded_short_post_settle_samples"],
                    "physical_planar_motion": window["planar"].report(),
                    "stationary_physical_planar_motion": window["stationary"].report(),
                    "rates": {key: [s.report() for s in signals] for key, signals in window["rates"].items()},
                    "tilt": window["tilt"].report(),
                    "mechanical_power_proxy": window["mechanical_power_proxy"].report(),
                    "actor_clipping_fraction": [v / count if count else None for v in window["actor_clipped"]],
                    "terminal_events": window["terminal"]}
            self._report = {"format": "transformer_rl.history_control_statistics", "schema_version": 2,
                "history_length": self.history_length, "minimum_full_age": self.history_length - 1,
                "age_semantics": "policy steps since reset, captured before actor inference",
                "full_history_rule": "pre_inference_episode_age >= history_length - 1",
                "policy_dt_s": self.dt, "settle_steps": self.settle_steps,
                "post_settle_rule": "pre_inference_episode_age >= settle_steps; minimum samples per environment/episode/history-window; command changes do not restart settling",
                "min_steady_samples": self.min_steady_samples, "windows": windows,
                "interval_rule": "both endpoints in the same episode, history window and unchanged-reference segment",
                "actuation_scope": "physical packet targets and effort are distinct from actor raw mean and issued action",
                "transport_arrival_and_high_frequency_current_verified": False}
        return self._report

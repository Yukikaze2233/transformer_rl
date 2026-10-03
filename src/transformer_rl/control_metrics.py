"""Bounded, policy-rate PRE-reset motion-control statistics.

No statistic here is an estimate of high-rate current-loop behavior. Commands
are the references effective at the physical sample, not transport requests.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import torch


AXES = ("vx", "wz", "height")


@dataclass
class _Scalar:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    absolute: float = 0.0
    peak: float = 0.0
    in_band: int = 0

    def add(self, value, tolerance=math.inf):
        self.count += 1
        difference = value - self.mean
        self.mean += difference / self.count
        self.m2 += difference * (value - self.mean)
        self.absolute += abs(value)
        self.peak = max(self.peak, abs(value))
        self.in_band += abs(value) <= tolerance

    def merge(self, other):
        if not other.count:
            return
        total = self.count + other.count
        difference = other.mean - self.mean
        self.m2 += other.m2 + difference * difference * self.count * other.count / total
        self.mean += difference * other.count / total
        self.count = total
        self.absolute += other.absolute
        self.peak = max(self.peak, other.peak)
        self.in_band += other.in_band

    def report(self):
        return {"count": self.count, "mean": self.mean if self.count else None,
                "rms": math.sqrt(max(0., self.m2 / self.count + self.mean**2)) if self.count else None,
                "mean_abs": self.absolute / self.count if self.count else None,
                "max_abs": self.peak if self.count else None}


@dataclass
class _Errors:
    axes: list[_Scalar] = field(default_factory=lambda: [_Scalar() for _ in AXES])
    all_in_band: int = 0
    count: int = 0

    def add(self, error, tolerances):
        self.count += 1
        self.all_in_band += all(abs(value) <= tolerance for value, tolerance in zip(error, tolerances))
        for signal, value, tolerance in zip(self.axes, error, tolerances):
            signal.add(value, tolerance)


@dataclass
class _ErrorPool:
    axes: list[_Scalar] = field(default_factory=lambda: [_Scalar() for _ in AXES])
    within_m2: list[float] = field(default_factory=lambda: [0.] * 3)
    means: list[_Scalar] = field(default_factory=lambda: [_Scalar() for _ in AXES])
    count: int = 0
    all_in_band: int = 0
    groups: int = 0

    def merge(self, errors):
        if not errors.count:
            return
        self.count += errors.count
        self.all_in_band += errors.all_in_band
        self.groups += 1
        for index, signal in enumerate(errors.axes):
            self.axes[index].merge(signal)
            self.within_m2[index] += signal.m2
            self.means[index].add(signal.mean)

    def report(self, centering):
        result = {}
        for index, name in enumerate(AXES):
            signal, means = self.axes[index], self.means[index]
            result[name] = {"bias": signal.mean if signal.count else None,
                "mae": signal.absolute / signal.count if signal.count else None,
                "rmse": signal.report()["rms"], "peak_abs_error": signal.peak if signal.count else None,
                "in_band_fraction": signal.in_band / signal.count if signal.count else None,
                "within_group_std": math.sqrt(max(0., self.within_m2[index]) / self.count) if self.count else None,
                "group_mean_std": math.sqrt(max(0., means.m2) / means.count) if means.count else None,
                "count": signal.count}
        return {"axes": result, "samples": self.count, "groups": self.groups, "centering": centering,
                "all_axes_in_band_fraction": self.all_in_band / self.count if self.count else None}


@dataclass
class _Response:
    start: float
    previous_reference: float
    target: float
    tolerance: float
    hold_s: float
    initial_tracking_in_band: bool
    previous_time: float | None = None
    previous_error: float | None = None
    iae: float = 0.
    t10: float | None = None
    t90: float | None = None
    band_since: float | None = None
    settling: float | None = None
    overshoot: float = 0.
    samples: int = 0

    def add(self, timestamp, actual):
        error = actual - self.target
        elapsed = timestamp - self.start
        if self.previous_time is not None:
            self.iae += .5 * (abs(error) + abs(self.previous_error)) * (timestamp - self.previous_time)
        amplitude = self.target - self.previous_reference
        progress = (actual - self.previous_reference) / amplitude
        if self.t10 is None and progress + 1e-12 >= .1:
            self.t10 = elapsed
        if self.t90 is None and progress + 1e-12 >= .9:
            self.t90 = elapsed
        self.overshoot = max(self.overshoot, math.copysign(1., amplitude) * error)
        if abs(error) <= self.tolerance:
            if self.band_since is None:
                self.band_since = timestamp
            if timestamp - self.band_since + 1e-12 >= self.hold_s:
                self.settling = self.band_since - self.start
        else:
            self.band_since = self.settling = None
        self.previous_time, self.previous_error = timestamp, error
        self.samples += 1

    def finish(self, reason, failure):
        duration = self.previous_time - self.start if self.previous_time is not None else 0.
        held_long_enough = duration + 1e-12 >= self.hold_s
        eligible = held_long_enough and self.initial_tracking_in_band
        complete = reason != "partial" and not failure and eligible
        rise = self.t90 - self.t10 if self.t10 is not None and self.t90 is not None and not failure and eligible else None
        settling = self.settling if not failure and eligible else None
        return {"end_reason": reason, "failure": failure, "samples": self.samples,
                "step_eligible": eligible,
                "held_long_enough": held_long_enough, "initial_tracking_in_band": self.initial_tracking_in_band,
                "observed_duration_s": duration, "previous_reference": self.previous_reference, "target": self.target,
                "observed_iae": self.iae, "complete_iae": self.iae if complete else None,
                "rise_10_90_s": rise, "rise_censored": rise is None,
                "absolute_overshoot": self.overshoot if not failure and eligible else None,
                "settling_time_s": settling, "settling_censored": settling is None,
                "partial": reason == "partial"}


@dataclass
class _Segment:
    reference: list[float]
    full: _Errors = field(default_factory=_Errors)
    steady: _Errors = field(default_factory=_Errors)
    responses: list[_Response | None] = field(default_factory=lambda: [None] * 3)


@dataclass
class _Episode:
    errors: _Errors = field(default_factory=_Errors)
    segment: _Segment | None = None
    previous_time: float | None = None
    previous_leg: list[float] | None = None
    previous_wheel: list[float] | None = None
    previous_effort: list[float] | None = None
    origin: list[float] | None = None
    previous_error: list[float] | None = None


class ControlMetrics:
    """Consume one vector of PRE-reset physical samples per policy step.

    Memory is O(num_envs + recent_response_limit), independent of elapsed time.
    Full-interval errors retain all samples, including transients and failures.
    A steady segment starts only after an unchanged reference has lasted
    settle_steps samples; continuous reference slews normally yield short,
    ineligible segments. Episode survival flags never imply tracking success.
    """

    def __init__(self, num_envs, policy_dt_s, *, settle_steps=200, min_steady_samples=200,
                 tracking_tolerance=(.15, .25, .03), command_tolerance=(1e-5, 1e-5, 1e-5),
                 response_min_step=(.1, .2, .02), settling_hold_s=.5,
                 saturation_tolerance=1e-6, recent_response_limit=32):
        for name, value, minimum in (("num_envs", num_envs, 1), ("settle_steps", settle_steps, 0),
                                    ("min_steady_samples", min_steady_samples, 1),
                                    ("recent_response_limit", recent_response_limit, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value, positive in (("policy_dt_s", policy_dt_s, True), ("settling_hold_s", settling_hold_s, True),
                                      ("saturation_tolerance", saturation_tolerance, False)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        for name, values in (("tracking_tolerance", tracking_tolerance), ("command_tolerance", command_tolerance),
                             ("response_min_step", response_min_step)):
            if not isinstance(values, (tuple, list)) or len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"{name} must contain three finite nonnegative numbers")
        self.num_envs, self.policy_dt_s = num_envs, float(policy_dt_s)
        self.settle_steps, self.min_steady_samples = settle_steps, min_steady_samples
        self.tolerances, self.command_tolerances = tuple(tracking_tolerance), tuple(command_tolerance)
        self.response_min_step = tuple(response_min_step)
        self.hold_s, self.saturation_tolerance = float(settling_hold_s), float(saturation_tolerance)
        self._episodes = [_Episode() for _ in range(num_envs)]
        self._full, self._steady = _ErrorPool(), _ErrorPool()
        self._episode_counts = {"completed": 0, "failed": 0, "success_flags": 0, "partial": 0,
                                "short": 0, "completed_all_samples_tracking_in_band": 0,
                                "partial_all_samples_tracking_in_band": 0}
        self._segment_counts = {"total": 0, "eligible": 0, "short": 0, "partial": 0, "failed": 0,
                                "command_changes": 0, "discarded_settle_samples": 0, "discarded_short_samples": 0}
        self._response_counts = {name: {"events": 0, "eligible_steps": 0, "short_hold_candidates": 0,
                                        "nonsteady_start": 0, "small_reference_changes": 0,
                                        "failed": 0, "partial": 0, "rise_censored": 0,
                                        "settling_censored": 0} for name in AXES}
        self._response_values = {name: {key: _Scalar() for key in ("observed_iae", "complete_iae", "rise_10_90_s",
                                                                  "absolute_overshoot", "settling_time_s")} for name in AXES}
        self._recent = deque(maxlen=recent_response_limit)
        self._actuation = {name: [_Scalar() for _ in range(width)] for name, width in
                           (("leg_target_rate", 4), ("wheel_target_acceleration", 2), ("effort_rate", 6),
                            ("actual_effort", 6), ("leg_position_error", 4), ("wheel_velocity_error", 2))}
        self._saturated = [0] * 6
        self._clipped = [0] * 6
        self._bounds_samples = [0] * 6
        self._actuation_samples = 0
        self._power, self._tilt, self._drift = _Scalar(), _Scalar(), _Scalar()
        self._full_iae = [0.] * 3
        self._band_time = [0.] * 3
        self._joint_band_time = self._duration = 0.
        self._request_error = [_Scalar() for _ in AXES]
        self._request_available = None
        self._finished = False
        self._cached = None

    def _validate(self, packet, done):
        if not isinstance(packet, dict) or not isinstance(done, torch.Tensor) or done.shape != (self.num_envs,) or done.dtype != torch.bool:
            raise ValueError("packet must be a dict and done must be a bool tensor [num_envs]")
        shapes = {"time_s": (), "command_reference": (3,), "actual": (3,), "position_xy": (2,), "tilt": (),
                  "leg_target": (4,), "wheel_target": (2,), "motor_position": (6,), "motor_velocity": (6,),
                  "motor_effort": (6,), "requested_motor_effort": (6,), "effort_bounds": (6, 2),
                  "failure": (), "success": ()}
        available = "request" in packet
        if self._request_available is not None and self._request_available != available:
            raise ValueError("request availability must remain constant")
        if available:
            shapes["request"] = (3,)
        blocks, offsets, width = [], {}, 0
        for name, shape in shapes.items():
            value = packet.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, *shape) or value.device != done.device:
                raise ValueError(f"{name} requires shape {(self.num_envs, *shape)} on done device")
            if name in ("failure", "success"):
                if value.dtype != torch.bool:
                    raise ValueError(f"{name} must be bool")
            elif not value.is_floating_point() or (name == "time_s" and value.dtype != torch.float64):
                raise ValueError(f"{name} must be floating point; time_s must be float64")
            flat = value.detach().double().reshape(self.num_envs, -1)
            offsets[name] = (width, width + flat.shape[1])
            width += flat.shape[1]
            blocks.append(flat)
        packed = torch.cat(blocks + [done.double().reshape(self.num_envs, 1)], dim=1).cpu()
        if not torch.isfinite(packed).all():
            raise ValueError("control packet must be finite")
        rows = []
        for row, episode in zip(packed.tolist(), self._episodes):
            data = {name: row[start:end] for name, (start, end) in offsets.items()}
            data["done"] = bool(row[-1])
            timestamp = data["time_s"][0]
            if timestamp < 0 or episode.previous_time is not None and timestamp <= episode.previous_time:
                raise ValueError("time_s must be nonnegative and strictly increasing within an episode")
            if any(data["effort_bounds"][i] > data["effort_bounds"][i + 1] for i in range(0, 12, 2)):
                raise ValueError("effort_bounds lower bound exceeds upper bound")
            if (data["failure"][0] or data["success"][0]) and not data["done"]:
                raise ValueError("terminal failure/success flags require done")
            if data["failure"][0] and data["success"][0]:
                raise ValueError("failure and success cannot both be true")
            rows.append(data)
        return rows, available

    def update(self, packet: dict[str, torch.Tensor], done: torch.Tensor):
        if self._finished:
            raise RuntimeError("cannot update after report")
        rows, self._request_available = self._validate(packet, done)
        for index, data in enumerate(rows):
            episode = self._episodes[index]
            timestamp, reference, actual = data["time_s"][0], data["command_reference"], data["actual"]
            error = [value - target for value, target in zip(actual, reference)]
            if episode.origin is None:
                episode.origin = data["position_xy"]
            changed = episode.segment is not None and any(abs(value - prior) > tolerance for value, prior, tolerance in
                                                           zip(reference, episode.segment.reference, self.command_tolerances))
            if episode.segment is None or changed:
                responses = [None] * 3
                if changed:
                    old_reference = episode.segment.reference
                    self._finish_segment(episode.segment, "command_change", False)
                    self._segment_counts["command_changes"] += 1
                    for axis, (prior, target, tolerance, command_tolerance, minimum) in enumerate(
                            zip(old_reference, reference, self.tolerances, self.command_tolerances, self.response_min_step)):
                        amplitude = abs(target - prior)
                        if amplitude > command_tolerance and amplitude >= minimum:
                            responses[axis] = _Response(timestamp, prior, target, tolerance, self.hold_s,
                                                        abs(episode.previous_error[axis]) <= tolerance)
                        elif amplitude > command_tolerance:
                            self._response_counts[AXES[axis]]["small_reference_changes"] += 1
                episode.segment = _Segment(reference[:], responses=responses)
            segment = episode.segment
            if episode.previous_time is not None:
                dt = timestamp - episode.previous_time
                self._duration += dt
                self._joint_band_time += dt * all(abs(value) <= tolerance for value, tolerance in zip(episode.previous_error, self.tolerances))
                for axis, (prior, value, tolerance) in enumerate(zip(episode.previous_error, error, self.tolerances)):
                    self._full_iae[axis] += .5 * (abs(prior) + abs(value)) * dt
                    self._band_time[axis] += dt * (abs(prior) <= tolerance)
            episode.errors.add(error, self.tolerances)
            segment.full.add(error, self.tolerances)
            if segment.full.count > self.settle_steps:
                segment.steady.add(error, self.tolerances)
            for response, value in zip(segment.responses, actual):
                if response is not None:
                    response.add(timestamp, value)
            self._record_actuation(episode, data, timestamp)
            self._tilt.add(data["tilt"][0])
            self._drift.add(math.dist(episode.origin, data["position_xy"]))
            if self._request_available:
                for signal, value, target in zip(self._request_error, reference, data["request"]):
                    signal.add(value - target)
            episode.previous_time = timestamp
            episode.previous_error = error
            if data["done"]:
                self._finish_episode(index, "reset", bool(data["failure"][0]), bool(data["success"][0]))

    def _record_actuation(self, episode, data, timestamp):
        def record(name, values):
            for signal, value in zip(self._actuation[name], values):
                signal.add(value)
        if episode.previous_time is not None:
            dt = timestamp - episode.previous_time
            for name, current, previous in (("leg_target_rate", data["leg_target"], episode.previous_leg),
                                            ("wheel_target_acceleration", data["wheel_target"], episode.previous_wheel),
                                            ("effort_rate", data["motor_effort"], episode.previous_effort)):
                record(name, [(now - before) / dt for now, before in zip(current, previous)])
        record("actual_effort", data["motor_effort"])
        record("leg_position_error", [math.atan2(math.sin(actual - target), math.cos(actual - target))
                                      for actual, target in zip(data["motor_position"][:4], data["leg_target"])])
        record("wheel_velocity_error", [actual - target for actual, target in zip(data["motor_velocity"][4:], data["wheel_target"])])
        self._power.add(sum(abs(effort * speed) for effort, speed in zip(data["motor_effort"], data["motor_velocity"])))
        for index, (effort, requested) in enumerate(zip(data["motor_effort"], data["requested_motor_effort"])):
            low, high = data["effort_bounds"][index * 2:index * 2 + 2]
            # A zero-width disabled actuator has no meaningful saturation ratio.
            if high > low:
                self._bounds_samples[index] += 1
                self._saturated[index] += effort <= low + self.saturation_tolerance or effort >= high - self.saturation_tolerance
                self._clipped[index] += requested < low - self.saturation_tolerance or requested > high + self.saturation_tolerance
        self._actuation_samples += 1
        episode.previous_leg, episode.previous_wheel, episode.previous_effort = data["leg_target"], data["wheel_target"], data["motor_effort"]

    def _finish_segment(self, segment, reason, failure):
        counts = self._segment_counts
        counts["total"] += 1
        counts["partial"] += reason == "partial"
        counts["failed"] += failure
        counts["discarded_settle_samples"] += segment.full.count - segment.steady.count
        if segment.steady.count >= self.min_steady_samples:
            counts["eligible"] += 1
            self._steady.merge(segment.steady)
        else:
            counts["short"] += 1
            counts["discarded_short_samples"] += segment.steady.count
        for name, response in zip(AXES, segment.responses):
            if response is None:
                continue
            event = response.finish(reason, failure)
            counts = self._response_counts[name]
            counts["events"] += 1
            counts["eligible_steps"] += event["step_eligible"]
            counts["short_hold_candidates"] += not event["held_long_enough"]
            counts["nonsteady_start"] += not event["initial_tracking_in_band"]
            for key in ("failed", "partial", "rise_censored", "settling_censored"):
                counts[key] += event["failure" if key == "failed" else key]
            for key, signal in self._response_values[name].items():
                if event[key] is not None:
                    signal.add(event[key])
            self._recent.append({"axis": name, **event})

    def _finish_episode(self, index, reason, failure=False, success=False):
        episode = self._episodes[index]
        if not episode.errors.count:
            return
        self._full.merge(episode.errors)
        counts = self._episode_counts
        counts["completed"] += reason == "reset"
        counts["partial"] += reason == "partial"
        counts["failed"] += failure
        counts["success_flags"] += success
        counts["short"] += episode.errors.count < self.settle_steps + self.min_steady_samples
        counts["partial_all_samples_tracking_in_band" if reason == "partial" else "completed_all_samples_tracking_in_band"] += (
            not failure and episode.errors.all_in_band == episode.errors.count)
        self._finish_segment(episode.segment, reason, failure)
        self._episodes[index] = _Episode()

    def report(self) -> dict:
        if self._cached is not None:
            return self._cached
        for index in range(self.num_envs):
            self._finish_episode(index, "partial")
        self._finished = True
        units = {"leg_target_rate": "rad/s", "wheel_target_acceleration": "rad/s^2", "effort_rate": "N*m/s",
                 "actual_effort": "N*m", "leg_position_error": "rad", "wheel_velocity_error": "rad/s"}
        actuation = {name: {"unit": units[name], "channels": [signal.report() for signal in signals]}
                     for name, signals in self._actuation.items()}
        actuation.update(sample_count=self._actuation_samples,
            active_bound_samples=self._bounds_samples,
            actual_bound_fraction=[count / samples if samples else None for count, samples in zip(self._saturated, self._bounds_samples)],
            requested_outside_bounds_fraction=[count / samples if samples else None for count, samples in zip(self._clipped, self._bounds_samples)],
            sampled_abs_mechanical_power={"unit": "W", **self._power.report()})
        full_interval = self._full.report("per_environment_episode")
        full_interval.update(observed_duration_s=self._duration,
            all_axes_in_band_time_fraction=self._joint_band_time / self._duration if self._duration else None)
        for axis, name in enumerate(AXES):
            full_interval["axes"][name].update(iae=self._full_iae[axis] if self._duration else None,
                in_band_time_fraction=self._band_time[axis] / self._duration if self._duration else None)
        self._cached = {"available": bool(self._full.count), "full_interval": full_interval,
            "steady": {"available": bool(self._steady.count), **self._steady.report("per_environment_command_hold_segment"),
                       **self._segment_counts}, "episodes": self._episode_counts,
            "response": {"axes": {name: {**self._response_counts[name], "metrics":
                         {key: signal.report() for key, signal in self._response_values[name].items()}} for name in AXES},
                         "recent_events": list(self._recent), "recent_event_capacity": self._recent.maxlen},
            "actuation": actuation, "tilt": {"unit": "rad", **self._tilt.report()},
            "distance_from_episode_origin": {"unit": "m", "scope": "path displacement; apply as drift only to stationary cases", **self._drift.report()},
            "request_reference_difference": {"available": bool(self._request_available),
                  "axes": {name: signal.report() for name, signal in zip(AXES, self._request_error)}},
            "protocol": {"policy_dt_s": self.policy_dt_s, "settle_steps": self.settle_steps,
                "min_steady_samples": self.min_steady_samples, "tracking_tolerance": list(self.tolerances),
                "command_tolerance": list(self.command_tolerances), "settling_hold_s": self.hold_s,
                "response_min_step": list(self.response_min_step),
                "threshold_scope": "simulation screening thresholds; not hardware acceptance limits",
                "axis_order": list(AXES), "signal_time": "PRE-reset physical sample time",
                "motor_order": "four leg targets followed by two wheel targets; position, velocity, effort and bounds share this order",
                "weighting": "equal sample weights within each scenario; aggregate scenarios and training seeds separately",
                "response_start": "first physical sample with changed effective reference; excludes unknown sub-sample latency",
                "response_iae": "trapezoidal integral from first sample after change; censored prefixes excluded from complete_iae",
                "full_interval_iae": "trapezoidal integral of sampled effective-reference error; no integration across reset",
                "in_band_time": "left-sample hold over observed within-episode intervals; first isolated sample has no duration",
                "step_eligibility": "reference change >= axis minimum; previous physical sample tracks old reference within tolerance; unchanged new reference for at least settling_hold_s; small slews are not step responses",
                "initial_tracking_in_band": "single physical sample screening before change, not proof of previous steady convergence",
                "rise": "first sampled 10% and 90% crossings of previous-reference to new-reference step; no interpolation",
                "settling": "final uninterrupted in-band run lasting at least hold_s; failures are censored",
                "scope": "policy-rate physical samples; no high-rate current-loop or electrical-energy claim",
                "leg_position_error": "shortest circular difference atan2(sin(actual-target),cos(actual-target)) for continuous revolute joints",
                "unsupported": ["transport request latency without request timestamps", "frequency-domain bandwidth", "continuous-time extrema between samples"]}}
        return self._cached

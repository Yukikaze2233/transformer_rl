"""Deterministic deployment: validated bundle, history, limiting and target mapping.

No simulator, exploration distribution, optimizer or motor driver is imported.
The caller supplies scaled observations and owns PC-side torque feedback control.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


class FrameRuntime:
    def __init__(self, directory, *, observation_schema, policy_dt_s, backend="onnx", threads=1,
                 priming_iterations=50):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != "transformer_rl.packed_policy" or self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported deployment bundle")
        control = self.manifest["control"]
        if observation_schema != control["observation_schema"] or policy_dt_s != control["policy_dt_s"]:
            raise ValueError("deployment observation schema or policy interval differs from training")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be positive")
        if type(priming_iterations) is not int or priming_iterations < 1:
            raise ValueError("priming_iterations must be a positive integer")
        self.dt = policy_dt_s
        policy = self.manifest["model"]["policy"]
        self.length, self.frame_dim, self.action_dim = policy["history_length"], policy["frame_dim"], policy["action_dim"]
        self.bounds = np.asarray(control["action_bounds"], dtype=np.float32)
        self.scale = np.asarray(control["target_scale"], dtype=np.float32)
        self.offset = np.asarray(control["target_offset"], dtype=np.float32)
        self.frames = np.zeros((1, self.length, self.frame_dim), dtype=np.float32)
        self.last_time = None
        name = {"onnx": "policy.onnx", "torchscript": "policy.pt"}.get(backend)
        if name is None or name not in self.manifest["files"]:
            raise ValueError("requested runtime is absent from the bundle")
        graph_path = self.directory / name
        if hashlib.sha256(graph_path.read_bytes()).hexdigest() != self.manifest["files"][name]:
            raise ValueError("deployment graph hash mismatch")
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(graph_path), sess_options=options, providers=["CPUExecutionProvider"])
        else:
            import torch
            self.torch = torch
            torch.set_num_threads(threads)
            self.session = torch.jit.load(str(graph_path), map_location="cpu").eval()
        self.preparation = self._prime(priming_iterations)

    def _mean_forward(self, frames):
        if self.backend == "onnx":
            mean = self.session.run(["mean"], {"frames": frames})[0]
        else:
            with self.torch.inference_mode():
                mean = self.session(self.torch.from_numpy(frames)).numpy()
        if (mean.dtype != np.float32 or mean.shape != (1, self.action_dim)
                or not np.isfinite(mean).all()):
            raise FloatingPointError("deployment policy produced invalid actions")
        return mean[0].copy()

    def _prime(self, iterations):
        """Exercise the loaded graph before arming, without issuing any actions."""
        frames = np.zeros_like(self.frames)
        started = time.perf_counter_ns()
        try:
            for _ in range(iterations):
                self._mean_forward(frames)
        finally:
            self.reset()
        return {"iterations": iterations,
                "duration_ms": (time.perf_counter_ns() - started) / 1e6,
                "scope": "prearm CPU mean graph only; excludes sensor I/O and action issuance"}

    def reset(self):
        self.last_time = None
        self.frames.fill(0)

    def step(self, frame, timestamp_s):
        frame = np.asarray(frame)
        if frame.dtype != np.float32 or frame.shape != (self.frame_dim,) or not np.isfinite(frame).all():
            raise ValueError("deployment frame must be finite float32 [F] in declared feature order")
        if type(timestamp_s) not in (float, int) or not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self.last_time is None:
            self.frames[:] = frame
        else:
            elapsed = timestamp_s - self.last_time
            if not math.isclose(elapsed, self.dt, rel_tol=0.2, abs_tol=1e-6):
                raise ValueError("policy sampling interval changed; reset history before resuming")
            self.frames[:, :-1] = self.frames[:, 1:].copy()
            self.frames[:, -1] = frame
        self.last_time = timestamp_s
        try:
            mean = self._mean_forward(self.frames)
        except Exception:
            self.reset()
            raise
        issued = np.clip(mean, -self.bounds, self.bounds)
        return {"mean": mean, "issued": issued, "targets": self.offset + self.scale * issued}

    def benchmark(self, *, warmup=50, iterations=1000):
        if type(warmup) is not int or warmup < 1 or type(iterations) is not int or iterations < 100:
            raise ValueError("benchmark requires positive warmup and at least 100 measured calls")
        frame = np.zeros(self.frame_dim, dtype=np.float32)
        self.reset()
        durations = []
        for index in range(warmup + iterations):
            started = time.perf_counter_ns()
            self.step(frame, index * self.dt)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                durations.append(elapsed_ms)
        self.reset()
        return {"backend": self.backend, "iterations": iterations,
                "mean_ms": float(np.mean(durations)), "p99_ms": float(np.percentile(durations, 99)),
                "max_ms": float(np.max(durations)),
                "deadline_misses": sum(ms >= self.dt * 1000 for ms in durations),
                "scope": "CPU history + mean inference + target mapping; excludes sensor I/O, transport and torque control"}


def run_control_loop(runtime, read_frame, send_targets, *, should_stop, on_fault, inference_budget_s=0.008):
    """Run until shutdown or the first fault, then hand control to the caller.

    read_frame returns (scaled float32 frame, sensor_capture_monotonic_seconds).
    send_targets receives targets and the issued action to echo in the next frame.
    on_fault handles both faults and intentional shutdown (InterruptedError).
    History is cleared before this callback. Restarting requires a new call by
    the controller; no automatic rearming or stale action resend is performed.
    Scheduling is soft and does not guarantee OS real-time deadlines.
    """
    if not 0 < inference_budget_s < runtime.dt:
        raise ValueError("inference budget must leave positive control-chain margin")
    deadline = time.monotonic()
    reason = InterruptedError("policy control loop stopped")
    try:
        while not should_stop():
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            release = time.monotonic()
            try:
                if release - deadline > runtime.dt * 0.2:
                    raise TimeoutError("missed policy release")
                frame, captured = read_frame()
                if not 0 <= time.monotonic() - captured <= runtime.dt:
                    raise TimeoutError("sensor frame is stale or has a future timestamp")
                result = runtime.step(frame, deadline)
                if time.monotonic() - release > inference_budget_s:
                    raise TimeoutError("observation and inference exceeded control budget")
                send_targets(result["targets"], result["issued"])
                if time.monotonic() - release > runtime.dt:
                    raise TimeoutError("control callback exceeded policy deadline")
            except Exception as error:
                reason = error
                return "fault"
            deadline += runtime.dt
        return "stopped"
    except BaseException as error:
        reason = error
        raise
    finally:
        runtime.reset()
        on_fault(reason)

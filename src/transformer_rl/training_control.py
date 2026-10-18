"""Private, file-backed control of explicitly managed packed-policy runs.

The frontend is standard-library only. A pause is acknowledged by the learner
at an update boundary; it never suspends an arbitrary PID or reloads an episode.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid


PROTOCOL = "transformer_rl.training_control.v1"
JSON_LIMIT = 256 * 1024


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def read_json(path, default=None):
    """Read a bounded regular file, without following a final symlink."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return default
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > JSON_LIMIT:
            raise ValueError(f"invalid control file: {path}")
        raw = stream.read(JSON_LIMIT + 1)
    if len(raw) > JSON_LIMIT:
        raise ValueError(f"control file exceeds limit: {path}")
    return json.loads(raw)


def write_json(path, value, *, replace=False):
    """Publish and fsync the directory before acknowledging a control action."""
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if len(raw) > JSON_LIMIT:
        raise ValueError("control document exceeds limit")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_pin(path):
    path = Path(path)
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(part)
        size = stream.tell()
    return {"path": str(path), "bytes": size, "sha256": state.hexdigest()}


def verify_pin(pin):
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve(strict=True) != path or file_pin(path) != pin:
        raise ValueError(f"artifact changed: {path}")
    return path


@contextmanager
def command_lock(root):
    with (Path(root) / "command.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def process_identity(pid):
    """Inspect only an explicitly owned handle, never enumerate /proc."""
    try:
        first = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        second = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if first[19] != second[19] or second[0] in ("Z", "X"):
            return None
        # A detached worker can be reparented after the submitting CLI exits.
        # PID/start/boot/argv identify it; the observed parent is diagnostic only.
        return {"pid": pid, "start_ticks": int(second[19]), "boot_id": boot_id(),
                "argv": [os.fsdecode(part) for part in argv if part]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _lease(root):
    descriptor = os.open(Path(root) / "run.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _lease_is_held(root):
    """Read only this lock's kernel entry, without probing unrelated processes."""
    path = Path(root) / "run.lock"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("session lease is not a regular file")
    expected = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    with Path("/proc/locks").open() as stream:
        raw = stream.read(JSON_LIMIT + 1)
    if len(raw) > JSON_LIMIT:
        raise ValueError("kernel lock table exceeds the bounded reader")
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK":
            major, minor, inode = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == expected:
                return True
    return False


class TrainingSession:
    """Immutable recipe, detached attempts and commands scoped to one directory."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        recipe_pin = read_json(self.root / "session.pin.json")
        if not recipe_pin:
            raise ValueError("not a managed training session; use init with a new directory")
        if recipe_pin.get("path") != str(self.root / "session.json"):
            raise ValueError("recipe pin escapes the session")
        verify_pin(recipe_pin)
        self.spec = read_json(self.root / "session.json")
        if not self.spec or self.spec.get("protocol") != PROTOCOL:
            raise ValueError("not a managed training session; use init with a new directory")

    @classmethod
    def create(cls, root, config, *, env_factory, updates, python=sys.executable,
               device="cpu", seed=0, rollout_steps=48, checkpoint_interval=40,
               max_seconds=3600., python_paths=(), name=None, tensorboard=False,
               learner_source=None, expected_initial_model_sha256=None):
        for key, value in (("updates", updates), ("rollout_steps", rollout_steps),
                           ("checkpoint_interval", checkpoint_interval)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", env_factory):
            raise ValueError("environment must be module:factory")
        if expected_initial_model_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_initial_model_sha256):
            raise ValueError("initial model identity must be a lowercase SHA256")
        python = str(Path(python).expanduser().absolute())
        if not os.access(python, os.X_OK):
            raise ValueError("training Python must be an executable absolute path")
        config = Path(config).expanduser().resolve(strict=True)
        document = read_json(config)
        if not isinstance(document, dict) or set(document) != {"model", "ppo", "control", "environment"}:
            raise ValueError("expected a packed-policy training configuration")
        paths = [str(Path(path).expanduser().resolve(strict=True)) for path in python_paths]
        root = Path(root).expanduser().absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        source = root / "source" / "transformer_rl"
        package = Path(learner_source).resolve(strict=True) if learner_source else Path(__file__).parent
        shutil.copytree(package, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(config, root / "config.json")
        (root / "attempts").mkdir()
        files = [root / "config.json", *sorted(source.rglob("*.py"))]
        pins = [file_pin(path.resolve()) for path in files]
        write_json(root / "session.json", {
            "protocol": PROTOCOL, "id": uuid.uuid4().hex, "name": name or root.name,
            "created_at": timestamp(), "created_boot": boot_id(), "python": python,
            "python_paths": paths, "env_factory": env_factory, "updates": updates,
            "rollout_steps": rollout_steps, "checkpoint_interval": checkpoint_interval,
            "seed": seed, "device": device, "max_seconds": float(max_seconds),
            "tensorboard": bool(tensorboard), "pins": pins,
            "expected_initial_model_sha256": expected_initial_model_sha256,
            "architecture": document.get("model", {}).get("policy", {}).get("architecture", "unknown"),
            "history_length": document.get("model", {}).get("policy", {}).get("history_length"),
            "num_envs": document.get("environment", {}).get("num_envs"),
        })
        write_json(root / "session.pin.json", file_pin(root.resolve() / "session.json"))
        return cls(root)

    def attempt(self):
        pointer = read_json(self.root / "current.json")
        if not pointer:
            return None
        name = pointer["attempt"]
        if not re.fullmatch(r"\d{4}-[0-9a-f]{12}", name):
            raise ValueError("invalid attempt pointer")
        path = (self.root / "attempts" / name).resolve(strict=True)
        if path.parent != self.root / "attempts":
            raise ValueError("attempt escapes the session")
        return path

    def status(self):
        result = {"name": self.spec["name"], "root": str(self.root), "phase": "ready", "active": False,
                  "target_update": self.spec["updates"], "update": 0, "architecture": self.spec["architecture"],
                  "device": self.spec["device"], "seed": self.spec["seed"],
                  "episode_state_restored": False, "pause_mode": "same_process_at_update_boundary"}
        attempt = self.attempt()
        if attempt is None:
            return result
        request = read_json(attempt / "request.json")
        control = read_json(attempt / "control" / "status.json", {})
        if not control:
            result.update(phase="starting", update=request.get("start_update", 0))
        if control and control.get("token") != request["token"]:
            raise ValueError("worker status belongs to another attempt")
        result.update(control)
        result.update(attempt=attempt.name, log=str(attempt / "worker.log"))
        intended = read_json(attempt / "control" / "request.json")
        result["requested"] = intended["operation"]
        result["request_sequence"] = intended["sequence"]
        final = read_json(attempt / "control" / "result.json")
        if final:
            if final.get("token") != request["token"]:
                raise ValueError("worker result belongs to another attempt")
            result.update(final["summary"])
        current_boot = boot_id()
        identity = control.get("identity") or read_json(attempt / "launch.json")
        exit_receipt = read_json(attempt / "exit.json")
        # Retired handles must not be polled on a different kernel boot.
        if current_boot != request["boot_id"]:
            result.update(phase=final["summary"]["phase"] if final else "interrupted", active=False, boot_changed=True)
        elif exit_receipt and exit_receipt.get("identity") == identity:
            result["active"] = False
            if not final:
                result.update(phase="failed", error="worker exited without a committed learning endpoint")
        elif identity and process_identity(identity["pid"]) == identity:
            result["active"] = True
            if final:
                result.update(phase="closing", result_phase=final["summary"]["phase"])
        elif final:
            result["active"] = False
        elif not identity and time.time() - request["submitted_epoch"] < 30:
            result.update(phase="starting", active=True)
        else:
            result.update(phase="failed" if control.get("phase") == "failed" else "interrupted", active=False,
                          error=control.get("error", "worker exited without a final checkpoint receipt"))
        if current_boot == request["boot_id"] and not result["active"] and _lease_is_held(self.root):
            # Linux can clear cmdline during exit before releasing the inherited
            # flock. Do not report shutdown or permit a new attempt in that gap.
            result.update(phase="closing" if final else "ownership_pending", active=True)
        result["checkpoint_resume"] = bool(final and final["summary"].get("phase") == "stopped")
        return result

    def start(self, *, paused=False, accept_new_boot=False, inherited_fds=()):
        with command_lock(self.root):
            state = self.status()
            if state["active"] or state["phase"] == "completed":
                return state
            previous = self.attempt()
            expected_boot = read_json(previous / "request.json")["boot_id"] if previous else self.spec["created_boot"]
            if boot_id() != expected_boot and not accept_new_boot:
                raise ValueError("boot changed; inspect the session, then explicitly pass --accept-new-boot")
            for pin in self.spec["pins"]:
                path = Path(pin["path"])
                if path != self.root / "config.json" and not path.is_relative_to(self.root / "source"):
                    raise ValueError("source pin escapes the session")
                verify_pin(pin)
            resume, start_update, spent, transitions = None, 0, 0., 0
            if previous:
                final = read_json(previous / "control" / "result.json")
                if not final or final["summary"].get("phase") != "stopped":
                    raise ValueError("unclean exit: automatic rollback/replay is refused; inspect the saved evidence")
                expected = [str(previous / "train/checkpoints/final.pt"),
                            str(previous / "train/checkpoints/final.pt.json")]
                if [pin["path"] for pin in final["checkpoint_pins"]] != expected:
                    raise ValueError("final checkpoint pins escape their attempt")
                for pin in final["checkpoint_pins"]:
                    verify_pin(pin)
                resume = final["checkpoint_pins"][0]["path"]
                start_update = final["summary"]["update"]
                spent = final["summary"]["session_active_elapsed_s"]
                transitions = final["summary"]["cumulative_transitions"]
            if spent >= self.spec["max_seconds"]:
                raise ValueError("session active-time budget exhausted; the original budget is not reset")
            descriptor = _lease(self.root)
            try:
                generation = len(list((self.root / "attempts").iterdir())) + 1
                attempt = self.root / "attempts" / f"{generation:04d}-{uuid.uuid4().hex[:12]}"
                attempt.mkdir()
                control = attempt / "control"
                control.mkdir()
                token = uuid.uuid4().hex
                binding = {"protocol": PROTOCOL, "token": token, "boot_id": boot_id(),
                           "submitted_at": timestamp(), "submitted_epoch": time.time(),
                           "lease_fd": descriptor, "root": str(self.root), "attempt": str(attempt),
                           "inherited_locks": [{"fd": fd, "device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino}
                                               for fd in inherited_fds],
                           "start_update": start_update, "prior_active_elapsed_s": spent,
                           "prior_transitions": transitions, "target_update": self.spec["updates"]}
                write_json(attempt / "request.json", binding)
                write_json(control / "request.json", {"token": token, "sequence": 0,
                           "operation": "pause" if paused else "resume", "at": timestamp()})
                write_json(self.root / "current.json", {"attempt": attempt.name}, replace=True)
                argv = [self.spec["python"], "-m", "transformer_rl.frame_process", "train",
                        "--config", str(self.root / "config.json"), "--env-factory", self.spec["env_factory"],
                        "--run-dir", str(attempt / "train"), "--updates", str(self.spec["updates"] - start_update),
                        "--seed", str(self.spec["seed"]), "--device", self.spec["device"],
                        "--rollout-steps", str(self.spec["rollout_steps"]),
                        "--checkpoint-interval", str(self.spec["checkpoint_interval"]),
                        "--max-seconds", str(self.spec["max_seconds"] - spent),
                        "--consumed-update-offset", str(start_update), "--control-dir", str(control)]
                if resume:
                    argv += ["--resume", resume]
                elif self.spec.get("expected_initial_model_sha256"):
                    argv += ["--expected-initial-model-sha256", self.spec["expected_initial_model_sha256"]]
                if not self.spec["tensorboard"]:
                    argv.append("--no-tensorboard")
                environment = dict(os.environ)
                environment["PYTHONPATH"] = os.pathsep.join([str(self.root / "source"), *self.spec["python_paths"]])
                environment["PYTHONUNBUFFERED"] = "1"
                with (attempt / "worker.log").open("xb") as log:
                    process = subprocess.Popen(argv, cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(descriptor, *inherited_fds))
                self.submitted_process = process
                identity = process_identity(process.pid)
                if identity:
                    write_json(attempt / "launch.json", identity)
            finally:
                # The worker inherited this open file description. Unlocking it
                # here would also release the worker's lifetime lease.
                os.close(descriptor)
            return self.status()

    def command(self, operation, *, wait_seconds=0.):
        if operation not in ("pause", "resume", "stop"):
            raise ValueError("unknown training operation")
        if not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("wait_seconds must be finite and nonnegative")
        with command_lock(self.root):
            state = self.status()
            if not state["active"]:
                if operation == "pause" and state["phase"] == "ready":
                    return state
                if state["phase"] in ("stopped", "completed") and operation == "stop":
                    return state
                raise ValueError("no live managed worker; use start for a new or checkpoint-resumed attempt")
            attempt = self.attempt()
            binding = read_json(attempt / "request.json")
            if binding["boot_id"] != boot_id():
                raise ValueError("worker boot changed; no command was sent")
            current = read_json(attempt / "control" / "request.json")
            if current["operation"] == "stop" and operation != "stop":
                raise ValueError("save-and-exit is already pending; wait for its final receipt")
            if current["operation"] != operation:
                current = {"token": binding["token"], "sequence": current["sequence"] + 1,
                           "operation": operation, "at": timestamp()}
                write_json(attempt / "control" / "request.json", current, replace=True)
            sequence = current["sequence"]
        deadline = time.monotonic() + wait_seconds
        while True:
            state = self.status()
            acknowledged = operation != "stop" and state.get("ack_sequence", -1) >= sequence
            if not state["active"] or acknowledged or time.monotonic() >= deadline:
                return state
            time.sleep(.1)


class TrainingControl:
    """Learner-side handshake. No model, optimizer or RNG operations occur here."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.binding = read_json(self.directory.parent / "request.json")
        if (not self.binding or self.binding.get("protocol") != PROTOCOL
                or self.directory != Path(self.binding["attempt"]) / "control"
                or self.binding["boot_id"] != boot_id()):
            raise ValueError("invalid training control binding")
        self.root = Path(self.binding["root"])
        descriptor = self.binding["lease_fd"]
        held, expected = os.fstat(descriptor), (self.root / "run.lock").stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("worker did not inherit its session lease")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for inherited in self.binding.get("inherited_locks", []):
            value = os.fstat(inherited["fd"])
            if (value.st_dev, value.st_ino) != (inherited["device"], inherited["inode"]):
                raise ValueError("inherited queue/resource lock changed")
        self.started = time.monotonic()
        self.paused_seconds = 0.
        self.pause_started = None
        self.state = {"token": self.binding["token"], "identity": process_identity(os.getpid()),
                      "ack_sequence": -1, "update": self.binding["start_update"],
                      "target_update": self.binding["target_update"], "cumulative_transitions": self.binding["prior_transitions"],
                      "phase": "initializing", "checkpoint": None, "checkpoint_update": None}
        self.state["inherited_lock_descriptors_verified"] = len(self.binding.get("inherited_locks", []))
        self.publish("initializing")

    def elapsed(self):
        paused = self.paused_seconds
        if self.pause_started is not None:
            paused += time.monotonic() - self.pause_started
        return time.monotonic() - self.started - paused, paused

    def publish(self, phase, **values):
        self.state.update(values, phase=phase, updated_at=timestamp())
        active, paused = self.elapsed()
        self.state.update(active_elapsed_s=active, paused_elapsed_s=paused,
                          session_active_elapsed_s=active + self.binding["prior_active_elapsed_s"])
        write_json(self.directory / "status.json", self.state, replace=True)

    def _request(self):
        request = read_json(self.directory / "request.json")
        if (not request or request.get("token") != self.binding["token"]
                or type(request.get("sequence")) is not int or request["sequence"] < self.state["ack_sequence"]
                or request.get("operation") not in ("pause", "resume", "stop")):
            raise ValueError("invalid or stale control request")
        return request

    def boundary(self, stop, save_checkpoint):
        request = self._request()
        if request["operation"] == "pause":
            self.publish("checkpointing")
            path = save_checkpoint(request["sequence"])
            self.pause_started = time.monotonic()
            self.publish("paused", ack_sequence=request["sequence"], checkpoint=str(path),
                         checkpoint_update=self.state["update"])
            try:
                last_publish = time.monotonic()
                while stop.reason is None:
                    request = self._request()
                    if request["operation"] != "pause":
                        break
                    time.sleep(.1)
                    if time.monotonic() - last_publish >= 1:
                        self.publish("paused")
                        last_publish = time.monotonic()
            finally:
                duration = time.monotonic() - self.pause_started
                self.paused_seconds += duration
                self.pause_started = None
                stop.deadline += duration
        if request["operation"] == "stop":
            stop.reason = "control_stop"
            # Stop is acknowledged only after final.pt and its receipt exist.
            self.publish("stopping")
            return False
        if stop.stopped():
            return False
        self.publish("collecting", ack_sequence=request["sequence"])
        return True

    def finish(self, report):
        phase = report["status"]
        request = self._request()
        checkpoint = Path(report["checkpoint"])
        active, paused = self.elapsed()
        self.state.update(phase=phase, update=report["final_update"], cumulative_transitions=report["cumulative_transitions"],
                          checkpoint=str(checkpoint), checkpoint_update=report["final_update"],
                          ack_sequence=request["sequence"], stop_reason=report["stop_reason"],
                          active_elapsed_s=active, paused_elapsed_s=paused,
                          session_active_elapsed_s=active + self.binding["prior_active_elapsed_s"])
        # The final result is the cold-resume commit marker. Failures or a lost
        # worker cannot promote a periodic/pause checkpoint to this receipt.
        write_json(self.directory / "result.json", {"token": self.binding["token"],
            "summary": dict(self.state), "checkpoint_pins": [file_pin(checkpoint), file_pin(Path(str(checkpoint) + ".json"))]})
        self.publish(phase)

    def fail(self, error):
        self.publish("failed", error=f"{type(error).__name__}: {error}")


_evaluation_control = None


def evaluation_control():
    return _evaluation_control


class EvaluationControl(TrainingControl):
    """Pause step-driven evaluation in memory; finish its window before exit."""

    def __init__(self, directory):
        super().__init__(directory)
        global _evaluation_control
        _evaluation_control = self
        self.publish("evaluating", update=0)

    def boundary(self, index, steps):
        request = self._request()
        if request["operation"] == "pause":
            self.pause_started = time.monotonic()
            self.publish("paused", update=index, target_update=steps, ack_sequence=request["sequence"],
                         pause_scope="evaluation_in_memory_no_disk_resume")
            try:
                while request["operation"] == "pause":
                    time.sleep(.1)
                    request = self._request()
            finally:
                self.paused_seconds += time.monotonic() - self.pause_started
                self.pause_started = None
        if index % 25 == 0 or request["sequence"] != self.state["ack_sequence"]:
            self.publish("stopping" if request["operation"] == "stop" else "evaluating",
                         update=index, target_update=steps,
                         ack_sequence=self.state["ack_sequence"] if request["operation"] == "stop" else request["sequence"])

    def finish_evaluation(self, steps):
        request = self._request()
        self.state.update(phase="completed", update=steps, target_update=steps, ack_sequence=request["sequence"])
        write_json(self.directory / "result.json", {"token": self.binding["token"], "summary": dict(self.state)})
        self.publish("completed")

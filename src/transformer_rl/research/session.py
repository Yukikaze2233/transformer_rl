"""旧 packed-policy 研究会话；通用状态与控制由基类负责。"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from ..common import training_control as control
from ..common.training_control import ManagedSession


class TrainingSession(ManagedSession):
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
        document = control.read_json(config)
        if not isinstance(document, dict) or set(document) != {"model", "ppo", "control", "environment"}:
            raise ValueError("expected a packed-policy training configuration")
        paths = [str(Path(path).expanduser().resolve(strict=True)) for path in python_paths]
        root = Path(root).expanduser().absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        source = root / "source" / "transformer_rl"
        package = Path(learner_source).resolve(strict=True) if learner_source else Path(__file__).parents[1]
        shutil.copytree(package, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(config, root / "config.json")
        (root / "attempts").mkdir()
        files = [root / "config.json", *sorted(source.rglob("*.py"))]
        pins = [control.file_pin(path.resolve()) for path in files]
        control.write_json(root / "session.json", {
            "protocol": control.PROTOCOL, "id": uuid.uuid4().hex, "name": name or root.name,
            "created_at": control.timestamp(), "created_boot": control.boot_id(), "python": python,
            "python_paths": paths, "env_factory": env_factory, "updates": updates,
            "rollout_steps": rollout_steps, "checkpoint_interval": checkpoint_interval,
            "seed": seed, "device": device, "max_seconds": float(max_seconds),
            "tensorboard": bool(tensorboard), "pins": pins,
            "expected_initial_model_sha256": expected_initial_model_sha256,
            "architecture": document.get("model", {}).get("policy", {}).get("architecture", "unknown"),
            "history_length": document.get("model", {}).get("policy", {}).get("history_length"),
            "num_envs": document.get("environment", {}).get("num_envs"),
        })
        control.write_json(root / "session.pin.json", control.file_pin(root.resolve() / "session.json"))
        return cls(root)

    def start(self, *, paused=False, accept_new_boot=False, inherited_fds=()):
        with control.command_lock(self.root):
            state = self.status()
            if state["active"] or state["phase"] == "completed":
                return state
            previous = self.attempt()
            expected_boot = control.read_json(previous / "request.json")["boot_id"] if previous else self.spec["created_boot"]
            if control.boot_id() != expected_boot and not accept_new_boot:
                raise ValueError("boot changed; inspect the session, then explicitly pass --accept-new-boot")
            for pin in self.spec["pins"]:
                path = Path(pin["path"])
                if path != self.root / "config.json" and not path.is_relative_to(self.root / "source"):
                    raise ValueError("source pin escapes the session")
                control.verify_pin(pin)
            resume, start_update, spent, transitions = None, 0, 0., 0
            if previous:
                final = control.read_json(previous / "control" / "result.json")
                if not final or final["summary"].get("phase") != "stopped":
                    raise ValueError("unclean exit: automatic rollback/replay is refused; inspect the saved evidence")
                expected = [str(previous / "train/checkpoints/final.pt"),
                            str(previous / "train/checkpoints/final.pt.json")]
                if [pin["path"] for pin in final["checkpoint_pins"]] != expected:
                    raise ValueError("final checkpoint pins escape their attempt")
                for pin in final["checkpoint_pins"]:
                    control.verify_pin(pin)
                resume = final["checkpoint_pins"][0]["path"]
                start_update = final["summary"]["update"]
                spent = final["summary"]["session_active_elapsed_s"]
                transitions = final["summary"]["cumulative_transitions"]
            if spent >= self.spec["max_seconds"]:
                raise ValueError("session active-time budget exhausted; the original budget is not reset")
            descriptor = control._lease(self.root)
            try:
                generation = len(list((self.root / "attempts").iterdir())) + 1
                attempt = self.root / "attempts" / f"{generation:04d}-{uuid.uuid4().hex[:12]}"
                attempt.mkdir()
                control_dir = attempt / "control"
                control_dir.mkdir()
                token = uuid.uuid4().hex
                binding = {"protocol": control.PROTOCOL, "token": token, "boot_id": control.boot_id(),
                           "submitted_at": control.timestamp(), "submitted_epoch": time.time(),
                           "lease_fd": descriptor, "root": str(self.root), "attempt": str(attempt),
                           "inherited_locks": [{"fd": fd, "device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino}
                                               for fd in inherited_fds],
                           "start_update": start_update, "prior_active_elapsed_s": spent,
                           "prior_transitions": transitions, "target_update": self.spec["updates"]}
                control.write_json(attempt / "request.json", binding)
                control.write_json(control_dir / "request.json", {"token": token, "sequence": 0,
                           "operation": "pause" if paused else "resume", "at": control.timestamp()})
                control.write_json(self.root / "current.json", {"attempt": attempt.name}, replace=True)
                worker_module = ('transformer_rl.research.frame_process'
                    if (self.root / 'source/transformer_rl/research/frame_process.py').is_file()
                    else 'transformer_rl.frame_process')
                argv = [self.spec["python"], "-m", worker_module, "train",
                        "--config", str(self.root / "config.json"), "--env-factory", self.spec["env_factory"],
                        "--run-dir", str(attempt / "train"), "--updates", str(self.spec["updates"] - start_update),
                        "--seed", str(self.spec["seed"]), "--device", self.spec["device"],
                        "--rollout-steps", str(self.spec["rollout_steps"]),
                        "--checkpoint-interval", str(self.spec["checkpoint_interval"]),
                        "--max-seconds", str(self.spec["max_seconds"] - spent),
                        "--consumed-update-offset", str(start_update), "--control-dir", str(control_dir)]
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
                identity = control.process_identity(process.pid)
                if identity:
                    control.write_json(attempt / "launch.json", identity)
            finally:
                # The worker inherited this open file description. Unlocking it
                # here would also release the worker's lifetime lease.
                os.close(descriptor)
            return self.status()

"""Serial research queues with explicit inheritance and managed child ownership."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .training_control import (TrainingSession, _lease, _lease_is_held, boot_id,
                               command_lock, file_pin, process_identity,
                               read_json, timestamp, verify_pin, write_json)


FORMAT = "transformer_rl.managed_research_queue.v1"


def document(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("research definition exceeds the explicit 2 MiB limit")
    return json.loads(raw)


def cell_key(cell):
    return f"{cell['rate_id']}/{cell['variant']}/seed_{cell['training_seed']}"


def open_session(value):
    root = Path(value).expanduser()
    if not root.exists():
        registry = read_json(Path.home() / ".config/transformer-rl/managed.json", {})
        if str(value) not in registry:
            raise ValueError(f"unknown managed session or queue: {value}")
        root = Path(registry[str(value)])
    if (root / 'structured.json').is_file():
        from ._structured_session import StructuredTrainingSession
        return StructuredTrainingSession(root)
    return TrainingQueue(root) if (root / "queue.json").is_file() else TrainingSession(root)


class TrainingQueue:
    cell_key = staticmethod(cell_key)

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        pin = read_json(self.root / "queue.pin.json")
        if not pin or pin["path"] != str(self.root / "queue.json"):
            raise ValueError("invalid managed research queue")
        verify_pin(pin)
        self.spec = read_json(self.root / "queue.json")
        if self.spec["format"] != FORMAT:
            raise ValueError("unsupported managed research queue")

    @property
    def parent_manifest(self):
        return document(self.root / "parent_manifest.json")

    @classmethod
    def adopt(cls, root, plan_path, *, name):
        plan = document(plan_path)
        manifest = document(verify_pin(plan["parent_manifest"]))
        inherited = document(verify_pin(plan["inherited_summary"]))
        cells = manifest["inputs"]["cells"]
        keys = [cell_key(cell) for cell in cells]
        if (not keys or len(set(keys)) != len(keys)
                or inherited["manifest_sha256"] != manifest["sha256"]
                or len(keys) != plan["expected_total_cells"]):
            raise ValueError("the original complete research matrix is required")
        if plan["authorized_boot"] != boot_id():
            raise ValueError("adoption authorization belongs to another boot")
        inherited_results = inherited["results"]
        retained = set(inherited_results) | set(plan["blocked"])
        if (set(inherited_results) & set(plan["blocked"]) or not retained <= set(keys)
                or retained != set(plan["retained_keys"])):
            raise ValueError("adoption must preserve all explicitly retained research cells")
        for key, row in inherited_results.items():
            cell = cells[keys.index(key)]
            if row["status"] == "incomplete":
                if row["training"]["charged_updates"] != cell["updates"]:
                    raise ValueError("historical reservation or failure changed")
            elif (row["status"] != "completed" or row["training"]["completed_updates"] != cell["updates"]
                  or set(row["development"]) != {str(seed) for seed in manifest["development_seeds"]}
                  or any(value is None or value["status"] != "completed" for value in row["development"].values())):
                raise ValueError("completed training/evaluation inheritance changed")
        pending = [cell for cell in cells if cell_key(cell) not in retained]
        if [cell_key(cell) for cell in pending] != plan["unstarted_keys"]:
            raise ValueError("unstarted cells differ from the audited adoption plan")
        for cell in pending:
            old = Path(manifest["output_root"]) / "cells" / cell_key(cell)
            if list(old.glob("training/attempt_*")) or list(old.glob("development/seed_*/attempt_*")):
                raise ValueError("a supposedly unstarted cell already has an attempt")
        root = Path(root).expanduser().absolute()
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        for path in ("sessions", "evaluations", "controllers"):
            (root / path).mkdir()
        shutil.copyfile(plan["parent_manifest"]["path"], root / "parent_manifest.json")
        shutil.copyfile(plan["inherited_summary"]["path"], root / "inherited_summary.json")
        shutil.copyfile(plan_path, root / "adoption_plan.json")
        package = Path(plan["learner_root"]) / "src/transformer_rl"
        files = {str(p.relative_to(package)): file_pin(p)["sha256"]
                 for p in package.rglob("*.py") if "__pycache__" not in p.parts}
        encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        spec = {"format": FORMAT, "name": name, "authorized_boot": plan["authorized_boot"],
                "created_at": timestamp(), "python": plan["python"], "device": manifest["device"],
                "runtime": plan["runtime"], "environment": plan["environment"], "locks": plan["locks"],
                "learner_root": plan["learner_root"], "learner_identity": {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()},
                "verifier": plan["verifier"], "verifier_dependencies": plan.get("verifier_dependencies", []),
                "blocked": plan["blocked"],
                "resource_gate": plan.get("resource_gate"),
                "cells": cells, "pending_keys": [cell_key(cell) for cell in pending], "development_seeds": manifest["development_seeds"],
                "evaluation_protocol": manifest["inputs"]["protocol"]["evaluation"],
                "parent_manifest": file_pin(root.resolve() / "parent_manifest.json"),
                "inherited_summary": file_pin(root.resolve() / "inherited_summary.json"),
                "adoption_plan": file_pin(root.resolve() / "adoption_plan.json"),
                "full_research_complete": False, "formal_selection": False, "budget_refund": False}
        write_json(root / "queue.json", spec)
        write_json(root / "queue.pin.json", file_pin(root.resolve() / "queue.json"))
        results = {key: {"status": row["status"], "origin": "inherited_without_raw_reread",
                        "training_updates": row["training"].get("completed_updates", 0),
                        "development": {seed: value["status"] if value else None for seed, value in row["development"].items()}}
                   for key, row in inherited_results.items()}
        for key, blocked in plan["blocked"].items():
            results[key] = {"status": "interrupted_no_checkpoint", "origin": "retained_not_restarted", **blocked}
        write_json(root / "results.json", results)
        write_json(root / "state.json", {"phase": "ready", "active_cell": None, "updated_at": timestamp()})
        registry_path = Path.home() / ".config/transformer-rl/managed.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with command_lock(root):
            registry = read_json(registry_path, {})
            if name in registry and registry[name] != str(root.resolve()):
                raise ValueError("managed queue name already points to another directory")
            registry[name] = str(root.resolve())
            write_json(registry_path, registry, replace=True)
        return cls(root)

    def status(self):
        state = read_json(self.root / "state.json", {})
        results = read_json(self.root / "results.json", {})
        receipt = read_json(self.root / "controller.json")
        active = False
        changed = boot_id() != self.spec["authorized_boot"]
        if receipt and not changed and not read_json(Path(receipt["directory"]) / "exit.json"):
            active = process_identity(receipt["identity"]["pid"]) == receipt["identity"] or _lease_is_held(self.root)
        phase = state.get("phase", "ready")
        if changed:
            phase = "interrupted"
        elif not active and phase not in ("ready", "stopped", "incomplete", "failed"):
            phase = "interrupted"
        if receipt and not changed and not active and _lease_is_held(self.root):
            phase, active = "closing", True
        live = state.get("worker", {})
        request = read_json(self.root / "request.json", {"operation": "resume", "sequence": -1})
        counts = {"completed": sum(row["status"] == "completed" for row in results.values()),
                  "retained_incomplete": sum(row["status"] in ("incomplete", "interrupted_no_checkpoint") for row in results.values()),
                  "not_started": sum(key not in results for key in self.spec["pending_keys"]), "total": len(self.spec["cells"])}
        return {"name": self.spec["name"], "root": str(self.root), "phase": phase, "active": active,
                "architecture": "研究队列", "seed": live.get("seed", "—"), "device": self.spec["device"],
                "update": live.get("update", 0), "target_update": live.get("target_update", 1200),
                "queue_counts": counts, "active_cell": state.get("active_cell"), "stage": state.get("stage"),
                "evaluation_seed": state.get("evaluation_seed"), "worker": live,
                "checkpoint_update": live.get("checkpoint_update"), "requested": request["operation"],
                "request_sequence": request["sequence"], "ack_sequence": state.get("ack_sequence", -1),
                "reward_mean": live.get("reward_mean"), "learning_rate": live.get("learning_rate"),
                "resources": state.get("resources"),
                "session_active_elapsed_s": live.get("session_active_elapsed_s", 0),
                "paused_elapsed_s": live.get("paused_elapsed_s", 0), "boot_changed": changed,
                "error": state.get("error"), "log": live.get("log") or (receipt.get("log") if receipt else None)}

    def jobs(self):
        results = read_json(self.root / "results.json", {})
        return [{"cell": cell_key(cell), **results.get(cell_key(cell), {"status": "not_started"})}
                for cell in self.spec["cells"]]

    def start(self, *, paused=False, accept_new_boot=False):
        with command_lock(self.root):
            if self.status()["active"]:
                return self.status()
            if boot_id() != self.spec["authorized_boot"]:
                raise ValueError("this research queue requires a new audited adoption for a new boot")
            for name in ("parent_manifest", "inherited_summary", "adoption_plan", "verifier"):
                verify_pin(self.spec[name])
            for pin in self.spec["verifier_dependencies"]:
                verify_pin(pin)
            package = Path(self.spec["learner_root"]) / "src/transformer_rl"
            actual = {str(path.relative_to(package)): file_pin(path)["sha256"]
                      for path in package.rglob("*.py") if "__pycache__" not in path.parts}
            if actual != self.spec["learner_identity"]["files"]:
                raise ValueError("the authenticated controlled learner changed")
            descriptor = _lease(self.root)
            try:
                directory = self.root / "controllers" / uuid.uuid4().hex
                directory.mkdir()
                write_json(self.root / "request.json", {"operation": "pause" if paused else "resume", "sequence": 0}, replace=True)
                argv = [self.spec["python"], "-m", "transformer_rl.training_queue", str(self.root), str(descriptor), str(directory)]
                environment = {**os.environ, **self.spec["environment"]}
                environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
                with (directory / "controller.log").open("xb") as log:
                    process = subprocess.Popen(argv, env=environment, cwd=self.root, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(descriptor,))
                identity = process_identity(process.pid)
                if not identity:
                    raise ValueError("queue controller exited before identity binding")
                write_json(self.root / "controller.json", {"directory": str(directory), "identity": identity,
                           "log": str(directory / "controller.log")}, replace=True)
                state = read_json(self.root / "state.json")
                state.update(phase="starting", updated_at=timestamp())
                write_json(self.root / "state.json", state, replace=True)
            finally:
                os.close(descriptor)
        return self.status()

    def command(self, operation, *, wait_seconds=0.):
        if operation not in ("pause", "resume", "stop"):
            raise ValueError("unknown research operation")
        with command_lock(self.root):
            status = self.status()
            if not status["active"]:
                raise ValueError("queue controller is not live; use start/resume after inspection")
            if operation == "stop" and status.get("stage") == "training" and status["update"] == 0:
                raise ValueError("the frozen research protocol rejects zero-update endpoints; keep paused or stop after the first update")
            request = read_json(self.root / "request.json")
            if request["operation"] == "stop" and operation != "stop":
                raise ValueError("save-and-exit is already pending")
            if request["operation"] != operation:
                request = {"operation": operation, "sequence": request["sequence"] + 1}
                write_json(self.root / "request.json", request, replace=True)
        deadline = time.monotonic() + wait_seconds
        while True:
            status = self.status()
            if not status["active"] or operation != "stop" and status["ack_sequence"] >= request["sequence"] or time.monotonic() >= deadline:
                return status
            time.sleep(.1)

    def publish(self, phase=None, **values):
        state = read_json(self.root / "state.json", {})
        if phase is not None:
            state["phase"] = phase
        state.update(values, updated_at=timestamp())
        write_json(self.root / "state.json", state, replace=True)

    def _boundary(self):
        request = read_json(self.root / "request.json")
        while request["operation"] == "pause":
            self.publish("paused", ack_sequence=request["sequence"], worker={})
            time.sleep(.2)
            request = read_json(self.root / "request.json")
        if request["operation"] == "stop":
            self.publish("stopped", ack_sequence=request["sequence"])
            return False
        self.publish(ack_sequence=request["sequence"])
        return True

    def _record_exit(self, process, directory, identity):
        code = process.wait()
        write_json(directory / "exit.json", {"identity": identity, "returncode": code,
                   "boot_id": self.spec["authorized_boot"], "at": timestamp(), "origin": "actual_parent_waitpid"})
        return code

    def _resources_ready(self):
        gate = self.spec.get("resource_gate")
        if not gate:
            return True
        memory = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.split(":")[0] in ("MemAvailable", "SwapFree")}
        result = subprocess.run([gate["nvidia_smi"], "-i", gate["gpu_uuid"],
            "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        if result.returncode or len(result.stdout.strip().splitlines()) != 1:
            raise ValueError("the fixed GPU resource query failed")
        gpu, free = [part.strip() for part in result.stdout.strip().split(",")]
        disk = os.statvfs(self.root)
        self.resource_observation = {"at": timestamp(), **memory,
            "gpu_free_mib": float(free), "minimum_gpu_free_mib": gate["minimum_gpu_free_mib"],
            "minimum_ram_available_bytes": gate["minimum_ram_available_bytes"],
            "disk_available_bytes": disk.f_bavail * disk.f_frsize,
            "minimum_disk_available_bytes": gate["minimum_disk_available_bytes"]}
        return (gpu == gate["gpu_uuid"] and float(free) >= gate["minimum_gpu_free_mib"]
                and memory["MemAvailable"] >= gate["minimum_ram_available_bytes"]
                and disk.f_bavail * disk.f_frsize >= gate["minimum_disk_available_bytes"])

    def _wait_training(self, session, descriptors):
        from .training_queue_validation import validate_training
        state = session.status()
        if state["phase"] != "completed":
            if not self._boundary():
                return None
            session.start(inherited_fds=descriptors)
            process = session.submitted_process
            attempt = session.attempt()
            identity = read_json(attempt / "launch.json")
            while process.poll() is None:
                request = read_json(self.root / "request.json")
                child = session.status()
                if request["operation"] != child["requested"]:
                    session.command(request["operation"])
                child = session.status()
                acknowledged = child.get("ack_sequence", -1) >= child.get("request_sequence", 0)
                self.publish(child["phase"], worker=child,
                             ack_sequence=request["sequence"] if acknowledged and request["operation"] != "stop" else -1)
                time.sleep(.2)
            self._record_exit(process, attempt, identity)
        evidence = None
        for attempt in sorted((session.root / "attempts").iterdir()):
            evidence = validate_training(self, session, attempt, evidence)
        state = session.status()
        if state["phase"] == "stopped":
            self.publish("stopped", worker=state, ack_sequence=read_json(self.root / "request.json")["sequence"])
            return None
        if state["phase"] != "completed" or evidence["final_update"] != 1200:
            raise ValueError("training did not satisfy the original full-cell endpoint")
        return evidence

    def _evaluation(self, job, checkpoint, seed, descriptors):
        from .training_queue_validation import load_helpers, validate_evaluation
        directory = self.root / "evaluations" / cell_key(job) / f"seed_{seed}"
        if (directory / "research_validation.json").exists():
            return read_json(directory / "research_validation.json")
        if directory.exists():
            raise ValueError("unsealed evaluation retained; retry/reselection is refused")
        manifest = self.parent_manifest
        module, _, _ = load_helpers(self.spec["verifier"], self.spec["verifier_dependencies"])
        configs, _ = module.evaluation_inputs(manifest, job, checkpoint)
        prepared = Path(manifest["inputs"]["root"])
        for path in configs.values():
            if file_pin(path)["sha256"] != manifest["inputs"]["files"][str(path.relative_to(prepared))]:
                raise ValueError("original evaluation configuration changed")
        protocol = self.spec["evaluation_protocol"]
        directory.mkdir(parents=True)
        (directory / "control").mkdir()
        lease = _lease(directory)
        try:
            token = uuid.uuid4().hex
            write_json(directory / "request.json", {"protocol": "transformer_rl.training_control.v1", "token": token,
                "boot_id": self.spec["authorized_boot"], "root": str(directory), "attempt": str(directory), "lease_fd": lease,
                "inherited_locks": [{"fd": fd, "device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino}
                                    for fd in descriptors],
                "start_update": 0, "target_update": protocol["steps"], "prior_active_elapsed_s": 0, "prior_transitions": 0})
            write_json(directory / "control/request.json", {"token": token, "sequence": 0, "operation": "resume"})
            argv = [self.spec["python"], "-m", "transformer_rl.frame_process", "evaluate-suite", "--checkpoint", checkpoint["path"],
                "--configs", *(str(p) for p in configs.values()), "--outputs", *(str(directory / f"{case}.json") for case in configs),
                "--steps", str(protocol["steps"]), "--seed", str(seed), "--device", self.spec["device"], "--control-output", str(directory / "control.json"),
                "--trace-output", str(directory / "trace.npz"), "--settle-steps", str(protocol["settle_steps"]), "--min-steady-samples", str(protocol["min_steady_samples"]),
                "--trace-replicas", "2", "--control-dir", str(directory / "control")]
            environment = {**os.environ, **self.spec["environment"], "PYTHONPATH": str(Path(self.spec["learner_root"]) / "src")}
            with (directory / "worker.log").open("xb") as log:
                process = subprocess.Popen(argv, env=environment, cwd=self.root, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease, *descriptors))
            identity = process_identity(process.pid)
            write_json(directory / "launch.json", identity)
            while process.poll() is None:
                request = read_json(self.root / "request.json")
                current = read_json(directory / "control/request.json")
                if current["operation"] != request["operation"]:
                    current = {"token": token, "sequence": current["sequence"] + 1, "operation": request["operation"]}
                    write_json(directory / "control/request.json", current, replace=True)
                child = read_json(directory / "control/status.json", {})
                self.publish(child.get("phase", "evaluating"), worker={**child, "seed": job["training_seed"], "log": str(directory / "worker.log")},
                             ack_sequence=request["sequence"] if child.get("ack_sequence", -1) >= current["sequence"] and request["operation"] != "stop" else -1)
                time.sleep(.2)
            if self._record_exit(process, directory, identity) != 0:
                raise ValueError("evaluation failed; the original seed/window will not be silently rerun")
            return validate_evaluation(self, job, checkpoint, seed, directory)
        finally:
            os.close(lease)

    def run(self, descriptor, controller_directory):
        resources = []
        try:
            if boot_id() != self.spec["authorized_boot"]:
                raise ValueError("queue boot changed before restoration")
            inherited, expected = os.fstat(descriptor), (self.root / "run.lock").stat()
            if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("controller did not inherit its queue lease")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for item in self.spec["locks"]:
                path = Path(item["path"])
                info = path.lstat()
                if path.is_symlink() or (info.st_dev, info.st_ino, info.st_uid) != (item["device"], item["inode"], item["uid"]):
                    raise ValueError("original resource-lock identity changed")
                stream = path.open("r+")
                resources.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors = (descriptor, *(stream.fileno() for stream in resources))
            while not self._resources_ready():
                self.publish("waiting_resources", worker={}, resources=self.resource_observation)
                if not self._boundary():
                    return 0
                time.sleep(10)
            results = read_json(self.root / "results.json")
            for job in self.spec["cells"]:
                key = cell_key(job)
                if key not in self.spec["pending_keys"] or results.get(key, {}).get("status") == "completed":
                    continue
                if not self._boundary():
                    return 0
                self.publish("training", active_cell=key, stage="training", evaluation_seed=None, worker={})
                session_root = self.root / "sessions" / key
                if not session_root.exists():
                    configuration = Path(self.parent_manifest["inputs"]["root"]) / job["training_config"]["path"]
                    if file_pin(configuration)["sha256"] != job["training_config"]["sha256"]:
                        raise ValueError("original training configuration changed")
                    session = TrainingSession.create(session_root, configuration, env_factory="transformer_rl.chassis_adapter:make_env",
                        updates=job["updates"], python=self.spec["python"], device=self.spec["device"], seed=job["training_seed"], rollout_steps=job["rollout_steps"],
                        checkpoint_interval=1200, max_seconds=604800., tensorboard=True,
                        learner_source=Path(self.spec["learner_root"]) / "src/transformer_rl",
                        expected_initial_model_sha256=job["initial_model_sha256"], name=key)
                else:
                    session = TrainingSession(session_root)
                results.setdefault(key, {"status": "training", "development": {str(seed): None for seed in self.spec["development_seeds"]}})
                write_json(self.root / "results.json", results, replace=True)
                evidence = self._wait_training(session, descriptors)
                if evidence is None:
                    return 0
                row = results[key]
                row.update(status="evaluating", training_updates=1200, checkpoint=evidence["checkpoint"])
                write_json(self.root / "results.json", results, replace=True)
                for seed in self.spec["development_seeds"]:
                    if row["development"][str(seed)] == "completed":
                        continue
                    if not self._boundary():
                        return 0
                    self.publish("evaluating", stage="evaluating", evaluation_seed=seed)
                    self._evaluation(job, evidence["checkpoint"], seed, descriptors)
                    row["development"][str(seed)] = "completed"
                    write_json(self.root / "results.json", results, replace=True)
                row["status"] = "completed"
                write_json(self.root / "results.json", results, replace=True)
            self.publish("incomplete", active_cell=None, worker={}, stage=None,
                         reason="historical incomplete and no-checkpoint obligations retained; full research remains incomplete")
            return 0
        except BaseException as error:
            self.publish("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for stream in reversed(resources):
                stream.close()
            write_json(Path(controller_directory) / "exit.json", {"at": timestamp(), "boot_id": self.spec["authorized_boot"]})


if __name__ == "__main__":
    queue = TrainingQueue(sys.argv[1])
    raise SystemExit(queue.run(int(sys.argv[2]), Path(sys.argv[3])))

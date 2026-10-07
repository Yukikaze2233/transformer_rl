"""Opt-in learning-state continuation with an isolated retention sampler.

This component owns a single branch, not teacher qualification or a campaign.
Episodes and fixed histories reset on every open; simulator state is not saved.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import platform
import re

import numpy as np
import torch

from .experiments import source_identity
from .frame_checkpoint import load_frame_checkpoint, restore_rng, save_frame_checkpoint
from .frame_config import FrameTrainConfig, digest
from .frame_training import FrameActorCritic, FrameCollector
from .frame_workflow import _model_state_sha256, _provenance, _seed
from .ppo import PPOTrainer
from .private_retention import PrivateAnchorRegularizer


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _file_sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def _session_arguments(config, env_factory, env_reference, rollout_steps,
                       training_seed, retention_seed, evaluation_seeds, device):
    if not isinstance(config, FrameTrainConfig) or not callable(env_factory):
        raise TypeError("continuation requires a parsed config and environment factory")
    if not isinstance(env_reference, str) or not env_reference:
        raise ValueError("environment reference is required")
    _integer(rollout_steps, "rollout_steps", minimum=1)
    for name, seed in (("training_seed", training_seed), ("retention_seed", retention_seed)):
        _integer(seed, name)
        if seed >= 2**32:
            raise ValueError(f"{name} must be an unsigned 32-bit integer")
    if (not isinstance(evaluation_seeds, (tuple, list))
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in evaluation_seeds)
            or retention_seed in (training_seed, *evaluation_seeds)):
        raise ValueError("retention seed must be independent of training and evaluation seeds")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda") or (target_device.type == "cuda" and target_device.index is None):
        raise ValueError("continuation device must be CPU or CUDA with an explicit index")
    return target_device


class FrameContinuation:
    """Continue complete actor/critic/Adam/RNG state and explicit consumed clocks.

    Start a guarded fresh learner, branch from a complete learning checkpoint,
    or resume exact state. A stage environment change requires explicit resume
    and environment_transition; optimizer and random streams are never reset.
    A failed PPO update invalidates this session: reload a sealed checkpoint
    rather than publishing a partially mutated optimizer as a complete update.
    """

    @classmethod
    def start(cls, config, env_factory, env_reference, *, rollout_steps,
              training_seed, retention_seed, expected_initial_model_sha256,
              evaluation_seeds=(), device="cpu"):
        """Start fresh learning, guarded on CPU before any environment or CUDA.

        The required tensor-content SHA binds seeded actor and critic weights.
        This entry point has no teacher or anchors: retention is exactly zero,
        with an independent private stream ready for complete-state saving.
        """
        target_device = _session_arguments(config, env_factory, env_reference,
            rollout_steps, training_seed, retention_seed, evaluation_seeds, device)
        # The parsed dataclass contains mutable mappings. Revalidate and copy
        # them before trusting a caller-supplied config or constructing a model.
        config = FrameTrainConfig.from_dict(config.to_dict())
        if (type(expected_initial_model_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_initial_model_sha256) is None):
            raise ValueError("expected_initial_model_sha256 requires a lowercase 64-hex SHA256")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(training_seed)
            with torch.device("cpu"):
                model = FrameActorCritic(config.model)
            actual = _model_state_sha256(model)
            post_construction_rng = torch.get_rng_state().clone()
        if actual != expected_initial_model_sha256:
            raise ValueError("initial model SHA mismatch before training: "
                             f"expected {expected_initial_model_sha256}, actual {actual}")
        # Adam configuration and the private stream are checked on CPU too.
        trainer = PPOTrainer(model, config.ppo)
        sampler = PrivateAnchorRegularizer(model.actor, config, (), 0., seed=retention_seed)
        instance = cls()
        instance.config, instance.model, instance.trainer = config, model, trainer
        instance.sampler = sampler
        instance.update = instance.initial_update = 0
        instance.prior_transitions = instance.prior_consumed_updates = 0
        instance.rollout_steps, instance.training_seed = rollout_steps, training_seed
        instance.attempted_updates = instance.discarded_transitions = 0
        instance.env = instance.collector = None
        instance.closed = instance.invalid = False
        instance.metadata = {"environment_factory": env_reference, "seed": training_seed,
            "evaluation_seeds": list(evaluation_seeds), "source": source_identity(),
            "continuation_device": str(target_device), "anchors": [], "retention_coef": 0.,
            "initial_model_sha256": actual, "initial_model_hash_format": "sorted_named_tensor_contents_v1",
            "initialization_guard": {"expected_sha256": expected_initial_model_sha256,
                                     "actual_sha256": actual, "verified": True},
            "initial_rng": {"algorithm": "seeded_global_learning_after_environment_reset_v1",
                            "seed": training_seed, "cpu": "seed_then_model_construction",
                            "python_numpy": "seed_after_reset",
                            "cuda": "seed_after_environment_device_and_reset"},
            "episode_state_restored": False, "history_reset": "repeat_first",
            "continuation_parent": None,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "cuda": torch.version.cuda,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
        try:
            # Construct the same guarded initial model as train_frame_policy.
            # Reusing it must retain the CPU random prefix consumed by the model.
            _seed(training_seed)
            torch.set_rng_state(post_construction_rng)
            instance.env = env_factory(model_config=config.model, environment_config=config.environment,
                                       device=target_device)
            instance.metadata["environment_provenance"] = _provenance(instance.env, config.control)
            model.to(target_device).eval()
            instance.trainer = PPOTrainer(model, config.ppo)
            instance.collector = FrameCollector(instance.env, model, config.ppo, config.control["action_bounds"])
            instance._set_progress()
            instance.collector.reset(seed=training_seed)
            # Environment construction/reset can consume any global stream.
            # CUDA exists only after the simulator application/device is ready;
            # seed it here rather than silently restoring an empty CPU-only RNG
            # capture. CPU learning begins after the seeded model prefix.
            _seed(training_seed)
            torch.set_rng_state(post_construction_rng)
            return instance
        except BaseException as error:
            instance.invalid = True
            try:
                instance.close()
            except BaseException as cleanup:
                error.add_note(f"continuation startup cleanup failed: {type(cleanup).__name__}: {cleanup}")
            raise

    @classmethod
    def open(cls, config, env_factory, env_reference, checkpoint, *,
             checkpoint_sha256, parent_update, cumulative_transitions,
             consumed_updates, rollout_steps, training_seed, retention_seed,
             anchors=(), retention_coef=0., retention_batch_size=256,
             evaluation_seeds=(), device="cpu", resume=False, environment_transition=False):
        target_device = _session_arguments(config, env_factory, env_reference,
            rollout_steps, training_seed, retention_seed, evaluation_seeds, device)
        config = FrameTrainConfig.from_dict(config.to_dict())
        if type(resume) is not bool:
            raise ValueError("environment reference and explicit resume mode are required")
        if type(environment_transition) is not bool or environment_transition and not resume:
            raise ValueError("environment_transition must be a boolean and requires resume=True")
        for name, value in (("parent_update", parent_update),
                            ("cumulative_transitions", cumulative_transitions),
                            ("consumed_updates", consumed_updates)):
            _integer(value, name)
        if (type(checkpoint_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
                or _file_sha(checkpoint) != checkpoint_sha256):
            raise ValueError("parent checkpoint SHA256 mismatch")
        with torch.device("cpu"):
            model, trainer, parent_config, update, metadata, rng = load_frame_checkpoint(checkpoint)
        if _file_sha(checkpoint) != checkpoint_sha256:
            raise ValueError("parent checkpoint changed during preflight")
        if (parent_config.model != config.model or parent_config.ppo != config.ppo
                or parent_config.control != config.control):
            raise ValueError("continuation requires the same model, PPO recipe and control")
        if metadata.get("environment_factory") != env_reference or metadata.get("seed") != training_seed:
            raise ValueError("parent environment factory or training seed differs")
        if (update != parent_update or consumed_updates < update
                or type(metadata.get("collected_transitions")) is not int
                or metadata["collected_transitions"] != cumulative_transitions):
            raise ValueError("parent update or transition clock differs from the audited clock")
        parent_state = metadata.get("continuation")
        if parent_state is not None and not resume:
            raise ValueError("a continuation checkpoint requires resume=True; sampler reset is not implicit")
        if parent_state is not None or resume:
            if (not isinstance(parent_state, dict)
                    or set(parent_state) != {"format", "schema_version", "clock", "retention"}
                    or parent_state["format"] != "transformer_rl.frame_continuation"
                    or type(parent_state["schema_version"]) is not int or parent_state["schema_version"] != 1):
                raise ValueError("continuation state has an unsupported format")
            expected_clock = {"consumed_updates": consumed_updates,
                              "collected_transitions": cumulative_transitions,
                              "rollout_steps": rollout_steps}
            if (not isinstance(parent_state["clock"], dict)
                    or any(type(value) is not int for value in parent_state["clock"].values())
                    or parent_state["clock"] != expected_clock):
                raise ValueError("continuation consumed clock or rollout length differs")
        if resume and not environment_transition and parent_config.to_dict() != config.to_dict():
            raise ValueError("resume requires the exact continuation configuration")
        if resume and "evaluation_seeds" in metadata and metadata["evaluation_seeds"] != list(evaluation_seeds):
            raise ValueError("resume evaluation seeds differ from checkpoint")
        current_source = source_identity()
        if resume and (metadata.get("source") != current_source
                       or metadata.get("continuation_device") != str(target_device)):
            raise ValueError("resume requires the exact producer source and device")
        if target_device.type == "cuda" and not rng["cuda"]:
            raise ValueError("CUDA continuation requires saved CUDA learning RNG state")
        sampler = PrivateAnchorRegularizer(model.actor, config, anchors, retention_coef,
                                           seed=retention_seed, batch_size=retention_batch_size)
        if resume:
            sampler.load_state_dict(parent_state["retention"])
        instance = cls()
        instance.config, instance.model, instance.trainer = config, model, trainer
        instance.sampler = sampler
        instance.update, instance.initial_update = update, update
        instance.prior_transitions, instance.prior_consumed_updates = cumulative_transitions, consumed_updates
        instance.rollout_steps, instance.training_seed = rollout_steps, training_seed
        instance.attempted_updates, instance.discarded_transitions = 0, 0
        instance.env = instance.collector = None
        instance.closed = instance.invalid = False
        instance.metadata = deepcopy(metadata)
        instance.metadata["parent_source"] = metadata.get("source")
        instance.metadata["source"] = current_source
        instance.metadata["continuation_device"] = str(target_device)
        instance.metadata["evaluation_seeds"] = list(evaluation_seeds)
        # This field describes this open only. Earlier transitions remain
        # auditable through the sealed parent checkpoint chain, not as a stale
        # assertion that an ordinary exact resume changed environments again.
        instance.metadata.pop("stage_transition", None)
        if environment_transition:
            instance.metadata["stage_transition"] = {
                "environment_transition": True,
                "parent_environment_sha256": digest(parent_config.environment),
                "environment_sha256": digest(config.environment)}
        instance.metadata.update(anchors=sampler.identities, retention_coef=retention_coef,
            initial_model_sha256=_model_state_sha256(model),
            initial_model_hash_format="sorted_named_tensor_contents_v1", episode_state_restored=False,
            history_reset="repeat_first", continuation_parent={"path": str(Path(checkpoint).resolve()),
                "sha256": checkpoint_sha256, "update": update, "resume": resume})
        try:
            # The CPU learner and anchors are fully checked before constructing a
            # simulator or initializing CUDA. Reset-time task sampling sees the
            # consumed clock, then global learning RNG is restored last.
            _seed(training_seed)
            instance.env = env_factory(model_config=config.model, environment_config=config.environment,
                                       device=target_device)
            provenance = _provenance(instance.env, config.control)
            if metadata.get("environment_provenance", {}).get("identity") != provenance["identity"]:
                raise ValueError("continuation environment source/assets differ from checkpoint")
            instance.metadata["environment_provenance"] = provenance
            optimizer = trainer.optimizer.state_dict()
            model.to(device).eval()
            instance.trainer = PPOTrainer(model, config.ppo)
            instance.trainer.optimizer.load_state_dict(optimizer)
            instance.collector = FrameCollector(instance.env, model, config.ppo, config.control["action_bounds"])
            instance._set_progress()
            instance.collector.reset(seed=training_seed)
            restore_rng(rng)
            return instance
        except BaseException as error:
            instance.invalid = True
            try:
                instance.close()
            except BaseException as cleanup:
                error.add_note(f"continuation startup cleanup failed: {type(cleanup).__name__}: {cleanup}")
            raise

    @property
    def collected_transitions(self):
        return self.prior_transitions + (self.collector.total_transitions if self.collector is not None else 0)

    @property
    def consumed_updates(self):
        return self.prior_consumed_updates + self.attempted_updates

    def _set_progress(self):
        progress = getattr(self.env, "set_training_progress", None)
        if progress is not None:
            progress(self.consumed_updates, self.collected_transitions)

    def _require_live(self):
        if self.closed or self.invalid:
            raise RuntimeError("continuation is closed or invalid; reload a sealed checkpoint")

    def step(self, *, should_stop=None):
        """Collect a full rollout, then apply one PPO update; discard partial tails."""
        self._require_live()
        if should_stop is not None and not callable(should_stop):
            raise TypeError("should_stop must be callable")
        try:
            self._set_progress()
            batch = self.collector.collect(self.rollout_steps, should_stop=should_stop)
            expected = self.rollout_steps * self.collector.num_envs
            if batch is None or len(batch) != expected:
                self.discarded_transitions += len(batch) if batch is not None else 0
                return None
            self.attempted_updates += 1
            metrics = self.trainer.update(batch, diagnostics=True,
                regularizer=self.sampler if self.sampler.coefficient > 0 else None)
            self.update += 1
            return {"update": self.update, "batch_samples": len(batch), "optimization": metrics,
                    "collection": deepcopy(self.collector.last_metrics),
                    "consumed_updates": self.consumed_updates,
                    "cumulative_transitions": self.collected_transitions}
        except BaseException:
            self.invalid = True
            raise

    def save(self, path):
        """Publish complete learning and private sampling state at a clean boundary."""
        self._require_live()
        self.metadata["collected_transitions"] = self.collected_transitions
        self.metadata["continuation_segment"] = {
            "start_update": self.initial_update,
            "successful_updates": self.update - self.initial_update,
            "attempted_updates": self.attempted_updates,
            "fresh_transitions": self.collector.total_transitions,
            "discarded_transitions": self.discarded_transitions}
        self.metadata["continuation"] = {
            "format": "transformer_rl.frame_continuation", "schema_version": 1,
            "clock": {"consumed_updates": self.consumed_updates,
                      "collected_transitions": self.collected_transitions,
                      "rollout_steps": self.rollout_steps},
            "retention": self.sampler.state_dict()}
        return save_frame_checkpoint(path, self.model, self.trainer, self.config, self.update, self.metadata)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.env is not None:
            self.env.close()

    def __enter__(self):
        self._require_live()
        return self

    def __exit__(self, *_):
        self.close()

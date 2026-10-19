"""Opt-in anchor KL with an independent, checkpointable CPU sampling stream.

The caller must keep the explicit sampling seed separate from frozen training
and evaluation seeds. This component does not qualify teachers or run branches.
Repository frame policies are deterministic; their forward pass introduces no
random draws. The legacy regularizer and its global-RNG behavior are unchanged.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch

from .frame_config import digest


_FORMAT = "transformer_rl.private_anchor_regularizer"
_ALGORITHM = "uniform_file_then_uniform_endpoint_with_replacement_v1"
_FIELDS = {"format", "schema_version", "algorithm", "torch_version", "seed",
           "coefficient", "batch_size", "identities", "pool_sizes", "draw_count",
           "endpoint_draw_count", "generator", "sha256"}
_ANCHOR_FIELDS = {"format", "schema_version", "control_sha256", "policy_config",
                  "teacher_checkpoint_sha256", "frames", "mean", "std"}


def _file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _uint32(value):
    return type(value) is int and 0 <= value < 2**32


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


class PrivateAnchorRegularizer:
    """The legacy Gaussian KL and sampling distribution, with private CPU RNG.

    ``draw_count`` counts sampled minibatches, each consisting of one file draw
    and ``min(batch_size, file_rows)`` endpoint draws with replacement. Loading
    replays only these private draws to verify the counter and generator state.
    This costs O(draw_count) time and O(batch_size + generator_state) memory;
    ``max_replay_calls`` is a caller-controlled work limit, never read from state.
    """

    def __init__(self, actor, config, paths, coefficient, *, seed, batch_size=256):
        try:
            valid_coefficient = type(coefficient) in (int, float) and math.isfinite(coefficient)
        except OverflowError:
            valid_coefficient = False
        if (not valid_coefficient or coefficient < 0
                or type(batch_size) is not int or batch_size < 1):
            raise ValueError("retention requires nonnegative coefficient and positive batch_size")
        if not _uint32(seed):
            raise ValueError("retention seed must be a uint32 integer")
        paths = list(paths)
        if coefficient > 0 and not paths:
            raise ValueError("positive retention coefficient requires anchor paths")
        try:
            next(actor.parameters())
        except StopIteration as error:
            raise ValueError("retention requires an actor parameter") from error
        self.actor, self.coefficient, self.batch_size = actor, float(coefficient), batch_size
        self.seed = seed
        self._pools, self._identities = [], []
        for path in paths:
            path = Path(path).resolve()
            identity = _file_sha256(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if _file_sha256(path) != identity:
                raise ValueError("anchor file changed while loading")
            if (not isinstance(payload, dict) or set(payload) != _ANCHOR_FIELDS
                    or payload["format"] != "transformer_rl.behavior_anchors"
                    or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                    or payload["control_sha256"] != digest(config.control)
                    or payload["policy_config"] != config.model.policy.to_dict()
                    or not _sha256(payload["teacher_checkpoint_sha256"])):
                raise ValueError("anchor observation, timing, teacher or architecture contract mismatch")
            frames, mean, std = (payload[key] for key in ("frames", "mean", "std"))
            if (any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.device.type != "cpu" or value.layout != torch.strided
                    or value.requires_grad or not torch.isfinite(value).all()
                    for value in (frames, mean, std))
                    or frames.ndim != 3
                    or frames.shape[1:] != (config.model.history_length, config.model.frame_dim)
                    or len(frames) < 1 or mean.shape != (len(frames), config.model.action_dim)
                    or std.shape != mean.shape or not (std > 0).all()):
                raise ValueError("invalid behavior anchor tensors")
            self._pools.append((frames, mean, std))
            self._identities.append({"path": str(path), "sha256": identity})
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._draw_count = self._endpoint_draw_count = 0

    @property
    def identities(self):
        return deepcopy(self._identities)

    def _indices(self, generator):
        pool_index = torch.randint(len(self._pools), (), generator=generator, device="cpu").item()
        rows = len(self._pools[pool_index][0])
        indices = torch.randint(rows, (min(self.batch_size, rows),),
                                generator=generator, device="cpu")
        return pool_index, indices

    def __call__(self):
        parameter = next(self.actor.parameters())
        if self.coefficient == 0:
            return parameter.new_zeros(())
        pool_index, indices = self._indices(self._generator)
        self._draw_count += 1
        self._endpoint_draw_count += indices.numel()
        frames, mean, std = (value[indices].to(parameter.device) for value in self._pools[pool_index])
        predicted = self.actor.policy(frames)
        current_std = self.actor.log_std.exp().expand_as(predicted)
        kl = (current_std.log() - std.log()
              + 0.5 * ((std / current_std).square()
                       + ((mean - predicted) / current_std).square() - 1.)).sum(-1)
        return self.coefficient * kl.mean()

    def state_dict(self):
        raw = bytes(self._generator.get_state().tolist())
        value = {"format": _FORMAT, "schema_version": 1, "algorithm": _ALGORITHM,
                 "torch_version": str(torch.__version__), "seed": self.seed,
                 "coefficient": self.coefficient, "batch_size": self.batch_size,
                 "identities": self.identities,
                 "pool_sizes": [len(pool[0]) for pool in self._pools],
                 "draw_count": self._draw_count, "endpoint_draw_count": self._endpoint_draw_count,
                 "generator": {"encoding": "base64_uint8", "bytes": len(raw),
                               "sha256": hashlib.sha256(raw).hexdigest(),
                               "data": base64.b64encode(raw).decode("ascii")}}
        return {**value, "sha256": digest(value)}

    def load_state_dict(self, state, *, max_replay_calls=1_000_000):
        """Validate a JSON state completely before replacing the live stream.

The trusted caller may raise the replay work limit to a frozen branch budget.
No global RNG state is read, reset or restored during validation or replay.
"""
        if type(max_replay_calls) is not int or max_replay_calls < 0:
            raise ValueError("max_replay_calls must be a nonnegative integer")
        if not isinstance(state, dict) or set(state) != _FIELDS:
            raise ValueError("invalid private retention state fields")
        unsigned = {key: value for key, value in state.items() if key != "sha256"}
        try:
            state_sha = digest(unsigned)
        except (TypeError, ValueError) as error:
            raise ValueError("private retention state must be finite JSON") from error
        if not _sha256(state["sha256"]) or state["sha256"] != state_sha:
            raise ValueError("private retention state checksum mismatch")
        expected = {"format": _FORMAT, "schema_version": 1, "algorithm": _ALGORITHM,
                    "torch_version": str(torch.__version__), "seed": self.seed,
                    "coefficient": self.coefficient, "batch_size": self.batch_size,
                    "identities": self.identities,
                    "pool_sizes": [len(pool[0]) for pool in self._pools]}
        if (not _uint32(state["seed"]) or type(state["schema_version"]) is not int
                or type(state["batch_size"]) is not int or type(state["coefficient"]) is not float
                or not isinstance(state["pool_sizes"], list)
                or any(type(size) is not int or size < 1 for size in state["pool_sizes"])
                or not isinstance(state["identities"], list)
                or any(not isinstance(identity, dict) or set(identity) != {"path", "sha256"}
                       or not isinstance(identity["path"], str) or not _sha256(identity["sha256"])
                       for identity in state["identities"])
                or any(state[key] != value for key, value in expected.items())):
            raise ValueError("private retention seed, algorithm, coefficient or anchor identity mismatch")
        count, endpoints = state["draw_count"], state["endpoint_draw_count"]
        if (type(count) is not int or type(endpoints) is not int or count < 0 or endpoints < 0
                or count > max_replay_calls or self.coefficient == 0 and (count or endpoints)):
            raise ValueError("invalid private retention draw count or replay work limit")
        sizes = [min(self.batch_size, len(pool[0])) for pool in self._pools]
        if (count == 0 and endpoints != 0
                or count > 0 and (not sizes or not count * min(sizes) <= endpoints <= count * max(sizes))):
            raise ValueError("private retention endpoint draw count mismatch")
        encoded = state["generator"]
        size = self._generator.get_state().numel()
        if (not isinstance(encoded, dict) or set(encoded) != {"encoding", "bytes", "sha256", "data"}
                or encoded["encoding"] != "base64_uint8" or type(encoded["bytes"]) is not int
                or encoded["bytes"] != size or not _sha256(encoded["sha256"])
                or not isinstance(encoded["data"], str) or len(encoded["data"]) != 4 * ((size + 2) // 3)):
            raise ValueError("invalid private retention generator encoding")
        try:
            raw = base64.b64decode(encoded["data"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("invalid private retention generator base64") from error
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != encoded["sha256"]:
            raise ValueError("private retention generator checksum mismatch")
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        candidate = torch.Generator(device="cpu")
        try:
            candidate.set_state(tensor)
        except RuntimeError as error:
            raise ValueError("invalid private retention generator state") from error
        if candidate.initial_seed() != self.seed:
            raise ValueError("private retention generator seed mismatch")
        replay = torch.Generator(device="cpu")
        replay.manual_seed(self.seed)
        replay_endpoints = 0
        for _ in range(count):
            _, indices = self._indices(replay)
            replay_endpoints += indices.numel()
        if replay_endpoints != endpoints or not torch.equal(replay.get_state(), tensor):
            raise ValueError("private retention generator state does not match draw count")
        for identity in self._identities:
            try:
                current_sha = _file_sha256(Path(identity["path"]))
            except OSError as error:
                raise ValueError("anchor file identity is unavailable") from error
            if current_sha != identity["sha256"]:
                raise ValueError("anchor file identity changed since loading")
        self._generator = candidate
        self._draw_count, self._endpoint_draw_count = count, endpoints

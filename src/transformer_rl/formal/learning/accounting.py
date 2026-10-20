"""Private durable reservations for the new learner, independent of weights.

A reservation is fsynced before collection. Missing observations after an
interruption remain reserved, never refunded or silently replayed. This file
does not authorize a worker, cross-boot launch, simulator or GPU allocation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from ...common.training_control import boot_id, read_json, write_json
from ..tasks.curriculum import STAGE_NAMES


@dataclass(frozen=True)
class LearningJournalSpec:
    run_id: str
    contract_sha256: str
    num_envs: int
    rollout_steps: int
    epochs: int
    minibatches: int
    target_rollouts: int = 30000

    def __post_init__(self):
        if (not isinstance(self.run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', self.run_id)
                or not isinstance(self.contract_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.contract_sha256)
                or any(type(v) is not int or v < 1 for v in
                       (self.num_envs, self.rollout_steps, self.epochs, self.minibatches, self.target_rollouts))
                or self.target_rollouts > 30000):
            raise ValueError('learning journal requires an explicit run, contract and positive nonrefundable budget')


class RecoveryConsumptionMismatch(RuntimeError):
    """Checkpoint and authoritative consumption differ; automatic resume refused."""


class LearningJournal:
    """Single writer and immutable hash-linked events; reads do not poll PIDs."""

    _FORMAT = 'private_learning_consumption'
    _ACTUAL = ('attempted_transitions', 'returned_transitions', 'ppo_updates', 'ppo_fresh_endpoints',
               'actor_optimizer_steps', 'actor_endpoint_uses', 'critic_optimizer_steps', 'critic_endpoint_uses')

    @classmethod
    def create(cls, directory, spec: LearningJournalSpec, *, authorized_boot):
        if not isinstance(spec, LearningJournalSpec) or authorized_boot != boot_id():
            raise ValueError('journal creation requires the caller-authorized actual boot and parsed specification')
        root = Path(directory).absolute()
        root.mkdir(parents=True, exist_ok=False)
        (root / 'events').mkdir()
        document = {'format': cls._FORMAT, 'spec': asdict(spec), 'created_boot': authorized_boot}
        write_json(root / 'identity.json', document)
        raw = (root / 'identity.json').read_bytes()
        write_json(root / 'identity.pin.json', {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        return cls(root, writable=True, authorized_boot=authorized_boot)

    def __init__(self, directory, *, writable=False, authorized_boot=None):
        self.directory = Path(directory).resolve(strict=True)
        self._fd, self._faulted = None, False
        pin = read_json(self.directory / 'identity.pin.json')
        document = read_json(self.directory / 'identity.json')
        raw = (self.directory / 'identity.json').read_bytes()
        if (not isinstance(pin, dict) or set(pin) != {'bytes', 'sha256'} or pin['bytes'] != len(raw)
                or pin['sha256'] != hashlib.sha256(raw).hexdigest()
                or not isinstance(document, dict) or set(document) != {'format', 'spec', 'created_boot'}
                or document['format'] != self._FORMAT):
            raise ValueError('learning journal identity is absent, changed or from another endpoint')
        self.spec = LearningJournalSpec(**document['spec'])
        self.identity_sha256 = pin['sha256']
        self.authorized_boot = authorized_boot
        if writable:
            if authorized_boot != boot_id():
                raise ValueError('journal writer needs the explicit actual boot binding')
            descriptor = os.open(self.directory / 'writer.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError('journal writer lease must be a regular file')
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = descriptor
            except BaseException:
                os.close(descriptor)
                raise
        self._state = {'rollouts_reserved': 0, 'reserved_transitions': 0,
                       'reserved_by_stage': [0] * len(STAGE_NAMES), **{key: 0 for key in self._ACTUAL},
                       'active_elapsed_s': 0., 'pending': None, 'failed': False}
        self._sequence, self._head = 0, None
        try:
            paths = sorted((self.directory / 'events').iterdir())
            if len(paths) > 3 * self.spec.target_rollouts + 10000:
                raise ValueError('journal event count exceeds the fixed run budget')
            for path in paths:
                if path.name.startswith('.'):
                    continue  # Incomplete writer temporaries are not commit markers.
                if path.name != f'{self._sequence:08d}.json':
                    raise ValueError('journal event sequence has a gap or an unexpected file')
                event = read_json(path)
                self._state = self._apply(self._state, event)
                data = path.read_bytes()
                self._head = {'sequence': self._sequence, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                self._sequence += 1
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _apply(self, state, event):
        if (not isinstance(event, dict) or set(event) != {'sequence', 'previous_sha256', 'boot', 'kind', 'payload', 'active_elapsed_s'}
                or type(event['sequence']) is not int or event['sequence'] != self._sequence
                or event['previous_sha256'] != (self._head['sha256'] if self._head else None)
                or not isinstance(event['boot'], str) or not re.fullmatch(r'[0-9a-f-]{36}', event['boot'])
                or type(event['active_elapsed_s']) not in (int, float) or not math.isfinite(event['active_elapsed_s'])
                or event['active_elapsed_s'] < state['active_elapsed_s']):
            raise ValueError('journal chain, clock or event fields differ')
        prepared = deepcopy(state)
        payload, kind = event['payload'], event['kind']
        if not isinstance(payload, dict):
            raise ValueError('journal event payload must be an object')
        if kind == 'reserve':
            if (set(payload) != {'index', 'stage'} or type(payload['index']) is not int
                    or payload['index'] != state['rollouts_reserved'] + 1 or type(payload['stage']) is not int
                    or not 0 <= payload['stage'] < len(STAGE_NAMES) or state['pending'] is not None or state['failed']
                    or payload['index'] > self.spec.target_rollouts):
                raise ValueError('journal cannot reserve a replayed, failed, pending or exhausted rollout')
            prepared['pending'] = dict(payload)
            prepared['rollouts_reserved'] += 1
            prepared['reserved_by_stage'][payload['stage']] += 1
            prepared['reserved_transitions'] += self.spec.num_envs * self.spec.rollout_steps
        elif kind == 'finish':
            if (set(payload) != {'index', 'actual', 'failed'} or state['pending'] is None
                    or type(payload['index']) is not int or payload['index'] != state['pending']['index']
                    or type(payload['failed']) is not bool
                    or not isinstance(payload['actual'], dict) or set(payload['actual']) != set(self._ACTUAL)
                    or any(type(v) is not int or v < state[k] for k, v in payload['actual'].items())):
                raise ValueError('journal finish needs its one pending reservation and monotonic actual counters')
            delta = {key: payload['actual'][key] - state[key] for key in self._ACTUAL}
            size, limit = self.spec.num_envs * self.spec.rollout_steps, self.spec.epochs * self.spec.minibatches
            if (not 0 <= delta['returned_transitions'] <= delta['attempted_transitions'] <= size
                    or delta['ppo_updates'] not in (0, 1)
                    or delta['ppo_fresh_endpoints'] != delta['ppo_updates'] * size
                    or delta['ppo_updates'] and delta['returned_transitions'] != size
                    or not payload['failed'] and delta['ppo_updates'] != 1
                    or any(not 0 <= delta[name + '_optimizer_steps'] <= limit
                           or not delta[name + '_optimizer_steps'] <= delta[name + '_endpoint_uses'] <= self.spec.epochs * size
                           for name in ('actor', 'critic'))
                    or not delta['ppo_updates'] and any(delta[key] for key in self._ACTUAL[4:])):
                raise ValueError('actual work exceeds its full-rollout reservation or contradicts PPO entry')
            prepared.update(payload['actual'], pending=None, failed=payload['failed'])
        elif kind == 'boundary':
            if payload or state['pending'] is not None or state['failed']:
                raise ValueError('checkpoint boundary requires a clean completed learning transaction')
        else:
            raise ValueError('unknown journal event kind')
        prepared['active_elapsed_s'] = float(event['active_elapsed_s'])
        return prepared

    def _append(self, kind, payload, active_elapsed_s):
        if self._fd is None or self._faulted or boot_id() != self.authorized_boot:
            raise RuntimeError('journal writer is absent, faulted or on a different boot')
        held, path = os.fstat(self._fd), (self.directory / 'writer.lock').stat(follow_symlinks=False)
        if (held.st_dev, held.st_ino) != (path.st_dev, path.st_ino):
            raise RuntimeError('journal writer lease changed')
        event = {'sequence': self._sequence, 'previous_sha256': self._head['sha256'] if self._head else None,
                 'boot': self.authorized_boot, 'kind': kind, 'payload': deepcopy(payload), 'active_elapsed_s': active_elapsed_s}
        prepared = self._apply(self._state, event)
        path = self.directory / 'events' / f'{self._sequence:08d}.json'
        try:
            write_json(path, event)
            data = path.read_bytes()
        except BaseException:
            self._faulted = True
            raise
        self._head = {'sequence': self._sequence, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        self._sequence += 1
        self._state = prepared
        return self.summary()

    def reserve(self, stage, *, active_elapsed_s):
        return self._append('reserve', {'index': self._state['rollouts_reserved'] + 1, 'stage': stage}, active_elapsed_s)

    def finish(self, actual, *, failed=False, active_elapsed_s):
        if self._state['pending'] is None:
            raise RuntimeError('there is no pending rollout to finish')
        return self._append('finish', {'index': self._state['pending']['index'], 'actual': actual, 'failed': failed}, active_elapsed_s)

    def record_boundary(self, *, active_elapsed_s):
        return self._append('boundary', {}, active_elapsed_s)

    def summary(self):
        return deepcopy({'identity_sha256': self.identity_sha256, 'head': self._head, **self._state})

    def checkpoint_boundary(self):
        summary = self.summary()
        if summary['pending'] is not None or summary['failed'] or self._faulted:
            raise RecoveryConsumptionMismatch('pending, failed or uncertain consumption is not a resumable boundary')
        return summary

    def require_exact_boundary(self, checkpoint_boundary):
        current = self.checkpoint_boundary()
        if checkpoint_boundary != current:
            raise RecoveryConsumptionMismatch('checkpoint is behind or differs from authoritative consumption; no refund or replay')

"""Private learner/update boundaries using the existing file control handshake.

This adapter does not launch a worker, admit a simulator, or register the new
checkpoint format with the production CLI. Its caller owns physical setup and
stage validation. A complete rollout is reserved before any environment step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat

from ._structured_checkpoint import LearningComponents, LearningRecipe, StructuredCheckpointStore
from .training_control import JSON_LIMIT, TrainingControl, boot_id, file_pin, read_json, write_json


class StageValidationRequired(RuntimeError):
    """The exposure boundary is intact; the independent evaluator must run."""


class StructuredLearningLoop:
    """One transaction per call, with cooperative control between updates.

    An update is never shortened to acknowledge a command. Pause retains the
    collector/environment; stop publishes a complete final learning endpoint.
    A failed transaction stays charged and cannot become a resumable endpoint.
    """

    def __init__(self, recipe: LearningRecipe, components: LearningComponents, collector,
                 journal, store: StructuredCheckpointStore, control: TrainingControl, stop):
        self.recipe, self.components, self.collector = recipe, components, collector
        self.journal, self.store, self.control, self.stop = journal, store, control, stop
        self._faulted, self._finished = False, False
        self._report = None
        self._last_checkpoint = None
        binding = control.binding
        boundary = journal.checkpoint_boundary()
        if (not isinstance(control, TrainingControl) or store.recipe != recipe or store.journal is not journal
                or collector.model is not components.model or collector.config != recipe.collection
                or collector.num_envs != recipe.num_envs or str(collector.device) != recipe.device
                or collector.attempted_vector_steps != components.attempted_vector_steps
                or collector.returned_vector_steps != components.returned_vector_steps
                or binding['start_update'] != components.trainer.update_index
                or binding['target_update'] != journal.spec.target_rollouts
                or binding['prior_transitions'] != boundary['returned_transitions']
                or type(binding['prior_active_elapsed_s']) not in (int, float)
                or not math.isfinite(binding['prior_active_elapsed_s'])
                or binding['prior_active_elapsed_s'] < boundary['active_elapsed_s']
                or binding['boot_id'] != journal.authorized_boot or binding['boot_id'] != boot_id()
                or not journal.directory.is_relative_to(Path(binding['root']))
                or store.directory != Path(binding['attempt']) / 'train' / 'checkpoints'):
            raise ValueError('private learner, collector, budget or managed attempt binding differs')
        components.validate_binding(recipe, journal)

    def _active_elapsed(self):
        return self.control.binding['prior_active_elapsed_s'] + self.control.elapsed()[0]

    def _publish(self, phase, **extra):
        state = self.journal.summary()
        self.control.publish(phase, update=self.components.trainer.update_index,
                             cumulative_transitions=state['returned_transitions'],
                             reserved_transitions=state['reserved_transitions'],
                             reserved_updates=state['rollouts_reserved'],
                             actor_optimizer_steps=state['actor_optimizer_steps'],
                             critic_optimizer_steps=state['critic_optimizer_steps'],
                             curriculum_stage=self.components.curriculum.stage,
                             curriculum_status=self.components.curriculum.status,
                             latest_checkpoint=self._last_checkpoint,
                             recipe_sha256=self.recipe.sha256, **extra)

    def _save(self, name, *, purpose, sequence):
        self.journal.record_boundary(active_elapsed_s=self._active_elapsed())
        receipt = self.store.save(name, self.components, metadata={
            'purpose': purpose, 'token': self.control.binding['token'], 'sequence': sequence,
            'boot': self.control.binding['boot_id'], 'physical_state_restored': False})
        path = self.store.directory / name
        marker = {'format': 'private_learning_latest', 'recipe_sha256': self.recipe.sha256,
                  'checkpoint': {'path': str(path), 'bytes': receipt['bytes'], 'sha256': receipt['sha256']},
                  'sidecar': file_pin(Path(str(path) + '.json')), 'update': self.components.trainer.update_index,
                  'purpose': purpose, 'sequence': sequence, 'physical_state_restored': False}
        # The marker is the final publication. A failed snapshot leaves the
        # previous marker untouched; old consumption still cannot be replayed.
        write_json(self.store.directory / 'latest.json', marker, replace=True)
        self._last_checkpoint = str(path)
        return path

    def _automatic_checkpoint(self):
        update = self.components.trainer.update_index
        if update % 25 == 0:
            sequence = self.control._request()['sequence']
            self._save(f'update_{update:05d}.pt', purpose='periodic_learning_resume', sequence=sequence)

    def _pause_checkpoint(self, sequence):
        return self._save(f'pause_{self.components.trainer.update_index:05d}_{sequence:08d}.pt',
                          purpose='same_process_pause', sequence=sequence)

    def _finish_stop(self):
        return self._finish('stopped', self.stop.reason)

    def _finish(self, status, reason):
        if self._report is not None:
            return self._report
        if status == 'completed' and self.components.trainer.update_index != self.journal.spec.target_rollouts:
            raise ValueError('training completion requires the entire original update budget')
        sequence = self.control._request()['sequence']
        purpose = 'cold_learning_resume' if status == 'stopped' else 'training_budget_complete'
        path = self._save('final.pt', purpose=purpose, sequence=sequence)
        self._report = {'status': status, 'checkpoint': str(path),
                        'final_update': self.components.trainer.update_index,
                        'cumulative_transitions': self.journal.summary()['returned_transitions'],
                        'stop_reason': reason}
        self._publish('stopping' if status == 'stopped' else 'checkpointing')
        self.control.finish(self._report)
        self._finished = True
        return self._report

    def _record_failure(self, error):
        self._faulted = True
        self.components.attempted_vector_steps = self.collector.attempted_vector_steps
        self.components.returned_vector_steps = self.collector.returned_vector_steps
        try:
            if self.journal.summary()['pending'] is not None:
                self.journal.finish(self.components.observed_consumption(self.recipe.num_envs),
                                    failed=True, active_elapsed_s=self._active_elapsed())
        except BaseException as accounting_error:
            error.add_note('Consumption remains uncertain: ' + repr(accounting_error))
        try:
            self.control.fail(error)
        except BaseException as control_error:
            error.add_note('Failure status publication also failed: ' + repr(control_error))

    def run_update(self):
        """Return metrics, or None after a committed cooperative stop.

        Exposure scheduling never invokes evaluation. The explicit older
        audited mode still leaves its validation boundary to the caller.
        """
        if self._faulted or self._finished:
            raise RuntimeError('private learner is faulted or already finished')
        self.components.validate_binding(self.recipe, self.journal)
        try:
            self._publish('between_updates')
            if not self.control.boundary(self.stop, self._pause_checkpoint):
                self._finish_stop()
                return None
            if self.components.curriculum.status == 'training_budget_complete':
                self._finish('completed', 'training_budget_complete')
                return None
            if self.components.curriculum.status != 'running':
                self._publish('validation_pending')
                raise StageValidationRequired('curriculum requires its independent validation before more learning')
            self.components.curriculum.prepare_rollout()
            self.journal.reserve(self.components.curriculum.stage, active_elapsed_s=self._active_elapsed())
            self.components.curriculum.charge_rollout()
            batch = self.collector.collect(self.recipe.rollout_steps)
            if len(batch) != self.recipe.num_envs * self.recipe.rollout_steps:
                raise ValueError('collector returned a partial or differently sized rollout')
            self._publish('optimizing')
            metrics = self.components.trainer.update(batch)
            self.components.attempted_vector_steps = self.collector.attempted_vector_steps
            self.components.returned_vector_steps = self.collector.returned_vector_steps
            self.journal.finish(self.components.observed_consumption(self.recipe.num_envs),
                                active_elapsed_s=self._active_elapsed())
            if self.components.curriculum.status == 'training_budget_complete':
                self._finish('completed', 'training_budget_complete')
                return metrics
            self._automatic_checkpoint()
            self._publish('between_updates')
            return metrics
        except StageValidationRequired:
            raise
        except BaseException as error:
            self._record_failure(error)
            raise


def _read_pinned_receipt(path, pin):
    if (not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}
            or pin['path'] != str(path)):
        raise ValueError('private receipt pin escapes the known attempt')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > JSON_LIMIT:
            raise ValueError('invalid private receipt')
        raw = stream.read(JSON_LIMIT + 1)
    if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('private receipt differs from its known pin')
    return json.loads(raw)


def load_stopped_attempt(store: StructuredCheckpointStore, control_directory, *, parent_exit_pin,
                         authorized_current_boot=None):
    """Validate a known parent's waitpid receipt before a private cold load.

    The parent pins its fresh own-child receipt after waitpid; no terminal PID
    is polled here. This only returns state. It does not launch, approve a new
    boot, restore physical state, or consume global RNG.
    """
    directory = Path(control_directory).resolve(strict=True)
    binding = read_json(directory.parent / 'request.json')
    result = read_json(directory / 'result.json')
    receipt = _read_pinned_receipt(directory.parent / 'exit.json', parent_exit_pin)
    current_authorized = (isinstance(binding, dict) and
                          (binding.get('boot_id') == boot_id() if authorized_current_boot is None
                           else authorized_current_boot == boot_id() == store.journal.authorized_boot))
    if (not isinstance(binding, dict) or not isinstance(result, dict)
            or receipt.keys() != {'format', 'token', 'boot', 'attempt', 'identity', 'returncode'}
            or receipt['format'] != 'private_structured_child_exit' or type(receipt['returncode']) is not int
            or receipt['returncode'] != 0 or receipt['token'] != binding['token']
            or receipt['boot'] != binding['boot_id'] or receipt['attempt'] != str(directory.parent)
            or not current_authorized or result.get('token') != binding['token']
            or result.get('summary', {}).get('identity') != receipt['identity']
            or not isinstance(receipt['identity'], dict)
            or receipt['identity'].get('boot_id') != binding['boot_id']
            or store.directory != directory.parent / 'train' / 'checkpoints'):
        raise ValueError('private endpoint lacks a clean known-parent waitpid binding')
    summary = result['summary']
    boundary = store.journal.checkpoint_boundary()
    path = store.directory / 'final.pt'
    if (summary.get('phase') != 'stopped' or summary.get('recipe_sha256') != store.recipe.sha256
            or summary.get('token') != binding['token'] or summary.get('checkpoint') != str(path)
            or summary.get('update') != boundary['ppo_updates']
            or summary.get('cumulative_transitions') != boundary['returned_transitions']
            or type(summary.get('session_active_elapsed_s')) not in (int, float)
            or not math.isfinite(summary['session_active_elapsed_s'])
            or summary['session_active_elapsed_s'] < boundary['active_elapsed_s']):
        raise ValueError('stopped learning endpoint and consumption differ')
    expected_paths = [str(path), str(path) + '.json']
    pins = result.get('checkpoint_pins')
    if (not isinstance(pins, list) or any(not isinstance(item, dict) for item in pins)
            or [item.get('path') for item in pins] != expected_paths):
        raise ValueError('stopped checkpoint pins escape the known attempt')
    sidecar = _read_pinned_receipt(Path(expected_paths[1]), pins[1])
    metadata = sidecar.get('metadata', {})
    if (pins[0].get('bytes') != sidecar['bytes'] or pins[0].get('sha256') != sidecar['sha256']
            or metadata.get('purpose') != 'cold_learning_resume' or metadata.get('token') != binding['token']
            or metadata.get('boot') != binding['boot_id'] or metadata.get('physical_state_restored') is not False
            or type(metadata.get('sequence')) is not int or metadata['sequence'] != summary.get('ack_sequence')):
        raise ValueError('final control result and snapshot publication differ')
    # The store bounds/hashes the model body before its sole deserialization.
    return store.load('final.pt'), float(summary['session_active_elapsed_s'])

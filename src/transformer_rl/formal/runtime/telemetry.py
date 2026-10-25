"""Best-effort event curves and device-resident rollout summaries.

Metrics JSON and the learning journal remain authoritative. TensorBoard never
reads an older log, changes a budget, or stores anything in learning state.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
import sys

import torch


class RolloutTelemetry:
    """Reduce interval densities and actual occupancy with one host transfer."""

    _FAMILIES = ('height', 'motion', 'rotation', 'terrain', 'request', 'recovery')
    _MODES = ('normal', 'stair', 'slope', 'recovery', 'jump')

    def __init__(self, num_envs, device, *, missing_terms=None):
        self.num_envs, self.device = num_envs, torch.device(device)
        self.missing_terms = dict(missing_terms or {})
        self._names = None
        self._sums = self._invalid = None
        self._steps = self._occupancy_steps = 0
        self._occupancy = torch.zeros(len(self._FAMILIES) + len(self._MODES) + 2,
                                      device=self.device, dtype=torch.float64)
        self._categories = (torch.arange(len(self._FAMILIES), device=self.device)[:, None],
                            torch.arange(len(self._MODES), device=self.device)[:, None])
        self._learning = torch.zeros(3, len(self._FAMILIES), device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def observe_occupancy(self, family, mode):
        """Call before step/reset so these labels describe the action interval."""
        if family is None or mode is None:
            return
        for value in (family, mode):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                    or value.device != self.device or value.dtype != torch.long):
                raise ValueError('telemetry occupancy requires device-resident integer row labels')
        start = 0
        for value, names, categories in zip((family, mode), (self._FAMILIES, self._MODES), self._categories):
            self._occupancy[start:start + len(names)].add_((value == categories).sum(-1))
            self._occupancy[start + len(names)].add_(((value < 0) | (value >= len(names))).sum())
            start += len(names) + 1
        self._occupancy_steps += 1

    @torch.no_grad()
    def observe_learning(self, family, actor, state_auxiliary, next_auxiliary):
        """Eligibility after ownership/query masks, before PPO minibatch updates."""
        if family is None:
            return
        groups = family == self._categories[0]
        for index, mask in enumerate((actor, state_auxiliary, next_auxiliary)):
            self._learning[index].add_((groups & mask[None]).sum(-1))

    @torch.no_grad()
    def observe_rewards(self, terms, *, series=None):
        if terms is None:
            return
        if not isinstance(terms, Mapping):
            raise TypeError('telemetry reward_terms must be a mapping of interval density tensors')
        sources = {'reward_density_per_s': terms, **(series or {})}
        selected = {}
        for group, mapping in sources.items():
            if not isinstance(group, str) or not isinstance(mapping, Mapping):
                raise TypeError('telemetry series require named mappings')
            for name, value in mapping.items():
                if not isinstance(name, str):
                    raise TypeError('telemetry reward names must be strings')
                # The physical producer prefixes paid components with
                # positive./cost.; unavailable placeholder terms must
                # remain absent from every scalar series.
                if name.rsplit('.', 1)[-1] not in self.missing_terms:
                    selected[f'{group}/{name}'] = value
        names = tuple(sorted(selected))
        if self._names is None:
            self._names = names
            self._sums = torch.zeros(len(names), device=self.device, dtype=torch.float64)
            self._invalid = torch.zeros_like(self._sums)
        elif names != self._names:
            raise ValueError('telemetry reward schema changed within a rollout')
        values = []
        for name in names:
            value = selected[name]
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,)
                    or value.device != self.device or not value.is_floating_point()):
                raise ValueError('telemetry reward terms require one floating value per environment')
            values.append(value.detach())
        if values:
            stacked = torch.stack(values)
            self._sums.add_(stacked.sum(-1, dtype=torch.float64))
            self._invalid.add_((~torch.isfinite(stacked)).sum(-1))
        self._steps += 1

    def summary(self):
        names = self._names or ()
        reward = (torch.cat((self._sums, self._invalid)) if names
                  else torch.empty(0, device=self.device, dtype=torch.float64))
        # This is the only device-to-host transfer in this observer.
        values = torch.cat((reward, self._occupancy, self._learning.flatten())).detach().cpu().tolist()
        size, samples = len(names), self._steps * self.num_envs
        densities, invalid = values[:size], values[size:2 * size]
        result = {
            'reward_density_per_s': {},
            'reward_nonfinite_samples': {},
            'reward_observed_samples': samples,
            'reward_source_available': self._names is not None,
            'missing_reward_terms': dict(self.missing_terms),
            'reward_scope': 'producer interval densities; formal positive./cost.* sum without a cost bound (SCUT); events are separate; ownership/recovery bonus may change learning reward',
        }
        for name, total, bad in zip(names, densities, invalid):
            group, term = name.split('/', 1)
            result.setdefault(group, {})[term] = (
                total / samples if samples and bad == 0 and math.isfinite(total) else None)
            if bad:
                result['reward_nonfinite_samples'][name] = int(bad)
        if self._occupancy_steps:
            denominator = self._occupancy_steps * self.num_envs
            occupancy = values[2 * size:2 * size + self._occupancy.numel()]
            offset = len(self._FAMILIES) + 1
            result['task_environment_step_fraction'] = dict(zip(
                (*self._FAMILIES, 'unknown'), (v / denominator for v in occupancy[:offset])))
            result['command_mode_before_step_fraction'] = dict(zip(
                (*self._MODES, 'unknown'), (v / denominator for v in occupancy[offset:])))
            learning = values[2 * size + self._occupancy.numel():]
            for index, name in enumerate(('ppo_eligible', 'state_auxiliary_eligible', 'next_auxiliary_eligible')):
                start = index * len(self._FAMILIES)
                result[name + '_task_steps'] = dict(zip(self._FAMILIES,
                    (int(v) for v in learning[start:start + len(self._FAMILIES)])))
            result['learning_exposure_scope'] = 'masked fresh endpoints; not retained optimizer steps or repeated sample uses'
        return result


class TensorBoardTelemetry:
    """Write only the row currently being published; failures cannot undo PPO."""

    _ALIASES = {
        'collection/reward_mean': 'reward/learning_step_mean',
        'collection/return_mean': 'return/gae_target_mean',
        'collection/advantage_std': 'advantage/raw_std',
    }

    def __init__(self, directory, *, scope, enabled=True, writer_factory=None):
        if scope not in ('formal', 'test') or type(enabled) is not bool:
            raise ValueError('telemetry requires an explicit formal/test scope and bool enable flag')
        self.scope, self.writer, self._last_step, self._test_step = scope, None, -1, 0
        self._notes = set()
        self._scope_recorded = False
        if enabled:
            if writer_factory is None:
                from torch.utils.tensorboard import SummaryWriter
                writer_factory = SummaryWriter
            # Every attempt/scope gets fresh event files. Never purge old data.
            self.writer = writer_factory(log_dir=str(Path(directory) / scope),
                                         max_queue=128, flush_secs=60)

    def __enter__(self):
        return self

    def _failure(self, error):
        print(f'TENSORBOARD_DISABLED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        writer = self.writer
        self.writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def __exit__(self, _exc_type=None, failure=None, _traceback=None):
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except BaseException as error:
                # Writer cleanup is secondary to an existing learning failure.
                # With no primary failure, preserve explicit interrupts.
                if failure is None and not isinstance(error, Exception):
                    raise
                if failure is not None:
                    failure.add_note(f'TensorBoard cleanup raised {type(error).__name__}: {error}')
                try:
                    self._failure(error)
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note('TensorBoard failure cleanup raised '
                                     f'{type(cleanup_error).__name__}: {cleanup_error}')

    def _write(self, name, value, step):
        if isinstance(value, Mapping):
            for key, nested in value.items():
                self._write(f'{name}/{key}' if name else str(key), nested, step)
        elif isinstance(value, Real) and math.isfinite(value):
            self.writer.add_scalar(self._ALIASES.get(name, name), value, global_step=step)
        elif value is None or isinstance(value, str):
            note = (name, str(value))
            if note not in self._notes:
                self.writer.add_text(name, 'unavailable' if value is None else value, global_step=step)
                self._notes.add(note)

    def record(self, row):
        if self.writer is None:
            return
        if self.scope == 'formal':
            step = row.get('budget_update')
            if type(step) is not int or step < 0 or step <= self._last_step:
                raise ValueError('formal TensorBoard step must strictly follow the actual budget_update')
        else:
            self._test_step += 1
            step = self._test_step
        try:
            if not self._scope_recorded:
                self._write('telemetry/scope', self.scope, step)
                self._write('telemetry/unconnected',
                            'No new closed-chain/contact/height/velocity sensor reductions; '
                            'existing emitted fields only. Checkpoint time remains within '
                            'accounting_checkpoint_publication_s when not separately emitted.', step)
                self._scope_recorded = True
            keys = ('collection', 'timing', 'optimization')
            if self.scope == 'formal':
                keys += ('consumption', 'total_consumption')
            for key in keys:
                self._write('ppo' if key == 'optimization' else key, row.get(key, {}), step)
            progress = (('update', 'budget_update', 'elapsed_s', 'curriculum_stage')
                        if self.scope == 'formal' else ('elapsed_s', 'curriculum_stage'))
            for key in progress:
                if key in row:
                    self._write('progress/' + key, row[key], step)
            if self.scope == 'test':
                self._write('progress/test_update', step, step)
                self._write('telemetry/formal_budget_credit', 0, step)
            self._last_step = step
        except Exception as error:
            # JSON/journal already own this update. A plotting failure must not
            # turn a committed learning update into a false training failure.
            self._failure(error)

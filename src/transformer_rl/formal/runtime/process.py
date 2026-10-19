"""Private structured learner entrypoint using the existing control handshake.

The launch parent owns admission, leases and waitpid. A separately selected
runtime factory owns the simulator lifecycle. This worker performs no automatic
evaluation and never substitutes the packed-policy worker or its checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from ..learning.accounting import LearningJournal
from ..learning.optimization import StructuredPPOTrainer
from ..learning.checkpoint import LearningComponents, LearningRecipe, StructuredCheckpointStore
from ..learning.collection import StructuredCollector
from ..learning.loop import StructuredLearningLoop, load_stopped_attempt
from ..policy import StructuredActorCritic
from ..tasks.curriculum import ContinuousTaskSampler, CurriculumBudget
from ..tasks.protocol import RequestProtocol
from ..tasks.terrain import ContinuousTerrainGenerator
from ...common.termination import _StopBudget
from ...common.training_control import TrainingControl, boot_id, read_json


def fresh_components(recipe: LearningRecipe):
    """Create the sole fresh initialization and check it before simulation."""
    random.seed(recipe.training_seed)
    np.random.seed(recipe.training_seed)
    torch.manual_seed(recipe.training_seed)
    model = StructuredActorCritic(recipe.policy).to(recipe.device)
    if StructuredCheckpointStore.model_sha256(model) != recipe.initial_model_sha256:
        raise ValueError('fresh learner differs from its frozen initial model')
    return LearningComponents(
        model, StructuredPPOTrainer(model, recipe.ppo),
        CurriculumBudget(promotion=recipe.curriculum_promotion),
        ContinuousTaskSampler(recipe.training_seed, device=recipe.device, domain=recipe.domain),
        ContinuousTerrainGenerator(recipe.training_seed, recipe.terrain),
        RequestProtocol(recipe.num_envs, device=recipe.device, config=recipe.protocol))


def run_worker(recipe, control_directory, runtime_factory, runtime_configuration,
               *, resume_attempt=None, parent_exit_pin=None, max_active_seconds=365 * 86400.):
    """Collect/update until the original budget or cooperative save-and-exit.

    Cold setup is intentionally outside the saved learning state: request
    identities remain retired while physics, sensor histories and contact state
    start fresh. The global action RNG is restored after that setup, once.
    """
    if not isinstance(recipe, LearningRecipe) or not callable(runtime_factory):
        raise TypeError('structured worker requires its parsed recipe and explicit runtime factory')
    control = TrainingControl(control_directory)
    binding = control.binding
    root, attempt = Path(binding['root']), Path(binding['attempt'])
    if (binding['target_update'] != recipe.journal_spec().target_rollouts
            or recipe.curriculum_promotion != 'exposure'
            or binding['boot_id'] != boot_id()
            or (resume_attempt is None) != (parent_exit_pin is None)):
        raise ValueError('worker budget, boot or explicit cold-resume receipt differs')
    restored = None
    try:
        with ExitStack() as stack:
            if resume_attempt is None:
                specification = read_json(root / 'structured.json') or {}
                if specification.get('source_transition_required'):
                    from .transition import load_transition

                    journal = stack.enter_context(LearningJournal(
                        root / 'journal', writable=True, authorized_boot=binding['boot_id']))
                    restored = load_transition(root, recipe, journal, binding)
                    components = restored.components
                else:
                    if binding['start_update'] != 0 or binding['prior_transitions'] != 0:
                        raise ValueError('fresh initialization cannot replace a consumed learning attempt')
                    components = fresh_components(recipe)
                    journal = stack.enter_context(LearningJournal.create(
                        root / 'journal', recipe.journal_spec(), authorized_boot=binding['boot_id']))
            else:
                journal = stack.enter_context(LearningJournal(
                    root / 'journal', writable=True, authorized_boot=binding['boot_id']))
                previous = Path(resume_attempt).resolve(strict=True)
                if previous.parent != attempt.parent or previous == attempt:
                    raise ValueError('cold resume must name a previous attempt of this same run')
                prior_store = StructuredCheckpointStore(previous / 'train/checkpoints', recipe, journal)
                restored, spent = load_stopped_attempt(prior_store, previous / 'control',
                                                       parent_exit_pin=parent_exit_pin,
                                                       authorized_current_boot=binding['boot_id'])
                if spent != binding['prior_active_elapsed_s']:
                    raise ValueError('active-time consumption differs from the stopped parent receipt')
                components = restored.components
            checkpoint_directory = attempt / 'train/checkpoints'
            checkpoint_directory.mkdir(parents=True, exist_ok=False)
            store = StructuredCheckpointStore(checkpoint_directory, recipe, journal)
            control.publish('initializing', recipe_sha256=recipe.sha256,
                            curriculum_promotion='exposure', automatic_evaluation=False,
                            solver_position_iterations=12, solver_velocity_iterations=6)
            runtime = stack.enter_context(runtime_factory(
                recipe, components, runtime_configuration, control=control))
            environment = getattr(runtime, 'environment', runtime)
            collector = StructuredCollector(environment, components.model, recipe.collection)
            collector.reset()
            collector.attempted_vector_steps = components.attempted_vector_steps
            collector.returned_vector_steps = components.returned_vector_steps
            if restored is not None:
                restored.restore_global_rng_after_cold_setup()
            components.validate_binding(recipe, journal)
            with _StopBudget(max_active_seconds) as stop, (attempt / 'train/metrics.jsonl').open('x') as metrics_log:
                loop = StructuredLearningLoop(recipe, components, collector, journal, store, control, stop)
                while not loop._finished:
                    started = time.perf_counter()
                    metrics = loop.run_update()
                    if metrics is not None:
                        row = {'update': components.trainer.update_index,
                               'curriculum_stage': components.curriculum.stage,
                               'elapsed_s': time.perf_counter() - started,
                               'collection': collector.last_metrics,
                               'timing': loop.last_timing,
                               'consumption': components.observed_consumption(recipe.num_envs),
                               'optimization': metrics}
                        metrics_log.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                        metrics_log.flush()
                return dict(loop._report)
    except BaseException as error:
        control.fail(error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Private structured training worker')
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--runtime-factory', required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--resume-attempt', type=Path)
    parser.add_argument('--parent-exit-pin', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', args.runtime_factory):
        raise ValueError('runtime factory must be module:factory')
    module, name = args.runtime_factory.split(':')
    report = run_worker(
        LearningRecipe.from_dict(read_json(args.recipe)), args.control_dir,
        getattr(importlib.import_module(module), name), read_json(args.runtime_config),
        resume_attempt=args.resume_attempt,
        parent_exit_pin=read_json(args.parent_exit_pin) if args.parent_exit_pin else None)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

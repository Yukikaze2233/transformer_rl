"""Freeze a new complete Lab learning session without starting a simulator.

This is fresh initialization only. Existing sessions resume through trainctl;
diagnostic lab checkpoints and historical learning budgets are never silently
converted, overwritten or replayed by this command.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch

from ...common.training_control import boot_id, read_json
from ..learning.checkpoint import StructuredCheckpointStore
from ..learning.settings import FormalTrainingSettings
from ..policy import StructuredActorCritic
from ..runtime.session import StructuredTrainingSession
from .runtime import parse_configuration
from .settings import load_asset_description


def prepare_session(root, settings, runtime_configuration, resources, *, training_seed,
                    include_jump=True, run_id=None, python=sys.executable, learner_source=None):
    """Create a distinct fresh source-bound run; never start/resume a worker."""
    if not isinstance(settings, FormalTrainingSettings) or type(include_jump) is not bool:
        raise TypeError('Lab preparation requires parsed formal settings and an explicit course choice')
    root = Path(root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('fresh Lab preparation cannot replace an existing session')
    if training_seed not in settings.training_seeds:
        raise ValueError('training seed is not in the formal plan')
    StructuredTrainingSession._resource_definition(resources)
    asset = load_asset_description(runtime_configuration['asset_directory'])
    package = Path(learner_source).resolve(strict=True) if learner_source else Path(__file__).resolve().parents[2]
    contract = {'runtime': runtime_configuration, 'settings': settings.to_dict(), 'include_jump': include_jump}
    task_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    # The worker creates the same CPU-seeded model before moving it to CUDA.
    # Preparation needs neither an SDK context nor a CUDA allocation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_seed)
        initial_sha = StructuredCheckpointStore.model_sha256(StructuredActorCritic(settings.policy))
    recipe = settings.recipe(run_id or root.name, training_seed,
        asset_manifest_sha256=asset.manifest_sha256,
        learner_source_sha256=StructuredTrainingSession.source_sha256(package),
        task_contract_sha256=task_sha, initial_model_sha256=initial_sha)
    if not include_jump:
        recipe = replace(recipe, target_rollouts=30000)
    parse_configuration(recipe, runtime_configuration, boot_id())
    return StructuredTrainingSession.create(root, recipe,
        runtime_factory='transformer_rl.formal.lab_env.runtime:runtime',
        runtime_configuration=runtime_configuration, resources=resources,
        python=python, learner_source=package)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--runtime-json', type=Path, required=True)
    parser.add_argument('--resources-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--no-jump', action='store_true', help='Create a distinct 30k run; never extend a prior run')
    args = parser.parse_args(argv)
    session = prepare_session(args.output, FormalTrainingSettings.load(args.settings),
        read_json(args.runtime_json), read_json(args.resources_json), training_seed=args.seed,
        include_jump=not args.no_jump, run_id=args.run_id)
    print(json.dumps({'root': str(session.root), 'target_updates': session.spec['updates'],
                      'recipe_sha256': session.spec['recipe_sha256'], 'training_started': False,
                      'next_command': ['trainctl', 'start', str(session.root)]}), flush=True)


if __name__ == '__main__':
    main()

"""Add explicit control boundaries to an authenticated frozen learner copy."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

from ..common.training_control import file_pin, write_json


def instrument_package(source, destination, expected_files):
    source, destination = Path(source).resolve(strict=True), Path(destination).absolute()
    actual = {str(path.relative_to(source)): file_pin(path)["sha256"]
              for path in source.rglob("*.py") if "__pycache__" not in path.parts}
    if actual != expected_files:
        raise ValueError("frozen learner files differ from the original campaign")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    edits = []

    def replace(name, before, after):
        path = destination / name
        text = path.read_text()
        if text.count(before) != 1:
            raise ValueError(f"unsupported frozen control boundary: {name}: {before[:65]}")
        path.write_text(text.replace(before, after))
        edits.append({"file": name, "before": before, "after": after})

    replace("frame_workflow.py", "expected_initial_model_sha256=None):",
            "expected_initial_model_sha256=None, control_dir=None):")
    replace("frame_workflow.py", "    attempted_updates = 0\n    try:\n",
            "    attempted_updates = 0\n    control = None\n    try:\n"
            "        if control_dir is not None:\n"
            "            from .training_control import TrainingControl\n"
            "            control = TrainingControl(control_dir)\n")
    replace("frame_workflow.py", '            with (run_dir / "metrics.jsonl").open("x", buffering=1) as log:\n',
            '            def save_pause_checkpoint(sequence):\n'
            '                metadata["collected_transitions"] = prior_transitions + collector.total_transitions\n'
            '                path = checkpoint_dir / f"pause_{sequence:06d}_update_{update:06d}.pt"\n'
            '                save_frame_checkpoint(path, model, trainer, config, update, metadata)\n'
            '                return path\n\n'
            '            with (run_dir / "metrics.jsonl").open("x", buffering=1) as log:\n')
    replace("frame_workflow.py", '                while update < start_update + updates and not stop.stopped():\n',
            '                while update < start_update + updates and not stop.stopped():\n'
            '                    if control is not None and not control.boundary(stop, save_pause_checkpoint):\n'
            '                        break\n')
    replace("frame_workflow.py", '                    metrics = trainer.update(batch, diagnostics=True, regularizer=regularizer)\n',
            '                    if control is not None:\n'
            '                        control.publish("optimizing")\n'
            '                    metrics = trainer.update(batch, diagnostics=True, regularizer=regularizer)\n')
    replace("frame_workflow.py", '            metadata["collected_transitions"] = prior_transitions + collector.total_transitions\n            final_path',
            '                    if control is not None:\n'
            '                        control.publish("running", update=update, cumulative_transitions=metadata["collected_transitions"],\n'
            '                            reward_mean=collector.last_metrics["reward_mean"], learning_rate=trainer.optimizer.param_groups[0]["lr"])\n'
            '            metadata["collected_transitions"] = prior_transitions + collector.total_transitions\n            final_path')
    replace("frame_workflow.py", '            _write_json(run_dir / "completion.json", report)\n            return report',
            '            if control is not None:\n'
            '                active, paused = control.elapsed()\n'
            '                report.update(active_elapsed_s=active, paused_elapsed_s=paused, resume_scope="learning_state_with_episode_reset")\n'
            '            _write_json(run_dir / "completion.json", report)\n'
            '            if control is not None:\n'
            '                control.finish(report)\n            return report')
    replace("frame_workflow.py", '    except BaseException as error:\n        # Preserve successfully applied updates,',
            '    except BaseException as error:\n        if control is not None:\n            control.fail(error)\n        # Preserve successfully applied updates,')
    replace("frame_workflow.py", '        for index in range(steps):\n',
            '        from .training_control import evaluation_control\n'
            '        live_control = evaluation_control()\n'
            '        for index in range(steps):\n'
            '            if live_control is not None:\n'
            '                live_control.boundary(index, steps)\n')
    replace("frame_cli.py", '    train.add_argument("--expected-initial-model-sha256", type=_initial_model_sha)\n',
            '    train.add_argument("--expected-initial-model-sha256", type=_initial_model_sha)\n'
            '    train.add_argument("--control-dir", type=Path, help=argparse.SUPPRESS)\n')
    replace("frame_cli.py", '            guard_options = {}\n',
            '            guard_options = {}\n'
            '            if args.control_dir is not None:\n'
            '                guard_options["control_dir"] = args.control_dir\n')
    replace("frame_cli.py", '    suite.add_argument("--checkpoint", type=Path, required=True)\n',
            '    suite.add_argument("--control-dir", type=Path, help=argparse.SUPPRESS)\n'
            '    suite.add_argument("--checkpoint", type=Path, required=True)\n')
    replace("frame_cli.py", '        elif args.operation == "evaluate-suite":\n',
            '        elif args.operation == "evaluate-suite":\n'
            '            from .training_control import EvaluationControl\n'
            '            live_control = EvaluationControl(args.control_dir) if args.control_dir is not None else None\n')
    replace("frame_cli.py", '                trace_output=args.trace_output, trace_replicas=args.trace_replicas)\n        elif args.operation == "export":',
            '                trace_output=args.trace_output, trace_replicas=args.trace_replicas)\n'
            '            if live_control is not None:\n'
            '                live_control.finish_evaluation(args.steps)\n        elif args.operation == "export":')
    shutil.copyfile(Path(__file__).resolve().parents[1] / "common/training_control.py", destination / "training_control.py")
    for name in ("frame_workflow.py", "frame_cli.py"):
        patched = (destination / name).read_text()
        ast.parse(patched)
        for edit in reversed([item for item in edits if item["file"] == name]):
            if patched.count(edit["after"]) != 1:
                raise ValueError("control instrumentation is not reversible")
            patched = patched.replace(edit["after"], edit["before"])
        if patched.encode() != (source / name).read_bytes():
            raise ValueError("unapproved learner change outside control boundaries")
    proof = {"original_files": actual, "changed_existing_files": ["frame_cli.py", "frame_workflow.py"],
             "added_files": ["training_control.py"], "edits": edits,
             "all_policy_optimizer_environment_history_checkpoint_code_unchanged": True,
             "original_sources_untouched": True,
             "controlled_files": {str(path.relative_to(destination)): file_pin(path)["sha256"]
                                  for path in destination.rglob("*.py")}}
    write_json(destination.parent / "control_compatibility.json", proof)
    return proof

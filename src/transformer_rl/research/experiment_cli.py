"""Plan, run, and summarize independent architecture/seed experiments."""
import argparse
import json
from pathlib import Path
import sys

from .experiments import plan, run, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--spec", type=Path, required=True)
    planner.add_argument("--root", type=Path, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--root", type=Path, required=True)
    runner.add_argument("--max-parallel", type=int)
    summary = commands.add_parser("summarize")
    summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "plan":
            result = plan(args.spec, args.root)
        elif args.operation == "run":
            result = run(args.root, args.max_parallel)
        else:
            result = summarize(args.root)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.operation == "run" and any(j is None or j["status"] != "completed" for j in result["jobs"]):
            return 1
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Explicit process owner: finish CLI artifacts before Sim6 fast shutdown."""
from __future__ import annotations

import sys
import traceback


_apps = []
_active = False


def register_app(app):
    if not _active:
        raise RuntimeError("run this backend with python -m examples._isaaclab_process")
    _apps.append(app)


def main():
    global _active
    _active = True
    exit_code = 1
    try:
        from transformer_rl.cli import main as cli_main
        exit_code = cli_main()
        exit_code = 0 if exit_code is None else exit_code
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        if isinstance(error.code, str):
            print(error.code, file=sys.stderr)
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for app in reversed(_apps):
            app.close(wait_for_replicator=False, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    # Factories import the canonical name, not a second registry instance.
    sys.modules["examples._isaaclab_process"] = sys.modules[__name__]
    raise SystemExit(main())

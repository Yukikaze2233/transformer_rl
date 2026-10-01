"""Own optional simulator shutdown after all packed workflow artifacts are written."""
import sys
import traceback

_apps = []
_active = False


def require_worker():
    if not _active:
        raise RuntimeError("use python -m transformer_rl.frame_process for simulator workers")


def register_app(app):
    require_worker()
    _apps.append(app)


def main():
    global _active
    _active = True
    code = 1
    try:
        from .frame_cli import main as cli_main
        code = cli_main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for app in reversed(_apps):
            app.close(wait_for_replicator=False, exit_code=code)
    return code


if __name__ == "__main__":
    sys.modules["transformer_rl.frame_process"] = sys.modules[__name__]
    raise SystemExit(main())

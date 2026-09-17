"""Real sleeping process for control tests; never imports a learner or simulator."""
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
import trainctl

root, attempt = Path(sys.argv[1]), Path(sys.argv[2])
stopped = False


def stop(*_):
    global stopped
    stopped = True


signal.signal(signal.SIGTERM, stop)
with trainctl.locked(root / 'control/run.lock', blocking=False):
    trainctl.save_json(attempt / 'launch.json', {'identity': trainctl.process_identity(__import__('os').getpid())})
    trainctl.save_json(attempt / 'status.json', {'status': 'running', 'stage': 'fixture'}, replace=True)
    deadline = time.monotonic() + 30
    while not stopped and time.monotonic() < deadline:
        time.sleep(0.02)
    trainctl.save_json(attempt / 'exit.json', {'status': 'stopped' if stopped else 'completed'})

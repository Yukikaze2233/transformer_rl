#!/usr/bin/env python3
"""Linux terminal control for frozen, checkpoint-resumable training studies.

The frontend uses only the standard library. Training runs in its registered
Python/runtime and imports the study's frozen learning source, not this checkout.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def latest_metric(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - 65536))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
            if 'update' in value and 'collection' in value:
                return value
        except ValueError:
            continue
    return None


def with_progress(root, state, *, legacy=False):
    state = dict(state)
    metric_path = state.get('metric_path')
    if not metric_path and legacy and state.get('stage') and state.get('job'):
        metric_path = root / state['stage'] / 'plan/jobs' / state['job'] / 'train/metrics.jsonl'
    if metric_path:
        metric_path = inside(root, metric_path)
        completion = read_json(metric_path.parent / 'completion.json')
        if completion:
            state.update(latest_update=completion['cumulative_update'],
                         latest_saved_update=completion['cumulative_update'],
                         collected_transitions=completion['collected_transitions'])
        else:
            metric = latest_metric(metric_path)
            if metric:
                state.update(latest_update=metric['update'],
                             collected_transitions=metric['collection']['total_transitions'])
    return state


def save_json(path, data, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity(pid):
    if type(pid) is not int or pid < 1:
        return None
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    argv=[os.fsdecode(v) for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def identity_alive(identity):
    return bool(identity) and process_identity(identity.get('pid')) == identity


def signal_checked(identity):
    """Never fall back to signaling an unverified numeric PID."""
    if not identity_alive(identity):
        return False
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        system = Path('/usr/bin/python3')
        if not system.is_file() or system.resolve() == Path(sys.executable).resolve():
            raise RuntimeError('pidfd support is required; run trainctl using a supported Linux system Python')
        result = subprocess.run([str(system), str(Path(__file__).resolve()), '_signal',
                                 '--identity', json.dumps(identity)], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'system Python could not signal the verified process')
        return json.loads(result.stdout)['signalled']
    try:
        fd = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return False
    try:
        if not identity_alive(identity):
            return False
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        return True
    finally:
        os.close(fd)


def registry_path(value=None):
    return Path(value or os.environ.get('TRAINCTL_REGISTRY',
                                       str(Path.home() / '.config/transformer-rl/trainctl.json'))).expanduser().absolute()


def registry(path):
    result = read_json(path, {'schema_version': 1, 'default': None, 'studies': {}})
    if result.get('schema_version') != 1 or not isinstance(result.get('studies'), dict):
        raise ValueError('invalid training registry')
    return result


def entry(path, name=None):
    data = registry(path)
    name = name or data['default']
    if name not in data['studies']:
        raise ValueError('unknown study; use register or list')
    return name, data['studies'][name]


def register(path, name, root, python, runtime):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('study name must use letters, digits, underscore or hyphen')
    root, python, runtime = root.resolve(strict=True), python.absolute(), runtime.resolve(strict=True)
    manifest = read_json(root / 'study.json')
    if not manifest or 'stage_plan_hashes' not in manifest:
        raise ValueError('register requires a prepared estimator study with study.json')
    source = Path(manifest['source']).resolve(strict=True)
    if not python.is_file() or not (source / 'tools/run_estimator_study.py').is_file():
        raise ValueError('training Python or frozen study source is unavailable')
    value = dict(root=str(root), source=str(source), python=str(python), runtime=str(runtime))
    with locked(path.with_suffix('.lock')):
        data = registry(path)
        if name in data['studies'] and data['studies'][name] != value:
            raise ValueError('name already refers to a different study; choose a new name')
        data['studies'][name] = value
        data['default'] = data['default'] or name
        save_json(path, data, replace=True)
    return value


def inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'artifact escapes study root: {path}')
    return path


def legacy_identity(root):
    launch = read_json(root / 'launch.json', {})
    actual = process_identity(launch.get('pid'))
    if (actual and launch.get('process_start_ticks') == actual['start_ticks']
        and launch.get('script') in actual['argv'] and str(root) in actual['argv']
        and 'run' in actual['argv']):
        return actual
    return None


def orphan_workers(root):
    workers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        identity = process_identity(int(path.name))
        if (not identity or len(identity['argv']) < 4 or identity['argv'][1] != '-m'
            or identity['argv'][3] not in ('train', 'evaluate')):
            continue
        argv = identity['argv']
        for operation, flag in (('train', '--run-dir'), ('evaluate', '--output')):
            if operation in argv and flag in argv:
                index = argv.index(flag)
                if (index + 1 < len(argv) and Path(argv[index + 1]).is_absolute()
                    and Path(argv[index + 1]).resolve().is_relative_to(root)):
                    workers.append(identity)
                    break
    return workers


def status(value):
    root = Path(value['root']).resolve(strict=True)
    pointer = read_json(root / 'control/current.json')
    if pointer:
        attempt = inside(root, pointer['attempt'])
        launch = read_json(attempt / 'launch.json', {})
        state = read_json(attempt / 'status.json', {})
        receipt = read_json(attempt / 'exit.json')
        live = identity_alive(launch.get('identity'))
        reported = (receipt or state).get('status', 'starting')
        if not live and receipt is None and launch:
            reported = 'stale'
        if not launch and receipt is None:
            request = read_json(attempt / 'request.json', {})
            if time.time() - request.get('requested_epoch', 0) > 45:
                reported = 'start_failed'
            if (attempt / 'start-error.json').exists():
                reported = 'start_failed'
        orphans = orphan_workers(root) if not live else []
        if orphans:
            reported = 'orphaned_worker'
        progress = with_progress(root, {**state, **(receipt or {})})
        return {**progress, 'status': reported, 'active': live or bool(orphans),
                'supervisor_active': live, 'orphaned_workers': orphans,
                'identity': launch.get('identity'), 'attempt': str(attempt), 'root': str(root),
                'log': state.get('worker_log') or str(attempt / 'control.log'), 'managed': True}
    original = with_progress(root, read_json(root / 'status.json', {}), legacy=True)
    identity = legacy_identity(root)
    reported = original.get('status', 'prepared')
    if reported == 'running' and not identity:
        reported = 'stale'
    orphans = orphan_workers(root) if not identity else []
    if orphans:
        reported = 'orphaned_worker'
    log = root / 'queue.log'
    if original.get('stage') and original.get('job'):
        directory = root / original['stage'] / 'plan/jobs' / original['job']
        if original.get('phase') == 'train':
            log = directory / 'train.log'
        else:
            match = re.fullmatch(r'evaluate_(\d+)_(.+)_(\d+)', original.get('phase', ''))
            if match:
                update, scenario, seed = match.groups()
                log = directory / f'evaluation_{int(update):06d}/{scenario}_{seed}.log'
    return {**original, 'status': reported, 'active': bool(identity) or bool(orphans),
            'supervisor_active': bool(identity), 'orphaned_workers': orphans, 'identity': identity,
            'root': str(root), 'log': str(log), 'managed': False}


def launch_command(value, attempt):
    script = str(Path(__file__).resolve())
    # Arguments are positional shell parameters, never interpolated shell code.
    bootstrap = ('unset PYTHONHOME PYTHONPATH CUDA_VISIBLE_DEVICES; source "$1" || exit; '
                 'export PYTHONPATH="$2/src:$2" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 '
                 'ENABLE_CAMERAS=0 LIVESTREAM=0; exec "$3" "$4" _run --root "$5" --attempt "$6"')
    return ['/bin/bash', '--noprofile', '--norc', '-c', bootstrap, 'trainctl',
            value['runtime'], value['source'], value['python'], script, value['root'], str(attempt)]


def start(value, name):
    root = Path(value['root']).resolve(strict=True)
    with locked(root / 'control/command.lock'):
        current = status(value)
        if current['active'] or current['status'] == 'starting':
            return current
        if current['status'] in ('completed', 'needs_review'):
            return current
        # This lease is held by the managed supervisor throughout its lifetime.
        with locked(root / 'control/run.lock', blocking=False):
            pass
        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        attempt = root / 'control/attempts' / token
        attempt.mkdir(parents=True, exist_ok=False)
        session = 'trainctl-' + hashlib.sha256(str(root).encode()).hexdigest()[:8] + '-' + token[-8:]
        command = launch_command(value, attempt)
        previous = read_json(root / 'control/current.json', {})
        generation = previous.get('generation', 0) + 1
        save_json(attempt / 'request.json', dict(requested_at=now(), requested_epoch=time.time(),
                  name=name, registration=value, session=session, argv=command,
                  generation=generation, controller_script=str(Path(__file__).resolve())))
        save_json(root / 'control/current.json', {'attempt': str(attempt), 'generation': generation}, replace=True)
        shell_command = shlex.join(command) + ' > ' + shlex.quote(str(attempt / 'control.log')) + ' 2>&1'
        try:
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', value['source'], shell_command],
                           check=True, capture_output=True, text=True, timeout=15)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                current = status(value)
                if current['active'] or (attempt / 'exit.json').exists():
                    return current
                time.sleep(0.1)
            raise TimeoutError(f'startup not acknowledged; inspect {attempt / "control.log"}')
        except BaseException as error:
            if not (attempt / 'launch.json').exists():
                save_json(attempt / 'start-error.json', {'error': str(error), 'at': now()})
            raise


def pause(value, wait_seconds=180):
    root = Path(value['root']).resolve(strict=True)
    with locked(root / 'control/command.lock'):
        current = status(value)
        deadline = time.monotonic() + wait_seconds
        if current['status'] == 'starting' and not current['active']:
            attempt = inside(root, current['attempt'])
            marker = attempt / 'pause-request.json'
            if not marker.exists():
                save_json(marker, {'requested_at': now()})
            while current['status'] == 'starting' and time.monotonic() < deadline:
                time.sleep(0.1)
                current = status(value)
            if current['status'] == 'starting':
                return {**current, 'status': 'stopping'}
        if not current['active']:
            return current
        identities = [current['identity']] if current['supervisor_active'] else current['orphaned_workers']
        for identity in identities:
            signal_checked(identity)
        while any(identity_alive(identity) for identity in identities) and time.monotonic() < deadline:
            time.sleep(0.2)
        result = status(value)
        if any(identity_alive(identity) for identity in identities) or result['active']:
            result['status'] = 'stopping'
        return result


def show(value, name=None, as_json=False):
    state = status(value)
    if as_json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print(f"{name or Path(value['root']).name}: {state['status']}  active={state['active']}")
        print(f"  stage={state.get('stage')}  job={state.get('job')}  phase={state.get('phase')}")
        print(f"  update={state.get('latest_update', '—')}  samples={state.get('collected_transitions', '—')}")
        print(f"  root={value['root']}")
        if state.get('error'):
            print(f"  error={state['error']}")
    return state


def logs(value, lines=40, follow=False):
    position = 0
    previous = None
    while True:
        state = status(value)
        path = inside(Path(value['root']), state['log'])
        if path.exists():
            with path.open(errors='replace') as stream:
                if previous != path:
                    print(f'[{path}]')
                    print(''.join(deque(stream, maxlen=lines)), end='')
                    position = stream.tell()
                else:
                    stream.seek(position)
                    print(stream.read(), end='', flush=True)
                    position = stream.tell()
            previous = path
        if not follow or not state['active']:
            return
        time.sleep(0.5)


def menu(value, name):
    while True:
        show(value, name)
        print('1 状态  2 启动/继续  3 正常暂停  4 最近日志  q 退出')
        choice = input('选择: ').strip().lower()
        if choice == 'q':
            return
        if choice == '2':
            print(json.dumps(start(value, name), indent=2, ensure_ascii=False))
        elif choice == '3':
            print(json.dumps(pause(value), indent=2, ensure_ascii=False))
        elif choice == '4':
            logs(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry')
    commands = parser.add_subparsers(dest='operation')
    reg = commands.add_parser('register', help='Register a prepared frozen study; does not start training')
    reg.add_argument('name')
    reg.add_argument('--root', type=Path, required=True)
    reg.add_argument('--python', type=Path, required=True)
    reg.add_argument('--runtime', type=Path, required=True)
    for action in ('status', 'start', 'resume', 'pause', 'logs', 'menu', 'check'):
        command = commands.add_parser(action)
        command.add_argument('name', nargs='?')
        if action == 'status':
            command.add_argument('--json', action='store_true')
        if action == 'pause':
            command.add_argument('--wait', type=float, default=180)
        if action == 'logs':
            command.add_argument('--lines', type=int, default=40)
            command.add_argument('--follow', action='store_true')
    commands.add_parser('list')
    worker = commands.add_parser('_run')
    worker.add_argument('--root', type=Path, required=True)
    worker.add_argument('--attempt', type=Path, required=True)
    check = commands.add_parser('_check')
    check.add_argument('--root', type=Path, required=True)
    signalling = commands.add_parser('_signal')
    signalling.add_argument('--identity', required=True)
    args = parser.parse_args()
    try:
        if args.operation == '_signal':
            print(json.dumps({'signalled': signal_checked(json.loads(args.identity))}))
            return 0
        if args.operation == '_run':
            if __package__:
                from ._managed_study import run_managed
            else:
                from _managed_study import run_managed
            return run_managed(args.root, args.attempt)
        if args.operation == '_check':
            if __package__:
                from ._managed_study import inspect_study
            else:
                from _managed_study import inspect_study
            print(json.dumps(inspect_study(args.root), indent=2))
            return 0
        path = registry_path(args.registry)
        if args.operation == 'register':
            print(json.dumps(register(path, args.name, args.root, args.python, args.runtime), indent=2))
            return 0
        if (args.operation == 'list' or (args.operation == 'status' and args.name is None)
            or (args.operation is None and not sys.stdin.isatty())):
            if getattr(args, 'json', False):
                print(json.dumps({'studies': {name: status(value) for name, value in registry(path)['studies'].items()}},
                                 indent=2, ensure_ascii=False))
                return 0
            for name, value in registry(path)['studies'].items():
                show(value, name, getattr(args, 'json', False))
            return 0
        name, value = entry(path, getattr(args, 'name', None))
        if args.operation in (None, 'menu'):
            menu(value, name)
        elif args.operation == 'status':
            show(value, name, args.json)
        elif args.operation in ('start', 'resume'):
            result = start(value, name)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2 if result['status'] in ('needs_review', 'failed', 'stale') else 0
        elif args.operation == 'pause':
            if not 0 <= args.wait <= 600:
                raise ValueError('--wait must be between 0 and 600 seconds')
            result = pause(value, args.wait)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2 if result['status'] == 'stopping' else 0
        elif args.operation == 'logs':
            if not 1 <= args.lines <= 10000:
                raise ValueError('--lines must be between 1 and 10000')
            logs(value, args.lines, args.follow)
        elif args.operation == 'check':
            # The check runs in the frozen training runtime, with no worker launch.
            bootstrap = ('unset PYTHONHOME PYTHONPATH; source "$1" || exit; '
                         'export PYTHONPATH="$2/src:$2" CUDA_VISIBLE_DEVICES=""; exec "$3" "$4" _check --root "$5"')
            subprocess.run(['/bin/bash', '-c', bootstrap, 'trainctl-check', value['runtime'], value['source'],
                            value['python'], str(Path(__file__).resolve()), value['root']], check=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'trainctl: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

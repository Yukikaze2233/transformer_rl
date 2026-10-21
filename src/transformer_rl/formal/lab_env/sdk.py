"""Isaac Lab 启动与设置边界；不改变环境、控制器或学习算法。"""
from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


_IDENTITY_FORMAT = 'private_native_sdk_identity_v1'
_REQUIRED_PACKAGES = {'isaaclab', 'isaaclab-physx', 'isaacsim', 'torch', 'warp-lang'}
_SDK_CONTRACTS = {'force_physx': True, 'use_newton_actuators': False, 'quaternion_order': 'xyzw'}
_ENVIRONMENT_PATHS = {'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PATH', 'ISAACSIM_PATH'}
_FATAL_NATIVE_REFERENCES = []


def _sdk_bytes(path, pin=None, *, limit=256 * 1024):
    """Read one bounded private seal without following a redirected path."""
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('SDK identity evidence requires a canonical absolute file')
    if pin is not None and (
            not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}
            or pin['path'] != str(path) or type(pin['bytes']) is not int or not 0 < pin['bytes'] <= limit
            or not isinstance(pin['sha256'], str) or re.fullmatch('[0-9a-f]{64}', pin['sha256']) is None):
        raise ValueError('SDK identity evidence pin differs')
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit
                or pin is not None and info.st_size != pin['bytes']):
            raise ValueError('SDK identity evidence file identity differs')
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or pin is not None and hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('SDK identity evidence bytes differ from their seal')
    return raw


def _sdk_document(path, pin=None, *, limit=256 * 1024):
    document = json.loads(_sdk_bytes(path, pin, limit=limit))
    if not isinstance(document, dict) or not document:
        raise ValueError('SDK identity evidence requires a nonempty document')
    json.dumps(document, allow_nan=False)
    return document


def _sdk_pin_document(pin):
    if not isinstance(pin, dict) or not isinstance(pin.get('path'), str):
        raise ValueError('SDK identity evidence pin differs')
    return _sdk_document(pin['path'], pin)


def verify_native_sdk_identity(pin):
    """Match the running interpreter and installed GA sources before Kit opens."""
    identity = _sdk_pin_document(pin)
    expected = {'format', 'interpreter', 'python', 'checkout', 'packages', 'required_environment', 'contracts'}
    python, checkout = identity.get('python'), identity.get('checkout')
    packages, required = identity.get('packages'), identity.get('required_environment')
    if (set(identity) != expected or identity['format'] != _IDENTITY_FORMAT
            or not isinstance(identity['interpreter'], str)
            or not Path(identity['interpreter']).is_absolute()
            or os.path.normpath(identity['interpreter']) != identity['interpreter']
            or not isinstance(python, dict) or set(python) != {'major', 'minor'}
            or any(type(python[name]) is not int or python[name] < 0 for name in python)
            or not isinstance(checkout, dict) or set(checkout) != {'directory', 'commit', 'uv_lock_pin'}
            or not isinstance(checkout['directory'], str)
            or not isinstance(checkout['commit'], str) or re.fullmatch('[0-9a-f]{40}', checkout['commit']) is None
            or not isinstance(checkout['uv_lock_pin'], dict)
            or not isinstance(packages, dict) or not _REQUIRED_PACKAGES <= packages.keys()
            or any(not isinstance(name, str) or not name or not isinstance(value, str) or not value
                   for name, value in packages.items())
            or not isinstance(required, dict) or not required or not set(required) <= _ENVIRONMENT_PATHS
            or not isinstance(required.get('LD_LIBRARY_PATH'), list) or not required['LD_LIBRARY_PATH']
            or identity['contracts'] != _SDK_CONTRACTS
            or identity['contracts'].get('force_physx') is not True
            or identity['contracts'].get('use_newton_actuators') is not False):
        raise ValueError('target SDK identity requires the exact private native contract')
    # Preserve the venv invocation path: resolving this symlink would collapse
    # distinct SDK environments onto their shared system CPython executable.
    if (sys.executable != identity['interpreter'] or not Path(sys.executable).is_file()
            or python != {'major': sys.version_info.major, 'minor': sys.version_info.minor}):
        raise ValueError('running SDK interpreter or Python version differs from the target')
    directory = Path(checkout['directory'])
    if (not directory.is_absolute() or directory.resolve(strict=True) != directory or not directory.is_dir()
            or checkout['uv_lock_pin'].get('path') != str(directory / 'uv.lock')):
        raise ValueError('target SDK checkout or lock path differs')
    lock = checkout['uv_lock_pin']
    # uv.lock is TOML, so bind its bytes rather than parsing it as a JSON document.
    _sdk_bytes(lock['path'], lock, limit=8 * 1024 * 1024)

    def git(*arguments):
        return subprocess.run(['git', '--no-optional-locks', '-C', str(directory), *arguments],
                              check=True, capture_output=True, text=True, timeout=15).stdout.strip()

    if (git('rev-parse', '--verify', 'HEAD') != checkout['commit']
            or git('status', '--porcelain', '--untracked-files=no')
            or git('status', '--porcelain', '--untracked-files=all', '--',
                   'source/isaaclab', 'source/isaaclab_physx')):
        raise ValueError('target SDK checkout source differs from the pinned clean commit')
    origins = {}
    for module in ('isaaclab', 'isaaclab_physx'):
        specification = util.find_spec(module)
        expected_origin = directory / 'source' / module / module / '__init__.py'
        if (specification is None or not isinstance(specification.origin, str)
                or Path(specification.origin).resolve(strict=True) != expected_origin):
            raise ValueError('running SDK module does not come from the target checkout: ' + module)
        origins[module] = str(expected_origin)
    observed_packages = {name: metadata.version(name) for name in packages}
    if observed_packages != packages:
        raise ValueError('installed SDK package versions differ from the pinned target')
    found = {}
    for name, paths in required.items():
        if (not isinstance(paths, list) or not paths
                or any(not isinstance(path, str) or not Path(path).is_absolute()
                       or os.path.normpath(path) != path or not Path(path).exists() for path in paths)
                or len(set(paths)) != len(paths)):
            raise ValueError('target SDK requires existing absolute process paths: ' + name)
        actual = set(re.split('[: ]', os.environ.get(name, '')) if name == 'LD_PRELOAD'
                     else os.environ.get(name, '').split(os.pathsep))
        if not set(paths) <= actual:
            raise ValueError('running SDK process paths differ from the target: ' + name)
        found[name] = list(paths)
    return {'format': 'private_native_sdk_identity_validation_v1',
            'status': 'matched_before_sdk_launch', 'target_sdk_pin': dict(pin),
            'interpreter': sys.executable, 'python': python, 'checkout': checkout,
            'packages': observed_packages, 'module_origins': origins,
            'required_environment_found': found, 'contracts': dict(_SDK_CONTRACTS),
            'tracked_checkout_clean': True, 'source_untracked_clean': True}


def verify_recovery_sdk_identity(session_root, *, authorized_boot, recipe_sha256, learner_source_sha256):
    """Only an explicitly sealed SDK migration introduces the new identity gate."""
    root = Path(session_root)
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError('recovery SDK identity requires its actual session directory')
    marker_path, marker_pin_path = root / 'recovery.json', root / 'recovery.pin.json'
    if not os.path.lexists(marker_path) and not os.path.lexists(marker_pin_path):
        return None
    marker_pin = _sdk_document(marker_pin_path, limit=4096)
    if marker_pin.get('path') != str(marker_path):
        raise ValueError('recovery SDK marker escapes its target session')
    marker = _sdk_pin_document(marker_pin)
    source = marker.get('source')
    if not isinstance(source, dict):
        raise ValueError('recovery SDK marker lacks its sealed source')
    transition_pin, summary = source.get('sdk_transition_pin'), source.get('sdk_transition')
    if transition_pin is None and summary is None:
        return None
    if (marker.get('format') != 'private_failed_learning_fork'
            or marker.get('target_session') != str(root)
            or marker.get('authorized_boot') != authorized_boot
            or marker.get('target_recipe_sha256') != recipe_sha256
            or not isinstance(summary, dict) or summary.get('evidence_pin') != transition_pin):
        raise ValueError('recovery SDK transition is not bound to this sealed session')
    transition = _sdk_pin_document(transition_pin)
    fields = {'format', 'authorized_boot', 'source_checkpoint_sha256', 'source_recipe_sha256',
              'source_learner_source_sha256', 'target_learner_source_sha256', 'source_sdk_pin',
              'target_sdk_pin', 'physical_compatibility_pin'}
    if (set(transition) != fields or transition['format'] != 'private_sdk_mechanical_transition_v1'
            or transition['authorized_boot'] != authorized_boot
            or transition['target_learner_source_sha256'] != learner_source_sha256
            or summary.get('target_sdk_pin') != transition['target_sdk_pin']):
        raise ValueError('recovery SDK transition differs from its target source or inventory')
    receipt = verify_native_sdk_identity(transition['target_sdk_pin'])
    return dict(receipt, recovery_marker_pin=marker_pin, sdk_transition_pin=transition_pin)


class LabSimulation:
    """显式选择 Kit PhysX；GA 与 beta 的启动差异集中处理。"""

    def __init__(self, device: str):
        from isaaclab import app

        self._scope = None
        self._application = None
        self.physics = None
        self._fatal_shutdown_deferred = False
        self._cuda_index = None
        if device.startswith('cuda'):
            self._cuda_index = int(device.split(':', 1)[1]) if ':' in device else 0
        self.declarative_scene = callable(getattr(app, 'launch_simulation', None))
        arguments = dict(headless=True, device=device, enable_cameras=False, livestream=0)
        argv = sys.argv
        try:
            sys.argv = [argv[0]]
            if self.declarative_scene:
                from isaaclab_physx.physics import PhysxCfg

                # 显式 PhysxCfg 避免 GA 自动选择其他求解器或 kitless 后端。
                self._scope = app.launch_simulation(PhysxCfg(), arguments)
                self.physics = self._scope.__enter__()
            else:
                self._application = app.AppLauncher(**arguments).app
        finally:
            sys.argv = argv

    def configure_settings(self):
        from isaaclab.app import settings_manager

        if self.declarative_scene:
            settings = settings_manager.get_settings_manager()
            if self._cuda_index is not None:
                settings.set('/physics/cudaDevice', self._cuda_index)
                settings.set('/physics/suppressReadback', True)
        else:
            settings = settings_manager.SettingsManager.instance()
            settings.initialize_carb_settings()
            if self._cuda_index is not None:
                settings.set_int('/physics/cudaDevice', self._cuda_index)
                settings.set_bool('/physics/suppressReadback', True)

    def close(self):
        # 失败路径由 worker 先保存原异常并非零退出，不在这里吞掉异常。
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None
        elif self._application is not None:
            self._application.close()
            self._application = None

    def defer_fatal_shutdown(self, environment=None):
        """Keep native finalizers unreachable until the worker's fatal os._exit.

        GA's generator context can close Kit with exit0 on GeneratorExit. Retain
        both owners so stack unwinding cannot run that context or env.__del__.
        Successful runtime shutdown never enters this fatal-only registry.
        """
        if not self._fatal_shutdown_deferred:
            _FATAL_NATIVE_REFERENCES.append((self, environment))
            self._fatal_shutdown_deferred = True

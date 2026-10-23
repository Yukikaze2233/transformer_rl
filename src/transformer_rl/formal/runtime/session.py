"""Private structured sessions behind the existing terminal dashboard.

Packed sessions and historical research queues retain their original formats.
This adapter freezes a new source/recipe pair and gives a detached parent sole
ownership of its new child. The child still performs all checkpoint acceptance.
"""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid

from .interop import native_process_environment
from ...common.training_control import (PROTOCOL, TrainingControl, ManagedSession, _lease,
                               boot_id, command_lock, file_pin, process_identity,
                               read_json, timestamp, verify_pin, write_json)


class StructuredTrainingSession(ManagedSession):
    _FORMAT = 'private_structured_training_session'
    requires_parent_exit = True

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        pin = read_json(self.root / 'structured.pin.json')
        if not pin or pin.get('path') != str(self.root / 'structured.json'):
            raise ValueError('structured session definition is not pinned inside its own run')
        verify_pin(pin)
        self.spec = read_json(self.root / 'structured.json')
        if self.spec.get('format') != self._FORMAT:
            raise ValueError('not a private structured training session')

    @staticmethod
    def _bind_child_identity(child, argv, authorized_boot, *, timeout_s=5.):
        """Bind an actual owned child after exec, before retiring its identity.

        Linux can briefly expose an empty cmdline while exec replaces it. Two
        full matching observations are required; an incomplete launch never
        becomes a resumable parent receipt by borrowing a later status file.
        """
        deadline = time.monotonic() + timeout_s
        previous = None
        while time.monotonic() < deadline:
            if boot_id() != authorized_boot or child.poll() is not None:
                return None
            identity = process_identity(child.pid)
            if (identity and identity.get('boot_id') == authorized_boot
                    and identity.get('argv') == argv):
                if identity == previous:
                    return identity
                previous = identity
            else:
                previous = None
            time.sleep(.001)
        return None

    @staticmethod
    def source_sha256(package):
        table = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(package).rglob('*.py'))}
        return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @classmethod
    def create_from_settings(cls, root, settings, *, run_id, training_seed,
                             asset_manifest_sha256, task_contract_sha256, initial_model_sha256,
                             runtime_factory, runtime_configuration, resources, device='cuda:0',
                             python=sys.executable, learner_source=None, name=None):
        """把正式配置展开为完整 recipe，再冻结到独立会话；不启动 worker。"""
        from ..learning.settings import FormalTrainingSettings

        if not isinstance(settings, FormalTrainingSettings):
            settings = FormalTrainingSettings.load(settings)
        package = Path(learner_source).resolve(strict=True) if learner_source else Path(__file__).resolve().parents[2]
        recipe = settings.recipe(
            run_id, training_seed, asset_manifest_sha256=asset_manifest_sha256,
            learner_source_sha256=cls.source_sha256(package), task_contract_sha256=task_contract_sha256,
            initial_model_sha256=initial_model_sha256, device=device)
        return cls.create(root, recipe, runtime_factory=runtime_factory,
                          runtime_configuration=runtime_configuration, resources=resources,
                          python=python, learner_source=package, name=name)

    @classmethod
    def create(cls, root, recipe, *, runtime_factory, runtime_configuration, resources,
               python=sys.executable, learner_source=None, name=None, source_transition_required=False,
               failed_recovery_required=False):
        from ..learning.checkpoint import LearningRecipe

        if (not isinstance(recipe, LearningRecipe) or recipe.curriculum_promotion != 'exposure'
                or type(source_transition_required) is not bool
                or type(failed_recovery_required) is not bool
                or source_transition_required and failed_recovery_required
                or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', runtime_factory)):
            raise ValueError('structured session requires the complete exposure recipe and runtime factory')
        executable = Path(python).expanduser().absolute()
        package = Path(learner_source).resolve(strict=True) if learner_source else Path(__file__).resolve().parents[2]
        if not os.access(executable, os.X_OK) or cls.source_sha256(package) != recipe.learner_source_sha256:
            raise ValueError('structured executable or frozen learner source differs')
        cls._resource_definition(resources)
        root = Path(root).expanduser().absolute()
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        shutil.copytree(package, root / 'source/transformer_rl',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (root / 'attempts').mkdir()
        write_json(root / 'recipe.json', recipe.to_dict())
        write_json(root / 'runtime.json', runtime_configuration)
        pins = [file_pin(p.resolve()) for p in
                (root / 'recipe.json', root / 'runtime.json', *sorted((root / 'source').rglob('*.py')))]
        write_json(root / 'structured.json', {
            'format': cls._FORMAT, 'id': uuid.uuid4().hex, 'name': name or root.name,
            'created_boot': boot_id(), 'created_at': timestamp(), 'python': str(executable),
            'updates': recipe.journal_spec().target_rollouts, 'seed': recipe.training_seed,
            'architecture': getattr(recipe.policy, 'architecture', 'structured_transformer'),
            'history_length': recipe.policy.history_length,
            'num_envs': recipe.num_envs, 'rollout_steps': recipe.rollout_steps, 'device': recipe.device,
            'recipe_sha256': recipe.sha256, 'runtime_factory': runtime_factory,
            'resources': resources, 'pins': pins, 'automatic_evaluation': False,
            **({'source_transition_required': True} if source_transition_required else {}),
            **({'failed_recovery_required': True} if failed_recovery_required else {})})
        write_json(root / 'structured.pin.json', file_pin(root.resolve() / 'structured.json'))
        return cls(root)

    @staticmethod
    def _resource_definition(value):
        required = {'gpu_uuid', 'nvidia_smi', 'driver_library_directory', 'minimum_ram_bytes',
                    'minimum_vram_mib', 'minimum_disk_bytes', 'locks'}
        if (not isinstance(value, dict) or set(value) != required
                or not re.fullmatch(r'GPU-[0-9a-f-]{36}', value['gpu_uuid'])
                or any(not Path(value[k]).is_absolute() for k in ('nvidia_smi', 'driver_library_directory'))
                or any(type(value[k]) is not int or value[k] <= 0
                       for k in ('minimum_ram_bytes', 'minimum_vram_mib', 'minimum_disk_bytes'))
                or value['minimum_ram_bytes'] < 7 * 1024**3 or value['minimum_vram_mib'] < 14336
                or value['minimum_disk_bytes'] < 10 * 1024**3
                or not isinstance(value['locks'], list) or not value['locks']):
            raise ValueError('structured admission needs explicit fixed-GPU, storage and original lock inputs')
        for item in value['locks']:
            if (not isinstance(item, dict) or set(item) != {'path', 'device', 'inode', 'uid'}
                    or not Path(item['path']).is_absolute()
                    or any(type(item[k]) is not int or item[k] < 0 for k in ('device', 'inode', 'uid'))):
                raise ValueError('resource lease definition must retain its original file identity')

    def _resources(self):
        gate = self.spec['resources']
        self._resource_definition(gate)
        memory = {line.split(':')[0]: int(line.split(':')[1].split()[0]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.split(':')[0] in ('MemAvailable', 'SwapFree')}
        gpu = subprocess.run([gate['nvidia_smi'], '-i', gate['gpu_uuid'],
                              '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=8, check=True)
        values = [v.strip() for v in gpu.stdout.strip().split(',')]
        if len(values) != 2 or values[0] != gate['gpu_uuid']:
            raise ValueError('fixed-GPU resource response differs')
        disk = os.statvfs(self.root)
        observed = {'at': timestamp(), **memory, 'gpu_uuid': values[0], 'gpu_free_mib': int(values[1]),
                    'disk_available_bytes': disk.f_bavail * disk.f_frsize,
                    'minimum_ram_available_bytes': gate['minimum_ram_bytes'],
                    'minimum_gpu_free_mib': gate['minimum_vram_mib'],
                    'minimum_disk_available_bytes': gate['minimum_disk_bytes']}
        ready = (memory['MemAvailable'] >= gate['minimum_ram_bytes']
                 and int(values[1]) >= gate['minimum_vram_mib']
                 and observed['disk_available_bytes'] >= gate['minimum_disk_bytes'])
        return ready, observed

    def status(self):
        state = super().status()
        admission = read_json(self.root / 'admission.json')
        if admission:
            state['resources'] = admission['resources']
            if (not admission['ready'] and admission['boot'] == boot_id() and not state['active']
                    and state['phase'] in ('ready', 'stopped', 'waiting_resources')):
                state['phase'] = 'waiting_resources'
        state['automatic_evaluation'] = False
        state.update(structured=True, history_length=self.spec['history_length'],
                     num_envs=self.spec['num_envs'], resources_resume_automatic=False)
        return state

    def start(self, *, paused=False, accept_new_boot=False, inherited_fds=()):
        with command_lock(self.root), ExitStack() as leases:
            state = self.status()
            if state['active'] or state['phase'] == 'completed':
                return state
            previous = self.attempt()
            expected = read_json(previous / 'request.json')['boot_id'] if previous else self.spec['created_boot']
            if boot_id() != expected and not accept_new_boot:
                raise ValueError('boot changed; explicit current-boot authorization is required')
            for pin in self.spec['pins']:
                path = Path(pin['path'])
                if path not in (self.root / 'recipe.json', self.root / 'runtime.json') and not path.is_relative_to(self.root / 'source'):
                    raise ValueError('structured source pin escapes its run')
                verify_pin(pin)
            start, spent, transitions, resume_pin = 0, 0., 0, None
            if previous:
                final, exited = read_json(previous / 'control/result.json'), read_json(previous / 'exit.json')
                if (not final or final['summary'].get('phase') != 'stopped' or not exited
                        or exited.get('returncode') != 0 or exited.get('identity') != final['summary'].get('identity')
                        or exited.get('token') != final.get('token')):
                    raise ValueError('unclean structured attempt cannot roll back or replay consumed work')
                start, spent, transitions = (final['summary'][key] for key in
                                            ('update', 'session_active_elapsed_s', 'cumulative_transitions'))
                resume_pin = file_pin(previous / 'exit.json')
            elif self.spec.get('source_transition_required'):
                from .transition import transition_metadata

                inherited = transition_metadata(self.root)
                start, spent, transitions = (inherited[key] for key in
                                             ('update', 'active_elapsed_s', 'returned_transitions'))
            elif self.spec.get('failed_recovery_required'):
                from .recovery import recovery_metadata

                inherited = recovery_metadata(self.root)
                start, spent, transitions = (inherited[key] for key in
                                             ('update', 'active_elapsed_s', 'returned_transitions'))
            descriptor = _lease(self.root)
            leases.callback(os.close, descriptor)
            held = []
            for item in self.spec['resources']['locks']:
                fd = os.open(item['path'], os.O_RDWR | os.O_NOFOLLOW)
                leases.callback(os.close, fd)
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode)
                        or (info.st_dev, info.st_ino, info.st_uid) != (item['device'], item['inode'], item['uid'])):
                    raise ValueError('original structured resource lease identity changed')
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append({'fd': fd, 'device': info.st_dev, 'inode': info.st_ino})
            ready, observed = self._resources()
            write_json(self.root / 'admission.json', {'ready': ready, 'boot': boot_id(),
                       'resources': observed, 'whole_system_gpu_exclusive': False}, replace=True)
            if not ready:
                # This call still owns a temporary admission lease. Querying
                # status here would mistake that lease for a closing learner.
                return dict(state, phase='waiting_resources', active=False, resources=observed)
            generation = len(list((self.root / 'attempts').iterdir())) + 1
            attempt = self.root / 'attempts' / f'{generation:04d}-{uuid.uuid4().hex[:12]}'
            (attempt / 'control').mkdir(parents=True)
            binding = {'protocol': PROTOCOL, 'token': uuid.uuid4().hex, 'boot_id': boot_id(),
                       'submitted_at': timestamp(), 'submitted_epoch': time.time(), 'lease_fd': descriptor,
                       'root': str(self.root), 'attempt': str(attempt), 'inherited_locks': held,
                       'start_update': start, 'prior_active_elapsed_s': spent,
                       'prior_transitions': transitions, 'target_update': self.spec['updates']}
            write_json(attempt / 'request.json', binding)
            write_json(attempt / 'control/request.json', {'token': binding['token'], 'sequence': 0,
                       'operation': 'pause' if paused else 'resume', 'at': timestamp()})
            if resume_pin:
                write_json(attempt / 'resume_exit.pin.json', resume_pin)
                write_json(attempt / 'resume.json', {'previous_attempt': str(previous)})
            write_json(self.root / 'current.json', {'attempt': attempt.name}, replace=True)
            env = dict(os.environ, PYTHONPATH=str(self.root / 'source'), PYTHONUNBUFFERED='1',
                       CUDA_VISIBLE_DEVICES=self.spec['resources']['gpu_uuid'],
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            env = native_process_environment(env, driver_library_directory=self.spec['resources']['driver_library_directory'])
            argv = [self.spec['python'], '-m', 'transformer_rl.formal.runtime.session', str(self.root), str(attempt)]
            with (attempt / 'parent.log').open('xb') as log:
                child = subprocess.Popen(argv, env=env, cwd=self.root, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(descriptor, *(v['fd'] for v in held), *inherited_fds))
            self.submitted_process = child
            identity = self._bind_child_identity(child, argv, binding['boot_id'])
            if identity:
                write_json(attempt / 'launch.json', identity)
            return self.status()

    def supervise(self, attempt):
        """Own exactly this new learner child through actual waitpid."""
        attempt = Path(attempt).resolve(strict=True)
        if attempt != self.attempt():
            raise ValueError('structured supervisor must own the current new attempt')
        control = TrainingControl(attempt / 'control')
        binding = control.binding
        control.publish('initializing', supervisor_identity=process_identity(os.getpid()))
        argv = [self.spec['python'], '-m', 'transformer_rl.formal.runtime.process',
                '--recipe', str(self.root / 'recipe.json'), '--control-dir', str(attempt / 'control'),
                '--runtime-factory', self.spec['runtime_factory'], '--runtime-config', str(self.root / 'runtime.json')]
        resume = read_json(attempt / 'resume.json')
        if resume:
            argv += ['--resume-attempt', resume['previous_attempt'],
                     '--parent-exit-pin', str(attempt / 'resume_exit.pin.json')]
        descriptors = (binding['lease_fd'], *(v['fd'] for v in binding['inherited_locks']))
        with (attempt / 'worker.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, pass_fds=descriptors)
            identity = self._bind_child_identity(child, argv, binding['boot_id'])
            if identity:
                write_json(attempt / 'worker_launch.json', identity)
            code = child.wait()
        write_json(attempt / 'exit.json', {'format': 'private_structured_child_exit',
                   'token': binding['token'], 'boot': binding['boot_id'], 'attempt': str(attempt),
                   'identity': identity, 'returncode': code})
        return code if identity is not None else (code or 2)


if __name__ == '__main__':
    raise SystemExit(StructuredTrainingSession(sys.argv[1]).supervise(sys.argv[2]))

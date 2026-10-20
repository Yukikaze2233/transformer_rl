"""保留原生物理错误，在采样提交前拒绝受损的物理区间。"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import uuid


class NativePhysicsHealth:
    """Read native output at learning boundaries without simulation callbacks.

    The callback adapter remains available for synthetic interfaces. Native
    workers read their inherited, dedicated stdout/stderr files instead.
    """

    def __init__(self, logging=None, *, log_path=None):
        if (logging is None) == (log_path is None):
            raise ValueError('physics health requires exactly one log source')
        self._logging = logging
        self.first_error = None
        self.messages_seen = 0
        self._handle = None
        self._logs = []
        self._probe = None
        self._probe_seen = False
        if logging is not None:
            self._handle = logging.add_logger(self._on_message)
        else:
            self._open_log(Path(log_path))

    def _open_log(self, path):
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
            os.close(descriptor)
            raise ValueError('physics log must be an owned regular file')
        self._logs.append({'path': path, 'descriptor': descriptor,
                           'identity': (value.st_dev, value.st_ino), 'cursor': 0, 'pending': b''})

    @classmethod
    def for_native_process(cls, directory):
        """Bind only this worker's existing dedicated output files."""
        directory = Path(directory).resolve(strict=True)
        health = None
        seen = set()
        try:
            for descriptor in (2, 1):
                value = os.fstat(descriptor)
                path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
                if (not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid()
                        or not path.is_absolute() or not path.is_relative_to(directory)):
                    raise ValueError('native physics requires dedicated stdout/stderr files inside its run directory')
                identity = value.st_dev, value.st_ino
                if identity in seen:
                    continue
                if health is None:
                    health = cls(log_path=path)
                else:
                    health._open_log(path)
                if health._logs[-1]['identity'] != identity:
                    raise RuntimeError('native output file changed while binding its inherited descriptor')
                seen.add(identity)
            health._probe = 'native physics stream monitor ' + uuid.uuid4().hex
            os.write(2, ('[python] ' + health._probe + '\n').encode())
            health.check()
            if not health._probe_seen:
                raise RuntimeError('native output monitor did not observe its descriptor probe')
            return health
        except BaseException:
            if health is not None:
                health.close()
            raise

    def _on_message(self, source, level, filename, line, message):
        self.messages_seen += 1
        if self.first_error is None and 'physx' in source.lower() and (
                'PhysX error:' in message or 'simulation will miss interactions' in message):
            self.first_error = message[:2048]

    def check(self):
        for source in self._logs:
            current = source['path'].stat(follow_symlinks=False)
            size = os.fstat(source['descriptor']).st_size
            if ((current.st_dev, current.st_ino) != source['identity']
                    or not stat.S_ISREG(current.st_mode) or size < source['cursor']):
                raise RuntimeError('native physics log was replaced or truncated')
            length = size - source['cursor']
            if length > 8 * 1024 * 1024:
                raise RuntimeError('native physics log exceeded its boundary read limit')
            raw = os.pread(source['descriptor'], length, source['cursor'])
            if len(raw) != length:
                raise RuntimeError('native physics log changed during its boundary read')
            source['cursor'] = size
            lines = (source['pending'] + raw).split(b'\n')
            source['pending'] = lines.pop()
            if len(source['pending']) > 65536:
                raise RuntimeError('native physics log contains an oversized partial line')
            self.messages_seen += len(lines)
            for line in (*lines, source['pending']):
                text = line.decode(errors='replace')
                self._probe_seen |= self._probe is not None and self._probe in text
                if (self.first_error is None and re.search(r'\[[^\]]*physx[^\]]*\]', text, re.I)
                        and ('PhysX error:' in text or 'simulation will miss interactions' in text)):
                    self.first_error = text[:2048]
        if self.first_error is not None:
            raise RuntimeError('native physics reported an invalid interval: ' + self.first_error)

    def close(self):
        if self._handle is not None:
            self._logging.remove_logger(self._handle)
            self._handle = None
        for source in self._logs:
            os.close(source['descriptor'])
        self._logs.clear()

"""保留原生物理错误，在采样提交前拒绝受损的物理区间。"""
from __future__ import annotations


class NativePhysicsHealth:
    """日志回调只记录首个错误；异常在采样线程抛出，不在SDK回调中抛出。"""

    def __init__(self, logging):
        self._logging = logging
        self.first_error = None
        self._handle = logging.add_logger(self._on_message)

    def _on_message(self, source, level, filename, function, line, message):
        if self.first_error is None and 'physx' in source.lower() and (
                'PhysX error:' in message or 'simulation will miss interactions' in message):
            self.first_error = message[:2048]

    def check(self):
        if self.first_error is not None:
            raise RuntimeError('native physics reported an invalid interval: ' + self.first_error)

    def close(self):
        if self._handle is not None:
            self._logging.remove_logger(self._handle)
            self._handle = None

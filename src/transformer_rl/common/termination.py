"""信号只设置停止标记，在完整采样或更新边界退出。"""
from __future__ import annotations

import signal
import time


class _StopBudget:
    """Flag-only signals and a soft deadline checked at collection/update boundaries."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds
        self.reason: str | None = None
        self.previous: dict[int, object] = {}

    def _signal(self, number, _frame):
        self.reason = signal.Signals(number).name

    def install(self) -> None:
        for number in (signal.SIGINT, signal.SIGTERM):
            if number not in self.previous:
                self.previous[number] = signal.getsignal(number)
            signal.signal(number, self._signal)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *_):
        for number, handler in self.previous.items():
            signal.signal(number, handler)

    def stopped(self) -> bool:
        if self.reason is None and time.monotonic() >= self.deadline:
            self.reason = "time_budget"
        return self.reason is not None

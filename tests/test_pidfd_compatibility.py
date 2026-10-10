"""Kernel process handles remain mandatory when userspace wrappers are absent."""
import ctypes
import errno
import os
import platform
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from transformer_rl import retention_campaign


def absent_python_wrappers(monkeypatch):
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)


def test_available_python_wrappers_take_precedence(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: calls.append(("open", pid)) or 17, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: calls.append(("send", fd, sig)), raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("unexpected libc fallback"))
    opening, sending = retention_campaign._pidfd_api()
    assert opening(123) == 17
    sending(17, 0)
    assert calls == [("open", 123), ("send", 17, 0)]


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"),
                    reason="actual syscall test requires a declared Linux 64-bit ABI")
def test_actual_kernel_syscalls_signal_only_the_owned_child_without_wrappers(monkeypatch):
    actual = ctypes.CDLL(None, use_errno=True)
    absent_python_wrappers(monkeypatch)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(syscall=actual.syscall))
    opening, sending = retention_campaign._pidfd_api()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    descriptor = None
    try:
        descriptor = opening(child.pid)
        sending(descriptor, 0)
        assert child.poll() is None
        sending(descriptor, signal.SIGTERM)
        assert child.wait(timeout=5) == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            sending(descriptor, 0)
    finally:
        if child.poll() is None:
            if descriptor is not None:
                sending(descriptor, signal.SIGKILL)
            else:
                child.kill()
            child.wait(timeout=5)
        if descriptor is not None:
            os.close(descriptor)


@pytest.mark.parametrize("number", (errno.ENOSYS, errno.EPERM))
def test_syscall_errors_propagate_and_never_use_numeric_pid_signals(monkeypatch, number):
    absent_python_wrappers(monkeypatch)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(retention_campaign.sys, "platform", "linux")
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("numeric PID signal is forbidden"))
    def unavailable(*args):
        ctypes.set_errno(number)
        return -1
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(syscall=unavailable))
    opening, sending = retention_campaign._pidfd_api()
    for call, arguments in ((opening, (123,)), (sending, (17, 0))):
        with pytest.raises(OSError) as error:
            call(*arguments)
        assert error.value.errno == number


def test_unknown_abi_is_rejected_before_any_syscall(monkeypatch):
    absent_python_wrappers(monkeypatch)
    monkeypatch.setattr(platform, "machine", lambda: "unknown")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(
        syscall=lambda *args: pytest.fail("unknown ABI cannot issue a syscall")))
    with pytest.raises(OSError, match="LP64"):
        retention_campaign._pidfd_api()

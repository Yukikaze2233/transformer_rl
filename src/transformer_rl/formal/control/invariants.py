"""Device-side value invariants without a host synchronization per check.

A value check such as ``isfinite(x).all()`` ends in a Python bool, which
forces the CPU to wait for the GPU. On the per-substep control path that is
several synchronizations per physics step. Owners running inside a policy
interval record violations in a ``DeferredInvariants`` instead; the
environment reads them once per interval and raises the same error with every
violated label. Layout checks (shape, dtype, device) stay immediate.
"""
from __future__ import annotations

import torch


class DeferredInvariants:
    """Violated value invariants collected on the device, read once per interval."""

    CAPACITY = 64

    def __init__(self, device):
        self.device = torch.device(device)
        self._labels: dict[str, int] = {}
        self._flags = torch.zeros(self.CAPACITY, dtype=torch.bool, device=self.device)

    def require(self, label, violated):
        """Record ``violated`` (a bool tensor on this device) under ``label`` without reading it."""
        index = self._labels.setdefault(label, len(self._labels))
        if index >= self.CAPACITY:
            raise RuntimeError('deferred invariants exceed their label capacity')
        self._flags[index] |= violated.any()

    def raise_if_violated(self):
        """One host read: raise every invariant violated since the last call."""
        if bool(self._flags.any()):
            flags = self._flags.tolist()
            failed = [label for label, index in self._labels.items() if flags[index]]
            self._flags.zero_()
            raise ValueError('control invariants violated: ' + '; '.join(failed))


def require(invariants: DeferredInvariants | None, label, violated):
    """Raise now, or record the violation when ``invariants`` defers checks to the interval end."""
    if invariants is None:
        if bool(violated.any()):
            raise ValueError(label)
    else:
        invariants.require(label, violated)

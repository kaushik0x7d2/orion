"""
Memory management for FHE tensor objects.

Provides context managers and utilities for deterministic cleanup of
Go-side ciphertext and plaintext objects. Python's GC is non-deterministic,
so without explicit cleanup the Go heap grows unbounded during batch
inference or server workloads.

Usage:
    # Context manager for automatic cleanup:
    with managed_cipher(scheme, ctxt_ids, shape) as ctxt:
        result = model(ctxt)
    # ctxt's Go-side ciphertexts are freed here

    # Memory monitor:
    stats = get_memory_stats(backend)
    print(f"Live ciphertexts: {stats['ciphertexts']}")

    # Bulk cleanup:
    cleanup_all(backend)
"""

import logging
from contextlib import contextmanager

logger = logging.getLogger("orion.memory")


class ManagedCipherTensor:
    """
    CipherTensor wrapper with guaranteed cleanup.

    Use as a context manager to ensure Go-side ciphertexts are freed
    when the block exits, even if an exception occurs.
    """
    def __init__(self, cipher_tensor):
        self._ct = cipher_tensor
        self._released = False

    def __enter__(self):
        return self._ct

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def release(self):
        """Explicitly free all Go-side ciphertexts."""
        if self._released:
            return
        self._released = True
        try:
            for idx in self._ct.ids:
                self._ct.backend.DeleteCiphertext(idx)
            # Prevent __del__ from double-freeing
            self._ct.ids = []
            logger.debug("Released %d ciphertexts", len(self._ct.ids))
        except Exception:
            pass

    @property
    def tensor(self):
        return self._ct

    # Delegate attribute access to the underlying CipherTensor
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._ct, name)


class ManagedPlainTensor:
    """PlainTensor wrapper with guaranteed cleanup."""
    def __init__(self, plain_tensor):
        self._pt = plain_tensor
        self._released = False

    def __enter__(self):
        return self._pt

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            for idx in self._pt.ids:
                self._pt.backend.DeletePlaintext(idx)
            self._pt.ids = []
        except Exception:
            pass

    @property
    def tensor(self):
        return self._pt

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._pt, name)


@contextmanager
def managed_cipher(cipher_tensor):
    """
    Context manager that ensures a CipherTensor's Go-side memory is freed.

    Args:
        cipher_tensor: A CipherTensor instance.

    Yields:
        The same CipherTensor, freed on block exit.

    Example:
        ptxt = orion.encode(data, level)
        ctxt = orion.encrypt(ptxt)
        with managed_cipher(ctxt) as ct:
            result = model(ct)
        # ct's Go-side memory is now freed
    """
    managed = ManagedCipherTensor(cipher_tensor)
    try:
        yield cipher_tensor
    finally:
        managed.release()


@contextmanager
def managed_plain(plain_tensor):
    """Context manager for PlainTensor cleanup."""
    managed = ManagedPlainTensor(plain_tensor)
    try:
        yield plain_tensor
    finally:
        managed.release()


def get_memory_stats(backend) -> dict:
    """
    Get current Go-side memory statistics.

    Args:
        backend: A LattigoLibrary instance.

    Returns:
        dict with live object counts.
    """
    try:
        live_pt = backend.GetLivePlaintexts()
        live_ct = backend.GetLiveCiphertexts()
        pt_count = len(live_pt) if isinstance(live_pt, list) else 0
        ct_count = len(live_ct) if isinstance(live_ct, list) else 0
    except Exception:
        pt_count = -1
        ct_count = -1

    return {
        "plaintexts": pt_count,
        "ciphertexts": ct_count,
        "total": pt_count + ct_count if pt_count >= 0 else -1,
    }


def cleanup_all(backend):
    """
    Free ALL live plaintexts and ciphertexts in the Go heap.

    WARNING: This invalidates all existing PlainTensor/CipherTensor
    objects. Use only during shutdown or between independent workloads.

    Args:
        backend: A LattigoLibrary instance.
    """
    freed = 0
    try:
        live_pt = backend.GetLivePlaintexts()
        if isinstance(live_pt, list):
            for pt_id in live_pt:
                backend.DeletePlaintext(pt_id)
                freed += 1
    except Exception:
        pass

    try:
        live_ct = backend.GetLiveCiphertexts()
        if isinstance(live_ct, list):
            for ct_id in live_ct:
                backend.DeleteCiphertext(ct_id)
                freed += 1
    except Exception:
        pass

    logger.info("Cleaned up %d Go-side objects", freed)
    return freed


class MemoryTracker:
    """
    Tracks Go-side memory usage over time for monitoring.

    Usage:
        tracker = MemoryTracker(backend)
        tracker.snapshot("before_inference")
        # ... do inference ...
        tracker.snapshot("after_inference")
        tracker.report()
    """
    def __init__(self, backend):
        self.backend = backend
        self.snapshots = []

    def snapshot(self, label=""):
        stats = get_memory_stats(self.backend)
        stats["label"] = label
        self.snapshots.append(stats)
        return stats

    def report(self):
        lines = ["Memory Usage Report:"]
        for snap in self.snapshots:
            label = snap.get("label", "?")
            lines.append(
                f"  [{label}] plaintexts={snap['plaintexts']}, "
                f"ciphertexts={snap['ciphertexts']}, total={snap['total']}"
            )
        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            delta = last["total"] - first["total"]
            lines.append(f"  Net change: {delta:+d} objects")
        return "\n".join(lines)

    def reset(self):
        self.snapshots = []

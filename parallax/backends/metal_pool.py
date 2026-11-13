"""Metal buffer pool and zero-copy view cache."""

from __future__ import annotations

import sys
import threading
from typing import Any, Optional
import numpy as np
from parallax.metal_engine import is_metal_available


class MetalBufferPool:
    """Thread-safe reusable Metal buffer pool."""

    def __init__(self, max_cached_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_cached_bytes = max_cached_bytes
        self._lock = threading.Lock()
        self._pool: dict[int, list[Any]] = {}
        self._allocated_bytes = 0
        self._cached_bytes = 0
        self._hits = 0
        self._misses = 0

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    @staticmethod
    def _bucket_size(size: int) -> int:
        if size <= 256:
            return 256
        return 1 << (size - 1).bit_length()

    def _allocate_raw(self, size: int) -> Optional[Any]:
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        buf = me._objc_send(
            me.ctypes.c_void_p, me._DEVICE, "newBufferWithLength:options:", size, 0
        )
        if buf:
            self._allocated_bytes += size
        return buf

    def _free_raw(self, buffer: Any, size: int) -> None:
        import parallax.metal_engine as me

        try:
            if me._objc_send is not None:
                me._objc_send(None, buffer, "release")
        except Exception:
            pass
        self._allocated_bytes = max(0, self._allocated_bytes - size)

    def acquire(self, size_bytes: int) -> Optional[Any]:
        if not self.available() or size_bytes <= 0:
            return None
        bucket = self._bucket_size(size_bytes)
        with self._lock:
            cached_list = self._pool.get(bucket)
            if cached_list:
                buf = cached_list.pop()
                self._cached_bytes -= bucket
                self._hits += 1
                return buf
            self._misses += 1
        return self._allocate_raw(bucket)

    def release(self, buffer: Any, size_bytes: int) -> None:
        if buffer is None or size_bytes <= 0:
            return
        bucket = self._bucket_size(size_bytes)
        with self._lock:
            if self._cached_bytes + bucket <= self.max_cached_bytes:
                self._pool.setdefault(bucket, []).append(buffer)
                self._cached_bytes += bucket
                return
        self._free_raw(buffer, bucket)

    def clear(self) -> None:
        with self._lock:
            for bucket, bufs in self._pool.items():
                for b in bufs:
                    self._free_raw(b, bucket)
            self._pool.clear()
            self._cached_bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            cached_count = sum(len(b) for b in self._pool.values())
            return {
                "hits": self._hits,
                "misses": self._misses,
                "cached_count": cached_count,
                "cached_bytes": self._cached_bytes,
                "allocated_bytes": self._allocated_bytes,
            }


class ZeroCopyViewCache:
    """Cache mapping ndarray data pointers to zero-copy Metal buffers."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[Any, int]] = {}

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    def _wrap_array(self, array: np.ndarray) -> Optional[Any]:
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        return me._objc_send(
            me.ctypes.c_void_p,
            me._DEVICE,
            "newBufferWithBytes:length:options:",
            array.ctypes.data_as(me.ctypes.c_void_p),
            array.nbytes,
            0,
        )

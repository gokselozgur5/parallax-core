"""Metal buffer pool and zero-copy view cache."""

from __future__ import annotations

import sys
import threading
import weakref
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
    """Cache mapping live ndarray views to zero-copy Metal buffers."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._dead: list[tuple[tuple[int, int], Any]] = []

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    @staticmethod
    def _owner(array: np.ndarray) -> np.ndarray:
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        return owner

    def _drain_dead_locked(self) -> list[Any]:
        stale: list[Any] = []
        while self._dead:
            key, dead_ref = self._dead.pop()
            entry = self._cache.get(key)
            if entry is not None and entry[1] is dead_ref:
                del self._cache[key]
                stale.append(entry[0])
        return stale

    def _wrap_array(self, array: np.ndarray) -> Optional[Any]:
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None
        return me._objc_send(
            me.ctypes.c_void_p,
            me._DEVICE,
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            array.ctypes.data_as(me.ctypes.c_void_p),
            array.nbytes,
            0,
            None,
        )

    def _release_raw(self, buffer: Any) -> None:
        import parallax.metal_engine as me

        try:
            if me._objc_send is not None and buffer:
                me._objc_send(None, buffer, "release")
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._dead.clear()
            bufs = [buf for buf, _ in self._cache.values()]
            self._cache.clear()
        for buf in bufs:
            self._release_raw(buf)

    def count(self) -> int:
        with self._lock:
            stale = self._drain_dead_locked()
            total = len(self._cache)
        for buf in stale:
            self._release_raw(buf)
        return total

    def get_or_create(self, array: np.ndarray) -> Optional[Any]:
        if not self.available() or array.size == 0 or not array.flags.c_contiguous:
            return None
        key = (int(array.ctypes.data), int(array.nbytes))
        owner = self._owner(array)
        hit: Optional[Any] = None
        with self._lock:
            stale = self._drain_dead_locked()
            entry = self._cache.get(key)
            if entry is not None:
                if entry[1]() is owner:
                    hit = entry[0]
                else:
                    del self._cache[key]
                    stale.append(entry[0])
        for b in stale:
            self._release_raw(b)
        if hit is not None:
            return hit
        buf = self._wrap_array(array)
        if buf is None or self.max_entries <= 0:
            return buf

        def _on_collect(dead_ref: Any, k: tuple[int, int] = key) -> None:
            self._dead.append((k, dead_ref))
            if self._lock.acquire(blocking=False):
                try:
                    evicted_dead = self._drain_dead_locked()
                finally:
                    self._lock.release()
                for b in evicted_dead:
                    self._release_raw(b)

        owner_ref = weakref.ref(owner, _on_collect)
        to_free: list[Any] = []
        with self._lock:
            to_free.extend(self._drain_dead_locked())
            existing = self._cache.get(key)
            if existing is not None and existing[1]() is owner:
                to_free.append(buf)
                buf = existing[0]
            else:
                if existing is not None:
                    del self._cache[key]
                    to_free.append(existing[0])
                if self._cache and len(self._cache) >= self.max_entries:
                    old_key, (old_buf, _) = next(iter(self._cache.items()))
                    del self._cache[old_key]
                    to_free.append(old_buf)
                self._cache[key] = (buf, owner_ref)
        for b in to_free:
            self._release_raw(b)
        return buf

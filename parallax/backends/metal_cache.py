"""Metal pipeline compilation cache."""

from __future__ import annotations

import hashlib
import sys
import threading
from typing import Any, Optional

from parallax.metal_engine import is_metal_available


class MetalPipelineCache:
    """Thread-safe cache for compiled Metal compute pipeline states."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Any] = {}
        self._hits = 0
        self._misses = 0

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()

    @staticmethod
    def _make_key(func_name: str, source: str) -> tuple[str, str]:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        return (func_name, digest)

    def _release_pso(self, pso: Any) -> None:
        import parallax.metal_engine as me

        try:
            if me._objc_send is not None and pso:
                me._objc_send(None, pso, "release")
        except Exception:
            pass

    def _compile_lib(self, src: str) -> tuple[Optional[Any], Optional[Any]]:
        import ctypes
        import parallax.metal_engine as me

        if me._DEVICE is None or me._objc_send is None:
            return None, None
        err = ctypes.c_void_p()
        s_cf = me._cf_string(src)
        lib = me._objc_send(
            ctypes.c_void_p,
            me._DEVICE,
            "newLibraryWithSource:options:error:",
            s_cf,
            None,
            ctypes.byref(err),
        )
        return lib, s_cf

    def _compile_pipeline(self, func_name: str, src: str) -> Optional[Any]:
        import ctypes
        import parallax.metal_engine as me

        lib, s_cf = self._compile_lib(src)
        if not lib:
            me._cf_release(s_cf)
            return None
        f_cf = me._cf_string(func_name)
        fn = me._objc_send(ctypes.c_void_p, lib, "newFunctionWithName:", f_cf)
        pso = None
        err = ctypes.c_void_p()
        if fn:
            pso = me._objc_send(
                ctypes.c_void_p,
                me._DEVICE,
                "newComputePipelineStateWithFunction:error:",
                fn,
                ctypes.byref(err),
            )
        me._cf_release(s_cf, f_cf)
        me._release_buffers(fn, lib)
        return pso

    def has(self, func_name: str, source: str) -> bool:
        key = self._make_key(func_name, source)
        with self._lock:
            return key in self._cache

    def get_or_compile(
        self, func_name: str, source: str
    ) -> Optional[Any]:
        if not self.available() or not func_name or not source:
            return None
        key = self._make_key(func_name, source)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
        pso = self._compile_pipeline(func_name, source)
        if pso is None:
            return None
        with self._lock:
            if len(self._cache) >= self.max_entries:
                old_k, old_p = next(iter(self._cache.items()))
                del self._cache[old_k]
                self._release_pso(old_p)
            self._cache[key] = pso
        return pso

    def clear(self) -> None:
        with self._lock:
            for pso in self._cache.values():
                self._release_pso(pso)
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "count": len(self._cache),
            }

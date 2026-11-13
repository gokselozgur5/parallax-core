"""AVX-512 SGEMM microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
from parallax.native.loader import bind

_FN_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512: Optional[Callable[..., Any]] = None


def _get_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_AVX512
    if _FN_AVX512 is None:
        args = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        _FN_AVX512 = bind("c_gemm_avx512_f32", args, ctypes.c_int)
    return _FN_AVX512


def is_avx512_available() -> bool:
    """Return True if AVX-512 is supported by the CPU."""
    global _FN_HAS_AVX512
    if _FN_HAS_AVX512 is None:
        _FN_HAS_AVX512 = bind("c_has_avx512_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512() != 0) if _FN_HAS_AVX512 else False

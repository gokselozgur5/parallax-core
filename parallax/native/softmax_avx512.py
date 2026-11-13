"""AVX-512 softmax microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
from parallax.native.loader import bind

_FN_SOFTMAX_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_SOFTMAX: Optional[Callable[..., Any]] = None


def _get_softmax_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_SOFTMAX_AVX512
    if _FN_SOFTMAX_AVX512 is None:
        args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        _FN_SOFTMAX_AVX512 = bind("c_softmax_avx512_f32", args, ctypes.c_int)
    return _FN_SOFTMAX_AVX512


def is_avx512_softmax_available() -> bool:
    """Return True if AVX-512 softmax is supported by the CPU."""
    global _FN_HAS_AVX512_SOFTMAX
    if _FN_HAS_AVX512_SOFTMAX is None:
        _FN_HAS_AVX512_SOFTMAX = bind("c_has_avx512_softmax_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512_SOFTMAX() != 0) if _FN_HAS_AVX512_SOFTMAX else False

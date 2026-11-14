"""AVX-512 RMSNorm microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
from parallax.native.loader import bind

_FN_RMSNORM_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_RMSNORM: Optional[Callable[..., Any]] = None


def is_avx512_rmsnorm_available() -> bool:
    """Return True if AVX-512 RMSNorm is supported by the CPU."""
    global _FN_HAS_AVX512_RMSNORM
    if _FN_HAS_AVX512_RMSNORM is None:
        _FN_HAS_AVX512_RMSNORM = bind("c_has_avx512_rmsnorm_f32", [], ctypes.c_int)
    return bool(_FN_HAS_AVX512_RMSNORM() != 0) if _FN_HAS_AVX512_RMSNORM else False

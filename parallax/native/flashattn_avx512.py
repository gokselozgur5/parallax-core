"""AVX-512 FlashAttention-1 single-head microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_FLASHATTN_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_FLASHATTN: Optional[Callable[..., Any]] = None


def _get_flashattn_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_FLASHATTN_AVX512
    if _FN_FLASHATTN_AVX512 is None:
        args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
        ]
        _FN_FLASHATTN_AVX512 = bind("c_flashattn_avx512_f32", args, ctypes.c_int)
    return _FN_FLASHATTN_AVX512

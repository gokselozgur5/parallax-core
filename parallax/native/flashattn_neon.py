"""Python wrapper for the NEON FlashAttention-1 single-head microkernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional

import numpy as np

from parallax.native.loader import bind

_FN_FLASHATTN_NEON: Optional[Callable[..., Any]] = None


def _get_flashattn_neon_fn() -> Optional[Callable[..., Any]]:
    global _FN_FLASHATTN_NEON
    if _FN_FLASHATTN_NEON is None:
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
        _FN_FLASHATTN_NEON = bind("c_flashattn_neon_f32", args, ctypes.c_int)
    return _FN_FLASHATTN_NEON

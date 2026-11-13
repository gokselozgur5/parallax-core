"""Python wrapper for the native C SwiGLU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import bind

_SWIGLU_FN: Optional[Callable[..., Any]] = None
_SWIGLU_INIT_ATTEMPTED: bool = False


def _get_swiglu_fn() -> Optional[Callable[..., Any]]:
    global _SWIGLU_FN, _SWIGLU_INIT_ATTEMPTED
    if not _SWIGLU_INIT_ATTEMPTED:
        _SWIGLU_FN = bind(
            "c_swiglu_f32",
            [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
            None,
        )
        _SWIGLU_INIT_ATTEMPTED = True
    return _SWIGLU_FN

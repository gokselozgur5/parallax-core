"""Python wrapper for the native C GELU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import bind

_GELU_FN: Optional[Callable[..., Any]] = None
_GELU_INIT_ATTEMPTED: bool = False


def _get_gelu_fn() -> Optional[Callable[..., Any]]:
    global _GELU_FN, _GELU_INIT_ATTEMPTED
    if not _GELU_INIT_ATTEMPTED:
        _GELU_FN = bind(
            "c_gelu_f32",
            [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ],
            None,
        )
        _GELU_INIT_ATTEMPTED = True
    return _GELU_FN

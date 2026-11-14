from __future__ import annotations

import ctypes
from typing import Tuple
import numpy as np
from parallax.native.loader import bind

_FP = ctypes.POINTER(ctypes.c_float)
_c_mha = bind(
    "c_mha_split_qkv_f32",
    [_FP] * 4 + [ctypes.c_int] * 5,
    ctypes.c_int,
)


def is_mha_proj_available() -> bool:
    """Return True if native c_mha_split_qkv_f32 is bound."""
    return _c_mha is not None

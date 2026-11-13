"""Loader and ctypes binding helper for native C and SIMD kernels."""

from __future__ import annotations

import ctypes
from typing import Optional

from parallax.c_engine import _find_or_build_library


def get_c_lib() -> Optional[ctypes.CDLL]:
    """Return the loaded CDLL handle from parallax.c_engine, or None."""
    return _find_or_build_library()

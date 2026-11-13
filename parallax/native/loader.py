"""Loader and ctypes binding helper for native C and SIMD kernels."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional, Sequence

from parallax.c_engine import _find_or_build_library


def get_c_lib() -> Optional[ctypes.CDLL]:
    """Return the loaded CDLL handle from parallax.c_engine, or None."""
    return _find_or_build_library()


def bind(
    name: str,
    argtypes: Sequence[Any],
    restype: Any = None,
) -> Optional[Callable[..., Any]]:
    """Bind a C symbol from the shared library with ctypes signatures."""
    lib = get_c_lib()
    if lib is None:
        return None
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.argtypes = list(argtypes)
    fn.restype = restype
    return fn

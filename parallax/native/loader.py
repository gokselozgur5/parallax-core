"""Loader, lazy ctypes binding, and int32 dimension checks for native kernels."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional, Sequence

from parallax.c_engine import _find_or_build_library

INT32_MAX: int = 2147483647
INT32_MIN: int = -2147483648


def get_c_lib() -> Optional[ctypes.CDLL]:
    """Return the loaded CDLL handle from parallax.c_engine, or None."""
    return _find_or_build_library()


def check_int32_bounds(*values: int, label: str = "Dimensions") -> None:
    """Raise ValueError if any dimension or element count overflows signed int32."""
    for val in values:
        if val < INT32_MIN or val > INT32_MAX:
            raise ValueError(f"{label} exceed INT32_MAX: {values}")


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


class LazySymbol:
    """Lazy ctypes symbol wrapper that rebinds automatically when CDLL changes."""

    __slots__ = ("name", "argtypes", "restype", "_int_indices", "_cached_lib", "_cached_fn")

    def __init__(self, name: str, argtypes: Sequence[Any], restype: Any = None) -> None:
        self.name = name
        self.argtypes = list(argtypes)
        self.restype = restype
        self._int_indices = tuple(i for i, t in enumerate(self.argtypes) if t is ctypes.c_int)
        self._cached_lib: Optional[ctypes.CDLL] = None
        self._cached_fn: Optional[Callable[..., Any]] = None

    def resolve(self) -> Optional[Callable[..., Any]]:
        """Resolve and cache the C function against the active CDLL handle."""
        lib = get_c_lib()
        if lib is None:
            self._cached_lib = None
            self._cached_fn = None
            return None
        if lib is not self._cached_lib:
            self._cached_lib = lib
            try:
                fn = getattr(lib, self.name)
                fn.argtypes = list(self.argtypes)
                fn.restype = self.restype
            except AttributeError:
                fn = None
            self._cached_fn = fn
        return self._cached_fn

    def __bool__(self) -> bool:
        return self.resolve() is not None

    def __call__(self, *args: Any) -> Any:
        for idx in self._int_indices:
            if idx < len(args) and isinstance(args[idx], int) and not isinstance(args[idx], bool):
                check_int32_bounds(args[idx])
        fn = self.resolve()
        if fn is None:
            raise RuntimeError(f"{self.name} symbol is not available")
        return fn(*args)


def lazy_bind(
    name: str,
    argtypes: Sequence[Any],
    restype: Any = None,
) -> LazySymbol:
    """Create a deferred symbol binding without loading or compiling at import time."""
    return LazySymbol(name, argtypes, restype)

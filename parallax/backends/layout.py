"""Layout guard that prepares arrays for native kernels."""

from __future__ import annotations

import numpy as np

# Object, complex, void, unicode and byte-string dtypes cannot become float32 safely.
_REJECTED_KINDS = frozenset("OcVUS")


def _check_dtype(arr: np.ndarray) -> None:
    """Raise TypeError when ``arr`` cannot be converted to float32."""
    if arr.dtype.kind in _REJECTED_KINDS:
        raise TypeError(f"native kernels need real numeric input, got dtype {arr.dtype}")


def ensure_f32_contiguous(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a C-contiguous float32 array and whether a copy was made."""
    _check_dtype(arr)
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr, False
    return np.ascontiguousarray(arr, dtype=np.float32), True

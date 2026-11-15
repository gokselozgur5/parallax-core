"""Layout guard for validating tensor memory layouts before native execution."""

from __future__ import annotations

from typing import Any
import numpy as np


class LayoutError(ValueError):
    """Raised when tensor memory layout violates native backend requirements."""


def _extract_array(tensor_or_arr: Any) -> np.ndarray:
    if hasattr(tensor_or_arr, "data") and isinstance(tensor_or_arr.data, np.ndarray):
        return tensor_or_arr.data
    return tensor_or_arr if isinstance(tensor_or_arr, np.ndarray) else np.asarray(tensor_or_arr)


def is_c_contiguous(tensor_or_arr: Any) -> bool:
    """Return True if tensor memory layout is C-contiguous."""
    return bool(_extract_array(tensor_or_arr).flags.c_contiguous)


def validate_layout(tensor_or_arr: Any, *, contiguous: bool = True) -> None:
    """Validate that tensor layout meets native execution constraints."""
    if contiguous and not is_c_contiguous(tensor_or_arr):
        raise LayoutError("tensor memory layout must be C-contiguous")

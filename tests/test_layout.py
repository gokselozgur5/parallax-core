from __future__ import annotations

import numpy as np
import pytest

from parallax.core.layout import (
    LayoutError,
    is_c_contiguous,
    validate_layout,
)
from parallax.tensor import Tensor


def test_is_c_contiguous():
    c_arr = np.ones((4, 4), dtype=np.float64)
    non_c_arr = c_arr[:, ::2]
    assert is_c_contiguous(c_arr) is True
    assert is_c_contiguous(non_c_arr) is False
    assert is_c_contiguous(Tensor(c_arr)) is True
    assert is_c_contiguous(Tensor(non_c_arr)) is False


def test_validate_layout_contiguity():
    non_c = np.ones((4, 4), dtype=np.float32)[:, ::2]
    with pytest.raises(LayoutError, match="contiguous"):
        validate_layout(non_c, contiguous=True)


def test_validate_layout_dtype():
    obj_arr = np.array(["a", "b"], dtype=object)
    with pytest.raises(LayoutError, match="dtype"):
        validate_layout(obj_arr)

    f64_arr = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(LayoutError, match="dtype"):
        validate_layout(f64_arr, allowed_dtypes=[np.float32])
    validate_layout(f64_arr, allowed_dtypes=[np.float64, np.float32])


from __future__ import annotations

import ctypes
import numpy as np
from parallax.metal.aligned import aligned_zeros, get_page_size, is_aligned
from parallax.metal.view import wrap_buffer


def test_wrap_buffer_strided_page_aligned_copies_logical_elements() -> None:
    ps = max(get_page_size(), 16384)
    elems = (ps // np.dtype(np.float32).itemsize) * 2
    base = aligned_zeros(elems, dtype=np.float32)
    base[:] = np.arange(elems, dtype=np.float32)
    arr = base[::2]
    assert is_aligned(arr)
    assert not arr.flags.c_contiguous

    with wrap_buffer(arr) as view:
        raw = np.ctypeslib.as_array((ctypes.c_float * arr.size).from_address(view.ptr))
        assert view.as_numpy().flags.c_contiguous
        assert not np.shares_memory(view.as_numpy(), base)
        assert np.array_equal(raw, arr)


def test_wrap_buffer_transposed_page_aligned_copies_row_major_order() -> None:
    ps = max(get_page_size(), 16384)
    cols = ps // (4 * np.dtype(np.float32).itemsize)
    base = aligned_zeros((4, cols), dtype=np.float32)
    base[:] = np.arange(base.size, dtype=np.float32).reshape(4, cols)
    arr = base.T
    assert is_aligned(arr)
    assert not arr.flags.c_contiguous

    with wrap_buffer(arr) as view:
        raw = np.ctypeslib.as_array((ctypes.c_float * arr.size).from_address(view.ptr))
        assert view.as_numpy().flags.c_contiguous
        assert not np.shares_memory(view.as_numpy(), base)
        assert np.array_equal(raw, arr.ravel())
        assert np.array_equal(view.as_numpy(), arr)

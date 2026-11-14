from __future__ import annotations

import numpy as np
from parallax.metal.aligned import (
    align_size,
    aligned_empty,
    aligned_zeros,
    get_page_size,
    is_aligned,
)


def test_page_size_and_align_size() -> None:
    ps = get_page_size()
    assert ps > 0
    assert align_size(0, ps) == 0
    assert align_size(1, ps) == ps
    assert align_size(ps, ps) == ps
    assert align_size(ps + 1, ps) == 2 * ps


def test_aligned_empty_properties() -> None:
    ps = get_page_size()
    arr = aligned_empty(1024, dtype=np.float32)
    assert arr.ctypes.data % ps == 0
    assert arr.nbytes % ps == 0
    assert is_aligned(arr)


def test_aligned_zeros_and_writability() -> None:
    arr = aligned_zeros(512, dtype=np.float32)
    assert np.all(arr == 0)
    arr[0] = 123.45
    assert arr[0] == np.float32(123.45)


def test_custom_page_size_alignment() -> None:
    custom_ps = 8192
    arr = aligned_empty(100, dtype=np.float32, page_size=custom_ps)
    assert is_aligned(arr, page_size=custom_ps)

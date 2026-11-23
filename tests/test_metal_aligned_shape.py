from __future__ import annotations

import numpy as np
from parallax.metal.aligned import aligned_empty, aligned_zeros, get_page_size, is_aligned
from parallax.metal.view import wrap_buffer


def test_aligned_zeros_returns_requested_shape() -> None:
    arr = aligned_zeros((3, 5), dtype=np.float32)
    assert arr.shape == (3, 5)
    assert np.all(arr == 0)


def test_aligned_empty_preserves_shapes_and_page_backing() -> None:
    ps = get_page_size()
    for shape in ((3, 5), (), 0, (3, 0), 7):
        arr = aligned_empty(shape, dtype=np.float32)
        expected = (shape,) if isinstance(shape, int) else shape
        assert arr.shape == expected
        assert arr.ctypes.data % ps == 0
        assert arr.base is not None and arr.base.nbytes % ps == 0
        assert is_aligned(arr)


def test_wrap_buffer_preserves_unaligned_shape() -> None:
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    with wrap_buffer(x) as view:
        assert view.as_numpy().shape == x.shape
        assert np.array_equal(view.as_numpy(), x)

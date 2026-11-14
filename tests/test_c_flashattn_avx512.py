import numpy as np
import pytest
from parallax.native.flashattn_avx512 import (
    _get_flashattn_avx512_fn,
    is_avx512_flashattn_available,
    flashattn_avx512,
)


def test_flashattn_avx512_symbols_bound():
    fn = _get_flashattn_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_flashattn_available(), bool)


def test_flashattn_avx512_dim_and_type_checks():
    q = np.ones((2, 2), dtype=np.float32)
    k = np.ones((2, 3), dtype=np.float32)
    v = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        flashattn_avx512(q, k, v)

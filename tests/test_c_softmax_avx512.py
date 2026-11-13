import numpy as np
import pytest
from parallax.native.softmax_avx512 import (
    _get_softmax_avx512_fn,
    is_avx512_softmax_available,
    softmax_avx512,
)


def test_softmax_avx512_symbols_bound():
    fn = _get_softmax_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_softmax_available(), bool)


def test_softmax_avx512_shape_checks():
    x = np.array(1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Input array must have at least 1 dimension"):
        softmax_avx512(x)

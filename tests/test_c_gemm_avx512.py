import numpy as np
import pytest
from parallax.native.gemm_avx512 import _get_avx512_fn, gemm_avx512, is_avx512_available


def test_avx512_symbols_bound():
    fn = _get_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_available(), bool)


def test_gemm_avx512_shape_checks():
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        gemm_avx512(a, b)

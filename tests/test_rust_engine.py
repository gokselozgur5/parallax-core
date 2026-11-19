import numpy as np
import pytest
from parallax.rust_engine import rust_gemm


def test_rust_gemm_does_not_round_int64_product_to_float32() -> None:
    a = np.array([[2**24 + 1]], dtype=np.int64)
    b = np.array([[1]], dtype=np.int64)
    try:
        out = rust_gemm(a, b)
    except TypeError:
        return
    assert int(out[0, 0]) == 16777217


def test_rust_gemm_complex_and_non_float32_buffer_rejected() -> None:
    with pytest.raises(TypeError):
        rust_gemm(np.array([[1 + 2j]]), np.array([[1 + 2j]]))
    with pytest.raises(TypeError):
        rust_gemm(np.zeros((2, 2)), np.zeros((2, 2)), c=np.zeros((2, 2), dtype=np.int64))

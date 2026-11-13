import numpy as np
import pytest
from parallax.backends.base import check_backend_contract
from parallax.backends.numpy_backend import NumpyBackend


def test_numpy_backend_contract():
    b = NumpyBackend()
    check_backend_contract(b)
    assert b.name == "numpy"
    assert b.available() is True
    assert b.supports("gemm", "float32", (2, 2)) is True
    assert b.supports("unknown_op", "float32", (2, 2)) is False


def test_numpy_gemm():
    b = NumpyBackend()
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    c = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    res = b.gemm(a, c)
    assert np.allclose(res, a @ c)

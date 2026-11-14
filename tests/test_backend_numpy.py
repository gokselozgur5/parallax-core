import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.numpy import NumpyBackend


def test_numpy_backend_contract():
    backend = NumpyBackend()
    check_backend_contract(backend)
    assert backend.name == "numpy"
    assert backend.available() is True
    assert backend.supports("gemm", "float32", (2, 2)) is True
    assert backend.supports("unknown", "float32", (2, 2)) is False


def test_numpy_backend_gemm():
    backend = NumpyBackend()
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    out = backend.gemm(a, b)
    assert np.allclose(out, a @ b)


def test_numpy_backend_softmax():
    backend = NumpyBackend()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = backend.softmax(x)
    expected = np.exp(x) / np.sum(np.exp(x))
    assert np.allclose(out, expected)

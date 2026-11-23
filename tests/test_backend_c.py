import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.c_backend import CBackend, CEngineBackend


def test_c_backend_contract():
    backend = CEngineBackend()
    check_backend_contract(backend)
    assert backend.name == "c"
    assert backend.available() is True
    assert backend.supports("gemm", "float32", (2, 2)) is True
    assert backend.supports("unknown", "float32", (2, 2)) is False
    assert backend.supports("gemm", "float64", (2, 2)) is False
    assert CBackend is CEngineBackend


def test_c_backend_gemm():
    backend = CEngineBackend()
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    out = backend.gemm(a, b)
    assert np.allclose(out, a @ b)


def test_c_backend_softmax():
    backend = CEngineBackend()
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = backend.softmax(x)
    expected = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    assert np.allclose(out, expected)


def test_c_backend_rmsnorm():
    backend = CEngineBackend()
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight = np.array([1.0, 1.0], dtype=np.float32)
    out = backend.rmsnorm(x, weight, eps=1e-5)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    expected = (x / np.sqrt(variance + 1e-5)) * weight
    assert np.allclose(out, expected, atol=1e-4)




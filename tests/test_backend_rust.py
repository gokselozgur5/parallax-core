import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.rust_backend import RustBackend, RustEngineBackend


def test_rust_backend_contract():
    backend = RustEngineBackend()
    check_backend_contract(backend)
    assert backend.name == "rust"
    assert isinstance(backend.available(), bool)
    assert backend.supports("unknown", "float32", (2, 2)) is False


def test_rust_backend_gemm():
    backend = RustEngineBackend()
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    out = backend.gemm(a, b)
    assert np.allclose(out, a @ b)


def test_rust_backend_softmax():
    backend = RustEngineBackend()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = backend.softmax(x)
    expected = np.exp(x) / np.sum(np.exp(x))
    assert np.allclose(out, expected)


def test_rust_backend_supports_and_alias():
    backend = RustBackend()
    assert isinstance(backend, RustEngineBackend)
    if backend.available():
        assert backend.supports("gemm", "float32", (2, 2)) is True
        assert backend.supports("softmax", "float32", (2, 2)) is True
        assert backend.supports("rmsnorm", "float32", (2, 2)) is False




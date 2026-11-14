import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.rust_engine import RustEngineBackend


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


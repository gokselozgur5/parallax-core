import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.c_engine import CEngineBackend, CBackend


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


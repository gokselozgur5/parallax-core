from parallax.backends.base import check_backend_contract
from parallax.backends.numpy import NumpyBackend


def test_numpy_backend_contract():
    backend = NumpyBackend()
    check_backend_contract(backend)
    assert backend.name == "numpy"
    assert backend.available() is True
    assert backend.supports("gemm", "float32", (2, 2)) is True
    assert backend.supports("unknown", "float32", (2, 2)) is False

from parallax.backends.base import check_backend_contract
from parallax.backends.rust_engine import RustEngineBackend


def test_rust_backend_contract():
    backend = RustEngineBackend()
    check_backend_contract(backend)
    assert backend.name == "rust"
    assert isinstance(backend.available(), bool)
    assert backend.supports("unknown", "float32", (2, 2)) is False

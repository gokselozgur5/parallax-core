import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.metal import MetalBackend


def test_metal_backend_contract():
    backend = MetalBackend()
    check_backend_contract(backend)
    assert backend.name == "metal"
    assert isinstance(backend.available(), bool)

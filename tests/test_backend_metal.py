import numpy as np
from parallax.backends.base import check_backend_contract
from parallax.backends.metal import MetalBackend


def test_metal_backend_contract():
    backend = MetalBackend()
    check_backend_contract(backend)
    assert backend.name == "metal"
    assert isinstance(backend.available(), bool)


def test_metal_backend_probe():
    from parallax.backends.metal import is_metal_available, probe_metal

    info = probe_metal()
    assert isinstance(info, dict)
    assert "available" in info
    assert "platform" in info
    assert is_metal_available() == info["available"]
    backend = MetalBackend()
    assert backend.probe() == info

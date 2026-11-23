import numpy as np
import pytest
from parallax.backends.base import check_backend_contract
from parallax.backends.metal_backend import MetalBackend, is_metal_available, probe_metal


def test_metal_backend_contract():
    backend = MetalBackend()
    check_backend_contract(backend)
    assert backend.name == "metal"
    assert isinstance(backend.available(), bool)


def test_metal_backend_probe():
    info = probe_metal()
    assert isinstance(info, dict)
    assert "available" in info
    assert "platform" in info
    assert is_metal_available() == info["available"]
    backend = MetalBackend()
    assert backend.probe() == info


def test_metal_backend_supports(monkeypatch):
    backend = MetalBackend()
    monkeypatch.setattr(backend, "available", lambda: False)
    assert not backend.supports("gemm", np.float32, (2, 2))
    monkeypatch.setattr(backend, "available", lambda: True)
    assert backend.supports("gemm", np.float32, (2, 2))
    assert backend.supports("rmsnorm", "float32", (2, 2))
    assert not backend.supports("softmax", np.float32, (2, 2))
    assert not hasattr(backend, "softmax")
    assert not backend.supports("unknown_op", np.float32, (2, 2))
    assert not backend.supports("gemm", np.float64, (2, 2))


def test_metal_backend_unavailable_execution():
    backend = MetalBackend()
    if not backend.available():
        with pytest.raises(RuntimeError, match="Metal backend is not available"):
            backend.gemm(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(RuntimeError, match="Metal backend is not available"):
            backend.rmsnorm(np.zeros((2, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32))


def test_metal_backend_mock_execution(monkeypatch):
    backend = MetalBackend()
    monkeypatch.setattr(backend, "available", lambda: True)
    import parallax.metal_engine as me

    monkeypatch.setattr(me, "metal_gemm", lambda a, b: a @ b)
    a = np.ones((2, 2), dtype=np.float32)
    out = backend.gemm(a, a)
    assert np.allclose(out, a @ a)


def test_metal_backend_mock_rmsnorm(monkeypatch):
    backend = MetalBackend()
    monkeypatch.setattr(backend, "available", lambda: True)
    import parallax.metal_engine as me

    monkeypatch.setattr(me, "metal_rmsnorm", lambda x, w, eps=1e-5: x * w)
    x = np.ones((2, 2), dtype=np.float32)
    w = np.ones((2,), dtype=np.float32)
    out = backend.rmsnorm(x, w)
    assert np.allclose(out, x * w)

from __future__ import annotations

import sys

import numpy as np
import pytest

from parallax.backends.metal_backend import MetalBackend


def test_metal_backend_rejects_ops_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    backend = MetalBackend()
    assert not backend.available()
    a = np.eye(2, dtype=np.float32)
    w = np.ones((2,), dtype=np.float32)
    with pytest.raises(RuntimeError, match="Metal backend is not available"):
        backend.gemm(a, a)
    with pytest.raises(RuntimeError, match="Metal backend is not available"):
        backend.rmsnorm(a, w)


def test_metal_backend_rejects_ops_without_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "parallax.backends.metal_backend.is_metal_available", lambda: False
    )
    backend = MetalBackend()
    assert not backend.available()
    with pytest.raises(RuntimeError, match="Metal backend is not available"):
        backend.gemm(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Metal backend is not available"):
        backend.rmsnorm(np.eye(2, dtype=np.float32), np.ones((2,), dtype=np.float32))

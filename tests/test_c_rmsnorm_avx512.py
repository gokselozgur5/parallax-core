import numpy as np
import pytest

from parallax.native.rmsnorm_avx512 import (
    is_avx512_rmsnorm_available,
    rmsnorm_avx512,
)


def test_avx512_rmsnorm_availability_probe():
    assert isinstance(is_avx512_rmsnorm_available(), bool)


def test_rmsnorm_avx512_shape_validation():
    x = np.ones((2, 16), dtype=np.float32)
    w_bad = np.ones((7,), dtype=np.float32)
    if not is_avx512_rmsnorm_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            rmsnorm_avx512(x, w_bad)
    else:
        with pytest.raises(ValueError, match="shape mismatch"):
            rmsnorm_avx512(x, w_bad)


def test_rmsnorm_avx512_raises_when_unavailable(monkeypatch):
    import parallax.native.rmsnorm_avx512 as mod

    monkeypatch.setattr(mod, "is_avx512_rmsnorm_available", lambda: False)
    x = np.ones((2, 16), dtype=np.float32)
    w = np.ones((16,), dtype=np.float32)
    with pytest.raises(RuntimeError, match="unavailable"):
        mod.rmsnorm_avx512(x, w)

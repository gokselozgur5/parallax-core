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


def _numpy_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return ((x / rms) * weight).astype(np.float32)


@pytest.mark.parametrize("shape", [(1, 16), (2, 32), (4, 64), (3, 77), (2, 128)])
def test_rmsnorm_avx512_numerical_parity(shape):
    if not is_avx512_rmsnorm_available():
        pytest.skip("AVX-512 RMSNorm not available on this platform")
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal(shape[1]).astype(np.float32)
    actual = rmsnorm_avx512(x, w, eps=1e-5)
    expected = _numpy_rmsnorm(x, w, eps=1e-5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

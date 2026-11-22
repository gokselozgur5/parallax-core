import numpy as np

import parallax.native.rmsnorm_avx512 as mod


def _ref_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x64 = x.astype(np.float64)
    rms = np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)
    return ((x64 / rms) * weight.astype(np.float64)).astype(np.float32)


def test_rmsnorm_avx512_scalar_fallback_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(mod, "is_avx512_rmsnorm_available", lambda: False)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 19)).astype(np.float32)
    w = rng.standard_normal((19,)).astype(np.float32)
    actual = mod.rmsnorm_avx512(x, w, eps=1e-5)
    expected = _ref_rmsnorm(x, w, eps=1e-5)
    assert actual.shape == x.shape and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_c_rmsnorm_avx512_f32_scalar_fallback_entry() -> None:
    fn = mod._get_rmsnorm_avx512_fn()
    assert fn is not None
    rng = np.random.default_rng(11)
    x = np.ascontiguousarray(rng.standard_normal((3, 21)), dtype=np.float32)
    w = np.ascontiguousarray(rng.standard_normal((21,)), dtype=np.float32)
    out = np.empty_like(x)
    status = fn(x.ctypes.data, w.ctypes.data, out.ctypes.data, 3, 21, 1e-5)
    assert status == 0
    np.testing.assert_allclose(out, _ref_rmsnorm(x, w, eps=1e-5), rtol=1e-5, atol=1e-5)

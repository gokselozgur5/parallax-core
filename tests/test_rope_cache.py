from __future__ import annotations
import numpy as np
import pytest
from parallax.attention import apply_rope
from parallax.native.rope_cache import RoPECache, is_rope_cache_available

def test_rope_cache_availability() -> None:
    assert is_rope_cache_available()

def test_rope_cache_validation() -> None:
    with pytest.raises(ValueError):
        RoPECache.create(dim=15, max_seq_len=64)
    with pytest.raises(ValueError):
        RoPECache.create(dim=16, max_seq_len=0)
    cache = RoPECache.create(dim=16, max_seq_len=32)
    with pytest.raises(ValueError):
        cache.apply_in_place(np.zeros((8, 4), dtype=np.float32))
    with pytest.raises(TypeError):
        cache.apply_in_place(np.zeros((4, 2, 16), dtype=np.float64))
    with pytest.raises(ValueError):
        cache.apply_in_place(np.zeros((4, 2, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        cache.apply_in_place(np.zeros((4, 2, 16), dtype=np.float32), start_pos=30)

def test_rope_cache_parity_interleaved() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((12, 4, 32)).astype(np.float32)
    cache = RoPECache.create(dim=32, max_seq_len=128, theta=10000.0)
    x_ref = np.transpose(x, (1, 0, 2))
    ref = apply_rope(x_ref, seq_dim=1, offset=10, theta=10000.0, interleaved=True)
    expected = np.transpose(ref, (1, 0, 2))
    actual = cache.apply_in_place(x.copy(), start_pos=10, interleaved=True)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

def test_rope_cache_parity_non_interleaved() -> None:
    rng = np.random.default_rng(123)
    x = rng.standard_normal((16, 2, 64)).astype(np.float32)
    cache = RoPECache.create(dim=64, max_seq_len=256, theta=50000.0)
    x_ref = np.transpose(x, (1, 0, 2))
    ref = apply_rope(x_ref, seq_dim=1, offset=5, theta=50000.0, interleaved=False)
    expected = np.transpose(ref, (1, 0, 2))
    actual = cache.apply_in_place(x.copy(), start_pos=5, interleaved=False)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

def test_apply_in_place_rotates_without_native_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    import math
    import parallax.native.rope_cache as rc
    monkeypatch.setattr(rc, "_c_build_rope_cache", None)
    monkeypatch.setattr(rc, "_c_apply_rope_cached", None)
    cache = rc.RoPECache.create(dim=4, max_seq_len=8)
    x = np.ones((1, 1, 4), dtype=np.float32)
    out = cache.apply_in_place(x, start_pos=1)
    assert out[0, 0, 0] == pytest.approx(math.cos(1.0) - math.sin(1.0), abs=1e-6)
    assert out[0, 0, 1] == pytest.approx(math.sin(1.0) + math.cos(1.0), abs=1e-6)


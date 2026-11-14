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

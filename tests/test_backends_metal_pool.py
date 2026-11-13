from __future__ import annotations

import sys
import numpy as np
import pytest
from parallax.backends.metal_pool import MetalBufferPool, ZeroCopyViewCache
from parallax.metal_engine import is_metal_available


def test_metal_pool_availability() -> None:
    pool = MetalBufferPool()
    if sys.platform != "darwin":
        assert not pool.available()
    else:
        assert pool.available() == is_metal_available()


def test_metal_pool_bucket_size() -> None:
    assert MetalBufferPool._bucket_size(100) == 256
    assert MetalBufferPool._bucket_size(256) == 256
    assert MetalBufferPool._bucket_size(257) == 512
    assert MetalBufferPool._bucket_size(1024) == 1024


def test_metal_pool_acquire_and_reuse() -> None:
    pool = MetalBufferPool()
    if not pool.available():
        pytest.skip("Metal unavailable")
    buf1 = pool.acquire(512)
    assert buf1 is not None
    pool.release(buf1, 512)
    s = pool.stats()
    assert s["cached_count"] == 1
    buf2 = pool.acquire(512)
    assert pool.stats()["hits"] == 1
    pool.release(buf2, 512)
    pool.clear()
    assert pool.stats()["cached_count"] == 0


def test_metal_pool_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    pool = MetalBufferPool()
    assert not pool.available()
    assert pool.acquire(1024) is None
    pool.release(None, 1024)


def test_zero_copy_view_cache() -> None:
    cache = ZeroCopyViewCache(max_entries=4)
    if not cache.available():
        pytest.skip("Metal unavailable")
    arr = np.ones((16, 16), dtype=np.float32)
    buf1 = cache.get_or_create(arr)
    assert buf1 is not None
    assert cache.count() == 1
    buf2 = cache.get_or_create(arr)
    assert buf1 == buf2
    cache.clear()
    assert cache.count() == 0


def test_zero_copy_view_cache_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    cache = ZeroCopyViewCache()
    assert not cache.available()
    arr = np.ones((8,), dtype=np.float32)
    assert cache.get_or_create(arr) is None
    cache.clear()

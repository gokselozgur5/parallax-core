"""Tests for dispatched KV Cache tensor buffer manager."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.kv_cache import (
    KVCacheBuffer,
    KVCacheManager,
    create_kv_cache,
)


def test_kv_cache_buffer_init_shapes():
    buf = KVCacheBuffer(batch_size=2, max_seq_len=64, num_heads=4, head_dim=16)
    assert buf.batch_size == 2
    assert buf.max_seq_len == 64
    assert buf.num_heads == 4
    assert buf.head_dim == 16
    assert buf.k.shape == (2, 4, 64, 16)
    assert buf.v.shape == (2, 4, 64, 16)


def test_kv_cache_memory_usage_bytes():
    buf = KVCacheBuffer(batch_size=1, max_seq_len=128, num_heads=2, head_dim=64)
    assert buf.memory_usage_bytes() == 131072


def test_kv_cache_single_step_update_and_view():
    buf = KVCacheBuffer(batch_size=1, max_seq_len=16, num_heads=2, head_dim=4)
    k_step = np.ones((1, 2, 1, 4), dtype=np.float32) * 3.0
    v_step = np.ones((1, 2, 1, 4), dtype=np.float32) * 5.0
    k_out, v_out = buf.update(k_step, v_step)
    assert buf.seq_len == 1
    assert k_out.shape == (1, 2, 1, 4)
    assert v_out.shape == (1, 2, 1, 4)
    np.testing.assert_allclose(k_out, k_step)
    np.testing.assert_allclose(v_out, v_step)


def test_kv_cache_prefill_and_decode_step():
    buf = KVCacheBuffer(batch_size=1, max_seq_len=16, num_heads=2, head_dim=4)
    k_pre = np.random.randn(1, 2, 4, 4).astype(np.float32)
    v_pre = np.random.randn(1, 2, 4, 4).astype(np.float32)
    k_out, v_out = buf.update(k_pre, v_pre)
    assert buf.seq_len == 4
    assert k_out.shape == (1, 2, 4, 4)
    k_dec = np.random.randn(1, 2, 1, 4).astype(np.float32)
    v_dec = np.random.randn(1, 2, 1, 4).astype(np.float32)
    k_out2, v_out2 = buf.update(k_dec, v_dec)
    assert buf.seq_len == 5
    assert k_out2.shape == (1, 2, 5, 4)
    np.testing.assert_allclose(k_out2[:, :, 4:5, :], k_dec)


def test_kv_cache_overflow_raises_value_error():
    buf = KVCacheBuffer(batch_size=1, max_seq_len=4, num_heads=2, head_dim=4)
    k_overflow = np.ones((1, 2, 5, 4), dtype=np.float32)
    v_overflow = np.ones((1, 2, 5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds maximum sequence length"):
        buf.update(k_overflow, v_overflow)


def test_kv_cache_manager_multi_layer():
    mgr = KVCacheManager(num_layers=3, batch_size=1, max_seq_len=16, num_heads=2, head_dim=4)
    assert len(mgr.layers) == 3
    assert mgr.memory_usage_bytes() == 3 * mgr.layers[0].memory_usage_bytes()
    k_in = np.ones((1, 2, 1, 4), dtype=np.float32)
    v_in = np.ones((1, 2, 1, 4), dtype=np.float32) * 2.0
    k_out, v_out = mgr.update(layer_idx=1, k_val=k_in, v_val=v_in)
    assert mgr.layers[1].seq_len == 1
    assert mgr.layers[0].seq_len == 0
    np.testing.assert_allclose(k_out, k_in)


def test_kv_cache_reset_clears_positions():
    buf = KVCacheBuffer(batch_size=1, max_seq_len=8, num_heads=2, head_dim=4)
    k = np.ones((1, 2, 2, 4), dtype=np.float32)
    buf.update(k, k)
    assert buf.seq_len == 2
    buf.reset()
    assert buf.seq_len == 0
    k_view, _ = buf.get_view()
    assert k_view.shape[2] == 0


def test_create_kv_cache_factory_dispatch():
    buf = create_kv_cache(batch_size=2, max_seq_len=32, num_heads=2, head_dim=8)
    assert isinstance(buf, KVCacheBuffer)
    mgr = create_kv_cache(batch_size=2, max_seq_len=32, num_heads=2, head_dim=8, num_layers=4)
    assert isinstance(mgr, KVCacheManager)
    assert mgr.num_layers == 4


def test_dispatch_kv_cache_package_exports():
    import parallax.dispatch as pd
    import python.parallax.dispatch as ppd

    for mod in (pd, ppd):
        assert hasattr(mod, "KVCacheBuffer")
        assert hasattr(mod, "KVCacheManager")
        assert hasattr(mod, "create_kv_cache")

"""Tests for dispatched KV Cache tensor buffer manager."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

import parallax

REPO_ROOT = Path(__file__).resolve().parent.parent
_dispatch_pkg = str(REPO_ROOT / "python" / "parallax")
if _dispatch_pkg not in parallax.__path__:
    parallax.__path__.append(_dispatch_pkg)

from parallax.dispatch.kv_cache import KVCacheBuffer


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

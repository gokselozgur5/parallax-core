"""Tests for dispatched KV Cache tensor buffer manager."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = str(REPO_ROOT / "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from parallax.dispatch.kv_cache import KVCacheBuffer


def test_kv_cache_buffer_init_shapes():
    buf = KVCacheBuffer(batch_size=2, max_seq_len=64, num_heads=4, head_dim=16)
    assert buf.batch_size == 2
    assert buf.max_seq_len == 64
    assert buf.num_heads == 4
    assert buf.head_dim == 16
    assert buf.k.shape == (2, 4, 64, 16)
    assert buf.v.shape == (2, 4, 64, 16)

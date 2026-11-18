import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("python"))

from parallax.dispatch.kv_cache import KVCacheBuffer

def test_kv_cache_update_rejects_inconsistent_writes():
    cache = KVCacheBuffer(batch_size=1, max_seq_len=8, num_heads=1, head_dim=2)
    two = np.ones((1, 1, 2, 2), dtype=np.float32)
    one = np.ones((1, 1, 1, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        cache.update(two, one)
    cache.reset()
    with pytest.raises(ValueError):
        cache.update(one, one, start_pos=5)
    assert cache.seq_len == 0

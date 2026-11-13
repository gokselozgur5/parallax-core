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

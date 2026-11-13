from __future__ import annotations

import sys
import pytest
from parallax.backends.metal_cache import MetalPipelineCache
from parallax.metal_engine import MSL_KERNELS, is_metal_available


def test_metal_cache_availability() -> None:
    cache = MetalPipelineCache()
    if sys.platform != "darwin":
        assert not cache.available()
    else:
        assert cache.available() == is_metal_available()


def test_metal_cache_key_generation() -> None:
    k1 = MetalPipelineCache._make_key("gemm_f32", "src1")
    k2 = MetalPipelineCache._make_key("gemm_f32", "src1")
    k3 = MetalPipelineCache._make_key("gemm_f32", "src2")
    assert k1 == k2
    assert k1 != k3
    assert k1[0] == "gemm_f32"

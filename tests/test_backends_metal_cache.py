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


def test_metal_cache_compilation_and_reuse() -> None:
    cache = MetalPipelineCache()
    if not cache.available():
        pytest.skip("Metal unavailable")
    assert not cache.has("rmsnorm_f32", MSL_KERNELS)
    pso1 = cache.get_or_compile("rmsnorm_f32", MSL_KERNELS)
    assert pso1 is not None
    assert cache.has("rmsnorm_f32", MSL_KERNELS)
    assert cache.stats()["misses"] == 1
    pso2 = cache.get_or_compile("rmsnorm_f32", MSL_KERNELS)
    assert pso1 == pso2
    assert cache.stats()["hits"] == 1
    cache.clear()
    assert cache.stats()["count"] == 0


def test_metal_cache_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    cache = MetalPipelineCache()
    assert not cache.available()
    assert cache.get_or_compile("func", "src") is None
    assert not cache.has("func", "src")
    cache.clear()
    assert cache.stats()["count"] == 0

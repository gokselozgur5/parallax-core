"""Tests for heuristic dispatch policy."""

from __future__ import annotations

from parallax.core.dispatch import (
    DispatchPolicy,
    PlatformInfo,
    get_candidate_backends,
    select_optimal_backend,
)


def test_heuristic_dispatch_optimal_metal():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    backend = policy.select_backend("gemm", (128, 128), platform=plat)
    assert backend == "metal"


def test_heuristic_dispatch_small_inputs_select_numpy():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    assert policy.select_backend("gemm", (8, 8), platform=plat) == "numpy"


def test_heuristic_dispatch_fallback_when_backend_unavailable():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    avail = {"c-neon", "numpy"}
    assert policy.select_backend("gemm", 10000, platform=plat, available_backends=avail) == "c-neon"


def test_heuristic_dispatch_x86_avx2():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="x86_64", system="Linux", avx2=True)
    assert policy.candidate_backends("gemm", 10000, platform=plat) == ["c-avx", "rust", "numpy"]
    assert policy.select_backend("gemm", 10000, platform=plat) == "c-avx"


def test_heuristic_dispatch_cuda():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="x86_64", system="Linux", cuda=True)
    assert policy.select_backend("gemm", 10000, platform=plat) == "cuda"


def test_select_optimal_backend_module_helper():
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    assert select_optimal_backend("gemm", 10000, platform=plat) == "metal"
    assert get_candidate_backends("gemm", 10000, platform=plat) == ["metal", "c-neon", "numpy"]


def test_custom_thresholds_override():
    policy = DispatchPolicy(thresholds={"gemm": 100})
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    assert policy.select_backend("gemm", 50, platform=plat) == "numpy"
    assert policy.select_backend("gemm", 150, platform=plat) == "metal"


class _Done:
    stdout = "hw.optional.neon: 1\n"


def test_dispatch_does_not_probe_the_cpu_on_every_call(monkeypatch):
    import numpy as np
    from parallax.backends import cpuinfo, registry
    from parallax.backends.dispatch import dispatch
    from parallax.backends.numpy_backend import NumpyBackend

    if hasattr(cpuinfo.detect_cpu, "cache_clear"):
        cpuinfo.detect_cpu.cache_clear()
    calls = []
    monkeypatch.setattr(cpuinfo.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cpuinfo.subprocess, "run", lambda *a, **k: calls.append(a) or _Done())
    added = "numpy" not in registry.list_backends()
    if added:
        registry.register("numpy", NumpyBackend())
    try:
        a = np.ones((4, 4), dtype=np.float32)
        for _ in range(20):
            dispatch("gemm", a, a)
    finally:
        if added:
            registry.unregister("numpy")
        if hasattr(cpuinfo.detect_cpu, "cache_clear"):
            cpuinfo.detect_cpu.cache_clear()
    assert len(calls) <= 1, f"sysctl was spawned {len(calls)} times for 20 dispatch calls"


def test_fallback_history_does_not_grow_without_bound(monkeypatch):
    import numpy as np

    import parallax.backends.dispatch as dd
    from parallax.backends import registry
    from parallax.backends.cpuinfo import CpuInfo
    from parallax.backends.numpy_backend import NumpyBackend

    monkeypatch.setattr(dd, "detect_cpu", lambda: CpuInfo(arch="x86_64", system="Linux"))
    monkeypatch.setenv("PARALLAX_BACKEND", "metal")
    added = "numpy" not in registry.list_backends()
    if added:
        registry.register("numpy", NumpyBackend())
    dd.clear_fallback_history()
    try:
        a = np.ones((2, 2), dtype=np.float64)
        for _ in range(20000):
            dd.dispatch("gemm", a, a)
        size = len(dd.get_fallback_history())
    finally:
        dd.clear_fallback_history()
        if added:
            registry.unregister("numpy")
    assert size < 20000, f"{size} fallback records kept after 20000 calls"




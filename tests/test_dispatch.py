"""Tests for heuristic dispatch policy."""

from __future__ import annotations

import pytest

from parallax.core.dispatch import DispatchPolicy, PlatformInfo


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

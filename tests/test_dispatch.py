"""Tests for heuristic dispatch policy."""

from __future__ import annotations

import pytest

from parallax.core.dispatch import DispatchPolicy, PlatformInfo


def test_heuristic_dispatch_optimal_metal():
    policy = DispatchPolicy()
    plat = PlatformInfo(arch="arm64", system="Darwin", neon=True)
    backend = policy.select_backend("gemm", (128, 128), platform=plat)
    assert backend == "metal"

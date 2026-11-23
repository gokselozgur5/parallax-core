"""Tests for the unified native kernel dispatcher, lazy rebinding, and bounds checks."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import parallax.native.dispatcher as disp
from parallax.backends.cpuinfo import CpuInfo
from parallax.native import KernelSelection, select_kernel
from parallax.native.dispatcher import (
    OPS,
    flashattn,
    gelu,
    gemm,
    layernorm,
    reset_dispatch_cache,
    rmsnorm,
    rope,
    softmax,
    swiglu,
)


def test_dispatcher_isa_priority_order() -> None:
    assert OPS == ("gemm", "rmsnorm", "softmax", "flashattn", "layernorm", "rope", "gelu", "swiglu")
    assert isinstance(select_kernel("gemm"), KernelSelection)
    with pytest.raises(ValueError, match="Unsupported kernel operation"):
        select_kernel("unknown_op")

    x86_all = CpuInfo(arch="x86_64", system="Linux", avx2=True, avx512f=True)
    x86_avx2 = CpuInfo(arch="x86_64", system="Linux", avx2=True, avx512f=False)
    arm_neon = CpuInfo(arch="arm64", system="Darwin", neon=True)
    plain = CpuInfo(arch="riscv64", system="Linux")

    with (
        patch.object(disp, "is_avx512_available", return_value=True),
        patch.object(disp, "is_avx2_gemm_available", return_value=True),
        patch.object(disp, "is_gemm_neon_available", return_value=True),
    ):
        assert select_kernel("gemm", x86_all).isa == "avx512"
        assert select_kernel("gemm", x86_avx2).isa == "avx2"
        assert select_kernel("gemm", arm_neon).isa == "neon"
        assert select_kernel("gemm", plain).isa in ("scalar_c", "numpy")

    with (
        patch.object(disp, "is_avx512_rmsnorm_available", return_value=True),
        patch.object(disp, "is_avx2_rmsnorm_available", return_value=True),
        patch.object(disp, "is_neon_rmsnorm_available", return_value=True),
    ):
        assert select_kernel("rmsnorm", x86_all).isa == "avx512"
        assert select_kernel("rmsnorm", x86_avx2).isa == "avx2"
        assert select_kernel("rmsnorm", arm_neon).isa == "neon"

    with (
        patch.object(disp, "is_avx512_softmax_available", return_value=True),
        patch.object(disp, "is_avx2_softmax_available", return_value=True),
        patch.object(disp, "is_softmax_neon_available", return_value=True),
    ):
        assert select_kernel("softmax", x86_all).isa == "avx512"
        assert select_kernel("softmax", x86_avx2).isa == "avx2"
        assert select_kernel("softmax", arm_neon).isa == "neon"

    with (
        patch.object(disp, "is_avx512_flashattn_available", return_value=True),
        patch.object(disp, "is_flashattn_neon_available", return_value=True),
    ):
        assert select_kernel("flashattn", x86_all).isa == "avx512"
        assert select_kernel("flashattn", arm_neon).isa == "neon"
        assert select_kernel("flashattn", plain).isa == "numpy"


def test_dispatched_ops_numerical_parity_against_independent_reference() -> None:
    reset_dispatch_cache()
    rng = np.random.default_rng(2026)
    a = rng.standard_normal((5, 7)).astype(np.float32)
    b = rng.standard_normal((7, 6)).astype(np.float32)
    ref_gemm = np.array(
        [[sum(float(a[i, k]) * float(b[k, j]) for k in range(7)) for j in range(6)] for i in range(5)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(gemm(a, b), ref_gemm, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(gemm(a, b, cpu=CpuInfo(arch="generic", system="Linux")), ref_gemm, rtol=1e-4, atol=1e-4)

    x = rng.standard_normal((2, 3, 9)).astype(np.float32)
    w = rng.standard_normal((9,)).astype(np.float32)
    bias = rng.standard_normal((9,)).astype(np.float32)
    x64 = x.astype(np.float64)
    ref_rms = ((x64 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + 1e-5)) * w.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(rmsnorm(x, w, eps=1e-5), ref_rms, rtol=1e-5, atol=1e-5)

    mean = np.mean(x64, axis=-1, keepdims=True)
    var = np.mean((x64 - mean) ** 2, axis=-1, keepdims=True)
    ref_ln = (((x64 - mean) / np.sqrt(var + 1e-5)) * w.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(layernorm(x, weight=w, bias=bias, eps=1e-5), ref_ln, rtol=1e-5, atol=1e-5)

    sm_ref = np.exp(x64 - np.max(x64, axis=1, keepdims=True))
    sm_ref = (sm_ref / np.sum(sm_ref, axis=1, keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(softmax(x, axis=1), sm_ref, rtol=1e-5, atol=1e-5)

    q, k, v = (rng.standard_normal((4, 8)).astype(np.float32) for _ in range(3))
    sc = 1.0 / math.sqrt(8.0)
    scores = (q.astype(np.float64) @ k.astype(np.float64).T) * sc
    scores = np.where(np.triu(np.ones((4, 4), dtype=bool), k=1), -np.inf, scores)
    p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    ref_fa = ((p / np.sum(p, axis=-1, keepdims=True)) @ v.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(flashattn(q, k, v, is_causal=True), ref_fa, rtol=1e-4, atol=1e-4)

    rx = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    c1, s1 = math.cos(1.0), math.sin(1.0)
    c2, s2 = math.cos(1.0 / 100.0), math.sin(1.0 / 100.0)
    ref_rope = np.array([[[1.0 * c1 - 2.0 * s1, 1.0 * s1 + 2.0 * c1, 3.0 * c2 - 4.0 * s2, 3.0 * s2 + 4.0 * c2]]], dtype=np.float32)
    np.testing.assert_allclose(rope(rx, start_pos=1, theta=10000.0, interleaved=True), ref_rope, rtol=1e-5, atol=1e-5)

    gx = np.array([-2.0, 0.0, 1.5], dtype=np.float32)
    c_k = math.sqrt(2.0 / math.pi)
    ref_gelu = np.array([0.5 * v * (1.0 + math.tanh(c_k * (v + 0.044715 * v**3))) for v in gx], dtype=np.float32)
    np.testing.assert_allclose(gelu(gx), ref_gelu, rtol=1e-5, atol=1e-6)

    ref_swiglu = np.array([(v / (1.0 + math.exp(-float(v)))) * 2.0 for v in gx], dtype=np.float32)
    np.testing.assert_allclose(swiglu(gx, np.full_like(gx, 2.0)), ref_swiglu, rtol=1e-5, atol=1e-6)



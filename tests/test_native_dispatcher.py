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


def test_all_wrappers_import_without_loading_c_library() -> None:
    code = (
        "import importlib, parallax.c_engine as ce\n"
        "calls = []\n"
        "ce._find_or_build_library = lambda: calls.append(1)\n"
        "mods = [\n"
        "  'avx2_rmsnorm', 'avx2_softmax', 'dispatcher', 'flashattn_avx512', 'flashattn_neon',\n"
        "  'gelu', 'gemm_avx2', 'gemm_avx512', 'gemm_neon', 'gqa_proj', 'layernorm', 'loader',\n"
        "  'mha_proj', 'neon_rmsnorm', 'rmsnorm_avx512', 'rope', 'rope_cache',\n"
        "  'softmax_avx512', 'softmax_neon', 'swiglu',\n"
        "]\n"
        "for m in mods:\n"
        "    importlib.import_module(f'parallax.native.{m}')\n"
        "assert calls == [], f'unexpected import-time loads: {calls}'\n"
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


def test_reset_c_lib_cache_falls_back_to_numpy_and_rebinds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from parallax.c_engine import reset_c_lib_cache
    from parallax.native.gqa_proj import is_gqa_proj_available
    from parallax.native.layernorm import is_layernorm_available
    from parallax.native.mha_proj import is_mha_proj_available
    from parallax.native.rope import is_rope_available
    from parallax.native.rope_cache import is_rope_cache_available

    reset_c_lib_cache()
    try:
        assert is_layernorm_available() and is_rope_available() and is_rope_cache_available()
        assert is_gqa_proj_available() and is_mha_proj_available()
        assert select_kernel("layernorm").isa == "scalar_c"

        monkeypatch.setenv("PARALLAX_C_LIB", str(tmp_path / "missing_libparallax_c.dylib"))
        reset_c_lib_cache()
        assert not is_layernorm_available() and not is_rope_available()
        assert not is_rope_cache_available() and not is_gqa_proj_available()
        for op in OPS:
            assert select_kernel(op).isa == "numpy"
        x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        w2 = np.array([1.0, 1.0], dtype=np.float32)
        assert gemm(x2, x2).shape == (2, 2)
        assert rmsnorm(x2, w2).shape == (2, 2)
        assert softmax(x2).shape == (2, 2)
        assert flashattn(x2, x2, x2).shape == (2, 2)
        assert layernorm(x2, weight=w2).shape == (2, 2)
        assert rope(np.ones((2, 1, 4), dtype=np.float32)).shape == (2, 1, 4)
        assert gelu(x2).shape == (2, 2)
        assert swiglu(x2, x2).shape == (2, 2)

        monkeypatch.delenv("PARALLAX_C_LIB", raising=False)
        reset_c_lib_cache()
        assert is_layernorm_available() and is_rope_available() and is_rope_cache_available()
        assert select_kernel("layernorm").isa == "scalar_c"
        assert select_kernel("gelu").isa == "scalar_c"
        assert select_kernel("swiglu").isa == "scalar_c"
    finally:
        monkeypatch.delenv("PARALLAX_C_LIB", raising=False)
        reset_c_lib_cache()


def test_dispatcher_rejects_int32_overflow_across_all_ops() -> None:
    class _BigArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = np.dtype(np.float32)
            self.size = math.prod(shape)

    big2d: Any = _BigArray((50000, 50000))
    b_small: Any = _BigArray((50000, 2))
    w_big = np.ones((50000,), dtype=np.float32)
    big3d: Any = _BigArray((1, 2147483648, 4))
    for call in (
        lambda: gemm(big2d, b_small),
        lambda: rmsnorm(big2d, w_big),
        lambda: softmax(big2d),
        lambda: flashattn(big2d, big2d, big2d),
        lambda: layernorm(big2d, weight=w_big),
        lambda: rope(big3d),
        lambda: gelu(big2d),
        lambda: swiglu(big2d, big2d),
    ):
        with pytest.raises(ValueError, match="INT32_MAX"):
            call()


def test_csrc_deduplicated_headers_and_isa_files() -> None:
    csrc = Path(__file__).resolve().parent.parent / "parallax" / "csrc"
    for c_file in sorted(csrc.glob("*.c")):
        text = c_file.read_text(encoding="utf-8")
        assert "#define PARALLAX_STATUS_" not in text, f"{c_file.name} redefines status macros"
    for dispatch_file in ("rmsnorm.c", "gemm.c", "softmax.c"):
        text = (csrc / dispatch_file).read_text(encoding="utf-8")
        assert "USE_NEON" not in text and "USE_AVX" not in text, f"{dispatch_file} duplicates SIMD loops"

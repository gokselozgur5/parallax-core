"""Regression tests for AVX-512 ISA probes, double accumulation, and size_t/INT32_MAX bounds."""

from __future__ import annotations

import ctypes
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from parallax.native.flashattn_avx512 import flashattn_avx512, is_avx512_flashattn_available
from parallax.native.flashattn_neon import flashattn_neon
from parallax.native.gemm_avx512 import gemm_avx512, is_avx512_available
from parallax.native.gemm_neon import gemm_neon
from parallax.native.layernorm import c_layernorm
from parallax.native.loader import bind
from parallax.native.rmsnorm_avx512 import is_avx512_rmsnorm_available, rmsnorm_avx512
from parallax.native.rope import rope_in_place
from parallax.native.rope_cache import RoPECache
from parallax.native.softmax_avx512 import is_avx512_softmax_available, softmax_avx512
from parallax.native.softmax_neon import softmax_neon

CSRC_DIR = Path(__file__).resolve().parent.parent / "parallax" / "csrc"
AVX512_FEATURES = ("avx512f", "avx512dq", "avx512vl", "avx512bw")
INDEX_CHECKED_KERNELS = (
    "gemm_neon.c", "gemm_avx512.c", "softmax_neon.c", "softmax_avx512.c",
    "rmsnorm_avx512.c", "flashattn_neon.c", "flashattn_avx512.c",
    "layernorm.c", "rope.c", "rope_cache.c", "gqa_proj.c", "mha_proj.c",
    "gelu.c", "swiglu.c",
)


class _MockArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(np.float32)
        self.size = math.prod(shape)


def _extract_c_function(source: str, func_name: str) -> str:
    match = re.search(rf"\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{", source)
    assert match is not None, f"Function {func_name} not found"
    depth = 0
    for idx in range(match.end() - 1, len(source)):
        depth += (source[idx] == "{") - (source[idx] == "}")
        if depth == 0:
            return source[match.start() : idx + 1]
    raise AssertionError(f"Unclosed brace in {func_name}")


def test_avx512_probes_check_all_compiled_extensions() -> None:
    simd_hdr = (CSRC_DIR / "parallax_simd.h").read_text(encoding="utf-8")
    probes = (
        ("gemm_avx512.c", "c_has_avx512_f32"),
        ("rmsnorm_avx512.c", "c_has_avx512_rmsnorm_f32"),
        ("softmax_avx512.c", "c_has_avx512_softmax_f32"),
        ("flashattn_avx512.c", "c_has_avx512_flashattn_f32"),
    )
    for filename, fn_name in probes:
        body = _extract_c_function((CSRC_DIR / filename).read_text(encoding="utf-8"), fn_name)
        expanded = body + (simd_hdr if "PARALLAX_HAS_AVX512_ISA" in body else "")
        for feat in AVX512_FEATURES:
            assert f'"{feat}"' in expanded, f"{filename}:{fn_name} missing {feat} check"
    results = [
        is_avx512_available(), is_avx512_rmsnorm_available(),
        is_avx512_softmax_available(), is_avx512_flashattn_available(),
    ]
    assert all(isinstance(r, bool) for r in results) and len(set(results)) == 1


def test_rmsnorm_avx512_double_accumulation_no_overflow() -> None:
    src = (CSRC_DIR / "rmsnorm_avx512.c").read_text(encoding="utf-8")
    kernel_body = _extract_c_function(src, "rmsnorm_avx512_kernel")
    assert "__m512d" in kernel_body and "_mm512_cvtps_pd" in kernel_body
    assert "_mm512_fmadd_pd" in kernel_body and "_mm512_fmadd_ps" not in kernel_body

    x = np.full((2, 33), 1e20, dtype=np.float32)
    w = np.full((33,), 2.0, dtype=np.float32)
    out = rmsnorm_avx512(x, w, eps=1e-5)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 2.0, rtol=1e-5, atol=1e-5)

    c_fn = bind(
        "c_rmsnorm_avx512_f32",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float],
        ctypes.c_int,
    )
    assert c_fn is not None
    out_c = np.empty_like(x)
    assert c_fn(x.ctypes.data, w.ctypes.data, out_c.ctypes.data, 2, 33, ctypes.c_float(1e-5)) == 0
    np.testing.assert_allclose(out_c, 2.0, rtol=1e-5, atol=1e-5)


def test_csrc_kernels_use_size_t_offsets() -> None:
    for filename in INDEX_CHECKED_KERNELS:
        src = (CSRC_DIR / filename).read_text(encoding="utf-8")
        assert "size_t" in src, f"{filename} does not use size_t indices"
        assert "for (int " not in src, f"{filename} still uses signed int loop counter"
        assert "(long)" not in src and "long " not in src, f"{filename} uses platform-width long index"


def test_native_wrappers_reject_int32_max_overflow() -> None:
    big_dim: Any = _MockArray((2147483648, 4))
    big_prod: Any = _MockArray((50000, 50000))
    b4 = np.zeros((4, 4), dtype=np.float32)
    b50k: Any = _MockArray((50000, 2))
    w4 = np.ones((4,), dtype=np.float32)
    big_rope: Any = _MockArray((1, 2147483648, 4))
    cache = RoPECache.create(dim=4, max_seq_len=8)
    calls = [
        lambda: gemm_neon(big_dim, b4), lambda: gemm_neon(big_prod, b50k),
        lambda: gemm_avx512(big_dim, b4), lambda: gemm_avx512(big_prod, b50k),
        lambda: softmax_neon(big_dim), lambda: softmax_neon(big_prod),
        lambda: softmax_avx512(big_dim), lambda: softmax_avx512(big_prod),
        lambda: rmsnorm_avx512(big_dim, w4),
        lambda: flashattn_neon(big_dim, big_dim, big_dim),
        lambda: flashattn_avx512(big_dim, big_dim, big_dim),
        lambda: c_layernorm(big_dim, weight=w4),
        lambda: rope_in_place(big_rope),
        lambda: RoPECache.create(dim=4, max_seq_len=1 << 62),
        lambda: cache.apply_in_place(big_rope),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="INT32_MAX"):
            call()

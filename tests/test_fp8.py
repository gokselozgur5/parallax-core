from parallax.fp8 import FP8Format


def test_fp8_format_constants() -> None:
    assert FP8Format.E4M3_MAX == 448.0
    assert FP8Format.E5M2_MAX == 57344.0

import numpy as np
from parallax.fp8 import quantize_fp8, dequantize_fp8, fp8_gemm


def test_fp8_quantize_dequantize_roundtrip() -> None:
    np.random.seed(42)
    x = np.random.randn(4, 16)
    q_blocks, scales = quantize_fp8(x, format_type="e4m3", block_size=8)
    reconstructed = dequantize_fp8(q_blocks, scales, x.shape)
    assert reconstructed.shape == x.shape
    cos_sim = np.sum(x * reconstructed) / (
        np.linalg.norm(x) * np.linalg.norm(reconstructed)
    )
    assert cos_sim > 0.99


def test_fp8_gemm_precision() -> None:
    np.random.seed(42)
    a = np.random.randn(8, 16)
    b = np.random.randn(16, 8)
    ref = np.matmul(a, b)
    out = fp8_gemm(a, b, format_type="e5m2", block_size=8)
    assert out.shape == ref.shape
    cos_sim = np.sum(ref * out) / (np.linalg.norm(ref) * np.linalg.norm(out))
    assert cos_sim > 0.99

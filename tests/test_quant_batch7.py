"""
Unit tests for Q4_0 scaling with signed extreme (Issue #297).
"""

import numpy as np
import pytest

from parallax.gguf_quant import (
    QK4_0,
    dequantize_q4_0,
    quantize_q4_0,
)


def test_q4_0_signed_extreme_scaling():
    """Issue #297: Q4_0 scale is picked from signed extreme matching ggml."""
    pos_block = np.zeros(QK4_0, dtype=np.float32)
    pos_block[0] = 7.0
    pos_bytes = quantize_q4_0(pos_block)
    pos_delta = np.frombuffer(pos_bytes[:2], dtype="<f2")[0]
    assert np.isclose(pos_delta, 1.0, atol=1e-3)
    pos_dequant = dequantize_q4_0(pos_bytes, (QK4_0,))
    assert np.max(np.abs(pos_block - pos_dequant)) <= float(pos_delta) / 2.0 + 1e-4

    neg_block = np.zeros(QK4_0, dtype=np.float32)
    neg_block[0] = -8.0
    neg_bytes = quantize_q4_0(neg_block)
    neg_delta = np.frombuffer(neg_bytes[:2], dtype="<f2")[0]
    assert np.isclose(neg_delta, 1.0, atol=1e-3)
    neg_dequant = dequantize_q4_0(neg_bytes, (QK4_0,))
    assert np.max(np.abs(neg_block - neg_dequant)) <= float(neg_delta) / 2.0 + 1e-4

    np.random.seed(42)
    for sign in [-1.0, 1.0]:
        for target_scale in [0.5, 1.0, 2.0, 3.5]:
            d = target_scale
            extreme = -8.0 * d if sign < 0 else 7.0 * d
            inner = np.linspace(-6.5 * d, 5.5 * d, QK4_0 - 1).astype(np.float32)
            inner += np.random.uniform(-0.4 * d, 0.4 * d, size=QK4_0 - 1).astype(np.float32)
            block = np.concatenate([[extreme], inner]).astype(np.float32)

            q_bytes = quantize_q4_0(block)
            extracted_d = float(np.frombuffer(q_bytes[:2], dtype="<f2")[0])
            deq = dequantize_q4_0(q_bytes, (QK4_0,))
            err = np.abs(block - deq)
            assert err[0] <= 1e-3
            assert np.max(err) <= abs(extracted_d) / 2.0 + 1e-3


from parallax.nn import Linear
from parallax.quant import QuantizedLinear
from parallax.tensor import Tensor


def test_quantized_linear_cache():
    """Issue #298: QuantizedLinear caches dequantized weight across forwards."""
    np.random.seed(42)
    lin = Linear(16, 8, bias=True)
    qlin = QuantizedLinear.from_linear(lin)

    assert qlin._cached_weight is None
    x1 = Tensor(np.random.randn(2, 4, 16))
    out1 = qlin(x1)
    assert qlin._cached_weight is not None
    cached_ref = qlin._cached_weight

    x2 = Tensor(np.random.randn(2, 4, 16))
    out2 = qlin(x2)
    assert qlin._cached_weight is cached_ref

    qlin.invalidate_cache()
    assert qlin._cached_weight is None
    out3 = qlin(x2)
    assert qlin._cached_weight is not None


from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.quant import calculate_compression_ratio, quantize_model


def test_calculate_compression_ratio_baseline_and_inplace():
    """Issue #299: calculate_compression_ratio supports baseline_dtype and in-place quant."""
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        tie_weights=False,
    )
    model = ParallaxLM(cfg)

    quant_model = quantize_model(model)
    stats_fp32 = calculate_compression_ratio(model, quant_model, baseline_dtype="float32")
    stats_fp16 = calculate_compression_ratio(model, quant_model, baseline_dtype="float16")

    assert stats_fp32["original_mb"] > 0
    assert stats_fp32["quantized_mb"] > 0
    assert stats_fp32["ratio"] > 1.5
    assert np.isclose(stats_fp16["original_mb"] * 2.0, stats_fp32["original_mb"], atol=1e-2)
    assert stats_fp16["quantized_mb"] == stats_fp32["quantized_mb"]

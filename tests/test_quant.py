"""
Unit tests for Pure NumPy INT8 Post-Training Quantization (M16).
"""

import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear
from parallax.quant import (
    QuantizedLinear,
    calculate_compression_ratio,
    dequantize_int8,
    quantize_int8,
    quantize_model,
)
from parallax.tensor import Tensor


def test_quantize_and_dequantize_int8_precision():
    np.random.seed(42)
    # Weights centered around 0 with range [-2.5, 2.5]
    w = np.random.uniform(-2.5, 2.5, size=(32, 32))

    w_int8, scale = quantize_int8(w)
    assert w_int8.dtype == np.int8
    assert np.max(np.abs(w_int8)) <= 127

    w_recon = dequantize_int8(w_int8, scale)
    assert w_recon.shape == w.shape

    # Relative error should be very small (< 1%)
    max_err = np.max(np.abs(w - w_recon))
    assert max_err < (2.5 / 127.0) * 1.5


def test_quantized_linear_forward():
    np.random.seed(42)
    lin = Linear(16, 8, bias=True)
    qlin = QuantizedLinear.from_linear(lin)

    x = Tensor(np.random.randn(2, 4, 16))
    out_orig = lin(x)
    out_quant = qlin(x)

    assert out_quant.shape == (2, 4, 8)
    # Mean absolute error between float and INT8 linear projection should be small
    mae = np.mean(np.abs(out_orig.data - out_quant.data))
    assert mae < 0.05


def test_quantize_entire_parallax_lm_model():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        use_rope=True,
    )
    model = ParallaxLM(cfg)
    idx = np.array([[1, 2, 3, 4]])

    orig_logits = model(idx).data

    quant_model = quantize_model(model)
    quant_logits = quant_model(idx).data

    assert quant_logits.shape == orig_logits.shape

    # Top-1 greedy token predictions should closely match
    pred_orig = np.argmax(orig_logits[0], axis=-1)
    pred_quant = np.argmax(quant_logits[0], axis=-1)
    # High concordance between predictions
    match_ratio = np.mean(pred_orig == pred_quant)
    assert match_ratio >= 0.75


def test_calculate_compression_ratio_includes_mlp_and_head():
    import copy
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        tie_weights=False,
    )
    orig_model = ParallaxLM(cfg)
    quant_model = quantize_model(copy.deepcopy(orig_model))

    stats = calculate_compression_ratio(orig_model, quant_model)
    assert stats["original_mb"] > 0
    assert stats["quantized_mb"] > 0
    assert stats["quantized_mb"] < stats["original_mb"]
    assert stats["ratio"] > 1.0


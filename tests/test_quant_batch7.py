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
